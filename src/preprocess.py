import os
import json
import pandas as pd
import re
import pickle
import io
import tokenize as py_tokenize
import ast

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

def load_data(folder_path, limit=2000):
    all_data = []

    # CodeSearchNet is stored as JSONL files, so each line is one example.
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, 'r') as f:
                for line in f:
                    if limit is not None and len(all_data) >= limit:
                        return pd.DataFrame(all_data)
                    all_data.append(json.loads(line))

    return pd.DataFrame(all_data)


def clean_data(df):
    # Keep only examples that have both source code and a docstring.
    df = df[df['docstring'].notna()]
    df = df[df['docstring'].str.strip() != ""]
    df = df[df['code'].notna()]
    df = df[df['code'].str.strip() != ""]

    if 'func_name' not in df.columns:
        df['func_name'] = ""
    df['func_name'] = df['func_name'].fillna("")

    return df[['code', 'docstring', 'func_name']]


def extract_func_name(code):
    # Prefer Python's AST when the snippet can be parsed.
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
    except SyntaxError:
        pass

    match = re.search(r"\bdef\s+([A-Za-z_]\w*)", code)
    return match.group(1) if match else ""


def strip_docstrings(code):
    try:
        # Remove embedded docstrings so the model cannot see the target summary.
        tokens = py_tokenize.generate_tokens(io.StringIO(code).readline)
        cleaned_tokens = []
        previous_type = py_tokenize.INDENT

        for token in tokens:
            token_type = token.type
            token_text = token.string

            is_docstring = (
                token_type == py_tokenize.STRING
                and previous_type in {
                    py_tokenize.INDENT,
                    py_tokenize.NEWLINE,
                    py_tokenize.NL,
                }
            )

            if not is_docstring:
                cleaned_tokens.append(token)

            previous_type = token_type

        return py_tokenize.untokenize(cleaned_tokens)
    except (py_tokenize.TokenError, IndentationError, SyntaxError):
        return code


def build_model_input(code, func_name=""):
    # The model sees the function name plus code without docstrings.
    code = strip_docstrings(code)
    func_name = func_name or extract_func_name(code)
    if func_name:
        return f"function name: {func_name}\ncode:\n{code}"
    return code


def extract_summary_target(docstring):
    """Keep the short descriptive part of a docstring as the training target."""
    docstring = re.sub(r"\s+", " ", str(docstring)).strip()
    if not docstring:
        return ""

    split_markers = [" :param", " :return", " Args:", " Returns:", " Parameters:"]
    for marker in split_markers:
        if marker in docstring:
            docstring = docstring.split(marker, 1)[0].strip()

    sentence_match = re.search(r"(.+?[.!?])(?:\s|$)", docstring)
    if sentence_match:
        return sentence_match.group(1).strip()

    return docstring


def split_identifier(identifier):
    # Split names like calculate_totalPrice into useful smaller parts.
    identifier = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", identifier)
    identifier = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", identifier)
    identifier = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", identifier)

    parts = [part for part in re.split(r"[_\s]+", identifier) if part]
    subtokens = []

    for part in parts:
        matches = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+", part)
        for match in matches:
            if match:
                subtokens.append(match.lower())

    return subtokens


def tokenize_doc(docstring):
    # Normalize common noisy pieces before extracting words.
    docstring = docstring.lower()
    docstring = re.sub(r"https?://\S+|www\.\S+", " <URL> ", docstring)
    docstring = re.sub(r"\d+(\.\d+)?", " <NUM> ", docstring)
    docstring = re.sub(r"['’]", " ", docstring)

    raw_tokens = re.findall(r"<URL>|<NUM>|[a-zA-Z_]\w*", docstring)
    tokens = []

    for token in raw_tokens:
        if token in {"<URL>", "<NUM>"}:
            tokens.append(token)
        else:
            tokens.extend(split_identifier(token))

    return tokens


def pad_sequence(seq, max_len, pad_value=0):
    # Pad or cut sequences so batches have fixed tensor sizes.
    if len(seq) < max_len:
        return seq + [pad_value] * (max_len - len(seq))
    return seq[:max_len]

def save_vocab(code_vocab, doc_vocab, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "code_vocab.pkl"), "wb") as f:
        pickle.dump(code_vocab, f)  # encoder/input vocabulary

    with open(os.path.join(output_dir, "doc_vocab.pkl"), "wb") as f:
        pickle.dump(doc_vocab, f)  # decoder/output vocabulary


def train_bpe_tokenizer(df, vocab_size=20000, min_frequency=2, output_path="outputs/tokenizer.json"):
    """Train one BPE tokenizer on both Python code and docstrings."""
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"],
    )

    def corpus_iterator():
        # Stream text into the tokenizer instead of building one huge string.
        for _, row in df.iterrows():
            yield row["model_input"] if "model_input" in row else row["code"]
            yield row["target_summary"] if "target_summary" in row else row["docstring"]

    tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    return tokenizer


def load_bpe_tokenizer(path="outputs/tokenizer.json"):
    tokenizer = Tokenizer.from_file(path)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def bpe_vocab(tokenizer):
    return tokenizer.get_vocab()


def bpe_encode(tokenizer, text):
    return tokenizer.encode(text).ids


def bpe_decode(tokenizer, ids):
    # Remove control tokens before turning IDs back into text.
    special_ids = {
        tokenizer.token_to_id("<PAD>"),
        tokenizer.token_to_id("<SOS>"),
        tokenizer.token_to_id("<EOS>"),
    }
    clean_ids = [token_id for token_id in ids if token_id not in special_ids]
    return tokenizer.decode(clean_ids).strip()
