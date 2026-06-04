import torch
import json
import os
import pickle
import sys
import nltk
from pathlib import Path
from rouge_score import rouge_scorer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import (
    BEAM_SIZE,
    DROPOUT,
    EMBED_DIM,
    HIDDEN_DIM,
    MAX_CODE_LEN,
    MAX_DOC_LEN,
    MIN_SUMMARY_LEN,
    OUTPUT_DIR,
    TEST_DATASET_PATH,
)
from src.preprocess import (
    bpe_decode,
    bpe_encode,
    build_model_input,
    clean_data,
    extract_summary_target,
    load_bpe_tokenizer,
    load_data,
    pad_sequence,
    tokenize_doc,
)
from models.seq2seq_model import Encoder, Decoder, Seq2Seq

nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD VOCAB --------
with open(os.path.join(OUTPUT_DIR, "code_vocab.pkl"), "rb") as f:
    code_vocab = pickle.load(f)

with open(os.path.join(OUTPUT_DIR, "doc_vocab.pkl"), "rb") as f:
    doc_vocab = pickle.load(f)

tokenizer = load_bpe_tokenizer(os.path.join(OUTPUT_DIR, "tokenizer.json"))

# -------- LOAD MODEL --------
# Load the saved model weights and recover the expected vocab sizes.
state_dict = torch.load(os.path.join(OUTPUT_DIR, "model.pt"), map_location=device)
INPUT_DIM = state_dict["encoder.embedding.weight"].shape[0]
OUTPUT_DIM = state_dict["decoder.embedding.weight"].shape[0]

if len(code_vocab) != INPUT_DIM:
    code_vocab = {token: idx for token, idx in code_vocab.items() if idx < INPUT_DIM}

if len(doc_vocab) != OUTPUT_DIM:
    doc_vocab = {token: idx for token, idx in doc_vocab.items() if idx < OUTPUT_DIM}

inv_doc_vocab = {v: k for k, v in doc_vocab.items()}

# Rebuild the model before loading the trained parameters.
encoder = Encoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, dropout=DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMBED_DIM, HIDDEN_DIM, dropout=DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

try:
    model.load_state_dict(state_dict)
except RuntimeError as exc:
    raise RuntimeError(
        "The saved model is incompatible with the current attention architecture. "
        f"Run `python scripts\\train.py` to retrain and create a new {os.path.join(OUTPUT_DIR, 'model.pt')}."
    ) from exc
model.eval()

def preprocess_input(code):
    # Use the same input format as training: function name plus cleaned code.
    ids = bpe_encode(tokenizer, build_model_input(code))
    padded = pad_sequence(ids, MAX_CODE_LEN)
    return torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)


def preprocess_target(docstring):
    # Encode the reference summary for loss and token accuracy.
    ids = bpe_encode(tokenizer, extract_summary_target(docstring))
    ids = [doc_vocab['<SOS>']] + ids + [doc_vocab['<EOS>']]
    padded = pad_sequence(ids, MAX_DOC_LEN)
    return torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)

# -------- INFERENCE --------
def generate_summary(code):
    src = preprocess_input(code)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

        # Beam search keeps several possible summaries while decoding.
        beams = [([doc_vocab['<SOS>']], hidden, cell, 0.0)]
        finished = []

        for _ in range(MAX_DOC_LEN):
            candidates = []

            for sequence, beam_hidden, beam_cell, score in beams:
                input_token = torch.tensor([sequence[-1]], device=device)
                output, next_hidden, next_cell = model.decoder(
                    input_token,
                    beam_hidden,
                    beam_cell,
                    encoder_outputs
                )
                log_probs = torch.log_softmax(output, dim=1)
                top_scores, top_tokens = torch.topk(log_probs, BEAM_SIZE, dim=1)

                # Extend each beam with the most likely next tokens.
                for token_score, token_id in zip(top_scores[0], top_tokens[0]):
                    token_id = token_id.item()
                    new_sequence = sequence + [token_id]
                    new_score = score + token_score.item()
                    generated_len = len(new_sequence) - 1

                    if token_id == doc_vocab['<EOS>']:
                        if generated_len >= MIN_SUMMARY_LEN:
                            finished.append((new_sequence, new_score))
                        continue

                    candidates.append((new_sequence, next_hidden, next_cell, new_score))

            if not candidates:
                break

            # Keep only the strongest candidates after each decoding step.
            beams = sorted(
                candidates,
                key=lambda item: item[3] / (len(item[0]) ** 0.7),
                reverse=True
            )[:BEAM_SIZE]

        if not finished:
            finished = [(sequence, score) for sequence, _, _, score in beams]

        # Choose the best completed summary.
        best_sequence, _ = max(
            finished,
            key=lambda item: item[1] / (len(item[0]) ** 0.7)
        )

    generated_ids = []
    for token_id in best_sequence[1:]:
        if token_id == doc_vocab['<EOS>']:
            break

        if token_id in {doc_vocab["<PAD>"], doc_vocab["<SOS>"], doc_vocab["<EOS>"]}:
            continue
        if generated_ids and generated_ids[-1] == token_id:
            continue
        generated_ids.append(token_id)

    # Convert generated BPE token IDs back to text.
    return bpe_decode(tokenizer, generated_ids)

# -------- LOAD TEST DATA --------
df = load_data(TEST_DATASET_PATH, limit=1000)  # official test split sample
df = clean_data(df)
df['target_summary'] = df['docstring'].apply(extract_summary_target)
df = df[df['target_summary'].str.strip() != ""]

# -------- EVALUATION --------
bleu_scores = []
loss_scores = []
rouge1_scores = []
rougeL_scores = []
token_correct = 0
token_total = 0
exact_matches = 0
examples = []
criterion = torch.nn.CrossEntropyLoss(ignore_index=doc_vocab["<PAD>"])
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

for i in range(len(df)):
    code = df.iloc[i]['code']
    docstring = df.iloc[i]['docstring']
    target_summary = df.iloc[i]['target_summary']
    reference_tokens = tokenize_doc(target_summary)
    reference = " ".join(reference_tokens)

    # Generate one summary for this code sample.
    generated = generate_summary(code)

    # Cross-entropy loss and perplexity
    src_tensor = preprocess_input(code)
    trg_tensor = preprocess_target(target_summary)

    with torch.no_grad():
        output = model(src_tensor, trg_tensor, teacher_forcing_ratio=0)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = trg_tensor[:, 1:].reshape(-1)
        loss = criterion(output, target)
        loss_scores.append(loss.item())

        predictions = output.argmax(dim=1)
        non_pad_mask = target != doc_vocab['<PAD>']
        token_correct += (predictions[non_pad_mask] == target[non_pad_mask]).sum().item()
        token_total += non_pad_mask.sum().item()

    # BLEU
    ref_tokens = [reference.split()]
    gen_tokens = generated.split()
    bleu = nltk.translate.bleu_score.sentence_bleu(ref_tokens, gen_tokens)
    bleu_scores.append(bleu)
    exact_matches += int(generated.strip() == reference.strip())

    # ROUGE
    scores = rouge.score(reference, generated)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

    if i < 5:
        examples.append({
            "code": code,
            "reference": reference,
            "generated": generated,
            "loss": loss.item(),
            "perplexity": torch.exp(loss).item(),
            "token_accuracy": (
                (predictions[non_pad_mask] == target[non_pad_mask]).sum().item()
                / max(non_pad_mask.sum().item(), 1)
            ),
            "exact_match": generated.strip() == reference.strip(),
            "bleu": bleu,
            "rouge1": scores['rouge1'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
        })

    if i < 3:  # print few examples
        print("\nCODE:\n", code)
        print("REFERENCE:", reference)
        print("GENERATED:", generated)
        print("LOSS:", loss.item())
        print("PERPLEXITY:", torch.exp(loss).item())
        print("TOKEN ACCURACY:", (
            (predictions[non_pad_mask] == target[non_pad_mask]).sum().item()
            / max(non_pad_mask.sum().item(), 1)
        ))
        print("EXACT MATCH:", generated.strip() == reference.strip())
        print("BLEU:", bleu)
        print("ROUGE-1:", scores['rouge1'].fmeasure)
        print("ROUGE-L:", scores['rougeL'].fmeasure)

# -------- FINAL RESULTS --------
# Average all collected metrics.
avg_loss = sum(loss_scores) / len(loss_scores)
avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
avg_token_accuracy = token_correct / max(token_total, 1)
exact_match_rate = exact_matches / len(df)

print("\nAverage Loss:", avg_loss)
print("Average Perplexity:", avg_perplexity)
print("Token Accuracy:", avg_token_accuracy)
print("Exact Match Rate:", exact_match_rate)
print("Average BLEU:", avg_bleu)
print("Average ROUGE-1:", avg_rouge1)
print("Average ROUGE-L:", avg_rougeL)

results = {
    "average_loss": avg_loss,
    "average_perplexity": avg_perplexity,
    "token_accuracy": avg_token_accuracy,
    "exact_match_rate": exact_match_rate,
    "average_bleu": avg_bleu,
    "average_rouge1": avg_rouge1,
    "average_rougeL": avg_rougeL,
    "num_examples": len(df),
    "beam_size": BEAM_SIZE,
    "examples": examples,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Store metrics and examples for the report.
with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"Evaluation results saved to {os.path.join(OUTPUT_DIR, 'evaluation_results.json')}")
