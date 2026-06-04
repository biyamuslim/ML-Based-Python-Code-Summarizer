import torch
import argparse
import pickle
import sys
import os
from pathlib import Path

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
)
from src.preprocess import (
    bpe_decode,
    bpe_encode,
    build_model_input,
    load_bpe_tokenizer,
    pad_sequence,
)
from models.seq2seq_model import Encoder, Decoder, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD VOCAB
with open(os.path.join(OUTPUT_DIR, "code_vocab.pkl"), "rb") as f:
    code_vocab = pickle.load(f)

with open(os.path.join(OUTPUT_DIR, "doc_vocab.pkl"), "rb") as f:
    doc_vocab = pickle.load(f)

tokenizer = load_bpe_tokenizer(os.path.join(OUTPUT_DIR, "tokenizer.json"))

# LOAD MODEL
state_dict = torch.load(os.path.join(OUTPUT_DIR, "model.pt"), map_location=device)
INPUT_DIM = state_dict["encoder.embedding.weight"].shape[0] # vocab size for encoder input
OUTPUT_DIM = state_dict["decoder.embedding.weight"].shape[0] # vocab size for decoder output

if len(code_vocab) != INPUT_DIM: # If the vocab size has changed, trim the vocab to fit the model's expected input size.
    code_vocab = {token: idx for token, idx in code_vocab.items() if idx < INPUT_DIM} 

if len(doc_vocab) != OUTPUT_DIM: # If the vocab size has changed, trim the vocab to fit the model's expected output size.
    doc_vocab = {token: idx for token, idx in doc_vocab.items() if idx < OUTPUT_DIM}

# Reverse vocab (for decoding)
inv_doc_vocab = {v: k for k, v in doc_vocab.items()}

# Recreate the same model structure before loading trained weights.
encoder = Encoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, dropout=DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMBED_DIM, HIDDEN_DIM, dropout=DROPOUT)

model = Seq2Seq(encoder, decoder, device).to(device)
try:
    model.load_state_dict(state_dict) # Load the trained model weights into the model architecture.
except RuntimeError as exc:
    raise RuntimeError(
        "The saved model is incompatible with the current attention architecture. "
        f"Run `python scripts\\train.py` to retrain and create a new {os.path.join(OUTPUT_DIR, 'model.pt')}."
    ) from exc
model.eval() # Set the model to evaluation mode

def preprocess_input(code):
    # Format the input the same way as during training.
    ids = bpe_encode(tokenizer, build_model_input(code))
    padded = pad_sequence(ids, MAX_CODE_LEN)
    return torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)


# -------- INFERENCE --------
def generate_summary(code):
    src = preprocess_input(code)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

        # Start decoding from the start-of-summary token.
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
                ) # Get the decoder's output probabilities for the next token.
                # Use log probabilities for better numerical stability during beam search.
                log_probs = torch.log_softmax(output, dim=1)  
                #Select top BEAM_SIZE
                top_scores, top_tokens = torch.topk(log_probs, BEAM_SIZE, dim=1)

                # Keep the most likely next-token options for beam search.
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

            # Prefer high-scoring sequences while avoiding very short outputs.
            beams = sorted(
                candidates,
                key=lambda item: item[3] / (len(item[0]) ** 0.7),
                reverse=True
            )[:BEAM_SIZE]

        if not finished:
            finished = [(sequence, score) for sequence, _, _, score in beams]

        # Pick the best complete summary candidate.
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

    # Convert generated token IDs back into readable text.
    return bpe_decode(tokenizer, generated_ids)

# -------- MAIN --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a natural language summary for Python code.")
    parser.add_argument("--input", type=str, help="Python code snippet to summarize.")
    args = parser.parse_args()

    if args.input:
        summary = generate_summary(args.input)
        print("Generated Summary:", summary)
        sys.exit(0)

    print("Enter Python code. Press Enter on an empty line to summarize.")
    print("Type 'exit' on the first line to quit.\n")

    while True:
        lines = []
        first_line = input(">>> ")

        if first_line.lower() == "exit":
            break

        lines.append(first_line)

        while True:
            line = input("... ")
            if line == "":
                break
            lines.append(line)

        code = "\n".join(lines)
        summary = generate_summary(code)
        print("Generated Summary:", summary)
        print("-" * 50)
