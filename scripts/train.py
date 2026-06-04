import torch
import os
import sys
import json
import random
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import (
    BATCH_SIZE,
    BPE_VOCAB_SIZE,
    CHECKPOINT_PATH,
    CLIP,
    DATA_LIMIT,
    DATASET_PATH,
    DROPOUT,
    EMBED_DIM,
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    LIGHT_CHECKPOINT_PATH,
    MAX_CODE_LEN,
    MAX_DOC_LEN,
    MIN_FREQ,
    OUTPUT_DIR,
    RESUME_TRAINING,
    SEED,
)
from src.preprocess import *
from src.dataset import CodeDataset
from models.seq2seq_model import Encoder, Decoder, Seq2Seq

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Save checkpoints carefully because large model files can fail on low disk space.
def save_checkpoint_safely(checkpoint, checkpoint_path, light_checkpoint_path):
    temp_path = checkpoint_path + ".tmp"
    light_temp_path = light_checkpoint_path + ".tmp"

    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    except Exception as exc:
        print(f"Warning: checkpoint save failed: {exc}")
        print("Trying a smaller checkpoint without optimizer state.")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

        light_checkpoint = {
            "epoch": checkpoint["epoch"],
            "model_state_dict": checkpoint["model_state_dict"],
            "best_val_loss": checkpoint["best_val_loss"],
            "history": checkpoint["history"],
            "config": checkpoint["config"],
            "optimizer_state_available": False,
        }

        try:
            torch.save(
                light_checkpoint,
                light_temp_path,
                _use_new_zipfile_serialization=False,
            )
            os.replace(light_temp_path, light_checkpoint_path)
            print(f"Saved lightweight checkpoint to {light_checkpoint_path}")
        except Exception as light_exc:
            print(f"Warning: lightweight checkpoint save failed too: {light_exc}")
            print("Training can continue because the best model is still saved separately.")
            if os.path.exists(light_temp_path):
                try:
                    os.remove(light_temp_path)
                except OSError:
                    pass


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save the setup so the final run can be reported and reproduced.
config = {
    "architecture": "LSTM Seq2Seq with attention",
    "data_limit": DATA_LIMIT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "gradient_clip": CLIP,
    "seed": SEED,
    "tokenizer": "Byte-Level BPE",
    "input_columns": ["func_name", "code"],
    "target_column": "target_summary",
    "bpe_vocab_size": BPE_VOCAB_SIZE,
    "min_freq": MIN_FREQ,
    "dropout": DROPOUT,
    "max_code_len": MAX_CODE_LEN,
    "max_doc_len": MAX_DOC_LEN,
    "resume_training": RESUME_TRAINING,
    "checkpoint_path": CHECKPOINT_PATH,
    "light_checkpoint_path": LIGHT_CHECKPOINT_PATH,
    "output_dir": OUTPUT_DIR,
}

print("Training config:", config)
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump(config, f, indent=2)

df = load_data(DATASET_PATH, limit=DATA_LIMIT)
df = clean_data(df)

# Build the actual model input: function name plus code without embedded docstrings.
df['model_input'] = df.apply(
    lambda row: build_model_input(row['code'], row['func_name']),
    axis=1
)

# Keep the target short by extracting only the first sentence of the docstring, which is often a concise summary.
df['target_summary'] = df['docstring'].apply(extract_summary_target)
df = df[df['target_summary'].str.strip() != ""]
print("Loaded examples:", len(df))

resume_checkpoint_path = None
if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    resume_checkpoint_path = CHECKPOINT_PATH
elif RESUME_TRAINING and os.path.exists(LIGHT_CHECKPOINT_PATH):
    resume_checkpoint_path = LIGHT_CHECKPOINT_PATH

resume_from_checkpoint = resume_checkpoint_path is not None

if resume_from_checkpoint:
    print(f"Found checkpoint: {resume_checkpoint_path}")
    tokenizer = load_bpe_tokenizer(os.path.join(OUTPUT_DIR, "tokenizer.json"))

    with open(os.path.join(OUTPUT_DIR, "code_vocab.pkl"), "rb") as f:
        code_vocab = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "doc_vocab.pkl"), "rb") as f:
        doc_vocab = pickle.load(f)
else:
    # Train one shared BPE tokenizer for both code inputs and summaries.
    tokenizer = train_bpe_tokenizer(
        df,
        vocab_size=BPE_VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        output_path=os.path.join(OUTPUT_DIR, "tokenizer.json")
    )
    code_vocab = bpe_vocab(tokenizer)
    doc_vocab = code_vocab #using one shared tokenizer for both code and summaries
    save_vocab(code_vocab, doc_vocab, output_dir=OUTPUT_DIR)

print("BPE vocab size:", tokenizer.get_vocab_size())

# Convert text into token IDs before padding and batching.
df['code_ids'] = df['model_input'].apply(lambda x: bpe_encode(tokenizer, x))
df['doc_ids'] = df['target_summary'].apply(lambda x: bpe_encode(tokenizer, x))

# Wrap each target summary with start and end tokens.
sos_id = tokenizer.token_to_id("<SOS>")
eos_id = tokenizer.token_to_id("<EOS>")
df['doc_ids'] = df['doc_ids'].apply(lambda x: [sos_id] + x + [eos_id])

df['code_ids_padded'] = df['code_ids'].apply(lambda x: pad_sequence(x, MAX_CODE_LEN))
df['doc_ids_padded'] = df['doc_ids'].apply(lambda x: pad_sequence(x, MAX_DOC_LEN))

# Use an 80/20 train-validation split with a fixed seed for reproducibility.
dataset = CodeDataset(df)
# 20% Validation split
val_size = max(1, int(0.2 * len(dataset))) 
# 80% Training split
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED) # to keep the split consistent across runs
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Build the LSTM encoder-decoder model with attention.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

encoder = Encoder(len(code_vocab), EMBED_DIM, HIDDEN_DIM, dropout=DROPOUT)
decoder = Decoder(len(doc_vocab), EMBED_DIM, HIDDEN_DIM, dropout=DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=doc_vocab["<PAD>"]) #loss Function that ignores padding tokens when calculating loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #optimizer that updates model weights

# Training loop: save the best model based on validation loss.
best_val_loss = float("inf")
history = []
history_path = os.path.join(OUTPUT_DIR, "training_history.json")
start_epoch = 0

if resume_from_checkpoint:
    try:
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print("Optimizer state not found; continuing with a fresh optimizer.")
        best_val_loss = checkpoint["best_val_loss"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
    except Exception as exc:
        print(f"Warning: could not load checkpoint: {exc}")
        print("Starting from epoch 0. The old checkpoint may be incomplete or corrupted.")
        resume_from_checkpoint = False

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_train_loss = 0

    for i, batch in enumerate(train_loader):
        src = batch['code'].to(device)
        trg = batch['doc'].to(device)

        optimizer.zero_grad() # reset accumulated gradients from previous batch
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Clip gradients  prevents the model weights from being updated too aggressively by limiting large gradients, which helps keep LSTM training stable..
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        total_train_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0

    # Validation is done without updating model weights.
    with torch.no_grad():
        for batch in val_loader:
            src = batch['code'].to(device)
            trg = batch['doc'].to(device)

            output = model(src, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    train_perplexity = torch.exp(torch.tensor(avg_train_loss))
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))
    print(
        f"Epoch {epoch} complete | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Perplexity: {train_perplexity:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Perplexity: {val_perplexity:.4f}"
    )

    epoch_record = {
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "train_perplexity": train_perplexity.item(),
        "val_loss": avg_val_loss,
        "val_perplexity": val_perplexity.item(),
        "best_val_loss": min(best_val_loss, avg_val_loss),
        "model_saved": avg_val_loss < best_val_loss,
    }
    history.append(epoch_record)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
        print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

    save_checkpoint_safely(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "config": config,
        },
        CHECKPOINT_PATH,
        LIGHT_CHECKPOINT_PATH
    )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

print(f"Training complete. Best model saved to {os.path.join(OUTPUT_DIR, 'model.pt')}")
print(f"Training history saved to {history_path}")
