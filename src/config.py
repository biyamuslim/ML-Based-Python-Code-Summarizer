import os

# Shared project paths
DATASET_PATH = "D:/datasets/codesearchnet/python/python/final/jsonl/train"
TEST_DATASET_PATH = "D:/datasets/codesearchnet/python/python/final/jsonl/test"
OUTPUT_DIR = os.environ.get("MLSA_OUTPUT_DIR", "outputs")
# OUTPUT_DIR = os.environ.get("MLSA_OUTPUT_DIR", "D:/mlsa_outputs")

# Data and tokenizer settings
DATA_LIMIT = 15000
BPE_VOCAB_SIZE = 8000
MIN_FREQ = 2
MAX_CODE_LEN = 160
MAX_DOC_LEN = 25

# Model settings
EMBED_DIM = 64
HIDDEN_DIM = 128
DROPOUT = 0.0

# Training settings
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
CLIP = 1.0
SEED = 42
RESUME_TRAINING = True

# Decoding settings
BEAM_SIZE = 5
MIN_SUMMARY_LEN = 2

# Output artifact paths
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")
LIGHT_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint_light.pt")
