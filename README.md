# ML-Based Python Code Summarization

This project implements a PyTorch-based Python code summarization tool. It trains a custom LSTM encoder-decoder model with attention on CodeSearchNet Python code/docstring pairs and generates short natural language summaries for Python functions.

The project does not use a pretrained full language model. The tokenizer and neural model are trained as part of the project pipeline.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

The project expects the Python split of CodeSearchNet in JSONL format. The dataset paths are configured in:

```text
src/config.py
```

Default paths:

```text
Training data: D:/datasets/codesearchnet/python/python/final/jsonl/train
Test data:     D:/datasets/codesearchnet/python/python/final/jsonl/test
```

If the dataset is stored somewhere else, update `DATASET_PATH` and `TEST_DATASET_PATH` in `src/config.py`.

## Training

Run training from the project root:

```powershell
python scripts\train.py
```

Training performs preprocessing, trains the Byte-Level BPE tokenizer, builds the dataset tensors, trains the Seq2Seq attention model, and saves the best model based on validation loss.

Main output files:

```text
outputs/model.pt
outputs/tokenizer.json
outputs/code_vocab.pkl
outputs/doc_vocab.pkl
outputs/training_history.json
outputs/training_config.json
```

## Evaluation

Run evaluation from the project root:

```powershell
python scripts\evaluate.py
```

Evaluation uses examples from the official CodeSearchNet test split and reports:

- cross-entropy loss
- perplexity
- token accuracy
- exact match rate
- BLEU
- ROUGE-1
- ROUGE-L

Results are saved to:

```text
outputs/evaluation_results.json
```

## Inference

Summarize a single-line Python function:

```powershell
python scripts\summarize.py --input "def add_numbers(a, b): return a + b"
```

For multi-line input, run:

```powershell
python scripts\summarize.py
```

Then paste the function into the terminal and press Enter on an empty line to generate the summary. Type `exit` on the first line to quit.

Example:

```text
>>> def is_even(n):
...     return n % 2 == 0
...
Generated Summary: Check if the
```

## Project Structure

```text
mlsaProject/
|-- data/       Dataset-related files
|-- models/     Seq2Seq model definition
|-- outputs/    Saved model, tokenizer, vocabularies, and results
|-- scripts/    Training, evaluation, and inference scripts
|-- src/        Preprocessing, dataset, and configuration code
|-- archive/    Optional/old experiments
|-- README.md
|-- requirements.txt
```

## Notes

The final model is a custom baseline trained from scratch on CPU. Its outputs are sometimes incomplete or generic, but the project includes the full workflow required for code summarization: preprocessing, tokenization, model training, validation, evaluation, and command-line inference.
