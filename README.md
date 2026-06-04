# ML-Based Python Code Summarization

A PyTorch-based Python code summarization tool that generates short natural language summaries for Python functions. The project uses a custom LSTM sequence-to-sequence model with attention and Byte-Level BPE tokenization.

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Configuration](#dataset-configuration)
- [Running the Application](#running-the-application)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Notes](#notes)
- [License](#license)

## Getting Started

This project trains and evaluates a custom neural code summarization model on the Python split of CodeSearchNet. The model is trained from scratch and does not use a pretrained full language model.

## Prerequisites

- Python 3.x
- PyTorch
- pandas
- numpy
- nltk
- rouge-score
- tokenizers

## Installation

Create and activate a virtual environment, then install the required dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset Configuration

The project expects the Python split of CodeSearchNet in JSONL format. The dataset used for this project was downloaded from Kaggle:

```text
https://www.kaggle.com/datasets/omduggineni/codesearchnet
```

Update the dataset paths in:

```text
src/config.py
```

Default paths:

```text
Training data: D:/datasets/codesearchnet/python/python/final/jsonl/train
Test data:     D:/datasets/codesearchnet/python/python/final/jsonl/test
```

The dataset itself is not included in the repository.

If your dataset is stored in a different location, update `DATASET_PATH` and `TEST_DATASET_PATH` in `src/config.py`.

## Running the Application

Train the model:

```powershell
python scripts\train.py
```

Run evaluation:

```powershell
python scripts\evaluate.py
```

Run inference:

```powershell
python scripts\summarize.py --input "def add_numbers(a, b): return a + b"
```

For multi-line input:

```powershell
python scripts\summarize.py
```

## Evaluation

The evaluation script uses examples from the official CodeSearchNet test split and reports:

- cross-entropy loss
- perplexity
- token accuracy
- exact match rate
- BLEU
- ROUGE-1
- ROUGE-L

Results are saved in:

```text
outputs/evaluation_results.json
```

## Usage

Example command:

```powershell
python scripts\summarize.py --input "def is_even(n): return n % 2 == 0"
```

Example output:

```text
Generated Summary: Check if the
```

For interactive multi-line input, paste the Python function into the terminal and press Enter on an empty line to generate the summary. Type `exit` on the first line to quit.

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

This is a custom baseline model trained from scratch on CPU. The generated summaries can be incomplete or generic, but the project includes the full workflow for preprocessing, tokenization, model training, validation, evaluation, and command-line inference.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
