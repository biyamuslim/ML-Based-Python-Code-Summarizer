import torch
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    def __init__(self, df):
        # Keep the padded ID sequences ready for PyTorch batches.
        self.code = df['code_ids_padded'].values
        self.doc = df['doc_ids_padded'].values

    def __len__(self):
        return len(self.code)

    def __getitem__(self, idx):
        # Each item returns one code input and its target summary.
        return {
            'code': torch.tensor(self.code[idx], dtype=torch.long),
            'doc': torch.tensor(self.doc[idx], dtype=torch.long)
        }
