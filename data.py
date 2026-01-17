from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterator, Optional

import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class ToyClientDataset(Dataset):
    """Toy dataset producing random token sequences for LM loss.

    This is only to validate the training loop and plumbing.
    Replace with real instruction datasets per silo in your experiments.
    """
    vocab_size: int
    seq_len: int
    num_samples: int = 256
    seed: int = 0

    def __post_init__(self):
        g = torch.Generator().manual_seed(self.seed)
        self.data = torch.randint(0, self.vocab_size, (self.num_samples, self.seq_len), generator=g)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        return {"input_ids": x, "labels": x.clone(), "attention_mask": torch.ones_like(x)}

def make_client_dataloaders(tokenizer, num_clients: int, seq_len: int, micro_batch: int):
    loaders = []
    for i in range(num_clients):
        ds = ToyClientDataset(
            vocab_size=tokenizer.vocab_size,
            seq_len=seq_len,
            num_samples=256,
            seed=1234 + i,
        )
        dl = DataLoader(ds, batch_size=micro_batch, shuffle=True)
        loaders.append(dl)
    return loaders
