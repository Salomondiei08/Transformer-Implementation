from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import random


def generate_sequence(length: int, vocab_size: int) -> torch.Tensor:
    """Generate a random integer sequence of given length."""
    ints = [random.randint(1, vocab_size - 2) for _ in range(length)]  # reserve 0 for PAD, vocab_size-1 maybe EOS
    return torch.tensor(ints, dtype=torch.long)


class CopyDataset(Dataset):
    """Dataset where target is identical to source (sequence copy)."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.data = [generate_sequence(seq_len, vocab_size) for _ in range(num_samples)]
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.data[idx]
        tgt = self.data[idx].clone()
        return src, tgt


class ReverseDataset(Dataset):
    """Dataset where target is the reversed source sequence."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.data = [generate_sequence(seq_len, vocab_size) for _ in range(num_samples)]
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.data[idx]
        tgt = torch.flip(src, dims=[0])
        return src, tgt


def collate_fn(batch, pad_token: int = 0):
    """Collate function to pad variable-length sequences."""
    srcs, tgts = zip(*batch)
    srcs = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_token)
    tgts = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_token)
    return srcs, tgts


def get_dataloader(task: str, batch_size: int, num_samples: int, seq_len: int, vocab_size: int):
    """Return a DataLoader for the specified toy task."""
    if task == "copy":
        dataset = CopyDataset(num_samples, seq_len, vocab_size)
    elif task == "reverse":
        dataset = ReverseDataset(num_samples, seq_len, vocab_size)
    else:
        raise ValueError(f"Unknown task: {task}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)