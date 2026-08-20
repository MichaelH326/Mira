"""Data loading and batching for character-level training."""

from pathlib import Path

import torch

from mira.tokenizer import CharTokenizer


def load_corpus(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def prepare_data(
    text: str, tokenizer: CharTokenizer, val_fraction: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(len(data) * val_fraction)
    return data[:-n_val], data[-n_val:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of contiguous (input, shifted-target) sequences."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
