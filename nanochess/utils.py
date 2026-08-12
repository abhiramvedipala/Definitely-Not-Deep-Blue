"""Seeding, device resolution and parameter accounting."""

from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    """Seed every RNG this project draws from.

    Reproducibility is a stated requirement, and it is worth being blunt about
    the limit: this makes a run repeatable on the same device, not across
    devices. MPS, CUDA and CPU kernels reduce in different orders.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str = "auto") -> torch.device:
    """Resolve a device string, preferring MPS on Apple Silicon then CUDA."""
    if choice != "auto":
        return torch.device(choice)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def human_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
