"""The training loop. Ours, not a framework's.

Nothing here is exotic — AdamW, warmup, cosine decay, gradient clipping,
checkpointing. It is written out because wrapping someone else's Trainer would
defeat the stated purpose of the project.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .data import get_batch
from .model import GPTConfig, NanoChessGPT
from .utils import count_parameters, human_count, resolve_device, set_seed


@dataclass
class TrainConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 5000
    batch_size: int = 32
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eval_interval: int = 250
    eval_batches: int = 40
    checkpoint_interval: int = 500
    log_interval: int = 25
    seed: int = 1337
    device: str = "auto"
    time_budget_s: float | None = None
    history: list[dict] = field(default_factory=list)


def lr_at_step(step: int, config: TrainConfig) -> float:
    """Linear warmup into cosine decay, floored at ``min_lr``.

    Warmup exists because attention with a freshly random init produces large,
    badly-scaled gradients in the first few dozen steps; stepping at full rate
    there is the classic way to send a small transformer to a bad basin it
    never leaves.
    """
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.lr - config.min_lr)
