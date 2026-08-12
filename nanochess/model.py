"""A decoder-only transformer, written out rather than assembled from
``nn.Transformer``.

Every tensor operation carries a shape assertion. That is not defensive
programming for its own sake: a from-scratch transformer with a broadcasting
bug still trains, still produces a falling loss curve, and still generates
plausible-looking garbage. The assertions turn a week of confusion into an
immediate failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit lower-triangular mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd {config.n_embd} must divide evenly by n_head {config.n_head}"
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert C == self.n_embd, f"expected embedding width {self.n_embd}, got {C}"
        assert T <= self.mask.shape[-1], f"sequence length {T} exceeds block size {self.mask.shape[-1]}"

        qkv = self.qkv(x)
        assert qkv.shape == (B, T, 3 * C), qkv.shape
        q, k, v = qkv.split(self.n_embd, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        assert q.shape == (B, self.n_head, T, self.head_dim), q.shape

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        assert att.shape == (B, self.n_head, T, T), att.shape

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))

        y = att @ v
        assert y.shape == (B, self.n_head, T, self.head_dim), y.shape

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        assert y.shape == (B, T, C), y.shape
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network with the conventional 4x expansion."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = F.gelu(self.fc(x))
        assert h.shape == (B, T, 4 * C), h.shape
        out = self.dropout(self.proj(h))
        assert out.shape == (B, T, C), out.shape
        return out
