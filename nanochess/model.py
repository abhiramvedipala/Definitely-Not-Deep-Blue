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


class Block(nn.Module):
    """Pre-norm transformer block: norm before each sublayer, residual around it."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        assert x.shape == shape, x.shape
        return x


PRESETS = {
    # 2 layers, 16 wide: fast enough that no test needs to be slow.
    "tiny": dict(block_size=32, n_layer=2, n_head=2, n_embd=16),
    # The real run. 256 tokens is ~128 plies, so a whole game fits in context
    # and the model never has to reason about a position whose origin has
    # scrolled out of the window.
    "small": dict(block_size=256, n_layer=6, n_head=6, n_embd=384),
}


def model_config(preset: str, vocab_size: int, **overrides) -> GPTConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return GPTConfig(vocab_size=vocab_size, **{**PRESETS[preset], **overrides})


class NanoChessGPT(nn.Module):
    """Decoder-only transformer over chess move tokens."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying. Beyond saving 0.76M parameters, the input and output
        # spaces here are literally the same set of moves, so forcing one
        # representation is the honest modelling choice.
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block size {self.config.block_size}"
        )

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        assert x.shape == (B, T, self.config.n_embd), x.shape

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        assert logits.shape == (B, T, self.config.vocab_size), logits.shape

        loss = None
        if targets is not None:
            assert targets.shape == (B, T), targets.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, self.config.vocab_size), targets.reshape(B * T)
            )
        return logits, loss
