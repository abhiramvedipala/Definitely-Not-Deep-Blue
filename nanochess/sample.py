"""Sampling: temperature, top-k, and a streaming generator."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.nn import functional as F

from .model import NanoChessGPT


def apply_top_k(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    """Mask all but the ``k`` largest logits to ``-inf``.

    Returned as a new tensor rather than modified in place, so callers can keep
    the raw logits — the eval harness wants them for diagnostics.
    """
    if k is None or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one token per row from ``logits`` (shape ``(B, vocab)``)."""
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    logits = apply_top_k(logits, top_k)
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: NanoChessGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
    stop_at_eos: bool = False,
    eos_id: int | None = None,
    forced_tokens: list[int] | None = None,
) -> torch.Tensor:
    """Autoregressively extend ``idx`` by up to ``max_new_tokens`` tokens.

    ``forced_tokens`` overrides sampling with a fixed script. It exists so the
    eval harness can be tested against a model with known behaviour instead of
    against whatever an untrained network happens to emit.
    """
    was_training = model.training
    model.eval()
    for i in range(max_new_tokens):
        cropped = idx[:, -model.config.block_size :]
        logits, _ = model(cropped)
        assert logits.shape[:2] == cropped.shape, logits.shape

        if forced_tokens is not None and i < len(forced_tokens):
            next_token = torch.full((idx.shape[0], 1), forced_tokens[i], dtype=torch.long,
                                    device=idx.device)
        else:
            next_token = sample_next(logits[:, -1, :], temperature, top_k, generator)

        idx = torch.cat([idx, next_token], dim=1)
        if stop_at_eos and eos_id is not None and (next_token == eos_id).all():
            break
    model.train(was_training)
    return idx


def stream_moves(
    model: NanoChessGPT,
    tokenizer,
    prompt: list[str] | None = None,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Iterator[str]:
    """Yield decoded moves one at a time, as they are generated.

    A generator rather than a list because the CLI prints moves as they appear;
    waiting for a whole game before showing anything makes a slow model feel
    broken.
    """
    device = device or next(model.parameters()).device
    ids = [tokenizer.bos_id] + (tokenizer.encode(prompt) if prompt else [])
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                cropped = idx[:, -model.config.block_size :]
                logits, _ = model(cropped)
                next_token = sample_next(logits[:, -1, :], temperature, top_k, generator)
                token_id = int(next_token.item())
                if token_id == tokenizer.eos_id:
                    return
                idx = torch.cat([idx, next_token], dim=1)
                decoded = tokenizer.decode([token_id])
                if decoded:  # skip pad/unk rather than emitting a bare "<unk>"
                    yield decoded[0]
    finally:
        model.train(was_training)
