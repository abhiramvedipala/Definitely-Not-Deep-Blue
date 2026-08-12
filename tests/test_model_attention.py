import pytest
import torch

from nanochess.model import MLP, CausalSelfAttention, GPTConfig
from nanochess.utils import set_seed

CFG = GPTConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16)


def test_attention_preserves_shape():
    attn = CausalSelfAttention(CFG)
    x = torch.randn(3, 8, 16)
    assert attn(x).shape == (3, 8, 16)


def test_attention_accepts_sequences_shorter_than_block_size():
    attn = CausalSelfAttention(CFG)
    assert attn(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_attention_rejects_sequences_longer_than_block_size():
    attn = CausalSelfAttention(CFG)
    with pytest.raises(AssertionError):
        attn(torch.randn(2, 9, 16))


def test_attention_rejects_wrong_embedding_width():
    attn = CausalSelfAttention(CFG)
    with pytest.raises(AssertionError):
        attn(torch.randn(2, 4, 12))


def test_attention_is_causal_future_tokens_cannot_affect_the_past():
    """The load-bearing test: perturbing token t must leave outputs 0..t-1 alone."""
    set_seed(0)
    attn = CausalSelfAttention(CFG).eval()
    x = torch.randn(1, 8, 16)
    perturbed = x.clone()
    perturbed[0, 5:] = torch.randn(3, 16)

    with torch.no_grad():
        a, b = attn(x), attn(perturbed)

    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-6), "information leaked backwards"
    assert not torch.allclose(a[0, 5:], b[0, 5:], atol=1e-6), "perturbation had no effect"


def test_n_embd_must_divide_evenly_by_n_head():
    with pytest.raises(AssertionError):
        CausalSelfAttention(GPTConfig(vocab_size=32, block_size=8, n_head=3, n_embd=16))


def test_mlp_preserves_shape_and_expands_four_fold():
    mlp = MLP(CFG)
    assert mlp(torch.randn(2, 8, 16)).shape == (2, 8, 16)
    assert mlp.fc.weight.shape == (4 * 16, 16)


def test_attention_gradients_reach_every_parameter():
    attn = CausalSelfAttention(CFG)
    attn(torch.randn(2, 8, 16)).sum().backward()
    for name, p in attn.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradients"
