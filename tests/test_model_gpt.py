import pytest
import torch

from nanochess.model import Block, GPTConfig, NanoChessGPT, model_config
from nanochess.utils import count_parameters, set_seed

CFG = GPTConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16)


def test_block_preserves_shape():
    assert Block(CFG)(torch.randn(2, 8, 16)).shape == (2, 8, 16)


def test_block_is_residual_so_output_differs_from_input_but_stays_finite():
    set_seed(0)
    x = torch.randn(2, 8, 16)
    out = Block(CFG)(x)
    assert not torch.allclose(out, x)
    assert torch.isfinite(out).all()


def test_forward_returns_logits_over_the_vocabulary():
    model = NanoChessGPT(CFG)
    logits, loss = model(torch.randint(0, 32, (4, 8)))
    assert logits.shape == (4, 8, 32)
    assert loss is None


def test_forward_with_targets_returns_a_scalar_loss():
    model = NanoChessGPT(CFG)
    idx = torch.randint(0, 32, (4, 8))
    logits, loss = model(idx, targets=idx)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_untrained_loss_is_near_uniform_entropy():
    """A fresh model should sit near ln(vocab_size); far off means broken init."""
    set_seed(0)
    model = NanoChessGPT(CFG)
    _, loss = model(torch.randint(0, 32, (16, 8)), targets=torch.randint(0, 32, (16, 8)))
    import math

    assert abs(loss.item() - math.log(32)) < 0.6


def test_output_head_shares_storage_with_the_token_embedding():
    model = NanoChessGPT(CFG)
    assert model.head.weight.data_ptr() == model.token_emb.weight.data_ptr()


def test_rejects_sequences_longer_than_block_size():
    model = NanoChessGPT(CFG)
    with pytest.raises(AssertionError):
        model(torch.randint(0, 32, (2, 9)))


def test_gradients_reach_every_parameter():
    model = NanoChessGPT(CFG)
    idx = torch.randint(0, 32, (2, 8))
    _, loss = model(idx, targets=idx)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"


def test_model_can_overfit_a_single_batch():
    """The end-to-end proof that forward, backward and the loss all agree."""
    set_seed(0)
    model = NanoChessGPT(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    idx = torch.randint(0, 32, (4, 8))
    first = None
    for _ in range(120):
        _, loss = model(idx, targets=idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.2


def test_tiny_preset_is_small_enough_for_tests():
    model = NanoChessGPT(model_config("tiny", vocab_size=1972))
    assert count_parameters(model) < 200_000


def test_small_preset_sits_in_the_specified_parameter_budget():
    model = NanoChessGPT(model_config("small", vocab_size=1972))
    n = count_parameters(model)
    assert 8_000_000 < n < 20_000_000, f"{n} parameters is outside the spec's budget"


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        model_config("enormous", vocab_size=1972)
