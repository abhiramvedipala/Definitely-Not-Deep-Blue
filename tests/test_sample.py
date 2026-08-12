import pytest
import torch

from nanochess.model import NanoChessGPT, model_config
from nanochess.sample import apply_top_k, generate, sample_next, stream_moves
from nanochess.tokenizer import MoveTokenizer
from nanochess.utils import set_seed


def tiny():
    tok = MoveTokenizer()
    return NanoChessGPT(model_config("tiny", vocab_size=tok.vocab_size)), tok


def test_top_k_keeps_exactly_k_finite_logits():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    out = apply_top_k(logits, k=2)
    assert torch.isfinite(out).sum() == 2


def test_top_k_keeps_the_largest_logits():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    out = apply_top_k(logits, k=2)[0]
    assert torch.isfinite(out[1]) and torch.isfinite(out[4])
    assert torch.isneginf(out[0])


def test_top_k_of_none_is_a_no_op():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    assert torch.equal(apply_top_k(logits, k=None), logits)


def test_top_k_larger_than_vocab_is_a_no_op():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    assert torch.equal(apply_top_k(logits, k=99), logits)


def test_zero_temperature_is_greedy():
    logits = torch.tensor([[1.0, 9.0, 3.0]])
    for _ in range(10):
        assert sample_next(logits, temperature=0.0).item() == 1


def test_high_temperature_produces_variety():
    set_seed(0)
    logits = torch.tensor([[1.0, 1.1, 1.05]])
    draws = {sample_next(logits, temperature=2.0).item() for _ in range(50)}
    assert len(draws) > 1


def test_sampling_is_deterministic_under_a_generator():
    logits = torch.randn(1, 50)
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    a = [sample_next(logits, 1.0, generator=g1).item() for _ in range(20)]
    b = [sample_next(logits, 1.0, generator=g2).item() for _ in range(20)]
    assert a == b


def test_negative_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature"):
        sample_next(torch.randn(1, 5), temperature=-1.0)


def test_generate_extends_the_prompt_by_max_new_tokens():
    model, tok = tiny()
    out = generate(model, torch.tensor([[tok.bos_id]]), max_new_tokens=7, temperature=1.0)
    assert out.shape == (1, 8)


def test_generate_preserves_the_prompt_prefix():
    model, tok = tiny()
    prompt = torch.tensor([[tok.bos_id, 10, 11]])
    out = generate(model, prompt, max_new_tokens=4)
    assert torch.equal(out[:, :3], prompt)


def test_generate_crops_context_to_the_block_size():
    """Prompts longer than the window must not blow up the model's assertion."""
    model, tok = tiny()
    long_prompt = torch.randint(4, tok.vocab_size, (1, 40))  # tiny block_size is 32
    out = generate(model, long_prompt, max_new_tokens=3)
    assert out.shape == (1, 43)


def test_generate_stops_at_eos_when_asked():
    model, tok = tiny()
    out = generate(model, torch.tensor([[tok.bos_id]]), max_new_tokens=5,
                   stop_at_eos=True, eos_id=tok.eos_id, forced_tokens=[tok.eos_id])
    assert out[0, -1].item() == tok.eos_id
    assert out.shape[1] == 2


def test_stream_moves_yields_decoded_moves_one_at_a_time():
    model, tok = tiny()
    set_seed(0)
    stream = stream_moves(model, tok, max_new_tokens=5, temperature=1.0)
    moves = list(stream)
    assert len(moves) <= 5
    assert all(isinstance(m, str) for m in moves)


def test_stream_moves_is_lazy_and_does_not_precompute_everything():
    model, tok = tiny()
    stream = stream_moves(model, tok, max_new_tokens=1000, temperature=1.0)
    first = next(stream)  # must return without generating all 1000
    assert isinstance(first, str)
    stream.close()
