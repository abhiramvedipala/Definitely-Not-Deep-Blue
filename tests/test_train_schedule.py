import pytest

from nanochess.train import TrainConfig, lr_at_step


def cfg(**kw):
    return TrainConfig(**{"lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 100, "max_steps": 1000, **kw})


def test_lr_starts_near_zero():
    assert lr_at_step(0, cfg()) == pytest.approx(0.0, abs=1e-9)


def test_lr_reaches_the_peak_at_the_end_of_warmup():
    assert lr_at_step(100, cfg()) == pytest.approx(1e-3)


def test_warmup_is_linear():
    c = cfg()
    assert lr_at_step(50, c) == pytest.approx(0.5 * 1e-3)
    assert lr_at_step(25, c) == pytest.approx(0.25 * 1e-3)


def test_warmup_is_monotonically_increasing():
    c = cfg()
    values = [lr_at_step(s, c) for s in range(0, 101)]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))


def test_decay_is_monotonically_decreasing():
    c = cfg()
    values = [lr_at_step(s, c) for s in range(100, 1001, 10)]
    assert all(b <= a for a, b in zip(values, values[1:], strict=False))


def test_lr_lands_exactly_on_min_lr_at_max_steps():
    assert lr_at_step(1000, cfg()) == pytest.approx(1e-4)


def test_lr_is_clamped_to_min_lr_beyond_max_steps():
    assert lr_at_step(5000, cfg()) == pytest.approx(1e-4)


def test_cosine_midpoint_is_halfway_between_peak_and_floor():
    c = cfg()
    midpoint = lr_at_step(550, c)  # halfway through the 100..1000 decay
    assert midpoint == pytest.approx((1e-3 + 1e-4) / 2, rel=1e-6)


def test_zero_warmup_starts_at_the_peak():
    assert lr_at_step(0, cfg(warmup_steps=0)) == pytest.approx(1e-3)
