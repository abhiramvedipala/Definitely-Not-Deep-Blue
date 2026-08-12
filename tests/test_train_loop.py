import torch

from nanochess.model import NanoChessGPT, model_config
from nanochess.train import (
    TrainConfig,
    configure_optimizer,
    estimate_loss,
    load_checkpoint,
    save_checkpoint,
    train,
)
from nanochess.utils import set_seed

VOCAB = 64


def tiny_model():
    return NanoChessGPT(model_config("tiny", vocab_size=VOCAB))


def fake_data(n=4000):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, VOCAB, (n,), generator=g)


def test_checkpoint_round_trips_model_optimizer_and_step(tmp_path):
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = TrainConfig(max_steps=10)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, opt, step=7, config=cfg, best_val=1.23)
    fresh = tiny_model()
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    state = load_checkpoint(path, fresh, fresh_opt, device=torch.device("cpu"))

    assert state.step == 7
    assert state.best_val == 1.23
    for a, b in zip(model.parameters(), fresh.parameters(), strict=True):
        assert torch.equal(a, b)


def test_checkpoint_stores_the_model_config_so_it_can_be_rebuilt(tmp_path):
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, step=1, config=TrainConfig(), best_val=None)

    blob = torch.load(path, weights_only=False)
    assert blob["model_config"]["n_layer"] == 2
    assert blob["model_config"]["vocab_size"] == VOCAB


def test_estimate_loss_returns_a_finite_positive_scalar():
    model = tiny_model()
    loss = estimate_loss(model, fake_data(), batch_size=4, block_size=32,
                         device=torch.device("cpu"), batches=3)
    assert loss > 0 and torch.isfinite(torch.tensor(loss))


def test_estimate_loss_does_not_leave_the_model_in_eval_mode():
    model = tiny_model()
    model.train()
    estimate_loss(model, fake_data(), 4, 32, torch.device("cpu"), batches=2)
    assert model.training


def test_training_reduces_loss_on_a_learnable_stream(tmp_path):
    """A repeating pattern is trivially learnable; failing this means the loop is broken."""
    set_seed(0)
    data = torch.arange(VOCAB).repeat(200)  # perfectly predictable
    cfg = TrainConfig(max_steps=200, warmup_steps=10, batch_size=8, lr=3e-3,
                      eval_interval=1000, checkpoint_interval=1000, log_interval=1000,
                      device="cpu")
    state = train(tiny_model(), data, data, cfg, out_dir=tmp_path)
    # 200 steps, not 60: a 16-dim model needs a little room before the drop is
    # unambiguous, and a threshold the loop can miss by noise is not a test.
    assert state.history[-1]["loss"] < state.history[0]["loss"] * 0.5


def test_training_writes_a_checkpoint(tmp_path):
    set_seed(0)
    data = fake_data()
    cfg = TrainConfig(max_steps=10, warmup_steps=2, batch_size=4, eval_interval=5,
                      checkpoint_interval=5, log_interval=100, device="cpu")
    train(tiny_model(), data, data, cfg, out_dir=tmp_path)
    assert (tmp_path / "ckpt.pt").exists()


def test_training_writes_a_history_file_for_the_loss_curve(tmp_path):
    set_seed(0)
    data = fake_data()
    cfg = TrainConfig(max_steps=10, warmup_steps=2, batch_size=4, eval_interval=5,
                      checkpoint_interval=100, log_interval=100, device="cpu")
    train(tiny_model(), data, data, cfg, out_dir=tmp_path)
    assert (tmp_path / "history.json").exists()


def test_resume_continues_from_the_saved_step(tmp_path):
    set_seed(0)
    data = fake_data()
    cfg = TrainConfig(max_steps=10, warmup_steps=2, batch_size=4, eval_interval=100,
                      checkpoint_interval=10, log_interval=100, device="cpu")
    train(tiny_model(), data, data, cfg, out_dir=tmp_path)

    model = tiny_model()
    opt = configure_optimizer(model, cfg)
    state = load_checkpoint(tmp_path / "ckpt.pt", model, opt, device=torch.device("cpu"))
    assert state.step == 10


def test_split_run_matches_a_continuous_run(tmp_path):
    """Train 20 steps, versus train 10 + resume 10. The weights must agree."""
    data = fake_data()
    common = dict(warmup_steps=2, batch_size=4, lr=1e-3, eval_interval=1000,
                  log_interval=1000, device="cpu", seed=99)

    set_seed(99)
    straight = tiny_model()
    train(straight, data, data, TrainConfig(max_steps=20, checkpoint_interval=1000, **common),
          out_dir=tmp_path / "a")

    set_seed(99)
    split = tiny_model()
    train(split, data, data, TrainConfig(max_steps=20, checkpoint_interval=10, **common),
          out_dir=tmp_path / "b", stop_after=10)
    opt = configure_optimizer(split, TrainConfig(max_steps=20, **common))
    state = load_checkpoint(tmp_path / "b" / "ckpt.pt", split, opt, device=torch.device("cpu"))
    train(split, data, data, TrainConfig(max_steps=20, checkpoint_interval=1000, **common),
          out_dir=tmp_path / "b", state=state, optimizer=opt)

    for a, b in zip(straight.parameters(), split.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6), "resume diverged from a continuous run"


def test_time_budget_stops_training_early(tmp_path):
    cfg = TrainConfig(max_steps=10_000_000, warmup_steps=2, batch_size=4,
                      eval_interval=10**9, checkpoint_interval=10**9, log_interval=10**9,
                      device="cpu", time_budget_s=0.5)
    state = train(tiny_model(), fake_data(), fake_data(), cfg, out_dir=tmp_path)
    assert state.step < 10_000_000
