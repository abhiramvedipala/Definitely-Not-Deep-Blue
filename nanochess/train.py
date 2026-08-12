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


@dataclass
class TrainState:
    step: int = 0
    best_val: float | None = None
    history: list[dict] = field(default_factory=list)


def configure_optimizer(model: NanoChessGPT, config: TrainConfig) -> torch.optim.AdamW:
    """Decay matrices, do not decay biases, LayerNorm gains or embeddings."""
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr, betas=config.betas)


@torch.no_grad()
def estimate_loss(
    model: NanoChessGPT,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    batches: int = 20,
) -> float:
    """Mean loss over a fixed number of random windows, with the RNG pinned.

    The generator is seeded from the batch index rather than left to drift, so
    two evaluations of the same weights give the same number and the val curve
    reflects the model rather than which windows it happened to draw.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(batches):
        generator = torch.Generator().manual_seed(10_000 + i)
        x, y = get_batch(data, batch_size, block_size, device, generator=generator)
        _, loss = model(x, targets=y)
        total += loss.item()
    model.train(was_training)
    return total / batches


def save_checkpoint(
    path: str | Path,
    model: NanoChessGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainConfig,
    best_val: float | None,
) -> None:
    """Persist everything needed to resume — including the model's own config.

    Storing the architecture alongside the weights means a checkpoint is
    self-describing: `sample` and `eval` rebuild the exact model that was
    trained instead of trusting a CLI flag to match.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
            "model_config": asdict(model.config),
            "train_config": {k: v for k, v in asdict(config).items() if k != "history"},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: NanoChessGPT,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> TrainState:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    if optimizer is not None and blob.get("optimizer"):
        optimizer.load_state_dict(blob["optimizer"])
    return TrainState(step=blob["step"], best_val=blob.get("best_val"))


def train(
    model: NanoChessGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: TrainConfig,
    out_dir: str | Path,
    state: TrainState | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    stop_after: int | None = None,
) -> TrainState:
    """Run the training loop, resumably.

    Batches are drawn from a generator seeded with ``seed + step``, not from a
    single long-lived RNG. That is what makes resume exact: step 4,000 sees the
    same batch whether it was reached in one run or three, so a resumed run is
    verifiably the same run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    model.to(device)
    model.train()

    optimizer = optimizer or configure_optimizer(model, config)
    state = state or TrainState()
    block_size = model.config.block_size
    started = time.time()
    steps_done = 0

    while state.step < config.max_steps:
        step = state.step
        lr = lr_at_step(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        generator = torch.Generator().manual_seed(config.seed + step)
        x, y = get_batch(train_data, config.batch_size, block_size, device, generator=generator)

        _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        entry = {"step": step, "loss": loss.item(), "lr": lr}
        state.step += 1
        steps_done += 1

        if config.eval_interval and (state.step % config.eval_interval == 0):
            val = estimate_loss(
                model, val_data, config.batch_size, block_size, device, config.eval_batches
            )
            entry["val_loss"] = val
            if state.best_val is None or val < state.best_val:
                state.best_val = val
            print(f"step {state.step:>6}  train {loss.item():.4f}  val {val:.4f}  lr {lr:.2e}")

        state.history.append(entry)

        if config.log_interval and (state.step % config.log_interval == 0):
            elapsed = time.time() - started
            rate = steps_done / max(elapsed, 1e-9)
            print(
                f"step {state.step:>6}/{config.max_steps}  loss {loss.item():.4f}  "
                f"lr {lr:.2e}  {rate:.2f} steps/s"
            )

        if config.checkpoint_interval and (state.step % config.checkpoint_interval == 0):
            save_checkpoint(out_dir / "ckpt.pt", model, optimizer, state.step, config, state.best_val)

        if stop_after is not None and steps_done >= stop_after:
            break
        if config.time_budget_s is not None and (time.time() - started) >= config.time_budget_s:
            print(f"time budget of {config.time_budget_s:.0f}s reached at step {state.step}")
            break

    save_checkpoint(out_dir / "ckpt.pt", model, optimizer, state.step, config, state.best_val)
    (out_dir / "history.json").write_text(json.dumps(state.history), encoding="utf-8")
    return state


def build_and_train(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    model_cfg: GPTConfig,
    train_cfg: TrainConfig,
    out_dir: str | Path,
    resume: bool = False,
) -> TrainState:
    """Entry point used by the CLI: construct, optionally resume, then train."""
    set_seed(train_cfg.seed)
    model = NanoChessGPT(model_cfg)
    device = resolve_device(train_cfg.device)
    model.to(device)
    print(f"model: {human_count(count_parameters(model))} parameters on {device}")

    optimizer = configure_optimizer(model, train_cfg)
    state = None
    ckpt = Path(out_dir) / "ckpt.pt"
    if resume and ckpt.exists():
        state = load_checkpoint(ckpt, model, optimizer, device)
        print(f"resumed from step {state.step}")

    return train(model, train_data, val_data, train_cfg, out_dir, state=state, optimizer=optimizer)
