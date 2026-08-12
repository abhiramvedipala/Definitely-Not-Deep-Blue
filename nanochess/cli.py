"""Command line interface: ``data``, ``train``, ``sample``, ``eval``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .baselines import BigramSource, RandomLegalSource, UniformRandomTokenSource
from .data import GameFilter, download_corpus, load_tokens, prepare_corpus
from .evaluate import evaluate_model, evaluate_source
from .model import NanoChessGPT, model_config
from .sample import stream_moves
from .tokenizer import MoveTokenizer, load_tokenizer
from .train import TrainConfig, build_and_train
from .utils import resolve_device, set_seed

DEFAULT_DATA_DIR = "data/prepared"
DEFAULT_OUT_DIR = "runs/nanochess"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanochess", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    data = sub.add_parser("data", help="download and prepare the corpus")
    data.add_argument("--pgn", default=None, help="local PGN (.pgn or .pgn.zst); downloads if unset")
    data.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    data.add_argument("--limit", type=int, default=None, help="max games to keep")
    data.add_argument("--min-elo", type=int, default=1500)
    data.add_argument("--min-plies", type=int, default=10)
    data.add_argument("--max-plies", type=int, default=300)
    data.add_argument("--val-fraction", type=float, default=0.01)
    data.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train", help="train a model")
    train.add_argument("--preset", default="small", choices=["tiny", "small"])
    train.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    train.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    train.add_argument("--max-steps", type=int, default=6000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--warmup-steps", type=int, default=200)
    train.add_argument("--dropout", type=float, default=0.0,
                       help="regularisation; worth raising when the run does many epochs")
    train.add_argument("--eval-interval", type=int, default=250)
    train.add_argument("--checkpoint-interval", type=int, default=500)
    train.add_argument("--hours", type=float, default=None, help="wall-clock budget")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=1337)

    sample = sub.add_parser("sample", help="generate a game")
    sample.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    sample.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    sample.add_argument("--prompt", default="", help="space-separated UCI moves to condition on")
    sample.add_argument("--max-moves", type=int, default=60)
    sample.add_argument("--temperature", type=float, default=1.0)
    sample.add_argument("--top-k", type=int, default=None)
    sample.add_argument("--device", default="auto")
    sample.add_argument("--seed", type=int, default=0)

    ev = sub.add_parser("eval", help="measure legal-move rate against the baselines")
    ev.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ev.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ev.add_argument("--games", type=int, default=100)
    ev.add_argument("--max-plies", type=int, default=40)
    ev.add_argument("--temperature", type=float, default=1.0)
    ev.add_argument("--top-k", type=int, default=None)
    ev.add_argument("--device", default="auto")
    ev.add_argument("--seed", type=int, default=0)

    return parser


def _load_model(out_dir: Path, device: torch.device):
    """Rebuild a model from a checkpoint's own stored config."""
    ckpt_path = out_dir / "ckpt.pt"
    if not ckpt_path.exists():
        print(f"no checkpoint at {ckpt_path}; run `nanochess train` first")
        return None, None
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    from .model import GPTConfig

    model = NanoChessGPT(GPTConfig(**blob["model_config"]))
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model, blob


def cmd_data(args) -> int:
    pgn = Path(args.pgn) if args.pgn else download_corpus()
    game_filter = GameFilter(
        min_elo=args.min_elo, min_plies=args.min_plies, max_plies=args.max_plies
    )
    stats = prepare_corpus(
        pgn, args.data_dir, MoveTokenizer(), game_filter,
        limit=args.limit, val_fraction=args.val_fraction, seed=args.seed,
    )
    print(json.dumps(stats, indent=2))
    return 0


def cmd_train(args) -> int:
    data_dir = Path(args.data_dir)
    if not (data_dir / "train.bin").exists():
        print(f"no corpus at {data_dir}; run `nanochess data` to prepare one first")
        return 1

    tokenizer = load_tokenizer(data_dir / "tokenizer.json")
    train_data = load_tokens(data_dir / "train.bin")
    val_data = load_tokens(data_dir / "val.bin")

    model_cfg = model_config(args.preset, vocab_size=tokenizer.vocab_size, dropout=args.dropout)
    train_cfg = TrainConfig(
        lr=args.lr, min_lr=args.lr / 10, warmup_steps=args.warmup_steps,
        max_steps=args.max_steps, batch_size=args.batch_size,
        eval_interval=args.eval_interval, checkpoint_interval=args.checkpoint_interval,
        seed=args.seed, device=args.device,
        time_budget_s=args.hours * 3600 if args.hours else None,
    )
    print(f"train tokens: {train_data.numel():,}   val tokens: {val_data.numel():,}")
    state = build_and_train(train_data, val_data, model_cfg, train_cfg, args.out_dir,
                            resume=args.resume)
    print(f"finished at step {state.step}, best val loss {state.best_val}")
    return 0


def cmd_sample(args) -> int:
    device = resolve_device(args.device)
    set_seed(args.seed)
    model, _ = _load_model(Path(args.out_dir), device)
    if model is None:
        return 1

    tokenizer_path = Path(args.data_dir) / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else MoveTokenizer()
    prompt = args.prompt.split() if args.prompt else None

    generator = torch.Generator().manual_seed(args.seed)
    for i, move in enumerate(
        stream_moves(model, tokenizer, prompt=prompt, max_new_tokens=args.max_moves,
                     temperature=args.temperature, top_k=args.top_k,
                     generator=generator, device=device)
    ):
        end = "\n" if i % 2 else " "
        print(move, end=end, flush=True)
    print()
    return 0


def cmd_eval(args) -> int:
    device = resolve_device(args.device)
    set_seed(args.seed)
    model, blob = _load_model(Path(args.out_dir), device)
    if model is None:
        return 1

    data_dir = Path(args.data_dir)
    tokenizer = load_tokenizer(data_dir / "tokenizer.json")
    val_path = data_dir / "val.bin"
    val_data = load_tokens(val_path) if val_path.exists() else None
    if val_data is not None and val_data.numel() <= model.config.block_size + 1:
        val_data = None  # too small to sample a window from; skip rather than crash

    reports = {}
    common = dict(games=args.games, max_plies=args.max_plies, seed=args.seed)

    model_report = evaluate_model(
        model, tokenizer, val_data=val_data, temperature=args.temperature,
        top_k=args.top_k, device=device, **common,
    )
    reports[model_report.source] = model_report

    games_path = data_dir / "train_games.json"
    if games_path.exists():
        corpus = json.loads(games_path.read_text())
        reports["bigram"] = evaluate_source(BigramSource.fit(corpus), **common)
    reports["uniform-random-token"] = evaluate_source(UniformRandomTokenSource(tokenizer), **common)
    reports["random-legal"] = evaluate_source(RandomLegalSource(), **common)

    print(f"\ncheckpoint step {blob['step']}   {args.games} games x {args.max_plies} plies\n")
    for report in reports.values():
        print(report.summary())
    if model_report.val_loss is not None:
        print(f"\nvalidation loss: {model_report.val_loss:.4f}")

    out_path = Path(args.out_dir) / "eval.json"
    out_path.write_text(
        json.dumps({k: v.to_dict() for k, v in reports.items()}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out_path}")
    return 0


COMMANDS = {"data": cmd_data, "train": cmd_train, "sample": cmd_sample, "eval": cmd_eval}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        raise SystemExit(2)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
