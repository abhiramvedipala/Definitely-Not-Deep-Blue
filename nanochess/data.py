"""Corpus ingest: PGN in, packed token windows out."""

from __future__ import annotations

import io
import json
import random
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import chess.pgn
import torch

NORMAL_TERMINATIONS = {"Normal", "Time forfeit"}


def iter_pgn_games(stream: TextIO) -> Iterator[chess.pgn.Game]:
    """Yield parsed games from a PGN stream until it is exhausted."""
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        yield game


def game_to_ucis(game: chess.pgn.Game) -> list[str]:
    """Replay a game's mainline and return it as UCI move strings.

    Replaying rather than string-munging is deliberate: python-chess validates
    every move as it goes, so a game that survives this function is guaranteed
    legal, and the corpus cannot teach the model illegal moves.
    """
    return [move.uci() for move in game.mainline_moves()]


@dataclass
class GameFilter:
    """Corpus quality gate.

    The Elo floor is the meaningful knob. Lichess bulk dumps skew towards very
    weak blitz, and a corpus of blunders teaches a model that blunders are
    normal — which does not hurt legality directly, but does make the move
    distribution noisier than it needs to be for a model this small.
    """

    min_elo: int = 1500
    min_plies: int = 10
    max_plies: int = 300

    def headers_ok(self, headers) -> bool:
        if headers.get("Termination", "Normal") not in NORMAL_TERMINATIONS:
            return False
        for key in ("WhiteElo", "BlackElo"):
            raw = headers.get(key, "")
            if not raw.isdigit() or int(raw) < self.min_elo:
                return False
        return True

    def moves_ok(self, ucis: list[str]) -> bool:
        return len(ucis) >= self.min_plies

    def truncate(self, ucis: list[str]) -> list[str]:
        return ucis[: self.max_plies]


def extract_games(
    stream: TextIO, game_filter: GameFilter, limit: int | None = None
) -> Iterator[list[str]]:
    """Stream games out of a PGN file, keeping only those that pass the filter."""
    kept = 0
    for game in iter_pgn_games(stream):
        if limit is not None and kept >= limit:
            return
        if not game_filter.headers_ok(game.headers):
            continue
        try:
            ucis = game_to_ucis(game)
        except ValueError:
            continue  # malformed mainline; drop it rather than guess
        if not game_filter.moves_ok(ucis):
            continue
        kept += 1
        yield game_filter.truncate(ucis)


def pack_games(games: list[list[str]], tokenizer) -> torch.Tensor:
    """Flatten games into one token stream: ``<bos> m1 … mN <eos>`` per game.

    Packing wastes no tokens on padding, which matters when the whole budget is
    four hours. The delimiters are what let the model learn where a game starts,
    so sampling from ``<bos>`` produces an opening rather than a middlegame.
    """
    stream: list[int] = []
    for ucis in games:
        stream.extend(tokenizer.encode(ucis, add_special=True))
    return torch.tensor(stream, dtype=torch.int64)


def split_games(
    games: list[list[str]], val_fraction: float = 0.01, seed: int = 0
) -> tuple[list[list[str]], list[list[str]]]:
    """Split *by game*, before packing.

    Splitting after packing would cut games in half and put the first 30 plies
    in train and the rest in val, which leaks and makes validation loss a lie.
    """
    shuffled = list(games)
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of ``(x, y)`` windows, where ``y`` is ``x`` shifted one token."""
    high = len(data) - block_size - 1
    assert high > 0, f"data of length {len(data)} is too short for block_size {block_size}"
    idx = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in idx])
    assert x.shape == (batch_size, block_size), x.shape
    assert y.shape == (batch_size, block_size), y.shape
    return x.to(device), y.to(device)


def save_tokens(path: str | Path, tokens: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, path)


def load_tokens(path: str | Path) -> torch.Tensor:
    return torch.load(Path(path), weights_only=True)


LICHESS_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
)


@contextmanager
def open_pgn(path: str | Path):
    """Open a PGN file, decompressing zstandard transparently and incrementally.

    Incrementally matters: Lichess dumps are tens of gigabytes decompressed, so
    the reader has to stream. The smallest month is used by default precisely so
    this stays a laptop-scale operation.
    """
    path = Path(path)
    if path.suffix == ".zst":
        import zstandard

        with path.open("rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            yield fh


def download_corpus(url: str = LICHESS_URL, dest: str | Path = "data/games.pgn.zst") -> Path:
    """Download the corpus if it is not already on disk."""
    dest = Path(dest)
    if dest.exists():
        print(f"{dest} already present ({dest.stat().st_size / 1e6:.1f} MB), skipping download")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        total = int(response.headers.get("content-length", 0))
        seen = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            seen += len(chunk)
            if total:
                print(f"\r  {seen / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    return dest


def prepare_corpus(
    pgn_path: str | Path,
    out_dir: str | Path,
    tokenizer,
    game_filter: GameFilter,
    limit: int | None = None,
    val_fraction: float = 0.01,
    seed: int = 0,
) -> dict:
    """Turn a PGN file into packed train/val token streams plus metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open_pgn(pgn_path) as fh:
        games = list(extract_games(fh, game_filter, limit=limit))

    if not games:
        raise ValueError(f"no games survived the filter in {pgn_path}")

    train_games, val_games = split_games(games, val_fraction=val_fraction, seed=seed)
    train_tokens = pack_games(train_games, tokenizer)
    val_tokens = pack_games(val_games, tokenizer)

    save_tokens(out_dir / "train.bin", train_tokens)
    save_tokens(out_dir / "val.bin", val_tokens)
    tokenizer.save(out_dir / "tokenizer.json")

    # The bigram baseline is fit on moves, not token ids, so keep the raw games.
    (out_dir / "train_games.json").write_text(json.dumps(train_games), encoding="utf-8")

    stats = {
        "source": str(pgn_path),
        "games_kept": len(games),
        "train_games": len(train_games),
        "val_games": len(val_games),
        "train_tokens": int(train_tokens.numel()),
        "val_tokens": int(val_tokens.numel()),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": tokenizer.name,
        "filter": asdict(game_filter),
        "seed": seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
