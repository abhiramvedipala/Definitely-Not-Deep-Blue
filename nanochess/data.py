"""Corpus ingest: PGN in, packed token windows out."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
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
