"""Corpus ingest: PGN in, packed token windows out."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TextIO

import chess.pgn

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
