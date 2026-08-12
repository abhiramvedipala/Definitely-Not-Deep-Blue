"""The eval harness — the reason this project exists.

Validation loss is reported because it is cheap, but it is not the headline.
The headline is: **what fraction of the moves this model generates are actually
legal in the position it is generating them for?** python-chess is the judge,
so the answer is a fact about chess rather than a fact about our code.

The one design decision worth defending: when the model emits an illegal move,
the walk does not stop. It records the failure, substitutes a random legal move,
and keeps going. Stopping instead would make the legal-move rate a restatement
of "how early did it first fail", which is already measured separately as
plies-to-first-illegal. Continuing keeps the two numbers independent, and it is
the only way to learn whether the model degrades at ply 30 or falls apart at
ply 5.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import chess


@dataclass
class GameWalk:
    """The record of one generated game, judged ply by ply."""

    per_ply: list[bool] = field(default_factory=list)
    played: list[str] = field(default_factory=list)
    board: chess.Board = field(default_factory=chess.Board)

    @property
    def total(self) -> int:
        return len(self.per_ply)

    @property
    def legal(self) -> int:
        return sum(self.per_ply)

    @property
    def first_illegal_ply(self) -> int | None:
        for i, ok in enumerate(self.per_ply, start=1):
            if not ok:
                return i
        return None


def walk_move_source(source, max_plies: int, rng: random.Random) -> GameWalk:
    """Play up to ``max_plies`` moves from ``source``, judging each against the board."""
    walk = GameWalk()
    board = walk.board

    for _ in range(max_plies):
        if board.is_game_over():
            break

        uci = source.next_move(list(walk.played), board, rng)
        try:
            move = chess.Move.from_uci(uci)
        except (ValueError, chess.InvalidMoveError):
            move = None

        legal = move is not None and move in board.legal_moves
        walk.per_ply.append(legal)

        if legal:
            walk.played.append(uci)
            board.push(move)
        else:
            # Substitute so the game continues and later plies stay measurable.
            fallback = rng.choice(list(board.legal_moves))
            walk.played.append(fallback.uci())
            board.push(fallback)

    return walk


@dataclass
class EvalReport:
    source: str
    games: int
    max_plies: int
    total_moves: int
    legal_moves: int
    legal_rate: float
    mean_plies_to_first_illegal: float
    legal_rate_by_ply: dict[int, float]
    val_loss: float | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "games": self.games,
            "max_plies": self.max_plies,
            "total_moves": self.total_moves,
            "legal_moves": self.legal_moves,
            "legal_rate": self.legal_rate,
            "mean_plies_to_first_illegal": self.mean_plies_to_first_illegal,
            "legal_rate_by_ply": {str(k): v for k, v in sorted(self.legal_rate_by_ply.items())},
            "val_loss": self.val_loss,
        }

    def summary(self) -> str:
        return (
            f"{self.source:<22} legal {self.legal_rate * 100:6.2f}%   "
            f"first illegal at ply {self.mean_plies_to_first_illegal:6.2f}   "
            f"({self.legal_moves}/{self.total_moves} moves over {self.games} games)"
        )


def evaluate_source(source, games: int, max_plies: int, seed: int = 0) -> EvalReport:
    """Sample ``games`` continuations from ``source`` and aggregate the legality metrics."""
    rng = random.Random(seed)
    walks = [walk_move_source(source, max_plies, rng) for _ in range(games)]

    total = sum(w.total for w in walks)
    legal = sum(w.legal for w in walks)

    # A game with no illegal move survived its whole length; crediting it 0
    # would punish the best possible behaviour.
    survivals = [
        (w.first_illegal_ply - 1) if w.first_illegal_ply is not None else w.total for w in walks
    ]

    by_ply: dict[int, float] = {}
    for ply in range(1, max_plies + 1):
        judged = [w.per_ply[ply - 1] for w in walks if w.total >= ply]
        if judged:
            by_ply[ply] = sum(judged) / len(judged)

    return EvalReport(
        source=getattr(source, "name", type(source).__name__),
        games=games,
        max_plies=max_plies,
        total_moves=total,
        legal_moves=legal,
        legal_rate=legal / total if total else 0.0,
        mean_plies_to_first_illegal=sum(survivals) / len(survivals) if survivals else 0.0,
        legal_rate_by_ply=by_ply,
    )
