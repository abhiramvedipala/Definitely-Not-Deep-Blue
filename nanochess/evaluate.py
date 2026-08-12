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

    def bucketed_legal_rate(self, width: int = 10) -> dict[str, float]:
        """Legality grouped into ply ranges.

        Forty per-ply numbers are data; four bucket numbers are a finding. This
        is the view that answers the question the aggregate rate hides — does
        the model hold up through the opening and then lose the thread, or is
        it unreliable from move one?
        """
        buckets: dict[str, float] = {}
        for start in range(1, self.max_plies + 1, width):
            end = min(start + width - 1, self.max_plies)
            rates = [self.legal_rate_by_ply[p] for p in range(start, end + 1)
                     if p in self.legal_rate_by_ply]
            if rates:
                buckets[f"{start}-{end}"] = sum(rates) / len(rates)
        return buckets

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


class ModelMoveSource:
    """Adapts the transformer to the move-source interface the harness expects.

    One judgement call is encoded here: if the model emits a special token
    (``<eos>``, ``<pad>``, ``<unk>``) when asked for a move, that is surfaced as
    the null move ``0000``, which is never legal, and therefore counts against
    the model. Re-sampling until a real move appeared would quietly inflate the
    headline number — we asked for a move and did not get one, and the metric
    should say so.
    """

    name = "nanochess"

    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 1.0,
        top_k: int | None = None,
        device=None,
        forced_token: int | None = None,
    ) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_k = top_k
        self.device = device or next(model.parameters()).device
        self.forced_token = forced_token
        self.last_prompt_length = 0
        self._torch = torch
        model.eval()

    def next_move(self, history: list[str], board: chess.Board, rng: random.Random) -> str:
        from .sample import sample_next

        torch = self._torch
        ids = [self.tokenizer.bos_id, *self.tokenizer.encode(history)]
        ids = ids[-self.model.config.block_size :]
        self.last_prompt_length = len(ids)

        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(idx)

        if self.forced_token is not None:
            token_id = self.forced_token
        else:
            # The harness owns randomness, so seed torch from its RNG to keep
            # the whole evaluation reproducible from a single seed.
            generator = torch.Generator(device="cpu").manual_seed(rng.randrange(2**31))
            token_id = int(
                sample_next(
                    logits[:, -1, :].cpu(), self.temperature, self.top_k, generator
                ).item()
            )

        decoded = self.tokenizer.decode([token_id])
        return decoded[0] if decoded else "0000"


def evaluate_model(
    model,
    tokenizer,
    val_data=None,
    games: int = 100,
    max_plies: int = 40,
    seed: int = 0,
    temperature: float = 1.0,
    top_k: int | None = None,
    device=None,
    batch_size: int = 16,
    eval_batches: int = 20,
) -> EvalReport:
    """Run the full model evaluation: legality metrics, plus validation loss."""
    source = ModelMoveSource(model, tokenizer, temperature, top_k, device)
    report = evaluate_source(source, games=games, max_plies=max_plies, seed=seed)

    if val_data is not None:
        from .train import estimate_loss

        report.val_loss = estimate_loss(
            model,
            val_data,
            batch_size,
            model.config.block_size,
            device or next(model.parameters()).device,
            batches=eval_batches,
        )
    return report
