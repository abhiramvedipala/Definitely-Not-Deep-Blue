"""Baselines, so the model's number means something.

A legal-move rate reported on its own is unreadable — is 88% good? These three
sources bracket the answer:

* ``uniform-random-token`` is the floor. It knows nothing, and scores about what
  chance allows: roughly 30 legal moves out of a 1968-token vocabulary.
* ``bigram`` is the bar. It is the cheapest thing that learns anything at all
  from the corpus, and a transformer that cannot beat it has not earned its
  parameters.
* ``random-legal`` is the ceiling, at 100% by construction. It is included
  partly as a reference point and partly as a self-check on the harness: if it
  ever scores below 1.0, the bug is in the eval, not the model.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import chess


class RandomLegalSource:
    """Plays a uniformly random legal move. Scores 100% by construction."""

    name = "random-legal"

    def next_move(self, history: list[str], board: chess.Board, rng: random.Random) -> str:
        return rng.choice(list(board.legal_moves)).uci()


class UniformRandomTokenSource:
    """Draws uniformly from the move vocabulary, ignoring the board entirely."""

    name = "uniform-random-token"

    def __init__(self, tokenizer) -> None:
        self.moves = list(tokenizer.moves)

    def next_move(self, history: list[str], board: chess.Board, rng: random.Random) -> str:
        return rng.choice(self.moves)


class BigramSource:
    """Count-based P(next move | previous move), backing off to unigram.

    Deliberately given the same interface as the model so the eval harness
    cannot accidentally treat them differently — the comparison is only fair if
    both are measured by exactly the same code path.
    """

    name = "bigram"

    def __init__(self, bigram: dict[str, Counter], unigram: Counter, openings: Counter) -> None:
        self.bigram = bigram
        self.unigram = unigram
        self.openings = openings

    @classmethod
    def fit(cls, games: list[list[str]]) -> BigramSource:
        bigram: dict[str, Counter] = defaultdict(Counter)
        unigram: Counter = Counter()
        openings: Counter = Counter()
        for moves in games:
            if moves:
                openings[moves[0]] += 1
            for i, move in enumerate(moves):
                unigram[move] += 1
                if i > 0:
                    bigram[moves[i - 1]][move] += 1
        return cls(dict(bigram), unigram, openings)

    def most_likely(self, previous: str) -> str:
        counts = self.bigram.get(previous) or self.unigram
        return counts.most_common(1)[0][0]

    def _sample(self, counts: Counter, rng: random.Random) -> str:
        population = list(counts)
        weights = [counts[m] for m in population]
        return rng.choices(population, weights=weights, k=1)[0]

    def next_move(self, history: list[str], board: chess.Board, rng: random.Random) -> str:
        if not history:
            return self._sample(self.openings or self.unigram, rng)
        counts = self.bigram.get(history[-1])
        return self._sample(counts or self.unigram, rng)
