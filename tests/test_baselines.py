import random

import chess

from nanochess.baselines import BigramSource, RandomLegalSource, UniformRandomTokenSource
from nanochess.evaluate import evaluate_source
from nanochess.tokenizer import MoveTokenizer

CORPUS = [
    ["e2e4", "e7e5", "g1f3"],
    ["e2e4", "e7e5", "b1c3"],
    ["e2e4", "c7c5", "g1f3"],
    ["d2d4", "d7d5"],
]


def test_random_legal_source_scores_exactly_one_hundred_percent():
    """The ceiling, and a self-check: if this is not 1.0, the harness is broken."""
    report = evaluate_source(RandomLegalSource(), games=10, max_plies=30, seed=0)
    assert report.legal_rate == 1.0


def test_random_legal_source_always_returns_a_move_legal_in_the_position():
    source = RandomLegalSource()
    board = chess.Board()
    rng = random.Random(0)
    for _ in range(20):
        uci = source.next_move([], board, rng)
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)


def test_uniform_random_token_source_is_close_to_useless():
    """The floor. ~30 legal moves out of 1968 tokens is about 1.5%."""
    tok = MoveTokenizer()
    report = evaluate_source(UniformRandomTokenSource(tok), games=20, max_plies=20, seed=0)
    assert report.legal_rate < 0.10


def test_uniform_random_token_source_only_emits_real_vocabulary_entries():
    tok = MoveTokenizer()
    source = UniformRandomTokenSource(tok)
    rng = random.Random(0)
    for _ in range(50):
        assert source.next_move([], chess.Board(), rng) in set(tok.moves)


def test_bigram_learns_the_most_common_continuation():
    model = BigramSource.fit(CORPUS)
    # e2e4 is followed by e7e5 twice and c7c5 once.
    assert model.most_likely("e2e4") == "e7e5"


def test_bigram_falls_back_to_unigram_for_an_unseen_context():
    model = BigramSource.fit(CORPUS)
    assert model.most_likely("h2h3") in model.unigram


def test_bigram_uses_the_opening_distribution_at_the_start_of_a_game():
    model = BigramSource.fit(CORPUS)
    rng = random.Random(0)
    first_moves = {model.next_move([], chess.Board(), rng) for _ in range(30)}
    assert first_moves <= {"e2e4", "d2d4"}


def test_bigram_conditions_on_the_previous_move_only():
    model = BigramSource.fit(CORPUS)
    rng = random.Random(1)
    seconds = {model.next_move(["e2e4"], chess.Board(), rng) for _ in range(40)}
    assert seconds <= {"e7e5", "c7c5"}


def test_bigram_beats_uniform_random_but_is_far_from_perfect():
    tok = MoveTokenizer()
    corpus = CORPUS * 10
    bigram = evaluate_source(BigramSource.fit(corpus), games=20, max_plies=20, seed=0)
    uniform = evaluate_source(UniformRandomTokenSource(tok), games=20, max_plies=20, seed=0)
    assert bigram.legal_rate > uniform.legal_rate
    assert bigram.legal_rate < 1.0


def test_bigram_is_deterministic_under_a_seed():
    model = BigramSource.fit(CORPUS)
    a = [model.next_move(["e2e4"], chess.Board(), random.Random(3)) for _ in range(5)]
    b = [model.next_move(["e2e4"], chess.Board(), random.Random(3)) for _ in range(5)]
    assert a == b


def test_every_source_reports_a_name_for_the_results_table():
    tok = MoveTokenizer()
    for source in (RandomLegalSource(), UniformRandomTokenSource(tok), BigramSource.fit(CORPUS)):
        assert isinstance(source.name, str) and source.name
