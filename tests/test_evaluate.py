import random

import chess
import pytest

from nanochess.evaluate import (
    EvalReport,
    GameWalk,
    evaluate_source,
    walk_move_source,
)


class ScriptedSource:
    """Plays a fixed list of moves, then falls back to a fixed filler move."""

    name = "scripted"

    def __init__(self, moves, filler="0000"):
        self.moves = list(moves)
        self.filler = filler
        self.calls = 0

    def next_move(self, history, board, rng):
        self.calls += 1
        return self.moves[len(history)] if len(history) < len(self.moves) else self.filler


LEGAL_OPENING = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]


def test_a_fully_legal_game_records_no_illegal_moves():
    walk = walk_move_source(ScriptedSource(LEGAL_OPENING), max_plies=6, rng=random.Random(0))
    assert walk.legal == 6
    assert walk.total == 6
    assert walk.first_illegal_ply is None


def test_legality_is_judged_against_the_real_board():
    """e2e4 twice: legal on ply 1, impossible on ply 3 since the pawn has moved."""
    source = ScriptedSource(["e2e4", "e7e5", "e2e4"])
    walk = walk_move_source(source, max_plies=3, rng=random.Random(0))
    assert walk.legal == 2
    assert walk.first_illegal_ply == 3


def test_an_unparseable_token_counts_as_illegal():
    walk = walk_move_source(ScriptedSource(["not-a-move"]), max_plies=1, rng=random.Random(0))
    assert walk.legal == 0
    assert walk.first_illegal_ply == 1


def test_walk_continues_past_an_illegal_move_so_later_plies_are_measured():
    """An illegal move must not end measurement, or the rate just re-encodes ply 1."""
    source = ScriptedSource(["e2e4", "e2e4", "e7e5", "g1f3"])
    walk = walk_move_source(source, max_plies=4, rng=random.Random(0))
    assert walk.total == 4
    assert walk.first_illegal_ply == 2
    assert walk.legal >= 1


def test_illegal_moves_are_replaced_by_a_legal_one_keeping_the_board_valid():
    source = ScriptedSource(["zzzz"] * 10)
    walk = walk_move_source(source, max_plies=10, rng=random.Random(0))
    assert walk.legal == 0
    assert walk.total == 10
    assert walk.board.is_valid()


def test_walk_stops_when_the_game_actually_ends():
    fools_mate = ["f2f3", "e7e5", "g2g4", "d8h4"]  # checkmate on ply 4
    walk = walk_move_source(ScriptedSource(fools_mate), max_plies=20, rng=random.Random(0))
    assert walk.total == 4
    assert walk.board.is_checkmate()


def test_per_ply_legality_is_recorded_for_bucketing():
    walk = walk_move_source(ScriptedSource(["e2e4", "e2e4"]), max_plies=2, rng=random.Random(0))
    assert walk.per_ply == [True, False]


def test_evaluate_source_aggregates_over_many_games():
    report = evaluate_source(ScriptedSource(LEGAL_OPENING), games=5, max_plies=6, seed=0)
    assert isinstance(report, EvalReport)
    assert report.games == 5
    assert report.legal_rate == 1.0
    assert report.total_moves == 30


def test_report_gives_mean_plies_before_the_first_illegal_move():
    report = evaluate_source(ScriptedSource(["e2e4", "e7e5", "zzzz"]), games=3, max_plies=5, seed=0)
    assert report.mean_plies_to_first_illegal == pytest.approx(2.0)


def test_fully_legal_source_reports_max_plies_for_first_illegal():
    """With no illegal move, 'plies survived' is the whole game, not zero."""
    report = evaluate_source(ScriptedSource(LEGAL_OPENING), games=2, max_plies=6, seed=0)
    assert report.mean_plies_to_first_illegal == 6.0


def test_report_buckets_legality_by_ply():
    report = evaluate_source(ScriptedSource(LEGAL_OPENING), games=2, max_plies=6, seed=0)
    assert report.legal_rate_by_ply[1] == 1.0
    assert set(report.legal_rate_by_ply) == set(range(1, 7))


def test_report_serialises_to_a_plain_dict():
    report = evaluate_source(ScriptedSource(LEGAL_OPENING), games=2, max_plies=4, seed=0)
    blob = report.to_dict()
    assert blob["legal_rate"] == 1.0
    assert blob["source"] == "scripted"
    assert "legal_rate_by_ply" in blob


def test_evaluation_is_reproducible_under_a_seed():
    a = evaluate_source(ScriptedSource(["zzzz"] * 8), games=4, max_plies=8, seed=7)
    b = evaluate_source(ScriptedSource(["zzzz"] * 8), games=4, max_plies=8, seed=7)
    assert a.to_dict() == b.to_dict()


def test_game_walk_exposes_the_move_list_actually_played():
    walk = walk_move_source(ScriptedSource(LEGAL_OPENING), max_plies=3, rng=random.Random(0))
    assert isinstance(walk, GameWalk)
    assert walk.played == LEGAL_OPENING[:3]
    board = chess.Board()
    for uci in walk.played:
        board.push(chess.Move.from_uci(uci))  # must replay cleanly


def test_bucketed_legality_groups_plies_into_ranges():
    report = evaluate_source(ScriptedSource(LEGAL_OPENING), games=4, max_plies=6, seed=0)
    buckets = report.bucketed_legal_rate(width=2)
    assert list(buckets) == ["1-2", "3-4", "5-6"]
    assert all(rate == 1.0 for rate in buckets.values())


def test_bucketing_shows_degradation_over_the_course_of_a_game():
    """Legal early, illegal later — the bucket table must make that visible."""
    source = ScriptedSource(["e2e4", "e7e5", "zzzz", "zzzz"])
    report = evaluate_source(source, games=4, max_plies=4, seed=0)
    buckets = report.bucketed_legal_rate(width=2)
    assert buckets["1-2"] == 1.0
    assert buckets["3-4"] == 0.0


def test_bucketing_handles_a_final_partial_bucket():
    report = evaluate_source(ScriptedSource(LEGAL_OPENING), games=2, max_plies=5, seed=0)
    assert list(report.bucketed_legal_rate(width=2)) == ["1-2", "3-4", "5-5"]
