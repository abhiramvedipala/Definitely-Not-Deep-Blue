import io

from nanochess.data import GameFilter, extract_games, game_to_ucis, iter_pgn_games

SCHOLARS_MATE = """[Event "Rated Classical game"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1750"]
[Termination "Normal"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""

LOW_RATED = """[Event "Rated Blitz game"]
[Result "0-1"]
[WhiteElo "1100"]
[BlackElo "1050"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 0-1
"""

ABORTED = """[Event "Rated Blitz game"]
[Result "*"]
[WhiteElo "1900"]
[BlackElo "1900"]
[Termination "Abandoned"]

1. e4 *
"""

UNRATED = """[Event "Casual game"]
[Result "1-0"]
[Termination "Normal"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 1-0
"""


def _stream(*games):
    return io.StringIO("\n\n".join(games))


def test_iter_pgn_games_yields_each_game():
    games = list(iter_pgn_games(_stream(SCHOLARS_MATE, LOW_RATED)))
    assert len(games) == 2


def test_game_to_ucis_converts_san_to_uci():
    game = next(iter_pgn_games(_stream(SCHOLARS_MATE)))
    assert game_to_ucis(game) == ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]


def test_game_to_ucis_expands_castling_to_king_move():
    game = next(iter_pgn_games(_stream(LOW_RATED)))
    assert "e1g1" in game_to_ucis(game)


def test_filter_rejects_games_below_the_elo_floor():
    f = GameFilter(min_elo=1500, min_plies=4)
    assert f.headers_ok(next(iter_pgn_games(_stream(SCHOLARS_MATE))).headers)
    assert not f.headers_ok(next(iter_pgn_games(_stream(LOW_RATED))).headers)


def test_filter_rejects_games_with_missing_elo():
    f = GameFilter(min_elo=1500, min_plies=4)
    assert not f.headers_ok(next(iter_pgn_games(_stream(UNRATED))).headers)


def test_filter_rejects_abnormal_terminations():
    f = GameFilter(min_elo=1500, min_plies=1)
    assert not f.headers_ok(next(iter_pgn_games(_stream(ABORTED))).headers)


def test_filter_rejects_games_that_are_too_short():
    f = GameFilter(min_elo=1500, min_plies=20)
    assert not f.moves_ok(["e2e4", "e7e5"])
    assert f.moves_ok(["e2e4"] * 20)


def test_filter_truncates_overlong_games():
    f = GameFilter(max_plies=5)
    assert f.truncate(["e2e4"] * 10) == ["e2e4"] * 5


def test_extract_games_applies_every_filter():
    stream = _stream(SCHOLARS_MATE, LOW_RATED, ABORTED, UNRATED)
    games = list(extract_games(stream, GameFilter(min_elo=1500, min_plies=4)))
    assert games == [["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]]


def test_extract_games_honours_the_limit():
    stream = _stream(SCHOLARS_MATE, SCHOLARS_MATE, SCHOLARS_MATE)
    games = list(extract_games(stream, GameFilter(min_elo=1500, min_plies=4), limit=2))
    assert len(games) == 2


def test_extracted_moves_are_all_replayable_on_a_real_board():
    import chess

    stream = _stream(SCHOLARS_MATE, LOW_RATED)
    for ucis in extract_games(stream, GameFilter(min_elo=0, min_plies=1)):
        board = chess.Board()
        for uci in ucis:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            board.push(move)
