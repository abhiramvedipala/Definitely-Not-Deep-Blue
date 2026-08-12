import chess
import pytest

from nanochess.tokenizer import BOS, EOS, PAD, UNK, MoveTokenizer, build_move_vocab


def test_vocab_has_expected_size():
    moves = build_move_vocab()
    assert len(moves) == 1968


def test_vocab_has_no_duplicates():
    moves = build_move_vocab()
    assert len(set(moves)) == len(moves)


def test_vocab_is_sorted_for_determinism():
    moves = build_move_vocab()
    assert moves == sorted(moves)


@pytest.mark.parametrize(
    "uci",
    [
        "e2e4",  # pawn double push
        "g1f3",  # knight
        "e1g1",  # white kingside castle
        "e1c1",  # white queenside castle
        "a1h8",  # long diagonal slide
        "e7e8q",  # promotion to queen
        "e7e8n",  # underpromotion
        "b2a1r",  # black promotion with capture geometry
    ],
)
def test_vocab_contains_known_moves(uci):
    assert uci in set(build_move_vocab())


def test_vocab_excludes_geometrically_impossible_moves():
    vocab = set(build_move_vocab())
    assert "e2e2" not in vocab  # null move
    assert "a1b4" not in vocab  # neither a queen line nor a knight leap
    assert "e2e5q" not in vocab  # promotion from the wrong rank


def test_every_legal_move_of_a_real_game_is_in_vocab():
    """The enumeration must cover every move a real board can actually produce."""
    vocab = set(build_move_vocab())
    board = chess.Board()
    for _ in range(40):
        legal = list(board.legal_moves)
        if not legal:
            break
        for move in legal:
            assert move.uci() in vocab, f"{move.uci()} missing from vocabulary"
        board.push(legal[0])


def test_specials_occupy_the_first_four_ids():
    tok = MoveTokenizer()
    assert tok.token_to_id[PAD] == 0
    assert tok.token_to_id[BOS] == 1
    assert tok.token_to_id[EOS] == 2
    assert tok.token_to_id[UNK] == 3


def test_vocab_size_is_specials_plus_moves():
    tok = MoveTokenizer()
    assert tok.vocab_size == 1972


def test_encode_decode_round_trip():
    tok = MoveTokenizer()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    assert tok.decode(tok.encode(moves)) == moves


def test_encode_adds_bos_and_eos_when_asked():
    tok = MoveTokenizer()
    ids = tok.encode(["e2e4"], add_special=True)
    assert ids[0] == tok.token_to_id[BOS]
    assert ids[-1] == tok.token_to_id[EOS]


def test_decode_strips_special_tokens():
    tok = MoveTokenizer()
    ids = tok.encode(["e2e4", "e7e5"], add_special=True)
    assert tok.decode(ids) == ["e2e4", "e7e5"]


def test_unknown_move_maps_to_unk():
    tok = MoveTokenizer()
    assert tok.encode(["zzzz"]) == [tok.token_to_id[UNK]]


def test_round_trip_over_a_full_real_game():
    tok = MoveTokenizer()
    board = chess.Board()
    ucis = []
    while not board.is_game_over() and len(ucis) < 120:
        move = list(board.legal_moves)[0]
        ucis.append(move.uci())
        board.push(move)
    assert tok.decode(tok.encode(ucis)) == ucis
