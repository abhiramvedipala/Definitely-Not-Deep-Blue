import pytest

from nanochess.tokenizer import CharTokenizer, MoveTokenizer, load_tokenizer


def test_move_tokenizer_save_load_round_trip(tmp_path):
    tok = MoveTokenizer()
    path = tmp_path / "tok.json"
    tok.save(path)
    restored = MoveTokenizer.load(path)
    assert restored.token_to_id == tok.token_to_id
    assert restored.id_to_token == tok.id_to_token
    assert restored.vocab_size == tok.vocab_size


def test_loaded_tokenizer_encodes_identically(tmp_path):
    tok = MoveTokenizer()
    path = tmp_path / "tok.json"
    tok.save(path)
    restored = MoveTokenizer.load(path)
    moves = ["e2e4", "e7e5", "g1f3"]
    assert restored.encode(moves) == tok.encode(moves)


def test_save_creates_parent_directories(tmp_path):
    tok = MoveTokenizer()
    tok.save(tmp_path / "nested" / "deeper" / "tok.json")
    assert (tmp_path / "nested" / "deeper" / "tok.json").exists()


def test_loading_wrong_tokenizer_type_raises(tmp_path):
    path = tmp_path / "tok.json"
    CharTokenizer().save(path)
    with pytest.raises(ValueError, match="not a move tokenizer"):
        MoveTokenizer.load(path)


def test_char_tokenizer_round_trips_moves():
    tok = CharTokenizer()
    moves = ["e2e4", "e7e5", "g1f3", "e7e8q"]
    assert tok.decode(tok.encode(moves)) == moves


def test_char_tokenizer_uses_far_fewer_tokens_in_vocab_but_more_per_game():
    move_tok, char_tok = MoveTokenizer(), CharTokenizer()
    moves = ["e2e4", "e7e5", "g1f3"]
    assert char_tok.vocab_size < move_tok.vocab_size
    assert len(char_tok.encode(moves)) > len(move_tok.encode(moves))


def test_char_tokenizer_save_load_round_trip(tmp_path):
    tok = CharTokenizer()
    path = tmp_path / "char.json"
    tok.save(path)
    restored = CharTokenizer.load(path)
    assert restored.encode(["e2e4"]) == tok.encode(["e2e4"])


@pytest.mark.parametrize("cls", [MoveTokenizer, CharTokenizer])
def test_both_tokenizers_satisfy_the_same_protocol(cls, tmp_path):
    tok = cls()
    for attr in ("vocab_size", "bos_id", "eos_id", "unk_id", "name"):
        assert hasattr(tok, attr)
    moves = ["e2e4", "e7e5"]
    ids = tok.encode(moves, add_special=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == moves


@pytest.mark.parametrize("cls", [MoveTokenizer, CharTokenizer])
def test_load_tokenizer_dispatches_on_saved_name(cls, tmp_path):
    path = tmp_path / "t.json"
    cls().save(path)
    assert isinstance(load_tokenizer(path), cls)
