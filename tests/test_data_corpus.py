import json

import zstandard

from nanochess.data import GameFilter, load_tokens, open_pgn, prepare_corpus
from nanochess.tokenizer import MoveTokenizer

GAME = """[Event "Rated game"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1800"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0
"""


def write_pgn(path, n_games):
    path.write_text("\n\n".join([GAME] * n_games), encoding="utf-8")
    return path


def write_zst(path, n_games):
    raw = "\n\n".join([GAME] * n_games).encode("utf-8")
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))
    return path


def test_open_pgn_reads_a_plain_text_file(tmp_path):
    path = write_pgn(tmp_path / "games.pgn", 3)
    with open_pgn(path) as fh:
        assert "1. e4 e5" in fh.read()


def test_open_pgn_transparently_decompresses_zstandard(tmp_path):
    path = write_zst(tmp_path / "games.pgn.zst", 3)
    with open_pgn(path) as fh:
        assert "1. e4 e5" in fh.read()


def test_open_pgn_streams_rather_than_loading_everything(tmp_path):
    """Lichess dumps do not fit in memory; the reader must be incremental."""
    path = write_zst(tmp_path / "games.pgn.zst", 50)
    with open_pgn(path) as fh:
        assert len(fh.readline()) < 200  # a single header line, not the whole file


def test_prepare_corpus_writes_train_val_tokenizer_and_metadata(tmp_path):
    src = write_zst(tmp_path / "games.pgn.zst", 40)
    out = tmp_path / "prepared"
    prepare_corpus(src, out, MoveTokenizer(), GameFilter(min_elo=1500, min_plies=4))

    for name in ("train.bin", "val.bin", "tokenizer.json", "meta.json"):
        assert (out / name).exists(), f"{name} was not written"


def test_prepared_metadata_records_how_the_corpus_was_built(tmp_path):
    src = write_zst(tmp_path / "games.pgn.zst", 40)
    out = tmp_path / "prepared"
    prepare_corpus(src, out, MoveTokenizer(), GameFilter(min_elo=1500, min_plies=4))

    meta = json.loads((out / "meta.json").read_text())
    assert meta["games_kept"] > 0
    assert meta["vocab_size"] == 1972
    assert meta["train_tokens"] > meta["val_tokens"]
    assert "filter" in meta and meta["filter"]["min_elo"] == 1500


def test_prepared_tokens_are_loadable_and_non_empty(tmp_path):
    src = write_zst(tmp_path / "games.pgn.zst", 40)
    out = tmp_path / "prepared"
    prepare_corpus(src, out, MoveTokenizer(), GameFilter(min_elo=1500, min_plies=4))

    train = load_tokens(out / "train.bin")
    assert train.numel() > 0
    assert train.max().item() < 1972


def test_prepare_corpus_honours_the_game_limit(tmp_path):
    src = write_zst(tmp_path / "games.pgn.zst", 100)
    out = tmp_path / "prepared"
    stats = prepare_corpus(src, out, MoveTokenizer(), GameFilter(min_elo=1500, min_plies=4),
                           limit=10)
    assert stats["games_kept"] == 10


def test_prepare_corpus_keeps_the_raw_games_for_the_bigram_baseline(tmp_path):
    """The bigram baseline needs moves, not token ids, so they are saved too."""
    src = write_zst(tmp_path / "games.pgn.zst", 20)
    out = tmp_path / "prepared"
    prepare_corpus(src, out, MoveTokenizer(), GameFilter(min_elo=1500, min_plies=4))

    games = json.loads((out / "train_games.json").read_text())
    assert isinstance(games, list) and games and games[0][0] == "e2e4"


def test_a_game_never_appears_in_both_splits(tmp_path):
    src = write_zst(tmp_path / "games.pgn.zst", 60)
    out = tmp_path / "prepared"
    prepare_corpus(src, out, MoveTokenizer(), GameFilter(min_elo=1500, min_plies=4),
                   val_fraction=0.25)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["train_games"] + meta["val_games"] == meta["games_kept"]
