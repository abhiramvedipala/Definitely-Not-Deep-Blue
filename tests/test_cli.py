import json

import pytest

from nanochess.cli import build_parser, main


def test_parser_exposes_the_four_subcommands():
    parser = build_parser()
    for command in ("data", "train", "sample", "eval"):
        assert parser.parse_args([command]).command == command


def test_train_accepts_the_documented_knobs():
    args = build_parser().parse_args(
        ["train", "--preset", "small", "--max-steps", "100", "--batch-size", "8",
         "--lr", "1e-3", "--device", "cpu", "--seed", "7", "--hours", "0.5"]
    )
    assert args.preset == "small"
    assert args.max_steps == 100
    assert args.hours == 0.5
    assert args.seed == 7


def test_sample_accepts_temperature_and_top_k():
    args = build_parser().parse_args(["sample", "--temperature", "0.8", "--top-k", "20"])
    assert args.temperature == 0.8
    assert args.top_k == 20


def test_eval_accepts_game_and_ply_counts():
    args = build_parser().parse_args(["eval", "--games", "50", "--max-plies", "30"])
    assert args.games == 50
    assert args.max_plies == 30


def test_no_command_exits_with_an_error():
    with pytest.raises(SystemExit):
        main([])


def test_unknown_command_exits_with_an_error():
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_train_reports_a_missing_corpus_instead_of_crashing(tmp_path, capsys):
    code = main(["train", "--data-dir", str(tmp_path / "nope"), "--device", "cpu"])
    assert code != 0
    assert "prepare" in capsys.readouterr().out.lower()


def test_sample_reports_a_missing_checkpoint_instead_of_crashing(tmp_path, capsys):
    code = main(["sample", "--out-dir", str(tmp_path / "nope")])
    assert code != 0
    assert "checkpoint" in capsys.readouterr().out.lower()


def test_end_to_end_data_train_sample_eval(tmp_path, capsys):
    """The whole pipeline on a tiny model — the smoke test that ties it together."""
    import zstandard

    game = (
        '[Event "t"]\n[Result "1-0"]\n[WhiteElo "1800"]\n[BlackElo "1800"]\n'
        '[Termination "Normal"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0\n"
    )
    pgn = tmp_path / "games.pgn.zst"
    pgn.write_bytes(zstandard.ZstdCompressor().compress(("\n\n".join([game] * 200)).encode()))

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "run"

    assert main(["data", "--pgn", str(pgn), "--data-dir", str(data_dir), "--min-elo", "1500"]) == 0
    assert (data_dir / "train.bin").exists()

    assert main(["train", "--preset", "tiny", "--max-steps", "10", "--batch-size", "4",
                 "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--device", "cpu"]) == 0
    assert (out_dir / "ckpt.pt").exists()

    assert main(["sample", "--out-dir", str(out_dir), "--data-dir", str(data_dir),
                 "--max-moves", "5", "--device", "cpu"]) == 0

    assert main(["eval", "--out-dir", str(out_dir), "--data-dir", str(data_dir),
                 "--games", "3", "--max-plies", "6", "--device", "cpu"]) == 0
    report = json.loads((out_dir / "eval.json").read_text())
    assert "nanochess" in report
    assert 0.0 <= report["nanochess"]["legal_rate"] <= 1.0


def test_eval_reports_every_baseline_alongside_the_model(tmp_path):
    import zstandard

    game = (
        '[Event "t"]\n[Result "1-0"]\n[WhiteElo "1800"]\n[BlackElo "1800"]\n'
        '[Termination "Normal"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0\n'
    )
    pgn = tmp_path / "g.pgn.zst"
    pgn.write_bytes(zstandard.ZstdCompressor().compress(("\n\n".join([game] * 200)).encode()))
    data_dir, out_dir = tmp_path / "d", tmp_path / "r"

    main(["data", "--pgn", str(pgn), "--data-dir", str(data_dir), "--min-plies", "4"])
    main(["train", "--preset", "tiny", "--max-steps", "5", "--batch-size", "4",
          "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--device", "cpu"])
    main(["eval", "--out-dir", str(out_dir), "--data-dir", str(data_dir),
          "--games", "2", "--max-plies", "4", "--device", "cpu"])

    report = json.loads((out_dir / "eval.json").read_text())
    assert set(report) >= {"nanochess", "bigram", "uniform-random-token", "random-legal"}
    assert report["random-legal"]["legal_rate"] == 1.0
