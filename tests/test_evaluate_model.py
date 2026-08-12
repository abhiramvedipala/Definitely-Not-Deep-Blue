import random

import chess
import torch

from nanochess.evaluate import ModelMoveSource, evaluate_model, evaluate_source
from nanochess.model import NanoChessGPT, model_config
from nanochess.tokenizer import MoveTokenizer
from nanochess.utils import set_seed


def tiny():
    tok = MoveTokenizer()
    model = NanoChessGPT(model_config("tiny", vocab_size=tok.vocab_size))
    return model, tok


def test_model_source_returns_a_string_from_the_move_vocabulary():
    set_seed(0)
    model, tok = tiny()
    source = ModelMoveSource(model, tok)
    move = source.next_move([], chess.Board(), random.Random(0))
    assert isinstance(move, str)


def test_model_source_conditions_on_the_history_it_is_given():
    """A longer history must reach the model as a longer prompt."""
    set_seed(0)
    model, tok = tiny()
    source = ModelMoveSource(model, tok)
    source.next_move(["e2e4", "e7e5"], chess.Board(), random.Random(0))
    assert source.last_prompt_length == 3  # <bos> + two moves


def test_model_source_crops_history_to_the_block_size():
    set_seed(0)
    model, tok = tiny()  # tiny block_size is 32
    source = ModelMoveSource(model, tok)
    long_history = ["e2e4"] * 50
    source.next_move(long_history, chess.Board(), random.Random(0))
    assert source.last_prompt_length <= model.config.block_size


def test_special_tokens_are_surfaced_as_non_moves_not_hidden():
    """Emitting <eos> when asked for a move is a failure, and is counted as one."""
    set_seed(0)
    model, tok = tiny()
    source = ModelMoveSource(model, tok, forced_token=tok.eos_id)
    move = source.next_move([], chess.Board(), random.Random(0))
    assert chess.Move.from_uci(move) not in chess.Board().legal_moves


def test_model_source_is_deterministic_at_zero_temperature():
    set_seed(0)
    model, tok = tiny()
    source = ModelMoveSource(model, tok, temperature=0.0)
    moves = [source.next_move(["e2e4"], chess.Board(), random.Random(i)) for i in range(5)]
    assert len(set(moves)) == 1


def test_untrained_model_scores_poorly_which_is_the_point_of_the_metric():
    set_seed(0)
    model, tok = tiny()
    report = evaluate_source(ModelMoveSource(model, tok), games=5, max_plies=10, seed=0)
    assert report.legal_rate < 0.5


def test_model_source_carries_a_name_for_the_results_table():
    model, tok = tiny()
    assert ModelMoveSource(model, tok).name == "nanochess"


def test_evaluate_model_attaches_validation_loss_to_the_report():
    set_seed(0)
    model, tok = tiny()
    data = torch.randint(0, tok.vocab_size, (2000,))
    report = evaluate_model(model, tok, val_data=data, games=3, max_plies=6, seed=0,
                            device=torch.device("cpu"))
    assert report.val_loss is not None and report.val_loss > 0
    assert report.source == "nanochess"


def test_evaluate_model_works_without_validation_data():
    set_seed(0)
    model, tok = tiny()
    report = evaluate_model(model, tok, val_data=None, games=2, max_plies=4, seed=0,
                            device=torch.device("cpu"))
    assert report.val_loss is None
