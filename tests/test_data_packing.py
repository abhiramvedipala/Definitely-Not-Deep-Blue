import torch

from nanochess.data import get_batch, load_tokens, pack_games, save_tokens, split_games
from nanochess.tokenizer import MoveTokenizer

GAMES = [
    ["e2e4", "e7e5", "g1f3"],
    ["d2d4", "d7d5"],
]


def test_pack_games_returns_a_flat_int64_tensor():
    tok = MoveTokenizer()
    packed = pack_games(GAMES, tok)
    assert packed.dtype == torch.int64
    assert packed.ndim == 1


def test_pack_games_length_is_moves_plus_two_specials_per_game():
    tok = MoveTokenizer()
    packed = pack_games(GAMES, tok)
    assert len(packed) == (3 + 2) + (2 + 2)


def test_pack_games_delimits_each_game_with_bos_and_eos():
    tok = MoveTokenizer()
    packed = pack_games(GAMES, tok).tolist()
    assert packed[0] == tok.bos_id
    assert packed[4] == tok.eos_id
    assert packed[5] == tok.bos_id
    assert packed[-1] == tok.eos_id


def test_packed_stream_decodes_back_to_the_original_games():
    tok = MoveTokenizer()
    packed = pack_games(GAMES, tok).tolist()
    assert tok.decode(packed) == GAMES[0] + GAMES[1]


def test_split_games_splits_by_game_not_by_token():
    games = [[f"m{i}"] for i in range(100)]
    train, val = split_games(games, val_fraction=0.1, seed=0)
    assert len(train) == 90
    assert len(val) == 10
    flat_train = {tuple(g) for g in train}
    flat_val = {tuple(g) for g in val}
    assert flat_train.isdisjoint(flat_val)


def test_split_is_deterministic_under_a_seed():
    games = [[f"m{i}"] for i in range(50)]
    a = split_games(games, val_fraction=0.2, seed=42)
    b = split_games(games, val_fraction=0.2, seed=42)
    assert a == b


def test_split_shuffles_so_val_is_not_just_the_tail():
    games = [[f"m{i}"] for i in range(100)]
    _, val = split_games(games, val_fraction=0.1, seed=3)
    assert val != [[f"m{i}"] for i in range(90, 100)]


def test_get_batch_returns_expected_shapes():
    data = torch.arange(1000)
    x, y = get_batch(data, batch_size=4, block_size=16, device=torch.device("cpu"))
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


def test_get_batch_targets_are_inputs_shifted_by_one():
    data = torch.arange(1000)
    x, y = get_batch(data, batch_size=4, block_size=16, device=torch.device("cpu"))
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_get_batch_is_deterministic_given_a_generator():
    data = torch.arange(1000)
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    x1, _ = get_batch(data, 4, 16, torch.device("cpu"), generator=g1)
    x2, _ = get_batch(data, 4, 16, torch.device("cpu"), generator=g2)
    assert torch.equal(x1, x2)


def test_get_batch_never_reads_past_the_end_of_the_data():
    data = torch.arange(20)
    for _ in range(50):
        x, y = get_batch(data, batch_size=2, block_size=8, device=torch.device("cpu"))
        assert x.max() < 20 and y.max() < 20


def test_tokens_survive_a_save_load_round_trip(tmp_path):
    data = torch.randint(0, 100, (500,))
    path = tmp_path / "train.bin"
    save_tokens(path, data)
    assert torch.equal(load_tokens(path), data)
