import torch

from nanochess.utils import count_parameters, human_count, resolve_device, set_seed


def test_same_seed_gives_identical_tensors():
    set_seed(1234)
    a = torch.randn(8, 8)
    set_seed(1234)
    b = torch.randn(8, 8)
    assert torch.equal(a, b)


def test_different_seeds_diverge():
    set_seed(1)
    a = torch.randn(8, 8)
    set_seed(2)
    b = torch.randn(8, 8)
    assert not torch.equal(a, b)


def test_set_seed_also_seeds_python_random():
    import random

    set_seed(7)
    a = [random.random() for _ in range(5)]
    set_seed(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_resolve_device_honours_explicit_choice():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_returns_an_available_device():
    device = resolve_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}
    torch.zeros(2, device=device)  # must actually be usable


def test_count_parameters_counts_only_trainable():
    layer = torch.nn.Linear(4, 5)  # 4*5 weights + 5 biases
    assert count_parameters(layer) == 25
    layer.bias.requires_grad_(False)
    assert count_parameters(layer) == 20


def test_human_count_formats_millions():
    assert human_count(11_400_000) == "11.4M"
    assert human_count(43_000) == "43.0K"
    assert human_count(512) == "512"
