import pytest

from nanochess.plot import render_curve_svg

HISTORY = [
    {"step": 0, "loss": 7.6, "lr": 0.0},
    {"step": 1, "loss": 6.2, "lr": 1e-4},
    {"step": 2, "loss": 5.1, "lr": 2e-4, "val_loss": 5.3},
    {"step": 3, "loss": 4.4, "lr": 3e-4},
    {"step": 4, "loss": 3.9, "lr": 3e-4, "val_loss": 4.1},
]


def test_renders_a_standalone_svg_document():
    svg = render_curve_svg(HISTORY)
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")


def test_plots_both_the_train_and_validation_series():
    svg = render_curve_svg(HISTORY)
    assert svg.count("<polyline") == 2


def test_omits_the_validation_series_when_there_is_none():
    svg = render_curve_svg([{"step": i, "loss": 5.0 - i * 0.1} for i in range(5)])
    assert svg.count("<polyline") == 1


def test_labels_the_axes_with_real_values():
    svg = render_curve_svg(HISTORY)
    assert "step" in svg.lower()
    assert "loss" in svg.lower()


def test_is_readable_in_both_colour_schemes():
    """Rendered on GitHub in dark mode too, so it cannot rely on a white page."""
    svg = render_curve_svg(HISTORY)
    assert "prefers-color-scheme" in svg


def test_empty_history_is_rejected_rather_than_producing_a_blank_chart():
    with pytest.raises(ValueError, match="empty"):
        render_curve_svg([])


def test_a_flat_curve_does_not_divide_by_zero():
    flat = [{"step": i, "loss": 4.0} for i in range(5)]
    assert "<polyline" in render_curve_svg(flat)


def test_downsamples_a_very_long_history():
    """A 5000-step run must not embed 5000 points into the repo."""
    long_history = [{"step": i, "loss": 8.0 - i / 1000} for i in range(5000)]
    svg = render_curve_svg(long_history, max_points=200)
    train_points = svg.split('<polyline')[1].split('"')[3]
    assert len(train_points.split()) <= 200


def test_training_writes_a_curve_alongside_the_history(tmp_path):
    import torch

    from nanochess.model import NanoChessGPT, model_config
    from nanochess.train import TrainConfig, train

    data = torch.randint(0, 64, (2000,))
    cfg = TrainConfig(max_steps=6, warmup_steps=1, batch_size=4, eval_interval=3,
                      checkpoint_interval=100, log_interval=100, device="cpu")
    train(NanoChessGPT(model_config("tiny", vocab_size=64)), data, data, cfg, out_dir=tmp_path)

    assert (tmp_path / "curve.svg").exists()
    assert (tmp_path / "curve.svg").read_text().startswith("<svg")
