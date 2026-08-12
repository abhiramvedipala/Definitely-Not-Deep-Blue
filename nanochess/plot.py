"""Render the training curve as a standalone SVG.

Hand-rolled rather than matplotlib. The dependency list in the spec is torch,
python-chess and zstandard, and pulling a plotting stack in to draw two
polylines would not be a good trade. The output is a small text file, which is
also the only kind of image worth committing to a repository.
"""

from __future__ import annotations

W, H = 720, 320
PAD_L, PAD_R, PAD_T, PAD_B = 60, 20, 20, 44

STYLE = """
  <style>
    .bg { fill: #ffffff; }
    .axis { stroke: #b0b0b0; stroke-width: 1; }
    .grid { stroke: #e6e6e6; stroke-width: 1; }
    .label { fill: #444444; font: 12px system-ui, sans-serif; }
    .train { stroke: #2f6fdb; fill: none; stroke-width: 1.6; }
    .val { stroke: #d1495b; fill: none; stroke-width: 1.8; }
    @media (prefers-color-scheme: dark) {
      .bg { fill: #0d1117; }
      .axis { stroke: #555c66; }
      .grid { stroke: #21262d; }
      .label { fill: #b9c0c8; }
      .train { stroke: #6ea8ff; }
      .val { stroke: #ff7b8a; }
    }
  </style>
"""


def _downsample(points: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    stride = len(points) / max_points
    return [points[int(i * stride)] for i in range(max_points)]


def render_curve_svg(history: list[dict], max_points: int = 400) -> str:
    """Draw train and validation loss against step."""
    if not history:
        raise ValueError("cannot plot an empty history")

    train = [(float(e["step"]), float(e["loss"])) for e in history if "loss" in e]
    val = [(float(e["step"]), float(e["val_loss"])) for e in history if "val_loss" in e]
    train = _downsample(train, max_points)

    xs = [x for x, _ in train + val]
    ys = [y for _, y in train + val]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # A perfectly flat curve would otherwise divide by zero.
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0
    y_lo, y_hi = y_min - 0.05 * y_span, y_max + 0.05 * y_span

    def project(x: float, y: float) -> tuple[float, float]:
        px = PAD_L + (x - x_min) / x_span * (W - PAD_L - PAD_R)
        py = PAD_T + (y_hi - y) / (y_hi - y_lo) * (H - PAD_T - PAD_B)
        return px, py

    def polyline(points, cls) -> str:
        coords = " ".join(f"{px:.1f},{py:.1f}" for px, py in (project(x, y) for x, y in points))
        return f'  <polyline class="{cls}" points="{coords}" />'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        STYLE,
        f'  <rect class="bg" width="{W}" height="{H}" />',
    ]

    for i in range(5):
        y = y_hi - i * (y_hi - y_lo) / 4
        _, py = project(x_min, y)
        parts.append(f'  <line class="grid" x1="{PAD_L}" y1="{py:.1f}" x2="{W - PAD_R}" y2="{py:.1f}" />')
        parts.append(f'  <text class="label" x="{PAD_L - 8}" y="{py + 4:.1f}" text-anchor="end">{y:.2f}</text>')

    for i in range(5):
        x = x_min + i * x_span / 4
        px, _ = project(x, y_lo)
        parts.append(
            f'  <text class="label" x="{px:.1f}" y="{H - PAD_B + 18}" text-anchor="middle">{int(x)}</text>'
        )

    parts.append(
        f'  <line class="axis" x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}" />'
    )
    parts.append(f'  <line class="axis" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}" />')

    parts.append(polyline(train, "train"))
    if val:
        parts.append(polyline(val, "val"))

    parts.append(
        f'  <text class="label" x="{W / 2}" y="{H - 6}" text-anchor="middle">step</text>'
    )
    parts.append(
        f'  <text class="label" x="14" y="{H / 2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {H / 2})">cross-entropy loss</text>'
    )
    parts.append(f'  <text class="label" x="{W - PAD_R - 130}" y="{PAD_T + 14}">train</text>')
    if val:
        parts.append(f'  <text class="label" x="{W - PAD_R - 70}" y="{PAD_T + 14}">validation</text>')
        parts.append(
            f'  <line class="val" x1="{W - PAD_R - 90}" y1="{PAD_T + 10}" '
            f'x2="{W - PAD_R - 76}" y2="{PAD_T + 10}" />'
        )
    parts.append(
        f'  <line class="train" x1="{W - PAD_R - 150}" y1="{PAD_T + 10}" '
        f'x2="{W - PAD_R - 136}" y2="{PAD_T + 10}" />'
    )

    parts.append("</svg>")
    return "\n".join(parts)
