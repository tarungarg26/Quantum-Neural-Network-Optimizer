"""Publication-grade Matplotlib style settings and deterministic export."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_SIZES = {
    "single": (3.4, 2.6),
    "double": (7.0, 3.5),
    "full_page": (7.0, 8.5),
}


def apply_style() -> None:
    """Set global rcParams for publication-quality figures."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 11.0,
        "lines.linewidth": 1.5,
        "lines.markersize": 5.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.constrained_layout.use": True,
        "svg.hashsalt": "qnnbench_seed_salt",
        "savefig.dpi": 300,
    })


def create_figure(size: Literal["single", "double", "full_page"] = "single") -> tuple[plt.Figure, plt.Axes]:
    """Create a new figure with standard publication dimensions."""
    apply_style()
    figsize = FIG_SIZES[size]
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig: plt.Figure, out_path: str | Path) -> None:
    """Save figure deterministically to SVG and PNG formats."""
    import re
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Save SVG (vector)
    svg_path = out.with_suffix(".svg")
    fig.savefig(svg_path, format="svg", metadata={"Creator": "qnnbench"})

    # Strip dynamic date tags to guarantee byte-level determinism (Section 5.1)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_text_stripped = re.sub(r"<dc:date>.*?</dc:date>", "<dc:date>1970-01-01T00:00:00</dc:date>", svg_text)
    svg_path.write_text(svg_text_stripped, encoding="utf-8")

    # Save PNG (raster at 300 DPI)
    png_path = out.with_suffix(".png")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
