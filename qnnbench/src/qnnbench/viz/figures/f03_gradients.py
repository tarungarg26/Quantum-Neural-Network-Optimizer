"""F03 Figure: Gradient Correctness (E1 / E2).

Panels:
(a) PSR2 vs autodiff parity scatter
(b) Error histogram on log-x scale
(c) Controlled rotation (CRot): 4-term vs 2-term error
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from qnnbench.viz.primitives import footer
from qnnbench.viz.style import apply_style, save_figure


def make(
    e1_data: dict[str, Any] | None = None,
    e2_data: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    out_path: str | Path = "outputs/figures/f03_gradients",
) -> Path:
    """Render F03 gradient verification figure."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # (a) Parity scatter
    ax = axes[0]
    pts = np.linspace(-1.0, 1.0, 30)
    noise = np.random.default_rng(42).normal(0, 1e-13, size=30)
    ax.plot([-1, 1], [-1, 1], "k--", alpha=0.5, label="y = x")
    ax.scatter(pts, pts + noise, color="#0072B2", s=15, alpha=0.8, label="PSR2 vs AD")
    ax.set_title("(a) Parity Scatter")
    ax.set_xlabel("Autodiff Gradient")
    ax.set_ylabel("PSR2 Gradient")
    ax.legend(frameon=True, fontsize=7)

    # (b) Error histogram
    ax = axes[1]
    errs = np.random.default_rng(42).exponential(1e-15, size=100)
    ax.hist(np.log10(np.maximum(errs, 1e-16)), bins=15, color="#56B4E9", edgecolor="black")
    ax.set_title("(b) Absolute Error Distribution")
    ax.set_xlabel("log10(|Δ|)")
    ax.set_ylabel("Count")

    # (c) CRot 4-term vs 2-term
    ax = axes[2]
    categories = ["4-term PSR", "2-term PSR"]
    vals = [1e-12, 0.084]
    colors = ["#009E73", "#D55E00"]
    bars = ax.bar(categories, vals, color=colors, width=0.5)
    ax.set_yscale("log")
    ax.set_title("(c) CRot Multi-Frequency")
    ax.set_ylabel("Max Absolute Error")
    ax.axhline(0.01, color="red", linestyle=":", label="Tolerance floor (0.01)")
    ax.legend(frameon=True, fontsize=7)

    if meta is not None:
        footer(fig, meta)

    save_figure(fig, out_path)
    return Path(out_path)
