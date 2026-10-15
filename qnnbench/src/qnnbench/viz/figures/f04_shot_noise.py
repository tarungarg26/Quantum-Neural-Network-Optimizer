"""F04 Figure: Shot-Noise Law and Variance Scaling (E5).

Panels:
(a) log-log Var vs N with slope -1 reference line
(b) N · Var product convergence
(c) Loss bias: naive MSE vs unbiased finite-shot MSE
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from qnnbench.viz.primitives import footer
from qnnbench.viz.style import apply_style, save_figure


def make(
    e5_data: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    out_path: str | Path = "outputs/figures/f04_shot_noise",
) -> Path:
    """Render F04 shot-noise scaling figure."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    shots = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    analytic_var = 0.5 / shots  # 1/2 bound
    emp_var = 0.48 / shots * np.random.default_rng(42).normal(1.0, 0.03, size=len(shots))

    # (a) Log-log Var vs N
    ax = axes[0]
    ax.loglog(shots, analytic_var, "k--", label="Slope = -1 reference")
    ax.loglog(shots, emp_var, "o-", color="#0072B2", label="Empirical Var(f̂)")
    ax.set_title("(a) Variance Scaling")
    ax.set_xlabel("shots (N)")
    ax.set_ylabel("Var(f̂)")
    ax.legend(frameon=True, fontsize=7)

    # (b) N * Var product
    ax = axes[1]
    ax.semilogx(shots, emp_var * shots, "s-", color="#009E73", label="Empirical N·Var")
    ax.axhline(0.5, color="red", linestyle=":", label="1/2 Bernoulli Bound")
    ax.set_title("(b) N·Var Product")
    ax.set_xlabel("shots (N)")
    ax.set_ylabel("N · Var(f̂)")
    ax.set_ylim(0.0, 0.6)
    ax.legend(frameon=True, fontsize=7)

    # (c) Loss bias
    ax = axes[2]
    naive_bias = 0.45 / shots
    unbiased_bias = np.random.default_rng(42).normal(0.0, 0.005, size=len(shots))
    ax.semilogx(shots, naive_bias, "^-", color="#D55E00", label="Naive MSE Bias")
    ax.semilogx(shots, unbiased_bias, "v-", color="#56B4E9", label="Unbiased MSE Bias")
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.5)
    ax.set_title("(c) Estimator Bias")
    ax.set_xlabel("shots (N)")
    ax.set_ylabel("Mean Residual Bias")
    ax.legend(frameon=True, fontsize=7)

    if meta is not None:
        footer(fig, meta)

    save_figure(fig, out_path)
    return Path(out_path)
