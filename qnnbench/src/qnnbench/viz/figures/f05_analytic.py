"""F05 Figure: Analytic Optimiser Comparison (E3).

Plots loss and accuracy trajectories vs optimization updates across all 5 optimizers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qnnbench.viz.primitives import band_line, footer
from qnnbench.viz.style import apply_style, save_figure


def make(
    trace_df: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
    out_path: str | Path = "outputs/figures/f05_analytic",
) -> Path:
    """Render F05 analytic comparison figure."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    if trace_df is None or trace_df.empty:
        # Synthetic mock data matching contract
        records = []
        for opt in ["gd", "momentum", "adam", "spsa", "qng"]:
            rate = {"adam": 0.05, "qng": 0.04, "momentum": 0.06, "gd": 0.08, "spsa": 0.09}[opt]
            for run_i in range(5):
                for u in range(0, 80, 5):
                    loss = 0.5 * np.exp(-rate * u) + np.random.normal(0, 0.01)
                    acc = min(1.0, 0.5 + 0.45 * (1 - np.exp(-rate * u)))
                    records.append({
                        "run_id": f"{opt}_{run_i}",
                        "optimiser": opt,
                        "update": u,
                        "val_mse": max(0.01, loss),
                        "val_acc": acc,
                    })
        trace_df = pd.DataFrame(records)

    # (a) Loss vs updates
    ax = axes[0]
    band_line(ax, trace_df, x_col="update", y_col="val_mse", group_col="optimiser")
    ax.set_title("(a) Validation Loss")
    ax.set_xlabel("updates")
    ax.set_ylabel("Validation MSE")
    ax.legend(frameon=True, fontsize=7)

    # (b) Accuracy vs updates
    ax = axes[1]
    band_line(ax, trace_df, x_col="update", y_col="val_acc", group_col="optimiser")
    ax.set_title("(b) Classification Accuracy")
    ax.set_xlabel("updates")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=True, fontsize=7)

    if meta is not None:
        footer(fig, meta)

    save_figure(fig, out_path)
    return Path(out_path)
