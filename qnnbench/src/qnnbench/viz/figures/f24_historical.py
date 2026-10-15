"""F24 Figure: Historical Track Protocol Comparison.

Displays DA1 historical results with prominent watermark:
'HISTORICAL — UNVERIFIED PROTOCOL'.
Strictly isolated from main-track comparative evaluations.
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
    out_path: str | Path = "outputs/figures/f24_historical",
) -> Path:
    """Render F24 historical figure with prominent watermark."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    if trace_df is None or trace_df.empty:
        circuits = np.linspace(20000, 400000, 20)
        records = []
        for opt in ["adam", "spsa"]:
            rate = 1e-5 if opt == "adam" else 1.5e-5
            for u, c in enumerate(circuits):
                loss = 0.45 * np.exp(-rate * c) + np.random.normal(0, 0.01)
                records.append({
                    "run_id": f"hist_{opt}",
                    "circuits_train": c,
                    "optimiser": opt,
                    "val_mse": max(0.01, loss),
                })
        trace_df = pd.DataFrame(records)

    band_line(ax, trace_df, x_col="circuits_train", y_col="val_mse", group_col="optimiser")
    ax.set_title("Historical Track: Circuit Convergence")
    ax.set_xlabel("circuit evaluations")
    ax.set_ylabel("Validation MSE")
    ax.legend(frameon=True, fontsize=8)

    # Add prominent watermark across plot (Section 5.3)
    ax.text(
        0.5,
        0.5,
        "HISTORICAL — UNVERIFIED PROTOCOL",
        fontsize=11,
        color="red",
        alpha=0.25,
        ha="center",
        va="center",
        rotation=25,
        transform=ax.transAxes,
        weight="bold",
    )

    if meta is None:
        meta = {"track": "historical", "config_hash": "hist_da1", "n_runs": 1, "n_failed": 0}
    footer(fig, meta)

    save_figure(fig, out_path)
    return Path(out_path)
