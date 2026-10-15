"""Reusable publication-grade visualization primitives (M16).

Implements:
- band_line: mean/median with 95% shaded confidence interval
- spaghetti: thin background individual run traces
- rank_heatmap: 2D matrix of optimizer rankings
- forest: point estimates with horizontal confidence intervals
- footer: standard metadata footer annotation
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qnnbench.viz.palette import get_style


def footer(fig: plt.Figure, meta: dict[str, Any]) -> None:
    """Add unobtrusive metadata footer to figure.

    Format: 'track | config_hash[:8] | n_runs | n_failed'
    """
    track = meta.get("track", "main")
    config_hash = str(meta.get("config_hash", "00000000"))[:8]
    n_runs = meta.get("n_runs", 0)
    n_failed = meta.get("n_failed", 0)

    label = f"{track} | hash:{config_hash} | runs:{n_runs} | failed:{n_failed}"
    fig.text(
        0.98,
        0.01,
        label,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#777777",
        family="sans-serif",
    )


def band_line(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    show_spaghetti: bool = True,
) -> None:
    """Plot mean trajectory with shaded 95% confidence interval per group."""
    groups = df[group_col].unique()

    for grp in groups:
        sub = df[df[group_col] == grp]
        style = get_style(grp)

        # Calculate mean and standard error / CI at each x coordinate
        grouped = sub.groupby(x_col)[y_col].agg(["mean", "std", "count"])
        xs = grouped.index.to_numpy()
        means = grouped["mean"].to_numpy()
        stds = grouped["std"].fillna(0.0).to_numpy()
        counts = grouped["count"].to_numpy()

        ci = 1.96 * (stds / np.sqrt(np.maximum(counts, 1)))

        if show_spaghetti and "run_id" in sub.columns:
            for _r_id, r_df in sub.groupby("run_id"):
                ax.plot(
                    r_df[x_col],
                    r_df[y_col],
                    color=style["color"],
                    alpha=0.15,
                    linewidth=0.7,
                )

        ax.plot(
            xs,
            means,
            label=str(grp),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            markersize=4,
        )
        ax.fill_between(
            xs,
            means - ci,
            means + ci,
            color=style["color"],
            alpha=0.2,
        )


def forest(
    ax: plt.Axes,
    labels: list[str],
    diffs: list[float],
    ci_lows: list[float],
    ci_highs: list[float],
) -> None:
    """Plot paired difference forest plot with confidence intervals."""
    y_positions = np.arange(len(labels))
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)

    for i, y_pos in enumerate(y_positions):
        diff = diffs[i]
        low = ci_lows[i]
        high = ci_highs[i]
        ax.plot([low, high], [y_pos, y_pos], color="#0072B2", linewidth=1.8)
        ax.plot(diff, y_pos, "o", color="#0072B2", markersize=6)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Paired Difference (Δ)")


def rank_heatmap(
    ax: plt.Axes,
    rank_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
) -> None:
    """Plot rank heatmap with annotated numeric values."""
    im = ax.imshow(rank_matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = rank_matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if val < 3.0 else "white",
                fontsize=8,
            )
