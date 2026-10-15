"""Analysis and comparison engine (M15).

Implements:
- Paired differences (arrays or DataFrame with matching split/init keys)
- Cluster and percentile bootstrap confidence intervals
- Average ranking with tie handling
- Log-log OLS fitting via fits.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from qnnbench.analysis.fits import LogLogFitResult, fit_log_log
from qnnbench.analysis.stats import friedman_test, holm_bonferroni, paired_wilcoxon

__all__ = [
    "LogLogFitResult",
    "bootstrap_ci",
    "compute_ranks",
    "fit_log_log",
    "friedman_test",
    "holm_bonferroni",
    "paired_differences",
    "paired_wilcoxon",
]


def paired_differences(
    a_or_df: np.ndarray | pd.DataFrame,
    b_or_opt_a: Any = None,
    opt_b: str | None = None,
    metric: str = "val_mse",
    opt_a: str | None = None,
) -> np.ndarray:
    """Compute paired differences Δ = a - b.

    Usage 1 (arrays):
      paired_differences(values_a, values_b) -> np.ndarray

    Usage 2 (DataFrame):
      paired_differences(df, opt_a="adam", opt_b="gd", metric="val_mse") -> np.ndarray
    """
    if isinstance(a_or_df, pd.DataFrame):
        df = a_or_df
        actual_opt_a = opt_a if opt_a is not None else b_or_opt_a
        if actual_opt_a is None or opt_b is None:
            raise ValueError("opt_a and opt_b must be provided when passing DataFrame")

        df_a = df[df["optimiser"] == actual_opt_a].set_index(["split_seed", "init_seed"])
        df_b = df[df["optimiser"] == opt_b].set_index(["split_seed", "init_seed"])

        common_keys = df_a.index.intersection(df_b.index)
        if len(common_keys) == 0:
            return np.array([], dtype=np.float64)

        vals_a = df_a.loc[common_keys, metric].to_numpy(dtype=np.float64)
        vals_b = df_b.loc[common_keys, metric].to_numpy(dtype=np.float64)
        return vals_a - vals_b

    # Array usage
    a = np.asarray(a_or_df)
    b = np.asarray(b_or_opt_a)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return a - b


def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and (1-alpha) percentile bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    """
    arr = np.asarray(data)
    rng = np.random.default_rng(seed)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    mean_val = float(np.mean(arr))
    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return mean_val, ci_low, ci_high


def compute_ranks(matrix: np.ndarray) -> np.ndarray:
    """Compute average ranks across rows (1 = lowest/best).

    matrix: (n_instances, n_methods)
    """
    matrix = np.asarray(matrix)
    ranks = np.empty_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        ranks[i] = stats.rankdata(matrix[i])
    return ranks
