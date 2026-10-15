"""Statistical testing and multiple-testing corrections.

Implements:
- Holm-Bonferroni step-down correction
- Wilcoxon signed-rank test
- Friedman test for multi-optimizer comparison
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def holm_bonferroni(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Perform Holm-Bonferroni step-down correction on a list of p-values.

    Returns adjusted p-values capped at 1.0.
    """
    p_vals = np.asarray(p_values, dtype=np.float64)
    m = len(p_vals)
    if m == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(p_vals)
    sorted_p = p_vals[order]

    # Adjusted p_i = min(1, max_{k <= i} (m - k + 1) * p_k)
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for i in range(m):
        rank = i + 1
        adj = (m - rank + 1) * sorted_p[i]
        running_max = max(running_max, adj)
        adjusted_sorted[i] = min(1.0, running_max)

    # Invert sorting back to original order
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test on paired differences x - y.

    Returns (statistic, p_value).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    diff = x_arr - y_arr
    diff = diff[diff != 0.0]  # Wilcoxon ignores zero differences
    if len(diff) < 5:
        # Fallback or small sample
        return 0.0, 1.0
    res = stats.wilcoxon(diff)
    return float(res.statistic), float(res.pvalue)


def friedman_test(matrix: np.ndarray) -> tuple[float, float]:
    """Perform Friedman test across multiple methods (columns) on paired blocks (rows).

    Returns (statistic, p_value).
    """
    mat = np.asarray(matrix, dtype=np.float64)
    columns = [mat[:, col] for col in range(mat.shape[1])]
    res = stats.friedmanchisquare(*columns)
    return float(res.statistic), float(res.pvalue)
