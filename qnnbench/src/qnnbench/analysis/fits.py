"""Log-log OLS fitting and scaling analysis for E5 and E6.

Invariants:
- Returns slope, intercept, and 95% CI.
- Forbids extrapolation beyond measured bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy import stats


@dataclass
class LogLogFitResult:
    """Result of a log-log linear fit log(y) = slope * log(x) + intercept."""

    slope: float
    intercept: float
    slope_ci: tuple[float, float]
    min_x: float
    max_x: float
    r_squared: float
    std_err: float

    def predict(self, x: float | np.ndarray) -> np.ndarray:
        """Predict y for given x. Refuses extrapolation beyond [min_x, max_x]."""
        x_arr = np.asarray(x, dtype=np.float64)
        if np.any(x_arr < self.min_x - 1e-9) or np.any(x_arr > self.max_x + 1e-9):
            raise ValueError(
                f"Refusing extrapolation request beyond measured domain [{self.min_x}, {self.max_x}]"
            )
        return np.exp(self.intercept) * (x_arr ** self.slope)

    def __iter__(self) -> Iterator[float | tuple[float, float]]:
        """Yield (slope, intercept, slope_ci) for tuple unpacking compatibility."""
        yield self.slope
        yield self.intercept
        yield self.slope_ci


def fit_log_log(x: np.ndarray, y: np.ndarray) -> LogLogFitResult:
    """Fit log(y) = slope * log(x) + intercept via OLS.

    Returns LogLogFitResult with slope, intercept, 95% CI on slope, and strict bounds.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")
    if np.any(x_arr <= 0) or np.any(y_arr <= 0):
        raise ValueError("x and y must be strictly positive for log-log fit")

    log_x = np.log(x_arr)
    log_y = np.log(y_arr)
    n = len(x_arr)

    slope, intercept, r_value, _p_value, std_err = stats.linregress(log_x, log_y)
    t_crit = stats.t.ppf(0.975, df=n - 2) if n > 2 else 1.96
    ci = (float(slope - t_crit * std_err), float(slope + t_crit * std_err))

    return LogLogFitResult(
        slope=float(slope),
        intercept=float(intercept),
        slope_ci=ci,
        min_x=float(np.min(x_arr)),
        max_x=float(np.max(x_arr)),
        r_squared=float(r_value ** 2),
        std_err=float(std_err),
    )
