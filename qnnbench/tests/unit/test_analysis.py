import numpy as np
import pytest

from qnnbench.analysis.compare import (
    bootstrap_ci,
    compute_ranks,
    fit_log_log,
    paired_differences,
)


def test_t_m15_01_paired_differences():
    """T-M15-01: Paired differences require equal shape."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 2.0, 1.0])
    diff = paired_differences(a, b)
    assert np.allclose(diff, [0.5, 0.0, 2.0])

    with pytest.raises(ValueError, match="Shape mismatch"):
        paired_differences(a, np.array([1.0, 2.0]))


def test_t_m15_02_bootstrap_ci_coverage():
    """T-M15-02: Bootstrap CI covers known sample mean with fixed seed."""
    data = np.array([1.2, 1.5, 1.1, 1.8, 1.4, 1.3, 1.6, 1.2, 1.4, 1.5])
    mean_val, low, high = bootstrap_ci(data, n_boot=1000, seed=42)
    assert low <= mean_val <= high
    assert np.isclose(mean_val, np.mean(data))


def test_t_m15_07_rank_computation_handles_ties():
    """T-M15-07: Rank computation properly assigns average ranks to ties."""
    matrix = np.array([
        [10.0, 20.0, 20.0, 40.0],  # rank: 1, 2.5, 2.5, 4
    ])
    ranks = compute_ranks(matrix)
    assert np.allclose(ranks[0], [1.0, 2.5, 2.5, 4.0])


def test_t_m15_09_fit_log_log_recovers_exponent():
    """T-M15-09: Log-log fit recovers known synthetic exponent within CI."""
    x = np.array([10.0, 20.0, 50.0, 100.0, 200.0, 500.0])
    true_slope = -1.0
    true_c = 5.0
    y = true_c * (x ** true_slope)

    slope, _intercept, ci = fit_log_log(x, y)
    assert np.isclose(slope, true_slope, atol=1e-5)
    assert ci[0] <= true_slope <= ci[1]


def test_t_m15_10_determinism():
    """T-M15-10: Bootstrap is deterministic given fixed seed."""
    data = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    res1 = bootstrap_ci(data, n_boot=500, seed=123)
    res2 = bootstrap_ci(data, n_boot=500, seed=123)
    assert res1 == res2
