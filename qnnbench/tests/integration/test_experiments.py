"""Integration tests for experiment modules E1 through E8 (T-M14-*)."""

from __future__ import annotations

import pytest

from qnnbench.experiments.runner import run_experiment


def test_t_m14_e1_smoke_and_correctness() -> None:
    """T-M14-E1: Two-term PSR smoke execution and algebraic equivalence check."""
    res = run_experiment("E1", tiny=True)
    assert res["experiment"] == "E1"
    assert res["passed"] is True
    assert res["max_error"] <= 1e-10


def test_t_m14_e2_smoke_and_four_term_distinction() -> None:
    """T-M14-E2: Four-term PSR smoke execution and multi-frequency distinction."""
    res = run_experiment("E2", tiny=True)
    assert res["experiment"] == "E2"
    assert res["passed"] is True
    assert res["max_psr4_error"] <= 1e-9
    assert res["mean_psr2_error"] > 0.01


def test_t_m14_e3_smoke_all_five_optimisers() -> None:
    """T-M14-E3: Analytic comparison across GD, Momentum, Adam, SPSA, and QNG."""
    runs = run_experiment("E3", tiny=True)
    assert len(runs) == 5
    optimisers = {r["optimiser"] for r in runs}
    assert optimisers == {"gd", "momentum", "adam", "spsa", "qng"}
    for r in runs:
        assert r["updates_completed"] == 3
        assert r["best_val_mse"] >= 0.0


def test_t_m14_e4_smoke_and_budget_invariants() -> None:
    """T-M14-E4: Equal-shot comparison respects training and monitoring limits."""
    runs = run_experiment("E4", tiny=True)
    assert len(runs) == 5
    for r in runs:
        assert r["total_train_shots_spent"] <= r["budget"]
        assert r["total_mon_shots_spent"] <= int(r["budget"] * 0.20) + 1000


def test_t_m14_e5_smoke_shot_noise_scaling() -> None:
    """T-M14-E5: Shot-noise scaling smoke test."""
    res = run_experiment("E5", tiny=True)
    assert res["experiment"] == "E5"
    assert len(res["variances"]) == 3
    # slope should be negative, around -1
    assert res["fitted_slope"] < 0.0


def test_t_m14_e6_smoke_trainability_and_no_extrapolation() -> None:
    """T-M14-E6: Trainability scan returns CIs and refuses extrapolation."""
    res = run_experiment("E6", tiny=True)
    assert res["experiment"] == "E6"
    assert len(res["widths"]) == 2
    fit_obj = res["fit_object_local"]
    with pytest.raises(ValueError, match="extrapolation"):
        fit_obj.predict(100.0)


def test_t_m14_e7_smoke_qng_diagnostics() -> None:
    """T-M14-E7: QNG condition number and GD baseline comparison."""
    results = run_experiment("E7", tiny=True)
    assert len(results) >= 2
    assert results[0]["optimiser"] == "gd"
    assert results[1]["optimiser"] == "qng"


def test_t_m14_e8_smoke_adaptive_allocation() -> None:
    """T-M14-E8: Adaptive vs uniform allocation comparison."""
    res = run_experiment("E8", tiny=True)
    assert res["experiment"] == "E8"
    assert res["matched_budget"] is True
    assert len(res["uniform_rmse"]) == 2
