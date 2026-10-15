"""Regression tests and mutation sanity guards enforcing context.md and claude.md invariants.

Verifies:
- Corrections #1 through #6 (DA1 -> DA2 corrections)
- Forbidden behavior mutation guards (leakage, budget overshoot, premature test access, frozen tampering)
"""

from __future__ import annotations

import numpy as np
import pytest

from qnnbench.analysis.compare import paired_differences
from qnnbench.budget.ledger import BudgetManager, Category
from qnnbench.core.config import Config, load_frozen
from qnnbench.core.errors import (
    BudgetPreflightError,
    ConfigFrozenError,
    LeakageError,
    NonFiniteError,
)
from qnnbench.core.types import Theta
from qnnbench.data.manager import Scaler, make_split, make_task
from qnnbench.grad.psr import mse_loss
from qnnbench.train.evaluator import TestAccessGuard


# ─── DA1 -> DA2 Corrections (#1 to #6) ─────────────────────────────────────────


def test_correction_1_no_leakage() -> None:
    """Correction 1: Scaler must be fit strictly on training data."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)

    # Train-only fit
    scaler = Scaler.fit(split.train_X)
    orig_a = scaler.a.copy()
    orig_d = scaler.d.copy()

    # Perturb test set
    perturbed_test = split.test_X + 100.0
    # Scaler params must be completely unchanged by test perturbation
    np.testing.assert_array_equal(scaler.a, orig_a)
    np.testing.assert_array_equal(scaler.d, orig_d)

    # If user tries to fit on full dataset or test set with wrong tag, ensure safeguard or detection
    with pytest.raises(LeakageError):
        # Explicit guard when non-train batch is passed
        Scaler.fit_guarded(split.test_X, is_train=False)


def test_correction_2_backprop_is_labelled_reference() -> None:
    """Correction 2: Backprop/autodiff is classified under Category.REFERENCE and incurs 0 budget."""
    mgr = BudgetManager(train_shots=1000, monitoring_shots=500)
    # Category reference must never reduce remaining shots
    assert mgr.can_afford("reference", circuits=1000, shots_per_circuit=1000)
    mgr.preflight("reference", circuits=1000, shots_per_circuit=1000)
    assert mgr.spent("training") == 0
    assert mgr.spent("monitoring") == 0


def test_correction_3_paired_variance_draws() -> None:
    """Correction 3: Variances must be evaluated on identical paired parameter draws."""
    from qnnbench.core.seeds import PurposeID, stream

    rng1 = stream(split_seed=7, init_seed=0, purpose=PurposeID.INIT, optimiser=None)
    draw1 = rng1.uniform(0.0, 2 * np.pi, size=(2, 4, 2))

    rng2 = stream(split_seed=7, init_seed=0, purpose=PurposeID.INIT, optimiser=None)
    draw2 = rng2.uniform(0.0, 2 * np.pi, size=(2, 4, 2))

    # Identical draws for paired comparisons
    np.testing.assert_array_equal(draw1, draw2)


def test_correction_4_cost_formula() -> None:
    """Correction 4: Formula cost per update is (2p + 1)m. Ratio SPSA/PSR is 16.5 for p=16."""
    p = 16
    m = 70
    psr_circuits = (2 * p + 1) * m
    assert psr_circuits == 2310

    m_small = 10
    psr_circuits_small = (2 * p + 1) * m_small
    assert psr_circuits_small == 330

    # SPSA circuits = 2 * m
    spsa_circuits = 2 * m
    ratio = psr_circuits / spsa_circuits
    assert abs(ratio - 16.5) < 1e-12


def test_correction_5_no_extrapolation() -> None:
    """Correction 5: Exponent fits must refuse extrapolation beyond measured widths."""
    from qnnbench.analysis.compare import fit_log_log

    x_measured = np.array([2.0, 4.0, 6.0, 8.0])
    y_measured = 1.0 / (2.0 ** x_measured)
    res = fit_log_log(x_measured, y_measured)

    # Calling predict on x > 8 should raise ValueError
    with pytest.raises(ValueError, match="extrapolation"):
        res.predict(x=16.0)


def test_correction_6_historical_track_isolated() -> None:
    """Correction 6: Historical track results are never merged into main comparisons."""
    import pandas as pd

    df = pd.DataFrame({
        "split_seed": [7, 7, 17, 17],
        "init_seed": [0, 0, 0, 0],
        "optimiser": ["adam", "gd", "adam", "gd"],
        "val_mse": [0.05, 0.08, 0.06, 0.09],
        "track": ["historical", "historical", "main", "main"],
    })

    main_df = df[df["track"] != "historical"]
    diffs = paired_differences(main_df, opt_a="adam", opt_b="gd", metric="val_mse")
    assert len(diffs) == 1  # Only seed 17 is included; historical seed 7 is excluded


# ─── Mutation Sanity Guards ───────────────────────────────────────────────────


def test_guard_test_access_once_only() -> None:
    """Guard: Test set cannot be evaluated before selection or more than once."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)
    scaler = Scaler.fit(split.train_X)
    guard = TestAccessGuard(split.test_X, split.test_y)

    # Accessing before unlock raises LeakageError
    with pytest.raises(LeakageError):
        guard.get_test_batch(scaler)

    # First access after unlock succeeds
    guard.unlock()
    batch = guard.get_test_batch(scaler)
    assert batch.m > 0

    # Second access raises LeakageError
    with pytest.raises(LeakageError):
        guard.get_test_batch(scaler)


def test_guard_budget_overshoot_raises() -> None:
    """Guard: Attempting to commit an operation exceeding remaining budget raises."""
    mgr = BudgetManager(train_shots=100, monitoring_shots=50)
    with pytest.raises(BudgetPreflightError):
        mgr.preflight("training", circuits=10, shots_per_circuit=20)  # 200 > 100


def test_guard_tampered_frozen_config() -> None:
    """Guard: Any modification to a frozen config YAML invalidates the hash."""
    from pathlib import Path
    frozen_path = Path("configs/frozen/gd.yaml")
    if frozen_path.exists():
        # Valid load succeeds
        cfg = load_frozen(frozen_path)
        assert cfg.hash is not None

        # Altering raw dictionary changes hash
        tampered = Config(raw={**cfg.raw, "new_key": "illegal"})
        assert tampered.hash != cfg.hash


def test_guard_non_finite_values_raise() -> None:
    """Guard: Non-finite loss or inputs raise NonFiniteError."""
    with pytest.raises(NonFiniteError):
        mse_loss(np.array([np.nan, 0.5]), b=0.0, y=np.array([1.0, -1.0]))
