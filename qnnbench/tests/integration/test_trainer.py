import numpy as np
import pytest

from qnnbench.budget.ledger import BudgetManager
from qnnbench.core.errors import LeakageError
from qnnbench.core.types import ANGLE_SHAPE, RunKey, Theta
from qnnbench.data.manager import fit_scaler, make_split, make_task
from qnnbench.optim.base import GradientDescent
from qnnbench.train.checkpoint import CheckpointTracker
from qnnbench.train.evaluator import TestAccessGuard
from qnnbench.train.trainer import train


def test_t_m11_01_test_access_guard_locked_raises():
    """T-M11-01: Attempting to access test set before unlock raises LeakageError."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)
    guard = TestAccessGuard(split.test_X, split.test_y)
    scaler = fit_scaler(split)

    with pytest.raises(LeakageError, match="before finalisation"):
        guard.get_test_batch(scaler)


def test_t_m11_02_test_evaluated_strictly_once():
    """T-M11-02: Second access to test set raises LeakageError."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)
    guard = TestAccessGuard(split.test_X, split.test_y)
    scaler = fit_scaler(split)

    guard.unlock()
    batch1 = guard.get_test_batch(scaler)
    assert batch1.m == 20

    with pytest.raises(LeakageError, match="already accessed"):
        guard.get_test_batch(scaler)


def test_t_m11_03_checkpoint_selects_argmin_val_mse():
    """T-M11-03: Checkpoint tracker selects minimum val MSE."""
    tracker = CheckpointTracker(patience=10)
    theta1 = Theta(np.zeros(ANGLE_SHAPE), 0.1)
    theta2 = Theta(np.zeros(ANGLE_SHAPE), 0.2)
    theta3 = Theta(np.zeros(ANGLE_SHAPE), 0.3)

    tracker.step(update=0, val_mse=0.45, val_acc=0.8, theta=theta1, optimiser_state={})
    tracker.step(update=1, val_mse=0.25, val_acc=0.9, theta=theta2, optimiser_state={})
    tracker.step(update=2, val_mse=0.35, val_acc=0.85, theta=theta3, optimiser_state={})

    assert tracker.best_checkpoint is not None
    assert tracker.best_checkpoint.val_mse == 0.25
    assert tracker.best_checkpoint.theta == theta2


def test_t_m11_04_budget_exhaustion_clean_termination():
    """T-M11-04: Budget exhaustion terminates cleanly with status='completed_budget'."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)

    # Budget that allows only 1 update (m=10 => 330 circuits * 32 shots = 10,560 shots)
    budget_mgr = BudgetManager(train_shots=11_000, monitoring_shots=5_000)
    opt = GradientDescent(lr=0.1)
    key = RunKey("test_budget", "gd", 7, 0, "hash1")

    res = train(
        run_key=key,
        split=split,
        optimiser=opt,
        budget_mgr=budget_mgr,
        shots=32,
        batch_size=10,
        max_updates=10,
    )

    assert res.status == "completed_budget"
    assert res.updates_completed <= 2


def test_t_m11_10_tiny_integration_run():
    """T-M11-10: Tiny end-to-end run (analytic, 2 updates) finishes cleanly."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)
    budget_mgr = BudgetManager(train_shots=100_000, monitoring_shots=50_000)

    opt = GradientDescent(lr=0.1)
    key = RunKey("smoke_e2e", "gd", 7, 0, "hash_smoke")

    res = train(
        run_key=key,
        split=split,
        optimiser=opt,
        budget_mgr=budget_mgr,
        shots=None,  # analytic
        batch_size=2,  # tiny batch for fast execution
        max_updates=2,
        eval_every=1,
    )

    assert res.updates_completed == 2
    assert res.test_mse >= 0.0
    assert 0.0 <= res.test_acc <= 1.0
