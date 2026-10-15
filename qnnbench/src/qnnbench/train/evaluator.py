"""
AGENT INSTRUCTIONS — train/evaluator.py
=======================================
M11 — Evaluator & TestAccessGuard.

Invariants:
- TestAccessGuard ensures test split is evaluated EXACTLY ONCE per run, at finalisation.
- Access before finalisation or multiple accesses raise LeakageError.
- Evaluates MSE and accuracy.
"""

from __future__ import annotations

import numpy as np

from qnnbench.core.errors import LeakageError
from qnnbench.core.types import Batch, Theta
from qnnbench.data.manager import Scaler
from qnnbench.quantum.backend import Backend, CircuitJob


class TestAccessGuard:
    """Guard to ensure test set is accessed strictly once per run."""
    __test__ = False

    def __init__(self, test_X: np.ndarray, test_y: np.ndarray) -> None:
        self._test_X = test_X
        self._test_y = test_y
        self._accessed = False
        self._unlocked = False

    def unlock(self) -> None:
        """Called only at finalisation."""
        self._unlocked = True

    def get_test_batch(self, scaler: Scaler) -> Batch:
        if not self._unlocked:
            raise LeakageError("Attempted to access test set before finalisation unlock!")
        if self._accessed:
            raise LeakageError("Test set was already accessed! Strictly 1 evaluation per run allowed.")
        self._accessed = True
        X_tilde, _ = scaler.transform(self._test_X)
        return Batch(X=X_tilde, y=self._test_y, idx=np.arange(len(self._test_y)))


def evaluate(
    theta: Theta,
    batch: Batch,
    backend: Backend,
    shots: int | None = None,
    category: str = "monitoring",
    tag: str = "eval",
) -> tuple[float, float]:
    """Evaluate model on batch. Returns (mse, accuracy)."""
    jobs = [
        CircuitJob(theta_angles=theta.angles, x_tilde=batch.X[i], observable="Z0")
        for i in range(batch.m)
    ]
    preds_circuit = backend.execute(jobs, shots=shots, category=category, tag=tag)
    preds = preds_circuit + theta.bias

    mse = float(np.mean((preds - batch.y) ** 2))
    # Binary accuracy using sign(preds) in {-1, +1}
    signs = np.where(preds >= 0.0, 1.0, -1.0)
    acc = float(np.mean(signs == batch.y))
    return mse, acc
