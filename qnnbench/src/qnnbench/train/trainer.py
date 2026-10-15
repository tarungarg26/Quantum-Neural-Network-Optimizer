"""
AGENT INSTRUCTIONS — train/trainer.py
======================================
M11 — Trainer loop.

Contract:
- Initialises θ₀ identically across optimisers (paired).
- Minibatch order identical across optimisers (paired).
- All quantum executions route strictly through Backend.
- Test set guarded by TestAccessGuard (evaluated exactly once post-selection).
- Clean stop on BudgetExhausted / BudgetPreflightError without overshoot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from qnnbench.budget.allocation import AllocationPolicy, UniformAllocation
from qnnbench.budget.ledger import BudgetManager, Ledger
from qnnbench.core.errors import BudgetExhausted, BudgetPreflightError
from qnnbench.core.seeds import PurposeID, stream
from qnnbench.core.types import ANGLE_SHAPE, Batch, RunKey, Theta
from qnnbench.data.manager import Split, fit_scaler
from qnnbench.optim.base import Optimiser
from qnnbench.quantum.backend import Backend, DeviceConfig
from qnnbench.train.checkpoint import CheckpointTracker
from qnnbench.train.evaluator import TestAccessGuard, evaluate


@dataclass
class TrainResult:
    run_key: RunKey
    status: str
    updates_completed: int
    best_val_mse: float
    best_val_acc: float
    test_mse: float
    test_acc: float
    trace: list[dict[str, Any]]
    final_theta: Theta
    best_theta: Theta


def train(
    run_key: RunKey,
    split: Split,
    optimiser: Optimiser,
    budget_mgr: BudgetManager,
    allocation: AllocationPolicy | None = None,
    shots: int | None = None,
    batch_size: int = 10,
    max_updates: int = 80,
    eval_every: int = 5,
    patience: int = 5,
) -> TrainResult:
    """Execute training run."""
    ledger = Ledger()
    budget_mgr.attach_ledger(ledger)

    scaler = fit_scaler(split)
    train_X_tilde, _ = scaler.transform(split.train_X)
    val_X_tilde, _ = scaler.transform(split.val_X) if len(split.val_X) > 0 else (np.empty((0, 4)), None)

    guard = TestAccessGuard(split.test_X, split.test_y)

    # 1. Paired initialization: uses INIT stream
    rng_init = stream(run_key.split_seed, run_key.init_seed, PurposeID.INIT)
    angles0 = rng_init.uniform(0, 2 * np.pi, ANGLE_SHAPE)
    theta = Theta(angles=angles0, bias=0.0)

    # 2. Paired minibatches: uses MINIBATCH stream
    rng_batch = stream(run_key.split_seed, run_key.init_seed, PurposeID.MINIBATCH)

    # 3. Optimiser-specific shot sampling backend
    rng_shots = stream(run_key.split_seed, run_key.init_seed, PurposeID.SHOT_SAMPLING, optimiser=run_key.optimiser)
    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=rng_shots, run_id=run_key.run_id())

    if allocation is None:
        allocation = UniformAllocation(shots=shots if shots is not None else 32)

    val_batch = Batch(X=val_X_tilde, y=split.val_y, idx=split.val_idx) if len(split.val_X) > 0 else None

    tracker = CheckpointTracker(patience=patience)
    trace: list[dict[str, Any]] = []

    m_train = len(split.train_X)
    status = "completed_updates"
    update_idx = 0

    t_start = time.perf_counter()

    for u in range(max_updates):
        update_idx = u

        # Generate minibatch indices (paired)
        if batch_size >= m_train:
            b_idx = np.arange(m_train)
        else:
            b_idx = rng_batch.choice(m_train, size=batch_size, replace=False)

        batch = Batch(X=train_X_tilde[b_idx], y=split.train_y[b_idx], idx=b_idx)

        # Preflight budget affordance check
        cost_circuits = optimiser.cost_per_update(batch.m, shots=shots)
        shots_per_circ = shots if shots is not None else 0
        if not budget_mgr.can_afford_update(cost_circuits, shots_per_circ, category="training"):
            status = "completed_budget"
            break

        try:
            step = optimiser.propose(theta, batch, backend, shots=shots, update_idx=u)
            theta = optimiser.apply(theta, step)
        except (BudgetPreflightError, BudgetExhausted):
            status = "completed_budget"
            break

        # Checkpoint evaluation on val
        val_mse = 0.0
        val_acc = 0.0
        if val_batch is not None and (u % eval_every == 0 or u == max_updates - 1):
            n_val_circuits = val_batch.m
            val_shots_per_circ = shots if shots is not None else 0
            if budget_mgr.can_afford("monitoring", n_val_circuits, val_shots_per_circ):
                val_mse, val_acc = evaluate(theta, val_batch, backend, shots=shots, category="monitoring", tag="val_eval")
                stop_early = tracker.step(u, val_mse, val_acc, theta, optimiser.state_dict())
                if stop_early:
                    status = "early_stopped"
                    break

        trace.append({
            "update": u,
            "val_mse": val_mse,
            "val_acc": val_acc,
            "wall_s": time.perf_counter() - t_start,
        })

    best_record = tracker.best_checkpoint
    best_theta = best_record.theta if best_record is not None else theta

    # 4. Final: unlock test guard and evaluate TEST exactly once (D-04: uncharged reference)
    guard.unlock()
    test_batch = guard.get_test_batch(scaler)
    test_mse, test_acc = evaluate(best_theta, test_batch, backend, shots=None, category="reference", tag="test_eval")

    return TrainResult(
        run_key=run_key,
        status=status,
        updates_completed=update_idx + 1,
        best_val_mse=best_record.val_mse if best_record is not None else 0.0,
        best_val_acc=best_record.val_acc if best_record is not None else 0.0,
        test_mse=test_mse,
        test_acc=test_acc,
        trace=trace,
        final_theta=theta,
        best_theta=best_theta,
    )
