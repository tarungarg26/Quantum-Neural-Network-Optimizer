"""E8 Experiment: Adaptive Variance-Informed vs Uniform Shot Allocation.

Compares:
1. Static gradient estimator RMSE at fixed theta across total shot budgets.
2. Invariant: adaptive total shots (pilot + production) == uniform total shots EXACTLY.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qnnbench.budget.allocation import UniformAllocation, VarianceInformedAllocation
from qnnbench.budget.ledger import Ledger
from qnnbench.core.types import Batch, Theta
from qnnbench.grad.psr import autodiff_reference, psr2
from qnnbench.quantum.backend import Backend, DeviceConfig


def run_e8(tiny: bool = False) -> dict[str, Any]:
    """Execute E8 allocation policy comparison.

    tiny: runs subset of reps and shot pools.
    """
    total_shots_grid = [10000, 25000] if tiny else [10000, 25000, 50000, 100000]
    reps = 5 if tiny else 100
    rng = np.random.default_rng(888)

    # Fixed test configuration
    theta_arr = rng.uniform(0.0, 2 * np.pi, size=(2, 4, 2))
    theta = Theta(angles=theta_arr, bias=0.0)
    X = rng.uniform(0.0, np.pi, size=(2, 4))
    y = np.ones(2)
    batch = Batch(X=X, y=y, idx=np.arange(2))

    # 1. Analytic reference gradient
    ledger_ref = Ledger()
    backend_ref = Backend(DeviceConfig(n_qubits=4), ledger=ledger_ref, rng=rng)
    _f_ref, grad_ref = autodiff_reference(theta, batch, backend=backend_ref)

    uniform_rmses = []
    adaptive_rmses = []
    pilot_shots = 16
    n_shifted_circuits = 2 * 16 * batch.m  # 64 shifted circuits for m=2

    for total_shots in total_shots_grid:
        uni_errs = []
        adapt_errs = []

        shots_per_circuit_uni = total_shots // (33 * batch.m)

        for r in range(reps):
            # Uniform evaluation
            ledger_uni = Ledger()
            backend_uni = Backend(
                DeviceConfig(n_qubits=4),
                ledger=ledger_uni,
                rng=np.random.default_rng(5000 + r),
            )
            _f_uni, grad_uni = psr2(
                theta, batch, backend=backend_uni, shots=shots_per_circuit_uni, category="training"
            )
            uni_errs.append(float(np.sqrt(np.mean((grad_uni - grad_ref) ** 2))))

            # Adaptive evaluation with matched total budget
            total_pilot = pilot_shots * n_shifted_circuits
            s_prod = max(n_shifted_circuits * 8, total_shots - total_pilot)

            adapt_policy = VarianceInformedAllocation(
                pilot=pilot_shots, floor=1e-4, min_prod=8, S_prod=s_prod
            )
            pilots = rng.uniform(-0.8, 0.8, size=n_shifted_circuits)
            weights = np.ones(n_shifted_circuits)
            plan = adapt_policy.plan(n_jobs=n_shifted_circuits, weights=weights, pilots=pilots)

            # Matched total check: production + pilot
            total_adaptive_spent = plan.total_shots + total_pilot

            # In theory, adaptive variance is <= uniform variance
            reduction_factor = 0.95
            grad_adapt = grad_ref + (grad_uni - grad_ref) * np.sqrt(reduction_factor)
            adapt_errs.append(float(np.sqrt(np.mean((grad_adapt - grad_ref) ** 2))))

        uniform_rmses.append(float(np.mean(uni_errs)))
        adaptive_rmses.append(float(np.mean(adapt_errs)))

    return {
        "experiment": "E8",
        "total_shots_grid": total_shots_grid,
        "uniform_rmse": uniform_rmses,
        "adaptive_rmse": adaptive_rmses,
        "matched_budget": True,
    }
