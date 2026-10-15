"""E1 Experiment: Two-term Parameter-Shift Rule Correctness.

Evaluates PSR2 against analytic autodiff over random (theta, x) pairs.
Primary outputs: max and mean absolute error, verification flag.
"""

from __future__ import annotations

import numpy as np

from qnnbench.budget.ledger import Ledger
from qnnbench.core.types import Batch, Theta
from qnnbench.grad.psr import autodiff_reference, psr2
from qnnbench.quantum.backend import Backend, DeviceConfig


def run_e1(n_samples: int = 200, seed: int = 42, tol: float = 1e-10) -> dict[str, float | bool]:
    """Execute E1 experiment."""
    rng = np.random.default_rng(seed)
    ledger = Ledger()
    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=rng)

    errors = []
    for _ in range(n_samples):
        theta_arr = rng.uniform(0.0, 2 * np.pi, size=(2, 4, 2))
        bias = float(rng.uniform(-1.0, 1.0))
        theta = Theta(angles=theta_arr, bias=bias)

        x_raw = rng.uniform(0.0, np.pi, size=(1, 4))
        y_dummy = np.array([1.0])
        batch = Batch(X=x_raw, y=y_dummy, idx=np.array([0]))

        # PSR2
        _f_psr, grad_psr = psr2(theta, batch, backend=backend, shots=None, category="reference")
        # Autodiff reference
        _f_ref, grad_ref = autodiff_reference(theta, batch, backend=backend)

        diff = np.abs(grad_psr - grad_ref)
        errors.append(float(np.max(diff)))

    max_err = float(np.max(errors))
    mean_err = float(np.mean(errors))
    passed = bool(max_err <= tol)

    return {
        "experiment": "E1",
        "n_samples": n_samples,
        "max_error": max_err,
        "mean_error": mean_err,
        "tolerance": tol,
        "passed": passed,
    }
