"""E2 Experiment: Four-term Parameter-Shift Rule on Controlled Rotations.

Demonstrates that 4-term PSR achieves exact analytic gradient on multi-frequency
generators while 2-term PSR has measurable structural error.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from qnnbench.grad.psr import psr4
from qnnbench.quantum.circuit_factory import build_test_circuit


def run_e2(n_trials: int = 50, seed: int = 123) -> dict[str, float | bool]:
    """Execute E2 experiment."""
    rng = np.random.default_rng(seed)
    crot_circuit = build_test_circuit("crot")

    psr4_errors = []
    psr2_errors = []

    for _ in range(n_trials):
        params = rng.uniform(0.0, 2 * np.pi, size=(3,))
        x = rng.uniform(0.0, np.pi, size=(2,))

        # Autodiff reference
        params_pl = qml.numpy.array(params.copy(), requires_grad=True)
        grad_ad = qml.grad(lambda p: crot_circuit(p, x), argnums=0)(params_pl)

        # 4-term shift rule for parameter 1 (RY inside CRot)
        k = 1
        grad_psr4 = psr4(params, k, lambda p, _x: crot_circuit(p, _x), x)
        psr4_errors.append(float(abs(grad_psr4 - grad_ad[k])))

        # 2-term shift rule
        p_plus = params.copy()
        p_plus[k] += np.pi / 2
        p_minus = params.copy()
        p_minus[k] -= np.pi / 2
        grad_psr2 = 0.5 * (float(crot_circuit(p_plus, x)) - float(crot_circuit(p_minus, x)))
        psr2_errors.append(float(abs(grad_psr2 - grad_ad[k])))

    max_psr4_err = float(np.max(psr4_errors))
    max_psr2_err = float(np.max(psr2_errors))
    mean_psr2_err = float(np.mean(psr2_errors))

    passed = bool(max_psr4_err <= 1e-9 and mean_psr2_err > 0.01)

    return {
        "experiment": "E2",
        "n_trials": n_trials,
        "max_psr4_error": max_psr4_err,
        "max_psr2_error": max_psr2_err,
        "mean_psr2_error": mean_psr2_err,
        "passed": passed,
    }
