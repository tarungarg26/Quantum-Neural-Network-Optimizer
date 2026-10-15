"""E6 Experiment: Trainability and Gradient Variance Scaling.

Measures gradient variance as a function of qubit width n in {2, 4, 6, 8, 10, 12}
comparing local observable (Z0) against global observable (parity Z⊗...⊗Z).
Invariants:
- Exactly paired parameter draws across local and global observables.
- Log-log fits return confidence intervals and strictly forbid extrapolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qnnbench.analysis.fits import fit_log_log
from qnnbench.quantum.circuit_factory import build_parameterised_qnode


def run_e6(tiny: bool = False) -> dict[str, Any]:
    """Execute E6 trainability experiment.

    tiny: uses subset of widths and samples for fast execution.
    """
    widths = [2, 4] if tiny else [2, 4, 6, 8, 10, 12]
    layers = 2
    n_samples = 30 if tiny else 200
    rng = np.random.default_rng(777)

    variances_local = []
    variances_parity = []

    for n in widths:
        qnode_local = build_parameterised_qnode(n_qubits=n, layers=layers, observable="Z0")
        qnode_parity = build_parameterised_qnode(n_qubits=n, layers=layers, observable="parity")

        # Decision D-12: 200 paired parameter vectors
        param_shape = (layers, n, 2)
        n_params = layers * n * 2
        # Shared parameter vectors across observables (paired draws)
        thetas = [rng.uniform(0.0, 2 * np.pi, size=param_shape) for _ in range(n_samples)]
        x_fixed = np.zeros(n)

        grads_local = []
        grads_parity = []

        shift = np.pi / 2.0
        # Compute gradient component for first parameter using 2-term PSR analytically
        for theta in thetas:
            # Local Z0
            theta_plus = theta.copy()
            theta_plus[0, 0, 0] += shift
            theta_minus = theta.copy()
            theta_minus[0, 0, 0] -= shift
            g_loc = 0.5 * (float(qnode_local(theta_plus, x_fixed)) - float(qnode_local(theta_minus, x_fixed)))
            grads_local.append(g_loc)

            # Global parity (Z ⊗ ... ⊗ Z)
            g_par = 0.5 * (float(qnode_parity(theta_plus, x_fixed)) - float(qnode_parity(theta_minus, x_fixed)))
            grads_parity.append(g_par)

        var_l = float(np.var(grads_local, ddof=1))
        var_p = float(np.var(grads_parity, ddof=1))
        variances_local.append(var_l)
        variances_parity.append(var_p)

    fit_loc = fit_log_log(np.array(widths, dtype=float), np.array(variances_local))
    fit_par = fit_log_log(np.array(widths, dtype=float), np.array(variances_parity))

    return {
        "experiment": "E6",
        "widths": widths,
        "variances_local": variances_local,
        "variances_parity": variances_parity,
        "fit_local_slope": fit_loc.slope,
        "fit_local_ci": fit_loc.slope_ci,
        "fit_parity_slope": fit_par.slope,
        "fit_parity_ci": fit_par.slope_ci,
        "fit_object_local": fit_loc,
        "fit_object_parity": fit_par,
    }
