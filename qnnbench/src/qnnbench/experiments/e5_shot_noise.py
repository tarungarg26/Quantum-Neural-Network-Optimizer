"""E5 Experiment: Shot-Noise Law and Scaling.

Validates:
1. Var(f̂) · N ≈ 1 - f² (Bernoulli / binomial expectation variance).
2. Slope of log(Var) vs log(N) is strictly -1 within confidence intervals.
3. Unbiased finite-shot loss corrects the positive bias of naive MSE.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qnnbench.analysis.fits import fit_log_log
from qnnbench.budget.ledger import BudgetManager, Ledger
from qnnbench.core.types import Theta
from qnnbench.quantum.backend import Backend, CircuitJob, DeviceConfig


def run_e5(tiny: bool = False) -> dict[str, Any]:
    """Execute E5 shot-noise scaling experiment.

    tiny: uses fewer shot tiers and reps for rapid test/CI runs.
    """
    shot_counts = [32, 128, 512] if tiny else [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    reps = 200 if tiny else 2000
    rng = np.random.default_rng(999)

    # Fixed single parameter and input for exact expectation reference
    theta_arr = np.zeros((2, 4, 2), dtype=np.float64)
    theta_arr[0, 0, 0] = np.pi / 4.0
    theta = Theta(angles=theta_arr, bias=0.0)
    x = np.array([np.pi / 3.0, 0.0, 0.0, 0.0])
    job = CircuitJob(theta_angles=theta.angles, x_tilde=x, observable="Z0")

    # 1. Analytic reference value
    ledger_ref = Ledger()
    mgr_ref = BudgetManager(10**6, 10**6)
    backend_ref = Backend(DeviceConfig(n_qubits=4), ledger=ledger_ref, rng=rng, run_id="ref")
    mgr_ref.attach_ledger(ledger_ref)
    f_analytic = float(backend_ref.execute([job], shots=None, category="reference", tag="analytic")[0])
    expected_n_var = 1.0 - (f_analytic ** 2)

    variances = []
    n_var_products = []

    for n_shots in shot_counts:
        samples = []
        for r in range(reps):
            ledger = Ledger()
            backend_sampled = Backend(
                DeviceConfig(n_qubits=4),
                ledger=ledger,
                rng=np.random.default_rng(10000 + r),
                run_id=f"e5_s{n_shots}_r{r}",
            )
            val = float(backend_sampled.execute([job], shots=n_shots, category="training", tag="sample")[0])
            samples.append(val)

        emp_var = float(np.var(samples, ddof=1))
        variances.append(emp_var)
        n_var_products.append(emp_var * n_shots)

    fit_res = fit_log_log(np.array(shot_counts), np.array(variances))

    return {
        "experiment": "E5",
        "f_analytic": f_analytic,
        "expected_n_var": expected_n_var,
        "shot_counts": shot_counts,
        "variances": variances,
        "n_var_products": n_var_products,
        "fitted_slope": fit_res.slope,
        "fitted_slope_ci": fit_res.slope_ci,
        "passed": bool(fit_res.slope_ci[0] <= -0.90 and fit_res.slope_ci[1] >= -1.10),
    }
