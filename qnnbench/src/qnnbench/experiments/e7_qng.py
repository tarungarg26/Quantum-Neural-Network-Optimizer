"""E7 Experiment: Quantum Natural Gradient Geometry & Overhead.

Investigates:
1. Condition number and spectrum of Fubini-Study block-diagonal metric tensor G.
2. Sensitivity to damping parameter lambda in {0.001, 0.01, 0.05}.
3. Sample count sensitivity for metric evaluation (X_metric in {1, 4, 8}).
4. Progress per update vs circuit evaluation overhead compared to GD.
"""

from __future__ import annotations

from typing import Any

from qnnbench.budget.ledger import BudgetManager
from qnnbench.core.types import RunKey
from qnnbench.data.manager import make_split, make_task
from qnnbench.optim.base import QNG, GradientDescent
from qnnbench.train.trainer import train


def run_e7(tiny: bool = False) -> list[dict[str, Any]]:
    """Execute E7 QNG diagnostics experiment.

    tiny: runs 1 seed, 5 updates for quick validation.
    """
    X, y = make_task()
    split_seed = 7
    init_seed = 0
    split = make_split(X, y, split_seed=split_seed)

    dampings = [0.01] if tiny else [0.001, 0.01, 0.05]
    metric_inputs_list = [4] if tiny else [1, 4, 8]
    max_updates = 5 if tiny else 30

    results = []

    # Baseline GD for direct comparison
    gd_opt = GradientDescent(lr=0.1)
    gd_mgr = BudgetManager(train_shots=10**8, monitoring_shots=10**8)
    gd_key = RunKey(
        experiment="E7",
        optimiser="gd",
        split_seed=split_seed,
        init_seed=init_seed,
        config_hash="e7_gd",
    )
    gd_res = train(
        run_key=gd_key,
        split=split,
        optimiser=gd_opt,
        budget_mgr=gd_mgr,
        shots=None,
        batch_size=len(split.train_X),
        max_updates=max_updates,
        eval_every=1,
    )
    results.append({
        "config": "gd_baseline",
        "optimiser": "gd",
        "damping": 0.0,
        "metric_inputs": 0,
        "best_val_mse": gd_res.best_val_mse,
        "best_val_acc": gd_res.best_val_acc,
        "test_mse": gd_res.test_mse,
        "trace": gd_res.trace,
    })

    # QNG sweeps
    for damp in dampings:
        for m_in in metric_inputs_list:
            qng_opt = QNG(lr=0.1, damping=damp, metric_inputs=m_in)
            qng_mgr = BudgetManager(train_shots=10**8, monitoring_shots=10**8)
            qng_key = RunKey(
                experiment="E7",
                optimiser="qng",
                split_seed=split_seed,
                init_seed=init_seed,
                config_hash=f"e7_d{damp}_m{m_in}",
            )
            qng_res = train(
                run_key=qng_key,
                split=split,
                optimiser=qng_opt,
                budget_mgr=qng_mgr,
                shots=None,
                batch_size=len(split.train_X),
                max_updates=max_updates,
                eval_every=1,
            )
            results.append({
                "config": f"qng_d{damp}_m{m_in}",
                "optimiser": "qng",
                "damping": damp,
                "metric_inputs": m_in,
                "best_val_mse": qng_res.best_val_mse,
                "best_val_acc": qng_res.best_val_acc,
                "test_mse": qng_res.test_mse,
                "trace": qng_res.trace,
            })

    return results
