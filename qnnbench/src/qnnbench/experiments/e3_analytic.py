"""E3 Experiment: Five Optimisers on Analytic Statevector Simulation.

Executes 80 full-batch updates for GD, Momentum, Adam, SPSA, and QNG
across 15 paired replication keys (5 split seeds x 3 init seeds).
"""

from __future__ import annotations

from typing import Any

from qnnbench.budget.ledger import BudgetManager
from qnnbench.core.config import BASE_BENCHMARK
from qnnbench.core.types import RunKey
from qnnbench.data.manager import make_split, make_task
from qnnbench.optim.base import Adam, GradientDescent, Momentum, QNG, SPSA, Optimiser
from qnnbench.train.trainer import train


def get_optimiser(name: str) -> Optimiser:
    """Instantiate optimizer with frozen default hyperparameters."""
    if name == "gd":
        return GradientDescent(lr=0.1)
    elif name == "momentum":
        return Momentum(lr=0.1, mu=0.9)
    elif name == "adam":
        return Adam(lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8)
    elif name == "spsa":
        return SPSA(a=0.5, c=0.15, A=8, alpha=0.602, gamma=0.101)
    elif name == "qng":
        return QNG(lr=0.1, damping=0.01, metric_inputs=8)
    raise ValueError(f"Unknown optimiser: {name}")


def run_e3(tiny: bool = False) -> list[dict[str, Any]]:
    """Execute E3 experiment runs.

    tiny: if True, runs 1 seed and 5 updates for quick verification.
    """
    X, y = make_task()
    split_seeds = [7] if tiny else BASE_BENCHMARK["replication"]["split_seeds"]
    init_seeds = [0] if tiny else BASE_BENCHMARK["replication"]["init_seeds"]
    optimisers = ["gd", "momentum", "adam", "spsa", "qng"]
    max_updates = 3 if tiny else 80

    results = []
    for s_seed in split_seeds:
        split = make_split(X, y, split_seed=s_seed)
        for i_seed in init_seeds:
            for opt_name in optimisers:
                opt = get_optimiser(opt_name)
                # Infinite budget for analytic run
                mgr = BudgetManager(train_shots=10**9, monitoring_shots=10**9)
                run_key = RunKey(
                    experiment="E3",
                    optimiser=opt_name,
                    split_seed=s_seed,
                    init_seed=i_seed,
                    config_hash="analytic_base",
                )
                res = train(
                    run_key=run_key,
                    split=split,
                    optimiser=opt,
                    budget_mgr=mgr,
                    shots=None,  # Analytic execution
                    batch_size=len(split.train_X),  # Full-batch
                    max_updates=max_updates,
                    eval_every=5 if not tiny else 1,
                )
                results.append({
                    "run_id": run_key.run_id(),
                    "optimiser": opt_name,
                    "split_seed": s_seed,
                    "init_seed": i_seed,
                    "best_val_mse": res.best_val_mse,
                    "best_val_acc": res.best_val_acc,
                    "test_mse": res.test_mse,
                    "test_acc": res.test_acc,
                    "updates_completed": res.updates_completed,
                })
    return results
