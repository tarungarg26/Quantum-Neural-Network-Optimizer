"""E4 Experiment: Equal-Shot Ranking.

Evaluates GD, Momentum, Adam, SPSA, and QNG under fixed finite-shot budgets
B in {1e5, 5e5, 1e6, 5e6} and per-circuit shot allocations N in {32, 128, 512}.
"""

from __future__ import annotations

from typing import Any

from qnnbench.budget.allocation import UniformAllocation
from qnnbench.budget.ledger import BudgetManager
from qnnbench.core.config import BASE_BENCHMARK
from qnnbench.core.types import RunKey
from qnnbench.data.manager import make_split, make_task
from qnnbench.experiments.e3_analytic import get_optimiser
from qnnbench.train.trainer import train


def run_e4(tiny: bool = False) -> list[dict[str, Any]]:
    """Execute E4 equal-shot experiment runs.

    tiny: if True, runs 1 budget, 1 shot-count, 1 seed, 5 updates.
    """
    X, y = make_task()
    budgets = [10000] if tiny else [100000, 500000, 1000000, 5000000]
    shot_counts = [32] if tiny else [32, 128, 512]
    split_seeds = [7] if tiny else BASE_BENCHMARK["replication"]["split_seeds"]
    init_seeds = [0] if tiny else BASE_BENCHMARK["replication"]["init_seeds"]
    optimisers = ["gd", "momentum", "adam", "spsa", "qng"]
    max_updates = 5 if tiny else 150

    results = []
    for budget in budgets:
        # Decision D-01: monitoring budget = 20% of training budget
        mon_budget = int(budget * 0.20)
        for shots in shot_counts:
            for s_seed in split_seeds:
                split = make_split(X, y, split_seed=s_seed)
                for i_seed in init_seeds:
                    for opt_name in optimisers:
                        opt = get_optimiser(opt_name)
                        mgr = BudgetManager(train_shots=budget, monitoring_shots=mon_budget)
                        alloc = UniformAllocation(shots=shots)
                        run_key = RunKey(
                            experiment="E4",
                            optimiser=opt_name,
                            split_seed=s_seed,
                            init_seed=i_seed,
                            config_hash=f"e4_b{budget}_s{shots}",
                        )
                        res = train(
                            run_key=run_key,
                            split=split,
                            optimiser=opt,
                            budget_mgr=mgr,
                            allocation=alloc,
                            shots=shots,
                            batch_size=10,
                            max_updates=max_updates,
                            eval_every=5,
                        )
                        results.append({
                            "run_id": run_key.run_id(),
                            "experiment": "E4",
                            "budget": budget,
                            "shots": shots,
                            "optimiser": opt_name,
                            "split_seed": s_seed,
                            "init_seed": i_seed,
                            "best_val_mse": res.best_val_mse,
                            "best_val_acc": res.best_val_acc,
                            "test_mse": res.test_mse,
                            "test_acc": res.test_acc,
                            "updates_completed": res.updates_completed,
                            "status": res.status,
                            "total_train_shots_spent": mgr.spent("training"),
                            "total_mon_shots_spent": mgr.spent("monitoring"),
                        })
    return results
