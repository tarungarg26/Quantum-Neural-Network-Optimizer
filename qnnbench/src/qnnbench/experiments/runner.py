"""Experiment runner dispatcher for E1 through E8.

API:
  run_experiment(name: str, tiny: bool = False) -> Any
"""

from __future__ import annotations

from typing import Any

from qnnbench.experiments.e1_two_term import run_e1
from qnnbench.experiments.e2_four_term import run_e2
from qnnbench.experiments.e3_analytic import run_e3
from qnnbench.experiments.e4_equal_shot import run_e4
from qnnbench.experiments.e5_shot_noise import run_e5
from qnnbench.experiments.e6_trainability import run_e6
from qnnbench.experiments.e7_qng import run_e7
from qnnbench.experiments.e8_allocation import run_e8


def run_experiment(name: str, tiny: bool = False) -> Any:
    """Execute experiment by ID string (e.g. 'E1', 'E2', ... 'E8')."""
    name_norm = name.strip().upper()
    if name_norm == "E1":
        return run_e1(n_samples=20 if tiny else 200)
    elif name_norm == "E2":
        return run_e2(n_trials=10 if tiny else 50)
    elif name_norm == "E3":
        return run_e3(tiny=tiny)
    elif name_norm == "E4":
        return run_e4(tiny=tiny)
    elif name_norm == "E5":
        return run_e5(tiny=tiny)
    elif name_norm == "E6":
        return run_e6(tiny=tiny)
    elif name_norm == "E7":
        return run_e7(tiny=tiny)
    elif name_norm == "E8":
        return run_e8(tiny=tiny)
    else:
        raise ValueError(f"Unknown experiment: {name}. Must be E1..E8.")
