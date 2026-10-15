"""
AGENT INSTRUCTIONS — io/logger.py & io/schemas.py
=================================================
M13 — IO, logger, schemas.

Handles writing:
- trace.csv
- events.jsonl
- final.json
- arrays.npz
- config.yaml
- env.json (live runtime versions)
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import sklearn
import yaml

TRACE_COLUMNS = [
    "run_id",
    "update",
    "circuits_train",
    "shots_train",
    "shots_monitor",
    "shots_pilot",
    "wall_s",
    "train_mse",
    "val_mse",
    "val_acc",
    "grad_norm",
    "step_norm",
    "lr_eff",
    "spsa_a_t",
    "spsa_c_t",
    "qng_damping",
    "qng_cond",
    "qng_fallback",
    "clipped_val",
    "clipped_test",
    "track",
]


def get_env_manifest() -> dict[str, str]:
    """Read live library versions from the active runtime environment."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pennylane": qml.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def write_run_artefacts(
    run_dir: str | Path,
    run_id: str,
    trace_rows: list[dict[str, Any]],
    events_jsonl: str,
    final_dict: dict[str, Any],
    arrays_dict: dict[str, np.ndarray],
    config_dict: dict[str, Any],
) -> None:
    """Atomically write all run artefacts to run_dir."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # 1. env.json
    env_info = get_env_manifest()
    with open(run_path / "env.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # 2. config.yaml
    with open(run_path / "config.yaml", "w") as f:
        yaml.dump(config_dict, f, sort_keys=True)

    # 3. events.jsonl
    with open(run_path / "events.jsonl", "w") as f:
        f.write(events_jsonl)

    # 4. final.json
    final_dict["run_id"] = run_id
    with open(run_path / "final.json", "w") as f:
        json.dump(final_dict, f, indent=2)

    # 5. arrays.npz
    np.savez(run_path / "arrays.npz", **arrays_dict)

    # 6. trace.csv
    with open(run_path / "trace.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in trace_rows:
            r["run_id"] = run_id
            writer.writerow(r)
