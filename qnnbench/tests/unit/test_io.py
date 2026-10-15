import json
from pathlib import Path

import numpy as np

from qnnbench.io.logger import TRACE_COLUMNS, write_run_artefacts


def test_t_m13_02_and_04_env_manifest_and_run_id(tmp_path: Path):
    """T-M13-02 & T-M13-04: Artefacts written with live env manifest and consistent run_id."""
    run_id = "test_run_123"
    trace_rows = [{
        "update": 0,
        "circuits_train": 330,
        "shots_train": 3300,
        "shots_monitor": 500,
        "shots_pilot": 0,
        "wall_s": 0.12,
        "train_mse": 0.45,
        "val_mse": 0.32,
        "val_acc": 0.85,
        "grad_norm": 0.15,
        "step_norm": 0.05,
        "lr_eff": 0.1,
        "spsa_a_t": 0.0,
        "spsa_c_t": 0.0,
        "qng_damping": 0.0,
        "qng_cond": 1.0,
        "qng_fallback": False,
        "clipped_val": 0,
        "clipped_test": 0,
        "track": "main",
    }]
    events = json.dumps({"run_id": run_id, "category": "training", "circuits": 330})
    final = {"status": "completed_updates", "test_mse": 0.28, "test_acc": 0.90}
    arrays = {"theta": np.ones((2, 4, 2))}
    config = {"task": "iris"}

    write_run_artefacts(
        run_dir=tmp_path,
        run_id=run_id,
        trace_rows=trace_rows,
        events_jsonl=events,
        final_dict=final,
        arrays_dict=arrays,
        config_dict=config,
    )

    # Verify env.json contains live python and pennylane
    with open(tmp_path / "env.json") as f:
        env_data = json.load(f)
    assert "pennylane" in env_data
    assert "python" in env_data

    # Verify final.json has matching run_id
    with open(tmp_path / "final.json") as f:
        final_data = json.load(f)
    assert final_data["run_id"] == run_id

    # Verify trace.csv has run_id and correct columns
    with open(tmp_path / "trace.csv") as f:
        header = f.readline().strip().split(",")
    assert header == TRACE_COLUMNS

    # Verify arrays.npz
    loaded_arrays = np.load(tmp_path / "arrays.npz")
    assert np.allclose(loaded_arrays["theta"], arrays["theta"])
