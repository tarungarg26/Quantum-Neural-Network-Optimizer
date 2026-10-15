"""
AGENT INSTRUCTIONS — core/config.py + core/hashing.py
=======================================================
M01 — Config loading and hashing.

API:
  load_config(path: str | Path) -> Config
  freeze(config: Config, path: str | Path) -> None   # writes YAML + hash
  load_frozen(path: str | Path) -> Config            # raises ConfigFrozenError on hash mismatch
  Config.validate() -> None                          # raises ValueError on invalid config

Invariants:
- config_hash = sha256(canonical_json(config)) — canonical means sorted keys, no floats that differ by repr
- Frozen configs in configs/frozen/ carry their hash embedded in YAML
- Loading a frozen config with a mismatched hash raises ConfigFrozenError

TESTS: tests/unit/test_config.py (T-M01-01..05)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qnnbench.core.errors import ConfigFrozenError

# ─── Base benchmark constants (Section 2.1 of FLIGHT_MANUAL) ─────────────────
BASE_BENCHMARK: dict[str, Any] = {
    "task": {
        "dataset": "iris",
        "classes": ["versicolor", "virginica"],
        "labels": {"versicolor": -1, "virginica": 1},
        "n_samples": 100,
    },
    "circuit": {
        "n_qubits": 4,
        "encoding": "angle_RY",
        "layers": 2,
        "rot": ["RY", "RZ"],
        "entangler": "cnot_ring",
        "readout": "Z0",
    },
    "model": {
        "score": "<Z0> + b",
        "loss": "mse_pm1",
        "param_shape": [2, 4, 2],
        "n_gate_params": 16,
        "n_total_params": 17,
    },
    "init": {
        "theta": "uniform_0_2pi",
        "b": 0.0,
        "paired_across_optimisers": True,
    },
    "splits": {
        "main": {
            "train": 0.6,
            "val": 0.2,
            "test": 0.2,
            "stratified": True,
            "expected_counts": [30, 30, 10, 10, 10, 10],
        },
        "historical": {"train": 0.7, "test": 0.3, "seed": 7},
    },
    "replication": {
        "split_seeds": [7, 17, 27, 37, 47],
        "init_seeds": [0, 1, 2],
        "runs_per_config": 15,
    },
}

VALID_OPTIMISERS = {"gd", "momentum", "adam", "spsa", "qng"}


def canonical_json(d: Any) -> str:
    """Return a canonical JSON string with sorted keys and no trailing whitespace."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(config_dict: dict[str, Any]) -> str:
    """sha256 of canonical JSON of config dict."""
    return hashlib.sha256(canonical_json(config_dict).encode()).hexdigest()


@dataclass
class Config:
    """Loaded benchmark configuration."""

    raw: dict[str, Any]
    hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.hash = config_hash(self.raw)

    def validate(self) -> None:
        """Validate config integrity. Raises ValueError on any violation."""
        raw = self.raw

        # n_total_params must be derivable, not just hardcoded
        n_gate = raw.get("model", {}).get("n_gate_params", 0)
        n_total = raw.get("model", {}).get("n_total_params", 0)
        if n_total != n_gate + 1:
            raise ValueError(
                f"n_total_params ({n_total}) must equal n_gate_params ({n_gate}) + 1 (bias)"
            )

        # split fractions must sum to 1
        main = raw.get("splits", {}).get("main", {})
        frac_sum = main.get("train", 0) + main.get("val", 0) + main.get("test", 0)
        if abs(frac_sum - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {frac_sum}")

        # must be stratified
        if not main.get("stratified", False):
            raise ValueError("Main split must be stratified=True")

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a Config object."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw=raw)
    cfg.validate()
    return cfg


def freeze(config: Config, path: str | Path) -> None:
    """Write config to YAML with embedded hash. Idempotent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_hash": config.hash, **config.raw}
    with open(path, "w") as f:
        yaml.dump(payload, f, sort_keys=True)


def load_frozen(path: str | Path) -> Config:
    """Load a frozen config, verifying its hash.

    Raises ConfigFrozenError if the file was tampered with.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    expected_hash = raw.pop("_hash", None)
    if expected_hash is None:
        raise ConfigFrozenError(str(path), "<missing>", "<not frozen>")
    cfg = Config(raw=raw)
    if cfg.hash != expected_hash:
        raise ConfigFrozenError(str(path), expected_hash, cfg.hash)
    return cfg
