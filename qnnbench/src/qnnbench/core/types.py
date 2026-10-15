"""
AGENT INSTRUCTIONS — core/types.py
====================================
M01 — Config & types. This file defines the immutable core data structures.

RULES:
- All dataclasses MUST be frozen=True.
- Theta.flat() returns shape (17,): 16 angles then bias. from_flat is the inverse.
- Batch carries pre-scaled X (angle-scaled to [0,π]). X shape (m,4), y shape (m,) in {-1,+1}.
- RunKey: experiment, optimiser, split_seed, init_seed, config_hash. run_id = sha1(RunKey)[:12].
- n_total_params == 17 must be DERIVED, not hardcoded twice.

TESTS: T-M01-01 to T-M01-05 live in tests/unit/test_types.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

# ─── Fixed shape constants (derived once, reused everywhere) ──────────────────
ANGLE_SHAPE: tuple[int, int, int] = (2, 4, 2)   # (layers, wires, {RY,RZ})
N_GATE_PARAMS: int = int(np.prod(ANGLE_SHAPE))  # 16
N_TOTAL_PARAMS: int = N_GATE_PARAMS + 1         # 17 (16 angles + 1 bias)


@dataclass(frozen=True)
class Theta:
    """Immutable model parameters: circuit angles + classical bias.

    angles: shape (2,4,2) — θ[layer, wire, {0:RY, 1:RZ}]
    bias:   scalar float
    """

    angles: np.ndarray   # (2,4,2) float64
    bias: float

    def __post_init__(self) -> None:
        if self.angles.shape != ANGLE_SHAPE:
            raise ValueError(
                f"angles must have shape {ANGLE_SHAPE}, got {self.angles.shape}"
            )
        if not np.issubdtype(self.angles.dtype, np.floating):
            raise TypeError(f"angles must be float, got {self.angles.dtype}")

    def flat(self) -> np.ndarray:
        """Return flat (17,) array: 16 angles then bias."""
        return np.concatenate([self.angles.ravel(), [self.bias]])

    @staticmethod
    def from_flat(v: np.ndarray) -> Theta:
        """Reconstruct Theta from flat (17,) array."""
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (N_TOTAL_PARAMS,):
            raise ValueError(f"Expected shape ({N_TOTAL_PARAMS},), got {v.shape}")
        return Theta(angles=v[: N_GATE_PARAMS].reshape(ANGLE_SHAPE).copy(), bias=float(v[-1]))

    # numpy arrays in frozen dataclasses need custom equality / hash
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Theta):
            return NotImplemented
        return bool(np.array_equal(self.angles, other.angles)) and self.bias == other.bias

    def __hash__(self) -> int:
        return hash((self.angles.tobytes(), self.bias))


@dataclass(frozen=True)
class Batch:
    """A mini-batch of pre-scaled data.

    X: (m,4) float64 — angle-scaled features in [0,π]
    y: (m,)  float64 — labels in {-1, +1}
    idx: (m,) int    — original row indices
    """

    X: np.ndarray    # (m,4)
    y: np.ndarray    # (m,)
    idx: np.ndarray  # (m,)

    def __post_init__(self) -> None:
        if self.X.ndim != 2 or self.X.shape[1] != 4:
            raise ValueError(f"X must be (m,4), got {self.X.shape}")
        m = self.X.shape[0]
        if self.y.shape != (m,):
            raise ValueError(f"y must have shape ({m},)")
        if self.idx.shape != (m,):
            raise ValueError(f"idx must have shape ({m},)")
        if not set(np.unique(self.y)).issubset({-1.0, 1.0}):
            raise ValueError(f"Labels must be in {{-1, +1}}, got {np.unique(self.y)}")

    @property
    def m(self) -> int:
        return self.X.shape[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Batch):
            return NotImplemented
        return (
            np.array_equal(self.X, other.X)
            and np.array_equal(self.y, other.y)
            and np.array_equal(self.idx, other.idx)
        )

    def __hash__(self) -> int:
        return hash((self.X.tobytes(), self.y.tobytes(), self.idx.tobytes()))


@dataclass(frozen=True)
class RunKey:
    """Unique identifier for one training run.

    run_id = sha1(canonical_json(RunKey))[:12]
    """

    experiment: str
    optimiser: str
    split_seed: int
    init_seed: int
    config_hash: str

    def run_id(self) -> str:
        canonical = json.dumps(
            {
                "experiment": self.experiment,
                "optimiser": self.optimiser,
                "split_seed": self.split_seed,
                "init_seed": self.init_seed,
                "config_hash": self.config_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha1(canonical.encode()).hexdigest()[:12]
