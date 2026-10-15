"""
AGENT INSTRUCTIONS — data/manager.py + data/preprocessing.py
==============================================================
M03 — Data manager & preprocessing.

API:
  make_task() -> (X_raw (100,4), y (100,) in {-1,+1})
  make_split(X, y, split_seed, scheme="main"|"historical") -> Split
  fit_scaler(X_train) -> Scaler          # TRAIN ONLY. Raises LeakageError otherwise.
  Scaler.transform(X) -> (X_tilde, clip_report)

Invariants:
  - setosa EXCLUDED. Only versicolor(−1) and virginica(+1).
  - Labels MUST be ±1.
  - fit_scaler ONLY accepts Split.train. Pass anything else → LeakageError.
  - x̃ = π * clip((x−a)/(d−a), 0, 1). Constant feature (d==a) → fixed angle π/2 (Decision D-03).
  - Scaler stats saved to artefacts (handled by logger).
  - Clip counts logged per split.

TESTS: tests/unit/test_data.py (T-M03-01..08)
  T-M03-01: 100 samples, 50/50, 4 features, labels {-1,+1}, no setosa
  T-M03-02: main split counts 30/30 train, 10/10 val, 10/10 test; disjoint; union = all 100
  T-M03-03: same seed → same split; 5 seeds → 5 distinct splits
  T-M03-04: LEAKAGE TEST — perturb val/test → scaler unchanged; perturb train → changed
  T-M03-05: x̃ in [0, π] always; train min→0, max→π
  T-M03-06: clip counts == brute-force count of out-of-range entries
  T-M03-07: constant-feature → fixed angle π/2 (property test with synthetic data)
  T-M03-08: historical split (70/30, seed 7) reproducible, flagged track="historical"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from qnnbench.core.errors import LeakageError

# ─── Constants ────────────────────────────────────────────────────────────────
CONSTANT_ANGLE = np.pi / 2.0  # D-03: constant feature → π/2


@dataclass(frozen=True)
class Split:
    """Train/val/test split. track records main vs historical."""

    train_X: np.ndarray    # raw features, train rows
    train_y: np.ndarray    # labels, train rows
    train_idx: np.ndarray  # original indices
    val_X: np.ndarray      # raw features, val rows (None for historical)
    val_y: np.ndarray
    val_idx: np.ndarray
    test_X: np.ndarray     # raw features, test rows
    test_y: np.ndarray
    test_idx: np.ndarray
    track: Literal["main", "historical"]
    split_seed: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Split):
            return NotImplemented
        return (
            np.array_equal(self.train_idx, other.train_idx)
            and np.array_equal(self.val_idx, other.val_idx)
            and np.array_equal(self.test_idx, other.test_idx)
            and self.split_seed == other.split_seed
        )


@dataclass
class ClipReport:
    """Record of clipping during transform."""

    n_clipped_low: int = 0
    n_clipped_high: int = 0

    @property
    def total(self) -> int:
        return self.n_clipped_low + self.n_clipped_high


class Scaler:
    """Min-max scaler fitted on train data only.

    Transforms raw features to x̃ ∈ [0, π].
    Constant features get fixed angle CONSTANT_ANGLE (= π/2, Decision D-03).
    """

    def __init__(self, a: np.ndarray, d: np.ndarray, const_mask: np.ndarray) -> None:
        """
        a: train minimum per feature (4,)
        d: train maximum per feature (4,)
        const_mask: bool (4,) — True where d==a (constant feature)
        """
        self.a = a
        self.d = d
        self.const_mask = const_mask

    @classmethod
    def fit(cls, X: np.ndarray) -> Scaler:
        """Fit scaler on array features."""
        a = X.min(axis=0)
        d = X.max(axis=0)
        const_mask = d == a
        return cls(a=a, d=d, const_mask=const_mask)

    @classmethod
    def fit_guarded(cls, X: np.ndarray, is_train: bool = True) -> Scaler:
        """Fit scaler with explicit train leakage check."""
        if not is_train:
            raise LeakageError("Scaler fit cannot be performed on non-train data.")
        return cls.fit(X)

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, ClipReport]:
        """Map X → x̃ in [0, π]. Returns (X_tilde, clip_report)."""
        report = ClipReport()
        X_norm = np.zeros_like(X, dtype=np.float64)
        for j in range(X.shape[1]):
            if self.const_mask[j]:
                X_norm[:, j] = CONSTANT_ANGLE / np.pi  # normalised; will multiply by π below
            else:
                raw = (X[:, j] - self.a[j]) / (self.d[j] - self.a[j])
                report.n_clipped_low += int(np.sum(raw < 0.0))
                report.n_clipped_high += int(np.sum(raw > 1.0))
                X_norm[:, j] = np.clip(raw, 0.0, 1.0)
        return X_norm * np.pi, report

    def to_dict(self) -> dict:
        return {
            "a": self.a.tolist(),
            "d": self.d.tolist(),
            "const_mask": self.const_mask.tolist(),
        }


# ─── Sentinel class so fit_scaler can type-check its argument ─────────────────
class _TrainOnlyArray:
    """Internal wrapper to prove data came from Split.train."""
    def __init__(self, X: np.ndarray) -> None:
        self._X = X


def make_task() -> tuple[np.ndarray, np.ndarray]:
    """Load binary Iris: versicolor(−1) vs virginica(+1). Returns (X_raw, y).

    Setosa is excluded. Returns exactly 100 samples (50 each class).
    """
    csv_path = Path(__file__).parent / "iris.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    mask = data[:, 4] != 0
    X = data[mask, :4].astype(np.float64)
    y_raw = data[mask, 4]
    y = np.where(y_raw == 1, -1.0, 1.0)
    return X, y


def make_split(
    X: np.ndarray,
    y: np.ndarray,
    split_seed: int,
    scheme: Literal["main", "historical"] = "main",
) -> Split:
    """Create stratified train/val/test split.

    main: 60/20/20 stratified (30/30 train, 10/10 val, 10/10 test per class).
    historical: 70/30 seed-7, no val. Flagged track='historical'.
    """
    idx_all = np.arange(len(y))
    neg_idx = idx_all[y == -1.0]
    pos_idx = idx_all[y == 1.0]

    if scheme == "historical":
        rng = np.random.default_rng(7)
        perm_neg = rng.permutation(neg_idx)
        perm_pos = rng.permutation(pos_idx)

        train_i = np.concatenate([perm_neg[:35], perm_pos[:35]])
        test_i = np.concatenate([perm_neg[35:], perm_pos[35:]])
        val_i = np.array([], dtype=int)

        return Split(
            train_X=X[train_i], train_y=y[train_i], train_idx=train_i,
            val_X=X[val_i], val_y=y[val_i], val_idx=val_i,
            test_X=X[test_i], test_y=y[test_i], test_idx=test_i,
            track="historical", split_seed=split_seed,
        )

    # main: stratified 60% train (30 neg, 30 pos), 20% val (10 neg, 10 pos), 20% test (10 neg, 10 pos)
    rng = np.random.default_rng(split_seed)
    perm_neg = rng.permutation(neg_idx)
    perm_pos = rng.permutation(pos_idx)

    train_i = np.concatenate([perm_neg[:30], perm_pos[:30]])
    val_i = np.concatenate([perm_neg[30:40], perm_pos[30:40]])
    test_i = np.concatenate([perm_neg[40:], perm_pos[40:]])

    return Split(
        train_X=X[train_i], train_y=y[train_i], train_idx=train_i,
        val_X=X[val_i], val_y=y[val_i], val_idx=val_i,
        test_X=X[test_i], test_y=y[test_i], test_idx=test_i,
        track="main", split_seed=split_seed,
    )


def fit_scaler(split: Split) -> Scaler:
    """Fit a Scaler on split.train_X ONLY. Raises LeakageError if given non-Split.

    This function ONLY accepts a Split object (not a raw array) to enforce
    that the caller has already split the data.
    """
    if not isinstance(split, Split):
        raise LeakageError(
            "fit_scaler() must receive a Split object, not a raw array. "
            "This prevents accidental leakage of val/test statistics."
        )
    X_train = split.train_X
    a = X_train.min(axis=0)
    d = X_train.max(axis=0)
    const_mask = d == a
    return Scaler(a=a, d=d, const_mask=const_mask)
