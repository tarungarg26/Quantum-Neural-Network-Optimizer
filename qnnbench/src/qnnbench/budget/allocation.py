"""
AGENT INSTRUCTIONS — budget/allocation.py
=========================================
M10 — Allocation policies.

API:
  UniformAllocation(shots: int)
  VarianceInformedAllocation(pilot: int = 16, floor: float = 1e-4, min_prod: int = 8, S_prod: int = 1000)

RULES:
- Uniform: fixed N shots for all circuits.
- VarianceInformedAllocation:
  1. Pilot shots: 16 pilot shots per shifted circuit.
  2. sigma_j = sqrt(max(1 - f_hat_j^2, floor)).
  3. Weight a_j = 0.5 * |r_i| / m.
  4. N_j = S_prod * (|a_j| * sigma_j) / sum_r(|a_r| * sigma_r).
  5. Floor: N_j >= 8.
  6. Largest-remainder rounding to ensure sum(N_j) == S_prod exactly.
  7. Pilots charged to ledger under category 'pilot', NEVER mixed into estimator.

TESTS: tests/unit/test_allocation.py (T-M10-01..08)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ShotPlan:
    """Shot allocation plan for a collection of circuit jobs."""
    shots_per_job: list[int]
    is_uniform: bool
    total_shots: int

    def __post_init__(self) -> None:
        if self.total_shots != sum(self.shots_per_job):
            object.__setattr__(self, "total_shots", sum(self.shots_per_job))


class AllocationPolicy(Protocol):
    def plan(self, n_jobs: int, weights: np.ndarray | None = None, pilots: np.ndarray | None = None) -> ShotPlan:
        ...


class UniformAllocation:
    """Fixed N shots across all circuits."""

    def __init__(self, shots: int = 32) -> None:
        self.shots = shots

    def plan(self, n_jobs: int, weights: np.ndarray | None = None, pilots: np.ndarray | None = None) -> ShotPlan:
        alloc = [self.shots] * n_jobs
        return ShotPlan(shots_per_job=alloc, is_uniform=True, total_shots=self.shots * n_jobs)


class VarianceInformedAllocation:
    """Adaptive variance-informed shot allocation using largest-remainder rounding."""

    def __init__(
        self,
        pilot: int = 16,
        floor: float = 1e-4,
        min_prod: int = 8,
        S_prod: int = 1056,
    ) -> None:
        self.pilot = pilot
        self.floor = floor
        self.min_prod = min_prod
        self.S_prod = S_prod

    def plan(
        self,
        n_jobs: int,
        weights: np.ndarray | None = None,
        pilots: np.ndarray | None = None,
    ) -> ShotPlan:
        """Compute integer allocation summing exactly to S_prod."""
        if weights is None:
            weights = np.ones(n_jobs)
        if pilots is None:
            pilots = np.zeros(n_jobs)

        # 1. Pilot std estimates
        # sigma_j = sqrt(max(1 - f_hat^2, floor))
        sigmas = np.sqrt(np.maximum(1.0 - pilots ** 2, self.floor))
        numerators = np.abs(weights) * sigmas
        denom = float(np.sum(numerators))
        if denom < 1e-12:
            raw_shares = np.full(n_jobs, 1.0 / n_jobs)
        else:
            raw_shares = numerators / denom

        # Ensure budget allows min_prod
        min_total = n_jobs * self.min_prod
        if self.S_prod < min_total:
            raise ValueError(f"S_prod ({self.S_prod}) must be at least {min_total} to satisfy min_prod={self.min_prod}")

        remaining_pool = self.S_prod - min_total
        continuous_alloc = self.min_prod + remaining_pool * raw_shares

        # 2. Largest-remainder rounding
        floored = np.floor(continuous_alloc).astype(int)
        remainders = continuous_alloc - floored
        surplus = self.S_prod - int(np.sum(floored))

        # Tie-breaking: sorted by remainder descending, then index ascending for determinism
        order = np.lexsort((np.arange(n_jobs), -remainders))
        alloc = floored.copy()
        for idx in order[:surplus]:
            alloc[idx] += 1

        alloc_list = alloc.tolist()
        return ShotPlan(
            shots_per_job=alloc_list,
            is_uniform=False,
            total_shots=int(np.sum(alloc)),
        )
