"""
AGENT INSTRUCTIONS — quantum/backend.py  *** THE SINGLE EXECUTION CHOKE POINT ***
==================================================================================
M05 — Backend. ALL circuit executions go through Backend.execute(). NO exceptions.

CRITICAL RULES:
1. Backend.execute() is the ONLY place that calls a QNode/device.
2. Every call MUST emit exactly one ledger event.
3. 'reference' category calls are NEVER charged to any budget.
4. Preflight check happens BEFORE any device call.
5. A lint rule (T-M05-06) ensures no other module imports qml.QNode directly.

API:
  class Backend:
    __init__(self, device_cfg, ledger: Ledger, rng: np.random.Generator)
    execute(self, jobs: Sequence[CircuitJob], *, shots: int|None,
            category: Category, tag: str) -> np.ndarray

  CircuitJob = namedtuple('CircuitJob', ['theta_angles', 'x_tilde', 'observable'])
  Category = Literal['training', 'monitoring', 'pilot', 'reference', 'search']

Modes:
  shots=None  → analytic (default.qubit)
  shots=N     → finite-shot (default.qubit with shots=N)
  device_cfg.type='mixed' → default.mixed (noise track, optional)

Ledger accounting:
  - Each CircuitJob = 1 circuit eval
  - Batch of K jobs → K circuits × shots
  - Reference category: emits event but does NOT decrement any budget

TESTS: tests/unit/test_backend.py (T-M05-01..08)
  T-M05-01: analytic output == circuit_factory golden values
  T-M05-02: sampled mean over many reps → analytic (z-test 5σ)
  T-M05-03: same seed → identical sampled; different optimiser stream → different
  T-M05-04: batch of K jobs → ledger shows K circuits and K*N shots
  T-M05-05: preflight failure leaves device untouched (mock count=0)
  T-M05-06: AST test — no module outside backend.py imports qml.QNode/device execution
  T-M05-07: 'reference' calls leave all budgets unchanged
  T-M05-08: noise device smoke test (default.mixed) tagged track='noise'
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import pennylane as qml

from qnnbench.budget.ledger import Ledger, LedgerEvent
from qnnbench.quantum.circuit_factory import _circuit_template, _get_observable

Category = Literal["training", "monitoring", "pilot", "reference", "search"]

# Category codes that DO consume budget
_BUDGET_CATEGORIES: frozenset[str] = frozenset({"training", "monitoring", "pilot", "search"})


class CircuitJob(NamedTuple):
    """A single circuit evaluation request."""
    theta_angles: np.ndarray   # (2,4,2)
    x_tilde: np.ndarray        # (4,)
    observable: str            # 'Z0' or 'parity'


@dataclass
class DeviceConfig:
    """Configuration for the PennyLane device."""
    n_qubits: int = 4
    device_type: str = "default.qubit"  # or 'default.mixed'
    shots: int | None = None             # analytic by default


class Backend:
    """The single execution choke point for all quantum circuit evaluations.

    ALL quantum work goes through Backend.execute(). This ensures:
      - Consistent ledger accounting
      - Budget enforcement (preflight before ANY device call)
      - Deterministic shot sampling via explicit RNG
    """

    def __init__(
        self,
        device_cfg: DeviceConfig,
        ledger: Ledger,
        rng: np.random.Generator,
        run_id: str = "unknown",
    ) -> None:
        self._cfg = device_cfg
        self._ledger = ledger
        self._rng = rng
        self._run_id = run_id
        # Analytic device (shots=None)
        self._analytic_dev = qml.device(device_cfg.device_type, wires=device_cfg.n_qubits)
        self._shot_dev_cache: dict[int, qml.Device] = {}

    def _get_shot_device(self, shots: int, seed: int | None = None) -> qml.Device:
        return qml.device(
            self._cfg.device_type, wires=self._cfg.n_qubits, shots=shots, seed=seed
        )

    def execute(
        self,
        jobs: Sequence[CircuitJob],
        *,
        shots: int | None,
        category: Category,
        tag: str,
    ) -> np.ndarray:
        """Execute a batch of circuit jobs and return results as (K,) array.

        Parameters
        ----------
        jobs:     Sequence of CircuitJob (each = 1 circuit eval)
        shots:    None → analytic; int → finite-shot
        category: budget category (reference = free)
        tag:      human-readable tag for the ledger event

        Returns
        -------
        np.ndarray of shape (len(jobs),) with expectation values.
        """
        n_jobs = len(jobs)
        shots_total = n_jobs * shots if shots is not None else 0

        # ── Preflight budget check (before ANY device call) ───────────────────
        if category in _BUDGET_CATEGORIES and shots is not None:
            self._ledger.preflight(
                category=category,
                circuits=n_jobs,
                shots_per_circuit=shots,
            )

        t0 = time.perf_counter()
        results = np.empty(n_jobs, dtype=np.float64)

        for i, job in enumerate(jobs):
            obs = _get_observable(job.observable, self._cfg.n_qubits)
            if shots is None:
                dev = self._analytic_dev
            else:
                job_seed = int(self._rng.integers(0, 2**31 - 1))
                dev = self._get_shot_device(shots, seed=job_seed)

            @qml.qnode(dev, diff_method="parameter-shift" if shots is None else "best")
            def _qnode(angles, x, target_obs=obs):
                _circuit_template(angles, x, n_qubits=self._cfg.n_qubits)
                return qml.expval(target_obs)

            results[i] = float(_qnode(job.theta_angles, job.x_tilde))

        wall_s = time.perf_counter() - t0

        # ── Emit ledger event ─────────────────────────────────────────────────
        event = LedgerEvent(
            run_id=self._run_id,
            category=category,
            tag=tag,
            circuits=n_jobs,
            shots_per_circuit=shots if shots is not None else 0,
            shots_total=shots_total,
            wall_s=wall_s,
        )
        self._ledger.commit(event)

        return results
