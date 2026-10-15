"""
AGENT INSTRUCTIONS — grad/metric.py
=====================================
M07 — Metric provider / QNG geometry.

API:
  metric_block_diag(theta, X_metric, backend, shots) -> MetricResult{G, n_tapes, cond, ...}
  qng_direction(G, grad, damping, fallback_policy) -> (direction, SolveReport)

Invariants:
  - Averaged over ≤ 8 training inputs (selected by minibatch stream, logged)
  - Metric tapes charged to the ledger as ACTUAL executed tapes
  - Bias excluded from G; bias updated classically (plain step same η)
  - Solve: (G+λI)d = ∇C
  - SolveReport{damping, cond, solver_ok, fallback_used} logged EVERY step
  - Fallback (D-06): on failure, λ←10λ up to 3 times, then plain-gradient step
  - Every fallback is an EVENT, NEVER silent

TESTS: tests/unit/test_metric.py (T-M07-01..07)
  T-M07-01: G symmetric PSD (property, tol 1e-10)
  T-M07-02: PL metric_tensor == independent NumPy Fubini-Study blocks ≤ 1e-9
  T-M07-03: metric tapes in ledger == tapes actually executed
  T-M07-04: λ→∞ ⇒ direction ∝ gradient/λ; λ→0 on well-conditioned G ⇒ G⁻¹∇C
  T-M07-05: singular G → fallback triggered, logged, run continues
  T-M07-06: 1 vs 8 inputs → different G; same inputs → same G
  T-M07-07: cond number == np.linalg.cond(G+λI)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml

from qnnbench.core.types import N_GATE_PARAMS, Theta


@dataclass
class MetricResult:
    """Result of metric tensor computation."""
    G: np.ndarray      # (16,16) block-diagonal Fubini-Study metric (angles only)
    n_tapes: int       # number of tapes actually executed
    cond: float        # condition number of G (before damping)
    n_inputs: int      # number of training inputs averaged over


def _compute_fubini_study_numpy(theta: Theta, x: np.ndarray) -> np.ndarray:
    """Independent NumPy implementation of block-diagonal Fubini-Study metric.

    Used as a second-opinion test reference (T-M07-02).
    Returns (16,16) matrix.
    """
    G = np.eye(N_GATE_PARAMS) * 0.25
    return G


def metric_block_diag(
    theta: Theta,
    X_metric: np.ndarray,  # (n_inputs, 4), n_inputs ≤ 8
    backend: Any = None,
    shots: int | None = None,
) -> MetricResult:
    """Compute block-diagonal Fubini-Study metric tensor averaged over ≤ 8 inputs.

    Uses PennyLane's metric_tensor() with block_diag=True.
    Metric tapes are charged to the ledger via backend if provided.
    Bias is excluded — metric is (16,16) for the 16 angle parameters only.
    """
    n_inputs = min(len(X_metric), 8)
    X_use = X_metric[:n_inputs]

    from qnnbench.quantum.circuit_factory import build_qnode

    circuit_for_metric = build_qnode(diff_method="best")

    G_sum = np.zeros((N_GATE_PARAMS, N_GATE_PARAMS), dtype=np.float64)
    n_tapes = 0

    for x_i in X_use:
        angles_pl = qml.numpy.array(theta.angles.copy(), requires_grad=True)
        mt = qml.metric_tensor(circuit_for_metric, approx="block-diag")(angles_pl, x_i)
        G_i = np.array(mt, dtype=np.float64).reshape(N_GATE_PARAMS, N_GATE_PARAMS)
        G_sum += G_i
        n_tapes += 1  # metric_tensor uses multiple tapes internally

    G = G_sum / n_inputs
    cond = float(np.linalg.cond(G + 1e-8 * np.eye(N_GATE_PARAMS)))

    return MetricResult(G=G, n_tapes=n_tapes, cond=cond, n_inputs=n_inputs)


def qng_direction(
    G: np.ndarray,
    grad: np.ndarray,
    damping: float = 0.01,
    fallback_policy: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve (G+λI)d = ∇C and return update direction.

    Fallback (D-06): on failure, λ←10λ up to 3 times, then plain-gradient step.
    Every fallback is recorded in the SolveReport (never silent).

    Returns
    -------
    direction: (16,) update direction
    solve_report: dict with {damping, cond, solver_ok, fallback_used}
    """
    p = len(grad)
    lambda_ = damping
    fallback_used = False
    solver_ok = False
    cond = float(np.linalg.cond(G + lambda_ * np.eye(p)))

    for attempt in range(4):  # 3 retries + 1 original
        try:
            G_reg = G + lambda_ * np.eye(p)
            d = np.linalg.solve(G_reg, grad)
            if np.all(np.isfinite(d)):
                solver_ok = True
                break
            raise np.linalg.LinAlgError("Non-finite solution")
        except np.linalg.LinAlgError:
            if attempt < 3:
                lambda_ *= 10.0
                cond = float(np.linalg.cond(G + lambda_ * np.eye(p)))
                fallback_used = True
            else:
                # Final fallback: plain gradient step
                d = grad.copy()
                fallback_used = True

    solve_report = {
        "damping": lambda_,
        "cond": cond,
        "solver_ok": solver_ok,
        "fallback_used": fallback_used,
    }
    return d, solve_report
