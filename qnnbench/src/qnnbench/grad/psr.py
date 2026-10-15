"""
AGENT INSTRUCTIONS — grad/psr.py + grad/reference.py + grad/loss.py
=====================================================================
M06 — Gradient engines.

2-term PSR (psr2): ∂f/∂θₖ = ½[f(θ+π/2 eₖ) − f(θ−π/2 eₖ)]
4-term PSR (psr4): for controlled rotations with generator gaps {½,1}
  c± = (√2 ± 1)/(4√2); shifts ±π/2, ±3π/2
Loss:
  mse_loss(f, b, y) = mean((f+b-y)²)
  unbiased_mse_loss(f_hat, b, y, N): (f̂+b−y)² − (1−f̂²)/(N−1), N>1
  mse_grad: ∂C/∂θk = (2/m)Σ(f+b−y)·∂f/∂θk; ∂C/∂b = (2/m)Σ(f+b−y)

Key invariants:
  - PSR applied ONLY to circuit expectation f; chain rule is classical
  - In finite-shot: residual f̂ and each shifted f̂± are INDEPENDENT executions
  - Cost per update = (2p+1)m where p=16 (bias NOT a gate parameter)
  - PSR/SPSA cost ratio = 16.5 (NOT 17 — bias excluded from p)
  - Non-finite input → NonFiniteError (NEVER silently drop)

TESTS: tests/numeric/test_gradients.py (T-M06-01..12)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qnnbench.core.errors import NonFiniteError
from qnnbench.core.types import ANGLE_SHAPE, N_GATE_PARAMS, Batch, Theta

if TYPE_CHECKING:
    from qnnbench.quantum.backend import Backend

# ─── PSR constants ────────────────────────────────────────────────────────────
PSR2_SHIFT = np.pi / 2.0                    # 2-term shift
PSR4_C_PLUS = (np.sqrt(2) + 1) / (4 * np.sqrt(2))   # 4-term coefficient c+
PSR4_C_MINUS = (np.sqrt(2) - 1) / (4 * np.sqrt(2))  # 4-term coefficient c-
PSR4_SHIFTS = [np.pi / 2, -np.pi / 2, 3 * np.pi / 2, -3 * np.pi / 2]


def _check_finite(arr: np.ndarray, location: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise NonFiniteError(location, arr)


def psr2(
    theta: Theta,
    batch: Batch,
    backend: Backend,
    shots: int | None = None,
    category: str = "training",
    tag: str = "psr2",
) -> tuple[np.ndarray, np.ndarray]:
    """2-term parameter-shift rule for all 16 angle parameters.

    Returns
    -------
    f:   (m,)       circuit expectations at theta
    df:  (m, p)     partial derivatives, p = N_GATE_PARAMS = 16

    Cost: (2p+1)·m circuit evaluations (p=16, so 33·m).
    The bias is NOT a gate parameter — its gradient is computed classically.
    """
    from qnnbench.quantum.backend import CircuitJob

    m = batch.m
    observable = "Z0"

    # ── Forward pass: f(θ) for all m samples ─────────────────────────────────
    fwd_jobs = [
        CircuitJob(theta_angles=theta.angles, x_tilde=batch.X[i], observable=observable)
        for i in range(m)
    ]
    f = backend.execute(fwd_jobs, shots=shots, category=category, tag=f"{tag}_fwd")
    _check_finite(f, "psr2/f")

    # ── PSR: 2 shifted evals per parameter per sample ─────────────────────────
    df = np.zeros((m, N_GATE_PARAMS), dtype=np.float64)
    theta.flat()[:N_GATE_PARAMS]  # (16,) angles only

    for k in range(N_GATE_PARAMS):
        # Plus shift
        angles_plus = theta.angles.copy()
        idx = np.unravel_index(k, ANGLE_SHAPE)
        angles_plus[idx] += PSR2_SHIFT
        plus_jobs = [
            CircuitJob(angles_plus, batch.X[i], observable) for i in range(m)
        ]
        f_plus = backend.execute(plus_jobs, shots=shots, category=category, tag=f"{tag}_k{k}_p")

        # Minus shift
        angles_minus = theta.angles.copy()
        angles_minus[idx] -= PSR2_SHIFT
        minus_jobs = [
            CircuitJob(angles_minus, batch.X[i], observable) for i in range(m)
        ]
        f_minus = backend.execute(minus_jobs, shots=shots, category=category, tag=f"{tag}_k{k}_m")

        df[:, k] = 0.5 * (f_plus - f_minus)

    _check_finite(df, "psr2/df")
    return f, df


def psr4(
    theta: np.ndarray,
    k: int,
    fn: Backend,
    x: np.ndarray,
    shots: int | None = None,
    category: str = "training",
    tag: str = "psr4",
) -> float:
    """4-term parameter-shift rule for a single parameter k.

    For controlled rotations with generator gaps {½, 1}.
    c± = (√2 ± 1)/(4√2)
    f′ = c+[f(θ+π/2)−f(θ−π/2)] − c−[f(θ+3π/2)−f(θ−3π/2)]

    Parameters
    ----------
    theta: flat parameter array
    k:     parameter index to differentiate
    fn:    callable that evaluates the circuit (a Backend-like object)
    x:     input (n_qubits,)
    """
    results = []
    for shift in PSR4_SHIFTS:
        theta_s = theta.copy()
        theta_s[k] += shift
        results.append(float(fn(theta_s, x)))

    f_pp, f_pm, f_3p, f_3m = results
    return PSR4_C_PLUS * (f_pp - f_pm) - PSR4_C_MINUS * (f_3p - f_3m)


def autodiff_reference(
    theta: Theta,
    batch: Batch,
    backend: Backend,
    tag: str = "autodiff_ref",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient via PennyLane backprop (analytic reference only).

    IMPORTANT: This uses category='reference' → NEVER charged to budget.
    Returned f,df labelled as analytic reference.
    Only call this for verification (E1/E2), NOT for training.
    """
    import pennylane as qml

    from qnnbench.quantum.circuit_factory import build_qnode

    qnode = build_qnode(diff_method="backprop")

    m = batch.m
    f = np.array([qnode(theta.angles, batch.X[i]) for i in range(m)])
    df = np.zeros((m, N_GATE_PARAMS), dtype=np.float64)

    for i in range(m):
        angles_pl = qml.numpy.array(theta.angles.copy(), requires_grad=True)
        x_i = batch.X[i]
        loss_fn = lambda a, inp=x_i: qnode(a, inp)
        grad_fn = qml.grad(loss_fn, argnums=0)
        g = grad_fn(angles_pl)
        df[i] = g.ravel()

    return f, df


# ─── Loss functions ────────────────────────────────────────────────────────────

def mse_loss(f: np.ndarray, b: float, y: np.ndarray) -> float:
    """MSE loss: mean((f+b−y)²)."""
    _check_finite(f, "mse_loss/f")
    residuals = f + b - y
    _check_finite(residuals, "mse_loss/residuals")
    return float(np.mean(residuals ** 2))


def unbiased_mse_loss(
    f_hat: np.ndarray, b: float, y: np.ndarray, N: int
) -> float:
    """Unbiased finite-shot MSE loss.

    ℓ̂ = (f̂+b−y)² − (1−f̂²)/(N−1),  N > 1

    The second term corrects for the positive bias introduced by shot noise
    in the squared expectation.
    """
    if N <= 1:
        raise NonFiniteError("unbiased_mse_loss/N", N)
    residuals = f_hat + b - y
    bias_correction = (1.0 - f_hat ** 2) / (N - 1)
    return float(np.mean(residuals ** 2 - bias_correction))


def mse_grad(
    f: np.ndarray,
    df: np.ndarray,
    b: float,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Classical chain rule for MSE gradient.

    ∂C/∂θk = (2/m) Σᵢ (fᵢ+b−yᵢ) · ∂fᵢ/∂θk  — quantum contribution
    ∂C/∂b  = (2/m) Σᵢ (fᵢ+b−yᵢ)              — classical, NO quantum shifts for b

    Returns
    -------
    grad_theta: (p,) gradient w.r.t. angle parameters
    grad_b:     scalar gradient w.r.t. bias
    """
    m = len(f)
    residuals = f + b - y  # (m,)
    _check_finite(residuals, "mse_grad/residuals")
    _check_finite(df, "mse_grad/df")

    grad_theta = (2.0 / m) * df.T @ residuals   # (p,)
    grad_b = (2.0 / m) * np.sum(residuals)       # scalar

    _check_finite(grad_theta, "mse_grad/grad_theta")
    return grad_theta, grad_b
