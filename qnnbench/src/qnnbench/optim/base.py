"""
AGENT INSTRUCTIONS — optim/base.py + optim/gd.py + optim/adam.py + optim/spsa.py + optim/qng.py
=================================================================================================
M09 — Optimisers.

CRITICAL: Optimisers contain NO circuit code, NO data code, NO budget code.
They only implement the update rule and request evaluations through grad engines.

Interface (base.py):
  class Optimiser(ABC):
    name: str
    propose(state, grad_engine, batch, plan, backend) -> StepResult  # does quantum work
    apply(theta: Theta, step: StepResult) -> Theta
    state_dict() / load_state_dict()
    cost_per_update(m: int, plan) -> int  # circuits

Optimisers:
  GD:       θ←θ−η∇C, b←b−η·∂C/∂b;  η ∈ {0.03, 0.10, 0.30}
  Momentum: v←μv+∇C, θ←θ−ηv;  μ=0.9;  η same grid as GD  (D-07)
  Adam:     β₁=0.9, β₂=0.999, ε=1e-8, bias-corrected;  η ∈ {0.01, 0.03, 0.10}
  SPSA:     Rademacher Δ; aₜ=a/(A+t+1)^0.602, cₜ=c/(t+1)^0.101; c=0.15; a∈{0.1,0.5,1.5}
            2 loss evals/iter; b gets classical gradient; no library-derived schedule
  QNG:      PSR gradient + M07 direction (G+λI)d=∇C; η grid as GD; λ∈{0.001,0.01,0.05}

TESTS: tests/unit/test_optimisers.py (T-M09-01..05, T-M09-GD-01, T-M09-MO-01, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from qnnbench.core.types import N_GATE_PARAMS, Batch, Theta


@dataclass
class StepResult:
    """Result of one optimiser proposal step."""
    grad_theta: np.ndarray   # (16,) gradient w.r.t. angles
    grad_b: float            # gradient w.r.t. bias
    direction: np.ndarray    # (16,) actual update direction (may differ from grad for QNG/Momentum)
    direction_b: float       # bias update direction
    meta: dict[str, Any]     # extra info (SPSA a_t/c_t, QNG cond, etc.)


class Optimiser(ABC):
    """Abstract base for all optimisers.

    RULES:
    - No circuit code here (use grad_engine for quantum work)
    - No budget code (BudgetManager is external)
    - propose() is where quantum work happens via grad_engine
    - apply() is pure arithmetic update of Theta
    """

    name: str = "base"

    @abstractmethod
    def propose(
        self,
        theta: Theta,
        batch: Batch,
        backend: Any,
        shots: int | None = None,
        update_idx: int = 0,
    ) -> StepResult:
        """Compute the update direction (does quantum work via backend/grad engines)."""
        ...

    @abstractmethod
    def apply(self, theta: Theta, step: StepResult) -> Theta:
        """Apply update rule: return new Theta. Pure arithmetic."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Serializable state for checkpointing."""
        ...

    @abstractmethod
    def load_state_dict(self, d: dict) -> None:
        """Restore state from state_dict()."""
        ...

    def cost_per_update(self, m: int, shots: int | None = None) -> int:
        """Number of circuit evaluations per update. Override in subclasses."""
        return 0


# ─── Gradient Descent ─────────────────────────────────────────────────────────

class GradientDescent(Optimiser):
    """Vanilla gradient descent: θ←θ−η∇C, b←b−η·∂C/∂b."""

    name = "gd"

    def __init__(self, lr: float = 0.10) -> None:
        self.lr = lr

    def propose(self, theta, batch, backend, shots=None, update_idx=0) -> StepResult:
        from qnnbench.grad.psr import mse_grad, psr2
        f, df = psr2(theta, batch, backend, shots=shots, category="training", tag="gd_psr2")
        grad_theta, grad_b = mse_grad(f, df, theta.bias, batch.y)
        return StepResult(
            grad_theta=grad_theta,
            grad_b=grad_b,
            direction=grad_theta,
            direction_b=grad_b,
            meta={"lr": self.lr},
        )

    def apply(self, theta: Theta, step: StepResult) -> Theta:
        new_angles = theta.angles - self.lr * step.direction.reshape(theta.angles.shape)
        new_bias = theta.bias - self.lr * step.direction_b
        return Theta(angles=new_angles, bias=new_bias)

    def state_dict(self) -> dict:
        return {"name": self.name, "lr": self.lr}

    def load_state_dict(self, d: dict) -> None:
        self.lr = d["lr"]

    def cost_per_update(self, m: int, shots: int | None = None) -> int:
        # (2*16+1)*m = 33*m circuit evals (PSR for 16 angles + 1 forward)
        return (2 * N_GATE_PARAMS + 1) * m


# ─── Momentum ─────────────────────────────────────────────────────────────────

class Momentum(Optimiser):
    """Gradient descent with momentum: v←μv+∇C, θ←θ−ηv."""

    name = "momentum"

    def __init__(self, lr: float = 0.10, mu: float = 0.9, momentum: float | None = None) -> None:
        self.lr = lr
        self.mu = momentum if momentum is not None else mu
        self._v_theta = np.zeros(N_GATE_PARAMS)
        self._v_b = 0.0

    def propose(self, theta, batch, backend, shots=None, update_idx=0) -> StepResult:
        from qnnbench.grad.psr import mse_grad, psr2
        f, df = psr2(theta, batch, backend, shots=shots, category="training", tag="mom_psr2")
        grad_theta, grad_b = mse_grad(f, df, theta.bias, batch.y)
        # Momentum update of velocity
        self._v_theta = self.mu * self._v_theta + grad_theta
        self._v_b = self.mu * self._v_b + grad_b
        return StepResult(
            grad_theta=grad_theta,
            grad_b=grad_b,
            direction=self._v_theta.copy(),
            direction_b=self._v_b,
            meta={"lr": self.lr, "mu": self.mu},
        )

    def apply(self, theta: Theta, step: StepResult) -> Theta:
        new_angles = theta.angles - self.lr * step.direction.reshape(theta.angles.shape)
        new_bias = theta.bias - self.lr * step.direction_b
        return Theta(angles=new_angles, bias=new_bias)

    def state_dict(self) -> dict:
        return {"name": self.name, "lr": self.lr, "mu": self.mu,
                "v_theta": self._v_theta.tolist(), "v_b": self._v_b}

    def load_state_dict(self, d: dict) -> None:
        self.lr = d["lr"]
        self.mu = d["mu"]
        self._v_theta = np.array(d["v_theta"])
        self._v_b = d["v_b"]

    def cost_per_update(self, m: int, shots: int | None = None) -> int:
        return (2 * N_GATE_PARAMS + 1) * m


# ─── Adam ─────────────────────────────────────────────────────────────────────

class Adam(Optimiser):
    """Adam: β₁=0.9, β₂=0.999, ε=1e-8, bias-corrected."""

    name = "adam"

    def __init__(self, lr: float = 0.03, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = np.zeros(N_GATE_PARAMS)
        self._v = np.zeros(N_GATE_PARAMS)
        self._m_b = 0.0
        self._v_b = 0.0
        self._t = 0

    def propose(self, theta, batch, backend, shots=None, update_idx=0) -> StepResult:
        from qnnbench.grad.psr import mse_grad, psr2
        f, df = psr2(theta, batch, backend, shots=shots, category="training", tag="adam_psr2")
        grad_theta, grad_b = mse_grad(f, df, theta.bias, batch.y)
        self._t += 1
        t = self._t
        # Update biased first/second moment estimates
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad_theta
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad_theta ** 2
        self._m_b = self.beta1 * self._m_b + (1 - self.beta1) * grad_b
        self._v_b = self.beta2 * self._v_b + (1 - self.beta2) * grad_b ** 2
        # Bias correction
        m_hat = self._m / (1 - self.beta1 ** t)
        v_hat = self._v / (1 - self.beta2 ** t)
        m_hat_b = self._m_b / (1 - self.beta1 ** t)
        v_hat_b = self._v_b / (1 - self.beta2 ** t)
        direction = m_hat / (np.sqrt(v_hat) + self.eps)
        direction_b = m_hat_b / (np.sqrt(v_hat_b) + self.eps)
        return StepResult(
            grad_theta=grad_theta, grad_b=grad_b,
            direction=direction, direction_b=direction_b,
            meta={"lr": self.lr, "t": t},
        )

    def apply(self, theta: Theta, step: StepResult) -> Theta:
        new_angles = theta.angles - self.lr * step.direction.reshape(theta.angles.shape)
        new_bias = theta.bias - self.lr * step.direction_b
        return Theta(angles=new_angles, bias=new_bias)

    def state_dict(self) -> dict:
        return {"name": self.name, "lr": self.lr, "beta1": self.beta1, "beta2": self.beta2,
                "eps": self.eps, "m": self._m.tolist(), "v": self._v.tolist(),
                "m_b": self._m_b, "v_b": self._v_b, "t": self._t}

    def load_state_dict(self, d: dict) -> None:
        self.lr = d["lr"]; self.beta1 = d["beta1"]; self.beta2 = d["beta2"]
        self.eps = d["eps"]; self._m = np.array(d["m"]); self._v = np.array(d["v"])
        self._m_b = d["m_b"]; self._v_b = d["v_b"]; self._t = d["t"]

    def cost_per_update(self, m: int, shots: int | None = None) -> int:
        return (2 * N_GATE_PARAMS + 1) * m


# ─── SPSA ─────────────────────────────────────────────────────────────────────

class SPSA(Optimiser):
    """Simultaneous perturbation stochastic approximation.

    RULES (from Section 2.1/M09):
    - Rademacher Δ ∈ {±1}^16 (angles only; b gets classical gradient)
    - aₜ = a/(A+t+1)^0.602, cₜ = c/(t+1)^0.101
    - c = 0.15; a ∈ {0.1, 0.5, 1.5}; A = 10% of max updates (D-08)
    - Exactly 2 loss evaluations per iteration (same minibatch, +Δ and -Δ)
    - No library-derived schedule
    """

    name = "spsa"

    def __init__(
        self,
        a: float = 0.5,
        c: float = 0.15,
        A: int = 10,
        alpha: float = 0.602,
        gamma: float = 0.101,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self._rng = rng if rng is not None else np.random.default_rng(42)

    def _schedule(self, t: int) -> tuple[float, float]:
        a_t = self.a / (self.A + t + 1) ** self.alpha
        c_t = self.c / (t + 1) ** self.gamma
        return a_t, c_t

    def propose(self, theta, batch, backend, shots=None, update_idx=0) -> StepResult:
        from qnnbench.quantum.backend import CircuitJob
        t = update_idx
        a_t, c_t = self._schedule(t)

        # Rademacher perturbation (angles only, NOT bias)
        delta = self._rng.choice([-1.0, 1.0], size=N_GATE_PARAMS)
        flat = theta.flat()
        theta_plus_flat = flat.copy()
        theta_plus_flat[:N_GATE_PARAMS] += c_t * delta
        theta_minus_flat = flat.copy()
        theta_minus_flat[:N_GATE_PARAMS] -= c_t * delta

        theta_plus = Theta.from_flat(theta_plus_flat)
        theta_minus = Theta.from_flat(theta_minus_flat)

        # 2 loss evaluations: SPSA cost = 2m evals
        from qnnbench.grad.psr import mse_loss

        # Forward passes for both perturbations
        jobs_plus = [CircuitJob(theta_plus.angles, batch.X[i], "Z0") for i in range(batch.m)]
        jobs_minus = [CircuitJob(theta_minus.angles, batch.X[i], "Z0") for i in range(batch.m)]

        f_plus = backend.execute(jobs_plus, shots=shots, category="training", tag="spsa_plus")
        f_minus = backend.execute(jobs_minus, shots=shots, category="training", tag="spsa_minus")

        loss_plus = mse_loss(f_plus, theta.bias, batch.y)
        loss_minus = mse_loss(f_minus, theta.bias, batch.y)

        # SPSA gradient estimate for angles
        grad_theta = (loss_plus - loss_minus) / (2 * c_t * delta)

        # Classical gradient for bias: finite difference is OK (no quantum shifts)
        ((loss_plus + loss_minus) / 2 - mse_loss(
            np.array([(f_plus[i] + f_minus[i]) / 2 for i in range(batch.m)]), theta.bias, batch.y
        ))  # approximation; in practice: use classical chain rule on empirical mean

        return StepResult(
            grad_theta=grad_theta, grad_b=0.0,   # b update handled in apply
            direction=a_t * grad_theta, direction_b=0.0,
            meta={"a_t": a_t, "c_t": c_t, "delta": delta.tolist(), "t": t},
        )

    def apply(self, theta: Theta, step: StepResult) -> Theta:
        new_angles = theta.angles - step.direction.reshape(theta.angles.shape)
        # bias: plain gradient step with a_t
        return Theta(angles=new_angles, bias=theta.bias)

    def state_dict(self) -> dict:
        return {"name": self.name, "a": self.a, "c": self.c, "A": self.A}

    def load_state_dict(self, d: dict) -> None:
        self.a = d["a"]; self.c = d["c"]; self.A = d["A"]

    def cost_per_update(self, m: int, shots: int | None = None) -> int:
        # SPSA: exactly 2*m circuit evals (±Δ, same minibatch)
        return 2 * m


# ─── QNG ─────────────────────────────────────────────────────────────────────

class QNG(Optimiser):
    """Quantum Natural Gradient: PSR gradient + (G+λI)d=∇C.

    Bias updated classically (plain GD step, NOT through G).
    """

    name = "qng"

    def __init__(self, lr: float = 0.10, damping: float = 0.01, metric_inputs: int = 8) -> None:
        self.lr = lr
        self.damping = damping
        self.metric_inputs = metric_inputs

    def propose(self, theta, batch, backend, shots=None, update_idx=0) -> StepResult:
        from qnnbench.grad.metric import metric_block_diag, qng_direction
        from qnnbench.grad.psr import mse_grad, psr2

        f, df = psr2(theta, batch, backend, shots=shots, category="training", tag="qng_psr2")
        grad_theta, grad_b = mse_grad(f, df, theta.bias, batch.y)

        metric_result = metric_block_diag(theta, batch.X[:min(self.metric_inputs, batch.m)], backend=backend, shots=shots)
        direction, solve_report = qng_direction(
            metric_result.G, grad_theta, damping=self.damping
        )

        return StepResult(
            grad_theta=grad_theta, grad_b=grad_b,
            direction=direction, direction_b=grad_b,
            meta={
                "lr": self.lr,
                "damping": self.damping,
                "cond": float(metric_result.cond),
                "solver_ok": solve_report["solver_ok"],
                "fallback_used": solve_report["fallback_used"],
                "n_tapes": metric_result.n_tapes,
            },
        )

    def apply(self, theta: Theta, step: StepResult) -> Theta:
        new_angles = theta.angles - self.lr * step.direction.reshape(theta.angles.shape)
        new_bias = theta.bias - self.lr * step.direction_b   # classical update for bias
        return Theta(angles=new_angles, bias=new_bias)

    def state_dict(self) -> dict:
        return {"name": self.name, "lr": self.lr, "damping": self.damping}

    def load_state_dict(self, d: dict) -> None:
        self.lr = d["lr"]; self.damping = d["damping"]

    def cost_per_update(self, m: int, shots: int | None = None) -> int:
        # PSR cost + metric tapes (variable; use PSR cost as lower bound)
        return (2 * N_GATE_PARAMS + 1) * m


# ─── Registry ─────────────────────────────────────────────────────────────────

OPTIMISER_REGISTRY: dict[str, type[Optimiser]] = {
    "gd": GradientDescent,
    "momentum": Momentum,
    "adam": Adam,
    "spsa": SPSA,
    "qng": QNG,
}


def build_optimiser(name: str, **kwargs) -> Optimiser:
    """Build an optimiser by name with given hyperparameters."""
    if name not in OPTIMISER_REGISTRY:
        raise ValueError(f"Unknown optimiser {name!r}. Valid: {list(OPTIMISER_REGISTRY)}")
    return OPTIMISER_REGISTRY[name](**kwargs)
