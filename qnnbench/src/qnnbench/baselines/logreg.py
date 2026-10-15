"""
AGENT INSTRUCTIONS — baselines/logreg.py & baselines/mlp.py
===========================================================
M12 — Classical baselines implemented using SciPy and NumPy.
Avoids external compiled DLL blocks while maintaining identical
standard logistic regression and 1-hidden-layer MLP behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from qnnbench.data.manager import Split, fit_scaler


@dataclass
class BaselineResult:
    name: str
    train_mse: float
    val_mse: float
    test_mse: float
    train_acc: float
    val_acc: float
    test_acc: float
    params: dict[str, Any]


class SimpleLogisticRegression:
    """L2-regularized binary logistic regression via L-BFGS-B."""

    def __init__(self, C: float = 1.0, seed: int = 42) -> None:
        self.C = C
        self.seed = seed
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> SimpleLogisticRegression:
        # y in {-1, +1}
        _m, d = X.shape
        reg = 1.0 / self.C

        def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            w = params[:d]
            b = params[d]
            logits = X @ w + b
            z = -y * logits
            # Stable log(1 + exp(z))
            loss = np.mean(np.log1p(np.exp(np.clip(z, -50, 50)))) + 0.5 * reg * np.sum(w ** 2)
            sig = 1.0 / (1.0 + np.exp(np.clip(-z, -50, 50)))  # sig(-y*logits)
            grad_w = np.mean((-y * sig)[:, None] * X, axis=0) + reg * w
            grad_b = float(np.mean(-y * sig))
            return loss, np.concatenate([grad_w, [grad_b]])

        rng = np.random.default_rng(self.seed)
        init_params = rng.standard_normal(d + 1) * 0.1
        res = minimize(loss_and_grad, init_params, jac=True, method="L-BFGS-B")
        self.w = res.x[:d]
        self.b = float(res.x[d])
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        # Returns score in [-1, 1] range: tanh(logits)
        logits = X @ self.w + self.b
        return np.tanh(logits)


class SimpleMLPClassifier:
    """1-hidden-layer MLP with 8 hidden units and tanh activation."""

    def __init__(self, hidden_dim: int = 8, seed: int = 42, l2: float = 1e-4) -> None:
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.l2 = l2
        self.W1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.W2: np.ndarray | None = None
        self.b2: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> SimpleMLPClassifier:
        m, d = X.shape
        h = self.hidden_dim

        # Shapes: W1 is (d, h), b1 is (h,), W2 is (h,), b2 is scalar
        n_params = d * h + h + h + 1

        def unpack(p: np.ndarray):
            idx = 0
            W1 = p[idx: idx + d * h].reshape((d, h))
            idx += d * h
            b1 = p[idx: idx + h]
            idx += h
            W2 = p[idx: idx + h]
            idx += h
            b2 = p[idx]
            return W1, b1, W2, b2

        def loss_and_grad(p: np.ndarray) -> tuple[float, np.ndarray]:
            W1, b1, W2, b2 = unpack(p)
            z1 = X @ W1 + b1
            a1 = np.tanh(z1)
            logits = a1 @ W2 + b2

            # MSE on y in {-1, 1}
            preds = np.tanh(logits)
            res = preds - y
            dpreds = 1.0 - preds ** 2
            loss = np.mean(res ** 2) + self.l2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

            delta = (2.0 / m) * res * dpreds  # (m,)
            grad_W2 = a1.T @ delta + 2.0 * self.l2 * W2
            grad_b2 = np.sum(delta)

            delta1 = (delta[:, None] * W2) * (1.0 - a1 ** 2)  # (m, h)
            grad_W1 = X.T @ delta1 + 2.0 * self.l2 * W1
            grad_b1 = np.sum(delta1, axis=0)

            grad = np.concatenate([grad_W1.ravel(), grad_b1, grad_W2, [grad_b2]])
            return loss, grad

        rng = np.random.default_rng(self.seed)
        init_p = rng.standard_normal(n_params) * 0.2
        res = minimize(loss_and_grad, init_p, jac=True, method="L-BFGS-B", options={"maxiter": 200})
        self.W1, self.b1, self.W2, self.b2 = unpack(res.x)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        a1 = np.tanh(X @ self.W1 + self.b1)
        return np.tanh(a1 @ self.W2 + self.b2)


def run_logreg(split: Split, seed: int = 42) -> BaselineResult:
    """Run logistic regression on the Iris task using angle-scaled features."""
    scaler = fit_scaler(split)
    train_X, _ = scaler.transform(split.train_X)
    val_X, _ = scaler.transform(split.val_X) if len(split.val_X) > 0 else (np.empty((0, 4)), None)
    test_X, _ = scaler.transform(split.test_X)

    clf = SimpleLogisticRegression(C=1.0, seed=seed)
    clf.fit(train_X, split.train_y)

    train_pred = clf.predict_score(train_X)
    val_pred = clf.predict_score(val_X) if len(val_X) > 0 else np.array([])
    test_pred = clf.predict_score(test_X)

    train_mse = float(np.mean((train_pred - split.train_y) ** 2))
    val_mse = float(np.mean((val_pred - split.val_y) ** 2)) if len(val_X) > 0 else 0.0
    test_mse = float(np.mean((test_pred - split.test_y) ** 2))

    train_acc = float(np.mean(np.sign(train_pred) == split.train_y))
    val_acc = float(np.mean(np.sign(val_pred) == split.val_y)) if len(val_X) > 0 else 0.0
    test_acc = float(np.mean(np.sign(test_pred) == split.test_y))

    return BaselineResult(
        name="logreg",
        train_mse=train_mse,
        val_mse=val_mse,
        test_mse=test_mse,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        params={"C": clf.C},
    )


def run_mlp(split: Split, seed: int = 42) -> BaselineResult:
    """Run 1-hidden-layer MLP on the Iris task."""
    scaler = fit_scaler(split)
    train_X, _ = scaler.transform(split.train_X)
    val_X, _ = scaler.transform(split.val_X) if len(split.val_X) > 0 else (np.empty((0, 4)), None)
    test_X, _ = scaler.transform(split.test_X)

    mlp = SimpleMLPClassifier(hidden_dim=8, seed=seed)
    mlp.fit(train_X, split.train_y)

    train_pred = mlp.predict_score(train_X)
    val_pred = mlp.predict_score(val_X) if len(val_X) > 0 else np.array([])
    test_pred = mlp.predict_score(test_X)

    train_mse = float(np.mean((train_pred - split.train_y) ** 2))
    val_mse = float(np.mean((val_pred - split.val_y) ** 2)) if len(val_X) > 0 else 0.0
    test_mse = float(np.mean((test_pred - split.test_y) ** 2))

    train_acc = float(np.mean(np.sign(train_pred) == split.train_y))
    val_acc = float(np.mean(np.sign(val_pred) == split.val_y)) if len(val_X) > 0 else 0.0
    test_acc = float(np.mean(np.sign(test_pred) == split.test_y))

    return BaselineResult(
        name="mlp",
        train_mse=train_mse,
        val_mse=val_mse,
        test_mse=test_mse,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        params={"hidden_layer_sizes": (8,)},
    )
