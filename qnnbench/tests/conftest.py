"""
conftest.py — shared pytest fixtures for all tests.

AGENT: Add fixtures here that are used across multiple test files.
Do NOT put experiment-specific fixtures here.
"""
from __future__ import annotations

import numpy as np
import pytest

from qnnbench.budget.ledger import BudgetManager, Ledger
from qnnbench.core.seeds import PurposeID, stream
from qnnbench.core.types import ANGLE_SHAPE, Batch, Theta
from qnnbench.data.manager import fit_scaler, make_split, make_task

# ─── RNG fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def seed_stream():
    return stream(split_seed=7, init_seed=0, purpose=PurposeID.INIT)


# ─── Model parameter fixtures ─────────────────────────────────────────────────

@pytest.fixture
def random_theta(rng):
    angles = rng.uniform(0, 2 * np.pi, ANGLE_SHAPE)
    return Theta(angles=angles, bias=0.0)


@pytest.fixture
def zero_theta():
    return Theta(angles=np.zeros(ANGLE_SHAPE), bias=0.0)


# ─── Data fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def iris_task():
    X_raw, y = make_task()
    return X_raw, y


@pytest.fixture(scope="session")
def iris_split(iris_task):
    X_raw, y = iris_task
    return make_split(X_raw, y, split_seed=7, scheme="main")


@pytest.fixture(scope="session")
def iris_scaler(iris_split):
    return fit_scaler(iris_split)


@pytest.fixture
def small_batch(iris_split, iris_scaler):
    """A small (10-sample) batch of scaled training data."""
    split = iris_split
    scaler = iris_scaler
    X_tilde, _ = scaler.transform(split.train_X[:10])
    return Batch(X=X_tilde, y=split.train_y[:10], idx=split.train_idx[:10])


# ─── Budget fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def small_budget():
    return BudgetManager(train_shots=10_000, monitoring_shots=2_000)


@pytest.fixture
def ledger():
    return Ledger()
