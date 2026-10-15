from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from qnnbench.core.types import (
    ANGLE_SHAPE,
    N_TOTAL_PARAMS,
    Batch,
    RunKey,
    Theta,
)


def test_t_m01_02_mutating_frozen_dataclass_raises():
    """T-M01-02: Mutating a frozen dataclass raises FrozenInstanceError."""
    angles = np.zeros(ANGLE_SHAPE)
    theta = Theta(angles=angles, bias=0.0)
    with pytest.raises(FrozenInstanceError):
        theta.bias = 1.0  # type: ignore

    batch = Batch(
        X=np.zeros((10, 4)),
        y=np.ones(10),
        idx=np.arange(10),
    )
    with pytest.raises(FrozenInstanceError):
        batch.X = np.ones((10, 4))  # type: ignore

    key = RunKey("E1", "gd", 7, 0, "hash123")
    with pytest.raises(FrozenInstanceError):
        key.experiment = "E2"  # type: ignore


def test_t_m01_04_theta_flat_roundtrip():
    """T-M01-04: Theta.flat / from_flat roundtrip property test across random arrays."""
    rng = np.random.default_rng(12345)
    for _ in range(50):
        angles = rng.uniform(0, 2 * np.pi, ANGLE_SHAPE)
        bias = float(rng.standard_normal())
        theta = Theta(angles=angles, bias=bias)

        flat = theta.flat()
        assert flat.shape == (N_TOTAL_PARAMS,)
        assert flat[-1] == bias

        reconstructed = Theta.from_flat(flat)
        assert np.allclose(reconstructed.angles, theta.angles, atol=1e-14)
        assert reconstructed.bias == theta.bias
        assert reconstructed == theta


def test_theta_validation():
    with pytest.raises(ValueError, match="angles must have shape"):
        Theta(angles=np.zeros((3, 4, 2)), bias=0.0)
    with pytest.raises(TypeError, match="angles must be float"):
        Theta(angles=np.zeros(ANGLE_SHAPE, dtype=np.int32), bias=0.0)


def test_batch_validation():
    with pytest.raises(ValueError, match="Labels must be in"):
        Batch(X=np.zeros((5, 4)), y=np.array([0, 1, -1, 1, -1]), idx=np.arange(5))
    with pytest.raises(ValueError, match="X must be"):
        Batch(X=np.zeros((5, 3)), y=np.ones(5), idx=np.arange(5))


def test_run_key_id_deterministic():
    k1 = RunKey("E1", "gd", 7, 0, "abc")
    k2 = RunKey("E1", "gd", 7, 0, "abc")
    assert k1.run_id() == k2.run_id()
    assert len(k1.run_id()) == 12
