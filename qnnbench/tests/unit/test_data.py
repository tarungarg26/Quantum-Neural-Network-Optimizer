import numpy as np
import pytest

from qnnbench.core.errors import LeakageError
from qnnbench.data.manager import (
    CONSTANT_ANGLE,
    Scaler,
    Split,
    fit_scaler,
    make_split,
    make_task,
)


def test_t_m03_01_task_structure():
    """T-M03-01: task has 100 samples, 50/50, 4 features, labels {-1,+1}, no setosa."""
    X, y = make_task()
    assert X.shape == (100, 4)
    assert y.shape == (100,)
    assert set(np.unique(y)) == {-1.0, 1.0}
    assert np.sum(y == -1.0) == 50
    assert np.sum(y == 1.0) == 50


def test_t_m03_02_main_split_counts_and_disjoint():
    """T-M03-02: main split counts = 30/30 train, 10/10 val, 10/10 test; disjoint; union = all 100."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7, scheme="main")

    assert split.train_X.shape == (60, 4)
    assert split.val_X.shape == (20, 4)
    assert split.test_X.shape == (20, 4)

    # Stratified balance check
    assert np.sum(split.train_y == -1.0) == 30
    assert np.sum(split.train_y == 1.0) == 30
    assert np.sum(split.val_y == -1.0) == 10
    assert np.sum(split.val_y == 1.0) == 10
    assert np.sum(split.test_y == -1.0) == 10
    assert np.sum(split.test_y == 1.0) == 10

    # Disjoint and union check
    s_train = set(split.train_idx)
    s_val = set(split.val_idx)
    s_test = set(split.test_idx)
    assert len(s_train & s_val) == 0
    assert len(s_train & s_test) == 0
    assert len(s_val & s_test) == 0
    assert (s_train | s_val | s_test) == set(range(100))


def test_t_m03_03_split_reproducibility_and_diversity():
    """T-M03-03: same seed -> same split; 5 replication seeds -> 5 distinct splits."""
    X, y = make_task()
    sp1 = make_split(X, y, split_seed=7)
    sp2 = make_split(X, y, split_seed=7)
    assert sp1 == sp2

    splits = [make_split(X, y, split_seed=s) for s in [7, 17, 27, 37, 47]]
    for i in range(5):
        for j in range(i + 1, 5):
            assert not np.array_equal(splits[i].train_idx, splits[j].train_idx)


def test_t_m03_04_leakage_test():
    """T-M03-04: LEAKAGE TEST: perturb val/test -> scaler unchanged; perturb train -> changed."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)

    # Passing raw array directly must raise LeakageError
    with pytest.raises(LeakageError):
        fit_scaler(X)  # type: ignore

    scaler1 = fit_scaler(split)

    # Perturb test and val in a modified copy
    split_perturbed_test = Split(
        train_X=split.train_X.copy(),
        train_y=split.train_y.copy(),
        train_idx=split.train_idx.copy(),
        val_X=split.val_X + 999.0,
        val_y=split.val_y.copy(),
        val_idx=split.val_idx.copy(),
        test_X=split.test_X - 500.0,
        test_y=split.test_y.copy(),
        test_idx=split.test_idx.copy(),
        track="main",
        split_seed=split.split_seed,
    )
    scaler_after_test_perturb = fit_scaler(split_perturbed_test)
    assert np.all(scaler1.a == scaler_after_test_perturb.a)
    assert np.all(scaler1.d == scaler_after_test_perturb.d)

    # Perturb train
    split_perturbed_train = Split(
        train_X=split.train_X + 5.0,
        train_y=split.train_y.copy(),
        train_idx=split.train_idx.copy(),
        val_X=split.val_X.copy(),
        val_y=split.val_y.copy(),
        val_idx=split.val_idx.copy(),
        test_X=split.test_X.copy(),
        test_y=split.test_y.copy(),
        test_idx=split.test_idx.copy(),
        track="main",
        split_seed=split.split_seed,
    )
    scaler_after_train_perturb = fit_scaler(split_perturbed_train)
    assert not np.all(scaler1.a == scaler_after_train_perturb.a)


def test_t_m03_05_scaled_range():
    """T-M03-05: x̃ ∈ [0, π] always; train min->0, max->π."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)
    scaler = fit_scaler(split)

    train_tilde, _ = scaler.transform(split.train_X)
    assert np.all(train_tilde >= 0.0 - 1e-12)
    assert np.all(train_tilde <= np.pi + 1e-12)
    assert np.allclose(train_tilde.min(axis=0), 0.0)
    assert np.allclose(train_tilde.max(axis=0), np.pi)

    val_tilde, _ = scaler.transform(split.val_X)
    assert np.all(val_tilde >= 0.0)
    assert np.all(val_tilde <= np.pi)


def test_t_m03_06_clip_counts():
    """T-M03-06: clip counts equal brute-force count of out-of-range val/test entries."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)
    scaler = fit_scaler(split)

    _, report_val = scaler.transform(split.val_X)

    # Brute force count
    expected_low = int(np.sum(split.val_X < scaler.a))
    expected_high = int(np.sum(split.val_X > scaler.d))

    assert report_val.n_clipped_low == expected_low
    assert report_val.n_clipped_high == expected_high


def test_t_m03_07_constant_feature_handling():
    """T-M03-07: constant-feature -> fixed angle π/2 (Decision D-03)."""
    # Synthetic data with column 1 constant
    X_synth = np.array([
        [1.0, 5.0, 10.0, 20.0],
        [2.0, 5.0, 12.0, 22.0],
        [3.0, 5.0, 14.0, 24.0],
    ])
    scaler = Scaler(
        a=X_synth.min(axis=0),
        d=X_synth.max(axis=0),
        const_mask=(X_synth.max(axis=0) == X_synth.min(axis=0)),
    )
    X_test = np.array([[1.5, 99.0, 11.0, 21.0]])
    tilde, _report = scaler.transform(X_test)
    assert np.isclose(tilde[0, 1], CONSTANT_ANGLE)


def test_t_m03_08_historical_split():
    """T-M03-08: historical split (70/30, seed 7) reproducible and flagged track='historical'."""
    X, y = make_task()
    split1 = make_split(X, y, split_seed=7, scheme="historical")
    split2 = make_split(X, y, split_seed=7, scheme="historical")
    assert split1 == split2
    assert split1.track == "historical"
    assert split1.train_X.shape == (70, 4)
    assert split1.test_X.shape == (30, 4)
    assert len(split1.val_idx) == 0
