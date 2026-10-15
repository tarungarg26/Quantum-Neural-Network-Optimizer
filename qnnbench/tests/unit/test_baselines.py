
from qnnbench.baselines.logreg import run_logreg, run_mlp
from qnnbench.data.manager import make_split, make_task


def test_t_m12_01_same_split_indices():
    """T-M12-01: Baselines run on identical split as quantum runs."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)

    res_lr = run_logreg(split, seed=42)
    res_mlp = run_mlp(split, seed=42)

    assert res_lr.train_acc > 0.70
    assert res_mlp.train_acc > 0.70
    assert res_lr.test_acc > 0.70
    assert res_mlp.test_acc > 0.70


def test_t_m12_03_determinism():
    """T-M12-03: Deterministic outputs given same seed."""
    X, y = make_task()
    split = make_split(X, y, split_seed=7)

    r1 = run_logreg(split, seed=123)
    r2 = run_logreg(split, seed=123)
    assert r1.test_mse == r2.test_mse
    assert r1.test_acc == r2.test_acc

    m1 = run_mlp(split, seed=456)
    m2 = run_mlp(split, seed=456)
    assert m1.test_mse == m2.test_mse
    assert m1.test_acc == m2.test_acc
