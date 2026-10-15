from pathlib import Path

import numpy as np

from qnnbench.core.seeds import PurposeID, stream
from qnnbench.core.types import ANGLE_SHAPE, Theta


def test_t_m02_01_same_args_identical_draws_different_purpose_diff():
    """T-M02-01: Same args yield identical draws; different purpose yields different draws."""
    rng1 = stream(7, 0, PurposeID.INIT)
    rng2 = stream(7, 0, PurposeID.INIT)
    assert np.all(rng1.uniform(size=10) == rng2.uniform(size=10))

    rng3 = stream(7, 0, PurposeID.MINIBATCH)
    assert not np.all(rng1.uniform(size=10) == rng3.uniform(size=10))


def test_t_m02_02_init_stream_paired_across_optimisers():
    """T-M02-02: init stream identical across optimisers; shot_sampling differs."""
    # init stream should be identical regardless of optimiser arg
    rng_gd_init = stream(7, 0, PurposeID.INIT, optimiser="gd")
    rng_adam_init = stream(7, 0, PurposeID.INIT, optimiser="adam")
    assert np.all(rng_gd_init.uniform(size=10) == rng_adam_init.uniform(size=10))

    # shot_sampling differs across optimisers
    rng_gd_shots = stream(7, 0, PurposeID.SHOT_SAMPLING, optimiser="gd")
    rng_adam_shots = stream(7, 0, PurposeID.SHOT_SAMPLING, optimiser="adam")
    assert not np.all(rng_gd_shots.uniform(size=10) == rng_adam_shots.uniform(size=10))


def test_t_m02_03_fifteen_replication_keys_distinct_theta0():
    """T-M02-03: 15 replication keys (5 split seeds x 3 init seeds) yield 15 distinct θ₀."""
    split_seeds = [7, 17, 27, 37, 47]
    init_seeds = [0, 1, 2]
    theta0_set = []

    for s in split_seeds:
        for i in init_seeds:
            rng = stream(s, i, PurposeID.INIT)
            angles = rng.uniform(0, 2 * np.pi, ANGLE_SHAPE)
            theta = Theta(angles=angles, bias=0.0)
            theta0_set.append(theta)

    assert len(theta0_set) == 15
    for i in range(15):
        for j in range(i + 1, 15):
            assert not np.allclose(theta0_set[i].angles, theta0_set[j].angles, atol=1e-8)


def test_t_m02_04_theta0_range_and_bias():
    """T-M02-04: θ₀ ∈ [0, 2π)¹⁶, b₀ = 0."""
    rng = stream(7, 0, PurposeID.INIT)
    angles = rng.uniform(0, 2 * np.pi, ANGLE_SHAPE)
    theta = Theta(angles=angles, bias=0.0)
    assert np.all(theta.angles >= 0.0)
    assert np.all(theta.angles < 2 * np.pi)
    assert theta.bias == 0.0


def test_t_m02_05_banned_global_rng_lint():
    """T-M02-05: Grep src/qnnbench for banned global RNG calls."""
    import ast
    src_dir = Path("src/qnnbench")
    violations = []

    for py_file in src_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Detect np.random.<func>() calls
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Attribute)
                    and getattr(func.value.value, "id", "") == "np"
                    and func.value.attr == "random"
                    and func.attr in {"seed", "rand", "randn", "randint", "choice", "uniform", "standard_normal"}
                ):
                    violations.append(f"{py_file}:{node.lineno}: np.random.{func.attr}()")
                # Detect random.seed()
                elif (
                    isinstance(func, ast.Attribute)
                    and getattr(func.value, "id", "") == "random"
                    and func.attr == "seed"
                ):
                    violations.append(f"{py_file}:{node.lineno}: random.seed()")

    assert not violations, "Found banned global RNG calls:\n" + "\n".join(violations)
