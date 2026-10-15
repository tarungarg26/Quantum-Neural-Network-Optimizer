"""Property-based tests using Hypothesis for qnnbench invariants.

Tests:
- Theta round-trip flat/from_flat
- Budget ledger: never overshoots budget under arbitrary sequence of requests
- Allocation: sum of allocated shots strictly equals target production shots
- Scaler: transforms always bounded in [0, pi]
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qnnbench.budget.allocation import VarianceInformedAllocation
from qnnbench.budget.ledger import BudgetManager, Category, Ledger, LedgerEvent
from qnnbench.core.types import Theta
from qnnbench.data.manager import Scaler


@settings(max_examples=50)
@given(
    angles=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=16,
        max_size=16,
    ),
    bias=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_theta_flat_roundtrip_property(angles: list[float], bias: float) -> None:
    arr = np.array(angles, dtype=np.float64).reshape((2, 4, 2))
    theta = Theta(angles=arr, bias=bias)
    flat = theta.flat()
    restored = Theta.from_flat(flat)
    np.testing.assert_allclose(theta.angles, restored.angles, atol=1e-12)
    assert abs(theta.bias - restored.bias) < 1e-12


@settings(max_examples=50)
@given(
    train_shots=st.integers(min_value=1000, max_value=50000),
    monitoring_shots=st.integers(min_value=500, max_value=10000),
    requests=st.lists(
        st.tuples(
            st.sampled_from(["training", "monitoring", "reference"]),
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=10, max_value=500),
        ),
        min_size=1,
        max_size=30,
    ),
)
def test_ledger_never_overshoots_property(
    train_shots: int,
    monitoring_shots: int,
    requests: list[tuple[Category, int, int]],
) -> None:
    mgr = BudgetManager(train_shots=train_shots, monitoring_shots=monitoring_shots)
    ledger = Ledger()
    mgr.attach_ledger(ledger)

    for cat, circuits, spc in requests:
        if mgr.can_afford(cat, circuits, spc):
            mgr.preflight(cat, circuits, spc)
            evt = LedgerEvent(
                run_id="test",
                category=cat,
                tag="prop_test",
                circuits=circuits,
                shots_per_circuit=spc,
                shots_total=circuits * spc,
                wall_s=0.0,
            )
            ledger.commit(evt)

    assert mgr.spent("training") <= train_shots
    assert mgr.spent("monitoring") <= monitoring_shots


@settings(max_examples=50)
@given(
    n_params=st.integers(min_value=2, max_value=32),
    s_prod=st.integers(min_value=500, max_value=10000),
    pilot_shots=st.integers(min_value=8, max_value=32),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_allocation_sum_exact_property(
    n_params: int,
    s_prod: int,
    pilot_shots: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n_jobs = n_params * 2
    if s_prod < n_jobs * 8:
        s_prod = n_jobs * 8 + 100
    alloc = VarianceInformedAllocation(pilot=pilot_shots, floor=1e-4, min_prod=8, S_prod=s_prod)
    # simulate random pilot expectation values in [-1, 1]
    pilots = rng.uniform(-1.0, 1.0, size=(n_jobs,))
    # simulate residual weights
    weights = rng.uniform(0.01, 1.0, size=(n_jobs,))

    plan = alloc.plan(n_jobs=n_jobs, weights=weights, pilots=pilots)
    assert plan.total_shots == s_prod
    assert sum(plan.shots_per_job) == s_prod
    assert all(shots >= 8 for shots in plan.shots_per_job)


@settings(max_examples=50)
@given(
    train_x=st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=50,
    ),
    test_x=st.lists(
        st.floats(min_value=-2000.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=30,
    ),
)
def test_scaler_range_property(train_x: list[float], test_x: list[float]) -> None:
    train_arr = np.array(train_x).reshape((-1, 1))
    test_arr = np.array(test_x).reshape((-1, 1))
    scaler = Scaler.fit(train_arr)
    scaled_train, _ = scaler.transform(train_arr)
    scaled_test, _ = scaler.transform(test_arr)

    assert np.all(scaled_train >= 0.0)
    assert np.all(scaled_train <= np.pi + 1e-12)
    assert np.all(scaled_test >= 0.0)
    assert np.all(scaled_test <= np.pi + 1e-12)
