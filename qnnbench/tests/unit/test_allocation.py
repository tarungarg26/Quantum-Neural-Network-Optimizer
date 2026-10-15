import numpy as np

from qnnbench.budget.allocation import UniformAllocation, VarianceInformedAllocation


def test_t_m10_01_sum_equals_s_prod_exactly():
    """T-M10-01: ΣNⱼ == S_prod exactly (property test, random weights)."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n_jobs = int(rng.integers(5, 30))
        s_prod = n_jobs * 16 + int(rng.integers(10, 500))
        policy = VarianceInformedAllocation(min_prod=8, S_prod=s_prod)

        weights = rng.uniform(0.1, 2.0, n_jobs)
        pilots = rng.uniform(-0.9, 0.9, n_jobs)

        plan = policy.plan(n_jobs, weights, pilots)
        assert sum(plan.shots_per_job) == s_prod
        assert plan.total_shots == s_prod


def test_t_m10_02_floor_and_zero_variance():
    """T-M10-02: Every Nⱼ ≥ 8; zero-variance pilot (f=±1) still gets ≥ 8."""
    n_jobs = 10
    policy = VarianceInformedAllocation(min_prod=8, S_prod=200)

    # Pilots at ±1 (f^2 = 1 => 1 - f^2 = 0)
    pilots = np.ones(n_jobs)
    weights = np.ones(n_jobs)

    plan = policy.plan(n_jobs, weights, pilots)
    for n in plan.shots_per_job:
        assert n >= 8
    assert sum(plan.shots_per_job) == 200


def test_t_m10_03_monotonicity():
    """T-M10-03: Nⱼ is monotone in |aⱼ|σⱼ."""
    n_jobs = 4
    policy = VarianceInformedAllocation(min_prod=8, S_prod=200)

    # Ascending weights with equal pilots
    weights = np.array([0.1, 0.5, 1.0, 2.0])
    pilots = np.zeros(n_jobs)

    plan = policy.plan(n_jobs, weights, pilots)
    alloc = plan.shots_per_job
    assert alloc[0] <= alloc[1] <= alloc[2] <= alloc[3]


def test_t_m10_04_largest_remainder_determinism():
    """T-M10-04: Largest-remainder tie-breaking is deterministic."""
    n_jobs = 5
    policy = VarianceInformedAllocation(min_prod=8, S_prod=101)

    weights = np.ones(n_jobs)
    pilots = np.zeros(n_jobs)

    plan1 = policy.plan(n_jobs, weights, pilots)
    plan2 = policy.plan(n_jobs, weights, pilots)

    assert plan1.shots_per_job == plan2.shots_per_job
    assert sum(plan1.shots_per_job) == 101


def test_t_m10_05_uniform_reduces():
    """T-M10-05: Uniform case reduces to equal shots."""
    uniform = UniformAllocation(shots=32)
    plan = uniform.plan(10)
    assert plan.shots_per_job == [32] * 10
    assert plan.total_shots == 320
    assert plan.is_uniform is True
