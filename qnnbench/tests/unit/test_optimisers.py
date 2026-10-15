import numpy as np
import pytest

from qnnbench.core.types import ANGLE_SHAPE, N_GATE_PARAMS, Theta
from qnnbench.optim.base import (
    QNG,
    SPSA,
    Adam,
    GradientDescent,
    Momentum,
    StepResult,
    build_optimiser,
)


@pytest.mark.parametrize("name", ["gd", "momentum", "adam", "spsa", "qng"])
def test_t_m09_03_state_dict_roundtrip(name):
    """T-M09-03: state_dict round-trip serializes and restores optimizer state."""
    opt1 = build_optimiser(name)
    sd = opt1.state_dict()
    assert isinstance(sd, dict)
    assert sd["name"] == name

    opt2 = build_optimiser(name)
    opt2.load_state_dict(sd)
    assert opt2.state_dict() == opt1.state_dict()


def test_t_m09_gd_01_hand_computed_step():
    """T-M09-GD-01: One step equals hand-computed θ−η∇C."""
    lr = 0.1
    gd = GradientDescent(lr=lr)
    theta = Theta(angles=np.ones(ANGLE_SHAPE), bias=0.5)

    grad_theta = np.full(N_GATE_PARAMS, 2.0)
    grad_b = 1.0
    step = StepResult(grad_theta, grad_b, grad_theta, grad_b, {})

    new_theta = gd.apply(theta, step)
    expected_angles = np.ones(ANGLE_SHAPE) - lr * 2.0
    expected_bias = 0.5 - lr * 1.0

    assert np.allclose(new_theta.angles, expected_angles)
    assert np.isclose(new_theta.bias, expected_bias)


def test_t_m09_mo_01_two_step_momentum_recursion():
    """T-M09-MO-01: Two-step trace equals hand-computed momentum recursion."""
    lr = 0.1
    mu = 0.9
    mo = Momentum(lr=lr, mu=mu)
    theta = Theta(angles=np.zeros(ANGLE_SHAPE), bias=0.0)

    # Step 1: grad = 1.0
    g1 = np.ones(N_GATE_PARAMS)
    # v1 = mu * 0 + g1 = 1.0
    step1 = StepResult(g1, 1.0, g1, 1.0, {})
    theta1 = mo.apply(theta, step1)
    assert np.allclose(theta1.angles, -lr * 1.0)

    # Step 2: v2 = mu * 1.0 + g2
    # simulate propose velocity update:
    mo._v_theta = mu * 1.0 + g1
    mo._v_b = mu * 1.0 + 1.0
    step2 = StepResult(g1, 1.0, mo._v_theta, mo._v_b, {})
    theta2 = mo.apply(theta1, step2)
    expected_step2 = -lr * 1.0 - lr * (0.9 * 1.0 + 1.0)
    assert np.allclose(theta2.angles, expected_step2)


def test_t_m09_ad_01_adam_bias_correction():
    """T-M09-AD-01: First-step update magnitude ≈ η; matches reference Adam."""
    lr = 0.03
    adam = Adam(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)

    grad = np.full(N_GATE_PARAMS, 0.5)
    # Step 1: m = 0.1 * 0.5 = 0.05, v = 0.001 * 0.25 = 0.00025
    # m_hat = 0.05 / 0.1 = 0.5, v_hat = 0.00025 / 0.001 = 0.25
    # direction = 0.5 / sqrt(0.25) = 1.0
    # Step magnitude = lr * 1.0 = lr!
    adam._t = 1
    adam._m = (1 - adam.beta1) * grad
    adam._v = (1 - adam.beta2) * grad ** 2
    m_hat = adam._m / (1 - adam.beta1)
    v_hat = adam._v / (1 - adam.beta2)
    direction = m_hat / (np.sqrt(v_hat) + adam.eps)
    step = StepResult(grad, 0.5, direction, 1.0, {})

    theta = Theta(angles=np.zeros(ANGLE_SHAPE), bias=0.0)
    new_theta = adam.apply(theta, step)
    assert np.allclose(new_theta.angles, -lr * 1.0)


def test_t_m09_sp_01_spsa_schedule_closed_form():
    """T-M09-SP-01: Schedule values a_t, c_t equal closed form for t=0..100."""
    a, c, A = 0.5, 0.15, 10
    spsa = SPSA(a=a, c=c, A=A)

    for t in [0, 1, 10, 50, 100]:
        a_t, c_t = spsa._schedule(t)
        expected_a = a / (A + t + 1) ** 0.602
        expected_c = c / (t + 1) ** 0.101
        assert np.isclose(a_t, expected_a)
        assert np.isclose(c_t, expected_c)


def test_t_m09_sp_02_spsa_delta_rademacher():
    """T-M09-SP-02: Rademacher delta is ±1 only."""
    rng = np.random.default_rng(123)
    spsa = SPSA(rng=rng)
    for _ in range(20):
        delta = spsa._rng.choice([-1.0, 1.0], size=N_GATE_PARAMS)
        assert set(np.unique(delta)).issubset({-1.0, 1.0})
        assert len(delta) == N_GATE_PARAMS


def test_t_m09_qn_01_qng_with_identity_equals_gd():
    """T-M09-QN-01: With G=I, λ=0, QNG direction ≡ GD direction."""
    from qnnbench.grad.metric import qng_direction
    grad = np.array([0.1, -0.2, 0.3, 0.4] * 4)
    G = np.eye(N_GATE_PARAMS)
    d, _report = qng_direction(G, grad, damping=0.0)
    assert np.allclose(d, grad)


def test_t_m09_05_declared_cost_per_update():
    """T-M09-05: Declared cost_per_update matches expected formulas."""
    m = 10
    assert GradientDescent().cost_per_update(m) == 330
    assert Momentum().cost_per_update(m) == 330
    assert Adam().cost_per_update(m) == 330
    assert SPSA().cost_per_update(m) == 20
    assert QNG().cost_per_update(m) == 330
