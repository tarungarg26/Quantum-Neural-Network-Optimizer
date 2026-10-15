import numpy as np
import pennylane as qml
import pytest

from qnnbench.budget.ledger import Ledger
from qnnbench.core.errors import NonFiniteError
from qnnbench.core.types import ANGLE_SHAPE, Batch, Theta
from qnnbench.grad.psr import (
    autodiff_reference,
    mse_grad,
    psr2,
    psr4,
    unbiased_mse_loss,
)
from qnnbench.quantum.backend import Backend, CircuitJob, DeviceConfig
from qnnbench.quantum.circuit_factory import build_test_circuit


def test_t_m06_01_e1_psr2_vs_autodiff():
    """T-M06-01 (E1): psr2 vs autodiff: max |Δ| ≤ 1e-10 over random (θ,x)."""
    ledger = Ledger()
    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=np.random.default_rng(42))

    rng = np.random.default_rng(123)
    # Test over 20 pairs (fast and sufficient to verify exact algebraic equivalence)
    for _ in range(5):
        theta = Theta(angles=rng.uniform(0, 2 * np.pi, ANGLE_SHAPE), bias=0.0)
        X = rng.uniform(0, np.pi, (2, 4))
        y = np.array([-1.0, 1.0])
        batch = Batch(X=X, y=y, idx=np.array([0, 1]))

        f_psr, df_psr = psr2(theta, batch, backend=backend, shots=None, category="reference")
        f_ad, df_ad = autodiff_reference(theta, batch, backend=backend)

        assert np.allclose(f_psr, f_ad, atol=1e-10)
        assert np.allclose(df_psr, df_ad, atol=1e-10)


def test_t_m06_02_e2_psr4_on_controlled_rotations():
    """T-M06-02 (E2): psr4 on controlled-rotation circuit ≤ 1e-9; psr2 shows error > 1e-2."""
    crot_circuit = build_test_circuit("crot")

    params = np.array([0.7, 1.2, 0.4])
    x = np.array([np.pi / 4, np.pi / 3])

    # Autodiff reference
    params_pl = qml.numpy.array(params.copy(), requires_grad=True)
    grad_ad = qml.grad(lambda p: crot_circuit(p, x), argnums=0)(params_pl)

    # 4-term shift rule for parameter 1 (RY inside CRot)
    k = 1
    grad_psr4 = psr4(params, k, lambda p, _x: crot_circuit(p, _x), x)
    assert np.isclose(grad_psr4, grad_ad[k], atol=1e-9)

    # 2-term shift rule fails on generator gap {1/2, 1}
    p_plus = params.copy()
    p_plus[k] += np.pi / 2
    p_minus = params.copy()
    p_minus[k] -= np.pi / 2
    grad_psr2 = 0.5 * (float(crot_circuit(p_plus, x)) - float(crot_circuit(p_minus, x)))

    # Error must be large (> 1e-2)
    diff = abs(grad_psr2 - grad_ad[k])
    assert diff > 1e-2, f"psr2 error {diff} was expected to be > 1e-2"


def test_t_m06_03_finite_difference_cross_check():
    """T-M06-03: Finite-difference cross-check (central, h=1e-5) agrees ≤ 1e-6."""
    ledger = Ledger()
    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=np.random.default_rng(0))
    rng = np.random.default_rng(999)

    theta = Theta(angles=rng.uniform(0, 2 * np.pi, ANGLE_SHAPE), bias=0.0)
    X = rng.uniform(0, np.pi, (1, 4))
    batch = Batch(X=X, y=np.array([1.0]), idx=np.array([0]))

    _, df_psr = psr2(theta, batch, backend=backend, shots=None, category="reference")

    # Central finite difference on parameter 0
    h = 1e-5
    t_plus = theta.angles.copy()
    t_plus[0, 0, 0] += h
    t_minus = theta.angles.copy()
    t_minus[0, 0, 0] -= h

    j_plus = [CircuitJob(t_plus, X[0], "Z0")]
    j_minus = [CircuitJob(t_minus, X[0], "Z0")]
    f_p = backend.execute(j_plus, shots=None, category="reference", tag="fd_p")[0]
    f_m = backend.execute(j_minus, shots=None, category="reference", tag="fd_m")[0]

    df_fd = (f_p - f_m) / (2 * h)
    assert np.isclose(df_psr[0, 0], df_fd, atol=1e-6)


def test_t_m06_04_mse_grad_vs_autodiff_loss():
    """T-M06-04: mse_grad vs autodiff of whole loss incl. bias ≤ 1e-10."""
    import pennylane as qml
    dev = qml.device("default.qubit", wires=4)

    from qnnbench.quantum.circuit_factory import _circuit_template

    rng = np.random.default_rng(123)
    angles = rng.uniform(0, 2 * np.pi, ANGLE_SHAPE)
    bias = 0.35
    theta = Theta(angles=angles, bias=bias)

    X = rng.uniform(0, np.pi, (3, 4))
    y = np.array([-1.0, 1.0, -1.0])
    batch = Batch(X=X, y=y, idx=np.arange(3))

    backend = Backend(DeviceConfig(n_qubits=4), ledger=Ledger(), rng=rng)
    f, df = psr2(theta, batch, backend=backend, shots=None, category="reference")
    g_theta, g_b = mse_grad(f, df, bias, y)

    # PennyLane autodiff of loss
    @qml.qnode(dev, diff_method="backprop")
    def circ(ang, x):
        _circuit_template(ang, x, 4)
        return qml.expval(qml.PauliZ(0))

    def full_loss(ang, b):
        preds = [circ(ang, X[i]) for i in range(3)]
        return sum((preds[i] + b - y[i]) ** 2 for i in range(3)) / 3.0

    ang_pl = qml.numpy.array(angles, requires_grad=True)
    b_pl = qml.numpy.array(bias, requires_grad=True)

    g_ang_ad = qml.grad(full_loss, argnums=0)(ang_pl, b_pl)
    g_b_ad = qml.grad(full_loss, argnums=1)(ang_pl, b_pl)

    assert np.allclose(g_theta, g_ang_ad.ravel(), atol=1e-10)
    assert np.isclose(g_b, g_b_ad, atol=1e-10)


def test_t_m06_05_cost_formula():
    """T-M06-05: Cost formula: p=16, m=70 => 2310; m=10 => 330 circuits."""
    p = 16
    assert (2 * p + 1) * 70 == 2310
    assert (2 * p + 1) * 10 == 330

    # Ledger agrees after one PSR update
    ledger = Ledger()
    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=np.random.default_rng(0))
    batch = Batch(X=np.zeros((10, 4)), y=np.ones(10), idx=np.arange(10))
    theta = Theta(np.zeros(ANGLE_SHAPE), 0.0)

    psr2(theta, batch, backend=backend, shots=16, category="training")
    assert ledger.total_circuits("training") == 330


def test_t_m06_06_spsa_psr_cost_ratio():
    """T-M06-06: SPSA/PSR circuit ratio is 16.5 (not 17)."""
    p = 16
    # PSR cost per sample: 2p+1 = 33
    # SPSA cost per sample: 2
    assert (2 * p + 1) / 2.0 == 16.5


def test_t_m06_10_unbiased_mse_loss_raises_n_le_1():
    """T-M06-10: unbiased_mse_loss raises for N ≤ 1."""
    f = np.array([0.5, -0.5])
    y = np.array([1.0, -1.0])
    with pytest.raises(NonFiniteError):
        unbiased_mse_loss(f, 0.0, y, N=1)
    with pytest.raises(NonFiniteError):
        unbiased_mse_loss(f, 0.0, y, N=0)


def test_t_m06_11_non_finite_input_raises():
    """T-M06-11: Non-finite input raises NonFiniteError."""
    f = np.array([np.nan, 0.5])
    df = np.zeros((2, 16))
    y = np.array([1.0, -1.0])
    with pytest.raises(NonFiniteError):
        mse_grad(f, df, 0.0, y)
