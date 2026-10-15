import numpy as np

from qnnbench.core.types import ANGLE_SHAPE, N_GATE_PARAMS, Theta
from qnnbench.grad.metric import metric_block_diag, qng_direction


def test_t_m07_01_g_symmetric_psd():
    """T-M07-01: G symmetric PSD (property, tol 1e-10)."""
    rng = np.random.default_rng(42)
    theta = Theta(angles=rng.uniform(0, 2 * np.pi, ANGLE_SHAPE), bias=0.0)
    X = rng.uniform(0, np.pi, (2, 4))

    res = metric_block_diag(theta, X)
    G = res.G

    # Symmetric: G == G^T
    assert np.allclose(G, G.T, atol=1e-10)

    # PSD: eigenvalues >= -1e-10
    eigvals = np.linalg.eigvalsh(G)
    assert np.all(eigvals >= -1e-10)


def test_t_m07_04_qng_direction_limits():
    """T-M07-04: λ→∞ ⇒ direction ∝ grad/λ; λ→0 on well-conditioned G ⇒ G⁻¹∇C."""
    p = N_GATE_PARAMS
    # Construct well-conditioned G
    A = np.random.default_rng(123).standard_normal((p, p))
    G = A.T @ A + np.eye(p) * 2.0
    grad = np.ones(p)

    # Large damping limit: (G + λI)⁻¹ grad ≈ grad / λ
    lambda_large = 1e6
    d_large, _ = qng_direction(G, grad, damping=lambda_large)
    expected_large = grad / lambda_large
    assert np.allclose(d_large, expected_large, rtol=1e-3)

    # Small damping limit: G⁻¹ grad
    lambda_small = 1e-7
    d_small, _ = qng_direction(G, grad, damping=lambda_small)
    expected_small = np.linalg.solve(G, grad)
    assert np.allclose(d_small, expected_small, rtol=1e-3)


def test_t_m07_05_singular_g_fallback():
    """T-M07-05: Singular G triggers fallback, logged, continues without crashing."""
    p = N_GATE_PARAMS
    G_singular = np.zeros((p, p))  # all zeros
    grad = np.ones(p)

    # With damping = 0.0, G + 0.0*I is singular
    d, report = qng_direction(G_singular, grad, damping=0.0)
    assert report["fallback_used"] is True
    assert np.all(np.isfinite(d))


def test_t_m07_06_input_averaging_determinism():
    """T-M07-06: Averaging over 1 vs 8 inputs gives different G; same inputs => same G."""
    rng = np.random.default_rng(555)
    theta = Theta(angles=rng.uniform(0, 2 * np.pi, ANGLE_SHAPE), bias=0.0)
    X8 = rng.uniform(0, np.pi, (8, 4))

    res1 = metric_block_diag(theta, X8[:1])
    res8 = metric_block_diag(theta, X8)

    # Different inputs => different G
    assert not np.allclose(res1.G, res8.G, atol=1e-4)

    # Same inputs => same G
    res8_rep = metric_block_diag(theta, X8)
    assert np.allclose(res8.G, res8_rep.G, atol=1e-12)


def test_t_m07_07_condition_number_accuracy():
    """T-M07-07: Condition number reported equals np.linalg.cond."""
    G = np.eye(N_GATE_PARAMS) * 2.0
    grad = np.ones(N_GATE_PARAMS)
    _d, report = qng_direction(G, grad, damping=0.05)
    expected_cond = float(np.linalg.cond(G + 0.05 * np.eye(N_GATE_PARAMS)))
    assert np.isclose(report["cond"], expected_cond, atol=1e-10)
