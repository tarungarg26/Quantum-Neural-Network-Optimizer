from pathlib import Path

import numpy as np

from qnnbench.quantum.circuit_factory import (
    build_qnode,
    n_gates_report,
)


def _numpy_simulate_benchmark(theta: np.ndarray, x: np.ndarray, observable: str = "Z0") -> float:
    """Independent NumPy statevector simulation of the benchmark circuit for second-opinion test."""
    # State |0000>
    psi = np.zeros((2, 2, 2, 2), dtype=complex)
    psi[0, 0, 0, 0] = 1.0

    def ry(angle: float) -> np.ndarray:
        c = np.cos(angle / 2.0)
        s = np.sin(angle / 2.0)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def rz(angle: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * angle / 2.0), 0.0],
            [0.0, np.exp(1j * angle / 2.0)]
        ], dtype=complex)

    letters = ["a", "b", "c", "d"]

    def apply_1q(state: np.ndarray, gate: np.ndarray, wire: int) -> np.ndarray:
        in_sub = list(letters)
        out_sub = list(letters)
        out_sub[wire] = "x"
        formula = f"x{in_sub[wire]},abcd->" + "".join(out_sub)
        return np.einsum(formula, gate, state)

    def apply_cnot(state: np.ndarray, ctrl: int, tgt: int) -> np.ndarray:
        # Reshape to 16
        flat = state.reshape(16).copy()
        out = np.zeros(16, dtype=complex)
        for i in range(16):
            c_bit = (i >> (3 - ctrl)) & 1
            (i >> (3 - tgt)) & 1
            if c_bit == 1:
                # flip target bit
                flipped = i ^ (1 << (3 - tgt))
                out[flipped] = flat[i]
            else:
                out[i] = flat[i]
        return out.reshape((2, 2, 2, 2))

    # Encoding
    for w in range(4):
        psi = apply_1q(psi, ry(x[w]), w)

    # Trainable layers
    for layer in range(2):
        for w in range(4):
            psi = apply_1q(psi, ry(theta[layer, w, 0]), w)
            psi = apply_1q(psi, rz(theta[layer, w, 1]), w)
        # CNOT ring: 0->1, 1->2, 2->3, 3->0
        for w in range(4):
            psi = apply_cnot(psi, w, (w + 1) % 4)

    # Expectation value
    flat_psi = psi.reshape(16)
    if observable == "Z0":
        # Z0 eigenvalues: +1 if bit 0 is 0, -1 if bit 0 is 1
        probs = np.abs(flat_psi) ** 2
        z0_diag = np.array([1 if ((i >> 3) & 1) == 0 else -1 for i in range(16)])
        return float(np.sum(z0_diag * probs))
    elif observable == "parity":
        probs = np.abs(flat_psi) ** 2
        # Parity = (-1)^(b0 + b1 + b2 + b3)
        par_diag = np.array([1 if ((i).bit_count() % 2 == 0) else -1 for i in range(16)])
        return float(np.sum(par_diag * probs))
    raise ValueError(f"Unknown observable: {observable}")


def test_t_m04_01_gate_counts():
    """T-M04-01: gate counts exactly 4 encoding, 16 trainable, 8 CNOT."""
    rep = n_gates_report(n_qubits=4, layers=2)
    assert rep["encoding_RY"] == 4
    assert rep["trainable"] == 16
    assert rep["cnot"] == 8


def test_t_m04_02_expectation_bounded_range():
    """T-M04-02: f in [-1, 1] for random θ, x (property test)."""
    qnode = build_qnode()
    rng = np.random.default_rng(999)
    for _ in range(15):
        theta = rng.uniform(0, 2 * np.pi, (2, 4, 2))
        x = rng.uniform(0, np.pi, 4)
        val = float(qnode(theta, x))
        assert -1.0 - 1e-12 <= val <= 1.0 + 1e-12


def test_t_m04_03_golden_values():
    """T-M04-03: golden values at 5 fixed (θ,x) vs tests/golden/circuit_f.npz (tol 1e-12)."""
    golden_path = Path("tests/golden/circuit_f.npz")
    assert golden_path.exists(), "circuit_f.npz missing"
    data = np.load(golden_path)
    qnode = build_qnode()

    for theta, x, expected_f in zip(data["thetas"], data["xs"], data["fs"]):
        val = float(qnode(theta, x))
        assert np.isclose(val, expected_f, atol=1e-12)


def test_t_m04_04_theta_zero_not_identity():
    """T-M04-04: θ=0 does NOT give identity because CNOT ring and encoding remain."""
    qnode = build_qnode()
    theta = np.zeros((2, 4, 2))
    x = np.array([np.pi / 3, np.pi / 4, np.pi / 5, np.pi / 6])
    val = float(qnode(theta, x))
    # It should not just trivially equal cos(x[0]) because CNOTs entangle
    trivial_cos = np.cos(x[0])
    assert not np.isclose(val, trivial_cos, atol=1e-5)


def test_t_m04_05_periodicity():
    """T-M04-05: f is 2π-periodic in each θ component."""
    qnode = build_qnode()
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, (2, 4, 2))
    x = rng.uniform(0, np.pi, 4)
    val1 = float(qnode(theta, x))

    # Shift each angle by 2pi
    theta_shifted = theta + 2 * np.pi
    val2 = float(qnode(theta_shifted, x))
    assert np.isclose(val1, val2, atol=1e-11)


def test_t_m04_06_numpy_statevector_cross_check():
    """T-M04-06: Independent NumPy statevector re-implementation agrees with PennyLane <= 1e-12."""
    qnode = build_qnode()
    rng = np.random.default_rng(777)
    for _ in range(5):
        theta = rng.uniform(0, 2 * np.pi, (2, 4, 2))
        x = rng.uniform(0, np.pi, 4)
        pl_val = float(qnode(theta, x))
        np_val = _numpy_simulate_benchmark(theta, x, observable="Z0")
        assert np.isclose(pl_val, np_val, atol=1e-12)


def test_t_m04_07_parameterised_builder():
    """T-M04-07: width/depth parameterised builder produces expected param counts."""
    for n_qubits in [2, 4, 6]:
        for layers in [1, 2, 3]:
            rep = n_gates_report(n_qubits=n_qubits, layers=layers)
            assert rep["encoding_RY"] == n_qubits
            assert rep["trainable"] == 2 * layers * n_qubits
            assert rep["cnot"] == layers * n_qubits


def test_t_m04_08_z0_vs_parity_different():
    """T-M04-08: Z0 vs parity give different values on random θ."""
    qnode_z0 = build_qnode(observable="Z0")
    qnode_par = build_qnode(observable="parity")
    rng = np.random.default_rng(888)
    theta = rng.uniform(0, 2 * np.pi, (2, 4, 2))
    x = rng.uniform(0, np.pi, 4)
    val_z0 = float(qnode_z0(theta, x))
    val_par = float(qnode_par(theta, x))
    assert not np.isclose(val_z0, val_par, atol=1e-4)
