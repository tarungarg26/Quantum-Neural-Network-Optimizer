"""
AGENT INSTRUCTIONS — quantum/circuit_factory.py
================================================
M04 — Circuit factory and observables.

API:
  build_qnode(device, n_qubits=4, layers=2, observable="Z0", diff_method="best") -> QNode
  build_test_circuit(kind="crot") -> QNode      # for E2 (controlled rotations)
  n_gates_report() -> {"encoding_RY":4, "trainable":16, "cnot":8}
  draw(qnode, theta, x) -> matplotlib.Figure

Circuit structure:
  |0000⟩
  → RY(x̃ᵢ) per wire (4 encoding gates)
  → 2 × [RY(θ[l,w,0]), RZ(θ[l,w,1]) per wire + CNOT ring 0→1,1→2,2→3,3→0]
  → ⟨Z₀⟩  (or parity for E6)

θ index convention: θ[layer, wire, {0:RY, 1:RZ}] — shape (2,4,2)
  - layer ∈ {0,1}
  - wire  ∈ {0,1,2,3}
  - gate  ∈ {0:RY, 1:RZ}

OBSERVABLES:
  Z0:     qml.PauliZ(0)                   — local readout
  parity: qml.PauliZ(0)@qml.PauliZ(1)@qml.PauliZ(2)@qml.PauliZ(3) — global, for E6

TESTS: tests/unit/test_circuit.py (T-M04-01..08)
  T-M04-01: gate counts exactly 4/16/8
  T-M04-02: f in [-1,1] for random θ, x (property test)
  T-M04-03: golden values at 5 fixed (θ,x) vs tests/golden/circuit_f.npz (tol 1e-12)
  T-M04-04: θ=0 does NOT give identity (CNOTs remain) — assert output ≠ trivial
  T-M04-05: f is 2π-periodic in each θ
  T-M04-06: independent NumPy statevector agrees with PennyLane within 1e-12
  T-M04-07: parameterised builder produces correct param counts for E6 grid
  T-M04-08: Z0 vs parity give DIFFERENT values on random θ (guards silent swap)
"""

from __future__ import annotations

from typing import Literal

import matplotlib
import numpy as np
import pennylane as qml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qnnbench.core.types import Theta

# ─── Observable registry ──────────────────────────────────────────────────────
_OBSERVABLE_REGISTRY: dict[str, qml.operation.Operator] = {}


def _get_observable(name: str, n_qubits: int = 4) -> qml.operation.Operator:
    key = f"{name}_{n_qubits}"
    if key not in _OBSERVABLE_REGISTRY:
        if name == "Z0":
            _OBSERVABLE_REGISTRY[key] = qml.PauliZ(0)
        elif name == "parity":
            obs = qml.PauliZ(0)
            for w in range(1, n_qubits):
                obs = obs @ qml.PauliZ(w)
            _OBSERVABLE_REGISTRY[key] = obs
        else:
            raise ValueError(f"Unknown observable: {name!r}. Choose 'Z0' or 'parity'.")
    return _OBSERVABLE_REGISTRY[key]


def _circuit_template(theta: np.ndarray, x: np.ndarray, n_qubits: int = 4, layers: int = 2) -> None:
    """PennyLane circuit template (no measurement — called inside a QNode).

    theta: (layers, n_qubits, 2) — gate parameters
    x:     (n_qubits,)           — angle-scaled input features
    """
    # Encoding layer
    for w in range(n_qubits):
        qml.RY(x[w], wires=w)

    # Trainable layers
    for layer in range(layers):
        for w in range(n_qubits):
            qml.RY(theta[layer, w, 0], wires=w)
            qml.RZ(theta[layer, w, 1], wires=w)
        # CNOT ring entangler: 0→1, 1→2, 2→3, 3→0
        for w in range(n_qubits):
            qml.CNOT(wires=[w, (w + 1) % n_qubits])


def build_qnode(
    device: qml.Device | None = None,
    n_qubits: int = 4,
    layers: int = 2,
    observable: str = "Z0",
    diff_method: str = "best",
) -> qml.QNode:
    """Build the benchmark QNode.

    Parameters
    ----------
    device:      PennyLane device. If None, uses default.qubit (analytic).
    n_qubits:    Number of qubits (default 4).
    layers:      Number of trainable layers (default 2).
    observable:  'Z0' or 'parity'.
    diff_method: PennyLane diff method ('best', 'parameter-shift', 'backprop').

    Returns
    -------
    QNode that takes (theta, x) → scalar expectation value.
    theta: (layers, n_qubits, 2) array
    x:     (n_qubits,) array
    """
    if device is None:
        device = qml.device("default.qubit", wires=n_qubits)

    obs = _get_observable(observable, n_qubits)

    @qml.qnode(device, diff_method=diff_method)
    def circuit(theta: np.ndarray, x: np.ndarray) -> float:
        _circuit_template(theta, x, n_qubits=n_qubits, layers=layers)
        return qml.expval(obs)

    return circuit


build_parameterised_qnode = build_qnode


def build_test_circuit(kind: Literal["crot"] = "crot") -> qml.QNode:
    """Build the E2 test circuit with controlled rotations.

    Used for verifying 4-term parameter shift rule (psr4).
    Generator gaps {½, 1} are present in CRot, requiring 4-term rule.
    """
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method="best")
    def crot_circuit(params: np.ndarray, x: np.ndarray) -> float:
        """Controlled rotation circuit for E2 verification."""
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.CRot(params[0], params[1], params[2], wires=[0, 1])
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    return crot_circuit


def n_gates_report(n_qubits: int = 4, layers: int = 2) -> dict[str, int]:
    """Return uncompiled gate counts for the benchmark circuit.

    Returns
    -------
    {'encoding_RY': n_qubits, 'trainable': 2*layers*n_qubits, 'cnot': layers*n_qubits}
    """
    return {
        "encoding_RY": n_qubits,
        "trainable": 2 * layers * n_qubits,
        "cnot": layers * n_qubits,
    }


def draw(qnode: qml.QNode, theta: Theta, x: np.ndarray) -> plt.Figure:
    """Draw the circuit as a Matplotlib figure."""
    fig, ax = qml.draw_mpl(qnode)(theta.angles, x)
    ax.set_title("Benchmark Circuit: 4-qubit VQC", fontsize=12)
    return fig
