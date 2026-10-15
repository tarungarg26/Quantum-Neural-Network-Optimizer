"""F01 Figure: Architecture and VQC Quantum Circuit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from qnnbench.core.types import Theta
from qnnbench.quantum.circuit_factory import build_qnode, draw, n_gates_report
from qnnbench.viz.primitives import footer
from qnnbench.viz.style import apply_style, save_figure


def make(meta: dict[str, Any] | None = None, out_path: str | Path = "outputs/figures/f01_circuit") -> Path:
    """Generate F01 architecture diagram figure."""
    apply_style()
    qnode = build_qnode(n_qubits=4, layers=2, observable="Z0")

    theta_dummy = Theta(angles=np.zeros((2, 4, 2)), bias=0.0)
    x_dummy = np.zeros(4)

    fig = draw(qnode, theta_dummy, x_dummy)
    gates = n_gates_report(4, 2)
    # Add subtitle with gate accounting
    fig.suptitle(
        f"4-Qubit VQC Architecture | Gates: {gates['encoding_RY']} RY(x) + {gates['trainable']} Trainable + {gates['cnot']} CNOT",
        fontsize=9,
        y=0.98,
    )
    if meta is not None:
        footer(fig, meta)

    save_figure(fig, out_path)
    return Path(out_path)
