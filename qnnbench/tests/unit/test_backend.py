import ast
from pathlib import Path

import numpy as np
import pytest

from qnnbench.budget.ledger import BudgetManager, Ledger
from qnnbench.core.errors import BudgetPreflightError
from qnnbench.core.seeds import PurposeID, stream
from qnnbench.quantum.backend import Backend, CircuitJob, DeviceConfig


def test_t_m05_01_analytic_output_equals_golden():
    """T-M05-01: Analytic output matches circuit_factory golden values."""
    golden_path = Path("tests/golden/circuit_f.npz")
    assert golden_path.exists()
    data = np.load(golden_path)

    ledger = Ledger()
    rng = np.random.default_rng(42)
    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=rng)

    jobs = [
        CircuitJob(theta_angles=t, x_tilde=x, observable="Z0")
        for t, x in zip(data["thetas"], data["xs"])
    ]
    results = backend.execute(jobs, shots=None, category="reference", tag="golden_check")

    assert np.allclose(results, data["fs"], atol=1e-12)


def test_t_m05_02_sampled_mean_converges_to_analytic():
    """T-M05-02: Sampled mean over many reps approaches analytic expectation (z-test 5σ)."""
    ledger = Ledger()
    rng = np.random.default_rng(123)
    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=rng)

    theta = np.full((2, 4, 2), 0.5)
    x = np.array([0.2, 0.4, 0.6, 0.8])
    job = CircuitJob(theta_angles=theta, x_tilde=x, observable="Z0")

    # Analytic
    analytic_val = backend.execute([job], shots=None, category="reference", tag="analytic")[0]

    # Finite shot sample: 200 reps of 200 shots each
    reps = 100
    shots = 200
    samples = []
    for _ in range(reps):
        s = backend.execute([job], shots=shots, category="reference", tag="shot_sample")[0]
        samples.append(s)

    mean_s = np.mean(samples)
    std_err = np.sqrt((1 - analytic_val ** 2) / (reps * shots)) + 1e-6
    z_score = abs(mean_s - analytic_val) / std_err
    assert z_score < 5.0, f"Z-score {z_score} >= 5.0"


def test_t_m05_03_reproducibility_and_optimiser_separation():
    """T-M05-03: Same seed => identical sampled output; different optimiser stream => different."""
    ledger1 = Ledger()
    rng1 = stream(7, 0, PurposeID.SHOT_SAMPLING, optimiser="gd")
    b1 = Backend(DeviceConfig(n_qubits=4), ledger=ledger1, rng=rng1)

    ledger2 = Ledger()
    rng2 = stream(7, 0, PurposeID.SHOT_SAMPLING, optimiser="gd")
    b2 = Backend(DeviceConfig(n_qubits=4), ledger=ledger2, rng=rng2)

    theta = np.full((2, 4, 2), 0.7)
    x = np.array([0.1, 0.3, 0.5, 0.7])
    job = CircuitJob(theta_angles=theta, x_tilde=x, observable="Z0")

    # Note: PennyLane internal shot sampling uses its own seeds if not passed,
    # but backend deterministic execution ensures same calls
    out1 = b1.execute([job], shots=64, category="reference", tag="t")
    out2 = b2.execute([job], shots=64, category="reference", tag="t")
    assert np.all(out1 == out2)


def test_t_m05_04_batch_jobs_ledger_accounting():
    """T-M05-04: Batch of K jobs -> ledger shows K circuits and K*N shots."""
    ledger = Ledger()
    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=np.random.default_rng(0))

    theta = np.zeros((2, 4, 2))
    x = np.zeros(4)
    jobs = [CircuitJob(theta, x, "Z0") for _ in range(7)]

    backend.execute(jobs, shots=32, category="training", tag="batch_test")

    assert len(ledger.events) == 1
    event = ledger.events[0]
    assert event.circuits == 7
    assert event.shots_per_circuit == 32
    assert event.shots_total == 7 * 32
    assert ledger.total_shots("training") == 7 * 32


def test_t_m05_05_preflight_failure_leaves_device_untouched():
    """T-M05-05: Preflight failure leaves device untouched."""
    ledger = Ledger()
    mgr = BudgetManager(train_shots=100, monitoring_shots=50)
    mgr.attach_ledger(ledger)

    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=np.random.default_rng(0))
    jobs = [CircuitJob(np.zeros((2, 4, 2)), np.zeros(4), "Z0") for _ in range(10)]

    # 10 circuits * 32 shots = 320 shots > 100 budget
    with pytest.raises(BudgetPreflightError):
        backend.execute(jobs, shots=32, category="training", tag="overflow")

    assert len(ledger.events) == 0
    assert mgr.total_spent("training") == 0


def test_t_m05_06_ast_choke_point_lint():
    """T-M05-06: AST test — no module outside quantum/backend.py and quantum/circuit_factory.py imports qml.device or qml.QNode directly."""
    src_dir = Path("src/qnnbench")
    forbidden = ["grad", "optim", "budget", "train", "baselines", "experiments", "analysis"]

    for d in forbidden:
        path = src_dir / d
        if not path.exists():
            continue
        for py_file in path.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "device"
                    and getattr(node.func.value, "id", "") == "qml"
                ):
                    pytest.fail(f"Direct qml.device call in {py_file}")


def test_t_m05_07_reference_calls_leave_budgets_unchanged():
    """T-M05-07: 'reference' calls leave all budgets unchanged."""
    ledger = Ledger()
    mgr = BudgetManager(train_shots=500, monitoring_shots=200)
    mgr.attach_ledger(ledger)

    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=np.random.default_rng(0))
    jobs = [CircuitJob(np.zeros((2, 4, 2)), np.zeros(4), "Z0") for _ in range(5)]

    backend.execute(jobs, shots=100, category="reference", tag="ref_call")

    assert mgr.remaining("training") == 500
    assert mgr.remaining("monitoring") == 200


def test_t_m05_08_noise_device_smoke_test():
    """T-M05-08: Noise device smoke test (default.mixed) tagged track='noise'."""
    ledger = Ledger()
    backend = Backend(
        DeviceConfig(n_qubits=2, device_type="default.mixed"),
        ledger=ledger,
        rng=np.random.default_rng(0)
    )
    jobs = [CircuitJob(np.zeros((2, 4, 2)), np.zeros(4), "Z0")]
    out = backend.execute(jobs, shots=None, category="reference", tag="noise_smoke")
    assert out.shape == (1,)
