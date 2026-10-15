# 🔬 qnnbench — Quantum Neural Network Benchmarking Engine

[![PennyLane](https://img.shields.io/badge/PennyLane-0.38+-teal.svg?style=flat-square)](https://pennylane.ai/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg?style=flat-square)](tests/)

`qnnbench` is the core Python benchmarking engine of the **QNNO** project. It provides audit-grade budget tracking, exact and noisy quantum gradient estimators, and standardized optimization experiments on Parameterized Quantum Circuits (VQCs).

---

## 📁 Module Organization

| Module | Location | Purpose |
| :--- | :--- | :--- |
| **Backend** | `src/qnnbench/quantum/backend.py` | **The Single Execution Choke-Point**: all PennyLane device executions pass through here; emits immutable ledger events |
| **Circuits** | `src/qnnbench/quantum/circuit_factory.py` | 4-qubit hardware-efficient ansatz, $R_y$ feature encoding, and ring CNOT entangling layers |
| **Gradients** | `src/qnnbench/grad/psr.py` | Exact and finite-shot 2-term and 4-term Parameter-Shift Rule (PSR) |
| **Metric Tensor** | `src/qnnbench/grad/metric.py` | Fubini-Study metric tensor approximation for Quantum Natural Gradient (QNG) |
| **Optimizers** | `src/qnnbench/optim/` | Gradient Descent (`gd.py`), Momentum (`momentum.py`), Adam (`adam.py`), SPSA (`spsa.py`), and QNG (`qng.py`) |
| **Budget Ledger** | `src/qnnbench/budget/ledger.py` | Append-only event log and preflight budget constraint manager |
| **Experiments** | `src/qnnbench/experiments/` | Standardized experimental protocols (E1 through E8) |
| **Data Engine** | `src/qnnbench/data/manager.py` | Deterministic stratified train/val/test splits and $[0, \pi]^4$ angle scaling |

---

## ⚡ Quickstart

### Automated Environment Setup
```powershell
.\setup_env.ps1
```

### Live Presentation Runner
```powershell
# Interactive walkthrough
.\demo.ps1

# Or specific modes:
.\demo.ps1 -Mode 1    # Quantum Analytic (shots=None)
.\demo.ps1 -Mode 2    # Quantum Finite-Shot (shots=128)
.\demo.ps1 -Mode 3    # Classical Baseline (LogReg + MLP)
```

### Running Test Suites
```powershell
.\run_tests.ps1 -Suite fast     # Quick unit and property checks
.\run_tests.ps1 -Suite all      # Full test suite with coverage
```

For full documentation and system flow, refer to [**`FLOW.md`**](file:///d:/QNNO/qnnbench/FLOW.md) and [**`context.md`**](file:///d:/QNNO/context.md).
