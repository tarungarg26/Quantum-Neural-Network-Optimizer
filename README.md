# ⚛️ Quantum Neural Network Optimizer (QNNO)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.38+-teal.svg?style=flat-square&logo=quantum)](https://pennylane.ai/)
[![React 19](https://img.shields.io/badge/React-19.2+-61DAFB.svg?style=flat-square&logo=react)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.3+-646CFF.svg?style=flat-square&logo=vite)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg?style=flat-square)](#-testing--validation)

> **A Rigorous Empirical Benchmark and Interactive Visualizer for First- and Second-Order Optimizers on Parameterized Quantum Circuits Under Finite-Shot Noise.**

Developed at **Vellore Institute of Technology (VIT Chennai)**  
**Authors**: Tarun Garg · Nigam Anand Nayak · Shivam Kumar  
**Project Guide**: Dr. Ilakiyaselvan N. (2026–2027)

---

## 🌟 Overview

Parameterized Quantum Circuits (Variational Quantum Circuits / QNNs) are leading candidates for near-term quantum advantage. However, on Noisy Intermediate-Scale Quantum (NISQ) devices, expectation values are not exact—they are estimated via a finite number of physical measurement shots $N$, introducing stochastic **quantum shot noise**:

$$\operatorname{Var}\left(\hat{f}\right) = \frac{1 - f^2}{N}$$

**QNNO** provides an end-to-end scientific benchmark and real-time visualization platform that:
1. **Audits Quantum Resource Consumption**: Enforces a strict, append-only budget ledger (`Ledger` & `BudgetManager`) that tracks every circuit evaluation and shot, preventing unmetered gradient evaluations.
2. **Benchmarks Multiple Optimizers**: Evaluates Gradient Descent (GD), Classical Momentum, Adam, SPSA (Simultaneous Perturbation Stochastic Approximation), and Quantum Natural Gradient (QNG with Fubini-Study metric tensor).
3. **Features Real-World Datasets**: Evaluates both quantum and classical architectures on real datasets, including **Formula 1 Grand Prix Telemetry** (OpenF1 / FastF1), **Wine Chemical Cultivars**, **Breast Cancer Diagnostic Biometrics**, and the canonical **Iris Flower Benchmark**.
4. **Interactive Browser Dashboard**: Real-time React + Vite web dashboard displaying live quantum state superposition rings, Bloch spheres, parameter weight heatmaps, and side-by-side QNN vs. classical MLP comparisons.

---

## 🏗️ Repository Architecture

```text
Quantum-Neural-Network-Optimizer/
├── qnnbench/                     # Scientific benchmarking engine (Python + PennyLane)
│   ├── src/qnnbench/
│   │   ├── quantum/             # Backend execution choke-point & 4-qubit circuit factory
│   │   ├── grad/                # 2-term & 4-term Parameter-Shift Rule (PSR) + Fubini metric
│   │   ├── optim/               # GD, Momentum, Adam, SPSA, and QNG optimizers
│   │   ├── budget/              # Append-only Ledger & preflight BudgetManager
│   │   ├── data/                # Data manager, scaling, and stratified splitting
│   │   ├── experiments/         # Standardized benchmark pipelines (E1–E8)
│   │   └── viz/                 # Publication-ready matplotlib & seaborn figures
│   ├── tests/                   # 100% passing test suite (Unit, Property, Regression, Numeric)
│   ├── demo.py                  # Live demonstration script (interactive terminal)
│   ├── demo.ps1                 # PowerShell demo wrapper with automatic environment discovery
│   └── setup_env.ps1            # Automated virtual environment setup script
│
├── qnn-visualizer/              # Interactive Web Visualizer (React + Vite + D3)
│   ├── src/
│   │   ├── engine/              # In-browser quantum simulator & real-world datasets
│   │   │   ├── datasets.js      # Real Formula 1, Wine, Cancer, and Iris datasets
│   │   │   └── quantumEngine.js # Statevector math, shot-noise sampling, and classical MLP
│   │   ├── components/          # Canvas network, Bloch spheres, metrics, and circuit schematics
│   │   └── hooks/               # useTrainingEngine reactive state hook
│   └── package.json
│
├── FLOW.md                      # Comprehensive system flow & phase gate documentation
├── context.md                   # Formal project specifications & scientific invariants
└── .gitignore                   # Multi-tier exclusion for Python, Node, and test caches
```

---

## ⚡ The Switching Mechanism: Quantum vs. Classical

A primary design requirement of this benchmark is transparent comparability without cherry-picking or data drift:

```text
shots = None     -->   QUANTUM ANALYTIC     (Exact statevector, zero noise)
shots = 128      -->   QUANTUM FINITE-SHOT  (Simulates real physical measurement shot noise)
(Classical net)  -->   CLASSICAL BASELINE   (Identical splits, features, and evaluation criteria)
```

1. **Exact Same Inputs**: Both QNN and MLP receive the identical angle-scaled features $\tilde{x} \in [0, \pi]^4$.
2. **Exact Same Splits**: Deterministic, seed-locked train ($60\%$), validation ($20\%$), and test ($20\%$) partitions.
3. **Audit Invariant**: Reference baseline computations never consume quantum shot allocations.

---

## 📊 Benchmark Datasets

The platform includes 4 real-world benchmark datasets:

| Dataset | Domain | 4 Selected Features | Classification Task | Model Dynamics |
| :--- | :--- | :--- | :--- | :--- |
| 🏎️ **Formula 1 Telemetry** | Motorsport (*FastF1 / OpenF1*) | Sector 1 (s), Sector 2 (s), Sector 3 (s), Speed Trap (km/h) | Attack / Push Lap pace vs. Tire Preservation / In-lap pace | Both learn well ($78\% - 85\%$) |
| 🧬 **Breast Cancer Wisconsin** | Oncology (*UCI ML / Clinical*) | Mean Radius, Mean Texture, Mean Perimeter, Mean Smoothness | Malignant vs. Benign diagnosis | **MLP excels ($87.5\%$ vs. $65.6\%$)** |
| 🍷 **Wine Chemical Profiling** | Chemistry (*UCI ML*) | Alcohol %, Flavanoids, Color Intensity, Proline (mg/L) | Barolo vs. Grignolino cultivars | High accuracy ($94\% - 98\%$) |
| 🌸 **Iris Flower Benchmark** | Botany (*Fisher Canonical*) | Sepal Length, Sepal Width, Petal Length, Petal Width | Versicolor vs. Virginica | $100\%$ accuracy on both |

---

## 🔬 Scientific Experiments (E1–E8)

The Python benchmark (`qnnbench`) implements eight standardized, reproducible experiment protocols:

- **E1 — Two-Term Parameter-Shift Rule (PSR)**: Evaluates analytic expectation value gradients via $\theta \pm \frac{\pi}{2}$ shifts.
- **E2 — Four-Term PSR**: Evaluates higher-order gate generators with multi-frequency spectra.
- **E3 — Analytic Baseline Benchmark**: Convergence curves under infinite budget (`shots=None`) across all 5 optimizers.
- **E4 — Equal-Shot Budgeted Comparison**: Strict $B = 50{,}000$ training shot ceiling across optimizers.
- **E5 — Shot Noise Scaling**: Empirical variance verification against theoretical $O(1/N)$ curves.
- **E6 — Trainability vs. Entanglement**: Barren plateau sensitivity under varying circuit depths and entanglement topologies.
- **E7 — Quantum Natural Gradient (QNG)**: Evaluates Fubini-Study block-diagonal metric tensor inversion cost.
- **E8 — Adaptive Shot Allocation**: Dynamic shot scheduling per training update.

---

## 🚀 Quick Start Guide

### 1. Python Scientific Benchmark (`qnnbench`)

#### Setup Environment
```powershell
cd D:\QNNO\qnnbench

# One-command automated setup (creates .venv, installs dependencies, runs smoke test)
.\setup_env.ps1
```

#### Run Interactive Live Demo (Recommended for Presentations)
```powershell
# Interactive mode (pauses between Analytic, Finite-Shot, and Classical sections)
.\demo.ps1

# Or run specific modes directly:
.\demo.ps1 -Mode 1   # Mode 1: Quantum Analytic (shots=None, exact)
.\demo.ps1 -Mode 2   # Mode 2: Finite-Shot Quantum (shots=128, realistic noise)
.\demo.ps1 -Mode 3   # Mode 3: Classical Baselines (LogReg + MLP)
.\demo.ps1 -All      # Unattended end-to-end run
```

#### Run Automated Test Suite
```powershell
.\run_tests.ps1 -Suite fast     # Run fast unit and property tests
.\run_tests.ps1 -Suite all      # Run complete test suite (unit, integration, regression)
```

---

### 2. Live Web Visualizer (`qnn-visualizer`)

The interactive web dashboard runs locally in any modern browser:

```powershell
cd D:\QNNO\qnn-visualizer
npm install
npm run dev
```

Open your browser at **`http://localhost:5173/`**.

#### Features in the Web Dashboard:
- **Dataset Switcher**: Toggle instantly between **Formula 1 Telemetry**, **Breast Cancer**, **Wine Quality**, and **Iris**.
- **Mode Toggle**: Switch between **Quantum (QNN)**, **Classical (MLP)**, or **Side-by-Side (Both)**.
- **Physical Shot Selector**: Choose between `∞` (Analytic statevector) and `32`, `128`, `512`, or `1024` shots to visualize physical noise variance.
- **Interactive Visualizations**:
  - **Neural Net Canvas**: Visualizes node activations and parameter edge weights with real-time pruning ghosting.
  - **Bloch Spheres**: Displays 3D qubit statevector rotations under variational gates.
  - **Circuit Schematic**: Interactive PennyLane 4-qubit layout with $R_y(\tilde{x})$ encodings and ring CNOT entanglers.
  - **Custom Inference Panel**: Test arbitrary feature inputs with instant classification confidence readouts.

---

## 🧪 Testing & Validation

The codebase enforces strict scientific integrity through multi-level automated test gates:

```powershell
# Run the complete test suite with coverage
pytest tests/ -v
```

- **Unit Tests**: Verifies quantum circuits, ledger balance conservation, gradient math, and optimizers.
- **Property-Based Tests (Hypothesis)**: Validates invariants across arbitrary random angles and input seeds.
- **Regression Guards**: Enforces architectural rules (e.g. guaranteeing `qml.QNode` is never imported outside `backend.py`).
- **Claims Linting**: Validates that no experiment asserts unproven "quantum advantage" claims without empirical statistical evidence.

---

## 👥 Authors & Academic Context

* **Project**: Quantum Neural Network Optimization Benchmark (QNNO)
* **Institution**: School of Computer Science and Engineering (SCOPE), **Vellore Institute of Technology, Chennai**
* **Academic Year**: 2026–2027
* **Team Members**:
  - **Tarun Garg** ([@tarungarg26](https://github.com/tarungarg26))
  - **Nigam Anand Nayak**
  - **Shivam Kumar**
* **Project Supervisor**: **Dr. Ilakiyaselvan N.**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
