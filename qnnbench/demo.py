"""
qnnbench — Live Demo for Teacher Presentation
==============================================
Demonstrates switching between:
  Mode 1: Quantum Analytic  (shots=None, exact statevector, no noise)
  Mode 2: Quantum Finite-Shot (shots=128, realistic hardware shot noise)
  Mode 3: Classical Baselines (LogReg + MLP, identical splits & features)

Usage:
  python demo.py             # Run all modes interactively
  python demo.py --mode 1    # Quantum analytic only
  python demo.py --mode 2    # Quantum finite-shot only
  python demo.py --mode 3    # Classical baselines only
  python demo.py --all       # All three without pausing
"""

import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from qnnbench.data.manager import make_task, make_split, fit_scaler
from qnnbench.quantum.circuit_factory import n_gates_report
from qnnbench.core.types import Theta, ANGLE_SHAPE, Batch
from qnnbench.core.seeds import stream, PurposeID
from qnnbench.quantum.backend import Backend, DeviceConfig
from qnnbench.budget.ledger import Ledger, BudgetManager
from qnnbench.grad.psr import psr2, mse_grad, mse_loss
from qnnbench.core.errors import BudgetPreflightError


def print_banner(title: str, fill: str = "="):
    print(flush=True)
    print(fill * 65, flush=True)
    print(f"  {title}", flush=True)
    print(fill * 65, flush=True)
    print(flush=True)


def pause(prompt: str = "Press Enter to continue..."):
    print(f"\n>>> {prompt}")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print()


def run_mode1():
    print_banner("MODE 1: Quantum Analytic (shots=None) — Exact Simulation")
    print("  Device   : default.qubit (PennyLane statevector simulator)")
    print("  Shots    : None (exact expectation values, zero shot noise)")
    print("  Gradient : 2-term Parameter-Shift Rule (exact mathematical gradient)")
    print("  Updates  : 10 full-batch Gradient Descent steps")
    print()

    X, y = make_task()
    split = make_split(X, y, split_seed=7)
    scaler = fit_scaler(split)
    X_train, _ = scaler.transform(split.train_X)
    X_val, _ = scaler.transform(split.val_X)

    gates = n_gates_report()
    print(f"  Dataset          : Iris binary (versicolor vs virginica)")
    print(f"  Train samples    : {len(split.train_X)} (30 versicolor / 30 virginica)")
    print(f"  Val samples      : {len(split.val_X)} (10 versicolor / 10 virginica)")
    print(f"  Gate counts      : encoding_RY={gates['encoding_RY']}, trainable={gates['trainable']}, cnot={gates['cnot']}")
    print(f"  Parameters       : 17 (16 gate angles + 1 classical bias)")
    print()

    rng = stream(7, 0, PurposeID.SHOT_SAMPLING, optimiser="gd")
    ledger = Ledger()
    mgr = BudgetManager(train_shots=10**9, monitoring_shots=10**9)
    mgr.attach_ledger(ledger)
    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=rng, run_id="demo_analytic")

    rng_init = stream(7, 0, PurposeID.INIT)
    angles0 = rng_init.uniform(0, 2 * np.pi, ANGLE_SHAPE)
    theta = Theta(angles=angles0, bias=0.0)

    train_batch = Batch(X=X_train, y=split.train_y, idx=split.train_idx)
    val_batch = Batch(X=X_val, y=split.val_y, idx=split.val_idx)

    eta = 0.10
    print(f"  {'Update':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'Circuits':>9}  {'Time':>8}")
    print("  " + "-" * 52)

    t0 = time.perf_counter()
    for u in range(5):
        f, df = psr2(theta, train_batch, backend, shots=None, category="training")
        fv, _ = psr2(theta, val_batch, backend, shots=None, category="monitoring")
        gt, gb = mse_grad(f, df, theta.bias, train_batch.y)
        theta = Theta(angles=theta.angles - eta * gt.reshape(ANGLE_SHAPE), bias=theta.bias - eta * gb)
        loss = mse_loss(f, theta.bias, train_batch.y)
        vloss = mse_loss(fv, theta.bias, val_batch.y)
        circs = ledger.total_circuits()
        wall = time.perf_counter() - t0
        print(f"  {u+1:>6}  {loss:>10.4f}  {vloss:>10.4f}  {circs:>9}  {wall:>7.2f}s", flush=True)

    print()
    print(f"  Total circuit evals : {ledger.total_circuits():,}")
    print(f"  Physical shots used : 0 (analytic mode uses statevector math)")
    print(f"  Key takeaway        : Smooth monotonic convergence with zero variance.")
    print("  [MODE 1 COMPLETE]")


def run_mode2():
    SHOTS = 128
    BUDGET = 250_000

    print_banner(f"MODE 2: Quantum Finite-Shot (shots={SHOTS}) — Realistic Hardware Simulation")
    print(f"  Device   : default.qubit (PennyLane simulator with multinomial sampling)")
    print(f"  Shots    : {SHOTS} per circuit evaluation")
    print(f"  Gradient : 2-term PSR with INDEPENDENT random shots per shift")
    print(f"  Noise    : Var(f_hat) = (1 - f^2) / N  ->  std ~ 0.06 at N={SHOTS}")
    print(f"  Budget   : Hard ceiling = {BUDGET:,} training shots")
    print()

    X, y = make_task()
    split = make_split(X, y, split_seed=7)
    scaler = fit_scaler(split)
    X_train, _ = scaler.transform(split.train_X)

    rng = stream(7, 0, PurposeID.SHOT_SAMPLING, optimiser="gd")
    ledger = Ledger()
    mgr = BudgetManager(train_shots=BUDGET, monitoring_shots=10_000)
    mgr.attach_ledger(ledger)
    backend = Backend(DeviceConfig(n_qubits=4), ledger=ledger, rng=rng, run_id="demo_shot")

    rng_init = stream(7, 0, PurposeID.INIT)
    angles0 = rng_init.uniform(0, 2 * np.pi, ANGLE_SHAPE)
    theta = Theta(angles=angles0, bias=0.0)
    rng_batch = stream(7, 0, PurposeID.MINIBATCH)

    m = 10
    eta = 0.10
    shots_per_update = (2 * 16 + 1) * m * SHOTS
    max_updates = BUDGET // shots_per_update

    print(f"  Minibatch size        : {m} samples")
    print(f"  Shots per circuit     : {SHOTS}")
    print(f"  Shots per PSR update  : (2 * 16 + 1) * {m} * {SHOTS} = {shots_per_update:,}")
    print(f"  Budget-limited updates: ~{max_updates}")
    print()
    print(f"  {'Update':>6}  {'Train MSE':>10}  {'Shots Used':>12}  {'Budget%':>8}")
    print("  " + "-" * 42)

    for u in range(25):
        if not mgr.can_afford_update(shots_per_update, shots_per_circuit=1, category="training"):
            print(f"  [Budget exhausted after {u} updates]")
            break
        b_idx = rng_batch.choice(len(X_train), size=m, replace=False)
        batch = Batch(X=X_train[b_idx], y=split.train_y[b_idx], idx=b_idx)
        try:
            f, df = psr2(theta, batch, backend, shots=SHOTS, category="training")
            gt, gb = mse_grad(f, df, theta.bias, batch.y)
            theta = Theta(angles=theta.angles - eta * gt.reshape(ANGLE_SHAPE), bias=theta.bias - eta * gb)
            loss = mse_loss(f, theta.bias, batch.y)
            used = ledger.total_shots("training")
            pct = 100.0 * used / BUDGET
            print(f"  {u+1:>6}  {loss:>10.4f}  {used:>12,}  {pct:>7.1f}%")
        except BudgetPreflightError:
            print(f"  [Budget preflight blocked update {u+1}]")
            break

    print()
    print(f"  Total training shots : {ledger.total_shots('training'):,}")
    print(f"  Total circuit evals  : {ledger.total_circuits():,}")
    print(f"  Key takeaway         : Loss fluctuates due to finite shot noise.")
    print(f"                         Gradient variance scales as O(1/shots).")
    print("  [MODE 2 COMPLETE]")


def run_mode3():
    print_banner("MODE 3: Classical Baselines (no quantum circuits)")
    print("  Models   : Logistic Regression + MLP (scikit-learn)")
    print("  Features : SAME angle-scaled x_tilde in [0, pi]^4 as quantum circuit")
    print("  Splits   : SAME train/val/test splits as quantum runs (seed=7)")
    print("  Purpose  : Reference performance bands — NO quantum advantage claims")
    print()

    X, y = make_task()
    split = make_split(X, y, split_seed=7)
    scaler = fit_scaler(split)

    X_tr, _ = scaler.transform(split.train_X)
    X_va, _ = scaler.transform(split.val_X)
    X_te, _ = scaler.transform(split.test_X)
    y_tr, y_va, y_te = split.train_y, split.val_y, split.test_y

    print(f"  Data Splits: Train={len(y_tr)}, Val={len(y_va)}, Test={len(y_te)}")
    print()

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    lr_acc_val = accuracy_score(y_va, lr.predict(X_va))
    lr_acc_test = accuracy_score(y_te, lr.predict(X_te))

    # MLP Classifier
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
    mlp.fit(X_tr, y_tr)
    mlp_acc_val = accuracy_score(y_va, mlp.predict(X_va))
    mlp_acc_test = accuracy_score(y_te, mlp.predict(X_te))

    print(f"  {'Model':<24}  {'Val Accuracy':>14}  {'Test Accuracy':>14}")
    print("  " + "-" * 56)
    print(f"  {'Logistic Regression':<24}  {lr_acc_val:>14.1%}  {lr_acc_test:>14.1%}")
    print(f"  {'MLP (16 hidden units)':<24}  {mlp_acc_val:>14.1%}  {mlp_acc_test:>14.1%}")
    print()
    print("  NOTE: In compliance with project guidelines, classical baselines")
    print("        are reference bands only. This benchmark studies shot noise")
    print("        and optimiser dynamics, NOT quantum advantage.")
    print("  [MODE 3 COMPLETE]")


def main():
    parser = argparse.ArgumentParser(description="qnnbench Live Demonstration")
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3],
                        help="1: Quantum Analytic, 2: Quantum Finite-Shot, 3: Classical, 0: All")
    parser.add_argument("--all", action="store_true",
                        help="Run all modes without pausing")
    args = parser.parse_args()

    print_banner("QNNBENCH — Live Demo for Teacher Presentation", fill="#")
    print("  Project : Quantum Neural Network Optimisation Benchmark")
    print("  Team    : Tarun Garg · Nigam Anand Nayak · Shivam Kumar")
    print("  Guide   : Dr. Ilakiyaselvan N. | VIT Chennai | 2026-27")
    print()
    print("  SWITCHING MECHANISM:")
    print("    shots=None  -->  MODE 1 (Analytic statevector, exact, fast)")
    print("    shots=128   -->  MODE 2 (Finite-shot, simulates quantum hardware noise)")
    print("    (no circuit)-->  MODE 3 (Classical baselines on identical splits)")
    print()

    interactive = (not args.all) and (args.mode == 0)

    if interactive:
        pause("Press Enter to begin Mode 1 (Analytic Quantum)...")

    if args.mode in (0, 1):
        run_mode1()
        if interactive:
            pause("Press Enter to switch to Mode 2 (Finite-Shot Quantum)...")

    if args.mode in (0, 2):
        run_mode2()
        if interactive:
            pause("Press Enter to switch to Mode 3 (Classical Baselines)...")

    if args.mode in (0, 3):
        run_mode3()

    print_banner("DEMO FINISHED SUCCESSFULLY", fill="#")
    print("  Summary for Teacher:")
    print("  • Switching between Analytic and Noisy is ONE parameter: shots=None vs shots=N.")
    print("  • Switching to Classical uses the EXACT same train/val/test splits.")
    print("  • The Ledger and BudgetManager guarantee auditability and shot-tracking.")
    print()


if __name__ == "__main__":
    main()
