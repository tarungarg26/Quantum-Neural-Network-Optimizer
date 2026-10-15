# QNNBENCH — Complete System Flow

> **Team:** Tarun Garg (24BCE5084), Nigam Anand Nayak (24BCE5202), Shivam Kumar (24BCE1001)
> **Guide:** Dr. Ilakiyaselvan N. · VIT Chennai · Software Engineering 2026-27

---

## Table of Contents

1. [Big Picture](#1-big-picture)
2. [Repository Layout](#2-repository-layout)
3. [Module-by-Module Flow](#3-module-by-module-flow)
4. [Data Flow Walkthrough (One Training Run)](#4-data-flow-walkthrough-one-training-run)
5. [Build Phases and Gates](#5-build-phases-and-gates)
6. [Test Strategy](#6-test-strategy)
7. [Hard Invariants (Never Violate)](#7-hard-invariants-never-violate)
8. [Key Mathematical Facts](#8-key-mathematical-facts)

---

## 1. Big Picture

```
                         +-----------------------------------------+
                         |           qnnbench benchmark            |
                         |                                         |
  Iris (binary) ------►  |  DataManager --► CircuitFactory         |
  versicolor / virginica  |       |               |                 |
  100 samples, 4 features |  Scaler (train-    QNode(4-qubit)      |
                         |  only fit)         <Z0> readout         |
                         |       |               |                 |
                         |       +-----► Backend.execute()  ◄──── THE ONLY place
                         |                       |                 |  QNodes run
                         |               LedgerEvent (every call)  |
                         |                       |                 |
                         |              BudgetManager.preflight    |
                         |                       |                 |
                         |         +─────────────+                 |
                         |         | GradEngines (PSR / SPSA)     |
                         |         | MetricProvider  (QNG)        |
                         |         |                               |
                         |    5 Optimisers ----► Trainer loop      |
                         |    GD/Mom/Adam       (80 updates        |
                         |    SPSA/QNG           analytic or       |
                         |                       finite-shot)      |
                         |                       |                 |
                         |               Checkpoints, best-val     |
                         |               TestAccess (once)         |
                         |                       |                 |
                         |               Logger --► runs/<id>/     |
                         +-----------------------------------------+
                                         |
                              Analysis --+--► Figures (F01-F24)
                              Stats             Tables (T01-T12)
                                               Report (HTML/PDF)
```

**Goal:** reproducible hybrid quantum-classical benchmark comparing 5 optimisers on one small variational classifier, with explicit shot/circuit-eval accounting. **No quantum-advantage claims.**

---

## 2. Repository Layout

```
qnnbench/
├─ pyproject.toml          # deps: pennylane, numpy, sklearn, hypothesis, ruff, pytest
├─ configs/
│  ├─ base.yaml            # FIXED benchmark (changing this voids all comparisons)
│  ├─ optimisers/          # {gd,momentum,adam,spsa,qng}.yaml
│  ├─ experiments/         # E1.yaml - E8.yaml
│  └─ frozen/              # hash-locked tuned configs
├─ src/qnnbench/
│  ├─ core/        types.py  errors.py  config.py  seeds.py  hashing.py
│  ├─ data/        manager.py  (iris.csv bundled)
│  ├─ quantum/     circuit_factory.py  backend.py
│  ├─ grad/        psr.py  metric.py
│  ├─ budget/      ledger.py  allocation.py
│  ├─ optim/       base.py  (GD/Mom/Adam/SPSA/QNG all in base.py currently)
│  ├─ train/       trainer.py  evaluator.py  checkpoint.py
│  ├─ baselines/   logreg.py  mlp.py
│  ├─ io/          logger.py
│  ├─ experiments/ e1_two_term.py - e8_allocation.py  runner.py
│  ├─ analysis/    compare.py  (tidy, stats, fits, tables)
│  ├─ viz/         (style, palette, figures F01-F24)
│  ├─ report/      build.py  claims_lint.py
│  └─ cli.py
├─ tests/
│  ├─ conftest.py           # shared fixtures
│  ├─ unit/                 # per-module tests
│  ├─ integration/
│  ├─ numeric/              # gradient correctness
│  ├─ property/             # hypothesis-based
│  ├─ regression/           # DA1->DA2 corrections + mutation guards
│  └─ golden/               # frozen reference arrays (npz/json)
└─ runs/<run_id>/           # (gitignored) trace.csv events.jsonl final.json
```

---

## 3. Module-by-Module Flow

### M01 — Config & Types

**Files:** `src/qnnbench/core/types.py`, `core/config.py`, `core/errors.py`

Defines immutable core dataclasses and the error taxonomy.

**Key types:**

| Type | Shape / fields | Purpose |
|------|---------------|---------|
| `Theta` | `angles (2,4,2) float64` + `bias float` | All 17 model parameters |
| `Batch` | `X (m,4)`, `y (m,)`, `idx (m,)` | Pre-scaled mini-batch |
| `RunKey` | `experiment`, `optimiser`, `split_seed`, `init_seed`, `config_hash` | Unique run identity |

`RunKey.run_id()` = `sha1(canonical_json(RunKey))[:12]` — one 12-char hex string ties together every CSV, JSON, NPZ for one run.

**Shape constants (derived once, not hardcoded twice):**
```python
ANGLE_SHAPE    = (2, 4, 2)      # (layers, wires, {RY, RZ})
N_GATE_PARAMS  = 16             # np.prod(ANGLE_SHAPE)
N_TOTAL_PARAMS = 17             # 16 angles + 1 classical bias
```

**Error taxonomy:**

| Error | When |
|-------|------|
| `BudgetExhausted` | Clean stop — run completed within budget |
| `BudgetPreflightError` | Op would exceed budget; raised *before* execution |
| `NonFiniteError` | NaN/Inf in loss, grad, or solve |
| `LeakageError` | Scaler fit on non-train data, or test accessed too early |
| `ConfigFrozenError` | Frozen config file tampered |
| `AccountingMismatchError` | Formula disagrees with ledger in a non-adaptive plan |

---

### M02 — Seeds (RNG Policy)

**File:** `src/qnnbench/core/seeds.py`

Ensures every random number comes from a deterministic seed tree — guaranteeing byte-identical replications.

```python
stream(split_seed, init_seed, purpose: PurposeID, optimiser=None) -> np.random.Generator
```

Uses `np.random.SeedSequence([split_seed, init_seed, PURPOSE_ID, optimiser_salt])`.

**Purpose IDs:**

| PurposeID | Shared across optimisers? | Used for |
|-----------|--------------------------|---------|
| `INIT` | Yes | theta_0 — IDENTICAL across all 5 optimisers |
| `MINIBATCH` | Yes | Minibatch index order — IDENTICAL across all 5 optimisers |
| `SHOT_SAMPLING` | No (keyed by optimiser) | Device seeds for shot sampling |
| `PILOT_SHOTS` | No | Allocation pilot shots |
| `SPSA_DELTA` | No | Rademacher perturbation vectors |
| `BOOTSTRAP` | Yes | Statistical bootstrap |
| `TUNING` | Yes | Hyperparameter search |

**Why pairing matters:** `INIT` and `MINIBATCH` being identical across optimisers means all 5 see the **same theta_0 and minibatch order** — differences in performance are purely due to the update rule.

**Banned (lint enforced):** `np.random.seed(...)`, `np.random.rand(...)`, `random.seed(...)`

---

### M03 — Data Manager & Preprocessing

**File:** `src/qnnbench/data/manager.py`

Loads Iris, creates binary task, performs stratified splits, applies train-only scaling.

**Pipeline:**
```
make_task()
  -> X_raw (100,4), y (100,) in {-1, +1}
  -> setosa excluded; versicolor=-1, virginica=+1

make_split(X, y, split_seed, scheme="main"|"historical")
  -> Split(train_X, train_y, val_X, val_y, test_X, test_y, ...)

fit_scaler(split)                   <- ONLY split.train_X is ever seen here
  -> Scaler(a, d, const_mask)

scaler.transform(X)
  -> x_tilde = pi * clip((x - a) / (d - a), 0, 1)  in [0, pi]
  -> clip_report (how many values were clipped)
```

**Split sizes (main scheme):**

| Subset | versicolor | virginica | Total |
|--------|-----------|----------|-------|
| Train  | 30        | 30       | 60    |
| Val    | 10        | 10       | 20    |
| Test   | 10        | 10       | 20    |

**Historical scheme:** 70/30, seed 7 — for DA1 carry-forward results only (labelled `track="historical"`).

**Leakage guards:** `Scaler.fit_guarded(X, is_train=False)` raises `LeakageError`.

---

### M04 — Circuit Factory

**File:** `src/qnnbench/quantum/circuit_factory.py`

Defines the fixed benchmark circuit.

**Circuit structure:**
```
|0000>
  -> RY(x_tilde_0)...RY(x_tilde_3)     [4 encoding gates]
  -> Layer 0:
       RY(theta[0,w,0]), RZ(theta[0,w,1]) for w in {0,1,2,3}  [8 gates]
       CNOT ring: 0->1, 1->2, 2->3, 3->0                       [4 gates]
  -> Layer 1:
       RY(theta[1,w,0]), RZ(theta[1,w,1]) for w in {0,1,2,3}  [8 gates]
       CNOT ring: 0->1, 1->2, 2->3, 3->0                       [4 gates]
  -> <Z_0>  (readout on wire 0 only)
```

**Gate count (uncompiled):**

| Gate type | Count |
|-----------|-------|
| Encoding RY | 4 |
| Trainable (RY + RZ per wire, 2 layers) | 16 |
| CNOT entanglers | 8 |
| **Total** | **28** |

**Parameter indexing:** `theta[layer, wire, {0:RY, 1:RZ}]` — shape `(2, 4, 2)`.

**Score and prediction:**
```
s = <Z_0> + b          (score)
y_hat = sign(s)        (prediction in {-1, +1})
loss = MSE vs +/-1 labels
```

**Observables:**
- `Z0` = `qml.PauliZ(0)` — local readout (baseline)
- `parity` = Z tensor Z tensor Z tensor Z — global, for E6 only

---

### M05 — Backend (Single Execution Choke Point)

**File:** `src/qnnbench/quantum/backend.py`

THE ONLY place in the codebase where a QNode is called. A lint test (T-M05-06) enforces this by AST-scanning all other modules.

**API:**
```python
class Backend:
    def execute(
        self,
        jobs: Sequence[CircuitJob],
        *,
        shots: int | None,     # None -> analytic; int -> finite-shot
        category: Category,    # "training"|"monitoring"|"pilot"|"reference"|"search"
        tag: str,
    ) -> np.ndarray            # shape (K,) expectation values
```

**Execution flow:**
```
Backend.execute(jobs, shots, category, tag)
  |
  +-- 1. Preflight budget check (BEFORE any device call)
  |        if category in {training, monitoring, pilot, search}:
  |            BudgetManager.preflight(category, circuits=K, shots_per_circuit=shots)
  |            -> raises BudgetPreflightError if K*shots > remaining
  |
  +-- 2. Execute K circuit jobs
  |        for each job:
  |            seed = rng.integers(...)     # deterministic from shot_sampling stream
  |            dev = qml.device("default.qubit", shots=shots, seed=seed)
  |            result = qnode(job.theta_angles, job.x_tilde)
  |
  +-- 3. Commit ledger event (AFTER successful execution)
           LedgerEvent(run_id, category, tag, circuits=K, shots_per_circuit, shots_total, wall_s)
           ledger.commit(event)
```

**Category reference chart:**

| Category | Charged? | Use case |
|----------|---------|---------|
| `training` | Yes | PSR forward + shifted circuits |
| `monitoring` | Yes | Validation evaluation at checkpoints |
| `pilot` | Yes | Variance-informed allocation pilots |
| `search` | Yes | Hyperparameter tuning |
| `reference` | **FREE** | Autodiff/backprop verification |

---

### M06 — Gradient Engines (PSR)

**File:** `src/qnnbench/grad/psr.py`

Implements the Parameter-Shift Rule. All quantum work goes through `Backend.execute()`.

**2-Term PSR:**
```
d_f/d_theta_k = (1/2) [f(theta + pi/2 * e_k) - f(theta - pi/2 * e_k)]
```
Cost per update: `(2p + 1) * m` circuit evaluations (p=16, m=batch size).

**4-Term PSR (E2 only, for CRot):**
```
c_plus  = (sqrt(2) + 1) / (4*sqrt(2))
c_minus = (sqrt(2) - 1) / (4*sqrt(2))
f' = c_plus*[f(theta+pi/2)-f(theta-pi/2)] - c_minus*[f(theta+3pi/2)-f(theta-3pi/2)]
```

**Loss functions:**
```python
mse_loss(f, b, y)              = mean((f+b-y)^2)
unbiased_mse_loss(f_hat, b, y, N) = mean((f_hat+b-y)^2 - (1-f_hat^2)/(N-1))
# N > 1 required; corrects positive bias from shot noise
```

**MSE Gradient (classical chain rule):**
```
d_C/d_theta_k = (2/m) * sum_i (f_i+b-y_i) * d_f_i/d_theta_k   # quantum contribution
d_C/d_b       = (2/m) * sum_i (f_i+b-y_i)                       # classical — NO shifts for bias
```

**Key invariant:** In finite-shot mode, forward `f_hat` and each shifted evaluation use **independent** random seeds — the product is unbiased.

---

### M07 — Metric Provider / QNG Geometry

**File:** `src/qnnbench/grad/metric.py`

Computes the Quantum Natural Gradient direction via the Fubini-Study metric.

**QNG update rule:**
```
(G + lambda*I) d = gradient_C    <- solve for direction d
theta <- theta - eta * d          <- angle params only
b     <- b - eta * d_C/d_b        <- classical step for bias (never through G)
```

**Metric:** Block-diagonal Fubini-Study, averaged over up to 8 training inputs. Metric tapes charged to ledger as actual executed tapes.

**Fallback (Decision D-06):** On solve failure, lambda <- 10*lambda, up to 3 retries, then plain gradient step. Every fallback is a ledger event — **never silent**.

---

### M08 — Budget Ledger & Manager

**Files:** `src/qnnbench/budget/ledger.py`

The authoritative accounting system. Ledger is the source of truth — formulas are cross-checked against it.

**Ledger event schema:**
```json
{
  "seq": 42,
  "run_id": "a3f1b2c4d5e6",
  "category": "training",
  "tag": "psr2_k3_p",
  "circuits": 10,
  "shots_per_circuit": 128,
  "shots_total": 1280,
  "update": 7,
  "wall_s": 0.043
}
```

**Budget enforcement flow:**
```
Backend.execute(jobs, shots, category, tag)
    |
    +-- ledger.preflight(category, circuits, shots)
    |       -> budget_manager.preflight(...)
    |           -> if circuits * shots > remaining: raise BudgetPreflightError
    |           -> device is NEVER called on failure
    |
    +-- [device executes K circuits]
    |
    +-- ledger.commit(event)
            -> budget_manager.commit(category, shots_total)
            -> remaining -= shots_total
```

**Key invariants:**
- Never overshoot, never silently shrink — ALL-OR-NOTHING
- Run ends when next **whole** update is unaffordable (`status = "completed_budget"`)
- Categories never leak across each other
- `reference` events never reduce any budget

**Arithmetic sanity:**
- N=32, m=10: PSR update = `(2*16+1)*10*32 = 10,560 shots` -> <= 9 updates in 1e5 budget
- SPSA: `2*10*32 = 640 shots/update` -> <= 156 updates in 1e5 budget

---

### M09 — Optimisers

**File:** `src/qnnbench/optim/base.py`

5 optimiser classes. **No circuit code, no data code, no budget code** — quantum work via grad engines only.

**Interface:**
```python
class Optimiser(ABC):
    def propose(self, theta, batch, backend, shots, update_idx) -> StepResult
    def apply(self, theta, step) -> Theta      # pure arithmetic
    def state_dict() / load_state_dict()
    def cost_per_update(m, shots) -> int       # circuits
```

**The 5 optimisers:**

| Optimiser | Update Rule | Hyperparams |
|-----------|-------------|-------------|
| **GD** | `theta <- theta - eta*grad` | eta in {0.03, 0.10, 0.30} |
| **Momentum** | `v <- mu*v + grad; theta <- theta - eta*v` | mu=0.9; same eta grid |
| **Adam** | Bias-corrected adaptive moments | beta1=0.9, beta2=0.999, eps=1e-8; eta in {0.01, 0.03, 0.10} |
| **SPSA** | 2-point simultaneous perturbation | a_t=a/(A+t+1)^0.602, c_t=c/(t+1)^0.101, c=0.15, a in {0.1,0.5,1.5} |
| **QNG** | Natural gradient via Fubini-Study | eta grid = GD grid; lambda in {0.001,0.01,0.05} |

**SPSA details:**
- Rademacher delta in {+/-1}^16 (angles only; bias gets classical gradient)
- Exactly **2 loss evaluations** per iteration on the **same** minibatch
- Schedule parameters logged explicitly in every trace row

---

### M10 — Allocation Policies

**File:** `src/qnnbench/budget/allocation.py`

**UniformAllocation(N):** Fixed N shots per circuit. Used for N in {32, 128, 512}.

**VarianceInformedAllocation (hypothesis under test, E8):**
```
1. Fire 16 pilot shots per shifted circuit -> estimate sigma_j = sqrt(max(1 - f_hat_j^2, floor))
2. Weight: a_j = (1/2) * |residual_i| / m
3. Allocate: N_j = S_prod * |a_j|*sigma_j / sum_r(|a_r|*sigma_r)
4. Floor: N_j >= 8 shots minimum
5. Largest-remainder integer rounding: sum(N_j) == S_prod exactly
6. Production samples are FRESH (pilots never reused in estimator)
7. Pilot shots charged to budget separately
```

**Rationale:** Minimises sum(a_j^2 * sigma_j^2 / N_j) subject to sum(N_j) = S_prod.

---

### M11 — Trainer, Evaluator, Checkpointing

**Files:** `src/qnnbench/train/trainer.py`, `evaluator.py`, `checkpoint.py`

Orchestrates the complete training loop for one run.

**Training loop (pseudocode):**
```python
# Paired init (INIT stream, shared across optimisers)
theta = Theta(angles=rng_init.uniform(0, 2*pi, (2,4,2)), bias=0.0)

# Paired minibatch (MINIBATCH stream, shared across optimisers)
rng_batch = stream(split_seed, init_seed, MINIBATCH)

# Optimiser-specific shot noise
backend = Backend(DeviceConfig(n_qubits=4), ledger, rng_shots)

for update in range(max_updates):
    batch = next_minibatch(rng_batch)

    # Budget preflight
    if not budget_mgr.can_afford_update(cost_circuits, shots):
        status = "completed_budget"; break

    step = optimiser.propose(theta, batch, backend, shots, update)
    theta = optimiser.apply(theta, step)

    # Checkpoint: evaluate val (monitoring budget)
    if update % eval_every == 0:
        val_mse, val_acc = evaluate(theta, val_batch, ...)
        tracker.step(update, val_mse, val_acc, theta, optimiser.state_dict())
        if early_stop: status = "early_stopped"; break

# Model selection: restore best-val-MSE checkpoint
best_theta = tracker.best_checkpoint.theta

# Test evaluation: ONCE, ONLY AFTER SELECTION
guard.unlock()
test_batch = guard.get_test_batch(scaler)   # raises on 2nd call
test_mse, test_acc = evaluate(best_theta, test_batch, category="reference")
```

**Analytic mode:** 80 full-batch updates (m=60)
**Finite-shot mode:** minibatch 10; shots in {32, 128, 512}; budgets in {1e5, 5e5, 1e6, 5e6}

**`TestAccessGuard`:** Raises `LeakageError` before `unlock()` and on 2nd call after unlock.

**Status values:**
- `"completed_updates"` — hit max_updates cleanly
- `"completed_budget"` — budget ran out
- `"early_stopped"` — patience exhausted
- `"failed"` — NonFiniteError or other exception (logged, never dropped)

---

### M12 — Classical Baselines

Logistic regression + 1-hidden-layer MLP (scikit-learn). Same splits, same train-only scaling. Test evaluated once. Used in F22 and T06 for reference (no advantage framing).

---

### M13 — IO, Logger, Schemas

**File:** `src/qnnbench/io/logger.py`

The only writer of run artefacts. Produces per-run output in `runs/<run_id>/`:

| File | Content |
|------|---------|
| `trace.csv` | Per-update/checkpoint rows |
| `events.jsonl` | Full ledger (all budget events) |
| `final.json` | Summary: status, metrics, totals |
| `arrays.npz` | theta trajectory, gradients, SPSA delta, QNG condition numbers |
| `config.yaml` | Config used for this run |
| `env.json` | Actual library versions from execution machine |

**`trace.csv` columns:**
`run_id, update, circuits_train, shots_train, shots_monitor, shots_pilot, wall_s, train_mse, val_mse, val_acc, grad_norm, step_norm, lr_eff, spsa_a_t, spsa_c_t, qng_damping, qng_cond, qng_fallback, clipped_val, clipped_test, track`

---

### M14 — Experiments (E1-E8)

Each experiment: `run(config) -> list[RunKey]`, writes only through M13.

| ID | Research Question | What it measures |
|----|------------------|-----------------|
| **E1** | RQ1: 2-term PSR correct? | PSR2 vs autodiff on 200 random (theta, x): max error <= 1e-10 |
| **E2** | RQ1: 4-term on CRot | PSR4 <= 1e-9; PSR2 on CRot shows error > 1e-2 (intentional) |
| **E3** | RQ2: Analytic comparison | 5 optimisers x 80 updates x 15 paired seeds; loss/acc curves |
| **E4** | RQ2: Equal-shot ranking | Budgets x shots x 5 optimisers x 15 seeds |
| **E5** | Shot-noise law | N in {8..4096}, R>=2000 reps; Var(f_hat)*N <= 1/2 |
| **E6** | RQ4: Trainability | Width {2,4,6,8,10,12}, depth, init, observable; paired draws |
| **E7** | RQ3: QNG geometry | Damping x metric inputs; cond(G) traces |
| **E8** | RQ5: Adaptive allocation | Variance-informed vs uniform at matched total shots |
| **E-hist** | Historical DA1 track | DA1 70/30 seed-7; labelled `track="historical"` |

---

### M15 — Analysis & Statistics

**File:** `src/qnnbench/analysis/compare.py`

The comparison engine used by every figure and table.

**Methods:**
- Paired differences on identical (split_seed, init_seed) blocks
- Cluster bootstrap by split seed (10k resamples, BCa) for confidence intervals
- Friedman test across 5 optimisers per (budget, shots)
- Wilcoxon signed-rank with Holm correction
- Performance profiles (Dolan-More style)
- Cost-to-target: shots to reach val MSE <= tau (right-censored, never dropped)
- Pareto front: (shots, val MSE) per optimiser

`fit_log_log(x, y).predict(x=16.0)` raises `ValueError("extrapolation")` if x > max(x_measured).

---

### M16 — Visualisation

24 figures (F01-F24), each a `def make(data: TidyBundle, out: Path) -> FigureArtifact`.

**Design rules (enforced by tests):**
- Okabe-Ito colour-blind-safe palette — fixed mapping per optimiser
- Every x-axis explicitly states `updates`, `circuit evaluations`, or `shots`
- Uncertainty always shown: mean line + 95% CI band (cluster-bootstrap)
- SVG/PDF (vector) + PNG @ 300 dpi
- Footer on every figure: `track | config_hash[:8] | n_runs | n_failed`
- F24 (historical) carries watermark `"HISTORICAL — UNVERIFIED PROTOCOL"`

**Claims lint (T-M16-09):** Scans captions, titles, templates for: `speedup`, `advantage`, `outperforms classical`, `energy`, `supremacy`, `faster than`.

---

### M17 — CLI, Orchestration, Report

**File:** `src/qnnbench/cli.py`

```
qnnbench run <E#> [--tiny]   Run experiment(s)
qnnbench tune                Hyperparameter tuning (validation only)
qnnbench freeze              Hash-lock tuned configs
qnnbench replicate           Run 15 paired replications per frozen config
qnnbench analyse             Load runs/, produce tidy DataFrames
qnnbench figures             Render all F01-F24
qnnbench report              Build HTML/PDF report
qnnbench doctor              Check env + write manifest
```

Parallelism by process pool over RunKey; per-run output dirs; resumable (skips complete runs).

---

## 4. Data Flow Walkthrough (One Training Run)

Complete data flow for a single GD run (split_seed=7, init_seed=0):

```
1. CLI: qnnbench run E3 --optimiser gd --split-seed 7 --init-seed 0
        |
        v
2. RunKey("E3", "gd", 7, 0, config_hash)
   run_id = sha1(RunKey)[:12]  ->  e.g. "a3f1b2c4d5e6"

3. make_task() -> X_raw (100,4), y (100,) in {-1,+1}
   make_split(X_raw, y, split_seed=7, scheme="main")
     -> train(60), val(20), test(20)  disjoint, stratified

4. fit_scaler(split)          <- ONLY split.train_X seen here
   -> Scaler(a, d)            <- train min/max per feature
   scaler.transform(train_X)  -> X_tilde_train in [0,pi]^4 per sample

5. stream(7, 0, INIT) -> rng_init
   theta_0 = Theta(angles=rng_init.uniform(0,2*pi,(2,4,2)), bias=0.0)
   <- IDENTICAL to every other optimiser with same (split_seed=7, init_seed=0)

6. BudgetManager(train_shots=inf, monitoring_shots=inf)  <- analytic run
   Ledger()  <- empty event log
   ledger.attach_budget_manager(budget_mgr)

7. Backend(DeviceConfig(n_qubits=4), ledger, rng_shots, run_id)
   <- rng_shots = stream(7, 0, SHOT_SAMPLING, optimiser="gd")
   <- creates qml.device("default.qubit", wires=4)

8. GDOptimiser(eta=0.10)

9. Training loop (80 updates, full-batch, analytic):
   for u in range(80):
     batch = Batch(X=X_tilde_train, y=train_y, idx=arange(60))  <- full batch

     +-- psr2(theta, batch, backend, shots=None, category="training")
     |     for k in 0..15:
     |       f_plus  = backend.execute([jobs+], shots=None, category="training", tag="psr2_k{k}_p")
     |         -> ledger.commit(LedgerEvent(circuits=60, shots_per_circuit=0, ...))
     |       f_minus = backend.execute([jobs-], ...)
     |       df[:,k] = 0.5 * (f_plus - f_minus)
     |     f = backend.execute([fwd_jobs], ...)
     +-- returns (f (60,), df (60,16))

     grad_theta, grad_b = mse_grad(f, df, theta.bias, batch.y)
     <- pure classical chain rule, NO quantum calls

     step = StepResult(grad_theta, grad_b, direction=grad_theta, ...)
     theta = GD.apply(theta, step)
     <- theta.angles -= eta * grad_theta.reshape(2,4,2)
     <- theta.bias   -= eta * grad_b

     if u % 5 == 0:   <- checkpoint
       val_mse, val_acc = evaluate(theta, val_batch, category="monitoring")
       tracker.step(u, val_mse, val_acc, theta, gd.state_dict())

     trace.append({update:u, val_mse, val_acc, wall_s})

10. best_theta = tracker.best_checkpoint.theta    <- argmin val_mse
    guard.unlock()
    test_batch = guard.get_test_batch(scaler)     <- ONLY call allowed
    test_mse, test_acc = evaluate(best_theta, test_batch, category="reference")
    <- reference: FREE, never charged to budget

11. logger.write_run(run_id, trace, ledger, final_theta, best_theta, config, env)
    -> runs/a3f1b2c4d5e6/
         trace.csv
         events.jsonl
         final.json
         arrays.npz
         config.yaml
         env.json
```

---

## 5. Build Phases and Gates

| Phase | Modules | Gate (must be green before next phase) |
|-------|---------|---------------------------------------|
| **P0** | Scaffold, CI, lint | `pytest tests/ -k "not slow"` passes on empty skeleton |
| **P1** | M01 M02 M13 M08 | All unit tests; ledger property tests |
| **P2** | M03 M04 M05 | Leakage test; circuit golden tests; ledger-reconciliation |
| **P3** | M06 M07 | **E1 passes** (PSR vs autodiff <= 1e-10); **E2 passes** (4-term <= 1e-9) |
| **P4** | M09 M11 M12 | Optimiser unit tests; E3 smoke (5 optimisers x 1 seed) |
| **P5** | Finite-shot, tuning | **E5: N*Var <= 1/2 + tolerance**; E4 smoke; budget-never-overshoot |
| **P6** | M10, E6 E7 E8 | Allocation invariants; paired-draw test; E8 matched-budget test |
| **P7** | M15 M16 M17 | All 24 figures render from fixture data; claims lint; schema tests |
| **P8** | Freeze + replicate | 15 runs/config; failures enumerated; report builds from `runs/` only |
| **P9** | Optional (noise, two-moons) | Separate track; never mixed into baseline |

**Coverage floors:** `core/ budget/ grad/ optim/` >= 95%; others >= 85%.

---

## 6. Test Strategy

| Marker | What | Example |
|--------|------|---------|
| `unit` | Single function, mocked deps | `test_theta_flat_round_trip` |
| `integration` | Multi-module path | `test_tiny_e2e` (3 updates, 2 optimisers) |
| `numeric` | Floating-point correctness | `test_psr2_vs_autodiff` (max delta <= 1e-10) |
| `property` | Hypothesis-based | `test_budget_never_overshoots` |
| `viz` | Figure rendering | All `fNN.make()` from fixture data |
| `slow` | > 10 s | Full 80-update E3 run |
| `regression` | DA1->DA2 corrections + guards | `test_correction_1_no_leakage` |

**DA1->DA2 Regression corrections (named tests):**
1. `test_correction_1_no_leakage` — Scaler fit on train only
2. `test_correction_2_backprop_is_labelled_reference` — autodiff is reference category
3. `test_correction_3_paired_variance_draws` — same INIT stream across optimisers
4. `test_correction_4_cost_formula` — (2p+1)m = 2310 for p=16, m=70
5. `test_correction_5_no_extrapolation` — fit_log_log.predict raises beyond measured x
6. `test_correction_6_historical_track_isolated` — historical rows excluded from main

**Mutation guards:**
- `test_guard_test_access_once_only` — TestAccessGuard raises on double/early access
- `test_guard_budget_overshoot_raises` — BudgetPreflightError on overshoot
- `test_guard_tampered_frozen_config` — ConfigFrozenError on hash mismatch
- `test_guard_non_finite_values_raise` — NonFiniteError on NaN input

---

## 7. Hard Invariants (Never Violate)

| Invariant | Where Enforced |
|-----------|---------------|
| All circuit executions through `Backend.execute()` | T-M05-06 (AST lint) |
| Budget never overshoots | T-M08-01 (property test) |
| Scaler fit on train-only | `LeakageError`, T-M03-04 |
| Test set accessed exactly once | `TestAccessGuard`, T-M11-02 |
| Failed runs logged, never dropped/reseeded | T-M11-09, T-M15-06 |
| `reference` category never charged | T-M05-07 |
| PSR/SPSA ratio = 16.5 (not 17) | T-M06-06 |
| Cost formula: (2p+1)m for PSR | T-M06-05 |
| Historical track isolated | T-M15-08, F24 watermark |
| No speedup/advantage/energy claims | T-M16-09 claims lint |
| Pairing: theta_0 and minibatches identical across optimisers | T-M09-02 |
| Config hash immutable after freeze | `ConfigFrozenError` |
| Determinism: same config+seeds -> byte-identical trace.csv | T-M14-E*-02 |

---

## 8. Key Mathematical Facts

| Fact | Value |
|------|-------|
| Circuit parameters p | 16 (angles only; bias excluded from PSR) |
| Total parameters | 17 (16 + bias) |
| PSR shift | pi/2 |
| PSR cost per update | (2*16+1)*m = 33m circuits |
| SPSA cost per update | 2m circuits |
| PSR/SPSA ratio | 16.5 |
| Shot noise variance | Var(f_hat) = (1-f^2)/N |
| Shot noise per gradient | Var(g_hat_k) <= 1/(2N) |
| Accuracy shots needed | N >= 1/(2*eps^2); eps=0.01 -> N=5000 |
| Unbiased loss correction | (f_hat+b-y)^2 - (1-f_hat^2)/(N-1) |
| 4-term PSR coefficients | c_plus/minus = (sqrt(2) +/- 1)/(4*sqrt(2)) |
| QNG update | (G+lambda*I)d = grad_C; theta <- theta - eta*d |
| Replication runs | 15 = 5 split seeds x 3 init seeds per frozen config |
| Training samples | 60 (stratified 30/30) |
| Val/Test samples | 20 each (10/10) |

---

*This document is authoritative on code structure. `context.md` is authoritative on science, constants, and corrections. If they conflict, stop and ask.*
