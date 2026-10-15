# FLIGHT MANUAL — `qnnbench` (Quantum Neural Network Optimisation Benchmark)

Low-level architecture, build order, per-module test specs, comparison + visualisation spec.
Companion to `context.md`. **`context.md` is authoritative on science, constants and corrections; this manual is authoritative on code structure, interfaces, tests, and deliverables.** If they conflict, stop and ask; do not guess.

---

## 0. AGENT OPERATING RULES (read first, obey always)

1. Read `context.md` fully before writing any code. Do not re-derive its math.
2. Work **phase by phase** (Section 3). A phase is closed only when its **gate** passes. Never start a phase with a red gate.
3. **Tests first or alongside.** Every module ships with its tests in the same commit. A module without tests is not done. Test IDs (`T-Mxx-nn`) in this manual are the minimum; add more freely.
4. **One choke point for execution.** Every circuit execution goes through `Backend.execute()`, which emits ledger events. Nothing else may call a device/QNode (enforced by test T-M05-06 and a lint rule).
5. **Never silently**: reseed, drop failed runs, shrink a budget, change a split, fit on test/val, or alter architecture/readout/dataset between compared optimisers.
6. **Claims lint.** No wording of speedup, quantum advantage, or energy savings anywhere (code comments, captions, report templates). Enforced by T-M16-09.
7. DA1 numbers (96.7 %, 2.7× wall-clock, SPSA-beats-Adam crossover, etc.) are **historical/unverified**. They may appear only on the `historical` track, always labelled, never as DA2 results.
8. Anything not specified here or in `context.md` is a **Decision**. Record it in `docs/DECISIONS.md` (ID, choice, rationale, date) *before* using it. Section 12 lists pre-seeded decisions with proposed defaults.
9. Determinism: same config + same seeds ⇒ byte-identical `trace.csv`. Non-determinism is a bug.
10. Keep functions pure where possible; side-effects (I/O, ledger, RNG) only in the modules designated for them.

---

## 1. REPOSITORY LAYOUT

```
qnnbench/
├─ pyproject.toml            # deps, pytest/ruff/mypy config, extras: [report]
├─ Makefile                  # make env|test|test-fast|lint|tune|run-E1..E8|replicate|report|all
├─ context.md  FLIGHT_MANUAL.md
├─ docs/  DECISIONS.md  ARTEFACT_SCHEMAS.md  ENV_MANIFEST.json (generated)
├─ configs/
│  ├─ base.yaml              # fixed benchmark (Section 2.1)
│  ├─ optimisers/{gd,momentum,adam,spsa,qng}.yaml
│  ├─ experiments/E1.yaml ... E8.yaml
│  └─ frozen/                # written by tuning; hash-locked
├─ src/qnnbench/
│  ├─ core/        types.py  errors.py  config.py  seeds.py  hashing.py
│  ├─ data/        manager.py  preprocessing.py
│  ├─ quantum/     circuit_factory.py  backend.py  observables.py
│  ├─ grad/        reference.py  psr.py  sampled.py  loss.py  metric.py
│  ├─ budget/      ledger.py  manager.py  allocation.py
│  ├─ optim/       base.py  gd.py  momentum.py  adam.py  spsa.py  qng.py  registry.py
│  ├─ train/       trainer.py  evaluator.py  checkpoint.py  tuning.py
│  ├─ baselines/   logreg.py  mlp.py
│  ├─ io/          logger.py  schemas.py  artefacts.py
│  ├─ experiments/ e1_two_term.py ... e8_allocation.py  runner.py
│  ├─ analysis/    tidy.py  stats.py  compare.py  fits.py  tables.py
│  ├─ viz/         style.py  palette.py  primitives.py  figures/f01_*.py ...  gallery.py
│  ├─ report/      build.py  templates/  claims_lint.py
│  └─ cli.py
├─ tests/
│  ├─ unit/  integration/  numeric/  property/  viz/  regression/  conftest.py
│  └─ golden/                # small frozen reference arrays (npz/json)
├─ runs/<run_id>/   (gitignored) trace.csv events.jsonl final.json arrays.npz config.yaml env.json
└─ outputs/  figures/ tables/ report/   (gitignored, rebuilt by `make report`)
```

Package manager: `pip` + `pyproject.toml`; Python 3.11. Dev tools: `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy --strict` on `core/ budget/ optim/ grad/`, `pre-commit`. Optional extra `[report]`: `jinja2`. No other plotting library; Matplotlib only.

---

## 2. GLOBAL CONVENTIONS

### 2.1 Fixed benchmark (`configs/base.yaml`, loaded into frozen dataclasses; changing it voids all comparisons)
```yaml
task: {dataset: iris, classes: [versicolor, virginica], labels: {versicolor: -1, virginica: 1}, n_samples: 100}
circuit: {n_qubits: 4, encoding: angle_RY, layers: 2, rot: [RY, RZ], entangler: cnot_ring, readout: Z0}
model: {score: "<Z0> + b", loss: mse_pm1, param_shape: [2,4,2], n_gate_params: 16, n_total_params: 17}
init: {theta: uniform_0_2pi, b: 0.0, paired_across_optimisers: true}
splits: {main: {train: 0.6, val: 0.2, test: 0.2, stratified: true, expected_counts: [30,30,10,10,10,10]},
         historical: {train: 0.7, test: 0.3, seed: 7}}
replication: {split_seeds: [7,17,27,37,47], init_seeds: [0,1,2], runs_per_config: 15}
```

### 2.2 Core types (`core/types.py`) — immutable, typed, no hidden state
```python
@dataclass(frozen=True)
class Theta:           # model parameters
    angles: np.ndarray # (2,4,2) float64
    bias: float
    def flat(self) -> np.ndarray            # (17,): 16 angles then bias
    @staticmethod def from_flat(v) -> "Theta"

@dataclass(frozen=True)
class Batch:  X: np.ndarray; y: np.ndarray; idx: np.ndarray   # X already angle-scaled (m,4)

@dataclass(frozen=True)
class RunKey: experiment: str; optimiser: str; split_seed: int; init_seed: int; config_hash: str
```
`run_id = sha1(RunKey)[:12]` — one ID across CSV/JSON/NPZ.

### 2.3 Error taxonomy (`core/errors.py`)
`BudgetExhausted` (clean stop, not a failure) · `BudgetPreflightError` (op would exceed budget; raised *before* execution) · `NonFiniteError` (loss/grad/solve) · `LeakageError` · `ConfigFrozenError` · `AccountingMismatchError`. Failed runs write `final.json {status:"failed", error, last_update}` and appear in every aggregate table as failures. They are **counted, never dropped or reseeded**.

### 2.4 RNG policy (`core/seeds.py`)
One `np.random.SeedSequence([split_seed, init_seed, PURPOSE_ID])` per purpose; `PURPOSE_ID` enum: `init, minibatch, spsa_delta, shot_sampling, pilot_shots, bootstrap, tuning`.
- `init` and `minibatch` streams depend **only** on `(split_seed, init_seed)` → identical across optimisers (pairing).
- `shot_sampling` stream also keyed by optimiser name (so optimisers do not share shot noise artificially) but is deterministic.
- Device seeds are derived from the `shot_sampling` stream, never from global state.
- Banned: `np.random.seed`, `np.random.rand*` global calls (lint rule).

### 2.5 Config & hashing (`core/config.py`, `core/hashing.py`)
Frozen dataclasses from YAML; `config_hash = sha256(canonical_json(config))`. Frozen configs in `configs/frozen/` carry their hash; loading with a mismatched hash raises `ConfigFrozenError`.

### 2.6 Logging
`io/logger.py` is the only writer of artefacts. Python `logging` for human messages; artefacts for data. Schemas in `docs/ARTEFACT_SCHEMAS.md` and enforced in `io/schemas.py` (tests T-M13-*).

---

## 3. BUILD ORDER, DEPENDENCY GRAPH, GATES

```
M01 config/types ─┐
M02 seeds ────────┤
M13 io/logger ────┼─► M08 ledger/budget ─► M05 backend ─► M06 grad engines ─► M09 optimisers ─► M11 trainer/eval
M03 data ─────────┤                         ▲   M04 circuit ─┘     │                 ▲
                  │                         └─ M10 allocation       └─ M07 metric(QNG)┘
M12 baselines (needs M03 only)      M14 experiments (needs all)     M15 analysis ─► M16 viz ─► M17 report/CLI
```

| Phase | Modules | Gate (must be green) |
|---|---|---|
| P0 | scaffold, CI, Makefile, lint | `make lint test-fast` runs on empty skeleton |
| P1 | M01 M02 M13 M08 | all unit tests; ledger property tests |
| P2 | M03 M04 M05 | leakage test, circuit golden tests, ledger-reconciliation test |
| P3 | M06 M07 | **E1 and E2 pass** (PSR ≡ autodiff ≤ 1e-10; 4-term ≤ 1e-9; 2-term on CRot fails as documented) |
| P4 | M09 M11 M12 M15(partial) | optimiser unit tests; E3 smoke (5 optimisers × 1 seed) |
| P5 | finite-shot path, tuning | **E5 N·Var ≤ ½ + tolerance**; E4 smoke; budget-never-overshoot property test |
| P6 | M10, E6 E7 E8 | allocation invariants; paired-draw test; E8 matched-budget test |
| P7 | M15 M16 M17 | all figures render from fixture data; claims lint; schema tests |
| P8 | freeze + replicate | 15 runs/config completed; failures enumerated; report builds from `runs/` only |
| P9 | optional (noise, two-moons) | separate track; never mixed into baseline |

Each phase end: update `docs/DECISIONS.md`, tag git `phase-N`, run `make test` (full) and save the coverage report. Coverage floors: `core/ budget/ grad/ optim/` ≥ 95 %, others ≥ 85 %.

---

## 4. MODULE SPECIFICATIONS

Format per module: **Purpose · Files · Public API · Invariants · Edge cases · Tests**. Tests listed are mandatory minimums.

### M01 — Config & types (`core/`)
**API:** `load_config(path)->Config`, `freeze(config, path)`, `load_frozen(path)->Config`, `Config.validate()`.
**Invariants:** base benchmark fields immutable once frozen; `n_total_params==17` derived, not hardcoded twice.
**Tests**
- T-M01-01 YAML round-trip preserves values and hash.
- T-M01-02 mutating a frozen dataclass raises.
- T-M01-03 tampering a frozen file ⇒ `ConfigFrozenError`.
- T-M01-04 `Theta.flat/from_flat` round-trip (property test, random arrays).
- T-M01-05 config validation rejects: unknown optimiser, negative budget, split fractions ≠ 1, non-stratified main split.

### M02 — Seeds (`core/seeds.py`)
**API:** `stream(split_seed, init_seed, purpose, optimiser=None)->np.random.Generator`.
**Tests**
- T-M02-01 same args ⇒ identical draws; different purpose ⇒ different draws.
- T-M02-02 `init` stream identical across optimisers; `shot_sampling` differs across optimisers.
- T-M02-03 15 replication keys yield 15 distinct θ₀ (no collisions).
- T-M02-04 θ₀ ∈ [0, 2π)¹⁶, b₀ = 0.
- T-M02-05 lint test: grep source tree for banned global-RNG calls ⇒ none.

### M03 — Data manager & preprocessing (`data/`)
**Purpose:** load Iris, build the binary task, stratified splits, train-only scaling.
**API**
```python
make_task() -> (X_raw (100,4), y (100,) in {-1,+1})
make_split(X, y, split_seed, scheme="main"|"historical") -> Split(train, val, test: Batch-like with raw X)
fit_scaler(X_train) -> Scaler(a, d, const_mask)           # TRAIN ONLY
Scaler.transform(X) -> (X_tilde, clip_report)             # x̃ = π·clip((x−a)/(d−a),0,1); const feature → fixed angle (Decision D-03)
```
**Invariants:** `fit_scaler` receives only train rows (type-enforced: accepts `Split.train`, raises `LeakageError` otherwise); setosa excluded; labels ±1; clip counts logged per split; scaler stats saved in artefacts.
**Edge cases:** constant feature in train (d==a); val/test values outside train range (clipped, counted).
**Tests**
- T-M03-01 task has 100 samples, 50/50, 4 features, labels {−1,+1}, no setosa.
- T-M03-02 main split counts = 30/30 train, 10/10 val, 10/10 test; disjoint; union = all 100.
- T-M03-03 same seed ⇒ same split; 5 replication seeds ⇒ 5 distinct splits.
- T-M03-04 **leakage test**: perturb val/test features arbitrarily ⇒ scaler params unchanged; perturb train ⇒ changed.
- T-M03-05 x̃ ∈ [0, π] always; train min→0, max→π.
- T-M03-06 clip counts equal brute-force count of out-of-range val/test entries.
- T-M03-07 constant-feature handling per Decision D-03 (property test with synthetic data).
- T-M03-08 historical split (70/30, seed 7) reproducible and flagged `track="historical"`.

### M04 — Circuit factory (`quantum/circuit_factory.py`, `observables.py`)
**API**
```python
build_qnode(device, n_qubits=4, layers=2, observable="Z0", diff_method=...) -> QNode
build_test_circuit(kind="crot") -> QNode      # for E2 (controlled rotations)
n_gates_report() -> {"encoding_RY":4,"trainable":16,"cnot":8}   # uncompiled
draw(qnode, theta, x) -> matplotlib Figure                        # for viz F01
```
Structure: |0000⟩ → RY(x̃ᵢ) per wire → 2×[RY,RZ per wire; CNOT 0→1,1→2,2→3,3→0] → ⟨Z₀⟩. θ index convention `θ[layer, wire, {0:RY,1:RZ}]` documented once, reused everywhere.
**Observables:** `Z0` (local), `parity` (Z⊗…⊗Z, for E6), registry-based; never swapped silently.
**Tests**
- T-M04-01 gate counts match 4/16/8.
- T-M04-02 f ∈ [−1,1] for random θ, x (property test).
- T-M04-03 golden values: f at 5 fixed (θ, x) pairs vs `tests/golden/circuit_f.npz` (tolerance 1e-12).
- T-M04-04 θ = 0 does **not** give identity: assert output equals independent hand-computed state (CNOTs remain).
- T-M04-05 f is 2π-periodic in each θ.
- T-M04-06 independent NumPy statevector re-implementation of the circuit agrees with PennyLane within 1e-12 (second opinion).
- T-M04-07 width/depth parameterised builder produces expected param counts for E6 grid.
- T-M04-08 `Z0` vs `parity` give different values on random θ (guards against silent observable swap).

### M05 — Backend (`quantum/backend.py`) — the single execution choke point
**API**
```python
class Backend:
    def __init__(self, device_cfg, ledger: Ledger, rng): ...
    def execute(self, jobs: Sequence[CircuitJob], *, shots: int | None, category: Category, tag: str) -> np.ndarray
    # CircuitJob = (theta_angles, x_tilde, observable)
    # shots=None => analytic; shots=N => sampled; category in {training, monitoring, pilot, reference, search}
```
Modes: `default.qubit` analytic; `default.qubit` finite-shot; `default.mixed` (optional, noise track). Param-broadcasting allowed internally for speed, but **ledger counts each CircuitJob as one circuit eval** × `shots`.
**Invariants:** (a) every call emits exactly one ledger event; (b) `reference` category is never charged to any budget; (c) preflight check happens before any device call (Section M08).
**Tests**
- T-M05-01 analytic output equals M04 golden values.
- T-M05-02 sampled mean over many reps → analytic (z-test, 5σ).
- T-M05-03 same seed ⇒ identical sampled output; different optimiser stream ⇒ different.
- T-M05-04 batch of K jobs ⇒ ledger shows K circuits and K·N shots regardless of broadcasting.
- T-M05-05 preflight failure leaves device untouched (mock device call count = 0).
- T-M05-06 **ledger reconciliation**: monkeypatch raw device to count executions; after a full mini-run, raw count == ledger circuit count; any bypass fails. Also AST test: no module outside `quantum/backend.py` imports/calls `qml.QNode`/`qml.device` execution paths.
- T-M05-07 `reference` calls leave all budgets unchanged.
- T-M05-08 noise device smoke test (`default.mixed`) runs and is tagged `track="noise"`.

### M06 — Gradient engines (`grad/`)
**Files:** `reference.py` (autodiff, analytic, labelled), `psr.py` (2-term, 4-term, exact/analytic), `sampled.py` (finite-shot PSR with independent shots per shifted circuit), `loss.py` (MSE, unbiased finite-shot loss, chain rule).
**API**
```python
class GradEngine(Protocol):
    def f_and_df(self, theta: Theta, batch: Batch, plan: ShotPlan) -> FGrad   # f (m,), df/dθ (m,16)
    cost(self, m: int, plan) -> CostEstimate                                    # (2p+1)·m circuits

autodiff_reference(theta, batch)           # diff_method="backprop", category=reference, flagged
psr2(theta, batch, plan)                   # ∂f/∂θk = ½[f(θ+π/2 eₖ) − f(θ−π/2 eₖ)]
psr4(fn, theta, k, plan)                   # c±=(√2±1)/(4√2); shifts ±π/2, ±3π/2  (E2 only)
mse_loss(f, b, y)                          # (f+b−y)²
unbiased_mse_loss(f_hat, b, y, N)          # (f̂+b−y)² − (1−f̂²)/(N−1), N>1
mse_grad(f, df, b, y)                      # ∂C/∂θk = (2/m)Σ(f+b−y)·∂f/∂θk ; ∂C/∂b = (2/m)Σ(f+b−y)  (no quantum shifts for b)
```
**Invariants:** shift rule applied to circuit expectation f only; chain rule is classical. In finite-shot mode, residual `f̂` and each shifted `f̂±` come from **separate, independent** executions (so the product is unbiased). Cost per update = `(2p+1)m` where p=16, +monitoring/metric overhead logged separately.
**Tests**
- T-M06-01 (**E1**) psr2 vs autodiff: max |Δ| ≤ 1e-10 over 200 random (θ,x).
- T-M06-02 (**E2**) psr4 on controlled-rotation circuit ≤ 1e-9; psr2 on the same circuit shows error ≫ (> 1e-2), asserted.
- T-M06-03 finite-difference cross-check (central, h=1e-5) agrees ≤ 1e-6.
- T-M06-04 `mse_grad` vs autodiff of whole loss incl. bias ≤ 1e-10.
- T-M06-05 cost formula: p=16, m=70 ⇒ 2310; m=10 ⇒ 330 circuits; ledger agrees exactly after one update.
- T-M06-06 SPSA/PSR ratio = 16.5 (not 17) from engine cost functions.
- T-M06-07 finite-shot gradient unbiased: mean over 2000 reps within 4 SE of analytic.
- T-M06-08 Var(ĝₖ) ≤ 1/(2N)·(1+tol) (E5 unit version).
- T-M06-09 unbiased loss is unbiased (mean over reps → analytic loss, 4 SE); naive loss shows measurable positive bias ≈ (1−f²)/N.
- T-M06-10 `unbiased_mse_loss` raises for N ≤ 1.
- T-M06-11 non-finite input ⇒ `NonFiniteError`.
- T-M06-12 gradient identical when computed via broadcasting vs one-by-one loop.

### M07 — Metric provider / QNG geometry (`grad/metric.py`)
**API:** `metric_block_diag(theta, X_metric (≤8 inputs)) -> MetricResult{G (16×16 block-diag), n_tapes, cond, ...}`; `qng_direction(G, grad, damping, fallback_policy) -> (d, SolveReport)`.
**Invariants:** averaged over ≤ 8 training inputs (selected by the `minibatch` stream, logged); metric tapes charged to the ledger as actual executed tapes (no fixed per-iteration constant); bias excluded from G and updated classically (plain step with the same η); solve `(G+λI)d = ∇C`; `SolveReport{damping, cond, solver_ok, fallback_used}` logged **every** step.
**Fallback (Decision D-06):** on failure, λ←10λ up to 3 times, then plain-gradient step; every fallback is an event, never silent.
**Tests**
- T-M07-01 G symmetric PSD (property, tol 1e-10).
- T-M07-02 PennyLane `metric_tensor(block-diag)` ≡ independent NumPy Fubini–Study blocks ≤ 1e-9.
- T-M07-03 metric tapes in ledger == tapes actually executed.
- T-M07-04 λ→∞ ⇒ direction ∝ gradient/λ (GD limit); λ→0 on well-conditioned G ⇒ G⁻¹∇C.
- T-M07-05 singular G ⇒ fallback triggered, logged, run continues.
- T-M07-06 averaging over 1 vs 8 inputs gives different G; same inputs ⇒ same G (determinism).
- T-M07-07 cond number reported equals `np.linalg.cond(G+λI)`.

### M08 — Budget ledger & manager (`budget/ledger.py`, `budget/manager.py`)
**Ledger:** append-only event log. Event: `{seq, run_id, category, tag, circuits, shots_per_circuit, shots_total, update, wall_s}`.
Categories: `training`, `monitoring`, `pilot` (allocation pilots), `search` (hyper-param tuning), `reference` (free, analytic only).
**Manager API**
```python
BudgetManager(train_shots, monitoring_shots, search_shots=None)
preflight(category, circuits, shots_per_circuit) -> None | raises BudgetPreflightError
commit(event)                       # only after successful execution
remaining(category) -> int
can_afford_update(plan) -> bool     # full update cost incl. pilots and metric tapes
```
**Invariants:** preflight before every op; **never overshoot, never silently shrink** an operation to fit; run ends when the next *whole* update is unaffordable (status `completed_budget`). Event ledger is authoritative over formulas; `AccountingMismatchError` if formula and ledger disagree beyond zero in a non-adaptive plan.
**Tests**
- T-M08-01 property (hypothesis): any random sequence of ops ⇒ spent ≤ budget per category, always.
- T-M08-02 refused op leaves ledger unchanged (no partial commit).
- T-M08-03 categories never leak (monitoring cannot consume training budget and vice-versa).
- T-M08-04 `reference` events never reduce any remaining budget.
- T-M08-05 Arithmetic sanity: N=32, m=10 ⇒ 10,560 shots/PSR update ⇒ ≤ 9 updates in 1e5; SPSA 640 shots/update ⇒ ≤ 156 updates (computed from ledger, not hard-coded).
- T-M08-06 pilot shots counted in the matched-budget total for adaptive allocation.
- T-M08-07 ledger serialises/deserialises losslessly; `seq` strictly increasing.
- T-M08-08 final totals in `final.json` equal sum of `events.jsonl`.

### M09 — Optimisers (`optim/`)
**Interface (`optim/base.py`)**
```python
class Optimiser(ABC):
    name: str
    def propose(self, state, grad_engine, batch, plan, backend) -> StepResult   # does the quantum work via engines
    def apply(self, theta: Theta, step: StepResult) -> Theta
    def state_dict(self) / load_state_dict(self)
    def cost_per_update(self, m: int, plan) -> int                # circuits
```
Optimisers contain **no** circuit code, data code, or budget code — only update rules; they request evaluations through the engine.
| Optimiser | Rule / specifics |
|---|---|
| GD | θ←θ−η∇C; b updated with same η. η ∈ {0.03, 0.10, 0.30} |
| Momentum | v←μv+∇C, θ←θ−ηv, μ=0.9 (η grid: reuse GD grid, Decision D-07) |
| Adam | β₁=0.9, β₂=0.999, ε=1e-8, bias-corrected; η ∈ {0.01, 0.03, 0.10} |
| SPSA | Rademacher Δ ∈ {±1}¹⁶ (angles only; b gets classical gradient); aₜ=a/(A+t+1)^0.602, cₜ=c/(t+1)^0.101, c=0.15, a∈{0.1,0.5,1.5}; A and exponents recorded explicitly (D-08 for A default 10% of max updates); 2 loss evals/iter on the **same** minibatch; no library-derived schedule |
| QNG | PSR gradient + M07 direction; η grid as GD; λ ∈ {0.001, 0.01, 0.05} |

**Tests (common, parametrised over all five)**
- T-M09-01 loss decreases on a convex quadratic toy problem through the same `apply` path (sanity).
- T-M09-02 identical θ₀ and minibatch order across optimisers (pairing).
- T-M09-03 `state_dict` round-trip ⇒ resumed run identical to uninterrupted run (bitwise).
- T-M09-04 non-finite gradient ⇒ `NonFiniteError`, run marked failed, not reseeded.
- T-M09-05 declared `cost_per_update` equals ledger circuits for one update.
**Specific**
- T-M09-GD-01 one step equals hand-computed θ−η∇C.
- T-M09-MO-01 two-step trace equals hand-computed momentum recursion.
- T-M09-AD-01 first-step update magnitude ≈ η (bias-correction check); matches reference NumPy Adam over 10 steps ≤ 1e-12.
- T-M09-SP-01 schedule values aₜ, cₜ equal closed form for t=0..100; logged in trace.
- T-M09-SP-02 Δ is ±1 only, resampled each iteration, uses `spsa_delta` stream.
- T-M09-SP-03 SPSA gradient estimate unbiased on a smooth quadratic (mean over many Δ → true gradient within 3 SE).
- T-M09-SP-04 exactly 2 loss evaluations per update (ledger).
- T-M09-QN-01 with G=I, λ=0, QNG step ≡ GD step.
- T-M09-QN-02 bias updated by classical rule, not through G.
- T-M09-QN-03 ledger = PSR cost + actual metric tapes.

### M10 — Allocation policies (`budget/allocation.py`)
**API**
```python
class AllocationPolicy(Protocol):
    def plan(self, jobs_meta, remaining_update_shots, pilot_results|None) -> ShotPlan
UniformAllocation(N)                         # fixed shots {32,128,512} per circuit
VarianceInformedAllocation(pilot=16, floor=eps, min_prod=8, S_prod=...)
```
Variance-informed rule (**hypothesis, not established**): 16 pilot shots per shifted circuit → σⱼ = √max(1−f̂ⱼ², floor); weight aⱼ = ½·|rᵢ|/m (residual-weighted PSR coefficient); `Nⱼ = S_prod·|aⱼ|σⱼ / Σᵣ|aᵣ|σᵣ`; positive floor; `Nⱼ ≥ 8`; **largest-remainder** rounding to integers with ΣNⱼ == S_prod exactly; **fresh** production samples (pilots never reused in the estimator, but charged to the budget). Rationale (put in docstring): minimises Σ aⱼ²σⱼ²/Nⱼ s.t. ΣNⱼ=S_prod.
**Tests**
- T-M10-01 ΣNⱼ == S_prod exactly (property test, random weights).
- T-M10-02 every Nⱼ ≥ 8; zero-variance pilot (σ=0) still gets ≥ 8 (floor works, no div-by-zero).
- T-M10-03 Nⱼ monotone in |aⱼ|σⱼ.
- T-M10-04 largest-remainder: tie-breaking deterministic; integer output.
- T-M10-05 uniform case reduces: equal weights ⇒ equal Nⱼ (±1).
- T-M10-06 pilot shots appear in ledger as `pilot` and in matched-total accounting.
- T-M10-07 pilot samples excluded from estimator (mock: estimator input contains only production results).
- T-M10-08 analytic optimum: on synthetic variances, adaptive allocation's predicted variance ≤ uniform's.

### M11 — Trainer, evaluator, checkpointing (`train/`)
**Trainer loop (pseudocode contract)**
```
state = init(run_key); ckpt_points = predeclared
while not budget.exhausted_for_next_update:
    batch = minibatch_stream.next()               # paired across optimisers
    plan  = allocation.plan(...)
    step  = optimiser.propose(...)                # all quantum execution via Backend
    theta = optimiser.apply(theta, step)
    if update in ckpt_points: evaluate_val(theta, monitoring budget); keep best-by-val-MSE
    log trace row
final: restore best-val θ; evaluate TEST once (see D-09); write final.json
```
Analytic mode: 80 full-batch updates (m = train size = 60). Finite-shot: minibatch 10, budgets {1e5, 5e5, 1e6, 5e6} training shots, fixed shots {32,128,512}. Early stop (optional): predeclared patience (default 5 checkpoints).
**Evaluator:** MSE, accuracy (sign(s)), per split; val evaluated at checkpoints only; **test evaluated exactly once per run**, post-selection; `TestAccessGuard` raises on a second access or on access before finalisation.
**Tests**
- T-M11-01 `TestAccessGuard`: optimiser/allocation/selection code paths never touch test (spy on data manager).
- T-M11-02 test evaluated exactly once per run (second call raises).
- T-M11-03 checkpoint selects argmin val MSE (synthetic trace).
- T-M11-04 budget exhaustion ⇒ clean termination, `status=completed_budget`, no overshoot (property over budgets × shots × optimisers).
- T-M11-05 resume-from-checkpoint equals uninterrupted run.
- T-M11-06 early stop triggers at predeclared patience and logs reason.
- T-M11-07 analytic run: exactly 80 updates, full-batch.
- T-M11-08 minibatch order identical across optimisers for same keys.
- T-M11-09 failed run produces `final.json{status:"failed"}` and is included by `analysis.tidy`.
- T-M11-10 integration: tiny end-to-end (analytic, 3 updates, 2 optimisers) in < 10 s, all artefacts present and schema-valid.

### M11b — Tuning (`train/tuning.py`)
Equal **pilot budget per optimiser**; grids from M09; selection by **validation only**; search cost logged to `search` category and reported separately; output = `configs/frozen/<optimiser>.yaml` (hash-locked). Replication refuses to start on unfrozen configs.
**Tests:** T-M11b-01 every optimiser receives identical pilot budget; T-M11b-02 test set never touched; T-M11b-03 frozen file hash verified at replication start; T-M11b-04 search shots not mixed into run totals.

### M12 — Classical baselines (`baselines/`)
Logistic regression + 1-hidden-layer MLP (scikit-learn), same splits, same train-only scaling (use the **same** x̃ features and also a standardised variant, labelled), no artificial shot counts, hyper-params tuned on val only, test evaluated once.
**Tests:** T-M12-01 same split indices as quantum runs; T-M12-02 no test access during fit; T-M12-03 deterministic given seed; T-M12-04 outputs in same artefact schema (`optimiser="logreg"|"mlp"`, shots = null).

### M13 — IO, logger, schemas (`io/`)
**Artefacts per run** (`runs/<run_id>/`): `trace.csv` (per-update/checkpoint rows), `events.jsonl` (ledger + fallbacks + failures), `final.json`, `arrays.npz` (θ trajectory, gradients, SPSA Δ, metric condition numbers), `config.yaml`, `env.json` (library versions from **actual** execution machine).
`trace.csv` columns: `run_id, update, circuits_train, shots_train, shots_monitor, shots_pilot, wall_s, train_mse, val_mse, val_acc, grad_norm, step_norm, lr_eff, spsa_a_t, spsa_c_t, qng_damping, qng_cond, qng_fallback, clipped_val, clipped_test, track`.
**Tests**
- T-M13-01 schema validation for every artefact type (missing/extra column fails).
- T-M13-02 one `run_id` across all files of a run.
- T-M13-03 atomic writes (kill-simulated partial write never leaves a corrupt final file).
- T-M13-04 `env.json` contains versions read from the live interpreter (not hard-coded DA1 table).
- T-M13-05 CSV/NPZ round-trip lossless (float64).
- T-M13-06 runs are append-safe under parallel workers (file-lock or per-run dirs).

### M14 — Experiments (`experiments/`)
Each experiment: a pure function `run(config)->list[RunKey]` + a declarative YAML. All write only through M13.
| ID | Question | Core design | Primary outputs |
|---|---|---|---|
| E1 | RQ1 2-term PSR correct? | 200 random (θ,x); psr2 vs autodiff | max/mean error, histogram |
| E2 | RQ1 4-term on controlled rotations | CRot test circuit; psr2 vs psr4 vs autodiff | error table, log-error plot |
| E3 | RQ2 five optimisers, analytic | 80 full-batch updates; frozen configs; 15 paired runs | loss/acc vs updates, final-metric tables |
| E4 | RQ2 equal-shot ranking | budgets {1e5,5e5,1e6,5e6} × shots {32,128,512} × 5 optimisers × 15 runs | val-MSE vs shots, rank tables, crossovers |
| E5 | shot-noise law | N ∈ {8…4096} geometric, R≥2000 reps; bias/var/RMSE of f̂, ĝ, ℓ̂ (naive vs unbiased); N·Var vs analytic and ½ bound | log-log slope −1 plot, N·Var table |
| E6 | RQ4 trainability | width n∈{2,4,6,8,10,12}, depth, init, observable local-vs-parity; **same parameter vectors reused across observables (paired draws)**; fits with uncertainty; descriptive only, no extrapolation | variance vs n, fitted exponents ± CI |
| E7 | RQ3 QNG | damping {0.001,0.01,0.05} × metric inputs {1,4,8} × block-diag vs GD; equal-update and equal-shot | cond(G) traces, overhead vs benefit |
| E8 | RQ5 adaptive vs uniform | static gradient RMSE at fixed θ **and** training outcomes; matched total shots incl. pilots | RMSE-vs-shots, allocation histograms, win-rate |
| E-hist | historical track | DA1 70/30 seed-7 split, DA1 checkpoints 20k–400k circuits | separate figures, labelled `historical`, never merged with E4 |
| opt | noise (`default.mixed`), two-moons | separate track | separate figures |
**Tests (per experiment)**
- T-M14-E*-01 smoke run at tiny scale (≤ 30 s) produces schema-valid artefacts.
- T-M14-E*-02 re-running with the same config reproduces identical results (hash of `trace.csv`).
- T-M14-E3-03 all 5 optimisers share θ₀ per (split, init) key.
- T-M14-E4-03 total training shots per run ≤ budget; monitoring ≤ monitoring allowance.
- T-M14-E5-03 empirical Var(f̂)·N ≈ 1−f² within 5 % at N ≥ 512.
- T-M14-E6-03 parameter vectors identical across local/global observable runs (array equality).
- T-M14-E6-04 exponent fits return CI; fit function refuses extrapolation request beyond measured n.
- T-M14-E7-03 QNG ledger overhead = actual metric tapes.
- T-M14-E8-03 matched-budget: adaptive total (production+pilot) == uniform total, exactly.
- T-M14-E-hist-01 outputs carry `track="historical"`; the tidy-merge for main analyses excludes them.

### M15 — Analysis & statistics (`analysis/`)
**`tidy.py`:** loads `runs/` into three tidy DataFrames: `runs` (one row per run), `trace` (per checkpoint), `events` (ledger). Includes failures (`status` column).
**`compare.py` — the comparison engine** (used by every table and figure):
- *Paired differences* on identical (split_seed, init_seed) blocks.
- *Effect size & uncertainty:* mean/median difference, **cluster bootstrap by split seed** (10k resamples, BCa) plus plain paired bootstrap; both reported.
- *Tests:* Friedman across the 5 optimisers per (budget, shots); post-hoc Wilcoxon signed-rank with **Holm** correction; rank-biserial effect size.
- *Ranking:* mean rank ± CI per cell; win/tie/loss matrix; critical-difference-style ordering.
- *Performance profiles* (Dolan–Moré style on final val MSE and on shots-to-target-MSE).
- *Cost-to-target:* shots / circuits / updates to reach val MSE ≤ τ (predeclared τ list), right-censored when never reached (reported, not dropped).
- *Pareto front:* (shots, val MSE) per optimiser.
- *Equal-update vs equal-shot divergence:* rank reversal detection between E3 and E4 (RQ2 headline).
- *Quantum vs classical:* test accuracy/MSE vs logreg/MLP reference bands (no advantage framing; wording enforced by claims lint).
**`fits.py`:** log-log OLS with CIs for exponents (E6) and slope-vs-(−1) check (E5); finite-width, descriptive.
**`tables.py`:** booktabs LaTeX + CSV + Markdown tables T01–T10 (Section 6).
**Tests**
- T-M15-01 paired differences computed only on matching keys (mismatched key set raises).
- T-M15-02 bootstrap CI covers known mean of synthetic data ≈ 95 % (coverage simulation, n=500, tol ±3 %).
- T-M15-03 Holm correction matches reference implementation on fixed p-values.
- T-M15-04 Wilcoxon/Friedman agree with SciPy on fixed fixtures.
- T-M15-05 censored cost-to-target keeps censored rows and flags them.
- T-M15-06 failed runs counted in failure table and excluded from means **only with explicit flag** printed in the table footer.
- T-M15-07 rank computation handles ties (average ranks).
- T-M15-08 historical-track rows never appear in main comparison outputs.
- T-M15-09 log-log fit recovers known synthetic exponent within CI.
- T-M15-10 deterministic with fixed bootstrap seed.

### M16 — Visualisation (`viz/`) — see Section 5 for the full spec
### M17 — CLI, orchestration, report (`cli.py`, `report/`)
`qnnbench run <E#> [--tiny]`, `tune`, `freeze`, `replicate`, `analyse`, `figures`, `report`, `doctor` (env check + manifest). Parallelism by process pool over `RunKey` with per-run output dirs; resumable (skips complete runs, re-runs partial).
**Tests:** T-M17-01 `doctor` writes manifest; T-M17-02 resume skips completed runs; T-M17-03 `report` builds from fixture runs without network; T-M17-04 `--tiny` full pipeline (E1→E8→report) < 5 min in CI; T-M17-05 report lists all failures and the historical/main separation.

---

## 5. VISUALISATION SPEC (publication-grade)

### 5.1 Design system (`viz/style.py`, `palette.py`) — one place, no per-figure styling
- Matplotlib `rcParams` style sheet: serif-compatible sans (`DejaVu Sans` fallback chain), 9 pt base, vector-first.
- **Colour-blind-safe Okabe–Ito palette.** Fixed mapping, defined once and imported everywhere:
  GD = blue, Momentum = sky, Adam = orange, SPSA = vermilion, QNG = green, LogReg = grey (dashed), MLP = dark grey (dotted), Adaptive = purple, Uniform = black. Also fixed **marker + linestyle per optimiser** so figures survive greyscale printing.
- Sizes: `single` (3.4 in), `double` (7.0 in), `full-page`; consistent margins via `constrained_layout`.
- Output: **SVG/PDF (vector) + PNG @ 300 dpi**; `svg.hashsalt` fixed and metadata stripped ⇒ deterministic bytes.
- **Axis rule:** every x-axis states its cost unit explicitly: *updates*, *circuit evaluations*, or *shots*. Never ambiguous. Log scales for shots/circuits. Budget checkpoints drawn as faint vertical guides.
- **Uncertainty always shown:** median/mean line + 95 % CI band (cluster-bootstrap); individual runs as thin low-alpha traces behind (spaghetti layer, toggleable).
- Annotate every figure with an unobtrusive footer: `track | config_hash[:8] | n_runs | n_failed`.
- Panel labels (a), (b)…; units in axis labels; legends ordered by final performance, not alphabetically.

### 5.2 Primitives (`viz/primitives.py`) — reusable, individually tested
`band_line(ax, df, x, y, group)`, `spaghetti(...)`, `rank_heatmap(...)`, `forest(ax, diffs)`, `perf_profile(...)`, `pareto(...)`, `loglog_with_slope(...)`, `annotate_crossover(...)`, `stacked_ledger_bar(...)`, `cd_diagram(...)`, `footer(fig, meta)`.

### 5.3 Figure catalogue (`viz/figures/fNN_*.py`) — each = `def make(data: TidyBundle, out: Path) -> FigureArtifact`
| ID | Figure | Content |
|---|---|---|
| F01 | Architecture & circuit | `draw_mpl` circuit + parameter-index map + gate counts (4/16/8) |
| F02 | Data & splits | 2×2 feature-pair scatter, train/val/test coloured, scaler range, clipped points marked |
| F03 | Gradient correctness (E1/E2) | (a) PSR vs autodiff parity scatter (y=x), (b) error histogram log-x, (c) CRot: 2-term vs 4-term error |
| F04 | Shot-noise law (E5) | log-log Var vs N with slope −1 reference and ½ bound; N·Var panel; bias of naive vs unbiased loss |
| F05 | Analytic optimiser comparison (E3) | loss and accuracy vs **updates**, bands over 15 runs |
| F06 | **Flagship: equal-shot comparison (E4)** | val MSE vs **shots** (log-x), per optimiser, CI bands, 3 shot-per-circuit panels, budget guides, crossover markers |
| F07 | Equal-update vs equal-shot | side-by-side ranking panels showing reversal (RQ2 headline) |
| F08 | Rank heatmap | optimisers × (budget, shots) cells, mean rank, significance stars (Holm) |
| F09 | Pairwise forest plot | paired differences with CIs for key contrasts |
| F10 | Performance profile | fraction of runs reaching target within shots τ |
| F11 | Cost-to-target | shots to reach val MSE ≤ τ, censored runs drawn as arrows |
| F12 | Pareto front | (shots, val MSE) with dominated region shaded |
| F13 | Budget anatomy | stacked bars of training / monitoring / pilot / search shots per optimiser |
| F14 | Updates-per-budget | updates achieved per optimiser per budget (explains F06 mechanically) |
| F15 | SPSA diagnostics | aₜ, cₜ schedules, gradient-estimate cosine similarity to exact gradient |
| F16 | QNG diagnostics (E7) | cond(G) trace, damping sensitivity, fallback events, overhead-vs-benefit |
| F17 | Trainability scan (E6) | variance vs width (local vs global), fitted exponents ± CI (labelled finite-width), depth/init panels |
| F18 | Adaptive vs uniform (E8) | gradient RMSE vs total shots, allocation histograms (Nⱼ distribution), win-rate bars |
| F19 | Loss landscape slices | 2-D slices through θ₀ → θ_final along two random orthonormal directions (same directions for all optimisers), trajectories overlaid |
| F20 | Optimiser trajectories | PCA of θ trajectories (shared basis) for one split |
| F21 | Decision boundary | in 2-D PCA of x̃, best model per optimiser vs classical baselines |
| F22 | Quantum vs classical | final test acc/MSE with classical reference bands (no advantage wording) |
| F23 | Robustness | per-(split, init) paired dot-plot; failure counts |
| F24 | Historical track | DA1-protocol curves, watermark "HISTORICAL — UNVERIFIED PROTOCOL" |

### 5.4 Gallery & dashboard
`viz/gallery.py` builds `outputs/figures/index.html` (static, no network, embedded SVG) with thumbnails, captions, track, hash, and links to the underlying tidy CSV for each figure. Optional `[report]` extra: Jinja2 template for a one-file HTML results report that reproduces all tables and figures.

### 5.5 Visualisation tests (`tests/viz/`, marker `viz`, Agg backend)
- T-M16-01 every `fNN.make` renders from **fixture data** (`tests/golden/fixture_runs/`) without error.
- T-M16-02 data-contract: each figure declares required tidy columns; missing column ⇒ explicit `SchemaError` (not a KeyError deep in Matplotlib).
- T-M16-03 palette/marker/linestyle mapping is unique per optimiser and identical across all figures (introspect Line2D properties).
- T-M16-04 every x-axis label contains one of {`updates`, `circuit evaluations`, `shots`} (or explicit non-cost unit for E5/E6/F02/F19–F21 whitelisted).
- T-M16-05 legend entries == set of optimisers present in data; ordered by final performance.
- T-M16-06 non-blank render (pixel variance > 0) and size within declared `single/double` width.
- T-M16-07 determinism: two renders of same data ⇒ identical SVG bytes.
- T-M16-08 uncertainty layer present (a `PolyCollection` band) in F05, F06, F08-style plots; spaghetti toggle works.
- T-M16-09 **claims lint**: scan all captions, titles, annotations, report templates for banned terms (`speedup`, `advantage`, `outperforms classical`, `energy`, `supremacy`, `faster than`) ⇒ fail unless in the explicit disclaimer allowlist.
- T-M16-10 footer annotation (track, hash, n_runs, n_failed) present on all main figures; F24 carries the historical watermark; historical data never plotted on main-track figures.
- T-M16-11 failure runs visibly counted in F23 (n_failed shown, not hidden).
- T-M16-12 optional baseline-image regression (`pytest-mpl`) for F03, F06, F08 on fixtures, tolerance documented.

---

## 6. TABLE CATALOGUE (`analysis/tables.py`; LaTeX booktabs + CSV + Markdown)
T01 benchmark spec & gate counts · T02 splits & clip counts · T03 gradient-verification errors (E1/E2) · T04 shot-noise statistics (E5) · T05 tuned hyper-parameters & search cost (frozen configs) · T06 analytic results mean±CI (E3) · T07 equal-shot results per (budget, shots) with ranks and Holm-adjusted p (E4) · T08 cost-to-target (censoring flagged) · T09 QNG overhead vs benefit (E7) · T10 adaptive vs uniform (E8) · T11 failures & fallbacks (every run, every reason) · T12 environment manifest.
Table tests: T-M15-T01 each table builds from fixtures; T-M15-T02 numeric cells match `compare.py` outputs; T-M15-T03 captions pass claims lint; T-M15-T04 failures table lists every failed `run_id`.

---

## 7. CROSS-CUTTING TEST STRATEGY

- **Markers:** `unit`, `integration`, `numeric`, `property`, `viz`, `slow`, `regression`. `make test-fast` = all but `slow`; CI runs `--tiny` pipeline.
- **Property tests (hypothesis):** ledger never overshoots; allocation sums exactly; Theta round-trip; metric PSD; scaler train-only.
- **Golden files:** tiny, versioned, regenerated only through a script that prints a diff and requires `--accept`.
- **Second-opinion tests:** independent NumPy statevector circuit; independent NumPy Adam; independent Fubini–Study blocks. If PennyLane and the second opinion disagree, the build is red.
- **Statistical tests are seeded and use wide tolerances** (≥ 4 SE) to avoid flakiness; each states its tolerance rationale in the docstring.
- **Mutation sanity:** `tests/regression/test_guards.py` deliberately injects each forbidden behaviour (fit scaler on all data, bypass backend, overshoot budget, access test early, reseed on failure, drop failed run) and asserts the corresponding guard fires.
- **Regression of corrections:** one named test per DA1→DA2 correction (#1–#6 in `context.md`): `test_correction_1_no_leakage`, `…_2_backprop_is_labelled_reference`, `…_3_paired_variance_draws`, `…_4_cost_formula`, `…_5_no_extrapolation`, `…_6_historical_track_isolated`.
- **Performance budget:** unit suite < 90 s; tiny pipeline < 5 min; any test > 10 s needs the `slow` marker.

---

## 8. OBSERVABILITY & REPRODUCIBILITY CHECKLIST
- `env.json` + `docs/ENV_MANIFEST.json` from the machine that actually ran the experiments (DA1 version table is unverified).
- `git` commit hash embedded in every run's `env.json`; refuse replication on a dirty tree unless `--allow-dirty` (logged).
- Every figure/table traceable: figure → tidy CSV → run IDs → config hash.
- `make all` from a clean clone reproduces everything from configs (tiny profile by default; `PROFILE=full` for the real thing).

---

## 9. HARD DO-NOTS → WHERE ENFORCED
| Rule (from `context.md`) | Enforcement |
|---|---|
| No speedup/advantage/energy claims | T-M16-09 claims lint, template scan |
| Uniform angles ≠ Haar; zero angles ≠ identity; Z₀ ≠ parity | T-M04-04, T-M04-08, docstrings |
| Separate ablations for amplitude encoding, re-uploading, cross-entropy, mitigation | separate `track` values; tidy merge excludes them from main analyses |
| Failed runs logged, never dropped/reseeded | T-M11-09, T-M15-06, T-M-guards |
| Fixed architecture/dataset/readout/accounting across optimisers | frozen base config hash embedded in every run; analysis refuses mixed hashes |
| Scaler/learned transforms train-only | T-M03-04, `LeakageError` |
| Backprop = labelled reference only | T-M05-07, category `reference`, T-M16 footer |
| Cost formula (2p+1)m; ratio 16.5 | T-M06-05/06, T-M08-05 |
| Historical checkpoints isolated | T-M14-E-hist-01, T-M15-08, F24 watermark |

---

## 10. DEFINITION OF DONE (project)
1. All phase gates green; coverage floors met; `make test` and tiny pipeline pass in CI.
2. 15 paired runs × every frozen config complete; failures enumerated in T11.
3. F01–F24 and T01–T12 regenerate from `runs/` alone; gallery + report build offline.
4. RQ1–RQ6 each have: a named experiment, a headline figure, a headline table, a stats statement (effect size + CI + Holm p), and a one-paragraph plain-language finding that passes claims lint.
5. `docs/DECISIONS.md` complete; every Section-12 decision resolved or confirmed.
6. README with the 5-command reproduction path and a limitations section (Iris scale, single-qubit readout, simulator-only, finite-width fits).

---

## 11. LIMITATIONS TO SURFACE IN THE REPORT (auto-included by the report builder)
Simulator only · 100-sample task · small width ⇒ exponents are finite-width descriptive · tuning grids coarse · 15 runs (5 split blocks) limit statistical power · adaptive allocation is a hypothesis under test · historical results unverified.

---

## 12. PRE-SEEDED DECISIONS (record in `docs/DECISIONS.md`; defaults are proposals — confirm before first full run)
| ID | Decision | Proposed default |
|---|---|---|
| D-01 | Monitoring allowance | 20 % of training shots, separate ledger; validation evaluated sampled at `N_mon=512` |
| D-02 | Checkpoint schedule | every 10 % of the budget (finite-shot) / every 5 updates (analytic) |
| D-03 | Constant-feature angle | π/2 |
| D-04 | Final test evaluation | analytic on the selected θ (reporting only, uncharged, labelled) **and** a sampled variant at 512 shots (labelled) |
| D-05 | Minibatch with shots | m=10 stratified-ish draw from the `minibatch` stream, shared across optimisers |
| D-06 | QNG solve fallback | λ×10 up to 3 times, then plain gradient step, always logged |
| D-07 | Momentum η grid | same as GD |
| D-08 | SPSA stability constant A | 10 % of expected max updates, recorded in config |
| D-09 | Early stopping | off for main results; patience=5 checkpoints as an ablation |
| D-10 | Failed-run policy in aggregates | reported in T11; main means computed on completed runs with an explicit footer, plus a sensitivity table treating failures as worst-case |
| D-11 | Tidy targets τ for cost-to-target | predeclared list, set after tuning, before replication |
| D-12 | E6 sample count | 200 paired parameter vectors per (n, depth, init) |
| D-13 | Parallelism | process pool, per-run directories, no shared mutable files |

---

## 13. KICK-OFF PROMPT FOR THE AGENT
> Read `context.md` and `FLIGHT_MANUAL.md` in full. Work strictly in phase order (Section 3). Start with P0: scaffold the repo per Section 1, set up pytest/ruff/mypy/Makefile, and open `docs/DECISIONS.md` with the Section-12 table. For every module, write the tests listed for it (plus any extra edge cases you identify) before or with the implementation, run them, and report: files created, tests added (IDs), pass/fail, coverage for that module, and any decision you had to make. Do not proceed to the next phase until the gate is green. Never violate Section 0 or Section 9. If anything is ambiguous or conflicts with `context.md`, stop and ask me rather than guess. Begin P0 now.