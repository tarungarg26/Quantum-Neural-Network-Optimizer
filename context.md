# context.md

```markdown
# CONTEXT — Quantum Neural Network Optimisation (compressed handoff; supersedes reading full docs)

## Project
VIT Chennai, Software Engineering, 2026-27. Team: Tarun Garg (24BCE5084), Nigam Anand Nayak (24BCE5202), Shivam Kumar (24BCE1001); guide Dr. Ilakiyaselvan N.
Goal: reproducible hybrid quantum–classical benchmark comparing 5 optimisers on one small variational classifier, with explicit shot/circuit-eval accounting. **No quantum-advantage claims.**
Doc lineage: Proposal (broad vision) → DA1 (survey + math + code + preliminary results) → DA2 (current: full methodology). **DA2 supersedes DA1 on any conflict.**

## Fixed benchmark (identical across all compared methods — never vary silently)
- Task: binary Iris, versicolor(−1) vs virginica(+1); 100 samples (50/50), 4 features; setosa excluded.
- Circuit: 4 qubits |0000⟩ → angle encoding RY(x̃ᵢ) per feature → 2 layers of [RY, RZ per wire + CNOT ring 0→1,1→2,2→3,3→0] → readout ⟨Z₀⟩.
- Model: score s = ⟨Z₀⟩ + b; predict sign(s); loss = MSE vs ±1 labels; f ∈ [−1,1].
- Params: θ shape (2,4,2) = 16 angles + 1 classical bias b (17 total). Init θ ~ U[0,2π), b=0, **paired across optimisers**.
- Uncompiled gates: 4 encoding RY + 16 trainable + 8 CNOT.

## Core math (derived & verified; don't re-derive)
- 2-term parameter-shift (Pauli-generated gates): ∂f/∂θₖ = ½[f(θ+π/2eₖ) − f(θ−π/2eₖ)]
- 4-term rule (controlled rotations, generator gaps {½,1}): f′ = c₊[f(θ+π/2)−f(θ−π/2)] − c₋[f(θ+3π/2)−f(θ−3π/2)], c±=(√2±1)/(4√2)
- Shift rule applies to circuit expectation f only → classical chain rule gives MSE gradient; bias gradient needs no quantum shifts.
- Shot noise: Var(f̂)=(1−f²)/N; Var(ĝₖ)≤1/(2N); ε-accuracy per shifted circuit ⇒ N≥1/(2ε²) (ε=0.01 → 5,000 shots).
- QNG: Fubini–Study metric, block-diagonal, averaged over ≤8 training inputs; solve (G+λI)d=∇C; θ←θ−ηd; bias updated classically; report damping, condition number, failed-solve fallback.
- SPSA: 2 loss evals/iter, Rademacher Δ; aₜ=a/(A+t+1)^0.602, cₜ=c/(t+1)^0.101; record a,c,A,exponents explicitly (no library-derived schedule changes).
- Unbiased finite-shot loss: ℓ̂=(f̂+b−y)² − (1−f̂²)/(N−1), N>1.

## Cost model (event ledger is authoritative over formulas)
- PSR update: (2p+1)m circuit evals; p=16, m=70 → 2,310 (+ monitoring/metric overhead). Uniform N shots ⇒ ×N.
- SPSA update: 2m evals. PSR/SPSA circuit ratio = 16.5 (**not 17** — bias is not a gate parameter).
- QNG: PSR cost + actual metric tapes (no fixed per-iter constant).
- Ledger separates: training / monitoring-validation / pilot shots. Preflight every op; refuse if it exceeds remaining budget; never overshoot or silently shrink.

## Optimisers (5) & tuning
GD (η∈{0.03,0.10,0.30}) · momentum (μ=0.9) · Adam (β₁=0.9, β₂=0.999, ε=1e−8; η∈{0.01,0.03,0.10}) · SPSA (pilot a∈{0.1,0.5,1.5}, c=0.15) · QNG (η grid as GD; damping λ sensitivity {0.001,0.01,0.05}). Equal pilot budget per optimiser; selection via validation only; search costs reported separately.

## DA1 preliminary results — carry forward as UNVERIFIED (single split, flawed protocol; never present as new DA2 results)
- PSR ≡ autodiff to 1.4e−17; 4-term exact to 3.4e−11; 2-term on CRot errs 1.7e−1.
- 96.7% test accuracy — **but scaler was fit pre-split (leakage)**.
- Gradient-variance fits: global 2^−0.98n vs local 2^−0.64n (small-n; refit with uncertainty in DA2).
- N·Var ≈ 0.42–0.52 (analytic bound ½).
- SPSA beat Adam below ~2×10⁵ circuit execs; ordering reverses above.
- QNG MSE 0.4318 @ epoch 20 vs Adam 0.5071, GD 0.6747 (~2.7× wall-clock).

## DA1→DA2 corrections (must honour)
1. Fit all learned transforms on train split only (fixes leakage).
2. Backprop runs = labelled analytic reference only; hardware-cost experiments use explicit PSR/sampled executions.
3. Local/global variance scan: reuse exact same parameter vectors (paired draws).
4. Cost formula (2p+1)m, not 2(p+1)M.
5. Plateau exponents = finite-width descriptive fits; no large-n extrapolation.
6. DA1 circuit-budget checkpoints (20k–400k) = historical track only; not interchangeable with finite-shot budget grid.

## DA2 protocol
- Splits: main 60/20/20 stratified (intended 30/30 train, 10/10 val, 10/10 test); historical 70/30 seed 7 (hyperparams fixed, not test-tuned). Replication: split seeds {7,17,27,37,47} × 3 init seeds = 15 paired runs per frozen config.
- Preprocessing: x̃ᵢ = π·clip[(xᵢ−aᵢ)/(dᵢ−aᵢ), 0, 1], a,d = train-split min/max; apply unchanged to val/test; log clipped counts; constant feature → fixed angle.
- Analytic training: 80 full-batch updates. Finite-shot: minibatch 10; fixed shots {32,128,512}; training budgets {1e5, 5e5, 1e6, 5e6} shots; separate monitoring allowance.
- Checkpointing: best val MSE at predeclared points; test set untouched by optimiser, allocation policy, and model selection. Optional early stop: predeclared patience (e.g., 5 checkpoints).
- Classical baselines: logistic regression + 1-hidden-layer MLP, same splits, no artificial shot counts.
- Modules: data manager / circuit factory / backend / estimator / metric provider / optimiser adapter / budget manager / evaluator / logger (one run ID across CSV/JSON/NPZ artefacts).
- Experiments: E1 two-term verify · E2 four-term verify · E3 five-optimiser analytic · E4 finite-shot budget compare · E5 shot-noise law (bias/var/RMSE, N·Var) · E6 trainability scan (width/depth/init/observable, paired draws) · E7 QNG geometry ablation · E8 adaptive-vs-uniform allocation · optional: noise (default.mixed), two-moons.
- RQs: RQ1 gradient correctness · RQ2 optimiser ranking at equal cost (equal-update ≠ equal-shot) · RQ3 QNG overhead vs benefit · RQ4 width/depth trainability · RQ5 adaptive allocation value · RQ6 robustness across paired seeds.

## Proposed extension (hypothesis to test, NOT established)
Variance-informed shot allocation: 16 pilot shots per shifted circuit → estimate σⱼ → allocate Nⱼ = S_prod·|aⱼ|σⱼ / Σᵣ|aᵣ|σᵣ; positive variance floor; ≥8 production shots each; largest-remainder integer rounding; fresh production samples (pilots charged to budget, not reused in estimator). Compare vs uniform at matched total shots incl. pilots.

## Stack
Python; PennyLane (default.qubit analytic + finite-shot; default.mixed optional noise); pennylane.numpy (requires_grad separates θ from x); scikit-learn (Iris, stratified splits, baselines); NumPy; Matplotlib (label x-axes: updates vs circuit evals vs shots); CPU sufficient; env manifest from actual execution machine (DA1 version table unverified: pennylane 0.45.1, numpy 2.4.4, sklearn 1.8.0, Python 3.11).

## Hard do-nots
- No speedup/advantage/energy claims; Iris is too small for that.
- Uniform angles ≠ Haar; zero angles ≠ identity block (CNOTs remain); single-qubit readout ≠ same minimiser as parity observable.
- Amplitude encoding, data re-uploading, cross-entropy, error mitigation = separate labelled ablations only; never mixed into baseline.
- Failed runs (nonfinite loss/grad/solve): log, never silently drop or reseed.
- Architecture, dataset, readout, and accounting conventions stay fixed across optimiser comparisons.
```

**Usage:** drop this file in the working directory and instruct the agent: *"Read `context.md` first; it is authoritative. Only consult the full DA1/DA2 docs if you need verbatim text (e.g., exact equations, references, or report wording)."* This compresses ~15k+ tokens of source material to ~1.4k tokens while preserving every decision, constant, correction, and constraint the agent needs.