# Architecture and Design Decisions (qnnbench)

This file records all architectural, statistical, and operational decisions in accordance with Rule 8 of the Flight Manual.

| ID | Choice | Rationale | Date | Status |
|---|---|---|---|---|
| D-01 | Monitoring allowance = 20% of training shots | Prevents monitoring from overwhelming small training budgets while maintaining equal validation fidelity across optimizers. Validations sampled at $N_{mon}=512$. | 2026-10-12 | Confirmed |
| D-02 | Checkpoint schedule: every 10% of budget (finite-shot) / every 5 updates (analytic) | Guarantees consistent validation frequency across distinct optimizer convergence speeds without excessive overhead. | 2026-10-12 | Confirmed |
| D-03 | Constant-feature angle = $\pi / 2$ | In the degenerate event that feature spread $d - a = 0$, mapped angle is fixed to $\pi/2$ (maximum superposition / unbiased sensitivity). | 2026-10-12 | Confirmed |
| D-04 | Final test evaluation: analytic on selected $\theta$ (reporting only, uncharged, labelled) and sampled variant at 512 shots | Isolates optimization performance from test-evaluation sampling noise while also benchmarking realistic finite-shot deployment. | 2026-10-12 | Confirmed |
| D-05 | Minibatch with shots: $m=10$ stratified draw from `minibatch` stream | Guarantees identical sample sequences across paired optimizer runs while matching standard batch size constraints. | 2026-10-12 | Confirmed |
| D-06 | QNG solve fallback: $\lambda \leftarrow 10\lambda$ up to 3 times, then GD fallback step | Prevents numeric divergence on rank-deficient or singular Fubini-Study metric tensors while explicitly logging every fallback event. | 2026-10-12 | Confirmed |
| D-07 | Momentum $\eta$ grid: $\{0.03, 0.10, 0.30\}$ (reusing GD grid) | Minimizes tuning degrees of freedom and establishes direct comparability with vanilla gradient descent. | 2026-10-12 | Confirmed |
| D-08 | SPSA stability constant $A$: 10% of expected max updates | Standard Spall (1998) guideline to avoid large step sizes during initial exploration. $a \in \{0.1, 0.5, 1.5\}$, $c=0.15$. | 2026-10-12 | Confirmed |
| D-09 | Early stopping: disabled for main benchmark; patience=5 checkpoints for ablation | Prevents premature stopping artifacts under heavy shot noise; enables fair total-budget comparison. | 2026-10-12 | Confirmed |
| D-10 | Failed-run policy: logged, counted, never silently dropped or reseeded | Full observability into optimizer stability and ill-conditioning; failed runs are preserved in failure tables (T11). | 2026-10-12 | Confirmed |
| D-11 | Tidy targets $\tau$ for cost-to-target: $\{0.25, 0.20, 0.15, 0.10, 0.05\}$ | Predeclared loss thresholds for time-to-target survival and Dolan-More performance profiles. | 2026-10-12 | Confirmed |
| D-12 | E6 sample count: 200 paired parameter vectors per $(n, \text{depth}, \text{init})$ | Sufficient statistical power to distinguish local vs global observable variance decay rates without excessive simulation overhead. | 2026-10-12 | Confirmed |
| D-13 | Parallelism: process pool with isolated per-run directories | Zero mutable shared state across workers; safe atomic writes and deterministic reproduction. | 2026-10-12 | Confirmed |
