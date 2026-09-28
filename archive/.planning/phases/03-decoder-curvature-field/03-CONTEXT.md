# Phase 3: Decoder & Curvature Field - Context

**Gathered:** 2026-08-13
**Status:** Ready for planning

<domain>
## Phase Boundary

A **per-point mean-curvature field** `‖H‖` over the PU manifold, computed by `torch.func`
autodiff through a C2-smooth **CAE chart decoder**. Swiss roll known-answer check first, PU
field second, seeds and sanity third, synthetic control last.

**Substrate is fixed:** the Chart Auto-Encoder, by user decision 2026-08-12. Selection is
**tabled, not resolved** — the choice rests on readiness and a clean defect ledger, **not** on
measured superiority. Full record including the evidence against:
`.planning/phases/02-eigenspectrum-audit-validity-gate/02-NOTE-phase-2-stage-on-hold.md`.

**The precondition is deliberately overridden.** Phase 3's `Depends on` line names a **PASS**
and no PASS exists anywhere in this milestone — 02, 02.2, 02.4 and 02.5 stage 1 are all FAIL.
The override and its consequence must be recorded in Phase 3's **own** artifacts, never
inherited as a silent green light. The consequence, restated so it is carried rather than
lost: **a curvature field decoded from an unvalidated parameterization conflates real
curvature with parameterization damage, and CURV-06/07's synthetic control provably cannot
detect that** — a synthetic manifold that trains cleanly never reproduces the fragmentation
pathology. Partial mitigation on record (`02.4-FINDINGS.md`): every FAIL in this milestone is
**global**-scoped and no local-scoped gate has ever failed here (02.2 T2, 02.4 T3 both
passed). Mean curvature is a local invariant. That is an argument for proceeding, not a
verdict.

**Scope: curvature is local.** `II_p` depends only on a neighbourhood of `p`. No global
parameterization is attempted and none is claimed.

**Requirements DEC-01..05 / CURV-01..08 are stale** — written against Isomap coordinates and a
global chart. Re-mint at plan time; do not re-point.

**What this phase does NOT do.** It does not reopen, soften, recompute or reinterpret any
sealed verdict. It does not resume Phases 02.3, 02.5, 02.6 or 02.7 (see D-05 for the single
scoped exception). It does not partition by curvature or run MKNN — that is Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Swiss roll step 1 — the known-answer check

- **D-01:** The step-1 statistic is the **median `rho_chart` over ≥5 torch seeds**, with the
  full spread reported alongside — never a single draw. Rationale: `02.5-09`'s 4-seed sweep
  measured `{−0.1444, −0.0604, 0.4250, 0.8665}` on identical code, data and fixture, so a
  single-seed gate is a draw from a distribution. Matches
  `02.5-PREREGISTRATION-AMENDMENT-01.md`'s 5-seed rule, itself adopted because stage 1 was
  seed-sensitive. — **Reversibility:** costly — the stopping rule, the sweep design and the
  compute budget all size off this choice.

- **D-02:** The gate is an **absolute floor: median `rho_chart` > 0.65**. The raw-point
  centroid baseline (`0.6712` at `k=30, d=2`) is **reported as context and gates nothing**.
  **Swiss roll only** — PU has no analytic `H`, so no equivalent gate exists there. *User
  decision, overriding all three options offered (beat-the-baseline / absolute-0.90 /
  two-tier).* Note for the planner: `02.5-09-SUMMARY.md` §3 warns `0.6712` "should not be read
  as a validated reference point" — it missed `02.5-05`'s own `>0.90` bar — which is
  consistent with demoting it to context. — **Reversibility:** one-way — the floor is declared
  in the plan before the sweep runs; changing it after seeing `rho` is post-hoc tuning, the
  exact forking-paths move this milestone's discipline exists to prevent.

- **D-03:** **Only the exact `g`-trace path may gate.** The Hutchinson randomized-trace
  estimator (`chart_curvature.randomized_trace_mean_curvature_nongating`) stays non-gating,
  per `02.5-NOTE-randomized-trace.md`'s demotion of it to a convergence check on the exact
  path. Report `K = 4, 8, 16` as context if run at all.

- **D-04:** Under an `n_charts` sweep the floor applies to the **best config** — the roll
  passes if median `rho_chart > 0.65` at **any** swept `n_charts`, with the **full sweep table
  printed** so the best-of-N is visible rather than hidden. This is the pipeline-validation
  reading: the question is whether this code can recover a known answer at all. **The
  multiple-comparisons loosening must be named as a caveat in the read-out** — N configs give
  N shots at a fixed bar.

- **D-05a (stopping rule):** If the roll cannot clear 0.65 at any swept config, **Phase 3 stops
  and reports**, per the ROADMAP. That is a complete outcome, not a failure to work around.

### Chart count — the phase's only measured lever

- **D-05:** **`n_charts` is an in-scope Phase 3 hyperparameter.** *Explicit user scope ruling,
  overriding the on-hold status of Phase 02.3 (CAE Iteration) for this knob only.* **The plan
  must record this ruling in its own artifacts, exactly as it records the gate override.**
  Rationale: `02.5-09`'s sweep found `rho_chart` **monotone in charts actually used** —
  3 → `0.8665`, 5 → `0.4250`, 8 → `−0.0604` / `−0.1444` — with `max cond(g)` tracking it in
  lockstep (3.26, 7.64, 63.19, 122.22). Mechanism is visible in the plots: the learned surface
  fragments into chart-sized pieces joined by near-straight chords, and `‖H‖` bands along those
  artificial seams rather than along the spiral. Each chart decoder is C2-smooth internally;
  the **atlas** is not. This is the only axis ever measured to move `rho`. — **Reversibility:**
  one-way — it crosses a phase boundary the hold note drew; retracting it means the phase has
  no lever.

- **D-06:** **Nothing from the roll constrains the PU fit.** The roll's job is solely to show
  the pipeline recovers a known answer at *some* `n_charts`. PU gets its **own independent
  sweep**, selected on model-side diagnostics alone; the roll's winning `n_charts` is reported
  as context and never used to pick the PU value. Rationale: `02.5-NOTE-substrate-selection.md`
  §4 — Swiss roll topology does not transfer to PU, and nothing measured on a 2-D sheet bounds
  a `d≈20` manifold's template, dimension or curvature.

- **D-07:** The PU-side `n_charts` selection rule keys on **four model-side diagnostics**, none
  of which needs ground truth: `max cond(g)`, argmax **chart occupancy**, held-out
  **reconstruction** error, and **persistent-homology agreement restricted to H0 and H1**.
  *User modification of the offered three-part rule.* **H2 is excluded deliberately:** it has
  **no power analysis anywhere in this milestone**, so every `beta_2 = 0` is an unbounded null,
  and it is the term that blew `02.7-07`'s compute budget. H1's power *is* bounded — measured
  against an `S¹ × B¹⁸` fixture at PU's own `d≈20, D=768`, it detects a cycle down to **3.3×**
  the manifold's transverse thickness at `n=383` and **2.0×** at `n=2000`. H0 is the cheap
  MST-merge structure. — **Reversibility:** costly — the rule picks the fit the whole
  downstream field is computed from.

### Autodiff cost at PU scale

- **D-08:** Add a **forward-mode toggle** to `chart_curvature.py`
  (`chart_mean_curvature`, `chart_curvature_field`). **Default stays reverse** — `jacrev` /
  `torch.func.hessian` bit-identical to today — so every existing call site, the `02.5-09`
  notebook and the sealed roll numbers reproduce untouched. Forward mode is **opted into**
  where it pays. Scope is `chart_curvature.py` **only**; not `decoder_curvature.py`, not
  `derivative_bridge.py`.

  **Why.** For `F: R^d → R^D`, reverse mode computes `vᵀJ` (one Jacobian **row** per pass, so
  `~D` passes); forward mode computes `Jv` (one **column** per pass, so `~d` passes). Reverse
  is the right default for scalar-loss ML (`D=1`, `d` huge) and is what fingers type. The
  decoder is the opposite regime. `torch.func.hessian = jacfwd(jacrev)` compounds it:

  | | Jacobian | Hessian |
  |---|---|---|
  | current (`jacrev`, `jacfwd∘jacrev`) | `~D = 768` | `~d·D = 15,360` |
  | forward (`jacfwd`, `jacfwd∘jacfwd`) | `~d = 20` | `~d² = 400` |

  Same tensor, same values, exact either way. It survived unnoticed because **every execution
  to date has been Swiss roll at `d=2, D=3`**, where reverse costs 3 and forward costs 2.

  **The ~38× is an operation-count ceiling, NOT a measured speedup.** PyTorch's forward-mode
  path is less optimized than its reverse path, `vmap` over dual numbers has its own constants,
  and forward-mode `vmap` may hit an unimplemented-batching-rule error on some decoder op and
  not run at all. That is precisely why the toggle defaults to reverse.

- **D-09:** **Equivalence must be proved, not asserted** — agreement to float64 round-off
  between the forward and reverse paths, in the same shape as the existing
  `test_chart_curvature_dxd_solve_matches_explicit_projector`, which is how the current
  `d×d`-solve optimization earned its place. The existing sphere known-answer test and shape
  assertions must still pass. Note for the planner: `02.5-09` recorded that `chart_curvature.py`
  "needed no change" and ran unmodified on first attempt; this phase is the first thing to edit
  it, so the reverse path staying bit-identical is what preserves the `−0.0604` reproduction.

### The PU fit

- **D-10:** **Fresh fits only.** Everything trains under Phase 3's own protocol; the sealed
  02.2 PU fit plays no part. *User decision, declining the offered option to carry it as a
  labelled sweep row.* For reference, that sealed fit is `D_CHART=20`, `L_EMBED=40`,
  `N_CHARTS_INIT=16` (all 16 survived pruning), SiLU, width 250, 8,000 train / 2,000 holdout at
  `SPLIT_SEED=20260803`.

- **D-11:** **`chart_dim = 20`**, with its justification **restated in Phase 3's own artifacts**
  rather than silently inherited: TwoNN 19.5, local-PCA median 25.0, local intrinsic dimension
  stable and tight at ~20–25 (std 2.0), median of 8 geometric estimators 18
  (`02-FINDINGS.md` §6.3, §6.4). **`d_frozen = 5` is explicitly rejected** and the rejection
  recorded — `02-FINDINGS.md` §6.4 flags it as suspect against three estimates clustering at
  18–25, and it must not be inherited downstream.

- **D-12:** **Escalate to a `d` sweep only if the best `d=20` config loses to a matched
  plain-AE control** on held-out reconstruction and PH H0/H1 agreement. *User decision.*
  **Known limitation, recorded rather than re-litigated:** reconstruction and topology do not
  predict curvature quality — this is `02.5-09`'s D-09 finding (seeds 0 and 3 differ by 0.29
  percentage points of reconstruction error and 0.49 of Spearman) and the
  `02.5-NOTE-randomized-trace.md` §C illustration (a decoder learning `y = 0.7ax²` where truth
  is `y = ax²` attenuates curvature 30% with essentially no reconstruction signal). **Expect
  this trigger to fire:** quick task `20260809-topoae-vs-cae-persistence` measured the CAE
  preserving H0 merge structure ~5× worse than a plain AE at the same embedding dimension
  (`cae_retained = 0.183246`) using ~8× the parameters. Different chart counts may change it;
  the prior is that the `d` sweep runs.

- **D-13:** **PU sweep budget: 3 `n_charts` values × 3 seeds = 9 fits, ~3–5 h.** Seed count
  drops from the roll's 5 because PU has **no gate to clear** — only a diagnostics table to
  read. Widen later if the trend is unclear. Anchor: 02.2's timing probe measured **1,941.2 s
  per PU fit** at `n_charts=16, d=20, width 250, 8,000 train` — training only, and it predates
  the forward-mode toggle; **curvature at `d=20, D=768` has never been timed.** A timing
  measurement before committing the sweep is sensible and is left to the planner.

### Verification and carried debt

- **D-14:** **`derivative_bridge` runs at PU scale**, and **WR-01/02/03 close inside Phase 3**.
  Rationale: the D-09 equivalence test compares two *autodiff* paths and cannot catch a bug
  shared by both; finite differences can, and this phase is the first to edit
  `chart_curvature.py`. The hold note §5 trigger 2 explicitly anticipates the bridge landing on
  Phase 3 now that `02.5-10` is not coming.
  - **WR-01** — `finite_difference_jacobian` / `finite_difference_hessian` / `calibrate_fd_step`
    pass a **bound method** to `chart_curvature._assert_float64`, which expects the *model*; the
    per-parameter float64 guard is silently skipped. Masked only because every current call site
    pre-casts with `.double()`.
  - **WR-02** — relative-error columns exceed 100% against near-zero references
    (`full_hess_max_abs_rel = 1.1351e+00` already visible in the recorded PU table). This is
    exactly the failure mode a curvature bridge table hits.
  - **WR-03** — `calibrate_fd_step` computes its autodiff Hessian unchunked.
  Source: `02.6-REVIEW.md`, commit `1d3f666`. None affects any number Phase 02.6 recorded.

- **D-15:** **Gate machinery: the roll floor is the only declared bar.** No `PREREGISTRATION.md`,
  no ratification commit, no git-ancestry proof script, no verdict JSON artifact, no threshold
  table. The 0.65 floor is written into the plan before anything runs; that is sufficient.
  Rationale: the ROADMAP entry's own "start simple — add gate machinery only when a step's
  result would otherwise be untrustworthy", and CLAUDE.md's keep-it-simple rule. — **
  Reversibility:** reversible — machinery can be added to a later step if one turns out to need
  it.

### Claude's Discretion

- Exact `n_charts` values in each sweep (roll and PU). The roll should span the measured
  monotone range (something like 2/3/5/8); PU picks 3 values under D-13's budget.
- Whether to run a timing probe before committing the PU sweep (D-13 notes it is sensible).
- Deliverable shape — the milestone pattern is a runner script under `notebooks/diagnostics/`
  for the expensive PU grid plus a presentation notebook, and CLAUDE.md **mandates** a new
  `notebooks/03_swiss_roll_*_check.ipynb`. Additive only: `02.5_swiss_roll_chart_curvature_check.ipynb`
  is not rewritten.
- How the four D-07 diagnostics combine into a single selection (weighted, lexicographic, or a
  printed table plus a stated rule).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The decision this phase runs under
- `.planning/phases/02-eigenspectrum-audit-validity-gate/02-NOTE-phase-2-stage-on-hold.md` —
  the tabling decision, the evidence for and against the CAE, §3 the gate override and its
  consequence, §4 exactly where each held phase stopped and the carried debt, §5 what ends the
  hold.
- `.planning/ROADMAP.md` § "Phase 3: Decoder & Curvature Field" — the four-step structure,
  success criteria, assets-to-reuse list and constraints. Rewritten 2026-08-12; four superseded
  amendment layers dropped.

### Why the curvature question is shaped this way
- `.planning/phases/02.5-local-curvature-feasibility-cae-re-gate/02.5-09-SUMMARY.md` — the
  `−0.0604` measurement, the four-axis scoring, the 4-seed sweep, the fragmentation mechanism,
  and §3's warning against reading `0.6712` as validated.
- `.planning/phases/02.5-local-curvature-feasibility-cae-re-gate/02.5-NOTE-randomized-trace.md` —
  the trace convention (`H = tr_g(II)`, **not** the averaged `(1/d)tr`), the factor-of-`d`
  normalization trap, why antithetic probes give exactly zero variance reduction, why Hutch++
  does not apply to a vector-valued `II`, and §C's amplitude-vs-rank requirement.
- `.planning/phases/02.5-local-curvature-feasibility-cae-re-gate/02.5-NOTE-high-d-curvature-approaches.md` §1, §1a, §2d —
  the `r/R = 0.906` locality breakdown at `d=20` and the direction/magnitude/calibration
  reporting format.
- `.planning/phases/02.5-local-curvature-feasibility-cae-re-gate/02.5-NOTE-substrate-selection.md` §4 —
  Swiss roll topology does not transfer to PU. The basis for D-06.
- `.planning/phases/02.5-local-curvature-feasibility-cae-re-gate/02.5-CONTEXT.md` — D-00
  (mean curvature is the **trace**, so `d(d+1)/2` underdetermination may not bind), D-01
  (Spearman gates, median relative error does not), D-05, D-07.

### Sealed verdicts and dimension evidence
- `.planning/phases/02-eigenspectrum-audit-validity-gate/02-FINDINGS.md` §6.3, §6.4 — the
  intrinsic-dimension estimates behind D-11 and the reason `d_frozen = 5` is suspect.
- `.planning/phases/02.2-chart-autoencoder-validity-test-inserted/02.2-FINDINGS.md` §2 — the
  sealed CAE pre-registered design table (`D_CHART`, `L_EMBED`, `N_CHARTS_INIT`, activation,
  split) and the sealed FAIL.
- `.planning/phases/02.4-topological-auto-encoder-validity-test-inserted/02.4-FINDINGS.md` — the
  every-FAIL-is-global-scoped finding the phase's local-curvature argument leans on.
- `.planning/phases/02.6-decoder-substrate-screening/02.6-FINDINGS-02.md` — no substrate
  promoted or eliminated; the two named confounds; the derivative-usability bridge tables.
- `.planning/phases/02.6-decoder-substrate-screening/02.6-REVIEW.md` (commit `1d3f666`) —
  WR-01/02/03 in full.

### Code to reuse, never rebuild
- `notebooks/pu_manifold/chart_curvature.py` — exact chart curvature via `torch.func`;
  `chart_mean_curvature`, `chart_curvature_field`, `chart_decoder_map`,
  `assert_c2_activation`, `curvature_fidelity_report`,
  `randomized_trace_mean_curvature_nongating`. **The one module this phase edits (D-08).**
- `notebooks/pu_manifold/curvature_probe.py` — `centroid_mean_curvature` (the raw-point
  baseline), `swiss_roll_analytic_H_scaled`, `make_swiss_roll_fixture`,
  `spearman_gate_statistic`, `median_relative_error`.
- `notebooks/pu_manifold/decoder_curvature.py` — `assert_c2_decoder` (CURV/DEC C2 guard),
  `swiss_roll_analytic_H_vector`.
- `notebooks/pu_manifold/derivative_bridge.py` — autodiff vs finite-difference agreement
  (D-14).
- `notebooks/pu_manifold/persistence_probe.py` — PH agreement for D-07's H0/H1 term.
- `notebooks/pu_manifold/cae.py` — **sealed: import, never edit.** `ChartAutoEncoder`,
  `train_cae`, `PlainAutoEncoder`, `train_plain_ae`, `reconstruction_stats`, `chart_survival`.
- `notebooks/02.5_swiss_roll_chart_curvature_check.ipynb` — the reference the new roll notebook
  copies the shape of. **Additive only — not rewritten.**
- `notebooks/02.2_swiss_roll_cae_check.ipynb` — CLAUDE.md's named reference implementation for
  Swiss roll checks.

### Standing rules
- `CLAUDE.md` — the mandatory Swiss roll check for every new manifold model (what the notebook
  must and must not do), `src/effdim/` frozen this milestone, additive-only rule, keep-it-simple
  rule.
- `.planning/REQUIREMENTS.md` DEC-01..05, CURV-01..08 — **stale**, written against Isomap
  coordinates and a global chart. Re-mint at plan time; do not re-point.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `chart_curvature.chart_mean_curvature` — returns `H_vec`, `H_norm`,
  `metric_condition_number`, both derivative shapes, and provenance. Runs float64 only
  (`_assert_float64` refuses otherwise). Chunked at `VMAP_CHUNK` with exact short-chunk padding.
  Already g-traces **before** projecting (the two commute), so no `(D,D)` projector and no full
  `II` tensor is ever materialized — at `D=768` that would be 151 MB + 78 MB per 32-point chunk.
- `chart_curvature.chart_decoder_map(model, chart_idx)` — the per-chart decoder as a plain
  `R^chart_dim → R^out_dim` map, which is what both autodiff modes differentiate.
- `curvature_probe.centroid_mean_curvature` — the raw-point Laplace–Beltrami baseline. Its
  `2d/r2` scale constant is **corrected** (not `2(d+2)/r2` as `02.5-RESEARCH.md` Pattern 1
  states) and pinned by a sphere known-answer test. `d` is a **required positional argument with
  no default**, deliberately, so `d_frozen=5` cannot be inherited by accident.
- `cae.PlainAutoEncoder` / `train_plain_ae` — the matched baseline CLAUDE.md requires and D-12's
  escalation trigger keys on.
- `cae.chart_survival` — **known-broken as a pruning criterion.** It thresholds a *ratio* to the
  largest chart, so decoupled decay shrinks live and dead charts together and cancels; quick task
  `20260809-cae-undertraining-test` measured it returning 8/8 in all 24 fits while argmax
  occupancy fell to 6/8. **Use argmax occupancy, not `chart_survival`, for D-07's occupancy
  term.**

### Established Patterns
- **Curvature convention is pinned:** `CURVATURE_CONVENTION = "trace"`, `H = tr_g(II)`, with a
  regression guard. A unit sphere gives `‖H‖ = d`, not 1. Transcribing the averaged convention
  `H = (1/d)tr_g B` from external sources introduces a factor-of-20 error against every fixture
  and every recorded number. This codebase has already shipped and fixed exactly one
  factor-of-`d` scale bug.
- **Four axes scored separately, never collapsed:** direction (cosine), magnitude (median ratio
  **and** per-point CV), calibration (slope, intercept, R²), rank (Spearman). At seed 0 the
  decoder scored cosine `0.9706` — direction nearly right — while calibration slope was
  `−0.1856` at R² `0.0533`. A rank statistic is blind to that split.
- **Expensive grids are runner scripts** under `notebooks/diagnostics/` (resumable, per-cell
  timed, `--smoke` / `--dry-run` / `--resume`), with notebooks reserved for presentation and for
  CLAUDE.md's Swiss roll checks. See `template_benchmark_run.py`, `cae_train_run.py`.
- **Notebooks are committed executed, with outputs**, and the roll check must train from scratch
  in-notebook, never touching `notebooks/.cache/`, targeting under two minutes on CPU.

### Integration Points
- Phase 4 consumes `‖H‖` by **quantile partition**, which is why ordering (Spearman) is the
  gating statistic and magnitude is not. A sparser or seed-selected field propagates there.
- `notebooks/pu_manifold/tests/` — the forward-mode equivalence test (D-09) lands here beside
  the existing projector-equivalence and sphere known-answer tests.

</code_context>

<specifics>
## Specific Ideas

- The forward-mode toggle exists **so it can be abandoned cheaply** — the user's framing was
  "just make it a toggle, so if it doesn't work we can go back to reverse easily." Default
  reverse, opt in to forward. Do not make forward the default even after equivalence passes.
- PH is wanted for the PU selection rule **at H0 and H1 only**. Do not compute H2.
- `0.65` is a deliberate floor, chosen with the `0.6712` baseline in view and set just below it.
  It is not the baseline in disguise — the baseline gates nothing.
- Both the **gate override** (inherited) and the **`n_charts` scope ruling** (D-05, new this
  discussion) must appear in Phase 3's own artifacts. Neither may be inherited silently.

</specifics>

<deferred>
## Deferred Ideas

- **PH H2 agreement** as a selection term — excluded from D-07 until a power analysis exists.
  Every `beta_2 = 0` in this milestone is currently an unbounded null.
- **Resuming Phases 02.3 / 02.5 / 02.7** — remains on hold. D-05 opens `n_charts` to Phase 3 and
  nothing else. Per the hold note §5, only an explicit user decision resumes a named phase; the
  likeliest trigger is Phase 3's curvature field failing with `−0.0604` implicated.
- **The 5 open windows** in `.planning/WINDOWS.md` — `/gsd-ship` blocks until each is closed or
  waived with a recorded reason. Not Phase 3's work unless it touches one.
- **`VERIFICATION.md` missing** on Phases 02, 02.1 and 02.6.
- **02.7's two Swiss roll defects** (GMST local-dispersion instability, inflated banded β₀) —
  must be resolved before that phase's ~17 h grid is worth running. Does not block Phase 3.
- **Two gray areas raised but not discussed:** the exact PU deliverable shape, and how the step-4
  synthetic control is built given it provably cannot detect parameterization damage. Left to
  planning; the second is the more consequential.

</deferred>

---

*Phase: 3-decoder-curvature-field*
*Context gathered: 2026-08-13*
