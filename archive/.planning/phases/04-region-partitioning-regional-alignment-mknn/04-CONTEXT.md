# Phase 4: Region Partitioning & Regional Alignment (MKNN) - Context

**Gathered:** 2026-08-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Freeze a pre-specified partition of the PU 10k subsample into two curvature regions, report the
density confound explicitly, then compute per-region crossmodal MKNN (HSC vs Legacy Survey)
against region-local permutation nulls with bootstrap CIs across `k = 5, 10, 20, 50`, and give an
explicit verdict on whether the regional difference is distinguishable from noise — where "no
detectable difference" is a valid reported outcome.

This phase clarifies HOW that is implemented. The phase boundary from ROADMAP.md is fixed:
requirements REGN-01..05 (+ new REGN-06) and MKNN-01..08. No new capability is added.

**Unblocking status.** Phase 4 was BLOCKED under D-11 pending a route out of Phase 03.1's
ordering defect. D4-02 (RESOLVED 2026-08-23) supplies it: the instrument is
`curvature_probe.centroid_mean_curvature` applied directly to the point cloud, which forms no
pullback metric and cannot suffer the `cond(g)` pathology. Phase 03.1's metric regularization is
therefore OPTIONAL, not blocking, and Phase 3's non-reproducing decoder field is off the critical
path. No sealed verdict in Phases 2, 02.x, 3 or 03.1 is reopened, softened, recomputed or
reinterpreted by anything in this phase.

</domain>

<decisions>
## Implementation Decisions

Decisions D4-01..D4-04 were taken at the Phase 3 close and are recorded in
`.planning/phases/03-decoder-curvature-field/03-NOTE-phase-4-decisions.md`. They are LOCKED and
are not relitigated here; they are restated below only where this discussion changes or extends
them. D4-05..D4-19 are new, taken in this discussion on 2026-08-23.

### Curvature field on PU, and `k`

- **D4-05:** PU's estimated `‖H‖` dynamic range is **~4.8x** (p95/p05; measured at 5.54 / 4.83 /
  4.79 / 4.86 for k = 30 / 60 / 120 / 231, i.e. flat in `k`, so not a `k` artifact). The runner's
  own calibration puts the unrankable `quadratic_bowl` at 1.4x (`rho +0.03`) and the rankable
  `cubic` / `ridge` at 28.2x / 34.3x (`rho +0.61` / `+0.41`). **PU sits far nearer the unrankable
  end.** Phase 4 reports this number with its calibration and partitions on direction anyway —
  direction is a unit vector and does not consume the magnitude spread, so this is treated as
  further confirmation of D4-01 rather than a blocker. **No new gate is added.**
  — **Reversibility:** reversible — a reported number, no code depends on it.

- **D4-06:** The `k` sweep is **extended past 231** before `k` is frozen. Median `R_H` was still
  rising monotonically at 231 (0.078 → 0.247 → 0.428 → 0.589), so the plateau has not been
  demonstrated. Run at least `k = 350, 500`.
  — **Reversibility:** reversible — adds sweep points, changes no interface.

- **D4-07:** `k` is frozen by a **spacing-free absolute-increment rule, declared before the new
  sweep points are run**: freeze at the smallest `k` where median `R_H` gains **less than +0.03**
  over the previous sweep point AND median `R_H >= 0.5`. Chosen specifically because Phase 1's
  plateau rule for `k*=15` failed on uneven spacing — `WINDOWS.md` records that `STAGE2_K` was
  unevenly spaced so the plateau was maximal in *index* space, not `k` space. An absolute
  increment never compares gaps across unevenly spaced points and is immune to that defect.
  — **Reversibility:** one-way — this is a pre-registration. Once the new `k` values have been
  run and seen, the rule cannot be changed without invalidating the freeze and requiring a fresh
  pre-registration plus full re-run, exactly as the 02.2 CAE pre-registration precedent requires.

- **D4-08:** Phase 4 does **NOT** run D4-02 Amendment 02's cheap cross-estimator mitigation on PU.
  D4-03 stands as taken. Rationale recorded as the developer gave it: the blind spot was accepted
  deliberately and recorded as such; adding the check invites treating its result as a gate it was
  never declared to be, and the decoder arm measured cosine ~0 (twice negative) on fixtures, so
  agreement with it would be uninformative in either direction. Phase 4 states the blind spot in
  its own words instead.
  — **Reversibility:** reversible — the check remains available later at one cell.

### Direction partition scheme

- **D4-09:** Regions are formed by a **sign split on the top eigenvector of the unit-`H`
  covariance**. Compute `Cov(H_i/‖H_i‖)`, take its leading eigenvector `v`, assign each point by
  the sign of `<H_i/‖H_i‖, v>`. Chosen over spherical k-means because it is deterministic (no
  seed, no initialization, no k-means restart sensitivity), yields exactly the two regions the
  MKNN comparison and REGN-05 expect, and has an exact known answer available on ridge fixtures
  where `v` should recover the construction's bending direction `w`.
  — **Reversibility:** costly — the frozen partition is the input to every regional MKNN number,
  its null, and its CI; changing the scheme after any regional number exists invalidates all of
  them and breaks the pre-specification discipline the Ordering constraint requires.

- **D4-10:** **No known-answer fixture validation runs before the PU split is frozen.** Neither
  `make_ridge_graph_control` nor `make_multinormal_ridge_control` is run. **This OVERRIDES the
  D4-01 body text that names the ridge check "a Phase 4 precondition."** Developer rationale, as
  given: D4-01 was adopted on partial evidence with the codimension gap explicitly recorded;
  running `m=4`/`m=8` narrows codimension 1 to 8 against PU's ~748 and risks reading as closure
  rather than narrowing.
  — **Reversibility:** reversible — both fixtures exist and are tested; the check can be run later
  without touching anything Phase 4 produces.

- **D4-11:** **Re-mint REGN-01, REGN-03 and REGN-04 in `REQUIREMENTS.md`, and add REGN-06.**
  REGN-01's "Isomap coordinate space", REGN-03's "by quantile" and REGN-04's "quantile threshold"
  are all written for the superseded `|H|`-quantile + decoder route. IDs are preserved and the
  discipline they encode is preserved; only the quantity changes. Follows the Phase 3 requirement
  re-mint precedent already recorded in `REQUIREMENTS.md`.
  - REGN-01 → local sample-density measure per point **in the ambient embedding space the
    curvature field is estimated in** (see D4-13), shown.
  - REGN-03 → points partitioned by a **data-derived direction criterion**, never by a fixed
    absolute threshold.
  - REGN-04 → the partition rule is specified and frozen **before** regional alignment is
    computed, and that ordering is visible in the notebook.
  - REGN-06 (new) → the eigenvector `v` and the resulting sign split are **recorded and frozen as
    artifacts** before any MKNN number is computed, so the split is auditable after the fact.
  — **Reversibility:** one-way — `REQUIREMENTS.md` is the project's requirement contract and its
  coverage table is consumed by the plan-checker's requirement gate; re-minting IDs after plans
  cite them forces a plan revision.

- **D4-12:** **No new Swiss roll notebook** for the direction-partition rule. `CLAUDE.md`'s
  standing rule targets models that map data to a lower-dimensional representation and back, or
  that claim to recover manifold structure; a sign split on an eigenvector of an already-computed
  field is neither, and the estimator underneath it is already covered by
  `notebooks/02.5_swiss_roll_curvature_probe_check.ipynb`. **Phase 4's record must state this
  reasoning explicitly** rather than silently omitting the notebook.
  — **Reversibility:** reversible.

### Density confound

- **D4-13:** Local density is measured in the **ambient 768-d embedding space** — the same space
  the estimator runs in. `pu_curvature_rankability_run.py` applies `centroid_mean_curvature`
  directly to the normalized 768-d embeddings from `subsample_*.npz`, not to Isomap coordinates;
  the centroid displacement that can masquerade as curvature is computed from 768-d
  neighbourhoods, so the density that could fake it is 768-d density. REGN-01 re-minted
  accordingly (D4-11).
  — **Reversibility:** reversible.

- **D4-14:** The density-confound battery is **the REGN-02 correlation only**. No centroid-distance
  check, no partial regression, no density-matched stratification, no density-matched null.
  REGN-02 asks for the correlation to be reported explicitly before the split is trusted; that is
  what is delivered, with the confound risk stated plainly in the record for the reader to judge.
  **Consequence Phase 4's record must carry:** MKNN is itself a k-NN statistic and therefore
  directly density-sensitive, so **without a density-matched null a regional MKNN difference
  cannot be separated from a regional density difference by anything in this phase.** REGN-02's
  correlation number is therefore the only evidence bearing on that, and MKNN-07's verdict must be
  worded to reflect it.
  — **Reversibility:** reversible — the additional controls can be added later without
  invalidating the frozen split.

- **D4-15:** The headline PU field is **density-corrected**: `density_correct=True`,
  **`k_density=30`** (the pre-registered value, already used throughout the codebase — no new
  constant is introduced, and per D-05 there is deliberately no continuous strength dial).
  Accepted cost: **the `k` sweep re-runs corrected from `k=30` upward**, so the four existing
  uncorrected `R_H` numbers (k = 30/60/120/231) are **superseded, not extended**. Budget ~2,100s
  to reproduce those four, plus `k = 350, 500` on top.
  **Recorded so the rationale is not overstated later:** `02.5-02-SUMMARY.md` amended D-06's
  original claim. On a flat fixture the correction is *provably* inert — the normal projection is
  mathematically exact there regardless of density skew, so there is nothing to remove — and it is
  inert under uniform sampling. Its real, measured effect is a **~8-10% reduction in median
  relative error on a genuinely curved, strongly-skewed fixture**. It is adopted on that basis,
  not on the retracted flat-fixture claim.
  — **Reversibility:** costly — the frozen `k` and its `R_H` come from this estimator
  configuration; switching after the freeze invalidates the sweep and the pre-registration.

### MKNN mechanics and statistical budget

- **D4-16:** Per-region MKNN k-NN sets are computed **within the region's own index set** — subset
  both embeddings to the region's rows, then compute `N_k` inside that subset. Score and
  permutation null then live in the same index set, satisfying MKNN-04 by construction. Accepted
  consequence: the regional number is not directly comparable to the global MKNN-02 figure because
  `k/n` differs; a smaller region raises chance alignment, which is exactly what that region's own
  null absorbs.
  — **Reversibility:** costly — changing the neighbour scope changes every regional number, null
  and CI in the phase.

- **D4-17:** **1,000 permutations and 1,000 bootstrap resamples** per cell (2 regions x 4 values of
  `k` = 8 cells). Resolves `p` to ~0.001, enough to call an effect 4-20x over the `k/n` chance
  floor, and gives stable percentile CIs. The permutation only shuffles row correspondence, so the
  per-modality k-NN index is built **once per cell and reused across all permutations** — this is
  what makes the budget affordable and the planner must implement it that way.
  — **Reversibility:** reversible.

- **D4-18:** The MKNN implementation's check is **MKNN-02's global reproduction on real data**. No
  `tests/test_mknn.py` is added, notwithstanding the package's one-test-file-per-module
  convention. Landing near the paper's published crossmodal range is treated as a stronger
  end-to-end check than synthetic unit tests, and MKNN-02 is a requirement regardless so it costs
  nothing extra.
  — **Reversibility:** reversible — a unit test can be added at any time.

- **D4-19:** MKNN-02 reports the **raw MKNN alongside the `k/n` chance floor at our `n`, and the
  paper's raw range alongside their `n`** (10,000 vs 101,725). The ratio-over-chance carries the
  comparison, not the raw number. No subsample-size sensitivity curve and no full-config
  101,725-row pass. This makes the `n` mismatch explicit rather than hiding it inside a range
  check — necessary because under D4-18 this reproduction is the only implementation check the
  phase has, and "outside the published range" would otherwise not separate a bug from the
  subsample.
  — **Reversibility:** reversible.

### Claude's Discretion

The following were raised and deliberately left to the planner. Each carries its constraint.

- **MKNN-07's verdict rule.** With 2 regions x 4 values of `k`, what counts as "the high-vs-low
  result holds" — all four `k`, a majority, or one pre-designated headline `k` with the rest as
  sensitivity — and whether the 8 comparisons take a multiplicity correction. **Hard constraint:
  the rule MUST be written into the notebook before the first regional MKNN number is computed,
  and that ordering must be visible in the notebook.** This is precisely what the ROADMAP's
  Ordering constraint (garden-of-forking-paths guard) exists to prevent; deciding it after seeing
  the numbers is the failure mode.
- **Near-zero `‖H‖` points.** Direction is ill-defined where `‖H‖ ≈ 0`. Exclusion policy and
  threshold are the planner's call; whatever is chosen must be declared before the split is frozen
  and the excluded count reported (REGN-05 already requires region point counts).
- **Unbalanced regions.** Behaviour if the sign split returns badly unbalanced regions (e.g.
  9,400 / 600) — whether a minimum region size aborts or merely caveats the comparison. Declare
  before freezing.
- **Whether `v` is computed on all 10k points** or on some admissible subset.
- **Field computation scope:** all 10k rows or a subsample; the anchor-point protocol for `R_H` at
  the new `k` values (the existing runner used `n_anchor=1000`); seed policy; whether `d=20`
  remains the estimator's working tangent dimension or is re-derived. Note D-07 bars inheriting
  Phase 2's frozen embedding dimension by accident — `d` is a required explicit call-site choice.
- **Correlation statistic for REGN-02** — Spearman on density vs the signed projection
  `<H_i/‖H_i‖, v>`, vs on `‖H‖`, or both.
- **Whether density is also compared between the two regions after the split**, in addition to
  being correlated with curvature before it. Given D4-14, a region-level density imbalance is the
  single most decision-relevant density number the phase can report.
- **Exact `k` grid past 231** beyond the `350, 500` floor set by D4-06.
- **MKNN-08's hubness caveat** — stated as a caveat, or substantiated with a hubness statistic
  computed on the two regions.
- **Shipped artifact shape** — notebook only, or notebook plus a runner in
  `notebooks/diagnostics/` following the established pattern.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Binding decisions and their evidence
- `.planning/phases/03-decoder-curvature-field/03-NOTE-phase-4-decisions.md` — D4-01..D4-04 plus
  Amendment 01 (D4-01's validation scoped out; the codimension gap) and Amendment 02 (D4-02
  RESOLVED to the point-cloud estimator; the undertrained-decoder caveat). **The single most
  important file for this phase.**
- `.planning/spikes/003-fixture-validity-audit/README.md` — the measurements D4-01 and D4-02 rest
  on: the three fidelity axes coming apart at `d=20`.
- `.claude/skills/spike-findings-effdim/SKILL.md` — non-negotiable requirements for any curvature
  work here: sealed scorer unmodified, `CURVATURE_CONVENTION = "trace"`, no shrinkage dial, never
  edit `notebooks/pu_manifold/` or `notebooks/diagnostics/` from a spike, anchor at low `d`, never
  report a rank statistic without the direction axis beside it, show `r/R`.
- `.planning/phases/03-decoder-curvature-field/03-FINDINGS.md` §1, §9 — the gate override and the
  Phase 4 handoff; §1 sets the standard for how an inherited gap must be stated in a phase's own
  words.
- `.planning/phases/03.1-decoder-metric-regularization-inserted/03.1-FINDINGS.md` §10 — why
  `scale` repairs the metric without repairing the ordering.
- `.planning/phases/02.5-local-curvature-feasibility-cae-re-gate/02.5-02-SUMMARY.md` — the amended
  D-06 density-correction claim that D4-15 rests on. Read before describing what the correction
  does.
- `.planning/WINDOWS.md` — the `k*=15` plateau-rule defect (uneven spacing) that D4-07 is designed
  around.

### Requirements and scope
- `.planning/ROADMAP.md` § Phase 4 — goal, success criteria, and the **Ordering constraint**
  (pre-specify the split, then compute).
- `.planning/REQUIREMENTS.md` — REGN-01..05, MKNN-01..08, the coverage table, and the Phase 3
  re-mint precedent D4-11 follows.
- `CLAUDE.md` — the standing Swiss roll rule (and D4-12's reasoning for why it is not triggered),
  the additive-only rule, and the KEEP THINGS SIMPLE FIRST rule.

### Code the phase consumes
- `notebooks/pu_manifold/curvature_probe.py` — `centroid_mean_curvature` (D-05 gating estimator,
  `d` a required positional arg per D-07), `local_density_weights` (D-06, `k_density` its only
  constant), `centroid_mean_curvature_both_densities`, `make_flat_fixture`.
- `notebooks/pu_manifold/mknn.py` — the Phase 4 stub contract: `mknn_score`, `permutation_null`,
  `bootstrap_ci`, all `NotImplementedError`. Docstrings carry the metric definition and the
  MKNN-08 caveat. No module-level faiss import.
- `notebooks/pu_manifold/varying_ii_controls.py` — `make_ridge_graph_control`,
  `make_multinormal_ridge_control` (not run this phase per D4-10; the module docstrings are the
  clearest statement of the codimension argument).
- `notebooks/pu_manifold/cross_split_curvature.py` — the split-half `R_H` machinery D4-06/D4-07
  operate on.
- `notebooks/diagnostics/pu_curvature_rankability_run.py` — the existing runner and its
  `--k 30 60 120 231` protocol; D4-06 extends it and D4-15 re-runs it corrected.
- `notebooks/.cache/03.2_pu_curvature_rankability.jsonl` — the four measured rows quoted in D4-05.
  Uncorrected; superseded as the freeze basis by D4-15.
- `notebooks/diagnostics/direction_partition_run.py`, `notebooks/diagnostics/estimator_headtohead_run.py`
  — built during the D4-01/D4-02 work.

### External
- arXiv:2509.19453 — Duraphe, Smith, Sourav & Wu, *The Platonic Universe*. Source of the MKNN
  metric and the 0.4-2% crossmodal Legacy Survey range MKNN-02 compares against.
- Chechik et al. 2010 — the MKNN metric itself.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `curvature_probe.centroid_mean_curvature(X, k, d, density_correct, k_density)` — the D4-02
  instrument, ready to use. Trace convention, `(n, D)` output. `d` has no default by design.
- `curvature_probe.local_density_weights(X, k_density, d)` — REGN-01/REGN-02's density measure,
  already implemented and pre-registered. `rho_i = k_density / (n * V_d * r_i^d)`, computed in log
  space via `gammaln` because a naive `gamma(d/2+1)` underflows before `d=20`.
- `curvature_probe.centroid_mean_curvature_both_densities(X, k, d, k_density)` — computes corrected
  and uncorrected in ONE pass, bit-identical to two separate calls (there is an
  `np.array_equal` test asserting this). Relevant because D4-15 makes the corrected field the
  headline; if the uncorrected field is ever wanted alongside, it is free here.
- `cross_split_curvature` — split-half `R_H` at disjoint halves evaluated on shared anchor points.
- `pu_curvature_rankability_run.py` — a working end-to-end PU field runner with a `--smoke` mode
  and JSONL output; the natural base for D4-06's extended sweep rather than new code.
- `notebooks/.cache/subsample_*.npz` — the frozen row-aligned 10k pair, L2-normalized, with both
  `dinov3_vitb16_hsc` and `dinov3_vitb16_legacysurvey` columns. This is MKNN's entire input.

### Established Patterns
- **Runner + JSONL cache + notebook.** Every prior phase ships
  `notebooks/diagnostics/<name>_run.py` writing `notebooks/.cache/<phase>_<name>.jsonl`, consumed
  by a notebook. Phase 4 should follow it.
- **One test file per module** in `notebooks/pu_manifold/tests/test_<module>.py`. `mknn.py` is the
  only module without one; D4-18 accepts that gap deliberately.
- **`notebooks/.cache/` is gitignored.** Every artifact there must be reproducible from a runner.
- **Pre-registration discipline.** Thresholds are declared, with a cell-index assertion proving the
  declaration precedes the fit (the 02.2 CAE precedent). D4-07 and REGN-04/REGN-06 both need this.
- **Sealed modules are imported unchanged, never edited from experimental code.**

### Integration Points
- `mknn.py`'s three stubs are the only new implementation surface in the package. Everything else
  is composition of existing functions plus a notebook and a runner.
- The partition (D4-09) is new code with no existing home; `mknn.py` or a small new module are both
  reasonable, and `direction_partition_run.py` already exists in diagnostics as prior art.
- `REQUIREMENTS.md` is edited by D4-11 (re-mint REGN-01/03/04, add REGN-06) — a planning-time task,
  not an execution-time one, and it must land before the plan-checker's requirement gate runs.

</code_context>

<specifics>
## Specific Ideas

- The 4.8x spread number and its bowl/cubic/ridge calibration are to appear **in Phase 4's record
  as a stated number**, not left implicit (D4-05).
- Phase 4's record must state **three separate accepted gaps in its own words**, to the standard
  `03-FINDINGS.md` §1 was held to — by its own account, not by reference:
  1. D4-03 — the field is accepted on split-half reliability alone; split-half cannot detect a
     bias both halves share (measured on the Swiss roll: `R_H = 0.990` alongside `rho = 0.469`).
  2. D4-01 + D4-10 — the direction partition rests on codimension-1 evidence applied to a
     codimension-~748 problem, and no fixture validation runs.
  3. D4-14 — the density confound is reported, not controlled; a regional MKNN difference cannot
     be separated from a regional density difference by anything in this phase.
  Taken together these mean **Phase 4 produces its result with no known-answer anchor at any point
  in the chain — estimator, field, or partition.** That sentence, or its equivalent, belongs in
  the phase record; it is a deliberate and consistently-made developer choice, and the record
  should present it as one.
- "No detectable difference" is an explicitly valid outcome (MKNN-07) and must not be treated as
  a failure of the phase.

</specifics>

<deferred>
## Deferred Ideas

- **Cross-estimator agreement on PU** (D4-02 Amendment 02's "cheap mitigation"): run the CAE chart
  decoder's `H` field alongside `centroid_mean_curvature` at the frozen `k` and report their rank
  agreement. Costs one cell, converts D4-03's accepted blind spot into a measured number.
  Declined for Phase 4 (D4-08); available at any time without relitigating D4-03.
- **Codimension-gap narrowing** via `make_multinormal_ridge_control` at `m = 4, 8`. Declined for
  Phase 4 (D4-10). The fixture exists and is tested; it would narrow codimension 1 to 8 against
  ~748, not close the gap.
- **Density-matched null / partial regression / centroid-distance checks** for the density
  confound. Declined for Phase 4 (D4-14). These are the controls that would let a regional MKNN
  difference be attributed to curvature rather than density.
- **`tests/test_mknn.py`** with exact known answers (identical embeddings → 1.0; independent
  embeddings → `k/n`; hand-computed `n=6, k=2` case; seed reproducibility). Declined for Phase 4
  (D4-18); would restore the package's one-test-per-module convention.
- **Full-config 101,725-row global MKNN** for a like-for-like reproduction of the paper's number.
  Declined for Phase 4 (D4-19).
- **Intramodal MKNN across a model-size ladder** — the paper's stronger 28-56% signal. Already in
  PROJECT.md's Deferred list; needs a second model size and is out of scope for v1.1.

</deferred>

---

*Phase: 4-region-partitioning-regional-alignment-mknn*
*Context gathered: 2026-08-23*
