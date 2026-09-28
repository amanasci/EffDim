# Phase 4: Region Partitioning & Regional Alignment (MKNN) - Research

**Researched:** 2026-08-23
**Domain:** k-NN-based crossmodal representation alignment (MKNN) on a pre-specified,
direction-clustered curvature partition of a high-codimension point cloud
**Confidence:** MEDIUM — the MKNN mechanics and statistics are well-grounded (scipy primitives,
one external paper read directly); the curvature-direction partition and the density-confound
posture are inherited, accepted, partial-evidence decisions from Phase 3/03.1/spike 003, not
something this research can strengthen further per the phase's own locked decisions.

## Summary

Phase 4 has two genuinely separable halves, and the research effort should not be spent evenly
across them. **Half one — the partition** — is almost entirely pre-decided by
`03-NOTE-phase-4-decisions.md` (D4-01, D4-02, D4-09 through D4-15) and by spike 003's measurements:
`centroid_mean_curvature` on the raw 768-d `legacysurvey` point cloud, density-corrected
(`k_density=30`), at `d=20`, with regions formed by a **diametrical sign-split** (Dhillon, Marcotte
& Roshan, *Bioinformatics* 2003 — clustering unit vectors into two antipodal groups via the sign of
the top eigenvector of their covariance) on `H/‖H‖`. Nothing in this research reopens that; the
useful research contribution here is naming the sign-split by its literature term, confirming the
`k`-freeze and near-zero-exclusion mechanics are internally consistent with everything already
built, and being explicit about the codimension caveat this phase inherits (D4-01 Amendment 01).

**Half two — MKNN itself** — is new implementation surface (`mknn.py`'s three
`NotImplementedError` stubs) and is where this research adds the most value. The exact formula is
now verified against the origin paper's own PDF, not just the stub's docstring:
`MKNN(z1,z2) = k^-1 |N_k(z1) ∩ N_k(z2)|` (Duraphe, Smith, Sourav & Wu 2025, arXiv:2509.19453,
§"Measuring representational alignment"; attributed there to Chechik et al. 2010, though that
attribution is itself imprecise — see Priority Question 1 below). The paper's own Legacy-vs-HSC
crossmodal table (Table 2, n≈102k) ranges **0.34%–2.25%** across sixteen model variants, which is
the primary source for the "0.4–2%" range MKNN-02 targets — read directly from the PDF, not from
training-data memory. Neither the exact `k` the paper used nor its self-neighbour-exclusion policy
is stated in the visible text; both are open literature gaps this document names rather than guesses
at. The permutation-null and bootstrap-CI machinery should be built on `scipy.stats.permutation_test`
and `scipy.stats.bootstrap`, mirroring the pattern `curvature_probe.permutation_null` already
established in this codebase, with the k-NN index built once per region×k cell and reused across
every resample — exactly what D4-17's compute budget requires, and achievable with one dense
boolean neighbour-membership matrix per cell rather than per-permutation set intersections.

**Primary recommendation:** implement `mknn.py`'s three functions on top of a per-cell dense
boolean `(n_region, n_region)` neighbour-membership matrix built once from `sklearn.neighbors.
NearestNeighbors` (self excluded, consistent with the rest of this codebase); drive the
permutation null and bootstrap CI through `scipy.stats.permutation_test`
(`permutation_type="pairings"`) and `scipy.stats.bootstrap` (`method="percentile"`) respectively,
never a hand-rolled shuffle loop; substantiate MKNN-08 with a k-occurrence skewness statistic
(Radovanović, Nanopoulos & Ivanović, JMLR 2010) computed from the same membership matrix at
zero extra k-NN cost; and treat the partition itself as frozen, inherited machinery this phase
assembles from already-sealed pieces (`curvature_probe.centroid_mean_curvature`,
`local_density_weights`, `cross_split_curvature`) rather than anything to re-derive.

## Project Constraints (from CLAUDE.md)

- **Swiss roll sanity check rule does not apply here.** CLAUDE.md's mandatory-notebook rule
  targets models that map data to a lower-dimensional representation and back, or that claim to
  recover manifold structure. D4-12 (locked) already rules this out for the direction-partition
  sign-split — it is neither a representation-learning model nor a manifold-recovery claim, and
  the estimator underneath it is already covered by `notebooks/02.5_swiss_roll_curvature_probe_check.
  ipynb`. **MKNN itself is also not a manifold-learning model** (no learned mapping, no latent
  space, no reconstruction), so it independently falls outside the rule's scope too. Phase 4's
  record must still state this reasoning explicitly (D4-12), not silently omit a notebook.
- **Additive only.** `notebooks/pu_manifold/mknn.py`'s three stubs are filled in; no existing
  sealed module (`curvature_probe.py`, `varying_ii_controls.py`, `cross_split_curvature.py`,
  `synthetic_controls.py`) is edited. `notebooks/diagnostics/pu_curvature_rankability_run.py` is
  the natural base to *extend* (new `k` values, `density_correct=True`) per D4-15, not replace.
- **`src/effdim/` and `pyproject.toml` untouched** — notebook-only milestone; all new code lives
  under `notebooks/pu_manifold/` and `notebooks/diagnostics/`.
- **Notebooks committed with outputs, executed end to end.**
- **KEEP THINGS SIMPLE FIRST.** Every recommendation below prefers the already-pinned, already-used
  library primitive (`scipy.stats.permutation_test`, `scipy.stats.bootstrap`, `sklearn.neighbors.
  NearestNeighbors`) over a new dependency or a hand-rolled routine. No new package is required by
  this phase (see Package Legitimacy Audit).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

Decisions D4-01..D4-04 were taken at the Phase 3 close (`03-NOTE-phase-4-decisions.md`) and are
restated here only where D4-05..D4-19 (taken 2026-08-23) change or extend them.

**Curvature field on PU, and `k`:**
- **D4-05:** PU's estimated `‖H‖` dynamic range is **~4.8x** (p95/p05; 5.54/4.83/4.79/4.86 at
  k=30/60/120/231 — flat in `k`). Calibration: unrankable `quadratic_bowl` ≈1.4x (`rho +0.03`);
  rankable `cubic`/`ridge` ≈28.2x/34.3x (`rho +0.61`/`+0.41`). PU sits near the unrankable end on
  spread alone, but Phase 4 partitions on **direction** (a unit vector, unaffected by magnitude
  spread), so this is reported as further confirmation of D4-01, not a blocker. No new gate added.
- **D4-06:** The `k` sweep is extended past 231 before `k` is frozen — median `R_H` was still
  rising monotonically at 231 (0.078→0.247→0.428→0.589). Run at least `k = 350, 500`.
- **D4-07:** `k` is frozen by a **spacing-free absolute-increment rule, declared before the new
  sweep points are run**: freeze at the smallest `k` where median `R_H` gains **less than +0.03**
  over the previous sweep point AND median `R_H >= 0.5`. Chosen because Phase 1's plateau rule for
  `k*=15` failed on uneven spacing (`WINDOWS.md` #1) — an absolute increment never compares gaps
  across unevenly spaced points. **One-way**: once the new `k` values are run and seen, the rule
  cannot change without invalidating the freeze and requiring a fresh pre-registration + re-run.
- **D4-08:** Phase 4 does **NOT** run D4-02 Amendment 02's cheap cross-estimator mitigation on PU.
  D4-03 stands as taken; the decoder arm measured cosine ~0 (twice negative) on fixtures, so
  agreement with it would be uninformative in either direction.

**Direction partition scheme:**
- **D4-09:** Regions are formed by a **sign split on the top eigenvector of the unit-`H`
  covariance**: compute `Cov(H_i/‖H_i‖)`, take its leading eigenvector `v`, assign each point by
  the sign of `<H_i/‖H_i‖, v>`. Chosen over spherical k-means — deterministic (no seed, no
  init-sensitivity), yields exactly two regions, has an exact known answer on ridge fixtures.
  **Costly to reverse** — the frozen partition is the input to every regional MKNN number, null,
  and CI; changing the scheme after any regional number exists invalidates all of them.
- **D4-10:** **No known-answer fixture validation runs before the PU split is frozen.** Neither
  `make_ridge_graph_control` nor `make_multinormal_ridge_control` is run. This **overrides** the
  D4-01 body text calling the ridge check "a Phase 4 precondition." Rationale: D4-01 was adopted
  on partial evidence with the codimension gap explicitly recorded; running `m=4`/`m=8` narrows
  codimension 1 to 8 against PU's ~748 and risks reading as closure rather than narrowing.
- **D4-11:** Re-mint REGN-01, REGN-03, REGN-04 in `REQUIREMENTS.md`; add REGN-06 (see
  `<phase_requirements>` below for the re-minted text). **One-way** — `REQUIREMENTS.md` is the
  project's requirement contract; re-minting after plans cite the IDs forces a plan revision.
- **D4-12:** **No new Swiss roll notebook** for the direction-partition rule (see Project
  Constraints above). Phase 4's record must state this reasoning explicitly.

**Density confound:**
- **D4-13:** Local density is measured in the **ambient 768-d embedding space** — the same space
  the estimator runs in (`pu_curvature_rankability_run.py` applies `centroid_mean_curvature`
  directly to the normalized 768-d `legacysurvey` embeddings, not Isomap coordinates).
- **D4-14:** The density-confound battery is **the REGN-02 correlation only**. No centroid-distance
  check, no partial regression, no density-matched stratification, no density-matched null.
  **Consequence that must carry into the record:** MKNN is itself a k-NN statistic and therefore
  directly density-sensitive, so **without a density-matched null a regional MKNN difference
  cannot be separated from a regional density difference by anything in this phase.** MKNN-07's
  verdict must be worded to reflect this.
- **D4-15:** The headline PU field is **density-corrected**: `density_correct=True`,
  `k_density=30` (pre-registered, no new constant). The `k` sweep **re-runs corrected from k=30
  upward** — the four existing uncorrected `R_H` numbers (k=30/60/120/231) are **superseded, not
  extended**. Budget ~2,100s to reproduce those four, plus k=350,500 on top. The correction's real,
  measured effect is a ~8–10% reduction in median relative error on a genuinely curved,
  strongly-skewed fixture (its earlier flat-fixture "provably inert" framing was retracted in
  `02.5-02-SUMMARY.md` — it is inert on a flat fixture as a mathematical certainty, which is a
  different and weaker claim).

**MKNN mechanics and statistical budget:**
- **D4-16:** Per-region MKNN k-NN sets are computed **within the region's own index set** — subset
  both embeddings to the region's rows, then compute `N_k` inside that subset. Score and
  permutation null then live in the same index set (satisfies MKNN-04 by construction). Accepted
  consequence: the regional number is not directly comparable to the global MKNN-02 figure because
  `k/n` differs; the region's own null absorbs that.
- **D4-17:** **1,000 permutations and 1,000 bootstrap resamples** per cell (2 regions × 4 values of
  `k` = 8 cells). Resolves `p` to ~0.001. **The permutation only shuffles row correspondence, so
  the per-modality k-NN index is built ONCE per cell and reused across all permutations** — the
  planner must implement it that way; this is what makes the budget affordable.
- **D4-18:** The MKNN implementation's check is **MKNN-02's global reproduction on real data**. No
  `tests/test_mknn.py` is added, despite the package's one-test-file-per-module convention. Landing
  near the paper's published crossmodal range is treated as a stronger end-to-end check than
  synthetic unit tests, and MKNN-02 is a requirement regardless.
- **D4-19:** MKNN-02 reports the **raw MKNN alongside the `k/n` chance floor at our `n`, and the
  paper's raw range alongside their `n`** (10,000 vs 101,725). The ratio-over-chance carries the
  comparison, not the raw number. No subsample-size sensitivity curve, no full-config
  101,725-row pass.

**Ordering constraint (from ROADMAP.md, restated because it governs the whole phase):**
Pre-specify the split, then compute. All upstream hyperparameters and the partition threshold must
be frozen using upstream-only diagnostics from Phases 1-3 *before* the first regional MKNN number
is computed — a garden-of-forking-paths guard against post-hoc tuning on a headline effect with
thin statistical headroom (0.4-2% in the origin paper).

### Claude's Discretion

Raised and deliberately left to the planner; this document's Common Pitfalls / Code Examples /
Architecture Patterns sections give concrete, prescriptive recommendations for each — see the
cross-references inline.

- **MKNN-07's verdict rule** (2 regions × 4 k). Hard constraint: written into the notebook before
  the first regional MKNN number is computed, ordering visible. → See Architecture Patterns,
  "Pattern: pre-registered verdict rule."
- **Near-zero `‖H‖` points.** Exclusion policy/threshold is the planner's call, declared before
  freezing, excluded count reported. → See Common Pitfalls, "Pitfall: near-zero `‖H‖` direction is
  undefined."
- **Unbalanced regions.** Behaviour on a badly unbalanced split. → See Common Pitfalls, "Pitfall:
  a k-NN statistic computed on too few points is not a measurement."
- **Whether `v` is computed on all 10k points** or a subset. → Recommend: all points surviving the
  near-zero exclusion; no principled reason to subsample further (see Architecture Patterns).
- **Field computation scope, `n_anchor` protocol at new `k`, seed policy, whether `d=20` is
  re-derived.** → Recommend: full 10k rows, `n_anchor=1000` (existing precedent), the existing
  runner's seed for continuity, `d=20` kept explicit per D-07 (never re-derived — see Architecture
  Patterns, "Recommended Project Structure").
- **Correlation statistic for REGN-02** — Spearman on density vs `‖H‖`, vs the signed projection
  `<H_i/‖H_i‖, v>`, or both. → Recommend: **both**, plain Spearman (no partial regression per
  D4-14) — see Common Pitfalls, "Pitfall: REGN-02 answers only half the confound question."
- **Whether density is compared between the two regions post-split.** → Recommend: yes, report
  median/IQR per region plus a Mann-Whitney U rank-sum test (`scipy.stats.mannwhitneyu`) — cheap,
  standard, and CONTEXT.md's own text calls this "the single most decision-relevant density number
  the phase can report" given D4-14's declined controls.
- **Exact `k` grid past 231** beyond the `350, 500` floor. → Recommend: run `k=350` first; apply
  D4-07's freeze rule immediately; only run `k=500` if the rule has not fired yet. Sequential, not
  batch — saves compute if 350 already clears.
- **MKNN-08's hubness caveat** — stated only, or substantiated with a hubness statistic. →
  Recommend: substantiate, via k-occurrence skewness (Radovanović et al. 2010), computed at zero
  extra k-NN cost from the same membership matrix MKNN itself needs — see Code Examples.
- **Shipped artifact shape** — notebook only, or notebook + runner. → Recommend: runner + JSONL
  cache + notebook, following every prior phase's established pattern (see Architecture Patterns).

### Deferred Ideas (OUT OF SCOPE)

- **Cross-estimator agreement on PU** (D4-02 Amendment 02's mitigation): run the CAE chart decoder's
  `H` field alongside `centroid_mean_curvature` at the frozen `k`, report rank agreement. Declined
  for Phase 4 (D4-08); available later without relitigating D4-03.
- **Codimension-gap narrowing** via `make_multinormal_ridge_control` at `m=4,8`. Declined (D4-10).
- **Density-matched null / partial regression / centroid-distance checks.** Declined (D4-14).
- **`tests/test_mknn.py`** with exact known answers. Declined (D4-18).
- **Full-config 101,725-row global MKNN.** Declined (D4-19).
- **Intramodal MKNN across a model-size ladder** (SCALE-01, PROJECT.md). Out of v1.1 scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

Per D4-11, REGN-01/03/04 are **re-minted** below (not the stale `REQUIREMENTS.md` text, which
still describes the superseded `|H|`-quantile route); REGN-06 is **new**. REGN-02/05 and all
MKNN-01..08 are unchanged. Editing `REQUIREMENTS.md` itself is the planner's job (D4-11 code_context:
"a planning-time task"), not this research pass's.

| ID | Description (re-minted where marked) | Research Support |
|----|----------|------------------|
| REGN-01 *(re-minted)* | Local sample-density measure per point **in the ambient 768-d embedding space the curvature field is estimated in**, shown | `curvature_probe.local_density_weights(X, k_density=30, d=20)` on `subsample["legacysurvey"]`, already implemented and pre-registered (D4-13) |
| REGN-02 | Correlation between local density and curvature reported explicitly, before any region split is trusted | Plain `scipy.stats.spearmanr` against both `‖H‖` and the signed projection `<H_i/‖H_i‖, v>` (D4-14: correlation only, no partial regression) |
| REGN-03 *(re-minted)* | Points partitioned by a **data-derived direction criterion**, never a fixed absolute threshold | D4-09's sign split on `Cov(H_i/‖H_i‖)`'s top eigenvector — a diametrical-clustering split (Dhillon, Marcotte & Roshan 2003), data-derived by construction |
| REGN-04 *(re-minted)* | The partition rule specified and frozen **before** regional alignment is computed, ordering visible in the notebook | Cell-index assertion pattern already established (02.2 CAE pre-registration precedent, `git merge-base --is-ancestor` style ordering proof) |
| REGN-05 | Each region's point count shown | `np.bincount` on the sign-split labels, reported alongside the near-zero-exclusion count |
| REGN-06 *(new)* | The eigenvector `v` and the resulting sign split recorded and frozen as artifacts before any MKNN number is computed | `cache.npz_cache` — the same config-hash-keyed artifact pattern every prior phase uses for frozen fits |
| MKNN-01 | MKNN score as k-normalized k-NN intersection size, matching the origin paper | `MKNN(z1,z2) = k^-1 |N_k(z1) ∩ N_k(z2)|`, verified against arXiv:2509.19453's own PDF text |
| MKNN-02 | Global crossmodal HSC-vs-Legacy-Survey MKNN reproduced and compared against the origin paper's published range | Paper's own Table 2, Legacy-vs-HSC column, range 0.34%–2.25% at n≈102k — read directly from the PDF |
| MKNN-03 | Per-region MKNN score for high/low-curvature regions shown | Same `mknn_score` call, subset to each region's index set (D4-16) |
| MKNN-04 | Each region gets its own permutation null, computed within that region's index set | `scipy.stats.permutation_test`, `permutation_type="pairings"`, on the region's own precomputed neighbour-membership matrices |
| MKNN-05 | Bootstrap CIs on every regional MKNN score | `scipy.stats.bootstrap`, `method="percentile"`, on the region's own per-point overlap fractions |
| MKNN-06 | Whether the high-vs-low result holds across k=5,10,20,50 shown | Loop the region×k grid (8 cells), same membership-matrix construction per cell |
| MKNN-07 | Explicit verdict on whether the regional difference is distinguishable from noise; "no detectable difference" is a valid outcome | Pre-registered verdict rule (see Architecture Patterns), worded per D4-14's density-confound caveat |
| MKNN-08 | Hubness caveat for k-NN-based alignment metrics stated alongside results | k-occurrence skewness (Radovanović, Nanopoulos & Ivanović, JMLR 2010), computed per region/side at zero extra k-NN cost |
</phase_requirements>

## Architectural Responsibility Map

This is a single-tier, notebook-scoped numerical-analysis phase — there is no browser/server/API
split to reason about. The map below is by **module responsibility** instead, which is the
analogous boundary question for this codebase.

| Capability | Primary Owner | Secondary Owner | Rationale |
|------------|-------------|----------------|-----------|
| Curvature field estimation | `curvature_probe.py` (sealed, reused) | — | Already implements `centroid_mean_curvature`, `local_density_weights`; Phase 4 must not edit it |
| `k` freeze / split-half reliability | `cross_split_curvature.py` (sealed, reused) + `pu_curvature_rankability_run.py` (extended) | — | D4-06/D4-07 extend the existing runner's `--k` sweep, not new machinery |
| Direction partition (sign split) | **new** `notebooks/pu_manifold/mknn.py` or a small new module (planner's call) | `cache.py` (artifact freezing, REGN-06) | No existing home for D4-09's eigenvector-sign-split logic |
| Density-confound correlation | Composition of `local_density_weights` + `scipy.stats.spearmanr` | — | REGN-02 is a report, not a new estimator |
| MKNN score / null / CI | `mknn.py`'s three stubs (**the only new implementation surface**) | `sklearn.neighbors.NearestNeighbors`, `scipy.stats` | This is where all new statistical code lives |
| Orchestration, caching, plotting | new `notebooks/diagnostics/<name>_run.py` + `notebooks/04_*.ipynb` | `cache.py` (JSONL/npz pattern) | Matches every prior phase's runner+cache+notebook pattern |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.5.1 (pinned, `requirements-notebooks.txt`) | Array ops, boolean membership matrices | Already the substrate for every module this phase reuses |
| scipy | 1.18.0 (pinned) | `stats.permutation_test`, `stats.bootstrap`, `stats.spearmanr`, `stats.skew`, `stats.mannwhitneyu` | `stats.permutation_test`/`stats.bootstrap` already the codebase's Don't-Hand-Roll choice (`curvature_probe.permutation_null`'s own docstring names this) |
| scikit-learn | 1.9.0 (pinned) | `neighbors.NearestNeighbors` for k-NN sets | Exactly the class `curvature_probe.py`'s estimators already use throughout |

**No new package is required.** All three are already installed, pinned, and exercised by sealed
modules this phase imports.

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| faiss-cpu | 1.14.3 (pinned, currently unused) | Approximate/exact k-NN at scale | Only if `NearestNeighbors` proves too slow on a 10k×768 brute-force query — not expected to be needed (see Common Pitfalls) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `sklearn.neighbors.NearestNeighbors(algorithm="brute")` | `faiss-cpu` (already pinned, `mknn.py`'s own docstring explicitly permits it as long as it's not a module-level import) | faiss is faster at larger scale, but at `n<=10,000`, `D=768` brute-force sklearn (already the pattern every other estimator in this package uses) is simplest and consistent; introduce faiss only if a timing probe shows it's needed |
| `scipy.stats.bootstrap(method="percentile")` | `method="BCa"` (scipy's own default) | BCa needs a jackknife pass (extra `O(n)` calls) and is the more "correct" interval for skewed statistics, but D4-17's own text says "stable percentile CIs" — percentile is simpler, matches the stated intent, and is standard for a bounded `[0,1]` mean-of-indicators statistic at `n_resamples=1000` |
| Diametrical sign-split (D4-09, locked) | Spherical k-means (`k=2`) on `H/‖H‖` | Locked out already — D4-09 rejected k-means for its seed/init sensitivity; diametrical sign-split is deterministic |

**Installation:** none required — no new packages.

**Version verification:** `requirements-notebooks.txt` header states these versions were pinned
from the actual Phase 1 tracer run (`numpy==2.5.1`, `scipy==1.18.0`, `scikit-learn==1.9.0`,
`faiss-cpu==1.14.3`), and the running environment was checked directly: `scipy.__version__ ==
"1.18.0"` [VERIFIED: local venv]. All three APIs used here (`permutation_test`, `bootstrap`,
`NearestNeighbors`) are present and stable at these versions — `scipy.stats.bootstrap`'s signature
was checked directly via `help()` in this session [VERIFIED: local venv] and confirmed to support
`method="percentile"`, `n_resamples`, `rng`.

## Package Legitimacy Audit

**Not applicable — Phase 4 introduces no new external package dependencies.** Every function this
phase needs (`sklearn.neighbors.NearestNeighbors`, `scipy.stats.permutation_test`,
`scipy.stats.bootstrap`, `scipy.stats.spearmanr`, `scipy.stats.skew`, `scipy.stats.mannwhitneyu`,
`numpy` array/boolean operations) is provided by packages already pinned in
`requirements-notebooks.txt` and already exercised by sealed modules this phase imports
(`curvature_probe.py` already imports `scipy.stats.permutation_test` and `scipy.stats.spearmanr`;
`sklearn.neighbors.NearestNeighbors` is used throughout). No `pip install`, `npm install`, or
registry lookup is required for this phase.

## Architecture Patterns

### System Architecture Diagram

```
subsample_*.npz (hsc, legacysurvey — 10k rows, row-aligned, L2-normalized)
        │
        ▼
[1] CURVATURE FIELD (sealed, reused)
    centroid_mean_curvature(legacysurvey, k, d=20,
                             density_correct=True, k_density=30)
    -- extends pu_curvature_rankability_run.py's existing --k sweep
    -- runs k = 30,60,120,231 (re-run corrected, D4-15) then 350, [500]
        │
        ▼
[2] k-FREEZE (sealed, reused)
    cross_split_curvature.reliability_summary(R_H)
    -- freeze at smallest k where ΔmedianR_H < +0.03 AND R_H >= 0.5 (D4-07)
        │
        ▼
[3] NEAR-ZERO EXCLUSION  (new, small)
    drop points with ‖H‖ below a within-config low percentile
    -- mirrors the CURV-04 percentile-exclusion precedent (never absolute)
        │
        ▼
[4] DIRECTION PARTITION  (new — D4-09, the diametrical sign-split)
    v = top eigenvector of Cov(H_i/‖H_i‖)
    label_i = sign(<H_i/‖H_i‖, v>)
        │
        ├──> REGN-06: freeze {v, labels, excluded_idx} via cache.npz_cache
        │
        ▼
[5] DENSITY DIAGNOSTICS (composition of sealed local_density_weights + scipy.stats)
    REGN-01: rho_i in ambient 768-d space
    REGN-02: Spearman(rho, ‖H‖) AND Spearman(rho, signed projection)
    (discretion) region-level density comparison: Mann-Whitney U
        │
        ▼
[6] MKNN  (new — mknn.py's three stubs, the phase's real deliverable)
    for region in {A, B}:
      for k_mknn in {5, 10, 20, 50}:
        subset hsc[region_idx], legacysurvey[region_idx]         (D4-16)
        N_k(hsc)_i, N_k(ls)_i  via NearestNeighbors (self excluded)
        build (n_region, n_region) boolean membership matrices ONCE
        mknn_score      = mean_i |N_k(hsc)_i ∩ N_k(ls)_i| / k
        permutation_null = scipy.stats.permutation_test, reusing the
                            SAME membership matrices across 1,000 resamples
        bootstrap_ci     = scipy.stats.bootstrap over per-point overlap
                            fractions, 1,000 resamples, method="percentile"
        hubness          = k-occurrence skewness, same matrices, free
        │
        ▼
[7] GLOBAL MKNN (D4-19)
    same mechanics, whole 10k cloud, no region subsetting
    report raw score + k/n chance floor + paper's raw range + their n
        │
        ▼
[8] PRE-REGISTERED VERDICT (MKNN-07)
    written BEFORE step [6]'s numbers exist — see Pattern below
        │
        ▼
notebooks/04_region_partition_mknn.ipynb  (read-out, plots, verdict)
notebooks/diagnostics/region_partition_mknn_run.py  (JSONL cache, --smoke)
```

A reader can trace the primary use case (does regional MKNN differ by curvature side) from the raw
`.npz` input through the frozen partition to the final printed verdict by following the arrows
above; every arrow left of step [6] is either fully sealed code or a small, well-precedented
composition, and step [6] is the phase's actual new work.

### Recommended Project Structure

```
notebooks/pu_manifold/
├── mknn.py                        # fill in the three stubs; add a small
│                                   #   `region_partition(...)` helper here or in
│                                   #   a new module (planner's call — no
│                                   #   existing home per code_context)
notebooks/diagnostics/
├── region_partition_mknn_run.py   # new: runs the full [1]-[7] pipeline,
│                                   #   JSONL-caches per-cell results, --smoke mode
notebooks/
├── 04_region_partition_mknn.ipynb # new: loads the runner's cache, plots,
│                                   #   prints the pre-registered verdict
```

No new Swiss roll notebook (D4-12, stated explicitly in the notebook per the locked decision).

### Pattern: reuse the k-NN index across every permutation (D4-17's explicit requirement)

**What:** Build one dense boolean `(n_region, n_region)` neighbour-membership matrix per
embedding side, per cell, and reuse it for both the observed statistic and all 1,000 permutations.
**When to use:** Every region×k cell in step [6] above — this is not optional, it is what makes
the 8-cell × 1,000-permutation × 1,000-bootstrap budget affordable.
**Example:**
```python
# Source: this document's own derivation from D4-17's stated requirement — no external
# reference implements this exact vectorization; the general technique (avoid recomputing
# a k-NN index inside a resampling loop) is standard practice, not novel.
import numpy as np
from sklearn.neighbors import NearestNeighbors

def _membership_matrix(Z: np.ndarray, k: int) -> np.ndarray:
    """(n, n) boolean matrix; row i marks i's k nearest neighbours, self excluded."""
    n = Z.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(Z)
    _, idx = nbrs.kneighbors(Z)          # idx[:, 0] is the point itself
    M = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    M[rows, idx[:, 1:].ravel()] = True
    return M

def mknn_score(z1: np.ndarray, z2: np.ndarray, k: int) -> float:
    A = _membership_matrix(z1, k)
    B = _membership_matrix(z2, k)
    return float((A & B).sum(axis=1).mean() / k)

def permutation_null(z1, z2, k, n_permutations, seed):
    A = _membership_matrix(z1, k)          # built ONCE
    B = _membership_matrix(z2, k)          # built ONCE
    n = A.shape[0]
    observed = float((A & B).sum(axis=1).mean() / k)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        null[i] = (A & B[perm]).sum() / k / n     # reuses A, B -- no re-query
    return {"observed": observed, "null_distribution": null}
```
The observed statistic and every permutation reuse `A` and `B`; only the row-permutation of `B`
changes. At `n_region <= 9,500` this is a `~90M`-boolean-element `&`+`sum` per permutation, which
numpy executes at native speed — well inside the affordability D4-17 requires. (Prefer
`scipy.stats.permutation_test` over the raw loop above for the production implementation — see
next pattern — this snippet exists to show *why* the membership-matrix precompute is the key move.)

### Pattern: don't hand-roll the permutation test or the bootstrap

**What:** Wrap the membership-matrix statistic above in `scipy.stats.permutation_test` and
`scipy.stats.bootstrap` rather than writing the resampling loop by hand.
**When to use:** `mknn.permutation_null` and `mknn.bootstrap_ci`.
**Example:**
```python
# Source: scipy.stats API, version-checked in this session (scipy==1.18.0). Pattern mirrors
# curvature_probe.permutation_null, which already uses permutation_test this same way and
# explicitly warns against a "hand-rolled loop" for exactly this reason.
from scipy.stats import permutation_test, bootstrap

def permutation_null(z1, z2, k, n_permutations, seed):
    A, B = _membership_matrix(z1, k), _membership_matrix(z2, k)
    n = A.shape[0]

    def _stat(idx1, idx2):
        # permutation_test's "pairings" mode permutes BOTH arguments independently;
        # treat them purely as index arrays into the precomputed A/B, never assume
        # either is the caller's original unshuffled array (same discipline
        # curvature_probe.permutation_null's own docstring states for statistic_fn).
        idx1 = idx1.astype(np.int64)
        idx2 = idx2.astype(np.int64)
        return float((A[idx1] & B[idx2]).sum(axis=1).mean() / k)

    rng = np.random.default_rng(seed)
    result = permutation_test(
        (np.arange(n), np.arange(n)), _stat,
        permutation_type="pairings", alternative="greater",
        n_resamples=n_permutations, rng=rng,
    )
    return {
        "observed_score": float(result.statistic),
        "null_distribution": result.null_distribution,
        "p_value": float(result.pvalue),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
    }

def bootstrap_ci(z1, z2, k, n_resamples, seed):
    A, B = _membership_matrix(z1, k), _membership_matrix(z2, k)
    per_point = (A & B).sum(axis=1) / k          # (n,) -- the resampling unit is the POINT
    rng = np.random.default_rng(seed)
    res = bootstrap((per_point,), np.mean, method="percentile",
                     n_resamples=n_resamples, confidence_level=0.95, rng=rng)
    return {
        "score": float(per_point.mean()),
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }
```
**Why the bootstrap resamples points, not permutations of `B`:** the bootstrap answers "how much
would this mean vary under resampling of the *population of points*", which is a claim about
sampling variability, not about chance pairing (that is the permutation null's job). Resampling
`per_point` (an `(n,)` array of per-point overlap fractions, computed once) is the standard
nonparametric bootstrap for a population mean and needs no re-query of the k-NN index at all —
even cheaper than the permutation null.

### Pattern: pre-registered verdict rule (MKNN-07)

**What:** Write the exact rule for "the high-vs-low result holds" into the notebook, in a cell
that executes and asserts its own position **before** the cell that computes any regional MKNN
number — mirroring the 02.2 CAE pre-registration precedent (`git merge-base --is-ancestor` /
cell-index-ordering discipline).
**When to use:** Before step [6] in the pipeline diagram runs for real (a `--smoke` dry run is fine
before the rule is written; the real numbers must not exist yet).
**Recommended rule (this document's own proposal — flagged `[ASSUMED]`, no literature precedent
for this exact multiplicity shape exists; the planner/discuss-phase should confirm it, not accept
it uncritically):**
```python
# Cell N: PRE-REGISTRATION -- run and print BEFORE any regional MKNN number exists.
HEADLINE_K = 20                 # one pre-designated k; the others are sensitivity, not
                                 # independent tests requiring separate multiplicity correction
K_GRID = [5, 10, 20, 50]
NULL_QUANTILE = 0.99            # matches the codebase's existing permutation-null convention
                                 # in curvature_probe.permutation_null
VERDICT_RULE = (
    "A region-pair is DISTINGUISHABLE at k if its bootstrap CI does not overlap the OTHER "
    "region's bootstrap CI at the same k, AND the higher region's score clears its own "
    "permutation null at the 99th percentile. The headline call is made at k=20; k in "
    "{5,10,50} are reported as sensitivity checks, not independently corrected. "
    "'No detectable difference' at the headline k is a complete, valid outcome and is not "
    "escalated by a majority vote across k."
)
print(VERDICT_RULE)
assert "HEADLINE_K" in dir()   # trivial but real ordering marker for a later ancestry check
```
**Why no formal multiplicity correction across the 2×4 grid:** the four `k` values are not
independent trials (they are a nested sensitivity sweep on the same two regions and the same
underlying embeddings), so a Bonferroni-style correction across all 8 cells would be
overly conservative and is not standard practice for a robustness sweep. Treating one
pre-designated `k` as the headline test and the rest as corroborating sensitivity — one of the
two options CONTEXT.md's own discretion list offers — is the more defensible reading and is what
this document recommends, but this is a judgment call, not a citation.

### Pattern: diametrical clustering (naming D4-09's already-locked method)

**What:** D4-09's "sign split on the top eigenvector of `Cov(H_i/‖H_i‖)`" is precisely
**diametrical clustering** for `k=2` clusters — the algorithm from Dhillon, Marcotte & Roshan,
*"Diametrical clustering for identifying anti-correlated gene clusters"*, Bioinformatics 19(13),
2003 [CITED: web search result, not independently read in full — the algorithmic description
(cluster unit-norm vectors into antipodal groups via the leading eigenvector of the cluster's own
covariance/correlation matrix) is corroborated by multiple independent secondary sources found in
this session]. For exactly two clusters, one iteration of that algorithm's assignment step *is*
D4-09's sign split — there is no further EM-style refinement needed at `k=2` since the "cluster
covariance" starts as the whole-field covariance and a single eigenvector split already partitions
into two antipodal cones.
**When to use:** D4-09 is locked; this is naming, not a new recommendation. Useful for the
planner's own documentation/citations and for anyone auditing whether the method has literature
grounding (it does).
**Confidence:** MEDIUM — the correspondence is exact by construction (both are "assign by sign of
projection onto the leading eigenvector of the unit-vector covariance"), but the citation itself
was not read past its abstract/secondary summaries this session.

### Anti-Patterns to Avoid

- **Recomputing the k-NN index inside the permutation loop.** This is exactly what D4-17 forbids
  and what would make the 1,000-permutation × 8-cell budget unaffordable. Always precompute the
  membership matrix (or index arrays) once per cell.
- **Using `scipy.stats.permutation_test`'s default `permutation_type`.** The default is not
  `"pairings"`; passing the wrong `permutation_type` silently computes a different null. Set it
  explicitly, as `curvature_probe.permutation_null` already does.
- **Recomputing the k-NN graph inside the bootstrap loop.** The bootstrap resamples *points*, not
  the pairing; recomputing k-NN on a resampled (with-replacement, hence duplicate-containing) point
  set would change what is being estimated and is far more expensive than necessary. Precompute
  `per_point` once, bootstrap the mean of that fixed array.
- **Applying a partial-regression or density-matched null to REGN-02.** D4-14 explicitly declines
  this (`cross_split_curvature.partial_spearman` exists in this codebase and must NOT be used
  here — it is a partial correlation, which is exactly what D4-14 rules out).
- **Re-deriving `d` from PU's own dimension estimates.** D-07 (Phase 2) bars inheriting a frozen
  dimension by accident; `d=20` must be an explicit, stated call-site argument to
  `centroid_mean_curvature`, matching every prior D4-0x measurement.
- **Quoting the `k=20` decoder-vs-cloud head-to-head numbers as if they compared the same
  fixture as the sealed saddle control.** Already flagged in `03-NOTE-phase-4-decisions.md` —
  not a Phase 4 risk directly, but a citation trap if the planner references those numbers.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Permutation null for a paired statistic | A manual shuffle-and-recompute Python loop | `scipy.stats.permutation_test(..., permutation_type="pairings")` | Already the codebase's own stated Don't-Hand-Roll choice (`curvature_probe.permutation_null`'s docstring); correct pairing semantics, vectorizable via a custom `statistic_fn` |
| Bootstrap CI on a bounded `[0,1]` statistic | A manual resample-and-recompute loop with manually sorted percentiles | `scipy.stats.bootstrap(..., method="percentile")` | Handles percentile-interval construction correctly (symmetric-about-median rule); avoids off-by-one / interpolation bugs in a hand-rolled `np.percentile` call |
| k-NN sets, self excluded | A hand-rolled nearest-neighbour loop or naive `argsort` on a distance matrix | `sklearn.neighbors.NearestNeighbors(n_neighbors=k+1)`, drop column 0 | Exactly the idiom every estimator in `curvature_probe.py` already uses; `n_neighbors=k+1` then dropping self is the established convention, not something to reinvent |
| Hubness/k-occurrence skewness | A hand-rolled in-degree count over a manually built adjacency list | Column-sum of the same boolean membership matrix already built for MKNN (`M.sum(axis=0)`), then `scipy.stats.skew` on that count vector | Zero extra k-NN queries; `scipy.stats.skew` is the standard Fisher-Pearson skewness estimator Radovanović et al. use |
| Two-sample density comparison between regions | A hand-rolled permutation test on the median | `scipy.stats.mannwhitneyu` | Standard, well-understood non-parametric rank-sum test; no distributional assumption needed for a density ratio comparison |

**Key insight:** every piece of statistical machinery this phase needs already has a `scipy.stats`
primitive that this exact codebase has already chosen once (`permutation_test` in
`curvature_probe.py`) or that is the field-standard choice (`bootstrap`, `skew`, `mannwhitneyu`).
The only genuinely new code is the *composition* — building the k-NN membership matrix once and
threading it through both the null and the CI — not any new statistical algorithm.

## Common Pitfalls

### Pitfall 1: REGN-02 answers only half the confound question

**What goes wrong:** Reporting only `Spearman(density, ‖H‖)` addresses the *original* REGN-02
concern (does density fake curvature *magnitude*), but Phase 4 no longer partitions on magnitude
(D4-01) — it partitions on *direction*. A reader could see a clean `Spearman(density, ‖H‖)` number
and wrongly conclude the confound is closed for the partition actually used.
**Why it happens:** REGN-02's original text predates D4-01's pivot; the re-mint (D4-11) fixes the
partition-criterion requirements (REGN-03/04) but REGN-02's own text is unchanged.
**How to avoid:** Report **both** Spearman correlations — density vs `‖H‖` (legacy confound axis)
and density vs the signed projection `<H_i/‖H_i‖, v>` (the axis Phase 4 actually splits on).
**Warning signs:** A notebook cell computing only one correlation number for REGN-02.

### Pitfall 2: near-zero `‖H‖` direction is undefined

**What goes wrong:** `H_i/‖H_i‖` is undefined (or numerically explosive) wherever `‖H_i‖ ≈ 0`. If
these points are silently included in `Cov(H_i/‖H_i‖)`, a handful of near-zero points with noisy,
essentially-random directions can dominate the covariance estimate (each contributes a unit vector
regardless of how small the true signal was) and destabilize `v`.
**Why it happens:** The centroid estimator can return an arbitrarily small `‖H‖` at points where
the true curvature is genuinely near zero (D4-05's own bowl-fixture finding: near-constant
curvature is a real, measured phenomenon, not just a numerical edge case).
**How to avoid:** Mirror the already-established CURV-04 percentile-exclusion pattern (Phase 3):
flag and exclude points below a **within-config low percentile** of the field's own `‖H‖`
distribution — never a fixed absolute magnitude threshold, since curvature scale depends on the
estimator's own run configuration (matches REQUIREMENTS.md's own "Fixed absolute curvature
threshold" prohibition, already written for a different requirement but the same principle
applies). Report the excluded count explicitly (extends REGN-05's spirit).
**Warning signs:** `v`'s direction changing substantially between two otherwise-identical runs
differing only in exclusion policy; a suspiciously large fraction of "outlier" unit vectors near
the covariance's smaller eigenvalues.

### Pitfall 3: a k-NN statistic computed on too few points is not a measurement

**What goes wrong:** If the sign split is badly unbalanced (e.g., 9,400/600), the smaller region's
MKNN at `k=50` uses `k/n ≈ 8.3%` of the region per neighbourhood — a very different regime from the
larger region's `k/n ≈ 0.5%`, and the region's own permutation null and bootstrap CI become
unstable or misleadingly narrow at small `n`.
**Why it happens:** D4-09's sign split has no balance guarantee; it splits by the sign of a
projection, and the field could genuinely be lopsided.
**How to avoid:** Pre-register a minimum region-size floor before freezing the split — a
defensible, stated default is **`n_region >= 10 * k_max`** (`k_max=50` here, so `>=500`), a common
rule-of-thumb multiplier keeping the neighbourhood-to-population ratio reasonable at the largest
evaluated `k` [ASSUMED — this exact multiplier is this document's own reasoned default, not drawn
from a specific paper; confirm at the discuss/plan checkpoint]. Below the floor, report "regional
MKNN undefined for this split" for the affected region/k rather than computing and quoting an
unstable number — this keeps "no detectable difference" a valid, complete outcome rather than
silently computing a number nobody should trust.
**Warning signs:** Region point counts differing by more than roughly an order of magnitude;
bootstrap CIs on the smaller region visibly wider than the larger region's at the same `k`.

### Pitfall 4: the paper's exact `k` and self-neighbour policy are not stated in the visible text

**What goes wrong:** Assuming the origin paper used a specific `k` (this session found no explicit
`k` value in the paper's main text or the visible GitHub excerpt) risks silently mismatching the
comparison basis for MKNN-02's range check.
**Why it happens:** The paper states the formula (`MKNN(z1,z2)=k^-1|N_k(z1)∩N_k(z2)|`) but not the
specific `k` used to produce Table 2's numbers, at least not in the sections this session could
access. A GitHub-fetched code snippet (`mknn(embeddings_1, embeddings_2, k=10)`) surfaced during
this research is **not trustworthy as a verified fact** — the fetch tool itself reported it could
not access the actual source file content, so this may be an AI-summarization artifact rather than
real code. It is **not used** as a claim anywhere in this document.
**How to avoid:** Do not tune Phase 4's `k` grid to match an unverified paper `k`. D4-19 already
handles this correctly — the ratio-over-chance-floor comparison, not raw-number matching, is what
makes the reproduction meaningful regardless of an exact `k` match. One forensic cross-check *is*
available and is reported here for context, not as a hard fact: the paper's own null column
(`π(HSC) vs HSC`, computed at `n≈18.6k`) reads ~0.03–0.05%, which is close to what `k/n` at
`k≈7–10` would predict at that `n` (`10/18600 ≈ 0.054%`) — consistent with, but not proof of, a
`k` in roughly that range. [ASSUMED — inference from the null column, not a stated fact.]
**Warning signs:** A discuss-phase or plan-review treating "the paper used k=X" as settled without
tracing back to this caveat.

### Pitfall 5: hidden `O(n²)` memory blowup on the global (non-regional) MKNN pass

**What goes wrong:** The membership-matrix pattern recommended above costs `O(n_region²)` bytes.
For the **global** MKNN-02 pass (full `n=10,000` cloud), that is `10,000² = 100,000,000` boolean
entries ≈ 100 MB per matrix, ×2 (both embedding sides) ≈ 200 MB transient — manageable on a typical
machine, but worth sizing explicitly rather than discovering it mid-run, especially since D4-19
also considers (and declines) a 101,725-row full-config pass, which at the same pattern would be
`101,725² ≈ 1.03×10^10` entries ≈ 10 GB — infeasible, and consistent with D4-19's decision not to
run it.
**Why it happens:** Dense boolean membership matrices scale quadratically in the region/cloud size.
**How to avoid:** Confirm the 10,000-row global pass's ~200 MB transient footprint is acceptable
(it should be, on any development machine) before running; do not attempt the same pattern at
101,725 rows — this reinforces, rather than merely restates, why D4-19's decision is the right one.
**Warning signs:** A `MemoryError` or severe swapping if anyone later attempts the full-config pass
with this pattern.

## Code Examples

### Membership-matrix build (self excluded, matching this codebase's own idiom)

```python
# Source: this codebase's own established idiom (curvature_probe.centroid_mean_curvature and
# every other estimator in that module query NearestNeighbors(n_neighbors=k+1) and drop the
# self column at idx[:, 0]). No external source needed; this is a direct continuation of an
# existing, verified pattern.
from sklearn.neighbors import NearestNeighbors
import numpy as np

def knn_indices(Z: np.ndarray, k: int) -> np.ndarray:
    """(n, k) neighbour-index array, self excluded."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(Z)
    _, idx = nbrs.kneighbors(Z)
    return idx[:, 1:]     # idx[:, 0] is always the point itself at distance 0
```

### k-occurrence skewness (MKNN-08's substantiation)

```python
# Source: Radovanović, Nanopoulos & Ivanović, "Hubs in Space: Popular Nearest Neighbors in
# High-Dimensional Data", JMLR 2010 [CITED: web search summary of the paper's own definition
# of k-occurrence as the in-degree of the reversed k-NN digraph; not independently read
# in full this session]. scipy.stats.skew implements the standard Fisher-Pearson coefficient.
from scipy.stats import skew

def k_occurrence_skewness(membership_matrix: np.ndarray) -> float:
    """Skewness of the k-occurrence distribution -- how many times each point appears as
    SOMEONE ELSE's nearest neighbour. Computed from the SAME matrix MKNN already builds --
    no extra k-NN query."""
    k_occurrence = membership_matrix.sum(axis=0)   # column sums = in-degree
    return float(skew(k_occurrence))
```

### Region-level density comparison (discretion item, recommended)

```python
# Source: scipy.stats API, standard non-parametric two-sample test.
from scipy.stats import mannwhitneyu

def region_density_comparison(density: np.ndarray, region_labels: np.ndarray) -> dict:
    a, b = density[region_labels == 0], density[region_labels == 1]
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "median_density_region_a": float(np.median(a)),
        "median_density_region_b": float(np.median(b)),
        "mannwhitneyu_statistic": float(stat),
        "mannwhitneyu_pvalue": float(p),
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Hand-rolled shuffle-and-recompute permutation loops | `scipy.stats.permutation_test` with `permutation_type` control | `scipy` 1.7 (2021) introduced `permutation_test` | Already adopted in this codebase (`curvature_probe.permutation_null`); Phase 4 should follow, not diverge |
| Manual percentile bootstrap via sorted resamples | `scipy.stats.bootstrap` (`percentile`/`basic`/`BCa`) | `scipy` 1.7 (2021) | Removes off-by-one/interpolation risk in hand-rolled percentile code |
| Magnitude-quantile curvature partitioning | Direction-clustering (diametrical sign-split) partitioning | D4-01, this milestone, 2026-08-23 | Already locked; not this research's contribution, restated here for completeness |

**Deprecated/outdated:** the phase's own originally-specified `|H|`-quantile partition (superseded
2026-08-23 by D4-01) — noted for the record, not re-litigated.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The origin paper's exact `k` value for Table 2 is in the range implied by its null column (`k≈7–10` at `n≈18.6k`) | Common Pitfalls, Pitfall 4 | Low — this is explicitly not used to set Phase 4's own `k` grid; purely contextual, already caveated |
| A2 | Chechik et al. (2010)'s original metric is not the exact intersection-based MKNN formula; the Platonic Universe paper's attribution is imprecise | Summary | Low — does not affect MKNN-01's implementation, only the accuracy of a citation; worth flagging to the planner so the citation isn't over-claimed |
| A3 | The GitHub-fetched `mknn(embeddings_1, embeddings_2, k=10)` code snippet reflects real repository code | Common Pitfalls, Pitfall 4 | Low — explicitly NOT relied upon anywhere; flagged and discarded as a fetch-tool artifact |
| A4 | A minimum region-size floor of `n_region >= 10 * k_max = 500` is a reasonable pre-registration default | Common Pitfalls, Pitfall 3 | Medium — if too strict, could force "undefined" verdicts on a real, meaningful regional difference; if too loose, could let an unstable small-n statistic through. Planner/discuss-phase should confirm this number explicitly rather than inherit it silently |
| A5 | MKNN-07's verdict rule (one headline `k=20`, others as sensitivity, no formal multiplicity correction across the 2×4 grid) is the right pre-registration shape | Architecture Patterns, "Pattern: pre-registered verdict rule" | Medium — this is a genuine judgment call with no literature precedent found this session; must be ratified at a checkpoint before any regional MKNN number is computed, per the locked ordering constraint |
| A6 | Diametrical clustering (Dhillon, Marcotte & Roshan 2003) is the correct literature name for D4-09's method | Architecture Patterns, "Pattern: diametrical clustering" | Low — a naming/citation claim only; D4-09's actual mechanics are already locked and unaffected either way |

**If this table is empty:** N/A — six assumptions recorded above, all low-to-medium risk, none
touching the phase's already-locked mechanics.

## Open Questions

1. **What `k` did arXiv:2509.19453 actually use for its published crossmodal table?**
   - What we know: the formula and the resulting range (0.34%–2.25% for Legacy-vs-HSC).
   - What's unclear: the exact `k`; not stated in the visible paper text or a reliably-fetched
     code source this session.
   - Recommendation: do not gate Phase 4's own `k` grid on this; D4-19's ratio-over-chance-floor
     framing already sidesteps the need to match it exactly.

2. **Does Chechik et al. (2010) actually define the intersection-based MKNN formula, or is that
   the Platonic Universe authors' own construction attributed loosely?**
   - What we know: Chechik et al. 2010 (OASIS) is primarily about online similarity ranking with a
     "precision at k" retrieval evaluation; the mutual-intersection formula was not confirmed in
     that paper directly this session.
   - What's unclear: whether the attribution in arXiv:2509.19453 is to the general k-NN-based
     evaluation idea or to this exact formula.
   - Recommendation: cite arXiv:2509.19453 as the direct, verified source for MKNN-01's formula
     (which is what matters for implementation correctness); treat the Chechik attribution as the
     paper's own citation choice, not independently re-verified here.

3. **Is `n_region >= 10*k_max` the right region-size floor, and should the phase abort or merely
   caveat an undersized region?**
   - What we know: no literature precedent found for this specific multiplicity/floor question in
     the k-NN-alignment-metric context.
   - What's unclear: whether 500 is too strict or too loose for PU's actual, as-yet-unmeasured
     split balance.
   - Recommendation: treat as a `Claude's Discretion` item requiring explicit ratification, not a
     silently-inherited default — flagged in the Assumptions Log (A4).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| numpy | All array ops | ✓ | 2.5.1 (pinned) | — |
| scipy | permutation_test, bootstrap, spearmanr, skew, mannwhitneyu | ✓ | 1.18.0 (pinned; confirmed live in this session) | — |
| scikit-learn | NearestNeighbors | ✓ | 1.9.0 (pinned) | — |
| faiss-cpu | Not required; available if sklearn proves too slow | ✓ | 1.14.3 (pinned, currently unused anywhere in this package) | sklearn brute-force is the default; faiss is the stated fallback if a timing probe shows a need |
| `notebooks/.cache/subsample_*.npz` | The entire MKNN input (hsc, legacysurvey columns) | ✓ (must be re-verified present at execution time — this research did not re-generate it) | — | None — this is the phase's only external data dependency and it already exists from Phase 1 |

**Missing dependencies with no fallback:** none identified.
**Missing dependencies with fallback:** none currently missing; faiss-cpu is a available-but-unused
fallback for the one dependency (k-NN search) that could in principle need scaling.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.1.1 (pinned, `requirements-notebooks.txt`) |
| Config file | none found under `notebooks/` — tests discovered by pytest's default `test_*.py` convention in `notebooks/pu_manifold/tests/` |
| Quick run command | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_curvature_probe.py notebooks/pu_manifold/tests/test_varying_ii_controls.py notebooks/pu_manifold/tests/test_cross_split_curvature.py -q` |
| Full suite command | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REGN-01/02 | Density/curvature correlation reported | notebook (manual read) | N/A — printed cell output, no automated assertion needed beyond "cell ran" | N/A |
| REGN-03/04/06 | Partition frozen before MKNN computed, artifacts saved | notebook (cell-index ordering assertion) + smoke test on the freeze artifact round-trip | `pytest -k "test_region_partition_roundtrip"` (new, if the planner adds one) | ❌ new, optional |
| REGN-05 | Region point counts shown | notebook (printed) | N/A | N/A |
| MKNN-01 | Formula correctness on a trivial known case (identical embeddings → score 1.0) | unit-adjacent sanity, inline in notebook per D4-18 (no `test_mknn.py`) | manual notebook assertion cell | ❌ — deliberately not a pytest file, per D4-18 |
| MKNN-02 | Global reproduction against paper's range | integration, notebook | manual read-out against 0.34%–2.25% (this document's verified range) | ✅ Wave 0 — the runner IS the test, per D4-18 |
| MKNN-03..07 | Regional scores, nulls, CIs, verdict | notebook | manual read-out | ✅ — no gap, D4-18 explicitly accepts this |
| MKNN-08 | Hubness statistic reported | notebook | manual read-out | ✅ — no gap |

### Sampling Rate

- **Per task commit:** quick run command above (existing estimator/fixture tests only — no
  `test_mknn.py` exists to run per D4-18).
- **Per wave merge:** full suite command.
- **Phase gate:** MKNN-02's global reproduction against the verified 0.34%–2.25% range IS the
  phase's end-to-end check (D4-18) — full suite green plus that reproduction, before
  `/gsd-verify-work`.

### Wave 0 Gaps

**None required for MKNN's new code**, per D4-18's explicit, locked decision not to add
`tests/test_mknn.py`. Existing test files already cover every sealed function this phase reuses:
- `test_curvature_probe.py` — covers `centroid_mean_curvature`, `local_density_weights` (already
  passing, not re-run by this phase per D4-10's "no fixture validation before freezing").
- `test_varying_ii_controls.py` — covers `make_ridge_graph_control`/`make_multinormal_ridge_control`
  (present and passing; **not exercised this phase**, per D4-10).
- `test_cross_split_curvature.py` — covers the `R_H` machinery D4-06/D4-07 extend.

If the planner elects to add a small round-trip test for the new `region_partition`/artifact-freeze
helper (recommended but not required — REGN-06's own artifact IS the audit trail), that would be
the only new test file this phase could reasonably add without contradicting D4-18's spirit (D4-18
is specific to `mknn.py`'s statistical functions, not to the partition-freezing helper).

## Security Domain

**`security_enforcement` is not set in `.planning/config.json`, so it defaults to enabled** — this
section is included per that default, but every ASVS category below is **N/A** for the same reason
the Phase 02.6 code review recorded zero security findings: this phase has **no network surface, no
authentication, no user-input path, and no persistence layer beyond gitignored local cache files**
it reads/writes itself. It reads a pre-existing local `.npz` file and writes local JSONL/npz cache
files under `notebooks/.cache/`, using the same `cache.py` containment guard
(`_assert_inside_cache`) every prior phase already relies on.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth surface — local notebook execution only |
| V3 Session Management | No | No sessions |
| V4 Access Control | No | No multi-user access model |
| V5 Input Validation | Marginal | `cache.py`'s existing `_assert_inside_cache`/`_manifest_matches` guards already cover the one real input-validation concern (path containment for cache reads/writes); no new file-path or network input is introduced |
| V6 Cryptography | No | No secrets, no crypto surface |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via a caller-supplied cache stem | Tampering | Already mitigated by `cache._assert_inside_cache`, reused unchanged (T-01-01 precedent from Phase 1) |
| Stale-cache silent reuse after a config change | Tampering (of results, not security) | Already mitigated by `cache._manifest_matches`'s hard-raise on mismatch, reused unchanged |

## Sources

### Primary (HIGH confidence)

- **arXiv:2509.19453** (Duraphe, Smith, Sourav & Wu, *The Platonic Universe: Do Foundation Models
  See the Same Sky?*, NeurIPS 2025 ML4PS Workshop) — fetched and read directly as a PDF in this
  session. Source of MKNN-01's exact formula, MKNN-02's 0.34%–2.25% Legacy-vs-HSC range at
  n≈102k, and the paper's own null-test convention (`π(HSC)` shuffled-embedding baseline).
  [VERIFIED: fetched arXiv PDF directly]
- **scipy 1.18.0**, live-checked in this session (`scipy.__version__`, `help(scipy.stats.bootstrap)`)
  — confirmed `permutation_test` and `bootstrap` APIs and their `method="percentile"`/
  `permutation_type="pairings"` options exist as used in this document's code examples.
  [VERIFIED: local venv]
- **This codebase's own sealed modules** (`curvature_probe.py`, `varying_ii_controls.py`,
  `cross_split_curvature.py`, `mknn.py`'s stub docstrings, `cache.py`,
  `pu_curvature_rankability_run.py`, `direction_partition_run.py`, `estimator_headtohead_run.py`)
  — read directly in full or in relevant part this session. [VERIFIED: local files]
- **`03-NOTE-phase-4-decisions.md`** (and its two Amendments) and **`spikes/003-fixture-validity-
  audit/README.md`** — read in full this session; the evidentiary basis for D4-01/D4-02/D4-05
  through D4-19. [VERIFIED: local files]

### Secondary (MEDIUM confidence)

- **Radovanović, Nanopoulos & Ivanović**, *"Hubs in Space: Popular Nearest Neighbors in
  High-Dimensional Data"*, JMLR 2010 — grounding for MKNN-08's k-occurrence skewness
  substantiation. [CITED: web search summary; not independently read in full this session]
- **Dhillon, Marcotte & Roshan**, *"Diametrical clustering for identifying anti-correlated gene
  clusters"*, Bioinformatics 19(13), 2003 — literature name for D4-09's already-locked sign-split
  method. [CITED: web search summary; not independently read in full this session]

### Tertiary (LOW confidence)

- A GitHub-fetched code excerpt (`mknn(embeddings_1, embeddings_2, k=10)`) from
  `github.com/UniverseTBD/platonic-universe` — the fetch tool itself reported it could not access
  the real file content, so this snippet is **explicitly not relied upon** anywhere in this
  document's recommendations (see Pitfall 4, Assumption A3). [UNVERIFIED — discarded as evidence]
- The inference that the paper's own null-column values (~0.03–0.05%) are consistent with `k≈7–10`
  at `n≈18.6k` — arithmetic consistency only, not a stated fact from the paper. [ASSUMED]

## Metadata

**Confidence breakdown:**
- MKNN metric definition and published range: HIGH — read directly from the primary source PDF.
- Statistical machinery (permutation test, bootstrap, hubness statistic): HIGH for the scipy
  primitives (version-checked live), MEDIUM for the specific hubness/diametrical-clustering
  literature attributions (secondary-source only).
- Curvature partition mechanics: not independently re-verified by this research pass — inherited
  from locked, already-evidenced decisions (D4-01/D4-02/D4-09 etc.); this document's contribution
  there is consistency-checking and naming, not new verification.
- Density-confound posture: LOW-to-MEDIUM by the phase's own explicit admission — D4-14 accepts
  that MKNN's own k-NN sensitivity to density is not separable from curvature by anything in this
  phase; this is a stated, accepted gap, not a research failure.

**Research date:** 2026-08-23
**Valid until:** 30 days (the MKNN mechanics and scipy APIs are stable; the origin paper is recent
and unlikely to be revised, but a v2 arXiv update is a plausible risk within a few months)
