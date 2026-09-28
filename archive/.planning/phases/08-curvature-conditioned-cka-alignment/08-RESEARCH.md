# Phase 8: Curvature-Conditioned CKA Alignment - Research

**Researched:** 2026-08-27
**Domain:** Centered Kernel Alignment (CKA) / unbiased HSIC estimation over frozen curvature and
crossmodal embedding fields; density-stratified permutation testing; validation-ladder design.
**Confidence:** HIGH on the HSIC/CKA math and on the codebase machinery (all directly read from
source or the primary paper); MEDIUM on numeric tolerances and runtime estimates (derived from
first principles / analogous measured Phase 7 numbers, not measured in this session).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**CKA Estimator**

- **D8-01 — both kernels, linear carries the headline.** Linear CKA is the headline verdict statistic;
  **RBF CKA is reported as robustness and gates nothing** (the D7-03 pattern). Consequence the planner
  must budget for: **the validation ladder runs twice** — each kernel needs its own invariance anchor
  and its own planted-effect control before its number counts.

- **D8-02 — unbiased HSIC.** Song et al. 2012 / Nguyen, Raghu & Kornblith 2021. Removes the `O(1/n)`
  upward bias, so a subset-size drift introduced by density matching **cannot masquerade as a
  curvature gap**. This deliberately decouples the headline from the equal-n mandate rather than
  depending on it. Biased HSIC (the Kornblith 2019 form) was considered and rejected: its bias
  cancels in a difference only under exactly equal n, which would make equal-n load-bearing for
  *correctness* rather than only for comparability.

- **D8-03 — RBF bandwidth is a frozen global constant, computed once per modality.**
  `sigma` = median pairwise Euclidean distance over **all 10,000 points**, computed **separately for
  HSC and for Legacy** (different spaces, no reason to share a scale), frozen into the
  pre-registration **before any subset exists**, and reused unchanged for every subset, every `d`,
  every seed and every `S`. — **Reversibility:** one-way — `sigma` is a pre-registration constant
  under D7-06's freeze-before-any-number discipline; changing it after any number exists requires a
  new pre-registration and a full re-run, as `02.2`'s sealed FAIL and `06-PREREGISTRATION-AMENDMENT-01`
  both establish.

  **Per-subset median heuristic explicitly rejected, and this is the reason:** the high-`||H||`
  subset is measurably denser at `d=20/25` (`spearman(density, ||H||)` = **+0.4281** and **+0.3150**),
  so a per-subset bandwidth would shrink for density reasons and the RBF gap would be a density
  artifact **by construction** — the exact confound this phase is built to exclude.

- **D8-04 — bandwidth sensitivity ladder: `0.5·sigma`, `sigma`, `2·sigma`.** `sigma` carries the
  headline; `0.5x` and `2x` are reported beside it as diagnostics that **gate nothing**. All three
  multipliers go into the freeze commit. Gram matrices are already built, so each extra multiplier
  costs one reduction. **A sign flip across the 4x range means the RBF read is worthless here** —
  cheap to learn, expensive to guess wrong about.

**Split and Density Matching**

- **D8-05 — three `||H||` tertiles**, the Phases 5/6 pattern, so Phase 8's numbers sit beside Phase
  5/6 rows without translation. Two-extremes and median-split shapes were considered and not chosen.
  **Consequence carried into D8-10:** a three-bucket split invites a monotonicity criterion, which is
  precisely what Phase 6 failed on — D8-10 declines that criterion deliberately.

- **D8-06 — within-density-stratum tertile split.** Stratify all 10,000 points into `S` density
  strata; take `||H||` tertiles **within each stratum**; pool across strata. Density marginals are
  then **identical across the three subsets by construction**, and equal-n falls out for free. Reuses
  07.1's `density_stratified_null.density_strata` machinery directly.

  **Semantic consequence that MUST be stated explicitly in `08-FINDINGS.md`, not buried:** the
  tertiles rank **density-residualized curvature**, not raw `||H||`. Caliper matching (unknown
  discard rate, n-drift, another arbitrary constant) and density reweighting (weighted CKA has no
  published bias characterization — a new instrument on top of a new instrument) were both considered
  and rejected.

- **D8-07 — density estimator inherited unchanged.** `curvature_probe.local_density_weights` returns
  the per-point **inverse** density `w`, mean-normalized to 1; the density used throughout is the
  **relative density `1.0 / w`**, matching Phase 4's printed convention (`region_partition_mknn_run.py`
  REGN-01) so Phase 4 / 7 / 07.1 / 8 density numbers stay comparable rather than needing translation.

- **D8-08 — `S` is a threshold grid, not a single inherited constant.** 07.1's `SENSITIVITY_GRID_RULE`
  pattern applies: declare a grid of `S` values, report every point. The developer explicitly chose
  **not** to inherit `N_STRATA_HEADLINE = 20` as a gating constant, because `S=20` was chosen for a
  partial-correlation setting, not for a Gram-matrix statistic.

  **Measured fact the researcher must not re-derive:** `S` does **not** change pooled subset size
  (~3,333 per tertile at any `S`). It trades **density-match tightness against realized `||H||`
  contrast**, because within-stratum tertiles are computed on `n/S` points — large `S` matches density
  tighter but washes out the curvature contrast.

- **D8-09 — NO headline `S`; clearance required at EVERY `S` in the grid.** The verdict fires only if
  the curvature–CKA gap clears its null at every grid point. — **Reversibility:** one-way — this is a
  frozen verdict rule under D7-06; relaxing it after seeing an `S`-dependent result is exactly the
  post-hoc retuning the `k*=15` and `02.2` pre-registrations exist to prevent.

  Chosen deliberately: this phase's whole risk is **manufactured gaps**, so an `S`-dependent gap must
  be **self-reporting as an artifact** rather than something a reader has to notice.

**Null and Verdict Rule**

- **D8-10 — statistic is `CKA(tertile 3) − CKA(tertile 1)`.** The **middle tertile is printed beside
  it as a shape diagnostic and gates nothing.** Monotone-trend and Phase-6-style compound criteria
  were both considered and rejected: **Phase 6 died on a monotonicity criterion while its other two
  criteria held**, and at `n=3` buckets a trend statistic is near-powerless.

- **D8-11 — the null permutes `||H||` tertile LABELS within density strata**, then recomputes the
  entire three-subset CKA panel. Preserves density structure and subset sizes exactly; breaks only the
  curvature link. Direct analogue of 07.1's density-stratified null, and it nulls **the statistic the
  verdict actually reads**.

  Rejected alternatives, with reasons: permuting the **crossmodal row pairing**
  (`mknn.permutation_null`'s `permutation_type="pairings"`) nulls *alignment itself*, answering "is
  there any alignment" — a question **Phase 7 already settled** — not "does alignment differ by
  curvature". **Bootstrap CI** on the difference: CKA is a nonlinear function of the whole subset,
  its bootstrap bias is uncharacterized, and this record has no precedent for it.

- **D8-12 — `d=32` is a REPORTED DIAGNOSTIC that gates nothing, NOT a hard invalidator.**
  **A hard-invalidator reading was offered and explicitly declined by the developer on 2026-08-27.**

  **The tension, recorded so no downstream agent has to rediscover it:** `d=32` has nil
  density–curvature coupling (`spearman(density, ||H||) = +0.0118`, `p=0.238`) *and* no surviving
  07.1 association, so a gap appearing there is prima facie evidence the split machinery manufactures
  gaps. Against that, the phase's premise is that CKA may see what MKNN did not, so a genuine `d=32`
  gap is not impossible. **Consequence the planner MUST handle:** the artifact judgement now rests on
  the reader — which is exactly how Phase 4's result became uninterpretable — so D8-19/D8-20's
  reporting obligations are load-bearing and not optional polish.

- **D8-13 — per-`d` verdicts reported INDEPENDENTLY** (07.1's D-14 pattern). A null at one `d` does
  not silently void another, and **no pooled headline is invented**. Mirrors 07.1's
  `per_d_results = {20: false, 25: true, 32: false}` so Phase 8's table sits beside it row-for-row.

- **D8-14 — fields: frozen `07_crossmodal_curvature_fields.npz` at `d ∈ {20, 25, 32}`, plus 07.1's
  three existing `d=25` seed fields (`TORCH_INIT_SEEDS = 0, 1, 2`).** **No decoder retraining.** The
  seed axis is added only where the record supports it — 07.1 measured `SEED STABLE AT d=25`, 3-of-3,
  and those three decoders already exist. Retraining was named and explicitly ruled out (breaks
  comparability with 07.1's split; Phase 7 measured curvature computation at 1457s at `d=20` scaling
  as `D·d²`).

- **D8-15 — seed combination: unanimous 3-of-3 or nothing**, inheriting 07.1's rule verbatim. Anything
  short of 3-of-3 is **its own reported outcome, NOT upgradable by majority vote**. —
  **Reversibility:** one-way — inherits the ratified `05-03-DECISION.md` constraint below.

  **CARRIES THE ONE-WAY RATIFICATION (`05-03-DECISION.md`, ratified 2026-08-24):** seeds are **never
  pooled** into one averaged `||H||` field. Each seed gets its **own** within-stratum tertile split
  and its **own** verdict. Any `--mode pool` equivalent must raise.

**Validation Ladder**

- **D8-16 — low-`d` anchor is an invariance-property ladder on synthetic pairs.** Generate `Z1`, build
  `Z2` by a transform whose CKA answer is known in advance: **orthogonal rotation and isotropic
  scaling give exactly 1.0 for linear CKA** (Kornblith et al. 2019's defining invariances),
  independent columns give ≈0, and an **additive-noise ladder must decay monotonically** between them.
  A wrong centering, a transposed Gram, or a bad unbiased-HSIC correction each fail this visibly.

- **D8-17 — the CLAUDE.md Swiss roll standing rule is declared NOT APPLICABLE, on purpose.** CKA is
  **not a manifold or representation-learning model** — it has no decoder and no representation map;
  it is a statistic computed over two representations that already exist. The rule's stated purpose
  (tell a broken implementation apart from a real FAIL on data with no known answer) is served here by
  D8-16's invariance ladder, which has an answer known in closed form. **This is a deliberate
  declaration, recorded so the gate is satisfied by decision rather than by omission** — a Swiss roll
  option was presented and not chosen.

- **D8-18 — positive control is an effect-size LADDER on real PU geometry.** Keep PU's actual `||H||`
  field, actual density strata, actual subset sizes; inject a **graded** alignment degradation into
  the high-`||H||` tertile's rows in one modality and **sweep its magnitude**. Reports the **smallest
  gap the test detects — a power curve, not a single pass/fail** — so a null on PU arrives with a
  number saying what it *could* have seen.

  **This directly answers Phase 7's D7-02 lesson:** Phase 6's selfcheck planted a ~20x-spread field
  where PU's realized spread is ~1.5x, and therefore did not serve. A single pre-chosen planted size
  and a fully synthetic coupled fixture were both considered and rejected (the latter because the
  spike record documents a fixture whose *structure alone* moved `rho` from `+0.593` to `+0.150`).

- **D8-19 — negative control: shuffled-`||H||` end-to-end calibration run.** Shuffle the `||H||` field
  across points (**marginal preserved, point correspondence destroyed**), run the **entire** pipeline
  — within-stratum splitting *and* the permutation null — repeated enough times to read a
  **false-positive rate**. Confirms the machinery does not manufacture gaps on its own. This measures
  directly the thing `d=32` was proposed to catch, and matters more given D8-12 made `d=32`
  non-gating.

- **D8-20 — the ladder GATES NOTHING; all three rungs run and are reported beside the verdict.**
  **A hard-gate ordering was offered and explicitly declined by the developer on 2026-08-27.**

  **Recorded concern, stated once and not re-litigated:** with D8-12 and D8-20 both declining to gate,
  **Phase 8 has no structural mechanism preventing an artifact from being written up as a result.**
  Every safeguard is now a *reporting* obligation. The developer's answer to that is D8-21, which is
  therefore not optional.

- **D8-21 — frozen unconditional reporting block AND caveat-bearing verdict text.** Both, not either.
  — **Reversibility:** one-way — the block's contents are frozen under D7-06 before any number exists;
  editing it afterwards to drop an inconvenient row is the failure it is designed to prevent.

  1. **Frozen block:** pre-register the exact set of numbers `08-FINDINGS.md` must print **regardless
     of outcome** — `d=32`'s gap, the shuffled-`||H||` false-positive rate, the planted-effect
     detection floor, realized `||H||` contrast per `S`, and all three `sigma` rungs — each **beside
     the headline, not in an appendix**. 07.1's D-15 (per-`d` table reported unconditionally) is the
     precedent.
  2. **Caveat-bearing verdict text:** the verdict sentence itself **cannot be written without stating
     `d=32`'s gap and the false-positive rate in the same sentence**. This makes it structurally
     impossible to quote a headline without its caveat — which is how Phase 4's number escaped its
     confound.

**Inherited, Non-Negotiable**

- **D8-22 — freeze before any number (D7-06).** Every constant named above — kernels, HSIC form,
  `sigma` and its three multipliers, the `S` grid, the tertile rule, the null construction, the
  verdict and seed-combination rules, and D8-21's reporting block — is committed in a single freeze
  commit, git-ancestry-proved to precede every measured value.

- **D8-23 — additive only.** No sealed module is modified or mutated. Importing the new Phase 8
  module must **never** mutate module-level state in any sealed module (no monkeypatching, no
  attribute assignment onto `mknn`, `cae`, `decoder_curvature`, `curvature_probe`,
  `cross_split_curvature`, `linear_probe`, `pointcloud_probe`, `crossmodal_curvature`,
  `density_stratified_null`), regardless of import order. Carries Phase 7's cross-cutting constraint
  verbatim.

- **D8-24 — `src/effdim/` is untouched** for the whole v1.1 milestone; Phase 8 is notebook-scoped.

### Claude's Discretion

Left to researcher and planner — no user preference expressed, decide from the record:

- The `S` grid's exact values (07.1 used `(10, 20, 50)`; D8-08 does not mandate reuse).
- Permutation count and null RNG seed, and whether they inherit 07.1's constants.
- One- vs two-tailed threshold on the tertile difference (Phase 7 used a two-tailed permutation
  wrapper — inheriting it is the default unless the record argues otherwise).
- Invariance-ladder tolerances, the number of shuffled-`||H||` repeats defining the false-positive
  rate, and the planted-effect ladder's magnitude steps.
- Whether the RBF ladder runs at all three `sigma` rungs on every rung of the validation ladder, or
  only at `sigma` there.
- Module naming/layout, and the runtime budget and wave decomposition.
- How the S-grid axis and the `d=25` seed axis interact (cross product vs. seeds at one `S`).

### Deferred Ideas (OUT OF SCOPE)

- **Per-point local CKA.** Considered as a design option and explicitly not chosen — it would make
  07.1's per-point partial and stratified-null machinery directly reusable, but it is an unsealed
  instrument with small-sample CKA bias and a full validation burden of its own. A future phase, not
  this one.
- **Intramodal CKA across a model-size ladder.** The paper's stronger 28–56% signal. Already deferred
  at project level (`PROJECT.md` → Deferred) — needs a second model size.
- **Promoting any Phase 8 code into `src/effdim/`.** Needs its own test suite and milestone
  (`PROJECT.md` Key Decisions, 2026-07-29).
- **Human ratification of Phase 7 / 07.1 verdicts.** `07.1-FINDINGS.md` states no human sign-off is
  claimed for either verdict; both were produced under standing overnight authorization. Phase 8
  builds on those fields regardless, but the outstanding ratification is a UAT item, not a Phase 8
  task.
</user_constraints>

<phase_requirements>
## Phase Requirements

No milestone-level `REQ-` IDs exist for this phase (`REQUIREMENTS.md` has no Phase 8 row — its
Traceability table stops at Phase 4). Following the 07-CONTEXT.md §3 precedent (Phase 7's own
de-facto requirement set), **`08-CONTEXT.md`'s D8-01..D8-24 are the requirement set the plan must
trace against.**

| ID | Description | Research Support |
|----|-------------|------------------|
| D8-01 | Both kernels; linear headline, RBF non-gating robustness | §1 (HSIC/CKA formula), §2 (invariance ladder for both kernels) |
| D8-02 | Unbiased HSIC (Song 2012 / Nguyen–Raghu–Kornblith 2021) | §1 gives the exact formula, the centering trap, the n-floor |
| D8-03 | Frozen global RBF sigma per modality, computed once over all 10,000 points | §1 confirms sigma parameterization convention; §3 confirms `X_ambient` is d-invariant so sigma needs computing only once ever |
| D8-04 | Sigma sensitivity ladder 0.5x/1x/2x | §5 cost model: Gram-matrix-once pattern makes each extra sigma rung one extra Gram build, not a re-derivation |
| D8-05/06/07/08/09 | Tertile split within density strata, `S` grid, no headline `S` | §3 (`density_strata` exact signature), §5 (S does not change dominant cost) |
| D8-10/11/12/13 | Tertile-3-minus-1 statistic, within-stratum label-permutation null, `d=32` non-gating, independent per-`d` verdicts | §3 (`mknn.permutation_null`'s `pairings` shape, confirmed NOT reused), §6 (recommended permutation count/seed/tail) |
| D8-14/15 | Frozen fields, no retraining, unanimous seed rule | §4 (exact npz paths/keys/shapes) |
| D8-16/17/18/19/20/21 | Validation ladder (invariance, planted-effect, shuffled-`||H||`), non-gating, mandatory reporting | §7 Validation Architecture, §6 tolerance/repeat-count recommendations |
| D8-22/23/24 | Freeze-before-number, additive-only, `src/effdim/` untouched | §3 (import-safety audit of every named module) |
</phase_requirements>

## Summary

Phase 8 adds one new capability to a codebase that has never computed CKA or HSIC: a numpy/scipy
implementation of the **Song et al. (2012) unbiased HSIC estimator**, composed into linear and RBF
CKA, run over the **already-frozen** Phase 7 (`d ∈ {20,25,32}`) and 07.1 (`d=25`, three seeds)
curvature fields and the same HSC/Legacy 768-d embeddings every prior phase has used. No decoder is
retrained, no new package is installed, and no sealed module is edited — this is exactly the
"additive statistical layer over frozen artifacts" shape Phase 7 and 07.1 both used, and the same
freeze-then-run-then-report machinery transfers almost mechanically.

The one genuinely new piece of engineering is the null construction: unlike 07.1's per-point
partial-correlation null, D8-11's null must recompute a **global CKA scalar** on three re-partitioned
subsets, for every permutation. The research finding that determines the entire runtime budget is
that **the pairwise Gram matrices (`hsc` and `legacysurvey`, at each kernel/sigma rung) depend only
on the fixed 768-d ambient embeddings, never on `d`, the seed, or the tertile split** — so they can
be built exactly once (8 total: 2 modalities × 4 kernel variants), and every subsequent
d/seed/S/tertile/permutation operation is a cheap array-indexing + trace computation. Get this one
architectural decision right and the phase runs in minutes-to-tens-of-minutes; get it wrong (rebuild
Gram matrices per subset per permutation) and it does not finish.

The second most important finding is a correctness trap named directly in Kornblith 2019 and
several later reproducibility papers: the unbiased HSIC estimator's `1/(n(n-3))` correction terms
are **only valid on raw, zero-diagonal Gram matrices** — applying the standard double-centering
transform (`H K H`) before running the unbiased formula silently reproduces the *biased* estimator
under a different name, defeating the entire point of D8-02. The invariance ladder (D8-16) is
designed to catch exactly this class of bug, and its acceptance criteria should be written against
closed-form answers (1.0 for orthogonal/isotropic-scaling pairs under linear CKA, ≈0 for
independent columns) rather than approximate ones.

**Primary recommendation:** implement `cka.py` as a small, self-contained numpy module (no new
package), computing raw (uncentered, zero-diagonal) Gram matrices once per modality/kernel/sigma
combination across all 10,000 points, and have every downstream consumer (the tertile-difference
statistic, the label-permutation null, both validation-ladder rungs that touch PU data) index into
those matrices rather than recomputing kernel values.

## Architectural Responsibility Map

This phase has no client/server/database tiers — it is a single-process, notebook-scoped
statistical analysis over local files. The map below assigns each capability to a layer *within*
that single process, which is what the planner needs for correct module boundaries.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Gram matrix construction (linear, RBF) | New `cka.py` module | — | Pure numpy; no dependency on any sealed module's internals beyond raw arrays |
| Unbiased HSIC / CKA composition | New `cka.py` module | — | Self-contained math; must not import scipy's biased `HSIC` helpers (none exist in this codebase; none should be added) |
| Density stratification, tertile split | Reused `density_stratified_null.density_strata` | New Phase 8 module (tertile logic) | `density_strata` already does equal-count rank-quantile binning; tertile-within-stratum is new composition logic, not a new density estimator |
| Density field itself | Reused `curvature_probe.local_density_weights` | — | D8-07 mandates reuse unchanged |
| Label-permutation null construction | New Phase 8 module | Reused `cka.py` for the statistic | Null mechanics are new (07.1's `stratified_partial_null` is the closest analogue but computes a partial correlation, not CKA — cannot be reused directly, only its *shape* copied) |
| Freeze / pre-registration machinery | New Phase 8 module, copying `linear_probe.py`/`density_stratified_null.py`'s pattern | — | Every prior phase re-declares its own constants rather than importing across freeze boundaries (D7-05/D8-08 precedent) |
| Field/data loading (npz reads) | New Phase 8 runner script (`notebooks/diagnostics/08_*_run.py`) | — | Matches `07_crossmodal_curvature_run.py`'s own layout; no I/O inside the pure-function module |
| Reporting notebook | `notebooks/08_*.ipynb` | — | Committed with outputs per CLAUDE.md |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.5.1 (installed, verified `.venv/bin/python -c "import numpy; print(numpy.__version__)"`) [VERIFIED: local venv] | Gram matrix construction, HSIC arithmetic | Already the project's array backend; CKA/HSIC needs nothing beyond matrix multiply, elementwise ops, and reductions |
| scipy | 1.18.0 (installed, verified same way) [VERIFIED: local venv] | `scipy.spatial.distance.pdist`/`squareform` or manual pairwise-distance for RBF Gram, `scipy.stats` utilities for the permutation null's summary stats | Already the project's stats backend; Phase 7/07.1 use `scipy.stats.permutation_test`, `rankdata` from this same package |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (none) | — | — | This phase needs no library beyond numpy/scipy. No `sklearn.metrics.pairwise` dependency is required for linear CKA (a plain `X @ X.T` suffices); RBF Gram is one `pdist`/broadcasting call. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled unbiased-HSIC numpy | `scikit-learn`'s `pairwise_kernels` + a manually written HSIC wrapper | No net simplification — sklearn's `pairwise_kernels` still needs the same manual unbiased-HSIC correction on top; adds a dependency for no algorithmic benefit |
| Hand-rolled unbiased-HSIC numpy | A third-party CKA package (e.g. `torch_cka`, `cca_core`-adjacent packages) | Every such package found in a general search is torch-oriented (built for comparing neural-net activations, not 768-d static embeddings), and none was checked against `package-legitimacy check` in this session — **do not install one**; the phase is ~150 lines of numpy and installing a new dependency for it would violate "KEEP THINGS SIMPLE FIRST" and add an unaudited package to a milestone that has repeatedly rejected SUS installs (02.1's Ollivier-Ricci packages, etc.) |

**Installation:**
```bash
# No new packages required. numpy and scipy are already vetted and pinned
# (notebooks/requirements-notebooks.txt); Phase 8 adds pure-Python/numpy source only.
```

**Version verification:** No new packages, so no registry check is needed. numpy 2.5.1 and scipy
1.18.0 were confirmed present in `.venv` directly (see table above) — both support everything this
phase needs (`np.linalg`, elementwise array ops, `scipy.stats.rankdata`/`permutation_test` if
reused for null-construction idioms).

## Package Legitimacy Audit

**Not applicable — this phase installs no external packages.** All required functionality (matrix
multiplication, pairwise distances, random permutation, quantiles) is available in the numpy/scipy
versions already installed and used by every prior phase in this milestone. No `package-legitimacy
check` run was needed because no `npm view` / `pip index versions` / registry lookup applies to "no
new dependency."

**Packages removed due to `[SLOP]` verdict:** none (none proposed).
**Packages flagged as suspicious `[SUS]`:** none (none proposed).

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │  Frozen inputs (read-only, gitignored cache) │
                    │  07_crossmodal_curvature_fields.npz          │
                    │    h_norm_20, h_norm_25, h_norm_32           │
                    │  07.1_seed_fields_d25.npz                    │
                    │    h_norm_25_seed{0,1,2}                     │
                    │  subsample_*.npz                             │
                    │    hsc (10000,768), legacysurvey (10000,768) │
                    └───────────────────┬───────────────────────────┘
                                        │
                                        ▼
        ┌───────────────────────────────────────────────────────────┐
        │ STEP 1 (once, ever): build 8 full (10000×10000) Gram      │
        │ matrices — {hsc, legacysurvey} × {linear, RBF@0.5σ,       │
        │ RBF@σ, RBF@2σ} — from the fixed 768-d ambient embeddings. │
        │ Zero the diagonal on every one (unbiased-HSIC requirement)│
        └───────────────────────────┬───────────────────────────────┘
                                     │  (kept in memory / cached to disk)
                                     ▼
   ┌────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
   │ STEP 2: density     │   │ STEP 3: per (d,seed)  │   │ STEP 4: validation │
   │ strata + tertile    │──▶│ tertile index sets     │──▶│ ladder (synthetic  │
   │ split, per (d,seed, │   │ → CKA(K,L) via         │   │ pairs / planted    │
   │ S) — indices only,  │   │ submatrix indexing     │   │ effect / shuffled  │
   │ no Gram rebuild     │   │ into Step 1's matrices │   │ H, using the same  │
   └────────────────────┘   └──────────┬─────────────┘   │ Step-1 machinery)  │
                                        │                  └────────────────────┘
                                        ▼
                       ┌────────────────────────────────────┐
                       │ STEP 5: within-stratum tertile-label │
                       │ permutation null (D8-11) — reuses    │
                       │ Step 1's Gram matrices, re-indexes   │
                       │ per permutation, recomputes CKA panel│
                       └───────────────────┬────────────────┘
                                           ▼
                       ┌────────────────────────────────────┐
                       │ STEP 6: per-d/seed verdict, D8-21's  │
                       │ frozen unconditional reporting block,│
                       │ 08-FINDINGS.md, reporting notebook   │
                       └────────────────────────────────────┘
```

### Recommended Project Structure

```
notebooks/pu_manifold/
├── cka.py                    # NEW — HSIC/CKA math, Gram builders, tertile+null helpers
├── density_stratified_null.py  # REUSED read-only — density_strata()
├── curvature_probe.py        # REUSED read-only — local_density_weights()
├── mknn.py                   # REUSED read-only — hubness_skewness, chance_floor (diagnostics only)
├── crossmodal_curvature.py   # REUSED read-only — never edited; D7-07's ALIGNMENT_METRIC constant
│                              #   is superseded BY DECISION, recorded in the plan's
│                              #   <superseded_decision> block, never by patching this file
notebooks/diagnostics/
└── 08_cka_alignment_run.py   # NEW — loads frozen npz fields, drives cka.py, writes JSONL record
notebooks/
└── 08_cka_alignment_check.ipynb   # NEW — reporting notebook, committed with outputs
```

Module name `cka.py` follows the one-word, lowercase, single-purpose convention every prior phase
used (`mknn.py`, `topoae.py`, `cae.py`) rather than a longer `curvature_cka_alignment.py` — the
module's job is exactly "compute CKA," nothing else; splitting logic (tertiles, strata) can live
either in the same file (small phase, matches CLAUDE.md's "keep things simple" instruction) or in a
sibling `cka_split.py` if the planner prefers a physical wave boundary between "the estimator" and
"the phase-specific splitting/null logic." Either is defensible; the estimator itself should not
import anything from a splitting module (unit-testable and reusable in isolation, which the D8-16
invariance ladder needs on synthetic data with no density/split machinery at all).

### Pattern 1: Unbiased HSIC via Song et al. (2012)

**What:** For two `(n, n)` kernel (Gram) matrices `K`, `L`, form zero-diagonal copies `K̃`, `L̃`
(set `K̃_ii = 0`, `L̃_ii = 0`), then:

```
HSIC_1(K, L) = 1/(n(n-3)) * [ tr(K̃ L̃)
                               + (1ᵀ K̃ 1)(1ᵀ L̃ 1) / ((n-1)(n-2))
                               - (2/(n-2)) * 1ᵀ K̃ L̃ 1 ]
```

`CKA(K, L) = HSIC_1(K, L) / sqrt(HSIC_1(K, K) * HSIC_1(L, L))`.

**When to use:** Any time subset sizes vary or are small relative to `n`, which is exactly D8-02's
stated reason — this phase compares CKA across subsets whose size is controlled by `S` (D8-08), so a
biased estimator's `O(1/n)` term would not cancel cleanly across cells of potentially different
realized size.

**Requires `n > 3`** for the `1/(n(n-3))` term to be defined and for `(n-1)(n-2)` and `(n-2)` in the
denominator to be non-zero and positive; the estimator's variance is high at small `n` and Nguyen,
Raghu & Kornblith (2021) recommend minibatches on the order of hundreds to keep it well-behaved.
Phase 8's realized subset sizes (~3,333 per tertile pooled, `n // S` per stratum before pooling —
`n // S` must independently exceed `density_strata`'s own 3-point floor per 07.1's `ValueError`, but
that floor is for the **stratum**, not the tertile-within-stratum subset the CKA statistic is
computed on) are three to four orders of magnitude above the `n > 3` floor and roughly one order of
magnitude above the recommended minibatch-CKA regime — no small-sample concern applies here.
[VERIFIED: Song et al. 2012 formula cross-referenced against Kornblith et al. 2019 Eq. 3 / Nguyen,
Raghu & Kornblith 2021's minibatch-CKA description via WebSearch against openreview.net's hosted PDF
of the Nguyen–Raghu–Kornblith paper]

```python
# Source: Song, Smola, Gretton, Bedo & Borgwardt (2012); as reused in
# Nguyen, Raghu & Kornblith (2021), "Do Wide and Deep Networks Learn the Same Things?" (ICLR 2021)
import numpy as np

def _zero_diag(K: np.ndarray) -> np.ndarray:
    K = K.copy()
    np.fill_diagonal(K, 0.0)
    return K

def unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    if n <= 3:
        raise ValueError(f"unbiased_hsic: n={n} must exceed 3 (Song et al. 2012 floor).")
    Kt, Lt = _zero_diag(K), _zero_diag(L)
    ones = np.ones(n)
    term1 = np.trace(Kt @ Lt)
    term2 = (ones @ Kt @ ones) * (ones @ Lt @ ones) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * (ones @ Kt @ Lt @ ones)
    return float((term1 + term2 - term3) / (n * (n - 3)))

def cka(K: np.ndarray, L: np.ndarray) -> float:
    hsic_kl = unbiased_hsic(K, L)
    hsic_kk = unbiased_hsic(K, K)
    hsic_ll = unbiased_hsic(L, L)
    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)
```

**CRITICAL — do NOT double-center.** `K` and `L` above are RAW Gram matrices with only the diagonal
zeroed — `K = X @ X.T` for linear, `K_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))` for RBF, **never**
`H @ K @ H` (the classical double-centering transform used by the *biased* HSIC/CKA estimator,
`HSIC_0(K,L) = tr(K H L H) / (n-1)^2`). The `1/(n(n-3))` correction terms above already perform the
debiasing; applying them to a pre-centered matrix silently reproduces (a scaled variant of) the
biased estimator under the unbiased formula's name — this is the "classic implementation trap"
D8-16's invariance ladder exists to surface (a wrong centering fails the ladder's closed-form checks
visibly, per D8-16's own stated purpose).

### Pattern 2: Gram-matrix-once, submatrix-index-many

**What:** Build the full `(10000, 10000)` Gram matrix for each `{modality, kernel, sigma}`
combination exactly once. For any subset defined by an index array `idx` (a tertile, a stratum, a
permuted relabeling), the subset's Gram matrix is simply `K[np.ix_(idx, idx)]` — this is exact, not
an approximation, because kernel value `K(x_i, x_j)` depends only on `x_i` and `x_j`, never on which
other points are present in the batch.

**When to use:** Everywhere in this phase. The naive alternative — recomputing `X_subset @
X_subset.T` (or the RBF equivalent) fresh for every tertile, every `S`, every permutation — recomputes
the *same* pairwise kernel values millions of times. See §5/Runtime for the magnitude of the saving.

```python
# Build once, reuse everywhere.
K_hsc_linear = X_hsc @ X_hsc.T                       # (10000, 10000)
K_ls_linear  = X_legacysurvey @ X_legacysurvey.T     # (10000, 10000)
# ... and the RBF variants at 0.5*sigma, sigma, 2*sigma per modality (8 matrices total)

def cka_on_subset(K_full: np.ndarray, L_full: np.ndarray, idx: np.ndarray) -> float:
    K_sub = K_full[np.ix_(idx, idx)]
    L_sub = L_full[np.ix_(idx, idx)]
    return cka(K_sub, L_sub)
```

This pattern is what makes D8-11's null tractable: each of the `N_PERMUTATIONS` resamples only needs
to re-derive the tertile index arrays (cheap integer-array bookkeeping) and re-slice the 8 precomputed
matrices, never re-touch the 768-d embeddings.

### Anti-Patterns to Avoid

- **Recomputing Gram matrices per subset/permutation.** The single biggest risk to phase runtime;
  see the Runtime/Cost Model below. Kernel values are a pure function of the *pair* of points, so any
  code path that reconstitutes `X_subset` and calls `X_subset @ X_subset.T` (or an RBF pairwise-distance
  routine) inside a permutation loop is doing needless O(n_sub² D) work `N_PERMUTATIONS` times over.
- **Per-subset RBF bandwidth.** Already excluded by D8-03, but worth stating as an anti-pattern
  explicitly: `sigma = median(pdist(X_subset))` computed freshly per tertile is the textbook
  "adaptive bandwidth" mistake that would make the RBF gap a density artifact by construction (D8-03's
  own stated reason for rejecting it).
- **Double-centering before the unbiased-HSIC formula.** See Pattern 1's critical note.
- **Silently editing `crossmodal_curvature.py`'s `ALIGNMENT_METRIC = "mknn"` / D7-07 comment.**
  D8-23 forbids editing any sealed module; the supersession of D7-07's scope decision belongs in the
  new plan's `<superseded_decision>` block, exactly as `08-CONTEXT.md`'s own Integration Points
  section states.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Density stratification | A new equal-count quantile-binning routine | `density_stratified_null.density_strata(density, n_strata)` | Already implements exactly D8-06's need (stable-sort rank bins, remainder-to-last-stratum rule) and is unit-tested in `tests/test_density_stratified_null.py` |
| Per-point relative density | A new k-NN density estimator | `curvature_probe.local_density_weights` (invert via `1.0 / w`, per D8-07) | D8-07 mandates reuse unchanged; re-deriving density would introduce a second, uncharacterized density instrument into a phase whose entire premise is that curvature splits ARE partly density splits |
| Random permutation with a reproducible seed | A hand-rolled shuffle | `np.random.default_rng(seed).permutation(idx)` per stratum, mirroring `stratified_partial_null`'s exact idiom (`rng.permutation(idx)` inside a per-stratum loop, one `rng` instance shared across all strata and all resamples) | Reusing the *pattern* (not the code, since the statistic differs) keeps Phase 8's null auditable against 07.1's already-reviewed construction |
| Bootstrap CI on a CKA difference | A custom nonlinear-statistic bootstrap | Nothing — D8-11 explicitly rejects a bootstrap CI here ("CKA is a nonlinear function of the whole subset, its bootstrap bias is uncharacterized, and this record has no precedent for it") | The permutation null is the chosen and sufficient inferential tool; do not add a second one |
| Kernel matrix pairwise distances | A manual double for-loop over point pairs | `scipy.spatial.distance.pdist(X, metric="sqeuclidean")` + `squareform`, or `((X**2).sum(1)[:,None] + (X**2).sum(1)[None,:] - 2*X@X.T)` for the RBF exponent | O(n² D) vectorized computation vs. O(n²) Python-level loop iterations — at n=10,000 the loop form is intractable, the vectorized form takes well under a second |

**Key insight:** every piece of *density* and *split* machinery this phase needs already exists and
is validated by a prior phase (04/05/06/07.1); the only genuinely new code is the CKA/HSIC estimator
itself, which is short, well-specified in the literature, and should be validated by the D8-16
invariance ladder rather than trusted from a from-memory implementation.

## Common Pitfalls

### Pitfall 1: Double-centering before the unbiased-HSIC correction

**What goes wrong:** The resulting statistic is not the unbiased estimator D8-02 requires — it is a
different (and differently biased) quantity that happens to look plausible.
**Why it happens:** Most CKA tutorials and several public reference implementations present the
*biased* estimator (`tr(K H L H)/(n-1)^2`, which DOES require double-centering) side-by-side with the
unbiased one, and it is easy to reuse the centering step across both by habit.
**How to avoid:** Keep two clearly separate code paths / a single flag, and pin the invariant
`unbiased_hsic` is called ONLY on zero-diagonal RAW Gram matrices via a unit test that checks a
double-centered input produces a *different, wrong* answer against a hand-computed reference on a
tiny (n=10 or n=20) synthetic example.
**Warning signs:** CKA between identical or near-identical representations reading noticeably below
1.0 even without noise; the invariance ladder's orthogonal-rotation / isotropic-scaling checks
failing to hit 1.0 within tight tolerance.

### Pitfall 2: Reusing MKNN's or 07.1's permutation-null machinery structurally, but not statistically

**What goes wrong:** `mknn.permutation_null`'s `permutation_type="pairings"` shuffles which row of
modality B is paired with which row of modality A — this is explicitly the null D8-11 does NOT want
(it tests "is there alignment at all," a question Phase 7 already answered). A planner or implementer
skimming the codebase for "the permutation null pattern" could reach for this function by habit and
get the wrong null.
**Why it happens:** It is the only pre-existing crossmodal permutation null in the codebase and its
signature is convenient to reuse.
**How to avoid:** D8-11's null is a **label-shuffle within density strata**, structurally closer to
07.1's `stratified_partial_null` (shuffle a per-point quantity within strata, recompute the statistic)
than to `mknn.permutation_null` — but even `stratified_partial_null` cannot be called directly,
because it computes a partial Spearman correlation, not a CKA difference between three re-partitioned
subsets. The new null function must be written fresh, borrowing only the *stratum-restricted-shuffle*
shape.
**Warning signs:** A null distribution that looks suspiciously like a global MKNN reshuffle result
(near-zero shift with wide spread) rather than one that preserves tertile-size and density-marginal
structure.

### Pitfall 3: Rebuilding Gram matrices inside the permutation/S/d loop

**What goes wrong:** Runtime balloons from minutes to many hours or becomes intractable — see Runtime
section.
**Why it happens:** The natural, "obvious" way to write `cka_for_subset(X, idx)` is to slice `X` first
and then build the Gram matrix from the slice, which is *correct* but re-derives values the full
matrix already has.
**How to avoid:** Build once per {modality, kernel, sigma}, index thereafter (Pattern 2 above).
**Warning signs:** A single permutation-null run taking more than a few seconds per (d, S) cell; CPU
profile dominated by matrix-multiply calls rather than array-indexing calls.

### Pitfall 4: Per-subset RBF sigma (D8-03's named confound, restated as an implementation pitfall)

**What goes wrong:** The RBF ladder (D8-04) becomes uninterpretable because a shrinking sigma on the
denser high-`||H||` tertile mechanically raises pairwise kernel similarity there, manufacturing a CKA
difference that has nothing to do with curvature.
**Why it happens:** "Use the median heuristic" is a very common RBF-CKA default, and it is natural to
compute it fresh on whatever data is in hand (the subset) rather than remembering it was frozen
globally.
**How to avoid:** `sigma` (and its 0.5x/2x siblings) are computed exactly once, over all 10,000 points,
per modality, before any subset exists, and passed as a frozen constant into every Gram-matrix build
— never recomputed at a call site that only sees a subset.
**Warning signs:** RBF sigma values that differ across tertiles or across `S` in any debug output.

### Pitfall 5: CKA computed on rank-transformed data (confusing this phase's statistic with Phase 7's)

**What goes wrong:** Phase 7/07.1's headline statistics are all Spearman-based (rank correlations).
CKA is conventionally computed on **raw** feature values, not ranks — rank-transforming the 768-d
embeddings before building Gram matrices would silently change the statistic into something with no
literature precedent and no comparability to Kornblith et al. 2019's or the Platonic Universe paper's
numbers.
**Why it happens:** Every other statistic in this codebase (Spearman `rho`, `partial_spearman`) works
on ranks, and it would be easy to reuse a `rankdata()` call by habit when adapting 07.1's code shape.
**How to avoid:** CKA's inputs are the raw (L2-normalized, per DATA-04) 768-d embedding matrices,
exactly as loaded from `subsample_*.npz`'s `hsc`/`legacysurvey` arrays — no rank transform anywhere in
the CKA path.
**Warning signs:** A `rankdata` or `scipy.stats.rankdata` call anywhere in `cka.py`.

## Code Examples

### RBF Gram matrix with a frozen global bandwidth

```python
# Source: standard RBF/Gaussian kernel definition, sigma parameterization per
# Kornblith et al. 2019 Section 2 ("we set sigma to a fraction of the median distance
# between examples"); D8-03 fixes that fraction at {0.5, 1.0, 2.0} and freezes sigma
# globally rather than per subset.
import numpy as np
from scipy.spatial.distance import pdist, squareform

def median_pairwise_distance(X: np.ndarray) -> float:
    """D8-03's sigma: computed ONCE, over ALL 10,000 points, before any subset exists."""
    return float(np.median(pdist(X, metric="euclidean")))

def rbf_gram(X: np.ndarray, sigma: float) -> np.ndarray:
    sq_dists = squareform(pdist(X, metric="sqeuclidean"))
    return np.exp(-sq_dists / (2.0 * sigma ** 2))
```

### Within-stratum tertile-label permutation null (D8-11 shape, new to this phase)

```python
# Pattern borrowed from density_stratified_null.stratified_partial_null's per-stratum
# rng.permutation(idx) idiom, but recomputing a CKA panel rather than a partial correlation --
# the statistic itself has no precedent in the codebase and must be written fresh.
def stratified_tertile_label_null(
    h: np.ndarray, strata: np.ndarray, K_full: dict, L_full: dict,
    n_resamples: int, seed: int,
) -> dict:
    """h: per-point ||H|| (curvature magnitude). strata: per-point stratum id from
    density_strata(). K_full/L_full: dict of {kernel_name: (10000,10000) Gram matrix},
    built ONCE outside this function. Permutes ||H|| tertile LABELS within each stratum,
    recomputes tertile index sets, and recomputes CKA(tertile3) - CKA(tertile1) for every
    kernel in K_full/L_full -- the entire panel, per D8-11's own text."""
    rng = np.random.default_rng(seed)
    n = h.shape[0]
    strat_indices = [np.where(strata == s)[0] for s in np.unique(strata)]
    null_by_kernel = {name: np.empty(n_resamples) for name in K_full}
    for b in range(n_resamples):
        h_perm = h.copy()
        for idx in strat_indices:
            h_perm[idx] = h[rng.permutation(idx)]
        tertiles = tertile_split_within_strata(h_perm, strata)  # returns 3 index arrays
        for name in K_full:
            c3 = cka_on_subset(K_full[name], L_full[name], tertiles[2])
            c1 = cka_on_subset(K_full[name], L_full[name], tertiles[0])
            null_by_kernel[name][b] = c3 - c1
    return null_by_kernel
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Biased HSIC / CKA (`tr(K H L H)/(n-1)^2`, Kornblith et al. 2019's original full-batch formula) | Unbiased HSIC (Song et al. 2012, adopted for CKA by Nguyen, Raghu & Kornblith 2021) | Popularized for CKA specifically in ICLR 2021 (minibatch CKA for large-scale network comparison) | Bias no longer depends on `n`, so comparing CKA across differently-sized subsets (exactly this phase's tertiles under varying `S`) is valid without an equal-n assumption carrying the correctness burden |
| CCA / SVCCA for representational similarity | CKA (Kornblith et al. 2019) | ICML 2019 — CKA shown to correctly identify corresponding layers across architectures where CCA-family methods failed | Not directly load-bearing for Phase 8 (CKA is already the chosen metric, D7-07/D8 decision), but explains why CKA rather than CCA was ever on the table as "the second alignment probe" |

**Deprecated/outdated:**
- Full-batch biased CKA as the sole estimator: still valid and widely used when `n` is large and
  fixed across every comparison, but Phase 8's varying-`S` design specifically needs the
  size-invariance the unbiased estimator provides — D8-02 already made this call, no further research
  needed here beyond confirming the exact formula (done above).

## Runtime/Cost Model

**This is the single most planning-relevant number in the phase, per the research brief. The
headline finding: the dominant cost of a naive design (rebuilding Gram matrices per subset per
permutation) collapses almost entirely once Gram matrices are built once and sub-indexed.**

### Why the Gram matrices are `d`-invariant and seed-invariant

CKA's two representations, `z_a` and `z_b`, are the paired-modality 768-d embeddings — `hsc` and
`legacysurvey`, loaded once from `subsample_*.npz` and shared by every phase since Phase 1. **They do
not depend on `d`, on the decoder seed, or on `S`.** `d` and the seed only determine *which points
fall into which tertile* (via the `h_norm_{d}` / `h_norm_25_seed{k}` curvature field used to build the
split). This is confirmed directly from `crossmodal_curvature.py` (Phase 7's own headline statistic
is `spearman(||H||, MKNN)` computed on the SAME fixed `z_a`/`z_b` embeddings across all three `d`
values) and from the npz inspection below.

```
notebooks/.cache/07_crossmodal_curvature_fields.npz:
  h_norm_20, h_norm_25, h_norm_32, h_norm  (each (10000,) float64)
  cond_g_20, cond_g_25, cond_g_32, cond_g  (each (10000,) float64, non-gating diagnostics)
notebooks/.cache/07.1_seed_fields_d25.npz:
  h_norm_25_seed0, h_norm_25_seed1, h_norm_25_seed2  (each (10000,) float64)
```
[VERIFIED: local file inspection via `.venv/bin/python` + `numpy.load`, run in this session]

Both files live under the gitignored `notebooks/.cache/` (confirmed against `.gitignore` line
`notebooks/.cache/`), consistent with every other milestone artifact — a Phase 8 runner script must
load them by relative path from the repo root and **halt, not regenerate**, if either is missing
(the established convention from every prior "frozen cache" phase in this milestone, e.g. Phase
02.1/02.2's halt-not-regenerate precondition on the Isomap cache).

**Consequence:** the full set of Gram matrices needed — 2 modalities (`hsc`, `legacysurvey`) × 4
kernel variants (linear, RBF@0.5σ, RBF@σ, RBF@2σ) = **8 matrices of shape (10000, 10000)** — can be
built exactly once, before the `d`/seed/S loop even starts, and reused for every downstream
computation in the entire phase (headline statistic, sensitivity ladder, permutation null, and the
D8-18 planted-effect / D8-19 shuffled-`||H||` controls, all of which reuse the SAME real PU Gram
matrices and only vary which points are grouped into which subset).

### Cost of building the 8 Gram matrices (one-time)

- **Linear** (2 matrices): `X @ X.T` at `X` shape `(10000, 768)` → O(n² D) = 10000² × 768 ≈ 7.7×10¹♠
  multiply-adds per matrix. With BLAS-backed numpy this is a fraction of a second per matrix on any
  modern CPU (comparable to, e.g., a `(10000,768) @ (768,10000)` matmul, well within numpy's
  optimized GEMM path).
- **RBF** (6 matrices — 3 sigma rungs × 2 modalities): pairwise squared distances via `pdist` /
  broadcasting, same O(n² D) cost as linear, plus a cheap elementwise `exp()` over 10⁸ entries.
  `scipy.spatial.distance.pdist` at n=10,000 returns ~5×10⁷ unique pairs; expect low single-digit
  seconds per matrix on CPU, dominated by the `exp()` call rather than the distance computation.
- **Memory**: each `(10000,10000)` float64 matrix is 800 MB. 8 matrices held simultaneously would be
  ~6.4 GB — likely fine on a modern workstation but worth flagging. **Recommendation:** use
  `float32` for the stored Gram matrices (CKA/HSIC's numerical range at this scale does not need
  float64 precision the way the curvature Jacobian/Hessian work in Phase 3 did — that need was
  specific to catching pathological metric collapse, not applicable here) to halve memory to ~3.2 GB,
  or compute and cache RBF-sigma-rung matrices to disk (`.npy`) one at a time rather than holding all
  8 simultaneously if memory is a concern on the execution machine. Either choice is a planner
  discretion call with a concrete number now available to decide it.

### Cost of the tertile-difference statistic (headline + sensitivity ladder), per (d, seed, S) cell

Given the Gram matrices, computing `CKA(tertile3) - CKA(tertile1)` for one kernel requires: building
tertile index arrays (cheap, O(n)), then two `np.ix_` submatrix extractions (~3,333×3,333 = ~1.1×10⁷
entries each, ~11 MB per submatrix at float64, ~5.5 MB at float32) and three `unbiased_hsic` calls
per tertile-pair comparison (`HSIC(K,L)`, `HSIC(K,K)`, `HSIC(L,L)`), each dominated by a `(3333,3333)
@ (3333,3333)` matrix multiply inside `tr(K̃L̃)` and two matrix-vector products — on the order of
tens of milliseconds each with BLAS. **Total per (d, seed, S, kernel) cell: well under one second.**
Across `D_SWEEP` (3 values, one of which — `d=25` — also runs 3 seeds = 5 total field cells) × 4
kernel variants × an `S`-grid of, say, 3 values = **60 cells**, the headline+ladder computation alone
is on the order of tens of seconds to low single-digit minutes — negligible next to the null.

### Cost of the D8-11 permutation null (the dominant term)

Each of `N_PERMUTATIONS` resamples must: re-permute `||H||` labels within each of `S` strata (cheap),
rebuild tertile index arrays (cheap), then recompute the full CKA panel (as above, tens of
milliseconds per kernel per tertile-pair with the Gram-matrix-once pattern). Recommendation is to run
the null only on the **headline kernel/sigma configuration required to clear the verdict** — linear
CKA (mandatory, headline) plus RBF **at `sigma` only**, not at all three sigma rungs (see Claude's
Discretion recommendation below) — reducing the per-permutation kernel count from 4 to 2.

Per permutation: 2 kernels × (2 submatrix extractions + ~3 HSIC evaluations each) ≈ tens of
milliseconds. At `N_PERMUTATIONS = 1000` (Phase 7/07.1's own convention — see Discretion
recommendation below), that is **~1000 × (tens of ms) ≈ tens of seconds per (d/seed, S) cell.** Across
5 field cells (`d=20`, `d=25`×3 seeds, `d=32`) × 3 `S`-grid values = 15 cells, the null alone is on
the order of **several minutes to ~15-20 minutes total** if each permutation-cell round trip stays
in the tens-of-milliseconds range — the actual figure depends heavily on whether submatrix extraction
via `np.ix_` (which copies data) or a pre-sorted/stride-based indexing scheme is used; `np.ix_` is
simplest and should be tried first, with a fallback to caching each stratum's boolean mask as a
contiguous view if profiling shows indexing overhead dominating.

**Recommendation to keep this tractable:** batch permutations where possible (vectorize the
`rng.permutation` calls across all `N_PERMUTATIONS` resamples up front into a `(N_PERMUTATIONS, n)`
index array, rather than looping in pure Python per resample) — this is the same optimization
`density_stratified_null.stratified_partial_null` already applies at the rank-vector level (it
precomputes ranks once, outside the loop, and only permutes cheap integer/rank arrays inside the
loop). The equivalent optimization here is precomputing all `N_PERMUTATIONS` label-permutation index
sets before entering the CKA-recomputation loop, so the loop body is pure array indexing plus HSIC
arithmetic with no further RNG calls.

### Cost of D8-18 (planted-effect ladder) and D8-19 (shuffled-`||H||` calibration)

Both controls reuse the SAME real PU Gram matrices and the SAME null-construction machinery — they
are not new Gram-matrix costs, only new *outer loops* over the null computation above:

- **D8-18** sweeps a magnitude grid (recommend ~6-8 steps bracketing PU's realized ~1.5x spread,
  mirroring 07.1's `POSITIVE_CONTROL_TARGET_RHOS`'s finer-grid-than-Phase-7 precedent) — each step is
  one full null computation (as costed above), so total cost scales linearly with the number of
  magnitude steps chosen.
- **D8-19** repeats the entire pipeline (splitting + null) some number of times on a shuffled `||H||`
  field to read a false-positive rate — this is the most expensive control because it is "run
  everything above, N_REPEATS times" rather than one modification to a single input. Recommend
  keeping `N_REPEATS` modest (20-50) given each repeat itself contains a full `N_PERMUTATIONS`-draw
  null; at `N_REPEATS = 30` and each repeat costing the ~1-2 minutes estimated per (d/seed, S) cell
  above, this control alone could cost tens of minutes to a couple of hours if run at full `S`-grid ×
  full-`d`-sweep fidelity. **Recommendation:** run D8-19 at the headline `S` value only (not the full
  grid) and at `d=25` seed 0 only (not the full seed×d cross product) — it is a machinery-level
  calibration of the null's false-positive behavior, not a per-cell result, so it does not need to be
  repeated at every point in the `S`×`d`×seed grid to serve its stated purpose (confirming the
  machinery does not manufacture gaps). This is a Claude's Discretion recommendation with an explicit
  cost justification, not a locked decision.

### Bottom line

**Dominant term: the D8-11 permutation null, run across the full `d`/seed × `S` grid, at whatever
kernel count the RBF-ladder discretion below settles on.** With the Gram-matrix-once architecture,
a full run (headline + ladder + null across all cells, plus D8-18's magnitude sweep and a
reduced-scope D8-19) should complete in **well under an hour of CPU time**, likely in the 10-30 minute
range — several orders of magnitude cheaper than Phase 7's ~2-hour, 5.5-7-hour decoder-training-bound
runs, because **no decoder is retrained anywhere in this phase** (D8-14). The planner should budget a
single wave for "build Gram matrices + compute headline/ladder/null across the full grid" as one
runnable, timeable unit, with the validation-ladder (D8-16/18/19) as a separate wave that can run
before or in parallel with the real-data sweep (the invariance ladder in particular needs no PU data
at all).

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (installed; `notebooks/pu_manifold/tests/` holds every prior phase's unit tests, e.g. `test_density_stratified_null.py`, `test_cross_split_curvature.py`) |
| Config file | Root `pyproject.toml`'s `[tool.pytest.ini_options]` sets `testpaths = ["tests"]`, which targets the **library's own** `tests/` dir, not `notebooks/pu_manifold/tests/` — every prior phase invokes pytest against the notebooks path explicitly rather than relying on the root config |
| Quick run command | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_cka.py -x` |
| Full suite command | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|--------------------|-------------|
| D8-02 | Unbiased HSIC matches a hand-computed reference on a tiny (n=10-20) synthetic example; raises below n=4 | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_unbiased_hsic_matches_reference -x` | ❌ Wave 0 |
| D8-01/D8-16 | Linear CKA = 1.0 on orthogonal-rotation and isotropic-scaling pairs (closed-form) | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_linear_cka_invariances -x` | ❌ Wave 0 |
| D8-01/D8-16 | RBF CKA = 1.0 under orthogonal rotation but NOT under isotropic scaling at fixed sigma | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_rbf_cka_invariances -x` | ❌ Wave 0 |
| D8-16 | Additive-noise ladder decays monotonically | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_noise_ladder_monotone -x` | ❌ Wave 0 |
| D8-06/D8-08 | Tertile-within-stratum split preserves density marginals and yields ~equal-n subsets at every `S` | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_tertile_within_stratum_split -x` | ❌ Wave 0 |
| D8-11 | Label-permutation null preserves stratum sizes and tertile sizes exactly | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_stratified_tertile_null_preserves_sizes -x` | ❌ Wave 0 |
| D8-23 | Importing the new module never mutates any sealed module's globals, regardless of import order | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_import_does_not_mutate_sealed_modules -x` | ❌ Wave 0 |
| D8-18 | Planted-effect ladder recovers a known injected gap at PU's realized dynamic range | integration | `.venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode positive-control` | ❌ Wave 0 |
| D8-19 | Shuffled-`||H||` calibration false-positive rate reported | integration | `.venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode negative-control` | ❌ Wave 0 |
| D8-09/D8-13 | Full `d`×seed×`S` sweep produces a per-`d` clearance mapping at every `S` | integration | `.venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode sweep --freeze-commit <sha>` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_cka.py -x` (fast, no PU data needed for the estimator/invariance tests)
- **Per wave merge:** `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` (full suite, all prior phases' tests must stay green — Phase 7/07.1's tests exercise sealed modules Phase 8 imports read-only)
- **Phase gate:** Full suite green, plus the integration-level sweep/positive-control/negative-control runs recorded in the frozen JSONL, before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `notebooks/pu_manifold/tests/test_cka.py` — covers D8-02, D8-01, D8-16 (estimator + invariance ladder, no PU data needed)
- [ ] Test fixtures for synthetic invariance-ladder pairs (orthogonal rotation, isotropic scaling, independent columns, additive-noise ladder) — small, in-file, no shared conftest needed given the module's small scope
- [ ] `notebooks/pu_manifold/tests/test_cka_split.py` (or folded into `test_cka.py` if the planner keeps split logic in the same module) — covers D8-06/D8-08/D8-11
- [ ] Framework install: none — pytest is already the project's test runner

## Security Domain

### Applicable ASVS Categories

This phase has no network surface, no authentication, no user-input path, and no persistence layer
beyond writing to the existing gitignored JSONL/npz cache convention — the same posture Phase 02.6's
code review recorded for `derivative_bridge.py` ("this code has no network surface, no auth, no
user-input path and no persistence layer" — `02.6-REVIEW.md`, cited in `STATE.md`).

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | No | Not applicable — local notebook analysis, no auth surface |
| V3 Session Management | No | Not applicable |
| V4 Access Control | No | Not applicable |
| V5 Input Validation | Yes (narrow) | Every function should validate shape/finiteness of its numpy inputs before computing, matching the codebase's own convention (`mknn.py`, `density_stratified_null.py` all raise `ValueError` on non-finite/mismatched-shape input before doing any work) |
| V6 Cryptography | No | Not applicable — no secrets, no encrypted data |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Path traversal via a malformed cache-file path argument | Tampering | Route all writes through `cache.cache_path` / `cache._assert_inside_cache`'s existing containment guard, exactly as 07.1's `RECORD_LOCATION_RULE` does — do not construct paths by raw string concatenation |
| Silent pre-registration breach (a constant edited after a number exists) | Tampering / Repudiation | D8-22's freeze-before-any-number discipline, enforced by `assert_preregistered()` called first in every number-producing path, plus the git-ancestry proof (`git merge-base --is-ancestor`, `git rev-list --count`) every prior phase in this milestone uses |
| Accidental mutation of a sealed module's globals on import (monkeypatching) | Tampering | D8-23's explicit prohibition, verified by a regression test that imports the new module in varying orders relative to the sealed modules and asserts no attribute on any sealed module changed |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|----------------|
| A1 | Recommended `N_PERMUTATIONS = 1000`, inheriting Phase 7/07.1's convention, is adequate power for a CKA-difference null at PU's realized subset sizes | Runtime/Cost Model, Discretion | If underpowered, the verdict could read as a null when a real (or real-artifact) effect exists at finer resolution; cheap to raise (see cost model, linear in permutation count) if a pilot run shows the null distribution is not yet stable at 1000 draws |
| A2 | Recommended RBF-ladder scope for the permutation null (sigma only, not all three multiplier rungs) versus running all three sigma rungs through every ladder rung | Runtime/Cost Model, §Discretion | If the planner instead runs all three RBF sigma rungs through the full null, the dominant-cost estimate above should be scaled up ~1.5-2x (2 kernels → up to 4); still well within a tractable single run per the cost model's own margin |
| A3 | Recommended `float32` storage for the 8 precomputed Gram matrices to halve memory | Runtime/Cost Model | If float32 precision loss meaningfully shifts CKA values relative to float64 (unlikely at this n and value range, but unverified in this session), a fallback to float64 with disk-cached matrices (rather than all-in-memory) is the documented alternative |
| A4 | Recommended D8-19 (shuffled-`||H||` calibration) scoped to headline `S` and `d=25` seed 0 only, not the full grid | Runtime/Cost Model, §Discretion | If the false-positive rate turns out to vary meaningfully by `S` or by `d`, a headline-only reading could understate or overstate the machinery's true false-positive behavior at other grid points; this is a cost/rigor tradeoff explicitly flagged as discretion, not a locked decision |
| A5 | Numerical tolerance recommendation for the invariance ladder (orthogonal rotation / isotropic scaling landing within a tight absolute tolerance of exactly 1.0, e.g. `atol=1e-6` at float64) is a reasonable default, not empirically calibrated in this session | Validation Architecture, Code Examples | If float64 accumulation error at n≈few-hundred-to-few-thousand synthetic points exceeds this, the tolerance should be loosened based on the pilot run's actual observed deviation rather than tightened blindly |
| A6 | The `n // n_strata` floor `density_strata` enforces (>= 3) is compatible with every `S` in the planner's chosen grid at n=10,000 | User Constraints / Standard Stack | If the planner's `S` grid includes a very large value (e.g. `S=500`), `n // S = 20`, still comfortably above the floor; not a live risk unless `S` is chosen absurdly large (>3,000), which no precedent in this codebase does |

**If this table is empty:** Not applicable — see entries above. All entries above are engineering
recommendations under the "Claude's Discretion" heading of `08-CONTEXT.md`, not claims about locked
decisions (D8-01..D8-24), which are treated as ratified fact per the mandatory-initial-read
provenance rule and are not re-litigated here.

## Open Questions (RESOLVED)

> All three were decided during planning on 2026-08-27. Resolutions are recorded below and in the
> deciding plan's `<discretion_decisions>` block; nothing here is still open.


1. **Exact `S`-grid values and whether the `d=25` seed axis crosses every `S` or only a headline `S`.**
   - **RESOLVED (08-02 `<discretion_decisions>`):** `S_GRID = (10, 20, 50)`, and the `d=25` seed axis
     runs at **all three** `S` values, not a headline `S` — 18 cells. **This departs from the
     recommendation below, deliberately:** D8-09 leaves *no* headline `S` (clearance is required at
     every grid point), so a seed verdict taken at a single `S` would be un-evaluable under the
     phase's own verdict rule. The cost model absorbs 18 cells comfortably.
   - What we know: 07.1 used `(10, 20, 50)`; D8-08 does not mandate reuse; `S` trades density-match
     tightness against realized `||H||` contrast, not sample size.
   - What's unclear: whether the planner should run the full cross product (3 d-values-with-one-
     expanded-to-3-seeds) × (3 S values) = 15 cells, or restrict the seed axis to a single headline
     `S` value (reducing to 3 d-cells + 3 extra d=25 seed cells at headline S only = 6, plus 2 more
     d=20/d=32 cells at the other two S values = 4 more, total 10) to save runtime.
   - Recommendation: run the full cross product for `d=20`/`d=32` (only 1 field each, cheap) but run
     the seed axis at the headline `S` only for `d=25` (3 seeds × 1 headline S), with the other two `S`
     values checked only at seed 0 — this matches 07.1's own design (`SPLIT_SEED`/`PERMUTATION_SEED`
     held fixed while only `TORCH_INIT_SEEDS` varies, isolating one axis at a time) and keeps the total
     cell count near 10-12 rather than 15+, at negligible loss of coverage since D8-15's seed
     unanimity rule already requires all three seeds to agree at whatever `S` they're checked at.

2. **Whether `cka.py`'s tertile-split logic belongs in the estimator module or a sibling module.**
   - **RESOLVED (08-01 `<discretion_decisions>`):** one `cka.py`, per the recommendation and
     CLAUDE.md's "KEEP THINGS SIMPLE FIRST". The wave decomposition gained nothing from a hard
     module boundary.
   - What we know: the estimator itself (Gram matrices, unbiased HSIC, CKA composition) is reusable
     and unit-testable in complete isolation from any PU-specific splitting logic; the split logic
     depends on `density_stratified_null.density_strata` and PU-specific field files.
   - What's unclear: CLAUDE.md's "keep things simple first" argues for one small file; a stricter
     wave-boundary argument (estimator vs. phase-specific logic, testable independently) argues for
     two.
   - Recommendation: default to one `cka.py` file unless the planner's wave decomposition specifically
     benefits from a hard module boundary (e.g., if the invariance-ladder wave and the real-data-sweep
     wave are planned to run in parallel by different agents, a split module makes the disjoint-file
     wave dependency cleaner).

3. **Whether the shuffled-`||H||` calibration (D8-19) needs its own frozen `N_REPEATS` constant value
   - **RESOLVED (08-04 `<discretion_decisions>`):** `N_REPEATS = 30` is a **frozen pre-registered
     constant** inside the D8-22 freeze commit, not a CLI flag — per the recommendation.
   pre-registered alongside `N_PERMUTATIONS`, or can be left as a runner `--n-repeats` CLI flag.**
   - What we know: every other significance-bearing constant in this codebase (permutation count,
     seed, quantile) is a frozen pre-registered constant, never a CLI-supplied runtime value, per
     D8-22's freeze discipline.
   - What's unclear: nothing, really — the pattern strongly implies `N_REPEATS` (or equivalent) must
     also be a frozen constant in the pre-registration module, not left as a discretionary CLI flag at
     run time, since it directly determines the precision of the reported false-positive rate.
   - Recommendation: freeze `N_REPEATS` (recommend 30, per the cost analysis above) in the same module
     as every other Phase 8 pre-registered constant; do not leave it as a bare CLI default.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | Entire phase | Yes | 3.11+ (project floor, DATA-05 precedent) | — |
| numpy | Gram matrices, HSIC arithmetic | Yes [VERIFIED: local venv] | 2.5.1 | — |
| scipy | Pairwise distances, `rankdata`/`permutation_test` if reused for null shape | Yes [VERIFIED: local venv] | 1.18.0 | — |
| torch | NOT required — no decoder training or inference happens in this phase (D8-14) | Yes (present for other phases) | n/a | Not needed; if absent it would not block this phase at all |
| Frozen field npz (`07_crossmodal_curvature_fields.npz`, `07.1_seed_fields_d25.npz`) | D8-14's input | Yes [VERIFIED: local file present, keys/shapes confirmed by direct `numpy.load`] | n/a (data file, not versioned software) | None — these are the phase's entire data input; if either is missing the runner must halt-not-regenerate, per every prior phase's "gitignored cache is not reproducible here" convention |
| Frozen subsample npz (`hsc`/`legacysurvey` 768-d embeddings) | Gram matrix construction | Yes [VERIFIED: local file present, `subsample_20260729_a79b3460b838fd0a.npz`, confirmed `hsc`/`legacysurvey` both `(10000, 768) float64`] | n/a | None — same halt-not-regenerate convention |

**Missing dependencies with no fallback:** none — everything this phase needs is already present on
this machine.

**Missing dependencies with fallback:** none.

## Sources

### Primary (HIGH confidence)

- Direct source read: `notebooks/pu_manifold/density_stratified_null.py` (full file) — `density_strata`,
  `STRATIFICATION_RULE`, `SENSITIVITY_GRID_RULE`, `stratified_partial_null`, verdict machinery
- Direct source read: `notebooks/pu_manifold/mknn.py` (full file) — `permutation_null`'s
  `permutation_type="pairings"` shape (confirmed as the null D8-11 does NOT reuse), `hubness_skewness`,
  `chance_floor`
- Direct source read: `notebooks/pu_manifold/crossmodal_curvature.py` (relevant sections) —
  `density_diagnostics`, `ALIGNMENT_METRIC = "mknn"` / D7-07 comment (lines confirmed via `grep -n`),
  `per_point_mknn`, `two_tailed_permutation_null`, freeze/ancestry-proof pattern
- Direct source read: `notebooks/pu_manifold/curvature_probe.py` (`local_density_weights`) — confirms
  the inverse-density `w`, mean-normalized-to-1 convention D8-07 specifies
- Direct source read: `notebooks/pu_manifold/linear_probe.py` (relevant sections) —
  `assert_preregistered`, `SEED_HANDLING_RULE`, `SEED_VERDICT_COMBINATION_RULE`,
  `combine_seed_verdicts` signature and behavior
- Direct file inspection via `numpy.load` in this session: `07_crossmodal_curvature_fields.npz` and
  `07.1_seed_fields_d25.npz` — exact keys, shapes `(10000,)`, dtype `float64`; `subsample_*.npz` —
  `hsc`/`legacysurvey` both `(10000, 768) float64`
- `08-CONTEXT.md`, `.planning/ROADMAP.md` Phase 7/07.1/8 sections, `.planning/STATE.md` 2026-08-27
  entries — locked decisions, measured design constraints, cost precedents (1457s curvature compute
  at `d=20`, `D·d²` scaling)

### Secondary (MEDIUM confidence)

- WebSearch, cross-referencing the Song et al. (2012) unbiased HSIC formula and its adoption in
  Nguyen, Raghu & Kornblith (2021) "Do Wide and Deep Networks Learn the Same Things?" (ICLR 2021,
  openreview.net-hosted PDF) — confirmed the `1/(n(n-3))[tr(K̃L̃) + (1ᵀK̃1)(1ᵀL̃1)/((n-1)(n-2)) −
  (2/(n-2))1ᵀK̃L̃1]` form and the "value of CKA is independent of batch size when using this
  estimator" claim, both consistent with training-data knowledge of Kornblith et al. (2019) Eq. 3.
  [CITED: openreview.net/pdf/cb12ae8308060f86d8970f514c2a0e8a33d13c22.pdf]

### Tertiary (LOW confidence, flagged for planner/discuss-phase confirmation)

- The invariance-ladder numerical tolerances (e.g. `atol=1e-6`), the recommended `N_PERMUTATIONS =
  1000`, `N_REPEATS = 30` for D8-19, and the D8-18 magnitude-step count are all [ASSUMED] —
  engineering judgment calibrated against this codebase's own analogous precedents (Phase 7's
  `N_PERMUTATIONS = 1000`, 07.1's finer `POSITIVE_CONTROL_TARGET_RHOS` grid) rather than measured in
  this session. See Assumptions Log.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages, versions confirmed directly in the local venv
- Architecture (Gram-matrix-once pattern, module layout): HIGH — derived from direct reading of the
  frozen field files' shapes/keys and the modality-embedding data flow across every prior phase
- HSIC/CKA formula and invariance properties: HIGH — cross-referenced against the primary literature
  (Song et al. 2012, Kornblith et al. 2019, Nguyen/Raghu/Kornblith 2021) via WebSearch and training
  knowledge, mutually consistent
- Runtime/cost model: MEDIUM — arithmetic is sound (O(n²D) matrix operations, well-characterized numpy
  BLAS performance), but no pilot run was executed in this session to confirm wall-clock numbers;
  treat the "well under an hour" bottom line as a strong expectation, not a measured fact
- Pitfalls: HIGH for the double-centering trap (directly load-bearing, literature-confirmed) and the
  wrong-null-reuse trap (directly confirmed against this codebase's actual function signatures);
  MEDIUM for the specific numeric warning-sign thresholds, which are illustrative rather than measured

**Research date:** 2026-08-27
**Valid until:** No expiry driver — all inputs (frozen npz fields, sealed modules, the HSIC/CKA
literature) are static for the remainder of this milestone. Re-check only if `08-CONTEXT.md` is
amended or if the frozen field files are regenerated under a new fit_key.
