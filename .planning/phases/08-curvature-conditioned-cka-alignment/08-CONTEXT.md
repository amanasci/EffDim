# Phase 8: Curvature-Conditioned CKA Alignment - Context

**Gathered:** 2026-08-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Add **CKA as a second alignment probe** and test whether Phase 7's curvature–MKNN association is
MKNN-specific. Points are split by `||H||` magnitude into density-matched, equal-n subsets; CKA is
computed on each subset; the tertile-3-minus-tertile-1 difference is tested against a
density-stratified permutation null.

**This phase is D7-07's deferred decision, taken.** `07-CONTEXT.md` §3 declared CKA out of Phase 7
scope as "a separate decision, not a Phase 7 task". The developer took that decision on 2026-08-27.
**No sealed verdict is reopened, softened, or recomputed** — not Phase 7's `ASSOCIATION DETECTED`,
not 07.1's `SURVIVES AT SUBSET OF d` / `SEED STABLE AT d=25`, not Phase 5's `SPLIT ACROSS SEEDS`.

**Design chosen from four options: global CKA, curvature-conditioned.** 07.1's per-point
partial-correlation design does **not** transfer. `mknn.mknn_score` yields an `(n,)` per-point array
(D7-04), which is what made 07.1's partial correlation and stratified null possible. Standard CKA is
a **single global scalar per dataset**, so 07.1's partial/stratified machinery has nothing to attach
to. A per-point local-CKA variant was considered and **not chosen** — unsealed instrument,
small-sample CKA bias, full validation burden. Phase 8 therefore reuses **Phases 5/6's magnitude-split
pattern** (Phase 5 already split on `||H||` magnitude rather than Phase 4's direction sign).

**In scope:** a new additive CKA module, the within-density-stratum tertile split, the stratified
permutation null, the validation ladder (invariance anchor, planted-effect ladder, shuffled-`||H||`
calibration), and the reporting notebook + `08-FINDINGS.md`.

**Out of scope:** decoder retraining; any modification of sealed modules; per-point local CKA;
reinterpreting any sealed verdict.

</domain>

<decisions>
## Implementation Decisions

### CKA Estimator

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

### Split and Density Matching

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

### Null and Verdict Rule

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

### Validation Ladder

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

### Inherited, Non-Negotiable

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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The deferred decision this phase executes
- `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-CONTEXT.md` §3 — **D7-07**
  declares CKA out of Phase 7 scope as "a separate decision, not a Phase 7 task". §3's locked
  decisions are Phase 7's de-facto requirement set. §4 (validated instrument), §5 (what PU measures),
  §6 (the Phase 4 cautionary record), §7 (cost model), §8 (what Phase 7 will not claim).
- `.planning/ROADMAP.md` — Phase 7, Phase 07.1 and Phase 8 entries.
- `.planning/STATE.md` — the 2026-08-27 Phase 8 design entry and its measured design constraints.

### Verdicts that must NOT be reopened
- `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-FINDINGS.md` — Phase 7's
  `ASSOCIATION DETECTED` on the raw statistic.
- `.planning/phases/07.1-density-stratified-null-and-seed-stability/07.1-FINDINGS.md` —
  `SURVIVES AT SUBSET OF d` (`per_d_results = {20: false, 25: true, 32: false}`) and
  `SEED STABLE AT d=25` (3-of-3). §1 states the independence rule (D-14); §2 carries the per-`d`
  clearance table. **Note: neither verdict has been ratified by a human — see the document's own
  opening statement.**
- `.planning/phases/05-curvature-conditioned-linear-decodability/05-03-DECISION.md` — the **one-way**
  do-not-pool-seeds ratification that D8-15 carries.

### Machinery to reuse
- `notebooks/pu_manifold/density_stratified_null.py` — `density_strata(density, n_strata)`,
  `N_STRATA_HEADLINE`, `STRATA_GRID`, `STRATIFICATION_RULE`, `SENSITIVITY_GRID_RULE`. The
  stratification and threshold-grid patterns D8-06/D8-08/D8-09 inherit.
- `notebooks/pu_manifold/crossmodal_curvature.py` — `density_diagnostics` and the D7-03 non-gating
  pattern; line 47 and line 110 carry D7-07's CKA-out-of-scope constant, which Phase 8 supersedes by
  decision (do **not** silently edit it — see D8-23).
- `notebooks/pu_manifold/curvature_probe.py` — `local_density_weights` (inverse density `w`); D8-07's
  `1.0 / w` relative-density convention.
- `notebooks/pu_manifold/mknn.py` — `permutation_null`'s `permutation_type="pairings"` shape, named in
  D8-11 as the null that is **not** used, and `hubness_skewness` / `chance_floor` for diagnostics.
- `notebooks/pu_manifold/linear_probe.py` — Phase 5's `SEED_HANDLING_RULE` /
  `SEED_VERDICT_COMBINATION_RULE` / `combine_seed_verdicts` and the `assert_preregistered` freeze
  machinery.

### Project rules
- `CLAUDE.md` — the Swiss roll standing rule (**declared not applicable by D8-17**), the additive-only
  rule, "KEEP THINGS SIMPLE FIRST", and the `src/effdim/` freeze.
- `.claude/skills/spike-findings-effdim/SKILL.md` — the estimator-validation protocol D8-16/D8-18
  follow: anchor at low `d` first, write the decision rule before running, clear the confound probes.
  Also `references/curvature-estimator-validation.md` and `references/high-d-curvature-feasibility.md`.
- `.planning/PROJECT.md` — Key Decisions table; the milestone definition.

### External
- Kornblith, Norouzi, Lee & Hinton, *Similarity of Neural Network Representations Revisited*,
  ICML 2019 — linear/RBF CKA, the invariance properties D8-16 tests against.
- Song, Smola, Gretton, Bedo & Borgwardt, *Feature Selection via Dependence Maximization*, JMLR 2012 —
  the unbiased HSIC estimator of D8-02.
- Nguyen, Raghu & Kornblith, *Do Wide and Deep Networks Learn the Same Things?*, ICLR 2021 — minibatch
  CKA with unbiased HSIC.
- Duraphe, Smith, Sourav & Wu, *The Platonic Universe*, NeurIPS 2025 ML4PS
  ([arXiv:2509.19453](https://arxiv.org/abs/2509.19453)) — the origin experiment.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `density_stratified_null.density_strata` — ready-made density stratification; D8-06's within-stratum
  split is built on it.
- `density_stratified_null`'s `STRATA_GRID` / `SENSITIVITY_GRID_RULE` — the threshold-grid idiom D8-08
  adopts, already exercised in 07.1.
- `curvature_probe.local_density_weights` — the density field, no new estimator needed.
- `linear_probe`'s freeze machinery (`assert_preregistered`, seed-combination helpers) — the D8-22
  freeze and D8-15 seed rule have a working precedent to copy.
- `07_crossmodal_curvature_fields.npz` and 07.1's three `d=25` seed fields — **all curvature input
  already exists**; Phase 8 trains nothing.

### Established Patterns
- **Freeze-before-any-number (D7-06)** — constants committed in a single commit, git-ancestry-proved
  to precede every measured value.
- **Non-gating diagnostics (D7-03)** — density and hubness reported beside every verdict, gating
  nothing. D8-01, D8-04, D8-10 and D8-12 all follow this shape.
- **Independent per-cell reporting (07.1 D-14)** — a null in one cell does not void another. D8-13.
- **Threshold grid, not point estimate (07.1 `SENSITIVITY_GRID_RULE`)** — D8-08/D8-09.
- **Do not pool seeds (`05-03-DECISION.md`, one-way)** — D8-15.
- **Additive only, no sealed-module mutation on import (Phase 7 cross-cutting)** — D8-23.
- **Report to full precision, never round a near-miss away** (07.1 §2 house style, inherited from
  `07-CONTEXT.md` §5's n=800 Betti draw-2 result).

### Integration Points
- New module under `notebooks/pu_manifold/` (name at planner's discretion), importing
  `density_stratified_null` and `curvature_probe` **read-only**.
- `crossmodal_curvature.py` lines 47 and 110 carry D7-07's "CKA is out of scope" constant. Phase 8
  supersedes that scope decision **by phase decision, not by editing the sealed module** — the
  supersession is recorded here and belongs in the plan's `<superseded_decision>` block.
- Reporting notebook committed with outputs, executed end to end (CLAUDE.md).

</code_context>

<specifics>
## Specific Ideas

- **Measured design constraints, taken 2026-08-27 before planning — do not re-derive:**
  `spearman(density, ||H||)` on the frozen Phase 7 fields is **+0.4281 at `d=20`**, **+0.3150 at
  `d=25`**, **+0.0118 (`p=0.238`, nil) at `d=32`**; `spearman(density, MKNN) = -0.2121` at every `d`.
  **Phase 4's `-0.0273` does NOT transfer to the v1.1 fields**, so at `d=20/25` a curvature-magnitude
  split **is** partly a density split and Phase 4's Gap-3 confound reappears through a different door.
  Consequences: density-matched subset construction is **load-bearing, not a robustness check**; and
  equal-n subsets are mandatory because CKA is biased upward on small samples — the analogue of Phase
  4's `k/n_region` chance-floor artifact that inflated region 1's raw MKNN.

- **`S` does not buy sample size.** Pooled subset size is ~3,333 per tertile at any `S`. `S` trades
  density-match tightness against realized `||H||` contrast (within-stratum tertiles are computed on
  `n/S` points). Any plan that justifies an `S` on sample-size grounds has the mechanism wrong.

- **PU's `||H||` spread is ~1.5x** (plain-AE decoder; Phase 4's centroid estimator read 3.94x). Any
  planted effect must sit at *that* dynamic range — D8-18's whole point.

- **The `d=32` reading must be prominent, not a table row** (D8-12 + D8-21).

</specifics>

<deferred>
## Deferred Ideas

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

</deferred>

---

*Phase: 8-curvature-conditioned-cka-alignment*
*Context gathered: 2026-08-27*
