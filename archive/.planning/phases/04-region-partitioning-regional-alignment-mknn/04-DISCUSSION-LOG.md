# Phase 4: Region Partitioning & Regional Alignment (MKNN) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-23
**Phase:** 4-region-partitioning-regional-alignment-mknn
**Areas discussed:** PU curvature field + k, Direction partition scheme, Density-confound controls, MKNN mechanics + budget

---

## PU curvature field + k

### D4-05 — PU's low ‖H‖ dynamic range

Measured evidence put to the developer: PU `h_spread` (p95/p05) = 5.54 / 4.83 / 4.79 / 4.86 at
k = 30 / 60 / 120 / 231, against the runner's own calibration of bowl 1.4x (unrankable, `rho +0.03`),
cubic 28.2x and ridge 34.3x (rankable, `rho +0.61` / `+0.41`). Unaddressed in the Phase 4 decisions note.

| Option | Description | Selected |
|--------|-------------|----------|
| Report it, partition on direction anyway | Low magnitude spread confirms D4-01 rather than blocking; direction is a unit vector and does not consume the magnitude spread. Record states the 4.8x number and its calibration. No new gate. | ✓ |
| Gate on a direction-spread analogue first | Measure a direction-dispersion statistic on PU against the same fixtures before freezing the split. | |
| Treat as a real risk to the whole phase | Pre-register a stopping rule: if the direction field is also near-degenerate, report "no partitionable curvature structure" and skip regional MKNN. | |

**User's choice:** Report it, partition on direction anyway.

---

### D4-06 — how `k` is chosen

Context: only k=231 clears the runner's `R_H >= 0.5` admissibility rule; `r/R = 1.0331` there and a
231-neighbourhood is 2.3% of the 10k cloud.

| Option | Description | Selected |
|--------|-------------|----------|
| Freeze k=231, state r/R alongside | Adopt the only admissible k from the existing sweep; state that locality is asserted, not established. | |
| Extend the sweep past 231 first | R_H still rising monotonically at 231 (0.078 → 0.247 → 0.428 → 0.589); run k = 350, 500 and pick by a pre-registered rule. | ✓ |
| Freeze k=231 + robustness at k=120 | Headline at 231, plus a cheap sensitivity partition at k=120 to see whether the verdict flips. | |

**User's choice:** Extend the sweep past 231 first.

---

### D4-07 — the pre-registered plateau rule

Context raised: Phase 1 froze `k*=15` by a plateau rule whose defect is recorded in `WINDOWS.md` —
`STAGE2_K` was unevenly spaced, so the plateau was maximal in index space, not `k` space. The same
trap is live here (30/60/120/231 is roughly geometric; 231/350/500 is not).

| Option | Description | Selected |
|--------|-------------|----------|
| Absolute increment in R_H, spacing-free | Freeze at smallest k where median R_H gains < a declared amount (e.g. +0.03) over the previous point AND R_H >= 0.5. Never compares gaps across unevenly spaced points. | ✓ |
| Per-unit-k slope threshold | Freeze where d(R_H)/dk falls below a declared slope; normalizes by spacing but is noisier. | |
| Grid geometrically, then use increment | Continue the ~2x ladder (231 → 460) so spacing stays uniform in log-k. | |
| Cap on r/R as a hard co-condition | Any of the above plus a declared r/R ceiling disqualifying a k regardless of R_H. | |

**User's choice:** Absolute increment in R_H, spacing-free.
**Notes:** Threshold recorded as **+0.03** (the value carried in the selected option's description),
declared before the k=350/500 runs. Developer was invited to name a different threshold and did not.

---

### D4-08 — D4-02 Amendment 02's cheap cross-estimator mitigation

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — run it, report rank agreement | One cell: CAE decoder H field alongside centroid_mean_curvature at the frozen k; report Spearman agreement and median direction cosine. Converts D4-03's blind spot into a number, gates nothing. | |
| No — D4-03 stands as accepted | Taken deliberately and recorded as an accepted blind spot; adding the check invites treating it as a gate it was never declared to be, and the decoder arm measured cosine ~0 (twice negative) so agreement would be uninformative either way. | ✓ |
| Run it, but only if the sealed CAE checkpoint is reused | Cheap version only, using an existing `03_converged_cae_pu_nc4_seed*.pt`; skip if none gives a usable field. | |

**User's choice:** No — D4-03 stands as accepted.

---

## Direction partition scheme

Raised before questioning: `REQUIREMENTS.md` REGN-03/04 still carry the superseded `|H|`-quantile
wording even though ROADMAP success criterion 2 was superseded by D4-01.

### D4-09 — how unit `H/‖H‖` becomes regions

| Option | Description | Selected |
|--------|-------------|----------|
| Sign split on top eigenvector of unit-H covariance | Deterministic (no seed, no init), exactly two regions, maps onto the high/low binary MKNN and REGN-05 expect, exact check available on ridge fixtures where v should recover w. | ✓ |
| Spherical k-means, k=2 | Closer to D4-01's literal "clustering" wording, does not assume a single axis, but adds seed sensitivity and can return wildly unbalanced regions. | |
| Spherical k-means, region count by pre-registered rule | Do not assume two regions; pick count by a declared rule. Richer but multiplies comparisons against thin headroom. | |
| Sign split plus spherical k-means robustness check | Sign split as headline, k-means ARI reported alongside. | |

**User's choice:** Sign split on top eigenvector of unit-H covariance.

---

### D4-10 — known-answer fixture validation before the PU split is frozen

Context: D4-01's body calls the `make_ridge_graph_control` check "a Phase 4 precondition", but
Amendment 01 scoped out the partition-fidelity run and named the codimension gap.

| Option | Description | Selected |
|--------|-------------|----------|
| Ridge + multinormal ridge, both | Exact known-w check at codim 1, plus `make_multinormal_ridge_control` at d=20, m=4 and m=8 (verified unit-H covariance rank 8). Narrows the gap from 1 to 8, record says so. | |
| Ridge only, codimension gap stated | D4-01's literal precondition; record states this establishes normal-orientation recovery only. | |
| Multinormal only | Skip codim-1 as uninformative (direction IS the normal there); run only the fixture family testing what Phase 4 needs. | |
| Neither — accept the gap as already named | D4-01 was adopted on partial evidence with the gap recorded; narrowing 1 to 8 against ~748 risks reading as closure. Freeze the PU split directly. | ✓ |

**User's choice:** Neither — accept the gap as already named.
**Notes:** This overrides D4-01's own body text naming the ridge check a precondition; recorded as an
explicit override in CONTEXT.md rather than a silent omission. Flagged to the developer at the time,
once, that combined with D4-03 and D4-05 this leaves the phase with no known-answer anchor at any
point in the chain — estimator, field, or partition.

---

### D4-11 — REGN-03/04's superseded quantile wording

| Option | Description | Selected |
|--------|-------------|----------|
| Re-mint REGN-03/04 for the direction scheme | Rewrite both texts preserving IDs and the pre-specification discipline; follows the Phase 3 re-mint precedent. | |
| Leave text, satisfy intent, document mismatch | Do not touch REQUIREMENTS.md; state the wording is superseded. Leaves a live contradiction for a later audit. | |
| Re-mint and add a new REGN-06 | Re-mint 03/04, plus REGN-06: eigenvector v and the sign split recorded and frozen as artifacts before any MKNN number, so the split is auditable after the fact. | ✓ |

**User's choice:** Re-mint and add a new REGN-06.
**Notes:** REGN-01's "Isomap coordinate space" was subsequently folded into the same re-mint, once
D4-13 established the field runs in ambient 768-d.

---

### D4-12 — the CLAUDE.md standing Swiss roll rule

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — one notebook for the partition scheme | `04_swiss_roll_direction_partition_check.ipynb`: import the sign-split code unchanged, run on a 3k Swiss roll, plot the two regions on the roll. Would be the only known-answer check left in Phase 4 after D4-10. | |
| No — rule does not reach a partition rule | The rule targets models that map data down and back or claim to recover manifold structure; a sign split on an eigenvector of a computed field is neither, and the estimator is already covered by `02.5_swiss_roll_curvature_probe_check.ipynb`. | ✓ |
| Yes, and cover MKNN in the same notebook | One notebook covering both new pieces, with MKNN validated on known answers (identical → 1.0, independent → null). | |

**User's choice:** No — rule does not reach a partition rule.
**Notes:** CONTEXT.md requires Phase 4's record to state this reasoning explicitly rather than
silently shipping without a notebook.

---

## Density-confound controls

Two facts surfaced before questioning: (1) `pu_curvature_rankability_run.py` runs the estimator
directly on 768-d embeddings, not Isomap coordinates, so REGN-01's wording is written for the
retired decoder route; (2) `local_density_weights` / `density_correct=True` already exist and the
PU field computed so far is uncorrected.

### D4-13 — which space density is measured in

| Option | Description | Selected |
|--------|-------------|----------|
| Ambient 768-d — same space as the estimator | The centroid displacement that masquerades as curvature is computed from 768-d neighbourhoods, so the density that could fake it is 768-d density. Re-mint REGN-01. | ✓ |
| Both — ambient for the confound, Isomap for REGN-01 as written | Satisfies the requirement text unmodified and shows whether the two notions agree; one extra cheap cell. | |
| Isomap coordinate space only, as written | Honours REGN-01 literally; cheaper and more interpretable but not where the confound arises. | |

**User's choice:** Ambient 768-d — same space as the estimator.

---

### D4-14 — how far the confound battery goes

| Option | Description | Selected |
|--------|-------------|----------|
| Correlation + density-matched null | REGN-02's correlation, plus a null resampled within density strata so a density-driven regional difference lands in the null rather than the effect. | |
| Full battery as SUMMARY.md listed it | Correlation, centroid-distance, partial regression, density-matched stratification/null. Most defensible; four original constructions with no reference implementation. | |
| Correlation only, per REGN-02 literally | Report the correlation, state the confound risk plainly, let the reader judge. Cheapest; consistent with the phase's other decisions to state rather than test. | ✓ |
| Correlation + density-matched null + flat-fixture check | The above plus `make_flat_fixture(density_skew=...)`, where true H = 0 so any partition structure found IS density by construction. | |

**User's choice:** Correlation only, per REGN-02 literally.
**Notes:** Flagged at the time, as a consequence distinct from the D4-10 one: MKNN is itself a k-NN
statistic and directly density-sensitive, so without a density-matched null a regional MKNN
difference cannot be separated from a regional density difference by anything in this phase.
REGN-02's correlation becomes the only evidence bearing on that and MKNN-07's verdict must be
worded to reflect it. Carried into CONTEXT.md.

---

### D4-15 — corrected or uncorrected headline field

Asked twice. The first framing drew on `curvature_probe.py`'s docstring rationale — that the
correction exists because a density gradient produces nonzero centroid displacement on an exactly
flat manifold. `02.5-02-SUMMARY.md` records that claim as **amended and mathematically unsatisfiable
for the shipped estimator**: on a flat fixture the normal projection is exact regardless of density
skew, so there is nothing to remove. The question was re-put on accurate grounds. Only the second
asking is authoritative; the first is recorded here for the audit trail.

| Option (second asking) | Description | Selected |
|--------|-------------|----------|
| Uncorrected — keep continuity with the sweep | Existing R_H numbers are all uncorrected; correction invalidates the sweep D4-06 extends. The correction cannot remove density-faked H (proven), so it does not discharge REGN-02's confound. | |
| Corrected — density_correct=True, k_density=30 | Real if modest improvement (~8-10% median relative-error reduction on curved+skewed fixtures); PU density is not known to be uniform; k_density=30 already pre-registered so no new constant. Cost: sweep re-runs corrected from k=30 up. | ✓ |
| Both via centroid_mean_curvature_both_densities | One pass computes both bit-identically; corrected headline with the uncorrected ARI and verdict-flip reported alongside. | |

**User's choice (both askings):** Corrected — density_correct=True, k_density=30.
**Notes:** Accepted cost recorded explicitly: the four existing uncorrected R_H rows
(k = 30/60/120/231) are **superseded, not extended**. Budget ~2,100s to reproduce them plus
k = 350, 500 on top.

---

## MKNN mechanics + budget

Raised before questioning: MKNN-04 fixes the *null* to the region's index set but MKNN-03 does not
say where the *score's* neighbours come from; inconsistency there makes effect and null incomparable.

### D4-16 — where per-region k-NN sets come from

| Option | Description | Selected |
|--------|-------------|----------|
| Within-region — k-NN inside the region's index set | Score and null live in the same index set, satisfying MKNN-04 by construction. Regional number not directly comparable to global MKNN-02 because k/n differs — which the region's own null absorbs. | ✓ |
| Global neighbours, averaged over region members | Preserves the manifold's real neighbourhood structure and keeps regions on MKNN-02's k/n footing, but requires redefining the null or MKNN-04 is violated. | |
| Within-region headline, global reported alongside | Within-region for MKNN-04, global alongside so the reader can see whether the framings agree. Doubles the cells. | |

**User's choice:** Within-region — k-NN computed inside the region's index set.

---

### D4-17 — statistical budget

Context: 2 regions x 4 values of k = 8 cells on ~5k rows each; the paper's Legacy Survey crossmodal
range is 0.4-2% against a k/n chance floor of 0.1% at k=5, n=5000.

| Option | Description | Selected |
|--------|-------------|----------|
| 1,000 permutations, 1,000 bootstrap resamples | Resolves p to ~0.001, enough for a 4-20x-over-chance effect, stable percentile CIs. Affordable because the permutation only shuffles row correspondence, so the k-NN index is built once per cell and reused. | ✓ |
| 10,000 / 10,000 | Resolves p to ~1e-4 and smooths CI tails; 10x cost for one decimal place. | |
| Pre-register a budget, escalate only where it matters | 1,000 as declared default; re-run at 10,000 only for cells landing in a pre-declared band near the threshold. | |

**User's choice:** 1,000 permutations, 1,000 bootstrap resamples.

---

### D4-18 — known-answer checking for the MKNN implementation

Context: every module in `notebooks/pu_manifold/` has a matching `tests/test_<module>.py` except
`mknn.py`.

| Option | Description | Selected |
|--------|-------------|----------|
| test_mknn.py with exact known answers | Identical embeddings → 1.0 at every k; independent random → chance k/n; hand-computed n=6, k=2 case; seed reproducibility; CI brackets the estimate. Pure numpy, milliseconds. | |
| Rely on the global MKNN-02 reproduction as the check | MKNN-02 already requires reproducing the global crossmodal number against the published 0.4-2% range; landing there on real data is a stronger end-to-end check and costs nothing extra. | ✓ |
| Both | Unit tests for the algebraic properties no real-data run can verify, plus MKNN-02 end-to-end. | |

**User's choice:** Rely on the global MKNN-02 reproduction as the check.

---

### D4-19 — making MKNN-02's comparison valid

Flagged: MKNN's chance level is k/n, so our 10,000 rows and the paper's 101,725 do not share a
baseline. Under D4-18 this reproduction is the only implementation check the phase has, so
"outside the published range" would otherwise not separate a bug from the subsample.

| Option | Description | Selected |
|--------|-------------|----------|
| Report raw alongside chance floor and n | State raw MKNN, k/n floor at our n, the paper's raw range and their n; the ratio-over-chance carries the comparison. No extra computation. | ✓ |
| Subsample-size sensitivity curve | Global MKNN at n = 2k, 5k, 10k to show which way n pushes the number; converts the mismatch into a measured slope. | |
| Match the paper's n on the global number only | Stream the full 101,725-row config for the global number alone. Strongest claim; costs a full stream and a 100k-row k-NN. | |
| Raw comparison only, mismatch stated | One-line statement that n differs and the comparison is indicative. Cheapest; weakest as the implementation check it now is. | |

**User's choice:** Report raw alongside chance floor and n.

---

## Claude's Discretion

Offered and declined at the area close, or explicitly routed to the planner:

- **MKNN-07's verdict rule** — what counts as "the high-vs-low result holds" across 2 regions x 4
  values of k, and whether the 8 comparisons take a multiplicity correction. Routed to the planner
  with the hard constraint that the rule **must be written into the notebook before the first
  regional MKNN number is computed**, per the ROADMAP's Ordering constraint. The developer was
  shown that constraint verbatim and chose planner discretion.
- Near-zero `‖H‖` exclusion policy; unbalanced-region handling / minimum region size.
- Whether `v` is computed on all 10k points or an admissible subset.
- Field scope (all 10k or subsample), R_H anchor-point protocol at the new k values, seed policy,
  whether `d=20` stays the working tangent dimension (D-07 bars inheriting it by accident).
- Which correlation statistic serves REGN-02; whether density is also compared *between* regions
  after the split.
- Exact k grid past the 350/500 floor.
- MKNN-08's hubness caveat — stated, or substantiated with a measured hubness statistic.
- Shipped artifact shape — notebook only, or notebook plus a `notebooks/diagnostics/` runner.

## Deferred Ideas

- Cross-estimator agreement on PU (D4-08 declined; one cell, available any time).
- Codimension-gap narrowing via `make_multinormal_ridge_control` at m = 4, 8 (D4-10 declined).
- Density-matched null, partial regression, centroid-distance checks (D4-14 declined).
- `tests/test_mknn.py` (D4-18 declined).
- Full-config 101,725-row global MKNN (D4-19 declined).
- Intramodal MKNN across a model-size ladder — already deferred at project level; needs a second
  model size, out of scope for v1.1.

**No scope creep arose.** Every area stayed inside the ROADMAP Phase 4 boundary; the only
requirement-level changes (D4-11) re-mint existing IDs to match a superseded scheme rather than
adding capability.
