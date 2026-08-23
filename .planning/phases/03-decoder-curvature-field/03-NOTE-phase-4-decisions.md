# Note — four Phase 4 decisions taken at the Phase 3 close

**Date:** 2026-08-23. **Status:** developer decisions, taken via structured prompt at the Phase 3
close. **Binding on Phase 4's scope and planning**, not on any sealed verdict. Nothing recorded in
Phases 2, 02.x, 3 or 03.1 is reopened, softened, recomputed or reinterpreted here.

Evidence base: `.planning/spikes/003-fixture-validity-audit/README.md` (2026-08-22).

---

## D4-01 — Phase 4 partitions on curvature DIRECTION, not `|H|` magnitude

**Decision.** Regions are formed by clustering the unit vectors `H/‖H‖`, not by binning `‖H‖`
quantiles. Success criterion 2 ("partitioned into high/low-curvature regions by a pre-specified
quantile threshold") is **superseded** and must be rewritten before Phase 4 is planned.

**Why.** Spike 003 measured that at `d = 20` the three fidelity axes come apart:

| axis | status at `d=20` |
|---|---|
| magnitude | ~50x attenuated (ratio 0.018) — unusable |
| **direction** | **cosine 0.77–1.000** — usable |
| rank of magnitude | `rho` 0.41–0.65, saturating — partial |

Direction is a unit vector, so estimating it is a subspace problem controlled by the tangent-space
estimate, which converges. Magnitude requires estimating a scale from `k` samples in `d`
dimensions, which is where the dimension-dependent bias of the literature bites. Phase 4 as
originally specified consumed the weaker functional.

**Validation available before PU.** `varying_ii_controls.make_ridge_graph_control` has a single
known bending direction `w` by construction (rank-one Hessian), so direction-clustering has an
**exact known answer** on it. That check is a Phase 4 precondition.

**What this does not license.** The quantile-threshold *discipline* survives: the partition is
still pre-specified and frozen before any regional MKNN number is computed. Only the QUANTITY
being partitioned changes. The garden-of-forking-paths guard in the Ordering constraint stands
unchanged.

---

## D4-02 — the estimator is decided by a fixture head-to-head, not by assertion

**Decision.** Before Phase 4 commits to an instrument, run `centroid_mean_curvature` (point cloud,
no training) and the CAE chart decoder **side by side on `cubic` at `d = 20`**, where the answer is
known and the fixture can distinguish them. Phase 4 adopts the winner.

**Why.** The two routes have never been compared on a fixture capable of separating them, because
none existed until spike 003. Every sealed control has a constant second fundamental form and
returns `rho ≈ 0` for both instruments regardless of merit. On `cubic`, the centroid estimator
scored `rho = +0.65` at `d = 20` with no training; the decoder route's sealed control scored
`-0.0151` — but on the saddle, which cannot rank at all, so **that pair of numbers is not a
comparison** and must not be quoted as one.

**Consequence if the point-cloud estimator wins.** Phase 03.1's metric-regularization work becomes
optional rather than blocking for Phase 4, since the point-cloud route forms no pullback metric and
cannot suffer the `cond(g)` pathology. 03.1's sealed findings are unaffected either way.

---

## D4-03 — split-half reliability is accepted as sufficient evidence about the PU field

**Decision.** The measured PU split-half reliability (`median R_H = 0.5894` at `k = 231`, zero sign
disagreement from `k = 120`, rising monotonically in `k`) is accepted as sufficient to proceed.
Cross-estimator agreement on PU is **not** required as a precondition.

**This is a deliberate acceptance of a known blind spot, recorded as such.** The recommendation at
the decision point was cross-estimator agreement, and it was declined. The risk, stated plainly so
it is never inherited silently:

> **Split-half reliability cannot detect a bias that both halves share.** Both halves use the same
> estimator, the same `k`, the same architecture. Spike 003 measured this failure directly on the
> Swiss roll, where the answer is known: `R_H = 0.990` with `rho = 0.469` — near-perfect
> reproducibility alongside mediocre accuracy. There is no ground truth on PU, so this gap cannot
> be closed by more of the same measurement.

**Any Phase 4 result therefore inherits an unvalidated field**, exactly as Phase 3's own results are
conditioned on the §1 gate override. Phase 4's record must state this in its own words rather than
by reference, on the same standard `03-FINDINGS.md` §1 was held to.

**Cheap partial mitigation, not required but nearly free.** D4-02 already produces both estimators
in working order. Running both on PU and reporting their rank agreement would cost one cell and
would convert this blind spot into a measured number. Available at any time without relitigating
this decision.

---

## D4-04 — commit strategy

Two commits: spike 003 (research), then the Phase 3 closure (phase record). Kept separately
revertable.

---

## Status of these decisions

**D4-01 and D4-02 change Phase 4's scope and must be reflected in the ROADMAP before Phase 4 is
planned.** D4-03 changes no scope; it records an accepted risk. None of the four has been executed.
Phase 4 remains **BLOCKED** — D-11 stands, and these decisions define the route out rather than
constituting it.

---

## Amendment 01 (2026-08-23) — D4-01's validation scoped out, and a caveat on the evidence it rests on

### The validation was built, then deliberately not run

`notebooks/diagnostics/direction_partition_run.py` and
`varying_ii_controls.make_multinormal_ridge_control` were built to validate D4-01 by measuring
partition fidelity — `ARI(partition from estimated field, partition from true field)` — for the
direction and magnitude schemes side by side.

**Developer decision: the question is too narrow to be worth the compute, and the run was
killed.** The reasoning, which is sound and is recorded rather than paraphrased: both schemes
read the *same* estimated field at the *same* points, so any error in **where** the decoder
places a point cancels out of the comparison. The test could therefore only answer "given
whatever field we end up with, which functional partitions more stably" — not "is the field
trustworthy", which is the question that actually gates Phase 4 and which the location check
(`reconstruction_truth`) answers more cheaply and more fundamentally.

**D4-01 stands**, accepted on spike 003's direct measurements: at `d=20`, direction survives
(cosine 0.77–1.000) while magnitude is ~50x attenuated and its ordering saturates at
`rho ~ 0.5–0.65`.

Preliminary evidence from the smoke cell before it was stopped, recorded for whoever revisits
this: direction ARI **0.351** against magnitude ARI **0.016** at `d=6`, `n=800`, `k=30`. That is
a 22x gap and it points the same way as D4-01, but it is a SMOKE-SCALE number at a dimension
well below PU's and must not be quoted as a result.

### The caveat that must not be lost: codimension

Building the validation surfaced a real limit on the evidence D4-01 rests on.

**On a codimension-1 graph, `H = H_scalar · n̂` — so "curvature direction" IS the surface
normal.** Clustering it clusters tangent-plane orientation, not curvature structure. Measured on
`ridge` at `d=8`: the unit-`H` covariance has rank **2**, not the full space.

Every fixture in spike 003 — `saddle`, `bowl`, `aniso`, `cubic`, `sine`, `ridge` — is a
codimension-1 graph. So **spike 003's cosine 1.000 establishes that the estimator recovers the
NORMAL ORIENTATION**, which is a tangent-space problem known to converge well. It does **not**
establish that the estimator resolves `H`'s direction *within* a high-dimensional normal space.

PU is `d ~ 20` inside `D = 768` — codimension **~748** — and that is precisely the regime where
the distinction matters. **D4-01 therefore rests on evidence from a regime one codimension wide,
applied to a problem 748 wide.** That gap is unmeasured and is not closed by anything in this
milestone.

`make_multinormal_ridge_control` (`f: R^d -> R^m`, `m` orthonormal ridge directions, `H` rotating
within an `m`-dimensional normal space; verified at `d=20, m=4` to give unit-`H` covariance rank
8 with eigenvalues 0.25 x 4) exists and is tested, should anyone want to close it later. It tops
out around `m=8` against PU's ~748, so it would narrow the gap rather than close it.

### Status

D4-01 is **adopted on partial evidence, with the codimension gap named**. Phase 4's record must
state this in its own words rather than by reference, on the same standard `03-FINDINGS.md` §1
was held to for the gate override — the same requirement D4-03 already carries.

---

## Amendment 02 (2026-08-23) — D4-02 RESOLVED: the point-cloud estimator

`estimator_headtohead_run.py`, three cells, both arms on **identical data** at `d=20`, `D=28`,
`n=5000`, `k=231`, on fixtures with a varying second fundamental form (so both instruments have
something to find):

| cell | cloud `rho` | cloud cosine | decoder `rho` | decoder cosine | cloud cost | decoder cost |
|---|---|---|---|---|---|---|
| `cubic`, 200 ep | **+0.6115** | +0.7700 | +0.0019 | −0.0319 | 2s | 358s |
| `cubic`, 400 ep | **+0.6115** | +0.7700 | +0.0072 | +0.0299 | 2s | 354s |
| `ridge`, 400 ep | **+0.4119** | +0.9173 | +0.0184 | −0.0378 | 2s | 358s |

**D4-02 resolves to `curvature_probe.centroid_mean_curvature` on the point cloud.** It wins on
rank and on direction in every cell, at ~179x less compute. The decoder's cosine is ~0 and twice
NEGATIVE — its `H` direction is no better than random — and its magnitude ratio of 12,000–42,000x
is mechanistically consistent with the measured `cond(g)` of `4e11`–`1.6e12` destroying the
`g^-1` contraction inside `H = sum_jk g^jk II_jk`. That is the same pathology Phase 03.1
attacked, where `scale` fully repaired the metric and moved rank `rho` only `-0.122 -> +0.116`.

### The caveat, stated rather than buried

**The decoder arm is UNDERTRAINED relative to Phase 3's sealed fits.** It reconstructs at
`mse_per_dim = 0.23–0.32` against the sealed `d=20` cell's `1.6e-02` — roughly 15x worse. The
sealed cells ran 300 epochs at `n=10000`, `D=768`, in 25-epoch blocks, and cost ~2.6h each;
this comparison ran at reduced `D`, `n` and epochs so it was affordable. **So this is not a
clean disqualification of a well-trained decoder**, and must not be quoted as one.

What makes it decisive anyway: doubling the budget from 200 to 400 epochs moved the decoder's
`rho` from `+0.0019` to `+0.0072` and its reconstruction from `0.32` to `0.23`. **Flat.** There
is no trajectory toward competitiveness, while the point-cloud arm reaches `+0.61` in two
seconds with no training, no pullback metric to condition, and no seed sensitivity.

### Consequences

1. **Phase 03.1's metric regularization is OPTIONAL, not blocking, for Phase 4.** The
   point-cloud route forms no pullback metric and cannot suffer the `cond(g)` pathology at all.
   03.1's sealed findings are unaffected; they simply stop gating.
2. **Phase 3's non-reproducing field (52x `||H||` median spread across seeds) stops being on the
   critical path.** Phase 4 does not have to inherit it.
3. **`k` becomes Phase 4's main free parameter.** At `d=20` the centroid estimator needs `k` in
   the hundreds (cubic: `+0.035` at `k=30`, `+0.648` at `k=231`). On PU's 10,000 rows `k=231`
   is 2.3% of the cloud per neighbourhood; whether that is still local on PU is unmeasured.
4. **A cheap re-run is now available for D4-03's declined mitigation.** Both instruments exist
   and run on PU; reporting their rank agreement costs one cell and would convert D4-03's
   accepted blind spot into a measured number. Still not required.
