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
