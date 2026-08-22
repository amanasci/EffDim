# Spike Manifest

## Idea

Test whether a **local-polynomial geometry teacher** `(P̂, ÎI)` is feasible at `d=20`, on the
synthetic saddle control where analytic mean curvature is known in closed form. Phase 03.1 sealed
with `Phase 4 blocked, no proposed route out`: regularizing the decoder's *parameterization*
repairs the pullback metric completely (`log10_det_g -83.9 → +0.037`, `cond(g) 1.7e8 → 5.7e2`) and
moves the curvature-ordering estimand only from `rho -0.122` to at most `+0.116`, against the
`rho = 0.989` the identical fixture yields at `d=4`. Parameterization is therefore demonstrably not
the missing ingredient.

The developer-directed next step (`03.1-FINDINGS.md` §10, 2026-08-21) is to score a
**geometry-supervised** signal *alone* on the same four axes, before any architecture change is
proposed — i.e. ask whether there is anything at `d=20` that could teach a decoder geometry, not
whether a particular decoder can learn it.

The teacher is `curvature_probe.quadric_mean_curvature`: `P̂` from `_quadric_tangent_basis` (SVD
tangent frame that tolerates `d > k`), `ÎI` from `quadric_fit_curvature` (minimum-norm least
squares over `1 + d + d(d+1)/2` columns), returning `H = tr_g(ÎI)` as an ambient `(n, D)` vector.
It already exists, sealed and unmodified. It has never been scored on the four axes — the record
only ever states that it is underdetermined at `d=20` (deficit 180 at `k=30`, 110 at `k=100`).

## Requirements

Design decisions that emerged during spiking. Non-negotiable for anything downstream.

- **Score with the sealed scorer, unmodified.** `synthetic_control_run._fidelity_axes` supplies all
  four axes (direction median cosine; magnitude median ratio *and* CV; calibration slope/intercept/
  `R²`; rank Spearman `rho`). Any teacher number must be comparable to the sealed decoder row
  `rank_spearman_rho == -0.015106571347065712` by construction, not by re-derivation.
- **`CURVATURE_CONVENTION = "trace"`.** `H = tr_g(II)`, unnormalized; a unit `d`-sphere gives
  `||H|| = d`. The averaged convention differs by a factor of `d = 20`
  (`02.5-NOTE-high-d-curvature-approaches.md` §2c).
- **No shrinkage dial in the sealed estimator.** D-05 rejected one for
  `quadric_fit_curvature`; minimum-norm least squares is the only fit it computes. A ridge variant,
  if ever built, lives in spike-local code and never edits `curvature_probe.py`. Held for now
  (user decision, 2026-08-21).
- **Never edit `notebooks/pu_manifold/`.** Spikes import it unchanged. A teacher that only works
  after the sealed module is rewritten is itself the finding.
- **Anchor before interpreting a FAIL.** A `d=20` FAIL with no low-`d` anchor cannot distinguish
  the dimension wall from broken wiring. Spike 001 exists for that reason and runs first.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | teacher-low-d-anchor | standard | Given the saddle control at `d ∈ {2, 4}` and the Swiss roll, when the unmodified teacher is scored by the unmodified four-axis scorer, then all four axes clear — making a later `d=20` FAIL attributable to dimension rather than to wiring | ✓ VALIDATED | curvature, anchor, swiss-roll, low-d |
| 002 | teacher-d20-four-axes | standard | Given the sealed `d=20` saddle (`n=10000`, `seed=20260816`), when the teacher fits `(P̂, ÎI)` across `k` spanning the underdetermined and determined regimes, then the four axes say whether it beats the sealed decoder's `rho = -0.0151` | PENDING | curvature, d20, kill-test |
