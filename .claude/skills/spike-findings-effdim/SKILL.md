---
name: spike-findings-effdim
description: Implementation blueprint from spike experiments on high-dimension curvature estimation in EffDim. Requirements, the estimator-validation protocol, measured dead ends, and the open fixture-validity question at d=20. Load before proposing any curvature estimator, prior, or control fixture.
---

<context>
## Project: EffDim — v1.1 PU Manifold Curvature

Spikes 001–002 tested whether a **local-polynomial geometry teacher** `(P̂, ÎI)` is feasible at
`d=20`, on the synthetic saddle control where analytic mean curvature is known in closed form.
Phase 03.1 had sealed with Phase 4 blocked and no proposed route out: regularizing the decoder's
*parameterization* repairs the pullback metric completely (`log10_det_g -83.9 → +0.037`,
`cond(g) 1.7e8 → 5.7e2`) while moving rank `rho` only from `-0.122` to at most `+0.116`, against
the `rho = 0.989` the identical fixture yields at `d=4`. The developer-directed next step
(`03.1-FINDINGS.md` §10) was to score a geometry-supervised signal **alone** on the same four axes
before any architecture change was proposed.

The answer was no — and the spike surfaced a larger question about the control fixture itself.

Spike sessions wrapped: 2026-08-21 (spikes 001, 002); wrap-up 2026-08-22.
</context>

<requirements>
## Requirements

Non-negotiable. Every reference file honors these.

- **Score with the sealed scorer, unmodified.** `synthetic_control_run._fidelity_axes` supplies all
  four axes (direction median cosine; magnitude median ratio *and* CV; calibration slope/intercept/
  `R²`; rank Spearman `rho`). Spike numbers must be comparable to sealed rows by construction.
- **`CURVATURE_CONVENTION = "trace"`.** `H = tr_g(II)` unnormalized; a unit `d`-sphere gives
  `||H|| = d`. The averaged convention differs by a factor of `d`, and this codebase has already
  shipped and fixed one factor-of-`d` bug.
- **No shrinkage dial in the sealed estimator.** D-05 rejected one for `quadric_fit_curvature`;
  minimum-norm least squares is the only fit it computes. Ridge variants live in spike-local code.
  Held by user decision, 2026-08-21.
- **Never edit `notebooks/pu_manifold/` or `notebooks/diagnostics/` from a spike.** Import
  unchanged. A model that only works after the sealed module is rewritten is itself the finding.
- **Anchor at low `d` before interpreting a failure at high `d`.** A FAIL with no anchor cannot
  distinguish the phenomenon from broken wiring.
- **Never report a rank statistic from this teacher without the direction axis beside it.**
  52–75% of points carry an anti-aligned `H` at `d=20`, including cells whose `rho` looks usable.
- **Compare fixtures spread-for-spread, not name-for-name.** `||H_true||` spans 1095× on the bump
  fixture and 33× on the saddle; that alone moves `rho` from `+0.593` to `+0.150`.
- **A `k` large enough to make the quadratic fit determined is not a neighbourhood at `d=20`.**
  `r/R = 1.0331` at `k=231`, `1.0992` at `k=500`. Any estimator claiming both must show its `r/R`.
- **`D = 28` substitutes for `D = 768` only under a measured invariance check**, and only for
  `quadric_mean_curvature` (worst disagreement `1.288e-14`, 204× speedup).
- **No sealed number is reinterpreted by anything here.** Phase 4 stays blocked; no route out is
  proposed.
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| Estimator validation protocol | `references/curvature-estimator-validation.md` | Anchor at low `d`, state the pass regime in `r/R`, write the decision rule before running, then clear three confound probes — fixture structure, local scale, dynamic range |
| High-`d` feasibility findings | `references/high-d-curvature-feasibility.md` | The teacher does not beat the decoder at `d=20` (`-0.028` vs `-0.015`); larger `k`, more data, and the `k=500` result are all measured dead ends; the saddle control may be unable to show ordering at all |

## Source Files

Complete spike sources, including recorded stdout, preserved in `sources/`:

- `sources/001-teacher-low-d-anchor/` — `run_anchor.py`, `probe_cv_and_n.py`, `.out` files, README
- `sources/002-teacher-d20-four-axes/` — `run_d20.py`, three `probe_*.py`, `.out` files, README

Every script is standalone. Run with `.venv/bin/python` — several sealed modules import torch at
module scope even when the spike path is pure numpy.
</findings_index>

<open_question>
## Open — for the developer, not for autonomous action

**The `d=20` saddle control may be unable to show curvature ordering at all.**
`make_saddle_control` sets `hess = np.repeat(np.diag(signs)[None, None, :, :], n, axis=0)` — the
analytic Hessian is constant at every point, so `||H||` varies only through the pullback metric,
never through the second fundamental form. Measured: the same unmodified teacher scores `-0.0281`
on the saddle and `+0.5934` on the Gaussian-bump fixture at the same `d`, `n`, `k` and seed; not
explained by local sampling scale (partial `rho = +0.6006`) and not fully by dynamic range (bumps
hold `+0.21`–`+0.34` at spread-matched windows where the saddle scores zero in every window).

This matters because the sealed `d=20` decoder verdict rests on the same fixture. It is an open
question raised at spike 002's closing checkpoint — **not** a reinterpretation of a sealed result,
and it must not be used as one. See `references/high-d-curvature-feasibility.md` § Constraints.
</open_question>

<metadata>
## Processed Spikes

- 001-teacher-low-d-anchor (VALIDATED)
- 002-teacher-d20-four-axes (PARTIAL)
</metadata>
