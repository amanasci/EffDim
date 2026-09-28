# Spike Wrap-Up Summary

**Date:** 2026-08-22
**Spikes processed:** 2
**Feature areas:** curvature-estimator-validation, high-d-curvature-feasibility
**Skill output:** `./.claude/skills/spike-findings-effdim/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001 | teacher-low-d-anchor | standard | ✓ VALIDATED | curvature-estimator-validation |
| 002 | teacher-d20-four-axes | standard | ⚠ PARTIAL | high-d-curvature-feasibility |

## Key Findings

**The directed question is answered: no.** At the sealed `d=20` saddle and the sealed `k=30`, the
local-polynomial geometry teacher scores `rank_spearman_rho = -0.0281` against the sealed decoder's
`-0.0151`. The teacher is not better than the decoder; both are indistinguishable from zero. A
geometry-supervised objective built on this teacher, on this control, has nothing to teach with.

**The teacher already existed and had never been scored.**
`curvature_probe.quadric_mean_curvature` is `(P̂, ÎI)` exactly — sealed since D-05, designated
non-gating on sample-complexity grounds, with only its underdetermination flag on record. No new
estimator was written to answer the question.

**Measured dead ends, so they are not re-proposed.**
- Raising `k` to make the fit determined: `r/R` crosses `1.0` at `k=231`. Determined and local are
  mutually exclusive at `d=20`.
- `rho = +0.393` at `k=500`: a fixture artifact. Rank gain `k=30 → 500` is `+0.421` on the globally
  quadratic saddle, `+0.051` on a non-quadratic fixture.
- More data: tripling `n` moved `rho` `+0.010` at `d=20`, against `+0.058` for the same lever at
  `d=4`. `r/R ~ (k/n)^(1/d)`.
- Rank statistics read alone: 52–75% of points carry an anti-aligned `H`, including in cells whose
  `rho` looks usable.

**Independent reproduction of the recorded locality table.** `r/R = 0.1158` on the Swiss roll
(recorded `0.115`) and `0.8915` at `d=20, k=30` (recorded `0.906`), with MRE `0.8494` against
`0.870` — recomputed from scratch rather than quoted.

**The finding that outranks the question — open, and not acted on.** The saddle control may be
unable to show curvature ordering at `d=20` at all: its analytic Hessian is constant by
construction, so `||H||` varies only through the pullback metric. The same unmodified teacher
scores `+0.5934` on the Gaussian-bump fixture at identical `d`, `n`, `k`, seed — not explained by
local sampling scale (partial `rho = +0.6006`) and not fully by dynamic range. This matters because
the sealed `d=20` decoder verdict rests on the same fixture. Raised at spike 002's closing
checkpoint as a question for the developer. **No sealed number is reinterpreted; Phase 4 stays
blocked; no route out is proposed.**

**Method that made the above trustworthy.** Anchor at low `d` first (spike 001 exists only for
this); state the pass regime in `r/R` rather than in `d`; write each probe's decision rule into its
source before running it. Two probes then refuted the hypotheses that motivated writing them —
credible only because the rules predated the data.

## Caveats

Single seed (`20260816`) throughout, one fixture per family, no repetition. `rho = +0.5934` sits
beside 87% median relative error and near-random direction: not a working estimator, only evidence
that something orderable survives where the saddle reports nothing.
