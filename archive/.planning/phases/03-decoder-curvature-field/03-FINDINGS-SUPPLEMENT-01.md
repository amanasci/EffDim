# 03-FINDINGS Supplement 01 — one supporting clause in §6 is withdrawn

**Date:** 2026-08-23. **Scope:** annotates `03-FINDINGS.md` §6 point 3. **Additive** — the
sealed section is not edited, and its conclusion is not changed.

## The clause

`03-FINDINGS.md` §6, CURV-07 point 3, reads:

> **PU's own accuracy is untested.** Every fixture with known curvature that reached PU's
> dimension failed to *train* to PU-comparable quality. **No curved control reached PU's
> conditioning**, so nothing bounds `‖H‖` — neither magnitude nor, after the saddle's `rho`,
> ordering.

The clause **"nor, after the saddle's `rho`, ordering"** is withdrawn. It treated the sealed
`d=20` saddle control's `rho = -0.0151` as evidence that the ordering of the curvature field
was unbounded at that dimension. That inference does not hold.

## Why

Spike 003 (`.planning/spikes/003-fixture-validity-audit/`) measured that **the `d=20` saddle
control cannot show ordering at all**, as a property of the fixture rather than of the
estimator or the decoder.

For a graph `M = {(x, f(x))}`, `H = tr_g(II)` with `g = I + ∇f ∇fᵀ`. The saddle is
`f(x) = ½ xᵀ diag(signs) x`, so `II = diag(signs)` is **constant** and every spatial variation
in `‖H‖` comes from the metric tilt `1/(1+|∇f|²)` — none from the geometry the estimator is
asked to measure. Holding `d = 20`, `D = 28`, `n` and `k` fixed and changing only the surface:

| fixture | `II` varies? | `rho` | cosine |
|---|---|---|---|
| `quadratic_saddle` (the sealed control) | no (`CV = 0`) | +0.0238 | −0.0966 |
| `cubic` | yes | **+0.6115** | +0.7700 |

Sweeping `k` from 60 to 800 — 13x — the saddle never leaves zero (`+0.040 → −0.036`) while the
cubic climbs to `+0.65`. So the saddle's null is not an undersampling artifact either.

A fixture that returns `rho ≈ 0` no matter how good the estimator is cannot be used as evidence
about the estimator's ordering. The saddle's `rho = -0.0151` is **uninformative** about
ordering, not evidence against it.

## What does NOT change

**§6 point 3's conclusion stands: PU's accuracy remains untested.** The structural gap it names
— that closing it "needs a CAE that fits a *curved* 20-manifold as well as it fits PU, and no
fit in this milestone has done that" — is **still open**. Spike 003 used
`curvature_probe.centroid_mean_curvature` directly on the point cloud, with no decoder and no
training, so it says nothing about whether a CAE can fit such a manifold.

Also unchanged:

- **§6 point 4** — the field does not reproduce across seeds (52x `‖H‖` median spread, two of
  three degenerate). Spike 003 does not touch this, and §6 itself calls it the strongest of the
  four points precisely because it needs no control and no analytic truth.
- **§6 point 1** — seed 13's 351x margin above the `cond(g)` artifact floor, and its stated
  non-generalisation to the other two draws.
- **The `cond(g)` → artifact-curvature band table**, which is measured on the *flat* fixture
  (analytic `‖H‖` exactly zero) and does not depend on any fixture's rankability.
- **§1's override** and every statement conditioned on it.
- **`CURV-07`'s answer**: "Neither has been established. The PU field is NOT validated."

## Reading note for Phase 4

Spike 003 additionally measured that at `d = 20` curvature **direction** is recoverable
(cosine → 1.000) while **magnitude ordering** saturates near `rho ≈ 0.5–0.65`. Phase 4 as
specified partitions by `‖H‖` quantiles — the weaker of the two functionals. That is a design
question for Phase 4's planning, not a change to this phase's record, and it is **not ratified**.
