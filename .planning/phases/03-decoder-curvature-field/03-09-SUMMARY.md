---
phase: 03-decoder-curvature-field
plan: 09
subsystem: curvature-instrumentation
tags: [pytorch, torch-func, chart-autoencoder, curvature, finite-difference, pu-manifold]

# Dependency graph
requires:
  - phase: 03-decoder-curvature-field (03-03)
    provides: derivative_bridge with WR-01/02/03 closed, used verbatim by --bridge at PU scale
  - phase: 03-decoder-curvature-field (03-08)
    provides: the selected n_charts=4 and, via its --converge supplement, the converged
      checkpoint the field is computed on
provides:
  - notebooks/diagnostics/curvature_field_pu_run.py --field and --bridge, executed -- the
    phase's deliverable per-point mean-curvature vector-norm field over the PU cloud with
    its conditioning distribution, near-singular flagging and no-extrapolation evidence,
    and the independent finite-difference agreement check at real scale
affects: [03-10 (synthetic controls -- the calibration this field has no scale without),
  03-11 (the phase record)]

tech-stack:
  added: []
  patterns:
    - "Percentile-based near-singular flagging with the flagged set reported separately, never
       averaged into the summary and never deleted"
    - "Joint 2-D histogram of the reported quantity against its own conditioning, so a small
       value from a degenerate metric is distinguishable from a small value that is real"

key-files:
  created: []
  modified:
    - notebooks/diagnostics/curvature_field_pu_run.py

key-decisions:
  - "The field is computed on the CONVERGED checkpoint (03-08-SUPPLEMENT-03), not on a
     40-epoch grid fit; --field loads it and refuses to train a replacement"
  - "near_zero_reference_fraction is printed beside EVERY relative column unconditionally --
     showing it only beside a large one would be a comparison against a constant, which the
     same plan's acceptance criteria forbid"
  - "--bridge imports decode_closure from derivative_bridge_run rather than copying the
     two-hop chart decode, keeping one composition pinned to the sealed map instead of two"
  - "ONE seed only: 03-09's three-seed spread is NOT delivered, by explicit developer scope"

requirements-completed: [CURV-03, CURV-04, CURV-05, CURV-08]

duration: ~1h execution (bridge 106.4s, field 3129.5s) after ~1.9h of upstream training
completed: 2026-08-16
status: partial
---

# Phase 3 Plan 09: The PU Curvature Field Summary

**The deliverable exists: a per-point mean-curvature vector-norm field over all 10,000 PU rows, median `||H|| = 1.3590e+03` with `cond(g)` median `9.93e+06` beside it, and an independent finite-difference bridge showing the derivative computation is sound (full Hessian agreeing to ~5e-08 relative) while the metric contraction amplifies that error roughly 750-fold -- delivered on ONE seed, not the three this plan requires.**

## Status: PARTIAL

Tasks 1 and 2 are complete and verified. **Task 3 is partially executed**: it requires
`--field` across three seeds, and only seed `20260813` has a converged checkpoint. This is an
explicit developer scope decision ("local CPU, one seed first"), not an oversight or a
shortfall discovered late. The runner reports the other two seeds as missing by name, prints a
probe notice, and deliberately prints **no spread table** for a single draw.

Converging the remaining two seeds costs roughly 3.8 h of CPU. Until that runs, every number
below describes one draw and **no dispersion is claimed anywhere**.

## Task 1 — `--field`

Executed over all 10,000 rows of the frozen subsample at `n_charts = 4`, seed `20260813`, on
the converged checkpoint from `03-08-SUPPLEMENT-03.md`.

### The field

`||H||`, the mean-curvature **VECTOR NORM**. Never Gaussian curvature, never a principal
curvature: at codimension `768 - 20` there is no canonical normal direction, so any signed
scalar reduction would flip sign with an arbitrary choice. Convention: `trace`.

| Statistic | Value (unflagged, n = 9,900) |
|---|---|
| min | 6.8133e+02 |
| p05 | 9.6184e+02 |
| p25 | 1.1579e+03 |
| **median** | **1.3590e+03** |
| p75 | 1.6539e+03 |
| p95 | 2.3913e+03 |
| max | 4.2839e+03 |
| mean | 1.4683e+03 |

Distribution: unimodal, right-skewed, modal bin `[1041.5, 1221.7]`, with a thin tail to 4,284.

```
hist_counts = [93, 1049, 2144, 2103, 1567, 1009, 679, 403, 257, 194,
               132, 101, 63, 50, 25, 13, 8, 4, 4, 2]
```

### `cond(g)` — reported as a distribution, beside `||H||` and never instead of it

| median | p90 | p99 | max |
|---|---|---|---|
| 9.9321e+06 | 1.5828e+07 | 2.3116e+07 | 3.8212e+07 |

### Flagging — and why it mattered

100 points (**1.00%**) exceed the within-config 99th percentile of `cond(g)`
(`> 2.3116e+07`). They are flagged, reported separately, and **not averaged into** the summary
above. None is deleted.

| Set | n | `||H||` mean |
|---|---|---|
| unflagged | 9,900 | 1.4683e+03 |
| **flagged** | **100** | **2.0574e+03** |

The near-singular points carry a **40% higher** mean `||H||` than the rest of the cloud
(flagged range 962.3 to 3,527.2). Averaging them in would have inflated the reported field.
The median is nearly untouched by their exclusion (1363.1 all-points versus 1359.0 unflagged),
which is exactly why this plan required a distribution rather than a single statistic: the
contamination is visible in the mean and invisible in the median.

### The joint view — the thing neither marginal can show

The 2-D histogram of `||H||` against `log10 cond(g)` shows the mass shifting **right as
`||H||` rises**: the lowest-curvature row peaks at `log10 cond ~ 6.9`, while rows three and
four peak at `~7.05-7.14`. Higher conditioning co-occurs with higher curvature.

That association is the reason the joint view was required. It means the large-`||H||` tail is
**at least partly conditioning-driven rather than purely geometric**, and a reader looking only
at the `||H||` marginal would have had no way to see it.

### No-extrapolation evidence (CURV-08)

- Chart assignment independently recomputed via `model.chart_probs(model.encode(x)).argmax(dim=1)`
  and compared against the field's own: **MATCHES**.
- **0 constructed points.** No grid, no interpolation, no synthetic coordinate exists anywhere
  in the mode; the source contains no `meshgrid` and no `linspace`.
- Every point is a real data row measured in the chart the model itself assigned it.

### Second-derivative evidence (CURV-05)

On 64 **held-out** rows, not training rows:

- `max|Hessian| = 6.6968e-01`
- `strictly_positive = True`
- `all_finite = True`

`assert_c2_activation` already refuses a ReLU-family decoder whose Hessian would be identically
zero; this is the positive evidence rather than the absence of a refusal.

### Chart occupancy

**2 of 4 charts used** over the full cloud (chart 1: 6,557 points; chart 3: 3,443). Two charts
receive no point at all. Unchanged from the grid cell and from the converged fit's held-out
occupancy — converging the fit did not revive a dead chart.

### Wall clock

**3,129.5 s (52.2 min)** for the full-cloud field at `d = 20, D = 768`. This measurement did
not exist anywhere in this milestone before now.

## Task 2 — `--bridge`

`derivative_bridge.derivative_agreement` at `BRIDGE_N_POINTS_PU = 96` held-out, chart-assigned
points — deliberately 3x `chart_curvature.VMAP_CHUNK = 32`, so WR-03's chunking fix is
exercised rather than assumed numerically equal by coincidence. Cost re-derived at `d = 20`:
`1 + 40 + 760 = 801` decoder evaluations per point against `MAX_FD_ROWS = 8192`, printed
rather than silently asserted. `fd_step_used = 1.0e-04`, `activation = silu`. Wall clock 106.4 s.

Points partition across the two live charts: chart 1 (65 points), chart 3 (31).

| Level | median rel. (chart 1 / chart 3) | max rel. (chart 1 / chart 3) | `near_zero_reference_fraction` |
|---|---|---|---|
| `full_hessian_agreement` | 4.6764e-08 / 5.3941e-08 | 1.8674e-01 / 3.2764e-01 | 0.0 / 0.0 |
| `reduced_..._agreement[H_vec]` | 4.6860e-04 / 5.0599e-04 | **6.2101e+00 / 2.3705e+01** | 0.0 / 0.0 |
| `reduced_..._agreement[H_norm]` | 3.5161e-05 / 3.6462e-05 | 1.4799e-04 / 2.2795e-04 | 0.0 / 0.0 |

No threshold is applied and no boolean is derived from any of these numbers.

**`near_zero_reference_fraction = 0.0` on every row.** This is the reading
`02.6-FINDINGS-02.md` needed and did not have when it recorded
`full_hess_max_abs_rel = 1.1351e+00`: every relative column here is a genuine ratio, not a
thin denominator. It is printed beside every relative column unconditionally.

Three findings:

1. **The derivative computation is sound.** Autodiff and an independent, non-`torch.func`
   finite-difference stencil agree on the raw Hessian to ~5e-08 relative. This phase is the
   first ever to edit `chart_curvature.py`, and plan 03-05's forward-versus-reverse test
   compares two autodiff paths and structurally cannot see a bug they share. The bridge can,
   and it does not see one.

2. **The metric contraction amplifies error roughly 750-fold.** Median relative disagreement
   runs 4.7e-08 at the Hessian and 3.5e-05 after the `g^-1` contraction. That is the
   `cond(g) ~ 1e7` tax, measured rather than argued. It still leaves `||H||` resolved to about
   five significant figures — the amplification is large but starts from a very small base, so
   the reported field is numerically precise.

3. **The norm is stable exactly where the vector is not.** `H_vec` max relative disagreement
   reaches **6.21 and 23.7** — 620% and 2370% — at points with no thin denominator to blame,
   while `H_norm` at the same points stays near 2e-04. The curvature vector's direction swings
   wildly while its magnitude holds; at codimension 748 the components redistribute without
   moving the norm. This independently vindicates reporting `||H||` as a vector norm and never
   a signed or component-wise reduction: the quantity this plan refuses to report is precisely
   the one the bridge shows is unstable.

## What these numbers do and do not mean

**Numerically precise, and that is a claim about the computation only.** The bridge establishes
agreement between two independent derivative implementations. Agreement shows the derivative
computation is stable. It does **not** show the learned surface is correct — a net that learned
a smooth but wrong surface passes this check cleanly.

**The magnitude has no scale to be read against yet.** `||H|| ~ 1.36e+03` on data standardized
to unit global standard deviation is a very tightly curved surface: on the codimension-1 sphere
heuristic `R ~ d / ||H||`, a radius of curvature around `20 / 1359 ~ 1.5e-02`, roughly two
orders of magnitude tighter than the cloud's own extent. Whether the PU manifold is genuinely
that curved, or the CAE learned a wiggly surface that happens to reconstruct well, is **not
separable by anything measured here**. Plan 03-10's synthetic controls — the same architecture
and protocol fitted to manifolds whose curvature is known analytically — are the calibration
this field has no scale without.

**The C0/C2 gap is now concrete rather than argued.** The converged fit reconstructs to
`mse_per_dim = 4.71e-05` and simultaneously carries these second derivatives. Excellent
reconstruction and enormous curvature coexist in the same model, which is the claim
`03-NOTE-d12-retirement.md` §3 made structurally and this plan measured directly.

**The gate override travels with the numbers.** Printed with the field, not filed away from it:
this field is decoded from a parameterization no gate in this milestone has validated, every
sealed verdict here is FAIL, and a field so decoded conflates real curvature with
parameterization damage. Nothing above separates the two.

## Deviations from Plan

1. **Three-seed spread not delivered.** One seed only, by explicit developer scope. Task 3 is
   partial; `status: partial` in this document's own frontmatter. The runner names the missing
   seeds and prints no spread table rather than presenting a single draw as a result.
2. **`--field` loads a converged checkpoint rather than retraining per seed.** The plan
   predates the convergence directive; retraining inside `--field` would have put a
   40-epoch truncated fit behind the deliverable.
3. **`near_zero_reference_fraction` printed beside every relative column, not only above 1.0.**
   The plan asked for the latter, but its own acceptance criteria forbid comparing a bridge
   statistic against a constant. Printing it unconditionally satisfies the intent without the
   contradiction.
4. **The Hutchinson randomized-trace estimator is not used anywhere** (D-03), as required; only
   the exact `g`-trace path produced these numbers.

## Verification

- `--dry-run` exits 0, names both modes, the selected `n_charts`, the three seeds, the
  99th-percentile flag policy and the `801` / `8192` arithmetic: **pass**.
- `--smoke` exits 0 with `h_norm`, `cond`, `flagged` and `joint_hist` all present: **pass**.
- `COND_FLAG_PERCENTILE = 99.0` present uncommented; `vector norm` present; no
  `gaussian_curvature` / `principal_curvature` token; no `meshgrid`; no `linspace`: **pass**.
- `BRIDGE_N_POINTS_PU = 96 > VMAP_CHUNK = 32`: **pass**.
- `pytest notebooks/pu_manifold/tests/test_derivative_bridge.py -x -q`: **19 passed**.
- `pytest notebooks/pu_manifold/tests/ -q`: **302 passed, 1 skipped**.
- Three seeds executed: **not met** — one seed, by scope.

---
*Phase: 03-decoder-curvature-field*
*Completed: 2026-08-16*
