# Spike 003 — [RESOLVED] the `d=20` dead end was the FIXTURE, not the dimension

**Date:** 2026-08-22 (overnight autonomous session)
**Status:** spike 002's open question **CLOSES**. Curvature ranks at `d=20`.

---

## Read this first

**Six questions were open when the session started. Five are now answered.**

| question | answer | where |
|---|---|---|
| Is `d=20` a hard limit for curvature estimation? | **No.** `rho = +0.65` at `d=20`. | k-threshold table |
| Then why did every `d=20` run return zero? | Every sealed control has **constant `II`** and is unrankable by construction. | "Why every sealed control..." |
| Does a non-minimal surface fix it (your hypothesis)? | **No — it makes it worse.** Bowl spread 1.4x vs saddle 31.4x. | "Non-minimality does not help" |
| Is it undersampling? | **No.** Sweep `k` 13x; the saddle never leaves zero. | k-threshold table |
| Was the decoder location mismatch the cause? | **No.** Swiss roll `|Δ| ≤ 0.018`; `d=20` `+0.104 → +0.039`. | "Corrections", item 3 |
| Is the PU manifold itself rankable? | **RUNNING** — the pre-Phase-4 gate. | "What this means for Phase 4" |

**The finding with the most leverage:** at `d=20`, curvature **direction** is recovered
essentially perfectly (cosine `1.000`) while **magnitude ordering** saturates near `rho ≈ 0.4-0.65`.
Phase 4 currently partitions by `|H|` quantiles — the weaker functional. Partitioning by
curvature *direction* would use the one that survives. That fork is worth settling before
Phase 4 is planned.

**Nothing was committed and Phase 4 was not started.** Sealed code untouched; the permission to
edit it proved unnecessary. 52 new known-answer tests, all passing; the 69 pre-existing tests
still pass.

---

## The one-line result

At **identical** `d=20`, `D=28`, `n=5000`, `k=231` and estimator, changing only the surface:

| fixture | `II` varies? | `‖H‖` spread | centroid `rho` |
|---|---|---|---|
| `quadratic_saddle` (**the sealed control**) | no (`CV = 0`) | 31.4x | **+0.0238** |
| `quadratic_bowl` | no (`CV = 0`) | 1.4x | +0.0302 |
| `quadratic_aniso` | no (`CV = 0`) | 2.4x | +0.0615 |
| `cubic` | yes (`CV = 0.104`) | 28.2x | **+0.6115** |
| `sine` | yes (`CV = 0.080`) | 30.4x | +0.1225 |

`+0.024` on the sealed saddle and `+0.611` on a cubic, same everything else. **The estimator
can rank curvature at `d=20`.** Every previous zero at this dimension was measured on a
fixture that had nothing rankable in it.

---

## Why every sealed control was unrankable

For a graph `M = {(x, f(x))}` the mean curvature is `H = tr_g(II)` with `g = I + ∇f ∇fᵀ`.
When `D²f` is constant, all spatial variation in `‖H‖` comes from the metric tilt
`1/(1+|∇f|²)` and none from the geometry the estimator is asked to measure.

| sealed control | defect |
|---|---|
| `make_flat_control` | `II = 0`. No curvature at all. |
| `make_sphere_control` | `II` constant by symmetry ⇒ `‖H‖` constant ⇒ rank **undefined**. The sealed `d=4` record literally stores `rho: null`. |
| `make_saddle_control` | `II = diag(signs)`, constant. `‖H‖` moves only through `g`. |

**Every quadratic graph has constant `D²f`.** No choice of eigenvalues escapes it — pinned by
`test_every_quadratic_has_exactly_constant_second_fundamental_form`.

### Non-minimality does not help — it hurts

The natural fix ("the saddle is minimal, `trace(Q)=0`; use a non-minimal surface") is wrong,
and the algebra says why. For `f = ½ xᵀdiag(a)x`,

```
tr_g(D²f) = tr(diag(a)) − (uᵀ diag(a) u)/(1+|u|²),    u = x ⊙ a
```

Raising the trace adds a **large constant to every point** and leaves the varying part alone,
so relative dynamic range *collapses*. Measured at `d=20`: the maximally non-minimal bowl has
spread **1.4x** against the minimal saddle's **31.4x**. Pinned by
`test_non_minimality_does_not_rescue_dynamic_range`.

---

## The concentration law (new, and it constrains all future fixture design)

Varying `II` is necessary but decays with dimension. For any **separable**
`f(x) = Σⱼ fⱼ(xⱼ)` the Hessian is diagonal, so

```
‖D²f‖²_F = Σⱼ fⱼ''(xⱼ)²
```

is a sum of `d` independent terms and concentrates. Measured CV of `‖D²f‖_F`:

| `d` | 2 | 4 | 8 | 16 | 20 | 40 |
|---|---|---|---|---|---|---|
| `cubic` | 0.371 | 0.243 | 0.163 | 0.112 | 0.099 | 0.069 |
| `sine` | 0.291 | 0.186 | 0.129 | 0.089 | 0.081 | 0.057 |
| `ridge` | 0.434 | 0.462 | 0.491 | 0.488 | 0.480 | 0.487 |

`cubic` and `sine` track `1/√d` to a constant ≈ 0.44. **At high `d` a separable surface is
nearly constant-curvature however it is built**, so asking an estimator to rank its curvature
is asking it to rank noise. The same argument applies to `‖H‖ = |tr_g(II)|`, itself a sum over
`d` coordinates.

`make_ridge_graph_control` escapes it: `f(x) = A sin(w·x)` has `D²f = −A wwᵀ sin(w·x)`, **rank
one**, so `‖D²f‖_F = A|sin(w·x)|` averages nothing over `d` and its CV is flat at ≈ 0.48 from
`d=2` to `d=40`. `|∇f|² ≤ A²` is bounded too, so the metric tilt cannot run away with `d`
either. Both pathologies removed at once, surface still analytic.

---

## Which estimator works

The **cheap** one. At `k=231`, `d=20`:

- `centroid_mean_curvature` (D-05's gating estimator): `rho = +0.6115` on cubic
- `quadric_mean_curvature` (the spike 001/002 "teacher"): `rho = −0.0157`, MRE 5–260

This is exactly the D-05 rationale, now confirmed empirically at `d=20`: centroid estimates
only `H`'s **trace** — one unknown from `k` samples — while the quadric fit needs
`d(d+1)/2 = 210` coefficients per normal direction. Spike 002 spent its budget on the
estimator that *cannot* work at this dimension.

---

## Literature agreement

- **Aamari & Levrard**, *Nonasymptotic rates for manifold, tangent space and curvature
  estimation* (Ann. Statist. 47(1), 2019; [arXiv:1705.00989](https://arxiv.org/abs/1705.00989)) —
  optimal rates for `II` via local polynomials are dimension-dependent; local-polynomial
  estimators are provably suboptimal once noise exceeds a `d`-dependent threshold.
- **Chen, Latifi Jebelli et al.**, *Curvature of high-dimensional data*
  ([arXiv:2511.02873](https://arxiv.org/abs/2511.02873)) — proves **bias increases drastically
  in higher dimensions, so much so that in high dimensions the probability that a naive
  curvature estimate lies in a small interval near the true curvature could be near zero.**
  Their own validation reaches **dimension twelve**; `d=20` is past the published state of the
  art. Note their fixtures are **spheres** — constant curvature — so even the literature's
  high-`d` validation cannot test *ranking*.
- **Cao & Li**, *Efficient Weingarten map and curvature estimation on manifolds*
  (Mach. Learn. 110, 2021) — convergence rate set by intrinsic dimension.

Consistent picture: **magnitude** estimation at `d=20` is hopeless (matches our MRE 5–260),
while **rank** is a weaker functional and survives — which is what this milestone actually
gates on.

---

## Corrections to earlier claims in this project

1. **Spike 002's "teacher does not beat the decoder" was mis-stated.** It beat both the sealed
   decoder and 03.1's best prior; it failed only the absolute `rho ≥ 0.90` supervision bar.
2. **The coworker branch *does* have a synthetic check.** `physics_curvature_scale_bias_variance/synthetic.py`.
   But it is a **`d=2` chart in `R³`**, self-labelled "design check only", and uses a **crude
   reimplementation** (`_kh_stat`) rather than the production `fit_nested_chart`. So the precise
   claim is: their *production* estimator has no known-answer test. Notably their synthetic
   contrasts `kappa = cos(3r)` against `kappa = 1` — **they independently identified the
   constant-vs-varying-curvature axis.**
3. **The decoder location mismatch is real but not the explanation.** Re-scoring at
   `F(z_chart(x_i))` moves Swiss roll `rho` by at most `0.018` (converged arms), and at `d=20`
   moves it `+0.104 → +0.039`. Fixed in `reconstruction_truth.py`; not the cause.

---

## Artifacts

| path | what |
|---|---|
| `notebooks/pu_manifold/varying_ii_controls.py` | fixture family + `second_fundamental_form_variation` |
| `notebooks/pu_manifold/reconstruction_truth.py` | re-score at the decoder's own point |
| `notebooks/pu_manifold/cross_split_curvature.py` | `K_H_cross` / `R_H` ported from `curvature-experiments` |
| `notebooks/diagnostics/varying_ii_teacher_sweep_run.py` | the table at the top |
| `notebooks/diagnostics/ridge_frequency_sweep_run.py` | isolates `r/R` at fixed dynamic range |
| `notebooks/diagnostics/saddle_d20_rescore_probe.py` | `d=20` drift check |
| `notebooks/03.2_swiss_roll_cross_split_curvature_check.ipynb` | CLAUDE.md sanity check |

Tests: 18 + 14 + 20 = **52 known-answer tests**, all passing.

---

## The direction guard — why `rho = +0.61` is not a position artifact

At `k=231` out of `n=5000` the neighbourhood is ~5% of the cloud, so "the estimator recovered
position along the surface, not curvature" is a live worry — it is the same cheat that
invalidated spike 002's `k=500` cell. On any deterministic analytic surface curvature *is* a
function of position, so the two cannot be separated by a rank statistic alone.

The **cosine** axis separates them. It compares the estimated mean-curvature **vector** against
the analytic one in ambient `R²⁸`; recovering position cannot orient a vector.

| fixture | `rho` | median cosine | MRE |
|---|---|---|---|
| `cubic` | +0.6115 | **+0.7700** | 0.982 |
| `sine` | +0.1225 | +0.3622 | 0.988 |
| `quadratic_saddle` | +0.0238 | −0.0966 | 0.782 |
| `quadratic_bowl` | +0.0302 | +0.0243 | 0.998 |
| `quadratic_aniso` | +0.0615 | +0.0492 | 0.997 |

Cosine tracks `rho` across the whole set and is ≈ 0 on every constant-`II` fixture. The signal
is genuine curvature.

**The split that matters:** at `d=20` **rank (+0.61) and direction (+0.77) survive; magnitude
does not (MRE 0.98 ≈ 100% error).** That is exactly what the literature predicts — bias
explodes with dimension, destroying magnitude — and rank/direction are weaker functionals that
survive it. `spearman_gate_statistic` gates this milestone on **rank**, so the surviving
functional is the one the project actually needs.

---

## `d`-sweep at `k=30` — `k` is the second lever

centroid `rho`, `n=5000`:

| `d` | `ridge` | `cubic` | `quadratic_saddle` |
|---|---|---|---|
| 4 | +0.746 | +0.758 | +0.371 |
| 8 | +0.387 | +0.536 | +0.131 |
| 16 | +0.086 | +0.158 | +0.042 |
| 20 | +0.039 | +0.035 | +0.016 |

Varying-`II` beats the sealed saddle 2–3x at every `d`, but at `k=30` everything collapses by
`d=16`. Raising `k` to 231 at the same `n` takes cubic from **+0.035 to +0.611**. So two
levers are needed together at `d=20`: a fixture with varying `II`, and `k` large enough to
average down the centroid estimator's variance. Neither alone is sufficient — which is why
every previous attempt, holding the sealed fixtures fixed, found nothing.

The local-quadric teacher, by contrast, is strong at `d=4` (`rho +0.93`, cosine +0.995) and
dead by `d=8`. Spike 002 spent its budget on the estimator that cannot reach `d=20`.

---

## Varying `II` is necessary but not sufficient — the *scale* of variation matters too

At `d=20`, `k=231`, `n=5000`:

| fixture | `II` CV | centroid `rho` | centroid cosine |
|---|---|---|---|
| `cubic` | 0.104 | **+0.6115** | +0.7700 |
| `ridge` | **0.483** | +0.4119 | **+0.9173** |

The ridge has **4.6x** the curvature variation and scores **lower on rank** -- while scoring
**higher on direction** (cosine `+0.917` vs `+0.770`). The reason is a second
`r/R`-style constraint, now on the variation rather than the magnitude: the ridge concentrates
all its curvature change into ONE direction `w`, so a neighbourhood of radius `r` spans `w·x`
over `±r`. At `r ≈ 2.5` against the `sin` period `2π ≈ 6.3`, each neighbourhood covers ~40% of
a full cycle and the estimator averages the variation away. The cubic spreads its variation
across all `d` directions, so each one changes slowly across a neighbourhood.

So the fixture-design rule is **not** "maximise `‖D²f‖` variation". It is:

> the curvature must vary on a length scale LONGER than the neighbourhood radius `r`, while
> still varying enough across the whole domain to be rankable.

`make_ridge_graph_control`'s `frequency` is the knob for exactly this, and
`ridge_frequency_sweep_run.py` sweeps it — lowering frequency raises the radius of curvature
`R` (and the variation length scale) while `Var(w·x) = L²/3` holds the domain-wide dynamic
range fixed, independent of `d`.

---

## What survives at `d=20`, precisely

Magnitude ratio `median(‖H_est‖ / ‖H_true‖)` at `d=20`, `k=231`, `n=5000`:

| fixture | `rho` (rank) | cosine (direction) | ratio (magnitude) |
|---|---|---|---|
| `cubic` | +0.6115 | +0.7700 | **0.0183** |
| `ridge` | +0.4119 | **+0.9173** | 0.0185 |
| `sine` | +0.1225 | +0.3622 | 0.0121 |
| `quadratic_saddle` | +0.0238 | −0.0966 | 0.3214 |

The estimator returns roughly **2% of the true curvature magnitude** — a ~50x attenuation,
consistent with `centroid_mean_curvature`'s own documented `O(r²)` bias at finite
neighbourhood radius (D-05 caveat 1), and with the published result that bias explodes with
dimension.

**So the three fidelity axes come apart cleanly at `d=20`:**

| axis | status at `d=20` | usable? |
|---|---|---|
| magnitude | ~50x attenuated | **no** |
| direction | cosine 0.77–0.92 | **yes** |
| rank | `rho` 0.41–0.61 | **partially** |

`curvature_fidelity_report` was built to keep these three separate and never collapse them
into one score. That design decision is what makes this readable at all: a single combined
score would have been dominated by the magnitude failure and would have reported `d=20` as a
flat dead end, hiding a direction signal at cosine 0.92.

---

## Second guard — the bowl is a built-in negative control

The direction/cosine argument above rules out position-recovery, and there is an independent
argument from the same table that needs no new run.

On `quadratic_bowl` the sample positions vary over the full domain exactly as they do on
`cubic` — same `n`, same box, same `d` — but the curvature is nearly constant (`‖H‖` spread
**1.4x**, `II` CV exactly 0). If the estimator were reporting position rather than curvature,
it would produce structured output there too, and `rho` against the (nearly constant) truth
would be erratic rather than null.

Measured: `rho = +0.0302`, cosine `+0.0243`.

So the estimator returns **nothing** when position varies and curvature does not, and returns
`rho +0.61` / cosine `+0.77` when both vary. Those two facts together are what license reading
the cubic result as curvature recovery.

`quadratic_bowl` was built to test the minimality hypothesis and turned out to be a better
negative control than that: it is a **varying-position, constant-curvature** fixture, which is
precisely the null the milestone never had. `make_flat_control` cannot serve this role — it has
no curvature *and* is used to measure the noise floor, so a null result there is
uninformative about whether the estimator tracks position.

**Recommendation:** promote `quadratic_bowl` to a standing negative control alongside `flat`.

---

## Implementation re-check (requested)

| item | status |
|---|---|
| `cross_split_curvature` algebra vs `effdim_curvature_metrics.cross_metric_pair` | faithful; row-wise port; 20 tests |
| `reconstruction_truth` inversion of `rotate_and_pad` | identity tests at `d=4` **and** `d=20`, `rtol=1e-9` |
| `varying_ii_controls` closed forms | central finite differences for cubic, sine, ridge |
| `make_quadratic_graph_control` reproduces sealed saddle | exact, `atol=0` |
| ridge Hessian is rank one | `matrix_rank(wwᵀ) == 1`, `‖wwᵀ‖_F == 1` |
| curvature convention | module-level guard; every fixture routes through sealed `graph_mean_curvature` |
| sealed code | **not modified** — permission was given but proved unnecessary; both preprocessing chains invert from recorded values alone |

**52 known-answer tests, all passing.**

---

## What this means for Phase 4 (do not start it yet)

Phase 4 partitions the manifold by `|H|` quantiles, so it consumes the curvature field's
**ordering** — the one functional tonight showed survives at `d=20`. That is encouraging, but
two things must be checked on the real PU data first, and both are cheap.

### 1. The PU manifold may itself be constant-curvature

Everything tonight says a `rho ≈ 0` can mean "the surface has no rankable curvature". That
reading has never been checked for the PU embedding, and if it holds, **Phase 4's quantile
partition is meaningless no matter how good the estimator is** — you would be binning noise.

Cheap test, no training: run `centroid_mean_curvature` on the frozen 10k PU subsample at
several `k`, and report
- the dynamic range of the estimated `‖H‖` (`p95/p05`), against the ~1.4x that the
  near-constant-curvature bowl produces and the ~28–34x the rankable fixtures produce;
- the split-half `R_H` from `cross_split_curvature`, using two disjoint halves of the cloud.

A spread near 1x, or an `R_H` near zero, halts Phase 4 on evidence rather than on suspicion.

### 2. The point-cloud estimator may be the better instrument than the decoder

The whole `03.x` line computed curvature **through a trained CAE decoder**, and the sealed
`d=20` saddle control scored `rho = −0.015` that way. Tonight the **centroid estimator applied
directly to the point cloud** scored `+0.61` at the same `d`, with no training at all, on a
fixture with rankable curvature.

The decoder route was never compared against the direct route on a fixture that could
distinguish them, because no such fixture existed until tonight. **Run
`synthetic_control_run`-style decoder fits and `centroid_mean_curvature` on `cubic` at `d=20`
side by side before committing Phase 4 to either.** If the direct estimator wins, Phase 4 does
not need the decoder, and the entire `03.1` metric-regularisation effort becomes optional
rather than blocking.

### 3. `k` must be chosen, and it is not free

At `d=20` the centroid estimator needs `k` in the hundreds (cubic: `+0.035` at `k=30`,
`+0.611` at `k=231`). With only 10,000 PU rows, `k=231` is 2.3% of the cloud per
neighbourhood. Whether that is still "local" on the PU manifold is an open question that the
`r/R` diagnostic in `ridge_frequency_sweep_run.py` can answer directly.

---

## Open threads left running

- `ridge_frequency_sweep_run.py` at `d=20` — isolates `r/R` at fixed dynamic range. Tests
  whether locality is the remaining lever once the fixture is fixed.
- `k`-threshold sweep at `d=20`, `n=8000` — locates the `k` where ranking switches on.

Both write to `notebooks/.cache/03.2_*.jsonl` and survive the session.

---

## The `k` threshold — and the cleanest proof that the fixture is the problem

`d=20`, `n=8000`, `D=28`. **centroid** `rho`:

| `k` | `cubic` | `ridge` | `quadratic_saddle` |
|---|---|---|---|
| 60 | +0.3811 | +0.1848 | +0.0402 |
| 120 | +0.5821 | +0.3477 | +0.0362 |
| 231 | +0.6481 | +0.4370 | +0.0304 |
| 400 | +0.6503 | +0.4860 | +0.0086 |
| 800 | +0.5886 | +0.4999 | **−0.0360** |

**Sweep `k` over 13x and the sealed saddle never leaves zero**, while `cubic` climbs from
+0.38 to +0.65 and saturates. This is the strongest form of the result: the saddle's null is
not an undersampling artifact that a bigger budget would fix. It is unrankable by
construction. Any future attempt to rescue a `d=20` result by raising `k` on the sealed
controls is now excluded on evidence.

`cubic` saturates around `k = 231-400`; `ridge` is still climbing at `k=800`, consistent with
its curvature varying on a single direction whose length scale the neighbourhood has not yet
resolved (see the section above).

### The cosine guard catches the quadric's cheat automatically

**quadric** on `quadratic_saddle`:

| `k` | `rho` | cosine | MRE |
|---|---|---|---|
| 231 | +0.0726 | −0.045 | 244.3 |
| 400 | **+0.2981** | **−0.319** | 17.8 |
| 800 | **+0.3789** | **−0.315** | 20.5 |

Rank climbs to +0.38 while direction goes NEGATIVE and magnitude is off by 20x. That is the
signature of the ball engulfing the manifold and recovering the fixture's *global* quadratic
model -- spike 002 diagnosed exactly this at its `k=500` cell and had to argue for it with a
separate confound probe. Here the cosine axis flags it with no extra run.

**Rule worth keeping:** a rank gain that arrives without a direction gain is not curvature
recovery. Never report `spearman_gate_statistic` without `median_cosine_similarity` beside it.

The quadric is also non-monotone in `k` throughout (`+0.44` at `k=60` while UNDERDETERMINED,
`−0.02` at `k=231` where the fit first becomes determined), which is further reason not to
build a gate on it at this dimension.

---

## Direction and magnitude come apart, and the split is the actionable finding

Ridge frequency sweep at `d=20`, `n=5000`. Frequency changes ONLY the radius of curvature `R`;
`Var(w·x) = L²/3` holds the domain-wide dynamic range fixed and is independent of `d`.

| `r/R` | `k=30` `rho` | `k=30` cos | `k=231` `rho` | `k=231` cos | MRE |
|---|---|---|---|---|---|
| ~3.0 | +0.0385 | +0.470 | +0.4119 | +0.917 | 0.98 |
| ~0.9 | +0.2263 | +0.984 | +0.3009 | 0.997 | 0.98 |
| ~0.22 | +0.2677 | +0.999 | +0.3617 | +1.000 | 0.98 |
| ~0.07 | +0.2678 | +1.000 | +0.3712 | +1.000 | 0.98 |
| ~0.01 | +0.2702 | **+1.000** | +0.3739 | **+1.000** | 0.98 |

At `k=30`, `rank corr(r/R, rho) = −1.0000`: locality is the binding constraint, and removing
it is worth 7x (`+0.039 → +0.270`). At `k=231` locality is already satisfied and buys nothing.

**The last row is the important one.** With a perfectly local neighbourhood (`r/R = 0.01`) on a
fixture with 29x dynamic range, the estimator recovers the curvature DIRECTION exactly
(cosine `1.000`) while `rho` saturates at `0.37` and magnitude stays 50x attenuated.

That is not noise in the fixture and not a locality failure. It is a clean separation:

> **At `d=20`, curvature DIRECTION is recoverable essentially perfectly. Curvature MAGNITUDE
> ORDERING is only partially recoverable, and no amount of locality or `k` closes the gap.**

The asymmetry has a reason. A direction is a unit vector -- estimating it is a subspace
problem whose difficulty is controlled by the tangent-space estimate, which converges well.
A magnitude requires estimating a SCALE from `k` samples in `d` dimensions, which is exactly
where the dimension-dependent bias of the literature bites.

### Consequence for Phase 4, and a redesign worth considering

Phase 4 partitions the manifold by `|H|` **quantiles** — that is magnitude ordering, the
WEAKER of the two functionals, capped around `rho ≈ 0.4-0.65` even under ideal conditions.

**A partition built on curvature DIRECTION instead would use the functional that survives at
`d=20` with cosine ≈ 1.0.** Concretely: cluster the unit vectors `H/‖H‖` (or partition by
principal curvature directions) rather than binning `‖H‖`. Regions would then be "parts of the
manifold bending the same way" rather than "parts bending equally hard".

This is a real design fork and it is cheap to test on the fixtures already built — `ridge` has
a single known bending direction `w` by construction, so direction-clustering has an exact
known answer there. **Recommended before Phase 4 is planned**, not after.

---

## SELF-CORRECTION — dynamic range is not the determinant, and my PU calibration premise was wrong

I built the PU gate around a spread calibration: bowl 1.4x = unrankable, cubic 28x = rankable,
PU measured 4.8x, so "what `rho` does 4.8x support?" That framing is **wrong** and the
calibration run refutes it.

Ridge, `phase = pi/2`, `d=20`, `n=8000`, spread tuned continuously:

| spread | `r/R` | `rho` @k=120 | `rho` @k=231 | cosine |
|---|---|---|---|---|
| 36.0x | 3.0 | +0.279 | +0.361 | +0.894 |
| 16.5x | 1.8 | +0.545 | **+0.634** | +0.975 |
| 5.0x | 1.05 | +0.521 | +0.592 | +0.993 |
| 2.1x | 0.64 | +0.472 | +0.541 | +0.998 |
| 1.3x | 0.30 | +0.433 | +0.503 | +1.000 |
| 1.1x | 0.11 | +0.413 | +0.484 | +1.000 |

**At spread 1.1x — essentially constant curvature — `rho = +0.48`.** At spread 36x, `+0.36`.
Rankability does not track dynamic range, and in this fixture family it runs mildly the other
way.

The reason is elementary and I should have seen it before building the gate: **Spearman is
scale-free.** It asks whether the ordering is RESOLVABLE, not how wide it is. A 2% spread with
a smooth, coherent, noise-free ordering ranks fine; a 30x spread whose ordering is dominated by
estimator noise does not.

### What this does to the bowl explanation

The bowl still fails (`rho +0.03`), but **not because its spread is 1.4x**. It fails because its
`II` is constant, so its `||H||` variation is a ~3% metric-tilt perturbation sitting on a large
constant `tr(a) = d = 20`. The ridge at 1.1x spread has no such constant pedestal and its `II`
genuinely varies. **Constant-vs-varying `II` remains the real axis; spread was a red herring I
introduced.** Every other conclusion in this document rests on the `II` axis, not on spread,
and is unaffected.

### `rho` is non-monotone in `r/R`

It peaks near `r/R ~ 1.8` and declines in BOTH directions. Lowering frequency buys locality but
shrinks the curvature magnitude (`|H| ~ A * freq^2`), so signal-to-noise eventually falls. My
earlier "locality is the lever" claim held at `k=30` (`rank corr(r/R, rho) = -1.0000`) but does
not generalise -- at `k=120/231` there is an interior optimum.

### Consequence for the PU gate

PU's measured 4.8x spread is **much less informative than I claimed** when I wrote the gate.
The gate's `summarize()` still contains a `spread < 3.0` branch that would declare
"NEAR-CONSTANT CURVATURE"; on this evidence that branch is unsound and should be removed
before anyone reads a verdict from it.

What survives as evidence about PU is the **reliability** column, which was always the stronger
diagnostic:

| `k` | median `R_H` | frac neg | `r_dir` |
|---|---|---|---|
| 30 | 0.078 | 0.213 | +0.083 |
| 60 | 0.247 | 0.006 | +0.255 |
| 120 | 0.428 | **0.000** | +0.436 |

Two disjoint halves of the cloud agree on the sign of the curvature at every one of 1,000
anchors by `k=120`, with reliability rising monotonically in `k`. That is reproducible
structure. It is **not** proof of correctness -- this document's own headline finding is that
reliability and correctness come apart -- but it does rule out the "PU is a null" reading that
would have halted Phase 4 outright.

---

## PU pre-Phase-4 gate — FINAL

`centroid_mean_curvature` on the frozen Phase 1 10k subsample (`legacysurvey`, `D=768`),
`d=20`, 1,000 anchors held out of both halves so the two estimates are independent:

| `k` | spread | median `R_H` | frac sign-disagree | `r_dir` |
|---|---|---|---|---|
| 30 | 5.54x | 0.0779 | 0.213 | +0.083 |
| 60 | 4.83x | 0.2474 | 0.006 | +0.255 |
| 120 | 4.79x | 0.4280 | **0.000** | +0.436 |
| 231 | 4.86x | **0.5894** | 0.000 | — |

Reliability rises monotonically with `k` and clears the pre-declared 0.5 bound at `k=231`.
Sign disagreement is **zero** from `k=120` onward: two independent estimates, from disjoint
halves of the cloud, agree on the direction of curvature at every one of 1,000 anchors.

**Verdict: the "PU is a null" reading is ruled out.** The embedding is not the
constant-curvature case that would have halted Phase 4 regardless of estimator quality.

**Three limits on that verdict, all of them load-bearing:**

1. **Reproducible is not correct.** This document's headline finding is that split-half
   reliability certifies reproduction and never correctness -- measured directly on the Swiss
   roll at `R_H = 0.990` with `rho = 0.469`. A bias both halves share is perfectly reliable.
   There is no ground truth on PU, so this can never be upgraded by more of the same
   measurement.
2. **`k=231` is 2.3% of the cloud per neighbourhood.** Whether that is still "local" on the PU
   manifold is unmeasured. The `r/R` diagnostic exists (`ridge_frequency_sweep_run.py`) and has
   not been pointed at PU.
3. **The printed verdict said "RANKABLE AND REPRODUCIBLE".** That run predates the patch
   removing the unsound spread branch; "rankable" is NOT established and the wording overclaims.
   The current runner prints "REPRODUCIBLE" with the reliability-is-not-correctness caveat
   attached. Re-run for a clean record before quoting it anywhere.
