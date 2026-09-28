---
spike: 002
name: teacher-d20-four-axes
type: standard
validates: "Given the sealed d=20 saddle control, when the unmodified local-polynomial teacher fits (P̂, ÎI) across k spanning the underdetermined and determined regimes, then the four axes say whether a geometry-supervised objective has anything to teach with at d=20"
verdict: PARTIAL
related: [001]
tags: [curvature, d20, kill-test, locality, confound, fixture-validity]
---

# Spike 002: Teacher at `d=20`, Four Axes

## What This Validates

**Given** the sealed `d=20` saddle control (`make_saddle_control`, `n=10000`, `seed=20260816`);
**when** `curvature_probe.quadric_mean_curvature` fits `(P̂, ÎI)` unmodified across `k ∈ {30, 100,
231, 500}`, spanning the underdetermined and determined regimes, and is scored by
`synthetic_control_run._fidelity_axes` unmodified;
**then** the four axes say whether a geometry-supervised objective has anything to teach with at
`d=20` — the developer-directed question from `03.1-FINDINGS.md` §10.

## How to Run

```bash
.venv/bin/python .planning/spikes/002-teacher-d20-four-axes/run_d20.py                 # ~30 min
.venv/bin/python .planning/spikes/002-teacher-d20-four-axes/probe_confound.py          # ~19 min
.venv/bin/python .planning/spikes/002-teacher-d20-four-axes/probe_scale_confound.py    # ~40 s
.venv/bin/python .planning/spikes/002-teacher-d20-four-axes/probe_dynamic_range.py     # ~40 s
```

Recorded output sits beside each script as `*.out`.

## What to Expect

`run_d20.py` prints Part A (the ambient-dimension licence), Part B (the `k` ladder), Part C (the
`n` lever), then a verdict comparing the best teacher `rho` against the sealed decoder row. The
three probes each end in a printed decision rule that was written before the result was seen.

## Research

**Ambient dimension is not free for this estimator, and that shaped the design.**
`_quadric_tangent_basis` calls `np.linalg.svd(..., full_matrices=True)`, which materializes a
`(D, D)` matrix at every point. Measured, `n=600`:

| `D` | `k` | wall | extrapolated to `n=10000` |
|---|---|---|---|
| 28 | 30 | 1.33s | ~22s |
| 28 | 231 | 13.2s | ~220s |
| 768 | 30 | 38.9s | ~390s |
| **768** | **231** | **1297.5s** | **~21,600s (6 h)** |

So the grid runs at `D=28` under a licence measured in Part A rather than inherited from
`02.5-NOTE` §2a's bit-identity claim for the *gating* estimator.

**`k = 231` is the first determined fit** (`1 + d + d(d+1)/2` columns). Including it separates
"underdetermined" from "non-local" as explanations — every prior `d=20` quadric run in the record
was underdetermined, so the two have never been distinguished.

## Investigation Trail

**Iteration 1 — Part A, the licence.** `D=768` vs `D=28` at `n=600`, `k=30`: worst absolute
disagreement across all eight reported quantities **`1.288e-14`**, at a **204×** speedup. The
argument behind it also holds a priori (`make_saddle_control` draws `X_local` before `Q`, so
`X_local` is identical across `D`; `Q` is orthogonal, so distances, norms, cosines and ranks are
invariant), but it was measured rather than assumed. Tolerance was set to `1e-6` rather than
bit-identity, because the two fixtures carry different `Q` and a kNN near-tie can flip a
neighbour; the realized agreement was eight orders tighter than needed.

**Iteration 2 — Part B, the `k` ladder, `n=10000`, `D=28`.**

| `k` | deficit | `r/R` | rank `rho` | median cosine | median ratio | CV floored | calib `R²` | MRE |
|---|---|---|---|---|---|---|---|---|
| 30 | 180 | 0.8915 | -0.028123 | -0.0261 | 1.4475 | 1.313 | 0.0010 | 0.8494 |
| 100 | 110 | 0.9701 | +0.022756 | -0.1510 | 6.6845 | 1.286 | 0.0003 | 5.6845 |
| 231 | **0** | **1.0331** | +0.083639 | -0.0302 | 223.99 | 21.745 | 0.0000 | 222.99 |
| 500 | 0 | **1.0992** | **+0.392984** | **-0.3895** | 17.316 | 1.074 | 0.1457 | 16.316 |

Two facts land immediately. **§1's `d=20` row reproduces independently**: `k=30` gives
`r/R = 0.8915` and `MRE = 0.8494` against the note's `0.906` and `0.870`, recomputed from scratch
rather than quoted. And **determinedness is bought by abandoning locality** — `r/R` crosses `1.0`
at `k=231`, so the "neighbourhood" is larger than the manifold's own median radius. §2a's "`k=210`
is not a local neighbourhood" is now a number: at the first determined `k`, the ball is 3% wider
than the manifold, and at `k=500` it is 10% wider.

**Iteration 3 — Part C, the `n` lever, tested rather than argued.** At `k=231`, tripling `n` from
10000 to 30000 moved `rho` from `+0.0836` to `+0.0938` (**+0.010**) and `r/R` from `1.0331` to
`0.9523` (−7.8%, against §1's predicted 11%). Spike 001 measured the identical lever moving `d=4`
by **+0.058**. The `1/d` exponent is doing exactly what §1 says it does.

**Iteration 4 — the headline looked like a result, and it was not.** `rho = +0.393` at `k=500` is
3.4× phase 03.1's best (`+0.116`) and would be the largest `d=20` ordering signal this project has
produced. Two reasons to distrust it: it arrives precisely as the ball grows past the whole
manifold, and **this fixture is globally a quadratic form** (`f(x) = 0.5 xᵀQx`), so a local-QUADRIC
estimator covering everything fits the fixture's exact global model. `probe_confound.py` reran the
ladder on `make_graph_of_function_fixture` (three Gaussian bumps, not globally quadratic), with the
decision rule written before the run:

| `k` | saddle (quadratic) | bumps (not quadratic) |
|---|---|---|
| 30 | -0.028123 | **+0.593417** |
| 231 | +0.083639 | +0.423580 |
| 500 | +0.392984 | +0.644656 |

Rank gain `k=30 → 500`: saddle **+0.421**, bumps **+0.051**. **The `k`-dependence is confirmed as a
saddle artifact.** But the bumps fixture did not collapse — it scores higher at every `k`, and its
best result is at the *sealed* `k=30`. The confound was real and the conclusion it was supposed to
support was wrong.

That probe also surfaced a fact the four-axis summary hides: **direction is close to a coin flip
everywhere.** Fraction of points whose estimated `H` is anti-aligned with the true `H`: saddle
`k=30` **52.3%**, saddle `k=500` **74.9%**, bumps `k=30` **53.4%**. The cosine distribution on the
bumps runs `p05 = -0.9991` to `p95 = +0.9994` — bimodal at ±1, not scattered around a centre.
Rank ordering can look respectable while the vectors point the wrong way, which is exactly the
blindness `02.5-NOTE` §2d warned rank statistics carry.

**Iteration 5 — is `+0.593` curvature, or local sampling scale?** §1a records that the `d=20`
error "behaves like noise while being bias", set by the size and shape of each neighbourhood — and
neighbourhood size is not independent of geometry, since a graph fixture thins out where it is
steep. So `||H_est||` could rank-correlate with `||H_true||` while estimating no curvature, if both
track the local kNN radius. `probe_scale_confound.py`, Spearman partial correlation:

| quantity | saddle | bumps |
|---|---|---|
| `rho(||H_est||, ||H_true||)` | -0.028123 | **+0.593417** |
| `rho(||H_est||, r_knn)` | +0.052408 | **+0.012026** |
| `rho(||H_true||, r_knn)` | +0.139921 | -0.135193 |
| **partial `rho(est, true \| r_knn)`** | -0.035858 | **+0.600600** |

**Excluded.** The estimate is essentially uncorrelated with local scale (`+0.012`), and removing
scale slightly *raises* the headline. The teacher is ranking something genuinely second-order.

**Iteration 6 — dynamic range, the last difference between the fixtures.** `||H_true||` spread
(p95/p05) is **1095×** on bumps and **33×** on the saddle, while median relative error is
near-identical on both (`0.8673` vs `0.8494`). Ranking is pairwise, so a field spanning three
orders of magnitude survives 87% error where a field spanning 33× does not.
`probe_dynamic_range.py` cuts the bumps field to saddle-like windows using the same estimates —
only which points are compared changes:

| window | saddle spread | saddle `rho` | bumps spread | bumps `rho` |
|---|---|---|---|---|
| full | 33.26 | -0.028123 | 1095.34 | +0.593417 |
| q0.00–q0.50 | 20.19 | -0.005811 | 5.72 | +0.280313 |
| q0.25–q0.75 | 3.12 | +0.005261 | 16.44 | **+0.340013** |
| q0.50–q1.00 | 3.27 | -0.030734 | 669.62 | +0.470471 |
| q0.35–q0.65 | 1.93 | +0.000865 | 4.74 | +0.210752 |
| q0.40–q0.60 | 1.54 | +0.005632 | 2.45 | **+0.149544** |

Read spread-for-spread rather than row-for-row. Dynamic range explains most of the bumps' absolute
level (`+0.593 → +0.150` as spread falls `1095× → 2.45×`). **It does not explain the saddle**,
which posts zero ordering — sign included — in every window at every spread.

## Results

**PARTIAL.** Three findings, and the third is the one that matters.

**1. On the directed question as posed, the answer is no.** At the sealed fixture and the sealed
`k=30`, the teacher scores `rho = -0.0281` against the sealed decoder's `-0.0151`. The teacher is
**not better than the decoder** — both are indistinguishable from zero. A geometry-supervised
objective built on this teacher, on this control, has nothing to teach with.

**2. The one number that looked like a route out is an artifact, and was killed by a rule written
before it was tested.** `rho = +0.393` at `k=500` requires `r/R = 1.0992` — a "neighbourhood"
wider than the manifold — and is specific to the saddle's global quadratic form (rank gain
`+0.421` on the saddle vs `+0.051` on a non-quadratic fixture). It must not be carried forward as
evidence that larger `k` helps at `d=20`.

**3. The saddle control cannot detect curvature ordering at `d=20`, and its own construction is a
candidate reason.** On the bump fixture, at the *sealed* `k=30`, the same unmodified teacher scores
`rho = +0.5934` — not explained by local sampling scale (partial `rho = +0.6006`), and still
`+0.21` to `+0.34` at windows whose dynamic range matches the saddle's, where the saddle scores
zero. The saddle differs in a way readable directly from its source: `make_saddle_control` sets
`hess = np.repeat(np.diag(signs)[None, None, :, :], n, axis=0)` — **the analytic Hessian is
constant at every point by construction.** Its `||H||` varies only through the pullback metric
`g = I + ∇f∇fᵀ`, not through the second fundamental form. An estimator that recovered `ÎI`
perfectly would still have almost nothing to order on it.

That property was deliberate and well-motivated: the saddle was chosen over the bump family
precisely so its near-zero region reads "positive and negative curvature cancelling in the trace"
rather than "flat here", to exercise the failure mode `cond(g)` exists to disambiguate. The cost
of that choice for the *ordering* axes appears not to have been measured anywhere in the record.

### Surprises

1. **The confound probe refuted its own hypothesis and found something better.** It was built to
   show `+0.393` was fixture-specific; it did, and simultaneously showed the teacher scoring
   `+0.593` at the sealed `k` on a fixture nobody had run it against.
2. **Direction is near-random on every fixture at every `k`** (52–75% of points anti-aligned),
   including cells whose rank `rho` looks usable. Any downstream use of this teacher that reads
   ordering without checking direction would be reading a coin flip.
3. **The magnitude axis is not merely wrong, it is wrong by orders of magnitude** — median ratio
   `224` at `k=231` on the saddle, `421` on the bumps. Reported alongside a floored CV of `21.7`.
   Spike 001's `d=4` warning (magnitude breaks last) inverts at `d=20`: here magnitude is
   catastrophic while rank sometimes survives.

### What this does NOT establish

- **It does not reinterpret any sealed number.** The sealed `d=20` decoder row stands exactly as
  recorded. Finding 3 raises a question about the *fixture's* sensitivity on the ordering axes; it
  is a question for the developer, not a re-reading, and nothing here touches Phase 2/02.1/02.2/
  02.4/02.5/03/03.1 verdicts.
- **It does not unblock Phase 4.** No route out is proposed. The teacher does not supervise at
  `d=20` on the control the roadmap gates on.
- **It does not establish that the bump fixture is the right control.** One fixture, one seed, no
  repetition. `rho = +0.593` with 87% magnitude error and near-random direction is not a working
  estimator; it is a signal that something orderable survives where the saddle says nothing does.
- **Single seed throughout.** Every cell here is `seed=20260816`. D-03's standing caution against
  reading three draws as an established property applies with more force to one.
