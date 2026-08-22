# High-Dimension Curvature Feasibility — What Is Settled at `d=20`

The substantive findings. Read this before proposing any estimator, prior, or architecture change
aimed at recovering curvature ordering at `d=20`, and before designing a control to test one on.

## Requirements

- **No shrinkage dial in `quadric_fit_curvature`.** D-05 rejected one because its strength becomes
  a value that must be chosen and pre-registered blind. A ridge variant, if ever built, lives in
  spike-local code and never edits `curvature_probe.py`. Held by user decision, 2026-08-21.
- **Nothing here reinterprets a sealed number.** The sealed `d=20` decoder row
  `rank_spearman_rho == -0.015106571347065712` stands as recorded. Phase 4 stays blocked; no route
  out is proposed.
- **Compare fixtures spread-for-spread, not name-for-name.**
- **Any future estimator claiming both determinedness and locality at `d=20` must show its `r/R`.**

## How to Build It

### The teacher already exists — do not write a new one

`curvature_probe.quadric_mean_curvature(X, k, d)` **is** the local-polynomial geometry teacher
`(P̂, ÎI)`, sealed and tested. Per point: `k`-NN excluding self; `P̂` from `_quadric_tangent_basis`
(SVD with `full_matrices=True`, so it returns exactly `d` rows even when `d > k`); tangent
coordinates `u = centered @ P̂ᵀ`; ambient normal residual `z = centered - u @ P̂`; then
`quadric_fit_curvature(u, z, d)` fits `z = q(u)` over `1 + d + d(d+1)/2` columns by minimum-norm
least squares and returns `H = tr(∇²q)` in ambient coordinates.

It was designated non-gating under D-05 on sample-complexity grounds and **had never been scored on
the four axes** — the record carried only its underdetermination flag. Spike 002 scored it.

### What the four axes say at `d=20` (`n=10000`, sealed saddle, `seed=20260816`)

| `k` | deficit | `r/R` | rank `rho` | median cosine | median ratio | calib `R²` | MRE |
|---|---|---|---|---|---|---|---|
| 30 | 180 | 0.8915 | **-0.028123** | -0.0261 | 1.4475 | 0.0010 | 0.8494 |
| 100 | 110 | 0.9701 | +0.022756 | -0.1510 | 6.6845 | 0.0003 | 5.6845 |
| 231 | **0** | **1.0331** | +0.083639 | -0.0302 | 223.99 | 0.0000 | 222.99 |
| 500 | 0 | **1.0992** | +0.392984 | **-0.3895** | 17.316 | 0.1457 | 16.316 |

**The directed question's answer: no.** At the sealed fixture and sealed `k=30`, teacher
`rho = -0.0281` against the sealed decoder's `-0.0151`. The teacher is not better; both are zero.

## What to Avoid

### Dead ends, measured — do not re-propose these

1. **Raising `k` until the quadratic fit is determined.** `r/R` crosses `1.0` at `k=231`; at
   `k=500` the "neighbourhood" is 10% wider than the manifold. Determined and local are mutually
   exclusive at `d=20`.
2. **Reading `rho = +0.393` at `k=500` as progress.** It is a fixture artifact. The saddle is
   globally a quadratic form (`f(x) = 0.5 xᵀQx`), so a local-quadric estimator whose ball covers
   everything fits the fixture's exact global model. Rank gain from `k=30 → 500`: **+0.421** on the
   saddle, **+0.051** on a non-quadratic fixture.
3. **More data.** `r/R ~ (k/n)^(1/d)`. At `k=231`, tripling `n` (10000 → 30000) moved `rho` by
   **+0.010** and `r/R` by −7.8%. The same lever moved `d=4` by **+0.058**. The `1/d` exponent is
   the whole mechanism.
4. **Trusting a rank statistic alone.** Direction is close to a coin flip at every `k` on every
   fixture: 52.3% of points anti-aligned at saddle `k=30`, **74.9%** at saddle `k=500`, 53.4% at
   bumps `k=30`. The cosine distribution is bimodal at ±1, not scattered about a centre.
5. **Trusting a median magnitude ratio.** It reaches `224` (saddle, `k=231`) and `421` (bumps,
   `k=231`) — wrong by orders of magnitude, beside a floored CV of `21.7`.

Also still ruled out, from `02.5-NOTE-high-d-curvature-approaches.md` and unchanged by these
spikes: graph smoothing / region averaging (§1a — error decorrelates at the signal's own spatial
scale), and any other point-cloud second-order estimator (§1b — all inherit the same `r/R` wall).

### The trap that cost two wrong conclusions

Spike 002's confound probe was built to show `+0.393` was fixture-specific. It did — **and
simultaneously refuted its own framing** by finding the same unmodified teacher scoring
`rho = +0.5934` at the *sealed* `k=30` on the Gaussian-bump fixture. A probe that only confirms
its hypothesis has not been run properly.

## Constraints

### The open question — for the developer, not for autonomous action

**The saddle control may be unable to show curvature ordering at `d=20` at all.**

| | saddle | bumps |
|---|---|---|
| `rho` at sealed `k=30` | **-0.0281** | **+0.5934** |
| partial `rho` controlling for local kNN radius | -0.0359 | **+0.6006** |
| `rho(||H_est||, r_knn)` | +0.0524 | +0.0120 |
| `||H_true||` spread (p95/p05) | 33.3× | 1095.3× |
| median relative error | 0.8494 | 0.8673 |

Not local sampling scale — removing it slightly *raises* the bumps headline. Not fully dynamic
range — at spread-matched windows the bumps hold `+0.21` to `+0.34` where the saddle scores zero in
**every** window, sign included:

| window | saddle spread → `rho` | bumps spread → `rho` |
|---|---|---|
| q0.25–q0.75 | 3.12 → +0.005261 | 16.44 → **+0.340013** |
| q0.35–q0.65 | 1.93 → +0.000865 | 4.74 → +0.210752 |
| q0.40–q0.60 | 1.54 → +0.005632 | 2.45 → **+0.149544** |

**Mechanism, readable from the fixture's source.** `synthetic_controls.make_saddle_control` sets
`hess = np.repeat(np.diag(signs)[None, None, :, :], n, axis=0)` — the analytic Hessian is
**constant at every point**. Its `||H||` varies only through the pullback metric
`g = I + ∇f∇fᵀ`, never through the second fundamental form. An estimator that recovered `ÎI`
perfectly would still have almost nothing to order on it.

That was a deliberate, well-motivated choice: the saddle was picked over the bump family precisely
so its near-zero region reads "positive and negative curvature cancelling in the trace" rather than
"flat here", exercising the failure mode `cond(g)` exists to disambiguate. **The cost of that
choice for the ordering axes appears nowhere in the record.** It matters because the sealed `d=20`
decoder verdict rests on the same fixture.

Treat this as an open question raised at spike 002's closing checkpoint. It is not a
reinterpretation of any sealed result and must not be used as one.

### Statistical limits of everything above

- **Single seed** (`20260816`) throughout, one fixture per family, no repetition. D-03's caution
  against reading three draws as an established property applies with more force to one.
- `rho = +0.5934` sits beside 87% median relative error and near-random direction. It is not a
  working estimator — only evidence that something orderable survives where the saddle reports
  nothing.
- Reference points for any future comparison: same fixture at `chart_dim = 4` yields `rho = 0.989`
  (`03-FINDINGS.md` §6); phase 03.1's best decoder-side result is `+0.116` (`scale`, strong rung),
  combination `+0.013`.

## Origin

Synthesized from spikes: 001, 002
Source files: `sources/001-teacher-low-d-anchor/`, `sources/002-teacher-d20-four-axes/`
