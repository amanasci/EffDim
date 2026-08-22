# Curvature-Estimator Validation Protocol

The reusable method. This is how to find out whether a curvature or geometry estimator works at
a given intrinsic dimension, and — more importantly — how to make its FAILURE interpretable. Both
spikes in this set produced results only because the protocol below was followed; the first two
conclusions each drew were wrong and were caught by it.

## Requirements

- **Score with the sealed scorer, unmodified.** `synthetic_control_run._fidelity_axes` supplies all
  four axes (direction median cosine; magnitude median ratio *and* CV; calibration slope/intercept/
  `R²`; rank Spearman `rho`). Numbers must be comparable to sealed rows by construction, not by
  re-derivation.
- **`CURVATURE_CONVENTION = "trace"`.** `H = tr_g(II)`, unnormalized; a unit `d`-sphere gives
  `||H|| = d`. The averaged convention differs by exactly a factor of `d`. This codebase has
  already shipped and fixed one factor-of-`d` bug.
- **Never edit `notebooks/pu_manifold/` or `notebooks/diagnostics/` from a spike.** Import
  unchanged. An estimator that only works after the sealed module is rewritten is itself the
  finding.
- **Anchor at low `d` before interpreting any failure at high `d`.**
- **Never report a rank statistic without the direction axis beside it.**

## How to Build It

### 1. Anchor first — the run that makes a later FAIL mean something

A FAIL at high `d` has at least four causes: the dimension wall, a mis-wired fixture, a convention
slip, or a scorer fed vectors in mismatched frames. Only a fixture whose answer is known separates
them. Run the estimator where it MUST succeed before running it where you expect it to fail.

```python
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "notebooks" / "diagnostics"))
import synthetic_control_run as scr        # self-bootstraps pu_manifold onto sys.path
from pu_manifold import curvature_probe, synthetic_controls

out  = curvature_probe.quadric_mean_curvature(X, k=30, d=2)
axes = scr._fidelity_axes(out["H_vec"], H_true)     # the four axes, unmodified
```

Use the **sealed protocol's own `n`** (`CONTROL_N = 10000`) and `k`, so `n` never confounds a `d`
comparison. Spike 001's first run used `n=3000` and produced a false failure at `d=4`.

### 2. State the pass regime in `r/R`, not in `d`

Every neighbourhood-based estimator inverts an identity exact only as `r → 0`. The regime where it
must succeed is the regime where the neighbourhood is genuinely local. Measure it, don't quote it:

```python
def measure_r_over_R(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    r_knn = float(np.median(dist[:, -1]))
    R = float(np.median(np.linalg.norm(X - X.mean(axis=0), axis=1)))
    return r_knn / R
```

Recomputing this independently reproduced `02.5-NOTE-high-d-curvature-approaches.md` §1's table
twice (Swiss roll `0.1158` vs recorded `0.115`; `d=20, k=30` `0.8915` vs recorded `0.906`), which
is what made the third, unrecorded measurement believable.

### 3. Write the decision rule into the script before running it

Every probe prints its own verdict from a threshold fixed in the source, with the reasoning in the
module docstring. Two probes then refuted the hypothesis that motivated writing them — only
credible because the rule predated the data.

```python
if gain_bumps < 0.5 * gain_saddle:
    print("CONFOUNDED. The rank gain is largely specific to the globally-quadratic saddle.")
else:
    print("SURVIVES. ...")
```

### 4. Run the three confound probes before believing any positive result

Each is one hypothesis, one file, named for what it isolates. All three were necessary; each
changed the conclusion.

**(a) Fixture-structure confound** (`probe_confound.py`). Does the result depend on the fixture's
own functional form? Rerun the identical ladder on a fixture from a different family — everything
else held (`d`, `n`, `D`, seed, estimator, scorer). A local-*quadric* estimator on a *globally
quadratic* fixture will recover the fixture's exact global model once its ball is large enough,
and report it as a win.

**(b) Local-scale confound** (`probe_scale_confound.py`). Is the ordering curvature, or local
sampling density? Neighbourhood size is not independent of geometry — a graph fixture thins out
where it is steep — so `||H_est||` can rank-correlate with `||H_true||` while estimating nothing.
Spearman partial correlation against the local kNN radius:

```python
def partial_spearman(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    Xc = np.stack([rc, np.ones_like(rc)], axis=1)
    res_a = ra - Xc @ np.linalg.lstsq(Xc, ra, rcond=None)[0]
    res_b = rb - Xc @ np.linalg.lstsq(Xc, rb, rcond=None)[0]
    return float(np.corrcoef(res_a, res_b)[0, 1])
```

**(c) Dynamic-range confound** (`probe_dynamic_range.py`). Ranking is pairwise, so difficulty
depends on how far apart the true values are relative to estimator error. Compare fixtures
**spread-for-spread, not name-for-name**: restrict to contiguous quantile windows of `||H_true||`
and report each window's realized `p95/p05` beside its `rho`. Same estimates, only which points
are compared changes — so it costs one extra pass, not one extra fit.

### 5. Time before gridding

`_quadric_tangent_basis` calls `svd(..., full_matrices=True)`, materializing a `(D, D)` matrix per
point. A single timing probe at `n=600` turned a ~20-hour grid into a 30-minute one.

| `D` | `k` | wall at `n=600` | extrapolated `n=10000` |
|---|---|---|---|
| 28 | 30 | 1.33s | ~22s |
| 28 | 231 | 13.2s | ~220s |
| 768 | 30 | 38.9s | ~390s |
| 768 | 231 | 1297.5s | **~6 h** |

Then license the cheap substitution rather than assuming it — see Constraints.

### 6. Record revised criteria; do not silently apply them

When spike 001's thresholds turned out wrong, the revision note went into the docstring of the file
that now passes, so the original failure is visible from the passing artifact.

## What to Avoid

- **Do not gate on the raw magnitude-ratio CV on a mixed-sign fixture.** The saddle's trace cancels
  by construction and `MIN_TRUE_NORM = 1e-12` excludes none of it. Measured at `d=2`: raw CV
  `1.31`, floored CV `0.079`, median ratio pinned at `0.998` throughout. Report both; gate the
  floored one.
- **Do not read a median ratio without its CV**, and do not read either without the direction axis.
  At `d=20`, 52–75% of points carried an anti-aligned `H` in cells whose `rho` looked usable.
- **Do not treat "the estimator degrades at high `d`" as a cliff at the target `d`.** It is a slope
  already visible at `d=4`: `r/R` `0.094 → 0.321`, `rho` `0.998 → 0.845`, calibration `R²`
  `0.996 → 0.750`.
- **Do not raise `k` until the fit is determined and call the result an improvement.** At `d=20`,
  `r/R` crosses `1.0` at `k=231` — determined and local are mutually exclusive.
- **Do not reach for more data at high `d`.** `r/R ~ (k/n)^(1/d)`. Tripling `n` moved `rho` by
  `+0.058` at `d=4` and `+0.010` at `d=20`.
- **Do not use the system python.** Several sealed modules import torch at module scope even when
  the spike path is pure numpy. Use `.venv/bin/python`.

## Constraints

- **`D = 28` is a licensed substitute for `D = 768` for `quadric_mean_curvature` only, and only
  under the Part A check.** Measured worst disagreement across all four axes: `1.288e-14`, at a
  204× speedup. Re-measure before relying on it for a different estimator. The a-priori argument
  (fixtures draw `X_local` before the rotation `Q`; `Q` is orthogonal, so norms, cosines and ranks
  are invariant) holds, but was measured anyway.
- Set the invariance tolerance at `1e-6`, not bit-identity: different `Q` means different float
  rounding, and a kNN near-tie can flip one neighbour. Realized agreement was eight orders tighter.
- `make_swiss_roll_fixture` returns `H_norm` only. The analytic `H` **vector** needed for the
  direction axis is derived in `001-teacher-low-d-anchor/run_anchor.py`, pinned against the
  fixture's own `H_norm` at `1e-12`.
- `_fidelity_axes` returns `rank_spearman_rho = None` and null calibration when the analytic field
  is constant (sphere). Handle the null; do not coerce it to a number.

## Origin

Synthesized from spikes: 001, 002
Source files: `sources/001-teacher-low-d-anchor/`, `sources/002-teacher-d20-four-axes/`
