# 09-SUPPLEMENT-07 — the probe-facing curvature split: sphere term vs manifold shape, and the label's Hessian from data

**Status:** post-hoc, supplementary. **Not pre-registered. Feeds no verdict.** Nothing here changes any
Wave A record, Supplement 01–06, or the phase verdict. **Written:** 2026-09-12 UTC

## The question

Supplements 05 and 06 used `pf_curv = ‖⟨w_N, II⟩‖_g` with `II` the *full* second fundamental form of
the sphere-projected surface in `R^768`. Every submanifold of the unit sphere carries the sphere's own
second fundamental form: `⟨x̂, II_ij⟩ = −g_ij` exactly (checked below to 1e-5 on the fixture, 1e-15 on
the smoke decoder). So

```
⟨w_N, II⟩ = ⟨w_N, II^S⟩ − (w·x̂) g,          ‖(w·x̂) g‖_g = √d |ŷ − b₀|
```

The second term is the prediction level at the anchor times the metric. It is a genuine part of the
Hessian of `w·x` restricted to `M` (a linear readout of a unit-normalized embedding is bent by the
sphere in proportion to its own value), but it says nothing about the shape of the embedding
manifold, and it was not separated in Supplements 05–06. This supplement splits the two and, on the
Physics anchors, estimates the label's intrinsic Hessian from the data so that the residual
expansion's variable `Δ = Hess_M y − ⟨w_N, II⟩` and the alignment `cos_g(Hess_M y, ⟨w_N, II⟩)` become
available on real data.

A second correction. The expansion quoted in Supplement 05 and in the manuscript's Eq. (2),
`E[r²] = c² + s²‖δ‖² + (s⁴/4)[(tr Δ)² + 2‖Δ‖²_F] + O(s⁵)`, omits the cross term `c s² tr Δ`
(`E[Δ(u,u)] = s² tr Δ ≠ 0`). Monte Carlo at `d = 6`, `s = 0.3`: measured 0.7307, quoted formula 0.6201.
With `c̄` the patch-mean residual the correct Gaussian-patch form is
`E[r²] = c̄² + s²‖δ‖² + ½ s⁴ ‖Δ‖²_F + O(s⁴‖δ‖) + O(s⁶)` (Monte Carlo 0.7312); the `(tr Δ)²` piece is
absorbed by `c̄`. Only the fixture column `pred_resid` of Supplement 05 used the wrong form; no
partial in any table depends on it. The manuscript equation is corrected.

## 1. Fixture (exact geometry)

Runner `notebooks/diagnostics/09_fixture_probe_facing_split_run.py` (imports
`09_fixture_probe_facing_run.py` unchanged; same pool, seeds, samples, labels, probe, controls;
Freedman–Lane with 1,000 draws), executed locally at 16 threads, `γ ∈ {−1, 0.6}` (the steep and the
zero-coupling samplings of Supplement 05). Record `notebooks/.cache/09_fixture_probe_facing_split.jsonl`.

Sealed three-control partial against local R²; `p ≤ 0.004` unless marked.

| γ | coupling | label | `pf_full` | `pf_rad = √d|ŷ−b₀|` | `pf_tan = ‖⟨w_N,II^S⟩‖` | `‖Δ‖` | ρ(`pf_full`,`pf_rad`) | median `pf_tan` / `pf_rad` |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| −1 | +0.79 | intrinsic-linear | −0.79 | −0.78 | −0.40 | −0.71 | 0.998 | 0.04 / 1.70 |
| 0.6 | +0.05 | intrinsic-linear | −0.78 | −0.79 | +0.18 | −0.58 | 0.996 | 0.07 / 1.40 |
| −1 | +0.79 | nonlinear | 0.00 n.s. | 0.00 n.s. | 0.00 n.s. | −0.19 | 1.000 | 0.001 / 6.6 |
| 0.6 | +0.05 | nonlinear | +0.07 n.s. | +0.07 n.s. | +0.16 | −0.56 | 1.000 | 0.05 / 3.5 |

Reading.

1. **The fixture's stable −0.74…−0.80 (Supplement 05, manuscript Fig. 1a) is the sphere term.**
   `pf_full` and `√d|ŷ−b₀|` are rank-identical (0.996–1.000). On this generator the in-sphere
   probe-facing curvature is 20–40× smaller than the sphere term (`‖w_N‖/‖w‖` median 0.13–0.14; the
   bumps span one in-sphere normal direction), so the fixture cannot resolve it. What the fixture
   validated is the expansion's prediction for the sphere term: an intrinsically linear label read
   by a global linear probe on a sphere loses local R² where `|ŷ − b₀|` is large, stably across
   samplings that flip the `‖H_tan‖` partial.
2. **Where the in-sphere term is measurable it follows `‖H_tan‖`** (ρ = 0.84–0.89 with `‖H_tan‖`,
   partial −0.40 at coupling +0.79, +0.18 at zero coupling): the one-normal-direction fixture has no
   room for the two to differ.
3. **For the nonlinear label the sphere term carries nothing** (partial 0.00–0.07, n.s.); `‖Δ‖`
   carries −0.19 to −0.56, as in Supplement 05.

## 2. Physics anchors (decoder geometry, label Hessian from data)

Runner `notebooks/diagnostics/09_physics_probe_facing_split_run.py`, executed on the pod
(`universetbd-0`, 16 threads) from the stored per-anchor geometry of Supplement 06
(`probe-facing-out/probe-facing/09_probe_facing_geometry_d{16,20}.npz`; no decoder refit), runner
sha256 `43547001…` copied onto the clone at `71914dd` (untracked there). Record
`09_physics_probe_facing_split.jsonl`.

Added per anchor and label, beside the split: a least-squares quadratic of `y` on the tangent-projected
coordinates `u = g⁻¹Jᵀ(x − x₀)` of the 2,048 neighbours. In those coordinates
`x − x₀ = J u + ½ II(u,u) + …` with `II` normal, so the fitted quadratic coefficient is the covariant
Hessian `Hess_M y` directly (Christoffel term absorbed). The same regression applied to the probe's own
prediction `w·x` returns a data-side estimate of `⟨w_N, II_data⟩`, which is compared with the decoder's
`⟨w_N, II⟩` (rank of norms; per-anchor metric cosine).

Provenance. The first two launches hung in `physics_labels.load_label_table` (pyarrow over `hf://`
through `HfFileSystem`; all threads in `futex_wait`, one in `ssl.read`, no progress for 8 minutes; the
same read also stalled from the local machine). The 16 label shards were instead fetched with
`huggingface_hub.hf_hub_download` (74 s), column-projected and concatenated in shard order into
`/mnt/ssd-cluster/effdim/labels_Smith42_galaxies_v2.0_test.parquet` (86,471 rows, sha256
`60f2f82e64e4036e…`), and the runner's `--label-table` option substitutes that frame for the
`hf://` read (recorded in the environment row). Global OOF R² per label reproduces the sealed run
exactly (0.516, 0.508, 0.534, 0.477). Wall clock 1,276 s. Record sha256
`09e869ff61a1626a24773437834aa66dd3de02c4fdae80ec4498ece243e7be65`, copied to
`notebooks/.cache/09_physics_probe_facing_split.jsonl`.

## 3. Results

Multi-scale density-control partial against local R² (sealed three-control partial in parentheses
where it differs in significance); `*` = p > 0.05.

| label | d | `‖H_tan‖` | `pf_full` | **`pf_tan` (shape)** | **`pf_rad` (sphere)** | `‖Hess_M y‖` | **`‖Δ‖` (emp)** | **align cos** | `cross` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mag_r | 16 | +0.25 | +0.10 | +0.12 | −0.26 | −0.28 | −0.39 | +0.35 | +0.23 |
| mag_r | 20 | −0.17 | +0.15 | +0.19 | −0.26 | −0.34 | −0.40 | +0.43 | +0.33 |
| photo_z | 16 | +0.28 | −0.13 | −0.06* | −0.33 | −0.37 | −0.45 | +0.26 | +0.06* |
| photo_z | 20 | +0.13 | −0.22 | −0.12 | −0.36 | −0.39 | −0.44 | +0.24 | −0.00* |
| smooth_fraction | 16 | +0.21 | +0.06* | +0.07* | −0.13 | −0.07* | −0.09* (p=0.058) | +0.19 | +0.14 |
| smooth_fraction | 20 | +0.15 | +0.13 | +0.16 | −0.15 | −0.17 | −0.17 | +0.07* | +0.08* |
| stellar_mass | 16 | −0.02* | −0.05* | +0.00* | −0.35 | −0.01* | −0.04* | +0.01* | +0.01* |
| stellar_mass | 20 | −0.00* | −0.04* | +0.03* | −0.35 | −0.06* | −0.09* (p=0.050) | +0.02* | −0.01* |

Checks per (label, d): `⟨x̂, II⟩ = −g` to 1.5e-8; `Jᵀx̂ = 0` to 2e-9; `H_rad` median −16 / −20;
rank(`pf_full`, `pf_tan`) 0.956–0.992; rank(`pf_full`, `pf_rad`) 0.07–0.65; median `pf_tan` / `pf_rad`
18.3/1.73 (mag_r), 1.77/0.21 (photo_z), 4.8/1.1 (smooth), 16.6/2.3 (stellar) at d=16, similar at d=20;
`‖w_N‖/‖w‖` median 0.80–0.87. Decoder `⟨w_N,II⟩` against the data-side quadratic coefficient of `w·x`:
median metric cosine 0.55–0.65, rank of norms 0.15–0.56; the quadratic term raises the fit of `w·x` on
the patch from R² 0.86–0.92 (linear) to 0.90–0.95. Label quadratic fits gain 0.10–0.17 in R² over linear.

## 4. Reading

1. **On the galaxies the shape term dominates** (rank 0.96–0.99 with the full norm; 8–14× larger
   than the sphere term), the reverse of the fixture. Supplement 06's `pf_curv_dec` partials are
   therefore shape-term partials, and its label-dependent sign stands: `pf_tan` +0.12/+0.19 (mag_r),
   −0.06*/−0.12 (photo_z), +0.07*/+0.16 (smooth), null (stellar).
2. **The sphere term is negative on every label and both d** (−0.13 to −0.36), the sign the fixture
   gave (−0.78). A global linear readout of a unit-normalized embedding loses local R² where its own
   value is far from the intercept. Caveat: a global ridge also shrinks extreme predictions, and this
   design does not separate the two.
3. **The expansion's variable `‖Δ‖ = ‖Hess_M y − ⟨w_N,II⟩‖` is negative wherever the label has local
   structure** (−0.39/−0.40, −0.45/−0.44, −0.09*/−0.17) and null for stellar mass, whose label has no
   local quadratic structure the probe can miss (`‖Hess_M y‖` partial −0.01/−0.06*). `‖Δ‖` tracks
   `‖Hess_M y‖` closely (the label's own Hessian is 6–60× the shape term), so most of this is "the
   linear probe fails where the label is intrinsically curved", as it must.
4. **The alignment explains the label-dependent sign of the shape term.** `cos_g(Hess_M y, ⟨w_N,II⟩)`
   is positive on every structured label (+0.19 to +0.43): local R² is higher where the manifold bends
   in the readout's direction the way the label bends. In `‖Δ‖² = ‖Hess y‖² − 2⟨Hess y, pf⟩ + ‖pf‖²`
   the shape term's own partial follows the cross term where that is nonzero (mag_r +0.23/+0.33 →
   shape +; smooth +0.14/+0.08 → shape +) and is null or negative where the cross term is null
   (photo_z, stellar), as the remaining `+‖pf‖²` predicts. This converts Supplement 06's
   "consistent with the theory, not tested by it" into a tested prediction.
5. **Decoder II in the readout direction agrees moderately with the point cloud's own quadratic
   coefficient** (median cosine 0.55–0.65; rank of norms 0.15–0.56). This is the first real-data check
   of the decoder's II and it is not strong; the 2,048-patch quadratic is itself a coarse estimate
   (`r/R ≈ 1`).

## 5. What this does not settle

The sphere term versus ridge shrinkage (point 2). The noise of the local-quadratic `Hess_M y` (label
R² gain 0.10–0.17). The `d = 25, 32` cells. Anything about the two-embedding alignment phases.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Supplement 07 — post-hoc, not pre-registered, feeds no verdict. Fixture record `09_fixture_probe_facing_split.jsonl`; Physics record sha256 `09e869ff…7be65`; runners untracked at `71914dd` on both hosts.*
