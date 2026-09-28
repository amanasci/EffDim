# 09-SUPPLEMENT-05 — probe-facing curvature: the residual theorem, term by term, on the fixture

**Status:** post-hoc, supplementary. **Not pre-registered. Feeds no verdict.** Nothing here changes
any Wave A record, Supplement 01–04, or the phase verdict. No sealed constant is reinterpreted.
**Written:** 2026-09-11 UTC

## The question

Supplement 03 and Supplement 04 § 8 measured the sealed verdict statistic (controlled partial of
exact `‖H_tan‖` against a global ridge probe's local R²) on the adjudication fixture across six
samplings, and found a curve in the density–curvature coupling: −0.29 at coupling +0.80, a
plateau of about +0.22 for coupling below +0.15. Why an intrinsic-linear label reads *positive* at
zero coupling was left open.

A second-order expansion of the residual says what a global affine probe can and cannot do on a
curved surface, and names the quantities involved. This supplement computes those quantities per
anchor from exact geometry and asks which of them carries the association.

## 1. The theorem

Global affine probe `ŷ = w·x + b₀` on a `d`-manifold `M ⊂ R^D`. At an anchor with orthonormal
tangent frame, second fundamental form `II` and normal part `w_N` of `w`:

```
∇_M ŷ = Jᵀw,        Hess_M ŷ = ⟨w_N, II⟩,        Δ_M ŷ = ⟨w, H⟩
```

(the restriction of a linear map to a submanifold is as non-linear as the manifold bends in the
direction of `w`; the last identity is `Δ_M x = H`). For the residual `r = y − ŷ` over an
isotropic patch with second moment `s²`, with `c` the residual at the anchor, `δ = ∇_M y − ∇_M ŷ`
and `Δ = Hess_M y − ⟨w_N, II⟩`, all in the induced metric:

```
E_patch[r²] = c² + s²‖δ‖² + (s⁴/4)[(tr Δ)² + 2‖Δ‖²_F] + O(s⁵),     local R² = 1 − E[r²]/Var_patch(y)
```

Consequences. The curvature that costs a linear probe is `⟨w_N, II⟩`, the second fundamental form
seen from the probe's normal direction. `‖H‖` is its trace's upper bound and need not correlate
with it when the normal space is large. The term scales as `s⁴`, so patch radius (density)
multiplies it. For a label with intrinsic Hessian, the relevant object is the mismatch `Δ`, not
the curvature alone.

## 2. What was computed

Runner `notebooks/diagnostics/09_fixture_probe_facing_run.py`, executed locally at 16 threads
(exact geometry only; no autoencoder, no colleague estimator). It imports
`09_fixture_probe_decodability_run.py` unchanged and rebuilds every sample bit-for-bit (same pool,
seeds, `‖H_tan‖^γ` weighting, anchors, labels, sealed OOF probe, sealed local R², sealed controls
and Freedman-Lane partial with 2,000 draws). `γ ∈ {−1, 0, 0.4, 0.6, 0.8, 1}`, the union of
Supplements 03 and 04 § 8. Record `notebooks/.cache/09_fixture_probe_facing.jsonl`, 19 rows, sha256
`d6396ca012a09006…`.

Geometry at the 512 anchors is computed in the generator's unrotated `R^18` by central finite
differences (first and second derivatives of `f(z) = normalize([stereo(z); 0.8·h(z)])`), which
Supplement 03 verified against the sealed autodiff at rank 1.000000. From them: `g`, `g⁻¹`, the
Christoffel symbols `Γ^k_ij = (g⁻¹Jᵀ∂_i∂_j f)^k`, `II = ∂²f − JΓ`, `H = tr_g II`. The ridge weight
`w` (whole-data fit at the frozen `α = 100`) is rotated into that frame by `Qᵀ`. Labels' chart
gradients and Hessians are analytic; `Hess_M y = ∂²y − Γ·∂y`.

Per-anchor columns, all exact:

| column | definition |
|---|---|
| `exact_H_tan` | `‖H_tan‖`, the sealed verdict field (reference) |
| `pf_curv` | `‖⟨w_N, II⟩‖_g`, probe-facing curvature |
| `pf_trace_tan` | `⟨w_N, H_tan⟩` (signed) |
| `pf_trace_rad` | `−d⟨w, x₀⟩`, the sphere's own contribution |
| `hess_mismatch` | `‖Δ‖_g = ‖Hess_M y − ⟨w_N, II⟩‖_g` |
| `grad_mismatch` | `‖δ‖_g = ‖∇_M y − Jᵀw‖_g` |
| `bias_sq` | `c²` |
| `pred_resid` | the expansion's `E[r²]` with `s² = r_k²/(d+2)` from the anchor's own k-NN radius |
| `pred_local_r2` | `1 − pred_resid / pred_var`, the expansion's prediction of local R² |

Each column is Spearman-correlated with the measured local R², with `exact_H_tan`, and with log
radius, and run through the sealed three-control partial against the measured local R².

## 3. Results, intrinsic-linear label (`y = a·z`)

Sealed three-control partial of each column against local R². Coupling is
`ρ(‖H_tan‖, log r)` at the anchors. Every entry with `|partial| > 0.1` is at the permutation
floor.

| γ | coupling | `exact_H_tan` | **`pf_curv`** | `hess_mismatch` | `grad_mismatch` | `bias_sq` | `pred_resid` | `pred_local_r2` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −1.0 | +0.79 | −0.29 | **−0.79** | −0.71 | −0.57 | −0.57 | −0.73 | +0.46 |
| 0.0 | +0.51 | 0.00 | **−0.74** | −0.68 | −0.35 | −0.34 | −0.50 | +0.20 |
| 0.4 | +0.14 | +0.19 | **−0.75** | −0.59 | −0.52 | −0.42 | −0.63 | +0.34 |
| 0.6 | +0.05 | +0.22 | **−0.78** | −0.58 | −0.49 | −0.41 | −0.63 | +0.31 |
| 0.8 | −0.16 | +0.26 | **−0.80** | −0.56 | −0.53 | −0.45 | −0.65 | +0.37 |
| 1.0 | −0.15 | +0.21 | **−0.76** | −0.48 | −0.61 | −0.45 | −0.70 | +0.45 |

Spearman of each column with `exact_H_tan`, same rows:

| γ | `pf_curv` | `pf_trace_tan` | `hess_mismatch` | `grad_mismatch` | `bias_sq` | `pred_resid` |
|---:|---:|---:|---:|---:|---:|---:|
| −1.0 | +0.24 | −0.96 | −0.74 | −0.61 | −0.24 | −0.51 |
| 0.0 | +0.16 | −0.92 | −0.50 | −0.52 | −0.32 | −0.47 |
| 0.4 | +0.09 | −0.87 | −0.20 | −0.28 | −0.21 | −0.29 |
| 0.6 | +0.09 | −0.84 | −0.13 | −0.23 | −0.17 | −0.22 |
| 0.8 | +0.07 | −0.81 | +0.07 | −0.11 | −0.16 | −0.13 |
| 1.0 | +0.02 | −0.75 | +0.02 | −0.11 | −0.14 | −0.17 |

Other facts from the record: `‖w_N‖/‖w‖` median 0.13–0.16 (the ridge weight is mostly tangent);
`hess_mismatch` is `ρ = −0.98` to `−0.99` with log radius at every `γ` (the label's intrinsic
Hessian on this chart, `−Γ·a`, is a function of position that tracks the stereographic stretch);
the expansion's `pred_local_r2` correlates `+0.34` to `+0.47` with the measured local R².

## 4. Reading

1. **The probe-facing curvature carries a strong, stable, negative association.** `pf_curv`
   partial `−0.74` to `−0.80` at every sampling, while the sealed field's partial travels from
   `−0.29` to `+0.26` across the same samplings. This is the theorem's prediction and it holds
   at production scale with exact geometry: what a global linear probe pays for is `‖⟨w_N, II⟩‖`,
   and the sealed statistic is sensitive to it once the right column is used.

2. **The sealed field does not see it.** `ρ(‖H_tan‖, pf_curv)` is `+0.02` to `+0.24`. In a
   750-dimensional normal space the norm of the mean curvature vector says almost nothing about
   the second fundamental form's projection onto the probe's normal direction. The signed trace
   `⟨w_N, H_tan⟩` is `−0.75` to `−0.96` with `‖H_tan‖`, so `w_N` points consistently against `H_tan`
   on this generator, but the trace is not the norm and its own partial is small and unstable.

3. **Why the plateau is +0.22.** At zero coupling (`γ = 0.6`) `‖H_tan‖` has small negative
   correlations with the terms that lower local R²: `hess_mismatch −0.13`, `grad_mismatch −0.23`,
   `bias_sq −0.17`, `pred_resid −0.22`, and essentially none with `pf_curv` (`+0.09`). On this
   generator, anchors with larger `‖H_tan‖` are, incidentally, anchors where the global linear
   map is slightly better aligned and less biased. That incidental alignment, not curvature,
   produces the positive plateau. At `γ = −1` the same incidental correlations are large
   (`−0.74`, `−0.61`) and of the sign that the density coupling imposes, which is the steep part
   of the curve.

4. **The nonlinear label follows the mismatch, not the curvature.** With
   `y = sin(2a₁·z) + (a₂·z)² − ½(a₁·z)(a₂·z)`, `Hess_M y` is an order of magnitude larger than
   `⟨w_N, II⟩` (median `hess_mismatch` 36–112 against `pf_curv` 3–7). `hess_mismatch` and
   `grad_mismatch` carry partials of `−0.5` to `−0.7`; `pf_curv` alone reads `0` to `+0.2`. The
   theorem's variable is `Δ` as a whole; splitting the curvature out is only meaningful when the
   label's own intrinsic Hessian is small.

5. **The ridge null is explained almost entirely.** Local R² sits at 0.997, and its small
   variation is `bias_sq` (partial `−0.86` to `−0.93`) plus `pf_curv` (`−0.57` to `−0.89`);
   `pred_local_r2` correlates `+0.89` to `+0.95` with the measured value.

6. **The small-patch expansion is crude at this regime.** `r/R ≈ 1`, so `s² = r_k²/(d+2)` is not
   a small isotropic moment; `pred_local_r2` tracks the measured local R² at `+0.34` to `+0.47`
   for the intrinsic-linear label, well below `pf_curv` alone. The individual terms are more
   reliable than their assembly.

## 5. Consequence for the Physics record

Neither production instrument produces the probe-facing quantity. The decoder yields the full
second fundamental form by autodiff (`P_N D²F`, the `II` inside `plain_decoder_curvature`), and
the colleague's `B^S` is his estimate of `II`; his branch already carries `probe_facing_scalar`,
which computes `((tr b)² + 2‖b‖²_F)/(d(d+2))` with `b = ⟨ŵ_N, B⟩`, i.e. the expansion's
curvature term under uniform-ball moments. Nothing in the Phase 9 record computed it for either
instrument.

The theorem and this fixture together predict that a partial of `‖⟨w_N, II⟩‖` against local R²
on the Physics data should be negative and stable across the density manipulations that flip the
`‖H_tan‖` partial, for whichever instrument's `II` tracks the truth. That is the experiment that
could give a sign stable across instrument, label and `d`, which Supplement 04 found no existing
design provides. It needs the decoder's `II` tensor at the anchors (a change to what the runner
stores, not to any sealed module) and the fitted probe weights.

## 6. What this does not settle

- Whether the real manifold's `II` seen from the probe is large enough to matter beside the
  label's own intrinsic Hessian (point 4). Unknown until measured.
- The scale question: `s⁴` multiplies the curvature term, and on the real data `s` is the k-NN
  radius, which varies with density. A probe-facing column would still need a density-robust
  design; it removes the instrument-specific coupling problem, not the radius interaction.
- Anything about the two-embedding alignment phases (7 and 8). There the analogous variable is
  the relative second fundamental form `II_G − A·II_F`, never computed.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Supplement 05 — post-hoc, not pre-registered, feeds no verdict. Record sha256 `d6396ca0…`.*
