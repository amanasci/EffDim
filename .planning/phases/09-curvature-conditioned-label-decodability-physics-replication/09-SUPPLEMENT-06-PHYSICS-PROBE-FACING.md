# 09-SUPPLEMENT-06 — probe-facing curvature on the Physics anchors, both instruments

**Status:** post-hoc, supplementary. **Not pre-registered. Feeds no verdict.** No sealed constant is
reinterpreted; no Wave A number moves. **Written:** 2026-09-11 UTC

## The question

Supplement 05 derived, and confirmed on the known surface, that a global linear probe's residual
depends on the probe-facing curvature `⟨w_N, II⟩` (the second fundamental form contracted with the
normal part of the probe's weight vector), not on `‖H‖`; its partial against local R² was `−0.74`
to `−0.80` across six samplings where `‖H_tan‖`'s swung from `−0.29` to `+0.26`. Neither production
instrument had produced that quantity on the Physics data. This supplement computes it on the
sealed 512 anchors from both instruments and asks: does it carry a sign, and do the two instruments,
which disagreed on every label with `‖H‖`-type fields, agree on it?

## Provenance

Executed on the same host as `09-EXECUTION-HOST.md` § 9 (`pod128`, 16 threads, CPU), 2026-09-11,
under the developer's instruction to run the real-data probe-facing experiment. Runner
`notebooks/diagnostics/09_physics_probe_facing_run.py`, commit `71914ddc065ca7bf0db6546fc15f0eb9d95c97c5`.
Colleague code from the read-only checkout at `97efb2eb`, `topology` shimmed as before. Wall clock
7,981 s. Record `notebooks/.cache/09_physics_probe_facing.jsonl`, sha256
`aefe4474d55b27c544734cf75075c8644310cd83fda8999754bea30fce1a4b31`, verified on both sides. Per-`d`
geometry at the anchors (`J`, `D²F`, image, latent codes; float32) kept on the host under
`/mnt/ssd-cluster/effdim/probe-facing-out/probe-facing/`, not transferred (≈400 MB each).

## 1. What was computed

Held fixed, the production pipeline's own calls: embeddings and labels (`physics_labels`), the
frozen anchor draw, `knn_panel` at `k = 2048`, the out-of-fold ridge probe at `α = 100`, local R²,
and the three sealed controls. Added: a whole-data ridge at the same `α` for the weight vector
`w` per label; log radius at `k ∈ {16, 64, 256, 1024, 2048}` from the same panel for the
multi-scale control of Supplement 04; and, per `d ∈ {16, 20}`:

- **Decoder.** The frozen fit protocol (600 epochs, seed 0; variance explained 0.95205 at `d=16`,
  0.95693 at `d=20`, matching Wave A). At the anchors' latent codes, Jacobian and Hessian of the
  sphere-projected decoder by `torch.func` in the sealed chunk width; `II = D²F − JΓ`;
  `H = tr_g II`. Check: median cosine between this `H` and the sealed `plain_decoder_curvature`
  output `1.000000000` at both `d`; `H_rad` median `−16.0000` / `−20.0000`. `cond(g)` p50/p95
  14.8/25.4 and 13.7/20.9.
- **Colleague.** His `nested_pca_frame` + `_fit_rank` per anchor, unchanged; the fitted `B^S` of
  both halves of each of three splits averaged. `w_N` by his own `project_normal` (orthogonal to
  `span(x₀, J)`); `‖⟨w_N, B^S⟩‖_F`; and his own `probe_facing_scalar` times `‖w_N‖`.
  `K_H^cross` reproduced as reference: rank against decoder `‖H_tan‖` `−0.461` at `d=16` (Supplement
  01: `−0.463`).

Columns per anchor: `H_tan_norm` (sealed verdict field), `pf_curv_dec = ‖⟨w_N, II⟩‖_g`,
`pf_trace_tan_dec = ⟨w_N, H_tan⟩`, `pf_trace_rad_dec = H_rad ⟨w_N, x̂⟩`, `bias_sq = (y − ŷ_oof)²`
at the anchor, `K_H_cross_col`, `pf_curv_col`, `K_w_dir_col`. Statistic: rank-partial Spearman
against local R² with Freedman-Lane `p` (2,000 draws), under the sealed controls and under the
multi-scale controls.

## 2. Reproduction of the sealed numbers

| quantity | this run | record it must match |
|---|---|---|
| decoder `‖H_tan‖`, mag_r, `d=16`, sealed partial | `+0.328` | Amendment 01 `+0.328` |
| same, multi-scale control | `+0.248` | Supplement 04 `+0.25` |
| colleague `K_H^cross`, mag_r, `d=16`, sealed | `−0.147` | Supplement 01 `−0.149` |
| same, multi-scale | `−0.010` | Supplement 04 `−0.01` |
| decoder `‖H_tan‖`, mag_r, `d=20`, multi-scale | `−0.167` | Supplement 04 negative under density-robust designs |

## 3. Results

Partial against local R², sealed / multi-scale. `*` = `p > 0.05`.

| label | `d` | decoder `‖H_tan‖` | his `K_H^cross` | **`‖⟨w_N,II⟩‖` decoder** | **`‖⟨w_N,B^S⟩‖` colleague** | his `K_w_dir·‖w_N‖` |
|---|---:|---|---|---|---|---|
| mag_r | 16 | +0.33 / +0.25 | −0.15 / −0.01* | **+0.20 / +0.10** | **+0.49 / +0.40** | +0.47 / +0.38 |
| mag_r | 20 | +0.02* / −0.17 | −0.24 / −0.13 | **+0.25 / +0.15** | **+0.42 / +0.32** | +0.42 / +0.32 |
| photo_z | 16 | +0.36 / +0.28 | −0.14 / −0.01* | **−0.01* / −0.13** | **−0.17 / −0.16** | −0.12 / −0.12 |
| photo_z | 20 | +0.31 / +0.13 | −0.10 / +0.01* | **−0.10 / −0.22** | **−0.17 / −0.17** | −0.09 / −0.10 |
| smooth_fraction | 16 | +0.34 / +0.21 | −0.16 / −0.02* | +0.14 / +0.06* | +0.36 / +0.18 | +0.35 / +0.19 |
| smooth_fraction | 20 | +0.33 / +0.15 | −0.15 / −0.02* | +0.26 / +0.13 | +0.35 / +0.20 | +0.35 / +0.21 |
| stellar_mass | 16 | +0.07* / −0.02* | +0.03* / +0.11 | −0.01* / −0.05* | +0.23 / +0.16 | +0.22 / +0.16 |
| stellar_mass | 20 | +0.12 / −0.00* | −0.04* / +0.02* | +0.02* / −0.04* | +0.15 / +0.07* | +0.14 / +0.08* |

Relations between columns (mag_r, `d=16`): `ρ(pf_curv_dec, log r) = −0.56`, `ρ(pf_curv_col, log r) = +0.43`,
`ρ(pf_curv_dec, pf_curv_col) = −0.004`, `ρ(pf_curv_dec, ‖H_tan‖) = +0.61`. Decoder `‖w_N‖/‖w‖`
median 0.80–0.87.

`pf_trace_rad_dec` reads `+0.24` to `+0.41` on every label and `d`. On the unit sphere
`H_rad = −d` exactly, so this column is `−d ⟨w_N, x̂⟩ = −d (ŷ − b₀)`: the prediction level at the
anchor, not geometry. Discarded from the interpretation; its partials say bright-end predictions
decode better.

## 4. Reading

1. **The fixture's uniform negative sign does not appear on the real data.** The probe-facing
   partial is label-dependent: positive for `mag_r`, negative for `photo_z`, positive for
   `smooth_fraction`, null (decoder) or weakly positive (colleague) for `stellar_mass`. This is
   the regime of Supplement 05's nonlinear-label arm: when the label's intrinsic Hessian on the
   manifold is not small, the residual is carried by the mismatch `Δ = Hess_M y − ⟨w_N, II⟩` and
   the curvature piece alone can take either sign. A positive sign means the manifold bends in the
   probe's direction the way the label bends intrinsically, and the linear map is helped.
   `Hess_M y` is not available on real data, so this reading is consistent with the theory, not
   tested by it.
2. **The two instruments agree on the probe-facing sign.** For three of four labels, at both `d`,
   under both control sets, the decoder's `‖⟨w_N, II⟩‖` and the colleague's `‖⟨w_N, B^S⟩‖` carry the
   same sign. With `‖H‖`-type fields the same two instruments disagreed on every label. The two
   probe-facing columns couple to radius in opposite directions (`−0.56` and `+0.43`) and are
   rank-uncorrelated with each other (`−0.004`), so the agreement is not a shared density artefact.
   His own `probe_facing_scalar` tracks his contraction within 0.03 everywhere.
3. **The sign is stable across `d` where `‖H_tan‖`'s is not.** For `mag_r` the sealed field goes
   from `+0.33` (`d=16`) to `+0.02` (`d=20`, `−0.17` under the multi-scale control); the
   probe-facing decoder column reads `+0.20 / +0.10` and `+0.25 / +0.15`.
4. **Magnitudes are modest.** The strongest probe-facing partial is the colleague's `+0.49` on
   `mag_r` (`+0.40` under the multi-scale control); the decoder's is `+0.25`. Nothing here is the
   `−0.74` to `−0.80` of the fixture, and nothing should be: that number was for a label with no
   intrinsic Hessian.

## 5. What this settles and what it does not

**Settles.** On the Physics data the probe-facing curvature is a quantity on which the two
independently built instruments agree in sign per label, survives the multi-scale density control,
and holds from `d=16` to `d=20`. The `‖H‖`-type fields have none of those properties, which is
consistent with Supplement 05's explanation of why: they do not measure what the probe responds to.

**Does not settle.** Whether the label-dependent sign is the mechanism Eq. (2) of Supplement 05
names (curvature in the probe's direction matching the label's intrinsic Hessian), since
`Hess_M y` on the real manifold is unknown. The `d = 25, 32` cells. The positive-control and
shuffled-label gates, which this diagnostic does not touch. The radius⁴ interaction, which the
multi-scale control addresses only linearly in ranks. The exchangeability of anchor-level
permutation `p`-values under overlapping patches.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Supplement 06 — post-hoc, not pre-registered, feeds no verdict. Run commit `71914dd`, colleague
commit `97efb2eb`, record sha256 `aefe4474…1a4b31`.*
