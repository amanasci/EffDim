# 09-SUPPLEMENT-11 — cross-encoder probe-facing test (experiment X)

**Status:** post-hoc. **Not pre-registered. Feeds no verdict.** **Written:** 2026-09-15 UTC.
Runner `notebooks/diagnostics/09_physics_probe_facing_split_run.py --mode physics --d-values 16 --fit-seed 0
--hessian-xfit --parquet-path <local parquet> --embedding-column <enc>_galaxies --label-table <cached labels>`
(runner sha256 `ce63a83ce4b0d2f6…`, commit 139950f), pod `universetbd-0`, one tmux session per encoder,
16 threads each, 48–52 min wall. Embeddings: `UniverseTBD/pu-embeddings` physics test split, files
`physics/{dinov3_vitb16,clip_base,convnext_base,vit_large}_test.parquet` (snapshot `bc081f8a…`), fetched
with `hf_hub_download`; labels from the cached parquet (sha256 `60f2f82e…`). Records
`notebooks/.cache/09_physics_probe_facing_split_<enc>.jsonl`, logs `notebooks/.cache/09_xenc_<enc>.log`
(sha256 verified on both sides). Manuscript Appendix D is generated from these by `appendix_gen.py`.

## Correctness gate (row order)
Every parquet has exactly 86,471 rows and a single column `<enc>_galaxies`; there is no id column to join
on. The gate used instead: the label probes' global out-of-sample R² per encoder (mag_r 0.57 / 0.50 /
0.52 / 0.52; photo_z 0.56 / 0.49 / 0.49 / 0.51; smooth 0.59 / 0.49 / 0.56 / 0.55; stellar 0.54 / 0.50 /
0.49 / 0.49 for DINOv3 / CLIP-B / ConvNeXt-B / ViT-L, vs 0.52 / 0.51 / 0.53 / 0.48 for ViT-B), which a
misaligned row order would collapse to ≈ 0. Widths 768 / 512 / 1,024 / 1,024 (`in_dim` from the data;
the sealed loader's 768 is bypassed by a runner-level shim). Decoders 250³, seed 0, variance explained
0.966 / 0.985 / 0.965 / 0.968 (ViT-B: 0.952).

## Result (d = 16, 512 anchors, k = 2,048, multi-scale density control; ViT-B from the main record)

| label | quantity | ViT-B | DINOv3 | CLIP-B | ConvNeXt-B | ViT-L |
|---|---|---|---|---|---|---|
| mag_r | mismatch | −0.39 | −0.57 | −0.34 | −0.32 | −0.44 |
| | alignment | +0.35 | +0.53 | +0.32 | +0.21 | +0.13 |
| | shape | +0.12 | −0.11 | +0.16 | +0.16 | −0.21 |
| | sphere | −0.26 | −0.23 | −0.40 | −0.24 | −0.08* |
| photo_z | mismatch | −0.45 | −0.61 | −0.45 | −0.48 | −0.46 |
| | alignment | +0.26 | +0.33 | +0.06* | +0.22 | +0.14 |
| | shape | −0.06* | −0.05* | −0.17 | +0.29 | −0.16 |
| | sphere | −0.33 | −0.21 | −0.29 | −0.46 | −0.44 |
| smooth_fraction | mismatch | −0.09* | −0.33 | −0.25 | −0.20 | −0.01* |
| | alignment | +0.19 | −0.02* | +0.24 | +0.04* | +0.14 |
| | shape | +0.07* | −0.08* | +0.20 | +0.24 | +0.06* |
| | sphere | −0.13 | −0.06* | +0.05* | −0.15 | −0.11 |
| stellar_mass | mismatch | −0.04* | +0.03* | −0.12 | −0.06* | −0.02* |
| | alignment | +0.01* | −0.05* | −0.01* | +0.07* | −0.03* |
| | shape | +0.00* | −0.15 | −0.10 | −0.11 | −0.13 |
| | sphere | −0.35 | −0.26 | −0.38 | −0.40 | −0.38 |

\* not significant at 0.05. Cross-fitted mismatch (fit A / score B, fit B / score A) for the four new
encoders: mag_r −0.60/−0.62, −0.28/−0.35, −0.36/−0.34, −0.46/−0.49; photo_z −0.56/−0.59, −0.41/−0.40,
−0.46/−0.45, −0.44/−0.42 (same order as the table). Split-half tensor cosine of the label Hessian
0.19–0.35 on every encoder and label. Median cosine between the decoder's ⟨w_N, II⟩ and the data-side
quadratic fit of the readout 0.55–0.63 on every encoder. ‖H_tan‖ partial on mag_r: +0.25, +0.03*, +0.23,
+0.21, +0.14.

## Reading
- **Encoder-stable, as predicted.** Mismatch negative for mag_r and photo_z on all five encoders (−0.32 to
  −0.61; cross-fitted −0.28 to −0.62), alignment positive for mag_r on all five (+0.13 to +0.53) and for
  photo_z on four of five (CLIP-B +0.06 n.s.). No significant negative alignment in any of the 20 cells.
- **Label-dependent, as before.** smooth_fraction: mismatch significant on three encoders, alignment on
  three (different three); stellar_mass: alignment null on every encoder; mismatch null except CLIP-B (−0.12).
- **Shape term has no encoder-stable sign** (mag_r: +, −, +, +, −; photo_z: n.s., n.s., −, +, −): the same
  conclusion Section 5 drew for the trace statistic now holds for the decoder's own shape term, and the
  paper's account (its sign is set by alignment with, and magnitude relative to, the label Hessian) is what
  the encoder-to-encoder flips look like. The sphere term is negative in 17 of 20 cells (3 n.s.).
- Manuscript: Appendix D (two tables, generated), the "Across encoders" paragraph extended with the
  result, one clause each in the abstract and Discussion.

---
*Phase 09 follow-up. Not pre-registered, feeds no verdict.*
