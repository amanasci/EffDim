# Review robustness: the paper's five galaxy encoders under the reviewer's concerns 2-5

Date: 2026-09-30
Branch: `review-robustness` (from `encoder-scaling@7769dbd`)

## Goal

Answer the ML4PS reviewer's concerns 2-5 for the rebuttal / camera-ready, on the exact decoders
behind the published numbers:

2. sensitivity to probe regularisation — repeat mismatch, alignment and the intervention with a
   validation-tuned probe;
3. unproven added value — control for target difficulty (label-Hessian norm, local label roughness)
   and test on held-out anchors whether geometry predicts local R^2 beyond it;
4. dependence across overlapping anchors — dependence-aware uncertainty for the main partials;
5. surrogate definition and fidelity — state which baseline each result uses and quantify how
   closely the t=1 surrogate reproduces the actual probe.

This is sub-project B of three (A tensor fidelity, done; C scale-out). Scope is lean (rebuttal):
the paper's five encoders only. The ~2x `hess_y` magnitude error found in A is stated in the report
as a known limitation (the paper's partials are rank-based, so it does not move them); its cause
is not investigated here.

Success criteria:

1. For each of the five encoders the runner reproduces, at alpha = 100 from the stored geometry,
   the published mismatch/alignment partials and counterfactual help/hurt fractions (tolerances
   below), and stops otherwise.
2. It reports, per encoder and label, every quantity below at alpha = 100 and alpha*, with the
   report regenerable from the records.
3. All new tests pass; the CPU equivalence gate passes; `paper/latex/main.tex` is unchanged; no
   existing runner changes.

## Decisions (agreed in brainstorming)

| Topic | Decision |
|---|---|
| Destination | Rebuttal / camera-ready; lean scope |
| Encoders | ViT-B, DINOv3-B, CLIP-B, ConvNeXt-B, ViT-L at d=16 (the paper's five) |
| Geometry | The paper's stored decoder geometry, sha256-verified; no refit |
| Architecture | One new additive runner importing the split and counterfactual runners unchanged |
| Platform | Pod CPU (data already there), tmux, 16 threads; local CPU for tests |
| Alpha | Nested RidgeCV over a fixed grid for OOF predictions; RidgeCV on all rows for the global w |
| Added value | Extra controls (hess_label, local label roughness) + held-out Delta R^2 with cluster-based splits |
| Dependence | Cluster bootstrap (primary), thinned-anchor partials (secondary) |
| hess_y magnitude | Limitation only |

## Inputs (read-only)

| Encoder | Geometry npz (pod) | sha256 |
|---|---|---|
| vit_base | `/mnt/ssd-cluster/effdim/probe-facing-out/probe-facing/09_probe_facing_geometry_d16.npz` | `477886ad7036ff6a18897409bdbb1db15262457bb618da64fadaaba138b82ff9` |
| clip_base | `/mnt/ssd-cluster/effdim/xenc-out/geometry_clip_base/09_probe_facing_geometry_d16_seed0.npz` | `602cf931f1dd5ff42a40beece81f2a724c6eece5238e15c40fe202dd4daead80` |
| convnext_base | `/mnt/ssd-cluster/effdim/xenc-out/geometry_convnext_base/09_probe_facing_geometry_d16_seed0.npz` | `14db5894cf8d6eecb7e3e50146032e95b1bea05ddda6e0c5df99a34449fb4904` |
| dinov3_vitb16 | `/mnt/ssd-cluster/effdim/xenc-out/geometry_dinov3_vitb16/09_probe_facing_geometry_d16_seed0.npz` | `b4176c43b0171869f16e4bb4a93290e18c4b298f25fc6a392219a1def9f299de` |
| vit_large | `/mnt/ssd-cluster/effdim/xenc-out/geometry_vit_large/09_probe_facing_geometry_d16_seed0.npz` | `f8c1015ea589e378e33d36529ca3ea0499246408f432cd289e1dcf53db181718` |

Embeddings: `/mnt/ssd-cluster/effdim/hf-cache/hub/datasets--UniverseTBD--pu-embeddings/snapshots/bc081f8a5db4767edcd958653d96efde9137de0b/physics/<enc>_test.parquet`, column `<enc>_galaxies`.
Labels: `/mnt/ssd-cluster/effdim/labels_Smith42_galaxies_v2.0_test.parquet`, sha256
`60f2f82e64e4036eb4eff9a448b15365dd1e974e52779fe7c1323db0d9dabfd9` (the table every paper record's environment row names).
Published reference values: the paper records in `notebooks/.cache/` read through
`curvature-experiment/sweep/extract.py` (the ported record readers; mismatch column
`hess_mismatch_emp`, alignment column `align_cos_tan`, multi-scale controls) and the
counterfactual npz `09_physics_normal_scaling_<enc>_d16.npz` / `_thin.npz`. Nothing under
`/mnt/ssd-cluster/effdim/` or `notebooks/.cache/` is written.

Protocol constants unchanged: labels mag_r, photo_z, smooth_fraction, stellar_mass; 512 anchors;
k = 2048; the paper's anchor draw, OOF folds and permutation seeds.

## Method

### Concern 2: validation-tuned alpha

- Grid `ALPHA_GRID = logspace(-3, 4, 15)` (contains 1 and 100).
- OOF predictions (for local R^2): in each of the paper's outer folds, `RidgeCV(alphas=ALPHA_GRID)`
  (efficient leave-one-out) is fit on that fold's training rows only; its prediction on the
  held-out fold is used. Per-fold alphas are recorded.
- Global probe `w` (sets `w_N`, the probe-facing tensors and the counterfactual): `RidgeCV` on all
  finite rows; alpha* recorded per label.
- Every paper quantity is computed at alpha = 100 and at alpha*: global OOF R^2; the mismatch
  (`hess_mismatch_emp`) and alignment (`align_cos_tan`) partials with the multi-scale controls
  (partial and permutation p); counterfactual help/hurt fractions, median t*, the random
  normal-direction null matched on the contracted tensor's metric norm, and the thinned-anchor sign test.

### Concern 3: added value

- Extended controls: the paper's multi-scale controls plus `hess_label` (metric Frobenius norm of
  the label Hessian, already a split-runner column) and local label roughness `1 - r2_lin(y)` (the
  local linear fit's R^2 of the label in the neighbourhood, already computed by `local_quadratics`).
  Mismatch and alignment partials reported under the paper controls and under the extended ones.
- Held-out test: anchors are split into two halves by overlap cluster (the G = 32 blocks below,
  half the blocks each), 20 seeded splits. On half A fit OLS of local R^2 on (paper controls +
  difficulty terms) and on (those + mismatch + alignment); score both on half B. Report the
  out-of-sample Delta R^2 (geometry model minus difficulty model): median, 5-95% range over
  splits, and the fraction of splits with Delta R^2 > 0.

### Concern 4: dependence

- Blocks: the 512 x 512 neighbourhood-overlap matrix (the thin runner's `overlap_matrix`),
  average-linkage clustering on `1 - overlap`, cut into G = 32 blocks (sensitivity G = 16, 64).
- Cluster bootstrap: resample blocks with replacement, 2000 replicates, recompute each partial;
  report the 95% percentile interval and whether it excludes 0. Stated caveat: adjacent blocks
  still share boundary points, so this is more honest than the anchor-level permutation, not exact.
- Secondary: partials on thinned anchors (pairwise overlap <= 0.10, about 20 anchors) with
  permutation p and n, labelled low-power.

### Concern 5: surrogate fidelity

- Per anchor, the change in local R^2 from t = 0 to t = 1 computed (i) exactly on the data points
  (the counterfactual runner's quadratic in t) and (ii) by the second-order surrogate the Section 4
  figure uses. Report the Spearman between them and the median absolute difference, per encoder and
  label.
- A short note of which baseline (tangent-plus-sphere quadratic vs exact ambient tangent-plus-radial)
  each figure and table uses, confirmed from the code during planning.

### Reproduction guard

At alpha = 100, before any new number is written, the runner compares with the published values:
partials within 1e-6 absolute, counterfactual help/hurt fractions exactly equal. Any difference stops
the run with the differing cells printed.

## Code

Additive only.

- `curvature-experiment/runners/11_review_robustness_run.py` — per encoder: load and verify
  geometry, embeddings, labels; compute the above; append records to
  `EFFDIM_CACHE_DIR/11_review_robustness_<enc>.jsonl` (environment row first). Flags: `--encoder`,
  `--geometry-npz`, `--geometry-sha256`, `--parquet-path`, `--embedding-column`, `--label-table`,
  `--reference-dir` (paper records), `--threads`, `--record-path`, `--n-boot`, `--smoke`.
- `curvature-experiment/runners/11_review_robustness_report.py` — records ->
  `curvature-experiment/results/review-robustness/REPORT.md` (one rebuttal-ready table per concern,
  the limitation note) ; records copied to `results/review-robustness/records/`.
- `curvature-experiment/tests/test_review_robustness.py`.

## Tests (TDD, local CPU)

1. Nested alpha on synthetic ridge data: strong signal picks a small alpha; the held-out fold's rows
   never enter its own alpha selection.
2. Cluster bootstrap on independent synthetic anchors: the 95% interval covers the true partial in
   roughly 95% of 200 repetitions (accept 0.90-0.99).
3. Held-out Delta R^2: > 0 when a synthetic geometry column carries signal, ~0 (median within
   +-0.02) when it is noise.
4. Reproduction guard: a perturbed reference value stops the run with the cell named.
5. Smoke end to end on the smoke fixture (`--smoke`): writes rows with finite values.
6. CPU gate passes; `main.tex` unchanged.

## Out of scope

Sub-project C; editing `main.tex`; investigating the `hess_y` magnitude; any change to existing
runners or to the paper records.
