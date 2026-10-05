# Code and logic audit, "Linear Probes on Curved Latent Spaces" (2026-10-05)

Independent audit for the project lead. Scope is the ML4PS paper ([`paper/latex/main.tex`](../../paper/latex/main.tex)), the runners and
modules behind it, the sweep and QM9 machinery, the result reports, and four earlier reviews (treated as claims to
check). Read-only except this file. No ssh, no git, [`paper/generate/appendix_gen.py`](../../paper/generate/appendix_gen.py) was read but never imported or
run.

How to read the citations. `file:line` points at the code as it is on disk today. "RR", "TF", "SC", "QM" are the
four reports under `curvature-experiment/results/` ([`review-robustness/REPORT.md`](../../curvature-experiment/results/review-robustness/REPORT.md), [`tensor-fidelity/REPORT.md`](../../curvature-experiment/results/tensor-fidelity/REPORT.md),
[`scaling/SCALING_REPORT.md`](../../curvature-experiment/results/scaling/SCALING_REPORT.md), [`qm9/QM9_REPORT.md`](../../curvature-experiment/results/qm9/QM9_REPORT.md)). "Sanity" is
[`docs/reviews/2026-10-03-results-sanity/sanity-review.md`](../reviews/2026-10-03-results-sanity/sanity-review.md), "Outline" is
[`docs/reviews/2026-10-05-tmlr-outline/review.md`](../reviews/2026-10-05-tmlr-outline/review.md), "Novelty" is
[`docs/novelty/2026-10-03-framing1-novelty.md`](../novelty/2026-10-03-framing1-novelty.md), "Test-1" is [`docs/novelty/2026-10-05-test1-design-review/review.md`](../novelty/2026-10-05-test1-design-review/review.md).
"Cache" is `notebooks/.cache/` (what `curvature-experiment/.cache` symlinks to).

What I checked myself, and how. Scratch scripts are in `docs/audit/2026-10-05-scripts/`.
- [`check_math.py`](2026-10-05-scripts/check_math.py) imports the real runner functions and checks them on toy problems with known answers
  (label-Hessian fit, the empirical-mismatch identity, the counterfactual SSE identity, tilted-tangent leak).
- [`check_records.py`](2026-10-05-scripts/check_records.py) and [`check_qm9.py`](2026-10-05-scripts/check_qm9.py) recompute claims from the local records and per-anchor arrays.
- A Monte Carlo check of the non-Gaussian patch formula, a block-size check of the bootstrap, and a centred
  variance check on a cached PU embedding, all inline commands.
- A targeted test run, 110 tests in five files, all passed ([`test_ii_rank.py`](../../curvature-experiment/tests/test_ii_rank.py), [`test_cross_split_curvature.py`](../../curvature-experiment/tests/test_cross_split_curvature.py),
  [`test_sweep_extract.py`](../../curvature-experiment/tests/test_sweep_extract.py), [`test_intrinsic_dim.py`](../../curvature-experiment/tests/test_intrinsic_dim.py), [`test_physics_curvature_probe.py`](../../curvature-experiment/tests/test_physics_curvature_probe.py)). The full suite (363 passed)
  and the CPU gate were not rerun; those results come from the session status.

Not available locally, so not checked by me. The galaxy embeddings for the physics config, the galaxy label
parquet, and every per-anchor geometry npz (J, Hess, image). They live on the pod only. Everything that needs them
is marked "unverified (pod-only)".

---

## 1. Executive summary

**What the project claims.** A linear probe ŷ = w·x + b restricted to a curved embedding manifold has intrinsic
Hessian K = ⟨w_N, II⟩ (classical). The paper estimates II per object with an autoencoder decoder, estimates the
label's intrinsic Hessian from neighbours, and reports (i) that the mismatch ‖Hess y − K‖ is negatively associated
with local probe accuracy for magnitude and redshift across five encoders, (ii) that alignment cos(Hess y, K) is
positively associated, and (iii) that in a local second-order surrogate, keeping the probe's in-sphere bending
(t = 1) beats removing it (t = 0) at most anchors while reversing it (t = −1) hurts, unlike a random bend.

**What holds.** The geometry code is correct. II, the sphere split, the label-Hessian design, the
Freedman–Lane null, the counterfactual SSE formula, the II spectrum and the reproduction guards all do what they
say, and I confirmed the key identities numerically. The reported numbers match the records. The
magnitude/redshift association between the tabulated mismatch column and local R² is real and survives the block
bootstrap (ten galaxy encoders, 9/10 and 10/10 at α = 100, 10/10 and 10/10 at α*).

**What is weakened.** What that association means. The tabulated "mismatch" is `hess_mismatch_emp`, which by
linearity of least squares equals the norm of the local quadratic part of the probe residual y − w·x. It contains
no decoder II, it tracks the label-Hessian norm alone within 0.05 of partial in 36 of 40 galaxy cells, and it was
never cross-fitted. The paper's cross-fit appendix cross-fits a different column (`hess_mismatch_dec`). Alignment
is much weaker under the bootstrap (magnitude 4/10, redshift 5/10 at α = 100). QM9 transfers for the mismatch on
three of four properties but not for the counterfactual.

**What is unsupported or should be retracted.** The causal reading of the counterfactual ("bending toward the
label helps"), because "hurt" is implied by "help" and the help pattern is what least squares produces for any
decodable label. The decoder surrogate carries only 0.38–0.51 of the probe's own in-sphere readout amplitude. Also
the rebuttal's "label Hessian is about 2x too large" and "low split-half cosines are label noise", the rebuttal's
"results get stronger with the tuned probe" as a general statement, and any "few dominant bending directions"
claim for the II spectrum.

**Confirmed defects.** No numerical bug in the experiment code. There are seven reporting or labelling defects
(the paper describes a different column or variant than the one it tabulates, and one hand-typed range is wrong),
one library bug in `src/effdim` (TLE is MLE) that affects the QM9 choice of d, and one misleading fit metric
("variance explained" is uncentred). All are listed in Section 3.12.

### Claim table

| # | Claim (where made) | Status | Evidence | Where it is checked |
|---|---|---|---|---|
| 1 | Hess_M(w·x) = ⟨w_N, II⟩ and on the sphere K = K_S − (ŷ−b₀)g ([main.tex:97-113](../../paper/latex/main.tex#L97-L113)) | **holds** | Classical. Code check `II_rad_vs_minus_g_max_rel` 1.5e-8, `JT_xhat_max` 2e-9 in the published split record. My toy (check 1b) recovers ⟨w_N,II⟩ from data to 3e-15 | [`09_physics_probe_facing_split_run.py:174-179`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L174-L179), [`198-200`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L198-L200); Cache `09_physics_probe_facing_split.jsonl`; `test_radial_decomposition_on_analytic_sphere` |
| 2 | Decoder II validated on a synthetic surface: trace cos 0.999, ratio 1.00, rank 0.94 ([main.tex:63](../../paper/latex/main.tex#L63)) | **holds** for the trace | Record 09_instrument_adjudication ([REPRODUCE.md](../../curvature-experiment/REPRODUCE.md) §3.1). Not recomputed by me | [`09_instrument_adjudication_run.py`](../../curvature-experiment/runners/09_instrument_adjudication_run.py) |
| 3 | Full probe contraction validated (rebuttal §1) | **weakened** | Paper scale, pf_tan cos 0.963–0.965 (noise 0), 0.675–0.843 (noise 0.25), TF:100-116. But the fixture has a one-dimensional in-sphere normal space, so how w_N selects among many normal directions is untested (Sanity I5). Paper-scale cells are one seed | [`10_tensor_fidelity_run.py`](../../curvature-experiment/runners/10_tensor_fidelity_run.py); [`09_instrument_adjudication_run.py:267-270`](../../curvature-experiment/runners/09_instrument_adjudication_run.py#L267-L270) per Sanity |
| 4 | Residual expansion Eq. 3 and the help condition 2⟨H,K⟩ > ‖K‖² ([main.tex:118-159](../../paper/latex/main.tex#L118-L159)) | **holds as stated (Gaussian patch)**, **weakened** for real kNN patches | For a uniform d-ball (d = 16) my Monte Carlo gives Cov 0.0212 vs Gaussian formula 0.121 for trace-heavy tensors; the λ-corrected formula of Outline I2 matches (0.0212) | Section 3.4 below |
| 5 | Mismatch ‖Hess y − K‖ negatively associated with local R² for mag/z, −0.39 to −0.45 ([main.tex:183-185](../../paper/latex/main.tex#L183-L185)) | **weakened** | Numbers match the record. But the tabulated column is `hess_mismatch_emp` = ‖quadratic part of (y − w·x)‖, not ‖Hess y − ⟨w_N,II⟩‖. Within 0.05 of the ‖Hess y‖-alone partial in 36/40 galaxy cells (median gap 0.024, max 0.111 at ViT-B mag_r) | [`09_physics_probe_facing_split_run.py:192`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L192); [`appendix_gen.py:19`](../../paper/generate/appendix_gen.py#L19); my [`check_records.py`](2026-10-05-scripts/check_records.py) |
| 6 | "These relationships persist under cross-fitting" ([main.tex:190](../../paper/latex/main.tex#L190), 424) | **weakened (mislabelled), confirmed** | The cross-fitted column in tab:sens and tab:xencx is `hess_mismatch_dec`; tab:real and tab:xenc use `hess_mismatch_emp`. Every xfit row in all 15 record files that have one stores only `align_cos_tan`, `hess_label`, `hess_mismatch_dec`. emp was never cross-fitted | [`09_physics_probe_facing_split_run.py:407-442`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L407-L442) (column list [:432](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L432)); [`appendix_gen.py:19`](../../paper/generate/appendix_gen.py#L19), [`62`](../../paper/generate/appendix_gen.py#L62), [`90`](../../paper/generate/appendix_gen.py#L90), [`100`](../../paper/generate/appendix_gen.py#L100), [`224`](../../paper/generate/appendix_gen.py#L224); Section 5 point 11 |
| 7 | Mag/z mismatch association is robust to anchor dependence (rebuttal §4) | **holds** | 32-block bootstrap excludes 0: ten encoders 9/10 (mag) and 10/10 (z) at α = 100, 10/10 and 10/10 at α*. QM9 gap/alpha/cv 8/8 at α = 100 | Cache `scaling/records/*__robust.jsonl`, `qm9/records/*__robust.jsonl` |
| 8 | Alignment cos(Hess y, K) positively associated, +0.24 to +0.43 ([main.tex:186-189](../../paper/latex/main.tex#L186-L189)) | **weakened** | Tabulated column is `align_cos_tan` (uses K_S not K; numerically within 0.02 of `align_cos_full`). Bootstrap at α = 100, ten encoders: mag 4/10, z 5/10. Label-Hessian direction is poorly estimated (split-half cos 0.19–0.35) | [`09_physics_probe_facing_split_run.py:195`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L195); Cache robust records |
| 9 | Smooth fraction weaker, stellar mass null ([main.tex:191-195](../../paper/latex/main.tex#L191-L195)) | **holds** | Five-encoder bootstrap at α = 100 excludes 0 for smooth and stellar in 0/10 | RR:106-177 |
| 10 | t = 1 helps at 79–99%, t = −1 hurts at 97–100%, random t = 1 helps at 22–36% ([main.tex:275-285](../../paper/latex/main.tex#L275-L285)) | **holds (numbers)** | Recomputed from all 24 published cf arrays | Cache `09_physics_normal_scaling_*_d*.npz` |
| 11 | That pattern shows the probe's bending is "toward the target's remaining curvature" ([main.tex:287-292](../../paper/latex/main.tex#L287-L292), abstract) | **unsupported** | Help ⇔ t* > ½ and hurt ⇔ t* > −½ (identity holds at 100% of anchors). Least squares makes help near-automatic for any decodable label (Novelty §6 simulation). Decoder q has 0.38–0.51 of the probe's own in-sphere readout amplitude | Section 3.9 |
| 12 | t* > 1 means the ridge probe under-uses a locally beneficial term (Table cf caption) | **contested** | Ridge pushes pooled t*_S above 1, but the published t* (S_model) also contains a fidelity slope β(p|q) that can exceed 1 with no shrinkage. Cannot be split without ⟨p,q⟩, which is not stored | Section 5, point 2 |
| 13 | Sphere term negative on every label ([main.tex:206-211](../../paper/latex/main.tex#L206-L211)) | **holds**, partly an artefact | −0.13 to −0.36 at α = 100; vanishes for z and stellar mass at α = 1 (paper discloses) | Cache split and `_alpha1` records |
| 14 | Shape term tracks ‖K‖ at rank 0.96–0.99; decoder-free K cos 0.55–0.65 ([main.tex:197-216](../../paper/latex/main.tex#L197-L216)) | **holds** | rank 0.96–0.99; cos (tan) 0.55–0.65, (full) 0.57–0.67 | Cache split record `checks` |
| 15 | Five-encoder and ten-encoder breadth (Appendix C, rebuttal) | **holds as robustness to embedding**, not as independent replication | Same 86,471 galaxies, labels and anchor indices for all encoders | [`11_review_robustness_run.py:410-412`](../../curvature-experiment/runners/11_review_robustness_run.py#L410-L412) |
| 16 | Tuned probe makes results stronger (rebuttal §2) | **weakened** | True on galaxies by permutation counts. Not on QM9 (tuned help 0.12–0.60, t* 0.11–0.60, ChemFM alpha/cv mismatch turns positive). α* at the grid floor in 5/20 galaxy and 28/32 QM9 pairs. Not cross-fitted | RR:17-36; my [`check_qm9.py`](2026-10-05-scripts/check_qm9.py) |
| 17 | Held-out added value beyond target difficulty (rebuttal §3) | **weakened** | α = 100 medians −0.006 to +0.147, p05 > 0 strictly in 9/20; α* +0.015 to +0.258, p05 > 0 in 18/20. Neighbourhood-residual baseline never run | RR:44-90 |
| 18 | Roughness control "over-controls if anything" (RR:40, rebuttal §3) | **retract** | Partial grows in 13/20 pairs at α* when |Hess y| and roughness are added | RR:17-36 vs RR:71-90 |
| 19 | Label Hessian "about 2x too large" at paper scale (RR:228, rebuttal §1) | **unsupported** | `relerr` is unsigned ([`10_tensor_fidelity_run.py:82-90`](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L82-L90)); a cos/relerr pair admits two norm ratios (lin: 1.83 or 0.12). No signed ratio recorded | TF:100-116 |
| 20 | Low real split-half cosines are label noise (rebuttal §1) | **unsupported** | Synthetic labels carry no noise; noise is added to X only | [`10_tensor_fidelity_run.py:153-175`](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L153-L175), [`211-216`](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L211-L216) |
| 21 | QM9: mismatch transfers, counterfactual "direction generalises, size depends on encoder" (rebuttal) | **weakened / unsupported** | Mismatch 8/8 for gap, alpha, cv (bootstrap); mu 5/8, ChemFM-3B mu significantly positive. Counterfactual surrogate per-anchor Spearman with the probe −0.30 to +0.23; data-side S helps at 0.88–1.00 while S_model helps at 0.28–0.74. The failure is surrogate fidelity | QM:62-141; Cache qm9 records and arrays |
| 22 | Mean-bending trace −0.24 to +0.27 across five encoders ([main.tex:585](../../paper/latex/main.tex#L585)) | **unsupported in this repo** | Not produced by any runner here ([REPRODUCE.md](../../curvature-experiment/REPRODUCE.md) §5) | [`curvature-experiment/REPRODUCE.md`](../../curvature-experiment/REPRODUCE.md) |
| 23 | Weak ridge raises global OOF R² "by 0.10 to 0.15" ([main.tex:352](../../paper/latex/main.tex#L352)) | **wrong** | Records give +0.136 to +0.158 | Cache `09_physics_probe_facing_split_alpha1.jsonl` vs main |
| 24 | Decoder fits are good, "variance explained 0.952–0.985" (Appendix A, C) | **weakened (metric)** | VE is 1 − MSE / mean‖x‖² = 1 − MSE for unit rows, not relative to centred variance. On a cached PU DINOv3 embedding the centred variance is only 0.19–0.22, so centred VE could be far lower | [`09_physics_probe_facing_run.py:119`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L119) |
| 25 | QM9 chart dimension d from four ID estimators (QM:6-29) | **weakened** | `tle_dimensionality` is the MLE formula and `mind_mlk` is the median of the same per-point estimate, so three of four "estimators" are one statistic | [`src/effdim/geometry.py:34-86`](../../src/effdim/geometry.py#L34-L86), [`246-290`](../../src/effdim/geometry.py#L246-L290), [`348-393`](../../src/effdim/geometry.py#L348-L393) |
| 26 | II spectrum is low rank, "bends in a few dominant directions" (TMLR outline) | **unsupported** | Pre-stated rule gives "partial" for 8/10, "low" for 2/10; erank 22.3–42.4 of 136; full rank everywhere (cond 41–140); no decoder-prior null; no multi-direction validation | `results/ii-rank/*.json`; [`12_ii_rank_run.py:95-105`](../../curvature-experiment/runners/12_ii_rank_run.py#L95-L105) |
| 27 | Every run reproduces its reference exactly (rebuttal, RR, SC, QM) | **holds** as determinism | max diff 0 (split), ≤ 2.2e-15 (cf). Self-consistency of the same code on the same inputs, not an independent replication | RR guard rows; SC:67-79; QM:170-180 |

---

## 2. Pipeline map

Notation. n rows (86,471 galaxies; 130,744 molecules), D ambient width (384 to 4,096), d chart dimension (16 or 20 for
galaxies; 8 to 11 for QM9 plus 16), b = 512 anchors, k = 2,048 neighbours, m = d(d+1)/2.

### 2.1 Data and labels

- **Math.** Rows x ∈ S^{D−1} after row L2 normalisation. Labels are a positional join with the embedding rows
  (no id column exists).
- **Code.** [`pu_manifold/physics_labels.py`](../../curvature-experiment/pu_manifold/physics_labels.py): `PHYSICS_PARQUET_PATH` [:55](../../curvature-experiment/pu_manifold/physics_labels.py#L55), `EXPECTED_N_PHYSICS_ROWS = 86471` [:61](../../curvature-experiment/pu_manifold/physics_labels.py#L61),
  `LABEL_REPO/REVISION` = `Smith42/galaxies@v2.0` [:74-79](../../curvature-experiment/pu_manifold/physics_labels.py#L74-L79), `LABEL_COLUMN_MAP` [:97-102](../../curvature-experiment/pu_manifold/physics_labels.py#L97-L102), `SENTINEL_VALUES = (-99.0,)`
  [:148](../../curvature-experiment/pu_manifold/physics_labels.py#L148), `ALIGNMENT_ASSUMED_OFFSET = 0` [:192](../../curvature-experiment/pu_manifold/physics_labels.py#L192), `canonical_label` [:337-355](../../curvature-experiment/pu_manifold/physics_labels.py#L337-L355), `shifted_pairing` [:358-363](../../curvature-experiment/pu_manifold/physics_labels.py#L358-L363).
  Loading in [`09_physics_probe_facing_run.py:180-189`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L180-L189); other encoders through runner-level shims
  ([`09_physics_probe_facing_split_run.py:254-267`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L254-L267), normalisation at [:263-264](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L263-L264)).
- **Inputs/outputs.** X (n, D) float64; y (n,) with NaN for sentinels. photo_z 92.56% and stellar_mass 91.93%
  populated ([`physics_labels.py:107-120`](../../curvature-experiment/pu_manifold/physics_labels.py#L107-L120)).
- **Assumptions.** Row order identical between the embedding parquet and the 16 label shards concatenated in shard
  order. Tested by the D9-06 shift proof for ViT-B (Cache `09_row_alignment.jsonl`: R² at shift 0 = 0.516, gap
  0.516, passed). Other encoders rely on global OOF R² ≈ 0.5 as the alignment check ([REPRODUCE.md](../../curvature-experiment/REPRODUCE.md) §3.8).

### 2.2 Embeddings (sweep and QM9 only)

- Galaxy embeddings are downloaded from `UniverseTBD/pu-embeddings` at snapshot `bc081f8a…`
  ([`encoders.yaml`](../../curvature-experiment/encoders.yaml), [`sweep/setup_pod.sh:168-175`](../../curvature-experiment/sweep/setup_pod.sh#L168-L175)). QM9 embeddings are produced by [`sweep/embed.py`](../../curvature-experiment/sweep/embed.py) (mean pool over
  tokens, sha256 pinned per encoder in [`molecules.yaml`](../../curvature-experiment/molecules.yaml)). The QM9 label table comes from [`sweep/qm9_prepare.py`](../../curvature-experiment/sweep/qm9_prepare.py)
  (pinned HF revision [:178-181](../../curvature-experiment/sweep/qm9_prepare.py#L178-L181) and sha256 [:181](../../curvature-experiment/sweep/qm9_prepare.py#L181),183, expected counts [:188-189](../../curvature-experiment/sweep/qm9_prepare.py#L188-L189), canonical-SMILES dedup [:233-236](../../curvature-experiment/sweep/qm9_prepare.py#L233-L236),
  gap Hartree to eV [:237](../../curvature-experiment/sweep/qm9_prepare.py#L237)).
- Duplicate embeddings: [`sweep/embedding_duplicates.py:123-127`](../../curvature-experiment/sweep/embedding_duplicates.py#L123-L127); ChemBERTa-2 has 3,144 of 130,744 rows in
  duplicate groups (QM:53-60).

### 2.3 Autoencoder decoder

- **Math.** Plain AE, encoder and decoder MLPs with three hidden SiLU layers of 250, latent d, trained on the 80%
  train split to minimise mean ‖x − F(E(x))‖². Curvature is taken of the sphere-projected decoder F̂(z) = F(z)/‖F(z)‖.
- **Code.** `cae.PlainAutoEncoder` [`pu_manifold/cae.py:93-122`](../../curvature-experiment/pu_manifold/cae.py#L93-L122); loss [`cae.py:178`](../../curvature-experiment/pu_manifold/cae.py#L178); AdamW [`cae.py:156`](../../curvature-experiment/pu_manifold/cae.py#L156);
  `fit_decoder` [`09_physics_probe_facing_run.py:100-121`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L100-L121); `SphereProjectedDecoder`
  [`09_physics_curvature_run.py:84-99`](../../curvature-experiment/runners/09_physics_curvature_run.py#L84-L99).
- **Constants** ([`pu_manifold/physics_curvature_probe.py`](../../curvature-experiment/pu_manifold/physics_curvature_probe.py)). `SPLIT_SEED = 20260813` [:81](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L81), `HOLDOUT_FRACTION = 0.2` [:84](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L84),
  `AE_HIDDEN = (250,250,250)` [:90](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L90), `AE_ACTIVATION = "silu"` [:93](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L93), `MAX_EPOCHS = 600` [:96](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L96), `TORCH_INIT_SEED = 0` [:99](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L99),
  `TRAIN_CFG` lr 1e-3, weight decay 1e-4, batch 128, early stopping disabled [:102-111](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L102-L111).
- **Outputs.** z_anchor = E(x_anchor) (b, d); `var_explained` on the holdout rows [:119](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L119).
- **Assumptions.** The decoder's image near E(x₀) is the data manifold; the anchor x₀ is close to F̂(E(x₀)) (the
  counterfactual runner logs cos(image, data row), [`09_physics_normal_scaling_run.py:215-216`](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L215-L216)).

### 2.4 Geometry at the anchors

- **Math.** J = DF̂ (D×d), g = JᵀJ, Γ = g⁻¹JᵀD²F̂, II = D²F̂ − JΓ = P_N D²F̂ with P_N = I − Jg⁻¹Jᵀ,
  H = tr_g II = g^{ij} II_ij. Sphere split: II_rad = ⟨x̂, II⟩ (should equal −g), II_tan = II − x̂ ⊗ II_rad.
- **Code.** `decoder_geometry` [`09_physics_probe_facing_run.py:124-152`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L124-L152) (II at [:148-149](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L148-L149)); stored-array version
  `geometry_from_arrays` [`09_physics_probe_facing_split_run.py:109-117`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L109-L117); sphere split inside `split_columns`
  [:171-181](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L171-L181). The sealed trace-only path `decoder_curvature.plain_decoder_curvature`
  [`pu_manifold/decoder_curvature.py:158-266`](../../curvature-experiment/pu_manifold/decoder_curvature.py#L158-L266) is used as a cross-check (`median_cos_H_geo_vs_sealed`,
  [`09_physics_probe_facing_run.py:278-280`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L278-L280)).
- **Shapes.** J (b, D, d), Hess (b, D, d, d), image (b, D), stored float32 in the geometry npz (ViT-B d=16 about
  400 MB). GPU runs use forward mode (`jacfwd`) and score from the float32 copy so the npz and record agree
  ([`09_physics_probe_facing_split_run.py:62-72`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L62-L72)).
- **Assumptions.** J full column rank (g invertible); C² activation (guarded by `assert_c2_decoder`
  [`decoder_curvature.py:77-131`](../../curvature-experiment/pu_manifold/decoder_curvature.py#L77-L131)).

### 2.5 Probe and local R²

- **Math.** Ridge probe at α = 100. Outcome per anchor is the local out-of-fold R² over its k neighbours,
  r2 = 1 − Σ(y − ŷ_oof)²/Σ(y − ȳ_patch)². The geometry uses a separate whole-data in-sample ridge w.
- **Code.** OOF wrapper `physics_curvature_probe.oof_ridge_predictions` :661-708 (KFold 5, seed 20260902, RidgeCV
  with a duplicated single alpha); finite-row wrapper [`09_physics_curvature_run.py:64-81`](../../curvature-experiment/runners/09_physics_curvature_run.py#L64-L81); whole-data ridge
  [`09_physics_probe_facing_split_run.py:323-328`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L323-L328); `local_r2_panel` [`physics_curvature_probe.py:758-806`](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L758-L806);
  `knn_panel` [:742-755](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L742-L755) (sklearn exact kNN, the anchor is its own first neighbour).
- **Constants.** `ALPHA_RIDGE = 100` [:185](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L185), `N_OOF_FOLDS = 5` [:200](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L200), `OOF_FOLD_SEED = 20260902` [:203](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L203),
  `K_NEIGHBOURS = 2048` [:52](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L52), `N_ANCHORS = 512` [:62](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L62), `ANCHOR_DRAW_SEED = 20260902` [:65](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L65), `MIN_FINITE_NEIGHBOURS = 32`
  [:223](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L223). Anchors are drawn only from AE holdout rows (`anchor_indices` [:711-739](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L711-L739)).
- **Controls.** Sealed Z = [log r_k, local label variance, local count]; multiscale Z replaces log r_k by
  log r at k ∈ {16, 64, 256, 1024, 2048} (`MULTISCALE_KS`, [`09_physics_probe_facing_run.py:65`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L65); built at
  [`09_physics_probe_facing_split_run.py:315`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L315), [`326-327`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L326-L327)). All paper tables use multiscale.

### 2.6 Label Hessian and probe-facing tensors

- **Math.** For each anchor, u_i = g⁻¹Jᵀ(x_i − x₀) (tangent-projected coordinates), then least squares of the
  target on [1, u, ½u_i², u_iu_j (i<j)] so that the coefficients are (c, gradient, Hessian B). In Monge
  coordinates at an on-manifold anchor the Christoffel symbols vanish, so B is the covariant Hessian to leading
  order. Targets are the label y and the probe's own prediction p = Xw.
- **Code.** `quad_design` [`09_physics_probe_facing_split_run.py:120-126`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L120-L126); `local_quadratics` [:135-161](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L135-L161) (u at [:146](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L146));
  `split_columns` [:168-209](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L168-L209) computes w_N [:172-173](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L172-L173), `pf_full` = ⟨w_N, II⟩ [:176](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L176), `pf_tan` = ⟨w_N, II_tan⟩ [:177](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L177),
  `pf_rad` = (w_N·x̂) II_rad [:178-179](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L178-L179), all norms in the g metric (`metric_norms`
  [`09_physics_probe_facing_run.py:155-158`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L155-L158)), `hess_mismatch_dec` [:191](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L191), `hess_mismatch_emp` [:192](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L192),
  `align_cos_full/tan` [:194-195](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L194-L195).
- **Shapes.** hess_y and probe_emp (b, d, d); every column (b,). 153 regressors at d = 16, 231 at d = 20.

### 2.7 Partial Spearman and inference

- **Math.** Rank-transform x, y and each control; residualise rank x and rank y on [1, ranked controls]; report
  the Pearson correlation of the residuals. Freedman–Lane null: regress rank y on the controls, permute the
  residuals, add the fit back, recompute the partial; p = (1 + #|null| ≥ |obs|)/(B + 1).
- **Code.** `cross_split_curvature.partial_spearman` [`pu_manifold/cross_split_curvature.py:74-130`](../../curvature-experiment/pu_manifold/cross_split_curvature.py#L74-L130);
  `freedman_lane_y` [`physics_curvature_probe.py:852-872`](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L852-L872); `p_value_from_null` [:875-886](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L875-L886); `permutation_fwer`
  [:889-920](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L889-L920); per-column wrapper `partial_row` [`09_physics_probe_facing_run.py:85-94`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L85-L94).
- **Constants.** Runners use 2,000 permutations ([`09_physics_probe_facing_split_run.py:292`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L292)), so the p floor is
  1/2001 ≈ 5.0e-4. `N_PERMUTATIONS = 10000` in [`physics_curvature_probe.py:262`](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L262) is not what the paper used.
  `PERMUTATION_SEED = 20260902` [:268](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L268).
- **Cross-fit.** `--hessian-xfit` splits each patch at random into halves (seed 20260913), fits Hess y on one and
  scores local R² on the other, for the columns `hess_mismatch_dec`, `align_cos_tan`, `hess_label` only
  ([`09_physics_probe_facing_split_run.py:407-442`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L407-L442), column list [:432](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L432)).

### 2.8 Counterfactual normal scaling

- **Math.** At each anchor split w = w_T + w_rad + w_S (tangent, radial along x̂, in-sphere normal). Score the
  neighbours by ŷ_t = base + t·q + c with c refit, so SSE(t) = ‖e₀ − t q_c‖², t* = ⟨e₀,q_c⟩/‖q_c‖²,
  ΔR²(1) = (2⟨e₀,q_c⟩ − ‖q_c‖²)/SST. Variants differ in base and q:

  | variant | base | scaled term q |
  |---|---|---|
  | `S` | X(w_T + w_rad) | X w_S (data-side) |
  | `S_model` (published) | X(w_T + w_rad) | ½⟨w_S, II^S⟩(u,u) |
  | `S_proj` (the formula in [main.tex:250-252](../../paper/latex/main.tex#L250-L252)) | uᵀJᵀw_T − ½(w_N·x̂)uᵀgu | ½⟨w_S, II^S⟩(u,u) |
  | `random_qmatched` (published null) | X(w_T + w_rad) | ½⟨v, II^S⟩(u,u), v random in-sphere normal, rescaled to the same ‖q_c‖ |
  | `full`, `full_model`, `random_wnorm`, `random_matched` | variants not in the paper | |

- **Code.** `scaling_at_anchor` [`09_physics_normal_scaling_run.py:65-104`](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L65-L104) (decomposition [:73-76](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L73-L76), random direction
  [:77-79](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L77-L79), u [:81](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L81), q_S [:84](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L84), random rescaling [:87-92](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L87-L92), S_proj base [:93](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L93), SSE algebra [:98-103](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L98-L103)). `T_GRID` [:61](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L61).
  Per-anchor loop [:243-252](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L243-L252). Ridge w at α = 100 [:230](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L230). Seed 20260915 [:140](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L140).
- **Summaries.** help = fraction with r2(1) > r2(0), hurt = fraction with r2(−1) < r2(0), t* median
  ([`sweep/extract.py:57-72`](../../curvature-experiment/sweep/extract.py#L57-L72), identical in [`appendix_gen.py:247-251`](../../paper/generate/appendix_gen.py#L247-L251)).
- **Thinning.** Pairwise neighbourhood overlap ([`09_physics_normal_scaling_thin_run.py:317-323`](../../curvature-experiment/runners/09_physics_normal_scaling_thin_run.py#L317-L323)); min-degree greedy
  independent set at overlap ≤ 5% of k for the paper's sign tests ([`appendix_gen.py:254-260`](../../paper/generate/appendix_gen.py#L254-L260), ported as
  [`sweep/extract.py:75-80`](../../curvature-experiment/sweep/extract.py#L75-L80)), one-sided binomial test [`extract.py:89-102`](../../curvature-experiment/sweep/extract.py#L89-L102).

### 2.9 Robustness battery (runner 11)

- **Code** [`11_review_robustness_run.py`](../../curvature-experiment/runners/11_review_robustness_run.py). Tuned α* by RidgeCV (GCV/LOO) over 15 log-spaced values 1e-3..1e4
  ([:52](../../curvature-experiment/runners/11_review_robustness_run.py#L52), [:83-87](../../curvature-experiment/runners/11_review_robustness_run.py#L83-L87)); nested OOF with the grid inside each outer fold ([:65-80](../../curvature-experiment/runners/11_review_robustness_run.py#L65-L80)); extended controls add `hess_label` and
  roughness = 1 − label linear R² in the chart ([:108-115](../../curvature-experiment/runners/11_review_robustness_run.py#L108-L115)); 32-block cluster bootstrap with average-linkage blocks on
  1 − overlap ([:122-155](../../curvature-experiment/runners/11_review_robustness_run.py#L122-L155), sensitivity 16 and 64, 2,000 resamples); thinned partial at overlap ≤ 10% ([:158-162](../../curvature-experiment/runners/11_review_robustness_run.py#L158-L162));
  held-out ΔR² of local R² from adding mismatch and alignment to the extended controls, fit on half the blocks, 20
  splits, OLS on ranks ([:165-189](../../curvature-experiment/runners/11_review_robustness_run.py#L165-L189)); surrogate fidelity = Spearman and median |ΔR²_S_model(1) − ΔR²_S(1)| ([:224-228](../../curvature-experiment/runners/11_review_robustness_run.py#L224-L228)).
- **Reproduction guard.** At α = 100 recompute the published split partials (tolerance 1e-6 "exact" or 0.02
  "refit", [:192](../../curvature-experiment/runners/11_review_robustness_run.py#L192)) and the cf summaries (1e-12, hard-coded [:258](../../curvature-experiment/runners/11_review_robustness_run.py#L258)); refuse to write on any difference or missing
  reference cell ([:245-281](../../curvature-experiment/runners/11_review_robustness_run.py#L245-L281)). Requires the same BLAS thread count as the cf run ([:381-385](../../curvature-experiment/runners/11_review_robustness_run.py#L381-L385)).

### 2.10 Tensor fidelity (runner 10)

- Synthetic in-sphere fixture with exact geometry, labels `lin`, `nonlin`, `lam{0,0.5,1,2}` (`make_labels`
  [`10_tensor_fidelity_run.py:153-175`](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L153-L175)), truth covariant Hessian d²f − Γᵏ∂_k f ([:178-185](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L178-L185)), chart-free comparison by
  ambient lift ([:66-90](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L66-L90)). Pass lines `TENSOR_COS_PASS = 0.8`, `MISMATCH_RHO_PASS = 0.7` ([:60-61](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L60-L61)). Probe α = 100
  ([:235](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L235)). `rho_mismatch` scores `hess_mismatch_dec` ([:249](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L249)); `rho_align` scores `align_cos_full` ([:250](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L250)).

### 2.11 Sweep and QM9

- [`sweep/jobs.py:102-222`](../../curvature-experiment/sweep/jobs.py#L102-L222) expands each encoder into `main_xfit` (seed 0, xfit, saves geometry), `seed1`, `seed2`,
  `cf`, `thin`, `robust`, plus `main_d16` for molecules when d_run ≠ 16. Split jobs refit the decoder on GPU with
  `--deterministic` ([:195](../../curvature-experiment/sweep/jobs.py#L195)). The robust job's guard compares against the sweep's own `main_xfit` and `cf` outputs
  ([:214-219](../../curvature-experiment/sweep/jobs.py#L214-L219)).
- [`sweep/run_queue.py`](../../curvature-experiment/sweep/run_queue.py) schedules jobs, marks done, and by default deletes the geometry npz once `thin` and
  `robust` are done (`prune_geometry` [:99-140](../../curvature-experiment/sweep/run_queue.py#L99-L140); `--keep-geometry` [:278](../../curvature-experiment/sweep/run_queue.py#L278)).
- [`sweep/aggregate.py`](../../curvature-experiment/sweep/aggregate.py) writes the reports (claim counts [:295-359](../../curvature-experiment/sweep/aggregate.py#L295-L359); GPU vs CPU comparison [:362-436](../../curvature-experiment/sweep/aggregate.py#L362-L436) with the ViT-B seed
  spread as tolerance [:367-374](../../curvature-experiment/sweep/aggregate.py#L367-L374)).
- QM9 d: [`sweep/intrinsic_dim.py`](../../curvature-experiment/sweep/intrinsic_dim.py), 10,000-row subsample (seed 20261001), k = 10, `d_ID = round(median(mle,
  two_nn, tle, mind_mlk))`, `d_run = min(d_ID, 20)` ([:33-48](../../curvature-experiment/sweep/intrinsic_dim.py#L33-L48)).

### 2.12 II spectrum (runner 12)

- **Math.** In an orthonormal frame (J = QR, u = Rz) II_on(p,q) = II(R⁻¹e_p, R⁻¹e_q); remove the radial part;
  flatten Frobenius-preserving to a D × m matrix; singular values; effective rank, participation ratio, k90, k99,
  condition number; Gaussian D × m reference.
- **Code.** [`12_ii_rank_run.py:55-105`](../../curvature-experiment/runners/12_ii_rank_run.py#L55-L105) (`ii_spectra` [:75-92](../../curvature-experiment/runners/12_ii_rank_run.py#L75-L92)). Decision rule `FULL_FRAC = 0.9`, `LOW_FRAC = 0.25`
  ([:50-51](../../curvature-experiment/runners/12_ii_rank_run.py#L50-L51), [:100-105](../../curvature-experiment/runners/12_ii_rank_run.py#L100-L105)). Reads the sweep geometry `geometry/<enc>/09_probe_facing_geometry_d16_seed0.npz` ([:147](../../curvature-experiment/runners/12_ii_rank_run.py#L147)).

---

## 3. Logic audit per stage

Each subsection says whether the math is right, whether the code implements it, what tests cover it, and what is
known to be biased or wrong.

### 3.1 Data and alignment

- Math and code agree. Sentinel masking happens before any statistic.
- Verified by tests: [`test_physics_labels.py`](../../curvature-experiment/tests/test_physics_labels.py) (shard order, sentinel masking, shifted pairing, alignment
  verdict), `test_load_physics_rejects_wrong_row_count`.
- Gap. The galaxy embeddings for the original five runs were read via `hf://` without a pinned revision
  ([`physics_labels.py:55`](../../curvature-experiment/pu_manifold/physics_labels.py#L55), [REPRODUCE.md](../../curvature-experiment/REPRODUCE.md) §2 says so). The sweep pins the snapshot. [`setup_pod.sh:171-173`](../../curvature-experiment/sweep/setup_pod.sh#L171-L173) skips the
  download when a non-empty file exists and does not re-verify it; galaxy parquet files have no sha256 in
  [`encoders.yaml`](../../curvature-experiment/encoders.yaml).

### 3.2 Decoder fit

- Correct as implemented. The C² guard prevents a ReLU decoder silently returning II = 0.
- **Questionable metric.** `var_explained = 1 − mse_total / mean‖x‖²` ([`09_physics_probe_facing_run.py:119`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L119)). For
  unit rows mean‖x‖² = 1, so this is 1 − MSE, the share of the raw second moment, not of the centred variance.
  On a locally cached PU embedding (`legacysurvey_dinov3_vitb16.parquet`, not the physics config) ‖mean‖² is
  0.78–0.81, so the centred total variance is 0.19–0.22. If the galaxy configs are similar, the quoted 0.966 would
  be roughly 1 − 0.034/0.2 ≈ 0.83 in centred terms. **Unverified on the actual galaxy data (pod-only).** Comparisons
  of real VE with the synthetic fixture's VE (Sanity I4) are not like for like for the same reason.
- Verified by tests: `test_plain_autoencoder_matches_eq22_shape`, `test_fit_decoder_cpu_device_matches_default`,
  `test_assert_c2_decoder_*`.

### 3.3 Geometry, II and the sphere split

- **Math right, code right.** II = Hess − JΓ with Γ = g⁻¹JᵀHess is exactly P_N Hess
  ([`09_physics_probe_facing_run.py:148-149`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L148-L149)). The runner 12 form Hess − Jg⁻¹JᵀHess ([`12_ii_rank_run.py:81-82`](../../curvature-experiment/runners/12_ii_rank_run.py#L81-L82)) is
  the same. The orthonormal-frame change II_on = R⁻ᵀ II R⁻¹ ([`12_ii_rank_run.py:83-85`](../../curvature-experiment/runners/12_ii_rank_run.py#L83-L85)) is correct.
- Sphere split. Because F̂ maps into the unit sphere, x̂ is normal (JᵀF̂ = 0) and ⟨x̂, II⟩ = −g. The record checks
  confirm this to float32 precision (`II_rad_vs_minus_g_max_rel` 1.5e-8, `JT_xhat_max` 2e-9; II spectrum
  `radial_dev_max` 3.7e-8 to 3.2e-7). pf_tan = ⟨w_N, II_tan⟩ equals the paper's ⟨w_S, II^S⟩ because II_tan ⊥ x̂.
- Verified by tests: `test_plain_decoder_curvature_dxd_solve_matches_explicit_projector`,
  `test_plain_decoder_curvature_sphere_known_answer`, `test_radial_decomposition_on_analytic_sphere`,
  `test_decoder_geometry_forward_mode_matches_reverse`, `test_recovers_rank_and_radial_part_in_a_skewed_chart`,
  `test_probe_facing_tensors_match_split_columns_norms`, Swiss roll notebook
  `notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb` (per `SWISS_ROLL_APPLICABILITY_RULE`,
  [`physics_curvature_probe.py:481-487`](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L481-L487)).
- Known limit. II is evaluated at F̂(E(x₀)), the decoder image, while neighbour coordinates are measured from the
  data anchor x₀ ([`09_physics_probe_facing_split_run.py:145-146`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L145-L146)). The offset enters every downstream quantity.

### 3.4 Residual expansion and help condition (theory)

- Eq. 3 is right for u ~ N(0, s²I): Var(½uᵀΔu) = ½s⁴‖Δ‖²_F and odd cross moments vanish.
- **Known bias, confirmed.** kNN patches are closer to uniform balls than Gaussians. For a spherically symmetric
  patch with E u₁⁴ = 3λs⁴, Cov(½A(u,u), ½B(u,u)) = (s⁴/4)[2λ⟨A,B⟩ + (λ−1) trA trB] (Outline I2). My Monte Carlo at
  d = 16 (400,000 draws, uniform ball) gives λ = 0.9003 against (d+2)/(d+4) = 0.9000, Cov 0.02116 against
  0.02117 from the corrected formula, and 0.121 from the Gaussian formula for trace-heavy test tensors. The paper's
  g-metric alignment and help condition weight the trace direction wrongly for real patches. The direction of the
  effect on the reported partials is not known.

### 3.5 Label Hessian

- **Math right in the ideal case, code right.** My check 1a puts an exact quadratic label on an exact Monge surface
  in R¹⁰. `local_quadratics` recovers the true Hessian to 9e-15. The ½ on the diagonal design columns
  ([`09_physics_probe_facing_split_run.py:125`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L125)) gives B_ii and B_ij directly.
- **Tilted-tangent leakage, confirmed.** With a linear label (true Hessian 0) and the decoder tangent tilted into
  the normal space by ε, the estimate is 0.072, 0.146, 0.310 at ε = 0.05, 0.1, 0.2 (my check 3). The bias is linear
  in ε and in the label gradient, and it is II-shaped (Test-1 C3). With real decoders at imperfect fit a few degrees
  of tilt is plausible and has not been measured (no J vs local-PCA angles anywhere in the code).
- **Magnitude error.** TF at paper scale: relerr hess_y 0.289–1.177 at noise 0, 0.463–1.247 at noise 0.25, with
  cosines 0.914–0.997 (TF:105-116). Direction is right; the norm is off by 29–118%. Whether it is inflated or
  shrunk cannot be read from the records, since relerr is unsigned ([`10_tensor_fidelity_run.py:82-90`](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L82-L90)). For `lin`
  (cos 0.978, relerr 0.880) both λ = 1.83 and λ = 0.12 satisfy the pair (medians are over anchors, so this is only
  indicative). The "about 2x too large" statement (RR:228; rebuttal §1) is unsupported.
- **Reliability on real labels.** Split-half tensor cosine p50 0.19–0.35; split-half norm rank 0.91–0.94 (Cache
  xfit record, verified). So the norm ranks well and the direction is weak.
- Tests. No unit test calls `local_quadratics` or `quad_design` directly. They are exercised end to end by
  `[test_tensor_fidelity.py](../../curvature-experiment/tests/test_tensor_fidelity.py)::test_end_to_end_smoke` and `[test_review_robustness.py](../../curvature-experiment/tests/test_review_robustness.py)::test_smoke_end_to_end`, and their
  output is validated against truth only through the tensor-fidelity records.

### 3.6 The two mismatch columns and the empirical identity

- `hess_mismatch_dec` = ‖hess_y − ⟨w_N, II⟩‖_g uses the decoder. `hess_mismatch_emp` = ‖hess_y − probe_emp‖_g where
  probe_emp is the local quadratic fit of p = Xw ([`09_physics_probe_facing_split_run.py:191-192`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L191-L192)).
- **Identity.** Least squares is linear in the target, so on the same rows hess_y − probe_emp is exactly the
  quadratic coefficient of the residual y − Xw. My check 1c confirms it to 1e-14. For photo_z and stellar_mass the
  masks differ slightly (the label has NaNs, p does not), so the identity is approximate there.
- So `hess_mismatch_emp` = ‖Hess_M(y − w·x)‖, the curvature of the probe residual in the decoder chart. The decoder
  enters only through J and g. In the ideal noise-free on-manifold case probe_emp equals ⟨w_N, II⟩ exactly (my check
  1b, 2.6e-15), so it is a decoder-free estimate of K. On galaxies its median cosine with the decoder K is 0.57–0.67
  (full) and 0.55–0.65 (in-sphere) (Cache split record `checks`).
- **Every paper table uses `hess_mismatch_emp`** ([`appendix_gen.py:19`](../../paper/generate/appendix_gen.py#L19), [`90`](../../paper/generate/appendix_gen.py#L90), [`99`](../../paper/generate/appendix_gen.py#L99); [`table_main_gen.py:9`](../../paper/generate/table_main_gen.py#L9)). The paper
  defines Δ = Hess_M y − ⟨w_N, II⟩ ([main.tex:127](../../paper/latex/main.tex#L127), 304). That is the dec column.
- **Mismatch ≈ label-Hessian norm.** Across the 40 galaxy cells of the ten-encoder sweep the |partial(emp) −
  partial(‖Hess y‖)| gap has median 0.024; 36/40 ≤ 0.05; 27/40 ≤ 0.03; largest 0.111 (ViT-B mag_r, −0.387 vs −0.276)
  and 0.079 (ViT-B photo_z). Recomputed from Cache `scaling/records/*__main_xfit.jsonl`. ViT-B, the headline
  encoder, is where the two are most separated.
- **Near-circularity.** The outcome is the local R² of the OOF residuals over the same 2,048 neighbours, and
  emp is the norm of the quadratic part of (nearly) those residuals. The cross-fit that would break this coupling
  exists only for the dec column (Section 3.7).

### 3.7 Partial Spearman, Freedman–Lane, cross-fitting

- **Math right, code right.** `partial_spearman` residualises ranks on ranked controls with intercept and
  correlates residuals ([`cross_split_curvature.py:74-130`](../../curvature-experiment/pu_manifold/cross_split_curvature.py#L74-L130)). Freedman–Lane permutes rank-residuals of the outcome under
  the reduced model and adds the fit back ([`physics_curvature_probe.py:852-872`](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L852-L872)). The surrogate is re-ranked inside
  `partial_spearman`; with ties absent this is harmless.
- Tests: `test_partial_spearman_*` (4), `test_controlled_partial_reproduces_colleague_numbers`,
  `test_freedman_lane_preserves_control_fit`, `test_p_value_never_zero`.
- **Known violated assumption.** Permutation treats the 512 anchors as exchangeable. They are not. Each point lies
  in about 12 neighbourhoods on average (512 × 2048 / 86,471 = 12.1). From the stored overlap matrices, 42–59% of
  anchor pairs share at least one point and each anchor shares points with 214–299 others on average (my block
  check). The paper's Limitations sentence ([main.tex:309](../../paper/latex/main.tex#L309)) acknowledges this; the tables still print permutation
  stars.
- **Reporting defect (confirmed).** The cross-fit loop only cross-fits `hess_mismatch_dec`, `align_cos_tan` and
  `hess_label` ([`09_physics_probe_facing_split_run.py:432`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L432)). Appendix B and Table `tab:xencx` print the dec column
  under the heading "mismatch, cross-fit" ([`appendix_gen.py:62`](../../paper/generate/appendix_gen.py#L62), [`224`](../../paper/generate/appendix_gen.py#L224)), and the Appendix C prose "cross-fitted −0.28
  to −0.62" is computed from it ([`appendix_gen.py:100`](../../paper/generate/appendix_gen.py#L100)). Main tables use emp. For ViT-B d = 16 the in-sample dec
  partial is −0.304 and its cross-fit is −0.381/−0.336, so the dec column survives cross-fitting. Nothing shows the
  emp column (−0.387) survives it.
- **Labelling defect (confirmed, small).** The paper's alignment is cos_g(Hess_M y, K) ([main.tex:187](../../paper/latex/main.tex#L187), 390). The
  tables use `align_cos_tan`, cosine with K_S ([`09_physics_probe_facing_split_run.py:195`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L195), [`appendix_gen.py:19`](../../paper/generate/appendix_gen.py#L19)).
  For ViT-B the two differ by at most 0.02 (my [`check_records.py`](2026-10-05-scripts/check_records.py)).

### 3.8 Probe and controls

- Ridge without feature standardisation in both the OOF and the geometry fits ([`linear_probe.py:106-109`](../../curvature-experiment/pu_manifold/linear_probe.py#L106-L109),
  `RidgeCV` with fit_intercept only). The two fits use the same α; the geometry w is fit on all finite rows, the
  outcome uses OOF predictions. Test `test_oof_predictions_are_out_of_fold`.
- The sphere term `pf_rad` ≈ √d |w·x̂| is a function of how extreme the prediction is. Its negative partial is
  partly shrinkage: at α = 1 it vanishes for photo_z and stellar mass (Appendix B; records verified). The paper says
  so ([main.tex:583](../../paper/latex/main.tex#L583)).

### 3.9 Counterfactual

- **SSE formula right, code right.** My check 2a computes r2 at each t by brute-force intercept refit and matches
  the runner's `r2_curve` to 2e-16. Check 2c confirms variant `S` at t = 1 is the global readout with a local
  intercept.
- **Help/hurt identity (confirmed).** SSE(t) − SSE(0) = ‖q_c‖²(t² − 2t t*). So help ⇔ t* > ½ and hurt ⇔ t* > −½,
  hence help ⊂ hurt. Holds at 100% of anchors in all 24 published cells (my [`check_records.py`](2026-10-05-scripts/check_records.py), also Sanity §2).
  The "hurt at 97–100%" is therefore not independent evidence. Random hurt is 0.64–0.80 at α = 100 because random t*
  sits near 0 (medians −0.16 to +0.06).
- **What t* contains.** With p = centred X w_S on the patch, r_c the centred residual of the global probe and
  q the decoder quadratic, the stored quantities satisfy t*_S = 1 + ⟨r_c, p⟩/‖p‖² and
  t*_model = ⟨r_c + p, q⟩/‖q‖² = β(p|q) + ⟨r_c, q⟩/‖q‖², with β = ⟨p,q⟩/‖q‖² (Outline, corrected proposition).
  I re-derived this from `scaling_at_anchor` (both variants share e₀ = r_c + p because they share the base). The
  ridge normal equations give a pooled t*_S ≥ 1, not a per-anchor one.
- **Surrogate amplitude (confirmed).** Median ‖q‖/‖p‖ = sqrt(qq_S_model/qq_S) is 0.38–0.51 over the 24 published
  cells (my recomputation; matches the controller's figure). On QM9 it is 0.29–0.55. So the decoder's second-order
  term reproduces under half of the probe's own in-sphere readout amplitude on a patch.
- **Surrogate fidelity.** Per-anchor Spearman between ΔR²(1) of S_model and of S is +0.35 to +0.82 at α = 100 and
  +0.16 to +0.66 at α* on galaxies (RR:185-224); −0.30 to +0.23 on QM9 (verified). On QM9 the data-side S helps at
  0.88–1.00 of anchors while S_model helps at 0.28–0.74 (verified).
- **Least-squares null.** The Novelty review's simulation (labels with no designed curvature relation) reproduces
  help 76–100%, hurt 80–100%, random help 15–45% and t* 2–3 at comparable shrinkage (Novelty §6 table). I did not
  rerun it. Its algebra agrees with mine.
- **Reporting defect (confirmed, numerically immaterial).** [main.tex:250-252](../../paper/latex/main.tex#L250-L252) writes the surrogate as
  c + aᵀu + ½K_sph(u,u) + (t/2)K_S(u,u), which is variant `S_proj`. The table, Figure 1 and the sign tests use
  `S_model` ([`make_fig_intervention.py:17`](../../paper/latex/figures/make_fig_intervention.py#L17), [`appendix_gen.py:247`](../../paper/generate/appendix_gen.py#L247)). From the stored arrays the two differ by ≤ 0.01
  in help and ≤ 0.02 in median t* in every cell. The rebuttal already promises to fix the wording.
- Tests. No unit test calls `scaling_at_anchor`. Covered end to end by `test_smoke_end_to_end` and the extract
  tests (`test_cf_summary_matches_tab_cf`), and by the 1e-12 reproduction guard.

### 3.10 Robustness battery

- **Tuned probe.** α* is chosen by GCV on all rows and sets w (so w_N, emp, alignment and the counterfactual);
  OOF uses fold-wise selection ([`11_review_robustness_run.py:65-87`](../../curvature-experiment/runners/11_review_robustness_run.py#L65-L87)). Leak-free for the OOF outcome
  (`test_tuned_oof_never_uses_held_out_rows`). α* is at the grid floor 0.001 in 5/20 galaxy pairs (RR:18-28) and
  in 28/32 QM9 pairs (my [`check_qm9.py`](2026-10-05-scripts/check_qm9.py)). The tensor-fidelity validation ran only at α = 100
  ([`10_tensor_fidelity_run.py:235`](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L235)), so the near-OLS w_N used at α* is unvalidated. Nothing in runner 11 is
  cross-fitted (`split_quantities` [:108-111](../../curvature-experiment/runners/11_review_robustness_run.py#L108-L111) fits on the full panel).
- **Block bootstrap.** Correct percentile cluster bootstrap. Blocks are unbalanced: with 32 blocks, the largest has
  50–57 anchors and the median 9–16, with 1–3 singletons (my check on the five published overlap matrices). Blocks
  are built from anchor overlap, so neighbouring blocks still share points. Tests:
  `test_cluster_bootstrap_covers_truth`, `test_bootstrap_skips_degenerate_replicates`, `test_overlap_blocks_count`.
- **Held-out ΔR².** Correct as written. Ranks are computed on all anchors before the split
  ([`11_review_robustness_run.py:178-179`](../../curvature-experiment/runners/11_review_robustness_run.py#L178-L179)), which leaks only the rank scale. "Held out" means other anchor blocks on
  the same dataset, which share boundary points. Test `test_heldout_positive_with_signal_zero_with_noise`.
- **Extended controls.** Adding |Hess y| and roughness makes the mismatch partial larger in magnitude in 13/20 pairs
  at α* (counted from RR:17-36 vs RR:71-90), the suppression pattern RR itself warns about. RR:40's
  "over-controls rather than under-controls" is not supported.
- **Reproduction guard.** Correct and strict (refuses on missing reference cells). Tests
  `test_guard_stops_on_perturbed_reference`, `test_guard_refuses_missing_or_incomplete_reference`,
  `test_reference_uses_d16_rows_only`. For four encoders the run used `--guard refit` (tolerance 0.02 for partials)
  rather than the spec's 1e-6 ([`docs/superpowers/specs/2026-09-30-review-robustness-design.md:118-120`](../superpowers/specs/2026-09-30-review-robustness-design.md#L118-L120)); the
  observed diffs are 0, so the relaxation is moot. In the sweep the guard compares the robust job with the sweep's
  own split and cf jobs, so it is a determinism check, not a replication.

### 3.11 II spectrum and intrinsic dimension

- **II spectrum math right, code right.** Tests `test_sym_flatten_keeps_frobenius_norm`,
  `test_spectrum_metrics_on_known_spectra`, `test_full_rank_when_enough_directions`,
  `test_random_reference_is_near_full`, `test_recovers_rank_and_radial_part_in_a_skewed_chart`.
- **Results** (verified from `results/ii-rank/*.json`). Median erank 22.3–42.4, participation ratio 9.9–22.1, k90
  29–50, k99 78–104 of m = 136, condition number 41–140 (Gaussian reference 1.4–3.9). Verdicts: "low" for
  dinov3_vits16 and dinov3_vits16plus, "partial" for the other eight.
- **Decoder-prior confound, not addressed.** The only reference is a single Gaussian D × m matrix
  ([`12_ii_rank_run.py:95-97`](../../curvature-experiment/runners/12_ii_rank_run.py#L95-L97)). No random-init decoder, no AE on a covariance-matched cloud, no column-permuted
  embeddings, no decoder-free II, no seed subspace overlap (singular vectors were not saved, [:152](../../curvature-experiment/runners/12_ii_rank_run.py#L152)), no d sweep. The
  tensor-fidelity fixture has a single in-sphere normal direction, so spectrum recovery is unvalidated.
- **Intrinsic dimension (library defect, confirmed).** `tle_dimensionality` computes (k−1)/Σ ln(r_k/r_j) averaged over
  points ([`src/effdim/geometry.py:348-393`](../../src/effdim/geometry.py#L348-L393)), which is the same formula as `mle_dimensionality` ([:34-86](../../src/effdim/geometry.py#L34-L86)).
  `mind_mlk_dimensionality` is the median of the same per-point estimate ([:246-290](../../src/effdim/geometry.py#L246-L290)), not the MiND-MLk likelihood. So
  `choose_d`'s median of four ([`sweep/intrinsic_dim.py:46-48`](../../curvature-experiment/sweep/intrinsic_dim.py#L46-L48)) is effectively a function of the Levina–Bickel
  statistic alone, with two_nn (broken by ChemBERTa duplicates, 0.85–1.04) as the low outlier. QM:19-29 discloses the
  tle = mle equality. The d = 16 replicate gives the same mismatch counts (QM:71-76), so the conclusion does not
  hinge on d.

### 3.12 Consolidated list of confirmed defects

| # | Defect | Where | Effect |
|---|---|---|---|
| D1 | Cross-fit appendix prints `hess_mismatch_dec` under "mismatch, cross-fit"; main tables use `hess_mismatch_emp`; emp never cross-fitted | [`09_physics_probe_facing_split_run.py:432`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L432); [`paper/generate/appendix_gen.py:62`](../../paper/generate/appendix_gen.py#L62), [`100`](../../paper/generate/appendix_gen.py#L100), [`224`](../../paper/generate/appendix_gen.py#L224); [`main.tex:190`](../../paper/latex/main.tex#L190), [`424`](../../paper/latex/main.tex#L424) | The paper's cross-fitting defence does not apply to the tabulated column |
| D2 | Δ defined with ⟨w_N, II⟩, tables use emp (no decoder II) | [`main.tex:127`](../../paper/latex/main.tex#L127), [`304`](../../paper/latex/main.tex#L304) vs [`09_physics_probe_facing_split_run.py:192`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L192), [`appendix_gen.py:19`](../../paper/generate/appendix_gen.py#L19) | The headline association is about residual curvature, not decoder probe-facing curvature (Sanity I6, Outline C4) |
| D3 | Alignment defined with K, tables use K_S | [`main.tex:187`](../../paper/latex/main.tex#L187), [`390`](../../paper/latex/main.tex#L390) vs [`09_physics_probe_facing_split_run.py:195`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L195) | ≤ 0.02 numerically |
| D4 | Main-text surrogate formula is `S_proj`, reported numbers are `S_model` | [`main.tex:250-252`](../../paper/latex/main.tex#L250-L252) vs [`make_fig_intervention.py:17`](../../paper/latex/figures/make_fig_intervention.py#L17), [`appendix_gen.py:247`](../../paper/generate/appendix_gen.py#L247) | ≤ 0.01 help, ≤ 0.02 t* |
| D5 | "Global out-of-sample R² rises by 0.10 to 0.15" at α = 1 | [`appendix_gen.py:51`](../../paper/generate/appendix_gen.py#L51) → [`main.tex:352`](../../paper/latex/main.tex#L352) | Records give +0.136 to +0.158 |
| D6 | `tle_dimensionality` equals MLE; `mind_mlk` is median Levina–Bickel | [`src/effdim/geometry.py:246-290`](../../src/effdim/geometry.py#L246-L290), [`348-393`](../../src/effdim/geometry.py#L348-L393) | QM9 d effectively from one statistic |
| D7 | "Variance explained" is uncentred | [`09_physics_probe_facing_run.py:119`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L119) | Overstates decoder fit; size unverified on galaxy data |
| D8 | Appendix E mean-bending trace range not produced by this repository | [`main.tex:585`](../../paper/latex/main.tex#L585); [REPRODUCE.md](../../curvature-experiment/REPRODUCE.md) §5 | Unverifiable from the code here |
| D9 | "Every significant sign of the main table is retained" | [`appendix_gen.py:25`](../../paper/generate/appendix_gen.py#L25) → [`main.tex:319`](../../paper/latex/main.tex#L319) | Signs are retained but mag_r shape loses significance in seed 1 and 400³ (+0.05*); wording is loose, not false |

---

## 4. Reproducibility and provenance

**Pins.** [`curvature-experiment/requirements.txt`](../../curvature-experiment/requirements.txt) pins torch 2.13.0+cpu, numpy 2.5.1, scipy 1.18.0, scikit-learn
1.9.0, faiss-cpu 1.14.3, pandas 3.0.5, pyarrow 25.0.0, rdkit 2026.3.6, transformers 5.16.1. The pod installs the
same list with a CUDA torch ([`sweep/setup_pod.sh:98-114`](../../curvature-experiment/sweep/setup_pod.sh#L98-L114)). `pyproject.toml` leaves `src/effdim` deps unpinned. Records
carry numpy, torch, sklearn, scipy and Python versions in their environment rows.

**Seeds.**

| seed | value | where |
|---|---|---|
| AE train/holdout split | 20260813 | [`physics_curvature_probe.py:81`](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L81) |
| anchor draw, OOF folds, permutation null | 20260902 | `:65, :203, :268` |
| decoder init | 0 (1, 2 for ablations) | `:99`; `--fit-seed` |
| xfit half split | 20260913 | [`09_physics_probe_facing_split_run.py:408`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L408) |
| counterfactual random direction | 20260915 | [`09_physics_normal_scaling_run.py:140`](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L140); [`11_review_robustness_run.py:61`](../../curvature-experiment/runners/11_review_robustness_run.py#L61) |
| bootstrap and held-out splits | 20260930 | [`11_review_robustness_run.py:60`](../../curvature-experiment/runners/11_review_robustness_run.py#L60) |
| tensor-fidelity fixture, labels | 20260905, 20260930 | `[10_tensor_fidelity_run.py:188](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L188), 141` |
| QM9 ID subsample | 20261001 | [`sweep/intrinsic_dim.py:26`](../../curvature-experiment/sweep/intrinsic_dim.py#L26) |

**sha256 checks in code.** Runner 11 refuses a geometry npz or label table whose sha differs from the one passed
([`11_review_robustness_run.py:400-405`](../../curvature-experiment/runners/11_review_robustness_run.py#L400-L405)) and records sha of the published references ([:444-445](../../curvature-experiment/runners/11_review_robustness_run.py#L444-L445)). Split runner records
the label-table sha and the saved geometry sha (`[09_physics_probe_facing_split_run.py:271-286](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L271-L286), 345-352`). Counterfactual
runner records geometry and label sha (`[09_physics_normal_scaling_run.py:205](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L205), 177`). QM9: pinned HF revision and
parquet sha ([`qm9_prepare.py:178-183`](../../curvature-experiment/sweep/qm9_prepare.py#L178-L183)), label table sha in [`molecules.yaml`](../../curvature-experiment/molecules.yaml), embedding sha per encoder, verified on the
pod ([`setup_pod.sh:147-166`](../../curvature-experiment/sweep/setup_pod.sh#L147-L166)), d-file sha in every split/cf/robust record (QM:186-197). `results/scaling/SHA256SUMS`
and `results/qm9/SHA256SUMS` list record and array hashes.

**Code version.** `repo_head` in records is the tree the run started from, not a pin. Records dated 2026-09-12 to
09-15 carry `71914dd` although they used flags added later ([REPRODUCE.md](../../curvature-experiment/REPRODUCE.md) §5). The published ViT-B split record was
produced by a runner that was untracked at the time ([REPRODUCE.md](../../curvature-experiment/REPRODUCE.md) §3.5). The reproduction guard at 1e-6 shows today's
code reproduces it exactly from the same geometry, which closes most of that gap.

**What is regenerable, and from what.**

| artefact | regenerable from | needs |
|---|---|---|
| Paper appendix tables and prose | Cache records via [`paper/generate/appendix_gen.py`](../../paper/generate/appendix_gen.py) (not run in this audit) | local Cache (8.9 GB, git-ignored) |
| [`SCALING_REPORT.md`](../../curvature-experiment/results/scaling/SCALING_REPORT.md), [`QM9_REPORT.md`](../../curvature-experiment/results/qm9/QM9_REPORT.md) | Cache `scaling/` and `qm9/` records and arrays via [`sweep/aggregate.py`](../../curvature-experiment/sweep/aggregate.py); tests `test_committed_results_regenerate`, `test_qm9_results_regenerate` | local Cache |
| `RR` and `TF` reports | committed records under `results/*/records/` via `11_…_report.py`, `10_…_report.py` | nothing else |
| Tensor-fidelity records | [`10_tensor_fidelity_run.py`](../../curvature-experiment/runners/10_tensor_fidelity_run.py) alone (synthetic) | CPU or GPU time; paper scale is long |
| Split, cf, robust records for galaxies | runners 09/11 | pod-only geometry npz, pod label parquet, embeddings at snapshot `bc081f8a` |
| Sweep and QM9 records | [`sweep/run_queue.py`](../../curvature-experiment/sweep/run_queue.py) on the pod | GPUs, embeddings, label tables |
| II-rank JSON | [`12_ii_rank_run.py`](../../curvature-experiment/runners/12_ii_rank_run.py) | sweep geometry npz, which `run_queue` deletes by default after `thin` and `robust` ([`run_queue.py:99-140`](../../curvature-experiment/sweep/run_queue.py#L99-L140)). Whether it still exists on the pod is unknown |

**Where things live.**
- Pod only: ViT-B geometry `/mnt/ssd-cluster/effdim/probe-facing-out/probe-facing/09_probe_facing_geometry_d{16,20}.npz`
  (sha 477886ad… for d16 in the RR environment row), cross-encoder geometry under `/mnt/ssd-cluster/effdim/xenc-out/`,
  galaxy label parquet `/mnt/ssd-cluster/effdim/labels_Smith42_galaxies_v2.0_test.parquet` (sha 60f2f82e…), sweep
  output `/mnt/ssd-cluster/EffDim/sweep-out/`. Note two different directory casings (`effdim` vs `EffDim`); [CLAUDE.md](../../CLAUDE.md)
  only names the second.
- Local only (git-ignored, `.gitignore` "Notebook analysis caches"): `notebooks/.cache/` with all 09 records, the
  published cf npz and thin npz, `scaling/` and `qm9/` records and arrays. No galaxy geometry npz locally (only a
  d = 4 smoke file in `probe-facing/`).
- Committed: `results/tensor-fidelity/records`, `results/review-robustness/records`, reports, `results/ii-rank` JSON
  and spectra, `results/qm9/sidecars`, SHA256SUMS files.
- No script builds the galaxy label parquet ([REPRODUCE.md](../../curvature-experiment/REPRODUCE.md) §2). No sha256 for the ViT-B geometry is in
  [`paper/latex/COMPLIANCE.md`](../../paper/latex/COMPLIANCE.md) ([REPRODUCE.md](../../curvature-experiment/REPRODUCE.md) §2).

---

## 5. Contested points

1. **Does `hess_mismatch_emp` contain the second fundamental form?** Sanity I6 and Outline C4 say no. [REPRODUCE.md](../../curvature-experiment/REPRODUCE.md)
   and the split runner docstring ([:14-16](../../curvature-experiment/REPRODUCE.md#L14-L16)) call it a data-side check of ⟨w, II⟩. Both are partly right. My check 1b
   shows probe_emp equals ⟨w_N, II⟩ exactly when the data lie on the manifold and the chart is correct, so it does
   estimate the data's II in the probe direction. My check 1c shows it is also exactly the quadratic part of y − w·x.
   **Judgement.** It is a decoder-free estimate of residual curvature. It does not use the decoder's II and agrees
   with it only at median cosine 0.55–0.67. The paper must not describe the tabulated mismatch as built from the
   decoder's K. The practical worry (it is mostly ‖Hess y‖ and nearly the outcome's own residual) stands.

2. **Why is t* > 1 at α = 100?** The paper caption, Sanity I2 and Novelty §6-7 say ridge shrinkage. Outline C1(e)
   says the published t* also contains the fidelity slope β(p|q), which can exceed 1 with no shrinkage since
   ‖p‖/‖q‖ ≈ 2-2.6. **Judgement.** The Outline's algebra is correct (re-derived above). The ridge argument is exact
   only pooled over all rows and for variant S. The stored arrays do not contain ⟨p, q⟩, so the two parts cannot be
   separated. That t*_model drops to 0.96–1.36 at α* (RR:17-36) is consistent with both. Unresolved until ⟨p,q⟩ is
   stored.

3. **Is "reversal hurts" evidence?** The paper presents it as half the asymmetry. Sanity I1 says it is automatic.
   **Judgement.** Sanity is right. hurt ⇔ t* > −½ is an identity, and help ⊂ hurt always. The informative contrast
   is help vs random help, and the Novelty simulation shows that contrast is expected from least squares for any
   decodable label. The surviving content is that the decoder's q correlates with the probe's own normal readout,
   which is an instrument check.

4. **Label-Hessian magnitude.** RR:228 and the rebuttal say "about 2x too large". Outline I6 says the sign is not
   established. **Judgement.** Outline is right. Remove the factor-of-two sentence or compute a signed norm ratio.

5. **Roughness "over-controls".** RR:40 and the rebuttal say so. Sanity I8 says the data contradict it.
   **Judgement.** Sanity is right (13/20 partials grow at α*).

6. **"Results get stronger with the tuned probe."** Rebuttal §2 vs Sanity C3. **Judgement.** True for galaxy
   permutation counts, false for QM9 (verified), and not like for like because α* changes w_N, sits at the grid edge
   in many pairs and is unvalidated by the fixture.

7. **Cause of QM9 counterfactual non-transfer.** Rebuttal: "size depends on the encoder". Sanity C3: surrogate
   fidelity. **Judgement.** Fidelity is the better-supported explanation. Data-side S helps at 0.88–1.00 on every QM9
   pair, S_model at 0.28–0.74, and per-anchor agreement is −0.30 to +0.23 (all verified).

8. **Mismatch partial vs ‖Hess y‖ alone.** Sanity C2 and Outline C4 agree with each other and with the records
   (36/40 within 0.05, verified). The paper and rebuttal do not report the comparison. Not contested on the facts.

9. **Held-out ΔR² range.** The TMLR outline draft said "+0.001 to +0.15" at α = 100. Outline I5 says −0.006 to
   +0.147. **Judgement.** Outline is right (RR:44-63, vit_large smooth_fraction −0.006).

10. **Minor factual slip in the Test-1 review.** It says the local machine has no `.cache`. The Cache exists (8.9 GB)
    with all records and cf arrays; what is missing locally is the geometry, embeddings and labels, as its own
    parenthesis says. This does not change any of its conclusions.

11. **Is the paper's cross-fitted "mismatch" the same statistic as the main-table mismatch?** Raised by the agent
    drafting [`paper/tmlr/main.tex`](../../paper/tmlr/main.tex); not flagged in Sanity or Outline. **Checked and confirmed.**
    - Code. The `--hessian-xfit` path ([`09_physics_probe_facing_split_run.py:407-442`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L407-L442)) refits only the label Hessian
      on each half ([:413-415](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L413-L415)), recomputes `split_columns` with the half Hessians but the full-panel probe Hessian
      ([:428-429](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L428-L429)), and scores exactly three columns, `("hess_mismatch_dec", "align_cos_tan", "hess_label")` ([:432](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L432)).
      `hess_mismatch_emp` is never cross-fitted.
    - Records. The published ViT-B record `notebooks/.cache/09_physics_probe_facing_split.jsonl` has no xfit rows at
      all; the ViT-B cross-fit lives in `09_physics_probe_facing_split_xfit.jsonl` (8 rows). The four cross-encoder
      split records and all 14 non-empty `scaling/records/*__main_xfit.jsonl` carry 4 xfit rows each. In all 15
      files every xfit row stores the same column set, {align_cos_tan, hess_label, hess_mismatch_dec}
      (`scaling__llava_15_13b__main_xfit.jsonl` is an environment-only stub with none).
    - Paper generator. tab:real and tab:xenc read `hess_mismatch_emp` ([`appendix_gen.py:19`](../../paper/generate/appendix_gen.py#L19), [`90`](../../paper/generate/appendix_gen.py#L90);
      [`table_main_gen.py:9`](../../paper/generate/table_main_gen.py#L9)). tab:sens and tab:xencx read `hess_mismatch_dec` under the heading "mismatch, cross-fit"
      ([`appendix_gen.py:62`](../../paper/generate/appendix_gen.py#L62), [`224`](../../paper/generate/appendix_gen.py#L224)), and the prose range "cross-fitted −0.28 to −0.62" is built from dec ([:100](../../paper/generate/appendix_gen.py#L100)).
    - Size of the gap. For ViT-B d = 16 mag_r, in-sample emp −0.387, in-sample dec −0.304, cross-fitted dec
      −0.381 / −0.336. So the dec statistic survives cross-fitting, and nothing tests whether emp does.
    **Conclusion.** The claim is correct. [main.tex:190](../../paper/latex/main.tex#L190) ("These relationships persist under cross-fitting") and
    [main.tex:424](../../paper/latex/main.tex#L424) compare a different statistic from the one in the main tables. Either cross-fit emp (needs a
    probe_emp fit per half at [:428-429](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L428-L429) plus `"hess_mismatch_emp"` in the [:432](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L432) tuple) or put the dec column in the
    main tables. This is defect D1.

12. **Does the counterfactual mean "the encoder bends toward physics"?** Abstract and [main.tex:287-292](../../paper/latex/main.tex#L287-L292) suggest it;
    Novelty §5-7, Outline C2 and Test-1 Scope say no. **Judgement.** No. The current design cannot separate geometry
    from least squares. The proposed Test-1 redesign (surrogate labels sharing the physical label's linear part) is
    the right kind of test and has not been run.

---

## 6. Open issues and recommended next steps (ranked)

1. **Fix the paper's labels before anyone quotes it** (hours). Say the tabulated mismatch is the residual-curvature
   column, or switch the main tables to `hess_mismatch_dec`, which does have a cross-fit. Fix D1-D5 and D8. Replace
   permutation stars by block-bootstrap intervals for the claims that matter.
2. **Report the two baselines the ML4PS reviewer asked for** (1-2 days, CPU, stored geometry and records). ‖Hess y‖
   alone, and a cross-fitted neighbourhood-residual baseline (half-A residual variance → half-B local R²), with
   block-bootstrap intervals. Claim explanation of local error, not prediction, unless mismatch beats the residual
   baseline.
3. **Cross-fit the emp column** (small code change in the xfit loop at [`09_physics_probe_facing_split_run.py:428-432`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L428-L432);
   needs a probe_emp fit per half). Also cross-fit the α* battery.
4. **Counterfactual decomposition and a geometry-isolating null** (1-2 days). Store ⟨p,q⟩, ‖p‖, ‖q‖ in
   `scaling_at_anchor`; report β(p|q) and ⟨r_c,q⟩/‖q‖² separately; add a mismatched-II null (II from another anchor
   or a Haar rotation of the tangent frame, keeping w_S). Present the existing result as an instrument check.
5. **Rebuttal corrections** (hours). Remove "2x too large", the label-noise reading, "over-controls", the general
   "stronger with tuning"; use bootstrap counts; report alignment's weaker counts; state that encoder counts share
   galaxies, labels and anchors.
6. **Centred variance explained** for every decoder (minutes on the pod).
7. **Instrument gaps.** A multi-direction in-sphere fixture with label noise and a signed norm ratio; a tilt-leak
   placebo on real data (labels v·x with v in the local PCA tangent) and J vs local-PCA principal angles; the
   Σ-metric (non-Gaussian patch) version of Eq. 3. A Swiss roll notebook is required by [CLAUDE.md](../../CLAUDE.md) for any new
   curvature measure.
8. **Before any II-spectrum claim.** Decoder-prior nulls, decoder-free local-polynomial II, seed and width subspace
   overlap (save singular vectors), d sweep. Use `--keep-geometry` or copy the sweep geometry off the pod first.
9. **Library fix** (issue only; [CLAUDE.md](../../CLAUDE.md) forbids editing `src/effdim` during v1.1). TLE and MiND-MLk are mislabelled.
10. **Provenance hygiene.** Commit the label-parquet build script, record geometry sha256 in [COMPLIANCE.md](../../paper/latex/COMPLIANCE.md), pin the
    revision in `PHYSICS_PARQUET_PATH`, add galaxy parquet sha256 to [`encoders.yaml`](../../curvature-experiment/encoders.yaml), and decide which pod directory
    (`effdim` or `EffDim`) is canonical.
11. **Unit tests** for `local_quadratics`, the emp identity, `scaling_at_anchor` and the xfit path. The checks in
    [`check_math.py`](2026-10-05-scripts/check_math.py) can be turned into tests directly.

---

## 7. Glossary

| term | meaning | code |
|---|---|---|
| x, X | unit-normalised embedding row, matrix (n × D) | [`09_physics_probe_facing_split_run.py:263-264`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L263-L264) |
| F, F̂ | decoder; sphere-projected decoder F/‖F‖ | [`09_physics_curvature_run.py:84-99`](../../curvature-experiment/runners/09_physics_curvature_run.py#L84-L99) |
| J, g, ginv | decoder Jacobian (D×d), metric JᵀJ, its inverse | [`09_physics_probe_facing_split_run.py:109-117`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L109-L117) |
| II, II_rad, II_tan / II^S | second fundamental form; its component along x̂ (≈ −g); the in-sphere remainder | :115, :174-175 |
| H, H_tan_norm | mean-curvature vector tr_g II; norm of its in-sphere part | :116, :187 |
| w, b0, w_T, w_N, w_rad, w_S | probe weight and intercept; tangent, normal, radial and in-sphere normal parts | :172-178; [`09_physics_normal_scaling_run.py:73-76`](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L73-L76) |
| K | probe-facing curvature ⟨w_N, II⟩ | |
| pf_full | ‖⟨w_N, II⟩‖_g (decoder) | :176, :188 |
| pf_tan | ‖⟨w_N, II_tan⟩‖_g, the "shape" term K_S | :177 |
| pf_rad | ‖(w_N·x̂) II_rad‖_g ≈ √d |w·x̂|, the "sphere" term | :178-179 |
| pf_trace_tan | w_N·H_tan | :189 |
| hess_y, hess_label | local quadratic label Hessian; its g-norm | :135-161, :190 |
| probe_emp | local quadratic Hessian of p = Xw on the data | :192 |
| hess_mismatch_dec | ‖hess_y − pf_full tensor‖_g (decoder) | :191 |
| hess_mismatch_emp | ‖hess_y − probe_emp‖_g = ‖Hess(y − w·x)‖ (the paper's "mismatch") | :192 |
| align_cos_full / align_cos_tan | g-cosine of hess_y with ⟨w_N,II⟩ / with ⟨w_N,II_tan⟩ (the paper's "alignment" is _tan) | :194-195 |
| cross_full / cross_tan | g-inner products behind those cosines | :185 |
| bias_sq | (y − ŷ_oof)² at the anchor | :329 |
| local R², r2 | 1 − SSR/SST of OOF predictions over the anchor's k neighbours | [`physics_curvature_probe.py:758-806`](../../curvature-experiment/pu_manifold/physics_curvature_probe.py#L758-L806) |
| sealed / multiscale controls, Z_multi | [log r_2048, label variance, count] / log r at 5 scales + the two | [`09_physics_probe_facing_split_run.py:326-327`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L326-L327) |
| Z_ext, roughness | Z_multi + hess_label + (1 − label linear R² in the chart) | [`11_review_robustness_run.py:111-115`](../../curvature-experiment/runners/11_review_robustness_run.py#L111-L115) |
| partial | rank-partial Spearman with Freedman–Lane p | [`cross_split_curvature.py:74`](../../curvature-experiment/pu_manifold/cross_split_curvature.py#L74) |
| xfit, fitA_scoreB | Hessian fitted on half A, local R² on half B | [`09_physics_probe_facing_split_run.py:407-442`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L407-L442) |
| u | tangent-projected coordinates g⁻¹Jᵀ(x − x₀) | :146 |
| p, q, q_c, r_c, e₀ | data-side in-sphere readout X w_S; decoder quadratic ½⟨w_S,II^S⟩(u,u); centred q; centred global residual; shape-flat residual | [`09_physics_normal_scaling_run.py:84`](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L84), [`98-99`](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L98-L99) |
| S, S_model, S_proj, full, full_model | counterfactual variants (table in 2.8) | :95-97 |
| random_wnorm, random_matched, random_qmatched | random in-sphere normal v matched on ‖v‖, on ‖⟨v,II^S⟩‖_g, or on ‖q_c‖ (published) | :77-92 |
| rand_ratio | ‖⟨r,II^S⟩‖_g / ‖⟨w_S,II^S⟩‖_g at equal norm | :89 |
| eq, qq, dR2, r2_curve | ⟨e₀,q_c⟩, ‖q_c‖², ΔR²(1), r2 at t ∈ T_GRID | :100-103 |
| t* | eq/qq, the SSE-minimising scale | :101 |
| help, hurt | fraction of anchors with r2(1) > r2(0); with r2(−1) < r2(0) | [`sweep/extract.py:66-68`](../../curvature-experiment/sweep/extract.py#L66-L68) |
| d_r2_plus / d_r2_minus | median ΔR² at t = +1 / −1 | :68 |
| dec_cross_tan, dec_t_star | ⟨hess_y, K_S⟩_g; that over ‖K_S‖² (decoder-predicted t*) | [`09_physics_normal_scaling_run.py:237-238`](../../curvature-experiment/runners/09_physics_normal_scaling_run.py#L237-L238) |
| thinned, keep | anchors with pairwise overlap ≤ 5% (sign tests) or ≤ 10% (RR partials) | [`extract.py:75-80`](../../curvature-experiment/sweep/extract.py#L75-L80); `11_…:158-162` |
| blocks | average-linkage clusters of anchors on 1 − overlap (16/32/64) | `11_…:122-127` |
| α*, fold alphas | GCV-selected ridge on all rows; per-fold selections for OOF | `11_…:65-87` |
| guard exact / refit | reproduction tolerance 1e-6 / 0.02 on partials, 1e-12 on cf | `11_…:192,258` |
| main_xfit, seed1, seed2, main_d16, cf, thin, robust | sweep job suffixes | [`sweep/jobs.py:102-113`](../../curvature-experiment/sweep/jobs.py#L102-L113) |
| d_ID, d_run | rounded median of four ID estimates; min(d_ID, 20) | [`sweep/intrinsic_dim.py:46-48`](../../curvature-experiment/sweep/intrinsic_dim.py#L46-L48) |
| erank, pr, k90, k99, cond | entropy effective rank, participation ratio, directions for 90/99% of squared spectrum, s₁/s_m | [`12_ii_rank_run.py:63-72`](../../curvature-experiment/runners/12_ii_rank_run.py#L63-L72) |
| m | d(d+1)/2 = 136 at d = 16 | |
| relerr, rho_mismatch, rho_align | TF unsigned relative error; Spearman of dec mismatch; Spearman of full alignment | [`10_tensor_fidelity_run.py:82-90`](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L82-L90), [`249-250`](../../curvature-experiment/runners/10_tensor_fidelity_run.py#L249-L250) |
| var_explained | 1 − MSE/mean‖x‖² on holdout (uncentred) | [`09_physics_probe_facing_run.py:119`](../../curvature-experiment/runners/09_physics_probe_facing_run.py#L119) |
| u_scale | RMS g-norm of the neighbours' chart coordinates | [`09_physics_probe_facing_split_run.py:147`](../../curvature-experiment/runners/09_physics_probe_facing_split_run.py#L147) |

---

## 8. Questions teammates are likely to ask

**Is there a bug that changes a number in the paper?** No numerical bug in the experiment code. One hand-typed
range is wrong (α = 1 OOF R² gain is +0.14 to +0.16, not 0.10 to 0.15). The bigger problems are that the paper
describes different columns and variants than it tabulates (D1-D4).

**Is the mismatch result real?** The association between the tabulated mismatch and local R² for magnitude and
redshift is real and survives the block bootstrap on ten encoders. What it measures is the curvature of the probe's
residual, which in rank is almost the label's own curvature. It is not shown to be about the decoder's probe-facing
bending.

**What does cross-fitting show, then?** That the decoder-based mismatch (`hess_mismatch_dec`) does not depend on the
Hessian and the outcome sharing rows. The column in the main tables was never cross-fitted.

**Does the counterfactual show the encoder bends toward physics?** No. "Hurt" follows from "help" by algebra, and
"help above random" is what least squares gives for any decodable label. The decoder surrogate carries under half of
the probe's own in-sphere readout. Treat it as a check that the decoder's second-order term correlates with the
probe's normal readout.

**Why is t* about 2 to 3?** Some of it is ridge shrinkage. Some may be that the decoder quadratic is smaller than the
probe's actual normal readout. We cannot split the two without storing one more inner product.

**Is the decoder instrument trustworthy?** For the trace and for the full shape tensor in a fixture with one normal
direction, yes (cos ≥ 0.96 at paper scale without noise). With realistic noise the in-sphere cosine drops to
0.68–0.84. Multi-direction selection and the II spectrum are not validated. The "variance explained" numbers are
uncentred and likely flatter the fit.

**How good is the label Hessian?** Its norm ranks well (split-half ρ 0.91–0.94). Its direction is poor on real labels
(split-half cosine 0.19–0.35). Its magnitude is off by 29–118% on the fixture, direction of the error unknown. It
also picks up a gradient-proportional, II-shaped bias if the decoder tangent is tilted.

**Can I trust the p-values?** Not the permutation ones. Anchors overlap heavily. Use the block-bootstrap counts, and
read them as robustness, since blocks still share points.

**Do ten encoders mean ten replications?** No. Same galaxies, same labels, same anchor indices. They show robustness
to the choice of embedding.

**Did QM9 replicate?** The mismatch did for gap, polarisability and heat capacity (8/8 under bootstrap), partly for
dipole moment (5/8, with a significant reversal on ChemFM-3B). The counterfactual did not, because the decoder
surrogate does not track the probe on molecules.

**Is the II spectrum result solid?** The numbers are computed correctly. The claim that the manifold bends in a few
directions is not supported: 8 of 10 encoders are "partial" by the pre-stated rule, and nothing yet separates the
decoder's own smoothness prior from the representation.

**Can I regenerate everything from the repo?** The reports and paper tables, yes, from the local git-ignored Cache.
The underlying records need pod-only geometry, the pod label parquet and pinned embeddings. The tensor-fidelity
experiment is fully synthetic and regenerable from code.

**Were the reproduction guards meaningful?** They prove the code is deterministic and that today's code reproduces
the published records from the same inputs to 0 or 1e-15. They are not independent replications.

**What should we do first?** Fix the labels in the paper (Section 6, item 1), then run the two baselines and the emp
cross-fit (items 2 and 3). Those three decide what the paper can honestly claim.
