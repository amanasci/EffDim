# REPORT — known-curvature instrument failure localization

## Decision

**quadratic_tangent_estimation_failure**

Q1 succeeds on the decisive F4 rank check; replacing the exact tangent frame
with local PCA is the principal loss. Ridge and split-half increments stay
below the frozen materiality thresholds. Decoder true-projector / finite-
difference stages could not run (no cached `.pt` weights), so D’s false
curvature is **not** classified as learned-surface Hessian non-identifiability.

This is a 64-anchor cached diagnostic, not another 62-cell benchmark.

## Runtime

- wall-clock: **1268 s (21.1 min)**, under the 45-minute cap
- unit tests: **10/10**
- `COMPLETE.json` written; all required bounded stages finished
- skipped only: `D1_true_projector` and `D2_finite_difference` (no decoder weights)
- 3200 Q rows = 6 cells × {512,2048} on primary / 2048 on stress × 5 stages × 64 anchors

## Phase 0 parity (64-anchor subset)

Artifacts aligned by `sample_id`. No `BLOCKER.md`.

| Check | Result |
| --- | --- |
| F0 production Q residual \|K_dir\| | 4.0×10⁻¹⁵ (≈0) |
| F4 decoder ρ(H) full / 64-subset | 0.949 / 0.957 |
| F4 decoder ρ(K_dir) full / 64-subset | 0.944 / 0.964 |
| F4 production Q vs T2 (64) | ρ=0.17 (weak) |
| C_F4 η=0.10 T2/T3 | unavailable (marked, no new oracle) |

## Does Q1 recover T2/T3?

**On varying F4, yes for rank; only moderately for magnitude.**

At S0/N0, k=2048, 64 anchors:

- Q1 ρ(K_dir, T2)=**0.932**, ρ(T1)=0.783, calibration slope 0.69, R²=0.71
- Q1 median rel(K_dir) vs T2 = 0.31 (above the 0.25 magnitude bar, but rank ≥ 0.70)
- F0 residual K_dir ≈ 10⁻³² (no false curvature)
- F1 false TF fraction 0.042 (below 0.10)
- F0–F2 T2 tensors equal T1; F4 T2 tensors were not stored (scalar T2/T3 only)

Q1 still has non-trivial magnitude error on the constant fixtures (F1 rel 0.45,
F2 rel 0.54): the exact-frame residual quadratic underestimates those
pointwise norms at production radius. That is **not** the production-Q
failure. Production Q’s F4 rank collapse appears only after the PCA frame.

Design at k=2048: n=2048, q=136, rank=136, cond≈9.3 (exact frame) / 5.7 (PCA).
Adequate. Ridge shrinkage Q2 edf/q≈0.86.

## Incremental error Q1→Q5 (S0/N0, k=2048)

| Step | Isolates | Δ median rel(K_dir) vs T2 | Δ ρ(K_dir, T2) on F4 |
| --- | --- | --- | --- |
| Q1→Q2 | production ridge | +0.025 | −0.036 |
| Q1→Q3 | estimated PCA frame | **+0.321** | **−0.621** (0.93→0.31) |
| Q3→Q4 | ridge in PCA frame | +0.002 | −0.046 |
| Q4→Q5 | A/B split + KHcross | +0.096 | −0.093 |

Per-fixture smoking guns at Q3:

- **F1:** exact-frame K_dir≈0.21 collapses to 0.003; TF fraction jumps 0.04→0.88
- **F2:** K_dir jumps 0.033→0.61 (false TF energy); max principal angle 0.32 rad
- **F4:** rank recovery dies (0.93→0.31); Procrustes alignment only cuts tensor
  rel-B 1.73→1.05, so this is not a coordinate-packing artefact
- **F0:** stays residual-flat through Q5 (10⁻¹⁴)

Q5 matches the frozen production estimator. Once the frame is estimated, ridge
and splitting add little.

## Why reducing k failed

This is a bias–variance probe, **not** an asymptotic test.

- expected r_512/r_2048 = (512/2048)^{1/16} = **0.917**
- observed median physical-radius ratio = **0.887**
- radius shrinks ~11%, not enough to remove finite-patch bias
- each production half has 1024 obs at k=2048 vs **256** at k=512, against q=136

On F4, exact-frame Q1 actually *improves* slightly at k=512 (rel T2 0.16 vs
0.31; ρ(T2) stays 0.93). Production Q5 remains weak at both radii
(ρ(T2)=0.17 at 2048, 0.30 at 512). Shrinking k cannot fix a tangent-frame
failure, and it spends most of the sample on variance.

## Density and noise (F4 only)

Independent density gradient, no noise (vs T2 and T3 scalars):

- Q1 ρ(T2)=0.933 (unchanged vs uniform 0.932); rel T2 0.37 vs 0.31
- deterioration is small on both T1 and T2 → not a pure sampling-weighted
  estimand drift; the PCA cliff is unchanged (Q3 ρ(T2)=0.44)

Uniform sampling + η=0.10 sphere-normal noise (T2/T3 **unavailable**):

- Q1 ρ(T1) 0.78→0.59 (exact frame, noisy residuals; still ranked)
- Q3 ρ(T1) 0.35→0.04 (PCA + noise destroys rank)

Noise acts primarily through tangent estimation, then residual fitting.

## Decoder (cached seeds only)

No `.pt` files, so D1 (true projector) and D2 (autodiff vs FD on the decoder)
are unavailable. Generator-path autodiff vs FD passed in unit tests
(rel 4×10⁻⁸); that does **not** certify the learned decoder Hessian.

D0 on the same 64 anchors:

| Fixture | recon R² range | seed Spearman K_dir | D vs T1 |
| --- | --- | --- | --- |
| F0 | 0.936–0.999 | **0.097** | false K_dir≈8.0 (truth 0) |
| F1 | 0.939–0.940 | 0.760 | mean collapsed (H_D≈0.002 vs 0.75) |
| F2 | 0.881–0.882 | 0.636 | false mean H_D≈0.71 (truth 0) |
| F4 | 0.999–0.999 | 0.955 | ρ(H)=0.957, ρ(K_dir)=0.964; cal tracks T1 |

F4 rank recovery coexists with F0/F2 hallucination and F0 seed instability.
Whether that false curvature would survive the true projector **cannot be
tested** without weights. The mechanical label is therefore not
`decoder_learned_surface_hessian_nonidentifiability`.

## Dominant source per instrument

- **Q (production):** tangent-frame estimation (Q1→Q3). Convention/oracle
  mismatch is not the production failure: Q1 recovers F4 T2 rank with full
  rank-136 designs. Ridge and splitting are secondary. Finite-patch T2 vs T1
  on F4 is only rel 0.075, so this is not `quadratic_patch_model_bias`.
- **D:** unresolved at the Hessian/projector split. D0 false curvature on F0
  and F2 is real in the cached scalars; D1/D2 must wait for stored weights.

## Output paths

Host (canonical):

`/home/angus/platonic-universe/outputs/geometry/known_curvature_instrument_failure_localization/`

Worktree copy:

`outputs/geometry/known_curvature_instrument_failure_localization/`

Required files: `summary.json`, `decision.json`, `parity.json`,
`reuse_manifest.json`, `q_ablation_table.csv`, `decoder_ablation_table.csv`,
`design_conditioning.csv`, `tangent_alignment.csv`,
`density_noise_diagnostics.csv`, `METHODS.md`, `REPORT.md`, `COMPLETE.json`.
Optional figure: `figures/q_ablation.png`.
