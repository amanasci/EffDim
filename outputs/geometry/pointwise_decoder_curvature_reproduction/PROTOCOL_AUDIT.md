# PROTOCOL_AUDIT — pointwise decoder-curvature reproduction

Read-only recovery. Colleague whose results are reproduced: **Austin Lutterbach**
(`rmhslutterbach@gmail.com`), author of `notebooks/pu_manifold/decoder_curvature.py`
and of the d=16 cubic/ridge table on `origin/fixture-validity-audit`.

The current worktree is on `curvature-experiments` at `97efb2eb6cd7dec7f2c568f53c534752ff3c32c8`.
That SHA equals `refs/heads/curvature-experiments` and
`refs/remotes/origin/curvature-experiments`. It was inspected in place (already
checked out). The d=16 table and `varying_ii_controls.py` live on
`origin/fixture-validity-audit` at `dcd2208803f27224ee182cce47fdee21c1bc6ba5` and are **not**
ancestors of HEAD. They were read with `git show` / vendored; that branch was
not checked out, merged, or rebased.

Angus's nested-chart `K_H^cross` / local-quadratic estimator on this branch is a
**different** instrument and was not run.

## Named sources

| Name | Where recovered | Role |
| --- | --- | --- |
| 02.6-FINDINGS | `.planning/phases/02.6-decoder-substrate-screening/02.6-FINDINGS.md` on HEAD | Early Swiss-roll decoder screen (150 epochs). **Not** the R1 numbers. |
| Supp. 02 brief | `09-SUPPLEMENT-02-INSTRUMENT-ADJUDICATION.md` on fixture-validity-audit | Quotes R1 0.553 / 0.9998 / 1.046 as `ours H_tan`. |
| 09-FIXTURE-FIDELITY-D16 | same branch, `09-FIXTURE-FIDELITY-D16.md` | Exact R2–R5 plain-decoder cells. |
| 07_plain_decoder_sweep | `notebooks/diagnostics/07_instrument_fixture_sweep_run.py` | Implementation of R2–R5. Default `--d 20`; d=16 is `--d 16`. |
| 07_low_spread_control | `notebooks/diagnostics/07_instrument_low_spread_control_run.py` | Control arm; not a primary cell. |
| Supp. 03 | `03-08-SUPPLEMENT-03.md` | Present; not the cubic/ridge table. |

Hardcoded path in the 07 runner (`/home/akagi/Documents/Projects/EffDim/notebooks`)
is Austin's machine. Compatibility fix: this package inserts **this** repo's
`notebooks/` on `sys.path`. No other colleague code was edited.

## Fixture generators

**R1 Swiss roll.** `sklearn.datasets.make_swiss_roll(n_samples=3000, noise=0.0,
random_state=0)`. Centre by the coordinate-wise mean, divide by one global
standard deviation (a single scalar). Latent dimension 2.

**R2–R4 (optional R5).** `varying_ii_controls.FAMILIES["cubic"|"ridge"](5000, 16,
D, 20260816)`. Parameter samples are uniform in a box (cubic `domain_radius=1.5`).
The graph `(x, f(x))` is assembled, then `synthetic_controls.rotate_and_pad`
applies a seeded rotation into ambient `D` and a single global std. Analytic
`H_vec` is `curvature_probe.graph_mean_curvature` (same unnormalized trace)
rotated with the cloud.

Sampling distribution for cubic/ridge is the fixture generator's uniform
parameter box, not a Gaussian on the sphere.

## Train / evaluation split

- R1: `split_indices(n, seed=20260813, holdout_fraction=0.2)` — permutation,
  first `round(n*0.2)` holdout, remainder train. Train the AE on the train
  split only. Evaluate curvature at 256 holdout anchors drawn by
  `subsample.draw_row_indices(len(holdout), 256, seed=20260902)` then sorted.
- R2–R4: **no split**. Train and evaluate on all 5000 points.

## Normalization

- Input: as above (global std). No per-feature standardization after that.
- Decoder output: **raw** `model.decode`. No sphere projection on R1–R4.
  `SphereProjectedDecoder` is the Amendment-01 Physics path and is not used
  by `07_instrument_fixture_sweep_run` or the swiss-roll ambient arm.

## Architecture, loss, optimizer

`cae.PlainAutoEncoder(in_dim=D, latent_dim=d, hidden=(250,250,250),
activation="silu")`. SiLU is C², required by `assert_c2_decoder`.

Reconstruction loss inside `cae._train_decoder_protocol`: batch mean of the
per-row sum of squares `||x-y||^2`. Optimizer AdamW, `lr=1e-3`,
`weight_decay=1e-4`, `batch=128`. Early stopping disabled
(`early_stop_patience > max_epochs`, `min_delta=1e-9`).

- R1: 300 epochs. `torch.manual_seed(0)` at construction. Train cfg seed
  defaults to 0 (pcp.TRAIN_CFG has no `seed` key).
- R2–R4: 400 epochs. `torch.manual_seed(0)` at construction, then
  `cfg["seed"]=20260816` reseeds the training shuffle.

Checkpoint: last in-memory state. Historical runners wrote JSONL metrics,
not `.pt` weights. No frozen-weight selection rule.

## Autodiff conventions

`plain_decoder_curvature` requires float64. Derivatives use
`torch.func.vmap(jacrev)` and `vmap(hessian)` in chunks of 32.

Jacobian convention: `J.shape = (D, d)` with `J[:, a] = ∂_a F`.
Hessian convention: `Q.shape = (D, d, d)` with `Q[:, a, b] = ∂_a ∂_b F`.
First fundamental form: `g = J^T J`. Inverse via `torch.linalg.solve`.

The map differentiated is **raw `model.decode`**, never `forward` (which would
compose the encoder) and never a sphere-normalized wrap on these cells.

## Second fundamental form and mean curvature

Documented and implemented as:

```
J    = DF
g    = J^T J
P_N  = I - J g^{-1} J^T
II   = P_N D²F
H    = tr_g(II) = g^{jk} II_jk
```

Implementation is trace-first-then-project (`raw = g^{jk} Hess_jk`, then
subtract the tangent component). Algebraically identical to projecting each
Hessian slice and then tracing.

**Is reported H equal to `g^{ab} II_ab` or `(1/d) g^{ab} II_ab`?**
`g^{ab} II_ab`. `CURVATURE_CONVENTION = "trace"`. A unit d-sphere has
`||H|| = d`, not 1.

**Is II equal to `(I-P_T) D²F` or `(I-xx^T-P_T) D²F`?**
`(I-P_T) D²F`. The normal projector removes the tangent space only.

**Is F differentiated before or after output normalization?**
Before, on R1–R4.

## Metric definitions (traced, not inferred from column names)

- **variance explained** = `1 - mse_total / mean(||X||^2)` where `mse_total`
  is the mean over rows of `||x-y||^2` from `cae.reconstruction_stats`.
  This is **not** centered variance R².
- **ρ** = Spearman correlation of `||H_est||` with `||H_true||`
  (`scipy.stats.spearmanr(...).statistic`).
- **cosine** = median over evaluation points of the pointwise cosine of the
  ambient vectors `H_est` and `H_true`.
- **ratio** = median of `||H_est|| / ||H_true||`.
- **true ||H|| spread** is not a 07 output column. The fixture diagnostic
  `ii_variation.hess_fro_cv` is the CV of `||D²f||_F` across points.

R1 Supp. 02 labels the row `ours H_tan`. The swiss-roll runner stores
`instrument: ours_H_ambient` and scores `H_vec` against
`swiss_roll_analytic_H_vector`. The label in the brief is a naming
discrepancy; this reproduction preserves the **runner** definition.

## Analytic truth

- Swiss roll: plane-curve curvature vector of the Archimedean spiral in the
  x–z plane, y-component zero, scaled by `global_std`.
- Cubic/ridge: `graph_mean_curvature` on the closed-form grad/hess, then the
  same `rotate_and_pad` as the cloud.

## Aggregation

Single frozen seed per cell. No multi-seed mean. Medians over the evaluation
set (256 anchors for R1; 5000 points for R2–R4).

## Cached sphere fixtures

Searched this worktree for `*86471*.pt` and `notebooks/.cache/**/*.pt`.
None found. Missing sphere checkpoints are not a failure of the four-cell
reproduction. Label: `cached_sphere_evaluation_skipped_missing_weights`.

## Blockers

Estimator, truth, and metric code were recovered. This audit does **not**
assign `reproduction_blocked_by_missing_definition_or_code`.
