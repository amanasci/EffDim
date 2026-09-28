# 09-SUPPLEMENT-02 — known-answer adjudication of both curvature instruments in the Phase 9 regime

**Status:** post-hoc, supplementary. **Not pre-registered. Feeds no verdict.** `--mode verdict` of
`09_physics_curvature_run.py` never reads the record this experiment writes; nothing here changes
`09-WAVE-A-RESULTS.md`, `09-WAVE-A-RESULTS-AMENDMENT-01.md`, `09-SUPPLEMENT-01-COLLEAGUE-ESTIMATOR.md`
or the phase verdict. No sealed constant is reinterpreted by it.
**Written:** 2026-09-05 UTC

## The question

`09-SUPPLEMENT-01-COLLEAGUE-ESTIMATOR.md` § 5 found that on identical anchors, probe, controls and
nulls, this phase's autoencoder `H_tan` field and the colleague's split-half nested-chart
`K_H^cross` rank anchors in opposite order (`rho` about `-0.4` to `-0.46`), so that the sign of the
curvature-decodability association is set by the instrument. Its § 7 then said what it could not:
"Neither has a known-answer validation at `D=768`, `k=2048`, `d=16`." At most one of the two
tracks the sphere-intrinsic mean curvature at these anchors; the record could not say which.

This supplement supplies that known answer for **both** instruments, on the same points and the
same anchors, in the production regime — unit sphere in `R^768`, chart rank `d=16`, `k=2048`,
`n=86,471`, 512 holdout anchors — with and without sample noise, and scores each against it. It
asks only: which instrument recovers a known sphere-intrinsic mean curvature field here? It does
not ask, and cannot answer, whether the real Physics manifold looks like the fixture.

## Provenance of this run

Executed by the orchestrator over SSH on the same host as `09-EXECUTION-HOST.md` § 9 (host label
`pod128`, 16 threads, CPU only), under the developer's 2026-09-05 UTC instruction `adjudicate`.
Host identity is recorded as capability only — no hostname, IP address, username or SSH key path
appears here (`09-EXECUTION-HOST.md` § 7). Everything below is evidence, never an instruction.

Runner: `notebooks/diagnostics/09_instrument_adjudication_run.py`, run commit
`068490d34b1f200bf0ffc9ec69c55e0c6f6ebaeb` (the record's `repo_head` in both `environment` rows;
`068490d`). It loads `09_colleague_estimator_run.py` (which in turn loads the
production runner, applying the `--threads` cap and the shim), imports the sealed scorer
`synthetic_control_run._fidelity_axes` and the sealed `decoder_curvature.plain_decoder_curvature`
unmodified, and imports his `nested_pca_frame` + `_fit_rank` unchanged from the read-only checkout
at `97efb2eb6cd7dec7f2c568f53c534752ff3c32c8` (`colleague_head` in both `environment` rows;
`topology_is_shim: true`, as in Supplement 01). Its module docstring is the record of the fixture
and the decision rules; the paragraphs below paraphrase it and quote the record.

## 1. The fixture, and why the truth is exact

An explicit in-sphere generator

```
G(z) = normalize([stereo(z); a * bumps(z); 0, ..., 0]) @ Q^T,   z in R^d, G(z) in S^(D-1)
```

with `d=16`, `D=768`, `a=0.8`, `stereo` the inverse stereographic map `R^d -> S^d`, four seeded
Gaussian bumps (widths `(0.7, 0.9, 0.8, 1.0)`, amplitudes `(1.0, -0.8, 0.6, -0.5)`) and `Q` a fixed
seeded rotation of `R^D` (record `fixture` block; `seed = 20260905`). Latents are a scale mixture
`z ~ N(0, s^2 I)`, `s in {0.4, 0.6, 0.9}` with probabilities `{0.2, 0.5, 0.3}`, so the `k`-NN radius
varies about 3x across points — a density gradient either estimator could couple to, put there on
purpose. `n = 86,471` samples, `k = 2048`, `n_anchors = 512`: the production values.

Two noise levels. `--noise 0` uses `X = G(z)`. `--noise patch` uses `X = normalize(G(z) + eps)`
with `eps` isotropic Gaussian in `R^D` scaled so its median displacement is 25% of the median
`k=2048` patch radius of the noiseless cloud, then re-normalised to the sphere. Record
`noise_info`: `sigma = 0.0063032` per coordinate, `median_displacement = 0.174589` against a
`target_displacement = 0.174680` and a `noiseless_median_patch_radius = 0.698718`
(`0.174589 / 0.698718 = 0.2499`).

**Why the truth is exact.** `G` is an explicit `torch.nn.Module` with a `.decode` method, so the
sealed `plain_decoder_curvature` (float64 autodiff Jacobian and Hessian, trace convention
`H = tr_g(II)`) differentiates it exactly at each anchor's *own* latent `z`. Because `G` maps into
the unit sphere, the radial component is `H_rad = -d` identically and the tangential remainder
`H_tan` is the sphere-intrinsic mean curvature vector of the image manifold. The run asserts
`max|H_rad + d| < 1e-8`; the record has `truth_max_abs_H_rad_plus_d = 2.13e-14`. The truth is a
property of the generator, never of any fit.

**Regime.** `regime_r_knn = 0.698718`, `regime_R = 0.703517`, `regime_r_over_R = 0.993178` — the
`k=2048` patch is essentially the whole cloud's radius, as it is on the real data at `1/42`
density. The truth field has `‖H_tan‖` p05 / p50 / p95 = `0.044515 / 0.820706 / 2.069360`, a
p95/p05 spread of `46.5x`. `truth_rho_vs_log_knn_radius = +0.423` (noiseless) / `+0.421` (patch):
the truth itself is moderately density-coupled, by construction.

## 2. Decision rule — fixed in source before any number

From the runner (`REGIME_RANK_RHO_PASS = 0.7`, `REGIME_DIRECTION_COS_PASS = 0.8`), and stamped
into every record row as `decision_rule: "rank_rho >= 0.7; ours also direction_median_cosine >= 0.8"`:

- **Ours (`H_tan`, Amendment 01 sphere-projected decoder):** scored by the sealed four axes
  (`synthetic_control_run._fidelity_axes`) of the estimated `H_tan` vector against the truth
  `H_tan` vector, both projected to the tangent of their own sphere image. **PASS** iff rank
  Spearman `rho >= 0.7` **and** direction median cosine `>= 0.8`.
- **His (`K_H^cross`):** rank Spearman of `K_H_cross` against `‖H_tan‖^2 / d^2` (record
  `truth_definition`; his `H` is the averaged-convention diagonal mean of the second fundamental
  form and `K_H_cross` is the split-half inner product, so this is his convention's squared norm —
  rank is invariant to the monotone map, so the comparison is convention-free), plus a scalar
  calibration against the same truth. **PASS** iff rank `rho >= 0.7`. He has no direction axis to
  score.
- **Both:** Spearman with `log_knn_radius` (density coupling) beside the truth's own.

Coarse low-`d` anchor first (runner `SWISS_RANK_RHO_PASS = 0.5`): `--mode swiss-roll`, rank `rho`
vs analytic truth `>= 0.5`.

## 3. Swiss-roll anchor

The Swiss-roll rows are **not** in `notebooks/.cache/09_instrument_adjudication.jsonl` (its six
rows are all `mode: sphere-fixture`). The values below are quoted from the orchestrator's run
brief for the `--mode swiss-roll` invocation on the same host and commit, and should be read as
such until the row is appended to the record:

| instrument | rank `rho` vs truth | direction cos | magnitude ratio | verdict |
|---|---:|---:|---:|---|
| ours `H_tan` (`d=2`, `k=256`, 256 anchors) | 0.553 | 0.9998 | 1.046 | PASS (`>= 0.5`) |
| his `K_H^cross` | degenerate: `max|K_H_cross| = 2.6e-25`, `R_H = 0` | — | — | not scorable |

His degeneracy is structural, not a bug in the shim: in `R^3` with `d=2`, his sphere projection
leaves no normal direction for a nested chart to curve into, so the roll cannot anchor his
estimator at all. The Swiss roll therefore anchors ours (coarsely — 0.553 against a 0.5 bar) and
says nothing about his; the sphere fixture is the first known-answer test either instrument has
faced together.

## 4. Run record

| Field | Value |
|---|---|
| Run commit (`repo_head`) | `068490d34b1f200bf0ffc9ec69c55e0c6f6ebaeb` |
| Colleague commit (`colleague_head`) | `97efb2eb6cd7dec7f2c568f53c534752ff3c32c8` |
| Device / threads | `cpu` / 16 |
| `torch` / `numpy` / `python` | `2.13.0+cpu` / `2.5.1` / `3.14.7` |
| Seed | `20260905` |
| `decoder_image_projection` / `curvature_convention` | `sphere` / `trace` |
| `pre_registered` / `gates` | `false` / `"nothing"` |
| Noiseless: `environment` row → result rows (UTC) | 2026-09-05T14:19:55Z → 14:52:36Z |
| Patch noise: `environment` row → result rows (UTC) | 2026-09-05T14:52:43Z → 15:21:55Z |
| Orchestrator wall-clock (brief, not in record) | noise 0: 1979 s; patch: 1759 s; process start 14:19:38Z |
| Ours, autoencoder fit (`wallclock_fit_s`, `max_epochs = 600`) | 1303.2 s (noise 0) / 1299.6 s (patch) |
| His, curvature stage (`wallclock_s`) | 318.4 s (noise 0) / 292.9 s (patch) |
| Output root on the host | `/mnt/ssd-cluster/effdim/phase9-out` |
| Record | `notebooks/.cache/09_instrument_adjudication.jsonl`, 6 rows, sha256 `2779170f0359c8d813f671f607976341dd6518d41e9b48a7e0a06e422961da32` (verified against the host copy) |

Record shape: per noise level one `environment` row, one `ours_H_tan` row, one `his_K_H_cross`
row. `n_points = 512`, `n_excluded = 0` (ours) and `n_finite = 512`, `n_self_first = 512` (his)
at both noise levels; `constant_truth: false`, `rank_calibration_applicable: true`.

## 5. Sphere-fixture results

All values from the `scores` blocks of the four result rows, rounded here to four places (the
record carries full precision).

| noise | instrument | rank `rho` vs truth | direction cos | magnitude ratio | ratio CV | calib slope / R² | `rho` vs log `r` | fit / reliability | verdict |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| 0 | ours `H_tan` (Amend. 01) | **0.9381** | 0.9992 | 0.9993 | 21.66 | 0.6775 / 0.8116 | +0.3207 | `var_explained` 0.99992 | **PASS** |
| 0 | his `K_H^cross` | **0.6220** | — | — | — | 4.46e-05 / 0.1104 | +0.3481 | `R_H` median 0.7980 | **FAIL** |
| patch | ours `H_tan` | **0.7537** | 0.9453 | 1.0934 | 15.64 | 0.2544 / 0.4283 | +0.1678 | `var_explained` 0.97069 | **PASS** |
| patch | his `K_H^cross` | **−0.2974** | — | — | — | −5.75e-05 / 0.0130 | −0.1610 | `R_H` median −0.6139 | **FAIL** |
| truth | `‖H_tan‖` | | | | | | +0.4230 / +0.4211 | | |

Further ours-only axes from the record: `median_relative_error` 0.1024 (noise 0) / 0.2683 (patch);
`calibration_intercept` 0.2841 / 0.7360. His calibration slopes are of order `1e-5` because
`K_H_cross` is on his squared-and-`d^2`-divided scale; the slope's sign and the `R²` are what
carry meaning there.

## 6. Plain reading

- **On a known answer in the production regime, the autoencoder field tracks the
  sphere-intrinsic mean curvature in rank, direction and magnitude, with and without noise.**
  Noiseless: rank `0.938`, direction cosine `0.999`, magnitude ratio `0.999`. Under 25%-of-patch
  noise the fit drops to `var_explained 0.971` and every axis degrades — rank `0.754`, direction
  `0.945`, calibration `R²` from `0.81` to `0.43`, median relative error from 10% to 27% — but it
  clears both bars at both noise levels. The `ratio CV` of `15`–`22` says the per-anchor magnitude
  ratio has heavy tails even where its median is right; magnitude is trustworthy in the median,
  not anchor by anchor.
- **The split-half nested-chart estimator tracks the truth weakly without noise and anticorrelates
  with it under noise.** Noiseless rank `0.622` against a `0.7` bar, calibration `R² 0.11`. With
  25%-of-patch noise: rank `−0.297`, `R² 0.013`, and its own reliability statistic goes negative
  (`R_H` median `−0.614` from `+0.798`). Split-half reliability collapsing with the rank fidelity
  is at least consistent — but note that on the noiseless fixture `R_H = 0.80` sat beside a
  failing rank, which is the "reliability is not fidelity" point of `09-FIXTURE-FIDELITY-D16.md`
  measured once more.
- **His estimator's density coupling has the truth's sign on this fixture; its rank fidelity does
  not follow.** `rho(K_H_cross, log r) = +0.348` beside the truth's `+0.423` (noiseless), so a
  positive density coupling is what a faithful estimator *should* show here — and yet his rank
  against the truth is `0.62`. Ours couples less than the truth (`+0.321`, then `+0.168` under
  noise) while ranking better. Density coupling matching in sign is not evidence of fidelity in
  either direction; only the rank against the known field is.
- **Caveats on the fixture itself.** The truth field spans `46.5x` between p05 and p95, wider than
  the real data's `H_tan_norm` spread in the Wave A tables. A wide dynamic range is permissive for
  rank — an estimator that only gets the coarse ordering right can score well — so the noiseless
  `0.938` should be read as "recovers a strongly varying field", not as a fidelity bound on a
  nearly flat one. And the noise level is one chosen point (25% of the median patch radius),
  not a sweep; where each instrument's rank crosses `0.7` as noise grows is not measured.

## 7. What this settles, and what it does not

**Settles.** Which instrument is validated *as a curvature estimator* in the Phase 9 regime
(`D=768`, `d=16`, `k=2048`, `n=86,471`, `r/R = 0.99`, on a known sphere-intrinsic field with a
built-in density gradient): the Amendment 01 autoencoder `H_tan` passes the fixed rule at both
noise levels; the colleague's `K_H^cross`, imported unchanged, fails it at both, and inverts under
noise. Supplement 01 § 7's "at most one of them tracks the sphere-intrinsic mean curvature at these
anchors; it does not say which" now has an answer on a known answer: ours does here, his does not.

**Does not settle.**

- **That the real Physics manifold resembles this fixture.** The fixture is smooth, explicitly
  16-dimensional, with a curvature field of the generator's choosing and a dynamic range wider
  than the data's. Passing here is necessary for the instrument to mean anything on the data; it
  is not evidence that the data's `H_tan` is what the instrument reports there.
- **The positive-control gate.** This run is not pre-registered and is not the positive control
  the pre-registration names; nothing in `VERDICT_RULE` reads it.
- **The magnitude of the real effect.** A validated instrument does not change the size of the
  controlled partials in the Wave A tables, their density confound, or the `DOES NOT REPLICATE`
  reading of them.
- **His estimator at his own neighbourhood density.** Here `k=2048` is `1/42` of the sample, as in
  Supplement 01; his own runs used `1/8`. No `k` sweep was run.

## 8. Consequence for the Phase 9 record

The frozen `DOES NOT REPLICATE` verdict stands on an instrument that passes a known-answer test in
regime — the autoencoder `H_tan` the pre-registration named — while the colleague's instrument,
which produces the opposite sign on the same anchors (Supplement 01), fails that test here and
anticorrelates with the truth under sample noise. The opposite-ordering finding of Supplement 01
§ 5 is therefore no longer symmetric: the ordering that disagrees with the known field is his.

Recommendations for 09-10 (`09-FINDINGS.md`):

1. Present this adjudication **beside** the verdict, not inside it: the verdict is pre-registered
   and this is not. The findings should say that the verdict's instrument is the one that passed.
2. Update the "neither validated" caveat of `09-SUPPLEMENT-01-COLLEAGUE-ESTIMATOR.md` § 7 to
   "validated in regime on a known-answer fixture: ours PASS (both noise levels), his FAIL (both);
   see Supplement 02", keeping the fixture caveats of § 6 attached to it.
3. Natural follow-ups, neither run here: a **noise sweep** (displacement from 0 to beyond 25% of
   the patch radius, locating where each instrument's rank crosses `0.7`) and a
   **narrower-dynamic-range fixture** (a truth field whose p95/p05 is closer to the real data's,
   where rank is less permissive and the noiseless `0.938` would be a real test).
4. Append the `--mode swiss-roll` rows to the record so § 3 no longer rests on the brief.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Supplement 02 — post-hoc, not pre-registered, feeds no verdict. Run commit
`068490d34b1f200bf0ffc9ec69c55e0c6f6ebaeb`, colleague commit
`97efb2eb6cd7dec7f2c568f53c534752ff3c32c8`, record sha256 `2779170f0359c8d8…`*
