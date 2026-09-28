# 09-SUPPLEMENT-03 — the probe pipeline on a known surface: curvature or density?

**Status:** post-hoc, supplementary. **Not pre-registered. Feeds no verdict.** Nothing here changes
`09-WAVE-A-RESULTS.md`, `09-WAVE-A-RESULTS-AMENDMENT-01.md`, `09-SUPPLEMENT-01-COLLEAGUE-ESTIMATOR.md`,
`09-SUPPLEMENT-02-INSTRUMENT-ADJUDICATION.md` or the phase verdict. No sealed constant is reinterpreted.
**Written:** 2026-09-09 UTC

## The question

Phase 9 measured the controlled partial Spearman between local curvature and the local
out-of-fold R² of a global ridge probe, and got `+0.328` with the decoder instrument (Amendment 01,
`d=16`) against the colleague's `−0.240` with his split-half quadratic instrument; his instrument
inside our pipeline gave `−0.149` on the same anchors (Supplement 01). Supplement 02 validated
the decoder instrument on a known-answer surface and found his instrument below the bar at the
baseline sampling. None of that says what the statistic itself measures. No experiment in the
record had run the sealed probe pipeline on a surface where curvature, tangent planes, density
and the label are all known.

Two Swiss-roll notebooks did that first (`notebooks/09.1_swiss_roll_probe_decodability_check.ipynb`,
`notebooks/09.2_swiss_roll_density_decoupling_check.ipynb`). On the roll, an intrinsic-linear
label gives a negative curvature partial that stays negative (`−0.49 / −0.76 / −0.52`) when the
density-curvature coupling is swung from `−0.72` to `+0.50`, while a ridge null (ambient-linear
label at `alpha = 100`) carries a partial that tracks the coupling sign for sign (`−0.25 / +0.02 /
+0.38`). The roll cannot go further: it is a one-parameter surface in `R³`, outside both
production instruments' regime.

This supplement repeats the design on the Phase 9 adjudication fixture at production scale, with
the density-curvature coupling set by construction and all four curvature columns scored on the
same anchors.

## Provenance

Executed over SSH on the same host as `09-EXECUTION-HOST.md` § 9 (host label `pod128`, 16 threads,
CPU only), under the developer's 2026-09-09 UTC instruction to set up an experiment separating
curvature from residual density leakage and to read the pod guide before touching the host.
Guide digest verified unchanged on the host before any command. Host identity recorded as
capability only (§ 7).

Runner: `notebooks/diagnostics/09_fixture_probe_decodability_run.py`, commit
`3c1260d640706e3c0dde714c020b83b07ff05b4e` (`repo_head` in the record's `environment` row). It loads
`09_instrument_adjudication_run.py` unchanged (generator, latent draw, sealed truth, decoder path,
colleague path, scorer) and the sealed `physics_curvature_probe` pipeline (`oof_ridge_predictions`,
`local_r2_panel`, `controlled_partial`, `permutation_fwer`). Colleague code from the read-only
checkout at `97efb2eb6cd7dec7f2c568f53c534752ff3c32c8`, `topology` shimmed as in Supplements 01
and 02. Wall clock 6,689 s. Record
`notebooks/.cache/09_fixture_probe_decodability.jsonl`, 40 rows, sha256
`bf455f03f979e7f151b4cfbaaa444fd80359eaa65855ce24a69d4687ce750a6b`, verified on both sides of
the transfer. Pod copy `/mnt/ssd-cluster/effdim/probe-decodability/`.

## 1. Design

**Surface.** The Supplement 02 generator, unchanged: `G(z) = normalize([stereo(z); 0.8·h(z); 0…]) Qᵀ`,
`d = 16`, `D = 768`, seed `20260905`. Image on the unit sphere, 18-dimensional span (see the
white paper's § 5.1 and memory `adjudication-fixture-spans-18-dims`).

**Density set by construction.** A latent pool of `3n = 259,413` points from the fixture's scale
mixture. For each `gamma ∈ {−1, 0, +1}` the sample is `n = 86,471` draws from the pool without
replacement with weight `‖H_tan‖^gamma`, `‖H_tan‖` clipped to its `[p05, p95]` band. `gamma = 0` is
a uniform subsample. The surface, the generator and the anchor rule are identical across `gamma`;
only where the points sit changes.

**Exact curvature for the pool.** The sealed autodiff costs 70–250 s per 512 points; a 259k pool
would take hours. Curvature is invariant under the fixed rotation `Q`, so `‖H_tan‖` for the pool is
computed by batched central finite differences of the unrotated map `f(z) = normalize([stereo(z);
0.8·h(z)])` in `R^18` with an independent numpy `H = tr_g(II)` (83 s for the pool). At the 512
anchors of every `gamma` the run compares it with the sealed autodiff: max relative error
`1.2e-5 / 1.4e-6 / 5.5e-7`, rank agreement `1.000000` at all three. The anchor column
`exact_point` is the sealed autodiff value; pool weights and the patch-mean column use the
finite-difference field.

**Four curvature columns on the same 512 anchors.**

| column | what |
|---|---|
| `exact_point` | sealed autodiff `‖H_tan‖` of `G` at the anchor's own latent |
| `exact_patch` | mean of the finite-difference `‖H_tan‖` over the anchor's 2048 neighbours |
| `decoder_H_tan` | Amendment 01 decoder instrument, frozen fit protocol (`600` epochs, seed `0`) |
| `colleague_K_H_cross` | his `nested_pca_frame` + `_fit_rank`, `n_splits = 3`, seed `0`, unchanged |

Pointwise against patch-mean tests the scale explanation; exact against decoder tests the
instrument explanation.

**Three label arms**, seeded unit vectors `a₁, a₂ ∈ R^16`, `w ∈ R^768`:

| arm | label | why |
|---|---|---|
| `ambient_linear_null` | `y = w·x` | linear in ambient space; the ridge null of notebook 09.1 |
| `intrinsic_linear` | `y = a₁·z` | linear in intrinsic coordinates, nonlinear in ambient; the main arm |
| `nonlinear` | `y = sin(2 a₁·z) + (a₂·z)² − ½ (a₁·z)(a₂·z)` | smooth, ridge fit deliberately poor |

**Statistic.** The sealed pipeline: five-fold OOF ridge at `alpha = 100`, local R² over the
`k = 2048` neighbours, controls `(log_knn_radius, local_label_variance, local_evaluation_count)`,
rank-partial Spearman, Freedman-Lane null with `N_PERMUTATIONS = 10,000`.

## 2. The coupling each sample actually has

`rho(column, log_knn_radius)` at the anchors. Positive means the column is large where
neighbourhoods are wide, i.e. where the sample is sparse.

| `gamma` | `r/R` | truth `‖H_tan‖` p05/p50/p95 | `exact_point` | `exact_patch` | `decoder_H_tan` | `colleague_K_H_cross` |
|---:|---:|---|---:|---:|---:|---:|
| −1 | 0.999 | 0.005 / 0.265 / 1.522 | **+0.797** | +0.865 | +0.537 | +0.626 |
| 0 | 0.978 | 0.044 / 0.780 / 1.803 | **+0.508** | +0.714 | +0.370 | +0.314 |
| +1 | 0.989 | 0.305 / 1.051 / 2.262 | **−0.153** | −0.149 | −0.218 | −0.162 |

The manipulation moved the exact-curvature coupling from `+0.80` through `+0.51` to `−0.15`. The
baseline fixture (`gamma = 0`) already has curvature high where the sample is sparse (`+0.51`,
Supplement 02 measured `+0.42` on its own uniform draw), and the `[p05, p95]` clip on the
weights limits how far `gamma = +1` can push the other way. Every instrument column carries the
same sign of coupling as the exact field at every `gamma`.

**Instrument fidelity on these samples** (rank Spearman against the exact truth at the anchors,
`fit_info` in the `sample` rows):

| `gamma` | decoder rank / direction cos / magnitude ratio / var. explained | colleague rank / `R_H` median |
|---:|---|---|
| −1 | 0.806 / 0.995 / 1.087 / 0.99989 | 0.737 / 0.787 |
| 0 | 0.940 / 0.999 / 1.068 / 0.99988 | 0.600 / 0.787 |
| +1 | 0.929 / 1.000 / 1.018 / 0.99991 | 0.780 / 0.965 |

The `gamma = 0` cell reproduces Supplement 02 (decoder `0.938`, colleague `0.622`, on a different
uniform draw). At `gamma = −1` and `+1` the colleague estimator ranks `0.74` and `0.78` against
the truth, above Supplement 02's `0.7` bar. Supplement 02's noiseless FAIL is therefore a
property of that sampling, not a fixed property of the estimator on this surface; the noisy-arm
inversion (`−0.30`) was not re-run here. Cross-column rank agreement is `0.60–0.94` throughout
(`column_cross_rank_rho` in the record).

## 3. The controlled partials

Raw Spearman and the sealed three-control partial of each column against local R², with the
Freedman-Lane `p` (floor `1e-4`). Global OOF R² and the median local R² of the arm beside it.

### `intrinsic_linear` — the main arm

| `gamma` | coupling (exact) | global R² | local R² p50 | `exact_point` | `exact_patch` | `decoder_H_tan` | `colleague_K_H_cross` |
|---:|---:|---:|---:|---|---|---|---|
| −1 | +0.80 | 0.717 | 0.641 | **−0.291** (p 1e-4) | −0.490 (p 1e-4) | −0.294 (p 1e-4) | −0.258 (p 1e-4) |
| 0 | +0.51 | 0.742 | 0.620 | **−0.002** (p 0.95) | +0.042 (p 0.34) | +0.011 (p 0.80) | +0.193 (p 1e-4) |
| +1 | −0.15 | 0.816 | 0.708 | **+0.207** (p 1e-4) | +0.222 (p 1e-4) | +0.215 (p 2e-4) | +0.210 (p 1e-4) |

### `ambient_linear_null` — the ridge null

| `gamma` | global R² | local R² p50 | `exact_point` | `exact_patch` | `decoder_H_tan` | `colleague_K_H_cross` |
|---:|---:|---:|---|---|---|---|
| −1 | 0.997 | 0.997 | +0.253 (p 1e-4) | +0.010 (p 0.81) | +0.091 (p 0.035) | +0.113 (p 0.012) |
| 0 | 0.998 | 0.998 | −0.153 (p 9e-4) | −0.310 (p 1e-4) | −0.180 (p 1e-4) | −0.080 (p 0.076) |
| +1 | 0.998 | 0.998 | +0.029 (p 0.52) | −0.033 (p 0.49) | +0.026 (p 0.56) | −0.009 (p 0.83) |

### `nonlinear`

| `gamma` | global R² | local R² p50 | `exact_point` | `exact_patch` | `decoder_H_tan` | `colleague_K_H_cross` |
|---:|---:|---:|---|---|---|---|
| −1 | 0.163 | 0.044 | −0.011 (p 0.81) | +0.056 (p 0.21) | +0.062 (p 0.16) | +0.006 (p 0.89) |
| 0 | 0.268 | 0.091 | +0.045 (p 0.31) | +0.087 (p 0.053) | +0.049 (p 0.27) | +0.087 (p 0.054) |
| +1 | 0.406 | 0.205 | +0.085 (p 0.045) | +0.159 (p 5e-4) | +0.100 (p 0.020) | +0.140 (p 1e-3) |

## 4. Reading

1. **On this surface the sign of the curvature-decodability partial is set by the sample, not
   by the curvature.** Same generator, same label, exact curvature at the anchors. Coupling
   `+0.80` gives `−0.29`; coupling `+0.51` gives `0.00`; coupling `−0.15` gives `+0.21`. Every
   cell is at the permutation floor except the middle one. The three controls, including
   `log_knn_radius`, do not remove it. The Swiss roll behaved differently (the partial stayed
   negative across the same manipulation), and the two surfaces differ in one relevant way: on
   the roll the global probe fails almost everywhere (global R² `0.07`, local R² down to `−60`),
   so the curvature effect on the probe is enormous; on this surface the intrinsic-linear label
   is fit at R² `0.72–0.82`, the surface is mildly curved (truth spread `7–300×`, but the bumps
   are shallow), and whatever curvature does to the probe is smaller than what the sampling
   does to the statistic.

2. **Neither scale nor instrument explains the real-data sign.** Pointwise and patch-mean exact
   curvature agree in sign at every `gamma`. The decoder column agrees with the exact column in
   sign and magnitude at every `gamma` (`−0.29 / +0.01 / +0.22` against `−0.29 / 0.00 / +0.21`).
   His column agrees in sign too. Candidates 1 (decoder field is not the curvature) and 3
   (pointwise versus patch) from the white paper's discussion are not supported here; candidate
   2 (a density mechanism the frozen controls do not remove) is.

3. **Both real-data signs are what this mechanism predicts.** The rule this fixture shows is:
   the partial takes the opposite sign to the column's coupling with `log_knn_radius`. On the
   real data the decoder field couples `−0.60` and gave `+0.328`; the colleague's field couples
   `+0.70` and gave `−0.240` (his own reanalysis: `+0.765` coupling). Both match. The two
   instruments disagree on the real data because they couple to density in opposite directions,
   and the statistic follows the coupling.

4. **The ridge null is not a clean leakage probe at this scale.** Local R² sits at `0.997`
   everywhere, its variation is small, and the partials (`+0.25 / −0.15 / +0.03` for the exact
   column) do not track the coupling monotonically as they did on the roll. The nonlinear arm's
   probe barely works (global R² `0.16–0.41`, local R² near zero) and shows little.

5. **The colleague's estimator is in regime on two of three samplings.** Rank against the exact
   truth `0.74 / 0.60 / 0.78`. Supplement 02's noiseless FAIL at `0.62` sits at the uniform
   sampling only. This does not revisit the noisy-arm inversion, which was not re-run.

## 5. What this settles and what it does not

**Settles.** In the production regime (`D = 768`, `d = 16`, `n = 86,471`, `k = 2048`,
`r/R ≈ 1`), with exact curvature and a label linear in the intrinsic coordinates, the frozen
statistic's sign is determined by how the sample's density relates to its curvature, and the
frozen controls do not remove that dependence. A field that couples negatively to neighbourhood
radius reads positive; one that couples positively reads negative. The Phase 9 `+0.328` and the
colleague's `−0.240` are both consistent with that rule and neither is evidence about curvature
on its own.

**Does not settle.**

- Whether the real Physics manifold is in the "weak curvature" regime this fixture is in, or the
  roll's "strong curvature" regime. The real field's `1.8×` spread and the real probe's
  behaviour (local R² falls with radius at `−0.23`) point to the former, but that is inference.
- What a density-robust statistic would read. Candidates, none run: a within-anchor design that
  compares curvature to local R² at matched neighbourhood radius; residualising local R² on a
  richer density model than one log radius; or a sample re-weighted to `gamma` such that the
  chosen instrument's coupling is zero, then reading the partial there.
- The noisy-arm behaviour of either instrument under these samplings.
- The positive-control gate, which this run does not touch.

## 6. Consequence for 09-10

`09-FINDINGS.md` should present the verdict sentence as pre-registered, and beside it state that
on a known surface at production scale the verdict statistic's sign follows the density coupling
of whichever curvature field is used, that both instruments' real-data signs match that rule,
and that the record therefore holds no positive or negative curvature-decodability finding on
the Physics data. The instrument adjudication (Supplement 02) still says which field tracks the
curvature; this supplement says the statistic that was pre-registered cannot turn that into a
curvature claim without a density-robust design.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Supplement 03 — post-hoc, not pre-registered, feeds no verdict. Run commit `3c1260d6`, colleague
commit `97efb2eb`, record sha256 `bf455f03…750a6b`.*
