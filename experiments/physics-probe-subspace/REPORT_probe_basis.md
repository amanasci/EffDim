# Does a supervised physics-probe subspace beat ambient cosine for cross-model mKNN?

**Run:** `outputs/probe_basis/results.json` · 38 `independent` probes, k=10
**Data:** `UniverseTBD/pu-embeddings` physics split, `vit_base` ↔ `dinov3_vitb16`, 768-d, L2-normalised, `max_n=16384` → `n_train=11468`, `n_test=4916`
**Code:** `probe_basis_mknn.py`, `analyse_probes.py`, labels via `_common.load_physics_labels`
**Headline:** mKNN drops from **0.1320** (ambient cosine) to **0.0564** (38-d probe subspace) — and the probe subspace is beaten by a *random* subspace of the same rank.

---

## 1. Motivation

Prior work in [`experiments/SAE-shared-basis/CONTEXT.md`](../SAE-shared-basis/CONTEXT.md) left a specific puzzle. The embedding cloud behaves like a **~10-d linear core inside a thick soft shell** (PCA-95 planes at median *d* ≈ 87). Every attempt to exploit that by reducing dimension *lost* cross-model neighbour agreement: PCA-40 falls below ambient cosine, autoencoder bottlenecks worse still. The soft shell is not noise — it carries the fine-grained distributed features that decide neighbour *ranking*. Meanwhile blind whitening fails in the opposite direction: dividing by λ^(−1/2) in raw 768-d space inflates low-variance directions by >30×.

So unsupervised variance hierarchies give a bad handle in both directions. The proposal in [`PHYSICS_PROBE_SUBSPACE_CONTEXT.md`](PHYSICS_PROBE_SUBSPACE_CONTEXT.md) was to replace variance with **supervision**: train *M* linear probes on ground-truth astronomical properties, take their normal vectors *w_m*, and use span{*w_m*} as the projection target. The argument for it:

- Each *w_m* points along a direction of genuine physical variation, so the subspace is defined by *what the galaxies are*, not by how much variance a given architecture happened to assign.
- Projecting onto it annihilates the (768 − *M*) directions of model-private background, so distance is computed strictly on actionable semantics.
- With *M* ≈ 50–100 targets the subspace could match the *d* ≈ 87 soft shell while staying noise-free — capturing what PCA-40 threw away.

**The hypothesis under test:** a task-anchored projection retains the neighbourhood structure that an unsupervised projection of the same rank destroys, and therefore matches or beats ambient cosine mKNN.

It does not. This report is about why, and the answer turns out to be about label coverage rather than about geometry.

---

## 2. Methodology

### 2.1 Pipeline as run

1. **Load and normalise.** Both models' 768-d embeddings, L2-normalised, truncated to `max_n=16384`, rows paired by index.
2. **Split.** `train_test_split(test_size=0.3, random_state=0)` → 11468 train / 4916 test. The same split and seed as the curvature run, so probe statistics are directly comparable.
3. **Probes.** 38 targets from `INDEPENDENT_PROBES` — Galaxy Zoo vote fractions, *grz* photometry, Sérsic/Petrosian structure, stellar mass, redshift, SFR/sSFR, plus 11 derived quantities. Per probe: drop rows with NaN targets, standardise *y* on the surviving rows, fit `LinearRegression` on the train split, record `r2_train` and 5-fold CV `r2_cv`. Column *m* of *W* ∈ ℝ^(768×38) is that probe's coefficient vector. Fitted independently per model.
4. **Orthonormalise.** `np.linalg.qr(W)` → *Q*, intended to remove collinearity between physically correlated properties before projection.
5. **Project and score.** Test embeddings → *Z*·*Q*; exact k=10 NN with Euclidean distance in the subspace, versus cosine in the 768-d ambient space; mKNN = mean fraction of shared neighbours across the two models.
6. **Sweep.** Repeat for *d* = 1…38, adding one orthogonalised column at a time.

Orthogonal Procrustes alignment of *W_A* to *W_B* was deliberately **dropped** from this run (`PHYSICS_PROBE_SUBSPACE_CONTEXT.md` §6) to isolate the projection effect. Each model gets its own independent basis and its own independent neighbour graph.

### 2.2 Controls added for this report

The run as scripted reports the subspace against exactly one baseline — ambient cosine. That is not enough to attribute a drop to anything. Three controls were added, all reproducing the run's split, seed, normalisation and mKNN code path, and validated by recovering ambient cosine to `0.13201790` against the run's reported `0.1320`:

- **Random subspace.** Two independent Haar-random orthonormal 768×*d* bases, one per model — the same *structure* as the experiment (independent per-model bases, Euclidean in projected coordinates) with the supervision removed. This is the control that says whether the probes contributed anything.
- **PCA-*d*.** Fitted on the train split per model. The unsupervised method the probe basis was meant to improve on.
- **Metric check.** Cosine and Euclidean scored in every projected space, since the run compares ambient *cosine* to subspace *Euclidean*.
- **Random-subspace principal angles.** 200 draws of two independent random *m*-d subspaces of ℝ^768, to calibrate the reported 78.97°.

---

## 3. Findings

### 3.1 The projection loses 57% of the neighbour agreement

| representation | dim | metric | mKNN@10 |
|---|---:|---|---:|
| **probe subspace** | 38 | Euclidean | **0.0564** |
| random subspace | 38 | Euclidean | 0.0739 |
| PCA | 38 | Euclidean | 0.1151 |
| PCA | 38 | cosine | 0.1227 |
| **dense ambient** | 768 | cosine | **0.1320** |
| dense ambient | 768 | Euclidean | 0.1320 |
| PCA | 64 | cosine | 0.1321 |
| PCA | 128 | cosine | 0.1391 |
| SAE codes + IDF *(prior work)* | 2048 | cosine | 0.172 |
| Ridge shared DINO basis + IDF *(prior work)* | — | cosine | 0.220 |

Ambient cosine and ambient Euclidean are **identical to 8 significant figures** — as they must be on L2-normalised vectors, since the two orderings are monotonically related. So the metric switch is not the confound; and inside the projected spaces the two metrics agree closely (random-38: 0.0739 vs 0.0733; PCA-38: 0.1151 vs 0.1227). The drop is the projection.

### 3.2 The supervised subspace is worse than a random one of the same rank

This is the finding that matters. At *d* = 38, mKNN is **0.0564** for the probe basis and **0.0739** for two independent random bases (mean of 3 draws). The probes did not merely fail to help — 38 directions chosen by supervision preserve *less* cross-model neighbourhood structure than 38 directions chosen by a random number generator.

The full ladder, at matched rank:

| *d* | random subspace | PCA (Euclid) | PCA (cosine) |
|---:|---:|---:|---:|
| 3 | 0.0035 | 0.0144 | 0.0097 |
| 8 | 0.0156 | 0.0567 | 0.0587 |
| 38 | 0.0739 | 0.1151 | 0.1227 |
| 64 | 0.0938 | 0.1243 | 0.1321 |
| 128 | 0.1114 | 0.1286 | 0.1391 |

Two things follow. First, the prior claim that unsupervised truncation loses signal is confirmed and quantified — PCA-38 sits at 0.115–0.123 against ambient 0.132, and you need ~64 components before cosine recovers ambient. Second, PCA-38 beats the probe basis by a factor of **2.0–2.2**. Whatever the probe directions are aligned with, it is not the structure that mKNN measures. Note also that the probe subspace's 0.0564 is close to random-8's neighbourhood and to PCA-8 (0.0567) — a hint at its *effective* rather than nominal rank, developed in §3.4.

*Caveat: three random draws, not a distribution. The 0.0564 vs 0.0739 gap is large relative to plausible seed spread at n=4916, but it has not been formally tested.*

### 3.3 30 of the 38 probes are interpolating, not learning

The probe diagnostics explain it. Only **8 of 38** probes reach `r2_cv > 0.1` for `vit_base` (9 for `dinov3`), and those are exactly the probes with near-complete label coverage. The rest have `n_valid` between 725 and 1187 against **D = 768 features** — at or below the ambient dimension, where OLS interpolates the training rows and the coefficient vector is set by the subsample rather than by the property.

| probe | n_valid | r2_train | r2_cv (vit) | r2_cv (dino) |
|---|---:|---:|---:|---:|
| `smooth` | 11468 | 0.706 | **+0.631** | **+0.693** |
| `featured_disk` | 11468 | 0.697 | **+0.623** | **+0.698** |
| `smooth_minus_disk` | 11468 | 0.697 | **+0.623** | **+0.694** |
| `redshift` | 10652 | 0.665 | **+0.594** | **+0.613** |
| `total_merger` | 11468 | 0.504 | **+0.398** | **+0.519** |
| `merging` | 11468 | 0.434 | **+0.315** | **+0.425** |
| `major_disturbance` | 11468 | 0.435 | **+0.304** | **+0.420** |
| `minor_disturbance` | 11468 | 0.394 | **+0.277** | **+0.340** |
| `edge_on` | 1187 | 0.954 | −0.003 | +0.252 |
| `stellar_mass` | 803 | 0.987 | −2.56 | −1.65 |
| `bulge_dominant` | 725 | **1.000** | −7.51 | −5.06 |
| `petro_th50` | 803 | 0.983 | −32.4 | −34.2 |
| `ssfr` | 966 | 0.939 | −31.4 | −24.2 |
| `sfr` | 967 | 0.876 | −82.7 | −51.2 |
| `elpetro_theta` | 803 | 0.993 | **−11117** | **−24463** |

The signature is unmistakable: `r2_train` = 1.000 exactly for all nine probes with `n_valid` = 725, with `r2_cv` between −1.5 and −7.5. `elpetro_theta` at `r2_cv` = −11117 is the same pathology unbounded. Every probe with useful CV performance draws on the full ~11.5k Galaxy Zoo coverage; every probe drawing on the NSA/MaNGA cross-match (725–1187 rows) is fitting noise.

So 30 of the 38 basis directions are **not physics directions at all**. They are minimum-norm interpolants of an ~800-row subsample, and which direction each one points is a property of that subsample.

### 3.4 The `independent` probe set is not linearly independent — *W* has rank ≤ 31

`INDEPENDENT_PROBES` was built to remove the sum-to-one redundancy in the Galaxy Zoo vote-fraction groups. It does that, then reintroduces exact dependencies through the derived block. Reading the definitions in `_common.load_physics_labels` against the list, seven targets are **exact** linear combinations of others already in it:

| derived target | identity | constituents in the list? |
|---|---|---|
| `g_minus_r` | `mag_g − mag_r` | yes (both, same 803 rows) |
| `g_minus_z` | `mag_g − mag_z` | yes |
| `r_minus_z` | `mag_r − mag_z` | yes |
| `smooth_minus_disk` | `smooth − featured_disk` | yes (both, same 11468 rows) |
| `bar_signal` | `bar_strong + bar_weak` | yes (both, same 725 rows) |
| `bulge_total` | `bulge_dominant + bulge_large` | yes |
| `total_merger` | `merging + major + minor` | yes (all three) |

OLS coefficients are linear in the target, and `LinearRegression`'s min-norm solution `pinv(X)y` is linear in *y* too — so these identities transfer exactly from the targets to the columns of *W*, provided the constituent probes were fit on the same rows. They were: each dependency's members share an identical `n_valid`, because they come from the same catalogue join. **rank(*W*) ≤ 31.** (The remaining derived targets — `concentration`, `log_petro_th50`, `log_petro_th90`, `log_sersic_n` — are nonlinear and escape this.)

`np.linalg.qr` does not detect rank deficiency. Householder QR on a (768, 38) matrix of rank 31 returns 38 orthonormal columns regardless; seven diagonal entries of *R* are ~0 and the corresponding columns of *Q* are directions determined by floating-point rounding in the dependent columns. **At least 7 of the 38 "physical" basis directions are numerical noise.** This is verifiable directly from the saved artefacts: inspect `|diag(R)|` in `probe_weights.npz` and expect ~7 entries orders of magnitude below the rest.

The pipeline could not have caught this, because the rank it reports is a tautology — `rank_a` is set from `Q_A.shape[1]`, the number of columns requested, so it prints 38 for any input.

Now stack §3.3 and §3.4 on the 8 probes that *do* work. Two of them (`smooth_minus_disk`, `total_merger`) are exact dependents of the other six, so the working set has rank **6**, not 8. Of the remainder, `featured_disk` ≈ 1 − `smooth` − `artifact` with `artifact` small, so those two are near-collinear; and `merging` / `major_disturbance` / `minor_disturbance` are three answers to one Galaxy Zoo question, with CV R² tracking each other to within 0.04. Counting genuinely distinct physical axes, the honest estimate is **about three**: a smooth↔featured morphology axis, a merger-disturbance axis, and redshift.

*(The rank-6 figure is exact; "about three distinct axes" is an argument from the target definitions and the correlated R² values, not a measurement. Confirming it needs the singular spectrum of the 8-column signal block — cheap, but not run here.)*

### 3.5 QR is what turns a weak basis into a harmful one

The three signal axes are still *in* the projected space. The reason mKNN lands below random is what QR does to their relative weight.

Orthonormalisation was included for numerical stability against collinear physical properties, and for that purpose it is correct. But it also **equalises** the columns: in *Q*-coordinates, roughly 3 signal directions and roughly 35 noise directions carry the same metric weight. Euclidean distance then spends ~92% of its budget on directions fitted to ~800-row subsamples, and the neighbour ranking is decided by them.

That is structurally the same failure as the blind-whitening result this design was built to avoid — low-information directions promoted to unit weight, drowning the informative ones. The difference is only that here it arrives through orthonormalisation of a rank-deficient, mostly-overfitted weight matrix rather than through λ^(−1/2). It also explains why the probe basis underperforms *random*: a random 38-d projection at least samples the ambient sphere isotropically and retains a fair share of the soft-shell variance that ambient cosine uses, whereas the probe basis concentrates its 38 directions in whatever subspace the ~800-row interpolants happen to occupy.

### 3.6 The 78.97° principal angle is exactly the random-subspace value — and the diagnostic is not well-posed

`analyse_probes.py` reports a mean principal angle of **78.97°** between the two models' bases. Calibrated against 200 draws of two independent random *m*-d subspaces of ℝ^768:

| *m* | mean principal angle, random subspaces | measured |
|---:|---:|---:|
| 3 | 86.97° ± 0.71° | — |
| 8 | 85.10° ± 0.50° | — |
| 38 | **79.03° ± 0.21°** | **78.97°** |

The measured value sits **0.3 standard deviations** from the random baseline. There is no detectable shared structure in this statistic.

Two separate statements are needed here, and only the first is a finding about the diagnostic:

**The statistic cannot detect shared structure even in principle.** It computes `svd(Q_A.T @ Q_B)`, but *Q_A* lives in `vit_base`'s coordinate frame and *Q_B* in `dinov3`'s. Those frames have no common basis; applying an arbitrary rotation to one model's embeddings would change the answer while changing nothing about the representation. Coming out at the random value is the expected behaviour of a statistic comparing two unrelated coordinate systems.

**Therefore it is *not* evidence that the two models' physical subspaces are unrelated.** That claim requires a frame-free comparison — Procrustes or CCA on the paired projections *Z_A Q_A* vs *Z_B Q_B*, which are indexed by the same galaxies and so *are* comparable. Procrustes was dropped from this run, and that test has not been done. The subspace-relation question in the plan is open, not answered.

### 3.7 Answering the diagnostic questions from the experiment plan

- **Better than standard mKNN?** No: 0.0564 vs 0.1320, a 57% loss.
- **Better than the SAE experiment?** No, by a wide margin. Ridge shared-basis + IDF reaches 0.220 and SAE + IDF 0.172; both work by *mapping between* representations at high rank rather than truncating either one.
- **Overfitting / too few probes?** Overfitting, yes — severely, in 30 of 38 probes. But the plan's diagnosis is inverted. The problem is not too few *probes*; it is too few *labelled rows per probe*. Adding probes drawn from the same NSA/MaNGA cross-match adds more interpolants, and each one makes the QR-equalised metric worse. This is why the run "doesn't work for 38 probes" and why going straight to more probes would not have fixed it.
- **Relationship with local curvature?** Tested in [`REPORT_density_curvature.md`](REPORT_density_curvature.md) §5.3–5.5, and the answer is no: calibrated curvature contributes nothing to probe error once local density `d_k` is partialled out (partial ρ = −0.032). Local off-subspace variance `rf_k` does survive (partial ρ = +0.139 / +0.295). That report also found the legacy probe-error target was substantially a label-availability artifact — ρ(mean residual, `n_valid_probes`) = **+0.434** — which is the same coverage problem diagnosed here, surfacing in a different measurement.

---

## 4. What this does and does not rule out

**Ruled out, for this probe suite:** a 38-d QR-orthonormalised probe basis as a metric for cross-model mKNN. It is worse than ambient, worse than PCA at matched rank, and worse than random at matched rank. Retrying it with more probes from the same catalogues would make it worse, not better.

**Not ruled out — untested, because this run could not test them:**

- **The core hypothesis at adequate coverage.** With ~3 effective signal directions out of 38 nominal, this run never constructed the object the proposal described. A basis of 50–100 probes each fitted on ≥10k rows remains untested.
- **Any weighting other than equal.** Orthonormalisation was the aggravating step. Scaling directions by √(R²) as the plan's accuracy-weighting suggested, or by projected data variance, was never run — and with `r2_cv` ranging from +0.69 to −24463, accuracy weighting alone would have suppressed most of the harmful directions.
- **Whether the two subspaces are related.** Requires a frame-free test (§3.6).
- **Projection as filter vs. as replacement.** Every variant here discards the orthogonal complement. Ambient cosine wins partly because it uses the soft shell; a metric that *reweights* toward probe directions while retaining the complement is a different, untested proposition.

---

## 5. Limitations

- The mKNN control numbers here were computed for this report, not by the run script. They reproduce the run's ambient cosine to `0.13201790` on the same split, seed, normalisation and mKNN code, but they are a re-derivation.
- Random-subspace mKNN used 3 draws per dimension, giving means without confidence intervals.
- The probe R² values are read from the curvature run's `probe_stats`, which shares the identical split, seed, probe list and 5-fold KFold(seed=42) and therefore matches — but is a different JSON file than `outputs/probe_basis/results.json`.
- **rank(*W*) ≤ 31 is derived from the target definitions, not measured.** Confirm via `|diag(R)|` in `probe_weights.npz`.
- The `mknn_vs_dim` sweep was produced by the run but is not analysed here. It is also hard to read as intended: QR orders columns by Gram-Schmidt order over `INDEPENDENT_PROBES`, which is neither an importance ordering nor stable, and the dependent columns contribute rounding-noise directions at whatever position they occupy.
- **8 of 38 probes were usable, so this run tested a ~3-effective-dimension subspace, not a 38-dimensional one.** Every negative result above should be read with that scope.

---

## 6. Recommended next steps

Ordered by information gained per unit effort.

1. **Fix coverage before touching anything else.** Phase 1 of [`catalogue_crossmatch_plan.md`](catalogue_crossmatch_plan.md) — the GZ DESI Advanced join on `dr8_id` — adds ~16 independent morphology fractions at **~95–99% coverage**, and Phase 2 (Tractor WISE via Astro Data Lab) adds ~10 more at ~100%. That is ~30 probes at full coverage against the 3 the current run actually has. This is the binding constraint; nothing else is worth varying until it is lifted.
2. **Gate probes on `r2_cv`, not on membership in a list.** `stage_b_connectback.py` already implements exactly this filter (`probe_r2_min = 0.1`). Build the basis only from probes that pass, and record how many did.
3. **Deduplicate the target list properly.** Drop the 7 exactly-dependent derived targets, or keep them and use a rank-revealing decomposition (`scipy.linalg.qr(pivoting=True)`, or SVD with a tolerance) instead of plain `np.linalg.qr`. Report the *numerical* rank rather than `Q.shape[1]`.
4. **Replace equal weighting.** Scale each direction by √(R²_A · R²_B) per the plan's accuracy weighting, or by projected data variance. Cheap, and it targets the mechanism in §3.5 directly.
5. **Always run the random-subspace control at matched rank.** It is the control that converted this run from "downprojection loses signal" into "supervision made it worse than chance", and it costs one QR.
6. **Test the subspace relation frame-free.** CCA between *Z_A Q_A* and *Z_B Q_B* on paired test galaxies, or restore Procrustes. Retire the cross-frame `Q_A.T @ Q_B` principal-angle plot, which cannot answer the question it is labelled with.
7. **Consider that the SAE line already answers this better.** The two methods that beat ambient — SAE + IDF (0.172) and Ridge shared basis + IDF (0.220) — both *reweight or map at high rank*. Every truncation tried, supervised or not, has lost. If the probe subspace is worth pursuing, the more promising form is as a reweighting of the ambient metric toward probe directions, keeping the complement, rather than as a projection that deletes it.
