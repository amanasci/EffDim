# EffDim

## What This Is / Core Value

EffDim is a research Python library computing "effective dimensionality" (ED) of data across modalities. Entry point `effdim.compute_dim(data)` returns spectral estimates (participation ratio, stable rank, numerical rank, Shannon/Renyi entropy dimensions, PCA explained-variance thresholds, cumulative eigenvalue ratio) and geometric/intrinsic estimates (MLE, TwoNN, DANCo, MiND-MLi, MiND-MLk, ESS, TLE, GMST) in one call, so a researcher can see how spectral and geometric notions of dimension agree or disagree on the same `(n_samples, n_features)` array. Targets researchers characterizing intrinsic structure of high-dimensional data — notably neural network representations and embeddings.

## Current Milestone: v1.1 PU Manifold Curvature

**Goal:** Reconstruct the PU embedding manifold via Isomap, build a smooth decoder parameterization of it, compute mean curvature, and test whether foundation-model representational alignment (MKNN) varies with local curvature.

**Target features:** stream `UniverseTBD/pu-embeddings` config `legacysurvey_dinov3_vitb16`, subsample 10k paired rows (`dinov3_vitb16_hsc` / `dinov3_vitb16_legacysurvey`); fit Isomap on `_legacysurvey`; audit the classical-MDS eigenspectrum (leading positive eigenvalues + dropoff, explicit negative-eigenvalue detection); train a torch MLP decoder (Isomap coords -> 768-d, C2-smooth activation); derive mean curvature H via first/second fundamental forms (`torch.func` Jacobian/Hessian); partition by |H| quantiles; compute crossmodal MKNN (HSC vs Legacy Survey) per region against a permutation null with bootstrap CIs.

## Requirements

### Validated

- `compute_dim` returns spectral + geometric ED estimates for a 2D numpy array; input validation (2D only, >=2 samples, finite, list-of-arrays vstacked, auto-centering); SVD path switches to randomized SVD when `min(n, d) >= 1000`; k-NN distances computed once (k=10), shared across geometric estimators; benchmark suite for runtime/accuracy vs vector size/count
- Reproducible, row-aligned 10k subsample of `legacysurvey_dinov3_vitb16`, cached and L2-normalized (DATA-01..05) — *Phase 1*
- Isomap fit validated for connectivity and `n_neighbors` stability; `k*=15` frozen by a pre-registered plateau rule (ISO-01..05) — *Phase 1*

### Active
- [ ] Isomap manifold reconstruction + MDS eigenspectrum validity audit
- [ ] Smooth decoder parameterization; mean curvature field derived analytically
- [ ] Regional MKNN alignment compared across high- vs low-curvature regions

### Deferred
- [ ] Validate estimates against known dimensionalities (noise -> D, Swiss Roll -> intrinsic); CI across platforms
- [ ] Intramodal MKNN across a model-size ladder (paper's stronger 28-56% signal) — needs a second model size

### Out of Scope
- torch stays out of `pyproject.toml` core deps; notebooks self-install
- Bundling large datasets — notebooks stream from source
- Modifying `src/effdim/` during v1.1

## Context

`src/effdim/`: `api.py` (orchestration/validation/SVD), `metrics.py` (spectral), `geometry.py` (intrinsic dimension). Core deps: numpy, scipy, scikit-learn, faiss-cpu. `benchmarks/`, `tests/`, `docs/` (mkdocs) exist. `notebooks/` holds `02_k_sensitivity_refit.ipynb` (standalone k-sensitivity gate re-fit) and `pu_manifold/`: `cache.py`+`subsample.py` implemented, `curvature.py`+`mknn.py` stubbed for Phases 3-4. Phase 1 artifacts in gitignored `notebooks/.cache/`: `isomap_43cf438bc944c509.joblib` (~1.55 GiB, carries `dist_matrix_`), `phase1_handoff_43cf438bc944c509.json` (14-key Phase 1->2 interface). Frozen: `k*=15`, `n_components=18`, `d_provisional=18`. `notebooks/requirements-notebooks.txt` duplicates/pins core deps for a user-supplied venv. `TODO.md` tracks testing/CI hardening as standing next work.

**Standing rule (`CLAUDE.md`):** every new manifold model — chart auto-encoder, topological auto-encoder, decoder parameterization, curvature estimator — ships with a Swiss roll sanity-check notebook that imports the model code unchanged, trains it in under two minutes, and plots the reconstruction against the original. A model is not done without it: on a manifold with no known answer, a FAIL cannot be told apart from a broken implementation. Reference: `notebooks/02.3_swiss_roll_cae_check.ipynb`.

### PU embeddings dataset (v1.1 subject)

`UniverseTBD/pu-embeddings` — foundation-model **image** embeddings for astronomy surveys. 163 configs, 7,050,003 rows, ~93 GB. Never materialize whole; stream one config.

| config family | rows | columns |
|---|---|---|
| `jwst_*` | 1,496 | `<model>_hsc`, `<model>_jwst` |
| `desi_*` | 20,465 | `<model>_hsc`, `<model>_desi` |
| `legacysurvey_*` | 101,725 | `<model>_hsc`, `<model>_legacysurvey` |
| `physics_*_test` | 86,471 | `<model>_galaxies` (unpaired) |

No labels, no `object_id`. Paired configs are **row-aligned** — the only join, and all MKNN needs.

### Origin experiment

Duraphe, Smith, Sourav & Wu, *The Platonic Universe: Do Foundation Models See the Same Sky?*, NeurIPS 2025 ML4PS workshop ([arXiv:2509.19453](https://arxiv.org/abs/2509.19453)). Tests the Platonic Representation Hypothesis via mutual k-nearest-neighbour score (Chechik et al. 2010): `MKNN(z1, z2) = k^-1 * |N_k(z1) ∩ N_k(z2)|` — label-free, training-free, vs a random-permutation null. Reported alignment: intramodal (two sizes, same architecture) 28-56%; crossmodal (HSC vs other) ~7% JWST, 0.4-2% Legacy Survey, 0.3-0.5% DESI, rising with model capacity (14/18 intramodal, 28/33 crossmodal comparisons). v1.1 asks a question the paper does not: is that convergence uniform across the manifold, or concentrated where it's flat?

## Constraints

Python >= 3.8. Core package installable without GPU/deep-learning stack. Notebook-only deps (torch, datasets, umap) install from within the notebook, never in core `dependencies`.

## Key Decisions

| Date | Decision | Why |
|------|----------|-----|
| 2026-07-27 | `.planning/` bootstrapped retroactively | GSD workflow requires ROADMAP.md; project predates GSD adoption |
| 2026-07-27 | Notebook deps installed in-notebook, not `pyproject.toml` | Keeps core package light |
| 2026-07-29 | v1.1 uses `legacysurvey_dinov3_vitb16`, 10k subsample | Row-aligned pair enables MKNN; 101,725 rows = best coverage of any paired config; 10k keeps dense geodesic matrix ~800 MB so Isomap stays exact |
| 2026-07-29 | Physics probe is MKNN, not a supervised head | (arXiv:2509.19453) probes label-free; pu-embeddings ships no labels/join key |
| 2026-07-29 | Single model in v1.1, no size ladder | Establishes curvature method before multiplying compute; only crossmodal MKNN computable, on Legacy the weakest signal (0.4-2%) — a null regional result is plausible |
| 2026-07-29 | Decoder is a torch MLP, C2-smooth activation | Mean curvature needs nonzero 2nd derivative (ReLU's is zero); `torch.func.jacrev`/`hessian` give fundamental forms directly |
| 2026-07-29 | Milestone is notebook-only | Promoting curvature operator into `src/effdim/` needs its own test suite/milestone |
| 2026-07-30 | `subsample_*.npz` caches normalized arrays + raw norms only, never raw 768-d vectors (D-05/D-06) | Prevents mixing normalized/raw embeddings; recovery means re-streaming the 553 MiB parquet |
| 2026-07-30 | `requirements-notebooks.txt` fully self-provisions, duplicating core deps | User runs notebooks in their own venv; reverses original exclusion policy |
| 2026-07-31 | `k*=15` frozen by pre-registered plateau rule, unchanged after seeing results | Thresholds fixed pre-fit with a cell-index assertion, preventing post-hoc retuning. Known limitation in `WINDOWS.md`: `STAGE2_K` unevenly spaced, plateau maximal in index space not k space |
| 2026-08-04 | Chart Auto-Encoder (arXiv:1912.10094) tested behind a ratified pre-registration and returned `CAE_VERDICT = FAIL` | T1 geodesic distortion 0.296981 (threshold `<0.15`) and T3 held-out reconstruction margin 3.586350 (threshold `<0.90`) both failed; T2 passed. Thresholds were git-ancestry-proved to predate every fit, so the negative result is publishable as measured. Validated in Phase 02.2 |
| 2026-08-04 | On the 02.2 FAIL the user chose to iterate, not adopt 02.1's Krein representation or stop | Phase 3 now depends on a proposed Phase 02.3 reaching PASS from a freshly re-registered protocol. Per D-02 no automatic Krein fallback fired; the 02.2 verdict is sealed and any changed constant requires a new pre-registration plus a full re-run |

## Evolution

Updated at phase transitions (`/gsd-transition`) and milestone boundaries (`/gsd-complete-milestone`): move requirements between Validated/Active/Out of Scope, log decisions, refresh Context.

---
*Last updated: 2026-08-04 after completing Phase 02.2: Chart Autoencoder Validity Test (`CAE_VERDICT = FAIL`; user elected to iterate)*
</content>
