# EffDim

## What This Is

EffDim is a research-oriented Python library that computes "effective dimensionality" (ED)
of data across modalities. A single entry point, `effdim.compute_dim(data)`, returns a
dict of ED estimates from many established estimators at once — spectral metrics
(participation ratio, stable rank, numerical rank, Shannon/Renyi entropy dimensions,
PCA explained-variance thresholds, cumulative eigenvalue ratio) and geometric/intrinsic
dimension estimators (MLE, TwoNN, DANCo, MiND-MLi, MiND-MLk, ESS, TLE, GMST).

Targets researchers characterizing intrinsic structure of high-dimensional data — notably
neural network representations and embeddings — without stitching together single-method
packages.

## Core Value

One call over an `(n_samples, n_features)` array returns a comparable panel of effective
dimensionality estimates, so a researcher can see how spectral and geometric notions of
dimension agree or disagree on the same data.

## Current Milestone: v1.1 PU Manifold Curvature

**Goal:** Reconstruct the PU embedding manifold via Isomap, build a smooth decoder
parameterization of it, compute mean curvature, and test whether foundation-model
representational alignment (MKNN) varies with local curvature.

**Target features:**
- Stream `UniverseTBD/pu-embeddings` config `legacysurvey_dinov3_vitb16` and subsample
  10k paired rows (columns `dinov3_vitb16_hsc` / `dinov3_vitb16_legacysurvey`)
- Fit Isomap on the `_legacysurvey` embeddings
- Audit the classical-MDS eigenspectrum of the Isomap geodesic matrix: large positive
  eigenvalues with a steep dropoff, and explicit detection of large negative eigenvalues
- Train a torch MLP decoder mapping Isomap coordinates back to the 768-d embedding,
  using a C2-smooth activation
- Derive mean curvature H from the decoder via first and second fundamental forms,
  using `torch.func` Jacobian/Hessian
- Partition the manifold into high- and low-curvature regions by |H| quantiles
- Compute crossmodal MKNN (HSC vs Legacy Survey) per region against a permutation
  null, with bootstrap confidence intervals

## Requirements

### Validated

- `compute_dim` returns spectral + geometric ED estimates for a 2D numpy array
- Input validation: 2D only, >=2 samples, finite values, list-of-arrays vstacked, auto-centering
- SVD path switches to randomized SVD when `min(n, d) >= 1000`
- k-NN distances computed once (k=10) and shared across geometric estimators
- Benchmark suite for runtime and accuracy vs vector size / count
- Reproducible, row-aligned 10k subsample of `legacysurvey_dinov3_vitb16`, cached and
  L2-normalized (DATA-01..05) — *Validated in Phase 1: Data Loading & Manifold Reconstruction*
- Isomap fit validated for connectivity and `n_neighbors` stability; `k*=15` frozen by a
  pre-registered plateau rule (ISO-01..05) — *Validated in Phase 1*

### Active

- [ ] Isomap manifold reconstruction of PU foundation-model embeddings, with MDS
      eigenspectrum validity audit
- [ ] Smooth decoder parameterization of the reconstructed manifold
- [ ] Mean curvature field derived analytically from the decoder
- [ ] Regional MKNN alignment compared across high- vs low-curvature regions

### Deferred

- [ ] Validate estimates against known dimensionalities (noise -> D, Swiss Roll -> intrinsic)
- [ ] CI across platforms, including any compiled extension
- [ ] Intramodal MKNN across a model-size ladder (the paper's stronger 28-56% signal);
      requires a second model size, deferred out of v1.1

### Out of Scope

- Full deep-learning framework dependency in the core package — torch stays out of
  `pyproject.toml` core deps; analysis notebooks may install it themselves
- Bundling large datasets in the repo — notebooks stream from source
- Modifying `src/effdim/` during v1.1 — notebook-only milestone; promoting the curvature
  operator into the library is a later decision

## Context

- Package layout: `src/effdim/` with `api.py` (orchestration + validation + SVD),
  `metrics.py` (spectral estimators), `geometry.py` (intrinsic dimension estimators).
- Core deps are deliberately light: numpy, scipy, scikit-learn, faiss-cpu.
- `benchmarks/`, `tests/`, `docs/` (mkdocs) exist.
- `notebooks/` holds `02_k_sensitivity_refit.ipynb` (the k-sensitivity gate re-fit,
  standalone) and the `pu_manifold/` support package: `cache.py` + `subsample.py`
  implemented, `curvature.py` + `mknn.py` stubbed for Phases 3-4.
- Phase 1's canonical artifacts live in the gitignored `notebooks/.cache/`:
  `isomap_43cf438bc944c509.joblib` (~1.55 GiB, the single k*=15 fit carrying
  `dist_matrix_`) and `phase1_handoff_43cf438bc944c509.json` (the 14-key Phase 1→2
  interface). Frozen: `k*=15`, `n_components=18`, `d_provisional=18`.
- `notebooks/requirements-notebooks.txt` deliberately duplicates and pins the core deps
  `pyproject.toml` also declares — notebooks are provisioned into a user-supplied venv.
- `TODO.md` tracks testing/CI hardening as the standing next work.

### PU embeddings dataset (v1.1 subject)

`UniverseTBD/pu-embeddings` — foundation-model **image** embeddings for astronomy surveys.
163 configs, 7,050,003 rows, ~93 GB. Never materialize whole; stream one config.

| config family | rows | columns |
|---|---|---|
| `jwst_*` | 1,496 | `<model>_hsc`, `<model>_jwst` |
| `desi_*` | 20,465 | `<model>_hsc`, `<model>_desi` |
| `legacysurvey_*` | 101,725 | `<model>_hsc`, `<model>_legacysurvey` |
| `physics_*_test` | 86,471 | `<model>_galaxies` (unpaired) |

No labels, no `object_id` anywhere in the dataset. Paired configs are **row-aligned** —
the only join, and all the MKNN metric needs.

### Origin experiment

Duraphe, Smith, Sourav & Wu, *The Platonic Universe: Do Foundation Models See the Same
Sky?*, NeurIPS 2025 ML4PS workshop ([arXiv:2509.19453](https://arxiv.org/abs/2509.19453)).
Tests the Platonic Representation Hypothesis in astronomy via the mutual k-nearest-neighbour
score (Chechik et al. 2010):

    MKNN(z1, z2) = k^-1 * |N_k(z1) ∩ N_k(z2)|

Label-free and training-free — compares the k-NN sets of two embeddings of the *same*
object, against a random-permutation null. Reported alignment: intramodal (two sizes, same
architecture) 28-56%; crossmodal (HSC vs other modality) ~7% for JWST, 0.4-2% for Legacy
Survey, 0.3-0.5% for DESI. Alignment rises with model capacity — 14/18 intramodal and
28/33 crossmodal comparisons.

v1.1 asks a question the paper does not: is that convergence uniform across the
manifold, or concentrated where the manifold is flat?

## Constraints

- Python >= 3.8
- Core package must stay installable without GPU or deep-learning stack
- Notebook-only dependencies (torch, datasets, umap) must be installed from within the
  notebook, never added to core `dependencies`

## Key Decisions

| Date | Decision | Why |
|------|----------|-----|
| 2026-07-27 | `.planning/` bootstrapped retroactively from existing repo state | GSD quick-task workflow requires ROADMAP.md; project predates GSD adoption |
| 2026-07-27 | Notebook deps installed in-notebook, not in `pyproject.toml` | Keeps core package light per Out of Scope |
| 2026-07-29 | v1.1 uses `legacysurvey_dinov3_vitb16`, 10k subsample | Paired HSC + Legacy columns are row-aligned so MKNN works; 101,725 rows give the best manifold coverage of any paired config; 10k keeps the dense geodesic matrix at ~800 MB so Isomap stays exact rather than landmark-approximated |
| 2026-07-29 | Physics probe is MKNN, not a supervised head | The origin paper (arXiv:2509.19453) probes alignment label-free; pu-embeddings ships no labels and no join key, so a supervised probe was never available |
| 2026-07-29 | Single model in v1.1, no size ladder | Establishes the curvature method before multiplying compute. Accepted cost: only crossmodal MKNN is computable, and on Legacy that is the paper's weakest signal (0.4-2%), so a null regional result is plausible and must be reportable |
| 2026-07-29 | Decoder is a torch MLP with C2-smooth activation | Mean curvature needs a nonzero second derivative; ReLU's is identically zero. `torch.func.jacrev`/`hessian` give the fundamental forms directly |
| 2026-07-29 | Milestone is notebook-only | ROADMAP Phase 3 framed this as applied analysis; promoting the curvature operator into `src/effdim/` would need unit tests against known-curvature surfaces, its own milestone |
| 2026-07-30 | `subsample_*.npz` caches normalized arrays + raw norms only, never the raw 768-d vectors (D-05/D-06) | Removes any way for a later phase to silently mix normalized and raw embeddings. Accepted one-way cost: recovering raw vectors means re-streaming the 553 MiB parquet |
| 2026-07-30 | `requirements-notebooks.txt` fully self-provisions, duplicating core `pyproject.toml` deps | User runs the notebooks in their own pre-existing venv; a partial requirements file left that venv underprovisioned. Reverses the original deliberate-exclusion policy |
| 2026-07-31 | `k*=15` frozen by the pre-registered plateau rule, unchanged after seeing results | The sweep thresholds were fixed in a cell that executes before the first fit, with a cell-index assertion. Retuning post-hoc is the garden-of-forking-paths failure the design exists to prevent. Known limitation logged in `WINDOWS.md`: `STAGE2_K` is unevenly spaced, so the plateau is maximal in index space, not k space |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-31 after completing Phase 1: Data Loading & Manifold Reconstruction*
