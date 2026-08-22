# Curvature–probe experiments — context and getting started

This document orients you to the **`curvature-experiments`** branch: a self-contained research line asking whether **local geometry of a ViT-B galaxy embedding** is associated with **how well a linear probe decodes physical information** at each patch.

Shorter index: [`experiments/curvature/README.md`](experiments/curvature/README.md).

---

## The question and the current answer

**Question.** After fitting a regularized local quadratic chart in a fixed-$k$ neighbourhood on the unit sphere, does the chart’s sphere-normal mean-curvature statistic co-vary with global-probe decodability evaluated locally?

**Answer (audited).** **Yes, but only at a particular finite neighbourhood scale and across an evaluated chart-rank range.** The relationship is **rank- and bandwidth-conditioned**, not a universal scalar property of the representation.

**Paper decision label:** `claim_supported_but_scale_dependent`

More precise wording:

> The association is supported at the frozen broad-support scale ($k{=}2048$, $n{=}512$ anchors) but is conditioned on chart rank and neighbourhood bandwidth.

---

## Core estimands (read this before touching data)

| Symbol / name | Meaning |
|---------------|---------|
| $K_H^{\mathrm{cross}}$ | Split-half inner product of sphere-normal mean-curvature vectors $H^{(A)}, H^{(B)}$ from a nested quadratic chart |
| $R^2_{\mathrm{local}}$ | Global ridge probe’s **local OOF** $R^2$: `mag_r_desi_local_oof_r2` |
| Catalog magnitude | `mag_r_desi_catalog_value` — **not** the probe outcome |
| Controls | `log_knn_radius`, `local_label_variance`, `local_evaluation_count` |
| Frozen geometry | $k{=}2048$ neighbours, 512 hash-stable anchors, ViT-B, five-fold OOF global probe ($\alpha{=}100$) |

**Critical pitfall.** An adaptive analysis once *appeared* to flip the curvature sign because the outcome was silently changed from **local OOF probe $R^2$** to **raw catalog magnitude** (Spearman $\approx -0.215$ between those vectors). Typed target definitions and the adaptive audit now prevent that substitution. Always check the target column name.

**What $K_H$ is.** A **finite-scale extrinsic** statistic of a fitted activation chart (cross-split mean-curvature energy), **not** intrinsic Riemannian curvature.

---

## Repository layout

```
experiments/
  curvature/           ← start here (README + paper_working notes)
  geometry/            ← all experiment packages and run_*.py CLIs
    physics_activation_atlas/     shared chart / curvature / probe code
    physics_*                     numbered experiment packages
  alignment/           ← global probe direction vs curvature subspace ($A_H$, $A_B$)

submissions/neurreps_2026/   ← NeurRePS extended abstract (4-page main body)
paper/                       ← earlier LaTeX drafts
outputs/geometry/            ← run artifacts (reports in git; caches gitignored)

src/effdim/curvature.py      ← bootstrap PCA tangent helper (not the paper pipeline)
```

**Not on this branch.** `experiments/SAE-shared-basis/`, bipartite-matching, topology, etc. are unrelated worktrees left untracked.

**Large files.** Embeddings, neighbour indices, geometry caches (`.parquet`, `.npz`) are **gitignored**. They live on the science host under `~/platonic-universe/outputs/geometry/`.

---

## Environment and first commands

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # or project’s usual install

export PYTHONPATH=experiments
```

Smoke-test imports:

```bash
pytest experiments/geometry/physics_curvature_probe_submission_validation/ -q
python experiments/geometry/run_curvature_probe_submission_validation.py --help
```

**Science host** (full GPU runs, canonical artifacts): `angus@100.97.36.119`, repo at `~/platonic-universe`. Sync code from a laptop worktree with `rsync` or `tar | ssh`.

---

## Artifact dependency graph

Experiments are layered. Lower stages produce frozen inputs for higher stages.

```
physics_multimodel_graph_prior_quadratic
  embeddings, folds, kNN neighbourhoods, global OOF probe preds
        ↓
physics_nested_dimension_curvature
  H vectors, K_H at nested ranks
        ↓
physics_quadratic_predictive_dimension
  held-out linear/quadratic risk → dimension-as-range argument
        ↓
physics_curvature_probe_rank_sweep
  frozen per-anchor K_H_cross + local probe table (PRIMARY TABLE)
        ↓
physics_curvature_probe_submission_validation  ← main claim gate
physics_adaptive_dataset_curvature_probe (+ audit)
physics_curvature_scale_bias_variance
physics_local_probe_adaptation (+ audit)
```

If you only reproduce **one** number, use **submission validation** against the rank-sweep table. Everything else is audit, scale, or exploratory follow-up.

---

## Experiments by importance

### Tier 1 — defines the paper claim

| Package | CLI | What it does |
|---------|-----|--------------|
| `physics_curvature_probe_rank_sweep` | `run_curvature_probe_rank_sweep.py` | Frozen $K_H$–$R^2_{\mathrm{local}}$ curve at $d\in\{12,16,20\}$; produces `per_anchor_rank_curve.parquet` |
| `physics_curvature_probe_submission_validation` | `run_curvature_probe_submission_validation.py` | Parity, permutation, bootstrap, direct-error vs SST, scale slice, label shuffle → `decision.json` |
| `physics_quadratic_predictive_dimension` | `run_quadratic_predictive_dimension.py` | Held-out linear/quadratic risk across ranks → “dimension as range” figure |
| `physics_nested_dimension_curvature` | `run_nested_dimension_curvature.py` | Quadratic chart fit, $H$ vectors, $K_H^{\mathrm{cross}}$ |

**Key frozen associations** ($k{=}2048$, $n{=}512$, controlled Spearman):

| Chart rank $d$ | $\rho(K_H, R^2_{\mathrm{local}})$ |
|----------------|-----------------------------------|
| 12 | +0.143 |
| 16 | **−0.240** |
| 20 | −0.233 |

At $d{=}16$: OOF MSE $\rho{=}+0.227$, SST/local variance $\rho{\approx}-0.025$ → **not** an $R^2$-denominator artefact.

### Tier 2 — audits and confound checks

| Package | Why it matters |
|---------|----------------|
| `physics_adaptive_dataset_curvature_probe_audit` | Proves geometry was identical; failure was **wrong target** (catalog vs OOF $R^2$) |
| `physics_adaptive_dataset_curvature_probe` | Typed targets, adaptive protocol |
| `physics_activation_atlas/` | Shared implementation: quadratic chart, sphere-normal $B^S$, probes, inference helpers |
| `physics_global_probe_curvature_alignment` (runner) | $A_H^G$, $A_B^G$ alignment of global probe with curvature directions |
| `run_full_curvature_audit.py`, `run_split_half_curvature_reliability.py` | Reliability and leakage checks |

### Tier 3 — neighbourhood scale

| Package | CLI | Result |
|---------|-----|--------|
| `physics_curvature_scale_bias_variance` | `run_curvature_scale_bias_variance.py` | Factorial over support radius $R$ and fit count $m$; label `mechanism_unresolved` |

Scale snapshot ($d{=}16$, controlled $\rho$, 128 hash anchors):

| $k$ | $\rho$ |
|-----|--------|
| 1024 | −0.027 |
| 1536 | −0.080 |
| 2048 | −0.171 (128 anc.) / **−0.240** (512 anc.) |

$k{=}512$ curvature estimator fails $R_H$ reliability — not interpretable.

### Tier 4 — exploratory (post hoc)

| Package | CLI | Status |
|---------|-----|--------|
| `physics_local_probe_adaptation` | `run_local_probe_adaptation.py` | $\rho_{\mathrm{ctl}}(K_H, \Delta\mathrm{MSE}_{G\to P}){=}+0.153$; mean $\Delta\mathrm{MSE}{\approx}-0.10$ |
| `physics_local_probe_adaptation_audit` | `run_local_probe_adaptation_audit.py` | Bounded audit: paired $\Delta\rho$, $A_H/A_B$ controls, pathway, label shuffle |

Interpretation: **relative adaptation** in high-curvature patches (smaller penalty from local fitting), **not** uniformly better patch probes. Direction-rotation mechanism **not** established. Final audit may still be running on the science host.

### Supporting / historical (not main-text drivers)

- `physics_stable_tangent_dimension`, `physics_order_stratified_geometry`, `physics_implicit_normal_inverse`
- `run_curvature_probe_screen.py` — early screen; one in-sample patch result was **invalid** (accidental positive)
- `run_cross_model_probe_curvature_coverage.py` — cross-model inventory
- `experiments/alignment/` — unpaired universal geometry (separate scaling paper thread in `paper_working/`)

---

## How curvature is computed (method sketch)

1. **Fixed-$k$ neighbours** on unit-normalized ViT embeddings (not radius-defined — avoids density confounding).
2. **Nested PCA** chart $f(u) = x_0 + Ju + \frac{1}{2}Q(u,u)$.
3. Decompose $Q$ into tangential nonlinearity, forced sphere-radial part, and sphere-normal second fundamental form $B^S$.
4. Mean-curvature vector $H^S_a = d^{-1}\sum_i B^S_{a,ii}$ per split half → $K_H^{\mathrm{cross}} = \langle H^{(A)}, H^{(B)}\rangle$.

Early radial-only proxies were **trivial** (sphere confound). The production statistic removes that forced radial component.

**Dimensionality.** Do **not** claim $d{=}12$ as intrinsic dimension. Linear risk keeps improving through $d{=}20$; quadratic risk plateaus around $d{=}18$–$19$. The paper evaluates $d{=}12$–$20$ as a **predictive range** (~80–85% held-out variance), not an eigengap.

---

## Typical workflows

### Reproduce the submission decision (read-only, fast)

```bash
export PYTHONPATH=experiments
python experiments/geometry/run_curvature_probe_submission_validation.py \
  --output-dir outputs/geometry/physics_curvature_probe_submission_validation
cat outputs/geometry/physics_curvature_probe_submission_validation/decision.json
```

Requires upstream artifacts in `outputs/geometry/` (rank sweep, multimodel pack, nested curvature).

### Full pipeline from embeddings (slow, GPU host)

Run in order on the science host:

```bash
python experiments/geometry/run_physics_multimodel_graph_prior_quadratic.py ...
python experiments/geometry/run_nested_dimension_curvature.py ...
python experiments/geometry/run_quadratic_predictive_dimension.py ...
python experiments/geometry/run_curvature_probe_rank_sweep.py ...
python experiments/geometry/run_curvature_probe_submission_validation.py ...
```

Each runner supports `--help` and usually `--smoke` for CPU sanity checks.

### Local probe adaptation audit

```bash
pytest experiments/geometry/physics_local_probe_adaptation_audit/test_audit.py -q
python experiments/geometry/run_local_probe_adaptation_audit.py --smoke --skip-shuffle
python experiments/geometry/run_local_probe_adaptation_audit.py   # full: ~hours (shuffle)
```

---

## Paper and claims

| Path | Contents |
|------|----------|
| `submissions/neurreps_2026/` | Camera-ready target; `CLAIMS.md`, `main.tex`, `VALIDATION_REPORT.md` |
| `experiments/curvature/paper_working/` | Claim hierarchy, provenance, figure inventory, audit markdown |
| `submissions/neurreps_2026/CLAIMS.md` | Permitted vs forbidden claims |

### We can claim

- Robust association at frozen $k{=}2048$ scale
- Direct correspondence with OOF probe **error** (not SST)
- Sign transition across evaluated chart ranks ($d{=}12$ vs $d{=}16$–$20$)
- Bandwidth dependence of effect magnitude
- Provisional relative local adaptation (with negative mean patch advantage disclosed)

### We cannot claim

- Unique intrinsic dimension; intrinsic manifold curvature
- Causality; scale invariance
- Patch probes outperforming global on average
- Proven probe-direction rotation because of curvature
- Independent dataset/encoder replication
- Valid DESI spectroscopic label results (identity alignment unproven)

---

## Outputs and git policy

| Committed | Gitignored (regenerate on host) |
|-----------|----------------------------------|
| `*.json`, `*.md`, `*.csv` reports | `*.parquet`, `*.npz`, `checkpoints/` |
| NeurRePS submission PDF sources | Full embedding matrices, geometry caches |
| Summary figures in `paper_working/` | `outputs/geometry/physics_multimodel_graph_prior_quadratic/` bulk |

After a run, check `outputs/geometry/<package>/decision.json`, `COMPLETE.json`, and `REPORT.md` / `AUDIT_REPORT.md` where present.

---

## Tests

Package-level tests live beside each experiment:

```bash
export PYTHONPATH=experiments
pytest experiments/geometry/physics_curvature_probe_submission_validation/ -q
pytest experiments/geometry/physics_local_probe_adaptation/ -q
pytest experiments/geometry/physics_curvature_scale_bias_variance/ -q
pytest experiments/geometry/physics_local_probe_adaptation_audit/ -q
```

Swiss-roll sanity checks for new **manifold models** belong in `notebooks/` per `CLAUDE.md`; these physics experiments validate against frozen real-data parity instead.

---

## Suggested reading order

1. This file (`CONTEXT.md`)
2. `submissions/neurreps_2026/CLAIMS.md`
3. `experiments/curvature/paper_working/claim_hierarchy.md`
4. `outputs/geometry/physics_curvature_probe_submission_validation/decision.json` (if present)
5. `experiments/geometry/physics_activation_atlas/` — implementation details
6. Package `README.md` / `REPORT.md` inside the specific experiment you are extending

---

## Branch notes

- **Branch:** `curvature-experiments` (from `isomap-curvature` @ `7b2401e`)
- **Commit:** `feat(curvature): add audited curvature–probe experiment stack`
- **Unrelated work** remains untracked locally (SAE-shared-basis, bipartite-matching, …)

When in doubt: **reproduce parity first**, then change one layer at a time. The frozen $d{=}16$, $k{=}2048$, $\rho{=}-0.240$ controlled association is the numerical anchor everything else is checked against.
