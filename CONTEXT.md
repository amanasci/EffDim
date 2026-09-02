# Curvature–probe experiments — context

This is the onboarding document for the **`curvature-experiments`** branch of EffDim. It is the map of the **curvature / photometric-decoding** research line: NeurReps 2026 extended abstracts, the ML4PS 2026 research-track paper, frozen experiment packages, and copied audit tables.

Shorter indexes: [`experiments/curvature/README.md`](experiments/curvature/README.md), [`paper/curvature_neurreps/README.md`](paper/curvature_neurreps/README.md).

**Science host** (canonical large artifacts): `angus@100.97.36.119`, repo `~/platonic-universe`. Laptop copies of *reports* live under `paper/curvature_neurreps/audit_outputs/`. Embeddings, neighbour graphs, and `.npz` / `.parquet` geometry caches are **gitignored** and are not on GitHub.

**Unrelated local worktrees** (`experiments/SAE-shared-basis/`, bipartite-matching, topology, scaling-paper figures under `paper/figures/`, …) are **not** part of this paper line. Do not commit them onto this branch.

---

## Start here (15 minutes)

1. Read **this file**.
2. Skim [`submissions/ml4ps_2026/main.tex`](submissions/ml4ps_2026/main.tex) (current scientific framing) and [`submissions/ml4ps_2026/CLAIM_SOURCE_MAP.md`](submissions/ml4ps_2026/CLAIM_SOURCE_MAP.md) (every number → frozen field).
3. If you need the older NeurReps claim gate: [`submissions/neurreps_2026/CLAIMS.md`](submissions/neurreps_2026/CLAIMS.md).
4. Compile check: `cd submissions/ml4ps_2026 && latexmk -pdf main.tex`.

Do **not** treat conversational notes as source of truth. If a frozen JSON/CSV exists, use it.

---

## Two manuscripts (do not overwrite either)

| Venue | Path | Limit | Framing |
|-------|------|-------|---------|
| NeurReps 2026 extended abstract | [`submissions/neurreps_2026/`](submissions/neurreps_2026/) | 4 pages + refs + appendix | Rank- and scale-conditioned curvature ↔ **local linear** decodability. Decision: `claim_supported_but_scale_dependent`. |
| NeurReps + LPA appendix | [`submissions/neurreps_2026_lpa_revision/`](submissions/neurreps_2026_lpa_revision/) | same + exploratory appendix | Adds local-probe-adaptation as **exploratory**, not confirmatory. |
| **ML4PS 2026 Research track** | [`submissions/ml4ps_2026/`](submissions/ml4ps_2026/) | **4 pages, no appendix**, NeurIPS 2026 template | How photometric information is **geometrically organized**; quadratic chart structure; Hessian–curvature alignment as an **anisotropic prior**. |

The NeurReps trees are **frozen**. New work goes in additive packages and in `submissions/ml4ps_2026/`.

ML4PS footer (exact):

`Submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026). Do not distribute.`

Do not call ML4PS an official NeurIPS workshop in the paper. Double-blind: no author names, paths, GitHub, or acknowledgements in the PDF.

**ML4PS title:** *How Representation Curvature Organizes Photometric Decoding in Astronomical Foundation Models*

The target is **apparent \(r\)-band magnitude** (an astronomical photometric observable), not a fundamental physical property. Do not title the work “physical-property decoding.”

---

## The scientific question

> How is scientifically meaningful information geometrically organized inside astronomical foundation-model representations, and when does its local decoding require second-order structure?

Three linked measurements, all at frozen \(d=16\), \(k=2048\), \(n=512\) hash-stable ViT-B anchors:

1. **Sphere-normal extrinsic bending** \(K_H^{\mathrm{cross}}\) (not intrinsic Riemannian curvature).
2. **Held-out quadratic label structure** in local chart coordinates (tangent-linear \(L\) vs unrestricted quadratic \(UQ\)).
3. **Alignment** of the label Hessian with high-energy sphere-normal bending modes of \(B^S\).

NeurReps asked a narrower question (does curvature co-vary with local linear-probe \(R^2\)?). That result is still true and is §3.1 of the ML4PS paper. It is no longer the whole paper.

---

## Target identity (read before touching any table)

| Name | Field / quantity | Use |
|------|------------------|-----|
| Primary **decodability** outcome | `mag_r_desi_local_oof_r2` | Local OOF \(R^2\) of a **global** five-fold ridge probe (\(\alpha=100\)) of apparent \(r\)-band magnitude |
| Direct error | local OOF MSE / SSE of the same probe | Rules out an \(R^2\)-denominator story |
| Catalog magnitude | `mag_r_desi_catalog_value` | **Never** a substitute for the probe outcome |
| QLCA label \(y\) | catalog `mag_r_desi` in the neighbourhood | The scalar being decoded **in chart coordinates** (different from local OOF \(R^2\)) |

**Proven join:** physics `vit_base_test_labels.npz` is row-aligned to the galaxies test parquet; `sample_id` is that row index; `selection.npz` indexes both. Equal row count is **not** the proof.

**Unproven / excluded:** DESI spectroscopic `spec_z`, DESI imaging `mag_r` on the DESI embedding table (`desi_label_alignment_unresolved`). Do not resurrect those associations.

**Historical bug:** an adaptive run correlated \(K_H\) with **catalog magnitude** instead of local OOF \(R^2\) (Spearman \(\approx -0.215\) between those two \(y\) vectors) and looked like a sign flip. Audit label: `probe_label_alignment_failure`. Typed targets exist to stop this.

Controls for Spearman associations: `log_knn_radius`, `local_label_variance`, `local_evaluation_count`.

---

## Frozen numbers (do not round from memory)

Authoritative copies: `paper/curvature_neurreps/audit_outputs/` (and host `~/platonic-universe/outputs/geometry/…`).

### Global linear decoding (confirmatory)

From `submission_validation/metric_associations.csv` / `decision.json` and QLCA `parity.json`:

| Estimand | Value | Rounding in papers |
|----------|-------|--------------------|
| \(\rho_{\mathrm{ctl}}(K_H^{\mathrm{cross}}, R_G^2)\) at \(d=16\) | −0.240484 | −0.240 |
| \(\rho_{\mathrm{ctl}}(K_H^{\mathrm{cross}}, \mathrm{MSE}_G)\) | +0.227048 | +0.227 |
| \(\rho_{\mathrm{ctl}}(K_H, \mathrm{SST})\) | −0.024557 | −0.025 |
| same \(k\), \(d=12\) / \(d=20\) \(R^2\) | +0.142990 / −0.233325 | +0.143 / −0.233 |

Result is **rank- and bandwidth-conditioned**. Not scale-invariant. \(k=512\) fails curvature reliability \(R_H\).

### Held-out quadratic structure (QLCA primary)

From `quadratic_label_chart_alignment/primary_inference.json`:

| Estimand | Value |
|----------|-------|
| median \(\Delta_Q=\mathrm{MSE}_L-\mathrm{MSE}_{UQ}\) | 0.020582 |
| bootstrap 95% CI | [0.019673, 0.021616] |
| fraction of 512 anchors with \(\Delta_Q>0\) | 1.0 |
| \(p_{\mathrm{MC}}\) (median \(\Delta_Q\), \(B=2000\)) | \(1/2001\) |
| \(\rho_{\mathrm{ctl}}(K_H^{\mathrm{cross}},\Delta_Q)\) | 0.111249 |
| \(p_{\mathrm{MC}}\) (Freedman–Lane, \(B=10^4\)) | 0.007499 |
| Holm both pass | true |

Mechanical workflow label `quadratic_chart_link_unresolved` is **not** a scientific result; it flagged a bad synthetic gate. Do not put it in a paper.

### Geometry-regularized quadratic decoding (QLCA audit)

From `quadratic_label_chart_alignment_audit/`:

- \(q=136\) packed Hessian coefficients at \(d=16\).
- \(B^S_{\mathrm{flat}}\) algebraic rank **136 at every anchor**.
- Median energy ranks: \(r_{90}=71\), \(r_{95}=90\), \(r_{99}=119\).
- Original BS map used **48** modes because of a **hard cap** (`implementation_cap_below_energy_rank`), not because 48 modes hold nearly all bending energy.
- Full-rank ridge on \(c\) \(\equiv\) UQ with penalty \(\gamma^\top(B^{S\top}B^S)^+\gamma\).
- **Interpretation:** `geometry_regularized_quadratic_decoding` — geometry is an **anisotropic prior**, not a low-dimensional function class.
- Fraction of UQ gain: **median of per-anchor ratios** \(\Delta_{B^S}/\Delta_Q\) (not a ratio of medians): cap-48 \(\approx 0.938\) (94%); \(r_{90}\approx 98\%\); \(r_{95}\approx 99\%\); \(r_{99}\approx 100\%\).

### Hessian–curvature alignment

From `alignment_summary.json` + audit `alignment_tests.json`:

- Observed \(A_B\) median **2.427**.
- Haar / isotropic / matched-anchor nulls \(\approx 0.986\) (\(p_{\mathrm{MC}}<1/2001\), \(B=2000\)).
- Orientation-sensitive (Haar preserves the singular spectrum).
- Foldwise Hessian cosine \(\approx 0.924\); all 512 stable; split-half Spearman \(\approx 0.819\).

### Null calibration

Use **192 real-design nested-CV shuffles**: shuffled \(\Delta_Q\approx -0.00038\), false-positive safe, well calibrated (`shuffle_cause.json` `real_nested_battery`).

Do **not** treat the original fixed-\(\alpha_Q=100\) synthetic (\(\Delta_Q\approx -7.56\)) as evidence against the nested estimator.

### Local probe adaptation (secondary, not mediation)

- \(\rho_{\mathrm{ctl}}(K_H,\Delta\mathrm{MSE}_{G\to P})\approx +0.153\).
- Conditioning on \(\Delta_Q\) **raises** it to \(\approx 0.205\) (QLCA `secondary_inference.json`).
- Patch probes are **worse on average**. Relative adaptation \(\neq\) uniformly better local probes. Not a causal mediation analysis.

### Multi-label screen (secondary; not in the ML4PS body)

Package: `experiments/geometry/physics_multilabel_chart_screen/`.  
Reports: `paper/curvature_neurreps/audit_outputs/multilabel_chart_screen/` (global OOF) and `…/multilabel_chart_screen_quadratic/`.

Eligible physics-table labels with proven `sample_id` join and frozen OOF probes: `mag_r_desi`, `photo_z`, `smooth_fraction`, `stellar_mass`. Excluded: `sfr` (underpowered), DESI fields.

At the same frozen charts (\(n=512\)):

| Label | \(\rho_{\mathrm{ctl}}(K_H,R_G^2)\) | median \(\Delta_Q\) | \(\rho_{\mathrm{ctl}}(K_H,\Delta_Q)\) | \(A_B\) |
|-------|-----------------------------------:|--------------------:|--------------------------------------:|--------:|
| \(r\)-band magnitude | **−0.240** | **+0.021** (all 512 > 0) | **+0.111** | **2.43** |
| photometric redshift | −0.047 (n.s.) | +0.0003 | **−0.111** | 2.03 |
| smooth fraction | −0.007 (n.s.) | +0.0018 | +0.081 (\(p\approx0.048\)) | 1.96 |
| stellar-mass proxy | −0.231 | **−0.115** (21% positive) | +0.127 | 1.13 |

The quadratic, curvature-aligned mechanism is **label-specific**. Stellar mass tracks **global linear error** somewhat like magnitude (and is likely entangled with brightness) but unrestricted quadratics **overfit**. Do not drop this into the four-page ML4PS paper without a deliberate rewrite.

---

## Geometry (minimum needed)

Local tangent coordinates \(u\) after removing the unit-sphere radial component. Nested PCA frame \(J\) of rank \(d\). Chart

\[
f(u)=x_0+Ju+\tfrac12 Q(u,u).
\]

Remove tangential warp and forced radial curvature → sphere-normal tensor \(B^S\). Mean-curvature vector \(H^S=d^{-1}\sum_a B^S_{aa}\). Production statistic \(K_H^{\mathrm{cross}}=\langle H^{(A)},H^{(B)}\rangle\) (split halves). The cross product avoids the positive noise bias of \(\|\widehat H\|^2\).

Call it **sphere-normal extrinsic bending / curvature**. Never “intrinsic manifold curvature.”

**Label Hessian:** matrix of second-order variation of the photometric target in chart coordinates \(\Gamma\) in \(\hat y_{UQ}(u)=a_0+a_1^\top u+\frac12 u^\top\Gamma u\).

---

## Repository map

```
CONTEXT.md                          ← this file
experiments/curvature/              ← short README + paper_working notes
experiments/geometry/
  physics_activation_atlas/         ← shared chart / B^S / probe code
  physics_nested_dimension_curvature/
  physics_quadratic_predictive_dimension/
  physics_curvature_probe_rank_sweep/
  physics_curvature_probe_submission_validation/
  physics_adaptive_dataset_curvature_probe(_audit)/
  physics_curvature_scale_bias_variance/
  physics_local_probe_adaptation(_audit)/
  physics_quadratic_label_chart_alignment/         ← QLCA (do not overwrite outputs)
  physics_quadratic_label_chart_alignment_audit/
  physics_multilabel_chart_screen/
  run_*.py                          ← CLIs
submissions/neurreps_2026/          ← frozen NeurReps abstract
submissions/neurreps_2026_lpa_revision/
submissions/ml4ps_2026/             ← ML4PS sources + compiled PDF
paper/curvature_neurreps/           ← notes, copied audit JSON/CSV/MD, draft TeX
outputs/geometry/                   ← host run trees (reports may be copied; caches gitignored)
```

### ML4PS submission tree

| File | Role |
|------|------|
| `main.tex` | Double-blind NeurIPS 2026 article; footer override only |
| `neurips_2026.sty` | Unmodified official style |
| `references.bib` | Five verified entries (same as NeurReps) |
| `main.pdf` | Compiled submission |
| `figures/fig1_global.pdf` | Confirmatory curvature–error (from NeurReps Fig. 2) |
| `figures/fig2_quadratic.pdf` | QLCA / audit 2×2; rebuilt by `figures/make_fig2.py` |
| `CLAIM_SOURCE_MAP.md` | Number → artifact |
| `CHANGELOG_FROM_EXISTING_MANUSCRIPT.md` | NeurReps → ML4PS diffs |
| `CITATION_AUDIT.md` / `ANONYMIZATION_AUDIT.md` / `BUILD_REPORT.md` | Venue gates |

Compile: `cd submissions/ml4ps_2026 && python3 figures/make_fig2.py && latexmk -pdf -interaction=nonstopmode main.tex`

`make_fig2.py` reads frozen CSV/JSON under `paper/curvature_neurreps/audit_outputs/` (not the gitignored host parquet).

### Experiment CLIs

```bash
export PYTHONPATH=experiments

# NeurReps claim gate (needs host outputs/geometry caches)
python experiments/geometry/run_curvature_probe_submission_validation.py --help

# QLCA (frozen charts; do not write into preserved output dirs)
python experiments/geometry/run_quadratic_label_chart_alignment.py --smoke
python experiments/geometry/run_quadratic_label_chart_alignment_audit.py --smoke

# Multi-label screen
python experiments/geometry/physics_multilabel_chart_screen/test_multilabel_chart_screen.py
python experiments/geometry/run_multilabel_chart_screen.py --smoke --skip-quadratic
```

QLCA `PRESERVED` paths include NeurReps submissions and the original QLCA output tree. New runs must use a **new** `--output-dir`.

---

## Artifact dependency graph

```
physics_multimodel_graph_prior_quadratic
  X, folds, kNN, global OOF probes (mag_r_desi, photo_z, smooth_fraction, stellar_mass)
        ↓
physics_nested_dimension_curvature
  per-anchor charts, H, B^S  (H_vectors/{sid}.npz)
        ↓
physics_quadratic_predictive_dimension     → “dimension as range”
physics_curvature_probe_rank_sweep         → per_anchor_rank_curve (K_H)
        ↓
physics_curvature_probe_submission_validation   ← NeurReps claim gate
physics_local_probe_adaptation (+ audit)
physics_quadratic_label_chart_alignment         ← L / UQ / BS / A_B on mag_r
physics_quadratic_label_chart_alignment_audit   ← rank, Haar, shuffles
physics_multilabel_chart_screen                 ← other eligible labels
```

Host roots:

- `~/platonic-universe/outputs/geometry/physics_curvature_probe_submission_validation/`
- `~/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment/`
- `~/platonic-universe/outputs/geometry/physics_quadratic_label_chart_alignment_audit/`
- `~/platonic-universe/outputs/geometry/physics_multilabel_chart_screen/`
- `~/platonic-universe/outputs/geometry/physics_multilabel_chart_screen_quadratic/`

---

## What we can and cannot claim

### Can (with the stated conditioning)

- At frozen \(d=16\), \(k=2048\), greater \(K_H^{\mathrm{cross}}\) ↔ worse **global** linear decoding of apparent \(r\)-band magnitude (direct MSE, not SST).
- Local labels show held-out quadratic structure in chart coordinates; that gain co-varies with \(K_H\) for **this** target.
- Label Hessians align with high-energy sphere-normal bending; geometry supplies an anisotropic quadratic **prior** (full rank 136).
- Leading 48 bending modes retain ~94% of UQ gain **as a median of per-anchor ratios**, with the cap caveat.

### Cannot

- Intrinsic dimension or intrinsic Riemannian curvature.
- Causality; scale invariance; “BS is a low-dimensional function class.”
- “BS explains 94%” without the full-rank / implementation-cap sentence.
- Patch probes better than global on average; causal mediation via \(\Delta_Q\).
- The quadratic mechanism as a generic property of all scientific labels (multi-label screen).
- DESI spectroscopic / unproven-join labels.
- Independent encoder/dataset replication (future work; already in the ML4PS discussion).

---

## Tests

```bash
export PYTHONPATH=experiments
pytest experiments/geometry/physics_curvature_probe_submission_validation/ -q
pytest experiments/geometry/physics_quadratic_label_chart_alignment/test_qlca.py -q
# audit + multilabel tests are unittest-style or pytest if installed
python experiments/geometry/physics_multilabel_chart_screen/test_multilabel_chart_screen.py
```

These physics experiments validate against **frozen real-data parity**, not Swiss-roll notebooks. Swiss-roll notebooks are required only when adding a new manifold **model** (`CLAUDE.md`).

---

## Suggested reading order

1. This file
2. `submissions/ml4ps_2026/main.tex` + `CLAIM_SOURCE_MAP.md`
3. `submissions/neurreps_2026/CLAIMS.md` (older, narrower claim)
4. `paper/curvature_neurreps/audit_outputs/quadratic_label_chart_alignment/REPORT.md`
5. `paper/curvature_neurreps/audit_outputs/quadratic_label_chart_alignment_audit/INTERPRETATION.md`
6. `experiments/geometry/physics_activation_atlas/` (implementation)

---

## Branch notes

- **Branch:** `curvature-experiments`
- **Remote:** `origin` (`github.com/amanasci/EffDim.git`)
- Large caches stay gitignored (`*.parquet`, `*.npz`). CSV/JSON/MD reports are the portable record.
- When in doubt: reproduce **parity** (\(\rho=-0.240\) / \(+0.227\) at \(d=16\), \(k=2048\)) before changing a layer.
