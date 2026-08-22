# Curvature–probe experiments (ViT-B galaxy embeddings)

Paper question: **does local geometry of a vision representation relate to how easily physical information is extracted with a linear probe?**

Answer (current): **yes, at a finite neighbourhood scale and across an evaluated chart-rank range**—but the relationship is rank- and bandwidth-conditioned, not a universal scalar property.

**Decision label:** `claim_supported_but_scale_dependent`

## Layout

| Path | Role |
|------|------|
| `../geometry/` | Experiment packages, runners, and shared atlas code |
| `../alignment/` | Global probe / curvature alignment controls ($A_H$, $A_B$) |
| `paper_working/` | Claim audits, inventories, figure scripts |
| `../../submissions/neurreps_2026/` | NeurRePS extended abstract package |
| `../../outputs/geometry/` | Run artifacts (reports committed; large caches gitignored) |
| `../../paper/` | Earlier LaTeX drafts |

## Experiment packages (`experiments/geometry/`)

| Package | Paper section |
|---------|----------------|
| `physics_activation_atlas/` | Quadratic chart, $K_H^{\mathrm{cross}}$, nested dimension, audits |
| `physics_stable_tangent_dimension/` | Dimensionality as a predictive range (not one eigengap) |
| `physics_quadratic_predictive_dimension/` | Linear vs quadratic held-out risk across ranks |
| `physics_curvature_probe_rank_sweep/` | Frozen $K_H$–probe curve at $d\in\{12,16,20\}$ |
| `physics_curvature_probe_submission_validation/` | Target-identity, denominator, shuffle, OOF audits |
| `physics_adaptive_dataset_curvature_probe/` | Adaptive analysis + typed target definitions |
| `physics_adaptive_dataset_curvature_probe_audit/` | Restored probe-$R^2$ vs catalog-magnitude audit |
| `physics_curvature_scale_bias_variance/` | Factorial $R\times m$ scale / bias–variance decomposition |
| `physics_local_probe_adaptation/` | Patch vs global OOF probe relative adaptation |
| `physics_local_probe_adaptation_audit/` | Final bounded audit (paired $\Delta\rho$, alignment, shuffle) |
| `physics_order_stratified_geometry/` | Order-stratified geometry controls |
| `physics_implicit_normal_inverse/` | Sphere-normal inverse / confirmatory geometry |

## Key frozen result ($k{=}2048$, $n{=}512$ anchors)

Controlled $\rho(K_H^{\mathrm{cross}}, R^2_{\mathrm{local}})$:

| $d$ | $\rho_{\mathrm{ctl}}$ |
|-----|----------------------|
| 12 | +0.143 |
| 16 | −0.240 |
| 20 | −0.233 |

Direct-error check at $d{=}16$: MSE $\rho{=}+0.227$, SST $\rho{\approx}-0.025$ (not denominator-driven).

## Local probe adaptation (exploratory, post hoc)

$\rho_{\mathrm{ctl}}(K_H,\Delta\mathrm{MSE}_{G\to P}){=}+0.153$; mean $\Delta\mathrm{MSE}_{G\to P}{\approx}-0.10$ (patch worse on average). Interpret as **relative adaptation** in high-curvature regions, not uniformly superior local decodability.

## What we do not claim

Intrinsic dimension; intrinsic Riemannian curvature; causality; scale invariance; patch probes outperforming globally on average; proven direction rotation; independent replication; valid DESI label results.

## Running

```bash
source .venv/bin/activate
export PYTHONPATH=experiments
python experiments/geometry/run_curvature_probe_submission_validation.py --help
```

Large intermediate artifacts (`.parquet`, `.npz`, embeddings) live under `outputs/` on the science host and are gitignored here.
