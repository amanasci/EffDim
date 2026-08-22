"""METHODS.md and REPORT.md answering the 16 required questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read(out: Path, name: str, default=None):
    p = out / name
    if not p.exists():
        return default
    if name.endswith(".json"):
        return json.loads(p.read_text())
    if name.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _fmt(x, nd=3):
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def write_methods(out: Path, cfg: Any, ctx: dict) -> None:
    text = """# Methods: adaptive per-dataset curvature–physics probe

## Estimand

The unit of analysis is a **dataset**, not a dataset pair. For each eligible
dataset $j$ the geometry-only interval $D_j$ is chosen from embeddings, then
frozen. Only afterwards are registered physics labels loaded.

For every $d\\in D_j$ the rank-conditioned statistic $K_{{H,j}}^{{(d)}}$ is the
frozen nested split-half inner product $\\langle H_A,H_B\\rangle$ of the
sphere-normal mean-curvature vector after removing $Q_R$ and whitening. The
maximizing rank is **not** an intrinsic dimension.

## Discovery vs confirmation

Discovery is the completed ViT-B / `mag_r_desi` rank sweep on Smith42/galaxies.
It is plotted as a reference curve and **excluded** from confirmatory aggregate
$p$-values.

## Inventory

Eligible catalogs are the Smith42 registries (`galaxies`, `desi_hsc_crossmatched`,
`jwst_hsc_crossmatched`, `legacysurvey_hsc_crossmatched`, `sdss_hsc_crossmatched`)
plus documented adapters. Labels are taken only from registered columns, never
from every numeric field. Orientation is taken from documented photometry /
redshift semantics and is never reversed from observed $\\rho$.

Primary encoder family: ViT-B imaging (`vit_base_galaxies` or `vit_base_hsc`).
Other encoders are listed in the inventory and are not mixed into the primary
replication.

## Neighbourhood scale

Primary $k$ is the largest preset in {256,512,768,1024,1536,2048} with
$k \\le 0.125 n$. This rule is sample-size only.

## Geometry range

Held-out linear $R^2_L(d)$ uses the complete spherical-log ambient energy.
Thresholds tau in {0.70,0.75,0.80,0.825,0.85,0.875,0.90,0.95} are not
extrapolated. Quadratic screening is coarse (fixed-coordinate) then refined
with the frozen closest-point map from `physics_quadratic_predictive_dimension`.
$D_j$ follows the predeclared min/max formula. If $d_{90}$ is not reached the
interval is marked right-truncated.

## Inference

Within each dataset×label, one label permutation is reused across all
$d\\in D_j$ ($B\\ge 10000$ unless smoke). Report $p<1/(B+1)$ on zero exceedances.
Paired-anchor bootstrap $B\\ge 2000$ unless smoke. Same-object labels share one
object permutation. Independent datasets enter a joint maximum statistic.
Discovery is excluded from that family.

Shared-core confounders: `log_knn_radius`, `local_label_variance`,
`local_evaluation_count`. Dataset-specific extras (DESI `ZERR`/`EBV`) are
secondary only and are not searched.

## Cross-dataset axes

Absolute rank $d\\mapsto\\rho$ and variance $\\tau=R^2_L(d)\\mapsto\\rho$.
Interpolation on $\\tau$ is only between observed reliable ranks.

Smoke=%s. Seed=%s.
""" % (cfg.smoke, cfg.seed)
    (out / "METHODS.md").write_text(text)


def write_report(out: Path, cfg: Any, ctx: dict) -> None:
    inv = _read(out, "dataset_inventory.csv", pd.DataFrame())
    labs = _read(out, "physics_label_manifest.csv", pd.DataFrame())
    inc = _read(out, "inclusion_manifest.json", {})
    ranges = _read(out, "geometry_dimension_ranges.csv", pd.DataFrame())
    rank = _read(out, "dataset_rank_associations.csv", pd.DataFrame())
    perm = _read(out, "dataset_permutation_results.csv", pd.DataFrame())
    glob = _read(out, "global_permutation_results.csv", pd.DataFrame())
    contr = _read(out, "replication_contrasts.csv", pd.DataFrame())
    lodo = _read(out, "leave_one_dataset_out.csv", pd.DataFrame())
    rel = _read(out, "curvature_reliability.csv", pd.DataFrame())
    scale = _read(out, "scale_sensitivity.csv", pd.DataFrame())
    summary = _read(out, "summary.json", {})
    freeze = _read(out, "geometry_freeze.json", {})

    def _tbl(df, cols=None):
        if df is None or not len(df):
            return "(none)"
        use = df[cols] if cols else df
        return use.to_string(index=False)

    included = inv[inv.inclusion_status == "included"] if len(inv) and "inclusion_status" in inv.columns else inv
    excluded = inv[inv.inclusion_status != "included"] if len(inv) and "inclusion_status" in inv.columns else pd.DataFrame()
    p_g = float(glob.iloc[0].p_global_ctl) if len(glob) else float("nan")

    q7 = _tbl(rank, [c for c in ("dataset_id", "label", "d", "raw", "controlled", "p_ctl_fwer", "r2_L") if rank is not None and c in rank.columns]) if rank is not None else "(none)"
    q8 = "(none)"
    if rank is not None and len(rank) and "p_ctl_fwer" in rank.columns:
        hit = rank[(rank.p_ctl_fwer <= 0.05) & (rank.is_discovery == False)]  # noqa: E712
        q8 = _tbl(hit, [c for c in ("dataset_id", "label", "d", "controlled", "p_ctl_fwer") if c in hit.columns]) or "(none survive)"

    text = f"""# Adaptive dataset curvature–physics probe

**Primary label:** `{summary.get("primary_label", "NA")}`

Discovery reference: ViT-B / `mag_r_desi` on Smith42/galaxies. It is shown and
is **not** counted as an independent confirmatory study.

Geometry freeze sha256 prefix: `{freeze.get("sha16", "NA")}`.

## 1. Which datasets possess valid aligned physics labels?

Smith42/galaxies (`physics_vit_base`) has a row-aligned `vit_base_test_labels.npz`
(`mag_r_desi`, `smooth_fraction`, `photo_z`, `stellar_mass`, `sfr`).
Smith42/DESI (`desi_vit_base_hsc`) has a cached catalog whose row count equals
the local ViT-B embedding table (n=20465): spectroscopic `Z` and `r_cmodel_mag`.

{_tbl(labs, [c for c in ("dataset_id","raw_column","canonical_label","valid_geometry_subset","include_in_association","underpowered") if labs is not None and c in labs.columns])}

## 2. Which were included or excluded, and why?

Included: {inc.get("included_datasets")}

Excluded (not because associations were weak):

{_tbl(excluded, [c for c in ("dataset_id","inclusion_status","exclusion_reason") if excluded is not None and c in excluded.columns])}

JWST is excluded because the Smith42 catalog has 1667 rows and the embedding
parquet has 1496 — positional join is refused. Legacy photometry exists in the
Smith42 hub cache but the processed datasets cache was not loadable. SDSS has
`Z` in the catalog and no local embeddings. CosmosWeb is not Smith42 and its
catalog columns are dropped by `prepare()`. Other encoders are inventory-only.

## 3. What dimensional interval was selected for each dataset using geometry alone?

{_tbl(ranges, [c for c in ("dataset_id","d_low","d_high","d_low_primary","d_high_primary","right_truncated","d_75","d_80","d_85","d_90","dL_plat","dQ_plat") if ranges is not None and c in ranges.columns])}

Labels were not loaded in this step. The range file was hashed in
`geometry_freeze.json` before associations.

## 4. Which datasets reached 80%, 85%, 90% and 95% held-out variance?

{_tbl(ranges, [c for c in ("dataset_id","d_80","d_85","d_90","d_95") if ranges is not None and c in ranges.columns])}

`not_reached` means the spectral pass never crossed that $\\tau$. No
extrapolation was used.

## 5. Where did linear and quadratic reconstruction plateau?

{_tbl(ranges, [c for c in ("dataset_id","dL_plat","dQ_plat","quadratic_source") if ranges is not None and c in ranges.columns])}

Physics quadratic numbers are reused from the completed closest-point
`physics_quadratic_predictive_dimension` experiment.

## 6. Was any curvature sweep truncated by estimator identifiability?

{_tbl(ranges, [c for c in ("dataset_id","d_curv_max","curvature_range_right_truncated","right_truncated") if ranges is not None and c in ranges.columns])}

{_tbl(rel, [c for c in ("dataset_id","d","valid_frac","median_R_H","fail_reliability","m_d") if rel is not None and c in rel.columns])}

## 7. Complete raw and controlled rank curves

{q7}

## 8. Which associations survive within-dataset correction?

{q8}

## 9. Which survive global dataset × label × dimension correction?

Global confirmatory $p$ (discovery excluded) = {_fmt(p_g, 4)}.
Family: {inc.get("confirmatory_family")}.
Same-object physics labels share one object permutation. DESI is an independent
sample and enters the joint maximum separately.

## 10. Does the ViT-B positive-core / negative-tail transition recur?

{_tbl(contr, [c for c in ("dataset_id","label","mag_like","d_80","d_85","delta_85_80_raw","delta_85_80_ctl","predicted_sign","sign_consistent_raw") if contr is not None and c in contr.columns])}

For magnitude labels with the same documented orientation as `mag_r_desi`, the
discovery-informed contrast $\\Delta^{{85-80}}$ is predicted negative. Redshift
and morphology labels are not assumed to share that sign.

## 11. Absolute rank vs variance explained?

The variance-axis plots (`figures/07_heatmap_variance.png`,
`figures/08_discovery_variance_overlay.png`) are the primary cross-dataset
comparison. Absolute-rank heatmaps leave out-of-range cells blank. A common
numerical $d$ need not mean a common fraction of held-out energy.

## 12. Distribution of $\\Delta^{{85-80}}$

{_tbl(contr, [c for c in ("dataset_id","label","delta_85_80_ctl","mag_like","is_discovery") if contr is not None and c in contr.columns])}

## 13. Leave-one-dataset-out stability

{_tbl(lodo)}

## 14. Neighbourhood scale

Primary $k$ follows the frozen $0.125 n$ rule. Scale-sensitivity ranks are the
predeclared geometry ranks ($d_{{80}}$, $d_{{85}}$, $d_{{90}}$, plateaus), never
chosen from probe $\\rho$.

{_tbl(scale, [c for c in ("dataset_id","k","d","label","controlled","source") if scale is not None and c in scale.columns])}

## 15. Reliability, shrinkage, effective degrees of freedom

See `curvature_reliability.csv` and figure 15. Differences that appear only
where $R_H$ fails or $m(d)/n$ is extreme are not interpreted as physics-label
geometry.

## 16. What can and cannot be concluded

- $K_H^{{(d)}}$ is rank-conditioned curvature under a $d$-chart, not one
  geometric object.
- A maximizing $d$ is **not** intrinsic dimension, tangent dimension, or a
  claim that the manifold is $d$-dimensional.
- Smith42/galaxies and Smith42/DESI are the aligned ViT-B physics-labelled
  datasets used here. JWST/Legacy/SDSS/CosmosWeb remain in the manifest with
  explicit exclusion reasons.
- Discovery `mag_r_desi` is a reference, not a confirmatory replicate.
- Other physics labels on the same 16k galaxies are dependent; they are not
  independent studies.

Runtime: {_fmt(summary.get("runtime_s"), 1)} s. Permutations: {summary.get("n_perm")}.
Bootstraps: {summary.get("n_boot")}.
"""
    (out / "REPORT.md").write_text(text)
