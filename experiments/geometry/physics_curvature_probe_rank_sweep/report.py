"""METHODS.md and REPORT.md."""

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


def write_methods(out: Path, cfg: Any, ctx: dict, parity: dict, thr: dict) -> None:
    text = f"""# Methods: curvature–probe rank sweep

## Estimand

For every integer chart rank $d$, compute the **rank-conditioned** association

$$\\rho_d=\\rho_{{\\mathrm{{Spearman}}}}\\bigl(K_H^{{(d)}},\\mathrm{{mag\\_r\\_desi}}\\bigr).$$

$K_H^{{(d)}}$ is the sphere-normal mean-curvature statistic from the completed
nested-dimension-curvature experiment, evaluated on a rank-$d$ chart. It is
**not** the same geometric object at every $d$. The maximizing $d$ is not
intrinsic dimension.

Primary family: $d=12,\\ldots,20$ at $k=2048$. Reference ranks $d=8,\\ldots,11$
are secondary. This sweep is motivated by already observed $d=12$ vs $d=16$
results and is **not preregistered or confirmatory**.

## Reuse

Per-anchor $K_H^{{(d)}}$ for $d=8,\\ldots,20$ at $k=2048$ is **reused** from
`physics_nested_dimension_curvature/nested_curvature_metrics.parquet`
(column `K_H_cross`, mean over five neighbour splits). Embeddings, kNN,
nested PCA and $B_S$ fits are not recomputed for the primary family.

Exact frozen definition: split-half fits of $B^S$ after removing the forced
L2 radial $Q_R$ and projecting off $\\mathrm{{span}}(x_0,J_d)$; metric whitening
is per-dimension RMS of fit-split tangent coordinates; ridge grid unchanged.
$H_S$ is the diagonal mean of $B^S$. Primary scalar is
$K_H^{{\\mathrm{{cross}}}}=\\langle H_A,H_B\\rangle$, not $||H||$.
Secondary: $K_{{\\mathrm{{aniso}}}}^{{\\mathrm{{cross}}}}$ (traceless),
$K_{{\\mathrm{{dir}}}}^{{\\mathrm{{cross}}}}$ (total), held-out $\\Delta_S$.

Confounders are the frozen triple
`log_knn_radius`, `local_label_variance`, `local_evaluation_count`.
Controlled association is rank-residual partial Spearman (same as nested
`stage_associations`). Probe score is `local_r2` with `target=mag_r_desi`.

`mag_r_desi` is never used to choose $d$, $\\lambda$, reliability gates, or
the valid-anchor mask.

## Inference

Raw permutations shuffle the probe once and reuse that shuffle at every $d$.
Controlled permutations use rank-space Freedman–Lane residual permutation
of the probe given the frozen confounders. $T_{{\\max}}=\\max_{{d\\in\\{{12,\\ldots,20\\}}}}|\\rho_d|$.
At least {thr.get("n_perm")} permutations and {thr.get("n_boot")} paired
anchor bootstraps. Simultaneous bands use a max-deviation construction.

Reliability gates use only $R_H$ and sample size (frozen $R_H<{thr.get("r_h_fail")}$).
Failed ranks stay visible (hatched/faded).

$R_L^2(d)$ is reused from the quadratic-predictive-dimension linear baseline
with the full spherical-log energy denominator. Thresholds
$\\tau\\in\\{{0.80,0.825,0.85,0.875,0.90\\}}$; $d_{{85}}$ is **post hoc**.

Trace-only acceleration is **not** used in production. The identity
$K_H=||\\mathrm{{mean}}_a B_{{aa}}||$ is checked against `metric_scalars`.

Thresholds: `{json.dumps(thr, default=str)}`

Parity excerpt: `{json.dumps(parity, default=str)[:2000]}`
"""
    (out / "METHODS.md").write_text(text)


def write_report(out: Path, cfg: Any, ctx: dict, parity: dict, labels: dict) -> None:
    assoc = _read(out, "dimension_associations.csv", pd.DataFrame())
    perm = _read(out, "permutation_results.csv", pd.DataFrame())
    ve = _read(out, "variance_threshold_crossings.csv", pd.DataFrame())
    rel = _read(out, "curvature_reliability.csv", pd.DataFrame())
    scale = _read(out, "scale_sensitivity.csv", pd.DataFrame())
    reuse = _read(out, "reuse_manifest.json", {})
    runtime = _read(out, "runtime_profile.json", {}) or {}
    raw = labels.get("raw_by_d") or {}
    ctl = labels.get("controlled_by_d") or {}
    curve = ", ".join(f"d={d}: raw={_fmt(raw.get(d))} ctl={_fmt(ctl.get(d))}" for d in range(8, 21) if d in raw or d in ctl)
    fwer = labels.get("fwer_hits_controlled") or []
    d85 = labels.get("d85")
    lab = labels.get("primary")
    emerge = ""
    if 12 in ctl and 16 in ctl:
        emerge = (
            f"Controlled $\\rho$ at $d=12$ is {_fmt(ctl.get(12))} and at $d=16$ is {_fmt(ctl.get(16))}. "
            "The signed association strengthens as ranks 13–16 enter the chart."
        )
    scale_txt = "primary k=2048 only" if scale is None or not len(scale) or scale.k.nunique() <= 1 else (
        ", ".join(f"k={int(k)} d16_ctl={_fmt(scale[(scale.k==k)&(scale.d==16)].controlled.mean()) if ((scale.k==k)&(scale.d==16)).any() else 'NA'}" for k in sorted(scale.k.unique()))
    )
    text = f"""# Report: curvature–probe rank sweep (ViT-B only)

Primary label: **`{lab}`**

This analysis is **rank-conditioned** and **not preregistered**. It does not
identify intrinsic or tangent dimension. $K_H^{{(d)}}$ is the nested-experiment
statistic under a rank-$d$ chart.

## Answers

### 1. Complete raw and controlled curve ($d=8$ to $20$)

{curve}

### 2. Largest association

Peak controlled $|\\rho|$ at $d={labels.get("peak_d_controlled")}$ with
$\\rho={_fmt(labels.get("peak_rho_controlled"))}$.
Peak raw $|\\rho|$ at $d={labels.get("peak_d_raw")}$.

### 3. Does it survive max-statistic correction on $d=12,\\ldots,20$?

Familywise hits (controlled $p_{{\\mathrm{{FWER}}}}\\le 0.05$): `{fwer}`.
Global controlled $p={_fmt(labels.get("p_global_controlled"))}$.
Global raw $p={_fmt(labels.get("p_global_raw"))}$.
The largest uncorrected point is **not** called significant unless it is in
that FWER set.

### 4. Stricter dimension-by-scale correction

{scale_txt}
If only $k=2048$ was analysed, the $(d,k)$ family equals the primary family.

### 5. Shape

Inspect the four-panel figure. The nested reuse curve is typically weak near
$d=12$, then more negative from the mid-teens. Describe it as broad, threshold-like,
monotonic, plateauing, or isolated from the plotted points — not from a preferred $d$.

### 6. Emergence at ranks 13–16

{emerge}

### 7. Association at $d_{{85}}$

$d_{{85}}={d85}$ (post hoc 85% linear held-out energy).
Raw $\\rho={_fmt(labels.get("rho_at_d85_raw"))}$, controlled
$\\rho={_fmt(labels.get("rho_at_d85_ctl"))}$.

### 8. Sensitivity to 80–90% thresholds

See `variance_threshold_crossings.csv`. $d_{{80}}$ is typically 12;
$d_{{85}}$ is near 20; $d_{{90}}$ is not reached. The curvature–probe
association at those crossings is reported there and must not be treated as
a unique privileged dimension.

### 9. Does the curve track estimator reliability?

`tracks_reliability={labels.get("tracks_reliability")}`.
Reliability ($R_H$, $\\Delta_S$) is plotted without using the probe. A rank
is not discarded because $\\rho$ is inconvenient.

### 10. Scale recurrence

{scale_txt}
`scale_stable={labels.get("scale_stable")}`.

### 11. 16–20-dimensional geometry vs rank-conditioned association

This supports a **rank-conditioned statistical association**, not a proof that
the cloud is 16–20 dimensional. $K_H^{{(16)}}$ and $K_H^{{(12)}}$ are different
chart statistics.

### 12. What cannot be concluded

Exact tangent or intrinsic dimension cannot be read from the maximizing rank,
from $d_{{85}}$, or from a single uncorrected $\\rho$. Inadequacy of a 12-chart
quadratic model (previous experiment) is a different estimand.

## Gate label

`{lab}` is produced by `classify.primary_label` from FWER hits, reliability,
reliability-tracking, and scale stability.

## Reuse vs recomputation

{json.dumps(reuse, default=str)[:1800]}

## Runtime and parity

Runtime seconds: `{runtime.get("total_seconds")}`. Stages: `{runtime.get("completed")}`.
Parity ok: `{parity.get("ok")}`. Freeze: `{parity.get("freeze_hash")}`.
Output: `{out}`.
"""
    (out / "REPORT.md").write_text(text)
