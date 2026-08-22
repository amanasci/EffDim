"""METHODS.md and REPORT.md for the stable-tangent-dimension audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .synthetics import closest_synthetic


def _read(out: Path, name: str, default=None):
    p = out / name
    if not p.exists():
        return default
    if name.endswith(".json"):
        return json.loads(p.read_text())
    if name.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def write_methods(out: Path, cfg: Any, ctx: dict, parity: dict, thr: dict) -> None:
    ks = ctx.get("ks", cfg.ks)
    text = f"""# Methods: stable tangent dimension and local-linearity audit

## Estimand

The primary scientific estimand is the number of **reproducible local tangent
directions supported by held-out variation** at a specified neighbourhood scale
$k$, denoted $d_T(i,k)$. Graph-geodesic dimension $d_G$, spectral rank, and
finite-scale tangent dimension describe different geometric objects and need
not agree. Curvature, probe labels, and mean-curvature associations are
**never** used to select $d_T$.

## Frozen inputs

Activations, neighbour rows, anchor IDs, OOF probe scores and graph $d_G$ are
loaded from the multimodel / effective-dimension / nested-dimension artifacts.
See `freeze_manifest.json`. Representations are used as stored; this run
records L2-normalization status rather than silently renormalizing.

## Coordinates

If activations are unit-normalized, displacements are spherical logarithms

$$
z_{{ij}} = \\frac{{\\theta_{{ij}}}}{{\\sin\\theta_{{ij}}}}\\,(x_j - (x_i^\\top x_j)\\,x_i),
$$

with a stable small-angle branch $\\theta/\\sin\\theta \\approx 1 + \\theta^2/6$.
Projected chords $(I-x_i x_i^\\top)(x_j-x_i)$ are a sensitivity analysis.
Local PCA is **uncentred through the anchor** (no neighbour-mean subtraction)
in the primary analysis.

## Nested cross-fitted PCA

One maximum-rank SVD is computed per (anchor, scale, split half). Candidate
ranks are nested prefixes $T_{{i,1}}\\subset\\cdots\\subset T_{{i,d_\\max}}$.
Prefix agreement $A_i(d,k)=\\frac1d\\mathrm{{tr}}(P^A_{{i,d}}P^B_{{i,d}})$ is
invariant to eigenvector signs and to rotations inside the prefix. Near-degenerate
eigengap blocks are accepted or rejected as a whole.

Held-out linear risk $R_i(d,k)$ is the symmetric cross-fitted reconstruction
error; $G_i(d,k)=(R_i(d-1)-R_i(d))/R_i(0)$ is the normalized incremental gain.

## Nulls and thresholds

Matched nulls: residual isotropic directions after removing a prefix; column
permutation / parallel analysis; split-schedule re-splits. Thresholds were
frozen on **calibration** synthetic seeds (`thresholds.json`) before any
inspection of real probe associations. Evaluation synthetic seeds were held out.

## Operational $d_T$

A rank (or degenerate block) enters the stable tangent prefix only if it
passes split stability vs null, held-out linear gain vs null, cross-scale
persistence, and tangent-like eigenvalue scaling ($\\alpha\\approx 2$ bands
calibrated on synthetics, not hard-wired to exactly 2). Isolated accepted
ranks above a failed block are forbidden:

$$
d_T(i,k)=\\text{{end of the largest consecutive accepted prefix}}.
$$

Model-level summaries: median, IQR, survival $p_d(k)$, paired-anchor bootstrap
intervals. Scale grid: `{ks}`. Reference curvature scale $k={cfg.primary_k}$.
Graph $d_G={cfg.d_core}$ is reported alongside $d_T$ and is not a selection input.

Cross-model replication applies this **same frozen ViT-B decision function**
(full-neighbourhood eigengaps, split $A/G$, cross-scale tracks, frozen
`thresholds.json`) to 128 shared anchors. Gates are not retuned per model.

## Curvature / distortion panel (after freeze)

At $d_T$, the $d_T$ confidence band, $d_G$, and common $d=16$:

- $D_{{\\mathrm{{lin}}}}$: held-out linear residual energy (distortion, not pure curvature)
- $Q_S$, $\\Delta_S$: held-out quadratic sphere-normal gain
- $K_{{\\mathrm{{dir}}}}^2=K_H^2+K_{{\\mathrm{{aniso}}}}^2$ with $K_{{\\mathrm{{aniso}}}}^2=\\frac{{2}}{{d(d+2)}}\\|B^\\circ\\|_F^2$
- $K_{{\\max}}=\\max_{{|v|=1}}|B^S(v,v)|$ (multi-start + MC lower bound)
- curvature spectrum of $\\mathrm{{Sym}}^2(T)$ with $\\sqrt{{2}}$ off-diagonal weights
- sphere-parallel-transported tangent rotation $K_{{\\mathrm{{rot}}}}$
- Gauss excess sectional summaries, kept separate from the ambient $+1$ baseline

Mean curvature $K_H$ is diagnostic anatomy, not the headline measure.
Squared tensor norms are split-cross debiased.

## Probe associations

Fixed five-fold OOF global-probe scores only. Multiplicity is handled by a
family max-statistic over the prespecified metric × dimension panel. Probe
results cannot alter $d_T$, metric definitions, scales, or synthetic thresholds.

## Parity

Nested-basis quantities at $d=12$ and $d=16$ are reproduced from frozen
artifacts before new analysis (`parity.json`). The log-map uncentred PCA is a
**documented coordinate convention** for this audit, not a silent change of
the frozen preprocessing used to build neighbour lists.
"""
    (out / "METHODS.md").write_text(text)


def write_report(out: Path, cfg: Any, ctx: dict, parity: dict, labels: dict) -> None:
    summary = _read(out, "tangent_dimension_summary.csv", pd.DataFrame())
    loc = _read(out, "local_tangent_dimensions.parquet", pd.DataFrame())
    e4 = _read(out, "e4_block_evidence.parquet", pd.DataFrame())
    trk = _read(out, "scale_tracking.parquet", pd.DataFrame())
    atlas = _read(out, "curvature_metric_atlas.parquet", pd.DataFrame())
    rel = _read(out, "metric_reliability.csv", pd.DataFrame())
    agr = _read(out, "metric_agreement.csv", pd.DataFrame())
    sens = _read(out, "dimension_sensitivity.csv", pd.DataFrame())
    assoc = _read(out, "probe_associations.csv", pd.DataFrame())
    syn_c = _read(out, "synthetic_calibration.csv", pd.DataFrame())
    syn_e = _read(out, "synthetic_evaluation.csv", pd.DataFrame())
    repl = _read(out, "cross_model_replication.csv", pd.DataFrame())
    thr = _read(out, "thresholds.json", {})
    k_ref = int(labels.get("k_ref", cfg.primary_k))
    primary = labels.get("primary", "tangent_dimension_unresolved")

    def _med_sum(k=None):
        if summary is None or not len(summary):
            return {}
        g = summary if k is None else summary[summary.k == k]
        return g.iloc[0].to_dict() if len(g) else {}

    ref = _med_sum(k_ref) or (_med_sum() if len(summary) else {})
    dT = float(ref.get("median_dT", float("nan")))
    lo, hi = float(ref.get("median_lo", float("nan"))), float(ref.get("median_hi", float("nan")))
    iqr = float(ref.get("iqr_dT", float("nan")))
    scale_dep = False
    if len(summary) >= 2:
        scale_dep = float(summary.median_dT.max() - summary.median_dT.min()) >= 2.0

    # E4 gates
    e4_pass = "unresolved"
    extra_lab = labels.get("extra_scaling")
    if len(e4):
        a = float(np.nanmedian(e4.A_block_13_16))
        g = float(np.nanmedian(e4.G_block_13_16))
        agree_q = float(thr.get("agree_null_q99", 0.55))
        gain_q = float(thr.get("gain_null_q99", 0.0))
        stab = a >= agree_q
        gain = g >= gain_q
        e4_pass = (
            "weak_tangent"
            if stab and gain and extra_lab == "tangent_like"
            else "thickness"
            if extra_lab == "scale_independent_thickness"
            else "curvature_normal"
            if extra_lab == "curvature_normal_like"
            else "stratification"
            if extra_lab in ("mixed_or_crossing", "finite_scale_stratification")
            else "fail_or_unresolved"
        )

    # reliability ranking
    rel_txt = "n/a"
    if len(rel) and "split_rho" in rel.columns:
        r2 = rel.groupby("metric")["split_rho"].median().sort_values(ascending=False)
        rel_txt = r2.to_string()

    agr_notes = []
    if len(agr):
        def pair(a, b):
            hit = agr[((agr.a == a) & (agr.b == b)) | ((agr.a == b) & (agr.b == a))]
            return float(hit.rho.median()) if len(hit) else float("nan")

        agr_notes = [
            f"D_lin vs Q_S: ρ={pair('D_lin','Q_S'):.3f}",
            f"K_dir vs Q_S: ρ={pair('K_dir_cross','Q_S'):.3f}",
            f"K_H vs K_dir: ρ={pair('K_H_cross','K_dir_cross'):.3f}",
            f"K_aniso vs K_H: ρ={pair('K_aniso_cross','K_H_cross'):.3f}",
            f"K_max vs K_dir: ρ={pair('K_max','K_dir_cross'):.3f}",
        ]

    # mag_r paths
    kh_specific = "unresolved"
    if len(assoc):
        def path(m):
            g = assoc[assoc.metric == m].sort_values("d")
            return g[["d", "raw", "+controls", "sign_recurrence"]].to_string(index=False) if len(g) else "n/a"

        kh = assoc[assoc.metric == "K_H_cross"]
        others = assoc[assoc.metric.isin(["D_lin", "Q_S", "K_dir_cross", "K_aniso_cross", "K_max"])]
        jump_kh = False
        if len(kh) >= 2:
            k12 = kh[kh.d == 12]
            k16 = kh[kh.d == 16]
            if len(k12) and len(k16):
                jump_kh = abs(float(k16.raw.iloc[0]) - float(k12.raw.iloc[0])) > 0.2
        jump_other = False
        for m, gm in others.groupby("metric"):
            a = gm[gm.d == 12]
            b = gm[gm.d == 16]
            if len(a) and len(b) and abs(float(b.raw.iloc[0]) - float(a.raw.iloc[0])) > 0.2:
                jump_other = True
        kh_specific = "largely_KH_specific" if jump_kh and not jump_other else (
            "shared_by_broader_metrics" if jump_kh and jump_other else "no_clear_rank16_jump"
        )
    else:
        path = lambda m: "n/a"  # noqa: E731

    # closest synthetic
    closest, cdist = "n/a", float("nan")
    if len(syn_e) and len(loc):
        real = {
            "median_dT": dT,
            "p_ge_12": float(ref.get("p12", np.nan)),
            "p_ge_16": float(ref.get("p16", np.nan)),
            "agree_13_16": float(np.nanmedian(e4.A_block_13_16)) if len(e4) else np.nan,
            "gain_13_16": float(np.nanmedian(e4.G_block_13_16)) if len(e4) else np.nan,
            "alpha_13_16": float(trk[trk.rank0 >= 12].alpha.median()) if len(trk) and (trk.rank0 >= 12).any() else np.nan,
            "var_share_13_16": np.nan,
            "Dlin_12": float(atlas[atlas.d == 12].D_lin.median()) if len(atlas) and (atlas.d == 12).any() else np.nan,
            "Dlin_16": float(atlas[atlas.d == 16].D_lin.median()) if len(atlas) and (atlas.d == 16).any() else np.nan,
        }
        rows = syn_e.groupby("kind").median(numeric_only=True).reset_index().to_dict("records")
        closest, cdist = closest_synthetic(real, rows)

    # interpretation
    interp = "tangent_dimension_unresolved"
    if labels.get("parity_failed"):
        interp = "tangent_dimension_unresolved"
    elif primary == "scale_dependent_tangent_dimension":
        interp = "scale_dependent_finite_scale_tangent_geometry"
    elif primary == "stable_tangent_dimension_identified":
        interp = "stable_tangent_geometry_identified"
    elif primary == "stable_thickness_beyond_tangent":
        interp = "stable_thickness_beyond_tangent"
    elif primary == "finite_scale_stratification":
        interp = "finite_scale_stratification"
    elif primary == "tangent_dimension_unresolved":
        interp = "tangent_dimension_unresolved"
    if len(sens) and (sens.label == "unresolved").all():
        interp = "stable_dimension_but_curvature_metrics_unresolved"

    synth_ok = "n/a"
    if len(syn_e):
        cov = []
        for kind, g in syn_e.groupby("kind"):
            td = float(g.true_d.median())
            rec = float(g.median_dT.median())
            cov.append(f"{kind}: true {td:.0f} recovered {rec:.1f}")
        synth_ok = "\n".join(cov)

    report = f"""# Stable tangent dimension and local-linearity audit (ViT-B)

## Question

What is the number of reproducible local tangent directions supported by
held-out variation at the established curvature neighbourhood scale, and do
complementary finite-radius distortion measures agree after that dimension is
frozen independently of curvature and probes?

## Parity

```json
{json.dumps(parity, indent=2, default=str)[:4000]}
```

## Model-level $d_T$

Reference scale $k={k_ref}$. Graph $d_G={cfg.d_core}$ (not used in selection).

```
{summary.to_string(index=False) if len(summary) else 'n/a'}
```

Primary operational label: **{primary}**

## Directions 13–16 (E4)

```
{e4.describe().to_string() if len(e4) else 'n/a'}
```

E4 classification (label-free): **{e4_pass}** (scaling={extra_lab})

## Synthetic evaluation

```
{syn_e.groupby('kind')[['true_d','median_dT','p_ge_12','p_ge_16']].median().to_string() if len(syn_e) else 'n/a'}
```

Closest synthetic mechanism (prespecified feature vector): **{closest}** (distance {cdist:.3f}).

## Curvature / distortion panel

Reliability (split-half Spearman):

```
{rel_txt}
```

Agreement notes:

- {chr(10)+'- '.join(agr_notes) if agr_notes else 'n/a'}

Dimension sensitivity:

```
{sens.to_string(index=False) if len(sens) else 'n/a'}
```

## Probe associations (secondary; after freeze)

Rank-16 mag_r transition vs broader metrics: **{kh_specific}**

### $K_H$
```
{path('K_H_cross')}
```
### $D_{{lin}}$
```
{path('D_lin')}
```
### $Q_S$
```
{path('Q_S')}
```
### $K_{{dir}}$
```
{path('K_dir_cross')}
```
### $K_{{aniso}}$
```
{path('K_aniso_cross')}
```
### $K_{{\\max}}$
```
{path('K_max')}
```

## Cross-model

```
{repl.to_string(index=False) if len(repl) else 'ViT-B primary complete; replication not yet run (frozen thresholds).'}
```

## Direct answers

1. **Operational $d_T$:** largest consecutive prefix of eigengap blocks that pass split-stability, held-out linear gain, cross-scale persistence, and tangent-like scaling, all vs matched nulls frozen on calibration synthetics. Mathematically $d_T(i,k)$ is a cross-fitted, multiscale subspace rank — not Isomap dimension and not an eigenvalue elbow.

2. **Scale:** primary estimate at the existing curvature neighbourhood $k={k_ref}$ (grid `{ctx.get('ks', cfg.ks)}`). Radii $r_i(k)$ are used for scaling regressions, not $k$ itself.

3. **Single vs scale-dependent:** {'scale-dependent $d_T(k)$' if scale_dep else 'approximately concentrated at the reference scale'}.

4. **ViT-B $d_T$:** median {dT:.2f} with bootstrap band [{lo:.2f}, {hi:.2f}], IQR {iqr:.2f}. $d_G=12$.

5. **Directions 13–16 gates:** split agreement median {float(np.nanmedian(e4.A_block_13_16)) if len(e4) else float('nan'):.3f}; gain median {float(np.nanmedian(e4.G_block_13_16)) if len(e4) else float('nan'):.3f}. Classification: {e4_pass}.

6. **Mechanism of 13–16:** {e4_pass}. Linear held-out gain and scaling — not $\\Delta_S$ or $K_H$–mag_r — decide tangent vs thickness vs curvature-normal vs stratification.

7. **$d_T$ vs $d_G$:** $d_G$ is a graph-geodesic freeze; $d_T$ is held-out local linear rank. They {'agree within ~1' if np.isfinite(dT) and abs(dT-12)<=1 else 'need not coincide; disagreement is expected if finite-scale directions exist'}.

8. **Most split-half reliable metrics:** see reliability table. Metrics with low split-ρ are not used for strong model claims.

9. **Agreement with departure from linearity:** $D_{{\\mathrm{{lin}}}}$ is the primary finite-radius distortion; $Q_S$ and $K_{{\\mathrm{{dir}}}}$ are the quadratic/tensor counterparts. See agreement notes.

10. **Mean curvature:** $K_H$ is one contraction of $B^S$. $K_{{\\mathrm{{aniso}}}}$ and $K_{{\\max}}$ capture saddle-like or concentrated bending that $K_H$ can miss.

11. **Dimension robustness:** see `dimension_sensitivity.csv`. A conclusion is `dimension_robust` only if it survives the prespecified $d_T$ band.

12. **Synthetics recovered:** {synth_ok}

13. **Closest synthetic to ViT-B:** {closest}.

14. **Cross-model recurrence:** {'see replication table' if len(repl) else 'not yet run; ViT-B primary is the frozen pipeline'}.

15. **Probe associations after multiplicity:** see `probe_associations.csv` (`family_pass`). These did not alter $d_T$.

16. **Rank-16 mag_r result:** {kh_specific}.

## Paper-level claim (warranted only if labels support it)

Foundation-model representations exhibit reproducible, scale-dependent
departures from local linearity. Analysis requires first identifying the
stable tangent dimension through cross-fitted, multiscale subspace evidence,
then evaluating complementary finite-radius, quadratic-tensor and
tangent-variation measures. Graph-geodesic dimension, stable tangent
dimension and spectral rank describe different geometric objects.

Ontology: a stable direction is not automatically an intrinsic tangent;
bootstrap-stable thickness can be non-tangent; weak mean curvature does not
imply local linearity; a probe-associated field is not automatically
geometrically valid.

## Primary interpretation

**{interp}**
"""
    (out / "REPORT.md").write_text(report)
    labels = dict(labels)
    labels["interpretation"] = interp
    labels["e4"] = e4_pass
    labels["closest_synthetic"] = closest
    labels["kh_specific"] = kh_specific
    (out / "decision_labels.json").write_text(json.dumps(labels, indent=2, default=str))
    print(f"[std] REPORT interpretation={interp}", flush=True)
