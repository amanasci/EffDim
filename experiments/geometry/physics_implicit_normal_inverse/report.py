"""METHODS.md and REPORT.md for the implicit normal-space inverse."""

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


def _med(df, col, default=float("nan")):
    if df is None or len(df) == 0 or col not in getattr(df, "columns", []):
        return default
    return float(pd.to_numeric(df[col], errors="coerce").median())


def write_methods(out: Path, cfg: Any, ctx: dict, parity: dict, thr: dict) -> None:
    text = f"""# Methods: implicit normal-space inverse

## Estimand

Estimate the **normal space first**. The local manifold inside a finite carrier
$S_R$ is the common zero set of $q$ implicit functions

$$F:\\mathbb{{R}}^R\\to\\mathbb{{R}}^q,\\qquad F_\\ell(y)=a_\\ell^\\top y+\\tfrac12 y^\\top H_\\ell y.$$

Only the normal projector $P_N=AA^\\top$ is identifiable. The tangent is
$T_0=\\ker A^\\top$, so $d_1=R-q$ when $J_F(0)$ has rank $q$.

Two different quantities are reported:

- $c_N=\\dim(N_x\\mathcal{{M}}\\cap S_R)$ is the **total carrier-normal codimension**.
- $q_2=\\mathrm{{rank}}(B:\\mathrm{{Sym}}^2(T)\\to N)$ is the number of normals
  activated at quadratic order. Always $q_2\\le c_N$.

Prior $q_{{2\\mid 12}}=1$ is **not** evidence that $c_N=1$. Fitting does not
assume $d=12$ or $d=16$. After fitting we compare $q=4$ ($d_1=16$) and
$q=8$ ($d_1=12$) inside $R={cfg.R}$.

## Frozen inputs

Activations, kNN rows, anchors, nested PCA frames, splits and OOF probes are
loaded from completed geometry artifacts. This package never writes into

`physics_stable_tangent_dimension`, `physics_nested_dimension_curvature`,
or `physics_order_stratified_geometry`.

Unit-normalized representations are mapped with the spherical logarithm so the
estimator cannot rediscover $|x|^2=1$. Coordinates are $y=U_{{i,R}}^\\top z$
with the **anchor as origin** (no neighbour-mean centering). Held-out energy
outside $S_R$ is recorded. Primary carrier $R={cfg.R}$; sensitivity
$R\\in{cfg.R_sens}$.

Neighbour weighting is the frozen uniform kNN. Splits are radial-stratified
halves. Ridge $\\lambda$ is chosen on inner training splits with a one-SE rule.

## Constraint spectra

Linear constraints are the bottom eigenspaces of the weighted local covariance.
They are **not** automatically called normal.

Quadratic constraints profile $\\mathrm{{vech}}(yy^\\top)$ with $\\sqrt{{2}}$
off-diagonal weights. Candidate gradients are the bottom eigenvectors of
$K_\\lambda$. Degenerate blocks are judged through projectors.

Held-out residuals, quadratic cancellation $R^2$, Sampson distance, split
recurrence, adjacent-scale recurrence and matched weak-tangent nulls are stored
for every candidate. Codimension is **not** selected by minimizing $d_F$ over
$q$ ($q=0$ is trivially zero).

Geometric Stiefel/QR refinement is optional and reported separately. It does
not change frozen classification thresholds.

## Classification and bounds

Directions are labelled `curvature_active_normal`, `approximately_flat_normal`,
`structured_thickness_normal_candidate`, `first_order_tangent`, `mixed_order`,
or `unresolved`. Certified $c_N^-$ is the consecutive prefix of curvature-active
or approximately-flat normals. Then $d_1^+=R-c_N^-$. $d_1^-$ counts positively
certified first-order tangent directions. Isolated normals past an unresolved
block are not added to the certified prefix.

$q_2$ is recovered from the tangent restriction $S_\\ell=-T^\\top H_\\ell T$
after $A$ is inferred. Gauss-map transport of neighbouring-anchor frames is
validation, not a selection rule. OOF `mag_r_desi` is loaded only after
classifications are frozen.

## Synthetics

Thirteen matched families in an $R=20$ carrier with random rotations. Thresholds
are frozen on `calibration_seeds` and evaluated on untouched `evaluation_seeds`.
The method must be able to output dimensions other than 12 before the ViT-B
estimate is interpreted.

Thresholds: `{json.dumps(thr, default=str)}`

Parity: `{json.dumps(parity, default=str)[:1800]}`
"""
    (out / "METHODS.md").write_text(text)


def write_report(out: Path, cfg: Any, ctx: dict, parity: dict, labels: dict) -> None:
    bounds = _read(out, "dimension_bounds.csv", pd.DataFrame())
    clas = _read(out, "normal_classification.parquet", pd.DataFrame())
    proj = _read(out, "normal_projectors.parquet", pd.DataFrame())
    curv = _read(out, "implicit_curvature_rank.csv", pd.DataFrame())
    tail = _read(out, "tail_classification.parquet", pd.DataFrame())
    gauss = _read(out, "gauss_validation.csv", pd.DataFrame())
    seval = _read(out, "synthetic_evaluation.csv", pd.DataFrame())
    assoc = _read(out, "probe_associations.csv", pd.DataFrame())
    scal = _read(out, "constraint_scaling.parquet", pd.DataFrame())
    refine = _read(out, "geometric_refine.parquet", pd.DataFrame())
    k_ref = cfg.primary_k
    if len(bounds) and "k" in bounds.columns:
        hit = bounds[bounds.k == k_ref]
        b = hit.iloc[-1] if len(hit) else (bounds.iloc[-1] if len(bounds) else None)
    else:
        b = bounds.iloc[-1] if len(bounds) else None
    cNm = float(b.median_cN_minus) if b is not None else float("nan")
    d1p = float(b.median_d1_plus) if b is not None else float("nan")
    d1m = float(b.median_d1_minus) if b is not None else float("nan")
    iqr = float(b.iqr_cN_minus) if b is not None and "iqr_cN_minus" in b.index else float("nan")
    p8 = float(b.p_cN_ge_8) if b is not None and "p_cN_ge_8" in b.index else float("nan")
    p4 = float(b.p_cN_ge_4) if b is not None and "p_cN_ge_4" in b.index else float("nan")

    clas_k = clas[clas.k == k_ref] if len(clas) and "k" in clas.columns else clas
    n_flat = float((clas_k.label == "approximately_flat_normal").mean()) if len(clas_k) and "label" in clas_k.columns else float("nan")
    n_curv = float((clas_k.label == "curvature_active_normal").mean()) if len(clas_k) and "label" in clas_k.columns else float("nan")
    n_flat_c = _med(proj[proj.k == k_ref] if len(proj) and "k" in proj.columns else proj, "n_flat_prefix")
    n_curv_c = _med(proj[proj.k == k_ref] if len(proj) and "k" in proj.columns else proj, "n_curv_prefix")

    q2_8 = float("nan")
    q2_4 = float("nan")
    if len(curv):
        if 8 in set(curv.q.tolist()):
            q2_8 = float(curv[curv.q == 8].q2.median())
        if 4 in set(curv.q.tolist()):
            q2_4 = float(curv[curv.q == 4].q2.median())
    e4n = _med(tail, "e4_normal_frac")
    gauss_ov = _med(gauss, "median_overlap")
    weing = _med(gauss, "weingarten_cos")
    not_only12 = bool(len(seval) and "synth_not_only12" in seval.columns and bool(seval.synth_not_only12.iloc[0]))
    rec12 = float(seval[seval.kind.str.contains("d12")].call_12_8.mean()) if len(seval) and "call_12_8" in seval.columns else float("nan")
    rec16 = float(seval[seval.kind.str.contains("d16")].call_16_4.mean()) if len(seval) and "call_16_4" in seval.columns else float("nan")
    iso_d1 = float(seval[seval.kind == "isotropic_carrier"].d1_hat.median()) if len(seval) and "kind" in seval.columns else float("nan")
    primary = labels.get("primary", "implicit_normal_inverse_unresolved")
    parity_ok = bool(parity.get("ok")) if parity else False
    prior_q2 = (parity or {}).get("prior_q2_given_12", {})
    q2_persist = bool(np.isfinite(q2_8) and abs(q2_8 - 1.0) <= 1.5)

    # exact d1 claim gates
    exact_ok = (
        np.isfinite(cNm)
        and abs(cNm - 8) <= 1.0
        and np.isfinite(d1m)
        and d1m >= 10
        and abs(d1p - 12) <= 1.5
        and bool(labels.get("synth_not_only12"))
        and np.isfinite(gauss_ov)
        and gauss_ov >= 0.35
    )
    strongest = (
        "Within the stable rank-20 local carrier, ViT-B satisfies eight reproducible "
        "implicit normal constraints, while only one normal direction is significantly "
        "activated at quadratic order. The resulting geometry has a 12-dimensional "
        "tangent space but a rank-one first normal space."
        if (exact_ok and q2_persist and primary == "normal_codimension_8_supports_tangent12")
        else (
            "The strongest eight-normal / twelve-tangent claim is **not** made: "
            "normal-projector, held-out, scale, Gauss-map and synthetic-calibration "
            "gates do not all pass together with independent first-order tangent support."
        )
    )

    assoc_txt = "none"
    if assoc is not None and len(assoc):
        top = assoc.sort_values("rho_mag_r", key=np.abs, ascending=False).head(5)
        assoc_txt = "; ".join(f"{r.metric} ρ={r.rho_mag_r:.3f}" for _, r in top.iterrows())

    text = f"""# Report: implicit normal-space inverse (ViT-B only)

Primary label: **`{primary}`**

Parity ok: `{parity_ok}`. Prior $q_{{2\\mid 12}}$ from the order-stratified experiment:
{json.dumps(prior_q2, default=str)}. Freeze and mag_r curvature associations are in
`parity.json`. Cross-model replication was **not** run.

k grid: `{list(ctx.get('ks', []))}`; primary k={k_ref}; carrier R={cfg.R};
n anchors={len(ctx.get('use_sids', []))}.

## Required conclusions

1. **Certified independent carrier-normal constraints** $c_N^-$: median **{cNm:.2f}**
   (IQR {iqr:.2f}) at k={k_ref}. Fraction of anchors with $c_N^-\\ge 8$: {p8:.3f};
   with $c_N^-\\ge 4$: {p4:.3f}.

2. **Tangent-dimension interval** from the complement:
   $d_1\\in[{d1m:.2f},{d1p:.2f}]$ inside $S_{{{cfg.R}}}$.
   Exact $d_1$ is claimed only if $c_N$ is identified **and** complementary
   directions have independent first-order support.

3. **Does the inverse support $(c_N=8,d_1=12)$?**
   {'Yes, as a certified prefix at the median.' if (np.isfinite(cNm) and abs(cNm-8)<=1.5 and abs(d1p-12)<=1.5) else 'Not as a positively certified identification.'}
   Pr[cN>=8]={p8:.3f}.

4. **Does it support $(c_N=4,d_1=16)$?**
   {'Yes, as a certified prefix at the median.' if (np.isfinite(cNm) and abs(cNm-4)<=1.5 and abs(d1p-16)<=1.5) else 'Not as a positively certified identification.'}
   Median d1+={d1p:.2f}.

5. **Approximately flat normals** (prefix count median): {n_flat_c:.2f}.
   Overall direction fraction labelled flat: {n_flat:.3f}.

6. **Curvature-active normals** (prefix count median): {n_curv_c:.2f}.
   Overall direction fraction labelled curvature-active: {n_curv:.3f}.

7. **Jointly inferred $q_2$** at the q=8 candidate: median **{q2_8:.2f}**.
   At the q=4 candidate: {q2_4:.2f}.

8. **Does $q_2=1$ persist after joint tangent/normal inference?**
   {'Yes, within ±1.5 of 1 at the q=8 (d1=12) candidate.' if q2_persist else 'Not stably at 1, or the q=8 candidate was not recovered.'}
   Prior explicit $q_{{2\\mid 12}}$ expected 1.

9. **Fraction of $E_4$ in the learned normal space:** median **{e4n:.3f}**.

10. **Previously unresolved ~85% tail residual:**
    $E_4$ energy classified as normal ≈ {e4n:.3f}; complementary (tangent / thickness / unresolved)
    ≈ {1.0-e4n if np.isfinite(e4n) else float('nan'):.3f}.
    This is a geometric classification; `mag_r_desi` was not used to assign labels.

11. **Normal-bundle recurrence:** split overlap is in figure 04; adjacent-scale persist is in
    `normal_classification.parquet`. Gauss-map median transported overlap **{gauss_ov:.3f}**,
    Weingarten cosine {weing:.3f}.

12. **Matched synthetics, weak tangents vs normals:** evaluation table
    `synthetic_evaluation.csv`. Call rate for (12,8) on d12 families: {rec12:.3f};
    (16,4) on d16 families: {rec16:.3f}. Isotropic median d1+={iso_d1:.2f}.

13. **Can the method output dimensions other than 12?** **{not_only12}**.
    If false, the ViT-B number is not interpreted as architecture-independent.

14. **Old rank-16 mag_r association, after freeze:** {assoc_txt}.
    Probe associations did not change thresholds or the selected normal model.

## Strongest possible claim

{strongest}

## Geometric refine

Spectral initialization is the selection basis. Refine comparison (q=4 and q=8 on a
parity subset) is in `geometric_refine.parquet`
(n={len(refine) if refine is not None else 0}). Thresholds were not updated from refine.

## Scale exponents

Median raw amplitude exponent {_med(scal, 'amp_exp_raw'):.2f}; corrected
{_med(scal, 'amp_exp_corr'):.2f}. Unresolved exponents are not treated as negative evidence.

## Primary label rationale

`{primary}` from median $c_N^-$={cNm:.2f}, $d_1^+$={d1p:.2f}, $d_1^-$={d1m:.2f},
$q_2$={q2_8:.2f}, $E_4$ normal fraction={e4n:.3f}, synth_not_only12={not_only12}.

ViT-B only. Do not run cross-model replication until this report has been reviewed.
"""
    (out / "REPORT.md").write_text(text)
