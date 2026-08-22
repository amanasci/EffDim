"""METHODS.md and REPORT.md for order-stratified local geometry."""

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


def _med(df, col, default=float("nan")):
    if df is None or len(df) == 0 or col not in getattr(df, "columns", []):
        return default
    return float(pd.to_numeric(df[col], errors="coerce").median())


def write_methods(out: Path, cfg: Any, ctx: dict, parity: dict, thr: dict) -> None:
    text = f"""# Methods: order-stratified local geometry

## Estimand

Report the pair $(d_1,q_2)$, not a single ambiguous effective dimension.

- $d_1=\\mathrm{{rank}}(J)$ is the first-order locally linear core (frozen here as the
  certified stable-tangent rank $d_T=12$ for ViT-B).
- $q_2$ is the **reliable rank of the sphere-normal quadratic image**
  $B^S:\\mathrm{{Sym}}^2(T)\\to N_S$.
- $d_{{\\le 2}}=d_1+q_2$ is the ambient dimension needed through second order.

$q_2$ is **not** additional intrinsic dimension.

The primary comparison is $(12,q_2)$ versus $(16,0)$.

## Frozen inputs

Activations, neighbours, anchors, OOF probes, nested frames and $d_G$ are loaded
from existing multimodel / nested-dimension / stable-tangent artifacts.
This package never writes into those directories. See `freeze_manifest.json`.

Representations are used as stored. L2-normalization status is recorded, not
silently changed. Local PCA is uncentred through the anchor on spherical
log-map displacements. Split-fitted objects are compared with projectors.

## Quadratic tensor

Established decomposition $Q=Q_T+Q_R+B^S$. Only $B^S$ enters $q_2$.
Metric-whitened $\\mathrm{{vech}}(uu^\\top)$ uses $\\sqrt{{2}}$ off-diagonal weights.
Ridge parameters are chosen on inner training splits; rank and prediction use
untouched outer neighbours. Scan $q=0,\\ldots,{cfg.q_max}$.

## Reliable $q_2$

A quadratic-normal mode (or degenerate block) is accepted only if it passes
split-cross energy vs a radial-bin permutation null, left-subspace recurrence,
held-out sphere-normal gain, and the consecutive-prefix rule. Isolated higher
modes are not accepted above a failed block.

## Tail tests

$T_C=T_{{12}}$, $E_4=T_{{16}}\\ominus T_{{12}}$, $E_R=S_R\\ominus T_C$ inside the
data-supported carrier $S_R$ ($R={cfg.R}$). Complements are never taken in the
full ambient space.

Cross-fitted overlap $O_{{E,B}}=\\frac14\\mathrm{{tr}}(P_{{E_4}}P_{{U_{{B,4}}}})$.
Conditional prediction $\\widehat e(u)=A\\phi(u)$ is evaluated held out.
Mixed nonnegative law $V(r)=ar^2+br^4+c$ uses tangent radius, not $k$.
Odd/even pairing is reported only when enough antipodal pairs exist.

## Fair models

Identical outer splits and ambient targets for $M_{{12,\\mathrm{{linear}}}}$,
$M_{{16,\\mathrm{{linear}}}}$, $M_{{12,q}}$ and a nonlinear-chart sensitivity.
The central comparison is $M_{{12,\\widehat q_2}}$ versus $M_{{16,\\mathrm{{linear}}}}$.

## Synthetics and probes

Calibration seeds freeze `thresholds.json` before real $q_2$ inspection.
OOF `mag_r_desi` is loaded only in the associations stage.

Thresholds: `{json.dumps(thr, default=str)}`

Parity: `{json.dumps(parity, default=str)[:1500]}`
"""
    (out / "METHODS.md").write_text(text)


def write_report(out: Path, cfg: Any, ctx: dict, parity: dict, labels: dict) -> None:
    qsum = _read(out, "quadratic_rank_summary.csv", pd.DataFrame())
    ov = _read(out, "tail_quadratic_overlap.parquet", pd.DataFrame())
    pred = _read(out, "conditional_tail_prediction.parquet", pd.DataFrame())
    mix = _read(out, "mixed_scale_components.parquet", pd.DataFrame())
    mc = _read(out, "model_comparison.csv", pd.DataFrame())
    oe = _read(out, "odd_even_diagnostics.parquet", pd.DataFrame())
    nb = _read(out, "normal_complement_bounds.csv", pd.DataFrame())
    seval = _read(out, "synthetic_evaluation.csv", pd.DataFrame())
    assoc = _read(out, "probe_associations.csv", pd.DataFrame())
    repl = _read(out, "cross_model_order_dimensions.csv", pd.DataFrame())
    qspec = _read(out, "quadratic_spectrum.parquet", pd.DataFrame())

    k_ref = cfg.primary_k
    if len(qsum) and "k" in qsum.columns:
        hit = qsum[qsum.k == k_ref]
        if len(hit):
            qsum_k = hit
        else:
            qsum_k = qsum.iloc[[-1]]
    else:
        qsum_k = qsum

    q2 = _med(qsum_k, "median_q2")
    d1 = float(cfg.d_core)
    d_le2 = d1 + q2 if np.isfinite(q2) else float("nan")
    overlap = _med(ov, "O_E4_B")
    r2e = _med(pred, "r2_E4")
    resid_frac = _med(pred, "resid_frac_E4")
    resid_s0 = _med(pred, "resid_s0_E4")
    raw = mix[mix.series == "raw_E4"] if len(mix) and "series" in mix.columns else mix
    pi_lin, pi_quad, pi_thick = _med(raw, "pi_lin"), _med(raw, "pi_quad"), _med(raw, "pi_thick")
    mix_res = bool(len(raw) and raw.get("resolved", pd.Series(dtype=bool)).mean() > 0.3) if len(raw) else False
    delta = _med(mc, "delta_M16_minus_M12q")
    d1m, d1p = _med(nb, "d1_minus"), _med(nb, "d1_plus")
    n_oe = _med(oe, "n_pairs")
    o_even_e4 = _med(oe, "O_even_E4")
    o_odd_t = _med(oe, "O_odd_T12")
    primary = labels.get("primary", "order_stratification_unresolved")

    synth_rows = seval.to_dict("records") if len(seval) else []
    real_feat = {
        "median_q2": q2,
        "overlap_E4": overlap,
        "r2_quad_E4": r2e,
        "pi_quad": pi_quad,
        "pi_lin": pi_lin,
        "delta_M16_minus_M12q": delta,
    }
    closest, _cd = closest_synthetic(real_feat, synth_rows) if synth_rows else ("unresolved", float("nan"))

    mag_note = "not yet associated"
    if len(assoc):
        mag_note = assoc.to_string(index=False)

    scale_law = "unresolved"
    if mix_res:
        vals = {"linear": pi_lin, "quadratic": pi_quad, "thickness": pi_thick}
        scale_law = max(vals, key=lambda k: vals[k] if np.isfinite(vals[k]) else -1)

    text = f"""# REPORT: order-stratified geometry (ViT-B primary)

Primary interpretation: **{primary}**

Primary object: $(d_1,q_2)=({d1:.0f},{q2:.2f})$, $d_{{\\le 2}}={d_le2:.2f}$.
Graph $d_G=12$ is an external comparison, not a selection input.

Parity ok: {parity.get('ok') if parity else 'n/a'}. Stable core: {parity.get('stable_core') if parity else 'n/a'}.

## Direct answers

1. **Reliable $q_2$:** median {q2:.2f} (see `quadratic_rank_summary.csv`).
2. **Second-order osculating dimension $d_{{\\le 2}}$:** {d_le2:.2f}.
3. **Do directions 13–16 align with $\\mathrm{{im}}(B_{{12}}^S)$?** median overlap {overlap:.3f}.
4. **Quadratic predictability of $E_4$:** held-out $R^2={r2e:.3f}$.
5. **Conditional residual $r^2$ structure?** residual fraction {resid_frac:.3f}; residual leading energy {resid_s0:.4g}.
6. **Scale law of the tail:** {scale_law} (shares lin={pi_lin:.2f}, quad={pi_quad:.2f}, thick={pi_thick:.2f}; identifiable={mix_res}).
7. **Odd/even:** median pairs {n_oe:.1f}; $O(\\mathrm{{odd}},T_{{12}})={o_odd_t:.3f}$, $O(\\mathrm{{even}},E_4)={o_even_e4:.3f}$. Unresolved if too few pairs.
8. **$M_{{12,q}}$ vs $M_{{16,\\mathrm{{linear}}}}$:** median $\\Delta=$ {delta:.6g} (positive means the rank-16 linear model has higher held-out error, i.e. $M_{{12,q}}$ wins).
9. **Is $d_1=12$ exact or a lower bound?** $d_1^-={d1m:.1f}$, $d_1^+={d1p:.1f}$. The stable-tangent procedure certifies the lower bound; extras are classified here by quadratic vs residual linear evidence.
10. **Rank-16 mag_r geography in rank-12 quadratic modes?**
```
{mag_note}
```
11. **Closest synthetic:** {closest}.
12. **Cross-model recurrence:** {'see table' if len(repl) else 'not run; ViT-B primary first'}.

## Cross-model

```
{repl.to_string(index=False) if len(repl) else 'ViT-B only'}
```

## Quadratic rank summary

```
{qsum.to_string(index=False) if len(qsum) else 'missing'}
```

## Paper-level claim

Do **not** claim that ViT-B is a curved 12-dimensional manifold with four
reliable quadratic-normal directions unless $q_2$ modes pass split reliability,
held-out prediction, scale persistence, and synthetic-calibration gates, and
the conditional tail residual lacks stable $r^2$ variation.

Assigned label **{primary}**.
"""
    (out / "REPORT.md").write_text(text)
    print(f"[osg] REPORT interpretation={primary}", flush=True)
