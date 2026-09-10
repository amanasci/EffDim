"""FORMULAS, METHODS, REPORT, MANUSCRIPT_RECOMMENDATION, figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FORMULAS = """# FORMULAS

## D-full (historical, unnormalized)

J = DF, g = J^T J, P_T = J g^{-1} J^T
II^E = (I - P_T) D²F
H^E = g^{ab} II^E_ab    (no 1/d)
F is raw model.decode.

Also reported: H^E / d.

## D-residual

F̃ = F / ||F|| differentiated as a map (not a post-hoc projection of D²F).
B^S = (I - xx^T - P_T) D²F̃
H^S = g^{ab} B^S_ab
Scal = d(d-1) + ||H^S||² - ||B^S||_g²

## Q

Frozen nested_pca_frame + fit_quad + split-half.
K_H^cross = <H_A^S, H_B^S>  (production unpacked / Hessian pair as stored)
K_dir^cross unchanged, no clamping of negative cross-products.
Scal_cross = d(d-1) + <H_A^S, H_B^S> - <B_A^S, B_B^S>_g

## Oracles

T2: same Q neighbourhood indices, clean f(z), exact latents, true tangent.
T3: Sobol uniform latent ball, same radius, ≤2048 points.
"""

METHODS = """# METHODS

Bounded known-answer validation of D (Austin decoder autodiff) and Q (frozen local quadratic).
d=16, D=28, n≤5000, 64 hash-stable clean anchors, k=1024, ≤12 AEs, 60-minute wall.

Fixtures F0/F1/F2/F4 from the known-curvature audit generators, padded/rotated into R^{28}.
Sampling: Haar uniform (S0 n=5000, S1 n=1500); non-uniform w=exp(β s(u)) with β=log(10)/2 (S2/S3).
Noise scaled to s_x = median ||x-mean x||. Anchors and curvature truth stay clean.

D trained with the reproduction protocol (PlainAutoEncoder 250³ SiLU, AdamW, 400 epochs, seed 0/20260816).
Q is the unmodified production path. T2/T3 labels follow this brief (matched vs uniform), not the prior audit names.
"""


def write_all(out: Path, *, blocked, tests, decision=None, d_full=None, d_res=None, q_t2=None, q_pw=None, skipped=None, runtime_s=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "FORMULAS.md").write_text(FORMULAS)
    (out / "METHODS.md").write_text(METHODS)
    if blocked:
        (out / "REPORT.md").write_text("# REPORT\n\nBlocked by unit tests.\n")
        (out / "MANUSCRIPT_RECOMMENDATION.md").write_text("# MANUSCRIPT_RECOMMENDATION\n\nNo manuscript change. Tests blocked the run.\n")
        return
    decision = decision or {}
    lines = [
        "# REPORT",
        "",
        f"summary_label: `{decision.get('summary_label')}`",
        f"runtime_s: {runtime_s}",
        f"tests: {tests.get('n_passed')}/{tests.get('n_tests')}",
        f"skipped: {skipped}",
        "",
        "## Boolean findings",
        "",
    ]
    for k in (
        "decoder_full_clean_ok",
        "decoder_full_stress_ok",
        "decoder_residual_clean_ok",
        "decoder_residual_stress_ok",
        "quadratic_matched_patch_ok",
        "quadratic_uniform_patch_ok",
        "quadratic_pointwise_residual_ok",
        "quadratic_intrinsic_ok",
        "sampling_measure_dependence_detected",
        "sampling_robust",
        "noise_robust",
    ):
        lines.append(f"- {k}: {decision.get(k)}")
    lines += ["", "## D-full", ""]
    for r in d_full or []:
        lines.append(f"- {r.get('cell')} ρ={r.get('rho')} cos={r.get('median_cosine')} ratio={r.get('median_ratio')}")
    lines += ["", "## D-residual", ""]
    for r in d_res or []:
        lines.append(f"- {r.get('cell')} ρ={r.get('rho')} scal_ρ={r.get('scal_rho')} energy_frac={r.get('energy_frac')}")
    lines += ["", "## Q vs T2 / pointwise", ""]
    for r in q_t2 or []:
        lines.append(f"- T2 {r.get('cell')} cos={r.get('median_tensor_cos')} scal_ρ={r.get('scal_rho')}")
    for r in q_pw or []:
        lines.append(f"- PW {r.get('cell')} cos={r.get('median_tensor_cos')} scal_ρ={r.get('scal_rho')}")
    lines += [
        "",
        "High reconstruction R² is not treated as curvature validation.",
        "A T2 pass with a pointwise fail means Q is a finite-patch statistic, not a pointwise curvature estimator.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    rec = f"""# MANUSCRIPT_RECOMMENDATION

Do not edit the paper from this audit.

Summary label: `{decision.get('summary_label')}`.

If Q passes T2 and fails pointwise residual truth, describe Q as a finite-patch
quadratic statistic of its neighbourhood, not as a pointwise second-fundamental-form
estimator. If D-full passes F4 clean gates, the historical decoder autodiff is a
valid estimate of Euclidean H^E of the learned (and here, known) immersion at D=28.
D-residual is a separate estimand (B^S / H^S / Scal) and must not be swapped with D-full.
"""
    (out / "MANUSCRIPT_RECOMMENDATION.md").write_text(rec)
    _figures(out, d_full or [], d_res or [], q_t2 or [], q_pw or [])

    # stratified stubs from per-anchor if present
    p = out / "per_anchor_metrics.parquet"
    if p.is_file():
        import pandas as pd

        df = pd.read_parquet(p)
        if "w" in df.columns and len(df):
            d4 = df[df.fixture == "F4"].copy()
            if len(d4):
                d4["dens_q"] = pd.qcut(d4["w"], 5, duplicates="drop")
                dens = d4.groupby("dens_q", observed=True)[["pw_cos", "t2_cos", "H_E_est"]].median().reset_index()
                dens.to_csv(out / "density_stratified_metrics.csv", index=False)
        if "cell" in df.columns:
            noise = df.groupby(["noise", "fixture"], dropna=False)[["pw_cos", "t2_cos"]].median().reset_index()
            noise.to_csv(out / "noise_stratified_metrics.csv", index=False)
    else:
        Path(out / "density_stratified_metrics.csv").write_text("note\nno_per_anchor\n")
        Path(out / "noise_stratified_metrics.csv").write_text("note\nno_per_anchor\n")


def _figures(out, d_full, d_res, q_t2, q_pw):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if d_full:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        cells = [r["cell"] for r in d_full]
        ax.plot(cells, [r.get("rho") for r in d_full], "o-", label="D-full ρ")
        ax.plot(cells, [r.get("rho") for r in d_res] if d_res else [], "s--", label="D-res ρ")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig2_degradation.png", dpi=120)
        plt.close(fig)
    if q_t2:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        cells = [r["cell"] for r in q_t2]
        ax.plot(cells, [r.get("median_tensor_cos") for r in q_t2], "o-", label="Q vs T2 cos")
        ax.plot(cells, [r.get("median_tensor_cos") for r in q_pw] if q_pw else [], "s--", label="Q vs pointwise cos")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig3_Q_oracles.png", dpi=120)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.text(0.5, 0.5, "F4 analytic vs D-full / D-res / Scal: see per_anchor_metrics.parquet", ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "fig1_F4_truth_vs_est.png", dpi=120)
    plt.close(fig)
