"""Required figures for quadratic predictive dimension. ASCII labels only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load(out: Path, name: str):
    p = out / name
    if not p.exists():
        return None
    if name.endswith(".parquet"):
        return pd.read_parquet(p)
    if name.endswith(".json"):
        import json

        return json.loads(p.read_text())
    return pd.read_csv(p)


def _save(fig, figdir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(figdir / name, dpi=140)
    plt.close(fig)


def write_figures(out: Path, cfg) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    curves = _load(out, "aggregate_risk_curves.csv")
    plat = _load(out, "plateau_bootstrap.csv")
    tail = _load(out, "tail_adequacy.csv")
    scale = _load(out, "scale_sensitivity.csv")
    seval = _load(out, "synthetic_evaluation.csv")
    raw = _load(out, "per_anchor_metrics.parquet")
    if raw is None:
        raw = _load(out, "per_anchor_metrics.csv")

    def has(df, *cols):
        return df is not None and len(df) and all(c in df.columns for c in cols)

    # 1. Linear vs quadratic held-out NMSE by d
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(curves, "d", "nmse_quad_med", "nmse_lin_med"):
        ax.plot(curves.d, curves.nmse_lin_med, "o--", color="C0", label="linear")
        ax.plot(curves.d, curves.nmse_quad_med, "s-", color="C1", label="quadratic closest-point")
        if has(curves, "nmse_quad_lo", "nmse_quad_hi"):
            ax.fill_between(curves.d, curves.nmse_quad_lo, curves.nmse_quad_hi, color="C1", alpha=0.2)
        if has(curves, "nmse_lin_lo", "nmse_lin_hi"):
            ax.fill_between(curves.d, curves.nmse_lin_lo, curves.nmse_lin_hi, color="C0", alpha=0.15)
        ax.axvline(12, color="0.5", ls=":", label="d=12")
        ax.axvline(16, color="0.5", ls="--", label="d=16")
        ax.legend(fontsize=8)
    ax.set_xlabel("coordinate dimension d")
    ax.set_ylabel("held-out NMSE")
    ax.set_title("Linear vs quadratic held-out NMSE")
    _save(fig, figdir, "01_nmse_linear_vs_quadratic.png")

    # 2. Total R^2 with 90/95/99 lines
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(curves, "d", "r2_quad_med"):
        ax.plot(curves.d, curves.r2_quad_med, "s-", color="C1", label="quadratic R^2")
        if has(curves, "r2_quad_lo", "r2_quad_hi"):
            ax.fill_between(curves.d, curves.r2_quad_lo, curves.r2_quad_hi, color="C1", alpha=0.2)
        if has(curves, "r2_lin_pooled") or "nmse_lin_med" in curves.columns:
            ax.plot(curves.d, 1.0 - curves.nmse_lin_med, "o--", color="C0", label="linear R^2")
        for y, lab in ((0.90, "90%"), (0.95, "95%"), (0.99, "99%")):
            ax.axhline(y, ls=":", color="k", alpha=0.6)
            ax.text(curves.d.min(), y + 0.005, lab, fontsize=8)
        ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("coordinate dimension d")
    ax.set_ylabel("held-out R^2")
    ax.set_title("Total explained energy with adequacy lines")
    _save(fig, figdir, "02_r2_adequacy_lines.png")

    # 3. Incremental gains with paired intervals
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(curves, "d", "delta_quad_med"):
        ax.axhline(0.0, color="0.5", lw=0.8)
        yerr = None
        if has(curves, "delta_lo", "delta_hi"):
            lo = curves.delta_quad_med - curves.delta_lo
            hi = curves.delta_hi - curves.delta_quad_med
            yerr = np.vstack([lo.clip(lower=0), hi.clip(lower=0)])
        ax.errorbar(curves.d, curves.delta_quad_med, yerr=yerr, fmt="s-", color="C1", label="Delta NMSE")
        ax.legend(fontsize=8)
    ax.set_xlabel("d")
    ax.set_ylabel("NMSE_{d-1} - NMSE_d")
    ax.set_title("Incremental held-out gains (paired bootstrap)")
    _save(fig, figdir, "03_incremental_gains.png")

    # 4. Fixed-coordinate vs closest-point
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(curves, "d", "nmse_quad_med", "nmse_quad_fixed_med"):
        ax.plot(curves.d, curves.nmse_quad_fixed_med, "o--", label="fixed-coordinate u0")
        ax.plot(curves.d, curves.nmse_quad_med, "s-", label="closest-point")
        ax.legend(fontsize=8)
    ax.set_xlabel("d")
    ax.set_ylabel("held-out NMSE")
    ax.set_title("Fixed-coordinate vs closest-point error")
    _save(fig, figdir, "04_fixed_vs_closest.png")

    # 5. Unrestricted vs normal-only
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(curves, "d", "nmse_quad_med", "nmse_quadN_med"):
        ax.plot(curves.d, curves.nmse_quad_med, "s-", label="unrestricted B")
        ax.plot(curves.d, curves.nmse_quadN_med, "o--", label="normal-only B^N")
        ax.legend(fontsize=8)
    ax.set_xlabel("d")
    ax.set_ylabel("held-out NMSE")
    ax.set_title("Unrestricted vs normal-only quadratic models")
    _save(fig, figdir, "05_unrestricted_vs_normal.png")

    # 6. Tail R^2
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(tail, "d"):
        for col, lab in (
            ("r2_T12_med", "T12"),
            ("r2_E4_med", "E4"),
            ("r2_U4_med", "U4"),
            ("r2_U8_med", "U8"),
        ):
            if col in tail.columns:
                ax.plot(tail.d, tail[col], "o-", label=lab)
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.legend(fontsize=8)
    ax.set_xlabel("d")
    ax.set_ylabel("held-out component R^2")
    ax.set_title("Tail reconstruction: T12, E4, U4, U8")
    _save(fig, figdir, "06_tail_r2.png")

    # 7. Ridge penalties and effective df
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(curves, "d", "lam_med"):
        ax.semilogy(curves.d, np.clip(curves.lam_med, 1e-12, None), "s-", color="C0", label="selected lambda")
        ax.set_ylabel("ridge lambda")
        ax2 = ax.twinx()
        if has(curves, "df_frac_med"):
            ax2.plot(curves.d, curves.df_frac_med, "o--", color="C1", label="df / p")
            ax2.set_ylabel("effective df fraction")
        ax.legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("d")
    ax.set_title("Selected ridge and effective degrees of freedom")
    _save(fig, figdir, "07_ridge_and_df.png")

    # 8. Train vs held-out
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(curves, "d", "train_nmse_med", "nmse_quad_med"):
        ax.plot(curves.d, curves.train_nmse_med, "o--", label="train (fixed-coord)")
        ax.plot(curves.d, curves.nmse_quad_med, "s-", label="held-out closest-point")
        ax.legend(fontsize=8)
    ax.set_xlabel("d")
    ax.set_ylabel("NMSE")
    ax.set_title("Training vs held-out error")
    _save(fig, figdir, "08_train_vs_heldout.png")

    # 9. Anchor-level plateau distribution
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(plat, "dQ"):
        bins = np.arange(3.5, 21.5, 1.0)
        ax.hist(plat.dQ.dropna(), bins=bins, alpha=0.7, label="quadratic plateau")
        if "dL" in plat.columns:
            ax.hist(plat.dL.dropna(), bins=bins, alpha=0.45, label="linear plateau")
        ax.legend(fontsize=8)
    ax.set_xlabel("per-anchor plateau d")
    ax.set_ylabel("count")
    ax.set_title("Anchor-level plateau distribution")
    _save(fig, figdir, "09_anchor_plateau_hist.png")

    # 10. Cross-scale plateau and adequacy
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(scale, "k", "dQ"):
        ks = sorted(scale.k.unique())
        meds = [float(scale[scale.k == k].dQ.median()) for k in ks]
        q25 = [float(scale[scale.k == k].dQ.quantile(0.25)) for k in ks]
        q75 = [float(scale[scale.k == k].dQ.quantile(0.75)) for k in ks]
        ax.errorbar(ks, meds, yerr=[np.array(meds) - np.array(q25), np.array(q75) - np.array(meds)], fmt="s-")
        if "r2_best" in scale.columns:
            ax2 = ax.twinx()
            rmed = [float(scale[scale.k == k].r2_best.median()) for k in ks]
            ax2.plot(ks, rmed, "o--", color="C1", label="best R^2")
            ax2.set_ylabel("best held-out R^2")
    ax.set_xlabel("neighbourhood k")
    ax.set_ylabel("median quadratic plateau")
    ax.set_title("Cross-scale plateau and adequacy")
    _save(fig, figdir, "10_scale_sensitivity.png")

    # 11. Synthetic recovery matrix
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if has(seval, "kind", "dQ", "true_d"):
        kinds = list(seval.kind.unique())
        rec = []
        tru = []
        for knd in kinds:
            g = seval[seval.kind == knd]
            rec.append(float(g.dQ.median()))
            tru.append(float(g.true_d.median()))
        x = np.arange(len(kinds))
        ax.bar(x - 0.18, tru, 0.36, label="true d")
        ax.bar(x + 0.18, rec, 0.36, label="estimated dQ")
        ax.set_xticks(x)
        ax.set_xticklabels(kinds, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8)
        ax.axhline(12, color="0.5", ls=":")
    ax.set_ylabel("dimension")
    ax.set_title("Synthetic recovery and failure-mode matrix")
    _save(fig, figdir, "11_synthetic_recovery.png")

    # 12. Paired d=12 vs d=16
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    if has(raw, "sample_id", "d", "quad_close_nmse"):
        k_ref = getattr(cfg, "primary_k", 2048)
        g = raw[raw.k == k_ref] if "k" in raw.columns else raw
        loc = g.groupby(["sample_id", "d"], as_index=False).mean(numeric_only=True)
        a = loc[loc.d == 12].set_index("sample_id")
        b = loc[loc.d == 16].set_index("sample_id")
        m = a.join(b, lsuffix="_12", rsuffix="_16", how="inner")
        if "quad_close_nmse_12" in m.columns and len(m):
            x = m.quad_close_nmse_12.to_numpy()
            y = m.quad_close_nmse_16.to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                ax.scatter(x[mask], y[mask], s=12, alpha=0.6)
                lo = float(np.min([x[mask].min(), y[mask].min()]))
                hi = float(np.max([x[mask].max(), y[mask].max()]))
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
                ax.set_xlabel("NMSE at d=12")
                ax.set_ylabel("NMSE at d=16")
    ax.set_title("Paired d=12 vs d=16 (anchors)")
    _save(fig, figdir, "12_paired_d12_d16.png")

    # 13. Residual vs neighbourhood radius
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if has(raw, "k", "quad_close_nmse"):
        med = raw.groupby("k")["quad_close_nmse"].median()
        ax.plot(med.index, med.values, "s-")
        if "d" in raw.columns:
            for dmark, ls in ((12, "-"), (16, "--")):
                sub = raw[raw.d == dmark]
                if len(sub):
                    md = sub.groupby("k")["quad_close_nmse"].median()
                    ax.plot(md.index, md.values, ls, label=f"d={dmark}")
            if any(len(raw[raw.d == dmark]) for dmark in (12, 16) if "d" in raw.columns):
                ax.legend(fontsize=8)
    ax.set_xlabel("neighbourhood k")
    ax.set_ylabel("median held-out NMSE")
    ax.set_title("Residual error vs neighbourhood radius")
    _save(fig, figdir, "13_residual_vs_radius.png")

    # 14. Optimizer diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    if has(curves, "d", "boundary_frac"):
        axes[0].plot(curves.d, curves.boundary_frac, "s-")
        axes[0].set_xlabel("d")
        axes[0].set_ylabel("boundary-hit fraction")
        axes[0].set_title("Closest-point boundary hits")
    if has(curves, "d", "mean_n_iter"):
        axes[1].plot(curves.d, curves.mean_n_iter, "o-")
        axes[1].set_xlabel("d")
        axes[1].set_ylabel("mean GN iterations")
        axes[1].set_title("Optimizer iterations")
    _save(fig, figdir, "14_optimizer_diagnostics.png")
