"""Figures for the curvature–probe rank sweep. ASCII labels only."""

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
    fig.savefig(figdir / name, dpi=150)
    plt.close(fig)


def write_figures(out: Path, cfg) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    assoc = _load(out, "dimension_associations.csv")
    allc = _load(out, "controlled_dimension_associations.csv")
    boot = _load(out, "bootstrap_results.csv")
    perm = _load(out, "permutation_results.csv")
    rel = _load(out, "curvature_reliability.csv")
    ve = _load(out, "variance_explained.csv")
    vx = _load(out, "variance_threshold_crossings.csv")
    scale = _load(out, "scale_sensitivity.csv")
    synth = _load(out, "synthetic_validation.csv")
    miss = _load(out, "cache/missingness.csv")
    peak = _load(out, "cache/peak_ctl_boot.csv")
    env = _load(out, "cache/perm_envelopes.csv")
    extra = _load(out, "cache/bootstrap_extra.json") or {}

    def kh(df):
        if df is None or not len(df):
            return df
        if "mask" in df.columns:
            df = df[df["mask"] == "complete_case"]
        if "metric" in df.columns:
            df = df[df.metric == "K_H"]
        return df.sort_values("d")

    a = kh(assoc)
    failed = set()
    if rel is not None and len(rel) and "fail_reliability" in rel.columns:
        failed = set(int(d) for d in rel.loc[rel.fail_reliability, "d"])

    d85 = None
    if vx is not None and len(vx):
        hit = vx[np.isclose(vx.tau, 0.85)]
        if len(hit) and str(hit.iloc[0].d_tau) not in ("not_reached", "nan"):
            try:
                d85 = int(float(hit.iloc[0].d_tau))
            except (TypeError, ValueError):
                d85 = None

    def marks(ax):
        ax.axvline(12, color="0.5", ls=":", lw=0.9)
        ax.axvline(16, color="0.5", ls="--", lw=0.9)
        if d85 is not None:
            ax.axvline(d85, color="C3", ls="-.", lw=0.9)

    def fade(ax, d, y, **kw):
        c = "0.6" if int(d) in failed else kw.pop("color", "C0")
        ax.plot([d], [y], "s" if int(d) in failed else "o", color=c, **kw)

    # 1. four-panel primary
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))
    ax = axes[0, 0]
    if boot is not None and len(boot):
        ax.fill_between(boot.d, boot.raw_sim_lo, boot.raw_sim_hi, color="C0", alpha=0.12, label="simultaneous")
        ax.fill_between(boot.d, boot.raw_lo, boot.raw_hi, color="C0", alpha=0.25, label="pointwise 95%")
        ax.plot(boot.d, boot.raw, "o-", color="C0")
        for _, r in boot.iterrows():
            if int(r.d) in failed:
                ax.plot(r.d, r.raw, "s", color="0.45", ms=7)
    marks(ax)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("A. Raw Spearman(K_H, mag_r_desi)")
    ax.set_ylabel("rho raw")
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    if boot is not None and len(boot):
        ax.fill_between(boot.d, boot.ctl_sim_lo, boot.ctl_sim_hi, color="C1", alpha=0.12)
        ax.fill_between(boot.d, boot.ctl_lo, boot.ctl_hi, color="C1", alpha=0.25)
        ax.plot(boot.d, boot.controlled, "o-", color="C1")
        for _, r in boot.iterrows():
            if int(r.d) in failed:
                ax.plot(r.d, r.controlled, "s", color="0.45", ms=7)
    marks(ax)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("B. Controlled / partial Spearman")
    ax.set_ylabel("rho controlled")
    ax = axes[1, 0]
    if rel is not None and len(rel):
        ax.plot(rel.d, rel.dS_med, "o-", color="C2", label="held-out Delta_S")
        ax2 = ax.twinx()
        ax2.plot(rel.d, rel.R_H_med, "s--", color="C4", label="split R_H")
        ax.set_ylabel("Delta_S")
        ax2.set_ylabel("R_H")
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="upper right")
    ax.set_title("C. Curvature reliability (no probe)")
    ax.set_xlabel("d")
    ax = axes[1, 1]
    if ve is not None and len(ve):
        ax.plot(ve.d, ve.r2_L_pooled, "o-", color="C5")
        if "r2_lo" in ve.columns:
            ax.fill_between(ve.d, ve.r2_lo, ve.r2_hi, color="C5", alpha=0.2)
        for t, ls in ((0.80, ":"), (0.825, ":"), (0.85, "--"), (0.875, ":"), (0.90, ":")):
            ax.axhline(t, color="k", ls=ls, lw=0.7, alpha=0.7)
    marks(ax)
    ax.set_ylim(0.65, 1.0)
    ax.set_title("D. Held-out linear R^2_L (post-hoc 85% marked)")
    ax.set_xlabel("d")
    ax.set_ylabel("R^2_L")
    _save(fig, figdir, "01_four_panel_rank_sweep.png")

    # 2. envelopes
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    if boot is not None and env is not None and len(boot) and len(env):
        axes[0].plot(boot.d, boot.raw, "o-")
        axes[0].plot(env.d, env.raw_null95, "--", label="+/- null 95%")
        axes[0].plot(env.d, -env.raw_null95, "--")
        axes[1].plot(boot.d, boot.controlled, "o-", color="C1")
        axes[1].plot(env.d, env.ctl_null95, "--", color="C1")
        axes[1].plot(env.d, -env.ctl_null95, "--", color="C1")
    for ax, t in zip(axes, ("Raw + perm envelope", "Controlled + perm envelope")):
        ax.axhline(0, color="k", lw=0.5)
        marks(ax)
        ax.set_title(t)
        ax.set_xlabel("d")
    _save(fig, figdir, "02_null_envelopes.png")

    # 3. heatmap controlled d x k
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if scale is not None and len(scale):
        pv = scale.pivot_table(index="d", columns="k", values="controlled", aggfunc="mean")
        im = ax.imshow(pv.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels(list(pv.index))
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels(list(pv.columns))
        fig.colorbar(im, ax=ax, label="controlled rho")
    ax.set_title("Controlled correlation by d and k")
    ax.set_xlabel("k")
    ax.set_ylabel("d")
    _save(fig, figdir, "03_heatmap_controlled.png")

    # 4. fwer heatmap (primary k only unless scale has p)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if perm is not None and len(perm):
        g = perm[perm.kind == "controlled"]
        ax.bar(g.d, 1.0 - g.p_fwer.clip(0, 1), color="C3")
        ax.axhline(0.95, ls="--", color="k")
        ax.set_ylabel("1 - p_FWER")
        ax.set_title("Familywise-adjusted support (controlled, k=2048)")
    _save(fig, figdir, "04_fwer_support.png")

    # 5. reliability
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if rel is not None and len(rel):
        ax.plot(rel.d, rel.R_H_med, "o-", label="R_H")
        ax.plot(rel.d, rel.dS_med / max(float(rel.dS_med.max()), 1e-12), "s--", label="Delta_S (scaled)")
        ax.legend(fontsize=8)
    ax.set_title("Reliability over dimension")
    ax.set_xlabel("d")
    _save(fig, figdir, "05_reliability_curve.png")

    # 6. variance crossings
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if ve is not None and len(ve):
        ax.plot(ve.d, ve.r2_L_pooled, "o-")
        for t in (0.80, 0.825, 0.85, 0.875, 0.90):
            ax.axhline(t, ls=":", color="k", lw=0.6)
    ax.set_title("Held-out variance explained and thresholds")
    _save(fig, figdir, "06_variance_explained.png")

    # 7. paired 12 vs 16
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    rawp = _load(out, "per_anchor_rank_curve.parquet")
    if rawp is not None and len(rawp):
        loc = rawp.groupby(["sample_id", "d"], as_index=False).mean(numeric_only=True)
        a12 = loc[loc.d == 12].set_index("sample_id")
        a16 = loc[loc.d == 16].set_index("sample_id")
        m = a12.join(a16, lsuffix="_12", rsuffix="_16", how="inner")
        if "K_H_cross_12" in m.columns:
            ax.scatter(m.K_H_cross_12, m.K_H_cross_16, s=10, alpha=0.5)
            ax.set_xlabel("K_H at d=12")
            ax.set_ylabel("K_H at d=16")
    ax.set_title("Paired anchors d=12 vs d=16")
    _save(fig, figdir, "07_paired_d12_d16.png")

    # 8. peak dimension hist
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    if peak is not None and len(peak) and "peak_ctl" in peak.columns:
        ax.hist(peak.peak_ctl.dropna(), bins=np.arange(7.5, 21.5, 1), color="C1")
    ax.set_title("Bootstrap peak |controlled rho| dimension")
    ax.set_xlabel("d")
    _save(fig, figdir, "08_peak_dimension_boot.png")

    # 9. perm diagnostics
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    if perm is not None and len(perm):
        g = perm[perm.kind == "controlled"]
        ax.plot(g.d, g.p_pointwise, "o--", label="pointwise p")
        ax.plot(g.d, g.p_fwer, "s-", label="FWER p")
        ax.axhline(0.05, color="k", ls=":")
        ax.legend(fontsize=8)
    ax.set_title("Permutation p-values (controlled)")
    _save(fig, figdir, "09_permutation_diagnostics.png")

    # 10. missingness
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    if miss is not None and len(miss):
        ax.bar(miss.d, miss.n_valid)
        if "n_expected" in miss.columns:
            ax.axhline(float(miss.n_expected.iloc[0]), ls="--", color="k")
    ax.set_title("Valid anchors by dimension")
    _save(fig, figdir, "10_missingness.png")

    # 11. |rho| vs reliability
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    if rel is not None and a is not None and len(rel) and len(a):
        m = a.merge(rel, on="d")
        ax.scatter(m.R_H_med, np.abs(m.controlled), c=m.d, cmap="viridis")
        ax.set_xlabel("median R_H")
        ax.set_ylabel("|controlled rho|")
    ax.set_title("Association magnitude vs reliability")
    _save(fig, figdir, "11_rho_vs_reliability.png")

    # 12. synthetics
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    if synth is not None and len(synth):
        ax.bar(np.arange(len(synth)), synth.p_global_ctl.fillna(1.0))
        ax.set_xticks(np.arange(len(synth)))
        ax.set_xticklabels(synth.kind, rotation=20, ha="right")
        ax.axhline(0.05, ls="--", color="k")
        ax.set_ylabel("global controlled p")
    ax.set_title("Synthetic familywise checks")
    _save(fig, figdir, "12_synthetic_validation.png")
