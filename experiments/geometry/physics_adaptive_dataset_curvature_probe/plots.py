"""Publication-quality figures. Out-of-range cells stay blank."""

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
    fig.savefig(figdir / name, dpi=160)
    plt.close(fig)


def write_figures(out: Path, cfg) -> None:
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    ranges = _load(out, "geometry_dimension_ranges.csv")
    lin = _load(out, "linear_risk_curves.csv")
    qs = _load(out, "quadratic_screening.csv")
    rank = _load(out, "dataset_rank_associations.csv")
    var = _load(out, "dataset_variance_associations.csv")
    perm = _load(out, "dataset_permutation_results.csv")
    boot = _load(out, "bootstrap_results.csv")
    rel = _load(out, "curvature_reliability.csv")
    scale = _load(out, "scale_sensitivity.csv")
    contr = _load(out, "replication_contrasts.csv")
    lodo = _load(out, "leave_one_dataset_out.csv")
    inv = _load(out, "dataset_inventory.csv")
    labs = _load(out, "physics_label_manifest.csv")

    # 1. geometry ranges + variance crossings
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    if ranges is not None and len(ranges):
        for i, r in ranges.iterrows():
            lo, hi = r.get("d_low_primary", r.get("d_low")), r.get("d_high_primary", r.get("d_high"))
            try:
                ax.plot([float(lo), float(hi)], [i, i], "o-", lw=2, label=str(r.dataset_id))
                for key, mk in (("d_80", "s"), ("d_85", "D"), ("d_90", "^")):
                    v = r.get(key)
                    if isinstance(v, (int, float)) and np.isfinite(float(v)):
                        ax.scatter([float(v)], [i], marker=mk)
            except (TypeError, ValueError):
                pass
        ax.set_yticks(range(len(ranges)))
        ax.set_yticklabels([str(x) for x in ranges.dataset_id])
    ax.set_xlabel("geometry-only rank")
    ax.set_title("Dataset-specific geometry intervals")
    ax.legend(fontsize=7)
    _save(fig, figdir, "01_geometry_ranges.png")

    # 2. linear / quadratic risk
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    if lin is not None and len(lin) and "r2_L_pooled" in lin.columns:
        for did, g in lin.groupby("dataset_id"):
            ax.plot(g.d, g.r2_L_pooled, "o-", label=f"{did} R2_L")
    if qs is not None and len(qs) and "quad_nmse" in qs.columns:
        for did, g in qs.groupby("dataset_id"):
            gg = g.groupby("d").mean(numeric_only=True)
            if "quad_nmse" in gg:
                ax.plot(gg.index, 1.0 - gg.quad_nmse, "s--", label=f"{did} R2_Q")
    ax.set_xlabel("d")
    ax.set_ylabel("held-out R^2")
    ax.set_title("Linear and quadratic held-out reconstruction")
    ax.legend(fontsize=7)
    _save(fig, figdir, "02_linear_quadratic_risk.png")

    # 3–4. per dataset raw / controlled curves
    if rank is None or not len(rank):
        for name in ("03_raw_rank_curves.png", "04_controlled_rank_curves.png", "05_small_multiples_rank.png", "06_heatmap_rank.png"):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "no rank associations", ha="center")
            _save(fig, figdir, name)
    else:
        for kind, col, name in (("raw", "raw", "03_raw_rank_curves.png"), ("controlled", "controlled", "04_controlled_rank_curves.png")):
            ids = list(rank.groupby(["dataset_id", "label"]).groups)
            n = max(len(ids), 1)
            fig, axes = plt.subplots(n, 1, figsize=(8.0, 2.2 * n), squeeze=False)
            for ax, (did, lab) in zip(axes[:, 0], ids):
                g = rank[(rank.dataset_id == did) & (rank.label == lab)].sort_values("d")
                ax.plot(g.d, g[col], "o-")
                if f"{'raw' if kind=='raw' else 'ctl'}_lo" in g.columns or (kind == "raw" and "raw_lo" in g.columns):
                    lo = g["raw_lo" if kind == "raw" else "ctl_lo"]
                    hi = g["raw_hi" if kind == "raw" else "ctl_hi"]
                    if lo.notna().any():
                        ax.fill_between(g.d, lo, hi, alpha=0.2)
                ax.axhline(0, color="0.5", lw=0.6)
                ax.set_ylabel(r"$\rho$")
                ax.set_title(f"{did} / {lab} ({kind})")
            axes[-1, 0].set_xlabel("d")
            _save(fig, figdir, name)
        # 5 small multiples
        keys = list(rank.groupby(["dataset_id", "label"]).groups)
        n = max(len(keys), 1)
        fig, axes = plt.subplots(n, 1, figsize=(8.0, 2.0 * n), squeeze=False)
        for ax, (did, lab) in zip(axes[:, 0], keys):
            g = rank[(rank.dataset_id == did) & (rank.label == lab)].sort_values("d")
            ax.plot(g.d, g.raw, "o-", label="raw")
            ax.plot(g.d, g.controlled, "s-", label="controlled")
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_title(f"{did} / {lab}")
            ax.legend(fontsize=7)
        _save(fig, figdir, "05_small_multiples_rank.png")

        # 6 heatmap dataset x rank (controlled, discovery excluded from confirmatory but shown)
        fig, ax = plt.subplots(figsize=(9.0, 4.0))
        keys = [f"{a}/{b}" for a, b in rank.groupby(["dataset_id", "label"]).groups]
        ds = sorted(rank.d.unique())
        M = np.full((len(keys), len(ds)), np.nan)
        for i, (did, lab) in enumerate(rank.groupby(["dataset_id", "label"]).groups):
            g = rank[(rank.dataset_id == did) & (rank.label == lab)]
            for _, r in g.iterrows():
                M[i, ds.index(int(r.d))] = r.controlled
        im = ax.imshow(M, aspect="auto", cmap="coolwarm", vmin=-0.4, vmax=0.4)
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels(ds)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=7)
        ax.set_title("Controlled association (blank = out of range)")
        fig.colorbar(im, ax=ax, fraction=0.03)
        _save(fig, figdir, "06_heatmap_rank.png")

    # 7 variance heatmap
    if var is not None and len(var):
        v = var[var.kind == "controlled"]
        keys = [f"{a}/{b}" for a, b in v.groupby(["dataset_id", "label"]).groups]
        taus = sorted(v.tau.unique())
        M = np.full((len(keys), len(taus)), np.nan)
        for i, (did, lab) in enumerate(v.groupby(["dataset_id", "label"]).groups):
            g = v[(v.dataset_id == did) & (v.label == lab)]
            for _, r in g.iterrows():
                if bool(r.in_range):
                    M[i, taus.index(float(r.tau))] = r.rho
        fig, ax = plt.subplots(figsize=(9.0, 4.0))
        im = ax.imshow(M, aspect="auto", cmap="coolwarm", vmin=-0.4, vmax=0.4)
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f"{t:.2f}" for t in taus], rotation=45)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=7)
        ax.set_title("Controlled association vs held-out variance (no extrapolation)")
        fig.colorbar(im, ax=ax, fraction=0.03)
        _save(fig, figdir, "07_heatmap_variance.png")

        # 8 discovery overlay on variance axis
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        for (did, lab), g in v.groupby(["dataset_id", "label"]):
            gg = g[g.in_range == True]  # noqa: E712
            disc = False
            if rank is not None:
                hit = rank[(rank.dataset_id == did) & (rank.label == lab)]
                disc = bool(hit.is_discovery.iloc[0]) if len(hit) else False
            ax.plot(gg.tau, gg.rho, "o-" if disc else "s--", lw=2 if disc else 1, label=f"{did}/{lab}" + (" [discovery]" if disc else ""))
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel(r"held-out $R^2_L$")
        ax.set_ylabel(r"controlled $\rho$")
        ax.set_title("Discovery vs replication on the variance axis")
        ax.legend(fontsize=7)
        _save(fig, figdir, "08_discovery_variance_overlay.png")
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no variance associations", ha="center")
        _save(fig, figdir, "07_heatmap_variance.png")
        _save(fig, figdir, "08_discovery_variance_overlay.png")

    # 9 crossings by dataset
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    if ranges is not None and len(ranges):
        xs = np.arange(len(ranges))
        for key, off in (("d_80", -0.2), ("d_85", 0.0), ("d_90", 0.2), ("dL_plat", 0.35)):
            vals = []
            for _, r in ranges.iterrows():
                v = r.get(key)
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            ax.scatter(xs + off, vals, label=key)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in ranges.dataset_id], rotation=20)
    ax.set_ylabel("rank")
    ax.set_title("Variance crossings and plateaus")
    ax.legend(fontsize=7)
    _save(fig, figdir, "09_crossings_by_dataset.png")

    # 10 reliability
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    if rel is not None and len(rel) and "median_R_H" in rel.columns:
        for did, g in rel.groupby("dataset_id"):
            ax.plot(g.d, g.median_R_H, "o-", label=f"{did} R_H")
            fail = g[g.fail_reliability == True] if "fail_reliability" in g.columns else g.iloc[0:0]  # noqa: E712
            if len(fail):
                ax.scatter(fail.d, fail.median_R_H, marker="x", s=60, c="k")
    ax.axhline(0.20, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("d")
    ax.set_ylabel("median R_H")
    ax.set_title("Curvature reliability across rank")
    ax.legend(fontsize=7)
    _save(fig, figdir, "10_reliability.png")

    # 11 forest Δ85-80
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    if contr is not None and len(contr) and "delta_85_80_ctl" in contr.columns:
        y = np.arange(len(contr))
        ax.errorbar(contr.delta_85_80_ctl.fillna(np.nan), y, fmt="o")
        ax.axvline(0, color="0.5", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a}/{b}" for a, b in zip(contr.dataset_id, contr.label)], fontsize=7)
    ax.set_xlabel(r"$\Delta^{85-80}$ controlled")
    ax.set_title("Discovery-informed 85-80 contrast")
    _save(fig, figdir, "11_delta_forest.png")

    # 12 LODO
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if lodo is not None and len(lodo) and "median_delta" in lodo.columns:
        ax.bar(np.arange(len(lodo)), lodo.median_delta.fillna(0))
        ax.set_xticks(np.arange(len(lodo)))
        ax.set_xticklabels([str(x) for x in lodo.left_out], rotation=20, fontsize=7)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_title("Leave-one-dataset-out median Δ85-80")
    _save(fig, figdir, "12_lodo.png")

    # 13 scale
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    if scale is not None and len(scale) and "controlled" in scale.columns:
        for (did, d), g in scale.groupby(["dataset_id", "d"]):
            gg = g.dropna(subset=["controlled"])
            if len(gg):
                ax.plot(gg.k, gg.controlled, "o-", label=f"{did} d={d}")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("k")
    ax.set_ylabel(r"controlled $\rho$")
    ax.set_title("Scale sensitivity at predeclared geometry ranks")
    ax.legend(fontsize=6)
    _save(fig, figdir, "13_scale_sensitivity.png")

    # 14 permutation diagnostics
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    if perm is not None and len(perm) and "p_global_ctl" in perm.columns:
        ax.bar(np.arange(len(perm)), perm.p_global_ctl.fillna(1.0))
        ax.axhline(0.05, color="C3", ls="--")
        ax.set_xticks(np.arange(len(perm)))
        ax.set_xticklabels([f"{a}/{b}" for a, b in zip(perm.dataset_id, perm.label)], rotation=25, fontsize=7)
    ax.set_ylabel("curve-level p")
    ax.set_title("Familywise / curve-level permutation p")
    _save(fig, figdir, "14_perm_diagnostics.png")

    # 15 df / identifiability
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    if rel is not None and len(rel) and "m_d" in rel.columns:
        for did, g in rel.groupby("dataset_id"):
            ax.plot(g.d, g.m_d, "o-", label=f"{did} m(d)")
            if "median_df_ratio" in g.columns:
                ax.plot(g.d, g.median_df_ratio, "s--", label=f"{did} df/n")
    ax.set_xlabel("d")
    ax.set_title("Quadratic feature count and df ratio")
    ax.legend(fontsize=7)
    _save(fig, figdir, "15_identifiability.png")

    # 16 inclusion / missingness
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    if labs is not None and len(labs) and "valid_geometry_subset" in labs.columns:
        ax.barh(np.arange(len(labs)), labs.valid_geometry_subset.to_numpy(float))
        ax.set_yticks(np.arange(len(labs)))
        ax.set_yticklabels([f"{a}/{b}" for a, b in zip(labs.dataset_id, labs.canonical_label)], fontsize=7)
        ax.axvline(64, color="C3", ls="--", label="min valid anchors")
    ax.set_xlabel("valid labelled rows on geometry subset")
    ax.set_title("Inclusion and valid-label counts")
    ax.legend(fontsize=7)
    _save(fig, figdir, "16_inclusion_missingness.png")
