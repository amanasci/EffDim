"""Screening: frozen sphere-normal curvature vs local mag_r_desi ridge-probe R²."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data import load_prepare
from .paths import platonic_root, resolve_path

EXPECTED_HASH = "d195dc3ed3b6ff0a"
PRIMARY_LABEL = "mag_r_desi"
PRIMARY_CURV = "rho_times_B_traceless_fro"
SCALES = (512, 1024, 2048)
LOCAL_DIM = 8  # frozen feature protocol
SECONDARY = [
    "rho_times_B_fro",
    "rho_times_H_norm",
    "rho_times_B_traceless_fro",
    "entropy_rank",
    "stable_rank",
    "delta_s",
    "delta_scal",
]


@dataclass
class ScreenConfig:
    stage: str = "all"
    output_dir: str = "outputs/geometry/physics_curvature_probe_screen"
    curvature_path: str = (
        "outputs/geometry/physics_quadratic_atlas_sphere_normal/"
        "object_curvature_features_aggregated.parquet"
    )
    prepare_dir: str = "outputs/geometry/physics_activation_atlas_geometry_ablation/prepare"
    labels_path: str = "data_hf/physics/vit_base_test_labels.npz"
    primary_label: str = PRIMARY_LABEL
    expected_hash: str = EXPECTED_HASH
    scales: list[int] = field(default_factory=lambda: list(SCALES))
    ridge_alphas: list[float] = field(
        default_factory=lambda: [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    )
    eval_frac: float = 0.3
    n_bootstrap: int = 1000
    seed: int = 0
    force: bool = False

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def ridge_r2(X_tr, y_tr, X_te, y_te, *, alpha: float) -> float:
    """Local ridge probe R² (matches hypersphere_curvature_vs_probes_gpu)."""
    if len(X_tr) < 8 or len(X_te) < 4:
        return float("nan")
    m_tr = np.isfinite(y_tr)
    m_te = np.isfinite(y_te)
    if m_tr.sum() < 8 or m_te.sum() < 4:
        return float("nan")
    xs = StandardScaler().fit(X_tr[m_tr])
    ys = StandardScaler().fit(y_tr[m_tr].reshape(-1, 1))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(xs.transform(X_tr[m_tr]), ys.transform(y_tr[m_tr].reshape(-1, 1)).ravel())
    pred = ys.inverse_transform(
        model.predict(xs.transform(X_te[m_te])).reshape(-1, 1)
    ).ravel()
    return float(r2_score(y_te[m_te], pred))


def load_frozen_curvature(root: Path, cfg: ScreenConfig) -> pd.DataFrame:
    path = resolve_path(root, cfg.curvature_path)
    df = pd.read_parquet(path)
    if "config_hash" not in df.columns:
        raise RuntimeError("Frozen curvature parquet missing config_hash")
    hashes = set(df["config_hash"].astype(str).unique())
    if hashes != {cfg.expected_hash}:
        raise RuntimeError(
            f"config_hash mismatch: got {hashes}, expected {{{cfg.expected_hash}}}"
        )
    if not bool(df["valid"].all()):
        # do not alter validity gates; just record
        pass
    return df


def load_labels_for_selection(root: Path, cfg: ScreenConfig, sample_ids: np.ndarray) -> np.ndarray:
    z = np.load(resolve_path(root, cfg.labels_path))
    if cfg.primary_label not in z.files:
        raise KeyError(f"{cfg.primary_label} not in {cfg.labels_path}")
    y_full = np.asarray(z[cfg.primary_label], dtype=np.float64)
    return y_full[sample_ids]


def select_ridge_alpha(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    anchors_local: np.ndarray,
    scales: list[int],
    alphas: list[float],
    *,
    eval_frac: float,
    seed: int,
    max_anchors: int = 48,
) -> tuple[float, pd.DataFrame]:
    """Choose ridge α by mean held-out R² over a probe-only subsample (no curvature)."""
    rng = np.random.default_rng(seed)
    use = anchors_local if len(anchors_local) <= max_anchors else np.sort(
        rng.choice(anchors_local, size=max_anchors, replace=False)
    )
    k_max = max(scales)
    nn = NearestNeighbors(n_neighbors=k_max, metric="euclidean")
    nn.fit(X[train_idx])
    dists, inds = nn.kneighbors(X[use])
    rows = []
    for alpha in alphas:
        scores = []
        for ai, a_local in enumerate(use):
            for k in scales:
                neigh = train_idx[inds[ai, :k]]
                neigh = neigh[neigh != a_local]
                if len(neigh) < 20:
                    continue
                rng_i = np.random.default_rng(seed + 17 * int(a_local) + k)
                order = neigh.copy()
                rng_i.shuffle(order)
                n_te = max(4, int(round(eval_frac * len(order))))
                te, tr = order[:n_te], order[n_te:]
                r2 = ridge_r2(X[tr], y[tr], X[te], y[te], alpha=alpha)
                if np.isfinite(r2):
                    scores.append(r2)
        rows.append(
            {
                "alpha": alpha,
                "mean_r2": float(np.mean(scores)) if scores else float("nan"),
                "n": int(len(scores)),
            }
        )
    tab = pd.DataFrame(rows)
    best = tab.loc[tab["mean_r2"].idxmax()]
    return float(best["alpha"]), tab


def compute_probes_at_anchors(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    sid_to_local: dict[int, int],
    sample_ids: np.ndarray,
    scales: list[int],
    alpha: float,
    *,
    eval_frac: float,
    seed: int,
) -> pd.DataFrame:
    anchors_sid = np.asarray(sample_ids, dtype=np.int64)
    anchors_local = np.array([sid_to_local[int(s)] for s in anchors_sid], dtype=np.int64)
    k_max = max(scales)
    nn = NearestNeighbors(n_neighbors=k_max, metric="euclidean")
    nn.fit(X[train_idx])
    dists, inds = nn.kneighbors(X[anchors_local])
    rows = []
    for ai, sid in enumerate(anchors_sid):
        a_local = anchors_local[ai]
        for k in scales:
            rho = float(dists[ai, k - 1])
            neigh = train_idx[inds[ai, :k]]
            neigh = neigh[neigh != a_local]
            y_n = y[neigh]
            finite = np.isfinite(y_n)
            local_var = float(np.nanvar(y_n)) if finite.sum() > 1 else float("nan")
            n_eff = float(finite.sum())
            failure = ""
            r2 = float("nan")
            if len(neigh) < 20:
                failure = "too_few_neighbors"
            else:
                rng = np.random.default_rng(seed + 31 * int(sid) + k)
                order = neigh.copy()
                rng.shuffle(order)
                n_te = max(4, int(round(eval_frac * len(order))))
                te, tr = order[:n_te], order[n_te:]
                r2 = ridge_r2(X[tr], y[tr], X[te], y[te], alpha=alpha)
                if not np.isfinite(r2):
                    failure = "probe_nan"
            rows.append(
                {
                    "sample_id": int(sid),
                    "scale_k": int(k),
                    "local_index": int(a_local),
                    "probe_r2": r2,
                    "probe_alpha": float(alpha),
                    "local_label_variance": local_var,
                    "probe_knn_radius": rho,
                    "probe_n_eff": n_eff,
                    "probe_n_neighbors": int(len(neigh)),
                    "probe_ok": bool(np.isfinite(r2)),
                    "probe_failure_reason": failure,
                    "label": PRIMARY_LABEL,
                }
            )
    return pd.DataFrame(rows)


def join_curvature_probes(curv: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    c = curv.copy()
    c["delta_scal"] = (LOCAL_DIM**2) * (c["H_norm"] ** 2) - (c["B_fro"] ** 2)
    c["log_knn_radius"] = np.log(np.maximum(c["knn_radius"].astype(np.float64), 1e-12))
    m = c.merge(probes, on=["sample_id", "scale_k"], how="left", suffixes=("", "_probe"))
    m["analysis"] = "screening"
    return m


def spearman_dict(x: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 8:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": n}
    rho, p = spearmanr(x[m], y[m])
    return {"rho": float(rho), "pvalue": float(p), "n": n}


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> dict:
    """Rank-transform, residualize x,y on ranked Z, correlate residuals."""
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    n = int(m.sum())
    if n < 12:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": n}
    xr = rankdata(x[m]).astype(np.float64)
    yr = rankdata(y[m]).astype(np.float64)
    Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    # add intercept
    A = np.column_stack([np.ones(n), Zr])
    bx, *_ = np.linalg.lstsq(A, xr, rcond=None)
    by, *_ = np.linalg.lstsq(A, yr, rcond=None)
    rx = xr - A @ bx
    ry = yr - A @ by
    rho, p = spearmanr(rx, ry)
    return {"rho": float(rho), "pvalue": float(p), "n": n}


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int, partial_Z: np.ndarray | None = None
) -> dict:
    m = np.isfinite(x) & np.isfinite(y)
    if partial_Z is not None:
        m = m & np.all(np.isfinite(partial_Z), axis=1)
    idx = np.where(m)[0]
    if len(idx) < 12:
        return {"ci95": [float("nan"), float("nan")], "boot_mean": float("nan")}
    rng = np.random.default_rng(seed)
    rhos = []
    for b in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        if partial_Z is None:
            r = spearman_dict(x[take], y[take])["rho"]
        else:
            r = partial_spearman(x[take], y[take], partial_Z[take])["rho"]
        if np.isfinite(r):
            rhos.append(r)
    if not rhos:
        return {"ci95": [float("nan"), float("nan")], "boot_mean": float("nan")}
    arr = np.asarray(rhos)
    return {
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "boot_mean": float(np.mean(arr)),
        "B": int(len(arr)),
    }


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        raw = pvals[i]
        if not np.isfinite(raw):
            adj[i] = float("nan")
            continue
        val = (m - rank) * raw
        running = max(running, val)
        adj[i] = min(1.0, running)
    # enforce monotonicity in sorted order
    for j in range(1, m):
        i_prev, i_cur = order[j - 1], order[j]
        if np.isfinite(adj[i_prev]) and np.isfinite(adj[i_cur]):
            adj[i_cur] = max(adj[i_cur], adj[i_prev])
    return adj


def correlate_scale(
    df: pd.DataFrame, k: int, cfg: ScreenConfig
) -> dict[str, Any]:
    g0 = df[df["scale_k"] == k].copy()
    mask = (
        g0["probe_ok"].astype(bool)
        & g0["valid"].astype(bool)
        & np.isfinite(g0[PRIMARY_CURV].to_numpy(dtype=np.float64))
        & np.isfinite(g0["probe_r2"].to_numpy(dtype=np.float64))
    )
    g = g0.loc[mask]
    x = g[PRIMARY_CURV].to_numpy(dtype=np.float64)
    y = g["probe_r2"].to_numpy(dtype=np.float64)
    Z = g[["log_knn_radius", "local_label_variance", "reconstruction_error"]].to_numpy(
        dtype=np.float64
    )
    raw = spearman_dict(x, y)
    part = partial_spearman(x, y, Z)
    boot_raw = bootstrap_spearman(x, y, n_boot=cfg.n_bootstrap, seed=cfg.seed + k)
    boot_part = bootstrap_spearman(
        x, y, n_boot=cfg.n_bootstrap, seed=cfg.seed + 1000 + k, partial_Z=Z
    )
    if len(x) >= 20:
        q_lo, q_hi = np.nanquantile(x, [0.01, 0.99])
        m_trim = (x >= q_lo) & (x <= q_hi)
    else:
        m_trim = np.ones(len(x), dtype=bool)
    raw_trim = spearman_dict(x[m_trim], y[m_trim])
    part_trim = partial_spearman(x[m_trim], y[m_trim], Z[m_trim])
    base = raw["rho"]
    influ = float("nan")
    if len(x) >= 20 and np.isfinite(base):
        deltas = []
        for i in range(len(x)):
            m = np.ones(len(x), dtype=bool)
            m[i] = False
            r = spearman_dict(x[m], y[m])["rho"]
            if np.isfinite(r):
                deltas.append(abs(r - base))
        influ = float(np.max(deltas)) if deltas else float("nan")
    curv_vs_rad = spearman_dict(x, g["knn_radius"].to_numpy(dtype=np.float64))
    r2_vs_var = spearman_dict(y, g["local_label_variance"].to_numpy(dtype=np.float64))
    secondary = {}
    for col in SECONDARY:
        if col not in g.columns:
            continue
        secondary[col] = spearman_dict(g[col].to_numpy(dtype=np.float64), y)
    return {
        "scale_k": k,
        "n_joined": int(mask.sum()),
        "n_total": int(len(g0)),
        "raw": raw,
        "partial": part,
        "boot_raw": boot_raw,
        "boot_partial": boot_part,
        "raw_trim_1pct": raw_trim,
        "partial_trim_1pct": part_trim,
        "max_loo_abs_delta_rho": influ,
        "curv_vs_knn_radius": curv_vs_rad,
        "r2_vs_local_label_var": r2_vs_var,
        "secondary": secondary,
        "C_circ_mean": float(np.mean(x)) if len(x) else float("nan"),
        "C_circ_std": float(np.std(x)) if len(x) else float("nan"),
        "probe_r2_mean": float(np.mean(y)) if len(y) else float("nan"),
        "probe_r2_std": float(np.std(y)) if len(y) else float("nan"),
    }


def choose_verdict(primary_rows: pd.DataFrame, joined: pd.DataFrame) -> tuple[str, str]:
    """Pick one primary conclusion from Holm-adjusted partial/raw results."""
    # underpowered?
    if (primary_rows["n"] < 50).any() or primary_rows["rho_partial"].isna().all():
        return (
            "screen_underpowered",
            "Too few valid joined anchors or undefined correlations for a firm screen.",
        )
    sig = primary_rows["p_partial_holm"] <= 0.05
    # density confounding: raw significant but partial not, and curv correlates with radius
    dens = []
    for _, r in primary_rows.iterrows():
        k = int(r["scale_k"])
        g = joined[(joined.scale_k == k) & joined.probe_ok & joined.valid]
        cv = spearman_dict(
            g[PRIMARY_CURV].to_numpy(float), g["knn_radius"].to_numpy(float)
        )
        dens.append(abs(cv["rho"]) > 0.25 and r["p_raw"] <= 0.05 and r["p_partial_holm"] > 0.05)
    if any(dens) and not sig.any():
        return (
            "density_confounded_association",
            "Raw curvature–probe association shrinks after controlling for radius/variance/reconstruction.",
        )
    if sig.sum() == 1:
        row = primary_rows.loc[sig].iloc[0]
        sign = "negative" if row["rho_partial"] < 0 else "positive"
        return (
            "scale_specific_association",
            f"Holm-significant partial Spearman only at k={int(row['scale_k'])} "
            f"(ρ={row['rho_partial']:.3f}, {sign}).",
        )
    if sig.sum() >= 2:
        signs = np.sign(primary_rows.loc[sig, "rho_partial"])
        if (signs < 0).all():
            return (
                "negative_curvature_probe_association",
                "Partial Spearman(C°, R²) negative and Holm-significant at multiple scales.",
            )
        if (signs > 0).all():
            return (
                "positive_curvature_probe_association",
                "Partial Spearman(C°, R²) positive and Holm-significant at multiple scales.",
            )
        return (
            "scale_specific_association",
            "Holm-significant partial associations disagree in sign across scales.",
        )
    # no Holm-significant partial
    if (primary_rows["p_raw"] > 0.05).all() and (primary_rows["p_partial"] > 0.05).all():
        return (
            "no_curvature_probe_association",
            "No significant Spearman association between C° and probe R² at any pre-specified scale.",
        )
    return (
        "no_curvature_probe_association",
        "No Holm-significant partial association after multiplicity correction across scales.",
    )


def make_plots(joined: pd.DataFrame, primary_rows: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for k in SCALES:
        g = joined[(joined.scale_k == k) & joined.probe_ok & joined.valid]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.scatter(
            g[PRIMARY_CURV],
            g["probe_r2"],
            s=12,
            alpha=0.55,
            c="#1f4e79",
            edgecolors="none",
        )
        row = primary_rows[primary_rows.scale_k == k].iloc[0]
        ax.set_xlabel(r"$C^\circ_k=\rho\,|B^\circ|_F$")
        ax.set_ylabel(r"probe $R^2$ (mag_r_desi)")
        ax.set_title(
            f"screening k={k}  ρ_raw={row['rho_raw']:.3f}  ρ_partial={row['rho_partial']:.3f}"
        )
        fig.tight_layout()
        fig.savefig(fig_dir / f"scatter_k{k}.png", dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ks = primary_rows["scale_k"].to_numpy()
    ax.errorbar(
        ks,
        primary_rows["rho_raw"],
        yerr=[
            primary_rows["rho_raw"] - primary_rows["ci95_raw_lo"],
            primary_rows["ci95_raw_hi"] - primary_rows["rho_raw"],
        ],
        fmt="o-",
        label="raw",
        color="#1f4e79",
        capsize=3,
    )
    ax.errorbar(
        ks,
        primary_rows["rho_partial"],
        yerr=[
            primary_rows["rho_partial"] - primary_rows["ci95_partial_lo"],
            primary_rows["ci95_partial_hi"] - primary_rows["rho_partial"],
        ],
        fmt="s--",
        label="partial",
        color="#b85c38",
        capsize=3,
    )
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("k")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Curvature–probe association vs scale (screening)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_vs_scale.png", dpi=140)
    plt.close(fig)


def write_report(
    out: Path,
    cfg: ScreenConfig,
    alpha_tab: pd.DataFrame,
    alpha: float,
    primary_rows: pd.DataFrame,
    secondary_rows: pd.DataFrame,
    sanity: dict,
    verdict: str,
    statement: str,
    deeper: bool,
) -> None:
    report = f"""# Curvature–linear-probe screening

**Label:** analysis=`screening` (earlier exploratory label analyses exist; this is the pre-specified screen).

**Primary label:** `{cfg.primary_label}`  
**Primary curvature:** `{PRIMARY_CURV}` = ρ|B°|_F  
**Frozen config_hash:** `{cfg.expected_hash}` (verified)  
**Ridge α (probe-only selection):** {alpha}

## Probe α grid (no curvature)

{alpha_tab.to_string(index=False)}

## Primary test (per scale, not pooled)

{primary_rows.to_string(index=False)}

Holm correction applied across the three primary scales on partial-Spearman p-values.

## Sanity checks

{json.dumps(sanity, indent=2)}

## Secondary / exploratory metrics

{secondary_rows.to_string(index=False)}

ΔScal = d²|H^S|² − |B^S|²_F with d={LOCAL_DIM}. Collinear curvature metrics are not independent evidence.

## Primary conclusion

`{verdict}`

{statement}

**Strongest result:** see primary table (largest |ρ_partial| among Holm-significant scales, else largest |ρ_partial|).

**Deeper probe-mechanism analysis justified:** {deeper}

## Exact next command (not run)

Do not launch retrieval Fisher / JS / probe-vector / tangent-only / mode-alignment from this screen.
"""
    (out / "SCREENING_REPORT.md").write_text(report)


def run_screen(cfg: ScreenConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()

    curv = load_frozen_curvature(root, cfg)
    data = load_prepare(resolve_path(root, cfg.prepare_dir))
    X = data["X"].astype(np.float64)
    sample_ids_all = data["sample_ids"]
    train_idx = data["train_local"]
    sid_to_local = {int(s): i for i, s in enumerate(sample_ids_all)}
    y = load_labels_for_selection(root, cfg, sample_ids_all)

    anchor_sids = (
        curv[curv["scale_k"] == cfg.scales[0]]["sample_id"].astype(np.int64).to_numpy()
    )
    missing = [int(s) for s in anchor_sids if int(s) not in sid_to_local]
    anchors_local = np.array([sid_to_local[int(s)] for s in anchor_sids if int(s) in sid_to_local])

    print(f"[screen] select ridge α (probe-only) n_anchors={len(anchors_local)}", flush=True)
    alpha, alpha_tab = select_ridge_alpha(
        X,
        y,
        train_idx,
        anchors_local,
        cfg.scales,
        cfg.ridge_alphas,
        eval_frac=cfg.eval_frac,
        seed=cfg.seed,
    )
    alpha_tab.to_csv(out / "ridge_alpha_selection.csv", index=False)
    print(f"[screen] chosen α={alpha}", flush=True)

    probe_path = out / "probe_metrics.parquet"
    if _done(probe_path, cfg.force):
        probes = pd.read_parquet(probe_path)
    else:
        print(f"[screen] computing probes for {len(anchor_sids)} anchors × {cfg.scales}", flush=True)
        probes = compute_probes_at_anchors(
            X,
            y,
            train_idx,
            sid_to_local,
            anchor_sids,
            cfg.scales,
            alpha,
            eval_frac=cfg.eval_frac,
            seed=cfg.seed,
        )
        probes.to_parquet(probe_path, index=False)

    joined = join_curvature_probes(curv, probes)
    joined_path = out / "joined_curvature_probe.parquet"
    joined.to_parquet(joined_path, index=False)

    # missing IDs in join
    miss_join = []
    for k in cfg.scales:
        c_ids = set(curv.loc[curv.scale_k == k, "sample_id"].astype(int))
        p_ids = set(probes.loc[probes.scale_k == k, "sample_id"].astype(int))
        miss_join.append(
            {
                "scale_k": k,
                "curvature_n": len(c_ids),
                "probe_n": len(p_ids),
                "missing_in_probe": sorted(c_ids - p_ids)[:20],
                "n_missing_in_probe": len(c_ids - p_ids),
                "joined_ok": int(
                    ((joined.scale_k == k) & joined.probe_ok & joined.valid).sum()
                ),
            }
        )

    scale_stats = [correlate_scale(joined, k, cfg) for k in cfg.scales]
    p_partial = [s["partial"]["pvalue"] for s in scale_stats]
    p_holm = holm_adjust(p_partial)

    primary_rows = []
    for s, ph in zip(scale_stats, p_holm):
        primary_rows.append(
            {
                "scale_k": s["scale_k"],
                "n": s["raw"]["n"],
                "rho_raw": s["raw"]["rho"],
                "p_raw": s["raw"]["pvalue"],
                "ci95_raw_lo": s["boot_raw"]["ci95"][0],
                "ci95_raw_hi": s["boot_raw"]["ci95"][1],
                "rho_partial": s["partial"]["rho"],
                "p_partial": s["partial"]["pvalue"],
                "p_partial_holm": ph,
                "ci95_partial_lo": s["boot_partial"]["ci95"][0],
                "ci95_partial_hi": s["boot_partial"]["ci95"][1],
                "rho_raw_trim1": s["raw_trim_1pct"]["rho"],
                "rho_partial_trim1": s["partial_trim_1pct"]["rho"],
                "max_loo_abs_delta_rho": s["max_loo_abs_delta_rho"],
                "curv_vs_radius_rho": s["curv_vs_knn_radius"]["rho"],
                "r2_vs_labelvar_rho": s["r2_vs_local_label_var"]["rho"],
            }
        )
    primary_df = pd.DataFrame(primary_rows)
    primary_df.to_csv(out / "primary_correlations.csv", index=False)
    primary_df.to_parquet(out / "primary_correlations.parquet", index=False)

    # bootstrap table
    boot_rows = []
    for s in scale_stats:
        boot_rows.append(
            {
                "scale_k": s["scale_k"],
                "kind": "raw",
                "ci95_lo": s["boot_raw"]["ci95"][0],
                "ci95_hi": s["boot_raw"]["ci95"][1],
                "boot_mean": s["boot_raw"]["boot_mean"],
                "B": s["boot_raw"].get("B", cfg.n_bootstrap),
            }
        )
        boot_rows.append(
            {
                "scale_k": s["scale_k"],
                "kind": "partial",
                "ci95_lo": s["boot_partial"]["ci95"][0],
                "ci95_hi": s["boot_partial"]["ci95"][1],
                "boot_mean": s["boot_partial"]["boot_mean"],
                "B": s["boot_partial"].get("B", cfg.n_bootstrap),
            }
        )
    pd.DataFrame(boot_rows).to_csv(out / "bootstrap_intervals.csv", index=False)

    # secondary table
    sec_rows = []
    for s in scale_stats:
        for col, st in s["secondary"].items():
            sec_rows.append(
                {
                    "scale_k": s["scale_k"],
                    "metric": col,
                    "rho": st["rho"],
                    "pvalue": st["pvalue"],
                    "n": st["n"],
                    "exploratory": True,
                }
            )
    secondary_df = pd.DataFrame(sec_rows)
    secondary_df.to_csv(out / "secondary_correlations.csv", index=False)

    holm_df = primary_df[
        ["scale_k", "rho_partial", "p_partial", "p_partial_holm", "rho_raw", "p_raw"]
    ].copy()
    holm_df.to_csv(out / "holm_adjusted_primary.csv", index=False)

    probe_fail = float((~probes["probe_ok"]).mean())
    sanity = {
        "joined_by_scale": miss_join,
        "missing_sample_ids_in_prepare": missing[:20],
        "n_missing_sample_ids": len(missing),
        "probe_failure_rate": probe_fail,
        "config_hash_verified": cfg.expected_hash,
        "n_anchors": int(len(anchor_sids)),
        "distributions": {
            str(k): {
                "C_circ_mean": next(s["C_circ_mean"] for s in scale_stats if s["scale_k"] == k),
                "C_circ_std": next(s["C_circ_std"] for s in scale_stats if s["scale_k"] == k),
                "probe_r2_mean": next(s["probe_r2_mean"] for s in scale_stats if s["scale_k"] == k),
                "probe_r2_std": next(s["probe_r2_std"] for s in scale_stats if s["scale_k"] == k),
            }
            for k in cfg.scales
        },
    }
    (out / "sanity.json").write_text(json.dumps(sanity, indent=2))

    make_plots(joined, primary_df, out)
    verdict, statement = choose_verdict(primary_df, joined)
    # deeper justified if any Holm-significant partial with |ρ|>=0.1 and not density-only
    deeper = bool(
        verdict
        in {
            "negative_curvature_probe_association",
            "positive_curvature_probe_association",
            "scale_specific_association",
        }
        and (primary_df["p_partial_holm"] <= 0.05).any()
        and (primary_df.loc[primary_df["p_partial_holm"] <= 0.05, "rho_partial"].abs() >= 0.1).any()
    )
    write_report(
        out, cfg, alpha_tab, alpha, primary_df, secondary_df, sanity, verdict, statement, deeper
    )

    # strongest
    if (primary_df["p_partial_holm"] <= 0.05).any():
        sub = primary_df[primary_df["p_partial_holm"] <= 0.05]
        strong = sub.iloc[sub["rho_partial"].abs().argmax()].to_dict()
    else:
        strong = primary_df.iloc[primary_df["rho_partial"].abs().argmax()].to_dict()

    analysis = {
        "analysis": "screening",
        "verdict": verdict,
        "statement": statement,
        "strongest": strong,
        "deeper_probe_mechanism_justified": deeper,
        "ridge_alpha": alpha,
        "primary": primary_rows,
        "config_hash": cfg.expected_hash,
        "seconds": time.time() - t0,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(f"[screen] verdict={verdict} strongest_k={strong.get('scale_k')} "
          f"ρ_partial={strong.get('rho_partial'):.4f}", flush=True)
    return analysis


STAGES = ["all"]
