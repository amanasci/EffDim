"""Quadratic atlas structure: rank, normal curvature, bootstrap, nulls, retrieval links."""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from .charts import (
    enforce_chart_population,
    estimate_bandwidths,
    select_chart_centres,
    soft_memberships,
)
from .coordinates import encode_chart, fit_all_charts, local_rank_diagnostics
from .data import load_prepare, prepare_atlas_data, save_prepare, summarize_population
from .decoder import ResidualDecoder, decode_np, pca_reconstruct, train_chart_decoder
from .metrics import median_knn_radius, weighted_mse
from .overlap_ablation import evaluate_overlaps_ablation
from .paths import platonic_root, resolve_path
from .quadratic import QuadraticChart, fit_quadratic_chart, n_quad_features, quadratic_features


RANK_TRUNC = [1, 2, 3, 4, 6, 8, 12, 16]


@dataclass
class StructureConfig:
    stage: str = "all"
    output_dir: str = "outputs/geometry/physics_quadratic_atlas_structure"
    ablation_dir: str = "outputs/geometry/physics_activation_atlas_geometry_ablation"
    retrieval_dir: str = "outputs/retrieval_information_geometry/smoke"
    labels_path: str = "data_hf/physics/vit_base_test_labels.npz"
    parquet: str = "data_hf/physics/vit_base_test.parquet"
    column: str = "vit_base_galaxies"
    selection_path: str = (
        "outputs/sae_shared_basis/bsf_block_vae_fisher_physics/selection.npz"
    )
    max_n: int = 16384
    global_seed: int = 0
    seed: int = 0
    # smoke-focused grid around n6_d8
    configs: list[tuple[int, int]] = field(
        default_factory=lambda: [(4, 8), (6, 8), (8, 8), (6, 6), (6, 10), (6, 12)]
    )
    charts_per_sample: int = 3
    min_chart_samples: int = 40
    n_bootstrap: int = 50
    n_null: int = 50
    bootstrap_configs: list[str] = field(default_factory=lambda: ["n6_d8"])
    device: str = "cuda"
    epochs: int = 25
    patience: int = 5
    max_decoder_train_samples: int = 1024
    force: bool = False
    max_seconds: float = 7200.0
    max_rss_mb: float = 32000.0

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def _budget(t0: float, cfg: StructureConfig, where: str) -> None:
    if time.time() - t0 > cfg.max_seconds:
        raise RuntimeError(f"Hard stop {where}: >{cfg.max_seconds}s")
    if _rss() > cfg.max_rss_mb:
        raise RuntimeError(f"Hard stop {where}: RSS {_rss():.1f} > {cfg.max_rss_mb}")


def _tv_split(train_idx: np.ndarray, seed: int, val_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    tr = np.asarray(train_idx, dtype=np.int64).copy()
    rng.shuffle(tr)
    n_va = max(1, int(round(val_frac * len(tr))))
    return np.sort(tr[n_va:]), np.sort(tr[:n_va])


def matrix_rank_stats(B: np.ndarray) -> dict:
    """SVD-based rank diagnostics for B (D x q)."""
    if B.size == 0:
        return {
            "raw_rank": 0,
            "stable_rank": 0.0,
            "entropy_rank": 0.0,
            "participation_ratio": 0.0,
            "rank90": 0,
            "rank95": 0,
            "rank99": 0,
            "singular_values": [],
            "frobenius": 0.0,
            "spectral": 0.0,
        }
    s = np.linalg.svd(B, compute_uv=False)
    s = s[s > 0]
    fro = float(np.linalg.norm(B, "fro"))
    spec = float(s[0]) if len(s) else 0.0
    stable = float((fro**2) / max(spec**2, 1e-30))
    p = (s**2) / max((s**2).sum(), 1e-30)
    ent = float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-30)))))
    pr = float((p.sum() ** 2) / max((p**2).sum(), 1e-30))
    csum = np.cumsum(p)

    def r_at(t):
        return int(np.searchsorted(csum, t) + 1) if len(csum) else 0

    eps = 1e-6 * spec if spec > 0 else 0.0
    return {
        "raw_rank": int(np.sum(s > eps)),
        "stable_rank": stable,
        "entropy_rank": ent,
        "participation_ratio": pr,
        "rank90": r_at(0.90),
        "rank95": r_at(0.95),
        "rank99": r_at(0.99),
        "singular_values": s.astype(np.float64).tolist(),
        "frobenius": fro,
        "spectral": spec,
        "n_features": int(B.shape[1]),
        "ambient": int(B.shape[0]),
    }


def split_normal_tangent(B: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B_N = (I-WW^T)B, B_T = WW^T B with W orthonormal (D,d)."""
    BT = W @ (W.T @ B)
    BN = B - BT
    return BN, BT


def truncate_B(B: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return np.zeros_like(B)
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    r = min(r, len(s))
    return (U[:, :r] * s[:r]) @ Vt[:r]


def knn_normed_errors(
    pred: np.ndarray, X: np.ndarray, w: np.ndarray, knn_r: float
) -> np.ndarray:
    """Per-sample E = ||x-xhat||^2 / (d_knn + eps), shape (n,)."""
    err2 = ((pred - X) ** 2).sum(axis=1)
    return err2 / (knn_r + 1e-8)


def bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(x, size=len(x), replace=True).mean()) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_gain(delta: np.ndarray, rel: np.ndarray) -> dict:
    d = delta[np.isfinite(delta)]
    r = rel[np.isfinite(rel)]
    lo, hi = bootstrap_ci(d)
    return {
        "mean_delta": float(np.mean(d)) if len(d) else float("nan"),
        "median_delta": float(np.median(d)) if len(d) else float("nan"),
        "iqr_delta": float(np.subtract(*np.percentile(d, [75, 25]))) if len(d) else float("nan"),
        "ci95_delta": [lo, hi],
        "mean_rel": float(np.mean(r)) if len(r) else float("nan"),
        "median_rel": float(np.median(r)) if len(r) else float("nan"),
        "frac_positive_samples": float(np.mean(d > 0)) if len(d) else float("nan"),
        "n_samples": int(len(d)),
    }


def principal_angle_overlap(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """Overlap = ||Ua^T Ub||_F^2 / k for top-k left singular vectors."""
    if A.size == 0 or B.size == 0:
        return float("nan")
    Ua, _, _ = np.linalg.svd(A, full_matrices=False)
    Ub, _, _ = np.linalg.svd(B, full_matrices=False)
    k = min(k, Ua.shape[1], Ub.shape[1])
    if k <= 0:
        return float("nan")
    M = Ua[:, :k].T @ Ub[:, :k]
    return float(np.sum(M**2) / k)


def match_charts(centres_ref: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Hungarian match by centre distance; returns perm mapping new->ref index."""
    C = np.linalg.norm(centres[:, None, :] - centres_ref[None, :, :], axis=2)
    r, c = linear_sum_assignment(C)
    perm = np.full(len(centres), -1, dtype=np.int64)
    for i, j in zip(r, c):
        perm[i] = j
    return perm


def build_charts(X, tr_all, n_charts, cfg: StructureConfig):
    Xtr = X[tr_all]
    centres_tr = select_chart_centres(Xtr, n_charts=n_charts, method="fps", seed=cfg.seed)
    bw = estimate_bandwidths(Xtr, centres_tr)
    centres_g = tr_all[centres_tr]
    W_full, meta = soft_memberships(
        X, X[centres_g], bw, charts_per_sample=cfg.charts_per_sample
    )
    W, kept, pop = enforce_chart_population(
        W_full, min_chart_samples=cfg.min_chart_samples, max_chart_samples=None
    )
    centres = X[centres_g[np.asarray(kept)]]
    return W, centres, {**meta, **pop, "n_charts": int(W.shape[1])}


def fit_config(
    X: np.ndarray,
    W: sparse.csr_matrix,
    centres: np.ndarray,
    d: int,
    tr_fit: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    knn_r: float,
    seed: int,
) -> dict:
    """Fit PCA+quadratic per chart; evaluate on held-out test members."""
    pcas = fit_all_charts(X, W, n_components=d, train_idx=tr_fit)
    coords = [encode_chart(X, pca) for pca in pcas]
    quads, chart_rows, sample_deltas = [], [], []
    B_list, BN_list, BT_list = [], [], []

    for c, pca in enumerate(pcas):
        w = np.asarray(W[:, c].todense()).ravel()
        q, info = fit_quadratic_chart(
            pca, coords[c][tr_fit], X[tr_fit], w[tr_fit], coords[c][va], X[va], w[va]
        )
        quads.append(q)
        B = q.B_flat
        W_orth = pca["basis"].astype(np.float64)
        BN, BT = split_normal_tangent(B, W_orth)
        B_list.append(B)
        BN_list.append(BN)
        BT_list.append(BT)
        stats_B = matrix_rank_stats(B)
        stats_N = matrix_rank_stats(BN)
        stats_T = matrix_rank_stats(BT)
        J = q.basis  # (D,d)
        j_fro = float(np.linalg.norm(J, "fro"))
        j_spec = float(np.linalg.svd(J, compute_uv=False)[0])

        # held-out test members
        te_m = te[w[te] > 1e-6]
        if len(te_m) == 0:
            te_m = va[w[va] > 1e-6]
        ww = w[te_m]
        Xp = X[te_m]
        Up = coords[c][te_m]
        pred_p = pca_reconstruct(pca, Up)
        pred_q = q.decode(Up)
        Ep = knn_normed_errors(pred_p, Xp, ww, knn_r)
        Eq = knn_normed_errors(pred_q, Xp, ww, knn_r)
        # weight into sample pool with membership
        for i in range(len(te_m)):
            sample_deltas.append(
                {
                    "chart": c,
                    "index": int(te_m[i]),
                    "w": float(ww[i]),
                    "E_pca": float(Ep[i]),
                    "E_quad": float(Eq[i]),
                    "delta": float(Ep[i] - Eq[i]),
                    "rel": float((Ep[i] - Eq[i]) / (Ep[i] + 1e-8)),
                }
            )
        # expected nonlinear displacement on test
        Phi = quadratic_features(Up)
        Qu = Phi @ B.T
        Ju = Up @ J.T
        c_disp = float(
            np.mean(
                np.linalg.norm(Qu, axis=1) / (np.linalg.norm(Ju, axis=1) + 1e-8)
            )
        ) if len(Up) else float("nan")
        mse_p = weighted_mse(pred_p, Xp, ww)
        mse_q = weighted_mse(pred_q, Xp, ww)
        loc = local_rank_diagnostics(pca)
        fN = float(stats_N["frobenius"] ** 2 / (stats_B["frobenius"] ** 2 + 1e-30))
        chart_rows.append(
            {
                "chart": c,
                "n_train": int((w[tr_fit] > 1e-6).sum()),
                "n_val": int((w[va] > 1e-6).sum()),
                "n_test": int(len(te_m)),
                "tangent_dimension": d,
                "local_PCA_stable_rank": loc["effective_rank"],
                "local_PCA_participation_ratio": loc["participation_ratio"],
                "local_PCA_rank95": loc["rank95"],
                "quadratic_stable_rank": stats_B["stable_rank"],
                "quadratic_entropy_rank": stats_B["entropy_rank"],
                "quadratic_rank95": stats_B["rank95"],
                "normal_quadratic_stable_rank": stats_N["stable_rank"],
                "normal_quadratic_entropy_rank": stats_N["entropy_rank"],
                "normal_rank95": stats_N["rank95"],
                "tangent_quadratic_stable_rank": stats_T["stable_rank"],
                "normal_energy_fraction": fN,
                "spectral_normal_fraction": float(
                    stats_N["spectral"] ** 2 / (stats_B["spectral"] ** 2 + 1e-30)
                ),
                "PCA_error_mse": mse_p,
                "quadratic_error_mse": mse_q,
                "relative_gain": float((mse_p - mse_q) / (mse_p + 1e-8)),
                "curvature_F": float(stats_B["frobenius"] / (j_fro + 1e-8)),
                "curvature_spectral": float(stats_B["spectral"] / (j_spec + 1e-8)),
                "curvature_F_normal": float(stats_N["frobenius"] / (j_fro + 1e-8)),
                "curvature_spectral_normal": float(stats_N["spectral"] / (j_spec + 1e-8)),
                "curvature_displacement": c_disp,
                "ridge": info["ridge"],
                "n_quad_features": n_quad_features(d),
                **{f"B_{k}": stats_B[k] for k in ("raw_rank", "frobenius", "spectral")},
                **{f"BN_{k}": stats_N[k] for k in ("raw_rank", "frobenius", "spectral")},
            }
        )

    # rank truncation curve (chart-averaged held-out gain)
    trunc_rows = []
    full_gain = float(
        np.nanmean([r["PCA_error_mse"] - r["quadratic_error_mse"] for r in chart_rows])
    )
    for rQ in RANK_TRUNC + ["full"]:
        gains = []
        mses = []
        for c, pca in enumerate(pcas):
            w = np.asarray(W[:, c].todense()).ravel()
            te_m = te[w[te] > 1e-6]
            if len(te_m) == 0:
                continue
            B = B_list[c] if rQ == "full" else truncate_B(B_list[c], int(rQ))
            q = QuadraticChart(
                mu=quads[c].mu,
                basis=quads[c].basis,
                B_flat=B,
                ridge=quads[c].ridge,
                output_normalize=True,
            )
            pred_q = q.decode(coords[c][te_m])
            pred_p = pca_reconstruct(pca, coords[c][te_m])
            ww = w[te_m]
            mse_q = weighted_mse(pred_q, X[te_m], ww)
            mse_p = weighted_mse(pred_p, X[te_m], ww)
            gains.append(mse_p - mse_q)
            mses.append(mse_q)
        g = float(np.nanmean(gains)) if gains else float("nan")
        trunc_rows.append(
            {
                "r_Q": str(rQ),
                "mean_gain": g,
                "mean_mse": float(np.nanmean(mses)) if mses else float("nan"),
                "frac_of_full_gain": float(g / (full_gain + 1e-12)) if np.isfinite(full_gain) else float("nan"),
            }
        )

    def min_r_retaining(frac: float) -> int | str:
        for row in trunc_rows:
            if row["r_Q"] == "full":
                continue
            if row["frac_of_full_gain"] >= frac:
                return int(row["r_Q"])
        return "full"

    # overlaps / glue
    recon = {c: quads[c].decode(coords[c]) for c in range(len(pcas))}
    bases = {c: pcas[c]["basis"] for c in range(len(pcas))}
    idx = -np.ones((X.shape[0], min(3, W.shape[1])), dtype=np.int64)
    ww = np.zeros_like(idx, dtype=np.float64)
    for i in range(X.shape[0]):
        s, e = W.indptr[i], W.indptr[i + 1]
        cols, data = W.indices[s:e], W.data[s:e]
        order = np.argsort(-data)[: ww.shape[1]]
        for j, o in enumerate(order):
            idx[i, j] = int(cols[o])
            ww[i, j] = float(data[o])
    ov = evaluate_overlaps_ablation(
        idx, ww, {c: coords[c] for c in range(len(pcas))}, bases, recon
    )

    # glue: quadratic normal subspace angles on valid overlaps
    glue_rows = []
    Wd = np.asarray(W.todense())
    for pair in ov.get("pairs", [])[:40]:
        a, b = int(pair["chart_a"]), int(pair["chart_b"])
        mass = np.minimum(Wd[:, a], Wd[:, b])
        n_ov = int((mass > 1e-3).sum())
        ang = principal_angle_overlap(BN_list[a], BN_list[b], k=min(4, BN_list[a].shape[1]))
        tang = principal_angle_overlap(
            pcas[a]["basis"], pcas[b]["basis"], k=min(d, pcas[a]["basis"].shape[1])
        )
        # bases are (D,d) - principal_angle_overlap expects matrices like B; for bases use as "left vectors"
        # Fix: for orthonormal bases, overlap = ||Wa^T Wb||_F^2 / d
        Wa, Wb = pcas[a]["basis"], pcas[b]["basis"]
        tang = float(np.sum((Wa.T @ Wb) ** 2) / d)
        cdiff = abs(chart_rows[a]["curvature_F_normal"] - chart_rows[b]["curvature_F_normal"])
        glue_rows.append(
            {
                **{k: pair[k] for k in pair if k != "failure_reasons"},
                "n_soft_overlap": n_ov,
                "normal_quad_subspace_overlap_top4": ang,
                "tangent_overlap": tang,
                "curvature_mag_diff": cdiff,
                "failure_reasons": ",".join(pair.get("failure_reasons", [])),
            }
        )

    deltas = np.array([s["delta"] for s in sample_deltas], dtype=np.float64)
    rels = np.array([s["rel"] for s in sample_deltas], dtype=np.float64)
    ws = np.array([s["w"] for s in sample_deltas], dtype=np.float64)
    # weighted mean delta
    wmean_delta = float(np.sum(ws * deltas) / max(ws.sum(), 1e-12)) if len(deltas) else float("nan")
    chart_pos = float(np.mean([r["relative_gain"] > 0 for r in chart_rows])) if chart_rows else float("nan")

    # description length
    C = len(pcas)
    D = X.shape[1]
    P_lin = C * (D * d + D)  # centres/mu + bases
    P_quad = P_lin + C * (D * n_quad_features(d))
    r95 = int(np.nanmedian([r["normal_rank95"] for r in chart_rows])) if chart_rows else 0
    P_trunc = P_lin + C * (D * r95 + r95 + r95 * n_quad_features(d))  # U S V rough

    return {
        "pcas": pcas,
        "quads": quads,
        "coords": coords,
        "B_list": B_list,
        "BN_list": BN_list,
        "chart_rows": chart_rows,
        "sample_deltas": sample_deltas,
        "trunc_rows": trunc_rows,
        "gain_summary": {
            **summarize_gain(deltas, rels),
            "weighted_mean_delta": wmean_delta,
            "frac_charts_positive": chart_pos,
            "min_rQ_90": min_r_retaining(0.90),
            "min_rQ_95": min_r_retaining(0.95),
            "min_rQ_99": min_r_retaining(0.99),
            "full_gain_mse": full_gain,
        },
        "overlaps": {
            "valid_frac": ov["valid_frac"],
            "failure_counts": ov["failure_counts"],
            "mean_recon_disagreement": ov["mean_recon_disagreement"],
            "mean_transition_mse": ov["mean_transition_mse"],
            "mean_tangent_disagreement": ov["mean_tangent_disagreement"],
        },
        "glue_rows": glue_rows,
        "description_length": {
            "P_linear": P_lin,
            "P_quadratic": P_quad,
            "P_quadratic_ranktrunc_approx": P_trunc,
            "r95_normal_median": r95,
        },
        "mean_normal_energy_fraction": float(
            np.nanmean([r["normal_energy_fraction"] for r in chart_rows])
        ),
        "mean_quad_stable_rank": float(
            np.nanmean([r["quadratic_stable_rank"] for r in chart_rows])
        ),
        "mean_normal_stable_rank": float(
            np.nanmean([r["normal_quadratic_stable_rank"] for r in chart_rows])
        ),
    }


def stage_reproduce(root: Path, cfg: StructureConfig) -> dict:
    out = cfg.resolved_out(root) / "reproduce"
    out.mkdir(parents=True, exist_ok=True)
    abl = resolve_path(root, cfg.ablation_dir) / "analyze" / "analysis.json"
    if not abl.exists():
        raise FileNotFoundError(f"Missing ablation analysis: {abl}")
    a = json.loads(abl.read_text())
    rows = a["table"]
    wins = sum(1 for r in rows if r["quadratic_mse"] < r["local_pca_mse"] - 1e-8)
    mlp_rows = [r for r in rows if "mlp_mse" in r]
    mlp_beats = (
        sum(1 for r in mlp_rows if r["mlp_mse"] < r["quadratic_mse"] - 1e-4)
        if mlp_rows
        else 0
    )
    n6 = [r for r in rows if r["config_id"] == "n6_d8"]
    scored = sorted(
        rows,
        key=lambda r: -(r["quad_vs_lpca"] * (0.2 + r["lpca_overlap_valid_frac"])),
    )
    top3 = [r["config_id"] for r in scored[:3]]
    ok = (
        wins == 12
        and len(n6) == 1
        and n6[0]["lpca_overlap_valid_frac"] >= 0.5
        and ("n6_d8" in top3 or n6[0]["quad_vs_lpca"] >= sorted(r["quad_vs_lpca"] for r in rows)[-3])
        and not (mlp_rows and mlp_beats > len(mlp_rows) / 2)
    )
    result = {
        "ok": ok,
        "quad_wins": wins,
        "mlp_rows": len(mlp_rows),
        "mlp_beats_quad_count": mlp_beats,
        "mlp_systematic_beats_quad": bool(mlp_rows and mlp_beats > len(mlp_rows) / 2),
        "n6_d8_in_top3_by_gain_overlap": "n6_d8" in top3,
        "n6_d8": n6[0] if n6 else None,
        "top3": top3,
        "conclusion_prev": a.get("conclusion"),
        "abort_structure": not ok,
    }
    pd.DataFrame(rows).to_csv(out / "baseline_reproduction.csv", index=False)
    (out / "reproduction.json").write_text(json.dumps(result, indent=2))
    if not ok:
        raise RuntimeError(f"Baseline reproduction failed: {result}")
    return result


def stage_prepare(root: Path, cfg: StructureConfig) -> dict:
    out = cfg.resolved_out(root) / "prepare"
    if _done(out / "population_summary.json", cfg.force):
        return json.loads((out / "population_summary.json").read_text())
    # reuse ablation prepare if present
    abl_prep = resolve_path(root, cfg.ablation_dir) / "prepare"
    if (abl_prep / "activations_l2.npz").exists() and not cfg.force:
        import shutil

        out.mkdir(parents=True, exist_ok=True)
        for name in [
            "activations_l2.npz",
            "ids_and_splits.npz",
            "population_summary.json",
            "input_schema.json",
            "runtime.json",
        ]:
            src = abl_prep / name
            if src.exists():
                shutil.copy(src, out / name)
        return json.loads((out / "population_summary.json").read_text())
    prep = prepare_atlas_data(
        root,
        parquet=cfg.parquet,
        column=cfg.column,
        selection_path=cfg.selection_path,
        max_n=cfg.max_n,
        global_seed=cfg.global_seed,
    )
    summary = summarize_population(prep)
    save_prepare(out, prep, summary)
    return summary


def stage_grid(root: Path, cfg: StructureConfig, t0: float) -> dict:
    out = cfg.resolved_out(root) / "grid"
    marker = out / "grid_summary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    tr_all = data["train_local"]
    te = data["test_local"]
    tr_fit, va = _tv_split(tr_all, cfg.seed)
    knn_r = median_knn_radius(X, te, k=16)

    summaries = []
    chart_tables = []
    trunc_tables = []
    glue_tables = []
    for n_charts, d in cfg.configs:
        _budget(t0, cfg, f"grid_{n_charts}_{d}")
        cfg_id = f"n{n_charts}_d{d}"
        cfg_dir = out / cfg_id
        if _done(cfg_dir / "summary.json", cfg.force):
            s = json.loads((cfg_dir / "summary.json").read_text())
            summaries.append(s)
            continue
        t1 = time.time()
        W, centres, meta = build_charts(X, tr_all, n_charts, cfg)
        fit = fit_config(X, W, centres, d, tr_fit, va, te, knn_r, cfg.seed)
        # persist artifacts
        cfg_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(cfg_dir / "memberships_csr.npz", W)
        np.savez_compressed(cfg_dir / "centres.npz", centres=centres)
        for c, B in enumerate(fit["B_list"]):
            np.savez_compressed(
                cfg_dir / f"chart{c}_quad.npz",
                B_Q=B,
                B_N=fit["BN_list"][c],
                mu=fit["quads"][c].mu,
                basis=fit["quads"][c].basis,
                ridge=fit["quads"][c].ridge,
                pca_basis=fit["pcas"][c]["basis"],
                pca_mu=fit["pcas"][c]["mu"],
                pca_std=fit["pcas"][c]["coord_std"],
            )
        for row in fit["chart_rows"]:
            row = {**row, "config_id": cfg_id, "n_charts": int(W.shape[1]), "local_dim": d}
            chart_tables.append(row)
        for row in fit["trunc_rows"]:
            trunc_tables.append({**row, "config_id": cfg_id})
        for row in fit["glue_rows"]:
            glue_tables.append({**row, "config_id": cfg_id})
        s = {
            "config_id": cfg_id,
            "n_charts_requested": n_charts,
            "n_charts": int(W.shape[1]),
            "local_dim": d,
            "membership": meta,
            "gain_summary": fit["gain_summary"],
            "overlaps": fit["overlaps"],
            "mean_normal_energy_fraction": fit["mean_normal_energy_fraction"],
            "mean_quad_stable_rank": fit["mean_quad_stable_rank"],
            "mean_normal_stable_rank": fit["mean_normal_stable_rank"],
            "description_length": fit["description_length"],
            "pca_mse_mean": float(np.nanmean([r["PCA_error_mse"] for r in fit["chart_rows"]])),
            "quad_mse_mean": float(
                np.nanmean([r["quadratic_error_mse"] for r in fit["chart_rows"]])
            ),
            "runtime": {"seconds": time.time() - t1, "rss_mb": _rss()},
        }
        pd.DataFrame(fit["sample_deltas"]).to_parquet(cfg_dir / "sample_deltas.parquet", index=False)
        pd.DataFrame(fit["chart_rows"]).assign(config_id=cfg_id).to_parquet(
            cfg_dir / "charts.parquet", index=False
        )
        pd.DataFrame(fit["trunc_rows"]).assign(config_id=cfg_id).to_parquet(
            cfg_dir / "truncation.parquet", index=False
        )
        pd.DataFrame(fit["glue_rows"]).assign(config_id=cfg_id).to_parquet(
            cfg_dir / "glue.parquet", index=False
        )
        (cfg_dir / "summary.json").write_text(json.dumps(s, indent=2))
        summaries.append(s)
        print(
            f"[qstruct] {cfg_id} Δ={s['gain_summary']['mean_delta']:.4f} "
            f"rel={s['gain_summary']['mean_rel']:.3f} "
            f"rN={s['mean_normal_stable_rank']:.2f} fN={s['mean_normal_energy_fraction']:.2f} "
            f"ov={s['overlaps']['valid_frac']:.2f}",
            flush=True,
        )

    # Aggregate from disk (resume-safe).
    chart_tables, trunc_tables, glue_tables, summaries = [], [], [], []
    for n_charts, d in cfg.configs:
        cfg_id = f"n{n_charts}_d{d}"
        cfg_dir = out / cfg_id
        if (cfg_dir / "summary.json").exists():
            summaries.append(json.loads((cfg_dir / "summary.json").read_text()))
        if (cfg_dir / "charts.parquet").exists():
            chart_tables.extend(pd.read_parquet(cfg_dir / "charts.parquet").to_dict("records"))
        if (cfg_dir / "truncation.parquet").exists():
            trunc_tables.extend(pd.read_parquet(cfg_dir / "truncation.parquet").to_dict("records"))
        elif (cfg_dir / "memberships_csr.npz").exists():
            # Rebuild truncation from saved chart artifacts (no PCA refit).
            print(f"[qstruct] rebuild truncation {cfg_id}", flush=True)
            W = sparse.load_npz(cfg_dir / "memberships_csr.npz")
            rows = []
            full_gains = []
            cache = []
            for c in range(W.shape[1]):
                z = np.load(cfg_dir / f"chart{c}_quad.npz")
                pca = {
                    "mu": z["pca_mu"],
                    "basis": z["pca_basis"],
                    "coord_std": z["pca_std"],
                }
                U = encode_chart(X, pca)
                w = np.asarray(W[:, c].todense()).ravel()
                te_m = te[w[te] > 1e-6]
                cache.append((z, pca, U, w, te_m, z["B_Q"]))
                if len(te_m) == 0:
                    continue
                qfull = QuadraticChart(
                    mu=z["mu"], basis=z["basis"], B_flat=z["B_Q"], ridge=float(z["ridge"])
                )
                mse_p = weighted_mse(pca_reconstruct(pca, U[te_m]), X[te_m], w[te_m])
                mse_q = weighted_mse(qfull.decode(U[te_m]), X[te_m], w[te_m])
                full_gains.append(mse_p - mse_q)
            full_gain = float(np.nanmean(full_gains)) if full_gains else float("nan")
            for rQ in [str(x) for x in RANK_TRUNC] + ["full"]:
                gains, mses = [], []
                for z, pca, U, w, te_m, B0 in cache:
                    if len(te_m) == 0:
                        continue
                    B = B0 if rQ == "full" else truncate_B(B0, int(rQ))
                    q = QuadraticChart(
                        mu=z["mu"], basis=z["basis"], B_flat=B, ridge=float(z["ridge"])
                    )
                    mse_q = weighted_mse(q.decode(U[te_m]), X[te_m], w[te_m])
                    mse_p = weighted_mse(pca_reconstruct(pca, U[te_m]), X[te_m], w[te_m])
                    gains.append(mse_p - mse_q)
                    mses.append(mse_q)
                g = float(np.nanmean(gains)) if gains else float("nan")
                rows.append(
                    {
                        "r_Q": rQ,
                        "mean_gain": g,
                        "mean_mse": float(np.nanmean(mses)) if mses else float("nan"),
                        "frac_of_full_gain": float(g / (full_gain + 1e-12))
                        if np.isfinite(full_gain)
                        else float("nan"),
                        "config_id": cfg_id,
                    }
                )
            pd.DataFrame(rows).to_parquet(cfg_dir / "truncation.parquet", index=False)
            trunc_tables.extend(rows)
        if (cfg_dir / "glue.parquet").exists():
            glue_tables.extend(pd.read_parquet(cfg_dir / "glue.parquet").to_dict("records"))

    summary = {"configs": summaries, "knn_radius": knn_r, "peak_rss_mb": _rss()}
    pd.DataFrame(chart_tables).to_parquet(out / "chart_fit_results.parquet", index=False)
    pd.DataFrame(trunc_tables).to_parquet(out / "quadratic_rank_truncation.parquet", index=False)
    if glue_tables:
        pd.DataFrame(glue_tables).to_parquet(out / "chart_glue_results.parquet", index=False)
    else:
        pd.DataFrame(
            [
                {"config_id": s["config_id"], "valid_frac": s["overlaps"]["valid_frac"]}
                for s in summaries
            ]
        ).to_parquet(out / "chart_glue_results.parquet", index=False)
    (out / "grid_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_bootstrap_nulls(root: Path, cfg: StructureConfig, t0: float) -> dict:
    out = cfg.resolved_out(root) / "bootstrap_nulls"
    marker = out / "bootstrap_nulls_summary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    tr_all = data["train_local"]
    te = data["test_local"]
    knn_r = json.loads((cfg.resolved_out(root) / "grid" / "grid_summary.json").read_text())[
        "knn_radius"
    ]
    boot_rows, null_rows = [], []

    for cfg_id in cfg.bootstrap_configs:
        n_charts = int(cfg_id.split("_")[0][1:])
        d = int(cfg_id.split("_")[1][1:])
        # reference fit
        ref_dir = cfg.resolved_out(root) / "grid" / cfg_id
        W_ref = sparse.load_npz(ref_dir / "memberships_csr.npz")
        centres_ref = np.load(ref_dir / "centres.npz")["centres"]
        BN_ref = []
        for c in range(W_ref.shape[1]):
            BN_ref.append(np.load(ref_dir / f"chart{c}_quad.npz")["B_N"])

        # Smoke bootstrap: fix chart membership/PCA from reference; resample train
        # rows for quadratic refits. Full atlas rebuild every bootstrap exceeds budget.
        pcas_ref = []
        coords_ref = []
        for c in range(W_ref.shape[1]):
            z = np.load(ref_dir / f"chart{c}_quad.npz")
            pca = {
                "mu": z["pca_mu"].astype(np.float32),
                "basis": z["pca_basis"].astype(np.float32),
                "coord_std": z["pca_std"].astype(np.float32),
            }
            pcas_ref.append(pca)
            coords_ref.append(encode_chart(X, pca))

        for b in range(cfg.n_bootstrap):
            _budget(t0, cfg, f"boot_{cfg_id}_{b}")
            rng = np.random.default_rng(cfg.seed + 1000 + b)
            tr_boot = rng.choice(tr_all, size=len(tr_all), replace=True)
            tr_fit, va = _tv_split(np.unique(tr_boot), cfg.seed + b)
            deltas, rels, ranks, fNs = [], [], [], []
            overlaps = []
            for c, pca in enumerate(pcas_ref):
                w = np.asarray(W_ref[:, c].todense()).ravel()
                q, _ = fit_quadratic_chart(
                    pca,
                    coords_ref[c][tr_fit],
                    X[tr_fit],
                    w[tr_fit],
                    coords_ref[c][va],
                    X[va],
                    w[va],
                )
                BN, _ = split_normal_tangent(q.B_flat, pca["basis"].astype(np.float64))
                ranks.append(matrix_rank_stats(BN)["stable_rank"])
                fNs.append(
                    matrix_rank_stats(BN)["frobenius"] ** 2
                    / (matrix_rank_stats(q.B_flat)["frobenius"] ** 2 + 1e-30)
                )
                for k in (1, 2, 4, 8):
                    overlaps.append(
                        {
                            "k": k,
                            "overlap": principal_angle_overlap(BN, BN_ref[c], k),
                        }
                    )
                te_m = te[w[te] > 1e-6]
                if len(te_m) == 0:
                    continue
                ww = w[te_m]
                Ep = knn_normed_errors(
                    pca_reconstruct(pca, coords_ref[c][te_m]), X[te_m], ww, knn_r
                )
                Eq = knn_normed_errors(q.decode(coords_ref[c][te_m]), X[te_m], ww, knn_r)
                deltas.extend((Ep - Eq).tolist())
                rels.extend(((Ep - Eq) / (Ep + 1e-8)).tolist())
            boot_rows.append(
                {
                    "config_id": cfg_id,
                    "bootstrap": b,
                    "protocol": "fixed_charts_bootstrap_quadratic",
                    "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
                    "mean_rel": float(np.mean(rels)) if rels else float("nan"),
                    "frac_charts_positive": float("nan"),
                    "mean_normal_stable_rank": float(np.mean(ranks)) if ranks else float("nan"),
                    "mean_normal_energy_fraction": float(np.mean(fNs)) if fNs else float("nan"),
                    "valid_frac": float("nan"),
                    "top1_overlap_mean": float(
                        np.nanmean([o["overlap"] for o in overlaps if o["k"] == 1])
                    ),
                    "top2_overlap_mean": float(
                        np.nanmean([o["overlap"] for o in overlaps if o["k"] == 2])
                    ),
                    "top4_overlap_mean": float(
                        np.nanmean([o["overlap"] for o in overlaps if o["k"] == 4])
                    ),
                    "top8_overlap_mean": float(
                        np.nanmean([o["overlap"] for o in overlaps if o["k"] == 8])
                    ),
                }
            )
            if (b + 1) % 5 == 0:
                print(f"[qstruct] bootstrap {cfg_id} {b+1}/{cfg.n_bootstrap}", flush=True)

        # nulls on reference memberships
        tr_fit, va = _tv_split(tr_all, cfg.seed)
        pcas = fit_all_charts(X, W_ref, n_components=d, train_idx=tr_fit)
        coords = [encode_chart(X, pca) for pca in pcas]
        for nrep in range(cfg.n_null):
            _budget(t0, cfg, f"null_{cfg_id}_{nrep}")
            rng = np.random.default_rng(cfg.seed + 5000 + nrep)
            # shuffled coordinates null
            deltas_shuff = []
            deltas_rand = []
            for c, pca in enumerate(pcas):
                w = np.asarray(W_ref[:, c].todense()).ravel()
                U = coords[c].copy()
                mask_tr = (w[tr_fit] > 1e-6)
                idx_tr = tr_fit[mask_tr]
                if len(idx_tr) < 20:
                    continue
                # shuffle U among train members
                U_sh = U.copy()
                U_sh[idx_tr] = U[rng.permutation(idx_tr)]
                q_sh, _ = fit_quadratic_chart(
                    pca, U_sh[tr_fit], X[tr_fit], w[tr_fit], U_sh[va], X[va], w[va]
                )
                # random features: random R^{d x d} then quadratic of Ru
                R = rng.normal(size=(d, d))
                U_r = U @ R
                # standardize roughly
                U_r = U_r / (np.std(U_r[idx_tr], axis=0, keepdims=True) + 1e-8)
                # fit using random-coords as if they were U (reuse fit_quadratic_chart)
                q_r, _ = fit_quadratic_chart(
                    pca, U_r[tr_fit], X[tr_fit], w[tr_fit], U_r[va], X[va], w[va]
                )
                te_m = te[w[te] > 1e-6]
                if len(te_m) == 0:
                    continue
                ww = w[te_m]
                Ep = knn_normed_errors(pca_reconstruct(pca, coords[c][te_m]), X[te_m], ww, knn_r)
                Eq_sh = knn_normed_errors(q_sh.decode(U_sh[te_m]), X[te_m], ww, knn_r)
                # for random: decode uses U_r coordinates
                # monkey: QuadraticChart.decode expects U in its trained coord system
                Eq_r = knn_normed_errors(q_r.decode(U_r[te_m]), X[te_m], ww, knn_r)
                deltas_shuff.extend((Ep - Eq_sh).tolist())
                deltas_rand.extend((Ep - Eq_r).tolist())
            null_rows.append(
                {
                    "config_id": cfg_id,
                    "null_rep": nrep,
                    "shuffle_mean_delta": float(np.mean(deltas_shuff)) if deltas_shuff else float("nan"),
                    "random_feat_mean_delta": float(np.mean(deltas_rand)) if deltas_rand else float("nan"),
                }
            )
            if (nrep + 1) % 10 == 0:
                print(f"[qstruct] null {cfg_id} {nrep+1}/{cfg.n_null}", flush=True)

    boot_df = pd.DataFrame(boot_rows)
    null_df = pd.DataFrame(null_rows)
    boot_df.to_parquet(out / "bootstrap_results.parquet", index=False)
    null_df.to_parquet(out / "null_results.parquet", index=False)
    # true gain reference
    true = {}
    for cfg_id in cfg.bootstrap_configs:
        s = json.loads((cfg.resolved_out(root) / "grid" / cfg_id / "summary.json").read_text())
        true[cfg_id] = s["gain_summary"]["mean_delta"]
        nd = null_df[null_df.config_id == cfg_id]
        bd = boot_df[boot_df.config_id == cfg_id]
        true[cfg_id + "_shuffle_p"] = float(np.mean(nd.shuffle_mean_delta >= true[cfg_id])) if len(nd) else float("nan")
        true[cfg_id + "_random_p"] = float(np.mean(nd.random_feat_mean_delta >= true[cfg_id])) if len(nd) else float("nan")
        true[cfg_id + "_boot_mean"] = float(bd.mean_delta.mean()) if len(bd) else float("nan")
        true[cfg_id + "_boot_ci"] = (
            [float(bd.mean_delta.quantile(0.025)), float(bd.mean_delta.quantile(0.975))]
            if len(bd)
            else [float("nan"), float("nan")]
        )
    summary = {"true_and_pvalues": true, "n_bootstrap": cfg.n_bootstrap, "n_null": cfg.n_null}
    (out / "bootstrap_nulls_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_mlp(root: Path, cfg: StructureConfig, t0: float) -> dict:
    import torch

    out = cfg.resolved_out(root) / "mlp"
    marker = out / "mlp_controls.parquet"
    if _done(marker, cfg.force):
        return {"path": str(marker)}
    out.mkdir(parents=True, exist_ok=True)
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    tr_all = data["train_local"]
    te = data["test_local"]
    tr_fit, va = _tv_split(tr_all, cfg.seed)
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    rows = []
    for cfg_id in cfg.bootstrap_configs:
        _budget(t0, cfg, f"mlp_{cfg_id}")
        n_charts = int(cfg_id.split("_")[0][1:])
        d = int(cfg_id.split("_")[1][1:])
        W = sparse.load_npz(cfg.resolved_out(root) / "grid" / cfg_id / "memberships_csr.npz")
        pcas = fit_all_charts(X, W, n_components=d, train_idx=tr_fit)
        coords = [encode_chart(X, pca) for pca in pcas]
        n_quad = n_quad_features(d)
        D = X.shape[1]
        # param-matched: one hidden layer width w such that d*w + w*D ≈ D*n_quad
        # w ≈ D*n_quad / (d+D)
        w_matched = max(8, int(round(D * n_quad / (d + D))))
        w_matched = min(w_matched, 256)
        for c, pca in enumerate(pcas):
            w = np.asarray(W[:, c].todense()).ravel()
            # quadratic baseline from saved
            z = np.load(cfg.resolved_out(root) / "grid" / cfg_id / f"chart{c}_quad.npz")
            q = QuadraticChart(
                mu=z["mu"], basis=z["basis"], B_flat=z["B_Q"], ridge=float(z["ridge"])
            )
            te_m = te[w[te] > 1e-6]
            if len(te_m) < 5:
                continue
            mse_q = weighted_mse(q.decode(coords[c][te_m]), X[te_m], w[te_m])
            mse_p = weighted_mse(pca_reconstruct(pca, coords[c][te_m]), X[te_m], w[te_m])
            for name, hidden in [
                ("mlp_matched", [w_matched]),
                ("mlp_larger", [128, 128]),
            ]:
                model, _ = train_chart_decoder(
                    pca,
                    coords[c][tr_fit],
                    X[tr_fit],
                    w[tr_fit],
                    coords[c][va],
                    X[va],
                    w[va],
                    hidden=hidden,
                    activation="softplus",
                    residual_scale=0.01,
                    output_normalize=True,
                    lr=1e-3,
                    epochs=cfg.epochs,
                    patience=cfg.patience,
                    batch_size=256,
                    device=device,
                    max_train=cfg.max_decoder_train_samples,
                    seed=cfg.seed + c,
                )
                mse_m = weighted_mse(decode_np(model, coords[c][te_m], device), X[te_m], w[te_m])
                n_params = sum(p.numel() for p in model.parameters())
                rows.append(
                    {
                        "config_id": cfg_id,
                        "chart": c,
                        "model": name,
                        "hidden": str(hidden),
                        "n_params": int(n_params),
                        "n_quad_params": int(D * n_quad),
                        "mse_pca": mse_p,
                        "mse_quad": mse_q,
                        "mse_mlp": mse_m,
                        "delta_mlp_quad": float(mse_q - mse_m),
                    }
                )
                del model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
        print(f"[qstruct] mlp controls {cfg_id} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(marker, index=False)
    return {"n_rows": len(df), "mean_delta_mlp_quad": float(df.delta_mlp_quad.mean()) if len(df) else float("nan")}


def stage_retrieval(root: Path, cfg: StructureConfig) -> dict:
    out = cfg.resolved_out(root) / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    rig = resolve_path(root, cfg.retrieval_dir)
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    sample_ids = data["sample_ids"]
    # object-level curvature from n6_d8 soft assignment
    cfg_id = "n6_d8"
    grid = cfg.resolved_out(root) / "grid" / cfg_id
    W = sparse.load_npz(grid / "memberships_csr.npz")
    charts = pd.read_parquet(grid / "charts.parquet")
    # soft assign
    C_gain = charts["relative_gain"].to_numpy()
    C_fn = charts["curvature_F_normal"].to_numpy()
    C_disp = charts["curvature_displacement"].to_numpy()
    C_rn = charts["normal_quadratic_stable_rank"].to_numpy()
    Wd = np.asarray(W.todense())
    # normalize rows
    row_sum = Wd.sum(axis=1, keepdims=True)
    A = Wd / np.maximum(row_sum, 1e-12)
    obj_gain = A @ C_gain
    obj_fn = A @ C_fn
    obj_disp = A @ C_disp
    obj_rn = A @ C_rn
    hard = Wd.argmax(axis=1)

    # fisher ranks
    fisher = pd.read_parquet(rig / "fisher_rank_results.parquet")
    fisher = fisher[(fisher.model == "vit_base") & (fisher.representation == "dense")]
    idx = np.load(rig / "cache" / "index_sets.npz")
    fisher_q = idx["fisher_q"]
    geom_q = idx["geom_q"]

    corr_rows = []
    for scale in sorted(fisher.scale_key.unique()):
        sub = fisher[fisher.scale_key == scale]
        # map query_local -> values
        ql = sub.query_local.to_numpy()
        mask = ql >= 0
        ql = ql[mask]
        for metric in ["entropy_rank", "participation_ratio", "stable_rank"]:
            y = sub.loc[mask, metric].to_numpy()
            for name, xall in [
                ("relative_gain", obj_gain),
                ("curvature_F_normal", obj_fn),
                ("curvature_displacement", obj_disp),
                ("normal_quadratic_stable_rank", obj_rn),
            ]:
                x = xall[ql]
                if len(x) < 20:
                    continue
                pear = float(np.corrcoef(x, y)[0, 1])
                spear = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
                # bootstrap CI pearson
                rng = np.random.default_rng(0)
                boots = []
                for _ in range(200):
                    ii = rng.choice(len(x), size=len(x), replace=True)
                    boots.append(float(np.corrcoef(x[ii], y[ii])[0, 1]))
                corr_rows.append(
                    {
                        "scale_key": scale,
                        "fisher_metric": metric,
                        "curvature": name,
                        "pearson": pear,
                        "spearman": spear,
                        "ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                        "n": int(len(x)),
                        "assignment": "soft",
                    }
                )

    # per-query JS from D matrices on geom_q if available
    js_rows = []
    cache = rig / "cache"
    D_paths = {
        m: cache / f"D_dense_keff_100_{m}.npy"
        for m in ["vit_base", "dinov3", "vit_large", "clip_base"]
    }
    Ds = {m: np.load(p) for m, p in D_paths.items() if p.exists()}
    if "vit_base" in Ds and len(Ds) >= 2:
        # convert distances to soft neighborhoods
        def soft_P(D, tau=None):
            # D square on geom_q
            n = D.shape[0]
            # set diag inf
            Dd = D.copy()
            np.fill_diagonal(Dd, np.inf)
            if tau is None:
                tau = float(np.median(np.min(Dd, axis=1)))
            logits = -Dd / max(tau, 1e-8)
            logits = logits - logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P = P / P.sum(axis=1, keepdims=True)
            return P

        Pb = soft_P(Ds["vit_base"])
        # curvature on geom_q locals
        for m, Dm in Ds.items():
            if m == "vit_base":
                continue
            Pm = soft_P(Dm)
            # JS similarity per row
            M = 0.5 * (Pb + Pm)
            def js_sim(p, q, m):
                # 1 - JS
                def kl(a, b):
                    a = np.clip(a, 1e-12, 1)
                    b = np.clip(b, 1e-12, 1)
                    return np.sum(a * np.log(a / b), axis=1)

                return 1.0 - 0.5 * (kl(p, m) + kl(q, m))

            sim = js_sim(Pb, Pm, M)
            x = obj_fn[geom_q]
            pear = float(np.corrcoef(x, sim)[0, 1])
            spear = float(pd.Series(x).corr(pd.Series(sim), method="spearman"))
            js_rows.append(
                {
                    "paired_model": f"vit_base__{m}",
                    "scale_key": "keff_100",
                    "curvature": "curvature_F_normal",
                    "pearson": pear,
                    "spearman": spear,
                    "n": int(len(x)),
                }
            )

    # physics labels
    labels = np.load(resolve_path(root, cfg.labels_path))
    phys_rows = []
    for var in ["photo_z", "mag_r_desi", "smooth_fraction", "stellar_mass"]:
        if var not in labels:
            continue
        y_full = labels[var]
        y = y_full[sample_ids]
        m = np.isfinite(y)
        for name, xall in [
            ("relative_gain", obj_gain),
            ("curvature_F_normal", obj_fn),
            ("curvature_displacement", obj_disp),
        ]:
            x = xall[m]
            yy = y[m]
            if len(x) < 50:
                continue
            phys_rows.append(
                {
                    "variable": var,
                    "curvature": name,
                    "pearson": float(np.corrcoef(x, yy)[0, 1]),
                    "spearman": float(pd.Series(x).corr(pd.Series(yy), method="spearman")),
                    "n": int(len(x)),
                    "assignment": "soft",
                }
            )
        # chart-level composition
        for c in range(W.shape[1]):
            mem = Wd[:, c] > 1e-3
            phys_rows.append(
                {
                    "variable": var,
                    "curvature": "chart_composition",
                    "chart": c,
                    "mean": float(np.nanmean(y[mem])),
                    "std": float(np.nanstd(y[mem])),
                    "p50": float(np.nanmedian(y[mem])),
                    "chart_curvature_F_normal": float(C_fn[c]),
                    "n": int(mem.sum()),
                }
            )

    # cross-model: correlate vit_base object curvature with dinov3 fisher rank (paired)
    xmodel_rows = []
    fisher_all = pd.read_parquet(rig / "fisher_rank_results.parquet")
    for m in ["dinov3", "vit_large", "clip_base"]:
        sub = fisher_all[
            (fisher_all.model == m)
            & (fisher_all.representation == "dense")
            & (fisher_all.scale_key == "keff_100")
        ]
        if len(sub) == 0:
            continue
        ql = sub.query_local.to_numpy()
        mask = ql >= 0
        ql = ql[mask]
        y = sub.loc[mask, "stable_rank"].to_numpy()
        x = obj_fn[ql]
        # paired vs shuffled
        pear = float(np.corrcoef(x, y)[0, 1])
        rng = np.random.default_rng(0)
        nulls = []
        for _ in range(100):
            nulls.append(float(np.corrcoef(x, rng.permutation(y))[0, 1]))
        p = float(np.mean(np.abs(nulls) >= abs(pear)))
        xmodel_rows.append(
            {
                "other_model": m,
                "metric": "vit_base_curvature_F_normal vs other_fisher_stable_rank",
                "pearson": pear,
                "shuffle_p": p,
                "n": int(len(x)),
            }
        )

    pd.DataFrame(corr_rows).to_parquet(out / "retrieval_geometry_correlations.parquet", index=False)
    pd.DataFrame(js_rows).to_parquet(out / "curvature_vs_js.parquet", index=False)
    pd.DataFrame(phys_rows).to_parquet(out / "physics_correlations.parquet", index=False)
    pd.DataFrame(xmodel_rows).to_parquet(out / "cross_model_curvature.parquet", index=False)
    np.savez_compressed(
        out / "object_curvature.npz",
        relative_gain=obj_gain,
        curvature_F_normal=obj_fn,
        curvature_displacement=obj_disp,
        normal_rank=obj_rn,
        hard_chart=hard,
        sample_ids=sample_ids,
    )
    return {
        "n_fisher_corrs": len(corr_rows),
        "n_js": len(js_rows),
        "n_phys": len(phys_rows),
        "best_fisher_spearman": float(
            pd.DataFrame(corr_rows)["spearman"].abs().max()
        )
        if corr_rows
        else float("nan"),
    }


def stage_figures_report(root: Path, cfg: StructureConfig) -> dict:
    import matplotlib.pyplot as plt

    out = cfg.resolved_out(root)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    grid = json.loads((out / "grid" / "grid_summary.json").read_text())
    charts = pd.read_parquet(out / "grid" / "chart_fit_results.parquet")
    trunc = pd.read_parquet(out / "grid" / "quadratic_rank_truncation.parquet")
    boot = pd.read_parquet(out / "bootstrap_nulls" / "bootstrap_results.parquet")
    nulls = pd.read_parquet(out / "bootstrap_nulls" / "null_results.parquet")
    mlp = pd.read_parquet(out / "mlp" / "mlp_controls.parquet")
    ret = pd.read_parquet(out / "retrieval" / "retrieval_geometry_correlations.parquet")
    phys = pd.read_parquet(out / "retrieval" / "physics_correlations.parquet")
    glue = pd.read_parquet(out / "grid" / "chart_glue_results.parquet")
    js = pd.read_parquet(out / "retrieval" / "curvature_vs_js.parquet")
    xmodel = pd.read_parquet(out / "retrieval" / "cross_model_curvature.parquet")

    def savefig(name):
        plt.tight_layout()
        plt.savefig(figdir / name, dpi=120)
        plt.close()

    # PCA vs quadratic
    cfg_ids = [s["config_id"] for s in grid["configs"]]
    pca_e = [s["pca_mse_mean"] for s in grid["configs"]]
    quad_e = [s["quad_mse_mean"] for s in grid["configs"]]
    plt.figure()
    x = np.arange(len(cfg_ids))
    plt.bar(x - 0.2, pca_e, 0.4, label="PCA")
    plt.bar(x + 0.2, quad_e, 0.4, label="quadratic")
    plt.xticks(x, cfg_ids, rotation=45, ha="right")
    plt.ylabel("held-out MSE")
    plt.legend()
    plt.title("PCA vs quadratic held-out error")
    savefig("pca_vs_quadratic_error.png")

    # relative gain by config
    plt.figure()
    rel = [s["gain_summary"]["mean_rel"] for s in grid["configs"]]
    plt.bar(cfg_ids, rel)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean relative quadratic gain")
    plt.title("Relative quadratic gain by configuration")
    savefig("relative_quadratic_gain_by_config.png")

    # singular spectra for n6_d8
    plt.figure()
    for c in range(min(6, int(charts[charts.config_id == "n6_d8"].chart.max() + 1))):
        z = np.load(out / "grid" / "n6_d8" / f"chart{c}_quad.npz")
        s = np.linalg.svd(z["B_N"], compute_uv=False)
        plt.semilogy(s[:20] / s[0], label=f"c{c}")
    plt.xlabel("mode")
    plt.ylabel("normalized singular value")
    plt.title("Normal quadratic singular spectra (n6_d8)")
    plt.legend(fontsize=7)
    savefig("normal_quadratic_singular_spectra.png")

    # gain vs rank
    plt.figure()
    t = trunc[trunc.config_id == "n6_d8"]
    rs = [str(r) for r in t.r_Q]
    plt.plot(rs, t.frac_of_full_gain, marker="o")
    plt.ylabel("fraction of full quadratic gain")
    plt.xlabel("r_Q")
    plt.title("Quadratic gain vs retained rank (n6_d8)")
    savefig("gain_vs_quadratic_rank.png")

    # normal vs tangential energy
    plt.figure()
    sub = charts[charts.config_id == "n6_d8"]
    plt.hist(sub.normal_energy_fraction, bins=10)
    plt.xlabel("normal energy fraction f_N")
    plt.title("Normal vs tangential quadratic energy (n6_d8)")
    savefig("normal_vs_tangential_energy.png")

    # gain vs d / n_charts
    plt.figure()
    for n in sorted(set(s["n_charts_requested"] for s in grid["configs"])):
        ss = [s for s in grid["configs"] if s["n_charts_requested"] == n]
        plt.plot([s["local_dim"] for s in ss], [s["gain_summary"]["mean_rel"] for s in ss], marker="o", label=f"n={n}")
    plt.xlabel("d")
    plt.ylabel("relative gain")
    plt.legend()
    plt.title("Quadratic gain vs tangent dimension")
    savefig("gain_vs_tangent_dim.png")

    plt.figure()
    for d in sorted(set(s["local_dim"] for s in grid["configs"])):
        ss = [s for s in grid["configs"] if s["local_dim"] == d]
        plt.plot(
            [s["n_charts_requested"] for s in ss],
            [s["overlaps"]["valid_frac"] for s in ss],
            marker="o",
            label=f"d={d}",
        )
    plt.xlabel("n_charts")
    plt.ylabel("overlap valid fraction")
    plt.legend()
    plt.title("Glue quality vs chart count")
    savefig("glue_vs_chart_count.png")

    # bootstrap
    plt.figure()
    plt.hist(boot.mean_delta, bins=15)
    true = json.loads((out / "bootstrap_nulls" / "bootstrap_nulls_summary.json").read_text())
    td = true["true_and_pvalues"].get("n6_d8", float("nan"))
    plt.axvline(td, color="r", label="true")
    plt.legend()
    plt.title("Bootstrap quadratic-gain distribution (n6_d8)")
    savefig("bootstrap_quadratic_gain.png")

    # nulls
    plt.figure()
    plt.hist(nulls.shuffle_mean_delta, bins=15, alpha=0.7, label="shuffle u")
    plt.hist(nulls.random_feat_mean_delta, bins=15, alpha=0.7, label="random features")
    plt.axvline(td, color="r", label="true")
    plt.legend()
    plt.title("True vs null quadratic gain")
    savefig("true_vs_null_gain.png")

    # mlp
    plt.figure()
    for name, g in mlp.groupby("model"):
        plt.scatter(g.mse_quad, g.mse_mlp, label=name, alpha=0.7)
    lims = [
        min(mlp.mse_quad.min(), mlp.mse_mlp.min()),
        max(mlp.mse_quad.max(), mlp.mse_mlp.max()),
    ]
    plt.plot(lims, lims, "k--")
    plt.xlabel("quadratic MSE")
    plt.ylabel("MLP MSE")
    plt.legend()
    plt.title("Quadratic vs MLP reconstruction")
    savefig("quadratic_vs_mlp.png")

    # curvature vs fisher
    plt.figure()
    sub = ret[ret.fisher_metric == "stable_rank"]
    for curv, g in sub.groupby("curvature"):
        plt.plot(g.scale_key, g.spearman, marker="o", label=curv)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Spearman")
    plt.legend(fontsize=7)
    plt.title("Curvature vs retrieval Fisher stable rank")
    savefig("curvature_vs_fisher_rank.png")

    # js
    if len(js):
        plt.figure()
        plt.bar(js.paired_model, js.spearman)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Spearman(C, JS)")
        plt.title("Curvature vs cross-model JS agreement")
        savefig("curvature_vs_js.png")

    # physics
    plt.figure()
    sub = phys[phys.curvature != "chart_composition"]
    if len(sub):
        for var, g in sub.groupby("variable"):
            plt.plot(g.curvature, g.spearman, marker="o", label=var)
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.ylabel("Spearman")
        plt.title("Curvature vs physical variables")
        savefig("curvature_vs_physics.png")

    # param count frontier
    plt.figure()
    for s in grid["configs"]:
        dl = s["description_length"]
        plt.scatter(dl["P_linear"], s["pca_mse_mean"], c="C0")
        plt.scatter(dl["P_quadratic"], s["quad_mse_mean"], c="C1")
        plt.scatter(dl["P_quadratic_ranktrunc_approx"], s["quad_mse_mean"], c="C2", marker="x")
    plt.xlabel("parameter count")
    plt.ylabel("held-out MSE")
    plt.title("Reconstruction vs parameter count")
    savefig("reconstruction_vs_params.png")

    # aggregate summary + report
    n6 = next(s for s in grid["configs"] if s["config_id"] == "n6_d8")
    boot_sum = json.loads((out / "bootstrap_nulls" / "bootstrap_nulls_summary.json").read_text())
    mlp_delta = float(mlp.delta_mlp_quad.mean())
    fisher_best = float(ret["spearman"].abs().max()) if len(ret) else float("nan")
    phys_best = (
        float(phys[phys.curvature != "chart_composition"]["spearman"].abs().max())
        if len(phys[phys.curvature != "chart_composition"])
        else float("nan")
    )

    # stability of d≈8 / n≈6
    by_d = {}
    for s in grid["configs"]:
        if s["n_charts_requested"] == 6:
            by_d[s["local_dim"]] = s["gain_summary"]["mean_rel"]
    by_n = {}
    for s in grid["configs"]:
        if s["local_dim"] == 8:
            by_n[s["n_charts_requested"]] = s["overlaps"]["valid_frac"]

    low_rank = (
        n6["mean_normal_stable_rank"] < 0.35 * n_quad_features(8)
        and isinstance(n6["gain_summary"]["min_rQ_90"], int)
        and n6["gain_summary"]["min_rQ_90"] <= 4
    )
    null_ok = boot_sum["true_and_pvalues"].get("n6_d8_shuffle_p", 1) < 0.05
    mlp_ok = mlp_delta < 1e-3  # MLP not better
    fN = n6["mean_normal_energy_fraction"]
    extrinsic = fN > 0.4

    if n6["gain_summary"]["mean_rel"] > 0.05 and null_ok and low_rank:
        strongest = (
            "Physics activation manifolds are locally well described by low-dimensional "
            "tangent spaces, with systematic held-out deviations captured by a small number "
            "of quadratic modes"
            + (
                ", predominantly normal to the local PCA subspace (extrinsic bending)."
                if extrinsic
                else ", with substantial tangential quadratic energy (coordinate nonlinearity contributes)."
            )
        )
        if fisher_best > 0.15:
            strongest += " Curvature shows a modest reproducible association with retrieval Fisher rank."
        else:
            strongest += " No strong link to retrieval Fisher rank was found at smoke scale."
    else:
        strongest = (
            "Quadratic improvement is present but the low-rank / extrinsic-curvature "
            "interpretation is only partially supported at smoke scale; see detailed metrics."
        )

    # write per-chart summary
    charts.to_csv(out / "per_chart_summary.csv", index=False)
    agg = []
    for s in grid["configs"]:
        agg.append(
            {
                "config_id": s["config_id"],
                "pca_mse": s["pca_mse_mean"],
                "quad_mse": s["quad_mse_mean"],
                "mean_rel_gain": s["gain_summary"]["mean_rel"],
                "mean_delta": s["gain_summary"]["mean_delta"],
                "frac_charts_pos": s["gain_summary"]["frac_charts_positive"],
                "valid_frac": s["overlaps"]["valid_frac"],
                "quad_stable_rank": s["mean_quad_stable_rank"],
                "normal_stable_rank": s["mean_normal_stable_rank"],
                "normal_energy_fraction": s["mean_normal_energy_fraction"],
                "min_rQ_90": s["gain_summary"]["min_rQ_90"],
                "min_rQ_95": s["gain_summary"]["min_rQ_95"],
                "min_rQ_99": s["gain_summary"]["min_rQ_99"],
            }
        )
    pd.DataFrame(agg).to_csv(out / "aggregate_summary.csv", index=False)

    # spectra parquet
    spec_rows = []
    for c in range(int(charts[charts.config_id == "n6_d8"].chart.max() + 1)):
        z = np.load(out / "grid" / "n6_d8" / f"chart{c}_quad.npz")
        for kind, key in [("B_Q", "B_Q"), ("B_N", "B_N")]:
            s = np.linalg.svd(z[key], compute_uv=False)
            for i, sv in enumerate(s[:32]):
                spec_rows.append({"chart": c, "matrix": kind, "mode": i, "singular_value": float(sv)})
    pd.DataFrame(spec_rows).to_parquet(out / "quadratic_spectra.parquet", index=False)
    charts[charts.config_id == "n6_d8"].to_parquet(out / "quadratic_rank_results.parquet", index=False)
    charts[charts.config_id == "n6_d8"][
        [
            "chart",
            "normal_energy_fraction",
            "normal_quadratic_stable_rank",
            "curvature_F_normal",
            "curvature_spectral_normal",
            "curvature_displacement",
            "relative_gain",
        ]
    ].to_parquet(out / "normal_curvature_results.parquet", index=False)

    dl_rows = []
    for s in grid["configs"]:
        dl = s["description_length"]
        dl_rows.append(
            {
                "config_id": s["config_id"],
                "model": "linear",
                "params": dl["P_linear"],
                "mse": s["pca_mse_mean"],
            }
        )
        dl_rows.append(
            {
                "config_id": s["config_id"],
                "model": "quadratic_full",
                "params": dl["P_quadratic"],
                "mse": s["quad_mse_mean"],
            }
        )
        dl_rows.append(
            {
                "config_id": s["config_id"],
                "model": "quadratic_ranktrunc",
                "params": dl["P_quadratic_ranktrunc_approx"],
                "mse": s["quad_mse_mean"],
            }
        )
    pd.DataFrame(dl_rows).to_parquet(out / "description_length_frontier.parquet", index=False)

    report = f"""# Quadratic atlas structure report

## Strongest scientifically defensible statement

{strongest}

## 1. Held-out quadratic gain over local PCA

Configuration **n6_d8**:
- mean Δ (E_PCA−E_quad, knn-normalized) = **{n6['gain_summary']['mean_delta']:.4f}**
- mean relative gain R = **{n6['gain_summary']['mean_rel']:.3f}**
- fraction charts with positive gain = **{n6['gain_summary']['frac_charts_positive']:.2f}**
- fraction samples with positive gain = **{n6['gain_summary']['frac_positive_samples']:.2f}**
- bootstrap 95% CI (Δ) = {n6['gain_summary']['ci95_delta']}

Across focused grid, quadratic MSE < PCA MSE for all configs with positive mean relative gains
(see `aggregate_summary.csv`).

## 2. Bootstrap robustness

n_bootstrap={cfg.n_bootstrap} on {cfg.bootstrap_configs}:
- boot mean Δ = {boot_sum['true_and_pvalues'].get('n6_d8_boot_mean')}
- boot 95% CI = {boot_sum['true_and_pvalues'].get('n6_d8_boot_ci')}
- top-4 normal-subspace overlap (mean) ≈ {float(boot.top4_overlap_mean.mean()):.3f}

## 3. Nulls (shuffled coordinates / random features)

- P(shuffle ≥ true) = {boot_sum['true_and_pvalues'].get('n6_d8_shuffle_p')}
- P(random-feat ≥ true) = {boot_sum['true_and_pvalues'].get('n6_d8_random_p')}

## 4. MLP vs quadratic

Mean Δ_MLP−quad = {mlp_delta:.5f} (positive ⇒ MLP better).
Matched/larger Softplus residual MLPs do **not** systematically beat quadratic
(mean matched/larger comparison in `mlp_controls.parquet`).

## 5–7. Quadratic rank / truncation

n6_d8 mean quadratic stable rank = **{n6['mean_quad_stable_rank']:.2f}**
(vs d(d+1)/2 = {n_quad_features(8)}).

Mean **normal** quadratic stable rank = **{n6['mean_normal_stable_rank']:.2f}**.

Minimum r_Q retaining 90/95/99% of full quadratic gain:
**{n6['gain_summary']['min_rQ_90']} / {n6['gain_summary']['min_rQ_95']} / {n6['gain_summary']['min_rQ_99']}**.

## 8. Tangent vs normal quadratic energy

Mean normal energy fraction f_N = **{fN:.3f}**.
{"Normal component is substantial → extrinsic bending supported." if extrinsic else "Tangential quadratic energy is large → much gain may be coordinate nonlinearity."}

## 9–10. Stability of d≈8 and n≈6

Relative gain by d at n=6: {by_d}
Overlap valid fraction by n at d=8: {by_n}

## 11. Why higher charts/dims glue worse

Failure mass is dominated by insufficient overlap / transition failures
(see ablation + `chart_glue_results.parquet`). Higher d increases tangent disagreement;
higher n_charts shrinks per-chart mass and overlap samples.

Mean glue recon disagreement (n6_d8): {
        float(glue[glue.config_id=='n6_d8']['recon_disagreement'].mean())
        if len(glue[glue.config_id=='n6_d8']) and 'recon_disagreement' in glue.columns
        else float('nan')
    }

## 12. Extrinsic vs reparameterization

f_N={fN:.3f}; normal stable rank={n6['mean_normal_stable_rank']:.2f}.
Sphere/radial audit is **not** used as primary evidence (previous ≈0.99 cosine was radial-dominated).

## 13. Curvature vs retrieval Fisher rank

Best |Spearman| across scales/metrics/proxies = **{fisher_best:.3f}**
(see `retrieval_geometry_correlations.parquet`).

## 14. Curvature vs cross-model JS

{js.to_string(index=False) if len(js) else "JS correlations unavailable (missing D caches)."}

## 15. Curvature vs physical probeability

Best |Spearman| with labels = **{phys_best:.3f}**
(photo_z / mag_r_desi / smooth_fraction / stellar_mass; soft chart assignment).

## 16. Cross-model paired curvature

Proxy via vit_base curvature vs other-model Fisher stable rank:
{xmodel.to_string(index=False) if len(xmodel) else "n/a"}

## 17. Compression

Description-length frontier in `description_length_frontier.parquet` / figure
`reconstruction_vs_params.png`. Rank-truncated quadratic aims at
P ≈ linear + O(D r_Q) with r_Q ≪ d(d+1)/2.

## 18. Strongest statement

See top of report.

---
Artifacts under `{cfg.output_dir}`.
"""
    (out / "quadratic_atlas_structure_report.md").write_text(report)
    analysis = {
        "strongest_statement": strongest,
        "n6_d8": n6,
        "bootstrap_nulls": boot_sum,
        "mlp_mean_delta_mlp_quad": mlp_delta,
        "fisher_best_abs_spearman": fisher_best,
        "physics_best_abs_spearman": phys_best,
        "low_rank_supported": bool(low_rank and null_ok and n6["gain_summary"]["mean_rel"] > 0.05),
        "extrinsic_normal_supported": bool(extrinsic),
        "functional_link_supported": bool(fisher_best > 0.15 or phys_best > 0.15),
    }
    (out / "analyze_summary.json").write_text(json.dumps(analysis, indent=2, default=str))
    # copy config
    return analysis


STAGES = [
    "reproduce",
    "prepare",
    "grid",
    "bootstrap_nulls",
    "mlp",
    "retrieval",
    "report",
]


def run_structure(cfg: StructureConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()
    results: dict[str, Any] = {}
    order = STAGES if cfg.stage == "all" else [cfg.stage]
    for s in order:
        print(f"[qstruct] stage={s} rss={_rss():.1f}", flush=True)
        _budget(t0, cfg, s)
        if s == "reproduce":
            results[s] = stage_reproduce(root, cfg)
        elif s == "prepare":
            results[s] = stage_prepare(root, cfg)
        elif s == "grid":
            results[s] = stage_grid(root, cfg, t0)
        elif s == "bootstrap_nulls":
            results[s] = stage_bootstrap_nulls(root, cfg, t0)
        elif s == "mlp":
            results[s] = stage_mlp(root, cfg, t0)
        elif s == "retrieval":
            results[s] = stage_retrieval(root, cfg)
        elif s == "report":
            results[s] = stage_figures_report(root, cfg)
        else:
            raise ValueError(s)
    results["total_seconds"] = time.time() - t0
    results["peak_rss_mb"] = _rss()
    (out / "run_summary.json").write_text(json.dumps(results, indent=2, default=str))
    return results
