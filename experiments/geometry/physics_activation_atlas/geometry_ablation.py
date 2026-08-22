"""Geometry-only atlas ablation: global/local PCA, quadratic, residual MLP."""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy import sparse

from .charts import (
    enforce_chart_population,
    estimate_bandwidths,
    select_chart_centres,
    soft_memberships,
)
from .coordinates import encode_chart, fit_all_charts
from .curvature import (
    mean_curvature_callable_fd,
    mean_curvature_from_J_J2,
    overlap_curvature_agreement_sphere,
    run_curvature_unit_tests,
    validate_fd_vs_autodiff,
)
from .data import load_prepare, prepare_atlas_data, save_prepare, summarize_population
from .decoder import ResidualDecoder, decode_np, jacobian_stats, pca_reconstruct, train_chart_decoder
from .metrics import (
    fit_global_pca,
    global_pca_reconstruct,
    jacobian_stats_numpy,
    median_knn_radius,
    rmse_over_knn_radius,
    variance_normalized_mse,
    weighted_cosine,
    weighted_mse,
)
from .overlap_ablation import evaluate_overlaps_ablation
from .paths import platonic_root, resolve_path
from .quadratic import QuadraticChart, fit_quadratic_chart


@dataclass
class AblationConfig:
    stage: str = "all"
    output_dir: str = "outputs/geometry/physics_activation_atlas_geometry_ablation"
    parquet: str = "data_hf/physics/vit_base_test.parquet"
    column: str = "vit_base_galaxies"
    selection_path: str = (
        "outputs/sae_shared_basis/bsf_block_vae_fisher_physics/selection.npz"
    )
    max_n: int = 16384
    global_seed: int = 0
    seed: int = 0
    n_charts_grid: list[int] = field(default_factory=lambda: [6, 12, 24, 48])
    local_dims: list[int] = field(default_factory=lambda: [8, 16, 32])
    charts_per_sample: int = 3
    chart_selection: str = "fps"
    chart_bandwidth_policy: str = "median_knn"
    min_chart_samples: int = 40
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    decoder_activation: str = "softplus"
    decoder_residual_scale: float = 0.01
    learning_rate: float = 1e-3
    epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    max_decoder_train_samples: int = 1536
    device: str = "cuda"
    top_k_mlp: int = 3
    curvature_anchors: int = 4
    fd_autodiff_anchors: int = 2
    force: bool = False
    max_seconds: float = 7200.0
    max_rss_mb: float = 32000.0

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(path: Path, force: bool) -> bool:
    return path.exists() and not force


def _train_val_split(train_idx: np.ndarray, *, seed: int, val_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    tr = np.asarray(train_idx, dtype=np.int64).copy()
    rng.shuffle(tr)
    n_va = max(1, int(round(val_frac * len(tr))))
    return np.sort(tr[n_va:]), np.sort(tr[:n_va])


def _W_to_idx_w(W: sparse.csr_matrix, r: int):
    n = W.shape[0]
    idx = -np.ones((n, r), dtype=np.int64)
    w = np.zeros((n, r), dtype=np.float64)
    for i in range(n):
        s, e = W.indptr[i], W.indptr[i + 1]
        cols, data = W.indices[s:e], W.data[s:e]
        order = np.argsort(-data)[:r]
        for j, o in enumerate(order):
            idx[i, j] = int(cols[o])
            w[i, j] = float(data[o])
    return idx, w


def _check_budget(t0: float, cfg: AblationConfig, where: str) -> None:
    elapsed = time.time() - t0
    rss = _rss_mb()
    if elapsed > cfg.max_seconds:
        raise RuntimeError(f"Hard stop at {where}: elapsed {elapsed:.1f}s > {cfg.max_seconds}s")
    if rss > cfg.max_rss_mb:
        raise RuntimeError(f"Hard stop at {where}: RSS {rss:.1f} MB > {cfg.max_rss_mb}")


def stage_prepare(root: Path, cfg: AblationConfig) -> dict:
    out = cfg.resolved_out(root) / "prepare"
    marker = out / "population_summary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
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
    (out / "runtime.json").write_text(
        json.dumps({"seconds": time.time() - t0, "peak_rss_mb": _rss_mb()}, indent=2)
    )
    return summary


def _build_charts(X: np.ndarray, tr: np.ndarray, n_charts: int, cfg: AblationConfig):
    Xtr = X[tr]
    centres_tr = select_chart_centres(
        Xtr, n_charts=n_charts, method=cfg.chart_selection, seed=cfg.seed
    )
    bw = estimate_bandwidths(Xtr, centres_tr, policy=cfg.chart_bandwidth_policy)
    centres_g = tr[centres_tr]
    W_full, mem_meta = soft_memberships(
        X, X[centres_g], bw, charts_per_sample=cfg.charts_per_sample
    )
    W, kept, pop_meta = enforce_chart_population(
        W_full, min_chart_samples=cfg.min_chart_samples, max_chart_samples=None
    )
    return W, {
        **mem_meta,
        **pop_meta,
        "n_charts_requested": n_charts,
        "n_charts": int(W.shape[1]),
        "bandwidths": bw[np.asarray(kept)].tolist(),
    }


def _aggregate_recon(per_chart: list[dict], key_mse: str, key_cos: str) -> dict:
    return {
        "mse": float(np.nanmean([c[key_mse] for c in per_chart])),
        "cosine": float(np.nanmean([c[key_cos] for c in per_chart])),
        "var_norm_mse": float(np.nanmean([c["var_norm_mse"] for c in per_chart])),
        "rmse_over_knn": float(np.nanmean([c["rmse_over_knn"] for c in per_chart])),
    }


def _eval_model_on_charts(
    name: str,
    decode_chart: Callable[[int, np.ndarray], np.ndarray],
    X: np.ndarray,
    W: sparse.csr_matrix,
    coords: list[np.ndarray],
    pcas: list[dict],
    eval_idx: np.ndarray,
    knn_radius: float,
    jac_sampler: Callable[[int], dict] | None = None,
) -> dict:
    per = []
    recon_full = {}
    bases = {}
    jac_ok = {}
    for c in range(W.shape[1]):
        w_all = np.asarray(W[:, c].todense()).ravel()
        w = w_all[eval_idx]
        U = coords[c][eval_idx]
        Xc = X[eval_idx]
        pred = decode_chart(c, U)
        recon_full[c] = decode_chart(c, coords[c])
        bases[c] = pcas[c]["basis"]
        row = {
            "chart": c,
            "mse": weighted_mse(pred, Xc, w),
            "cosine": weighted_cosine(pred, Xc, w),
            "var_norm_mse": variance_normalized_mse(pred, Xc, w),
            "rmse_over_knn": rmse_over_knn_radius(pred, Xc, w, knn_radius=knn_radius),
            "n_eff": float(w.sum()),
        }
        if jac_sampler is not None:
            js = jac_sampler(c)
            row.update(js)
            jac_ok[c] = js
        per.append(row)
    agg = _aggregate_recon(per, "mse", "cosine")
    idx_w, ww = _W_to_idx_w(W, min(3, W.shape[1]))
    # restrict overlap eval to eval members by zeroing others? use full for gluing
    ov = evaluate_overlaps_ablation(
        idx_w,
        ww,
        {c: coords[c] for c in range(W.shape[1])},
        bases,
        recon_full,
        jacobians_ok=jac_ok if jac_ok else None,
        min_overlap_mass=5.0,
    )
    return {
        "model": name,
        "heldout": agg,
        "per_chart": per,
        "overlaps": {
            "valid_frac": ov["valid_frac"],
            "n_valid": ov["n_valid"],
            "n_pairs_evaluated": ov["n_pairs_evaluated"],
            "failure_counts": ov["failure_counts"],
            "mean_recon_disagreement": ov["mean_recon_disagreement"],
            "mean_transition_mse": ov["mean_transition_mse"],
            "mean_tangent_disagreement": ov["mean_tangent_disagreement"],
        },
        "overlaps_full": ov,
    }


def stage_grid(root: Path, cfg: AblationConfig, run_t0: float) -> dict:
    out = cfg.resolved_out(root) / "grid"
    marker = out / "grid_summary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    tr_all = data["train_local"]
    te = data["test_local"]
    tr, va = _train_val_split(tr_all, seed=cfg.seed)
    knn_r = median_knn_radius(X, te, k=16)

    rows = []
    for n_charts in cfg.n_charts_grid:
        _check_budget(run_t0, cfg, f"charts_{n_charts}")
        chart_dir = out / f"charts_n{n_charts}"
        w_path = chart_dir / "memberships_csr.npz"
        if _done(w_path, cfg.force):
            W = sparse.load_npz(w_path)
            meta = json.loads((chart_dir / "meta.json").read_text())
        else:
            t0 = time.time()
            W, meta = _build_charts(X, tr_all, n_charts, cfg)
            chart_dir.mkdir(parents=True, exist_ok=True)
            sparse.save_npz(w_path, W)
            meta["runtime"] = {"seconds": time.time() - t0, "peak_rss_mb": _rss_mb()}
            (chart_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        for d in cfg.local_dims:
            _check_budget(run_t0, cfg, f"cfg_{n_charts}_{d}")
            cfg_id = f"n{n_charts}_d{d}"
            cfg_dir = out / cfg_id
            result_path = cfg_dir / "result.json"
            if _done(result_path, cfg.force):
                rows.append(json.loads(result_path.read_text()))
                continue
            t0 = time.time()
            pcas = fit_all_charts(X, W, n_components=d, train_idx=tr_all)
            coords = [encode_chart(X, pca) for pca in pcas]
            # global PCA baseline (same d)
            gpca = fit_global_pca(X[tr_all], d)

            # Local PCA decode
            def dec_lpca(c, U, _pcas=pcas):
                return pca_reconstruct(_pcas[c], U)

            # Fit quadratics on train fit, select ridge on va
            quads = []
            quad_meta = []
            for c, pca in enumerate(pcas):
                w = np.asarray(W[:, c].todense()).ravel()
                q, info = fit_quadratic_chart(
                    pca, coords[c][tr], X[tr], w[tr], coords[c][va], X[va], w[va]
                )
                quads.append(q)
                quad_meta.append(info)

            def dec_quad(c, U, _q=quads):
                return _q[c].decode(U)

            # Jacobian samplers
            def jac_lpca(c, _pcas=pcas, _coords=coords, _te=te):
                # J of Normalize(mu + W u) ≈ sphere projection of basis
                pca = _pcas[c]
                basis = (pca["basis"] * pca["coord_std"]).astype(np.float64)
                # sample a few held-out members
                w = np.asarray(W[:, c].todense()).ravel()
                idx = np.where(w[te] > 1e-4)[0]
                if len(idx) == 0:
                    return {"frac_full_rank": float("nan"), "median_condition": float("nan")}
                rng = np.random.default_rng(cfg.seed + c)
                take = idx[rng.choice(len(idx), size=min(6, len(idx)), replace=False)]
                ranks, conds = [], []
                for ii in take:
                    u = _coords[c][te][ii]
                    # unnormalized J = basis; apply sphere proj at recon point
                    y = pca_reconstruct(pca, u[None, :])[0]
                    J = ((np.eye(len(y)) - np.outer(y, y)) @ basis)
                    js = jacobian_stats_numpy(J)
                    ranks.append(js["full_rank"])
                    conds.append(js["condition"])
                return {
                    "frac_full_rank": float(np.mean(ranks)),
                    "median_condition": float(np.median(conds)),
                }

            def jac_quad(c, _q=quads, _coords=coords, _te=te):
                w = np.asarray(W[:, c].todense()).ravel()
                idx = np.where(w[te] > 1e-4)[0]
                if len(idx) == 0:
                    return {"frac_full_rank": float("nan"), "median_condition": float("nan")}
                rng = np.random.default_rng(cfg.seed + c)
                take = idx[rng.choice(len(idx), size=min(6, len(idx)), replace=False)]
                ranks, conds = [], []
                for ii in take:
                    J = _q[c].jacobian_at(_coords[c][te][ii])
                    js = jacobian_stats_numpy(J)
                    ranks.append(js["full_rank"])
                    conds.append(js["condition"])
                return {
                    "frac_full_rank": float(np.mean(ranks)),
                    "median_condition": float(np.median(conds)),
                }

            # Global PCA held-out (not chart-weighted average of charts — single global)
            g_pred = global_pca_reconstruct(gpca, X[te])
            w_te_uniform = np.ones(len(te), dtype=np.float64)
            global_metrics = {
                "model": "global_pca",
                "heldout": {
                    "mse": weighted_mse(g_pred, X[te], w_te_uniform),
                    "cosine": weighted_cosine(g_pred, X[te], w_te_uniform),
                    "var_norm_mse": variance_normalized_mse(g_pred, X[te], w_te_uniform),
                    "rmse_over_knn": rmse_over_knn_radius(
                        g_pred, X[te], w_te_uniform, knn_radius=knn_r
                    ),
                },
            }

            # Val metrics for model selection (no test)
            def val_mse_for(decode_fn):
                mses = []
                for c in range(W.shape[1]):
                    w = np.asarray(W[:, c].todense()).ravel()
                    pred = decode_fn(c, coords[c][va])
                    mses.append(weighted_mse(pred, X[va], w[va]))
                return float(np.nanmean(mses))

            lpca_val = val_mse_for(dec_lpca)
            quad_val = val_mse_for(dec_quad)

            lpca_res = _eval_model_on_charts(
                "local_pca", dec_lpca, X, W, coords, pcas, te, knn_r, jac_lpca
            )
            quad_res = _eval_model_on_charts(
                "quadratic", dec_quad, X, W, coords, pcas, te, knn_r, jac_quad
            )

            g_mse = global_metrics["heldout"]["mse"]
            for res in (lpca_res, quad_res):
                res["improvement_vs_global_pca"] = float(g_mse - res["heldout"]["mse"])
                res["improvement_vs_local_pca"] = float(
                    lpca_res["heldout"]["mse"] - res["heldout"]["mse"]
                )

            result = {
                "config_id": cfg_id,
                "n_charts_requested": n_charts,
                "n_charts": int(W.shape[1]),
                "local_dim": d,
                "knn_radius_test": knn_r,
                "val_mse_local_pca": lpca_val,
                "val_mse_quadratic": quad_val,
                "quadratic_beats_pca_val": bool(quad_val < lpca_val - 1e-4),
                "quad_ridge_mean": float(np.nanmean([m["ridge"] for m in quad_meta])),
                "global_pca": global_metrics,
                "local_pca": {
                    k: v for k, v in lpca_res.items() if k != "overlaps_full"
                },
                "quadratic": {
                    k: v for k, v in quad_res.items() if k != "overlaps_full"
                },
                "local_pca_failure_counts": lpca_res["overlaps"]["failure_counts"],
                "quadratic_failure_counts": quad_res["overlaps"]["failure_counts"],
                "selection_score": float(
                    -lpca_val
                    + 0.02 * lpca_res["overlaps"]["valid_frac"]
                    + 0.5 * max(0.0, lpca_val - quad_val)
                ),
                "runtime": {"seconds": time.time() - t0, "peak_rss_mb": _rss_mb()},
            }
            # persist heavy overlap separately
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "overlaps_local_pca.json").write_text(
                json.dumps(lpca_res["overlaps_full"], indent=2)
            )
            (cfg_dir / "overlaps_quadratic.json").write_text(
                json.dumps(quad_res["overlaps_full"], indent=2)
            )
            # save PCA/quad artifacts lightly
            np.savez_compressed(
                cfg_dir / "pcas.npz",
                **{f"mu_{c}": pcas[c]["mu"] for c in range(len(pcas))},
                **{f"basis_{c}": pcas[c]["basis"] for c in range(len(pcas))},
                **{f"std_{c}": pcas[c]["coord_std"] for c in range(len(pcas))},
            )
            np.savez_compressed(
                cfg_dir / "quad_B.npz",
                **{f"B_{c}": quads[c].B_flat for c in range(len(quads))},
                ridges=np.array([q.ridge for q in quads]),
            )
            (result_path).write_text(json.dumps(result, indent=2))
            rows.append(result)
            print(
                f"[ablation] {cfg_id} lpca_mse={lpca_res['heldout']['mse']:.4f} "
                f"quad_mse={quad_res['heldout']['mse']:.4f} "
                f"ov={lpca_res['overlaps']['valid_frac']:.2f} rss={_rss_mb():.0f}",
                flush=True,
            )

    # select top-k for MLP without looking at test-held metrics for ranking:
    # use selection_score (val-based)
    ranked = sorted(rows, key=lambda r: -r["selection_score"])
    top = [
        {"config_id": r["config_id"], "selection_score": r["selection_score"],
         "val_mse_local_pca": r["val_mse_local_pca"], "val_mse_quadratic": r["val_mse_quadratic"],
         "n_charts": r["n_charts"], "local_dim": r["local_dim"]}
        for r in ranked[: cfg.top_k_mlp]
    ]
    summary = {
        "n_configs": len(rows),
        "rows": rows,
        "top_mlp_configs": top,
        "knn_radius_test": knn_r,
        "peak_rss_mb": _rss_mb(),
    }
    (out / "grid_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_mlp(root: Path, cfg: AblationConfig, run_t0: float) -> dict:
    out = cfg.resolved_out(root) / "mlp"
    marker = out / "mlp_summary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    grid = json.loads((cfg.resolved_out(root) / "grid" / "grid_summary.json").read_text())
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    tr_all = data["train_local"]
    te = data["test_local"]
    tr, va = _train_val_split(tr_all, seed=cfg.seed)
    knn_r = grid["knn_radius_test"]
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"

    results = []
    for sel in grid["top_mlp_configs"]:
        _check_budget(run_t0, cfg, f"mlp_{sel['config_id']}")
        cfg_id = sel["config_id"]
        n_charts = int(cfg_id.split("_")[0][1:])
        d = int(cfg_id.split("_")[1][1:])
        W = sparse.load_npz(cfg.resolved_out(root) / "grid" / f"charts_n{n_charts}" / "memberships_csr.npz")
        # reload pcas
        pcas = fit_all_charts(X, W, n_components=d, train_idx=tr_all)
        coords = [encode_chart(X, pca) for pca in pcas]
        models = []
        chart_metrics = []
        t0 = time.time()
        for c, pca in enumerate(pcas):
            w = np.asarray(W[:, c].todense()).ravel()
            model, metrics = train_chart_decoder(
                pca,
                coords[c][tr],
                X[tr],
                w[tr],
                coords[c][va],
                X[va],
                w[va],
                hidden=cfg.decoder_hidden_dims,
                activation=cfg.decoder_activation,
                residual_scale=cfg.decoder_residual_scale,
                output_normalize=True,
                lr=cfg.learning_rate,
                epochs=cfg.epochs,
                patience=cfg.patience,
                batch_size=cfg.batch_size,
                device=device,
                max_train=cfg.max_decoder_train_samples,
                seed=cfg.seed + c,
            )
            models.append(model)
            # jacobian on test members
            idx = np.where(w[te] > 1e-4)[0]
            ranks, conds = [], []
            if len(idx):
                rng = np.random.default_rng(cfg.seed + c)
                take = idx[rng.choice(len(idx), size=min(6, len(idx)), replace=False)]
                for ii in take:
                    js = jacobian_stats(model, coords[c][te][ii], device)
                    ranks.append(js["full_rank"])
                    conds.append(js["condition"])
            chart_metrics.append(
                {
                    **metrics,
                    "frac_full_rank": float(np.mean(ranks)) if ranks else float("nan"),
                    "median_condition": float(np.median(conds)) if conds else float("nan"),
                }
            )
            torch.save({"state_dict": model.state_dict(), "metrics": metrics}, out / f"{cfg_id}_chart{c}.pt")

        def dec_mlp(c, U, _m=models, _dev=device):
            return decode_np(_m[c], U, _dev)

        def jac_mlp(c):
            return {
                "frac_full_rank": chart_metrics[c]["frac_full_rank"],
                "median_condition": chart_metrics[c]["median_condition"],
            }

        mlp_res = _eval_model_on_charts(
            "mlp", dec_mlp, X, W, coords, pcas, te, knn_r, jac_mlp
        )
        # baselines from grid
        grid_row = next(r for r in grid["rows"] if r["config_id"] == cfg_id)
        mlp_res["improvement_vs_global_pca"] = float(
            grid_row["global_pca"]["heldout"]["mse"] - mlp_res["heldout"]["mse"]
        )
        mlp_res["improvement_vs_local_pca"] = float(
            grid_row["local_pca"]["heldout"]["mse"] - mlp_res["heldout"]["mse"]
        )
        mlp_res["improvement_vs_quadratic"] = float(
            grid_row["quadratic"]["heldout"]["mse"] - mlp_res["heldout"]["mse"]
        )
        row = {
            "config_id": cfg_id,
            "n_charts": int(W.shape[1]),
            "local_dim": d,
            "mlp": {k: v for k, v in mlp_res.items() if k != "overlaps_full"},
            "failure_counts": mlp_res["overlaps"]["failure_counts"],
            "baselines": {
                "global_pca": grid_row["global_pca"]["heldout"],
                "local_pca": grid_row["local_pca"]["heldout"],
                "quadratic": grid_row["quadratic"]["heldout"],
            },
            "runtime": {"seconds": time.time() - t0, "peak_rss_mb": _rss_mb()},
        }
        (out / f"{cfg_id}_result.json").write_text(json.dumps(row, indent=2))
        (out / f"{cfg_id}_overlaps.json").write_text(json.dumps(mlp_res["overlaps_full"], indent=2))
        results.append(row)
        # free
        del models
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[ablation] mlp {cfg_id} mse={row['mlp']['heldout']['mse']:.4f}", flush=True)

    summary = {"results": results, "peak_rss_mb": _rss_mb()}
    (out / "mlp_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_curvature(root: Path, cfg: AblationConfig, run_t0: float) -> dict:
    out = cfg.resolved_out(root) / "curvature"
    marker = out / "curvature_summary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    unit = run_curvature_unit_tests(device="cpu")
    grid = json.loads((cfg.resolved_out(root) / "grid" / "grid_summary.json").read_text())
    mlp = json.loads((cfg.resolved_out(root) / "mlp" / "mlp_summary.json").read_text())
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    tr_all = data["train_local"]
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # pick best MLP config for detailed curvature
    best = min(mlp["results"], key=lambda r: r["mlp"]["heldout"]["mse"])
    cfg_id = best["config_id"]
    n_charts = int(cfg_id.split("_")[0][1:])
    d = int(cfg_id.split("_")[1][1:])
    W = sparse.load_npz(cfg.resolved_out(root) / "grid" / f"charts_n{n_charts}" / "memberships_csr.npz")
    pcas = fit_all_charts(X, W, n_components=d, train_idx=tr_all)
    coords = [encode_chart(X, pca) for pca in pcas]
    # reload quads
    from .quadratic import QuadraticChart

    qb = np.load(cfg.resolved_out(root) / "grid" / cfg_id / "quad_B.npz")
    quads = []
    for c, pca in enumerate(pcas):
        quads.append(
            QuadraticChart(
                mu=pca["mu"].astype(np.float64),
                basis=(pca["basis"] * pca["coord_std"]).astype(np.float64),
                B_flat=qb[f"B_{c}"],
                ridge=float(qb["ridges"][c]),
                output_normalize=True,
            )
        )
    # load MLP chart 0 for FD validation
    models = []
    for c in range(W.shape[1]):
        path = cfg.resolved_out(root) / "mlp" / f"{cfg_id}_chart{c}.pt"
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        m = ResidualDecoder(
            d=d,
            ambient=X.shape[1],
            mu=pcas[c]["mu"],
            basis=pcas[c]["basis"] * pcas[c]["coord_std"],
            hidden=cfg.decoder_hidden_dims,
            activation=cfg.decoder_activation,
            residual_scale=cfg.decoder_residual_scale,
            output_normalize=True,
        ).to(device)
        m.load_state_dict(ckpt["state_dict"])
        models.append(m)

    # FD vs autodiff: nested autodiff is feasible only for small d; otherwise FD step-size only.
    w0 = np.asarray(W[:, 0].todense()).ravel()
    cand = np.where(w0 > 1e-3)[0]
    rng = np.random.default_rng(cfg.seed)
    take = cand[rng.choice(len(cand), size=min(cfg.fd_autodiff_anchors, len(cand)), replace=False)]
    if d <= 8:
        fd_val = validate_fd_vs_autodiff(
            models[0],
            [coords[0][i] for i in take],
            device=device,
            hs=[1e-2, 3e-3, 1e-3],
        )
    else:
        # step-size sensitivity via FD only
        rows = []
        for ai, i in enumerate(take):
            Hs = []
            for h in [1e-2, 3e-3, 1e-3]:
                fd = mean_curvature_callable_fd(
                    lambda u, _m=models[0], _dev=device: decode_np(_m, np.asarray(u)[None, :], _dev)[0],
                    coords[0][i],
                    h=h,
                )
                Hs.append(fd)
                rows.append(
                    {
                        "anchor": ai,
                        "h": h,
                        "H_norm": fd["H_norm"],
                        "H_sphere_norm": fd["H_sphere_norm"],
                        "radial_fraction": fd["radial_fraction"],
                    }
                )
            # compare h=1e-3 vs 3e-3
            err = float(np.linalg.norm(Hs[2]["H_sphere"] - Hs[1]["H_sphere"]))
            rows.append({"anchor": ai, "sphere_step_sensitivity_1e3_vs_3e3": err})
        fd_val = {
            "note": f"autodiff skipped for d={d}>8; reporting FD step-size sensitivity only",
            "comparisons": rows,
            "mean_sphere_error_h1e3": float("nan"),
        }

    def _curv_lpca(c: int, u: np.ndarray) -> dict:
        x = pca_reconstruct(pcas[c], np.asarray(u)[None, :])[0].astype(np.float64)
        basis = (pcas[c]["basis"] * pcas[c]["coord_std"]).astype(np.float64)
        J = (np.eye(len(x)) - np.outer(x, x)) @ basis
        # normalized linear chart: second fundamental form from sphere constraint only
        # Use FD on analytic J of Normalize(mu+Wu)
        h = 1e-3
        dloc = u.shape[0]
        J2 = np.zeros((len(x), dloc, dloc))
        for j in range(dloc):
            uj = np.asarray(u, dtype=np.float64).copy()
            uj[j] += h
            xj = pca_reconstruct(pcas[c], uj[None, :])[0].astype(np.float64)
            Jj = (np.eye(len(xj)) - np.outer(xj, xj)) @ basis
            J2[:, :, j] = (Jj - J) / h
        J2 = 0.5 * (J2 + J2.transpose(0, 2, 1))
        H = mean_curvature_from_J_J2(J, J2)
        return {"H": H, "x": x}

    def _curv_quad(c: int, u: np.ndarray) -> dict:
        u = np.asarray(u, dtype=np.float64)
        x = quads[c].decode(u[None, :])[0].astype(np.float64)
        J = quads[c].jacobian_at(u)
        J2 = quads[c].hessian_at(u)
        H = mean_curvature_from_J_J2(J, J2)
        return {"H": H, "x": x}

    def _curv_mlp(c: int, u: np.ndarray) -> dict:
        fd = mean_curvature_callable_fd(
            lambda uu, _c=c: decode_np(models[_c], np.asarray(uu)[None, :], device)[0],
            u,
            h=1e-3,
        )
        return {"H": fd["H"], "x": fd["x"]}

    def agreement_for(curv_fn, model_name: str, max_pairs: int = 3, max_anchors: int = 2):
        ov = json.loads(
            (cfg.resolved_out(root) / "grid" / cfg_id / "overlaps_local_pca.json").read_text()
        )
        pairs = [p for p in ov.get("pairs", []) if p.get("valid")][:max_pairs]
        if not pairs:
            pairs = sorted(ov.get("pairs", []), key=lambda p: -p.get("overlap_mass", 0))[:max_pairs]
        agreements = []
        Wd = np.asarray(W.todense())
        for pair in pairs:
            a, b = int(pair["chart_a"]), int(pair["chart_b"])
            mass = np.minimum(Wd[:, a], Wd[:, b])
            idxs = np.where(mass > 1e-3)[0]
            if len(idxs) < 2:
                continue
            samp = idxs[rng.choice(len(idxs), size=min(max_anchors, len(idxs)), replace=False)]
            Ha, Hb, Xa, Xb = {}, {}, {}, {}
            for i in samp:
                try:
                    ca = curv_fn(a, coords[a][i])
                    cb = curv_fn(b, coords[b][i])
                except Exception:  # noqa: BLE001
                    continue
                Ha[int(i)] = ca["H"]
                Hb[int(i)] = cb["H"]
                Xa[int(i)] = ca["x"]
                Xb[int(i)] = cb["x"]
            if Ha and Hb:
                agreements.append(overlap_curvature_agreement_sphere(Ha, Hb, Xa, Xb))
        if not agreements:
            return {"n_pairs": 0, "model": model_name}
        return {
            "model": model_name,
            "n_pairs": len(agreements),
            "full_H_cosine_mean": float(np.nanmean([a["full_H_cosine_mean"] for a in agreements])),
            "radial_fraction_mean": float(np.nanmean([a["radial_fraction_mean"] for a in agreements])),
            "H_sphere_cosine_mean": float(np.nanmean([a["H_sphere_cosine_mean"] for a in agreements])),
            "H_sphere_rel_norm_diff_mean": float(
                np.nanmean([a["H_sphere_rel_norm_diff_mean"] for a in agreements])
            ),
            "radial_dominated": bool(
                np.nanmean([a["radial_fraction_mean"] for a in agreements]) > 0.8
            ),
            "pairs": agreements,
        }

    payload = {
        "unit_tests": unit,
        "config_id": cfg_id,
        "fd_vs_autodiff": fd_val,
        "agreement_local_pca": agreement_for(_curv_lpca, "local_pca"),
        "agreement_quadratic": agreement_for(_curv_quad, "quadratic"),
        "agreement_mlp": agreement_for(_curv_mlp, "mlp", max_pairs=2, max_anchors=2),
        "runtime": {"peak_rss_mb": _rss_mb()},
    }
    (out / "curvature_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


def stage_analyze(root: Path, cfg: AblationConfig) -> dict:
    out = cfg.resolved_out(root)
    grid = json.loads((out / "grid" / "grid_summary.json").read_text())
    mlp = json.loads((out / "mlp" / "mlp_summary.json").read_text())
    curv = json.loads((out / "curvature" / "curvature_summary.json").read_text())
    prep = json.loads((out / "prepare" / "population_summary.json").read_text())

    table = []
    for r in grid["rows"]:
        table.append(
            {
                "config_id": r["config_id"],
                "n_charts": r["n_charts"],
                "local_dim": r["local_dim"],
                "global_mse": r["global_pca"]["heldout"]["mse"],
                "local_pca_mse": r["local_pca"]["heldout"]["mse"],
                "quadratic_mse": r["quadratic"]["heldout"]["mse"],
                "local_pca_cos": r["local_pca"]["heldout"]["cosine"],
                "quadratic_cos": r["quadratic"]["heldout"]["cosine"],
                "local_pca_var_norm_mse": r["local_pca"]["heldout"]["var_norm_mse"],
                "quadratic_var_norm_mse": r["quadratic"]["heldout"]["var_norm_mse"],
                "local_pca_rmse_knn": r["local_pca"]["heldout"]["rmse_over_knn"],
                "quad_vs_lpca": r["quadratic"]["improvement_vs_local_pca"],
                "lpca_vs_global": r["local_pca"]["improvement_vs_global_pca"],
                "lpca_overlap_valid_frac": r["local_pca"]["overlaps"]["valid_frac"],
                "quad_overlap_valid_frac": r["quadratic"]["overlaps"]["valid_frac"],
                "lpca_fail": r["local_pca_failure_counts"],
                "quad_fail": r["quadratic_failure_counts"],
            }
        )
    for r in mlp["results"]:
        for row in table:
            if row["config_id"] == r["config_id"]:
                row["mlp_mse"] = r["mlp"]["heldout"]["mse"]
                row["mlp_cos"] = r["mlp"]["heldout"]["cosine"]
                row["mlp_vs_quad"] = r["mlp"]["improvement_vs_quadratic"]
                row["mlp_vs_lpca"] = r["mlp"]["improvement_vs_local_pca"]
                row["mlp_overlap_valid_frac"] = r["mlp"]["overlaps"]["valid_frac"]
                row["mlp_fail"] = r["failure_counts"]

    # absolute adequacy: local PCA RMSE / knn radius
    lpca_rmses = [r["local_pca_rmse_knn"] for r in table]
    mean_lpca_rel = float(np.nanmean(lpca_rmses))
    local_pca_adequate = mean_lpca_rel < 1.0  # error below typical neighbour scale

    quad_wins = sum(1 for r in table if r["quad_vs_lpca"] > 1e-4)
    quad_beats = quad_wins >= max(1, len(table) // 3)

    mlp_beats_quad = False
    mlp_rows = [r for r in table if "mlp_mse" in r]
    if mlp_rows:
        mlp_beats_quad = bool(np.nanmean([r.get("mlp_vs_quad", 0) for r in mlp_rows]) > 1e-4)

    best_overlap = max(table, key=lambda r: r["lpca_overlap_valid_frac"])
    # failure breakdown aggregate
    fail_agg = {}
    for r in table:
        for k, v in r["lpca_fail"].items():
            fail_agg[k] = fail_agg.get(k, 0) + v

    agr_mlp = curv.get("agreement_mlp", {})
    agr_lpca = curv.get("agreement_local_pca", {})
    radial_dom = bool(
        agr_mlp.get("radial_dominated")
        or agr_lpca.get("radial_dominated")
        or (agr_mlp.get("radial_fraction_mean", 0) > 0.8)
    )
    sph_ok = bool(agr_mlp.get("H_sphere_cosine_mean", 0) > 0.5) if agr_mlp.get("n_pairs", 0) else False

    # conclusion
    gluing_bad = best_overlap["lpca_overlap_valid_frac"] < 0.5
    if gluing_bad and not local_pca_adequate:
        conclusion = "smooth_manifold_assumption_doubtful"
    elif gluing_bad:
        conclusion = "atlas_gluing_failure"
    elif mlp_beats_quad and quad_beats:
        conclusion = "nonlinear_decoder_needed"
    elif quad_beats:
        conclusion = "quadratic_geometry_detected"
    else:
        conclusion = "locally_affine_at_measured_scale"

    # runtime
    rt = {
        "prepare": json.loads((out / "prepare" / "runtime.json").read_text())
        if (out / "prepare" / "runtime.json").exists()
        else {},
        "grid_peak_rss": grid.get("peak_rss_mb"),
        "mlp_peak_rss": mlp.get("peak_rss_mb"),
        "grid_seconds": float(sum(r.get("runtime", {}).get("seconds", 0) for r in grid["rows"])),
        "mlp_seconds": float(sum(r.get("runtime", {}).get("seconds", 0) for r in mlp["results"])),
        "peak_rss_mb": max(
            float(grid.get("peak_rss_mb") or 0),
            float(mlp.get("peak_rss_mb") or 0),
            _rss_mb(),
        ),
    }
    rt["total_seconds_approx"] = rt["grid_seconds"] + rt["mlp_seconds"] + float(
        rt["prepare"].get("seconds", 0)
    )

    analysis = {
        "conclusion": conclusion,
        "local_pca_adequate_absolute": local_pca_adequate,
        "mean_lpca_rmse_over_knn": mean_lpca_rel,
        "quadratic_beats_pca": quad_beats,
        "n_configs_quad_wins": quad_wins,
        "mlp_beats_quadratic": mlp_beats_quad,
        "best_overlap_config": {
            "config_id": best_overlap["config_id"],
            "valid_frac": best_overlap["lpca_overlap_valid_frac"],
        },
        "failure_counts_lpca_sum": fail_agg,
        "table": table,
        "top_mlp_configs": grid["top_mlp_configs"],
        "fd_vs_autodiff": curv.get("fd_vs_autodiff"),
        "curvature_agreements": {
            "local_pca": agr_lpca,
            "quadratic": curv.get("agreement_quadratic"),
            "mlp": agr_mlp,
        },
        "previous_0.99_was_radial_dominated": radial_dom,
        "sphere_tangent_agreement_stable": sph_ok,
        "unit_tests_pass": curv.get("unit_tests", {}).get("all_pass"),
        "data": {
            "n_selected": prep.get("n_selected"),
            "ambient_dim": prep.get("ambient_dim"),
            "n_train": prep.get("n_train"),
            "n_test": prep.get("n_test"),
        },
        "runtime": rt,
    }

    report = [
        "# Physics activation atlas — geometry-only ablation",
        "",
        f"**Conclusion:** `{conclusion}`",
        "",
        "## Runtime",
        "```json",
        json.dumps(rt, indent=2),
        "```",
        "",
        "## Reconstruction table",
        "```json",
        json.dumps(table, indent=2)[:12000],
        "```",
        "",
        "## Overlap failures (local PCA, summed)",
        "```json",
        json.dumps(fail_agg, indent=2),
        "```",
        "",
        "## Curvature",
        "```json",
        json.dumps(analysis["curvature_agreements"], indent=2)[:8000],
        "```",
        "",
        "## FD vs autodiff",
        "```json",
        json.dumps(curv.get("fd_vs_autodiff"), indent=2)[:6000],
        "```",
        "",
        f"Local PCA absolute adequacy (RMSE/knn < 1): {local_pca_adequate} "
        f"(mean={mean_lpca_rel:.3f})",
        f"Quadratic beats PCA: {quad_beats} ({quad_wins}/{len(table)} configs)",
        f"MLP beats quadratic: {mlp_beats_quad}",
        f"Best overlap: {best_overlap['config_id']} valid_frac={best_overlap['lpca_overlap_valid_frac']:.3f}",
        f"Previous ~0.99 cosine radial-dominated: {radial_dom}",
        "",
    ]
    (out / "analyze").mkdir(parents=True, exist_ok=True)
    (out / "analyze" / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (out / "analyze" / "REPORT.md").write_text("\n".join(report))
    return analysis


STAGES = ["prepare", "grid", "mlp", "curvature", "analyze"]


def run_ablation(cfg: AblationConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out / "schema_version.json").write_text(
        json.dumps({"schema_version": 1, "experiment": "physics_activation_atlas_geometry_ablation"}, indent=2)
    )
    run_t0 = time.time()
    results: dict[str, Any] = {}
    order = STAGES if cfg.stage == "all" else [cfg.stage]
    for s in order:
        print(f"[ablation] stage={s} rss={_rss_mb():.1f}", flush=True)
        _check_budget(run_t0, cfg, s)
        if s == "prepare":
            results[s] = stage_prepare(root, cfg)
        elif s == "grid":
            results[s] = stage_grid(root, cfg, run_t0)
        elif s == "mlp":
            results[s] = stage_mlp(root, cfg, run_t0)
        elif s == "curvature":
            results[s] = stage_curvature(root, cfg, run_t0)
        elif s == "analyze":
            results[s] = stage_analyze(root, cfg)
        else:
            raise ValueError(s)
    return results
