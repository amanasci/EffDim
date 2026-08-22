"""Resumable staged pipeline for the Physics activation atlas experiment."""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse

from .boundary import chart_boundary_diagnostics, save_boundary
from .charts import (
    enforce_chart_population,
    estimate_bandwidths,
    save_charts,
    select_chart_centres,
    soft_memberships,
)
from .coordinates import encode_chart, fit_all_charts, save_coordinates
from .curvature import (
    evaluate_chart_curvature,
    overlap_curvature_agreement,
    run_curvature_unit_tests,
    save_curvature,
)
from .data import (
    load_prepare,
    prepare_atlas_data,
    save_prepare,
    summarize_population,
)
from .decoder import (
    decode_np,
    jacobian_stats,
    pca_reconstruct,
    save_decoder,
    train_chart_decoder,
    ResidualDecoder,
)
from .nerve import run_nerve_analysis, save_nerve
from .overlaps import evaluate_overlaps, save_overlaps
from .patch_topology import patchwise_topology, save_patch_topology
from .paths import platonic_root, resolve_path
from .priors import DiagGMM, save_priors, select_prior, weighted_loglik
from .plots import write_smoke_plots
from .report import write_report
from .synthetic import save_synthetic, validate_synthetic_atlas


@dataclass
class AtlasConfig:
    stage: str = "all"
    output_dir: str = "outputs/geometry/physics_activation_atlas"
    parquet: str = "data_hf/physics/vit_base_test.parquet"
    column: str = "vit_base_galaxies"
    selection_path: str = (
        "outputs/sae_shared_basis/bsf_block_vae_fisher_physics/selection.npz"
    )
    max_n: int = 16384
    global_seed: int = 0
    seed: int = 0
    n_charts: int = 12
    charts_per_sample: int = 3
    chart_selection: str = "fps"
    chart_bandwidth_policy: str = "median_knn"
    min_chart_samples: int = 40
    max_chart_samples: int | None = None
    candidate_dims: list[int] = field(default_factory=lambda: [8, 16])
    latent_dim: int | None = None  # if None, choose from candidates on val
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    decoder_activation: str = "softplus"
    decoder_residual_scale: float = 0.01
    decoder_output_normalization: bool = True
    learning_rate: float = 1e-3
    epochs: int = 40
    patience: int = 6
    batch_size: int = 256
    max_decoder_train_samples: int = 2048
    device: str = "cuda"
    prior_family: str = "all"
    mixture_components: list[int] = field(default_factory=lambda: [1, 2, 4])
    cf_frequency_count: int = 256
    cf_frequency_scales: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    cf_loss_weight: float = 1.0
    prior_learning_rate: float = 0.02
    prior_epochs: int = 40
    prior_patience: int = 8
    patch_max_points: int = 400
    nerve_maxdim: int = 2
    curvature_anchors: int = 8
    n_seeds: int = 1
    smoke: bool = False
    force: bool = False
    synth_n: int = 400

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(path: Path, force: bool) -> bool:
    return path.exists() and not force


def save_config(out: Path, cfg: AtlasConfig) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out / "schema_version.json").write_text(
        json.dumps({"schema_version": 1, "experiment": "physics_activation_atlas"}, indent=2)
    )


def _stage_timer(name: str, t0: float) -> dict:
    return {"stage": name, "seconds": time.time() - t0, "peak_rss_mb": _rss_mb()}


def _W_to_idx_w(W: sparse.csr_matrix, r: int) -> tuple[np.ndarray, np.ndarray]:
    n, C = W.shape
    idx = -np.ones((n, r), dtype=np.int64)
    w = np.zeros((n, r), dtype=np.float64)
    for i in range(n):
        s, e = W.indptr[i], W.indptr[i + 1]
        cols = W.indices[s:e]
        data = W.data[s:e]
        order = np.argsort(-data)[:r]
        for j, o in enumerate(order):
            idx[i, j] = int(cols[o])
            w[i, j] = float(data[o])
    return idx, w


def _dense_weights(W: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(W.todense(), dtype=np.float64)


def _train_val_split(train_idx: np.ndarray, *, seed: int, val_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Internal validation from training only; test/holdout remain final-eval."""
    rng = np.random.default_rng(seed)
    tr = np.asarray(train_idx, dtype=np.int64)
    rng.shuffle(tr)
    n_va = max(1, int(round(val_frac * len(tr))))
    return np.sort(tr[n_va:]), np.sort(tr[:n_va])


def stage_prepare(root: Path, cfg: AtlasConfig) -> dict:
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
    (out / "runtime.json").write_text(json.dumps(_stage_timer("prepare", t0), indent=2))
    return summary


def stage_synthetic(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root) / "synthetic"
    marker = out / "synthetic_validation.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
    result = validate_synthetic_atlas(
        n=cfg.synth_n if not cfg.smoke else min(cfg.synth_n, 300),
        n_charts=8 if cfg.smoke else 12,
        charts_per_sample=3,
        latent_dim=8,
        seed=cfg.seed,
        ambient=32,
    )
    result["runtime"] = _stage_timer("synthetic", t0)
    save_synthetic(out, result)
    return result


def stage_charts(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root) / "charts"
    marker = out / "charts_meta.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    tr = data["train_local"]
    Xtr = X[tr]
    centres_tr = select_chart_centres(
        Xtr, n_charts=cfg.n_charts, method=cfg.chart_selection, seed=cfg.seed
    )
    bw = estimate_bandwidths(Xtr, centres_tr, policy=cfg.chart_bandwidth_policy)
    centres_global = tr[centres_tr]
    W_full, mem_meta = soft_memberships(
        X, X[centres_global], bw, charts_per_sample=cfg.charts_per_sample
    )
    W, kept, pop_meta = enforce_chart_population(
        W_full,
        min_chart_samples=cfg.min_chart_samples,
        max_chart_samples=cfg.max_chart_samples,
    )
    # remapped bandwidths/centres
    bw_kept = bw[np.asarray(kept)]
    centres_tr_kept = centres_tr[np.asarray(kept)]
    centres_g_kept = centres_global[np.asarray(kept)]
    pi = np.asarray(W.sum(axis=0)).ravel() / max(W.shape[0], 1)
    meta = {
        **mem_meta,
        **pop_meta,
        "n_charts": int(W.shape[1]),
        "pi": pi.tolist(),
        "bandwidths": bw_kept.tolist(),
        "runtime": _stage_timer("charts", t0),
    }
    save_charts(
        out,
        {
            "centres_train_local": centres_tr_kept,
            "centres_global_local": centres_g_kept,
            "bandwidths": bw_kept,
            "kept_original_ids": np.asarray(kept),
            "W": W,
            "meta": meta,
        },
    )
    return meta


def stage_coordinates(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root) / "coordinates"
    marker = out / "coordinates_meta.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    tr_fit, va = _train_val_split(data["train_local"], seed=cfg.seed)
    W = sparse.load_npz(cfg.resolved_out(root) / "charts" / "memberships_csr.npz")
    dims = cfg.candidate_dims if cfg.latent_dim is None else [cfg.latent_dim]
    # select dim by mean weighted val PCA recon
    dim_scores = []
    for d in dims:
        pcas = fit_all_charts(X, W, n_components=d, train_idx=tr_fit)
        scores = []
        for c, pca in enumerate(pcas):
            w = np.asarray(W[:, c].todense()).ravel()
            U = encode_chart(X, pca)
            pred = pca_reconstruct(pca, U[va])
            ww = w[va]
            if ww.sum() <= 0:
                continue
            mse = float(np.sum(ww * ((pred - X[va]) ** 2).sum(1)) / ww.sum())
            scores.append(mse)
        dim_scores.append({"dim": d, "mean_val_pca_mse": float(np.mean(scores)) if scores else float("inf")})
    best_dim = int(sorted(dim_scores, key=lambda z: z["mean_val_pca_mse"])[0]["dim"])
    if cfg.latent_dim is not None:
        best_dim = cfg.latent_dim
    # Final PCA bases use full training memberships (not the internal val slice).
    pcas = fit_all_charts(X, W, n_components=best_dim, train_idx=data["train_local"])
    coords = {}
    for c, pca in enumerate(pcas):
        U = encode_chart(X, pca)
        coords[f"U_chart{c}"] = U
        coords[f"w_chart{c}"] = np.asarray(W[:, c].todense()).ravel().astype(np.float32)
    meta = {
        "selected_dim": best_dim,
        "dim_scores": dim_scores,
        "n_train_fit": int(len(tr_fit)),
        "n_train_val": int(len(va)),
        "per_chart": [
            {
                "chart": c,
                **pca["diagnostics"],
                "n_members": pca["n_members"],
                "n_effective": pca["n_effective"],
            }
            for c, pca in enumerate(pcas)
        ],
        "runtime": _stage_timer("coordinates", t0),
    }
    save_coordinates(out, pcas, coords, meta)
    return meta


def stage_train(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root) / "train"
    marker = out / "train_summary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    tr, va = _train_val_split(data["train_local"], seed=cfg.seed)
    W = sparse.load_npz(cfg.resolved_out(root) / "charts" / "memberships_csr.npz")
    coord_dir = cfg.resolved_out(root) / "coordinates"
    meta_c = json.loads((coord_dir / "coordinates_meta.json").read_text())
    d = int(meta_c["selected_dim"])
    z = np.load(coord_dir / "coords.npz")
    chart_metrics = []
    jac_rows = []
    for c in range(W.shape[1]):
        pca_z = np.load(coord_dir / f"pca_chart{c}.npz")
        pca = {
            "mu": pca_z["mu"],
            "basis": pca_z["basis"],
            "coord_std": pca_z["coord_std"],
            "eigenvalues": pca_z["eigenvalues"],
        }
        U = z[f"U_chart{c}"]
        w = z[f"w_chart{c}"].astype(np.float64)
        model, metrics = train_chart_decoder(
            pca,
            U[tr],
            X[tr],
            w[tr],
            U[va],
            X[va],
            w[va],
            hidden=cfg.decoder_hidden_dims,
            activation=cfg.decoder_activation,
            residual_scale=cfg.decoder_residual_scale,
            output_normalize=cfg.decoder_output_normalization,
            lr=cfg.learning_rate,
            epochs=cfg.epochs if not cfg.smoke else min(cfg.epochs, 25),
            patience=cfg.patience,
            batch_size=cfg.batch_size,
            device=device,
            max_train=cfg.max_decoder_train_samples,
            seed=cfg.seed + c,
        )
        save_decoder(out / f"decoder_chart{c}.pt", model, metrics)
        # jacobian on a few val anchors
        mask = w[va] > 1e-4
        idxs = np.where(mask)[0]
        full_rank = 0
        conds = []
        if len(idxs):
            rng = np.random.default_rng(cfg.seed + c)
            take = idxs[rng.choice(len(idxs), size=min(8, len(idxs)), replace=False)]
            for ii in take:
                js = jacobian_stats(model, U[va][ii], device)
                full_rank += int(js["full_rank"])
                conds.append(js["condition"])
        jac = {
            "chart": c,
            "n_anchors": int(min(8, len(idxs))),
            "frac_full_rank": float(full_rank / max(min(8, len(idxs)), 1)),
            "median_condition": float(np.median(conds)) if conds else float("nan"),
        }
        jac_rows.append(jac)
        chart_metrics.append({"chart": c, **metrics, **jac})
        # free GPU
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    summary = {
        "device": device,
        "n_charts": int(W.shape[1]),
        "latent_dim": d,
        "mean_val_mse_decoder": float(np.nanmean([m["val_mse_decoder"] for m in chart_metrics])),
        "mean_val_mse_pca": float(np.nanmean([m["val_mse_pca"] for m in chart_metrics])),
        "mean_improvement_vs_pca": float(np.nanmean([m["improvement_vs_pca"] for m in chart_metrics])),
        "mean_frac_full_rank": float(np.nanmean([m["frac_full_rank"] for m in chart_metrics])),
        "median_condition": float(np.nanmedian([m["median_condition"] for m in chart_metrics])),
        "charts": chart_metrics,
        "runtime": _stage_timer("train", t0),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_priors(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root) / "priors"
    marker = out / "prior_selection.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    tr, va = _train_val_split(data["train_local"], seed=cfg.seed)
    W = sparse.load_npz(cfg.resolved_out(root) / "charts" / "memberships_csr.npz")
    z = np.load(cfg.resolved_out(root) / "coordinates" / "coords.npz")
    per_chart = []
    for c in range(W.shape[1]):
        U = z[f"U_chart{c}"]
        w = z[f"w_chart{c}"].astype(np.float64)
        table = select_prior(
            U[tr],
            w[tr],
            U[va],
            w[va],
            ks=cfg.mixture_components,
            use_cf=True,
            n_freq=cfg.cf_frequency_count,
            scales=cfg.cf_frequency_scales,
            seed=cfg.seed + 17 * c,
        )
        per_chart.append({"chart": c, **table})
    # aggregate comparison
    def _name_stats(key_substr: str, metric: str):
        vals = []
        for ch in per_chart:
            for cand in ch["candidates"]:
                if key_substr in cand["name"]:
                    vals.append(cand[metric])
        return float(np.nanmean(vals)) if vals else float("nan")

    agg = {
        "mean_val_ll_standard_gaussian": _name_stats("standard_gaussian", "val_ll"),
        "mean_val_ll_mle_gmm": _name_stats("mle_gmm", "val_ll"),
        "mean_val_ll_cf_gmm": _name_stats("cf_gmm", "val_ll")
        if any("cf" in c["name"] for ch in per_chart for c in ch["candidates"])
        else _name_stats("cf", "val_ll"),
        "mean_val_cf_mle": _name_stats("mle_gmm", "val_cf"),
        "mean_val_cf_cfmatched": float(
            np.nanmean(
                [
                    c["val_cf"]
                    for ch in per_chart
                    for c in ch["candidates"]
                    if "cf" in c["name"]
                ]
            )
        ),
        "charts": per_chart,
        "runtime": _stage_timer("priors", t0),
    }
    # chosen families
    chosen_names = [ch["chosen"]["name"] for ch in per_chart]
    agg["chosen_names"] = chosen_names
    save_priors(out, agg)
    return agg


def stage_overlaps(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root) / "overlaps"
    marker = out / "overlaps.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    W = sparse.load_npz(cfg.resolved_out(root) / "charts" / "memberships_csr.npz")
    z = np.load(cfg.resolved_out(root) / "coordinates" / "coords.npz")
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    train_dir = cfg.resolved_out(root) / "train"
    coords, bases, recon = {}, {}, {}
    for c in range(W.shape[1]):
        pca_z = np.load(cfg.resolved_out(root) / "coordinates" / f"pca_chart{c}.npz")
        pca = {
            "mu": pca_z["mu"],
            "basis": pca_z["basis"],
            "coord_std": pca_z["coord_std"],
        }
        U = z[f"U_chart{c}"]
        coords[c] = U
        bases[c] = pca["basis"]
        try:
            ckpt = torch.load(train_dir / f"decoder_chart{c}.pt", map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(train_dir / f"decoder_chart{c}.pt", map_location=device)
        model = ResidualDecoder(
            d=U.shape[1],
            ambient=X.shape[1],
            mu=pca["mu"],
            basis=pca["basis"] * pca["coord_std"],
            hidden=cfg.decoder_hidden_dims,
            activation=cfg.decoder_activation,
            residual_scale=cfg.decoder_residual_scale,
            output_normalize=cfg.decoder_output_normalization,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        recon[c] = decode_np(model, U, device)
        del model
    idx, ww = _W_to_idx_w(W, cfg.charts_per_sample)
    result = evaluate_overlaps(
        idx, ww, coords, bases, recon, min_overlap_mass=5.0 if not cfg.smoke else 3.0, max_pairs=80
    )
    result["runtime"] = _stage_timer("overlaps", t0)
    valid_frac = float(np.mean([p["valid"] for p in result["pairs"]])) if result["pairs"] else float("nan")
    result["valid_frac"] = valid_frac
    save_overlaps(out, result)
    return result


def stage_nerve(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root) / "nerve"
    marker = out / "nerve.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
    W = sparse.load_npz(cfg.resolved_out(root) / "charts" / "memberships_csr.npz")
    idx, ww = _W_to_idx_w(W, cfg.charts_per_sample)
    result = run_nerve_analysis(
        idx, ww, W.shape[1], maxdim=cfg.nerve_maxdim, seed=cfg.seed
    )
    result["runtime"] = _stage_timer("nerve", t0)
    save_nerve(out, result)
    return result


def stage_patch_topology(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root) / "patch_topology"
    marker = out / "patch_topology.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
    z = np.load(cfg.resolved_out(root) / "coordinates" / "coords.npz")
    W = sparse.load_npz(cfg.resolved_out(root) / "charts" / "memberships_csr.npz")
    coords, weights = {}, {}
    for c in range(W.shape[1]):
        coords[c] = z[f"U_chart{c}"]
        weights[c] = z[f"w_chart{c}"].astype(np.float64)
    result = patchwise_topology(
        coords,
        weights,
        max_points=cfg.patch_max_points,
        seed=cfg.seed,
    )
    result["runtime"] = _stage_timer("patch-topology", t0)
    save_patch_topology(out, result)
    return result


def stage_boundary(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root) / "boundary"
    marker = out / "boundary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    W = sparse.load_npz(cfg.resolved_out(root) / "charts" / "memberships_csr.npz")
    pcas = []
    for c in range(W.shape[1]):
        z = np.load(cfg.resolved_out(root) / "coordinates" / f"pca_chart{c}.npz")
        pcas.append({"basis": z["basis"], "mu": z["mu"]})
    Wd = _dense_weights(W)
    result = chart_boundary_diagnostics(
        X,
        pcas,
        Wd,
        max_points_per_chart=100 if cfg.smoke else 200,
        seed=cfg.seed,
    )
    result["runtime"] = _stage_timer("boundary", t0)
    save_boundary(out, result)
    return result


def stage_curvature(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root) / "curvature"
    marker = out / "curvature.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    t0 = time.time()
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    unit = run_curvature_unit_tests(device="cpu")
    data = load_prepare(cfg.resolved_out(root) / "prepare")
    X = data["X"]
    W = sparse.load_npz(cfg.resolved_out(root) / "charts" / "memberships_csr.npz")
    z = np.load(cfg.resolved_out(root) / "coordinates" / "coords.npz")
    priors = json.loads((cfg.resolved_out(root) / "priors" / "prior_selection.json").read_text())
    per_chart = []
    for c in range(W.shape[1]):
        pca_z = np.load(cfg.resolved_out(root) / "coordinates" / f"pca_chart{c}.npz")
        pca = {
            "mu": pca_z["mu"],
            "basis": pca_z["basis"],
            "coord_std": pca_z["coord_std"],
        }
        U = z[f"U_chart{c}"]
        w = z[f"w_chart{c}"].astype(np.float64)
        path_c = cfg.resolved_out(root) / "train" / f"decoder_chart{c}.pt"
        try:
            ckpt = torch.load(path_c, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path_c, map_location=device)
        model = ResidualDecoder(
            d=U.shape[1],
            ambient=X.shape[1],
            mu=pca["mu"],
            basis=pca["basis"] * pca["coord_std"],
            hidden=cfg.decoder_hidden_dims,
            activation=cfg.decoder_activation,
            residual_scale=cfg.decoder_residual_scale,
            output_normalize=cfg.decoder_output_normalization,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        # prior logp
        chosen = priors["charts"][c]["chosen"]["model"]
        gmm = DiagGMM(
            weights=np.asarray(chosen["weights"]),
            means=np.asarray(chosen["means"]),
            variances=np.asarray(chosen["variances"]),
            family=chosen["family"],
        )
        lp = gmm.log_prob(U)
        curv = evaluate_chart_curvature(
            model,
            U,
            w,
            n_anchors=cfg.curvature_anchors,
            device=device,
            seed=cfg.seed + c,
            prior_logp=lp,
        )
        curv["chart"] = c
        per_chart.append(curv)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    # Overlap curvature agreement on shared high-membership anchors.
    from .curvature import mean_curvature_vector

    agreements = []
    ov = json.loads((cfg.resolved_out(root) / "overlaps" / "overlaps.json").read_text())
    Wd = _dense_weights(W)
    for pair in ov.get("pairs", [])[:12]:
        if not pair.get("valid"):
            continue
        a, b = int(pair["chart_a"]), int(pair["chart_b"])
        mass = np.minimum(Wd[:, a], Wd[:, b])
        cand = np.where(mass > 1e-3)[0]
        if len(cand) < 2:
            continue
        rng = np.random.default_rng(cfg.seed + 1000 + a + b)
        take = cand[rng.choice(len(cand), size=min(4, len(cand)), replace=False)]
        anchors_a, anchors_b = [], []
        models = {}
        for c_id in (a, b):
            pca_z = np.load(cfg.resolved_out(root) / "coordinates" / f"pca_chart{c_id}.npz")
            U = z[f"U_chart{c_id}"]
            path_c = cfg.resolved_out(root) / "train" / f"decoder_chart{c_id}.pt"
            try:
                ckpt = torch.load(path_c, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(path_c, map_location=device)
            model = ResidualDecoder(
                d=U.shape[1],
                ambient=X.shape[1],
                mu=pca_z["mu"],
                basis=pca_z["basis"] * pca_z["coord_std"],
                hidden=cfg.decoder_hidden_dims,
                activation=cfg.decoder_activation,
                residual_scale=cfg.decoder_residual_scale,
                output_normalize=cfg.decoder_output_normalization,
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            models[c_id] = (model, U)
        for i in take:
            try:
                oa = mean_curvature_vector(models[a][0], models[a][1][i], device)
                ob = mean_curvature_vector(models[b][0], models[b][1][i], device)
            except Exception:  # noqa: BLE001
                continue
            if oa["valid_geometry"] and ob["valid_geometry"]:
                anchors_a.append({"index": int(i), "H": oa["H"].tolist(), "H_norm": oa["H_norm"]})
                anchors_b.append({"index": int(i), "H": ob["H"].tolist(), "H_norm": ob["H_norm"]})
        del models
        if anchors_a and anchors_b:
            agreements.append(
                {
                    "pair": [a, b],
                    **overlap_curvature_agreement({"anchors": anchors_a}, {"anchors": anchors_b}),
                }
            )
    payload = {
        "unit_tests": unit,
        "charts": per_chart,
        "overlap_agreements": agreements,
        "mean_agreement_cosine": float(
            np.nanmean([a["cosine_mean"] for a in agreements if a.get("n_common", 0) > 0])
        )
        if agreements
        else float("nan"),
        "runtime": _stage_timer("curvature", t0),
    }
    if agreements and np.isfinite(payload["mean_agreement_cosine"]):
        payload["label"] = (
            "curvature_parameterization_stable"
            if payload["mean_agreement_cosine"] > 0.5
            else "curvature_not_parameterization_stable"
        )
    else:
        payload["label"] = "curvature_agreement_unavailable"
    save_curvature(out, payload)
    return payload


def stage_analyze(root: Path, cfg: AtlasConfig) -> dict:
    out = cfg.resolved_out(root)
    t0 = time.time()
    prep = json.loads((out / "prepare" / "population_summary.json").read_text())
    charts = json.loads((out / "charts" / "charts_meta.json").read_text())
    coords = json.loads((out / "coordinates" / "coordinates_meta.json").read_text())
    train = json.loads((out / "train" / "train_summary.json").read_text())
    priors = json.loads((out / "priors" / "prior_selection.json").read_text())
    overlaps = json.loads((out / "overlaps" / "overlaps.json").read_text())
    nerve = json.loads((out / "nerve" / "nerve.json").read_text())
    patch = json.loads((out / "patch_topology" / "patch_topology.json").read_text())
    boundary = json.loads((out / "boundary" / "boundary.json").read_text())
    curv = json.loads((out / "curvature" / "curvature.json").read_text())
    synth = json.loads((out / "synthetic" / "synthetic_validation.json").read_text())

    improve = train["mean_improvement_vs_pca"]
    full_rank = train["mean_frac_full_rank"]
    ov_valid = overlaps.get("valid_frac", float("nan"))
    ll_std = priors.get("mean_val_ll_standard_gaussian", float("nan"))
    ll_mle = priors.get("mean_val_ll_mle_gmm", float("nan"))
    ll_cf = priors.get("mean_val_ll_cf_gmm", float("nan"))
    cf_mle = priors.get("mean_val_cf_mle", float("nan"))
    cf_cfm = priors.get("mean_val_cf_cfmatched", float("nan"))

    decisions = {}
    if improve > 1e-4:
        decisions["reconstruction"] = "decoder_beats_local_pca"
    else:
        decisions["reconstruction"] = "atlas_failed_reconstruction"
    decisions["jacobian"] = (
        "atlas_rank_ok" if full_rank >= 0.8 and train["median_condition"] < 1e3 else "atlas_rank_deficient"
    )
    decisions["overlap"] = (
        "atlas_overlap_consistent" if ov_valid >= 0.6 else "atlas_overlap_inconsistent"
    )
    if np.isfinite(ll_mle) and np.isfinite(ll_std) and ll_mle > ll_std + 0.05:
        decisions["prior"] = "learned_prior_improves_coverage"
    else:
        decisions["prior"] = "learned_prior_not_needed"
    decisions["cf_matching"] = (
        "cf_improves_over_mle"
        if np.isfinite(cf_cfm) and np.isfinite(cf_mle) and cf_cfm < cf_mle * 0.95
        else "cf_no_material_gain"
    )
    decisions["nerve"] = "exploratory_nerve_cycles" if nerve.get("persistence", {}).get("betti_counts", {}).get("1", 0) > 0 else "exploratory_nerve_contractible_proxy"
    decisions["patch"] = patch.get("label", "patch_topology_not_detected")
    decisions["boundary"] = boundary.get("label", "boundary_density_confounded")
    decisions["curvature"] = curv.get("label", "curvature_agreement_unavailable")

    ready = (
        decisions["reconstruction"] == "decoder_beats_local_pca"
        and decisions["jacobian"] == "atlas_rank_ok"
        and decisions["overlap"] == "atlas_overlap_consistent"
        and synth.get("all_curvature_tests_pass", False)
    )
    decisions["scaleup"] = "atlas_ready_for_scaleup" if ready else "atlas_not_ready_for_scaleup"

    interpretation = (
        "Dense Physics activations admit a smoke-scale overlapping atlas with local PCA charts "
        "and smooth residual Softplus decoders. Results are conservative: the overlap nerve and "
        "patch PH are exploratory and do not establish global topology. Priors are fit after "
        "geometry (decoupled). Curvature is only interpreted at valid anchors and across overlaps."
    )
    next_cmd = (
        "cd ~/platonic-universe && source .venv/bin/activate && "
        "PYTHONPATH=experiments python -m geometry.physics_activation_atlas.run_physics_activation_atlas "
        "--stage all --output-dir outputs/geometry/physics_activation_atlas "
        "--n-charts 48 --charts-per-sample 3 --candidate-dims 8 16 32 "
        "--decoder-hidden-dims 256 256 --epochs 80 --curvature-anchors 32 "
        "--patch-max-points 800 --device cuda --seed 0"
    )

    # failure rates
    n_curv_excl = sum(c.get("n_excluded", 0) for c in curv.get("charts", []))
    n_curv_valid = sum(c.get("n_valid", 0) for c in curv.get("charts", []))
    analysis = {
        "decisions": decisions,
        "data": prep,
        "charts": {k: charts[k] for k in charts if k != "pi"},
        "dimension": coords,
        "synthetic": {
            "label": synth.get("label"),
            "all_curvature_tests_pass": synth.get("all_curvature_tests_pass"),
            "manifolds": {k: {"qualitative_ok": v.get("qualitative_ok"), "overlap_valid_frac": v.get("overlap_valid_frac")} for k, v in synth.get("manifolds", {}).items()},
        },
        "decoders": {
            "mean_val_mse_decoder": train["mean_val_mse_decoder"],
            "mean_val_mse_pca": train["mean_val_mse_pca"],
            "mean_improvement_vs_pca": train["mean_improvement_vs_pca"],
        },
        "jacobians": {
            "mean_frac_full_rank": train["mean_frac_full_rank"],
            "median_condition": train["median_condition"],
        },
        "priors": {
            "mean_val_ll_standard_gaussian": ll_std,
            "mean_val_ll_mle_gmm": ll_mle,
            "mean_val_ll_cf_gmm": ll_cf,
            "mean_val_cf_mle": cf_mle,
            "mean_val_cf_cfmatched": cf_cfm,
            "chosen_names": priors.get("chosen_names"),
        },
        "overlaps": {
            "n_pairs": overlaps.get("n_pairs"),
            "valid_frac": ov_valid,
            "mean_recon_disagreement": float(
                np.nanmean([p["recon_disagreement"] for p in overlaps.get("pairs", [])])
            )
            if overlaps.get("pairs")
            else float("nan"),
            "mean_tangent_disagreement": float(
                np.nanmean([p["tangent_disagreement"] for p in overlaps.get("pairs", [])])
            )
            if overlaps.get("pairs")
            else float("nan"),
        },
        "nerve": {
            "label": nerve.get("label"),
            "betti_counts": nerve.get("persistence", {}).get("betti_counts"),
            "shuffled_controls": nerve.get("shuffled_controls"),
        },
        "patch_topology": {
            "label": patch.get("label"),
            "n_patches": patch.get("n_patches"),
            "n_excess_H1": patch.get("n_excess_H1"),
        },
        "boundary": {
            "label": boundary.get("label"),
            "mean_excess_vs_control": boundary.get("mean_excess_vs_control"),
            "mean_corr_with_dtm": boundary.get("mean_corr_with_dtm"),
        },
        "curvature": {
            "unit_tests_pass": curv.get("unit_tests", {}).get("all_pass"),
            "label": curv.get("label"),
            "mean_agreement_cosine": curv.get("mean_agreement_cosine"),
            "n_valid_anchors": n_curv_valid,
            "n_excluded_anchors": n_curv_excl,
        },
        "runtime": {
            "analyze_seconds": time.time() - t0,
            "peak_rss_mb": _rss_mb(),
            "train": train.get("runtime"),
            "prepare_peak_note": "see per-stage runtime.json",
        },
        "interpretation": interpretation,
        "next_command": next_cmd,
        "failure_exclusion": {
            "curvature_valid": n_curv_valid,
            "curvature_excluded": n_curv_excl,
            "overlap_invalid_frac": float(1.0 - ov_valid) if np.isfinite(ov_valid) else float("nan"),
        },
    }
    try:
        analysis["plots"] = write_smoke_plots(out)
    except Exception as e:  # noqa: BLE001
        analysis["plots"] = {"error": str(e)}
    write_report(out / "analyze", analysis)
    return analysis


STAGES = [
    "prepare",
    "synthetic",
    "charts",
    "coordinates",
    "train",
    "priors",
    "overlaps",
    "nerve",
    "patch-topology",
    "boundary",
    "curvature",
    "analyze",
]


def run_pipeline(cfg: AtlasConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    save_config(out, cfg)
    stage = cfg.stage
    order = STAGES if stage == "all" else [stage]
    results: dict[str, Any] = {}
    dispatch = {
        "prepare": stage_prepare,
        "synthetic": stage_synthetic,
        "charts": stage_charts,
        "coordinates": stage_coordinates,
        "train": stage_train,
        "priors": stage_priors,
        "overlaps": stage_overlaps,
        "nerve": stage_nerve,
        "patch-topology": stage_patch_topology,
        "boundary": stage_boundary,
        "curvature": stage_curvature,
        "analyze": stage_analyze,
    }
    for s in order:
        print(f"[atlas] stage={s} rss_mb={_rss_mb():.1f}", flush=True)
        results[s] = dispatch[s](root, cfg)
        if _rss_mb() > 32000:
            raise RuntimeError(
                f"Hard stop: peak RSS {_rss_mb():.1f} MB exceeded 32 GB at stage {s}"
            )
    return results
