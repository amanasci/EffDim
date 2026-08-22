"""Multi-model OOF global Physics probe geography + graph/quadratic dim + gated curvature.

No permanent holdout. Shared 5-fold CV across models. Local scores use OOF predictions only.
"""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, StratifiedKFold

from .curvature_probe_alignment import B0_flat_for_svd, traceless_B0
from .curvature_probe_screen import partial_spearman, spearman_dict
from .global_probe_curvature_alignment import (
    CANONICAL_PROBE_LOCAL,
    CANONICAL_PROBE_TRAIN,
    fit_global_probe,
    local_r2_fixed_predictions,
    ridge_multi_intercept_torch,
    weighted_r2,
)
from .paths import platonic_root, resolve_path
from .quadratic import quadratic_features
from .sphere_normal_quadratic import sphere_project_basis

EPS = 1e-12
MODELS = {
    "vit_base": ("data_hf/physics/vit_base_test.parquet", "vit_base_galaxies"),
    "dinov3": ("data_hf/physics/dinov3_vitb16_test.parquet", "dinov3_vitb16_galaxies"),
    "clip_base": ("data_hf/physics/clip_base_test.parquet", "clip_base_galaxies"),
    "convnext_base": ("data_hf/physics/convnext_base_test.parquet", "convnext_base_galaxies"),
    "vit_large": ("data_hf/physics/vit_large_test.parquet", "vit_large_galaxies"),
}
TARGETS = ["mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass"]
SCALES = (256, 512, 1024, 2048)


def _load_embedding_col(path: Path, col: str) -> np.ndarray:
    """Load list/fixed-size embedding column via canonical SAE helper."""
    from topology.physics_activation_density_ph.paths import load_col

    return load_col(path, col, l2=False).astype(np.float32)
ISOMAP_DIMS_REL = (
    "outputs/sae_shared_basis/pipeline_isomap_sae_shared_mknn_physics_holdout20/isomap_dims.json"
)
ISOMAP_KEY = {
    "vit_base": "vit_base_test::vit_base_galaxies",
    "dinov3": "dinov3_vitb16_test::dinov3_vitb16_galaxies",
    "clip_base": "clip_base_test::clip_base_galaxies",
    "convnext_base": "convnext_base_test::convnext_base_galaxies",
    "vit_large": "vit_large_test::vit_large_galaxies",
}
GRAPH_PRIOR_SOURCE = (
    "experiments/SAE-shared-basis/pipeline_isomap_sae_shared_mknn.py::estimate_isomap_dim "
    "(d_residual_elbow; kNN-graph geodesics → landmark MDS residual elbow); "
    "local prior: PCA energy_rank_90 + participation_ratio on model kNN neighbourhoods "
    "(Tyagi/effdim-style; not SAE decoder directions)."
)


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, 1e-8)).astype(np.float32)


def energy_rank(evals: np.ndarray, frac: float = 0.9) -> int:
    ev = np.maximum(np.asarray(evals, float), 0.0)
    if ev.sum() <= 0:
        return 1
    c = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(c, frac) + 1)


def participation_ratio(evals: np.ndarray) -> float:
    ev = np.maximum(np.asarray(evals, float), 0.0)
    if ev.sum() <= 0:
        return 1.0
    return float((ev.sum() ** 2) / max(np.sum(ev**2), EPS))


@dataclass
class MultiModelConfig:
    output_dir: str = "outputs/geometry/physics_multimodel_graph_prior_quadratic"
    selection_path: str = "outputs/sae_shared_basis/bsf_block_vae_fisher_physics/selection.npz"
    labels_path: str = "data_hf/physics/vit_base_test_labels.npz"
    isomap_dims_path: str = ISOMAP_DIMS_REL
    models: list[str] = field(default_factory=lambda: list(MODELS.keys()))
    targets: list[str] = field(default_factory=lambda: list(TARGETS))
    scales: list[int] = field(default_factory=lambda: list(SCALES))
    n_folds: int = 5
    n_anchors: int = 512
    screen_anchors: int = 96
    probe_alpha: float = 100.0
    select_alpha: bool = False  # nested alpha optional; default fixed 100 for comparability
    seed: int = 0
    force: bool = False
    device: str = "cuda"
    stage: str = "all"
    sketch_dim: int = 128
    n_sketches: int = 3
    neff_df_min: float = 5.0
    primary_target: str = "mag_r_desi"
    graph_screen_models: list[str] = field(
        default_factory=lambda: ["vit_base", "dinov3", "clip_base"]
    )

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


# -------------------- Stage: prepare --------------------


def stage_prepare(root: Path, cfg: MultiModelConfig) -> dict[str, Any]:
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "prepare" / "ready.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())

    sel = np.load(resolve_path(root, cfg.selection_path))
    selected = np.asarray(sel["selected"], dtype=np.int64)
    assert len(selected) == 16384, f"expected 16384 selected, got {len(selected)}"

    lab = np.load(resolve_path(root, cfg.labels_path))
    label_schema = {k: {"shape": list(lab[k].shape), "finite_frac": float(np.isfinite(lab[k]).mean())} for k in lab.files}
    y_full = {t: np.asarray(lab[t], dtype=np.float64) for t in cfg.targets if t in lab.files}
    missing = [t for t in cfg.targets if t not in lab.files]
    if missing:
        raise KeyError(f"targets missing from labels: {missing}")

    # join by sample_id (= row index in parquet / labels)
    sample_ids = selected.copy()
    y = {t: y_full[t][sample_ids] for t in cfg.targets}

    # stratified folds on primary target quantiles where possible
    primary = y[cfg.primary_target]
    finite = np.isfinite(primary)
    fold_ids = np.full(len(sample_ids), -1, dtype=np.int64)
    rng = np.random.default_rng(cfg.seed)
    if finite.sum() > 50:
        bins = pd.qcut(primary[finite], 10, labels=False, duplicates="drop")
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        idx_f = np.where(finite)[0]
        for fi, (_, te) in enumerate(skf.split(idx_f, bins)):
            fold_ids[idx_f[te]] = fi
    # assign remaining (nan primary) randomly
    rest = np.where(fold_ids < 0)[0]
    if len(rest):
        fold_ids[rest] = rng.integers(0, cfg.n_folds, size=len(rest))

    # anchors: deterministic, ignore probe/curvature — stratify by fold only
    order = np.argsort(sample_ids, kind="mergesort")
    anchors = []
    per = cfg.n_anchors // cfg.n_folds
    for fi in range(cfg.n_folds):
        cand = order[fold_ids[order] == fi]
        anchors.extend(cand[:per].tolist())
    if len(anchors) < cfg.n_anchors:
        extra = [i for i in order if i not in set(anchors)]
        anchors.extend(extra[: cfg.n_anchors - len(anchors)])
    anchors_local = np.asarray(sorted(anchors[: cfg.n_anchors]), dtype=np.int64)
    anchors_sid = sample_ids[anchors_local]

    schema = {"models": {}, "labels": label_schema, "n_selected": int(len(sample_ids))}
    prep_dir = out / "prepare"
    prep_dir.mkdir(exist_ok=True)
    model_dir = prep_dir / "models"
    model_dir.mkdir(exist_ok=True)

    for m in cfg.models:
        pq_rel, col = MODELS[m]
        X_all = _load_embedding_col(resolve_path(root, pq_rel), col)
        if X_all.shape[0] != lab[cfg.primary_target].shape[0]:
            raise RuntimeError(f"{m}: row count {X_all.shape[0]} != labels {lab[cfg.primary_target].shape[0]}")
        X = l2_normalize(X_all[sample_ids])
        norms = np.linalg.norm(X, axis=1)
        # effective rank on subsample
        rs = rng.choice(len(X), size=min(4096, len(X)), replace=False)
        _, s, _ = np.linalg.svd(X[rs] - X[rs].mean(0), full_matrices=False)
        ev = (s**2) / max(len(rs), 1)
        schema["models"][m] = {
            "parquet": pq_rel,
            "column": col,
            "ambient_dim": int(X.shape[1]),
            "n": int(X.shape[0]),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "missing_rows": 0,
            "effective_rank_pr": participation_ratio(ev),
            "energy_rank_90": energy_rank(ev, 0.9),
        }
        np.savez_compressed(model_dir / f"{m}.npz", X=X.astype(np.float32))
        print(f"[prep] {m} D={X.shape[1]} PR={schema['models'][m]['effective_rank_pr']:.1f}", flush=True)

    folds_df = pd.DataFrame(
        {
            "local_index": np.arange(len(sample_ids)),
            "sample_id": sample_ids,
            "fold": fold_ids,
            "is_anchor": np.isin(np.arange(len(sample_ids)), anchors_local),
        }
    )
    for t in cfg.targets:
        folds_df[f"y_{t}"] = y[t]
        folds_df[f"finite_{t}"] = np.isfinite(y[t])
    folds_df.to_parquet(out / "sample_folds.parquet", index=False)
    np.savez_compressed(
        prep_dir / "anchors.npz",
        anchors_local=anchors_local,
        anchors_sample_id=anchors_sid,
    )
    (out / "resolved_model_schema.json").write_text(json.dumps(schema, indent=2))
    meta = {
        "n_selected": len(sample_ids),
        "n_folds": cfg.n_folds,
        "n_anchors": len(anchors_local),
        "targets": cfg.targets,
        "models": cfg.models,
        "scales": cfg.scales,
        "graph_prior_source": GRAPH_PRIOR_SOURCE,
        "canonical_train_fn": CANONICAL_PROBE_TRAIN,
        "canonical_local_fn": CANONICAL_PROBE_LOCAL,
    }
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(json.dumps(meta, indent=2))
    return meta


def load_model_X(out: Path, model: str) -> np.ndarray:
    return np.load(out / "prepare" / "models" / f"{model}.npz")["X"].astype(np.float32)


# -------------------- Stage: global probes (OOF) --------------------


def stage_global_probes(root: Path, cfg: MultiModelConfig) -> None:
    out = cfg.resolved_out(root)
    marker = out / "global_probes" / "ready.json"
    if _done(marker, cfg.force):
        return
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    folds = pd.read_parquet(out / "sample_folds.parquet")
    fold_ids = folds.fold.to_numpy(int)
    Y = np.column_stack([folds[f"y_{t}"].to_numpy(float) for t in cfg.targets])
    weight_rows = []
    cv_rows = []
    pred_dir = out / "global_probes" / "oof_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for m in cfg.models:
        X = load_model_X(out, m)
        oof = np.full_like(Y, np.nan)
        fold_W = {t: [] for t in cfg.targets}
        for fi in range(cfg.n_folds):
            te = fold_ids == fi
            tr = ~te
            # mask groups
            groups: dict[bytes, dict] = {}
            for j, t in enumerate(cfg.targets):
                mtr = np.isfinite(Y[tr, j])
                if mtr.sum() < 32:
                    continue
                key = mtr.tobytes()
                groups.setdefault(key, {"mask": mtr, "cols": []})
                groups[key]["cols"].append(j)
            for g in groups.values():
                mtr = g["mask"]
                cols = g["cols"]
                Xt = torch.tensor(X[tr][mtr], device=device, dtype=torch.float32)
                Yt = torch.tensor(Y[tr][mtr][:, cols], device=device, dtype=torch.float32)
                W, b, ok = ridge_multi_intercept_torch(Xt, Yt, alpha=cfg.probe_alpha)
                if not ok:
                    Xt64, Yt64 = Xt.double(), Yt.double()
                    W, b, ok = ridge_multi_intercept_torch(Xt64, Yt64, alpha=cfg.probe_alpha)
                    W, b = W.float(), b.float()
                Wc = W.detach().cpu().numpy()
                bc = b.detach().cpu().numpy()
                Xte = X[te]
                for li, j in enumerate(cols):
                    oof[te, j] = Xte @ Wc[:, li] + bc[li]
                    fold_W[cfg.targets[j]].append(Wc[:, li])
                    weight_rows.append(
                        {
                            "model": m,
                            "target": cfg.targets[j],
                            "fold": fi,
                            "coef_norm": float(np.linalg.norm(Wc[:, li])),
                            "intercept": float(bc[li]),
                            "n_train": int(mtr.sum()),
                            "alpha": cfg.probe_alpha,
                        }
                    )
            print(f"[probes] {m} fold {fi} done", flush=True)

        # full-data consensus weights
        for j, t in enumerate(cfg.targets):
            mtr = np.isfinite(Y[:, j])
            coef, intercept = fit_global_probe(X[mtr], Y[mtr, j], cfg.probe_alpha)
            # fold stability
            Ws = fold_W[t]
            cos_pairs = []
            for a in range(len(Ws)):
                for b in range(a + 1, len(Ws)):
                    na, nb = np.linalg.norm(Ws[a]), np.linalg.norm(Ws[b])
                    if na > EPS and nb > EPS:
                        cos_pairs.append(float(np.dot(Ws[a], Ws[b]) / (na * nb)))
            cos_full = [
                float(np.dot(w, coef) / (np.linalg.norm(w) * np.linalg.norm(coef) + EPS))
                for w in Ws
                if np.linalg.norm(w) > EPS
            ]
            stable = bool(len(cos_pairs) and np.nanmedian(cos_pairs) > 0.9)
            # OOF metrics
            yt, yp = Y[:, j], oof[:, j]
            mm = np.isfinite(yt) & np.isfinite(yp)
            r2 = float(1 - np.sum((yt[mm] - yp[mm]) ** 2) / max(np.sum((yt[mm] - yt[mm].mean()) ** 2), EPS))
            cv_rows.append(
                {
                    "model": m,
                    "target": t,
                    "oof_r2": r2,
                    "oof_pearson": float(np.corrcoef(yt[mm], yp[mm])[0, 1]) if mm.sum() > 5 else float("nan"),
                    "n_eval": int(mm.sum()),
                    "fold_cosine_median": float(np.nanmedian(cos_pairs)) if cos_pairs else float("nan"),
                    "fold_cosine_min": float(np.nanmin(cos_pairs)) if cos_pairs else float("nan"),
                    "cosine_to_full_median": float(np.nanmedian(cos_full)) if cos_full else float("nan"),
                    "direction_stable": stable,
                    "alpha": cfg.probe_alpha,
                }
            )
            weight_rows.append(
                {
                    "model": m,
                    "target": t,
                    "fold": -1,
                    "coef_norm": float(np.linalg.norm(coef)),
                    "intercept": float(intercept),
                    "n_train": int(mtr.sum()),
                    "alpha": cfg.probe_alpha,
                    "direction_stable": stable,
                }
            )
            np.savez_compressed(
                pred_dir / f"{m}_{t}.npz",
                oof=oof[:, j],
                w_full=coef,
                b_full=np.array([intercept]),
                fold_cos_median=np.array([np.nanmedian(cos_pairs) if cos_pairs else np.nan]),
            )
        # save OOF matrix
        np.savez_compressed(pred_dir / f"{m}_all_targets.npz", oof=oof, targets=np.array(cfg.targets))

    pd.DataFrame(weight_rows).to_parquet(out / "global_probe_weights.parquet", index=False)
    pd.DataFrame(cv_rows).to_parquet(out / "global_probe_cv_metrics.parquet", index=False)
    marker.write_text(json.dumps({"models": cfg.models, "alpha": cfg.probe_alpha}, indent=2))
    print(f"[probes] done rss={_rss():.0f}", flush=True)


# -------------------- Stage: neighbourhoods --------------------


def knn_torch_ip(X: np.ndarray, queries: np.ndarray, k: int, device: torch.device, batch: int = 256) -> np.ndarray:
    """Exact top-k by inner product (= Euclidean on unit sphere). Returns indices (nq, k)."""
    Xt = torch.tensor(X, device=device, dtype=torch.float32)
    # pre-normalize safeguard
    Xt = Xt / torch.clamp(Xt.norm(dim=1, keepdim=True), min=1e-8)
    outs = []
    for i0 in range(0, len(queries), batch):
        q = torch.tensor(queries[i0 : i0 + batch], device=device, dtype=torch.float32)
        q = q / torch.clamp(q.norm(dim=1, keepdim=True), min=1e-8)
        sims = q @ Xt.T
        _, idx = torch.topk(sims, k=k + 1, dim=1)  # +1 to drop self
        outs.append(idx.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def stage_neighbourhoods(root: Path, cfg: MultiModelConfig) -> None:
    out = cfg.resolved_out(root)
    marker = out / "model_neighbourhoods" / "ready.json"
    if _done(marker, cfg.force):
        return
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    folds = pd.read_parquet(out / "sample_folds.parquet")
    anchors_local = np.load(out / "prepare" / "anchors.npz")["anchors_local"]
    k_max = max(cfg.scales)
    ndir = out / "model_neighbourhoods"
    ndir.mkdir(exist_ok=True)

    # parity on vit_base small subset
    X0 = load_model_X(out, cfg.models[0])
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=32, metric="euclidean").fit(X0)
    _, idx_cpu = nn.kneighbors(X0[anchors_local[:8]])
    idx_gpu = knn_torch_ip(X0, X0[anchors_local[:8]], 31, device)
    # compare after removing self
    overlap = []
    for i in range(8):
        a = set(idx_cpu[i, 1:].tolist())
        b = set(idx_gpu[i].tolist())
        b.discard(int(anchors_local[i]))
        b = set(list(b)[:31])
        overlap.append(len(a & b) / 31)
    parity = {"knn_overlap_mean": float(np.mean(overlap))}
    (ndir / "knn_parity.json").write_text(json.dumps(parity, indent=2))
    print(f"[knn] parity overlap={parity['knn_overlap_mean']:.3f}", flush=True)

    for m in cfg.models:
        X = load_model_X(out, m)
        idx = knn_torch_ip(X, X[anchors_local], k_max, device, batch=64)
        # remove self if present; ensure k_max neighbours
        clean = np.zeros((len(anchors_local), k_max), dtype=np.int64)
        for i, a in enumerate(anchors_local):
            row = [int(j) for j in idx[i] if int(j) != int(a)]
            if len(row) < k_max:
                # rare; pad from random
                pad = [j for j in range(len(X)) if j != a and j not in row]
                row.extend(pad[: k_max - len(row)])
            clean[i] = row[:k_max]
        # distances for radius
        dists = np.linalg.norm(X[clean] - X[anchors_local][:, None, :], axis=2)
        np.savez_compressed(
            ndir / f"{m}_kmax{k_max}.npz",
            anchors_local=anchors_local,
            neigh=clean,
            dists=dists.astype(np.float32),
            sample_ids=folds.sample_id.to_numpy(),
        )
        print(f"[knn] {m} cached", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    marker.write_text(json.dumps({"k_max": k_max, "parity": parity}, indent=2))


# -------------------- Stage: probe geography --------------------


def stage_probe_geography(root: Path, cfg: MultiModelConfig) -> None:
    out = cfg.resolved_out(root)
    marker = out / "probe_geography" / "ready.json"
    if _done(marker, cfg.force):
        return
    folds = pd.read_parquet(out / "sample_folds.parquet")
    anchors_local = np.load(out / "prepare" / "anchors.npz")["anchors_local"]
    anchors_sid = np.load(out / "prepare" / "anchors.npz")["anchors_sample_id"]
    rows = []
    # vit_base neighbourhoods for shared sensitivity
    vit_pack = np.load(out / "model_neighbourhoods" / f"vit_base_kmax{max(cfg.scales)}.npz")
    vit_neigh = vit_pack["neigh"]

    for m in cfg.models:
        pack = np.load(out / "model_neighbourhoods" / f"{m}_kmax{max(cfg.scales)}.npz")
        neigh = pack["neigh"]
        dists = pack["dists"]
        oof_all = np.load(out / "global_probes" / "oof_predictions" / f"{m}_all_targets.npz")["oof"]
        Y = np.column_stack([folds[f"y_{t}"].to_numpy(float) for t in cfg.targets])
        gvar = {t: float(np.nanvar(Y[:, j])) for j, t in enumerate(cfg.targets)}
        for ai, a in enumerate(anchors_local):
            for k in cfg.scales:
                for neigh_src, N in [("model", neigh[ai, :k]), ("vit_shared", vit_neigh[ai, :k])]:
                    rho = float(dists[ai, k - 1]) if neigh_src == "model" else float("nan")
                    for j, t in enumerate(cfg.targets):
                        y = Y[N, j]
                        pred = oof_all[N, j]
                        msk = np.isfinite(y) & np.isfinite(pred)
                        if msk.sum() < 4:
                            continue
                        r2 = local_r2_fixed_predictions(y, pred)
                        mse = float(np.mean((y[msk] - pred[msk]) ** 2)) / max(gvar[t], EPS)
                        pear = float(np.corrcoef(y[msk], pred[msk])[0, 1])
                        rows.append(
                            {
                                "model": m,
                                "target": t,
                                "sample_id": int(anchors_sid[ai]),
                                "anchor_local": int(a),
                                "scale_k": int(k),
                                "neighbourhood": neigh_src,
                                "local_r2": r2,
                                "normalized_mse": mse,
                                "local_prediction_correlation": pear,
                                "local_label_variance": float(np.var(y[msk])),
                                "local_evaluation_count": int(msk.sum()),
                                "knn_radius": rho if neigh_src == "model" else float("nan"),
                                "log_knn_radius": float(np.log(max(rho, EPS))) if neigh_src == "model" else float("nan"),
                            }
                        )
        print(f"[geo] {m} scored", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "local_probe_fields.parquet", index=False)

    # cross-model correlations of local_r2 (model-specific neigh)
    xref = []
    sub = df[(df.neighbourhood == "model")]
    for t in cfg.targets:
        for k in cfg.scales:
            g = sub[(sub.target == t) & (sub.scale_k == k)]
            piv = g.pivot_table(index="sample_id", columns="model", values="local_r2")
            models = [m for m in cfg.models if m in piv.columns]
            for i, m1 in enumerate(models):
                for m2 in models[i + 1 :]:
                    a, b = piv[m1].to_numpy(float), piv[m2].to_numpy(float)
                    st = spearman_dict(a, b)
                    # weak region overlap: bottom quartile
                    m = np.isfinite(a) & np.isfinite(b)
                    if m.sum() > 20:
                        qa, qb = np.nanquantile(a[m], 0.25), np.nanquantile(b[m], 0.25)
                        wa, wb = (a <= qa) & m, (b <= qb) & m
                        jacc = float((wa & wb).sum() / max((wa | wb).sum(), 1))
                    else:
                        jacc = float("nan")
                    xref.append(
                        {
                            "target": t,
                            "scale_k": k,
                            "model_a": m1,
                            "model_b": m2,
                            "spearman_local_r2": st["rho"],
                            "weak_jaccard": jacc,
                            "n": st["n"],
                        }
                    )
    pd.DataFrame(xref).to_parquet(out / "crossmodel_probe_geography.parquet", index=False)
    (out / "probe_geography").mkdir(exist_ok=True)
    marker.write_text(json.dumps({"n_rows": len(df)}, indent=2))


# -------------------- Stage: graph prior --------------------


def _local_pca_scores(dx: np.ndarray, q_max: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues (descending) and top score coords (n, q) via Gram eigh."""
    n, D = dx.shape
    q_max = min(q_max, n - 1, D)
    dx = np.asarray(dx, dtype=np.float32)
    if n <= D:
        G = dx @ dx.T
        evals, evecs = np.linalg.eigh(G.astype(np.float64))
        evals = np.maximum(evals[::-1], 0.0)
        evecs = evecs[:, ::-1]
        s = np.sqrt(evals)
        u = (evecs[:, :q_max] * s[:q_max]).astype(np.float32)
        return (evals / max(n, 1)).astype(np.float64), u
    G = dx.T @ dx
    evals, evecs = np.linalg.eigh(G.astype(np.float64))
    evals = np.maximum(evals[::-1], 0.0)
    evecs = evecs[:, ::-1].astype(np.float32)
    u = dx @ evecs[:, :q_max]
    return (evals / max(n, 1)).astype(np.float64), u


def _graph_metrics_from_scores(ev: np.ndarray, u_all: np.ndarray, d_er: int) -> tuple[float, float, float]:
    """support turnover, boundary imbalance, participation ratio helpers."""
    d_pr = participation_ratio(ev)
    q = max(int(d_er), 2)
    u = u_all[:, : min(q, u_all.shape[1])]
    ru = np.linalg.norm(u, axis=1)
    qs = np.quantile(ru, [0.33, 0.66])
    bands = [(ru <= qs[0]), (ru > qs[0]) & (ru <= qs[1]), (ru > qs[1])]
    turns = []
    thr = np.quantile(np.abs(u), 0.1)
    for b0, b1 in zip(bands, bands[1:]):
        if b0.sum() < 3 or b1.sum() < 3:
            continue
        s0 = np.abs(u[b0]).mean(0) < thr
        s1 = np.abs(u[b1]).mean(0) < thr
        union = np.logical_or(s0, s1).sum()
        turns.append(1 - np.logical_and(s0, s1).sum() / max(union, 1))
    imb = float(abs((u[:, 0] > 0).mean() - 0.5)) if u.shape[1] >= 1 else float("nan")
    return d_pr, float(np.mean(turns)) if turns else float("nan"), imb


def stage_graph_prior(root: Path, cfg: MultiModelConfig) -> None:
    """Local PCA dimension prior on model kNN neighbourhoods (batched GPU svd_lowrank)."""
    out = cfg.resolved_out(root)
    marker = out / "graph_prior" / "ready.json"
    if _done(marker, cfg.force):
        return
    (out / "graph_prior").mkdir(exist_ok=True)
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    isomap_path = resolve_path(root, cfg.isomap_dims_path)
    isomap = json.loads(isomap_path.read_text()) if isomap_path.exists() else {}
    anchors_local = np.load(out / "prepare" / "anchors.npz")["anchors_local"]
    anchors_sid = np.load(out / "prepare" / "anchors.npz")["anchors_sample_id"]
    k_max = max(cfg.scales)
    q_max = 64
    batch = 16 if device.type == "cuda" else 4
    rows = []
    global_rows = []
    for m in cfg.models:
        t_m = time.time()
        key = ISOMAP_KEY.get(m)
        gstat = isomap.get(key, {}) if key else {}
        d_graph_global = gstat.get("d_residual_elbow", gstat.get("d_primary", float("nan")))
        has_gp = bool(
            isinstance(d_graph_global, (int, float, np.integer, np.floating))
            and np.isfinite(float(d_graph_global))
        )
        global_rows.append(
            {
                "model": m,
                "isomap_key": key,
                "d_residual_elbow": d_graph_global,
                "d_pr": gstat.get("d_pr", float("nan")),
                "d_cum90": gstat.get("d_cum90", float("nan")),
                "source": GRAPH_PRIOR_SOURCE,
                "has_graph_prior": has_gp,
            }
        )
        pack = np.load(out / "model_neighbourhoods" / f"{m}_kmax{k_max}.npz")
        X = load_model_X(out, m)
        neigh = pack["neigh"]
        Xt = torch.tensor(X, device=device, dtype=torch.float32)
        for k in cfg.scales:
            print(f"[graph] {m} k={k}", flush=True)
            for i0 in range(0, len(anchors_local), batch):
                ids = np.arange(i0, min(i0 + batch, len(anchors_local)))
                a_idx = anchors_local[ids]
                N = neigh[ids][:, :k]  # (B, k)
                xb = Xt[torch.as_tensor(a_idx, device=device)]  # (B, D)
                xn = Xt[torch.as_tensor(N, device=device)]  # (B, k, D)
                dx = xn - xb[:, None, :]
                x0u = xb / torch.clamp(xb.norm(dim=1, keepdim=True), min=1e-8)
                dx = dx - (dx * x0u[:, None, :]).sum(dim=-1, keepdim=True) * x0u[:, None, :]
                # per-anchor svd_lowrank (batched loop; still far faster than CPU eigh*4)
                for bi, ai in enumerate(ids):
                    dxi = dx[bi]
                    q = min(q_max, k - 1, dxi.shape[1])
                    try:
                        U, S, _ = torch.svd_lowrank(dxi, q=q, niter=2)
                    except RuntimeError:
                        ev, u_all = _local_pca_scores(dxi.detach().cpu().numpy(), q_max=q_max)
                    else:
                        ev = (S.detach().cpu().numpy() ** 2) / max(k, 1)
                        u_all = (U * S).detach().cpu().numpy()
                    d_er = energy_rank(ev, 0.9)
                    d_pr, turn, imb = _graph_metrics_from_scores(ev, u_all, d_er)
                    if has_gp:
                        d_graph = int(
                            np.clip(round(0.5 * float(d_graph_global) + 0.5 * d_er), 2, min(32, k // 8))
                        )
                        unc = float(abs(d_er - float(d_graph_global)))
                    else:
                        d_graph = int(np.clip(d_er, 2, min(32, k // 8)))
                        unc = float("nan")
                    rows.append(
                        {
                            "model": m,
                            "sample_id": int(anchors_sid[ai]),
                            "anchor_local": int(a_idx[bi]),
                            "scale_k": int(k),
                            "d_graph": d_graph,
                            "d_energy_rank_90": d_er,
                            "d_participation_ratio": d_pr,
                            "d_isomap_global": float(d_graph_global) if has_gp else float("nan"),
                            "graph_dimension_uncertainty": unc,
                            "graph_support_turnover": turn,
                            "graph_boundary_imbalance": imb,
                            "source": GRAPH_PRIOR_SOURCE,
                        }
                    )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[graph] {m} done in {time.time()-t_m:.1f}s", flush=True)
    pd.DataFrame(rows).to_parquet(out / "graph_dimension_prior.parquet", index=False)
    pd.DataFrame(global_rows).to_csv(out / "graph_prior" / "global_isomap_dims.csv", index=False)
    marker.write_text(json.dumps({"source": GRAPH_PRIOR_SOURCE, "n": len(rows)}, indent=2))


# -------------------- Stage: quadratic dimension screen --------------------


def jl_sketch(D: int, s: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(D, s)).astype(np.float32)
    S /= np.linalg.norm(S, axis=0, keepdims=True) + 1e-8
    S *= np.sqrt(D / s)
    return S


def oof_quad_errors(
    Xn: np.ndarray,
    d: int,
    sketch: np.ndarray | None,
    n_folds: int,
    seed: int,
) -> dict:
    """Single-d wrapper around multi-d screen (kept for tests / parity checks)."""
    out = oof_quad_errors_multi(Xn, [d], sketch, n_folds, seed)
    return out.get(d, {"E_linear": float("nan"), "E_quadratic": float("nan"), "ok": False})


def oof_quad_errors_multi(
    Xn: np.ndarray,
    dims: list[int],
    sketch: np.ndarray | None,
    n_folds: int,
    seed: int,
) -> dict[int, dict]:
    """Activation-only OOF linear/quadratic errors for many d, reusing max tangent basis."""
    dims = sorted(set(int(d) for d in dims if d >= 2))
    n = len(Xn)
    empty = {d: {"E_linear": float("nan"), "E_quadratic": float("nan"), "ok": False} for d in dims}
    if not dims or n < max(40, 5 * min(dims)):
        return empty
    d_max = max(dims)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    acc: dict[int, dict[str, list]] = {
        d: {"el": [], "eq": [], "df": [], "de": []} for d in dims
    }
    for fi, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        if len(tr) < 2 * d_max or len(te) < 4:
            continue
        x0a = Xn[tr].mean(0)
        x0a = x0a / max(np.linalg.norm(x0a), EPS)
        Xc = Xn[tr] - x0a
        Xc = Xc - np.outer(Xc @ x0a, x0a)
        try:
            # Gram eigh is cheaper when n_tr < D
            if len(tr) <= Xn.shape[1]:
                G = Xc @ Xc.T
                evals, evecs = np.linalg.eigh(G)
                order = np.argsort(evals)[::-1]
                # V from Xc.T @ U / s
                s = np.sqrt(np.maximum(evals[order[:d_max]], EPS))
                U = evecs[:, order[:d_max]]
                Vt = ((Xc.T @ U) / s).T
            else:
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                Vt = Vt[:d_max]
        except np.linalg.LinAlgError:
            continue
        Jmax = sphere_project_basis(x0a, Vt[:d_max].T)
        if Jmax.shape[1] < 2:
            continue
        Utr_max = (Xn[tr] - x0a) @ Jmax
        Ute_max = (Xn[te] - x0a) @ Jmax
        if sketch is not None:
            target_tr = Xn[tr] @ sketch
            target_te = Xn[te] @ sketch
        else:
            target_tr, target_te = Xn[tr], Xn[te]
        # build max quadratic design once; slice monomials for smaller d
        Phi_tr_max = quadratic_features(Utr_max)
        Phi_te_max = quadratic_features(Ute_max)
        for d in dims:
            if Jmax.shape[1] < max(2, d // 2) or Utr_max.shape[1] < d:
                continue
            J = Jmax[:, :d]
            Utr, Ute = Utr_max[:, :d], Ute_max[:, :d]
            Ltr = x0a + Utr @ J.T
            Lte = x0a + Ute @ J.T
            if sketch is not None:
                Ltr_s, Lte_s = Ltr @ sketch, Lte @ sketch
            else:
                Ltr_s, Lte_s = Ltr, Lte
            el = float(np.mean(np.sum((target_te - Lte_s) ** 2, axis=1)))
            # column subset of quadratic features for first d coords
            # quadratic_features packs [u_i u_j for i<=j] in order of increasing i,j over all d_max
            # Safer: recompute for d (small) — monomial count is tiny vs ambient
            Phi = quadratic_features(Utr)
            Phi_te = quadratic_features(Ute)
            resid = target_tr - Ltr_s
            lam = 1e-2
            Gphi = Phi.T @ Phi + lam * np.eye(Phi.shape[1], dtype=Phi.dtype)
            try:
                B = np.linalg.solve(Gphi, Phi.T @ resid)
                df_lam = float(np.trace(np.linalg.solve(Gphi, Phi.T @ Phi)))
            except np.linalg.LinAlgError:
                try:
                    G64 = Gphi.astype(np.float64)
                    B = np.linalg.solve(G64, (Phi.T @ resid).astype(np.float64)).astype(Phi.dtype)
                    df_lam = float(np.trace(np.linalg.solve(G64, (Phi.T @ Phi).astype(np.float64))))
                except np.linalg.LinAlgError:
                    B = np.linalg.lstsq(Gphi, Phi.T @ resid, rcond=None)[0]
                    df_lam = float(Phi.shape[1])
            pred_q = Lte_s + Phi_te @ B
            eq = float(np.mean(np.sum((target_te - pred_q) ** 2, axis=1)))
            acc[d]["el"].append(el)
            acc[d]["eq"].append(eq)
            acc[d]["df"].append(df_lam)
            acc[d]["de"].append(d)
    out: dict[int, dict] = {}
    for d in dims:
        if not acc[d]["eq"]:
            out[d] = {"E_linear": float("nan"), "E_quadratic": float("nan"), "ok": False}
            continue
        df_mean = float(np.mean(acc[d]["df"]))
        out[d] = {
            "E_linear": float(np.mean(acc[d]["el"])),
            "E_quadratic": float(np.mean(acc[d]["eq"])),
            "quadratic_gain": float(np.mean(acc[d]["el"]) - np.mean(acc[d]["eq"])),
            "neff": float(n),
            "df_lambda": df_mean,
            "neff_over_df": float(n / max(df_mean, EPS)),
            "d_eff": int(np.median(acc[d]["de"])),
            "ok": True,
        }
    return out


def stage_quadratic_dimension(root: Path, cfg: MultiModelConfig) -> None:
    out = cfg.resolved_out(root)
    marker = out / "quadratic_dimension" / "ready.json"
    if _done(marker, cfg.force):
        return
    (out / "quadratic_dimension").mkdir(exist_ok=True)
    graph = pd.read_parquet(out / "graph_dimension_prior.parquet")
    anchors_local = np.load(out / "prepare" / "anchors.npz")["anchors_local"]
    anchors_sid = np.load(out / "prepare" / "anchors.npz")["anchors_sample_id"]
    # screen subset: 96 stratified by sample_id only (model-independent)
    rng = np.random.default_rng(cfg.seed + 3)
    order = np.argsort(anchors_sid)
    screen_idx = order[:: max(1, len(order) // cfg.screen_anchors)][: cfg.screen_anchors]
    screen_local = anchors_local[screen_idx]
    screen_sid = anchors_sid[screen_idx]

    rows = []
    agree_rows = []
    scale_rows = []
    for m in cfg.graph_screen_models:
        if m not in cfg.models:
            continue
        X = load_model_X(out, m)
        pack = np.load(out / "model_neighbourhoods" / f"{m}_kmax{max(cfg.scales)}.npz")
        D = X.shape[1]
        sketches = [jl_sketch(D, cfg.sketch_dim, cfg.seed + 17 * s) for s in range(cfg.n_sketches)]
        g_m = graph[graph.model == m]
        for ki, k in enumerate(cfg.scales):
            # candidates from median graph prior on screen
            g_sk = g_m[(g_m.scale_k == k) & (g_m.sample_id.isin(screen_sid))]
            d0 = int(np.nanmedian(g_sk.d_graph)) if len(g_sk) else 8
            cands = sorted(set([
                max(2, d0 - 4), max(2, d0 - 2), d0, d0 + 2, d0 + 4,
                max(2, d0 - 6), d0 + 6, 8,
            ]))
            cands = [d for d in cands if d <= min(32, k // 6)]
            if not cands:
                cands = [4, 6, 8]
            err_by_d = {d: [] for d in cands}
            per_anchor_eq: dict[int, dict[int, list[float]]] = {}
            underpowered_flags = []
            for ai_s, (a, sid) in enumerate(zip(screen_local, screen_sid)):
                if ai_s % 8 == 0:
                    print(f"[quad] {m} k={k} screen {ai_s}/{len(screen_local)}", flush=True)
                ai = int(np.where(anchors_local == a)[0][0])
                Xn = X[pack["neigh"][ai, :k]]
                d_graph_i = int(g_m[(g_m.sample_id == sid) & (g_m.scale_k == k)].d_graph.iloc[0]) if (
                    ((g_m.sample_id == sid) & (g_m.scale_k == k)).any()
                ) else d0
                unverifiable = bool(d_graph_i > max(cands)) if cands else False
                per_anchor_eq[int(sid)] = {d: [] for d in cands}
                mean_by_d: dict[int, list[float]] = {d: [] for d in cands}
                last_ok: dict[int, dict] = {}
                for si, S in enumerate(sketches):
                    multi = oof_quad_errors_multi(
                        Xn, cands, S, n_folds=5, seed=cfg.seed + ai + 17 * si + k
                    )
                    for d, r in multi.items():
                        if not r["ok"]:
                            continue
                        mean_by_d[d].append(r["E_quadratic"])
                        per_anchor_eq[int(sid)][d].append(r["E_quadratic"])
                        last_ok[d] = r
                        rows.append(
                            {
                                "model": m,
                                "sample_id": int(sid),
                                "scale_k": k,
                                "d": d,
                                "d_graph": d_graph_i,
                                "quadratically_unverifiable_at_scale": unverifiable,
                                "underpowered": bool(
                                    np.isfinite(r.get("neff_over_df", np.nan))
                                    and r["neff_over_df"] < cfg.neff_df_min
                                ),
                                **{kk: r[kk] for kk in r if kk != "ok"},
                                "sketch": True,
                                "sketch_id": si,
                            }
                        )
                for d, es in mean_by_d.items():
                    if es:
                        err_by_d[d].append(float(np.mean(es)))
                        if d in last_ok and np.isfinite(last_ok[d].get("neff_over_df", np.nan)):
                            underpowered_flags.append(last_ok[d]["neff_over_df"] < cfg.neff_df_min)

            # full-ambient parity on up to 16 anchors at two extremal ds
            parity_ok = True
            if cands and len(screen_local) >= 4:
                check_sids = screen_sid[: min(16, len(screen_sid))]
                d_lo, d_hi = min(cands), max(cands)
                rank_sk, rank_full = [], []
                for sid in check_sids:
                    a = int(anchors_local[np.where(anchors_sid == sid)[0][0]])
                    ai = int(np.where(anchors_local == a)[0][0])
                    Xn = X[pack["neigh"][ai, :k]]
                    esk_multi = [
                        oof_quad_errors_multi(Xn, [d_lo, d_hi], S, n_folds=5, seed=cfg.seed + 999 + si)
                        for si, S in enumerate(sketches)
                    ]
                    e_sk = []
                    for d in (d_lo, d_hi):
                        vals = [m[d]["E_quadratic"] for m in esk_multi if m[d]["ok"]]
                        e_sk.append(float(np.mean(vals)) if vals else np.nan)
                    full = oof_quad_errors_multi(Xn, [d_lo, d_hi], None, n_folds=5, seed=cfg.seed + 1001)
                    e_full = [
                        full[d]["E_quadratic"] if full[d]["ok"] else np.nan for d in (d_lo, d_hi)
                    ]
                    if np.all(np.isfinite(e_sk)) and np.all(np.isfinite(e_full)):
                        rank_sk.append(int(np.argmin(e_sk)))
                        rank_full.append(int(np.argmin(e_full)))
                if rank_sk and rank_full:
                    agree = float(np.mean(np.asarray(rank_sk) == np.asarray(rank_full)))
                    parity_ok = agree >= 0.75
                    if not parity_ok:
                        print(
                            f"[quad] WARNING sketch/full ranking disagree ({agree:.2f}) for {m} k={k}",
                            flush=True,
                        )

            means = {d: float(np.mean(v)) for d, v in err_by_d.items() if v}
            ses = {
                d: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                for d, v in err_by_d.items()
                if v
            }
            if means:
                d_best = min(means, key=means.get)
                thresh = means[d_best] + ses.get(d_best, 0.0)
                d_star = min([d for d, e in means.items() if e <= thresh])
                # per-anchor d_quad via 1-SE on sketch-mean errors
                dquad_map = {}
                regret = []
                for sid, ded in per_anchor_eq.items():
                    means_i = {d: float(np.mean(v)) for d, v in ded.items() if v}
                    if not means_i:
                        continue
                    ses_i = {
                        d: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                        for d, v in ded.items()
                        if v
                    }
                    db = min(means_i, key=means_i.get)
                    th = means_i[db] + ses_i.get(db, 0.0)
                    dq = min([d for d, e in means_i.items() if e <= th])
                    dquad_map[sid] = dq
                    dg_i = g_sk.set_index("sample_id").d_graph
                    if sid in dg_i.index and int(dg_i.loc[sid]) in means_i:
                        regret.append(means_i[int(dg_i.loc[sid])] - means_i[dq])
                if dquad_map:
                    dquad = pd.Series(dquad_map)
                    dg = g_sk.set_index("sample_id").d_graph.reindex(dquad.index)
                    diff = dquad.to_numpy(float) - dg.to_numpy(float)
                    agree_rows.append(
                        {
                            "model": m,
                            "scale_k": k,
                            "median_d_graph": float(np.nanmedian(dg)),
                            "median_d_quad": float(np.nanmedian(dquad)),
                            "spearman": spearman_dict(dg.to_numpy(float), dquad.to_numpy(float))["rho"],
                            "median_abs_diff": float(np.nanmedian(np.abs(diff))),
                            "agree_exact": float(np.nanmean(diff == 0)),
                            "agree_pm2": float(np.nanmean(np.abs(diff) <= 2)),
                            "agree_pm4": float(np.nanmean(np.abs(diff) <= 4)),
                            "median_graph_regret": float(np.nanmedian(regret)) if regret else float("nan"),
                            "d_star": int(d_star),
                            "E_at_d_star": means[d_star],
                            "E_at_d_best": means[d_best],
                            "sketch_full_parity_ok": parity_ok,
                            "frac_underpowered": float(np.mean(underpowered_flags)) if underpowered_flags else float("nan"),
                        }
                    )
                scale_rows.append(
                    {
                        "model": m,
                        "scale_k": k,
                        "d_star": int(d_star),
                        "d_best": int(d_best),
                        "candidates": cands,
                        "d_graph_median": int(d0),
                        "quadratically_unverifiable_at_scale": bool(int(d0) > max(cands)) if cands else False,
                        "sketch_full_parity_ok": parity_ok,
                        "dimension_unresolved": not parity_ok,
                    }
                )
            print(f"[quad] {m} k={k} d_star={scale_rows[-1]['d_star'] if scale_rows else None}", flush=True)
    pd.DataFrame(rows).to_parquet(out / "quadratic_dimension_cv.parquet", index=False)
    pd.DataFrame(agree_rows).to_parquet(out / "graph_quadratic_agreement.parquet", index=False)
    pd.DataFrame(scale_rows).to_parquet(out / "scale_dimension_summary.parquet", index=False)
    marker.write_text(json.dumps({"n_rows": len(rows), "screen_anchors": len(screen_sid)}, indent=2))


# -------------------- Stage: curvature (gated) --------------------


def stage_curvature(root: Path, cfg: MultiModelConfig) -> None:
    out = cfg.resolved_out(root)
    marker = out / "curvature" / "ready.json"
    if _done(marker, cfg.force):
        return
    (out / "curvature").mkdir(exist_ok=True)
    scale_sum = pd.read_parquet(out / "scale_dimension_summary.parquet")
    agree = pd.read_parquet(out / "graph_quadratic_agreement.parquet")
    geo = pd.read_parquet(out / "local_probe_fields.parquet")
    anchors_local = np.load(out / "prepare" / "anchors.npz")["anchors_local"]
    anchors_sid = np.load(out / "prepare" / "anchors.npz")["anchors_sample_id"]
    # only screen anchors that appear in quadratic cv
    qcv = pd.read_parquet(out / "quadratic_dimension_cv.parquet")
    screen_sids = sorted(qcv.sample_id.unique())

    feat_rows = []
    assoc_rows = []
    for _, srow in scale_sum.iterrows():
        m = srow.model
        k = int(srow.scale_k)
        d_star = int(srow.d_star)
        if bool(getattr(srow, "dimension_unresolved", False)) or not bool(
            getattr(srow, "sketch_full_parity_ok", True)
        ):
            print(f"[curv] skip {m} k={k}: sketch/full parity failed", flush=True)
            continue
        if bool(getattr(srow, "quadratically_unverifiable_at_scale", False)):
            print(f"[curv] skip {m} k={k}: graph prior unverifiable at scale", flush=True)
            continue
        ag = agree[(agree.model == m) & (agree.scale_k == k)]
        if ag.empty:
            continue
        sub = qcv[(qcv.model == m) & (qcv.scale_k == k) & (qcv.d == d_star)]
        if sub.empty or float(sub.quadratic_gain.mean()) <= 0:
            print(f"[curv] skip {m} k={k}: no quadratic gain", flush=True)
            continue
        if float(sub.neff_over_df.median()) < cfg.neff_df_min:
            print(f"[curv] skip {m} k={k}: underpowered", flush=True)
            continue
        X = load_model_X(out, m)
        pack = np.load(out / "model_neighbourhoods" / f"{m}_kmax{max(cfg.scales)}.npz")
        # consensus weight for primary target if stable
        wmeta = pd.read_parquet(out / "global_probe_weights.parquet")
        cvmeta = pd.read_parquet(out / "global_probe_cv_metrics.parquet")
        for d in [max(2, d_star - 2), d_star, d_star + 2]:
            for sid in screen_sids:
                ai = int(np.where(anchors_sid == sid)[0][0])
                N = pack["neigh"][ai, :k]
                rho = float(pack["dists"][ai, k - 1])
                from .confirmatory_object_curvature import _fit_neighborhood

                chart, _, info, _, _, reason = _fit_neighborhood(X, N, d, seed=cfg.seed + ai + 17 * k + d)
                if chart is None:
                    continue
                B0, H = traceless_B0(chart.BS_flat, chart.J.shape[1])
                d_eff = chart.J.shape[1]
                K_mean = float(np.linalg.norm(H))
                K_tr = float(np.linalg.norm(B0))
                mean_frac = float(d_eff * K_mean**2 / max(K_tr**2 + d_eff * K_mean**2, EPS))
                # thickness
                dx = X[N] - chart.x0
                u = dx @ chart.J
                resid = dx - u @ chart.J.T
                x0u = chart.x0 / max(np.linalg.norm(chart.x0), EPS)
                resid = resid - np.outer(resid @ x0u, x0u)
                thick = float(np.var(resid) / max(np.var(u) + np.var(resid), EPS))
                row = {
                    "model": m,
                    "sample_id": int(sid),
                    "scale_k": k,
                    "d": d,
                    "d_star": d_star,
                    "d_eff": d_eff,
                    "K_mean": K_mean,
                    "K_traceless": K_tr,
                    "mean_fraction": mean_frac,
                    "normal_thickness": thick,
                    "recon_error": float(info.get("val_E_TRS", np.nan)),
                    "knn_radius": rho,
                    "log_knn_radius": float(np.log(max(rho, EPS))),
                }
                # probe facing for stable targets
                for t in cfg.targets:
                    st = cvmeta[(cvmeta.model == m) & (cvmeta.target == t)]
                    if st.empty or not bool(st.iloc[0].direction_stable):
                        continue
                    wz = np.load(out / "global_probes" / "oof_predictions" / f"{m}_{t}.npz")
                    w = wz["w_full"]
                    from .global_probe_curvature_falsification import unit_normal_component, probe_facing
                    from .global_probe_curvature_alignment import projection_energies

                    wN = unit_normal_component(w, chart.J, chart.x0)
                    pf = probe_facing(B0, H, wN, d_eff)
                    Bflat = B0_flat_for_svd(B0, d_eff)
                    Ub, s, _ = np.linalg.svd(Bflat, full_matrices=False)
                    keep = s > 1e-8 * (s[0] if len(s) else 1)
                    UB = Ub[:, keep] if np.any(keep) else Ub
                    en = projection_energies(w, chart.J, chart.x0, UB, UB)
                    row[f"K_probe_{t}"] = pf["K_probe"]
                    row[f"A_B_normal_{t}"] = en["A_B_normal"]
                feat_rows.append(row)
        print(f"[curv] {m} k={k} nfeat={len([r for r in feat_rows if r['model']==m and r['scale_k']==k])}", flush=True)

        # associations at d_star with OOF local performance
        feats = pd.DataFrame([r for r in feat_rows if r["model"] == m and r["scale_k"] == k and r["d"] == d_star])
        if feats.empty:
            continue
        ggeo = geo[(geo.model == m) & (geo.scale_k == k) & (geo.neighbourhood == "model")]
        for t in cfg.targets:
            g = ggeo[ggeo.target == t].merge(
                feats.drop(columns=[c for c in ["model"] if c in feats.columns]),
                on=["sample_id", "scale_k"],
                how="inner",
                suffixes=("", "_feat"),
            )
            if len(g) < 20:
                continue
            gp = pd.read_parquet(out / "graph_dimension_prior.parquet")
            g = g.merge(
                gp[(gp.model == m) & (gp.scale_k == k)][
                    ["sample_id", "graph_support_turnover", "graph_boundary_imbalance"]
                ],
                on="sample_id",
                how="left",
            )
            log_r = g["log_knn_radius_feat"].to_numpy(float) if "log_knn_radius_feat" in g.columns else g["log_knn_radius"].to_numpy(float)
            C = np.column_stack(
                [
                    log_r,
                    g.local_label_variance.to_numpy(float),
                    g.recon_error.to_numpy(float),
                    g.local_evaluation_count.to_numpy(float),
                    g.graph_support_turnover.fillna(0).to_numpy(float),
                    g.graph_boundary_imbalance.fillna(0).to_numpy(float),
                ]
            )
            for metric in ["local_r2", "normalized_mse", "local_prediction_correlation"]:
                yv = g[metric].to_numpy(float)
                if metric == "normalized_mse":
                    yv = -yv
                raw = spearman_dict(g.K_mean.to_numpy(float), yv)
                part = partial_spearman(g.K_mean.to_numpy(float), yv, C)
                assoc_rows.append(
                    {
                        "model": m,
                        "target": t,
                        "scale_k": k,
                        "d_star": d_star,
                        "metric": metric,
                        "raw_rho_K_mean": raw["rho"],
                        "partial_rho_K_mean": part["rho"],
                        "p_partial": part["pvalue"],
                        "n": part["n"],
                    }
                )

    pd.DataFrame(feat_rows).to_parquet(out / "curvature_features.parquet", index=False)
    pd.DataFrame(assoc_rows).to_parquet(out / "curvature_probe_associations.parquet", index=False)
    marker.write_text(json.dumps({"n_feat": len(feat_rows), "n_assoc": len(assoc_rows)}, indent=2))


# -------------------- Stage: sample-bootstrap inference --------------------


def stage_inference(root: Path, cfg: MultiModelConfig) -> None:
    out = cfg.resolved_out(root)
    path = out / "sample_bootstrap_inference.parquet"
    if _done(path, cfg.force):
        return
    if not (out / "curvature_features.parquet").exists():
        pd.DataFrame([]).to_parquet(path, index=False)
        return
    feats = pd.read_parquet(out / "curvature_features.parquet")
    if feats.empty:
        pd.DataFrame([]).to_parquet(path, index=False)
        return
    folds = pd.read_parquet(out / "sample_folds.parquet")
    n = len(folds)
    rng = np.random.default_rng(cfg.seed + 9)
    rows = []
    primary = cfg.primary_target
    tj = cfg.targets.index(primary)
    y_all = folds[f"y_{primary}"].to_numpy(float)
    for m in sorted(feats.model.unique()):
        oof = np.load(out / "global_probes" / "oof_predictions" / f"{m}_all_targets.npz")["oof"][:, tj]
        pack = np.load(out / "model_neighbourhoods" / f"{m}_kmax{max(cfg.scales)}.npz")
        sid_of_anchor = np.load(out / "prepare" / "anchors.npz")["anchors_sample_id"]
        fold_ids = folds.fold.to_numpy(int)
        for k in sorted(int(x) for x in feats.scale_k.unique()):
            f = feats[(feats.model == m) & (feats.scale_k == k) & (feats.d == feats.d_star)].reset_index(drop=True)
            if len(f) < 20:
                continue
            local_r2, Ks, ctrls, ais, K_rows = [], [], [], [], []
            for _, row in f.iterrows():
                sid = int(row.sample_id)
                ai = int(np.where(sid_of_anchor == sid)[0][0])
                N = pack["neigh"][ai, :k]
                msk = np.isfinite(y_all[N]) & np.isfinite(oof[N])
                if msk.sum() < 4:
                    continue
                local_r2.append(local_r2_fixed_predictions(y_all[N], oof[N]))
                Ks.append(float(row.K_mean))
                ctrls.append(
                    [
                        float(row.log_knn_radius),
                        float(np.var(y_all[N][msk])),
                        float(row.recon_error),
                        float(msk.sum()),
                    ]
                )
                ais.append(ai)
                K_rows.append(row)
            if len(local_r2) < 20:
                continue
            Ks_a = np.asarray(Ks, float)
            r2_a = np.asarray(local_r2, float)
            C0 = np.asarray(ctrls, float)
            real = partial_spearman(Ks_a, r2_a, C0)["rho"]
            boots = []
            for _ in range(200):
                take = rng.integers(0, n, size=n)
                counts = np.bincount(take, minlength=n).astype(np.float64)
                r2_b, Kb, Cb = [], [], []
                for ai, row in zip(ais, K_rows):
                    N = pack["neigh"][ai, :k]
                    w = counts[N]
                    msk = (w > 0) & np.isfinite(y_all[N]) & np.isfinite(oof[N])
                    if msk.sum() < 4:
                        continue
                    r2_b.append(weighted_r2(y_all[N][msk], oof[N][msk], w[msk]))
                    yv = y_all[N][msk]
                    mu = np.average(yv, weights=w[msk])
                    Kb.append(float(row.K_mean))
                    Cb.append(
                        [
                            float(row.log_knn_radius),
                            float(np.average((yv - mu) ** 2, weights=w[msk])),
                            float(row.recon_error),
                            float(w[msk].sum()),
                        ]
                    )
                if len(r2_b) < 15:
                    continue
                boots.append(partial_spearman(np.asarray(Kb), np.asarray(r2_b), np.asarray(Cb))["rho"])
            boots_a = np.asarray([b for b in boots if np.isfinite(b)], float)
            fold_rhos = []
            for fi in range(cfg.n_folds):
                keep = [
                    ii
                    for ii, ai in enumerate(ais)
                    if (fold_ids[pack["neigh"][ai, :k]] == fi).mean() < 0.5
                ]
                if len(keep) < 15:
                    continue
                fold_rhos.append(partial_spearman(Ks_a[keep], r2_a[keep], C0[keep])["rho"])
            rows.append(
                {
                    "model": m,
                    "target": primary,
                    "scale_k": int(k),
                    "partial_K_mean_local_r2": real,
                    "boot_lo": float(np.quantile(boots_a, 0.025)) if len(boots_a) else float("nan"),
                    "boot_hi": float(np.quantile(boots_a, 0.975)) if len(boots_a) else float("nan"),
                    "fold_sensitivity_median": float(np.nanmedian(fold_rhos)) if fold_rhos else float("nan"),
                    "n": int(len(local_r2)),
                    "n_boot": int(len(boots_a)),
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


# -------------------- Stage: analyze / REPORT --------------------


def stage_analyze(root: Path, cfg: MultiModelConfig) -> None:
    out = cfg.resolved_out(root)
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    cv = pd.read_parquet(out / "global_probe_cv_metrics.parquet")
    xref = pd.read_parquet(out / "crossmodel_probe_geography.parquet")
    labels = []

    # shared vs model-specific geography
    xref_p = xref[(xref.scale_k == 2048) & (xref.target == cfg.primary_target)]
    med = float(xref_p.spearman_local_r2.median()) if len(xref_p) else float("nan")
    if np.isfinite(med):
        labels.append("shared_global_probe_geography" if med >= 0.4 else "model_specific_probe_geography")
    else:
        labels.append("inconclusive")

    if (out / "graph_quadratic_agreement.parquet").exists():
        ag = pd.read_parquet(out / "graph_quadratic_agreement.parquet")
        if len(ag):
            if float(ag.agree_pm2.mean()) >= 0.5:
                labels.append("graph_prior_validated")
            elif float(ag.median_abs_diff.mean()) <= 4:
                labels.append("graph_prior_low_regret")
            else:
                labels.append("graph_prior_mismatch")

    if (out / "curvature_probe_associations.parquet").exists():
        assoc = pd.read_parquet(out / "curvature_probe_associations.parquet")
        prim = assoc[
            (assoc.target == cfg.primary_target)
            & (assoc.metric == "local_r2")
            & (assoc.model == "vit_base")
        ]
        if len(prim):
            # negative at multiple scales?
            neg = prim[prim.partial_rho_K_mean < -0.1]
            if len(neg) >= 2:
                labels.append("mean_curvature_replication")
            elif len(neg) == 1 and int(neg.iloc[0].scale_k) == 2048:
                labels.append("large_scale_only_curvature")
            # other models
            others = assoc[
                (assoc.target == cfg.primary_target)
                & (assoc.metric == "local_r2")
                & (assoc.model != "vit_base")
                & (assoc.partial_rho_K_mean < -0.1)
            ]
            if others.empty and len(neg):
                labels.append("vit_specific_mean_curvature")

    unstable = cv[cv.direction_stable == False]
    if len(unstable):
        labels.append("global_probe_direction_unstable")

    # plots
    fig1, ax = plt.subplots(figsize=(8, 4))
    piv = cv.pivot(index="target", columns="model", values="oof_r2")
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("OOF R²")
    ax.set_title("Global probe OOF R² by model/target")
    fig1.tight_layout()
    fig1.savefig(fig / "global_probe_oof_r2.png", dpi=140)
    plt.close(fig1)

    if len(xref):
        fig2, ax = plt.subplots(figsize=(6, 4))
        sub = xref[xref.target == cfg.primary_target]
        for (a, b), g in sub.groupby(["model_a", "model_b"]):
            ax.plot(g.scale_k, g.spearman_local_r2, marker="o", label=f"{a[:4]}-{b[:4]}")
        ax.legend(fontsize=7)
        ax.set_xlabel("k")
        ax.set_ylabel("Spearman local R²")
        ax.set_title(f"Cross-model geography ({cfg.primary_target})")
        fig2.tight_layout()
        fig2.savefig(fig / "crossmodel_geography.png", dpi=140)
        plt.close(fig2)

    if (out / "graph_quadratic_agreement.parquet").exists():
        ag = pd.read_parquet(out / "graph_quadratic_agreement.parquet")
        fig3, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(ag.median_d_graph, ag.median_d_quad, c=ag.scale_k, cmap="viridis")
        ax.plot([0, 32], [0, 32], "k--", lw=0.8)
        ax.set_xlabel("median d_graph")
        ax.set_ylabel("median d_quad")
        fig3.tight_layout()
        fig3.savefig(fig / "graph_vs_quad_dim.png", dpi=140)
        plt.close(fig3)

    if (out / "curvature_probe_associations.parquet").exists():
        assoc = pd.read_parquet(out / "curvature_probe_associations.parquet")
        sub = assoc[(assoc.target == cfg.primary_target) & (assoc.metric == "local_r2")]
        fig4, ax = plt.subplots(figsize=(7, 4))
        for m, g in sub.groupby("model"):
            ax.plot(g.scale_k, g.partial_rho_K_mean, marker="o", label=m)
        ax.axhline(0, color="gray", lw=0.8)
        ax.legend(fontsize=8)
        ax.set_title("Partial corr(K_mean, local OOF R²)")
        fig4.tight_layout()
        fig4.savefig(fig / "Kmean_association_by_model_scale.png", dpi=140)
        plt.close(fig4)

    schema = json.loads((out / "resolved_model_schema.json").read_text())
    report = f"""# Multi-model OOF global probe × graph/quadratic dimension × gated curvature

## Protocol

- No permanent holdout. Selection n={schema['n_selected']}.
- Shared 5-fold CV; local scores use **OOF predictions only**.
- Canonical train: `{CANONICAL_PROBE_TRAIN}`; local score: `{CANONICAL_PROBE_LOCAL}`; α={cfg.probe_alpha}.
- Graph prior: {GRAPH_PRIOR_SOURCE}

## Labels / models

Labels schema: `{cfg.labels_path}` keys={list(schema['labels'].keys())}.
Models: {cfg.models}

## Decision labels

{chr(10).join(f"- `{lab}`" for lab in labels)}

## Global probe OOF performance

{cv.to_string(index=False)}

## Cross-model geography (spearman of local OOF R²)

{xref[(xref.target==cfg.primary_target)].to_string(index=False) if len(xref) else "n/a"}

## Graph vs quadratic

{(pd.read_parquet(out/"graph_quadratic_agreement.parquet").to_string(index=False) if (out/"graph_quadratic_agreement.parquet").exists() else "not run / empty")}

## Scale-wide d_star

{(pd.read_parquet(out/"scale_dimension_summary.parquet").to_string(index=False) if (out/"scale_dimension_summary.parquet").exists() else "n/a")}

## Curvature–probe associations (gated)

{(pd.read_parquet(out/"curvature_probe_associations.parquet").to_string(index=False) if (out/"curvature_probe_associations.parquet").exists() else "no cells passed gates")}

## Primary replication question

Does negative K_mean association for vit_base / mag_r_desi / k=2048 replicate at smaller k and other models?
See associations table and labels (`mean_curvature_replication` / `large_scale_only_curvature` / `vit_specific_mean_curvature`).

## Stage C confirmation commands

Curvature was run on the 96-anchor screen for cells passing quadratic gates only.
To confirm a cell on all 512 anchors:

```bash
# example after inspecting scale_dimension_summary.parquet
PYTHONPATH=experiments python -m geometry.run_physics_multimodel_graph_prior_quadratic \\
  --stage curvature --force --seed 0
# then extend screen_anchors / implement full-anchor flag as needed
```

## Strongest defensible conclusion

Multi-model OOF global-probe geography and graph/quadratic dimension screening are the primary deliverables.
Mean-curvature–performance links are reported only for gated cells and must not be read as causal.
Interaction/alignment claims remain exploratory after prior falsification.

## Exact command

```bash
cd ~/platonic-universe && source .venv/bin/activate && \\
PYTHONPATH=experiments python -m geometry.run_physics_multimodel_graph_prior_quadratic \\
  --stage all --seed 0
```
"""
    (out / "REPORT.md").write_text(report)
    pd.DataFrame([{"label": lab} for lab in labels]).to_csv(out / "decision_labels.csv", index=False)
    print(f"[analyze] labels={labels}", flush=True)


# -------------------- orchestrator --------------------


def run(cfg: MultiModelConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()
    profile: dict[str, Any] = {"stages": {}}
    stages = {
        "prepare": stage_prepare,
        "global_probes": stage_global_probes,
        "neighbourhoods": stage_neighbourhoods,
        "probe_geography": stage_probe_geography,
        "graph_prior": stage_graph_prior,
        "quadratic_dimension": stage_quadratic_dimension,
        "curvature": stage_curvature,
        "inference": stage_inference,
        "analyze": stage_analyze,
    }
    order = list(stages.keys())
    want = order if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    # dependencies
    deps = {
        "global_probes": ["prepare"],
        "neighbourhoods": ["prepare"],
        "probe_geography": ["global_probes", "neighbourhoods"],
        "graph_prior": ["neighbourhoods"],
        "quadratic_dimension": ["graph_prior", "neighbourhoods"],
        "curvature": ["quadratic_dimension", "probe_geography", "global_probes"],
        "inference": ["curvature"],
        "analyze": ["probe_geography"],
    }
    run_set = set(want)
    for s in want:
        for d in deps.get(s, []):
            run_set.add(d)
    for s in order:
        if s not in run_set:
            continue
        t1 = time.time()
        print(f"[run] stage={s}", flush=True)
        stages[s](root, cfg)
        profile["stages"][f"{s}_s"] = time.time() - t1
    profile.update(
        {
            "total_seconds": time.time() - t0,
            "peak_rss_mb": _rss(),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1024**2)
            if torch.cuda.is_available()
            else 0.0,
            "graph_prior_source": GRAPH_PRIOR_SOURCE,
        }
    )
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[run] done in {profile['total_seconds']:.1f}s", flush=True)
    return profile
