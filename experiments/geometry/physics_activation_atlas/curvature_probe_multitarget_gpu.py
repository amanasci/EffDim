"""GPU-batched multi-target curvature alignment + shared-feature ablation."""

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
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .confirmatory_object_curvature import _fit_neighborhood, select_anchors
from .curvature_probe_alignment import B0_flat_for_svd, alignment_from_w, traceless_B0
from .curvature_probe_screen import (
    EXPECTED_HASH,
    LOCAL_DIM,
    ScreenConfig,
    load_frozen_curvature,
    partial_spearman,
    spearman_dict,
)
from .curvature_probe_subspace_ablation import (
    ambient_quadratic_form,
    build_features,
    haar_normal_basis,
    normal_pca_basis,
    orthonormalize_mutually,
    phi_weighted,
    rematerialize_chart,
    reproject_into_normal,
)
from .data import load_prepare
from .paths import platonic_root, resolve_path

# Canonical Physics probe registry (bipartite-matching/_shared.DEFAULT_PROPERTIES)
CANONICAL_TARGETS = ["mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass", "sfr"]
PRIMARY_TARGET = "mag_r_desi"
EPS = 1e-12
PROBE_ALPHA = 100.0
SCALES = (1024, 2048)


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


@dataclass
class MultiTargetConfig:
    output_dir: str = "outputs/geometry/physics_curvature_probe_multitarget"
    curvature_path: str = (
        "outputs/geometry/physics_quadratic_atlas_sphere_normal/"
        "object_curvature_features_aggregated.parquet"
    )
    prepare_dir: str = "outputs/geometry/physics_activation_atlas_geometry_ablation/prepare"
    labels_path: str = "data_hf/physics/vit_base_test_labels.npz"
    expected_hash: str = EXPECTED_HASH
    scales: list[int] = field(default_factory=lambda: list(SCALES))
    primary_k: int = 2048
    primary_target: str = PRIMARY_TARGET
    probe_alpha: float = PROBE_ALPHA
    n_folds: int = 5
    n_random_bases: int = 20  # projection nulls (not probe refits)
    batch_anchors: int = 16
    n_bootstrap: int = 1000
    n_permute: int = 500
    parity_anchors: int = 8
    seed: int = 0
    force: bool = False
    device: str = "cuda"
    min_finite_frac_global: float = 0.5
    min_local_train: int = 8
    min_local_eval: int = 4

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


# -------------------- target inventory --------------------


def build_target_inventory(
    y_all: dict[str, np.ndarray], train_idx: np.ndarray, cfg: MultiTargetConfig
) -> pd.DataFrame:
    rows = []
    for name in CANONICAL_TARGETS:
        if name not in y_all:
            rows.append(
                {
                    "target": name,
                    "included": False,
                    "reason": "missing_from_labels_npz",
                    "finite_frac_global": float("nan"),
                    "finite_frac_train": float("nan"),
                    "std_train": float("nan"),
                    "role": "excluded",
                }
            )
            continue
        y = y_all[name]
        fg = float(np.isfinite(y).mean())
        yt = y[train_idx]
        ft = float(np.isfinite(yt).mean())
        std = float(np.nanstd(yt))
        reason = "ok"
        included = True
        if fg < cfg.min_finite_frac_global:
            included, reason = False, f"global_finite_frac<{cfg.min_finite_frac_global}"
        elif ft < 0.3:
            included, reason = False, "train_finite_frac<0.3"
        elif not np.isfinite(std) or std < 1e-8:
            included, reason = False, "effectively_constant"
        elif name == "sfr" and fg < 0.2:
            included, reason = False, "sparse_sfr_unsupported_by_local_protocol"
        role = "primary" if name == cfg.primary_target and included else (
            "secondary" if included else "excluded"
        )
        # family tags
        if name in ("mag_r_desi",):
            family = "photometry"
        elif name in ("photo_z",):
            family = "redshift"
        elif name in ("smooth_fraction",):
            family = "morphology"
        elif name in ("stellar_mass", "sfr"):
            family = "stellar_population"
        else:
            family = "other"
        rows.append(
            {
                "target": name,
                "included": included,
                "reason": reason,
                "finite_frac_global": fg,
                "finite_frac_train": ft,
                "std_train": std,
                "role": role,
                "family": family,
            }
        )
    return pd.DataFrame(rows)


# -------------------- GPU ridge --------------------


def _torch_device(cfg: MultiTargetConfig) -> torch.device:
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def standardize_xy_torch(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Column-standardize X and Y (Y may be multi-col)."""
    x_mean = X.mean(dim=0, keepdim=True)
    x_std = X.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    y_mean = Y.mean(dim=0, keepdim=True)
    y_std = Y.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    Xs = (X - x_mean) / x_std
    Ys = (Y - y_mean) / y_std
    return Xs, Ys, {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def ridge_multi_solve(
    X: torch.Tensor, Y: torch.Tensor, *, alpha: float
) -> tuple[torch.Tensor, bool]:
    """
    Multi-output ridge: W = solve(X'X + aI, X'Y), shapes X(n,f), Y(n,t) -> W(f,t).
    Returns (W, ok).
    """
    n, f = X.shape
    XtX = X.T @ X
    XtX = XtX + alpha * torch.eye(f, device=X.device, dtype=X.dtype)
    XtY = X.T @ Y
    L, info = torch.linalg.cholesky_ex(XtX)
    if int(info.item()) != 0:
        return torch.zeros(f, Y.shape[1], device=X.device, dtype=X.dtype), False
    W = torch.cholesky_solve(XtY, L)
    return W, True


def ridge_r2_multi_torch(
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return per-target R² (t,), ambient/feature weights W_feat (f,t) mapped to raw X scale, ok."""
    if X_tr.shape[0] < 8 or X_te.shape[0] < 4:
        t = Y_tr.shape[1]
        return np.full(t, np.nan), np.zeros((X_tr.shape[1], t)), False
    Xs, Ys, st = standardize_xy_torch(X_tr, Y_tr)
    W_std, ok = ridge_multi_solve(Xs, Ys, alpha=alpha)
    if not ok:
        # float64 fallback on CPU
        X64 = Xs.double()
        Y64 = Ys.double()
        W_std, ok = ridge_multi_solve(X64, Y64, alpha=alpha)
        W_std = W_std.to(Xs.dtype)
        if not ok:
            t = Y_tr.shape[1]
            return np.full(t, np.nan), np.zeros((X_tr.shape[1], t)), False
    # predict on test in standardized space then invert y
    Xs_te = (X_te - st["x_mean"]) / st["x_std"]
    Ys_hat = Xs_te @ W_std
    Y_hat = Ys_hat * st["y_std"] + st["y_mean"]
    y_te = Y_te.detach().cpu().numpy()
    y_hat = Y_hat.detach().cpu().numpy()
    t = y_te.shape[1]
    r2 = np.full(t, np.nan)
    for j in range(t):
        m = np.isfinite(y_te[:, j]) & np.isfinite(y_hat[:, j])
        if m.sum() < 4 or np.var(y_te[m, j]) < 1e-12:
            continue
        r2[j] = float(r2_score(y_te[m, j], y_hat[m, j]))
    # weights on raw X: w_raw = w_std / x_std
    W_raw = (W_std / st["x_std"].reshape(-1, 1)).detach().cpu().numpy()
    return r2, W_raw, True


def sklearn_ridge_r2_weight(X_tr, y_tr, X_te, y_te, *, alpha: float) -> tuple[float, np.ndarray]:
    m_tr = np.isfinite(y_tr)
    m_te = np.isfinite(y_te)
    if m_tr.sum() < 8 or m_te.sum() < 4:
        return float("nan"), np.zeros(X_tr.shape[1])
    xs = StandardScaler().fit(X_tr[m_tr])
    ys = StandardScaler().fit(y_tr[m_tr].reshape(-1, 1))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(xs.transform(X_tr[m_tr]), ys.transform(y_tr[m_tr].reshape(-1, 1)).ravel())
    pred = ys.inverse_transform(model.predict(xs.transform(X_te[m_te])).reshape(-1, 1)).ravel()
    r2 = float(r2_score(y_te[m_te], pred))
    w = (model.coef_ / np.maximum(xs.scale_, EPS)).astype(np.float64)
    return r2, w


# -------------------- alignment from energy --------------------


def projection_energies(w: np.ndarray, T: np.ndarray, x0u: np.ndarray, UB: np.ndarray, UN: np.ndarray) -> dict:
    e_total = float(np.dot(w, w))
    if e_total < EPS:
        return {k: float("nan") for k in (
            "e_total", "e_T", "e_R", "e_N", "e_B", "e_NPCA",
            "A_T", "A_N", "A_B_total", "A_B_normal", "A_PCA_total", "A_PCA_normal",
        )}
    e_T = float(np.sum((T.T @ w) ** 2))
    e_R = float((np.dot(x0u, w)) ** 2)
    e_N = max(e_total - e_T - e_R, 0.0)
    e_B = float(np.sum((UB.T @ w) ** 2)) if UB.size else 0.0
    e_P = float(np.sum((UN.T @ w) ** 2)) if UN.size else 0.0
    return {
        "e_total": e_total,
        "e_T": e_T,
        "e_R": e_R,
        "e_N": e_N,
        "e_B": e_B,
        "e_NPCA": e_P,
        "A_T": e_T / (e_total + EPS),
        "A_N": e_N / (e_total + EPS),
        "A_B_total": e_B / (e_total + EPS),
        "A_B_normal": e_B / (e_N + EPS),
        "A_PCA_total": e_P / (e_total + EPS),
        "A_PCA_normal": e_P / (e_N + EPS),
    }


# -------------------- stats helpers --------------------


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.ones(n, dtype=float)
    prev = 1.0
    for i, idx in enumerate(order[::-1]):
        rank = n - i
        val = min(prev, p[idx] * n / rank) if np.isfinite(p[idx]) else float("nan")
        adj[idx] = val
        if np.isfinite(val):
            prev = val
    return adj


def fisher_meta(rhos: np.ndarray, ns: np.ndarray) -> dict:
    m = np.isfinite(rhos) & np.isfinite(ns) & (ns > 3)
    if m.sum() == 0:
        return {"z": float("nan"), "rho": float("nan"), "n_targets": 0}
    z = np.arctanh(np.clip(rhos[m], -0.999, 0.999))
    w = ns[m] - 3.0
    zbar = float(np.sum(w * z) / max(w.sum(), EPS))
    return {"z": zbar, "rho": float(np.tanh(zbar)), "n_targets": int(m.sum())}


def bootstrap_spearman_ci(x, y, n_boot, seed):
    m = np.isfinite(x) & np.isfinite(y)
    idx = np.where(m)[0]
    if len(idx) < 12:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        r, _ = spearmanr(x[take], y[take])
        if np.isfinite(r):
            rhos.append(float(r))
    if not rhos:
        return [float("nan"), float("nan")]
    return [float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975))]


def mc_p_twosided(real, nulls):
    nulls = np.asarray(nulls, float)
    nulls = nulls[np.isfinite(nulls)]
    B = len(nulls)
    if B == 0 or not np.isfinite(real):
        return float("nan"), 0
    return float((1 + np.sum(np.abs(nulls) >= abs(real))) / (B + 1)), B


# -------------------- cache geometry --------------------


def ensure_geometry_cache(
    root: Path, cfg: MultiTargetConfig, X: np.ndarray, train: np.ndarray, anchors: np.ndarray, data: dict
) -> Path:
    cache = cfg.resolved_out(root) / "geometry_cache"
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / "ready.json"
    if _done(marker, cfg.force):
        return cache
    k_max = max(cfg.scales)
    nn = NearestNeighbors(n_neighbors=k_max, metric="euclidean")
    nn.fit(X[train])
    dists, inds = nn.kneighbors(X[anchors])
    UB_pool: dict[int, list] = {k: [] for k in cfg.scales}
    # first pass for shuffle pool
    packs_tmp = {}
    for k in cfg.scales:
        for ai, a_local in enumerate(anchors):
            neigh = train[inds[ai, :k]]
            neigh = neigh[neigh != a_local]
            pack = rematerialize_chart(X, neigh, ai, k, cfg.seed)
            if pack is None:
                continue
            dx = X[neigh] - pack["x0u"][None, :]
            UN = normal_pca_basis(dx, pack["x0u"], pack["T"], pack["UB"].shape[1])
            packs_tmp[(ai, k)] = {
                **pack,
                "neigh": neigh,
                "rho": float(dists[ai, k - 1]),
                "UNPCA": UN,
                "sample_id": int(data["sample_ids"][a_local]),
            }
            UB_pool[k].append(pack["UB"])
        print(f"[multitarget] cached rematerialize k={k} n={sum(1 for a,kk in packs_tmp if kk==k)}", flush=True)

    rng = np.random.default_rng(cfg.seed + 9)
    for (ai, k), pack in packs_tmp.items():
        r = pack["UB"].shape[1]
        U_rands = np.stack(
            [haar_normal_basis(X.shape[1], pack["x0u"], pack["T"], r, rng) for _ in range(cfg.n_random_bases)],
            axis=2,
        )  # D,r,nrand
        # shuffled from another anchor
        pool = UB_pool[k]
        other = pool[rng.integers(0, len(pool))]
        if other.shape[1] >= r:
            Uo = other[:, :r]
        else:
            Uo = np.column_stack([other, haar_normal_basis(X.shape[1], pack["x0u"], pack["T"], r - other.shape[1], rng)])[:, :r]
        U_shuf = reproject_into_normal(Uo, pack["x0u"], pack["T"])
        if U_shuf.shape[1] < r:
            U_shuf = np.column_stack(
                [U_shuf, haar_normal_basis(X.shape[1], pack["x0u"], pack["T"], r - U_shuf.shape[1], rng)]
            )[:, :r]
        path = cache / f"k{k}_ai{ai:04d}.npz"
        np.savez_compressed(
            path,
            sample_id=pack["sample_id"],
            ai=ai,
            scale_k=k,
            rho=pack["rho"],
            neigh=pack["neigh"],
            T=pack["T"],
            x0u=pack["x0u"],
            UB=pack["UB"],
            UNPCA=pack["UNPCA"],
            B0=pack["B0"],
            U_rands=U_rands,
            U_shuf=U_shuf[:, :r],
            B0_fro=pack["B0_fro"],
        )
    marker.write_text(json.dumps({"n": len(packs_tmp), "scales": cfg.scales}, indent=2))
    return cache


# -------------------- per-anchor multi-target --------------------


def process_anchor(
    X: np.ndarray,
    y_mat: np.ndarray,
    target_names: list[str],
    pack_path: Path,
    *,
    alpha: float,
    n_folds: int,
    seed: int,
    device: torch.device,
) -> list[dict]:
    z = np.load(pack_path)
    neigh = z["neigh"]
    T, x0u, UB, UN, B0 = z["T"], z["x0u"], z["UB"], z["UNPCA"], z["B0"]
    rho = float(z["rho"])
    sid = int(z["sample_id"])
    k = int(z["scale_k"])
    U_rands = z["U_rands"]
    U_shuf = z["U_shuf"]
    Xn = X[neigh]
    Yn = y_mat[neigh]  # (n, n_targets)
    n = len(neigh)
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    folds = np.array_split(order, n_folds)

    # ambient CV multi-target with mask groups per fold aggregation
    # We'll compute fold-wise then average
    fold_r2 = {t: [] for t in target_names}
    fold_w = {t: [] for t in target_names}
    fold_align = {t: [] for t in target_names}

    for fi, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        X_tr, X_te = Xn[tr], Xn[te]
        Y_tr, Y_te = Yn[tr], Yn[te]
        # group targets by identical finite mask on train+need eval finite
        masks = {}
        for j, tname in enumerate(target_names):
            m_tr = np.isfinite(Y_tr[:, j])
            m_te = np.isfinite(Y_te[:, j])
            if m_tr.sum() < 8 or m_te.sum() < 4:
                fold_r2[tname].append(float("nan"))
                fold_w[tname].append(None)
                fold_align[tname].append(None)
                continue
            key = m_tr.tobytes()
            masks.setdefault(key, {"m_tr": m_tr, "targets": []})
            masks[key]["targets"].append(j)

        # solve each mask group on GPU
        solved_r2 = {}
        solved_w = {}
        for g in masks.values():
            m_tr = g["m_tr"]
            cols = g["targets"]
            # eval mask per target separately after joint train solve on shared train mask
            Xt = torch.tensor(X_tr[m_tr], device=device, dtype=torch.float32)
            Yt = torch.tensor(Y_tr[m_tr][:, cols], device=device, dtype=torch.float32)
            Xte = torch.tensor(X_te, device=device, dtype=torch.float32)
            Yte = torch.tensor(Y_te[:, cols], device=device, dtype=torch.float32)
            # For test, we still predict all rows then score finite
            r2, W, ok = ridge_r2_multi_torch(Xt, Yt, Xte, Yte, alpha=alpha)
            if not ok:
                Xt = Xt.double()
                Yt = Yt.double()
                Xte = Xte.double()
                Yte = Yte.double()
                r2, W, ok = ridge_r2_multi_torch(Xt, Yt, Xte, Yte, alpha=alpha)
            for li, j in enumerate(cols):
                solved_r2[j] = r2[li]
                solved_w[j] = W[:, li]

        for j, tname in enumerate(target_names):
            if j not in solved_r2:
                continue
            r2 = solved_r2[j]
            w = solved_w[j]
            fold_r2[tname].append(float(r2))
            fold_w[tname].append(w)
            en = projection_energies(w, T, x0u, UB, UN)
            # C_w via existing definition (uses B0 and normal component)
            cw = alignment_from_w(w, x0u, T, B0, rho).get("C_w", float("nan"))
            # null alignments: mean A_B_normal onto random / shuffled
            a_rand = []
            for ri in range(U_rands.shape[2]):
                er = projection_energies(w, T, x0u, U_rands[:, :, ri], UN)
                a_rand.append(er["A_B_normal"])
            a_shuf = projection_energies(w, T, x0u, U_shuf, UN)["A_B_normal"]
            fold_align[tname].append(
                {
                    **en,
                    "C_w": cw,
                    "A_B_random_median": float(np.nanmedian(a_rand)) if a_rand else float("nan"),
                    "A_B_shuffled": float(a_shuf),
                }
            )

    # ablation features shared across targets
    feats = build_features(Xn, x0u, T, UB, B0)
    feat_mats = {
        "MT": feats["z_T"],
        "MTQ": np.column_stack([feats["z_T"], feats["phi"]]),
        "MTBpred": np.column_stack([feats["z_T"], feats["z_B_pred"]]),
        "MTBobs": np.column_stack([feats["z_T"], feats["z_B_obs"]]),
        "MTNPCA": np.column_stack([feats["z_T"], feats["dx"] @ UN]),
        "Mfull": feats["ambient"],
    }

    def cv_r2_features_multi(Z: np.ndarray) -> np.ndarray:
        out = np.zeros(len(target_names))
        acc = [[] for _ in target_names]
        for fi, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
            # group by mask
            groups: dict[bytes, dict] = {}
            for j in range(len(target_names)):
                m_tr = np.isfinite(Yn[tr, j])
                m_te = np.isfinite(Yn[te, j])
                if m_tr.sum() < 8 or m_te.sum() < 4:
                    acc[j].append(float("nan"))
                    continue
                key = m_tr.tobytes()
                groups.setdefault(key, {"m_tr": m_tr, "cols": []})
                groups[key]["cols"].append(j)
            for g in groups.values():
                m_tr = g["m_tr"]
                cols = g["cols"]
                Xt = torch.tensor(Z[tr][m_tr], device=device, dtype=torch.float32)
                Yt = torch.tensor(Yn[tr][m_tr][:, cols], device=device, dtype=torch.float32)
                Xte = torch.tensor(Z[te], device=device, dtype=torch.float32)
                Yte = torch.tensor(Yn[te][:, cols], device=device, dtype=torch.float32)
                r2, _, ok = ridge_r2_multi_torch(Xt, Yt, Xte, Yte, alpha=alpha)
                for li, j in enumerate(cols):
                    acc[j].append(float(r2[li]) if ok else float("nan"))
        for j in range(len(target_names)):
            vals = [v for v in acc[j] if np.isfinite(v)]
            out[j] = float(np.mean(vals)) if vals else float("nan")
        return out

    abl_r2 = {name: cv_r2_features_multi(Z) for name, Z in feat_mats.items()}

    rows = []
    for j, tname in enumerate(target_names):
        r2s = np.asarray(fold_r2[tname], float)
        # stability of directions
        ws = [w for w in fold_w[tname] if w is not None and np.linalg.norm(w) > EPS]
        stab = float("nan")
        if len(ws) >= 2:
            sims = []
            for a in range(len(ws)):
                for b in range(a + 1, len(ws)):
                    ua, ub = ws[a] / np.linalg.norm(ws[a]), ws[b] / np.linalg.norm(ws[b])
                    sims.append(abs(float(np.dot(ua, ub))))
            stab = float(np.mean(sims))
        # mean alignments
        als = [a for a in fold_align[tname] if a is not None]
        def mean_key(key):
            vals = [a[key] for a in als if np.isfinite(a.get(key, np.nan))]
            return float(np.mean(vals)) if vals else float("nan")

        y_local = Yn[:, j]
        local_var = float(np.nanvar(y_local)) if np.isfinite(y_local).sum() > 1 else float("nan")
        r_MT = abl_r2["MT"][j]
        row = {
            "sample_id": sid,
            "scale_k": k,
            "target": tname,
            "probe_r2": float(np.nanmean(r2s)),
            "probe_r2_std": float(np.nanstd(r2s)),
            "n_folds_ok": int(np.isfinite(r2s).sum()),
            "probe_dir_stability": stab,
            "knn_radius": rho,
            "log_knn_radius": float(np.log(max(rho, EPS))),
            "local_target_variance": local_var,
            "A_T": mean_key("A_T"),
            "A_N": mean_key("A_N"),
            "A_B_total": mean_key("A_B_total"),
            "A_B_normal": mean_key("A_B_normal"),
            "A_PCA_total": mean_key("A_PCA_total"),
            "A_PCA_normal": mean_key("A_PCA_normal"),
            "C_w": mean_key("C_w"),
            "A_B_random_median": mean_key("A_B_random_median"),
            "A_B_shuffled": mean_key("A_B_shuffled"),
            "R2_MT": float(r_MT),
            "R2_MTQ": float(abl_r2["MTQ"][j]),
            "R2_MTBpred": float(abl_r2["MTBpred"][j]),
            "R2_MTBobs": float(abl_r2["MTBobs"][j]),
            "R2_MTNPCA": float(abl_r2["MTNPCA"][j]),
            "R2_Mfull": float(abl_r2["Mfull"][j]),
            "delta_Q": float(abl_r2["MTQ"][j] - r_MT),
            "delta_Bpred": float(abl_r2["MTBpred"][j] - r_MT),
            "delta_Bobs": float(abl_r2["MTBobs"][j] - r_MT),
            "delta_NPCA": float(abl_r2["MTNPCA"][j] - r_MT),
            "delta_full": float(abl_r2["Mfull"][j] - r_MT),
            "specificity": float(abl_r2["MTBobs"][j] - abl_r2["MTNPCA"][j]),
        }
        rows.append(row)
    return rows


# -------------------- parity --------------------


def run_parity(
    root: Path,
    cfg: MultiTargetConfig,
    X: np.ndarray,
    y_mat: np.ndarray,
    target_names: list[str],
    cache: Path,
    device: torch.device,
) -> dict:
    """Compare GPU multi-output vs sklearn on a small subset."""
    rows = []
    for k in cfg.scales:
        paths = sorted(cache.glob(f"k{k}_ai*.npz"))[: cfg.parity_anchors]
        for p in paths:
            z = np.load(p)
            neigh = z["neigh"]
            Xn = X[neigh]
            Yn = y_mat[neigh]
            n = len(neigh)
            rng = np.random.default_rng(cfg.seed + int(z["ai"]))
            idx = np.arange(n)
            rng.shuffle(idx)
            n_te = max(4, int(0.3 * n))
            te, tr = idx[:n_te], idx[n_te:]
            # pick up to 3 targets with enough finite
            used = []
            for j, t in enumerate(target_names):
                if np.isfinite(Yn[tr, j]).sum() >= 8 and np.isfinite(Yn[te, j]).sum() >= 4:
                    used.append(j)
                if len(used) >= 3:
                    break
            if not used:
                continue
            # sklearn per target
            for j in used:
                r_sk, w_sk = sklearn_ridge_r2_weight(
                    Xn[tr], Yn[tr, j], Xn[te], Yn[te, j], alpha=cfg.probe_alpha
                )
                # GPU joint on this mask group (single target for fair compare)
                m_tr = np.isfinite(Yn[tr, j])
                Xt = torch.tensor(Xn[tr][m_tr], device=device, dtype=torch.float32)
                Yt = torch.tensor(Yn[tr][m_tr, j : j + 1], device=device, dtype=torch.float32)
                Xte = torch.tensor(Xn[te], device=device, dtype=torch.float32)
                Yte = torch.tensor(Yn[te, j : j + 1], device=device, dtype=torch.float32)
                r_gpu, W_gpu, ok = ridge_r2_multi_torch(Xt, Yt, Xte, Yte, alpha=cfg.probe_alpha)
                w_gpu = W_gpu[:, 0]
                # cosine of weights
                cos = float("nan")
                if np.linalg.norm(w_sk) > EPS and np.linalg.norm(w_gpu) > EPS:
                    cos = float(np.dot(w_sk, w_gpu) / (np.linalg.norm(w_sk) * np.linalg.norm(w_gpu)))
                rows.append(
                    {
                        "scale_k": k,
                        "sample_id": int(z["sample_id"]),
                        "target": target_names[j],
                        "r2_sklearn": r_sk,
                        "r2_gpu": float(r_gpu[0]),
                        "r2_abs_diff": abs(r_sk - float(r_gpu[0])) if np.isfinite(r_sk) else float("nan"),
                        "weight_cosine": cos,
                        "ok": ok,
                    }
                )
    df = pd.DataFrame(rows)
    max_diff = float(df.r2_abs_diff.max()) if len(df) else float("nan")
    min_cos = float(df.weight_cosine.min()) if len(df) else float("nan")
    ok = bool(len(df) and max_diff < 0.02 and min_cos > 0.98)
    return {
        "ok": ok,
        "n_comparisons": int(len(df)),
        "max_r2_abs_diff": max_diff,
        "min_weight_cosine": min_cos,
        "mean_r2_abs_diff": float(df.r2_abs_diff.mean()) if len(df) else float("nan"),
        "rows": df.to_dict(orient="records"),
    }


# -------------------- summaries --------------------


def summarize_target_scale(df: pd.DataFrame, target: str, k: int, cfg: MultiTargetConfig) -> dict:
    g = df[(df.target == target) & (df.scale_k == k)].copy()
    # join reconstruction_error from frozen curvature if present
    r2 = g.probe_r2.to_numpy(float)
    abn = g.A_B_normal.to_numpy(float)
    abt = g.A_B_total.to_numpy(float)
    apn = g.A_PCA_normal.to_numpy(float)
    apt = g.A_PCA_total.to_numpy(float)
    cw = g.C_w.to_numpy(float)
    log_r = g.log_knn_radius.to_numpy(float)
    labvar = g.local_target_variance.to_numpy(float)
    # reconstruction_error column optional
    if "reconstruction_error" in g.columns:
        recon = g.reconstruction_error.to_numpy(float)
    else:
        recon = np.zeros(len(g))

    def corr(x, y):
        return spearman_dict(x, y)

    raw = {
        "corr_R2_A_B_normal": corr(r2, abn),
        "corr_R2_A_B_total": corr(r2, abt),
        "corr_R2_A_PCA_normal": corr(r2, apn),
        "corr_R2_A_PCA_total": corr(r2, apt),
        "corr_R2_C_w": corr(r2, cw),
    }
    C0 = np.column_stack([log_r, labvar, recon])
    C1 = np.column_stack([log_r, labvar, recon, g.A_T.to_numpy(float), g.A_N.to_numpy(float)])
    C2 = np.column_stack([C1, apn])
    part = {
        "partial_A_B_normal_C0": partial_spearman(abn, r2, C0),
        "partial_A_B_normal_C1": partial_spearman(abn, r2, C1),
        "partial_A_B_normal_C2": partial_spearman(abn, r2, C2),  # main
        "partial_A_PCA_normal_C1_AB": partial_spearman(apn, r2, np.column_stack([C1, abn])),
    }
    # permutation for main partial: shuffle R2
    rng = np.random.default_rng(cfg.seed + 17 * k + hash(target) % 1000)
    nulls = []
    for _ in range(cfg.n_permute):
        r_perm = r2.copy()
        m = np.isfinite(r_perm)
        r_perm[m] = rng.permutation(r_perm[m])
        nulls.append(partial_spearman(abn, r_perm, C2)["rho"])
    p_main, B = mc_p_twosided(part["partial_A_B_normal_C2"]["rho"], np.asarray(nulls))
    ci = bootstrap_spearman_ci(abn, r2, cfg.n_bootstrap, cfg.seed + k)

    # control: A_B_normal vs random/shuffled association difference
    a_rand = g.A_B_random_median.to_numpy(float)
    a_shuf = g.A_B_shuffled.to_numpy(float)

    return {
        "target": target,
        "scale_k": k,
        "n": int(np.isfinite(r2).sum()),
        "mean_probe_r2": float(np.nanmean(r2)),
        "mean_stability": float(np.nanmean(g.probe_dir_stability)),
        "rho_R2_A_B_normal": raw["corr_R2_A_B_normal"]["rho"],
        "p_R2_A_B_normal": raw["corr_R2_A_B_normal"]["pvalue"],
        "ci95_raw_A_B_normal": ci,
        "rho_R2_A_B_total": raw["corr_R2_A_B_total"]["rho"],
        "rho_R2_A_PCA_normal": raw["corr_R2_A_PCA_normal"]["rho"],
        "rho_R2_A_PCA_total": raw["corr_R2_A_PCA_total"]["rho"],
        "rho_R2_C_w": raw["corr_R2_C_w"]["rho"],
        "rho_partial_C0": part["partial_A_B_normal_C0"]["rho"],
        "rho_partial_C1": part["partial_A_B_normal_C1"]["rho"],
        "rho_partial_C2": part["partial_A_B_normal_C2"]["rho"],
        "p_partial_C2": part["partial_A_B_normal_C2"]["pvalue"],
        "p_perm_partial_C2": p_main,
        "B_perm": B,
        "rho_partial_PCA_given_AB": part["partial_A_PCA_normal_C1_AB"]["rho"],
        "rho_R2_A_B_random": corr(r2, a_rand)["rho"],
        "rho_R2_A_B_shuffled": corr(r2, a_shuf)["rho"],
        "mean_delta_Bobs": float(np.nanmean(g.delta_Bobs)),
        "mean_delta_NPCA": float(np.nanmean(g.delta_NPCA)),
        "mean_specificity": float(np.nanmean(g.specificity)),
        "mean_delta_Q": float(np.nanmean(g.delta_Q)),
        "mean_delta_Bpred": float(np.nanmean(g.delta_Bpred)),
        "mean_delta_full": float(np.nanmean(g.delta_full)),
        "frac_delta_Bobs_pos": float(np.mean(g.delta_Bobs.to_numpy(float) > 0)),
        "frac_specificity_pos": float(np.mean(g.specificity.to_numpy(float) > 0)),
    }


def classify_target(s: dict) -> tuple[str, str]:
    if not np.isfinite(s["mean_probe_r2"]) or s["mean_probe_r2"] < 0.05:
        return "not_locally_probeable", "Mean local probe R² too low."
    if s["mean_stability"] < 0.3:
        return "inconclusive", "Probe directions unstable across folds."
    ab = s["rho_partial_C2"]
    ap = s["rho_partial_PCA_given_AB"]
    spec = s["mean_specificity"]
    dB = s["mean_delta_Bobs"]
    dQ = s["mean_delta_Q"]
    # tangent dominated if high A_T correlation path: use mean A via deltas
    if dB < 0.02 and dQ < 0.02 and s["mean_delta_full"] > 0.1:
        # might still be ambient
        pass
    if abs(ab) >= 0.15 and s["p_perm_partial_C2"] <= 0.05 and spec > 0.01 and dB > s["mean_delta_NPCA"]:
        if abs(ab) > abs(s.get("rho_R2_A_PCA_normal", 0)) and s["rho_R2_A_B_normal"] > s.get("rho_R2_A_B_random", -1):
            return (
                "curvature_alignment_specific",
                "Partial A_B_normal predicts R² after PCA controls; ΔBobs>ΔNPCA; beats random/shuffle alignment.",
            )
    if abs(s["rho_R2_A_PCA_normal"]) >= 0.15 and (spec <= 0.01 or abs(ap) >= abs(ab)):
        return "generic_normal_alignment", "Normal-PCA alignment explains probe performance as well or better."
    if dQ >= 0.03 and dB <= dQ + 0.02 and spec <= 0.02:
        return "generic_quadratic_accessibility", "Quadratic tangent features account for gains without curvature specificity."
    if s["mean_delta_full"] > 0.1 and dB < 0.03 and abs(ab) < 0.1:
        return "tangent_dominated", "Tangent-linear probe dominates; curvature alignment weak."
    if abs(ab) >= 0.1 or dB >= 0.05:
        return "target_specific_mixed", "Mixed alignment/ablation signals without full specificity gates."
    return "inconclusive", "No clear alignment or ablation pattern."


# -------------------- main --------------------


def run_multitarget(cfg: MultiTargetConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()
    profile: dict[str, Any] = {"stages": {}}
    device = _torch_device(cfg)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"[multitarget] device={device} gpu={gpu_name}", flush=True)

    scfg = ScreenConfig(curvature_path=cfg.curvature_path, expected_hash=cfg.expected_hash)
    curv = load_frozen_curvature(root, scfg)
    data = load_prepare(resolve_path(root, cfg.prepare_dir))
    X = data["X"].astype(np.float64)
    train = np.asarray(data["train_local"])
    anchors = select_anchors(data, 384)
    sample_ids = data["sample_ids"]

    # labels
    lab = np.load(resolve_path(root, cfg.labels_path))
    y_all = {k: np.asarray(lab[k], dtype=np.float64)[sample_ids] for k in lab.files}
    inv = build_target_inventory(y_all, train, cfg)
    inv.to_csv(out / "target_inventory.csv", index=False)
    targets = inv.loc[inv.included, "target"].tolist()
    if cfg.primary_target not in targets:
        raise RuntimeError(f"primary target {cfg.primary_target} not included")
    print(f"[multitarget] targets={targets}", flush=True)
    y_mat = np.column_stack([y_all[t] for t in targets])

    # attach reconstruction_error from frozen curv (per sample_id,scale)
    recon_map = {
        (int(r.sample_id), int(r.scale_k)): float(r.reconstruction_error)
        for r in curv.itertuples()
    }

    t1 = time.time()
    cache = ensure_geometry_cache(root, cfg, X, train, anchors, data)
    profile["stages"]["geometry_cache_s"] = time.time() - t1

    # parity
    t1 = time.time()
    parity_path = out / "gpu_parity_checks.json"
    if _done(parity_path, cfg.force):
        parity = json.loads(parity_path.read_text())
    else:
        parity = run_parity(root, cfg, X, y_mat, targets, cache, device)
        parity_path.write_text(json.dumps(parity, indent=2))
    profile["stages"]["parity_s"] = time.time() - t1
    print(f"[multitarget] parity ok={parity['ok']} max|ΔR²|={parity['max_r2_abs_diff']}", flush=True)
    if not parity["ok"]:
        raise RuntimeError(f"GPU/CPU parity failed: {parity}")

    # process batches with checkpoint
    t1 = time.time()
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    all_rows = []
    for k in cfg.scales:
        paths = sorted(cache.glob(f"k{k}_ai*.npz"))
        for b0 in range(0, len(paths), cfg.batch_anchors):
            shard = shard_dir / f"k{k}_b{b0:04d}.parquet"
            if _done(shard, cfg.force):
                all_rows.append(pd.read_parquet(shard))
                continue
            batch_rows = []
            for p in paths[b0 : b0 + cfg.batch_anchors]:
                rows = process_anchor(
                    X,
                    y_mat,
                    targets,
                    p,
                    alpha=cfg.probe_alpha,
                    n_folds=cfg.n_folds,
                    seed=cfg.seed + int(np.load(p)["ai"]) + k,
                    device=device,
                )
                for r in rows:
                    r["reconstruction_error"] = recon_map.get((r["sample_id"], r["scale_k"]), float("nan"))
                    r["config_hash"] = cfg.expected_hash
                batch_rows.extend(rows)
            bdf = pd.DataFrame(batch_rows)
            bdf.to_parquet(shard, index=False)
            all_rows.append(bdf)
            print(f"[multitarget] wrote {shard.name} n={len(bdf)} rss={_rss():.0f}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    probe_df = pd.concat(all_rows, ignore_index=True)
    probe_df.to_parquet(out / "anchor_target_probe_results.parquet", index=False)
    profile["stages"]["probe_align_ablate_s"] = time.time() - t1

    # summaries
    t1 = time.time()
    align_rows = []
    for t in targets:
        for k in cfg.scales:
            align_rows.append(summarize_target_scale(probe_df, t, k, cfg))
    align_df = pd.DataFrame(align_rows)
    align_df["p_partial_C2_fdr"] = np.nan
    # BH-FDR on secondary targets at primary k for partial C2
    sec_mask = (align_df.scale_k == cfg.primary_k) & (align_df.target != cfg.primary_target)
    if sec_mask.any():
        align_df.loc[sec_mask, "p_partial_C2_fdr"] = bh_fdr(
            align_df.loc[sec_mask, "p_perm_partial_C2"].to_numpy(dtype=float)
        )
    prim_mask = (align_df.scale_k == cfg.primary_k) & (align_df.target == cfg.primary_target)
    align_df.loc[prim_mask, "p_partial_C2_fdr"] = align_df.loc[prim_mask, "p_perm_partial_C2"].to_numpy()
    align_df.to_parquet(out / "target_alignment_summary.parquet", index=False)
    align_df.to_csv(out / "target_alignment_summary.csv", index=False)

    abl_rows = []
    for t in targets:
        for k in cfg.scales:
            g = probe_df[(probe_df.target == t) & (probe_df.scale_k == k)]
            abl_rows.append(
                {
                    "target": t,
                    "scale_k": k,
                    "mean_delta_Q": float(g.delta_Q.mean()),
                    "mean_delta_Bpred": float(g.delta_Bpred.mean()),
                    "mean_delta_Bobs": float(g.delta_Bobs.mean()),
                    "mean_delta_NPCA": float(g.delta_NPCA.mean()),
                    "mean_delta_full": float(g.delta_full.mean()),
                    "mean_specificity": float(g.specificity.mean()),
                    "frac_spec_pos": float((g.specificity > 0).mean()),
                }
            )
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_parquet(out / "target_ablation_summary.parquet", index=False)

    # classify at primary k
    class_rows = []
    for t in targets:
        s = align_df[(align_df.target == t) & (align_df.scale_k == cfg.primary_k)].iloc[0].to_dict()
        lab, why = classify_target(s)
        fam = inv.loc[inv.target == t, "family"].iloc[0]
        class_rows.append(
            {
                "target": t,
                "family": fam,
                "role": "primary" if t == cfg.primary_target else "secondary",
                "label": lab,
                "reason": why,
                "rho_partial_C2": s["rho_partial_C2"],
                "p_perm_partial_C2": s["p_perm_partial_C2"],
                "p_partial_C2_fdr": s.get("p_partial_C2_fdr", s["p_perm_partial_C2"]),
                "mean_specificity": s["mean_specificity"],
                "mean_probe_r2": s["mean_probe_r2"],
            }
        )
    class_df = pd.DataFrame(class_rows)
    class_df.to_parquet(out / "target_classification.parquet", index=False)
    class_df.to_csv(out / "target_classification.csv", index=False)

    # family summary
    fam_rows = []
    for fam, sub in class_df.groupby("family"):
        rhos = align_df[
            (align_df.scale_k == cfg.primary_k) & (align_df.target.isin(sub.target))
        ]
        meta = fisher_meta(rhos.rho_partial_C2.to_numpy(float), rhos.n.to_numpy(float))
        fam_rows.append(
            {
                "family": fam,
                "n_targets": int(len(sub)),
                "fisher_rho_partial_C2": meta["rho"],
                "n_curvature_specific": int((sub.label == "curvature_alignment_specific").sum()),
                "n_generic_normal": int((sub.label == "generic_normal_alignment").sum()),
                "mean_specificity": float(
                    abl_df[(abl_df.scale_k == cfg.primary_k) & (abl_df.target.isin(sub.target))].mean_specificity.mean()
                ),
            }
        )
    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_parquet(out / "target_family_summary.parquet", index=False)

    # max-statistic permutation across secondary targets at primary k
    sec_targets = [t for t in targets if t != cfg.primary_target]
    maxstat = {"n_secondary": len(sec_targets), "p_maxstat": float("nan"), "obs_max_abs": float("nan")}
    if sec_targets:
        # build per-target arrays aligned by sample_id
        g0 = probe_df[probe_df.scale_k == cfg.primary_k]
        ids = sorted(g0.sample_id.unique())
        r2_mat = np.full((len(ids), len(sec_targets)), np.nan)
        ab_mat = np.full_like(r2_mat, np.nan)
        ctrl_list = []
        for ti, t in enumerate(sec_targets):
            sub = g0[g0.target == t].set_index("sample_id").reindex(ids)
            r2_mat[:, ti] = sub.probe_r2.to_numpy(float)
            ab_mat[:, ti] = sub.A_B_normal.to_numpy(float)
            C1 = np.column_stack(
                [
                    sub.log_knn_radius.to_numpy(float),
                    sub.local_target_variance.to_numpy(float),
                    sub.reconstruction_error.to_numpy(float),
                    sub.A_T.to_numpy(float),
                    sub.A_N.to_numpy(float),
                ]
            )
            C2 = np.column_stack([C1, sub.A_PCA_normal.to_numpy(float)])
            ctrl_list.append(C2)
        obs = []
        for ti in range(len(sec_targets)):
            obs.append(abs(partial_spearman(ab_mat[:, ti], r2_mat[:, ti], ctrl_list[ti])["rho"]))
        obs_max = float(np.nanmax(obs)) if obs else float("nan")
        rng = np.random.default_rng(cfg.seed + 91)
        null_max = []
        for _ in range(cfg.n_permute):
            # joint permutation of R² rows (shared across targets)
            perm = rng.permutation(len(ids))
            mxs = []
            for ti in range(len(sec_targets)):
                mxs.append(abs(partial_spearman(ab_mat[:, ti], r2_mat[perm, ti], ctrl_list[ti])["rho"]))
            null_max.append(float(np.nanmax(mxs)) if mxs else float("nan"))
        p_ms, Bms = mc_p_twosided(obs_max, np.asarray(null_max))
        # one-sided greater for max |ρ|
        nulls = np.asarray(null_max, float)
        nulls = nulls[np.isfinite(nulls)]
        p_ms = float((1 + np.sum(nulls >= obs_max)) / (len(nulls) + 1)) if len(nulls) and np.isfinite(obs_max) else float("nan")
        maxstat = {
            "n_secondary": len(sec_targets),
            "obs_max_abs": obs_max,
            "p_maxstat": p_ms,
            "B_perm": int(len(nulls)),
            "secondary_obs_abs": {t: float(o) for t, o in zip(sec_targets, obs)},
        }
    (out / "maxstat_secondary.json").write_text(json.dumps(maxstat, indent=2))
    profile["stages"]["summaries_s"] = time.time() - t1

    # plots / heatmaps
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    def _heatmap(mat: np.ndarray, row_labels, col_labels, title: str, path: Path, vmin=-0.5, vmax=0.5):
        fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(col_labels), 1.2 + 0.45 * len(row_labels)))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    scales_sorted = list(cfg.scales)
    # raw / partial corr heatmaps (targets × scales)
    for metric, title, fname in [
        ("rho_R2_A_B_normal", "raw corr(R², A_B_normal)", "heatmap_raw_corr_R2_AB.png"),
        ("rho_partial_C2", "partial corr(R², A_B_normal | C2)", "heatmap_partial_corr_R2_AB.png"),
        ("rho_R2_A_PCA_normal", "raw corr(R², A_PCA_normal)", "heatmap_raw_corr_R2_APCA.png"),
        ("mean_stability", "probe-direction stability", "heatmap_probe_stability.png"),
    ]:
        mat = np.full((len(targets), len(scales_sorted)), np.nan)
        for i, t in enumerate(targets):
            for j, k in enumerate(scales_sorted):
                row = align_df[(align_df.target == t) & (align_df.scale_k == k)]
                if len(row):
                    mat[i, j] = float(row.iloc[0][metric])
        vmin, vmax = (0.0, 1.0) if metric == "mean_stability" else (-0.5, 0.5)
        _heatmap(mat, targets, [f"k={k}" for k in scales_sorted], title, fig_dir / fname, vmin, vmax)

    # curvature vs PCA side-by-side at primary k
    prim_k = align_df[align_df.scale_k == cfg.primary_k].set_index("target").reindex(targets)
    mat = np.column_stack([prim_k.rho_R2_A_B_normal.to_numpy(float), prim_k.rho_R2_A_PCA_normal.to_numpy(float)])
    _heatmap(mat, targets, ["A_B_normal", "A_PCA_normal"], f"Alignment vs R² (k={cfg.primary_k})", fig_dir / "heatmap_AB_vs_APCA.png")

    abl_k = abl_df[abl_df.scale_k == cfg.primary_k].set_index("target").reindex(targets)
    mat = np.column_stack(
        [
            abl_k.mean_delta_Bobs.to_numpy(float),
            abl_k.mean_delta_NPCA.to_numpy(float),
            abl_k.mean_specificity.to_numpy(float),
        ]
    )
    _heatmap(
        mat,
        targets,
        ["ΔBobs", "ΔNPCA", "ΔBobs−ΔNPCA"],
        f"Ablation deltas (k={cfg.primary_k})",
        fig_dir / "heatmap_ablation_deltas.png",
        vmin=-0.1,
        vmax=0.4,
    )

    # family × scale Fisher meta of partial C2
    families = sorted(inv.loc[inv.included, "family"].unique())
    mat = np.full((len(families), len(scales_sorted)), np.nan)
    for i, fam in enumerate(families):
        fam_targets = inv.loc[(inv.included) & (inv.family == fam), "target"].tolist()
        for j, k in enumerate(scales_sorted):
            rhos = align_df[(align_df.scale_k == k) & (align_df.target.isin(fam_targets))]
            meta = fisher_meta(rhos.rho_partial_C2.to_numpy(float), rhos.n.to_numpy(float))
            mat[i, j] = meta["rho"]
    _heatmap(mat, families, [f"k={k}" for k in scales_sorted], "Fisher meta partial C2 by family", fig_dir / "heatmap_family_scale.png")

    # mag_r replication check vs prior alignment
    mag = align_df[(align_df.target == "mag_r_desi") & (align_df.scale_k == cfg.primary_k)].iloc[0]
    # prior A_B was A_B_normal
    prior_path = resolve_path(root, "outputs/geometry/physics_curvature_probe_alignment/alignment_associations.csv")
    prior_rho = float("nan")
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        pr = prior[(prior.scale_k == cfg.primary_k) & (prior.feature == "A_B")]
        if len(pr):
            prior_rho = float(pr.iloc[0]["rho"])

    peak_vram = (
        float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
    )
    profile.update(
        {
            "gpu": gpu_name,
            "device": str(device),
            "peak_vram_mb": peak_vram,
            "peak_rss_mb": _rss(),
            "total_seconds": time.time() - t0,
            "n_targets": len(targets),
            "n_anchors": 384,
        }
    )
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    n_sec = int((class_df.role == "secondary").sum())
    n_same_sign = int(
        (
            np.sign(align_df[(align_df.scale_k == cfg.primary_k) & (align_df.target != cfg.primary_target)].rho_partial_C2)
            == np.sign(mag.rho_partial_C2)
        ).sum()
    ) if n_sec else 0
    n_fdr = int(
        (
            (class_df.role == "secondary")
            & (class_df.p_partial_C2_fdr <= 0.05)
            & (class_df.rho_partial_C2.abs() >= 0.1)
        ).sum()
    )

    report = f"""# Multi-target curvature–probe alignment (GPU)

Frozen hash `{cfg.expected_hash}`. Device `{gpu_name}`. Alpha={cfg.probe_alpha}.
No curvature refit / retrieval / Fisher / JS.

## Target inventory

{inv.to_string(index=False)}

## GPU parity

ok={parity['ok']} max|ΔR²|={parity['max_r2_abs_diff']:.4g} min weight cosine={parity['min_weight_cosine']:.4g}

## Confirmatory replication (`mag_r_desi`, k={cfg.primary_k})

- raw corr(R², A_B_normal) = {mag.rho_R2_A_B_normal:.4f} (prior A_B raw ≈ {prior_rho:.4f})
- partial C2 corr = {mag.rho_partial_C2:.4f} (perm p={mag.p_perm_partial_C2:.4g})
- mean specificity ΔBobs−ΔNPCA = {mag.mean_specificity:.4f}
- mean probe R² = {mag.mean_probe_r2:.4f}; stability = {mag.mean_stability:.4f}

## Secondary targets

- same sign of partial C2 as mag_r: {n_same_sign}/{n_sec}
- survive BH-FDR (p_FDR≤0.05, |ρ|≥0.1): {n_fdr}/{n_sec}
- max-statistic |partial C2| across secondaries: obs={maxstat.get('obs_max_abs', float('nan')):.4f}, p={maxstat.get('p_maxstat', float('nan')):.4g}

## Classification (k={cfg.primary_k})

{class_df.to_string(index=False)}

## Family Fisher meta (partial C2; dependent targets noted)

{fam_df.to_string(index=False)}

## Answers

1. mag_r_desi alignment replicates: raw ρ={mag.rho_R2_A_B_normal:.3f} vs prior ~{prior_rho:.3f}; partial C2={mag.rho_partial_C2:.3f}.
2. Secondary same-sign count: {n_same_sign}/{n_sec}.
3. FDR survivors: {n_fdr}/{n_sec}.
4. Beyond normal-PCA: specificity={mag.mean_specificity:.4f}; PCA-controlled partial={mag.rho_partial_C2:.4f}; reverse PCA|AB={mag.rho_partial_PCA_given_AB:.4f}.
5. Generality: see classifications / families — not claimed as universal.
6. Alignment vs ablation agreement: compare tables `target_alignment_summary` and `target_ablation_summary`.
7. Per-target regimes: see classification column.
8. Runtime: {profile['total_seconds']:.1f}s; peak RSS={profile['peak_rss_mb']:.1f} MB; peak VRAM={peak_vram:.1f} MB.

## Exact command

```bash
cd ~/platonic-universe && source .venv/bin/activate && \\
PYTHONPATH=experiments python -m geometry.run_curvature_probe_multitarget \\
  --force --seed 0
```
"""
    (out / "REPORT.md").write_text(report)
    analysis = {
        "mag_r_desi_primary": mag.to_dict(),
        "prior_A_B_raw": prior_rho,
        "n_secondary_same_sign": n_same_sign,
        "n_secondary_fdr": n_fdr,
        "maxstat_secondary": maxstat,
        "classifications": class_df.to_dict(orient="records"),
        "parity_ok": parity["ok"],
        "runtime": profile,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(f"[multitarget] done in {profile['total_seconds']:.1f}s label_mag={class_df.loc[class_df.target=='mag_r_desi','label'].iloc[0]}", flush=True)
    return analysis
