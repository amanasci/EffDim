"""Global (UniverseTBD-style) ridge probe × local curvature alignment.

CORRECT PROTOCOL
----------------
Train one GLOBAL ridge probe per target on the canonical train split, freeze
(w_y, b_y), generate predictions once, and score local geographic performance
of those fixed predictions in frozen anchor neighbourhoods.

Canonical sources reused (recorded in REPORT.md):
  - fit_global_probe  ← SAE-shared-basis/run_physical_probe_field_geometry.py
  - weighted_r2       ← same module (uniform weights ⇒ ordinary local R²;
                        field geometry uses this for global_probe_local_r2)
  - alpha=100         ← frozen from geometry/.../curvature_probe_screen.select_ridge_alpha

Do NOT refit probes at anchors. Do NOT refit in T/B/NPCA subspaces.
Fixed-probe decomposition projects the frozen global w_y onto local bases.
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
from sklearn.linear_model import Ridge

from .confirmatory_object_curvature import select_anchors
from .curvature_probe_alignment import alignment_from_w
from .curvature_probe_screen import (
    EXPECTED_HASH,
    ScreenConfig,
    load_frozen_curvature,
    partial_spearman,
    spearman_dict,
)
from .curvature_probe_subspace_ablation import (
    haar_normal_basis,
    normal_pca_basis,
    rematerialize_chart,
    reproject_into_normal,
)
from .data import load_prepare
from .paths import platonic_root, resolve_path

EPS = 1e-12
CANONICAL_TARGETS = ["mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass", "sfr"]
PRIMARY_TARGET = "mag_r_desi"
PROBE_ALPHA = 100.0  # frozen from curvature_probe_screen.select_ridge_alpha
SCALES = (1024, 2048)

# Canonical helpers — byte-stable copies of
# experiments/SAE-shared-basis/run_physical_probe_field_geometry.py
# (importing that module pulls optional SAE stack; keep the probe API isolated).

CANONICAL_PROBE_TRAIN = (
    "experiments/SAE-shared-basis/run_physical_probe_field_geometry.py::fit_global_probe"
)
CANONICAL_PROBE_LOCAL = (
    "experiments/SAE-shared-basis/run_physical_probe_field_geometry.py::weighted_r2 "
    "(uniform weights; same helper as eval_probes_at_windows → global_probe_local_r2)"
)


def fit_global_probe(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Canonical global Ridge train (run_physical_probe_field_geometry.fit_global_probe)."""
    ridge = Ridge(alpha=float(alpha), fit_intercept=True)
    ridge.fit(X, y)
    return ridge.coef_.astype(np.float64), float(ridge.intercept_)


def weighted_r2(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    """Canonical local score helper (run_physical_probe_field_geometry.weighted_r2)."""
    ww = w / max(w.sum(), EPS)
    ym = float(np.dot(ww, y))
    ss_tot = float(np.dot(ww, (y - ym) ** 2))
    ss_res = float(np.dot(ww, (y - yhat) ** 2))
    if ss_tot < EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


@dataclass
class GlobalProbeAlignConfig:
    output_dir: str = "outputs/geometry/physics_global_probe_curvature_alignment"
    geometry_cache: str = (
        "outputs/geometry/physics_curvature_probe_multitarget/geometry_cache"
    )
    curvature_path: str = (
        "outputs/geometry/physics_quadratic_atlas_sphere_normal/"
        "object_curvature_features_aggregated.parquet"
    )
    prepare_dir: str = "outputs/geometry/physics_activation_atlas_geometry_ablation/prepare"
    labels_path: str = "data_hf/physics/vit_base_test_labels.npz"
    prior_ablation_report: str = (
        "outputs/geometry/physics_curvature_probe_subspace_ablation/REPORT.md"
    )
    expected_hash: str = EXPECTED_HASH
    scales: list[int] = field(default_factory=lambda: list(SCALES))
    primary_k: int = 2048
    primary_target: str = PRIMARY_TARGET
    probe_alpha: float = PROBE_ALPHA
    n_random_bases: int = 20
    n_bootstrap: int = 1000
    n_permute: int = 500
    parity_targets: int = 3
    seed: int = 0
    force: bool = False
    device: str = "cuda"
    min_finite_frac_global: float = 0.5

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


# -------------------- inventory --------------------


def build_target_inventory(y_all: dict[str, np.ndarray], train_idx: np.ndarray, cfg: GlobalProbeAlignConfig) -> pd.DataFrame:
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
                    "family": "other",
                }
            )
            continue
        y = y_all[name]
        fg = float(np.isfinite(y).mean())
        yt = y[train_idx]
        ft = float(np.isfinite(yt).mean())
        std = float(np.nanstd(yt))
        included, reason = True, "ok"
        if fg < cfg.min_finite_frac_global:
            included, reason = False, f"global_finite_frac<{cfg.min_finite_frac_global}"
        elif ft < 0.3:
            included, reason = False, "train_finite_frac<0.3"
        elif not np.isfinite(std) or std < 1e-8:
            included, reason = False, "effectively_constant"
        elif name == "sfr":
            included, reason = False, "sparse_sfr_unsupported_by_local_protocol"
        family = {
            "mag_r_desi": "photometry",
            "photo_z": "redshift",
            "smooth_fraction": "morphology",
            "stellar_mass": "stellar_population",
            "sfr": "stellar_population",
        }.get(name, "other")
        role = "primary" if name == cfg.primary_target and included else (
            "secondary" if included else "excluded"
        )
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


# -------------------- GPU multi-output ridge (match sklearn fit_intercept) --------------------


def ridge_multi_intercept_torch(
    X: torch.Tensor, Y: torch.Tensor, *, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Multi-output Ridge with intercept, matching sklearn.linear_model.Ridge(fit_intercept=True).
    Returns W (f,t), b (t,), ok.
    """
    n, f = X.shape
    t = Y.shape[1]
    x_mean = X.mean(dim=0)
    y_mean = Y.mean(dim=0)
    Xc = X - x_mean
    Yc = Y - y_mean
    XtX = Xc.T @ Xc + alpha * torch.eye(f, device=X.device, dtype=X.dtype)
    XtY = Xc.T @ Yc
    L, info = torch.linalg.cholesky_ex(XtX)
    if int(info.item()) != 0:
        return torch.zeros(f, t, device=X.device, dtype=X.dtype), y_mean, False
    W = torch.cholesky_solve(XtY, L)
    b = y_mean - x_mean @ W
    return W, b, True


def train_global_probes_gpu(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    target_names: list[str],
    *,
    alpha: float,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Group targets by identical train finite masks; solve jointly on GPU."""
    n, f = X_train.shape
    out: dict[str, dict[str, Any]] = {}
    groups: dict[bytes, dict] = {}
    for j, name in enumerate(target_names):
        m = np.isfinite(Y_train[:, j])
        if m.sum() < 32:
            out[name] = {"coef": np.zeros(f), "intercept": 0.0, "ok": False, "n_train": int(m.sum())}
            continue
        key = m.tobytes()
        groups.setdefault(key, {"mask": m, "cols": []})
        groups[key]["cols"].append(j)

    for g in groups.values():
        m = g["mask"]
        cols = g["cols"]
        Xt = torch.tensor(X_train[m], device=device, dtype=torch.float32)
        Yt = torch.tensor(Y_train[m][:, cols], device=device, dtype=torch.float32)
        # pinned-friendly: already on device
        W, b, ok = ridge_multi_intercept_torch(Xt, Yt, alpha=alpha)
        if not ok:
            Xt = Xt.double()
            Yt = Yt.double()
            W, b, ok = ridge_multi_intercept_torch(Xt, Yt, alpha=alpha)
            W = W.float()
            b = b.float()
        Wc = W.detach().cpu().numpy().astype(np.float64)
        bc = b.detach().cpu().numpy().astype(np.float64)
        for li, j in enumerate(cols):
            out[target_names[j]] = {
                "coef": Wc[:, li],
                "intercept": float(bc[li]),
                "ok": bool(ok),
                "n_train": int(m.sum()),
            }
    return out


def train_global_probes_sklearn(
    X_train: np.ndarray, Y_train: np.ndarray, target_names: list[str], *, alpha: float
) -> dict[str, dict[str, Any]]:
    out = {}
    for j, name in enumerate(target_names):
        m = np.isfinite(Y_train[:, j])
        if m.sum() < 32:
            out[name] = {"coef": np.zeros(X_train.shape[1]), "intercept": 0.0, "ok": False, "n_train": int(m.sum())}
            continue
        coef, intercept = fit_global_probe(X_train[m], Y_train[m, j], alpha)
        out[name] = {"coef": coef, "intercept": intercept, "ok": True, "n_train": int(m.sum())}
    return out


def local_r2_fixed_predictions(y: np.ndarray, yhat: np.ndarray) -> float:
    """Geographic local R² of a fixed global probe (uniform-weight weighted_r2)."""
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 4:
        return float("nan")
    w = np.ones(m.sum(), dtype=np.float64)
    return float(weighted_r2(y[m], yhat[m], w))


# -------------------- alignment / decomposition --------------------


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


def fixed_probe_decomposition(
    Xn: np.ndarray,
    y: np.ndarray,
    x_i: np.ndarray,
    w: np.ndarray,
    b: float,
    T: np.ndarray,
    UB: np.ndarray,
    UN: np.ndarray,
) -> dict[str, float]:
    """Decompose fixed global probe locally — no label refits."""
    dx = Xn - x_i[None, :]
    yhat_anchor = float(b + np.dot(w, x_i))
    w_T = T @ (T.T @ w)
    w_B = UB @ (UB.T @ w) if UB.size else np.zeros_like(w)
    w_P = UN @ (UN.T @ w) if UN.size else np.zeros_like(w)
    yhat_T = yhat_anchor + dx @ w_T
    yhat_TB = yhat_anchor + dx @ (w_T + w_B)
    yhat_TPCA = yhat_anchor + dx @ (w_T + w_P)
    yhat_full = b + Xn @ w
    r_full = local_r2_fixed_predictions(y, yhat_full)
    r_T = local_r2_fixed_predictions(y, yhat_T)
    r_TB = local_r2_fixed_predictions(y, yhat_TB)
    r_TPCA = local_r2_fixed_predictions(y, yhat_TPCA)
    return {
        "R2_full": r_full,
        "R2_T": r_T,
        "R2_TB": r_TB,
        "R2_TPCA": r_TPCA,
        "delta_B_fixed": r_TB - r_T if np.isfinite(r_TB) and np.isfinite(r_T) else float("nan"),
        "delta_PCA_fixed": r_TPCA - r_T if np.isfinite(r_TPCA) and np.isfinite(r_T) else float("nan"),
    }


# -------------------- stats --------------------


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


def mc_p_greater(real, nulls):
    nulls = np.asarray(nulls, float)
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls) == 0 or not np.isfinite(real):
        return float("nan"), 0
    return float((1 + np.sum(nulls >= real)) / (len(nulls) + 1)), len(nulls)


def mc_p_twosided(real, nulls):
    nulls = np.asarray(nulls, float)
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls) == 0 or not np.isfinite(real):
        return float("nan"), 0
    return float((1 + np.sum(np.abs(nulls) >= abs(real))) / (len(nulls) + 1)), len(nulls)


def fisher_meta(rhos: np.ndarray, ns: np.ndarray) -> dict:
    m = np.isfinite(rhos) & np.isfinite(ns) & (ns > 3)
    if m.sum() == 0:
        return {"z": float("nan"), "rho": float("nan"), "n_targets": 0}
    z = np.arctanh(np.clip(rhos[m], -0.999, 0.999))
    w = ns[m] - 3.0
    zbar = float(np.sum(w * z) / max(w.sum(), EPS))
    return {"z": zbar, "rho": float(np.tanh(zbar)), "n_targets": int(m.sum())}


# -------------------- geometry cache (reuse / complete) --------------------


def ensure_geometry_cache(
    root: Path,
    cfg: GlobalProbeAlignConfig,
    X: np.ndarray,
    train: np.ndarray,
    anchors: np.ndarray,
    data: dict,
) -> Path:
    """Reuse existing rematerialized neighbourhood packs; complete missing only."""
    cache = resolve_path(root, cfg.geometry_cache)
    cache.mkdir(parents=True, exist_ok=True)
    from sklearn.neighbors import NearestNeighbors

    k_max = max(cfg.scales)
    # Never wipe frozen neighbourhood packs under --force (outputs only).
    need = []
    for k in cfg.scales:
        for ai in range(len(anchors)):
            p = cache / f"k{k}_ai{ai:04d}.npz"
            if not p.exists():
                need.append((k, ai))
    if not need:
        print(f"[global-align] geometry cache complete at {cache}", flush=True)
        return cache

    print(f"[global-align] completing geometry cache: {len(need)} missing packs", flush=True)
    nn = NearestNeighbors(n_neighbors=k_max, metric="euclidean")
    nn.fit(X[train])
    dists, inds = nn.kneighbors(X[anchors])

    # pool of UB for shuffle (from existing + new)
    UB_pool: dict[int, list] = {k: [] for k in cfg.scales}
    for k in cfg.scales:
        for p in sorted(cache.glob(f"k{k}_ai*.npz")):
            try:
                UB_pool[k].append(np.load(p)["UB"])
            except Exception:
                pass

    rng = np.random.default_rng(cfg.seed + 9)
    for k, ai in need:
        a_local = anchors[ai]
        neigh = train[inds[ai, :k]]
        neigh = neigh[neigh != a_local]
        pack = rematerialize_chart(X, neigh, ai, k, cfg.seed)
        if pack is None:
            continue
        dx = X[neigh] - pack["x0u"][None, :]
        UN = normal_pca_basis(dx, pack["x0u"], pack["T"], pack["UB"].shape[1])
        r = pack["UB"].shape[1]
        U_rands = np.stack(
            [haar_normal_basis(X.shape[1], pack["x0u"], pack["T"], r, rng) for _ in range(cfg.n_random_bases)],
            axis=2,
        )
        pool = UB_pool[k] if UB_pool[k] else [pack["UB"]]
        other = pool[rng.integers(0, len(pool))]
        if other.shape[1] >= r:
            Uo = other[:, :r]
        else:
            Uo = np.column_stack(
                [other, haar_normal_basis(X.shape[1], pack["x0u"], pack["T"], r - other.shape[1], rng)]
            )[:, :r]
        U_shuf = reproject_into_normal(Uo, pack["x0u"], pack["T"])
        if U_shuf.shape[1] < r:
            U_shuf = np.column_stack(
                [U_shuf, haar_normal_basis(X.shape[1], pack["x0u"], pack["T"], r - U_shuf.shape[1], rng)]
            )[:, :r]
        path = cache / f"k{k}_ai{ai:04d}.npz"
        np.savez_compressed(
            path,
            sample_id=int(data["sample_ids"][a_local]),
            ai=ai,
            scale_k=k,
            rho=float(dists[ai, k - 1]),
            neigh=neigh,
            T=pack["T"],
            x0u=pack["x0u"],
            UB=pack["UB"],
            UNPCA=UN,
            B0=pack["B0"],
            U_rands=U_rands,
            U_shuf=U_shuf[:, :r],
            B0_fro=pack["B0_fro"],
        )
        UB_pool[k].append(pack["UB"])
        if len(UB_pool[k]) % 64 == 0:
            print(f"[global-align] cache wrote k={k} progress rss={_rss():.0f}", flush=True)
    (cache / "ready.json").write_text(
        json.dumps({"scales": cfg.scales, "n_anchors": len(anchors)}, indent=2)
    )
    return cache


# -------------------- parity --------------------


def run_parity(
    X: np.ndarray,
    y_mat: np.ndarray,
    targets: list[str],
    train: np.ndarray,
    probes_gpu: dict,
    probes_sk: dict,
    cache: Path,
    cfg: GlobalProbeAlignConfig,
) -> dict:
    rows = []
    for name in targets[: cfg.parity_targets]:
        wg, bg = probes_gpu[name]["coef"], probes_gpu[name]["intercept"]
        ws, bs = probes_sk[name]["coef"], probes_sk[name]["intercept"]
        cos = float(
            np.dot(wg, ws) / (np.linalg.norm(wg) * np.linalg.norm(ws) + EPS)
        )
        pred_g = X @ wg + bg
        pred_s = X @ ws + bs
        pred_diff = float(np.nanmean(np.abs(pred_g - pred_s)))
        # local scores on a few packs
        for k in cfg.scales:
            paths = sorted(cache.glob(f"k{k}_ai*.npz"))[:8]
            for p in paths:
                z = np.load(p)
                neigh = z["neigh"]
                y = y_mat[neigh, targets.index(name)]
                r_g = local_r2_fixed_predictions(y, pred_g[neigh])
                r_s = local_r2_fixed_predictions(y, pred_s[neigh])
                rows.append(
                    {
                        "target": name,
                        "scale_k": k,
                        "sample_id": int(z["sample_id"]),
                        "weight_cosine": cos,
                        "intercept_abs_diff": abs(bg - bs),
                        "pred_mae": pred_diff,
                        "local_r2_gpu": r_g,
                        "local_r2_sklearn": r_s,
                        "local_r2_abs_diff": abs(r_g - r_s)
                        if np.isfinite(r_g) and np.isfinite(r_s)
                        else float("nan"),
                    }
                )
    df = pd.DataFrame(rows)
    max_r2 = float(np.nanmax(df.local_r2_abs_diff)) if len(df) else float("nan")
    min_cos = float(df.weight_cosine.min()) if len(df) else float("nan")
    ok = bool(len(df) and max_r2 < 0.02 and min_cos > 0.98)
    return {
        "ok": ok,
        "n_comparisons": int(len(df)),
        "max_local_r2_abs_diff": max_r2,
        "min_weight_cosine": min_cos,
        "mean_local_r2_abs_diff": float(np.nanmean(df.local_r2_abs_diff)) if len(df) else float("nan"),
        "canonical_train_fn": CANONICAL_PROBE_TRAIN,
        "canonical_local_fn": CANONICAL_PROBE_LOCAL,
        "rows": df.to_dict(orient="records"),
    }


# -------------------- per-anchor scoring --------------------


def score_anchor_all_targets(
    pack_path: Path,
    X: np.ndarray,
    y_mat: np.ndarray,
    targets: list[str],
    probes: dict[str, dict],
    yhat_all: np.ndarray,
    recon_map: dict,
    sid_to_local: dict[int, int],
) -> list[dict]:
    try:
        z = np.load(pack_path)
        neigh = z["neigh"]
        T, x0u, UB, UN, B0 = z["T"], z["x0u"], z["UB"], z["UNPCA"], z["B0"]
        rho = float(z["rho"])
        sid = int(z["sample_id"])
        k = int(z["scale_k"])
        U_rands = z["U_rands"]
        U_shuf = z["U_shuf"]
    except Exception as exc:
        print(f"[global-align] skip corrupt pack {pack_path.name}: {exc}", flush=True)
        return []
    # Fixed-probe decomposition uses the true anchor activation x_i (not chart centre).
    x_i = X[sid_to_local[sid]]
    Xn = X[neigh]
    rows = []
    for j, name in enumerate(targets):
        w = probes[name]["coef"]
        b = probes[name]["intercept"]
        y = y_mat[neigh, j]
        yhat = yhat_all[neigh, j]
        r2 = local_r2_fixed_predictions(y, yhat)
        en = projection_energies(w, T, x0u, UB, UN)
        cw = alignment_from_w(w, x0u, T, B0, rho).get("C_w", float("nan"))
        a_rand = [
            projection_energies(w, T, x0u, U_rands[:, :, ri], UN)["A_B_normal"]
            for ri in range(U_rands.shape[2])
        ]
        a_shuf = projection_energies(w, T, x0u, U_shuf, UN)["A_B_normal"]
        decomp = fixed_probe_decomposition(Xn, y, x_i, w, b, T, UB, UN)
        local_var = float(np.nanvar(y)) if np.isfinite(y).sum() > 1 else float("nan")
        rows.append(
            {
                "sample_id": sid,
                "scale_k": k,
                "target": name,
                "local_r2": r2,
                "knn_radius": rho,
                "log_knn_radius": float(np.log(max(rho, EPS))),
                "local_target_variance": local_var,
                "reconstruction_error": recon_map.get((sid, k), float("nan")),
                "config_hash": EXPECTED_HASH,
                **en,
                "C_w": cw,
                "A_B_random_median": float(np.nanmedian(a_rand)) if a_rand else float("nan"),
                "A_B_shuffled": float(a_shuf),
                **decomp,
                "specificity_fixed": (
                    decomp["delta_B_fixed"] - decomp["delta_PCA_fixed"]
                    if np.isfinite(decomp["delta_B_fixed"]) and np.isfinite(decomp["delta_PCA_fixed"])
                    else float("nan")
                ),
            }
        )
    return rows


# -------------------- summaries / classify --------------------


def summarize_target_scale(df: pd.DataFrame, target: str, k: int, cfg: GlobalProbeAlignConfig) -> dict:
    g = df[(df.target == target) & (df.scale_k == k)].copy()
    r2 = g.local_r2.to_numpy(float)
    abn = g.A_B_normal.to_numpy(float)
    abt = g.A_B_total.to_numpy(float)
    apn = g.A_PCA_normal.to_numpy(float)
    apt = g.A_PCA_total.to_numpy(float)
    cw = g.C_w.to_numpy(float)
    log_r = g.log_knn_radius.to_numpy(float)
    labvar = g.local_target_variance.to_numpy(float)
    recon = g.reconstruction_error.to_numpy(float)
    an = g.A_N.to_numpy(float)

    def corr(x, y):
        return spearman_dict(x, y)

    # Primary continuity: C0 = radius (+ existing screen controls optionally)
    C0 = log_r.reshape(-1, 1)
    # Curvature-specific: radius, labvar, recon, A_N, A_PCA_normal
    C_spec = np.column_stack([log_r, labvar, recon, an, apn])
    part_primary = partial_spearman(abn, r2, C0)
    part_spec = partial_spearman(abn, r2, C_spec)
    part_pca = partial_spearman(apn, r2, np.column_stack([log_r, labvar, recon, an, abn]))

    rng = np.random.default_rng(cfg.seed + 17 * k + (hash(target) % 1000))
    nulls = []
    for _ in range(cfg.n_permute):
        r_perm = r2.copy()
        m = np.isfinite(r_perm)
        r_perm[m] = rng.permutation(r_perm[m])
        nulls.append(partial_spearman(abn, r_perm, C_spec)["rho"])
    p_perm, B = mc_p_twosided(part_spec["rho"], np.asarray(nulls))

    return {
        "target": target,
        "scale_k": k,
        "n": int(np.isfinite(r2).sum()),
        "mean_local_r2": float(np.nanmean(r2)),
        "rho_R2_A_B_normal": corr(r2, abn)["rho"],
        "p_R2_A_B_normal": corr(r2, abn)["pvalue"],
        "rho_R2_A_B_total": corr(r2, abt)["rho"],
        "rho_R2_A_PCA_normal": corr(r2, apn)["rho"],
        "rho_R2_A_PCA_total": corr(r2, apt)["rho"],
        "rho_R2_C_w": corr(r2, cw)["rho"],
        "rho_partial_radius": part_primary["rho"],
        "p_partial_radius": part_primary["pvalue"],
        "rho_partial_curv_specific": part_spec["rho"],
        "p_partial_curv_specific": part_spec["pvalue"],
        "p_perm_curv_specific": p_perm,
        "B_perm": B,
        "rho_partial_PCA_given_AB": part_pca["rho"],
        "rho_R2_A_B_random": corr(r2, g.A_B_random_median.to_numpy(float))["rho"],
        "rho_R2_A_B_shuffled": corr(r2, g.A_B_shuffled.to_numpy(float))["rho"],
        "mean_delta_B_fixed": float(np.nanmean(g.delta_B_fixed)),
        "mean_delta_PCA_fixed": float(np.nanmean(g.delta_PCA_fixed)),
        "mean_specificity_fixed": float(np.nanmean(g.specificity_fixed)),
        "mean_R2_full": float(np.nanmean(g.R2_full)),
        "mean_R2_T": float(np.nanmean(g.R2_T)),
        "mean_R2_TB": float(np.nanmean(g.R2_TB)),
        "mean_R2_TPCA": float(np.nanmean(g.R2_TPCA)),
        "mean_A_N": float(np.nanmean(an)),
        "mean_A_B_normal": float(np.nanmean(abn)),
    }


def classify_target(s: dict) -> tuple[str, str]:
    if not np.isfinite(s["mean_local_r2"]) or s["mean_local_r2"] < 0.05:
        return "not_locally_probeable", "Mean local R² of global probe too low."
    ab = s["rho_partial_curv_specific"]
    ap = s["rho_partial_PCA_given_AB"]
    spec = s["mean_specificity_fixed"]
    dB = s["mean_delta_B_fixed"]
    dP = s["mean_delta_PCA_fixed"]
    if (
        abs(ab) >= 0.15
        and s["p_perm_curv_specific"] <= 0.05
        and spec > 0.005
        and dB > dP
        and abs(ab) > abs(s.get("rho_R2_A_PCA_normal", 0)) * 0.8
        and s["rho_R2_A_B_normal"] > s.get("rho_R2_A_B_random", -1)
    ):
        return (
            "curvature_alignment_specific",
            "Fixed-probe A_B_normal predicts local R² after A_N/PCA controls; ΔB_fixed>ΔPCA_fixed.",
        )
    if abs(s["rho_R2_A_PCA_normal"]) >= 0.15 and (spec <= 0.005 or abs(ap) >= abs(ab)):
        return "generic_normal_alignment", "Normal-PCA alignment explains local performance as well or better."
    if s["mean_R2_T"] > 0.8 * s["mean_R2_full"] and abs(ab) < 0.1 and dB < 0.02:
        return "tangent_dominated", "Tangent projection of fixed global probe recovers most local R²."
    if abs(ab) >= 0.1 or dB >= 0.02:
        return "target_specific_mixed", "Mixed fixed-probe alignment/decomposition signals."
    return "inconclusive", "No clear fixed-probe curvature alignment pattern."


def annotate_prior_ablation(root: Path, cfg: GlobalProbeAlignConfig) -> None:
    """Relabel prior local-refit ablation conclusion without mixing tables."""
    path = resolve_path(root, cfg.prior_ablation_report)
    if not path.exists():
        return
    text = path.read_text()
    banner = (
        "\n\n---\n\n"
        "## Protocol note (added by global-probe correction)\n\n"
        "This ablation **refit probes inside local subspaces** at each anchor. "
        "Its scientific label is therefore "
        "`local_subspace_information_capacity` — it measures whether curvature "
        "coordinates carry locally accessible label information, **not** whether "
        "they explain the geographic performance of the fixed UniverseTBD global "
        "ridge probe. See "
        "`outputs/geometry/physics_global_probe_curvature_alignment/` for the "
        "corrected global-probe analysis.\n"
    )
    if "local_subspace_information_capacity" not in text:
        path.write_text(text.rstrip() + banner)


# -------------------- main --------------------


def run_global_probe_alignment(cfg: GlobalProbeAlignConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()
    profile: dict[str, Any] = {"stages": {}}
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"[global-align] device={device} gpu={gpu_name}", flush=True)
    print(f"[global-align] train_fn={CANONICAL_PROBE_TRAIN}", flush=True)
    print(f"[global-align] local_fn={CANONICAL_PROBE_LOCAL}", flush=True)

    scfg = ScreenConfig(curvature_path=cfg.curvature_path, expected_hash=cfg.expected_hash)
    curv = load_frozen_curvature(root, scfg)
    data = load_prepare(resolve_path(root, cfg.prepare_dir))
    X = data["X"].astype(np.float64)
    train = np.asarray(data["train_local"])
    anchors = select_anchors(data, 384)
    sample_ids = data["sample_ids"]

    lab = np.load(resolve_path(root, cfg.labels_path))
    y_all = {k: np.asarray(lab[k], dtype=np.float64)[sample_ids] for k in lab.files}
    inv = build_target_inventory(y_all, train, cfg)
    inv.to_csv(out / "target_inventory.csv", index=False)
    targets = inv.loc[inv.included, "target"].tolist()
    if cfg.primary_target not in targets:
        raise RuntimeError(f"primary {cfg.primary_target} not included")
    print(f"[global-align] targets={targets}", flush=True)
    y_mat = np.column_stack([y_all[t] for t in targets])

    recon_map = {
        (int(r.sample_id), int(r.scale_k)): float(r.reconstruction_error)
        for r in curv.itertuples()
    }

    # geometry cache (reuse / complete — no curvature refit)
    t1 = time.time()
    cache = ensure_geometry_cache(root, cfg, X, train, anchors, data)
    profile["stages"]["geometry_cache_s"] = time.time() - t1

    # train global probes once
    t1 = time.time()
    Xtr, Ytr = X[train], y_mat[train]
    probes_gpu = train_global_probes_gpu(Xtr, Ytr, targets, alpha=cfg.probe_alpha, device=device)
    probes_sk = train_global_probes_sklearn(Xtr, Ytr, targets, alpha=cfg.probe_alpha)
    # freeze predictions on full evaluation population once
    yhat = np.full_like(y_mat, np.nan)
    for j, name in enumerate(targets):
        w = probes_gpu[name]["coef"]
        b = probes_gpu[name]["intercept"]
        yhat[:, j] = X @ w + b
    probe_meta = {
        name: {
            "intercept": probes_gpu[name]["intercept"],
            "n_train": probes_gpu[name]["n_train"],
            "coef_norm": float(np.linalg.norm(probes_gpu[name]["coef"])),
            "alpha": cfg.probe_alpha,
            "ok": probes_gpu[name]["ok"],
        }
        for name in targets
    }
    (out / "global_probes.json").write_text(json.dumps(probe_meta, indent=2))
    np.savez_compressed(
        out / "global_probe_weights.npz",
        targets=np.array(targets),
        **{f"w_{t}": probes_gpu[t]["coef"] for t in targets},
        **{f"b_{t}": np.array([probes_gpu[t]["intercept"]]) for t in targets},
    )
    profile["stages"]["global_probe_train_s"] = time.time() - t1

    # parity
    t1 = time.time()
    parity_path = out / "gpu_parity_checks.json"
    if _done(parity_path, cfg.force):
        parity = json.loads(parity_path.read_text())
    else:
        parity = run_parity(X, y_mat, targets, train, probes_gpu, probes_sk, cache, cfg)
        parity_path.write_text(json.dumps(parity, indent=2))
    profile["stages"]["parity_s"] = time.time() - t1
    print(
        f"[global-align] parity ok={parity['ok']} max|ΔR²|={parity['max_local_r2_abs_diff']}",
        flush=True,
    )
    if not parity["ok"]:
        raise RuntimeError(f"GPU/CPU parity failed: {parity}")

    # score all anchors (all targets share the same frozen w_y per target)
    t1 = time.time()
    sid_to_local = {int(s): i for i, s in enumerate(np.asarray(sample_ids))}
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    all_rows = []
    for k in cfg.scales:
        paths = sorted(cache.glob(f"k{k}_ai*.npz"))
        shard = shard_dir / f"k{k}.parquet"
        if _done(shard, cfg.force):
            all_rows.append(pd.read_parquet(shard))
            continue
        batch = []
        for i, p in enumerate(paths):
            batch.extend(
                score_anchor_all_targets(
                    p, X, y_mat, targets, probes_gpu, yhat, recon_map, sid_to_local
                )
            )
            if (i + 1) % 64 == 0:
                print(f"[global-align] scored k={k} {i+1}/{len(paths)}", flush=True)
        bdf = pd.DataFrame(batch)
        bdf.to_parquet(shard, index=False)
        all_rows.append(bdf)
    probe_df = pd.concat(all_rows, ignore_index=True)
    probe_df.to_parquet(out / "anchor_target_probe_results.parquet", index=False)
    profile["stages"]["score_align_s"] = time.time() - t1

    # summaries
    t1 = time.time()
    align_rows = [summarize_target_scale(probe_df, t, k, cfg) for t in targets for k in cfg.scales]
    align_df = pd.DataFrame(align_rows)
    align_df["p_curv_specific_fdr"] = np.nan
    sec_mask = (align_df.scale_k == cfg.primary_k) & (align_df.target != cfg.primary_target)
    if sec_mask.any():
        align_df.loc[sec_mask, "p_curv_specific_fdr"] = bh_fdr(
            align_df.loc[sec_mask, "p_perm_curv_specific"].to_numpy(float)
        )
    prim_mask = (align_df.scale_k == cfg.primary_k) & (align_df.target == cfg.primary_target)
    align_df.loc[prim_mask, "p_curv_specific_fdr"] = align_df.loc[
        prim_mask, "p_perm_curv_specific"
    ].to_numpy()
    align_df.to_parquet(out / "target_alignment_summary.parquet", index=False)
    align_df.to_csv(out / "target_alignment_summary.csv", index=False)

    decomp_rows = []
    for t in targets:
        for k in cfg.scales:
            g = probe_df[(probe_df.target == t) & (probe_df.scale_k == k)]
            decomp_rows.append(
                {
                    "target": t,
                    "scale_k": k,
                    "mean_R2_full": float(g.R2_full.mean()),
                    "mean_R2_T": float(g.R2_T.mean()),
                    "mean_R2_TB": float(g.R2_TB.mean()),
                    "mean_R2_TPCA": float(g.R2_TPCA.mean()),
                    "mean_delta_B_fixed": float(g.delta_B_fixed.mean()),
                    "mean_delta_PCA_fixed": float(g.delta_PCA_fixed.mean()),
                    "mean_specificity_fixed": float(g.specificity_fixed.mean()),
                    "frac_delta_B_pos": float((g.delta_B_fixed > 0).mean()),
                }
            )
    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df.to_parquet(out / "target_fixed_decomp_summary.parquet", index=False)

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
                "rho_partial_curv_specific": s["rho_partial_curv_specific"],
                "p_perm_curv_specific": s["p_perm_curv_specific"],
                "p_curv_specific_fdr": s.get("p_curv_specific_fdr", s["p_perm_curv_specific"]),
                "mean_specificity_fixed": s["mean_specificity_fixed"],
                "mean_local_r2": s["mean_local_r2"],
            }
        )
    class_df = pd.DataFrame(class_rows)
    class_df.to_parquet(out / "target_classification.parquet", index=False)
    class_df.to_csv(out / "target_classification.csv", index=False)

    fam_rows = []
    for fam, sub in class_df.groupby("family"):
        rhos = align_df[
            (align_df.scale_k == cfg.primary_k) & (align_df.target.isin(sub.target))
        ]
        meta = fisher_meta(rhos.rho_partial_curv_specific.to_numpy(float), rhos.n.to_numpy(float))
        fam_rows.append(
            {
                "family": fam,
                "n_targets": int(len(sub)),
                "fisher_rho_partial_curv_specific": meta["rho"],
                "n_curvature_specific": int((sub.label == "curvature_alignment_specific").sum()),
                "n_generic_normal": int((sub.label == "generic_normal_alignment").sum()),
                "mean_specificity_fixed": float(
                    decomp_df[
                        (decomp_df.scale_k == cfg.primary_k) & (decomp_df.target.isin(sub.target))
                    ].mean_specificity_fixed.mean()
                ),
            }
        )
    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_parquet(out / "target_family_summary.parquet", index=False)

    # max-stat secondary
    sec_targets = [t for t in targets if t != cfg.primary_target]
    maxstat = {"n_secondary": len(sec_targets)}
    if sec_targets:
        g0 = probe_df[probe_df.scale_k == cfg.primary_k]
        ids = sorted(g0.sample_id.unique())
        obs = []
        mats = []
        for t in sec_targets:
            sub = g0[g0.target == t].set_index("sample_id").reindex(ids)
            r2 = sub.local_r2.to_numpy(float)
            ab = sub.A_B_normal.to_numpy(float)
            C = np.column_stack(
                [
                    sub.log_knn_radius.to_numpy(float),
                    sub.local_target_variance.to_numpy(float),
                    sub.reconstruction_error.to_numpy(float),
                    sub.A_N.to_numpy(float),
                    sub.A_PCA_normal.to_numpy(float),
                ]
            )
            obs.append(abs(partial_spearman(ab, r2, C)["rho"]))
            mats.append((ab, r2, C))
        obs_max = float(np.nanmax(obs))
        rng = np.random.default_rng(cfg.seed + 91)
        null_max = []
        for _ in range(cfg.n_permute):
            perm = rng.permutation(len(ids))
            mxs = [
                abs(partial_spearman(ab, r2[perm], C)["rho"]) for ab, r2, C in mats
            ]
            null_max.append(float(np.nanmax(mxs)))
        p_ms, Bms = mc_p_greater(obs_max, np.asarray(null_max))
        maxstat.update(
            {
                "obs_max_abs": obs_max,
                "p_maxstat": p_ms,
                "B_perm": Bms,
                "secondary_obs_abs": {t: float(o) for t, o in zip(sec_targets, obs)},
            }
        )
    (out / "maxstat_secondary.json").write_text(json.dumps(maxstat, indent=2))
    profile["stages"]["summaries_s"] = time.time() - t1

    # plots
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    def _heatmap(mat, row_labels, col_labels, title, path, vmin=-0.5, vmax=0.5):
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

    for metric, title, fname, vr in [
        ("rho_R2_A_B_normal", "raw corr(local R², A_B_normal)", "heatmap_raw_corr_R2_AB.png", (-0.5, 0.5)),
        ("rho_partial_curv_specific", "partial corr | C_spec", "heatmap_partial_corr_R2_AB.png", (-0.5, 0.5)),
        ("rho_R2_A_PCA_normal", "raw corr(local R², A_PCA_normal)", "heatmap_raw_corr_R2_APCA.png", (-0.5, 0.5)),
    ]:
        mat = np.full((len(targets), len(cfg.scales)), np.nan)
        for i, t in enumerate(targets):
            for j, k in enumerate(cfg.scales):
                row = align_df[(align_df.target == t) & (align_df.scale_k == k)]
                if len(row):
                    mat[i, j] = float(row.iloc[0][metric])
        _heatmap(mat, targets, [f"k={k}" for k in cfg.scales], title, fig_dir / fname, *vr)

    prim_k = align_df[align_df.scale_k == cfg.primary_k].set_index("target").reindex(targets)
    mat = np.column_stack(
        [prim_k.rho_R2_A_B_normal.to_numpy(float), prim_k.rho_R2_A_PCA_normal.to_numpy(float)]
    )
    _heatmap(mat, targets, ["A_B_normal", "A_PCA_normal"], f"Alignment vs local R² (k={cfg.primary_k})", fig_dir / "heatmap_AB_vs_APCA.png")

    dek = decomp_df[decomp_df.scale_k == cfg.primary_k].set_index("target").reindex(targets)
    mat = np.column_stack(
        [
            dek.mean_delta_B_fixed.to_numpy(float),
            dek.mean_delta_PCA_fixed.to_numpy(float),
            dek.mean_specificity_fixed.to_numpy(float),
        ]
    )
    _heatmap(
        mat,
        targets,
        ["ΔB_fixed", "ΔPCA_fixed", "ΔB−ΔPCA"],
        f"Fixed-probe decomposition (k={cfg.primary_k})",
        fig_dir / "heatmap_fixed_decomp.png",
        -0.05,
        0.15,
    )

    families = sorted(inv.loc[inv.included, "family"].unique())
    mat = np.full((len(families), len(cfg.scales)), np.nan)
    for i, fam in enumerate(families):
        fam_t = inv.loc[(inv.included) & (inv.family == fam), "target"].tolist()
        for j, k in enumerate(cfg.scales):
            rhos = align_df[(align_df.scale_k == k) & (align_df.target.isin(fam_t))]
            meta = fisher_meta(rhos.rho_partial_curv_specific.to_numpy(float), rhos.n.to_numpy(float))
            mat[i, j] = meta["rho"]
    _heatmap(mat, families, [f"k={k}" for k in cfg.scales], "Fisher meta partial (family×scale)", fig_dir / "heatmap_family_scale.png")

    annotate_prior_ablation(root, cfg)

    peak_vram = float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
    profile.update(
        {
            "gpu": gpu_name,
            "device": str(device),
            "peak_vram_mb": peak_vram,
            "peak_rss_mb": _rss(),
            "total_seconds": time.time() - t0,
            "n_targets": len(targets),
            "n_anchors": 384,
            "canonical_train_fn": CANONICAL_PROBE_TRAIN,
            "canonical_local_fn": CANONICAL_PROBE_LOCAL,
            "probe_alpha": cfg.probe_alpha,
        }
    )
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    mag = align_df[(align_df.target == "mag_r_desi") & (align_df.scale_k == cfg.primary_k)].iloc[0]
    n_sec = int((class_df.role == "secondary").sum())
    n_same_sign = (
        int(
            (
                np.sign(
                    align_df[
                        (align_df.scale_k == cfg.primary_k) & (align_df.target != cfg.primary_target)
                    ].rho_partial_curv_specific
                )
                == np.sign(mag.rho_partial_curv_specific)
            ).sum()
        )
        if n_sec
        else 0
    )
    n_fdr = int(
        (
            (class_df.role == "secondary")
            & (class_df.p_curv_specific_fdr <= 0.05)
            & (class_df.rho_partial_curv_specific.abs() >= 0.1)
        ).sum()
    )

    report = f"""# Global-probe × curvature alignment (CORRECTED PROTOCOL)

Frozen hash `{cfg.expected_hash}`. Device `{gpu_name}`. Alpha=`{cfg.probe_alpha}`
(frozen from `curvature_probe_screen.select_ridge_alpha`).

**No local probe refits. No curvature / atlas / kNN estimator changes.**

## Canonical functions

| Role | Source |
|------|--------|
| Global ridge train | `{CANONICAL_PROBE_TRAIN}` |
| Local performance of fixed predictions | `{CANONICAL_PROBE_LOCAL}` |
| Alpha selection (frozen) | `geometry.physics_activation_atlas.curvature_probe_screen.select_ridge_alpha` → 100 |

One global probe `(w_y, b_y)` is trained per target on `train_local`.
Predictions `yhat = X @ w_y + b_y` are generated once for the evaluation population.
At each frozen anchor / scale, local R² uses the same fixed predictions
(uniform-weight `weighted_r2` on the geographic neighbourhood).

## Target inventory

{inv.to_string(index=False)}

## GPU parity (vs sklearn `fit_global_probe`)

ok={parity['ok']} max|Δ local R²|={parity['max_local_r2_abs_diff']:.4g}
min weight cosine={parity['min_weight_cosine']:.4g}

## Confirmatory (`mag_r_desi`, k={cfg.primary_k})

- raw corr(local R², A_B_normal) = {mag.rho_R2_A_B_normal:.4f}
- partial | radius = {mag.rho_partial_radius:.4f}
- curvature-specific partial | (radius, labvar, recon, A_N, A_PCA_normal) = {mag.rho_partial_curv_specific:.4f}
  (perm p={mag.p_perm_curv_specific:.4g})
- mean ΔB_fixed = {mag.mean_delta_B_fixed:.4f}; ΔPCA_fixed = {mag.mean_delta_PCA_fixed:.4f};
  specificity = {mag.mean_specificity_fixed:.4f}
- mean local R² (fixed global) = {mag.mean_local_r2:.4f}

## Secondary targets

- same sign of curv-specific partial as mag_r: {n_same_sign}/{n_sec}
- survive BH-FDR (p_FDR≤0.05, |ρ|≥0.1): {n_fdr}/{n_sec}
- max-statistic |partial|: obs={maxstat.get('obs_max_abs', float('nan')):.4f}, p={maxstat.get('p_maxstat', float('nan')):.4g}

## Classification (k={cfg.primary_k})

{class_df.to_string(index=False)}

## Family Fisher meta (dependent targets noted)

{fam_df.to_string(index=False)}

## Fixed-probe decomposition (not local refits)

{decomp_df[decomp_df.scale_k == cfg.primary_k].to_string(index=False)}

## Protocol separation

The earlier subspace ablation at
`outputs/geometry/physics_curvature_probe_subspace_ablation/` used **per-anchor
refits**. Its conclusion is relabelled
`local_subspace_information_capacity` and must **not** be read as explaining
geographic performance of the fixed global probe. Tables here contain only
fixed-global-probe quantities.

## Geometric reading

Weak regions are locations where the local tangent / curvature subspace rotates
away from the fixed global direction `w_y`. The global probe cannot adapt its
orientation from patch to patch.

## Answers

1. mag_r_desi curv-specific partial ρ={mag.rho_partial_curv_specific:.3f} (perm p={mag.p_perm_curv_specific:.4g}).
2. Secondary same-sign: {n_same_sign}/{n_sec}.
3. FDR survivors: {n_fdr}/{n_sec}.
4. Beyond normal-PCA: specificity_fixed={mag.mean_specificity_fixed:.4f}; reverse PCA|AB={mag.rho_partial_PCA_given_AB:.4f}.
5. Generality: see classifications / families.
6. Alignment vs fixed decomposition: see `target_alignment_summary` vs `target_fixed_decomp_summary`.
7. Per-target regimes: classification column.
8. Runtime: {profile['total_seconds']:.1f}s; peak RSS={profile['peak_rss_mb']:.1f} MB; peak VRAM={peak_vram:.1f} MB.

## Exact command

```bash
cd ~/platonic-universe && source .venv/bin/activate && \\
PYTHONPATH=experiments python -m geometry.run_global_probe_curvature_alignment \\
  --force --seed 0
```
"""
    (out / "REPORT.md").write_text(report)
    analysis = {
        "mag_r_desi_primary": mag.to_dict(),
        "n_secondary_same_sign": n_same_sign,
        "n_secondary_fdr": n_fdr,
        "maxstat_secondary": maxstat,
        "classifications": class_df.to_dict(orient="records"),
        "parity_ok": parity["ok"],
        "runtime": profile,
        "canonical_functions": {
            "train": CANONICAL_PROBE_TRAIN,
            "local_score": CANONICAL_PROBE_LOCAL,
            "alpha_source": "curvature_probe_screen.select_ridge_alpha → 100",
        },
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(
        f"[global-align] done in {profile['total_seconds']:.1f}s "
        f"label_mag={class_df.loc[class_df.target=='mag_r_desi','label'].iloc[0]}",
        flush=True,
    )
    return analysis
