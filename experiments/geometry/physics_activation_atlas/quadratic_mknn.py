"""Per-patch quadratic flattening vs tangent mKNN (Physics multi-model).

Reuses NestedChart / fit_nested_chart from sphere_normal_quadratic.py without
redesigning the geometry model. Maps paper names:

  Q_T  ↔ NestedChart.A_flat (tangential warp)
  Q_R  ↔ forced sphere radial via Normalize(x0 + J u)  (decode_R)
  B^S  ↔ NestedChart.BS_flat (sphere-normal bending)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from .confirmatory_object_curvature import _fit_neighborhood, decompose_BS
from .multimodel_graph_prior_quadratic import (
    ISOMAP_KEY,
    MODELS,
    energy_rank,
    l2_normalize,
    oof_quad_errors_multi,
)
from .paths import platonic_root, resolve_path
from .quadratic import quadratic_features
from .sphere_normal_quadratic import NestedChart, normalize_rows, sphere_project_basis

EPS = 1e-12

PRIMARY_METRICS = (
    "ambient_l2",
    "ambient_cosine",
    "tangent",
    "quadratic_flat_full",
    "local_mahalanobis",
    "shuffled_Q",
)

PHASE2_METRICS = (
    "radial_only",
    "tangential_warp_only",
    "sphere_normal_only",
    "tangent_plus_radial",
    "tangent_plus_tangential",
    "tangent_plus_sphere_normal",
    "tangent_plus_tangential_plus_radial",
    "full_quadratic",
    "matched_random_normal_Q",
    "quadratic_geo_full",
)


SAE_PATHS = {
    "vit_base": ("outputs/sae/vit_base_test/vit_base_galaxies", "vit_base_galaxies"),
    "dinov3": (
        "outputs/sae/dinov3_vitb16_test/dinov3_vitb16_galaxies",
        "dinov3_vitb16_galaxies",
    ),
    "clip_base": ("outputs/sae/clip_base_test/clip_base_galaxies", "clip_base_galaxies"),
}
SAE_TAG_PREFER = (
    "F2048_k20_seed0",
    "F2048_k22_seed0",
    "F2048_k19_seed0",
    "F2048_k64_seed0",
    "F2048_k32_seed0",
)


@dataclass
class QuadraticMKNNConfig:
    output_dir: str = "outputs/geometry/quadratic_mknn/smoke"
    selection_path: str = (
        "outputs/sae_shared_basis/bsf_block_vae_fisher_physics/selection.npz"
    )
    isomap_dims_path: str = (
        "outputs/sae_shared_basis/pipeline_isomap_sae_shared_mknn_physics_holdout20/"
        "isomap_dims.json"
    )
    models: list[str] = field(
        default_factory=lambda: ["vit_base", "dinov3", "clip_base"]
    )
    # Representation for charts / distances:
    #   dense     — L2-normalized embeddings (original experiment)
    #   sae       — TopK SAE codes, then L2-normalized
    #   sae_idf   — SAE codes × train IDF, then L2-normalized
    space: str = "dense"
    sae_tag: str = ""  # empty → first available from SAE_TAG_PREFER
    chart_scales_K: list[int] = field(
        default_factory=lambda: [256, 512, 1024, 2048]
    )
    retrieval_ks: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    n_anchors: int = 96
    seed: int = 0
    device: str = "cuda"
    candidate_multipliers: list[int] = field(default_factory=lambda: [4, 10, 20])
    candidate_pool_min: int = 512
    primary_multiplier: int = 20
    invert_lambda: float = 1e-2
    invert_iters: int = 8
    dim_screen_anchors: int = 24
    patch_mode: str = "model_specific"  # or "shared"
    shared_reference_model: str = "vit_base"
    phase2: bool = False
    force: bool = False
    row_batch: int = 256

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


# -------------------- data --------------------


def _load_col(path: Path, col: str) -> np.ndarray:
    from topology.physics_activation_density_ph.paths import load_col

    return load_col(path, col, l2=False).astype(np.float32)


def _resolve_sae_dir(root: Path, model: str, tag: str) -> Path:
    if model not in SAE_PATHS:
        raise KeyError(f"no SAE path mapping for model={model}")
    base_rel, _ = SAE_PATHS[model]
    base = resolve_path(root, base_rel)
    if tag:
        p = base / tag
        if not (p / "model.pt").is_file():
            raise FileNotFoundError(p)
        return p
    for t in SAE_TAG_PREFER:
        p = base / t
        if (p / "model.pt").is_file():
            return p
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / "model.pt").is_file():
            return p
    raise FileNotFoundError(f"no SAE checkpoint under {base}")


def _idf_np(C: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    df = (C > 0).sum(axis=0).astype(np.float64)
    return (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)


def _encode_sae_codes(
    root: Path,
    cfg: QuadraticMKNNConfig,
    dense_by: dict[str, np.ndarray],
    train_local: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Encode dense activations → TopK SAE codes; optional IDF; L2-normalize."""
    import sys

    sae_dir = Path(__file__).resolve().parents[2] / "SAE-shared-basis"
    if str(sae_dir) not in sys.path:
        sys.path.insert(0, str(sae_dir))
    from _common import ensure_sae_import  # noqa: E402

    ensure_sae_import()
    # Prefer functions from affine runner (vendored TopKSAE load/encode)
    from sae_affine_basis_mknn_gpu import encode, load_sae  # noqa: E402

    codes_by: dict[str, np.ndarray] = {}
    tags: dict[str, str] = {}
    for m, X in dense_by.items():
        sae_path = _resolve_sae_dir(root, m, cfg.sae_tag)
        tags[m] = sae_path.name
        print(f"[quadratic_mknn] encoding {m} with {sae_path}", flush=True)
        bundle = load_sae(sae_path, device)
        C = encode(bundle, X, device)
        if cfg.space == "sae_idf":
            idf = _idf_np(C[train_local])
            C = C * idf[None, :]
        codes_by[m] = l2_normalize(C)
    return codes_by, tags


def _sae_dim_prior(X: np.ndarray, seed: int = 0) -> int:
    """Cheap energy-rank prior on a code subsample (post L2-normalize)."""
    rng = np.random.default_rng(seed)
    n = min(2048, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    Xc = X[idx] - X[idx].mean(0)
    # sketch via random projection if F is large
    F = Xc.shape[1]
    if F > 256:
        R = rng.normal(size=(F, 256)).astype(np.float64)
        R /= np.linalg.norm(R, axis=0, keepdims=True) + EPS
        Y = Xc.astype(np.float64) @ R
    else:
        Y = Xc.astype(np.float64)
    try:
        _, s, _ = np.linalg.svd(Y, full_matrices=False)
    except np.linalg.LinAlgError:
        return 8
    return int(np.clip(energy_rank(s**2, 0.9), 4, 24))


def load_physics_bundle(root: Path, cfg: QuadraticMKNNConfig) -> dict[str, Any]:
    sel = np.load(resolve_path(root, cfg.selection_path))
    selected = np.asarray(sel["selected"], dtype=np.int64)
    train_local = np.asarray(sel["train_idx"], dtype=np.int64)
    test_local = np.asarray(sel["test_idx"], dtype=np.int64)
    assert len(selected) == 16384
    assert set(train_local).isdisjoint(set(test_local))

    raw_by: dict[str, np.ndarray] = {}
    for m in cfg.models:
        pq_rel, col = MODELS[m]
        X_all = _load_col(resolve_path(root, pq_rel), col)
        raw_by[m] = X_all[selected].astype(np.float32)

    device = torch.device(
        cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    sae_tags: dict[str, str] = {}
    if cfg.space in ("sae", "sae_idf"):
        # SAE scalers expect raw (pre-L2) activations
        X_by, sae_tags = _encode_sae_codes(root, cfg, raw_by, train_local, device)
        dense_by = {m: l2_normalize(X) for m, X in raw_by.items()}
    elif cfg.space == "dense":
        dense_by = {m: l2_normalize(X) for m, X in raw_by.items()}
        X_by = dense_by
    else:
        raise ValueError(f"unknown space={cfg.space!r}; expected dense|sae|sae_idf")

    d_graph: dict[str, int] = {}
    if cfg.space == "dense":
        iso_path = resolve_path(root, cfg.isomap_dims_path)
        if iso_path.is_file():
            iso = json.loads(iso_path.read_text())
            for m in cfg.models:
                key = ISOMAP_KEY[m]
                block = iso.get(key, {})
                d_graph[m] = int(
                    block.get("d_residual_elbow")
                    or block.get("d_primary")
                    or 8
                )
        else:
            d_graph = {m: 8 for m in cfg.models}
    else:
        for m in cfg.models:
            d_graph[m] = _sae_dim_prior(X_by[m], seed=cfg.seed)

    # Deterministic anchors from test_local (sorted sample_ids, take first n)
    order = np.argsort(selected[test_local], kind="mergesort")
    anchors = test_local[order[: cfg.n_anchors]]

    return {
        "selected": selected,
        "train_local": train_local,
        "test_local": test_local,
        "X_by": X_by,
        "dense_by": dense_by,
        "d_graph": d_graph,
        "anchors": np.asarray(anchors, dtype=np.int64),
        "sample_ids": selected,
        "space": cfg.space,
        "sae_tags": sae_tags,
    }


# -------------------- knn helpers --------------------


@torch.inference_mode()
def knn_indices(
    X: np.ndarray,
    queries: np.ndarray,
    k: int,
    device: torch.device,
    row_batch: int = 256,
) -> np.ndarray:
    """Cosine/IP knn on L2-normalized rows. queries are rows of X (indices)."""
    Z = torch.as_tensor(X, device=device, dtype=torch.float32)
    Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    qidx = torch.as_tensor(queries, device=device, dtype=torch.long)
    n = Z.shape[0]
    k = min(k, n - 1)
    out = torch.empty(len(queries), k, device=device, dtype=torch.long)
    for s in range(0, len(queries), row_batch):
        e = min(len(queries), s + row_batch)
        qi = qidx[s:e]
        sim = Z[qi] @ Z.T
        sim[torch.arange(e - s, device=device), qi] = -torch.inf
        out[s:e] = torch.topk(sim, k=k, dim=1).indices
    return out.cpu().numpy()


@torch.inference_mode()
def knn_in_pool(
    X: np.ndarray,
    query_idx: int,
    pool_idx: np.ndarray,
    k: int,
    device: torch.device,
) -> np.ndarray:
    """Top-k among pool_idx for a single query (by cosine)."""
    Z = torch.as_tensor(X, device=device, dtype=torch.float32)
    Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    pool = torch.as_tensor(pool_idx, device=device, dtype=torch.long)
    sim = (Z[query_idx] @ Z[pool].T).float()
    kk = min(k, len(pool_idx))
    top = torch.topk(sim, k=kk).indices
    return pool_idx[top.cpu().numpy()]


def mknn_from_sets(nn_a: np.ndarray, nn_b: np.ndarray, k: int) -> float:
    return float(
        np.mean([len(set(nn_a[i, :k]) & set(nn_b[i, :k])) / k for i in range(len(nn_a))])
    )


def paired_bootstrap_ci(
    diffs: np.ndarray, n_boot: int = 400, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    diffs = np.asarray(diffs, dtype=np.float64)
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(diffs)
    for b in range(n_boot):
        means[b] = diffs[rng.integers(0, n, size=n)].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(diffs.mean()), float(lo), float(hi)


# -------------------- dimension selection --------------------


def candidate_dims(d_g: int, d_max: int = 24) -> list[int]:
    base = [d_g - 4, d_g - 2, d_g, d_g + 2, d_g + 4, 4, 6, max(8, d_g // 2)]
    out = sorted({int(np.clip(d, 2, d_max)) for d in base})
    return out


def select_d_one_se(errors: dict[int, dict]) -> int:
    """Smallest d with E_quad within 1 SE of best (over OOF folds if present)."""
    ok = {d: v for d, v in errors.items() if v.get("ok") and np.isfinite(v.get("E_quadratic", np.nan))}
    if not ok:
        return min(errors.keys()) if errors else 8
    best_d = min(ok, key=lambda d: ok[d]["E_quadratic"])
    best_e = ok[best_d]["E_quadratic"]
    # approximate SE from neff if available
    se = float(ok[best_d].get("se_quadratic", 0.0) or 0.0)
    if se <= 0:
        # fallback: 5% relative band
        se = 0.05 * abs(best_e) + 1e-6
    thresh = best_e + se
    eligible = [d for d, v in ok.items() if v["E_quadratic"] <= thresh]
    return int(min(eligible))


def screen_common_dimension(
    X: np.ndarray,
    train_local: np.ndarray,
    anchors: np.ndarray,
    K: int,
    d_g: int,
    device: torch.device,
    cfg: QuadraticMKNNConfig,
) -> tuple[int, pd.DataFrame]:
    """Estimate d_M*(K) from a screen of anchors via OOF quadratic recon."""
    screen = anchors[: min(cfg.dim_screen_anchors, len(anchors))]
    # neighbourhood indices into train
    nn = knn_indices(X, screen, K, device, cfg.row_batch)
    # map returned global indices — knn over full X; keep only train members
    train_set = set(int(i) for i in train_local)
    rows = []
    votes: list[int] = []
    dims = candidate_dims(d_g)
    for ai, a in enumerate(screen):
        neigh = [int(j) for j in nn[ai] if int(j) in train_set]
        if len(neigh) < K:
            # top-up: take train knn only
            nn_tr = knn_indices(X[train_local], np.array([a]), min(K, len(train_local) - 1), device, cfg.row_batch)
            # nn_tr indices into train_local array
            neigh = [int(train_local[j]) for j in nn_tr[0, :K]]
        neigh = np.asarray(neigh[:K], dtype=np.int64)
        Xn = X[neigh].astype(np.float64)
        errs = oof_quad_errors_multi(Xn, dims, None, n_folds=5, seed=cfg.seed + ai)
        d_star = select_d_one_se(errs)
        votes.append(d_star)
        for d, v in errs.items():
            rows.append(
                {
                    "anchor_local": int(a),
                    "K": K,
                    "d": d,
                    "E_linear": v.get("E_linear"),
                    "E_quadratic": v.get("E_quadratic"),
                    "quadratic_gain": v.get("quadratic_gain"),
                    "ok": v.get("ok"),
                    "selected": d == d_star,
                }
            )
    d_common = int(np.median(votes)) if votes else int(d_g)
    return d_common, pd.DataFrame(rows)


# -------------------- inverse coordinate / distances --------------------


def numerical_jacobian(
    decode: Callable[[np.ndarray], np.ndarray], u: np.ndarray, eps: float = 1e-4
) -> np.ndarray:
    d = u.shape[0]
    y0 = decode(u[None, :])[0]
    J = np.zeros((y0.shape[0], d), dtype=np.float64)
    for a in range(d):
        du = np.zeros(d)
        du[a] = eps
        J[:, a] = (decode((u + du)[None, :])[0] - decode((u - du)[None, :])[0]) / (
            2 * eps
        )
    return J


def invert_coordinates(
    decode: Callable[[np.ndarray], np.ndarray],
    xj: np.ndarray,
    u0: np.ndarray,
    *,
    lam: float,
    n_iter: int,
) -> dict[str, Any]:
    """Single-point wrapper around ``invert_coordinates_batch``."""
    out = invert_coordinates_batch(
        decode, xj[None, :], u0[None, :], lam=lam, n_iter=n_iter
    )
    return {
        "u": out["U"][0],
        "ok": bool(out["ok"][0]),
        "iters": int(out["iters"]),
        "residual": float(out["residual"][0]),
        "du": float(out["du"][0]),
    }


def invert_coordinates_batch(
    decode: Callable[[np.ndarray], np.ndarray],
    Xc: np.ndarray,
    U0: np.ndarray,
    *,
    lam: float,
    n_iter: int,
    eps: float = 1e-4,
) -> dict[str, Any]:
    """Batch Gauss–Newton inverse coords for many candidates."""
    U = np.asarray(U0, dtype=np.float64).copy()
    Xc = np.asarray(Xc, dtype=np.float64)
    U0 = np.asarray(U0, dtype=np.float64)
    n, d = U.shape
    ok = np.ones(n, dtype=bool)
    eye = np.eye(d, dtype=np.float64)
    for _ in range(n_iter):
        Y = decode(U)
        R = Xc - Y
        # numerical Jacobian (n, D, d)
        Jf = np.empty((n, Y.shape[1], d), dtype=np.float64)
        for a in range(d):
            du = np.zeros_like(U)
            du[:, a] = eps
            Jf[:, :, a] = (decode(U + du) - decode(U - du)) / (2 * eps)
        # Solve (J^T J + lam I) delta = J^T r - lam (u-u0) per row
        JT = np.transpose(Jf, (0, 2, 1))
        H = JT @ Jf + lam * eye
        g = np.einsum("nij,nj->ni", JT, R) - lam * (U - U0)
        try:
            delta = np.linalg.solve(H, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            ok[:] = False
            break
        U = U + delta
        if float(np.max(np.linalg.norm(delta, axis=1))) < 1e-5:
            break
    Y = decode(U)
    residual = np.sum((Xc - Y) ** 2, axis=1)
    ok &= np.isfinite(residual)
    return {
        "U": U,
        "ok": ok,
        "iters": n_iter,
        "residual": residual,
        "du": np.linalg.norm(U - U0, axis=1),
    }


def gauss_legendre_3() -> tuple[np.ndarray, np.ndarray]:
    # nodes on [-1,1] → map to [0,1]
    xi = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
    w = np.array([5 / 9, 8 / 9, 5 / 9])
    t = 0.5 * (xi + 1.0)
    wt = 0.5 * w
    return t, wt


def pullback_path_length(
    chart: NestedChart, u_star: np.ndarray, decode_name: str = "TRS"
) -> float:
    """Approximate ∫ ||Df(tu) u|| dt with 3-point Gauss–Legendre."""
    decode = {
        "R": chart.decode_R,
        "TR": chart.decode_TR,
        "RS": chart.decode_RS,
        "TRS": chart.decode_TRS,
    }[decode_name]
    t, w = gauss_legendre_3()
    acc = 0.0
    for ti, wi in zip(t, w):
        u = ti * u_star
        Jf = numerical_jacobian(decode, u)
        v = Jf @ u_star
        acc += wi * float(np.linalg.norm(v))
    return acc


def decode_for_metric(chart: NestedChart, metric: str) -> Callable[[np.ndarray], np.ndarray]:
    if metric in ("quadratic_flat_full", "full_quadratic", "shuffled_Q", "matched_random_normal_Q"):
        return chart.decode_TRS
    if metric in ("radial_only", "tangent_plus_radial"):
        return chart.decode_R
    if metric in ("tangential_warp_only", "tangent_plus_tangential"):
        return chart.decode_TR
    if metric == "sphere_normal_only":
        return chart.decode_RS
    if metric == "tangent_plus_sphere_normal":
        return chart.decode_RS
    if metric == "tangent_plus_tangential_plus_radial":
        return chart.decode_TR
    if metric == "quadratic_geo_full":
        return chart.decode_TRS
    return chart.decode_TRS


def shuffle_chart(chart: NestedChart, rng: np.random.Generator) -> NestedChart:
    A = chart.A_flat.copy()
    BS = chart.BS_flat.copy()
    if A.size:
        A = A[:, rng.permutation(A.shape[1])]
    if BS.size:
        BS = BS[:, rng.permutation(BS.shape[1])]
    return NestedChart(
        x0=chart.x0,
        J=chart.J,
        A_flat=A,
        BS_flat=BS,
        ridge_A=chart.ridge_A,
        ridge_BS=chart.ridge_BS,
        coord_scale=chart.coord_scale,
    )


def random_normal_chart(chart: NestedChart, rng: np.random.Generator) -> NestedChart:
    """Matched Frobenius-norm random B^S in sphere-normal subspace."""
    BS = chart.BS_flat.copy()
    target_norm = float(np.linalg.norm(BS))
    R = rng.normal(size=BS.shape)
    # project each column into normal space
    from .sphere_normal_quadratic import normal_projector_apply

    R = normal_projector_apply(R, chart.x0, chart.J)
    nrm = float(np.linalg.norm(R))
    if nrm > EPS:
        R = R * (target_norm / nrm)
    return NestedChart(
        x0=chart.x0,
        J=chart.J,
        A_flat=chart.A_flat.copy(),
        BS_flat=R,
        ridge_A=chart.ridge_A,
        ridge_BS=chart.ridge_BS,
        coord_scale=chart.coord_scale,
    )


# -------------------- patch fit --------------------


def train_neighbours_for_anchor(
    X: np.ndarray,
    anchor: int,
    train_local: np.ndarray,
    K: int,
    device: torch.device,
    row_batch: int,
) -> np.ndarray:
    """K nearest training points to anchor (by cosine), excluding the anchor if present."""
    # knn among train only
    tr = train_local
    if anchor in set(int(i) for i in tr):
        # still ok — knn_indices masks self when querying full X; query in train subspace
        pass
    Ztr = X[tr]
    # find position: query is X[anchor], gallery is train
    device_t = device
    Zq = torch.as_tensor(X[anchor], device=device_t, dtype=torch.float32)
    Zq = Zq / Zq.norm().clamp_min(1e-12)
    Zg = torch.as_tensor(Ztr, device=device_t, dtype=torch.float32)
    Zg = Zg / Zg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = Zg @ Zq
    # exclude self if anchor is in train
    for j, g in enumerate(tr):
        if int(g) == int(anchor):
            sim[j] = -torch.inf
            break
    kk = min(K, len(tr) - (1 if int(anchor) in set(map(int, tr)) else 0))
    top = torch.topk(sim, k=kk).indices.cpu().numpy()
    return tr[top].astype(np.int64)


def fit_patch(
    X: np.ndarray,
    neigh: np.ndarray,
    d: int,
    seed: int,
) -> tuple[NestedChart | None, NestedChart | None, dict, str]:
    chart, chart_RS, info, _U, _glob, reason = _fit_neighborhood(X, neigh, d, seed)
    return chart, chart_RS, info, reason


def mahalanobis_distances(
    xi: np.ndarray, Xcand: np.ndarray, Xpatch: np.ndarray, J: np.ndarray
) -> np.ndarray:
    """Shrunk covariance in tangent coords of the patch."""
    U = (Xpatch - xi) @ J
    if len(U) < 3:
        return np.linalg.norm(Xcand - xi, axis=1)
    cov = np.cov(U.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    d = cov.shape[0]
    # Ledoit-like shrink to identity
    tr = float(np.trace(cov)) / d
    cov = 0.8 * cov + 0.2 * tr * np.eye(d)
    try:
        prec = np.linalg.inv(cov + 1e-6 * np.eye(d))
    except np.linalg.LinAlgError:
        prec = np.eye(d)
    Uc = (Xcand - xi) @ J
    return np.einsum("ij,jk,ik->i", Uc, prec, Uc)


# -------------------- core evaluation --------------------


def pool_size_for_k(k: int, cfg: QuadraticMKNNConfig) -> int:
    return int(max(cfg.candidate_pool_min, cfg.primary_multiplier * k))


def evaluate_smoke(root: Path, cfg: QuadraticMKNNConfig) -> Path:
    t0 = time.time()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)

    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print("[quadratic_mknn] loading Physics bundle…", flush=True)
    bundle = load_physics_bundle(root, cfg)
    X_by = bundle["X_by"]
    train_local = bundle["train_local"]
    anchors = bundle["anchors"]
    d_graph = bundle["d_graph"]
    device = torch.device(
        cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    print(f"[quadratic_mknn] device={device} anchors={len(anchors)} models={cfg.models}", flush=True)

    # model manifest
    man_rows = []
    for m in cfg.models:
        pq, col = MODELS[m]
        man_rows.append(
            {
                "model": m,
                "parquet": pq,
                "column": col,
                "space": cfg.space,
                "sae_tag": bundle.get("sae_tags", {}).get(m, ""),
                "n": int(X_by[m].shape[0]),
                "dim": int(X_by[m].shape[1]),
                "d_graph": int(d_graph[m]),
            }
        )
    pd.DataFrame(man_rows).to_csv(out / "model_manifest.csv", index=False)
    pd.DataFrame(
        {
            "anchor_local": anchors,
            "sample_id": bundle["sample_ids"][anchors],
        }
    ).to_parquet(out / "anchor_manifest.parquet", index=False)

    # Precompute ambient candidate pools (shared across metrics): for each anchor,
    # top-P neighbours in each model (P = max pool needed).
    max_k = max(cfg.retrieval_ks)
    max_pool = pool_size_for_k(max_k, cfg)
    print(f"[quadratic_mknn] precomputing ambient pools (P={max_pool})…", flush=True)
    ambient_nn: dict[str, np.ndarray] = {}
    for m in cfg.models:
        ambient_nn[m] = knn_indices(X_by[m], anchors, max_pool, device, cfg.row_batch)

    # Dimension screening + patch fits
    dim_rows: list[dict] = []
    patch_rows: list[dict] = []
    dist_rows: list[dict] = []
    recall_rows: list[dict] = []
    null_rows: list[dict] = []

    # Store per (model,K,anchor) distances for primary metrics → knn later
    # distances[model][K][metric] -> (n_anchors, max_pool) aligned to ambient_nn order
    distances: dict[str, dict[int, dict[str, np.ndarray]]] = {
        m: {} for m in cfg.models
    }
    patch_diag: dict[tuple[str, int, int], dict] = {}

    d_star_by: dict[tuple[str, int], int] = {}

    metrics_run = list(PRIMARY_METRICS)
    if cfg.phase2:
        for mname in PHASE2_METRICS:
            if mname not in metrics_run:
                metrics_run.append(mname)

    for m in cfg.models:
        X = X_by[m]
        for K in cfg.chart_scales_K:
            print(f"[quadratic_mknn] {m} K={K}: screening d…", flush=True)
            d_common, dim_df = screen_common_dimension(
                X, train_local, anchors, K, d_graph[m], device, cfg
            )
            d_star_by[(m, K)] = d_common
            dim_df = dim_df.assign(model=m, d_common=d_common, d_graph=d_graph[m])
            dim_rows.extend(dim_df.to_dict("records"))
            print(f"  d_M*(K)={d_common} (graph prior {d_graph[m]})", flush=True)

            n_a = len(anchors)
            dist_buf = {
                met: np.full((n_a, max_pool), np.nan, dtype=np.float64)
                for met in metrics_run
            }
            inv_fail = 0
            inv_total = 0

            # shared patch IDs from reference model if requested
            shared_neigh: dict[int, np.ndarray] | None = None
            if cfg.patch_mode == "shared":
                Xref = X_by[cfg.shared_reference_model]
                shared_neigh = {}
                for ai, a in enumerate(anchors):
                    shared_neigh[ai] = train_neighbours_for_anchor(
                        Xref, int(a), train_local, K, device, cfg.row_batch
                    )

            for ai, a in enumerate(anchors):
                if shared_neigh is not None:
                    neigh = shared_neigh[ai]
                else:
                    neigh = train_neighbours_for_anchor(
                        X, int(a), train_local, K, device, cfg.row_batch
                    )
                chart, chart_RS, info, reason = fit_patch(
                    X, neigh, d_common, seed=cfg.seed + 17 * ai + K
                )
                xi = X[int(a)].astype(np.float64)
                cand = ambient_nn[m][ai]  # (max_pool,)
                Xc = X[cand].astype(np.float64)

                # ambient distances
                dist_buf["ambient_l2"][ai] = np.linalg.norm(Xc - xi, axis=1)
                dist_buf["ambient_cosine"][ai] = 1.0 - (Xc @ xi)

                if chart is None:
                    # fallback: ambient only; tangent = ambient_l2
                    dist_buf["tangent"][ai] = dist_buf["ambient_l2"][ai]
                    for met in metrics_run:
                        if met.startswith("quadratic") or met in (
                            "shuffled_Q",
                            "local_mahalanobis",
                            "radial_only",
                            "tangential_warp_only",
                            "sphere_normal_only",
                            "tangent_plus_radial",
                            "tangent_plus_tangential",
                            "tangent_plus_sphere_normal",
                            "tangent_plus_tangential_plus_radial",
                            "full_quadratic",
                            "matched_random_normal_Q",
                        ):
                            dist_buf[met][ai] = dist_buf["ambient_l2"][ai]
                    patch_rows.append(
                        {
                            "model": m,
                            "K": K,
                            "anchor_local": int(a),
                            "d": d_common,
                            "ok": False,
                            "reason": reason,
                            "n_fit": 0,
                        }
                    )
                    continue

                J = chart.J
                # tangent orthogonality
                tang_rad_err = float(np.linalg.norm(J.T @ xi))
                u0 = (Xc - xi) @ J
                dist_buf["tangent"][ai] = np.linalg.norm(u0, axis=1)
                dist_buf["local_mahalanobis"][ai] = mahalanobis_distances(
                    xi, Xc, X[neigh], J
                )

                # reconstruction diagnostics on val-ish: use neighbourhood itself
                U_n = (X[neigh] - chart.x0) @ J
                E_R = float(np.mean(np.sum((chart.decode_R(U_n) - X[neigh]) ** 2, axis=1)))
                E_TRS = float(
                    np.mean(np.sum((chart.decode_TRS(U_n) - X[neigh]) ** 2, axis=1))
                )
                dec = decompose_BS(chart.BS_flat, J.shape[1])
                patch_rows.append(
                    {
                        "model": m,
                        "K": K,
                        "anchor_local": int(a),
                        "sample_id": int(bundle["sample_ids"][int(a)]),
                        "d": int(J.shape[1]),
                        "n_fit": int(info.get("n_fit", 0)),
                        "n_validation": int(info.get("n_val", 0)),
                        "ok": True,
                        "reason": "",
                        "PCA_recon_E_R": E_R,
                        "quadratic_recon_E_TRS": E_TRS,
                        "quadratic_gain": E_R - E_TRS,
                        "val_E_R": info.get("val_E_R"),
                        "val_E_TR": info.get("val_E_TR"),
                        "val_E_TRS": info.get("val_E_TRS"),
                        "tangent_radial_orth_err": tang_rad_err,
                        "B_fro": dec["B_fro"],
                        "H_norm": dec["H_norm"],
                        "B_traceless_fro": dec["B_traceless_fro"],
                        "patch_radius": float(
                            np.median(np.linalg.norm(X[neigh] - xi, axis=1))
                        ),
                        "ridge_A": chart.ridge_A,
                        "ridge_BS": chart.ridge_BS,
                        "patch_mode": cfg.patch_mode,
                    }
                )
                patch_diag[(m, K, int(a))] = {
                    "quadratic_gain": E_R - E_TRS,
                    "B_fro": dec["B_fro"],
                    "H_norm": dec["H_norm"],
                }

                rng = np.random.default_rng(cfg.seed + ai + 1009 * K)
                charts_for_metric: dict[str, NestedChart] = {
                    "quadratic_flat_full": chart,
                    "full_quadratic": chart,
                    "shuffled_Q": shuffle_chart(chart, rng),
                }
                if cfg.phase2:
                    charts_for_metric["matched_random_normal_Q"] = random_normal_chart(
                        chart, rng
                    )
                    charts_for_metric["radial_only"] = chart
                    charts_for_metric["tangential_warp_only"] = chart
                    charts_for_metric["sphere_normal_only"] = chart
                    charts_for_metric["tangent_plus_radial"] = chart
                    charts_for_metric["tangent_plus_tangential"] = chart
                    charts_for_metric["tangent_plus_sphere_normal"] = chart
                    charts_for_metric["tangent_plus_tangential_plus_radial"] = chart
                    charts_for_metric["quadratic_geo_full"] = chart

                for met, ch in charts_for_metric.items():
                    if met not in dist_buf:
                        continue
                    decode = decode_for_metric(ch, met)
                    inv = invert_coordinates_batch(
                        decode,
                        Xc,
                        u0,
                        lam=cfg.invert_lambda,
                        n_iter=cfg.invert_iters,
                    )
                    inv_total += int(len(Xc))
                    inv_fail += int((~inv["ok"]).sum())
                    dvec = np.linalg.norm(inv["U"], axis=1)
                    if met == "quadratic_geo_full":
                        dvec = np.array(
                            [
                                pullback_path_length(ch, inv["U"][j])
                                if inv["ok"][j]
                                else dist_buf["tangent"][ai, j]
                                for j in range(len(Xc))
                            ],
                            dtype=np.float64,
                        )
                    else:
                        dvec = np.where(inv["ok"], dvec, dist_buf["tangent"][ai])
                    dist_buf[met][ai] = dvec

                if (ai + 1) % 16 == 0:
                    print(f"  {m} K={K} anchors {ai+1}/{n_a}", flush=True)

            distances[m][K] = dist_buf
            null_rows.append(
                {
                    "model": m,
                    "K": K,
                    "invert_fail_rate": inv_fail / max(inv_total, 1),
                    "invert_total": inv_total,
                    "d_common": d_common,
                }
            )

            # candidate recall diagnostics vs larger ambient top-k (approx)
            for mult in cfg.candidate_multipliers:
                for k in cfg.retrieval_ks:
                    P = int(max(cfg.candidate_pool_min, mult * k))
                    P = min(P, max_pool)
                    # recall of true tangent top-k inside ambient pool of size P
                    # (proxy: how often tangent-ranked top-k ⊂ ambient pool — tautological
                    #  for ambient; for quadratic, check overlap of quad top-k with ambient P)
                    hits = []
                    for ai in range(n_a):
                        order_q = np.argsort(dist_buf["quadratic_flat_full"][ai])[:k]
                        # pool is already ambient top max_pool; recall of quad top-k in first P
                        hits.append(float(np.mean(order_q < P)))
                    recall_rows.append(
                        {
                            "model": m,
                            "K": K,
                            "k": k,
                            "multiplier": mult,
                            "pool_size": P,
                            "mean_frac_quad_topk_in_pool_prefix": float(np.mean(hits)),
                        }
                    )

    pd.DataFrame(dim_rows).to_parquet(out / "dimension_selection.parquet", index=False)
    pd.DataFrame(patch_rows).to_parquet(out / "patch_fit_summary.parquet", index=False)
    pd.DataFrame(recall_rows).to_parquet(out / "candidate_recall.parquet", index=False)
    pd.DataFrame(null_rows).to_parquet(out / "null_results.parquet", index=False)

    # -------------------- mKNN over pairs --------------------
    print("[quadratic_mknn] scoring mKNN…", flush=True)
    pairs = [
        (cfg.models[i], cfg.models[j])
        for i in range(len(cfg.models))
        for j in range(i + 1, len(cfg.models))
    ]
    mknn_rows: list[dict] = []
    diff_rows: list[dict] = []
    per_anchor_gain: list[dict] = []

    for K in cfg.chart_scales_K:
        for k in cfg.retrieval_ks:
            P = min(pool_size_for_k(k, cfg), max_pool)
            for met in metrics_run:
                # build neighbour sets per model
                nn_by: dict[str, np.ndarray] = {}
                ok_met = True
                for m in cfg.models:
                    if K not in distances[m] or met not in distances[m][K]:
                        ok_met = False
                        break
                    D = distances[m][K][met][:, :P]
                    # argsort → map through ambient candidate ids
                    order = np.argsort(D, axis=1)[:, :k]
                    nn = np.take_along_axis(ambient_nn[m][:, :P], order, axis=1)
                    nn_by[m] = nn
                if not ok_met:
                    continue
                for a, b in pairs:
                    score = mknn_from_sets(nn_by[a], nn_by[b], k)
                    # per-query overlaps for bootstrap
                    overlaps = np.array(
                        [
                            len(set(nn_by[a][i]) & set(nn_by[b][i])) / k
                            for i in range(len(anchors))
                        ]
                    )
                    mean, lo, hi = paired_bootstrap_ci(overlaps, seed=cfg.seed + K + k)
                    mknn_rows.append(
                        {
                            "model_a": a,
                            "model_b": b,
                            "K": K,
                            "k": k,
                            "metric": met,
                            "mknn": score,
                            "mknn_boot_mean": mean,
                            "mknn_boot_lo": lo,
                            "mknn_boot_hi": hi,
                            "n_queries": len(anchors),
                            "pool_size": P,
                            "patch_mode": cfg.patch_mode,
                            "d_a": d_star_by.get((a, K)),
                            "d_b": d_star_by.get((b, K)),
                        }
                    )

            # paired diffs vs tangent / ambient for primary metrics
            for met in metrics_run:
                if met in ("tangent", "ambient_cosine"):
                    continue
                for a, b in pairs:
                    def _overlaps(metric_name: str) -> np.ndarray:
                        nn_m = {}
                        for m in (a, b):
                            D = distances[m][K][metric_name][:, :P]
                            order = np.argsort(D, axis=1)[:, :k]
                            nn_m[m] = np.take_along_axis(
                                ambient_nn[m][:, :P], order, axis=1
                            )
                        return np.array(
                            [
                                len(set(nn_m[a][i]) & set(nn_m[b][i])) / k
                                for i in range(len(anchors))
                            ]
                        )

                    if met not in distances[a][K] or "tangent" not in distances[a][K]:
                        continue
                    o_m = _overlaps(met)
                    o_t = _overlaps("tangent")
                    o_c = _overlaps("ambient_cosine")
                    d_t = o_m - o_t
                    d_c = o_m - o_c
                    mt, lt, ht = paired_bootstrap_ci(d_t, seed=cfg.seed)
                    mc, lc, hc = paired_bootstrap_ci(d_c, seed=cfg.seed + 1)
                    diff_rows.append(
                        {
                            "model_a": a,
                            "model_b": b,
                            "K": K,
                            "k": k,
                            "metric": met,
                            "delta_vs_tangent": mt,
                            "delta_vs_tangent_lo": lt,
                            "delta_vs_tangent_hi": ht,
                            "delta_vs_ambient": mc,
                            "frac_anchors_improved_vs_tangent": float(np.mean(d_t > 0)),
                            "median_anchor_gain_vs_tangent": float(np.median(d_t)),
                        }
                    )
                    if met == "quadratic_flat_full":
                        for i, anc in enumerate(anchors):
                            diag = patch_diag.get((a, K, int(anc)), {})
                            per_anchor_gain.append(
                                {
                                    "model_a": a,
                                    "model_b": b,
                                    "K": K,
                                    "k": k,
                                    "anchor_local": int(anc),
                                    "delta_m": float(d_t[i]),
                                    "m_quad": float(o_m[i]),
                                    "m_tangent": float(o_t[i]),
                                    "quadratic_gain": diag.get("quadratic_gain"),
                                    "B_fro": diag.get("B_fro"),
                                    "H_norm": diag.get("H_norm"),
                                }
                            )

    mknn_df = pd.DataFrame(mknn_rows)
    diff_df = pd.DataFrame(diff_rows)
    gain_df = pd.DataFrame(per_anchor_gain)
    mknn_df.to_parquet(out / "mknn_results.parquet", index=False)
    diff_df.to_parquet(out / "mknn_pairwise_differences.parquet", index=False)
    gain_df.to_parquet(out / "distance_results.parquet", index=False)  # per-anchor gains

    # Component ablation table (from mknn if phase2 metrics present)
    abl_rows = []
    if cfg.phase2 and not mknn_df.empty:
        for K in cfg.chart_scales_K:
            for k in cfg.retrieval_ks:
                sub = mknn_df[(mknn_df.K == K) & (mknn_df.k == k)]

                def mean_met(name: str) -> float:
                    s = sub[sub.metric == name]["mknn"]
                    return float(s.mean()) if len(s) else float("nan")

                abl_rows.append(
                    {
                        "K": K,
                        "k": k,
                        "mknn_T": mean_met("tangent"),
                        "mknn_T_QR": mean_met("tangent_plus_radial"),
                        "mknn_T_QT": mean_met("tangent_plus_tangential"),
                        "mknn_T_BS": mean_met("tangent_plus_sphere_normal"),
                        "mknn_T_QT_QR": mean_met("tangent_plus_tangential_plus_radial"),
                        "mknn_full": mean_met("quadratic_flat_full"),
                        "delta_QT": mean_met("tangent_plus_tangential") - mean_met("tangent"),
                        "delta_QR": mean_met("tangent_plus_radial") - mean_met("tangent"),
                        "delta_BS": mean_met("quadratic_flat_full")
                        - mean_met("tangent_plus_tangential_plus_radial"),
                    }
                )
    pd.DataFrame(abl_rows).to_parquet(
        out / "quadratic_component_ablation.parquet", index=False
    )

    # Correlations / quantiles
    corr_rows = []
    curv_q_rows = []
    recon_q_rows = []
    if not gain_df.empty:
        for (K, k), g in gain_df.groupby(["K", "k"]):
            for col in ("quadratic_gain", "B_fro", "H_norm"):
                x = g[col].to_numpy(dtype=float)
                y = g["delta_m"].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 8:
                    continue
                sp = spearmanr(x[mask], y[mask])
                pr = pearsonr(x[mask], y[mask])
                corr_rows.append(
                    {
                        "K": K,
                        "k": k,
                        "feature": col,
                        "spearman": float(sp.correlation) if sp.correlation is not None else float("nan"),
                        "spearman_p": float(sp.pvalue) if sp.pvalue is not None else float("nan"),
                        "pearson": float(pr[0]),
                        "pearson_p": float(pr[1]),
                        "n": int(mask.sum()),
                    }
                )
            # curvature quartiles on B_fro
            g2 = g.dropna(subset=["B_fro", "delta_m"]).copy()
            if len(g2) >= 16:
                g2["q"] = pd.qcut(g2["B_fro"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
                for qn, sg in g2.groupby("q", observed=True):
                    curv_q_rows.append(
                        {
                            "K": K,
                            "k": k,
                            "curvature_quartile": str(qn),
                            "mean_delta_m": float(sg["delta_m"].mean()),
                            "n": int(len(sg)),
                        }
                    )
            g3 = g.dropna(subset=["quadratic_gain", "delta_m"]).copy()
            if len(g3) >= 16:
                g3["q"] = pd.qcut(
                    g3["quadratic_gain"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
                )
                for qn, sg in g3.groupby("q", observed=True):
                    recon_q_rows.append(
                        {
                            "K": K,
                            "k": k,
                            "recon_gain_quartile": str(qn),
                            "mean_delta_m": float(sg["delta_m"].mean()),
                            "n": int(len(sg)),
                        }
                    )

    pd.DataFrame(corr_rows).to_parquet(
        out / "patchwise_gain_correlations.parquet", index=False
    )
    pd.DataFrame(curv_q_rows).to_parquet(
        out / "curvature_quantile_results.parquet", index=False
    )
    pd.DataFrame(recon_q_rows).to_parquet(
        out / "reconstruction_quantile_results.parquet", index=False
    )

    # Aggregate summary
    if not mknn_df.empty:
        agg = (
            mknn_df.groupby(["metric", "K", "k"])["mknn"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg.to_csv(out / "aggregate_summary.csv", index=False)
    else:
        agg = pd.DataFrame()

    # Figures + report
    _make_figures(mknn_df, diff_df, gain_df, pd.DataFrame(patch_rows), figdir, cfg)
    _write_report(out, cfg, mknn_df, diff_df, gain_df, agg, null_rows, t0)

    print(f"[quadratic_mknn] done in {time.time()-t0:.1f}s → {out}", flush=True)
    return out


def _pivot_delta(diff_df: pd.DataFrame, metric: str, value_col: str) -> pd.DataFrame | None:
    sub = diff_df[diff_df.metric == metric]
    if sub.empty:
        return None
    g = sub.groupby(["K", "k"])[value_col].mean().unstack("k")
    return g


def _make_figures(
    mknn_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    patch_df: pd.DataFrame,
    figdir: Path,
    cfg: QuadraticMKNNConfig,
) -> None:
    if mknn_df.empty:
        return

    # mKNN by metric (mean over pairs), facets by k at fixed mid K
    for K in cfg.chart_scales_K:
        sub = mknn_df[mknn_df.K == K]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for met, g in sub.groupby("metric"):
            gg = g.groupby("k")["mknn"].mean()
            ax.plot(gg.index, gg.values, "o-", label=met)
        ax.set_xlabel("retrieval k")
        ax.set_ylabel("mean mKNN")
        ax.set_title(f"mKNN by metric (K={K})")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figdir / f"mknn_by_metric_K{K}.png", dpi=140)
        plt.close(fig)

    # heatmaps
    for metric, col, name in [
        ("quadratic_flat_full", "delta_vs_tangent", "quad_minus_tangent"),
        ("quadratic_flat_full", "delta_vs_ambient", "quad_minus_ambient"),
        ("tangent", None, "tangent_minus_ambient"),
        ("local_mahalanobis", "delta_vs_tangent", "mahalanobis_minus_tangent"),
        ("shuffled_Q", "delta_vs_tangent", "shuffled_minus_tangent"),
    ]:
        if col is None:
            # build from mknn
            amb = (
                mknn_df[mknn_df.metric == "ambient_cosine"]
                .groupby(["K", "k"])["mknn"]
                .mean()
            )
            tang = (
                mknn_df[mknn_df.metric == "tangent"].groupby(["K", "k"])["mknn"].mean()
            )
            if amb.empty or tang.empty:
                continue
            heat = (tang - amb).unstack("k")
        else:
            heat = _pivot_delta(diff_df, metric, col)
        if heat is None or heat.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4))
        im = ax.imshow(heat.values, aspect="auto", cmap="RdBu_r", origin="lower")
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(list(heat.columns))
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(list(heat.index))
        ax.set_xlabel("k")
        ax.set_ylabel("K")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(figdir / f"heatmap_{name}.png", dpi=140)
        plt.close(fig)

    # scatter gains
    if not gain_df.empty:
        g = gain_df[gain_df.k == 10]
        if not g.empty and g["quadratic_gain"].notna().any():
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(g["quadratic_gain"], g["delta_m"], alpha=0.35, s=12)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("quadratic reconstruction gain")
            ax.set_ylabel("ΔmKNN (quad − tangent)")
            ax.set_title("mKNN gain vs recon gain (k=10)")
            fig.tight_layout()
            fig.savefig(figdir / "gain_vs_recon.png", dpi=140)
            plt.close(fig)
        if not g.empty and g["B_fro"].notna().any():
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(g["B_fro"], g["delta_m"], alpha=0.35, s=12)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("‖B^S‖_F")
            ax.set_ylabel("ΔmKNN (quad − tangent)")
            ax.set_title("mKNN gain vs sphere-normal curvature (k=10)")
            fig.tight_layout()
            fig.savefig(figdir / "gain_vs_BS.png", dpi=140)
            plt.close(fig)

    if not patch_df.empty and "quadratic_gain" in patch_df:
        fig, ax = plt.subplots(figsize=(5, 4))
        for m, g in patch_df[patch_df.ok == True].groupby("model"):  # noqa: E712
            gg = g.groupby("K")["quadratic_gain"].mean()
            ax.plot(gg.index, gg.values, "o-", label=m)
        ax.set_xlabel("K")
        ax.set_ylabel("mean quadratic recon gain")
        ax.set_title("Quadratic reconstruction gain vs K")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / "recon_gain_vs_K.png", dpi=140)
        plt.close(fig)


def _write_report(
    out: Path,
    cfg: QuadraticMKNNConfig,
    mknn_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    agg: pd.DataFrame,
    null_rows: list[dict],
    t0: float,
) -> None:
    lines = [
        "# Quadratic per-patch mKNN report",
        "",
        "Per-anchor nested quadratic charts (NestedChart R/T/S) vs tangent / ambient.",
        "",
        f"- space: `{cfg.space}` "
        "(dense embeddings | TopK SAE codes | SAE×IDF; charts fit after L2-normalize)",
        f"- models: `{cfg.models}`",
        f"- K (chart): `{cfg.chart_scales_K}`",
        f"- k (retrieval): `{cfg.retrieval_ks}`",
        f"- n_anchors: {cfg.n_anchors}",
        f"- patch_mode: `{cfg.patch_mode}`",
        f"- phase2 components: {cfg.phase2}",
        f"- wall time: {time.time() - t0:.1f}s",
        "",
        "## Gates",
        "",
    ]

    def mean_metric(met: str) -> float:
        if mknn_df.empty:
            return float("nan")
        return float(mknn_df[mknn_df.metric == met]["mknn"].mean())

    amb = mean_metric("ambient_cosine")
    tang = mean_metric("tangent")
    quad = mean_metric("quadratic_flat_full")
    shuf = mean_metric("shuffled_Q")
    mah = mean_metric("local_mahalanobis")

    gate_a = tang > amb
    gate_b = quad > tang
    gate_c = quad > shuf
    lines += [
        f"- **Gate A** (tangent > ambient): `{gate_a}` "
        f"(tangent={tang:.4f}, ambient={amb:.4f})",
        f"- **Gate B** (quadratic_flat > tangent): `{gate_b}` "
        f"(quad={quad:.4f}, tangent={tang:.4f})",
        f"- **Gate C** (quad > shuffled_Q): `{gate_c}` "
        f"(quad={quad:.4f}, shuffled={shuf:.4f})",
        f"- Mahalanobis mean mKNN: {mah:.4f}",
        "",
        "## Answers to report questions",
        "",
    ]

    # Q1–Q2
    lines.append(
        f"1. Tangent vs ambient: mean mKNN tangent={tang:.4f}, ambient={amb:.4f}, "
        f"Δ={tang-amb:+.4f}."
    )
    lines.append(
        f"2. Quadratic-flat vs tangent: mean mKNN quad={quad:.4f}, tangent={tang:.4f}, "
        f"Δ={quad-tang:+.4f}."
    )

    if not diff_df.empty:
        sub = diff_df[diff_df.metric == "quadratic_flat_full"]
        if not sub.empty:
            byK = sub.groupby("K")["delta_vs_tangent"].mean()
            byk = sub.groupby("k")["delta_vs_tangent"].mean()
            bestK = int(byK.idxmax()) if len(byK) else -1
            bestk = int(byk.idxmax()) if len(byk) else -1
            lines.append(
                f"3. Largest quadratic−tangent gain by chart scale K: "
                f"K={bestK} (Δ={byK.max():+.4f})."
            )
            lines.append(
                f"4. Largest gain by retrieval k: k={bestk} (Δ={byk.max():+.4f})."
            )
            lines.append(
                "5. Intermediate-scale pattern: "
                + ", ".join(f"K={int(K)}:{v:+.4f}" for K, v in byK.items())
            )
            # per pair
            lines.append("6. Per-pair mean Δ(quad−tangent):")
            for (a, b), g in sub.groupby(["model_a", "model_b"]):
                lines.append(f"   - {a}↔{b}: {g['delta_vs_tangent'].mean():+.4f}")

    lines.append(
        f"7. Learned Q vs shuffled: Δ(mKNN)={quad - shuf:+.4f} "
        f"({'beats null' if gate_c else 'does not beat null'})."
    )
    lines.append(
        f"8. Mahalanobis control: mah={mah:.4f}, quad={quad:.4f}, "
        f"tangent={tang:.4f}."
    )
    if cfg.phase2:
        lines.append(
            "9–11. See `quadratic_component_ablation.parquet` for Q_T / Q_R / B^S contributions."
        )
        geo = mean_metric("quadratic_geo_full")
        lines.append(
            f"12. Pullback geodesic mean mKNN={geo:.4f} vs flat={quad:.4f}."
        )
    else:
        lines.append(
            "9–12. Component / geodesic ablations not run (pass `--phase2`)."
        )

    if not gain_df.empty and gain_df["quadratic_gain"].notna().any():
        sp = spearmanr(
            gain_df["quadratic_gain"].to_numpy(float),
            gain_df["delta_m"].to_numpy(float),
            nan_policy="omit",
        )
        lines.append(
            f"13. Spearman(recon gain, ΔmKNN)={sp.correlation:.3f} (p={sp.pvalue:.3g})."
        )
    else:
        lines.append("13. Recon-gain correlation: insufficient data.")

    if not gain_df.empty and gain_df["B_fro"].notna().any():
        sp = spearmanr(
            gain_df["B_fro"].to_numpy(float),
            gain_df["delta_m"].to_numpy(float),
            nan_policy="omit",
        )
        lines.append(
            f"14. Spearman(‖B^S‖, ΔmKNN)={sp.correlation:.3f} (p={sp.pvalue:.3g})."
        )
    else:
        lines.append("14. Curvature correlation: insufficient data.")

    lines.append(
        "15. Candidate recall: see `candidate_recall.parquet` "
        "(fraction of quadratic top-k indices within ambient pool prefix)."
    )
    lines.append(
        f"16. Patch populations: this run used `{cfg.patch_mode}` "
        "(re-run with `--patch-mode shared` for the alternate condition)."
    )

    # Strongest statement
    if gate_b and gate_a:
        stmt = (
            "Support for local rank-collapse + second-order flattening: "
            "quadratic_flat > tangent > ambient on aggregate smoke."
        )
    elif gate_a and not gate_b:
        stmt = (
            "Tangent/PCA rank-collapse helps vs ambient, but quadratic flattening "
            "does not clearly add beyond tangent in this smoke."
        )
    elif gate_b and not gate_a:
        stmt = (
            "Quadratic exceeds tangent, but tangent does not beat ambient — "
            "interpret carefully."
        )
    else:
        stmt = (
            "Smoke does not support mKNN_Q > mKNN_T > mKNN_ambient on aggregate; "
            "inspect K×k heatmaps and curvature strata before concluding."
        )
    lines += ["", f"17. **Strongest defensible statement:** {stmt}", ""]

    if null_rows:
        lines += ["## Invert diagnostics", ""]
        for r in null_rows:
            lines.append(
                f"- {r['model']} K={r['K']}: invert_fail_rate={r['invert_fail_rate']:.4f}, "
                f"d*={r['d_common']}"
            )

    lines += ["", "## Aggregate mean mKNN", "", "```"]
    if not agg.empty:
        lines.append(
            agg.pivot_table(index=["metric", "K"], columns="k", values="mean").to_string()
        )
    else:
        lines.append("(empty)")
    lines.append("```")

    (out / "quadratic_mknn_report.md").write_text("\n".join(lines) + "\n")
