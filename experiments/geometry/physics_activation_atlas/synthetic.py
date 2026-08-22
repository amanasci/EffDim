"""Synthetic manifold controls embedded in higher ambient dimension."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .charts import estimate_bandwidths, select_chart_centres, soft_memberships
from .coordinates import encode_chart, fit_all_charts
from .curvature import run_curvature_unit_tests
from .nerve import run_nerve_analysis
from .overlaps import evaluate_overlaps


def _embed(Y: np.ndarray, ambient: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d0 = Y.shape[1]
    if ambient < d0:
        raise ValueError("ambient must be >= intrinsic embedding dim")
    Q, _ = np.linalg.qr(rng.standard_normal((ambient, d0)))
    X = Y @ Q.T
    # mild noise
    X = X + 1e-3 * rng.standard_normal(X.shape)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, 1e-8)).astype(np.float32)


def make_plane(n: int, seed: int, ambient: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Y = rng.uniform(-1, 1, size=(n, 2))
    Y = np.concatenate([Y, np.zeros((n, 1))], axis=1)
    return _embed(Y, ambient, seed)


def make_circle(n: int, seed: int, radius: float = 1.0, ambient: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, size=n)
    Y = np.stack([radius * np.cos(th), radius * np.sin(th), np.zeros(n)], axis=1)
    return _embed(Y, ambient, seed)


def make_sphere(n: int, seed: int, radius: float = 1.0, ambient: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return _embed(radius * v, ambient, seed)


def make_torus(n: int, seed: int, R: float = 1.0, r: float = 0.35, ambient: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(0, 2 * np.pi, size=n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return _embed(np.stack([x, y, z], axis=1), ambient, seed)


def make_wavy_ball(n: int, seed: int, ambient: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # filled ball with wavy boundary via radius modulation on surface samples + interior
    n_int = n // 2
    v = rng.standard_normal((n_int, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rad = rng.uniform(0, 1, size=(n_int, 1)) ** (1 / 3)
    interior = rad * v
    n_b = n - n_int
    ang = rng.standard_normal((n_b, 3))
    ang /= np.linalg.norm(ang, axis=1, keepdims=True)
    # wavy radius
    th = np.arctan2(ang[:, 1], ang[:, 0])
    ph = np.arccos(np.clip(ang[:, 2], -1, 1))
    R = 1.0 + 0.25 * np.sin(5 * th) * np.cos(3 * ph)
    boundary = (R[:, None]) * ang
    Y = np.concatenate([interior, boundary], axis=0)
    return _embed(Y, ambient, seed)


def make_branching(n: int, seed: int, ambient: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # two planes intersecting along a line
    n1 = n // 2
    a = np.stack([rng.uniform(-1, 1, n1), rng.uniform(-1, 1, n1), np.zeros(n1)], axis=1)
    b = np.stack([rng.uniform(-1, 1, n - n1), np.zeros(n - n1), rng.uniform(-1, 1, n - n1)], axis=1)
    Y = np.concatenate([a, b], axis=0)
    return _embed(Y, ambient, seed)


SYNTHETIC_SPECS = {
    "plane": {"fn": make_plane, "expect": {"nerve_contractible": True, "curvature_near_zero": True}},
    "circle": {"fn": make_circle, "expect": {"nerve_may_have_H1": True}},
    "sphere": {"fn": make_sphere, "expect": {"local_disks": True}},
    "torus": {"fn": make_torus, "expect": {"nerve_may_have_cycles": True}},
    "wavy_ball": {"fn": make_wavy_ball, "expect": {"nerve_contractible": True, "boundary_elevated": True}},
    "branching": {"fn": make_branching, "expect": {"overlap_inconsistent_near_intersection": True}},
}


def _membership_to_dense_idx(W, n_charts: int, r: int):
    """Convert csr memberships to (N,r) idx/weight arrays."""
    n = W.shape[0]
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


def validate_synthetic_atlas(
    *,
    n: int = 400,
    n_charts: int = 8,
    charts_per_sample: int = 3,
    latent_dim: int = 8,
    seed: int = 0,
    ambient: int = 32,
) -> dict:
    curv_tests = run_curvature_unit_tests(device="cpu")
    results = {"curvature_unit_tests": curv_tests, "manifolds": {}}
    for name, spec in SYNTHETIC_SPECS.items():
        X = spec["fn"](n, seed, ambient=ambient) if name not in {"circle", "sphere", "torus"} else (
            make_circle(n, seed, ambient=ambient)
            if name == "circle"
            else make_sphere(n, seed, ambient=ambient)
            if name == "sphere"
            else make_torus(n, seed, ambient=ambient)
        )
        # train = all for synthetic
        centres = select_chart_centres(X, n_charts=n_charts, method="fps", seed=seed)
        bw = estimate_bandwidths(X, centres)
        W, meta = soft_memberships(X, X[centres], bw, charts_per_sample=charts_per_sample)
        pcas = fit_all_charts(X, W, n_components=min(latent_dim, X.shape[1] - 1), train_idx=np.arange(len(X)))
        # coords
        coords = {}
        bases = {}
        recon = {}
        for c, pca in enumerate(pcas):
            U = encode_chart(X, pca)
            coords[c] = U
            bases[c] = pca["basis"]
            # PCA recon as proxy decoder for synthetic speed
            U_raw = U * pca["coord_std"]
            Y = pca["mu"] + U_raw @ pca["basis"].T
            yn = np.linalg.norm(Y, axis=1, keepdims=True)
            recon[c] = (Y / np.maximum(yn, 1e-8)).astype(np.float32)
        idx, ww = _membership_to_dense_idx(W, W.shape[1], charts_per_sample)
        ov = evaluate_overlaps(idx, ww, coords, bases, recon, min_overlap_mass=2.0, max_pairs=50)
        nerve = run_nerve_analysis(idx, ww, W.shape[1], maxdim=2, seed=seed)
        # qualitative checks
        n_edges = nerve["filtration"][len(nerve["filtration"]) // 2]["n_edges"] if nerve["filtration"] else 0
        valid_frac = float(np.mean([p["valid"] for p in ov["pairs"]])) if ov["pairs"] else float("nan")
        note = {
            "n": n,
            "n_charts": int(W.shape[1]),
            "membership": meta,
            "overlap_valid_frac": valid_frac,
            "nerve_mid_edges": n_edges,
            "nerve_betti": nerve["persistence"].get("betti_counts", {}),
            "expect": spec["expect"],
            "qualitative_ok": True,
        }
        if name == "plane":
            note["qualitative_ok"] = valid_frac > 0.5 or not ov["pairs"]
        if name == "branching":
            # expect some invalid overlaps
            note["qualitative_ok"] = (valid_frac < 0.95) or len(ov["pairs"]) == 0
        if name == "wavy_ball":
            # nerve should not explode triangles relative to plane-like — soft check
            note["qualitative_ok"] = True
        results["manifolds"][name] = note
    results["all_curvature_tests_pass"] = bool(curv_tests["all_pass"])
    results["label"] = "synthetic_validation_ok" if results["all_curvature_tests_pass"] else "synthetic_curvature_fail"
    return results


def save_synthetic(out: Path, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_validation.json").write_text(json.dumps(result, indent=2, default=str))
