"""Intrinsic dimension of each molecule encoder's embeddings, and the estimators' bias on unit spheres.

Per encoder: a 10,000-row subsample (seed 20261001) of the row-normalised embeddings, one shared k-NN of squared
distances with k = 10 (as effdim.api does), then mle, two_nn, tle and mind_mlk from effdim.geometry;
d_ID = round(median), d_run = min(d_ID, 20). The synthetic check runs the same estimate on unit spheres of known
dimension; the estimators read low at higher true dimension, which the report states.

Usage:
    python -m sweep.intrinsic_dim --manifest molecules.yaml --root <qm9-out> --out <molecules_d.json>
    python -m sweep.intrinsic_dim --synthetic --out data/qm9/id_synthetic.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from effdim.geometry import (compute_knn_distances, mind_mlk_dimensionality, mle_dimensionality,
                             tle_dimensionality, two_nn_dimensionality)

from sweep.jobs import Layout
from sweep.manifest import MOLECULES_PATH, load_manifest

SEED = 20261001
N_SUB = 10_000
K = 10
D_CAP = 20
ESTIMATORS = ("mle", "two_nn", "tle", "mind_mlk")


def estimate(X: np.ndarray, n_sub: int = N_SUB, seed: int = SEED, k: int = K) -> Dict[str, float]:
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=min(n_sub, X.shape[0]), replace=False))
    Xs = np.asarray(X[idx], dtype=np.float64)
    Xs = np.ascontiguousarray(Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-12), dtype=np.float32)
    knn = compute_knn_distances(Xs, k)
    return {"mle": float(mle_dimensionality(Xs, precomputed_knn_dist_sq=knn)),
            "two_nn": float(two_nn_dimensionality(Xs, precomputed_knn_dist_sq=knn)),
            "tle": float(tle_dimensionality(Xs, precomputed_knn_dist_sq=knn)),
            "mind_mlk": float(mind_mlk_dimensionality(Xs, precomputed_knn_dist_sq=knn))}


def choose_d(estimates: Dict[str, float], cap: int = D_CAP) -> Tuple[int, int]:
    d_id = int(round(float(np.median([estimates[k] for k in ESTIMATORS]))))
    return d_id, min(d_id, cap)


def sphere(n: int, true_d: int, D: int, seed: int) -> np.ndarray:
    """n points uniform on the unit sphere S^true_d (intrinsic dimension true_d) in R^(true_d+1), rotated into R^D."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, true_d + 1)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(rng.standard_normal((D, true_d + 1)))
    return g @ Q.T


def synthetic_check(true_dims=(8, 16, 24), Ds=(384, 768, 3072), n: int = N_SUB, seed: int = SEED) -> dict:
    rows = []
    for td in true_dims:
        for D in Ds:
            est = estimate(sphere(n, td, D, seed + td), n_sub=n, seed=seed)
            rows.append({"true_d": int(td), "D": int(D), "estimates": est, "bias": {k: est[k] - td for k in ESTIMATORS}})
    return {"n": int(n), "seed": int(seed), "rows": rows}


def load_embeddings(path, column: str) -> np.ndarray:
    import pyarrow.parquet as pq
    col = pq.read_table(path, columns=[column]).column(column).combine_chunks()
    return np.asarray(col.flatten(), dtype=np.float32).reshape(len(col), -1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(MOLECULES_PATH))
    ap.add_argument("--root", default=None, help="output root holding hf/<parquet_file>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--synthetic", action="store_true", help="run the unit-sphere bias check instead")
    a = ap.parse_args()
    if a.synthetic:
        res = synthetic_check()
        for r in res["rows"]:
            print(f"true d {r['true_d']:2d} D {r['D']:4d}: " + " ".join(f"{k} {r['bias'][k]:+.2f}" for k in ESTIMATORS), flush=True)
        a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        return
    m = load_manifest(a.manifest); lay = Layout(Path(a.root))
    out = {}
    for e in m.encoders:
        est = estimate(load_embeddings(lay.hf_parquet(e), e.column))
        d_id, d_run = choose_d(est)
        out[e.name] = {"d_ID": d_id, "d_run": d_run, "estimates": est}
        print(f"{e.name}: d_ID {d_id} d_run {d_run} " + " ".join(f"{k} {est[k]:.2f}" for k in ESTIMATORS), flush=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
