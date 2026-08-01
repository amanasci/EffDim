"""
Cross-model sweep of the Phase 2 validity gate.

Executes 02-MODEL-SWEEP-PREREGISTRATION.md. Fits Isomap and computes the full classical-MDS
eigenspectrum for every legacysurvey_* model config in UniverseTBD/pu-embeddings, holding the
object population, k, n_components, and every threshold constant. Only the model varies.

Resumable: results append to a JSONL file and completed models are skipped on re-run.
Reads only the legacysurvey column of each parquet (the paired hsc column is skipped).
"""
import gc
import json
import os
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from scipy.linalg import eigvalsh
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

CACHE = "notebooks/.cache"
OUT = f"{CACHE}/model_sweep_results.jsonl"
REPO = "datasets/UniverseTBD/pu-embeddings"

# --- pre-registered constants, copied verbatim from 02-MODEL-SWEEP-PREREGISTRATION.md ---
R_MAX_PASS, M_MAX_PASS = 0.10, 0.05
R_MAX_MARGINAL, M_MAX_MARGINAL = 0.25, 0.15
K, NCOMP = 15, 18

MODELS = [
    "astropt_015M", "astropt_095M", "astropt_850M",
    "clip_base", "clip_large",
    "convnext_nano", "convnext_tiny", "convnext_base", "convnext_large",
    "dino_small", "dino_base", "dino_large", "dino_giant",
    "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vitl16",
    "dinov3_vith16plus", "dinov3_vit7b16",
    "ijepa_giant", "ijepa_huge",
    "llava_15_7b", "llava_15_13b",
    "paligemma_3b", "paligemma_10b", "paligemma_28b",
    "vit_base", "vit_large", "vit_huge",
    "vit-mae_base", "vit-mae_large", "vit-mae_huge",
    "vjepa_large", "vjepa_giant", "vjepa_huge",
]


def classify(r, m):
    if r < R_MAX_PASS and m < M_MAX_PASS:
        return "PASS"
    if r < R_MAX_MARGINAL and m < M_MAX_MARGINAL:
        return "MARGINAL"
    return "FAIL"


def gate_stats(ev):
    neg, pos = ev[ev < 0], ev[ev > 0]
    return dict(
        r=float(abs(neg.min()) / pos.max()),
        m=float(np.abs(neg).sum() / np.abs(ev).sum()),
        n_positive=int(pos.size), n_negative=int(neg.size),
        lambda_max_pos=float(pos.max()), lambda_min_neg=float(neg.min()),
        noise_floor=float(ev.size * np.finfo(np.float64).eps * pos.max()),
    )


def spectrum(D):
    D2 = np.array(D, dtype=np.float64, copy=True)
    D2 **= 2
    D2 -= D2.mean(axis=1, keepdims=True)
    D2 -= D2.mean(axis=0, keepdims=True)
    D2 += D2.mean()
    D2 *= -0.5
    sym = float(np.abs(D2 - D2.T).max())
    ev = eigvalsh(D2)
    del D2
    gc.collect()
    return ev, sym


def twonn(X, sample=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    d, _ = NearestNeighbors(n_neighbors=3, n_jobs=-1).fit(X).kneighbors(X[idx])
    r1, r2 = d[:, 1], d[:, 2]
    ok = (r1 > 0) & (r2 > r1)
    return float(ok.sum() / np.log(r2[ok] / r1[ok]).sum())


def local_pca_dim(X, n_centers=600, k=60, var=0.90, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.choice(len(X), size=n_centers, replace=False)
    _, nbr = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X).kneighbors(X[centers])
    out = np.empty(n_centers, dtype=np.int64)
    for i, row in enumerate(nbr):
        P = X[row] - X[row].mean(axis=0)
        e = np.linalg.svd(P, compute_uv=False) ** 2
        out[i] = int(np.searchsorted(np.cumsum(e) / e.sum(), var) + 1)
    return out


def load_model_rows(fs, model, row_indices):
    """Read ONLY the legacysurvey column, take the pre-registered rows, L2-normalize."""
    path = f"{REPO}/legacysurvey/legacysurvey_{model}.parquet"
    col = f"{model}_legacysurvey"
    with fs.open(path, "rb") as fh:
        tb = pq.ParquetFile(fh).read(columns=[col])
    arr = tb.column(0).combine_chunks()
    dim = arr.type.list_size
    # flatten the fixed_size_list to its backing values buffer, reshape, then take only the
    # pre-registered rows. Stays in the parquet's own dtype (float32) until after subsetting,
    # so the biggest models never materialize 101725 x D in float64.
    flat = arr.flatten().to_numpy(zero_copy_only=False)
    X = flat.reshape(-1, dim)[row_indices].astype(np.float64)
    del tb, arr, flat
    gc.collect()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    assert (norms > 0).all(), f"{model}: zero-norm row present"
    return X / norms, float(norms.mean()), float(norms.std())


def main():
    orig = np.load(f"{CACHE}/subsample_20260729_a79b3460b838fd0a.npz")
    row_indices = orig["row_indices"]
    assert row_indices.shape == (10000,)

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["model"])
                except Exception:
                    pass
    print(f"pre-registered models: {len(MODELS)}   already done: {len(done)}")

    fs = HfFileSystem()
    for i, model in enumerate(MODELS, 1):
        if model in done:
            print(f"[{i:2d}/{len(MODELS)}] {model:24s} SKIP (done)")
            continue
        t_all = time.perf_counter()
        rec = {"model": model, "k": K, "n_components": NCOMP, "n_rows": 10000}
        try:
            t0 = time.perf_counter()
            X, nmean, nstd = load_model_rows(fs, model, row_indices)
            rec["load_seconds"] = round(time.perf_counter() - t0, 1)
            rec["ambient_dim"] = int(X.shape[1])
            rec["raw_norm_mean"], rec["raw_norm_std"] = nmean, nstd

            g = kneighbors_graph(X, n_neighbors=K, mode="distance", n_jobs=-1)
            ncomp, _ = connected_components(g, directed=False)
            rec["graph_n_components"] = int(ncomp)
            del g
            if ncomp != 1:
                rec["status"] = "skipped_disconnected"
                print(f"[{i:2d}/{len(MODELS)}] {model:24s} DISCONNECTED ({ncomp} components)")
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                del X
                gc.collect()
                continue

            t0 = time.perf_counter()
            iso = Isomap(n_neighbors=K, n_components=NCOMP, eigen_solver="dense", n_jobs=-1)
            iso.fit(X)
            rec["fit_seconds"] = round(time.perf_counter() - t0, 1)

            t0 = time.perf_counter()
            ev, sym = spectrum(iso.dist_matrix_)
            rec["eig_seconds"] = round(time.perf_counter() - t0, 1)
            rec["symmetry_max"] = sym
            del iso
            gc.collect()

            assert ev.shape == (10000,) and ev.dtype == np.float64
            rec.update(gate_stats(ev))
            rec["verdict"] = classify(rec["r"], rec["m"])

            rec["twonn_dim"] = twonn(X)
            dims = local_pca_dim(X)
            rec["localpca_median"] = float(np.median(dims))
            rec["localpca_mean"] = float(dims.mean())
            rec["localpca_std"] = float(dims.std())
            rec["status"] = "ok"
            del X, ev, dims
            gc.collect()

            print(f"[{i:2d}/{len(MODELS)}] {model:24s} D={rec['ambient_dim']:5d}  "
                  f"r={rec['r']:.6f}  m={rec['m']:.6f}  {rec['verdict']:8s}  "
                  f"ID={rec['twonn_dim']:5.1f}/{rec['localpca_median']:4.0f}  "
                  f"({time.perf_counter()-t_all:.0f}s)")
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-1500:]
            print(f"[{i:2d}/{len(MODELS)}] {model:24s} ERROR {type(e).__name__}: {e}")
            gc.collect()
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"\nsweep complete -> {OUT}")


if __name__ == "__main__":
    main()
