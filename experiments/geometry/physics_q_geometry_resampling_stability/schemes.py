"""Conditional-support repartition (A) and object-support subsample (B)."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

_EXEC: ProcessPoolExecutor | None = None

import numpy as np

from .config import D, HALF_PRIME, K, K_PRIME, KEEP_FRAC
from .q_fit import design_condition, fit_explicit_halves, hash_seed, partition_halves

_WORKER: dict[str, Any] = {}


def _device(name: str):
    try:
        import torch

        if str(name).startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception:
        return None


def _init_fit_worker(X, neigh, frames, sids):
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _WORKER["X"] = X
    _WORKER["neigh"] = neigh
    _WORKER["frames"] = frames
    _WORKER["sids"] = sids


def start_fit_pool(n_workers: int, X, neigh, frames, sids) -> ProcessPoolExecutor | None:
    """Spawn once; do not fork after CUDA. Reuse across replicates."""
    global _EXEC
    stop_fit_pool()
    if int(n_workers) <= 1:
        _init_fit_worker(X, neigh, frames, sids)
        return None
    ctx = mp.get_context("spawn")
    _EXEC = ProcessPoolExecutor(
        max_workers=int(n_workers),
        mp_context=ctx,
        initializer=_init_fit_worker,
        initargs=(X, neigh, frames, sids),
    )
    return _EXEC


def stop_fit_pool() -> None:
    global _EXEC
    if _EXEC is not None:
        _EXEC.shutdown(wait=True, cancel_futures=False)
        _EXEC = None


def _cpu():
    try:
        import torch

        return torch.device("cpu")
    except Exception:
        return None


def _conditional_one(payload: tuple[int, int]) -> dict[str, Any]:
    i, replicate = payload
    X = _WORKER["X"]
    neigh = _WORKER["neigh"]
    x0, J = _WORKER["frames"][i]
    sid = int(_WORKER["sids"][i])
    N = np.asarray(neigh[i, :K], dtype=np.int64)
    Xloc = np.asarray(X[N], dtype=np.float64)
    seed = hash_seed("A", replicate, sid)
    halfA, halfB = partition_halves(K, seed)
    fit = fit_explicit_halves(Xloc, x0, J, halfA, halfB, d=D, seed=seed, device=_cpu())
    return {
        "scheme": "conditional_support",
        "replicate": int(replicate),
        "sample_id": sid,
        "anchor_i": i,
        "k": K,
        "k_prime": K,
        "n_A": 1024,
        "n_B": 1024,
        "disjoint": True,
        "union_is_full": True,
        "support_overlap": 1.0,
        "jaccard": 1.0,
        "radius": _radius(Xloc, x0),
        "q_design_cond": design_condition(Xloc, x0, J[:, :D]),
        "projector_rel": 0.0,
        "retained_n": int(X.shape[0]),
        **fit,
    }


def _object_one(payload: tuple) -> dict[str, Any]:
    i, replicate, idx, retained_n, orig_neigh = payload
    X = _WORKER["X"]
    x0, J = _WORKER["frames"][i]
    sid = int(_WORKER["sids"][i])
    idx = np.asarray(idx, dtype=np.int64)
    Xloc = np.asarray(X[idx], dtype=np.float64)
    seed = hash_seed("B", replicate, sid)
    halfA, halfB = partition_halves(len(idx), seed)
    fit = fit_explicit_halves(Xloc, x0, J, halfA, halfB, d=D, seed=seed, device=_cpu())
    orig = set(np.asarray(orig_neigh, dtype=np.int64).tolist())
    new = set(idx.tolist())
    inter = len(orig & new)
    union = len(orig | new)
    return {
        "scheme": "object_support",
        "replicate": int(replicate),
        "sample_id": sid,
        "anchor_i": i,
        "k": K,
        "k_prime": int(len(idx)),
        "n_A": int(len(halfA)),
        "n_B": int(len(halfB)),
        "disjoint": len(set(halfA) & set(halfB)) == 0,
        "support_overlap": float(inter / K),
        "jaccard": float(inter / max(union, 1)),
        "radius": _radius(Xloc, x0),
        "q_design_cond": design_condition(Xloc, x0, J[:, :D]),
        "projector_rel": float("nan"),
        "retained_n": int(retained_n),
        "self_excluded": True,
        **fit,
    }


def global_inclusion_mask(n: int, replicate: int) -> np.ndarray:
    rng = np.random.default_rng(hash_seed("mask", int(replicate)))
    n_keep = int(round(KEEP_FRAC * n))
    take = rng.choice(n, size=n_keep, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[take] = True
    return mask


def knn_from_candidates(X: np.ndarray, query: np.ndarray, cand_idx: np.ndarray, k: int, device=None) -> np.ndarray:
    import torch

    dev = device or torch.device("cpu")
    Xt = torch.as_tensor(X[cand_idx], device=dev, dtype=torch.float32)
    qt = torch.as_tensor(query[None, :], device=dev, dtype=torch.float32)
    d = torch.cdist(qt, Xt)[0]
    k_use = min(int(k), int(d.numel()))
    idx = torch.topk(d, k=k_use, largest=False).indices.detach().cpu().numpy()
    return np.asarray(cand_idx[idx], dtype=np.int64)


def knn_batch(X: np.ndarray, queries: np.ndarray, cand_idx: np.ndarray, k: int, device=None) -> np.ndarray:
    import torch

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    Xt = torch.as_tensor(X[cand_idx], device=dev, dtype=torch.float32)
    Qt = torch.as_tensor(queries, device=dev, dtype=torch.float32)
    out = np.empty((len(queries), min(int(k), len(cand_idx))), dtype=np.int64)
    bs = 64
    kk = out.shape[1]
    for s in range(0, len(queries), bs):
        d = torch.cdist(Qt[s : s + bs], Xt)
        idx = torch.topk(d, k=kk, largest=False).indices.detach().cpu().numpy()
        out[s : s + bs] = cand_idx[idx]
    return out


def projector_rel(Ja: np.ndarray, Jb: np.ndarray) -> float:
    Qa, _ = np.linalg.qr(np.asarray(Ja, dtype=np.float64), mode="reduced")
    Qb, _ = np.linalg.qr(np.asarray(Jb, dtype=np.float64), mode="reduced")
    Pa, Pb = Qa @ Qa.T, Qb @ Qb.T
    return float(np.linalg.norm(Pa - Pb) / max(np.linalg.norm(Pa), 1e-12))


def _radius(Xloc: np.ndarray, x0: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(Xloc - x0[None, :], axis=1)))


def eligible_candidates(mask: np.ndarray, query_row: int) -> np.ndarray:
    """Global inclusion mask minus the query itself. Query stays a query even if dropped from candidates."""
    cand = np.flatnonzero(np.asarray(mask, dtype=bool))
    return cand[cand != int(query_row)]


def run_conditional_replicate(
    *,
    X: np.ndarray,
    neigh: np.ndarray,
    frames: list[tuple[np.ndarray, np.ndarray]],
    sids: list[int],
    replicate: int,
    device: str,
    n_workers: int = 1,
) -> list[dict[str, Any]]:
    jobs = [(i, int(replicate)) for i in range(len(sids))]
    if _EXEC is not None:
        return list(_EXEC.map(_conditional_one, jobs, chunksize=8))
    _init_fit_worker(X, neigh, frames, sids)
    if int(n_workers) > 1:
        with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
            return list(ex.map(_conditional_one, jobs, chunksize=4))
    return [_conditional_one(j) for j in jobs]


def run_object_support_replicate(
    *,
    X: np.ndarray,
    neigh: np.ndarray,
    frames: list[tuple[np.ndarray, np.ndarray]],
    sids: list[int],
    query_rows: np.ndarray,
    replicate: int,
    device: str,
    n_workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = X.shape[0]
    mask = global_inclusion_mask(n, replicate)
    retained_n = int(mask.sum())
    Q = X[query_rows]
    cand = np.flatnonzero(mask)
    knn = knn_batch(X, Q, cand, K_PRIME + 8, device=_device(device))
    payloads = []
    for i, sid in enumerate(sids):
        qrow = int(query_rows[i])
        idx = knn[i]
        idx = idx[idx != qrow][:K_PRIME]
        if idx.size < K_PRIME:
            extra = eligible_candidates(mask, qrow)
            extra = extra[~np.isin(extra, idx)]
            need = K_PRIME - idx.size
            if need > 0 and extra.size:
                idx = np.concatenate([idx, extra[:need]])
        idx = idx[:K_PRIME]
        payloads.append((i, int(replicate), idx, retained_n, np.asarray(neigh[i, :K], dtype=np.int64)))
    if _EXEC is not None:
        rows = list(_EXEC.map(_object_one, payloads, chunksize=8))
    else:
        _init_fit_worker(X, neigh, frames, sids)
        if int(n_workers) > 1:
            with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
                rows = list(ex.map(_object_one, payloads, chunksize=4))
        else:
            rows = [_object_one(p) for p in payloads]
    for row, i in zip(rows, range(len(sids))):
        qrow = int(query_rows[i])
        row["query_in_candidates"] = bool(mask[qrow])
        row["self_excluded"] = True
        row["query_always_available"] = True
    meta = {"replicate": int(replicate), "retained_n": retained_n, "mask_sum": retained_n, "k_prime": K_PRIME, "half": HALF_PRIME}
    return rows, meta
