"""Deterministic nested neighbour selection and A/B splits."""

from __future__ import annotations

import hashlib

import numpy as np


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def rank_keys(ids: np.ndarray, *, tag: str, seed: int, sample_id: int) -> np.ndarray:
    keys = np.array([_digest(f"{tag}:{int(seed)}:{int(sample_id)}:{int(i)}") for i in ids])
    return np.argsort(keys, kind="stable")


def select_m(pool: np.ndarray, m: int, *, seed: int, sample_id: int) -> np.ndarray:
    """Take the first m of a hash ranking of `pool` (nested in m)."""
    pool = np.asarray(pool, dtype=np.int64)
    m = int(m)
    if m > len(pool):
        raise ValueError(f"m={m} exceeds pool {len(pool)}")
    if m == len(pool):
        return pool.copy()
    order = rank_keys(np.arange(len(pool)), tag="select", seed=seed, sample_id=sample_id)
    return pool[order[:m]]


def split_ab(idx: np.ndarray, *, seed: int, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    order = rank_keys(np.arange(len(idx)), tag="split", seed=seed, sample_id=sample_id)
    half = len(idx) // 2
    return idx[order[:half]], idx[order[half:]]


def fit_val_seed(*, seed: int, sample_id: int, tag: str) -> int:
    return int(_digest(f"fitval:{int(seed)}:{int(sample_id)}:{tag}")[:8], 16)


select_m = select_m
split_ab = split_ab
