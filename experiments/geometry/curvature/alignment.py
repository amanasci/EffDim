"""Strict sample_id joins. Never align by row position."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def assert_unique_sample_ids(ids: Iterable[int], *, name: str = "sample_id") -> np.ndarray:
    arr = np.asarray(list(ids), dtype=np.int64)
    if len(arr) != len(set(int(x) for x in arr)):
        raise ValueError(f"duplicate {name} values")
    return arr


def join_by_sample_id(left: pd.DataFrame, right: pd.DataFrame, *, how: str = "inner") -> pd.DataFrame:
    if "sample_id" not in left.columns or "sample_id" not in right.columns:
        raise KeyError("both frames must have sample_id")
    L = left.copy()
    R = right.copy()
    L["sample_id"] = L["sample_id"].astype(int)
    R["sample_id"] = R["sample_id"].astype(int)
    assert_unique_sample_ids(L["sample_id"], name="left.sample_id")
    # right may have one row per target; caller must filter first if so
    out = L.merge(R, on="sample_id", how=how, validate="one_to_one" if how == "inner" and R.sample_id.is_unique else None)
    if how == "inner" and len(out) != len(L):
        raise RuntimeError(f"sample_id join dropped rows {len(L)}->{len(out)}")
    return out


def same_order(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)))
