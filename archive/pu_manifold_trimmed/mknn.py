"""Phase 4 regional crossmodal MKNN stubs.

Metric (Chechik et al. 2010, adopted verbatim by arXiv:2509.19453):

    MKNN(z1, z2) = k^-1 * |N_k(z1) intersect N_k(z2)|

with N_k(z) the k-NN index set within z's own embedding space. Label-free and
training-free — works on this dataset's row-aligned, join-key-free HSC/Legacy pairing.

Known caveat (MKNN-08): k-NN alignment metrics are hubness-sensitive in high
dimensions; state this alongside any MKNN result. No module-level faiss import.
"""

from typing import Any, Dict

import numpy as np
from scipy.stats import bootstrap, permutation_test, skew
from sklearn.neighbors import NearestNeighbors


def _membership_matrix(Z: np.ndarray, k: int) -> np.ndarray:
    """(n, n) boolean k-NN membership matrix: row i marks i's k nearest neighbours,
    self excluded. Always `NearestNeighbors(n_neighbors=k + 1)`, never `n_neighbors=k`
    — this codebase's fixed, three-times-repeated convention (curvature_probe.py)."""
    Z = np.asarray(Z, dtype=np.float64)
    if not np.all(np.isfinite(Z)):
        raise ValueError("_membership_matrix: Z contains a non-finite value.")
    n = Z.shape[0]
    if n < 2:
        raise ValueError(f"_membership_matrix: Z has n={n} rows, need at least 2.")
    if k < 1:
        raise ValueError(f"_membership_matrix: k={k} must be >= 1.")
    if k + 1 > n:
        raise ValueError(
            f"_membership_matrix: k={k} + 1 exceeds n={n} rows; cannot find k neighbours."
        )
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(Z)
    _, idx = nbrs.kneighbors(Z)  # idx[:, 0] is the point itself
    neighbor_idx = idx[:, 1:]  # (n, k), self excluded

    membership = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    membership[rows, neighbor_idx.ravel()] = True
    return membership


def mknn_score(z1: Any, z2: Any, k: Any) -> Any:
    """Mean MKNN score over all points. Caller guarantees row alignment — there is
    no object_id in this dataset to catch a mismatch."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    if z1.shape[0] != z2.shape[0]:
        raise ValueError(
            f"mknn_score: z1 has {z1.shape[0]} rows but z2 has {z2.shape[0]} rows; "
            "rows must be row-aligned."
        )
    A = _membership_matrix(z1, k)
    B = _membership_matrix(z2, k)
    return float(((A & B).sum(axis=1) / k).mean())


def permutation_null(
    z1: Any, z2: Any, k: Any, n_permutations: Any, seed: Any, quantile: Any
) -> Dict[str, Any]:
    """Permutation-null MKNN distribution, drawn *within* the region's own index set —
    a global null would not control for the region's local density.

    `quantile` has NO default value: a pre-registered constant, echoed into the
    returned dict rather than silently inherited (mirrors
    `curvature_probe.permutation_null`'s own `quantile` argument).

    Builds both membership matrices ONCE, then drives
    `scipy.stats.permutation_test` with `permutation_type="pairings"` (passed
    explicitly — the default silently computes a different null). `_stat(idx1,
    idx2)` treats both arguments purely as index arrays into the precomputed
    membership matrices and never assumes either is the caller's original,
    unshuffled array — no k-NN query occurs inside the resampling loop.
    """
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    if z1.shape[0] != z2.shape[0]:
        raise ValueError(
            f"permutation_null: z1 has {z1.shape[0]} rows but z2 has {z2.shape[0]} rows; "
            "rows must be row-aligned."
        )
    n = z1.shape[0]
    A = _membership_matrix(z1, k)
    B = _membership_matrix(z2, k)

    def _stat(idx1: np.ndarray, idx2: np.ndarray) -> float:
        idx1 = np.asarray(idx1, dtype=np.intp)
        idx2 = np.asarray(idx2, dtype=np.intp)
        return float(((A[idx1] & B[idx2]).sum(axis=1) / k).mean())

    rng = np.random.default_rng(seed)
    result = permutation_test(
        (np.arange(n), np.arange(n)),
        _stat,
        permutation_type="pairings",
        alternative="greater",
        n_resamples=n_permutations,
        rng=rng,
    )

    observed_score = float(result.statistic)
    null_threshold = float(np.quantile(result.null_distribution, quantile))

    return {
        "observed_score": observed_score,
        "null_distribution": result.null_distribution,
        "p_value": float(result.pvalue),
        "null_mean": float(np.mean(result.null_distribution)),
        "null_std": float(np.std(result.null_distribution)),
        "null_threshold": null_threshold,
        "null_quantile": float(quantile),
        "clears_null": bool(observed_score > null_threshold),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "n": int(n),
        "k": int(k),
    }


def bootstrap_ci(
    z1: Any, z2: Any, k: Any, n_resamples: Any, seed: Any, confidence_level: Any
) -> Dict[str, Any]:
    """Bootstrap (low, high) CI on the regional MKNN score, resampling within region.

    `n_resamples` and `confidence_level` are required arguments with no default.
    Resamples POINTS (never the pairing) from a fixed `(n,)` per-point overlap
    array computed once — no k-NN query occurs inside the resampling loop.
    """
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    if z1.shape[0] != z2.shape[0]:
        raise ValueError(
            f"bootstrap_ci: z1 has {z1.shape[0]} rows but z2 has {z2.shape[0]} rows; "
            "rows must be row-aligned."
        )
    A = _membership_matrix(z1, k)
    B = _membership_matrix(z2, k)
    n = z1.shape[0]
    per_point = (A & B).sum(axis=1) / k

    rng = np.random.default_rng(seed)
    result = bootstrap(
        (per_point,),
        np.mean,
        method="percentile",
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )
    ci_low = float(result.confidence_interval.low)
    ci_high = float(result.confidence_interval.high)

    return {
        "score": float(per_point.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "degenerate": bool(ci_low == ci_high),
        "confidence_level": float(confidence_level),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "n": int(n),
        "k": int(k),
    }


def hubness_skewness(Z: Any, k: Any) -> float:
    """k-occurrence skewness (Radovanovic, Nanopoulos and Ivanovic, JMLR 2010) —
    the skewness of the column sums (k-occurrence in-degree) of the k-NN membership
    matrix, computed at zero extra k-NN cost since the matrix is already built."""
    membership = _membership_matrix(Z, k)
    return float(skew(membership.sum(axis=0)))


def chance_floor(n: Any, k: Any) -> float:
    """The k/n chance floor an MKNN score is compared against."""
    return float(k) / float(n)
