"""
notebooks/pu_manifold/local_dimension.py -- D-09, D-10, D-12: the dimension layer.

Phase 02.7 manifold-template-inference-front-end-inserted. Calls all eight of
``src/effdim/geometry.py``'s FROZEN intrinsic-dimension estimators over a ``k`` sweep
(D-09), resolves a single ``d_hat`` only when a ratified majority of them plateau at the
same integer over a non-trivial contiguous ``k`` range (D-10, ``plateau_consensus``: no
consensus emits ``d_hat=None`` and a populated ``reason`` -- never a fallback median), and
computes anchor-neighbourhood local estimates whose dispersion feeds abstain condition (b)
(D-12). ``src/effdim/geometry.py`` is FROZEN -- every estimator is called here, none is
edited, duplicated or reimplemented.

**Two planning-time corrections, verified against source and measured this session, both
pinned as regression tests in ``tests/test_local_dimension.py`` rather than left as
prose:**

1. ``tle_dimensionality`` is mathematically identical to ``mle_dimensionality`` -- MLE
   computes ``(k-1) / (sum_j[log r_k - log r_j] + 1e-10)``; TLE computes
   ``u_j = r_j / r_k`` then ``(k-1) / (-sum_j log u_j + 1e-10)``: the same expression, the
   same epsilon, the same ``np.mean`` reduction. Both read from the SAME
   ``precomputed_knn_dist_sq`` array in this module. Measured this session on three fixed
   seeded regimes -- ``(n=500, d=3, D=10)``, ``(n=400, d=2, D=768)``, ``(n=300, d=5,
   D=20)`` -- ``mle == tle`` bit-for-bit at the chosen seeds. **This equality is
   data-dependent, not a mathematical guarantee**: ``geometry.compute_knn_distances``
   casts through FAISS's float32 path, and ``log(a) - log(b)`` is not bit-identical to
   ``-log(b/a)`` in floating point in general -- a scan over 20 seeds at the first regime
   found 15/20 exactly equal and 5/20 differing in the last 1-2 float32 ULPs. The specific
   seeds pinned in the test file were chosen because they reproduce exact equality, and
   that choice -- not a universal claim -- is what the regression test asserts. The
   practical consequence for D-09/D-10 is unaffected either way: MLE and TLE are the same
   estimator under two names, so the reported spread is over **seven** distinct estimators
   with one vote doubled, and D-10's majority always receives two correlated votes unless
   ``plateau_consensus``'s ``count_distinct=True`` collapses the pair.
2. ``two_nn_dimensionality`` and ``mind_mli_dimensionality`` are invariant in ``k`` by
   construction -- ``two_nn`` reads only columns 0 and 1 of ``precomputed_knn_dist_sq``
   (``geometry.py:111-113``); ``mind_mli`` reads only column 0 (``geometry.py:227``).
   Measured this session at ``k in {5, 10, 20, 30}`` on a fixed fixture: ``two_nn`` held at
   ``5.237297915123046`` and ``mind_mli`` at ``0.7714744210243225`` at every ``k``, while
   ``mle`` moved ``6.425092697143555 -> 5.545888900756836 -> 4.975106716156006 ->
   4.719031810760498`` across the same sweep. Together with ``gmst_dimensionality``, which
   accepts neither ``precomputed_knn_dist_sq`` nor ``k`` at all (``geometry.py:396-400``),
   that is **three of eight estimators that cannot vary with k** -- two of them plateau
   perfectly at every k for reasons that have nothing to do with the data. ``ESTIMATOR_NAMES``,
   ``K_INVARIANT_ESTIMATORS`` and ``DUPLICATE_ESTIMATOR_PAIRS`` below make both corrections
   machine-readable so ``plateau_consensus``'s ``count_distinct`` rule (and any future
   ratification in plan ``02.7-08``) is written against them explicitly rather than
   silently rediscovering them.

**GMST's separate call path (D-12).** ``gmst_dimensionality`` accepts neither
``precomputed_knn_dist_sq`` nor ``k`` -- it cannot be sliced onto the shared precomputed
array like the other seven. ``global_estimates`` calls it once on the raw coordinates and
repeats that single value across every swept ``k`` with ``k_applicable["gmst"] = False``,
so no reader mistakes a repeated value for a measured plateau. ``local_estimates`` calls it
directly on each anchor's raw local-subset coordinates (``provenance="recomputed"``) rather
than slicing a precomputed array (``provenance="sliced"`` for the other seven) -- its local
value is therefore NOT bit-identical in provenance to its own global call, because it
rebuilds its own internal kNN structure on the smaller subset. This asymmetry is recorded
in the data, not smoothed over.

**This module emits no mean, median, mode, or any other aggregate over the eight
estimators anywhere.** D-09's whole point is that disagreement is the signal --
``02-FINDINGS.md`` Sec 6.4 records ``d_frozen = 5`` as the outlier of four estimates, three
of which clustered at 18-25, and averaging that disagreement away is what produced the
suspect number. ``spread`` and ``dispersion`` below are descriptive only and apply no bar
of their own; every threshold that decides whether a spread or a dispersion counts as
"too wide" is a ratified constant plan ``02.7-08`` fixes and the caller supplies.

Arrays in, dicts out -- no file I/O, no cache handling, per ``persistence_probe.py``'s own
module contract.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from effdim import geometry

from . import geodesic_graph

# --- the eight frozen estimators and the two verified corrections, machine-readable ------

ESTIMATOR_NAMES: Tuple[str, ...] = (
    "mle",
    "two_nn",
    "danco",
    "mind_mli",
    "mind_mlk",
    "ess",
    "tle",
    "gmst",
)
"""The ordered 8-tuple of every frozen estimator this module calls, exactly as named in
``src/effdim/geometry.py``."""

K_INVARIANT_ESTIMATORS: Tuple[str, ...] = ("two_nn", "mind_mli", "gmst")
"""Verified this session (see module docstring correction 2): these three cannot vary with
``k`` by construction. ``plateau_consensus`` excludes them from the run-length requirement
under ``count_distinct=True`` -- their "plateau" is vacuous, not evidence about the data."""

DUPLICATE_ESTIMATOR_PAIRS: Tuple[Tuple[str, str], ...] = (("mle", "tle"),)
"""Verified this session (see module docstring correction 1): ``tle`` is Levina-Bickel MLE
under another name. ``plateau_consensus`` collapses each pair to one vote under
``count_distinct=True``."""

_GMST_MIN_POINTS = 10
"""``geometry.gmst_dimensionality`` returns ``0.0`` -- not a real estimate -- for fewer
than this many points (``geometry.py:406-408``). ``local_estimates`` validates
``neighbourhood_size`` against this floor with an explicit ``ValueError``."""

_ESTIMATOR_FUNCS: Dict[str, Any] = {
    "mle": geometry.mle_dimensionality,
    "two_nn": geometry.two_nn_dimensionality,
    "danco": geometry.danco_dimensionality,
    "mind_mli": geometry.mind_mli_dimensionality,
    "mind_mlk": geometry.mind_mlk_dimensionality,
    "ess": geometry.ess_dimensionality,
    "tle": geometry.tle_dimensionality,
    "gmst": geometry.gmst_dimensionality,
}
"""Private dispatch table. Every value here is imported unchanged from the frozen
``src/effdim/geometry.py`` -- none is wrapped, edited or reimplemented."""


# --- D-09: all eight, every k, no aggregation ----------------------------------------------


def global_estimates(
    data: np.ndarray, k_values: Sequence[int]
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, bool]]:
    """All eight frozen estimators, evaluated at every ``k`` in ``k_values``, with **no**
    mean, median, mode or any other reduction over them -- D-09's spread-reported,
    no-aggregation rule.

    ``geometry.compute_knn_distances(data, max(k_values))`` is computed exactly ONCE, then
    sliced to each swept ``k``'s first ``k`` columns and passed as
    ``precomputed_knn_dist_sq`` to the seven estimators that accept it (including
    ``two_nn`` and ``mind_mli``, which accept the array but structurally ignore all but its
    first one or two columns -- see module docstring correction 2). ``gmst_dimensionality``
    accepts neither parameter; it is called once on the raw coordinates and that single
    value is repeated across the sweep.

    Returns ``(estimates_by_k, k_applicable)``: ``estimates_by_k`` is
    ``{k: {estimator_name: value}}``; ``k_applicable`` is ``{estimator_name: bool}``,
    ``False`` only for ``"gmst"`` -- so no caller mistakes GMST's repeated value for a
    measured plateau.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(
            f"global_estimates: data must be a 2-d (n, d) array, got shape {data.shape!r}"
        )
    k_values = list(k_values)
    if len(k_values) == 0:
        raise ValueError("global_estimates: k_values must be non-empty")
    if any(kk < 2 for kk in k_values):
        raise ValueError(f"global_estimates: every k must be >= 2, got {k_values!r}")

    k_max = max(k_values)
    dist_sq_global = geometry.compute_knn_distances(data, k_max)
    if dist_sq_global.shape[1] < k_max:
        raise ValueError(
            f"global_estimates: requested k_max={k_max} exceeds the available neighbour "
            f"width ({dist_sq_global.shape[1]}); reduce k_values or supply more points"
        )

    gmst_value = float(geometry.gmst_dimensionality(data))

    k_applicable: Dict[str, bool] = {name: (name != "gmst") for name in ESTIMATOR_NAMES}

    estimates_by_k: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        dist_sq_k = dist_sq_global[:, :k]
        values: Dict[str, float] = {}
        for name in ESTIMATOR_NAMES:
            if name == "gmst":
                values[name] = gmst_value
            else:
                func = _ESTIMATOR_FUNCS[name]
                values[name] = float(func(data, precomputed_knn_dist_sq=dist_sq_k))
        estimates_by_k[int(k)] = values

    return estimates_by_k, k_applicable


def spread(estimates_at_k: Dict[str, float]) -> Dict[str, Any]:
    """Descriptive spread of one ``k``'s eight named estimates -- ``{"min", "max",
    "range", "distinct_count", "values"}``. Applies NO bar of its own; the spread bound
    that fires abstain condition (b) is a ratified constant the caller supplies and
    compares against separately."""
    names = list(estimates_at_k.keys())
    if len(names) == 0:
        raise ValueError("spread: estimates_at_k must be non-empty")
    vals = np.array([estimates_at_k[name] for name in names], dtype=np.float64)
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.shape[0] == 0:
        raise ValueError("spread: estimates_at_k contains no finite values")
    return {
        "min": float(finite_vals.min()),
        "max": float(finite_vals.max()),
        "range": float(finite_vals.max() - finite_vals.min()),
        "distinct_count": int(len(np.unique(finite_vals))),
        "values": dict(estimates_at_k),
    }


# --- D-10: plateau-and-consensus, no fallback aggregate -------------------------------------


def _bin_to_integer(value: float, tolerance: float) -> float:
    """Round ``value`` to the nearest integer if it lands within ``tolerance`` of one,
    else ``nan`` (meaning: does not bin cleanly to any integer at this tolerance, and so
    cannot join any plateau run)."""
    if not np.isfinite(value):
        return float("nan")
    nearest = round(value)
    if abs(value - nearest) <= tolerance:
        return float(nearest)
    return float("nan")


def _apply_count_distinct(supporters: List[str]) -> Tuple[List[str], List[str]]:
    """Collapse each :data:`DUPLICATE_ESTIMATOR_PAIRS` pair present in full within
    ``supporters`` down to its first-named member. Returns ``(counted, excluded)``."""
    supporters_set = set(supporters)
    dropped: set = set()
    for pair in DUPLICATE_ESTIMATOR_PAIRS:
        if all(member in supporters_set for member in pair):
            dropped.update(pair[1:])
    counted = [name for name in supporters if name not in dropped]
    excluded = [name for name in supporters if name in dropped]
    return counted, excluded


def plateau_consensus(
    estimates_by_k: Dict[int, Dict[str, float]],
    majority: int,
    min_run: int,
    tolerance: float,
    count_distinct: bool,
) -> Dict[str, Any]:
    """D-10's plateau-and-consensus rule. All five arguments are REQUIRED, no defaults --
    none of them is a value this module chooses.

    Each estimator's value at every swept ``k`` is rounded/binned to an integer under
    ``tolerance`` (:func:`_bin_to_integer`); each estimator's longest contiguous run of
    equal binned integers is found via ``geodesic_graph.contiguous_stable_range`` (reused,
    not reimplemented). ``d_hat`` is set to the integer at which at least ``majority``
    estimators plateau over a run of at least ``min_run`` -- never a fallback median.

    When ``count_distinct`` is ``True``: any :data:`DUPLICATE_ESTIMATOR_PAIRS` pair that
    both support the same integer counts once toward the majority tally, and the three
    :data:`K_INVARIANT_ESTIMATORS` are exempted from the ``min_run`` requirement (their
    "run" spans the whole sweep vacuously, by construction, not because the data
    stabilised).

    Returns ``{"d_hat", "reason", "supporting_estimators", "run", "excluded_estimators"}``.
    ``d_hat`` is ``None`` with a populated ``reason`` whenever no single integer reaches
    the required majority -- D-10 is explicit that no consensus means no number is
    emitted and abstain (d) fires, never a fallback aggregate.
    """
    k_sorted = sorted(estimates_by_k.keys())
    if len(k_sorted) == 0:
        raise ValueError("plateau_consensus: estimates_by_k must be non-empty")

    runs: Dict[str, Dict[str, Any]] = {}
    support: Dict[str, Optional[float]] = {}

    for name in ESTIMATOR_NAMES:
        binned_seq = [_bin_to_integer(estimates_by_k[k][name], tolerance) for k in k_sorted]
        run = geodesic_graph.contiguous_stable_range(binned_seq)
        runs[name] = run

        run_value = run["value"]
        if isinstance(run_value, float) and np.isnan(run_value):
            support[name] = None
            continue

        exempt_from_min_run = count_distinct and (name in K_INVARIANT_ESTIMATORS)
        if exempt_from_min_run or run["length"] >= min_run:
            support[name] = run_value
        else:
            support[name] = None

    votes: Dict[float, List[str]] = {}
    for name, val in support.items():
        if val is None:
            continue
        votes.setdefault(val, []).append(name)

    excluded_estimators: List[str] = []
    effective_votes: Dict[float, List[str]] = {}
    for val, supporters in votes.items():
        if count_distinct:
            counted, excluded = _apply_count_distinct(supporters)
            excluded_estimators.extend(excluded)
        else:
            counted = list(supporters)
        effective_votes[val] = counted

    qualifying = [
        (val, supporters)
        for val, supporters in effective_votes.items()
        if len(supporters) >= majority
    ]

    if len(qualifying) != 1:
        vote_counts = {val: len(supporters) for val, supporters in effective_votes.items()}
        if len(qualifying) == 0:
            reason = (
                f"no integer reached the required majority of {majority} "
                f"(effective vote counts: {vote_counts!r})"
            )
        else:
            reason = (
                f"more than one integer reached the required majority of {majority} "
                f"(effective vote counts: {vote_counts!r}); no unambiguous consensus"
            )
        return {
            "d_hat": None,
            "reason": reason,
            "supporting_estimators": [],
            "run": {},
            "excluded_estimators": sorted(set(excluded_estimators)),
        }

    best_value, _ = qualifying[0]
    winning_raw_supporters = sorted(votes[best_value])

    return {
        "d_hat": int(round(best_value)),
        "reason": "",
        "supporting_estimators": winning_raw_supporters,
        "run": {name: runs[name] for name in winning_raw_supporters},
        "excluded_estimators": sorted(set(excluded_estimators)),
    }


# --- D-12: local d_hat by anchor-neighbourhood slicing, gmst's separate path ----------------


def local_estimates(
    data: np.ndarray,
    k: int,
    anchor_indices: Sequence[int],
    neighbourhood_size: int,
    precomputed_knn_dist_sq: np.ndarray,
) -> Dict[str, Any]:
    """Local ``d_hat`` at each of ``anchor_indices``, by calling the frozen estimators on
    anchor-neighbourhood subsets -- D-12.

    For the seven estimators accepting ``precomputed_knn_dist_sq``, this takes the
    IDENTICAL global array the caller already computed (e.g. via :func:`global_estimates`)
    and passes a row-slice, ``precomputed_knn_dist_sq[local_idx]``, alongside
    ``data[local_idx]`` -- never a recomputation, so a local-versus-global disagreement is
    about the data, not about two implementations. Each anchor's neighbourhood is the
    ``neighbourhood_size`` spatially nearest points to it (by ambient Euclidean distance,
    including the anchor itself).

    ``gmst_dimensionality`` is the verified exception (see module docstring): it accepts
    neither ``precomputed_knn_dist_sq`` nor ``k``, so it is called directly on the raw
    local-subset coordinates -- its provenance is ``"recomputed"``, not ``"sliced"``, and
    it is NOT bit-identical in provenance to its own global call, because it rebuilds its
    own internal kNN structure on the smaller subset. Because ``gmst_dimensionality``
    returns ``0.0`` (not a real estimate) under 10 points, ``neighbourhood_size`` is
    validated against that floor with an explicit ``ValueError``.

    Returns ``{"anchor_indices", "k", "neighbourhood_size", "by_anchor"}``, where
    ``by_anchor`` is ``{anchor_idx: {estimator_name: {"value", "provenance"}}}`` --
    ``anchor_indices`` is echoed back in the result so a run is auditable against whatever
    seed the caller used to choose them.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(
            f"local_estimates: data must be a 2-d (n, d) array, got shape {data.shape!r}"
        )
    n = data.shape[0]
    precomputed_knn_dist_sq = np.asarray(precomputed_knn_dist_sq)
    if precomputed_knn_dist_sq.shape[0] != n:
        raise ValueError(
            "local_estimates: precomputed_knn_dist_sq must have one row per data point "
            f"(data has {n} rows, precomputed_knn_dist_sq has "
            f"{precomputed_knn_dist_sq.shape[0]})"
        )
    if neighbourhood_size < _GMST_MIN_POINTS:
        raise ValueError(
            f"local_estimates: neighbourhood_size={neighbourhood_size} is below "
            f"gmst_dimensionality's own floor of {_GMST_MIN_POINTS} points -- gmst "
            "returns 0.0 (not a real estimate), not an error, under that floor, so this "
            "guard prevents silently reporting a fake zero as a local estimate"
        )
    anchor_indices = list(anchor_indices)
    if len(anchor_indices) == 0:
        raise ValueError("local_estimates: anchor_indices must be non-empty")

    by_anchor: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for anchor_idx in anchor_indices:
        distances_to_anchor = np.linalg.norm(data - data[anchor_idx], axis=1)
        local_idx = np.argsort(distances_to_anchor)[:neighbourhood_size]

        entry: Dict[str, Dict[str, Any]] = {}
        for name in ESTIMATOR_NAMES:
            if name == "gmst":
                value = float(geometry.gmst_dimensionality(data[local_idx]))
                provenance = "recomputed"
            else:
                func = _ESTIMATOR_FUNCS[name]
                value = float(
                    func(
                        data[local_idx],
                        precomputed_knn_dist_sq=precomputed_knn_dist_sq[local_idx],
                    )
                )
                provenance = "sliced"
            entry[name] = {"value": value, "provenance": provenance}
        by_anchor[int(anchor_idx)] = entry

    return {
        "anchor_indices": [int(a) for a in anchor_indices],
        "k": int(k),
        "neighbourhood_size": int(neighbourhood_size),
        "by_anchor": by_anchor,
    }


def dispersion(local_by_anchor: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Per-estimator dispersion of :func:`local_estimates`'s ``by_anchor`` result across
    anchors -- ``{estimator_name: {"min", "max", "range", "iqr", "n_anchors"}}``. Reported
    strictly per estimator, never pooled across estimators, and applying no bar of its own
    -- the dispersion bound that fires abstain condition (b) is a ratified constant the
    caller supplies and compares against separately."""
    by_anchor = local_by_anchor["by_anchor"]
    anchor_ids = list(by_anchor.keys())
    if len(anchor_ids) == 0:
        raise ValueError("dispersion: local_by_anchor has no anchors")

    result: Dict[str, Dict[str, float]] = {}
    for name in ESTIMATOR_NAMES:
        vals = np.array(
            [by_anchor[anchor_id][name]["value"] for anchor_id in anchor_ids],
            dtype=np.float64,
        )
        q1, q3 = np.percentile(vals, [25, 75])
        result[name] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "range": float(vals.max() - vals.min()),
            "iqr": float(q3 - q1),
            "n_anchors": int(vals.shape[0]),
        }
    return result
