"""Phase 5 curvature-conditioned linear decodability: probe fit/score, seed pooling, bucketing
and verdict functions, plus the pre-registration constants block and its guard.

(a) **D5-03's own citation is corrected here, not silently followed.** D5-03 names
``notebooks/pu_manifold/decoder_curvature.py`` as the decoder-side curvature source. That
module's own docstring states it is ``chart_curvature.py`` with the
``chart_decoders[chart_idx]`` two-hop composition removed, built for Phase 02.6's
no-chart-index substrates (a plain autoencoder and a ``PlainAutoEncoder`` trained under
``topoae.train_topoae``, both decoding through one smooth MLP with no chart index at all).
``ChartAutoEncoder`` is chart-routed and has no bare single-hop decode entry point matching
that module's signature. The function this phase actually uses, matching the CAE Phase 3
built, is ``chart_curvature.chart_curvature_field(model, x, mode="reverse")`` -- frozen as
the named constant ``CURVATURE_SOURCE_FUNCTION`` (unset here, filled at the 05-04 freeze) so
the correction is auditable rather than buried in a comment. This module imports nothing
from ``decoder_curvature``, and the runner does not either.

(b) **A raw ``np.mean`` across the three seeds' fields would not be a fair average.** The
measured ``||H||`` medians across the three sealed CAE seeds are 1359.0, 51437.9 and 70794.1
-- a 52-fold range, with two of the three fields piecewise-constant on collapsed metrics.
Weighted naively by raw magnitude that is roughly 1.1 percent / 41.6 percent / 57.3 percent
seed influence: one seed would be almost entirely averaged away and the "pooled" field would
mostly reflect whichever seed happens to have the largest raw scale, not a genuine consensus.
:func:`pool_seed_fields` therefore takes its normalization method as a REQUIRED argument with
no default -- a default is exactly how a pooling rule would get inherited by accident instead
of chosen explicitly and frozen at 05-04 (D5-04).

(c) **D5-11's accepted gap.** The field this phase splits on has no demonstrated relationship
to true curvature: the sealed ``d=20`` decoder row is
``rank_spearman_rho = -0.015106571347065712`` against the only analytic-curvature control
that tests it (spike-findings-effdim, ``high-d-curvature-feasibility.md``), and direction is
near a coin flip (52-75 percent of points anti-aligned). A Swiss roll / low-``d`` anchor was
offered and declined for this phase. Any relationship Phase 5 measures between ``||H||`` and
probe residual therefore cannot be attributed to curvature by anything in this phase --
stated here, in this module's own words, and not only by cross-reference.

(d) **D5-12's inherited chain.** The CAE underlying every field this module consumes failed
its own validity gate (``CAE_VERDICT = FAIL``, Phase 02.2); Phase 3 ran on a deliberate
override of that gate; Phase 03.1 found the pullback metric fully repaired by a ``scale``
prior while the curvature ordering only partially and non-seed-consistently moved. Every
number this module's functions eventually help produce inherits that chain.

No file I/O happens in this module -- every function is pure, operates on arrays the caller
already has in memory, and every pre-registered value is passed in by the caller with no
default (mirroring ``region_partition.py``'s own stated convention: a default is how a
pre-registered value gets inherited by accident instead of by an explicit call-site choice).
"""

from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import bootstrap
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score


# --- Pre-registration (D5-09, ratified at plan 05-04's blocking checkpoint) -----------------
#
# WRITTEN UNSET. Every constant below, and VERDICT_RULE's full text, are filled EXACTLY ONCE
# at plan 05-04's blocking decision checkpoint -- BEFORE any PU probe number exists. Amending
# any of them after a PU probe number has been computed invalidates the phase: a rule chosen
# after seeing the numbers is a rationalization, not a pre-registration. See
# `.planning/phases/05-curvature-conditioned-linear-decodability/05-PREREGISTRATION.md` for
# the full committed record once it exists.
#
# Unset convention: descriptive-text string constants are "", numeric constants are None,
# tuple constants are None. POOLING_METHOD, BUCKET_EDGES and N_BUCKETS -- the three constants
# that gate whether the bucketed path can run at all -- are all `None` while unset, so no
# truthy-but-meaningless placeholder can be mistaken for a real value. With every constant
# below unset, `assert_preregistered()` raises today, which is the point: the bucketed path
# is structurally dead until the freeze.

TRAIN_FRACTION = None
SPLIT_SEED = None
SPLIT_RULE = ""
RIDGE_ALPHA_GRID = None
RIDGE_SELECTION_RULE = ""
ALPHA_PER_TARGET = None
FIT_INTERCEPT = None
EMBEDDING_PREPROCESSING = ""
RESIDUAL_METRIC = ""
R2_MULTIOUTPUT = ""
N_BUCKETS = None
BUCKET_RULE = ""
BUCKET_EDGES = None
POOLING_METHOD = None
SEED_STEMS = None
N_CHARTS = None
CURVATURE_MODE = ""
CURVATURE_CONVENTION = ""
CURVATURE_SOURCE_FUNCTION = ""
SIZE_MATCH_RULE = ""
SIZE_MATCH_N_REPEATS = None
SIZE_MATCH_SEED = None
N_BOOTSTRAP = None
BOOTSTRAP_SEED = None
CONFIDENCE_LEVEL = None
K_DENSITY = None
FIELD_D = None
VERDICT_RULE = ""
PREREGISTRATION_PATH = ""


def assert_preregistered() -> None:
    """Raise ``RuntimeError`` unless the pre-registration is intact. Checks, in order, one
    check per constant, raising on the FIRST failing check (``region_partition.py``'s own
    idiom): ``VERDICT_RULE`` is a non-empty string naming ``N_BUCKETS``; ``N_BUCKETS`` is a
    positive int; ``TRAIN_FRACTION`` is a float strictly inside ``(0, 1)``; ``SPLIT_SEED`` is
    a positive int; ``RIDGE_ALPHA_GRID`` is a non-empty tuple of positive floats;
    ``POOLING_METHOD`` is a non-empty string; ``BUCKET_EDGES`` is a tuple of
    ``N_BUCKETS - 1`` finite floats in strictly ascending order; ``SEED_STEMS`` is a tuple of
    three positive ints; ``CURVATURE_CONVENTION`` equals ``"trace"``;
    ``CURVATURE_SOURCE_FUNCTION`` is a non-empty string. Called at the top of the runner's
    ``--mode bucketed`` branch so that path fails loudly rather than computing anything when
    the pre-registration is absent or malformed.
    """
    if not isinstance(VERDICT_RULE, str) or not VERDICT_RULE.strip():
        raise RuntimeError(
            f"assert_preregistered: VERDICT_RULE={VERDICT_RULE!r} is empty or not a string."
        )
    if "N_BUCKETS" not in VERDICT_RULE:
        raise RuntimeError(
            f"assert_preregistered: VERDICT_RULE={VERDICT_RULE!r} does not name N_BUCKETS."
        )
    if not isinstance(N_BUCKETS, int) or isinstance(N_BUCKETS, bool) or N_BUCKETS <= 0:
        raise RuntimeError(f"assert_preregistered: N_BUCKETS={N_BUCKETS!r} is not a positive int.")
    if not isinstance(TRAIN_FRACTION, float) or not (0.0 < TRAIN_FRACTION < 1.0):
        raise RuntimeError(
            f"assert_preregistered: TRAIN_FRACTION={TRAIN_FRACTION!r} is not a float strictly "
            "inside (0, 1)."
        )
    if not isinstance(SPLIT_SEED, int) or isinstance(SPLIT_SEED, bool) or SPLIT_SEED <= 0:
        raise RuntimeError(
            f"assert_preregistered: SPLIT_SEED={SPLIT_SEED!r} is not a positive int."
        )
    if (
        not isinstance(RIDGE_ALPHA_GRID, tuple)
        or len(RIDGE_ALPHA_GRID) == 0
        or not all(isinstance(v, float) and v > 0.0 for v in RIDGE_ALPHA_GRID)
    ):
        raise RuntimeError(
            f"assert_preregistered: RIDGE_ALPHA_GRID={RIDGE_ALPHA_GRID!r} is not a non-empty "
            "tuple of positive floats."
        )
    if not isinstance(POOLING_METHOD, str) or not POOLING_METHOD.strip():
        raise RuntimeError(
            f"assert_preregistered: POOLING_METHOD={POOLING_METHOD!r} is empty or not a string."
        )
    if (
        not isinstance(BUCKET_EDGES, tuple)
        or len(BUCKET_EDGES) != N_BUCKETS - 1
        or not all(isinstance(v, float) and np.isfinite(v) for v in BUCKET_EDGES)
        or list(BUCKET_EDGES) != sorted(BUCKET_EDGES)
        or len(set(BUCKET_EDGES)) != len(BUCKET_EDGES)
    ):
        raise RuntimeError(
            f"assert_preregistered: BUCKET_EDGES={BUCKET_EDGES!r} is not a tuple of "
            f"{None if N_BUCKETS is None else N_BUCKETS - 1} finite floats in strictly "
            "ascending order."
        )
    if (
        not isinstance(SEED_STEMS, tuple)
        or len(SEED_STEMS) != 3
        or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in SEED_STEMS)
    ):
        raise RuntimeError(
            f"assert_preregistered: SEED_STEMS={SEED_STEMS!r} is not a tuple of three positive "
            "ints."
        )
    if CURVATURE_CONVENTION != "trace":
        raise RuntimeError(
            f"assert_preregistered: CURVATURE_CONVENTION={CURVATURE_CONVENTION!r} does not "
            'equal "trace".'
        )
    if not isinstance(CURVATURE_SOURCE_FUNCTION, str) or not CURVATURE_SOURCE_FUNCTION.strip():
        raise RuntimeError(
            f"assert_preregistered: CURVATURE_SOURCE_FUNCTION={CURVATURE_SOURCE_FUNCTION!r} "
            "is empty or not a string."
        )


# --- D5-01/D5-02: the probe itself -----------------------------------------------------------


def train_test_split_indices(
    n: int, train_fraction: float, split_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """One permutation of ``np.arange(n)`` under ``np.random.default_rng(split_seed)``; the
    first ``round(n * train_fraction)`` indices (of the permutation) are train, the rest test.
    Both returned index arrays are sorted ascending so downstream row alignment is readable.
    ``train_fraction`` and ``split_seed`` are required arguments with no default.
    """
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"train_test_split_indices: n={n} must be an int >= 2.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(
            f"train_test_split_indices: train_fraction={train_fraction} must be in (0, 1)."
        )
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(int(n))
    n_train = int(round(n * train_fraction))
    n_train = min(max(n_train, 1), n - 1)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])
    return train_idx, test_idx


def fit_probe(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha_grid: Any,
    alpha_per_target: bool,
    fit_intercept: bool,
) -> Dict[str, Any]:
    """Wraps ``sklearn.linear_model.RidgeCV(alphas=alpha_grid,
    alpha_per_target=alpha_per_target, fit_intercept=fit_intercept)``. Never hand-rolls a CV
    loop or a least-squares solver. Returns a flat dict carrying the fitted estimator and
    everything a caller or test might want to inspect without refitting.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    if X_train.ndim != 2:
        raise ValueError(f"fit_probe: X_train must be two-dimensional, got shape {X_train.shape}.")
    if Y_train.ndim != 2:
        raise ValueError(f"fit_probe: Y_train must be two-dimensional, got shape {Y_train.shape}.")
    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError(
            f"fit_probe: X_train has {X_train.shape[0]} rows but Y_train has "
            f"{Y_train.shape[0]} rows."
        )
    if not np.all(np.isfinite(X_train)):
        raise ValueError("fit_probe: X_train contains a non-finite value.")
    if not np.all(np.isfinite(Y_train)):
        raise ValueError("fit_probe: Y_train contains a non-finite value.")
    alpha_grid = tuple(float(a) for a in alpha_grid)
    if len(alpha_grid) == 0:
        raise ValueError("fit_probe: alpha_grid must be non-empty.")

    estimator = RidgeCV(
        alphas=alpha_grid, alpha_per_target=alpha_per_target, fit_intercept=fit_intercept
    )
    estimator.fit(X_train, Y_train)

    return {
        "estimator": estimator,
        "coef_shape": tuple(np.asarray(estimator.coef_).shape),
        "intercept_shape": tuple(np.asarray(estimator.intercept_).shape),
        "alpha_": estimator.alpha_,
        "alpha_grid": alpha_grid,
        "alpha_per_target": bool(alpha_per_target),
        "fit_intercept": bool(fit_intercept),
        "n_train": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_targets": int(Y_train.shape[1]),
    }


def predict_probe(fit: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """``fit["estimator"].predict(X)``."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"predict_probe: X must be two-dimensional, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("predict_probe: X contains a non-finite value.")
    return fit["estimator"].predict(X)


def per_point_residuals(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """The length-``n`` array ``np.sum((Y_true - Y_pred) ** 2, axis=1)``, the per-point
    squared L2 residual. Pairing this with :func:`aggregate_r2` is what satisfies
    05-CONTEXT.md's "R-squared and per-point residual derivable from one underlying quantity"
    constraint.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    if Y_true.shape != Y_pred.shape:
        raise ValueError(
            f"per_point_residuals: Y_true has shape {Y_true.shape} but Y_pred has shape "
            f"{Y_pred.shape}."
        )
    if Y_true.ndim != 2:
        raise ValueError(
            f"per_point_residuals: Y_true/Y_pred must be two-dimensional, got shape "
            f"{Y_true.shape}."
        )
    if not np.all(np.isfinite(Y_true)) or not np.all(np.isfinite(Y_pred)):
        raise ValueError("per_point_residuals: Y_true or Y_pred contains a non-finite value.")
    return np.sum((Y_true - Y_pred) ** 2, axis=1)


def aggregate_r2(Y_true: np.ndarray, Y_pred: np.ndarray, multioutput: str) -> float:
    """``sklearn.metrics.r2_score(Y_true, Y_pred, multioutput=multioutput)``. RESEARCH A3
    claims this equals ``1 - per_point_residuals(...).sum() / sum((Y_true - Y_true.mean(0))
    ** 2)`` exactly when ``multioutput="variance_weighted"`` -- this plan's test file pins
    that identity numerically rather than trusting the citation.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    if Y_true.shape != Y_pred.shape:
        raise ValueError(
            f"aggregate_r2: Y_true has shape {Y_true.shape} but Y_pred has shape {Y_pred.shape}."
        )
    if not np.all(np.isfinite(Y_true)) or not np.all(np.isfinite(Y_pred)):
        raise ValueError("aggregate_r2: Y_true or Y_pred contains a non-finite value.")
    return float(r2_score(Y_true, Y_pred, multioutput=multioutput))


# --- D5-04: seed pooling -----------------------------------------------------------------------


def pool_seed_fields(fields: Dict[Any, np.ndarray], method: str) -> Dict[str, Any]:
    """Pool multiple seeds' length-``n`` ``||H||`` fields into one field. ``method`` is
    REQUIRED with no default -- see this module's docstring (b) for why a raw average is not
    a fair pooling rule on this milestone's measured seed spread. Supports exactly two
    methods, raising ``ValueError`` naming the offending value on anything else:

    * ``"per_seed_median_divide"`` -- divide each seed's field by its own median, then average
      the normalized fields across seeds.
    * ``"per_seed_rank_uniform"`` -- map each seed's field to its own percentile rank on
      ``[0, 1]`` via a stable argsort, then average the ranks across seeds.

    Returns a dict with ``pooled``, ``per_seed_normalized`` (dict keyed by seed),
    ``per_seed_median`` (dict keyed by seed, the RAW per-seed median before normalization),
    ``method``, ``n_seeds``, ``n``.
    """
    if not isinstance(fields, dict) or len(fields) == 0:
        raise ValueError("pool_seed_fields: fields must be a non-empty dict mapping seed to an array.")
    seeds = sorted(fields.keys())
    arrays = {s: np.asarray(fields[s], dtype=np.float64) for s in seeds}
    n = arrays[seeds[0]].shape[0]
    for s in seeds:
        a = arrays[s]
        if a.ndim != 1:
            raise ValueError(
                f"pool_seed_fields: field for seed {s} must be one-dimensional, got shape "
                f"{a.shape}."
            )
        if a.shape[0] != n:
            raise ValueError(
                f"pool_seed_fields: field for seed {s} has {a.shape[0]} rows, expected {n}."
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"pool_seed_fields: field for seed {s} contains a non-finite value.")

    per_seed_median: Dict[Any, float] = {s: float(np.median(arrays[s])) for s in seeds}
    per_seed_normalized: Dict[Any, np.ndarray] = {}

    if method == "per_seed_median_divide":
        for s in seeds:
            med = per_seed_median[s]
            per_seed_normalized[s] = arrays[s] / max(med, 1e-12)
    elif method == "per_seed_rank_uniform":
        denom = max(n - 1, 1)
        for s in seeds:
            order = np.argsort(arrays[s], kind="stable")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(n, dtype=np.float64)
            per_seed_normalized[s] = ranks / denom
    else:
        raise ValueError(
            f"pool_seed_fields: method={method!r} is not recognized; must be "
            '"per_seed_median_divide" or "per_seed_rank_uniform".'
        )

    stacked = np.stack([per_seed_normalized[s] for s in seeds], axis=0)
    pooled = stacked.mean(axis=0)

    return {
        "pooled": pooled,
        "per_seed_normalized": per_seed_normalized,
        "per_seed_median": per_seed_median,
        "method": method,
        "n_seeds": len(seeds),
        "n": int(n),
    }


# --- D5-07/D5-09: bucketing --------------------------------------------------------------------


def bucket_edges_from_field(values: np.ndarray, n_buckets: int) -> Tuple[float, ...]:
    """Equal-frequency rank partition over ALL supplied values: stable-argsort, then
    ``np.array_split`` into ``n_buckets`` groups. The edge between adjacent groups is the
    value of the first element of the later group. Returns ``n_buckets - 1`` edges as a tuple
    of floats.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"bucket_edges_from_field: values must be one-dimensional, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("bucket_edges_from_field: values contains a non-finite value.")
    if not isinstance(n_buckets, (int, np.integer)) or isinstance(n_buckets, bool) or n_buckets < 2:
        raise ValueError(f"bucket_edges_from_field: n_buckets={n_buckets} must be an int >= 2.")
    if values.shape[0] < n_buckets:
        raise ValueError(
            f"bucket_edges_from_field: values has {values.shape[0]} points, fewer than "
            f"n_buckets={n_buckets}."
        )
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, n_buckets)
    return tuple(float(values[g[0]]) for g in groups[1:])


def assign_buckets(values: np.ndarray, edges: Any) -> np.ndarray:
    """``np.searchsorted(edges, values, side="right")`` -- a value exactly equal to an edge
    lands in the HIGHER bucket. Stated, deterministic tie rule.
    """
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"assign_buckets: values must be one-dimensional, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("assign_buckets: values contains a non-finite value.")
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def bucket_by_field(values: np.ndarray, n_buckets: int) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Composes :func:`bucket_edges_from_field` and :func:`assign_buckets`. Returns
    ``(labels, edges)``.
    """
    edges = bucket_edges_from_field(values, n_buckets)
    labels = assign_buckets(values, edges)
    return labels, edges


def bucket_counts(labels: np.ndarray, n_buckets: int) -> Dict[str, Any]:
    """``np.bincount(labels, minlength=n_buckets)`` as plain ints plus fractions, plus
    ``n_total``. The counts sum exactly to ``labels.shape[0]``.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"bucket_counts: labels must be one-dimensional, got shape {labels.shape}.")
    counts = np.bincount(labels, minlength=n_buckets).astype(np.int64)
    n_total = int(labels.shape[0])
    fractions = counts.astype(np.float64) / n_total if n_total > 0 else counts.astype(np.float64)
    return {
        "counts": counts,
        "fractions": fractions,
        "n_total": n_total,
        "n_buckets": int(n_buckets),
    }


# --- D5-08: bootstrap CI and the size-matched check ---------------------------------------------


def bucket_residual_ci(
    residuals: np.ndarray, n_resamples: int, seed: int, confidence_level: float
) -> Dict[str, Any]:
    """Copies ``mknn.bootstrap_ci``'s shape: required args with no default,
    ``scipy.stats.bootstrap((residuals,), np.mean, method="percentile",
    n_resamples=n_resamples, confidence_level=confidence_level,
    rng=np.random.default_rng(seed))``, flat result dict including ``degenerate``.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 1 or residuals.shape[0] < 2:
        raise ValueError(
            f"bucket_residual_ci: residuals must be one-dimensional with at least 2 points, "
            f"got shape {residuals.shape}."
        )
    if not np.all(np.isfinite(residuals)):
        raise ValueError("bucket_residual_ci: residuals contains a non-finite value.")

    rng = np.random.default_rng(seed)
    result = bootstrap(
        (residuals,),
        np.mean,
        method="percentile",
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )
    ci_low = float(result.confidence_interval.low)
    ci_high = float(result.confidence_interval.high)

    return {
        "score": float(residuals.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "degenerate": bool(ci_low == ci_high),
        "confidence_level": float(confidence_level),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "n": int(residuals.shape[0]),
    }


_SIZE_MATCH_CI_N_RESAMPLES = 300
"""Internal CI resample count used inside :func:`size_matched_check`'s per-draw disjointness
check. Not itself a pre-registered constant -- it is an implementation detail of how the
size-match diagnostic is computed, not a value that changes what question is asked."""


def size_matched_check(
    residuals: np.ndarray, labels: np.ndarray, n_repeats: int, seed: int, confidence_level: float
) -> Dict[str, Any]:
    """D5-08 / RESEARCH Pitfall 4. ``residuals`` and ``labels`` MUST be the REALIZED
    TEST-SPLIT arrays -- never the full-field bucket assignment. ``n_match`` is
    ``min(bucket_counts(labels, ...))`` computed over the REALIZED test-split counts, never
    over the full-field bucket counts: the edges are cut on all points, so equal full-field
    counts say nothing about the counts a plain random train/test split leaves in each
    bucket, and using the full-field count here is exactly the artifact that undercut Phase
    4's verdict.

    For each of ``n_repeats`` draws under one ``np.random.default_rng(seed)``, samples
    ``n_match`` residuals without replacement from the lowest and the highest bucket, records
    the highest-minus-lowest mean difference and whether the two buckets' percentile CIs at
    ``confidence_level`` are disjoint in that draw. Returns ``n_match``,
    ``realized_bucket_counts`` (a tuple, one count per bucket label 0..n_buckets-1),
    ``median_diff``, ``sign_stable`` (all repeats share the sign of ``median_diff``),
    ``ci_disjoint_fraction``, ``n_repeats``, ``seed``.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    labels = np.asarray(labels)
    if residuals.shape[0] != labels.shape[0]:
        raise ValueError(
            f"size_matched_check: residuals has {residuals.shape[0]} rows but labels has "
            f"{labels.shape[0]} rows."
        )
    if not np.all(np.isfinite(residuals)):
        raise ValueError("size_matched_check: residuals contains a non-finite value.")
    if residuals.shape[0] == 0:
        raise ValueError("size_matched_check: residuals/labels must be non-empty.")

    n_buckets = int(labels.max()) + 1
    counts_info = bucket_counts(labels, n_buckets)
    realized_counts = tuple(int(c) for c in counts_info["counts"])
    n_match = int(min(realized_counts))
    if n_match < 1:
        raise ValueError(
            f"size_matched_check: smallest realized test-split bucket count is {n_match}, "
            "need at least 1."
        )

    bucket_indices = {b: np.flatnonzero(labels == b) for b in range(n_buckets)}
    lowest, highest = 0, n_buckets - 1

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_repeats, dtype=np.float64)
    disjoint = 0
    for i in range(n_repeats):
        low_sample = rng.choice(bucket_indices[lowest], size=n_match, replace=False)
        high_sample = rng.choice(bucket_indices[highest], size=n_match, replace=False)
        low_res = residuals[low_sample]
        high_res = residuals[high_sample]
        diffs[i] = float(high_res.mean() - low_res.mean())

        low_ci = bucket_residual_ci(
            low_res,
            n_resamples=_SIZE_MATCH_CI_N_RESAMPLES,
            seed=int(rng.integers(0, 2**31 - 1)),
            confidence_level=confidence_level,
        )
        high_ci = bucket_residual_ci(
            high_res,
            n_resamples=_SIZE_MATCH_CI_N_RESAMPLES,
            seed=int(rng.integers(0, 2**31 - 1)),
            confidence_level=confidence_level,
        )
        if low_ci["ci_high"] < high_ci["ci_low"] or high_ci["ci_high"] < low_ci["ci_low"]:
            disjoint += 1

    median_diff = float(np.median(diffs))
    sign_stable = bool(np.all(diffs > 0) or np.all(diffs < 0))

    return {
        "n_match": n_match,
        "realized_bucket_counts": realized_counts,
        "median_diff": median_diff,
        "sign_stable": sign_stable,
        "ci_disjoint_fraction": float(disjoint) / n_repeats if n_repeats > 0 else 0.0,
        "n_repeats": int(n_repeats),
        "seed": int(seed),
    }


# --- D5-09/D5-10: the verdict --------------------------------------------------------------


def apply_verdict_rule(
    bucket_stats: Any, size_match: Dict[str, Any], verdict_rule: str
) -> Dict[str, Any]:
    """Applies the frozen rule mechanically. ``bucket_stats`` is a sequence of per-bucket
    stat dicts in bucket-index order (each carrying at least ``score``, ``ci_low``,
    ``ci_high`` -- :func:`bucket_residual_ci`'s own return shape). ``size_match`` is
    :func:`size_matched_check`'s return dict. Raises ``RuntimeError`` if ``verdict_rule`` is
    empty, so this cannot run before the freeze. The two terminal verdicts are the strings
    ``"HOLDS"`` and ``"NO DETECTABLE RELATIONSHIP"``; there is no third outcome and no
    near-miss: HOLDS requires ALL of -- the lowest and highest bucket's CIs are disjoint, the
    highest bucket's mean residual exceeds the lowest bucket's, and the size-matched check's
    sign is stable across repeats.
    """
    if not isinstance(verdict_rule, str) or not verdict_rule.strip():
        raise RuntimeError(
            "apply_verdict_rule: verdict_rule is empty; cannot apply before the "
            "pre-registration freeze."
        )
    if bucket_stats is None or len(bucket_stats) < 2:
        raise ValueError("apply_verdict_rule: bucket_stats must carry at least 2 buckets.")

    lowest = bucket_stats[0]
    highest = bucket_stats[-1]
    ci_disjoint = bool(
        lowest["ci_high"] < highest["ci_low"] or highest["ci_high"] < lowest["ci_low"]
    )
    residual_higher_at_high_curvature = bool(highest["score"] > lowest["score"])
    sign_stable = bool(size_match.get("sign_stable", False))

    holds = ci_disjoint and residual_higher_at_high_curvature and sign_stable
    verdict = "HOLDS" if holds else "NO DETECTABLE RELATIONSHIP"

    return {
        "verdict": verdict,
        "criteria": {
            "ci_disjoint": ci_disjoint,
            "residual_higher_at_high_curvature": residual_higher_at_high_curvature,
            "size_match_sign_stable": sign_stable,
        },
    }
