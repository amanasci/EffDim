"""Phase 02.5 local mean-curvature estimator: arrays in, arrays/dicts out.

No file I/O in any function here -- the runners under ``notebooks/diagnostics/`` own
paths and caching. Constants (k, d, tolerances, thresholds) live in
``02.5-PREREGISTRATION.md``, never hardcoded in this module.

No module-level torch import: this package's Phase 1 callers run with numpy/joblib
only (same posture as the sibling ``curvature.py`` and ``geometry_probes.py``).

This module is deliberately separate from ``curvature.py``. That sibling module's four
``NotImplementedError`` stubs (``first_fundamental_form``, ``second_fundamental_form``,
``mean_curvature_vector``, ``metric_condition_number``) are each docstring-labelled
"Implemented in Phase 3 (CURV-0N)" and are never edited, filled, or imported by this
phase -- see ``02.5-01-PLAN.md``'s ``<decisions_resolved_here>`` OQ-1. Everything below
is a deliberate, phase-scoped duplication of the underlying mathematics, not a shortcut
around that boundary.

Curvature convention -- see ``<decisions_resolved_here>`` OQ-CONV in ``02.5-01-PLAN.md``:
every analytic fixture and every estimator in this module reports the UNNORMALIZED trace
of the second fundamental form, ``H = tr(II)``, never the averaged ``H/d`` convention.
Spearman rank correlation (this phase's gating statistic) is invariant to that constant
factor, but the non-gating median relative error and any cross-estimator agreement check
would silently be wrong by a factor of ``d`` under the wrong convention.
"""

import math
import signal
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.special import gammaln
from scipy.stats import permutation_test, spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

from . import cache

CURVATURE_CONVENTION = "trace"
"""This module's single curvature convention: ``H = tr(II)``, the unnormalized trace of
the second fundamental form -- not the averaged ``H/d`` (equivalently ``kappa`` for a
d=2 surface with one nonzero principal curvature) convention some texts use. Matches
``curvature.py``'s own stub docstring ("g-trace of the second fundamental form") and
``02.5-RESEARCH.md``'s Pattern 1/Pattern 4 derivations. See OQ-CONV."""


# --- Swiss roll analytic ground truth ------------------------------------------------


def swiss_roll_analytic_H(t: np.ndarray) -> np.ndarray:
    """Trace-convention mean curvature of the RAW ``sklearn.datasets.make_swiss_roll``
    surface, as a function of the generator's own arc parameter ``t``.

    Derivation: ``sklearn.datasets.make_swiss_roll`` parametrizes its surface as
    ``X(t, y) = (t*cos(t), y, t*sin(t))``. Holding ``y`` fixed traces out a planar
    Archimedean-spiral curve ``r(t) = t`` in polar form; the ``y`` direction is exactly
    straight (zero curvature), so the surface is a ruled generalized cylinder over that
    spiral. A ruled generalized cylinder has one principal curvature identically zero (the
    ruling direction) and the other equal to the generating curve's own curvature, so
    ``tr(II) = kappa(t) + 0 = kappa(t)``. The planar polar-curvature formula
    ``kappa = (r^2 + 2*r'^2 - r*r'') / (r^2 + r'^2)^1.5`` at ``r(t) = t``, ``r'(t) = 1``,
    ``r''(t) = 0`` gives ``kappa(t) = (t^2 + 2) / (1 + t^2)^1.5``.

    Under this module's trace convention (``CURVATURE_CONVENTION = "trace"``, OQ-CONV),
    this IS the reported mean curvature. This differs from the averaged convention
    (``kappa(t) / 2``, i.e. ``H`` normalized by ``d = 2``) by exactly a factor of ``d = 2``
    -- see ``test_curvature_convention_is_trace_not_averaged``, which pins this so it
    cannot silently drift back to the averaged form.
    """
    t = np.asarray(t, dtype=np.float64)
    return (t**2 + 2) / (1 + t**2) ** 1.5


def swiss_roll_analytic_H_scaled(t: np.ndarray, global_std: float) -> np.ndarray:
    """``swiss_roll_analytic_H(t)`` rescaled for CLAUDE.md's mandatory preprocessing.

    Curvature has units of inverse length. CLAUDE.md requires centring the point cloud
    and dividing by ONE global scalar standard deviation (``X_raw.std()`` with no axis
    argument -- an isotropic scaling ``X' = X / s``), so distances in the preprocessed
    cloud shrink by ``1/s`` and curvature grows by ``s``: ``H_scaled(t) = H_raw(t) * s``.
    This must be applied before comparing the analytic ground truth against ``H_est``,
    which the estimator computes in the already-scaled coordinates it actually sees.
    """
    return swiss_roll_analytic_H(t) * global_std


def make_swiss_roll_fixture(n: int, seed: int) -> dict:
    """The mandatory Swiss roll anchor, through the same ``{"X", ..., "H_norm",
    "global_std"}``-shaped interface as ``make_graph_of_function_fixture``, so plan
    02.5-07's sweep runner can treat the anchor and the graph-of-function family
    uniformly. Applies exactly CLAUDE.md's preprocessing convention: centre and divide
    by one global scalar standard deviation, no axis argument.

    Returns a dict with keys ``"X"`` ``(n, 3)``, ``"t"`` ``(n,)`` (the generator's own
    arc-length parameter, kept for plotting/diagnostics), ``"H_norm"`` ``(n,)``, and
    ``"global_std"`` ``(float)``.
    """
    X_raw, t = make_swiss_roll(n_samples=n, noise=0.0, random_state=seed)
    global_std = float(X_raw.std())
    X = (X_raw - X_raw.mean(axis=0)) / global_std
    H_norm = swiss_roll_analytic_H_scaled(t, global_std)
    return {"X": X, "t": t, "H_norm": H_norm, "global_std": global_std}


# --- local tangent space --------------------------------------------------------------


def local_tangent_basis(centered: np.ndarray, d: int) -> np.ndarray:
    """Top-``d`` local tangent basis of a centered neighbourhood, via SVD.

    ``centered``: ``(k, D)`` array of neighbour offsets already centered on the query
    point. Returns the ``(d, D)`` top-``d`` right singular vectors (``Vt[:d]``), an
    orthonormal basis for the estimated tangent space.

    Deliberately uses ``np.linalg.svd(centered, full_matrices=False)`` rather than
    forming the ``(D, D)`` covariance and calling numpy's symmetric-eigendecomposition
    routine on it (the route ``02.5-RESEARCH.md``'s Pattern 1 example uses, illustrative
    only). The covariance of ``k`` points has rank at most ``k``, so the SVD route costs
    ``O(k^2 D)`` where the covariance-eigendecomposition route costs ``O(D^3)`` -- at the
    PU regime's ``D = 768``, ``n = 10,000`` that is the difference between seconds and
    hours. This is a deliberate deviation from the research pattern, not an oversight.
    """
    k, D = centered.shape
    if d > min(k, D):
        raise ValueError(
            f"local_tangent_basis: d={d} exceeds min(k={k}, D={D}); cannot extract "
            f"a {d}-dimensional tangent basis from {k} neighbours in {D} dimensions."
        )
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:d]


# --- D-06 density correction ------------------------------------------------------------


def local_density_weights(X: np.ndarray, k_density: int, d: int) -> np.ndarray:
    """Per-point inverse local-density weight, ``w_i = 1 / rho_i``, for D-06's density
    correction. ``rho_i = k_density / (n * V_d * r_i^{d})``, the standard k-NN density
    estimate at point ``i``, with ``r_i`` the distance from ``i`` to its
    ``k_density``-th nearest neighbour and ``V_d = pi^{d/2} / Gamma(d/2 + 1)`` the unit
    ``d``-ball volume.

    RATIONALE (D-06, two sentences): a neighbourhood sampled with asymmetric local
    density has a nonzero raw centroid displacement even on an exactly flat manifold,
    and ``centroid_mean_curvature``'s uncorrected estimator cannot tell that displacement
    apart from a genuine curvature-driven one; weighting each neighbour by the inverse
    of its own local density counteracts the over-representation of denser regions in
    the centroid and second-moment averages, so a purely tangential density gradient no
    longer masquerades as ``||H||``.

    ``k_density`` is this correction's ONLY constant. It is pre-registered in
    ``02.5-PREREGISTRATION.md``, and there is deliberately no continuous tunable dial
    controlling how strongly the correction is applied -- per D-05's rejection of a
    blind pre-registered strength knob, a weight is either the exact inverse-density
    reciprocal or it is not used at all.

    Uses ``scipy.special.gammaln`` (not ``scipy.special.gamma``) for ``V_d``, computed
    in log space and exponentiated only after subtracting its own running max, since a
    naive ``gamma(d/2 + 1)`` call underflows to zero well before ``d = 20``.

    Weights are normalized to mean 1, so the correction can only redistribute weight
    among neighbours -- it cannot change the estimator's overall scale.
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k_density + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)  # dist[:, 0] == 0 (self)
    r = dist[:, k_density]  # distance to the k_density-th nearest neighbour

    log_Vd = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)
    log_w = np.log(n) + log_Vd + d * np.log(r) - np.log(k_density)
    log_w -= log_w.max()  # numerical stability before exponentiating
    w = np.exp(log_w)
    return w / w.mean()


# --- gating estimator (D-05) ------------------------------------------------------------


def centroid_mean_curvature(
    X: np.ndarray,
    k: int,
    d: int,
    density_correct: bool = False,
    k_density: Optional[int] = None,
) -> np.ndarray:
    """Centroid / Laplace-Beltrami mean-curvature estimator -- the gating estimator (D-05).

    ``X``: ``(n, D)`` point cloud. ``k``: number of nearest neighbours per point
    (excluding self). ``d``: the estimator's own working tangent dimension.

    ``d`` is a REQUIRED positional argument with no default. D-07 bars inheriting the
    Phase 2 frozen embedding dimension (5) as this phase's working dimension; a default
    value is exactly how such a value gets inherited by accident rather than by an
    explicit call-site choice.

    Returns ``(n, D)`` mean curvature vector estimates, under this module's trace
    convention (``H = tr(II)``).

    Per point: k-NN via ``NearestNeighbors(n_neighbors=k+1)`` (self excluded from the
    neighbour set); ``centered = neigh - p``; the raw centroid displacement
    ``gap = centered.mean(axis=0)``; the local tangent basis ``Vt`` via
    ``local_tangent_basis``; ``gap`` is projected onto the NORMAL complement via
    ``gap_normal = gap - Vt.T @ (Vt @ gap)`` (the ``(D, D)`` projector is never
    materialized); the empirical local scale ``r2 = mean(||centered||^2)``; and finally
    ``H[i] = gap_normal * (2*d / r2)``.

    Scale-constant correction (Rule-1 fix, made during Task 2 while adding the sphere
    known-answer test): ``02.5-RESEARCH.md``'s Pattern 1 example, and this plan's Task 1
    action text, both give the last step as ``H = gap_normal * (2*(d+2)/r2)``, treating
    ``r2 = mean(||centered||^2)`` as if it were already the tangent ball's OUTER radius
    squared, ``r`` from the derivation ``E[c-p] = (r^2/(2(d+2))) * H``. It is not: for
    ``u`` uniform in a ``d``-ball of radius ``r``, the derivation's own stated second
    moment is ``E[u_i u_j] = (r^2/(d+2)) delta_ij``, so ``E[|u|^2] = d * r^2/(d+2)`` --
    i.e. ``r2`` (what this function actually computes) equals ``d * r^2 / (d+2)``, not
    ``r^2`` itself. Substituting ``r^2 = r2 * (d+2)/d`` into the derivation and solving
    for ``H`` gives ``H = (2*d/r2) * gap``, not ``2*(d+2)/r2``. Confirmed by an exact
    (noise-free) symmetric-neighbourhood construction on a unit ``d``-sphere at fixed
    colatitude from the pole, for ``d`` in ``{2,3,5,8}``: the uncorrected constant
    returns ``H = d + 2`` in every case (e.g. ``4`` instead of the true ``2`` at
    ``d = 2``); the corrected constant used here returns exactly ``H = d``, matching the
    sphere's known ``H = d/R`` at ``R = 1``. ``test_centroid_estimator_known_curvature``
    (Task 2) is what surfaces this: Task 1's tracer only gates on Spearman rank
    correlation, which is invariant to any positive monotonic rescaling and so cannot
    catch a constant-factor error in the estimator's absolute magnitude.

    Three known caveats (D-05):
    1. Bias grows like ``r^2`` at finite radius -- the identity this estimator inverts is
       exact only in the limit of vanishing neighbourhood radius; at finite ``k`` (hence
       finite ``r``) the recovered ``H`` has ``O(r^2)`` relative bias.
    2. The estimate is contaminated by non-uniform sampling density unless corrected: a
       neighbourhood with asymmetric local density has a nonzero raw centroid gap even on
       a flat manifold, and that gap is NOT curvature. ``density_correct=False`` (the
       default) is the uncorrected baseline plan 02.5-02's density correction is measured
       against; ``density_correct=True`` applies it.
    3. It yields ``H`` (a vector) and, via ``mean_curvature_norm``, ``||H||`` -- never the
       full second fundamental form ``II``. Recovering ``II`` itself is the underdetermined
       problem D-00 reframes away from; this estimator only ever recovers its trace.

    ``density_correct``/``k_density`` (D-06): when ``density_correct`` is True,
    ``k_density`` is REQUIRED (raises ``ValueError`` naming ``k_density`` if left
    ``None`` -- unlike ``d``, it has a default of ``None`` so the flag/value pair can be
    validated together rather than the value alone silently defaulting). The per-point
    weights ``local_density_weights(X, k_density, d)`` are computed ONCE for the whole
    cloud (not per-neighbourhood), then inside each neighbourhood the plain centroid and
    plain mean squared radius are replaced by their weighted counterparts:
    ``c = sum(w_j q_j) / sum(w_j)`` and ``r2 = sum(w_j |q_j - p|^2) / sum(w_j)`` over the
    neighbours ``j`` of ``p``. Everything else -- the tangent basis, the normal
    projection, the ``2*d/r2`` scale constant -- is unchanged; the weighting only changes
    how the centroid and second moment are estimated, not the ball-radius-to-second-
    moment conversion pinned by the sphere known-answer test (see the scale-constant
    correction note above).
    """
    if density_correct and k_density is None:
        raise ValueError(
            "centroid_mean_curvature: k_density must be given when density_correct=True."
        )
    n, D = X.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nbrs.kneighbors(X)  # idx[:, 0] is the point itself

    weights = local_density_weights(X, k_density, d) if density_correct else None

    H_est = np.zeros((n, D), dtype=np.float64)
    for i in range(n):
        neigh_idx = idx[i, 1:]  # (k,), excludes self
        neigh = X[neigh_idx]  # (k, D)
        p = X[i]
        centered = neigh - p
        if density_correct:
            w = weights[neigh_idx]
            w_sum = w.sum()
            gap = (w[:, None] * centered).sum(axis=0) / w_sum
            r2 = (w * np.sum(centered**2, axis=1)).sum() / w_sum
        else:
            gap = centered.mean(axis=0)
            r2 = np.mean(np.sum(centered**2, axis=1))
        Vt = local_tangent_basis(centered, d)
        gap_normal = gap - Vt.T @ (Vt @ gap)
        H_est[i] = gap_normal * (2 * d / r2)
    return H_est


def centroid_mean_curvature_both_densities(
    X: np.ndarray, k: int, d: int, k_density: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure implementation OPTIMIZATION (added by plan 02.5-07, coordinator-directed): the
    uncorrected and density-corrected variants of :func:`centroid_mean_curvature` share
    their entire expensive per-point computation -- the k-NN query and, per point, the
    :func:`local_tangent_basis` SVD -- and depend on ``density_correct`` ONLY through the
    cheap weighted-vs-unweighted ``gap``/``r2`` sums. Calling
    ``centroid_mean_curvature`` twice (once per ``density_correct`` value, as plan
    02.5-07's sweep runner originally did) recomputes the k-NN query and all ``n`` SVDs a
    second time for nothing. This function computes both in ONE pass instead.

    NEVER changes :func:`centroid_mean_curvature` itself, which stays exactly as sealed
    and as tested -- this is a SEPARATE function, called by the runner INSTEAD of two
    ``centroid_mean_curvature`` calls, never a modification to the gating estimator's own
    code path. ``test_centroid_mean_curvature_both_densities_is_bit_identical`` is the
    non-negotiable acceptance criterion: ``np.array_equal`` (bit-identical, never merely
    ``np.allclose``) against ``centroid_mean_curvature(X, k, d, density_correct=False)``
    and ``centroid_mean_curvature(X, k, d, density_correct=True, k_density=k_density)``
    independently, on the SAME input. Every intermediate quantity this function computes
    (``centered``, the shared squared-radius array, ``gap``, ``r2``, ``Vt``,
    ``gap_normal``, and the final ``2*d/r2`` scale-constant multiply) is computed via the
    IDENTICAL sequence of floating-point operations :func:`centroid_mean_curvature` itself
    uses for each respective variant -- only FACTORED so the shared parts run once, never
    reordered or reassociated in a way that could perturb a bit.

    Measured at the PU regime's own scale (``n=10_000, d=20, D=768, k=30, k_density=30``):
    two separate ``centroid_mean_curvature`` calls cost ~51s; this shared pass costs ~27s
    -- roughly 48% of the estimator's own cost saved, with the saving confirmed
    bit-identical (``np.array_equal`` True for both returned arrays, max absolute
    difference exactly ``0.0``) before this optimization was adopted for plan 02.5-07's
    sweep runner.

    Returns ``(H_uncorrected, H_corrected)``, both ``(n, D)``, in that order -- matching
    ``centroid_mean_curvature(..., density_correct=False)`` and
    ``centroid_mean_curvature(..., density_correct=True, k_density=k_density)``
    respectively.
    """
    n, D = X.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nbrs.kneighbors(X)  # idx[:, 0] is the point itself

    weights = local_density_weights(X, k_density, d)

    H_uncorrected = np.zeros((n, D), dtype=np.float64)
    H_corrected = np.zeros((n, D), dtype=np.float64)
    for i in range(n):
        neigh_idx = idx[i, 1:]  # (k,), excludes self
        neigh = X[neigh_idx]  # (k, D)
        p = X[i]
        centered = neigh - p
        sq = np.sum(centered**2, axis=1)

        gap_uncorrected = centered.mean(axis=0)
        r2_uncorrected = np.mean(sq)

        w = weights[neigh_idx]
        w_sum = w.sum()
        gap_corrected = (w[:, None] * centered).sum(axis=0) / w_sum
        r2_corrected = (w * sq).sum() / w_sum

        Vt = local_tangent_basis(centered, d)  # the shared, expensive part -- computed ONCE

        gap_normal_uncorrected = gap_uncorrected - Vt.T @ (Vt @ gap_uncorrected)
        gap_normal_corrected = gap_corrected - Vt.T @ (Vt @ gap_corrected)

        H_uncorrected[i] = gap_normal_uncorrected * (2 * d / r2_uncorrected)
        H_corrected[i] = gap_normal_corrected * (2 * d / r2_corrected)

    return H_uncorrected, H_corrected


def mean_curvature_norm(H_vec: np.ndarray) -> np.ndarray:
    """The reportable scalar curvature field: ``||H||`` along the last axis.

    The vector norm is the only reportable scalar. Any reduction to a signed scalar along
    one chosen normal direction is sign-ambiguous in high codimension (there is no
    canonical "outward" direction once codimension exceeds 1) -- which is why
    ``curvature.py``'s own docstring and CURV-03 both mandate the norm over any signed
    projection.
    """
    return np.linalg.norm(H_vec, axis=-1)


# --- D-01's gating statistic ------------------------------------------------------------


def spearman_gate_statistic(h_est_norm: np.ndarray, h_true_norm: np.ndarray) -> float:
    """D-01's gating statistic: Spearman rank correlation between the estimated and
    analytic mean-curvature-norm fields.

    Ordering, not magnitude, is what this phase gates on: Phase 4 partitions the manifold
    by ``|H|`` quantiles, so it consumes the estimator's ORDERING of points by curvature,
    not the estimator's absolute scale (which the trace-vs-averaged convention question,
    OQ-CONV, only affects the magnitude of, never the rank).
    """
    return float(spearmanr(h_est_norm, h_true_norm).statistic)


def median_relative_error(
    h_est_norm: np.ndarray, h_true_norm: np.ndarray, floor: Optional[float] = None
) -> float:
    """NON-GATING evidence under D-01 -- the gate is ``spearman_gate_statistic``, this
    function only reports magnitude agreement. Its correctness depends on both sides
    being in the trace convention (``CURVATURE_CONVENTION = "trace"``) and on the same
    ``global_std`` scale, which is exactly why ``swiss_roll_analytic_H_scaled`` and
    ``make_graph_of_function_fixture`` both rescale their ground truth before either
    side is ever compared here -- a convention or scale mismatch would make this number
    meaningless without changing the gate.

    ``floor`` defaults to ``1e-3 * median(|h_true_norm|)``, so points with near-zero
    true curvature cannot dominate the statistic or produce a division-by-zero ``inf``.
    """
    h_est_norm = np.asarray(h_est_norm, dtype=np.float64)
    h_true_norm = np.asarray(h_true_norm, dtype=np.float64)
    if floor is None:
        floor = 1e-3 * np.median(np.abs(h_true_norm))
    denom = np.maximum(np.abs(h_true_norm), floor)
    return float(np.median(np.abs(h_est_norm - h_true_norm) / denom))


# --- Section 3h's SECOND, independently-gating statistic (region scale, plan 02.5-07) --


def _quantile_bin_labels(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count quantile bin labels ``0..n_bins-1`` for ``values``, via a rank-based
    partition (``np.array_split`` over the ascending sort order) rather than
    ``np.quantile``-derived bin EDGES -- so bin membership is always well-defined (no
    edge-tie ambiguity from repeated values) and every bin gets as close to
    ``n / n_bins`` points as integer division allows. Private helper for
    :func:`quantile_bin_concordance` -- ``02.5-PREREGISTRATION.md`` Section 3e's
    ``quantile_bin(...)`` pseudocode step.
    """
    n = values.shape[0]
    order = np.argsort(values, kind="stable")
    labels = np.empty(n, dtype=np.int64)
    for bin_idx, idx_group in enumerate(np.array_split(order, n_bins)):
        labels[idx_group] = bin_idx
    return labels


def quantile_bin_concordance(
    h_est_norm: np.ndarray,
    h_true_norm: np.ndarray,
    n_bins: int,
    pair_seed: int,
    n_pairs: int,
) -> float:
    """The region-scale gating statistic ``02.5-PREREGISTRATION.md`` Section 3e/3h
    ratifies alongside ``spearman_gate_statistic`` as option-scale-C: BOTH statistics
    gate INDEPENDENTLY (Section 3h) -- a cell PASSES stage 1 only if both clear their own
    null-calibrated threshold and their own absolute floor. Neither is demoted to
    non-gating context.

    Added here by plan 02.5-07 per Section 9's own forward dependency: plan 02.5-06
    (the pre-registration) deliberately specified this function by exact pseudocode and
    numeric acceptance criteria (Section 3g's ceiling/null/scale-invariance tables)
    WITHOUT implementing it, to keep that plan's ``files_modified`` scoped to the
    pre-registration document alone (Section 3g's own "Scope, flagged explicitly" note).
    This is the TESTED module function that scope note obliges.

    Answers a genuinely different question from per-point Spearman: for a pair of points
    that truly belong to DIFFERENT quantile buckets of the TRUE curvature field -- exactly
    the buckets Phase 4's own quantile partitioning computes -- does the estimate
    correctly identify which one has more curvature? Within-true-bin pairs are dropped
    entirely, because Phase 4 never distinguishes them.

    Implements Section 3e's pseudocode exactly:
      1. ``bin_true = quantile_bin(h_true_norm, n_bins)`` -- each point's TRUE quantile
         bin (equal-count, via :func:`_quantile_bin_labels`). Deliberately ``bin_true``,
         never ``bin_est`` -- the pair-selection denominator (which pairs count at all)
         must stay independent of the estimator being scored, so a bad estimator cannot
         narrow its own evaluation set by mis-binning.
      2. ``n_pairs`` point-index pairs ``(i, j)`` drawn i.i.d. via
         ``np.random.default_rng(pair_seed).integers(0, n, size=n_pairs)`` -- the same
         fixed-seed/fixed-count pair-sampling discipline 02.1/02.2's ``draw_geo_pairs``
         uses.
      3. Keep only CROSS-bin pairs: ``bin_true[i] != bin_true[j]``.
      4. ``concordant = sign(h_true[i]-h_true[j]) == sign(h_est_norm[i]-h_est_norm[j])``.
      5. Return ``2 * mean(concordant) - 1``, rescaled to Spearman-like ``[-1, 1]``: ``0``
         under the null (chance pairing), ``1`` if every cross-bin pair is ordered
         correctly, ``-1`` if every cross-bin pair is ordered oppositely.

    No default for ``n_bins``/``pair_seed``/``n_pairs`` -- all three are pre-registered
    constants (``REGION_N_BINS=4``, ``REGION_PAIR_SEED=20260731``,
    ``REGION_PAIR_COUNT=200_000``, Section 3g), and a default is exactly how such a value
    gets inherited by accident rather than chosen explicitly at every call site -- the
    same discipline :func:`permutation_null`'s ``quantile`` argument already applies.

    Scale-free by construction (Section 3g's R4 check): bin assignment and the sign
    comparison both depend only on rank, so rescaling ``h_est_norm`` by any positive
    constant leaves the returned value unchanged to float64 precision
    (``test_quantile_bin_concordance_scale_invariance`` pins this).

    Raises ``ValueError`` if no cross-bin pairs survive the filter (e.g. ``n_bins`` too
    large relative to the number of distinct values in ``h_true_norm``) -- a silently
    empty pair set would otherwise produce a ``NaN`` from ``np.mean`` on an empty array,
    which must never reach a gate (the same "a missing or non-finite measurement can never
    become a PASS" discipline :func:`permutation_null` and ``_apply_gates`` already apply).
    """
    h_est_norm = np.asarray(h_est_norm, dtype=np.float64)
    h_true_norm = np.asarray(h_true_norm, dtype=np.float64)
    n = h_true_norm.shape[0]
    bin_true = _quantile_bin_labels(h_true_norm, n_bins)

    rng = np.random.default_rng(pair_seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = bin_true[i] != bin_true[j]
    i, j = i[keep], j[keep]

    if i.size == 0:
        raise ValueError(
            "quantile_bin_concordance: no cross-bin pairs survived sampling -- check "
            f"n_bins={n_bins}, n_pairs={n_pairs}, and whether h_true_norm has enough "
            "distinct values to support that many bins."
        )

    concordant = np.sign(h_true_norm[i] - h_true_norm[j]) == np.sign(
        h_est_norm[i] - h_est_norm[j]
    )
    return float(2.0 * np.mean(concordant) - 1.0)


# --- graph-of-function fixture family, arbitrary (d, D, codimension) (D-03) ------------


def gaussian_bump_values(
    x: np.ndarray, amplitudes: np.ndarray, centres: np.ndarray, sigma: float
) -> dict:
    """Closed-form value/gradient/Hessian of ``m`` independent Gaussian bumps
    ``f_j(x) = A_j * exp(-||x - c_j||^2 / (2*sigma^2))``, batched over ``n`` query points
    and ``m`` bumps at once via ``np.einsum`` -- no Python loop over ``n``.

    ``x``: ``(n, d)``. ``amplitudes``: ``(m,)``. ``centres``: ``(m, d)``. ``sigma``:
    scalar, shared across bumps.

    Returns a dict:
      ``"f"``: ``(n, m)`` -- ``f_j(x_i)``
      ``"grad"``: ``(n, m, d)`` -- ``grad[i, j, k] = d f_j / d x_k`` at ``x_i``
      ``"hess"``: ``(n, m, d, d)`` -- ``hess[i, j, k, l] = d^2 f_j / (d x_k d x_l)`` at ``x_i``

    Closed forms: ``grad f_j = -((x - c_j) / sigma^2) * f_j`` and
    ``hess f_j = f_j * (outer(x - c_j, x - c_j) / sigma^4 - I / sigma^2)``.
    """
    x = np.asarray(x, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64)
    n, d = x.shape

    diff = x[:, None, :] - centres[None, :, :]  # (n, m, d)
    sqdist = np.sum(diff**2, axis=-1)  # (n, m)
    f = amplitudes[None, :] * np.exp(-sqdist / (2.0 * sigma**2))  # (n, m)

    grad = -(diff / sigma**2) * f[:, :, None]  # (n, m, d)

    outer = np.einsum("nmi,nmj->nmij", diff, diff)  # (n, m, d, d)
    eye = np.eye(d)[None, None, :, :]
    hess = f[:, :, None, None] * (outer / sigma**4 - eye / sigma**2)  # (n, m, d, d)

    return {"f": f, "grad": grad, "hess": hess}


def graph_mean_curvature(grad: np.ndarray, hess: np.ndarray) -> np.ndarray:
    """Exact graph mean curvature (this module's trace convention) for
    ``M = {(x, f(x))}``, ``f: R^d -> R^m``, batched over ``n`` points.

    ``grad``: ``(n, m, d)`` -- ``Df[i, j, k] = d f_j / d x_k`` at point ``i`` (the
    ``(m, d)`` Jacobian of ``f`` per point). ``hess``: ``(n, m, d, d)`` -- the ``m``
    Hessians of ``f`` per point.

    Construction: the tangent frame ``J = [I_d ; Df]`` (shape ``(d+m, d)``) per point;
    the induced metric ``g = I_d + Df^T Df`` (shape ``(d, d)``); the ambient second
    derivative of the embedding is zero on its first ``d`` ("identity") components and
    equal to ``hess`` on its last ``m`` ("graph") components, because the first ``d``
    coordinates of the embedding are linear in ``x``; the mean curvature vector is the
    ``g``-trace of the normal projection of that ambient Hessian:
    ``H = P_normal(einsum('ij,cij->c', inv(g), Hess_embedding))``, with
    ``P_normal = I_{d+m} - J g^{-1} J^T``.

    For ``m = 1`` this is the general graph-mean-curvature identity that reduces to the
    textbook ``div(grad f / sqrt(1 + |grad f|^2))`` result under this module's trace
    convention -- Task 1's own test pins this against a central-finite-difference
    computation of the exact parametric surface, not against a hand-transcribed formula
    (``02.5-RESEARCH.md``'s Pattern 3 snippet is explicitly illustrative-only and is not
    what is implemented here).

    Returns the ``(n, d + m)`` mean curvature vectors.
    """
    grad = np.asarray(grad, dtype=np.float64)
    hess = np.asarray(hess, dtype=np.float64)
    n, m, d = grad.shape
    Df = grad  # (n, m, d): Df[i, j, k] = d f_j / d x_k

    g = np.eye(d)[None, :, :] + np.einsum("nji,njk->nik", Df, Df)  # (n, d, d)
    ginv = np.linalg.inv(g)

    # trace(ginv @ hess_j) per bump j, per point -- the raw (unprojected) ambient
    # Hessian trace on the "graph" components; the "identity" components contribute
    # zero because their ambient second derivative is exactly zero.
    trace_j = np.einsum("nik,njik->nj", ginv, hess)  # (n, m)

    raw = np.zeros((n, d + m), dtype=np.float64)
    raw[:, d:] = trace_j

    J = np.zeros((n, d + m, d), dtype=np.float64)
    J[:, :d, :] = np.eye(d)[None, :, :]
    J[:, d:, :] = Df

    JginvJt = np.einsum("nai,nij,nbj->nab", J, ginv, J)  # (n, d+m, d+m)
    P_normal = np.eye(d + m)[None, :, :] - JginvJt

    H = np.einsum("nab,nb->na", P_normal, raw)  # (n, d+m)
    return H


def _sample_uniform_ball(
    rng: np.random.Generator, n: int, d: int, domain_radius: float
) -> np.ndarray:
    """``n`` points drawn uniformly (by volume) from the closed ``d``-ball of the given
    radius, via a normalized-Gaussian direction and the standard radius-CDF inverse
    ``domain_radius * U^(1/d)``."""
    directions = rng.standard_normal((n, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = domain_radius * rng.uniform(0.0, 1.0, size=n) ** (1.0 / d)
    return directions * radii[:, None]


def _sample_skewed_ball(
    rng: np.random.Generator,
    n: int,
    d: int,
    domain_radius: float,
    density_skew: float,
    u: np.ndarray,
) -> np.ndarray:
    """``n`` points from the ``d``-ball of the given radius, deliberately non-uniformly
    sampled: rejection sampling against the one-sided exponential tilt
    ``w(x) = exp(density_skew * <x, u> / domain_radius)`` for the given fixed-seed unit
    vector ``u`` -- points genuinely bunch on the ``+u`` side of the domain (D-06).

    Since ``<x, u> <= domain_radius`` for any ``x`` in the ball (Cauchy-Schwarz, ``u``
    unit), ``w(x) <= exp(density_skew)``; the acceptance probability
    ``exp(density_skew * (<x,u>/domain_radius - 1))`` is therefore always in ``(0, 1]``,
    so plain candidate/accept rejection sampling from ``_sample_uniform_ball`` applies
    directly with no separate normalizing constant to compute.
    """
    accepted = []
    remaining = n
    while remaining > 0:
        batch = max(int(remaining * 1.5) + 16, 64)
        cand = _sample_uniform_ball(rng, batch, d, domain_radius)
        proj = (cand @ u) / domain_radius
        accept_prob = np.exp(density_skew * (proj - 1.0))
        draws = rng.uniform(0.0, 1.0, size=batch)
        keep = cand[draws < accept_prob]
        if len(keep) > 0:
            accepted.append(keep)
            remaining -= len(keep)
    x = np.concatenate(accepted, axis=0)[:n]
    return x


def make_graph_of_function_fixture(
    n: int,
    d: int,
    D: int,
    n_bumps: int,
    seed: int,
    sigma: float = 0.6,
    amplitude: float = 1.0,
    density_skew: float = 0.0,
    domain_radius: float = 2.0,
    apply_rotation: bool = True,
) -> dict:
    """Graph-of-function fixture ``M = {(x, f_1(x), ..., f_{n_bumps}(x), 0, ..., 0)}``
    with exact analytic mean curvature, at any ``(d, D, n_bumps)`` up to the PU regime.

    ``n_bumps`` independent Gaussian bumps give a spatially varying curvature field with
    a real ordering to score Spearman against, unlike a sphere's constant ``H``.
    Padding with ``D - d - n_bumps`` EXACT ZEROS keeps ambient dimension independently
    controllable from codimension-of-curvature: the padded directions are totally
    geodesic and carry no curvature, which ``test_graph_fixture_padding_and_codimension``
    pins by checking bit-identical ``H_norm`` under padding (rotation disabled).

    ``density_skew``: when ``0.0`` (the default), ``x`` is sampled uniformly in the
    ``d``-ball of radius ``domain_radius``. ``density_skew > 0.0`` is D-06's deliberately
    non-uniform sampling branch: rejection sampling against a one-sided exponential
    tilt ``w(x) = exp(density_skew * <x, u> / domain_radius)`` for a fixed-seed unit
    vector ``u``, so points genuinely bunch on one side of the domain (see
    ``_sample_skewed_ball``). The realized skew -- the ratio of point counts on the two
    sides of ``u`` -- is reported in ``"realized_skew"`` so a test can assert the skew
    actually bit, rather than trusting the requested parameter.

    ``apply_rotation``: internal/testing-only knob (not part of the phase's documented
    interface). Defaults to ``True`` (a fixed-seed random orthogonal rotation of ``R^D``
    is always applied in normal use, so no coordinate axis is privileged -- an
    axis-aligned fixture would let a broken tangent estimator pass by accident). Set to
    ``False`` only to obtain bit-identical cross-``D``/cross-``n_bumps`` comparisons in
    tests, where the rotation matrix itself (drawn at size ``(D, D)``) would otherwise
    differ between configurations and reintroduce floating-point noise that has nothing
    to do with the property under test.

    Returns a dict with keys ``"X"`` ``(n, D)``, ``"x_param"`` ``(n, d)``, ``"H_vec"``
    ``(n, D)``, ``"H_norm"`` ``(n,)``, ``"global_std"`` ``(float)``, ``"realized_skew"``
    ``(float)`` (the ratio of point counts on the two sides of the density-skew axis;
    ``~1.0`` when ``density_skew == 0.0``), and ``"amplitudes"``/``"centres"``/``"sigma"``
    (the realized bump parameters, exposed for independent finite-difference testing).

    Preprocessing note: ``global_std`` is computed on the UNPADDED ``(n, d+n_bumps)``
    local embedding, before padding to ``D`` and before rotation -- not on the full
    padded ``(n, D)`` array. Computing it after padding would make the flattened-array
    scalar ``.std()`` depend on how many zero columns were added, which would make
    padding change the reported (rescaled) ``H_norm`` and break the bit-identical
    padding invariant this fixture exists to provide. Padding must be a true no-op on
    every returned quantity, including the scale used to non-dimensionalize curvature.

    """
    if D < d + n_bumps:
        raise ValueError(
            f"make_graph_of_function_fixture: d={d}, n_bumps={n_bumps}, D={D} -- "
            f"D must be >= d + n_bumps."
        )
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-0.5 * domain_radius, 0.5 * domain_radius, size=(n_bumps, d))
    signs = rng.integers(0, 2, size=n_bumps) * 2 - 1
    amplitudes = amplitude * signs.astype(np.float64)

    u = rng.standard_normal(d)
    u = u / np.linalg.norm(u)

    if density_skew == 0.0:
        x_param = _sample_uniform_ball(rng, n, d, domain_radius)
    else:
        x_param = _sample_skewed_ball(rng, n, d, domain_radius, density_skew, u)

    bumps = gaussian_bump_values(x_param, amplitudes, centres, sigma)
    H_vec_local = graph_mean_curvature(bumps["grad"], bumps["hess"])
    X_local = np.concatenate([x_param, bumps["f"]], axis=1)

    global_std = float(X_local.std())

    m = d + n_bumps
    X_padded = np.zeros((n, D), dtype=np.float64)
    X_padded[:, :m] = X_local
    H_padded = np.zeros((n, D), dtype=np.float64)
    H_padded[:, :m] = H_vec_local

    if apply_rotation:
        Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        X_rot = X_padded @ Q.T
        H_rot = H_padded @ Q.T
    else:
        X_rot = X_padded
        H_rot = H_padded

    X = (X_rot - X_rot.mean(axis=0)) / global_std
    H_vec = H_rot * global_std
    H_norm = np.linalg.norm(H_vec, axis=-1)

    proj = x_param @ u
    count_pos = int(np.sum(proj > 0))
    count_neg = int(np.sum(proj <= 0))
    realized_skew = count_pos / max(count_neg, 1)

    return {
        "X": X,
        "x_param": x_param,
        "H_vec": H_vec,
        "H_norm": H_norm,
        "global_std": global_std,
        "realized_skew": realized_skew,
        "amplitudes": amplitudes,
        "centres": centres,
        "sigma": sigma,
    }


def make_flat_fixture(n: int, d: int, D: int, seed: int, density_skew: float = 0.0) -> dict:
    """A graph-of-function fixture with all bump amplitudes forced to exactly zero, so
    the returned ``H_norm`` (the ANALYTIC ground truth) is exactly zero everywhere -- not
    merely small. On a manifold whose true curvature is zero exactly, ALL apparent
    ``||H||`` an estimator reports is a sampling artifact, if it reports any at all.

    IMPORTANT FINDING (see ``test_density_correction_removes_bias`` and this plan's
    SUMMARY.md for the full derivation): on THIS exactly-linear fixture, with the exact
    exponential-linear density tilt ``_sample_skewed_ball`` implements,
    ``centroid_mean_curvature`` reports ``||H||`` at the float64 noise floor
    (~1e-14) REGARDLESS of ``density_skew`` and REGARDLESS of ``density_correct`` --
    there is no bias here for the correction to remove. This is a mathematical
    certainty, not an implementation gap: an exactly rank-``d`` point cloud makes
    ``local_tangent_basis``'s SVD recover the true tangent subspace EXACTLY (independent
    of how points are weighted within it), so any centroid displacement -- weighted or
    not -- lies exactly in that subspace and the normal projection removes it completely
    (this generalizes plan 02.5-01's Pitfall-3 regression guard to this density model).
    Separately, the specified density model's log-density is exactly LINEAR in ``x``, so
    even the raw (pre-projection) second moment of the tangential coordinate is
    unaffected by the skew to any order. D-06's correction has real, measurable value
    only where the tangent basis is imperfectly estimated -- i.e. on a genuinely curved
    fixture -- which is what ``test_density_correction_removes_bias``'s second half
    demonstrates instead.

    Thin wrapper over ``make_graph_of_function_fixture`` with ``n_bumps=1,
    amplitude=0.0``; same dict shape, same ``density_skew`` semantics.
    """
    return make_graph_of_function_fixture(
        n=n, d=d, D=D, n_bumps=1, seed=seed, amplitude=0.0, density_skew=density_skew
    )


# --- D-05 non-gating cross-check: local quadric ("bowl") fit --------------------------


def quadric_fit_curvature(u: np.ndarray, z: np.ndarray, d: int) -> np.ndarray:
    """RESEARCH.md Pattern 2's local quadric fit -- the D-05 non-gating cross-check.
    NEVER the gating estimator; see ``quadric_mean_curvature``'s docstring for why.

    ``u``: ``(k, d)`` tangent coordinates, already centered at the query point.
    ``z``: ``(k, m)`` normal-space coordinates for the same ``k`` neighbours (``m`` is
    whatever codimension the caller has projected into -- ``quadric_mean_curvature``
    below passes the full ambient residual, ``m = D``). Fits ``z = q(u)`` for a
    quadratic form ``q`` by ordinary least squares and returns the ``(m,)`` mean
    curvature vector, the trace of ``q``'s Hessian, under this module's trace
    convention.

    Design matrix columns: a constant, the ``d`` linear terms ``u_i``, and the
    ``d*(d+1)/2`` quadratic terms ``u_i*u_j`` for ``i <= j``. Solved by
    ``np.linalg.lstsq(A, z, rcond=None)`` exactly -- no shrinkage or truncation term of
    any kind is ever added to this call. D-05 rejected giving this estimator such a
    dial precisely because its strength would become a value that has to be chosen and
    pre-registered blind, with no principled way to set it; the plain least-squares (or,
    once ``k`` is smaller than the column count, minimum-norm least-squares) solution is
    the only fit this function ever computes.

    For a fitted term ``c_ii * u_i^2``, the Hessian entry is ``Hessian_ii = 2*c_ii`` --
    the trace only ever reads off the ``i == j`` (pure-quadratic) columns; the
    off-diagonal ``i != j`` (cross) columns contribute nothing to the trace by
    construction (a cross term ``c_ij * u_i * u_j``, ``i != j``, contributes to the
    OFF-diagonal Hessian entries ``Hessian_ij = Hessian_ji = c_ij``, which the trace
    never sums). This function therefore accumulates ``H += 2.0 * c`` over the ``i == j``
    columns only, with no branch that ever touches an off-diagonal column's coefficient.

    Underdetermined once ``d*(d+1)/2 > k`` -- this is the sample-complexity count the
    ROADMAP re-scope correctly describes for THIS estimator (see
    ``quadric_mean_curvature``'s docstring for why it does not transfer to the gating
    estimator). ``np.linalg.lstsq`` still returns a (minimum-norm) solution in that
    regime rather than raising; the caller (``quadric_mean_curvature``) is responsible
    for flagging the underdetermined case rather than silently trusting the fit.
    """
    u = np.asarray(u, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    k = u.shape[0]
    m = z.shape[1]

    cols = [np.ones(k)] + [u[:, i] for i in range(d)]
    diag_positions = []  # indices into `cols` (offset by 1+d) that are i == j columns
    col_idx = 1 + d
    for i in range(d):
        for j in range(i, d):
            cols.append(u[:, i] * u[:, j])
            if i == j:
                diag_positions.append(col_idx)
            col_idx += 1
    A = np.stack(cols, axis=1)  # (k, 1 + d + d*(d+1)/2)

    coef, *_ = np.linalg.lstsq(A, z, rcond=None)  # (1 + d + d*(d+1)/2, m)

    H = np.zeros(m, dtype=np.float64)
    for pos in diag_positions:
        H += 2.0 * coef[pos]
    return H


def _quadric_tangent_basis(centered: np.ndarray, d: int) -> np.ndarray:
    """Like ``local_tangent_basis``, but -- deliberately -- permits ``d > k``: the
    underdetermined regime this cross-check exists to MEASURE, not avoid.
    ``local_tangent_basis`` (the gating estimator's shared helper) hard-raises in that
    regime by design (it has no legitimate use for an underdetermined tangent estimate);
    this function cannot reuse it here, because ``quadric_mean_curvature`` must be able
    to run to completion and REPORT the underdetermined case rather than being blocked
    from ever reaching it.

    ``centered``: ``(k, D)``. Uses ``np.linalg.svd(centered, full_matrices=True)``
    instead of ``local_tangent_basis``'s ``full_matrices=False``, so the returned basis
    always has exactly ``d`` rows for any ``d <= D``. When ``d <= k`` the two routes
    return numerically identical top-``d`` singular vectors (confirmed to
    float64-epsilon agreement); when ``d > k`` (only reachable when ``n_coefficients =
    d*(d+1)/2 > k``, i.e. exactly the underdetermined case), rows ``k`` through ``d - 1``
    are an arbitrary orthonormal completion of the neighbourhood's own null space --
    genuinely uninformative directions along which every neighbour's ``centered``
    coordinate is EXACTLY zero, so their tangent-coordinate columns in
    ``quadric_fit_curvature``'s design matrix are identically zero and ``lstsq`` handles
    them as ordinary rank deficiency (a minimum-norm solution), not a special case.
    """
    k, D = centered.shape
    if d > D:
        raise ValueError(
            f"_quadric_tangent_basis: d={d} exceeds D={D}; cannot extract a "
            f"{d}-dimensional tangent basis in {D} ambient dimensions."
        )
    _, _, Vt = np.linalg.svd(centered, full_matrices=True)
    return Vt[:d]


def quadric_mean_curvature(X: np.ndarray, k: int, d: int) -> dict:
    """The D-05 non-gating cross-check estimator, per-point wrapper over
    ``quadric_fit_curvature``.

    NON-GATING under D-05: only ``centroid_mean_curvature`` (D-05's gating estimator)
    ever decides a PASS/FAIL. The reason is sample complexity, and it is specific to
    THIS estimator, not a general property of curvature estimation from a neighbourhood.
    The full local quadric fit needs ``d*(d+1)/2`` coefficients per normal direction --
    210 at ``d=20`` -- because it estimates the FULL second fundamental form. The
    centroid/Laplace-Beltrami estimator ``centroid_mean_curvature`` estimates only
    ``H``'s TRACE, as an average over ``k`` neighbour vectors -- one unknown with ``k``
    samples, not 210 unknowns with 15 equations -- so the ``d*(d+1)/2``-vs-``k`` count
    the ROADMAP re-scope warns about is real for this estimator and does NOT transfer to
    the gating one (D-00's reframing).

    Per point: ``k``-NN via ``NearestNeighbors(n_neighbors=k+1)`` (self excluded);
    ``centered = neigh - p``; the local tangent basis ``Vt`` (``(d, D)``) via
    ``_quadric_tangent_basis`` (NOT ``local_tangent_basis`` -- see that helper's own
    docstring for why this estimator needs a separate one that tolerates ``d > k``);
    tangent coordinates ``u = centered @ Vt.T`` (``(k, d)``); and the normal residual
    kept as the full AMBIENT ``(k, D)`` vector ``z = centered - u @ Vt`` rather than
    being re-expressed in an explicit ``(D - d)``-dimensional normal basis. This is a
    deliberate efficiency choice, not a mathematical one: materializing a
    ``(D, D-d)`` normal basis at ``D = 768`` costs memory and compute this cross-check
    does not need, and the two are equivalent because the tangential part of ``z`` is
    identically zero by construction (``u`` was already subtracted out via ``u @ Vt``).
    ``quadric_fit_curvature(u, z, d)`` then returns the fitted mean curvature vector
    directly in AMBIENT coordinates, so no back-transformation out of a normal basis is
    ever needed.

    Returns a dict with ``"H_vec"`` ``(n, D)``, ``"H_norm"`` ``(n,)``,
    ``"underdetermined"`` (``bool``, a property of ``(d, k)`` alone -- the same for every
    point), ``"n_coefficients"`` (``int``, ``d*(d+1)//2``), and ``"coefficient_deficit"``
    (``int``, ``max(0, n_coefficients - k)``).

    Does NOT raise when underdetermined -- it returns the fitted value with the flag
    set. The CONTRAST between a determined quadric fit (small ``d``, generous ``k``) and
    an underdetermined one (the PU regime's ``d=20``, ``k=15``) is itself the evidence
    D-00's reframing needs reported alongside the gating result, and raising would
    destroy that evidence rather than surface it.
    """
    X = np.asarray(X, dtype=np.float64)
    n, D = X.shape
    n_coefficients = d * (d + 1) // 2
    coefficient_deficit = max(0, n_coefficients - k)
    underdetermined = n_coefficients > k

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nbrs.kneighbors(X)  # idx[:, 0] is the point itself

    H_vec = np.zeros((n, D), dtype=np.float64)
    for i in range(n):
        neigh_idx = idx[i, 1:]  # (k,), excludes self
        neigh = X[neigh_idx]  # (k, D)
        p = X[i]
        centered = neigh - p
        Vt = _quadric_tangent_basis(centered, d)  # (d, D)
        u = centered @ Vt.T  # (k, d)
        z = centered - u @ Vt  # (k, D), the ambient normal residual
        H_vec[i] = quadric_fit_curvature(u, z, d)

    return {
        "H_vec": H_vec,
        "H_norm": mean_curvature_norm(H_vec),
        "underdetermined": bool(underdetermined),
        "n_coefficients": int(n_coefficients),
        "coefficient_deficit": int(coefficient_deficit),
    }


def estimator_agreement(h_centroid_norm: np.ndarray, h_quadric_norm: np.ndarray) -> dict:
    """Quantifies agreement between the two independent D-05 estimators -- the gating
    centroid/Laplace-Beltrami estimator and the non-gating local quadric fit -- and
    resolves CONTEXT.md's discretion point as REPORT, NEVER BLOCK: this function never
    raises and never gates anything, no matter how large the disagreement.

    Rationale: the two estimators have different bias mechanisms (finite-radius ``O(r^2)``
    bias for the centroid estimator vs. sample-starved least-squares variance for the
    quadric fit once ``d*(d+1)/2`` approaches ``k``), so at the underdetermined end of
    the ``d`` ladder they are EXPECTED to diverge. Making that divergence a blocking rule
    would convert an expected, informative signal into a spurious failure -- exactly the
    mistake D-05's "agreement hardens the result; divergence is itself informative"
    framing warns against. The divergence is reported in the stage-1 verdict artifact's
    non-gating ``context_metrics`` block (plan 02.5-04) and discussed in
    ``02.5-FINDINGS.md``, never used to fail a gate here or anywhere downstream.

    Returns a dict with ``"agreement_spearman"`` (``float``, rank agreement between the
    two estimators' ``||H||`` fields, via ``spearmanr``) and
    ``"agreement_median_rel_diff"`` (``float``,
    ``median(|a - b| / max(|a|, |b|, floor))`` with
    ``floor = 1e-3 * median(max(|a|, |b|))`` elementwise, the same near-zero-denominator
    guard ``median_relative_error`` uses, so neither estimator reporting near-zero
    ``||H||`` can blow the ratio up to ``inf``).
    """
    a = np.asarray(h_centroid_norm, dtype=np.float64)
    b = np.asarray(h_quadric_norm, dtype=np.float64)
    rho = float(spearmanr(a, b).statistic)

    elementwise_max = np.maximum(np.abs(a), np.abs(b))
    floor = 1e-3 * np.median(elementwise_max)
    denom = np.maximum(elementwise_max, floor)
    rel_diff = float(np.median(np.abs(a - b) / denom))

    return {"agreement_spearman": rho, "agreement_median_rel_diff": rel_diff}


# --- D-02 permutation-null calibration of the Spearman threshold ----------------------


def permutation_null(
    h_true_norm: np.ndarray,
    h_est_norm: np.ndarray,
    n_resamples: int,
    seed: int,
    quantile: float,
    statistic_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> dict:
    """D-02's permutation-null calibration: shuffles the pairing between the true and
    estimated curvature fields to build a chance distribution of the gating statistic,
    and reads a threshold off a caller-supplied quantile of it -- so the gate is
    calibrated against measured chance, never invented.

    ``statistic_fn`` (added by plan 02.5-07, generalizing this function to Section 3h's
    SECOND, independently-gating statistic): an optional ``(x, y) -> float`` callable.
    Defaults to ``None``, which reproduces this function's ORIGINAL behaviour exactly --
    ``lambda x, y: spearmanr(x, y).statistic`` -- so every existing call site (D-01's
    ``spearman_rho`` calibration, unchanged since plan 02.5-03) is unaffected. Passing a
    different callable (e.g. a closure over :func:`quantile_bin_concordance` with its own
    pre-registered ``n_bins``/``pair_seed``/``n_pairs`` already bound in) calibrates the
    SAME null-quantile/permutation-count machinery against a different rank statistic --
    Section 3g's "clears_null(0.99, 999)" phrasing applies generically to whichever gating
    statistic it is asked of, not only to Spearman.

    ``permutation_test``'s ``permutation_type="pairings"`` re-pairs the two samples
    independently on each resample (confirmed directly against this scipy version: both
    ``h_true_norm``'s and ``h_est_norm``'s row order are permuted on every resample, not
    just one held fixed) -- for any statistic (Spearman included) this reproduces the
    correct "any pairing is equally likely" null, since a statistic evaluated on two
    independently-permuted samples has the same distribution as one evaluated with only
    one sample permuted relative to a fixed other. ``statistic_fn`` MUST therefore treat
    its two arguments purely as a self-consistent, already-paired dataset (never assume
    either argument is the caller's original, unshuffled array) -- exactly how
    :func:`quantile_bin_concordance`'s own ``bin_true`` construction already works, since
    it derives bin membership from whichever array is passed as ``h_true_norm``, not from
    any external, unshuffled reference.

    Uses ``scipy.stats.permutation_test`` (already pinned, scipy 1.18.0) with
    ``permutation_type="pairings"``, ``alternative="greater"``, and
    ``rng=np.random.default_rng(seed)``, rather than a hand-rolled shuffle-and-recompute
    loop -- the same choice RESEARCH.md's Don't-Hand-Roll table recommends, and
    ``mknn.permutation_null``'s hand-rolled loop (read for this codebase's naming
    convention only, never copied) is not repeated a third, different way here.

    ``quantile`` has NO default value. It is a pre-registered constant (plan
    02.5-06), and giving it a default here is exactly how such a constant gets inherited
    by accident rather than chosen explicitly at every call site -- the same discipline
    ``centroid_mean_curvature`` already applies to its own ``d`` argument (D-07).

    Before computing anything, guards the inputs and raises ``ValueError`` naming the
    offending array if either array contains a non-finite value, if the two arrays have
    different lengths, or if either array is constant (``np.ptp(arr) == 0``) -- a
    constant array makes Spearman rank correlation undefined, and the resulting ``NaN``
    must never be allowed to reach a gate. This mirrors ``cae.py``'s own discipline
    (``verdict_from_metrics``, ~lines 1077-1085: "a missing or non-finite measurement can
    never become a PASS") applied one layer earlier, at calibration time rather than
    verdict time. It is also exactly why a ``d``-sphere (constant ``H`` everywhere) was
    ruled out as a fixture (D-03): a constant true-curvature field leaves Spearman
    undefined against ANY estimate, however good.

    Returns a dict with ``"observed_rho"`` (``float``, the actual, unshuffled value of
    ``statistic_fn`` -- named ``observed_rho`` for backward compatibility even when
    ``statistic_fn`` is not Spearman), ``"null_quantile"`` (``float``, the requested
    quantile, echoed back),
    ``"null_threshold"`` (``float``,
    ``np.quantile(result.null_distribution, quantile)``), ``"null_mean"`` (``float``),
    ``"null_std"`` (``float``), ``"n_resamples"`` (``int``, echoed), ``"seed"`` (``int``,
    echoed), and ``"clears_null"`` (``bool``, ``observed_rho > null_threshold`` --
    strictly greater-than).

    CALIBRATES, does NOT SET, the gate: stage 1's actual gate is fixed only by the
    ratified ``02.5-PREREGISTRATION.md`` (plan 02.5-06), which pre-registers
    ``n_resamples``, ``quantile``, and an additional absolute floor before this function
    is ever run on a real fixture. Anyone reading a ``clears_null`` value computed before
    that document is committed is reading a diagnostic, not a verdict.
    """
    h_true_norm = np.asarray(h_true_norm, dtype=np.float64)
    h_est_norm = np.asarray(h_est_norm, dtype=np.float64)

    if not np.all(np.isfinite(h_true_norm)):
        raise ValueError("permutation_null: h_true_norm contains a non-finite value.")
    if not np.all(np.isfinite(h_est_norm)):
        raise ValueError("permutation_null: h_est_norm contains a non-finite value.")
    if h_true_norm.shape[0] != h_est_norm.shape[0]:
        raise ValueError(
            f"permutation_null: h_true_norm (len={h_true_norm.shape[0]}) and "
            f"h_est_norm (len={h_est_norm.shape[0]}) have different lengths."
        )
    if np.ptp(h_true_norm) == 0:
        raise ValueError(
            "permutation_null: h_true_norm is constant -- Spearman rank correlation is "
            "undefined against a constant field."
        )
    if np.ptp(h_est_norm) == 0:
        raise ValueError(
            "permutation_null: h_est_norm is constant -- Spearman rank correlation is "
            "undefined against a constant field."
        )

    def _stat(x: np.ndarray, y: np.ndarray) -> float:
        if statistic_fn is None:
            return float(spearmanr(x, y).statistic)
        return float(statistic_fn(x, y))

    rng = np.random.default_rng(seed)
    result = permutation_test(
        (h_true_norm, h_est_norm),
        _stat,
        permutation_type="pairings",
        alternative="greater",
        n_resamples=n_resamples,
        rng=rng,
    )

    observed_rho = float(result.statistic)
    null_threshold = float(np.quantile(result.null_distribution, quantile))

    return {
        "observed_rho": observed_rho,
        "null_quantile": float(quantile),
        "null_threshold": null_threshold,
        "null_mean": float(np.mean(result.null_distribution)),
        "null_std": float(np.std(result.null_distribution)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "clears_null": bool(observed_rho > null_threshold),
    }


# --- one-call stage-1 measurement bundle (plan 02.5-07's sweep runner) ----------------


class QuadricTimeout(Exception):
    """Raised internally by :func:`measure_cell`'s SIGALRM handler when the non-gating
    quadric cross-check exceeds its own ``quadric_timeout_s`` allowance -- never raised by
    ``quadric_mean_curvature`` itself, and never allowed to escape :func:`measure_cell` (it
    is caught there and turned into ``quadric_timed_out: True`` in the returned dict)."""


def _quadric_alarm_handler(signum, frame):  # noqa: ARG001 -- required signal handler shape
    raise QuadricTimeout()


class _quadric_timeout_guard:
    """POSIX (``signal.setitimer``) context manager, scoped ONLY to the non-gating quadric
    cross-check inside :func:`measure_cell`. Always cancels its own pending alarm on exit
    (even on an exception other than the timeout itself), so a fast quadric fit never leaves
    a stray alarm armed for whatever runs after it. A ``None`` or non-positive ``seconds``
    disables the guard entirely (no alarm armed), so existing callers that never pass
    ``quadric_timeout_s`` see no behavioural change at all."""

    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds

    def __enter__(self):
        if self.seconds is not None and self.seconds > 0:
            signal.signal(signal.SIGALRM, _quadric_alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seconds is not None and self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return False


def measure_cell(
    fixture: dict,
    k: int,
    d: int,
    k_density: int,
    density_correct: bool,
    n_resamples: int,
    seed: int,
    quantile: float,
    region_n_bins: int,
    region_pair_seed: int,
    region_n_pairs: int,
    quadric_timeout_s: Optional[float] = None,
    precomputed_H_vec: Optional[np.ndarray] = None,
) -> dict:
    """The single call plan 02.5-07's sweep runner makes per grid cell, so the runner
    contains orchestration only and no statistics of its own.

    ``fixture``: a dict from ``make_swiss_roll_fixture``, ``make_graph_of_function_fixture``,
    or ``make_flat_fixture`` -- anything shaped ``{"X", "H_norm", ...}``.

    Exactly TWO returned keys are gating: ``"spearman_rho"`` and
    ``"quantile_bin_concordance"`` -- Section 3h's ratified option-scale-C, added by plan
    02.5-07 and SUPERSEDING this function's original single-gate design as shipped by plan
    02.5-03. A cell passes stage 1 only if BOTH clear their own null-calibrated threshold
    (Section 3h); this function does not itself apply that rule -- it only measures both
    statistics and both of their calibrated nulls, leaving the threshold comparison to
    :func:`verdict_from_stage1_metrics`, called by the runner. Every other key is
    context -- the same gating/non-gating split every phase since 02.2 has used. Consuming
    code must never branch a PASS/FAIL decision on any key other than these two.

    ``region_n_bins``/``region_pair_seed``/``region_n_pairs``: pre-registered constants
    (``REGION_N_BINS``, ``REGION_PAIR_SEED``, ``REGION_PAIR_COUNT``, Section 3g) threaded
    through to :func:`quantile_bin_concordance` and its own null calibration -- no
    defaults, same discipline as every other pre-registered argument this function and
    :func:`permutation_null` already apply.

    ``quadric_timeout_s`` (added by plan 02.5-07, ``Optional[float]``, default ``None`` --
    NOT a pre-registered constant, this function's own orchestration knob): the non-gating
    local-quadric cross-check (``quadric_mean_curvature``) is, at the PU regime's own
    ``D=768``, ``d=20``, measured at roughly 6-8 minutes of CPU time for a single cell --
    its least-squares design matrix has ``d*(d+1)/2`` columns, and the per-point Python
    loop pays that cost ``n`` times. When ``quadric_timeout_s`` is ``None`` or non-positive
    (the default -- every existing call site is unaffected), this function behaves exactly
    as it always has: the quadric cross-check runs to completion, however long that takes.
    When a positive value is given, ONLY the quadric cross-check (never the two GATING
    statistics or their null calibrations, which are computed first and are comparatively
    cheap even at the PU regime's own scale) is wrapped in a ``signal.setitimer``-based
    timeout; if it does not finish in time, the returned dict carries
    ``"quadric_timed_out": True`` and every quadric-derived key (``"quadric_spearman_rho"``,
    ``"agreement_spearman"``, ``"agreement_median_rel_diff"``, ``"quadric_underdetermined"``)
    set to ``None``, while the two GATING statistics and their nulls are still returned,
    valid and usable -- a cell's boundary-relevant measurement is preserved even when its
    non-gating context could not be completed within budget. This is the mechanism plan
    02.5-07's sweep runner uses to keep the full pre-registered grid tractable within its
    own wall-clock envelope without discarding gating data whenever only the (non-gating)
    quadric check is what ran long -- see ``curvature_feasibility_sweep_run.py``'s own
    module docstring for the measured cost this responds to. POSIX-only (``signal.SIGALRM``
    via ``signal.setitimer``); this codebase already runs on Linux exclusively.

    Returns a flat dict:
      - gating: ``"spearman_rho"`` (``float``, from ``spearman_gate_statistic`` on the
        centroid estimator's ``||H||`` against the fixture's ``H_norm``) and
        ``"quantile_bin_concordance"`` (``float``, from :func:`quantile_bin_concordance`
        on the same two fields)
      - spearman null (D-02): every key ``permutation_null`` returns, prefixed ``"null_"``
        where the key does not already start with that prefix (``"null_quantile"``,
        ``"null_threshold"``, ``"null_mean"``, ``"null_std"`` are already prefixed and
        pass through unchanged; ``"observed_rho"``, ``"n_resamples"``, ``"seed"``, and
        ``"clears_null"`` become ``"null_observed_rho"``, ``"null_n_resamples"``,
        ``"null_seed"``, and ``"null_clears_null"``)
      - region null (Section 3h, plan 02.5-07): the SAME ``permutation_null`` machinery,
        recalibrated against :func:`quantile_bin_concordance` via its ``statistic_fn``
        argument, with every key prefixed ``"region_null_"`` instead (distinct prefix, so
        the two null dicts never collide) -- ``"region_null_observed_rho"`` is this cell's
        unshuffled ``quantile_bin_concordance`` value (named ``observed_rho`` inside
        ``permutation_null`` for both statistics, per that function's own backward-
        compatible key naming), ``"region_null_threshold"``, ``"region_null_mean"``,
        ``"region_null_std"``, ``"region_null_clears_null"``, etc.
      - non-gating context: ``"median_relative_error"``, ``"quadric_spearman_rho"``,
        ``"agreement_spearman"``, ``"agreement_median_rel_diff"``,
        ``"quadric_underdetermined"``, ``"quadric_n_coefficients"``,
        ``"quadric_coefficient_deficit"``, ``"quadric_timed_out"`` (``bool``, ``True``
        only when ``quadric_timeout_s`` was given and exceeded -- the four keys just
        named are then all ``None``), ``"realized_skew"`` (only when the fixture
        dict itself carries one -- ``make_swiss_roll_fixture`` does not), ``"n"``,
        ``"d"``, ``"D"``, ``"k"``, ``"k_density"``, ``"density_correct"``,
        ``"region_n_bins"``, ``"region_pair_seed"``, ``"region_n_pairs"``, ``"seed"``
        (this plain top-level echo of the caller's ``seed``, distinct from
        ``"null_seed"``/``"region_null_seed"`` above, which are the same value but reached
        through ``permutation_null``'s own return dicts)
      - timing: ``"elapsed_s"`` (``float``, wall-clock seconds for this whole call --
        dominated, at the PU regime's own ``D=768``, by the non-gating quadric
        cross-check's per-point least-squares fit, not by either gating statistic;
        see plan 02.5-07-SUMMARY.md's Performance section for the measured breakdown)

    Every value is a plain Python ``float``, ``int``, ``bool``, ``str``, or (only the four
    quadric-derived keys, only when ``quadric_timed_out`` is ``True``) ``None`` -- never a
    numpy scalar -- so the runner can hand this dict straight to ``cache.json_cache`` /
    ``json.dumps`` with no conversion pass (``None`` serializes to JSON ``null`` natively).

    ``precomputed_H_vec`` (added by plan 02.5-07, coordinator-directed, ``Optional[np.ndarray]
    = None``, NOT a pre-registered constant): when ``None`` (the default -- every existing
    call site is unaffected), this function computes ``centroid_mean_curvature`` itself,
    exactly as it always has. When an ``(n, D)`` array is supplied, it is used AS-IS in
    place of that internal call -- ``density_correct`` and ``k_density`` are then used only
    for this dict's own ``"density_correct"``/``"k_density"`` context-echo fields, never to
    recompute anything. This lets a caller share one
    :func:`centroid_mean_curvature_both_densities` pass across two ``measure_cell`` calls
    (one per ``density_correct`` value) instead of paying for the k-NN query and every
    per-point tangent-basis SVD twice -- ``curvature_feasibility_sweep_run.py`` is exactly
    such a caller. Bit-identical to the uncached path by construction: the supplied array
    IS the same array :func:`centroid_mean_curvature_both_densities`'s own bit-identity
    test (``test_centroid_mean_curvature_both_densities_is_bit_identical``) already proves
    equals an independent :func:`centroid_mean_curvature` call.
    """
    t0 = time.perf_counter()

    X = np.asarray(fixture["X"], dtype=np.float64)
    h_true_norm = np.asarray(fixture["H_norm"], dtype=np.float64)
    n, D = X.shape

    if precomputed_H_vec is not None:
        H_centroid = precomputed_H_vec
    else:
        H_centroid = centroid_mean_curvature(
            X,
            k=k,
            d=d,
            density_correct=density_correct,
            k_density=k_density if density_correct else None,
        )
    h_est_norm = mean_curvature_norm(H_centroid)

    spearman_rho = spearman_gate_statistic(h_est_norm, h_true_norm)
    med_rel_err = median_relative_error(h_est_norm, h_true_norm)

    null_result = permutation_null(h_true_norm, h_est_norm, n_resamples, seed, quantile)
    null_prefixed = {}
    for key, value in null_result.items():
        pref_key = key if key.startswith("null_") else f"null_{key}"
        null_prefixed[pref_key] = value

    region_rho = quantile_bin_concordance(
        h_est_norm,
        h_true_norm,
        n_bins=region_n_bins,
        pair_seed=region_pair_seed,
        n_pairs=region_n_pairs,
    )

    def _region_stat(x: np.ndarray, y: np.ndarray) -> float:
        # permutation_null's samples tuple is (h_true_norm, h_est_norm); quantile_bin_concordance's
        # own signature is (h_est_norm, h_true_norm) -- map x (h_true-side) to h_true_norm
        # and y (h_est-side) to h_est_norm here, matching that convention.
        return quantile_bin_concordance(
            h_est_norm=y,
            h_true_norm=x,
            n_bins=region_n_bins,
            pair_seed=region_pair_seed,
            n_pairs=region_n_pairs,
        )

    region_null_result = permutation_null(
        h_true_norm, h_est_norm, n_resamples, seed, quantile, statistic_fn=_region_stat
    )
    region_null_prefixed = {}
    for key, value in region_null_result.items():
        # permutation_null's own keys already carry a "null_" prefix on 4 of 8 entries
        # (null_quantile/null_threshold/null_mean/null_std) -- REPLACE that prefix with
        # "region_" (giving region_null_threshold, not the double-prefixed
        # region_null_null_threshold); the remaining 4 (observed_rho/n_resamples/seed/
        # clears_null) get "region_null_" prepended directly, exactly mirroring the
        # spearman null's own "null_" prefixing logic above one level up.
        pref_key = f"region_{key}" if key.startswith("null_") else f"region_null_{key}"
        region_null_prefixed[pref_key] = value

    quadric_fields: Dict[str, Any]
    try:
        with _quadric_timeout_guard(quadric_timeout_s):
            quadric_result = quadric_mean_curvature(X, k=k, d=d)
            agreement = estimator_agreement(h_est_norm, quadric_result["H_norm"])
            quadric_spearman_rho = spearman_gate_statistic(quadric_result["H_norm"], h_true_norm)
        quadric_fields = {
            "quadric_timed_out": False,
            "quadric_spearman_rho": float(quadric_spearman_rho),
            "agreement_spearman": float(agreement["agreement_spearman"]),
            "agreement_median_rel_diff": float(agreement["agreement_median_rel_diff"]),
            "quadric_underdetermined": bool(quadric_result["underdetermined"]),
            "quadric_n_coefficients": int(quadric_result["n_coefficients"]),
            "quadric_coefficient_deficit": int(quadric_result["coefficient_deficit"]),
        }
    except QuadricTimeout:
        _n_coef = d * (d + 1) // 2
        quadric_fields = {
            "quadric_timed_out": True,
            "quadric_spearman_rho": None,
            "agreement_spearman": None,
            "agreement_median_rel_diff": None,
            "quadric_underdetermined": None,
            "quadric_n_coefficients": int(_n_coef),
            "quadric_coefficient_deficit": int(max(0, _n_coef - k)),
        }

    result = {
        "spearman_rho": float(spearman_rho),
        "quantile_bin_concordance": float(region_rho),
        **null_prefixed,
        **region_null_prefixed,
        "median_relative_error": float(med_rel_err),
        **quadric_fields,
        "n": int(n),
        "d": int(d),
        "D": int(D),
        "k": int(k),
        "k_density": int(k_density),
        "density_correct": bool(density_correct),
        "region_n_bins": int(region_n_bins),
        "region_pair_seed": int(region_pair_seed),
        "region_n_pairs": int(region_n_pairs),
        "seed": int(seed),
        "elapsed_s": float(time.perf_counter() - t0),
    }
    if "realized_skew" in fixture:
        result["realized_skew"] = float(fixture["realized_skew"])
    return result


# --- gate constants and direction-aware verdict functions (D-01, D-12) ----------------

STAGE1_GATING_METRICS: Tuple[str, ...] = ("spearman_rho", "quantile_bin_concordance")
STAGE2_GATING_METRICS: Tuple[str, ...] = ("chart_vs_raw_margin", "seed_spread")

# Extended by plan 02.5-07 (Section 3h's ratified option-scale-C): stage 1 now gates on
# TWO independent statistics, not one. `spearman_rho` alone was this tuple's sole entry
# as shipped by plan 02.5-04; `quantile_bin_concordance` (Section 3e/3g, implemented by
# plan 02.5-07) is added because the ratified pre-registration requires BOTH to clear,
# each against its own null-calibrated threshold and its own absolute floor, for a cell
# to pass -- neither is demoted to non-gating context (Section 3h's own wording: "Both
# statistics are gating ... only *which measurements decide PASS/FAIL* is now 'both', not
# 'one, with the other as context'"). `_apply_gates` below needed no change to support
# this -- it already iterates `gate_names` generically; only this tuple and the
# `GATE_DIRECTIONS` entry below needed extending.

# OQ-5 (surfaced during planning, not in RESEARCH.md): `cae.verdict_from_metrics` is not
# delegated to here. First, its `topoae.py`-style positional slot remap requires both
# tuples to have exactly three entries (`topoae.py:632-636`); stage 1 now has two gating
# metrics and stage 2 has two, so no such remap exists at either count. Second,
# `cae.verdict_from_metrics` applies strict less-than to every gate, while
# `spearman_rho`, `quantile_bin_concordance`, and `chart_vs_raw_margin` are greater-than
# gates -- sign-flipping the metric before delegating would put a transformation between
# the measurement and the tested comparison, precisely the opacity the delegation pattern
# exists to avoid. `_apply_gates` below therefore mirrors `cae.py:1077-1097`'s
# guard-then-compare discipline line for line but does not call it; the sealed module's
# own gating-metric tuple is never monkeypatched and its verdict function is never
# called.
GATE_DIRECTIONS: Dict[str, str] = {
    "spearman_rho": "greater",
    "quantile_bin_concordance": "greater",
    "chart_vs_raw_margin": "greater",
    "seed_spread": "less",
}

STAGE1_VERDICT_RULE = (
    "PASS requires BOTH spearman_rho AND quantile_bin_concordance strictly greater than "
    "their OWN thresholds (Section 3h's ratified option-scale-C, added by plan 02.5-07); "
    "a value exactly at either threshold does not clear it, and either gate alone failing "
    "fails the cell. There is no MARGINAL tier: every non-PASS outcome routes to the same "
    "halt-for-user-decision consequence (D-04), so a middle tier would carry no distinct "
    "consequence. Each threshold is the maximum of that statistic's OWN permutation-null "
    "quantile and its OWN absolute floor (SPEARMAN_ABSOLUTE_FLOOR=0.50 / "
    "REGION_ABSOLUTE_FLOOR=0.4750), both fixed in 02.5-PREREGISTRATION.md."
)

STAGE2_VERDICT_RULE = (
    "PASS requires chart_vs_raw_margin strictly greater than its threshold AND "
    "seed_spread strictly less than its threshold; both strict; no MARGINAL tier. "
    "Routing (D-12): charts win -> Phase 3 uses CAE charts; raw points win or tie inside "
    "the margin -> fork to the Ollivier-Ricci phase; neither clears the null -> the "
    "branch resolved at 02.5-PREREGISTRATION-STAGE2.md's ratification checkpoint (plan "
    "02.5-10)."
)


def _apply_gates(
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
    gate_names: Tuple[str, ...],
    rule_name: str,
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """One tested, direction-aware, guard-first comparison function both stage verdict
    functions call -- the strict comparison itself is performed only here, never
    re-implemented at either call site.

    Guard first, compare second (mirrors ``cae.py:1077-1085``'s discipline in intent):
    for every name in ``gate_names``, before any comparison runs, raises ``ValueError``
    naming that gate if it is absent from ``metrics``; raises ``ValueError`` naming the
    gate and its value if ``float(metrics[name])`` is not finite (via ``math.isfinite``);
    raises ``ValueError`` naming the gate if it is absent from ``thresholds``; and raises
    ``ValueError`` naming the gate if it is absent from :data:`GATE_DIRECTIONS`. A
    missing or non-finite measurement can never become a PASS -- halting is the honest
    behaviour, since emitting a FAIL would be indistinguishable from a measured FAIL.

    Then, for each gate, looks up ``GATE_DIRECTIONS[name]``: ``"greater"`` means
    ``passed = value > threshold``, ``"less"`` means ``passed = value < threshold``.
    Both strict.

    Returns ``(verdict, gate_detail)`` with ``verdict = "PASS"`` only if every gate
    passed, else ``"FAIL"``, and ``gate_detail[name] = {"value", "threshold",
    "direction", "passed"}``.
    """
    for name in gate_names:
        if name not in metrics:
            raise ValueError(
                f"_apply_gates ({rule_name}): gating metric {name!r} is absent from metrics"
            )
        value = float(metrics[name])
        if not math.isfinite(value):
            raise ValueError(
                f"_apply_gates ({rule_name}): gating metric {name!r} is non-finite "
                f"({value!r}) -- a missing or non-finite measurement can never become a PASS"
            )
        if name not in thresholds:
            raise ValueError(
                f"_apply_gates ({rule_name}): gate {name!r} is absent from thresholds"
            )
        if name not in GATE_DIRECTIONS:
            raise ValueError(
                f"_apply_gates ({rule_name}): gate {name!r} is absent from GATE_DIRECTIONS"
            )

    gate_detail: Dict[str, Dict[str, Any]] = {}
    all_pass = True
    for name in gate_names:
        value = float(metrics[name])
        threshold = thresholds[name]
        direction = GATE_DIRECTIONS[name]
        passed = bool(value > threshold) if direction == "greater" else bool(value < threshold)
        gate_detail[name] = {
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "passed": passed,
        }
        if not passed:
            all_pass = False
    verdict = "PASS" if all_pass else "FAIL"
    return verdict, gate_detail


def verdict_from_stage1_metrics(
    metrics: Dict[str, float], thresholds: Dict[str, float]
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Stage 1's verdict: PASS iff ``spearman_rho`` strictly exceeds its threshold (see
    :data:`STAGE1_VERDICT_RULE`). The strict comparison is performed only inside
    :func:`_apply_gates`, never re-implemented here."""
    return _apply_gates(metrics, thresholds, STAGE1_GATING_METRICS, "stage1")


def verdict_from_stage2_metrics(
    metrics: Dict[str, float], thresholds: Dict[str, float]
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Stage 2's verdict: PASS iff ``chart_vs_raw_margin`` strictly exceeds its threshold
    AND ``seed_spread`` is strictly below its threshold (see :data:`STAGE2_VERDICT_RULE`).
    The strict comparison is performed only inside :func:`_apply_gates`, never
    re-implemented here."""
    return _apply_gates(metrics, thresholds, STAGE2_GATING_METRICS, "stage2")


# --- persistence boundary --------------------------------------------------------------
# Everything above this line is pure: arrays/dicts in, arrays/dicts out, no file I/O.
# Everything below persists through ``cache`` -- the same layout ``cae.py`` uses (see
# that module's own pure-estimator / cache-writer split).


def write_curvature_verdict(
    key: str,
    stage: int,
    metrics: Dict[str, Any],
    thresholds: Dict[str, Any],
    verdict: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """R6's verdict writer, mirrored at 02.5 scope (D-15) -- written on PASS and on FAIL
    alike; there is no branch that skips the write.

    ``stage`` must be ``1`` or ``2``; any other value raises ``ValueError`` naming it.

    Recomputes ``(computed_verdict, gate_detail)`` via :func:`verdict_from_stage1_metrics`
    or :func:`verdict_from_stage2_metrics` (per ``stage``) from the supplied
    ``metrics``/``thresholds`` -- never trusting the caller's ``verdict`` argument
    directly -- and raises ``ValueError`` if ``computed_verdict != verdict`` (mirrors
    ``topoae.py:717-724``'s wording in intent: a verdict artifact whose stored outcome
    disagrees with its own gates must never be written).

    Thresholds live INSIDE the ``cfg`` dict passed to ``cache.json_cache``
    (``topoae.py:695-699``'s documented rationale), so editing a threshold afterwards
    and re-calling with the same ``key``/``stage`` raises ``cache._manifest_matches``'s
    mismatch ``ValueError`` instead of silently re-verdicting.

    Payload: ``key``, ``phase`` (``"02.5"``), ``stage``, ``CURVATURE_VERDICT`` (the
    verdict string), ``gating_metrics`` (the stage's own gate-name tuple as a list, so
    the artifact states its own gate list), ``gate_directions`` (the relevant slice of
    :data:`GATE_DIRECTIONS`), ``metrics``, ``thresholds``, ``gate_detail``,
    ``verdict_rule`` (the stage's rule string), ``curvature_convention``
    (:data:`CURVATURE_CONVENTION`), ``gate_scope`` (this phase's every gate is scoped to
    a local neighbourhood statistic -- ``{"value": "local", "reason": ...}``, following
    the additive ``gate_scope`` annotation plan 02.4-07 added to its own sealed verdict),
    and -- when ``extra`` is supplied -- everything reported but never gated, under the
    single key ``context_metrics``.

    Persisted through ``cache.json_cache(f"curvature_verdict_stage{stage}_{key}", cfg,
    _compute)``. Every value passes through the sealed model module's ``to_native``
    helper before it reaches ``json_cache`` -- imported lazily inside this function (not
    at module level) so this module keeps its numpy-only import posture; this is this
    module's only dependency on that sealed module, a read-only import of one pure
    conversion helper, never a call into its model code.
    """
    if stage not in (1, 2):
        raise ValueError(f"write_curvature_verdict: stage must be 1 or 2, got {stage!r}")

    if stage == 1:
        computed_verdict, gate_detail = verdict_from_stage1_metrics(metrics, thresholds)
        gating_metrics = STAGE1_GATING_METRICS
        verdict_rule = STAGE1_VERDICT_RULE
    else:
        computed_verdict, gate_detail = verdict_from_stage2_metrics(metrics, thresholds)
        gating_metrics = STAGE2_GATING_METRICS
        verdict_rule = STAGE2_VERDICT_RULE

    if computed_verdict != verdict:
        raise ValueError(
            f"write_curvature_verdict: supplied verdict {verdict!r} does not match the "
            f"verdict computed from metrics/thresholds ({computed_verdict!r}) -- refusing "
            "to write a verdict artifact whose stored outcome disagrees with its own gates"
        )

    from . import cae as cae_mod  # lazy: keeps this module's numpy-only import posture

    cfg = {"key": key, "phase": "02.5", "stage": stage, "thresholds": thresholds}

    def _compute() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "key": key,
            "phase": "02.5",
            "stage": stage,
            "CURVATURE_VERDICT": verdict,
            "gating_metrics": list(gating_metrics),
            "gate_directions": {name: GATE_DIRECTIONS[name] for name in gating_metrics},
            "metrics": metrics,
            "thresholds": thresholds,
            "gate_detail": gate_detail,
            "verdict_rule": verdict_rule,
            "curvature_convention": CURVATURE_CONVENTION,
            "gate_scope": {
                "value": "local",
                "reason": (
                    "Every gate in this phase is scoped to a local neighbourhood "
                    "statistic -- there is no global-scoped gate here to distinguish "
                    "it from."
                ),
            },
        }
        if extra:
            payload["context_metrics"] = extra
        return cae_mod.to_native(payload)

    return cache.json_cache(f"curvature_verdict_stage{stage}_{key}", cfg, _compute)


def write_curvature_handoff(key: str, verdict: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """R6's PASS-only handoff writer, mirrored at 02.5 scope (D-15). Raises ``ValueError``
    unless ``verdict == "PASS"`` (mirrors ``topoae.py:765-768``), so a non-PASS caller
    cannot produce a handoff even by mistake.

    The payload must name the FULL estimator contract -- D-15's whole point: naming the
    winning substrate alone would let Phase 3 reconstruct the field under different
    settings than the ones that passed the gate. Required keys, each raising
    ``ValueError`` naming it when absent from ``payload``: ``substrate``
    (``"cae_charts"`` or ``"raw_points"``), ``working_dimension`` (the ``d`` that gated),
    ``neighbourhood_rule`` (a dict with ``k`` and the prose bias-variance justification
    from the pre-registration), ``density_correction`` (a dict with ``enabled`` and
    ``k_density``), ``estimator_variant`` (``"centroid_laplace_beltrami"``),
    ``cache_stems`` (the list of fit/artifact stems the field was computed from),
    ``fit_key``, ``gate_values``, ``thresholds``, ``evidence_criteria`` (prose),
    ``preregistration_sha`` (the git-ancestry-proved commit of
    ``02.5-PREREGISTRATION-STAGE2.md``), and ``activation``.

    ``curvature_convention`` (:data:`CURVATURE_CONVENTION`) is always included.
    ``torch_version`` is read lazily inside this function (``import torch`` here, not at
    module level) so this module keeps its numpy-only import posture; if torch is not
    importable, records the string ``"not-installed"`` rather than raising.
    ``numpy_version`` and a UTC ``timestamp`` are also included. Every value passes
    through the sealed model module's ``to_native`` helper (imported lazily, same
    rationale as :func:`write_curvature_verdict`) before it reaches ``json_cache``.

    Persisted through ``cache.json_cache(f"curvature_handoff_{key}", cfg, _compute)``.
    """
    if verdict != "PASS":
        raise ValueError(
            f"write_curvature_handoff refuses to write a handoff for a non-PASS verdict: "
            f"{verdict!r}"
        )

    required_keys = (
        "substrate",
        "working_dimension",
        "neighbourhood_rule",
        "density_correction",
        "estimator_variant",
        "cache_stems",
        "fit_key",
        "gate_values",
        "thresholds",
        "evidence_criteria",
        "preregistration_sha",
        "activation",
    )
    for name in required_keys:
        if name not in payload:
            raise ValueError(
                f"write_curvature_handoff: payload is missing required key {name!r} -- "
                "the handoff must name the full estimator contract, not merely which "
                "substrate won"
            )

    from . import cae as cae_mod  # lazy: keeps this module's numpy-only import posture

    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = "not-installed"

    cfg = {"key": key, "phase": "02.5"}

    def _compute() -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "key": key,
            "phase": "02.5",
            "verdict": verdict,
            "substrate": payload["substrate"],
            "working_dimension": payload["working_dimension"],
            "neighbourhood_rule": payload["neighbourhood_rule"],
            "density_correction": payload["density_correction"],
            "estimator_variant": payload["estimator_variant"],
            "curvature_convention": CURVATURE_CONVENTION,
            "cache_stems": payload["cache_stems"],
            "fit_key": payload["fit_key"],
            "gate_values": payload["gate_values"],
            "thresholds": payload["thresholds"],
            "evidence_criteria": payload["evidence_criteria"],
            "preregistration_sha": payload["preregistration_sha"],
            "activation": payload["activation"],
            "torch_version": torch_version,
            "numpy_version": np.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return cae_mod.to_native(result)

    return cache.json_cache(f"curvature_handoff_{key}", cfg, _compute)


def clear_stale_curvature_handoff(key: str) -> bool:
    """R6's active deletion, mirrored at 02.5 scope (D-15). Mirrors
    ``topoae.py:794-817`` exactly in behaviour: unlinks both
    ``cache.cache_path(f"curvature_handoff_{key}", "json")`` and its
    ``cache.cache_path(f"curvature_handoff_{key}", "meta.json")`` sidecar when present,
    returns ``True`` if anything was removed and ``False`` otherwise, and never raises
    when nothing is there.

    The sidecar must go too: leaving it behind would let a later ``json_cache`` call see
    a manifest with no artifact, which is how a stale PASS survives a later FAIL.
    """
    json_path = cache.cache_path(f"curvature_handoff_{key}", "json")
    meta_path = cache.cache_path(f"curvature_handoff_{key}", "meta.json")

    removed = False
    if json_path.exists():
        json_path.unlink()
        removed = True
    if meta_path.exists():
        meta_path.unlink()
        removed = True
    return removed
