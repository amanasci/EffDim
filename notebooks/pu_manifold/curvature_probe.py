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

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

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


# --- gating estimator (D-05) ------------------------------------------------------------


def centroid_mean_curvature(X: np.ndarray, k: int, d: int) -> np.ndarray:
    """Centroid / Laplace-Beltrami mean-curvature estimator -- the gating estimator (D-05).

    ``X``: ``(n, D)`` point cloud. ``k``: number of nearest neighbours per point
    (excluding self). ``d``: the estimator's own working tangent dimension.

    ``d`` is a REQUIRED positional argument with no default. D-07 bars inheriting
    ``D_FROZEN = 5`` as this phase's working dimension; a default value is exactly how
    such a value gets inherited by accident rather than by an explicit call-site choice.

    Returns ``(n, D)`` mean curvature vector estimates, under this module's trace
    convention (``H = tr(II)``).

    Per point: k-NN via ``NearestNeighbors(n_neighbors=k+1)`` (self excluded from the
    neighbour set); ``centered = neigh - p``; the raw centroid displacement
    ``gap = centered.mean(axis=0)``; the local tangent basis ``Vt`` via
    ``local_tangent_basis``; ``gap`` is projected onto the NORMAL complement via
    ``gap_normal = gap - Vt.T @ (Vt @ gap)`` (the ``(D, D)`` projector is never
    materialized); the empirical local scale ``r2 = mean(||centered||^2)``; and finally
    ``H[i] = gap_normal * (2*(d + 2) / r2)``, the identity ``E[c - p] = (r^2 / (2(d+2))) *
    H`` inverted for ``H``.

    Three known caveats (D-05):
    1. Bias grows like ``r^2`` at finite radius -- the identity this estimator inverts is
       exact only in the limit of vanishing neighbourhood radius; at finite ``k`` (hence
       finite ``r``) the recovered ``H`` has ``O(r^2)`` relative bias.
    2. The estimate is contaminated by non-uniform sampling density unless corrected: a
       neighbourhood with asymmetric local density has a nonzero raw centroid gap even on
       a flat manifold, and that gap is NOT curvature. This function is the uncorrected
       baseline; plan 02.5-02 adds the density correction measured against it.
    3. It yields ``H`` (a vector) and, via ``mean_curvature_norm``, ``||H||`` -- never the
       full second fundamental form ``II``. Recovering ``II`` itself is the underdetermined
       problem D-00 reframes away from; this estimator only ever recovers its trace.
    """
    n, D = X.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nbrs.kneighbors(X)  # idx[:, 0] is the point itself

    H_est = np.zeros((n, D), dtype=np.float64)
    for i in range(n):
        neigh = X[idx[i, 1:]]  # (k, D), excludes self
        p = X[i]
        centered = neigh - p
        gap = centered.mean(axis=0)
        Vt = local_tangent_basis(centered, d)
        gap_normal = gap - Vt.T @ (Vt @ gap)
        r2 = np.mean(np.sum(centered**2, axis=1))
        H_est[i] = gap_normal * (2 * (d + 2) / r2)
    return H_est


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
