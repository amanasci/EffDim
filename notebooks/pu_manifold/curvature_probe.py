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

from typing import Optional

import numpy as np
from scipy.special import gammaln
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
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
