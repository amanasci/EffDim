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
from scipy.stats import permutation_test, spearmanr
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


def permutation_null(h_true_norm: np.ndarray, h_est_norm: np.ndarray, n_resamples: int, seed: int, quantile: float) -> dict:
    """D-02's permutation-null calibration: shuffles the pairing between the true and
    estimated curvature fields to build a chance distribution of Spearman values, and
    reads a threshold off a caller-supplied quantile of it -- so D-01's gate is
    calibrated against measured chance, never invented.

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

    Returns a dict with ``"observed_rho"`` (``float``, the actual, unshuffled Spearman
    statistic), ``"null_quantile"`` (``float``, the requested quantile, echoed back),
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
        return float(spearmanr(x, y).statistic)

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
