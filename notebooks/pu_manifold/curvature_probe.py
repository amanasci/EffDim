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


# --- gating estimator (D-05) ------------------------------------------------------------


def centroid_mean_curvature(X: np.ndarray, k: int, d: int) -> np.ndarray:
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
    non-uniform sampling branch, added by plan 02.5-02 Task 2.

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
        raise NotImplementedError(
            "make_graph_of_function_fixture: density_skew > 0.0 sampling is added by "
            "plan 02.5-02 Task 2."
        )

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
