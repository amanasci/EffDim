"""Analytic graph controls whose SECOND FUNDAMENTAL FORM actually varies across the domain.

**Why this module exists.** Every control the milestone has used to test curvature RANKING has
a constant second fundamental form, and that is a property of the fixtures rather than of the
estimator:

    ``synthetic_controls.make_flat_control``    ``II = 0``                     -- no rank at all
    ``synthetic_controls.make_sphere_control``  ``II`` constant by symmetry    -- ``||H||`` constant,
                                                                                 rank UNDEFINED
                                                                                 (the sealed d=4
                                                                                 record literally
                                                                                 stores ``rho: None``)
    ``synthetic_controls.make_saddle_control``  ``II = diag(signs)``, constant -- ``||H||`` varies
                                                                                 ONLY through ``g``

For any graph ``M = {(x, f(x))}`` the mean curvature is ``H = tr_g(II)`` with
``g = I + grad grad^T``, so when ``D^2 f`` is constant the whole spatial variation of ``||H||``
comes from the metric tilt ``1/(1 + |grad|^2)`` and nothing from the geometry the estimator is
supposed to be measuring. **Every quadratic graph has constant ``D^2 f``.** Making a quadratic
non-minimal does not fix this -- see :func:`make_quadratic_graph_control`'s docstring for the
measured consequence, which runs the opposite way to intuition.

So a ``rho ~ 0`` on those fixtures is consistent with an estimator that works perfectly and a
fixture that has nothing rankable in it, and the milestone has never been able to tell those
apart. This module supplies the missing arm: analytic surfaces, still closed-form, whose
``D^2 f`` genuinely varies point to point.

**Delegation, deliberately total.** No curvature mathematics is written here. Every fixture
computes ``grad`` and ``hess`` in closed form, hands them to the sealed
``curvature_probe.graph_mean_curvature``, and embeds via the sealed
``synthetic_controls.rotate_and_pad`` -- the same two calls ``make_saddle_control`` makes, in
the same order, so a result on these fixtures is directly comparable to the sealed controls
and any difference is the surface rather than the pipeline. Each fixture's hand-derived
``grad``/``hess`` is cross-checked against central finite differences in this module's tests,
matching ``synthetic_controls``' own standard.

Returns the same dict shape every control in this family returns (``X``, ``x_param``,
``H_vec``, ``H_norm``, ``global_std``) plus per-fixture provenance, so existing runners can
consume these without changes.
"""

from typing import Any, Dict, Optional

import numpy as np

from . import curvature_probe, synthetic_controls

CURVATURE_CONVENTION = "trace"
"""Drift guard, matching ``chart_curvature.CURVATURE_CONVENTION`` and
``synthetic_controls.CURVATURE_CONVENTION``. Every value here comes from the sealed
``graph_mean_curvature`` and is never rescaled in this module."""

if CURVATURE_CONVENTION != synthetic_controls.CURVATURE_CONVENTION:
    raise ImportError(
        f"varying_ii_controls.CURVATURE_CONVENTION={CURVATURE_CONVENTION!r} disagrees with "
        f"synthetic_controls.CURVATURE_CONVENTION="
        f"{synthetic_controls.CURVATURE_CONVENTION!r}. Two conventions in one comparison is "
        f"a silent scale error, not a style difference."
    )


def second_fundamental_form_variation(hess: np.ndarray) -> Dict[str, float]:
    """How much ``D^2 f`` actually varies over the sampled domain -- the diagnostic that
    separates "the estimator cannot rank this" from "there is nothing here to rank".

    ``hess``: ``(n, m, d, d)`` as handed to ``curvature_probe.graph_mean_curvature``.

    Returns the coefficient of variation of ``||D^2 f||_F`` across points, its spread ratio,
    and the raw percentiles. **A CV of exactly 0 means the surface has no second-order
    spatial structure at all**, so any ``||H||`` variation it does show is metric tilt. Both
    the saddle and every other quadratic graph score exactly 0 here.
    """
    h = np.asarray(hess, dtype=np.float64)
    fro = np.linalg.norm(h.reshape(h.shape[0], -1), axis=1)
    mean = float(fro.mean())
    p05, p95 = float(np.percentile(fro, 5)), float(np.percentile(fro, 95))
    return {
        "hess_fro_mean": mean,
        "hess_fro_cv": float(fro.std(ddof=1) / mean) if mean > 0 else 0.0,
        "hess_fro_p05": p05,
        "hess_fro_p95": p95,
        "hess_fro_spread": float(p95 / p05) if p05 > 0 else float("inf"),
    }


def _finish(
    x: np.ndarray,
    f_x: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    D: int,
    seed: int,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """The shared tail every control in this family runs: sealed curvature, graph assembly,
    sealed rotate-and-pad. Identical in order and content to ``make_saddle_control``'s tail."""
    H_local = curvature_probe.graph_mean_curvature(grad, hess)
    X_local = np.concatenate([x, f_x[:, None]], axis=1)
    X, H_vec, global_std = synthetic_controls.rotate_and_pad(X_local, H_local, D, seed)
    out = {
        "X": X,
        "x_param": x,
        "H_vec": H_vec,
        "H_norm": np.linalg.norm(H_vec, axis=-1),
        "global_std": global_std,
        "ii_variation": second_fundamental_form_variation(hess),
        "curvature_convention": CURVATURE_CONVENTION,
    }
    out.update(extra)
    return out


def make_quadratic_graph_control(
    n: int,
    d: int,
    D: int,
    seed: int,
    eigenvalues: Optional[np.ndarray] = None,
    domain_radius: float = 2.0,
) -> Dict[str, Any]:
    """General quadratic graph ``f(x) = 0.5 x^T diag(a) x``, ``a = eigenvalues``.

    Generalises ``synthetic_controls.make_saddle_control``, which is the special case
    ``a = shuffled(+1 x d//2, -1 x rest)``. Provided so the "is the problem MINIMALITY?"
    hypothesis can be tested directly rather than argued: set ``a = ones(d)`` for a bowl with
    ``trace = d``, maximally non-minimal.

    **The measured answer runs opposite to intuition, and the algebra says why.** For a
    diagonal quadratic, ``grad = x * a`` and ``D^2 f = diag(a)`` is CONSTANT, so

        ``tr_g(D^2 f) = tr(diag(a)) - (u^T diag(a) u) / (1 + |u|^2)``,   ``u = x * a``

    -- a constant plus a bounded tilt term. Making the surface non-minimal raises ``tr(a)``,
    which adds a LARGE CONSTANT to every point and leaves the varying part untouched, so the
    relative dynamic range of ``||H||`` COLLAPSES. The isotropic bowl at ``d=20`` is the worst
    case in this family, not the best: its ``||H||`` sits near ``d`` everywhere with a spread
    of order ``1.00x``, against the minimal saddle's ``33x``. Minimality is not the defect;
    the constant ``D^2 f`` is, and no choice of ``eigenvalues`` escapes it.

    ``eigenvalues`` defaults to the saddle's own construction so this function reproduces
    ``make_saddle_control`` when called without one.
    """
    rng = np.random.default_rng(seed)
    if eigenvalues is None:
        n_pos = d // 2
        a = np.array([1.0] * n_pos + [-1.0] * (d - n_pos), dtype=np.float64)
        rng.shuffle(a)
    else:
        a = np.asarray(eigenvalues, dtype=np.float64)
        if a.shape != (d,):
            raise ValueError(f"eigenvalues must have shape ({d},); got {a.shape}.")

    x = rng.uniform(-domain_radius, domain_radius, size=(n, d))
    grad = (x * a)[:, None, :]
    hess = np.repeat(np.diag(a)[None, None, :, :], n, axis=0)
    f_x = 0.5 * np.einsum("ij,j,ij->i", x, a, x)
    return _finish(
        x, f_x, grad, hess, D, seed,
        {"eigenvalues": a, "domain_radius": domain_radius, "trace": float(a.sum()),
         "family": "quadratic"},
    )


def make_cubic_graph_control(
    n: int,
    d: int,
    D: int,
    seed: int,
    coeffs: Optional[np.ndarray] = None,
    domain_radius: float = 1.5,
) -> Dict[str, Any]:
    """Separable cubic graph ``f(x) = sum_j a_j x_j^3 / 3`` -- the simplest analytic surface in
    this family whose second fundamental form VARIES.

        ``df/dx_j    = a_j x_j^2``
        ``d2f/dx_j^2 = 2 a_j x_j``   (diagonal, and linear in ``x``)

    So ``D^2 f`` is diagonal but no longer constant: it scales with position and CHANGES SIGN
    across each coordinate hyperplane. ``tr(D^2 f) = 2 sum_j a_j x_j`` sweeps a genuine range
    over the domain rather than sitting at a fixed value, which is precisely what the saddle
    and every other quadratic cannot do.

    Non-minimal almost everywhere and minimal exactly on ``sum_j a_j x_j = 0`` -- a hyperplane
    rather than the saddle's cone, and it is a zero of the geometry rather than a cancellation
    of a constant, so points near it are genuinely flat instead of merely summing to zero.

    ``domain_radius`` defaults to ``1.5`` rather than the saddle's ``2.0``: ``|grad| ~ a x^2``
    grows quadratically here, and a wider box drives ``|grad|^2`` large enough that the metric
    tilt ``1/(1+|grad|^2)`` dominates again -- reintroducing exactly the pathology this
    fixture exists to avoid.

    ``coeffs`` defaults to all ones.
    """
    rng = np.random.default_rng(seed)
    if coeffs is None:
        a = np.ones(d, dtype=np.float64)
    else:
        a = np.asarray(coeffs, dtype=np.float64)
        if a.shape != (d,):
            raise ValueError(f"coeffs must have shape ({d},); got {a.shape}.")

    x = rng.uniform(-domain_radius, domain_radius, size=(n, d))
    grad = (a * x**2)[:, None, :]
    hess = np.zeros((n, 1, d, d), dtype=np.float64)
    diag_idx = np.arange(d)
    hess[:, 0, diag_idx, diag_idx] = 2.0 * a * x
    f_x = np.einsum("j,ij->i", a, x**3) / 3.0
    return _finish(
        x, f_x, grad, hess, D, seed,
        {"coeffs": a, "domain_radius": domain_radius, "family": "cubic"},
    )


def make_sine_graph_control(
    n: int,
    d: int,
    D: int,
    seed: int,
    amplitude: float = 0.5,
    frequency: float = 1.0,
    domain_radius: float = np.pi,
) -> Dict[str, Any]:
    """Separable sinusoidal graph ``f(x) = A sum_j sin(w x_j)``.

        ``df/dx_j    =  A w cos(w x_j)``
        ``d2f/dx_j^2 = -A w^2 sin(w x_j)``   (diagonal, bounded, sign-changing)

    A second varying-``II`` surface with a different character to the cubic: its second
    derivative is BOUNDED rather than growing with ``|x|``, so ``|grad|`` stays ``O(A w
    sqrt(d))`` no matter how wide the domain and the metric tilt cannot run away. That makes
    it the cleaner high-``d`` test of the two -- the cubic's usefulness is capped by its own
    gradient growth, this one's is not.

    Defaults keep ``A w = 0.5`` so ``|grad|^2 ~ 0.25 d``, comparable to the saddle's own
    gradient scale at matched ``d``, which keeps the metric-tilt contribution similar and
    isolates the varying-``II`` term as the difference between the fixtures.
    """
    rng = np.random.default_rng(seed)
    A, w = float(amplitude), float(frequency)
    x = rng.uniform(-domain_radius, domain_radius, size=(n, d))

    grad = (A * w * np.cos(w * x))[:, None, :]
    hess = np.zeros((n, 1, d, d), dtype=np.float64)
    diag_idx = np.arange(d)
    hess[:, 0, diag_idx, diag_idx] = -A * w**2 * np.sin(w * x)
    f_x = A * np.sin(w * x).sum(axis=1)
    return _finish(
        x, f_x, grad, hess, D, seed,
        {"amplitude": A, "frequency": w, "domain_radius": domain_radius, "family": "sine"},
    )


def make_ridge_graph_control(
    n: int,
    d: int,
    D: int,
    seed: int,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    domain_radius: float = 3.0,
    phase: float = 0.0,
) -> Dict[str, Any]:
    """Ridge (single-index) graph ``f(x) = A sin(w . x + phase)`` for one fixed unit vector ``w``.

    **This fixture exists because every SEPARABLE surface degenerates at high ``d``, and that
    is a concentration law rather than a bad choice of ``f``.** For separable
    ``f(x) = sum_j f_j(x_j)`` the Hessian is diagonal and

        ``||D^2 f||_F^2 = sum_j f_j''(x_j)^2``

    is a sum of ``d`` independent terms, so by concentration of measure its coefficient of
    variation falls like ``1 / sqrt(d)`` no matter what the ``f_j`` are. Measured on this
    module's own ``cubic`` and ``sine`` families: CV ``0.371 -> 0.243 -> 0.163 -> 0.112 ->
    0.099`` at ``d = 2, 4, 8, 16, 20``, tracking ``1/sqrt(d)`` to a constant factor of about
    ``0.44``. So at ``d = 20`` a separable surface is nearly constant-curvature however it is
    built, and asking an estimator to RANK its curvature is asking it to rank noise. The same
    argument applies to ``||H|| = |tr_g(II)|``, which is likewise a sum over ``d`` coordinates.

    A ridge function escapes the averaging entirely:

        ``grad    =  A w cos(w . x)``
        ``D^2 f   = -A w w^T sin(w . x)``      RANK ONE
        ``||D^2 f||_F = A |sin(w . x)|``       (``||w w^T||_F = 1``)

    The curvature is carried by ONE direction, so nothing is summed over ``d`` and the
    coefficient of variation of ``||D^2 f||_F`` is independent of dimension. ``|grad|^2 =
    A^2 cos^2(w . x) <= A^2`` is bounded too, so the metric tilt ``1/(1+|grad|^2)`` cannot run
    away with ``d`` the way it does for a separable surface whose gradient norm grows like
    ``sqrt(d)``. Both pathologies are removed at once, and the surface stays analytic.

    ``w`` is drawn once from ``default_rng(seed)`` and normalised, so the fixture is
    reproducible from ``(d, seed)`` alone and is returned under ``"w"`` for provenance.

    **``phase`` tunes DYNAMIC RANGE independently of everything else, and exists because
    nothing else could.** With ``phase = 0`` the curvature ``|H| ~ A f^2 |sin(s)|`` passes
    through zero, and for small ``s`` it behaves like ``|s|`` whose ``p95/p05`` is
    scale-free -- so neither ``domain_radius`` nor ``frequency`` moves the spread off ~27x
    (measured: 25.1x to 27.4x as ``L`` goes 3.0 to 0.35). Setting ``phase = pi/2`` centres
    the curvature near its maximum instead, where ``|cos| ~ 1 - s^2/2`` stays bounded away
    from zero, and then ``frequency * domain_radius`` sets the spread continuously from ~1x
    upward. That is what makes a fixture matched to a MEASURED spread possible -- needed to
    calibrate a real dataset whose own spread is neither ~1x nor ~28x.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    w = w / np.linalg.norm(w)
    A, freq = float(amplitude), float(frequency)

    x = rng.uniform(-domain_radius, domain_radius, size=(n, d))
    s = freq * (x @ w) + float(phase)
    grad = (A * freq * np.cos(s)[:, None] * w[None, :])[:, None, :]
    hess = (-A * freq**2 * np.sin(s))[:, None, None, None] * np.outer(w, w)[None, None, :, :]
    f_x = A * np.sin(s)
    return _finish(
        x, f_x, grad, hess, D, seed,
        {"w": w, "amplitude": A, "frequency": freq, "domain_radius": domain_radius,
         "phase": float(phase), "family": "ridge"},
    )


def make_multinormal_ridge_control(
    n: int,
    d: int,
    D: int,
    seed: int,
    n_normal: int = 2,
    amplitude: float = 1.0,
    frequency: float = 0.5,
    domain_radius: float = 3.0,
) -> Dict[str, Any]:
    """HIGHER-CODIMENSION ridge bundle: ``f: R^d -> R^m``, component ``j`` a ridge along its own
    direction ``w_j``. The only fixture here whose mean-curvature VECTOR genuinely rotates.

    **Why codimension 1 is not enough, and why every other fixture in this module is.** For a
    graph ``M = {(x, f(x))}`` with scalar ``f``, the normal space is ONE-dimensional, so
    ``H = H_scalar * n_hat`` and the direction of ``H`` is just the surface normal -- the Gauss
    map. Clustering those directions clusters TANGENT-PLANE ORIENTATION, not curvature
    structure. Measured on ``ridge`` at ``d=8``: the unit-``H`` covariance has rank 2 (the
    normal tilts in the plane spanned by ``w`` and the graph axis) rather than filling any
    larger space.

    That matters because the PU manifold is ``d ~ 20`` inside ``D = 768`` -- codimension ~748 --
    where ``H`` lives in a high-dimensional normal space and its direction carries real
    information about WHICH normal direction the manifold bends in. A direction result measured
    on a codimension-1 fixture does not transfer to that question.

    Here, with ``m = n_normal`` output components,

        ``f_j(x)      =  A sin(freq * w_j . x)``
        ``grad[:,j,:] =  A freq cos(s_j) w_j``
        ``hess[:,j]   = -A freq^2 sin(s_j) w_j w_j^T``     (rank one, per normal)

    the ``j``-th normal component of ``H`` scales with ``sin(s_j)``, so as ``x`` moves the
    curvature vector ROTATES within the ``m``-dimensional normal space rather than merely
    flipping sign along one axis. Which component dominates is a known, analytic region label --
    returned as ``dominant_normal`` -- which is exactly the ground truth a direction-based
    partition (D4-01) needs in order to be validated before it is trusted on PU.

    The ``w_j`` are drawn from ``default_rng(seed)`` and orthonormalised by QR, so the normals
    are genuinely distinct and the fixture is reproducible from ``(d, seed, n_normal)``.
    ``frequency`` defaults to 0.5 so ``|grad|^2 <= m A^2 freq^2`` stays modest and the metric
    tilt cannot dominate as ``d`` grows.
    """
    if n_normal < 2:
        raise ValueError(
            f"n_normal must be at least 2 -- with one normal this is a codimension-1 graph and "
            f"H's direction is just the Gauss map. Got {n_normal}."
        )
    if n_normal > d:
        raise ValueError(f"n_normal={n_normal} cannot exceed d={d}.")

    rng = np.random.default_rng(seed)
    W, _ = np.linalg.qr(rng.standard_normal((d, n_normal)))  # (d, m), orthonormal columns
    A, freq = float(amplitude), float(frequency)

    x = rng.uniform(-domain_radius, domain_radius, size=(n, d))
    S = freq * (x @ W)  # (n, m)

    grad = (A * freq * np.cos(S))[:, :, None] * W.T[None, :, :]           # (n, m, d)
    outer = np.einsum("aj,bj->jab", W, W)                                  # (m, d, d)
    hess = (-A * freq**2 * np.sin(S))[:, :, None, None] * outer[None, :, :, :]
    f_x = A * np.sin(S)                                                    # (n, m)

    H_local = curvature_probe.graph_mean_curvature(grad, hess)             # (n, d + m)
    X_local = np.concatenate([x, f_x], axis=1)
    X, H_vec, global_std = synthetic_controls.rotate_and_pad(X_local, H_local, D, seed)

    return {
        "X": X,
        "x_param": x,
        "H_vec": H_vec,
        "H_norm": np.linalg.norm(H_vec, axis=-1),
        "global_std": global_std,
        "ii_variation": second_fundamental_form_variation(hess),
        "curvature_convention": CURVATURE_CONVENTION,
        "W": W,
        "n_normal": int(n_normal),
        "dominant_normal": np.argmax(np.abs(np.sin(S)), axis=1),
        "amplitude": A,
        "frequency": freq,
        "domain_radius": domain_radius,
        "family": "multinormal_ridge",
    }


FAMILIES = {
    "quadratic_saddle": lambda n, d, D, seed: make_quadratic_graph_control(n, d, D, seed),
    "quadratic_bowl": lambda n, d, D, seed: make_quadratic_graph_control(
        n, d, D, seed, eigenvalues=np.ones(d)
    ),
    "quadratic_aniso": lambda n, d, D, seed: make_quadratic_graph_control(
        n, d, D, seed, eigenvalues=np.logspace(-1, 1, d)
    ),
    "cubic": lambda n, d, D, seed: make_cubic_graph_control(n, d, D, seed),
    "sine": lambda n, d, D, seed: make_sine_graph_control(n, d, D, seed),
    "ridge": lambda n, d, D, seed: make_ridge_graph_control(n, d, D, seed),
    "multinormal_ridge": lambda n, d, D, seed: make_multinormal_ridge_control(n, d, D, seed),
}
"""The comparison set, ordered so each entry changes exactly one thing from the last.

    quadratic_saddle  minimal,      II constant, separable  -- the sealed fixture
    quadratic_bowl    NON-minimal,  II constant, separable  -- isolates minimality
    quadratic_aniso   NON-minimal,  II constant, separable  -- minimality + anisotropy
    cubic             non-minimal,  II VARIES,   separable  -- isolates varying II
    sine              non-minimal,  II VARIES,   separable  -- same, bounded II
    ridge             non-minimal,  II VARIES,   RANK ONE   -- isolates separability

The three quadratics score ``hess_fro_cv == 0`` exactly, so they answer "does non-minimality
help?" (it does not -- the bowl is worse than the saddle). ``cubic`` and ``sine`` answer "does
varying II help?" and are themselves defeated at high ``d`` by the ``1/sqrt(d)``
concentration of any separable Hessian. ``ridge`` answers "is separability the remaining
barrier?" and is the only entry whose curvature variation does not decay with dimension."""
