"""
Phase 3 Wave 0 ground truth: three known-geometry synthetic control fixtures -- flat plane,
sphere, saddle -- constructed at PU's actual working scale (``d = 20``, ``D = 768``), used by
step 4 as manifolds whose analytic mean curvature is known in advance rather than assumed.

**What these controls can establish, and what they cannot -- read this before quoting a
result from this module anywhere else.** A synthetic manifold sampled cleanly and fitted
under this protocol trains to a clean, unfragmented atlas. It has none of the pathology the
phase's gate override is worried about -- ``02.5-09``'s chart-count-driven fragmentation,
``max cond(g)`` climbing from ``3.26`` to ``122.22``, second derivatives banding along
artificial chart seams. A control that passes therefore establishes only that the
decoder-curvature *pipeline* is correct on a manifold that is easy to fit. It is necessary,
not sufficient, and it MUST NOT be presented as evidence that the PU curvature field is free
of parameterization damage.

This module writes NO new curvature mathematics. ``curvature_probe.graph_mean_curvature`` is
the only curvature computation used anywhere below, imported unmodified; the flat control
delegates to ``curvature_probe.make_flat_fixture`` outright; the saddle's only new arithmetic
is the pair of hand-computed ``grad``/``hess`` arrays for its own quadratic form, and those
are cross-checked against an independent finite-difference computation before being trusted
(``test_synthetic_saddle_control_matches_finite_difference``) rather than trusted by
inspection. ``curvature_probe.py`` is imported and never edited.

Curvature convention: every quantity here is the UNNORMALIZED trace ``H = tr_g(II)``,
matching ``chart_curvature.CURVATURE_CONVENTION`` and ``curvature_probe.CURVATURE_CONVENTION``
exactly. A unit ``d``-sphere gives ``||H|| = d`` under this convention -- never ``1`` and
never the averaged ``(d + 2) / d``. This module declares its own ``CURVATURE_CONVENTION``
constant (rather than merely importing a sealed one) and asserts at import time that all
three agree, so a future drift in either sealed module would break this module's import
instead of silently propagating a factor-of-``d`` error into step 4.
"""

from typing import Dict

import numpy as np

from . import chart_curvature, curvature_probe

CURVATURE_CONVENTION = "trace"
"""``H = tr_g(II)``, the unnormalized ``g``-trace of the second fundamental form -- never
the averaged ``(1/d) tr_g(II)``. Declared here (rather than merely imported) so that a drift
in either sealed module's own convention constant breaks this module's import instead of
silently propagating a factor-of-``d`` error into step 4's ground truth. A unit ``d``-sphere
gives ``||H|| = d`` under this convention, never ``1``."""

if CURVATURE_CONVENTION != chart_curvature.CURVATURE_CONVENTION:
    raise ValueError(
        f"synthetic_controls.CURVATURE_CONVENTION={CURVATURE_CONVENTION!r} disagrees with "
        f"chart_curvature.CURVATURE_CONVENTION={chart_curvature.CURVATURE_CONVENTION!r}. Two "
        f"modules computing the same mathematics must never silently diverge on which "
        f"convention they report under."
    )
if CURVATURE_CONVENTION != curvature_probe.CURVATURE_CONVENTION:
    raise ValueError(
        f"synthetic_controls.CURVATURE_CONVENTION={CURVATURE_CONVENTION!r} disagrees with "
        f"curvature_probe.CURVATURE_CONVENTION={curvature_probe.CURVATURE_CONVENTION!r}. Two "
        f"modules computing the same mathematics must never silently diverge on which "
        f"convention they report under."
    )


# --- Shared embedding step -----------------------------------------------------------


def rotate_and_pad(X_local: np.ndarray, H_local: np.ndarray, D: int, seed: int):
    """Embed a local ``(n, m)`` manifold-plus-curvature pair into ambient dimension ``D``,
    the shared final step of every control below and of
    ``curvature_probe.make_graph_of_function_fixture``'s own pipeline: zero-pad both arrays
    from ``m`` to ``D``, apply one fixed random orthogonal rotation ``Q`` drawn from
    ``np.random.default_rng(seed)`` via a QR factorization of a Gaussian ``(D, D)`` matrix
    (the only ``(D, D)`` array this module or any of its controls ever materializes), centre
    ``X`` and divide it by one global scalar standard deviation, and multiply ``H_local`` by
    that same scalar -- curvature scales inversely with length, so rescaling ``X`` down by
    ``global_std`` scales ``H`` up by ``global_std``, exactly
    ``make_graph_of_function_fixture``'s own convention, not re-derived here.

    Padding with exact zeros keeps the padded directions totally geodesic, so ambient
    dimension ``D`` stays independent of the manifold's own codimension of curvature.

    Returns ``(X, H_vec, global_std)``: ``X`` and ``H_vec`` both ``(n, D)`` float64,
    ``global_std`` a Python float.
    """
    X_local = np.asarray(X_local, dtype=np.float64)
    H_local = np.asarray(H_local, dtype=np.float64)
    if X_local.shape != H_local.shape:
        raise ValueError(
            f"rotate_and_pad: X_local.shape={X_local.shape} != H_local.shape={H_local.shape}"
        )
    n, m = X_local.shape
    if D < m:
        raise ValueError(f"rotate_and_pad: D={D} must be >= local width m={m}")

    global_std = float(X_local.std())

    X_padded = np.zeros((n, D), dtype=np.float64)
    X_padded[:, :m] = X_local
    H_padded = np.zeros((n, D), dtype=np.float64)
    H_padded[:, :m] = H_local

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    X_rot = X_padded @ Q.T
    H_rot = H_padded @ Q.T

    X = (X_rot - X_rot.mean(axis=0)) / global_std
    H_vec = H_rot * global_std
    return X, H_vec, global_std


# --- Flat control ----------------------------------------------------------------------


def make_flat_control(n: int, d: int, D: int, seed: int) -> Dict[str, np.ndarray]:
    """Flat-plane control: a thin wrapper over ``curvature_probe.make_flat_fixture``, which
    already produces an exactly linear embedding whose analytic ``H_norm`` is exactly zero
    at every point -- not merely small. Delegates outright; no new geometry is written here.

    Any apparent ``||H||`` a decoder reports on this fixture is entirely artifact, since the
    analytic answer is zero exactly.

    Returns the same dict ``curvature_probe.make_graph_of_function_fixture`` returns:
    ``X``, ``x_param``, ``H_vec``, ``H_norm``, ``global_std``, plus that function's other
    keys (``realized_skew``, ``amplitudes``, ``centres``, ``sigma``).
    """
    return curvature_probe.make_flat_fixture(n=n, d=d, D=D, seed=seed)


# --- Sphere control --------------------------------------------------------------------


def make_sphere_control(
    n: int, d: int, D: int, seed: int, R: float = 1.0
) -> Dict[str, np.ndarray]:
    """Radius-``R`` ``d``-sphere control, sampled in ``R^{d+1}`` then rotated and padded to
    ``D``. The analytic mean curvature vector under this module's trace convention points
    inward with magnitude ``d / R`` at every point: the unit outward normal is
    ``n_hat = x / R``, every principal curvature of a radius-``R`` sphere equals ``1 / R``,
    and the trace convention sums ``d`` copies of that value and points along ``-n_hat``,
    giving ``H_local = -(d / R**2) * X_local`` and ``||H_local|| = d / R`` exactly -- ``d``
    at ``R = 1``, never ``1`` and never the averaged ``(d + 2) / d``. This is derived from
    the ``d``-sphere's own second fundamental form, not transcribed from an external
    reference; the regression test pinning it against ``d`` (and explicitly rejecting ``1``
    and ``(d + 2) / d``) is the check that it is right.

    Returns the standard dict (``X``, ``x_param``, ``H_vec``, ``H_norm``, ``global_std``)
    plus ``R``.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, d + 1))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    X_local = R * raw / norms

    H_local = -(d / R**2) * X_local

    X, H_vec, global_std = rotate_and_pad(X_local, H_local, D, seed)
    H_norm = np.linalg.norm(H_vec, axis=-1)

    return {
        "X": X,
        "x_param": X_local,
        "H_vec": H_vec,
        "H_norm": H_norm,
        "global_std": global_std,
        "R": R,
    }
