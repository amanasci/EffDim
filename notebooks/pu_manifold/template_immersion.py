"""
notebooks/pu_manifold/template_immersion.py -- D-14: canonical embedding -> random
orthogonal lift -> ratified smooth nonlinear warp.

Phase 02.7 manifold-template-inference-front-end-inserted. The immersion pipeline that
turns a template name into a synthetic point cloud in `R^D`: a closed-form canonical
embedding at the template's own intrinsic dimension, a random orthogonal lift into `R^D`,
and a named, deterministic smooth nonlinear warp -- never a randomly initialised neural
net, which D-14 rules out because nothing constrains a net to be an immersion.

**This plan implements the `S^1` path only.** `S^1` uses `p(t) = (cos t, sin t)` with the
analytic tangent `v(t) = (-sin t, cos t)` -- no coordinate singularity anywhere, safe to
autodiff directly. The other three templates (`S^2`, `T^2`, `ball`) AND the Jacobian rank
check (D-14's own verification that `lift ∘ warp` restricted to the canonical manifold
stays an immersion, for every template including `S^1`) land in plan `02.7-05`; both raise
`NotImplementedError` naming that plan here rather than a silent partial implementation.

Arrays in, dicts and arrays out -- no file I/O, no cache handling.
"""

from typing import Any, Dict, Optional

import numpy as np

_UNIMPLEMENTED_TEMPLATES = ("S2", "T2", "ball")


def canonical_sample(template: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` points sampled from `template`'s canonical embedding, `(n, d_from)` float64.

    Only `"S1"` is implemented this plan: `p(t) = (cos t, sin t)`, `t` drawn uniformly on
    `[0, 2*pi)`. `"S2"`, `"T2"`, `"ball"` raise `NotImplementedError` naming plan
    `02.7-05`, which lands them alongside D-14's Jacobian rank check -- not built for any
    template in this plan, `S1` included.
    """
    if template == "S1":
        t = rng.uniform(0.0, 2.0 * np.pi, size=n)
        return np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float64)
    if template in _UNIMPLEMENTED_TEMPLATES:
        raise NotImplementedError(
            f"canonical_sample: template {template!r} lands in plan 02.7-05, alongside "
            "D-14's Jacobian rank check -- not built in this plan."
        )
    raise ValueError(f"canonical_sample: unknown template {template!r}")


def random_orthogonal_lift(d_from: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """A random `D x d_from` frame with orthonormal columns -- the `Q` factor of a QR
    decomposition of a `(D, d_from)` standard Gaussian matrix. Applying it to a
    `d_from`-dimensional canonical embedding is an isometry into `R^D`.
    """
    if D < d_from:
        raise ValueError(f"random_orthogonal_lift: D={D} must be >= d_from={d_from}")
    G = rng.standard_normal(size=(D, d_from))
    Q, _ = np.linalg.qr(G)
    return Q.astype(np.float64)


def smooth_warp(points: np.ndarray, strength: float, freq: float, seed: int) -> np.ndarray:
    """A named, deterministic smooth nonlinear warp -- NOT a randomly initialised net
    (D-14 rules that out: nothing constrains a net to be an immersion).

    For each point, perturbs it along a fixed random orthonormal frame `Q` (drawn once
    from `seed`) by `strength * sin(freq * <point, Q>)`, then maps the perturbation back
    into ambient coordinates via `Q`. Deterministic and reproducible given `seed`;
    `strength=0.0` returns `points` unperturbed to float64 precision.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"smooth_warp: points must be a 2-d (n, D) array, got shape {points.shape!r}")
    n, D = points.shape

    rng = np.random.default_rng(seed)
    W = rng.standard_normal(size=(D, D))
    Q, _ = np.linalg.qr(W)

    projections = points @ Q
    warp = strength * np.sin(freq * projections)
    return (points + warp @ Q.T).astype(np.float64)


def immerse(
    template: str,
    n: int,
    D: int,
    noise: float,
    seed: int,
    warp_params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """The full D-14 immersion pipeline for `template`: canonical embedding -> random
    orthogonal lift into `R^D` -> the ratified smooth nonlinear warp -> additive noise.

    `warp_params` carries `{"strength", "freq"}` (required keys) and an optional `"seed"`
    (defaults to `seed` when absent). This is a data-shape argument, not a ratifiable
    threshold -- the caller supplies its own warp configuration explicitly either way.

    Returns a dict carrying `points` (float64, shape `(n, D)`), `template`, `d_true`,
    `seed`, and the warp parameters, so provenance travels with the cloud.

    Only `"S1"` (`d_true=1`) is implemented this plan; the other three templates raise
    `NotImplementedError` via `canonical_sample`, naming plan `02.7-05`.
    """
    if warp_params is None:
        raise ValueError("immerse: warp_params is required (no default warp configuration)")
    if "strength" not in warp_params or "freq" not in warp_params:
        raise ValueError(
            f"immerse: warp_params must carry 'strength' and 'freq', got keys "
            f"{sorted(warp_params.keys())!r}"
        )

    rng = np.random.default_rng(seed)
    canonical = canonical_sample(template, n, rng)
    d_from = canonical.shape[1]

    lift = random_orthogonal_lift(d_from, D, rng)
    lifted = canonical @ lift.T

    warp_seed = warp_params.get("seed", seed)
    warped = smooth_warp(lifted, warp_params["strength"], warp_params["freq"], warp_seed)

    if noise > 0.0:
        warped = warped + rng.normal(scale=noise, size=warped.shape)

    d_true_by_template = {"S1": 1}
    d_true = d_true_by_template.get(template)

    return {
        "points": np.asarray(warped, dtype=np.float64),
        "template": template,
        "d_true": d_true,
        "seed": seed,
        "warp_params": dict(warp_params),
    }
