"""
notebooks/pu_manifold/template_immersion.py -- D-14: canonical embedding -> random
orthogonal lift -> ratified smooth nonlinear warp, with the Jacobian rank checked at
sampled points, never assumed.

Phase 02.7 manifold-template-inference-front-end-inserted. The immersion pipeline that
turns a template name into a synthetic point cloud in `R^D`: a closed-form canonical
embedding at the template's own intrinsic dimension, a random orthogonal lift into `R^D`,
and a named, deterministic smooth nonlinear warp -- never a randomly initialised neural
net, which D-14 rules out because nothing constrains a net to be an immersion.

**The library is exactly `{S1, S2, T2, ball}`.** `S1`'s path was built by plan `02.7-01`
(the tracer); this plan (`02.7-05`) completes `S2`/`T2`/`ball` and D-14's own verification
step -- the Jacobian rank check -- for all four, including `S1`.

**Why the rank check is not optional (D-14).** A "random smooth map" that stops being an
immersion self-intersects or collapses a dimension, and the ground-truth label attached to
that cloud is then silently wrong -- the benchmark would be scoring the front end against
an answer that is itself incorrect, with no way to detect that after the fact. Every cloud
this module can check carries `canonical_points`, `tangent_basis`, `lift`, `warp_params`
and `warp_seed` -- everything :func:`jacobian_rank` needs to differentiate literally the
same map that generated the cloud, so the check is never a re-derived approximation of it.

**The `S^2` coordinate-singularity trap (Pattern 3, 02.7-RESEARCH.md).** The canonical
spherical-angle parametrization `(theta, phi) -> (sin(theta) cos(phi), ...)` has a genuine
coordinate singularity at the poles -- its own Jacobian drops rank there even though `S^2`
itself has no singularity -- which would misreport a good rank-2 immersion as rank-1 near
the poles. This module never uses that parametrization anywhere. `S^2` is instead sampled
via a normalized Gaussian (`u = g / ||g||`, `g ~ N(0, I_3)`, no chart at all to be
singular), and its tangent basis is built by Duff, Burgess, Christensen, Hery, Kensler,
Liani & Villemin's branchless orthonormal-basis-from-normal construction (JCGT 2017),
verified numerically this session to be singularity-free everywhere on `S^2` including at
the exact poles `(0,0,+-1)` -- `|b . n|` and `|1 - ||b|||` both ~1e-16 there, and at 10,000
random unit-vector draws.

**`A1` [ASSUMED -- standard differential geometry, not verified against a specific
citation this session]:** the per-template canonical-embedding recipes and the `S^2`
tangent-space-projection rank-check recipe come from `02.7-RESEARCH.md` Pattern 3, itself
flagged there as standard differential geometry rather than source-checked. Cite it as
such; do not assert it as verified against a paper.

**The named warp.** `smooth_warp` perturbs a lifted point along a fixed random orthonormal
frame `W` (drawn once from a seed) by `strength * sin(freq * <point, W>)`, mapped back to
ambient coordinates via `W`. It is explicit, deterministic and closed-form -- D-14's whole
reason for ruling out a randomly initialised net is that nothing constrains a net to be an
immersion, and this warp's Jacobian (`I + strength*freq*W diag(cos(...)) W^T`, a symmetric
perturbation of the identity) stays invertible at the small `strength` levels this phase
uses. The exact `strength`/`freq` LEVELS the benchmark grid uses are ratified constants
plan `02.7-08` fixes, not defaults set here -- `smooth_warp` and `immerse` take them as
required arguments.

**Non-uniform sampling density (D-15's density arm).** `immerse`'s `density` argument
selects `n` of an oversampled candidate pool via a weight `distance(x, anchor) **
(density - 1.0)`, `anchor` being the pool's own first-drawn point -- `density=1.0` gives
uniform weights (a plain subsample), `density != 1.0` biases toward or away from the
anchor. This is an explicit, recorded non-uniformity (a uniformly sampled fixture would
never exercise D-15's density arm); the exact density LEVELS the grid uses are, again,
ratified by plan `02.7-08`, not fixed here. See :func:`_density_select`.

Arrays in, dicts and arrays out for the numpy-only surface (`canonical_sample`,
`random_orthogonal_lift`, `smooth_warp`, `immerse`) -- no file I/O, no cache handling; the
runners under `notebooks/diagnostics/` own paths and seeds. `jacobian_rank` additionally
imports `torch` (`torch.func.jacrev`/`vmap`) for the rank check itself -- this module is
therefore, like `chart_curvature.py`/`decoder_curvature.py`, NOT re-exported from
`pu_manifold/__init__.py`'s eager imports, so Phase-1-only callers do not need torch
installed to import the package.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.func import jacrev, vmap

from .chart_curvature import VMAP_CHUNK, _pad_to_chunk

T2_MAJOR_RADIUS = 2.0
T2_MINOR_RADIUS = 1.0
"""`R > r > 0`, fixed module constants for the standard torus embedding -- matches the
`(R=2, r=1)` pair `02.7-RESEARCH.md` Pitfall 1 measures `ripser` cost against. The
Jacobian of this embedding never drops rank for any `(u, v)` under `R > r > 0` (Pattern
3), so these two constants are the only thing that needs to be fixed to keep that
guarantee -- not a ratifiable threshold, since no per-template shape choice is what plan
`02.7-08` ratifies."""

DENSITY_OVERSAMPLE_FACTOR = 4.0
"""How many extra candidate points `immerse` draws from `canonical_sample` before
`_density_select` subsamples down to the requested `n` under the requested `density` bias.
An internal implementation constant controlling the candidate pool size -- NOT a
ratifiable threshold. D-15's density LEVELS (what plan `02.7-08` ratifies) are a separate
question from how large a pool this module draws to honour whichever level it is asked
for."""

_SELF_CHECK_N_CHECK = 30
_SELF_CHECK_RANK_TOL = 1e-6
"""`immerse(..., check_immersion=True)`'s own convenience defaults, used only when the
caller passes the bare flag rather than calling :func:`jacobian_rank` directly. **NOT
ratified** -- unlike `jacobian_rank`'s own `n_check`/`rank_tol`, which carry no default
and must never acquire one (Task 2 acceptance criteria assert this by
`inspect.signature`). A caller that needs the check's own real, ratified levels should call
`jacobian_rank` directly, exactly as this plan's own acceptance criteria and
`tests/test_template_immersion.py` do."""


def _onb_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Duff et al. (JCGT 2017) branchless orthonormal basis from a batch of unit normals
    `n`, `(rows, 3)` -> two `(rows, 3)` arrays `(b1, b2)` spanning each row's tangent
    plane. Singularity-free everywhere on `S^2`, including at the exact poles a
    spherical-angle chart would call singular -- verified numerically this session:
    `|b1|`/`|b2|` within `2e-16` of `1.0`, `|b1.n|`/`|b2.n|`/`|b1.b2|` within `5e-16` of
    `0.0`, at 10,000 random draws and at `n = (0,0,+-1)` explicitly.
    """
    sign = np.copysign(1.0, n[:, 2])
    a = -1.0 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    b1 = np.stack([1.0 + sign * n[:, 0] * n[:, 0] * a, sign * b, -sign * n[:, 0]], axis=1)
    b2 = np.stack([b, sign + n[:, 1] * n[:, 1] * a, -n[:, 1]], axis=1)
    return b1.astype(np.float64), b2.astype(np.float64)


def canonical_sample(
    template: str, n: int, rng: np.random.Generator, d: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """`n` points sampled from `template`'s canonical embedding, plus each point's tangent
    basis, `(points, tangent)`:

      - `points`: `(n, d_from)` float64, the canonical embedding.
      - `tangent`: `(n, d_from, d_true)` float64, a basis of the tangent space at each
        point, expressed in the SAME `d_from`-dimensional ambient coordinates as `points`
        -- so :func:`jacobian_rank` never has to re-derive it.

    - `"S1"`: `p(t) = (cos t, sin t)`, analytic tangent `v(t) = (-sin t, cos t)` -- no
      coordinate singularity anywhere.
    - `"S2"`: a normalized Gaussian sample, `u = g / ||g||`, `g ~ N(0, I_3)` -- deliberately
      NOT the spherical-angle parametrization (see module docstring). Tangent basis via
      :func:`_onb_from_normal`.
    - `"T2"`: the standard torus embedding `((R + r cos v) cos u, (R + r cos v) sin u,
      r sin v)`, `R > r > 0` (`T2_MAJOR_RADIUS`, `T2_MINOR_RADIUS`). Analytic tangent
      `(d/du, d/dv)`, each independently unit-normalized -- they are already orthogonal
      for this parametrization (a standard fact: the cross term `d/du . d/dv` vanishes
      identically), so no Gram-Schmidt step is needed.
    - `"ball"`: `B^d` sits in `R^d` already, no embedding step beyond sampling: a
      normalized-Gaussian direction times a `U^(1/d)`-distributed radius (the standard
      uniform-in-volume construction). Tangent basis is `R^d`'s own coordinate directions
      (the identity matrix) -- there is no embedding step to differentiate, so it can
      never drop rank. `d` is REQUIRED for `"ball"` and ignored otherwise.
    """
    if template == "S1":
        t = rng.uniform(0.0, 2.0 * np.pi, size=n)
        points = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float64)
        tangent = np.stack([-np.sin(t), np.cos(t)], axis=1).astype(np.float64)[:, :, None]
        return points, tangent

    if template == "S2":
        g = rng.standard_normal(size=(n, 3))
        points = (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float64)
        b1, b2 = _onb_from_normal(points)
        tangent = np.stack([b1, b2], axis=2).astype(np.float64)
        return points, tangent

    if template == "T2":
        R, r = T2_MAJOR_RADIUS, T2_MINOR_RADIUS
        u = rng.uniform(0.0, 2.0 * np.pi, size=n)
        v = rng.uniform(0.0, 2.0 * np.pi, size=n)
        tube = R + r * np.cos(v)
        points = np.stack(
            [tube * np.cos(u), tube * np.sin(u), r * np.sin(v)], axis=1
        ).astype(np.float64)
        du = np.stack([-tube * np.sin(u), tube * np.cos(u), np.zeros_like(u)], axis=1)
        du = du / np.linalg.norm(du, axis=1, keepdims=True)
        dv = np.stack(
            [-r * np.sin(v) * np.cos(u), -r * np.sin(v) * np.sin(u), r * np.cos(v)], axis=1
        )
        dv = dv / np.linalg.norm(dv, axis=1, keepdims=True)
        tangent = np.stack([du, dv], axis=2).astype(np.float64)
        return points, tangent

    if template == "ball":
        if d is None or d < 1:
            raise ValueError(
                f"canonical_sample: template 'ball' requires an integer d >= 1, got {d!r}"
            )
        g = rng.standard_normal(size=(n, d))
        directions = g / np.linalg.norm(g, axis=1, keepdims=True)
        radii = rng.uniform(0.0, 1.0, size=(n, 1)) ** (1.0 / d)
        points = (directions * radii).astype(np.float64)
        tangent = np.broadcast_to(np.eye(d, dtype=np.float64), (n, d, d)).copy()
        return points, tangent

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


def _warp_frame(D: int, seed: int) -> np.ndarray:
    """Regenerate the exact `D x D` orthonormal warp frame `smooth_warp` uses, given
    `seed` -- bit-identical, so :func:`jacobian_rank` differentiates literally the same
    map that generated the cloud, not a re-derived approximation of it."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(size=(D, D))
    Q, _ = np.linalg.qr(W)
    return Q.astype(np.float64)


def smooth_warp(points: np.ndarray, strength: float, freq: float, seed: int) -> np.ndarray:
    """A named, deterministic smooth nonlinear warp -- NOT a randomly initialised net
    (D-14 rules that out: nothing constrains a net to be an immersion).

    For each point, perturbs it along a fixed random orthonormal frame `W` (drawn once
    from `seed`, via :func:`_warp_frame`) by `strength * sin(freq * <point, W>)`, then maps
    the perturbation back into ambient coordinates via `W`. Deterministic and reproducible
    given `seed`; `strength=0.0` returns `points` unperturbed to float64 precision.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"smooth_warp: points must be a 2-d (n, D) array, got shape {points.shape!r}")
    n, D = points.shape

    Q = _warp_frame(D, seed)
    projections = points @ Q
    warp = strength * np.sin(freq * projections)
    return (points + warp @ Q.T).astype(np.float64)


def _density_select(
    canonical_points: np.ndarray, density: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Choose `n` of `canonical_points`'s rows without replacement, weighted by
    `distance(x, anchor) ** (density - 1.0)`, `anchor` being the pool's own first-drawn
    point (a fixed, deterministic reference given `rng`'s seed). `density=1.0` gives
    uniform weights (a plain uniform subsample); `density > 1.0` favours points far from
    `anchor`; `density < 1.0` favours points near it -- the explicit, recorded
    non-uniformity D-15's density arm needs. See module docstring for why `density=1.0` is
    not itself ratified here."""
    n_candidates = canonical_points.shape[0]
    if n_candidates < n:
        raise ValueError(
            f"_density_select: only {n_candidates} candidates available, need >= {n}"
        )
    if n_candidates == n:
        return np.arange(n)
    anchor = canonical_points[0]
    distances = np.linalg.norm(canonical_points - anchor, axis=1)
    distances = np.maximum(distances, 1e-12)
    weights = distances ** (density - 1.0)
    probabilities = weights / weights.sum()
    return rng.choice(n_candidates, size=n, replace=False, p=probabilities)


def immerse(
    template: str,
    n: int,
    D: int,
    noise: float,
    density: float,
    seed: int,
    warp_params: Dict[str, float],
    d: Optional[int] = None,
    check_immersion: bool = False,
) -> Dict[str, Any]:
    """The full D-14 immersion pipeline for `template`: canonical embedding -> density
    subsample -> random orthogonal lift into `R^D` -> the named smooth nonlinear warp ->
    additive noise.

    `warp_params` carries `{"strength", "freq"}` (required keys) and an optional `"seed"`
    (defaults to `seed` when absent). `d` is required for `"ball"` and ignored otherwise.
    `noise`/`density` carry no default (D-14/D-15 ratifiable thresholds); `check_immersion`
    is a plain feature flag, not a threshold, and defaults to `False`.

    Returns a dict carrying `points` (float64, shape `(n, D)`), `template`, `d_true`, `D`,
    `n`, `noise`, `density`, `seed`, `warp_params`, `warp_seed`, `lift`,
    `canonical_points`, `tangent_basis` and `ground_truth_valid` -- everything
    :func:`jacobian_rank` needs to check the SAME pipeline that produced this cloud, and
    everything a benchmark cell needs to reconstruct ground truth from the cloud alone.

    When `check_immersion=True`, runs :func:`jacobian_rank` at this module's own
    (unratified) self-check defaults and attaches the result at `"immersion_check"`; when
    the check fails, `"ground_truth_valid"` is set `False` rather than silently returning a
    labelled cloud that is not what its label says (D-14's whole point).
    """
    if not isinstance(warp_params, dict):
        raise ValueError(f"immerse: warp_params must be a dict, got {type(warp_params)!r}")
    if "strength" not in warp_params or "freq" not in warp_params:
        raise ValueError(
            f"immerse: warp_params must carry 'strength' and 'freq', got keys "
            f"{sorted(warp_params.keys())!r}"
        )
    if noise < 0.0:
        raise ValueError(f"immerse: noise must be >= 0.0, got {noise}")

    canonical_seed, lift_seed, density_seed, noise_seed = np.random.SeedSequence(seed).spawn(4)
    rng_canonical = np.random.default_rng(canonical_seed)
    rng_lift = np.random.default_rng(lift_seed)
    rng_density = np.random.default_rng(density_seed)
    rng_noise = np.random.default_rng(noise_seed)

    n_candidates = max(n, int(np.ceil(n * DENSITY_OVERSAMPLE_FACTOR)))
    canonical_all, tangent_all = canonical_sample(template, n_candidates, rng_canonical, d=d)

    idx = _density_select(canonical_all, density, n, rng_density)
    canonical_points = canonical_all[idx]
    tangent_basis = tangent_all[idx]

    d_from = canonical_points.shape[1]
    lift = random_orthogonal_lift(d_from, D, rng_lift)
    lifted = canonical_points @ lift.T

    warp_seed = int(warp_params.get("seed", seed))
    warped = smooth_warp(lifted, float(warp_params["strength"]), float(warp_params["freq"]), warp_seed)

    if noise > 0.0:
        warped = warped + rng_noise.normal(scale=noise, size=warped.shape)

    d_true_by_template = {"S1": 1, "S2": 2, "T2": 2, "ball": d}
    d_true = d_true_by_template.get(template)

    cloud: Dict[str, Any] = {
        "points": np.asarray(warped, dtype=np.float64),
        "template": template,
        "d_true": d_true,
        "D": D,
        "n": n,
        "noise": noise,
        "density": density,
        "seed": seed,
        "warp_params": dict(warp_params),
        "warp_seed": warp_seed,
        "lift": lift,
        "canonical_points": canonical_points,
        "tangent_basis": tangent_basis,
        "ground_truth_valid": True,
    }

    if check_immersion:
        n_check = min(_SELF_CHECK_N_CHECK, n)
        result = jacobian_rank(cloud, n_check=n_check, rank_tol=_SELF_CHECK_RANK_TOL)
        cloud["immersion_check"] = result
        cloud["ground_truth_valid"] = bool(result["is_immersion"])

    return cloud


def jacobian_rank(cloud: Dict[str, Any], n_check: int, rank_tol: float) -> Dict[str, Any]:
    """Verify that `lift . warp` (as recorded on `cloud`) is a genuine immersion at
    `n_check` of `cloud`'s own sampled points -- D-14's non-negotiable check, never
    assumed. `n_check` and `rank_tol` are REQUIRED, no defaults (verified by
    `inspect.signature` in this plan's own acceptance criteria) -- unlike
    `immerse(..., check_immersion=True)`'s convenience self-check, which picks its own
    (unratified) values, this is the primary API and must never silently degrade to a
    default rank tolerance.

    Computes the Jacobian of `lift . warp` (NOT of any parametrization -- see `A1` and the
    module docstring's `S^2` note) with `torch.func.jacrev` at float64, at the ambient
    canonical-embedding coordinates (`R^{d_from}`) of the first `n_check` rows of
    `cloud["canonical_points"]`, then right-multiplies each point's `D x d_from` Jacobian
    by that point's `d_from x d_true` tangent basis (`cloud["tangent_basis"]`) -- by the
    chain rule this equals the derivative of the WHOLE composite (embed -> lift -> warp)
    with respect to the template's own intrinsic coordinates, without ever differentiating
    through an angle chart. Rank is `torch.linalg.matrix_rank` (SVD-based) at the supplied
    `rank_tol`, applied as an absolute tolerance (`atol=rank_tol, rtol=0.0`).

    Returns `{"ranks", "min_rank", "expected_rank", "is_immersion", "n_check", "rank_tol",
    "worst_point_index"}` -- the full per-point rank array, not only the minimum, so a
    marginal case stays visible.
    """
    if n_check < 1:
        raise ValueError(f"jacobian_rank: n_check must be >= 1, got {n_check}")

    canonical_points = cloud["canonical_points"]
    tangent_basis = cloud["tangent_basis"]
    lift = cloud["lift"]
    warp_params = cloud["warp_params"]
    warp_seed = cloud["warp_seed"]
    d_true = cloud["d_true"]
    D = cloud["D"]

    n_total = canonical_points.shape[0]
    if n_check > n_total:
        raise ValueError(
            f"jacobian_rank: n_check={n_check} exceeds the cloud's {n_total} available points"
        )

    check_points = np.asarray(canonical_points[:n_check], dtype=np.float64)
    check_tangent = np.asarray(tangent_basis[:n_check], dtype=np.float64)
    d_from = check_points.shape[1]

    W_frame = _warp_frame(D, warp_seed)
    Q_t = torch.from_numpy(np.ascontiguousarray(lift))
    W_t = torch.from_numpy(np.ascontiguousarray(W_frame))
    strength = float(warp_params["strength"])
    freq = float(warp_params["freq"])

    def composed_map(x: torch.Tensor) -> torch.Tensor:
        lifted = x @ Q_t.T
        projections = lifted @ W_t
        warp = strength * torch.sin(freq * projections)
        return lifted + warp @ W_t.T

    x_batch = torch.from_numpy(np.ascontiguousarray(check_points))
    tb_batch = torch.from_numpy(np.ascontiguousarray(check_tangent))

    parts = []
    for start in range(0, n_check, VMAP_CHUNK):
        real = x_batch[start : start + VMAP_CHUNK]
        chunk = _pad_to_chunk(real)
        J_chunk = vmap(jacrev(composed_map))(chunk)[: real.shape[0]]
        parts.append(J_chunk.detach())
    J = torch.cat(parts, dim=0)

    if tuple(J.shape) != (n_check, D, d_from):
        raise ValueError(
            f"jacobian_rank: expected Jacobian shape {(n_check, D, d_from)}, got "
            f"{tuple(J.shape)}."
        )

    proj = torch.einsum("bDf,bft->bDt", J, tb_batch)
    ranks_t = torch.linalg.matrix_rank(proj, atol=rank_tol, rtol=0.0)
    ranks = ranks_t.numpy().astype(np.int64)

    min_rank = int(ranks.min())
    worst_point_index = int(ranks.argmin())

    return {
        "ranks": ranks,
        "min_rank": min_rank,
        "expected_rank": int(d_true),
        "is_immersion": bool(min_rank == d_true),
        "n_check": n_check,
        "rank_tol": rank_tol,
        "worst_point_index": worst_point_index,
    }
