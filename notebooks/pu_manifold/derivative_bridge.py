"""
Phase 02.6 decoder-substrate screening: the derivative-usability bridge (SC-5; D-16, D-17,
D-18). This module answers a question the halted run never tested -- are a trained decoder's
own second derivatives trustworthy -- by comparing the autodiff Hessian
``decoder_curvature.py``'s own tracer already computes against an independently-implemented
finite-difference Hessian of the SAME map. Because the comparison needs no analytic ground
truth (both sides differentiate the trained model itself, not a known surface), it transfers
unchanged from the Swiss roll to the PU regime (D-17) -- the whole reason this instrument was
chosen over pullback-metric conditioning or post-hoc curvature read-outs.

The autodiff half is reused-unchanged-from-sealed-code, not re-derived: every ``torch.func``
call here mirrors ``decoder_curvature.plain_decoder_curvature``'s own chunking discipline
exactly, importing ``chart_curvature.VMAP_CHUNK``/``_pad_to_chunk`` rather than re-deriving
them, and :func:`reduce_to_H_vec` below mirrors ``chart_curvature.chart_mean_curvature``'s
trace-first-then-project ``d x d`` solve verbatim rather than re-deriving the algebra. The
finite-difference half is genuinely new -- no finite-difference code exists anywhere else in
this project (``02.6-PATTERNS.md`` "No Analog Found") -- and its step size is calibrated
against this project's own known-answer fixtures (:func:`calibrate_fd_step`) rather than
assumed, per assumption A-06.

Tensors in, dicts and tensors out -- no file I/O, no cache handling; the runner that reports
this module's numbers (``notebooks/diagnostics/derivative_bridge_run.py``, plan 02.6-14) owns
paths and cache stems.

Like ``cae.py``, ``chart_curvature.py``, ``curvature_probe.py`` and ``decoder_curvature.py``,
this module imports ``torch`` at module level. For the same reason those modules are excluded
from ``pu_manifold/__init__.py``'s eager imports (so Phase-1-only callers do not need torch
installed to import the package), this module is deliberately NOT re-exported there either.

Curvature convention: every reduced quantity here is the UNNORMALIZED trace ``H = tr_g(II)``,
identical to ``chart_curvature.CURVATURE_CONVENTION`` and
``decoder_curvature.CURVATURE_CONVENTION``. This module declares its own
``CURVATURE_CONVENTION`` constant (rather than merely importing one of the sealed ones) and
asserts at import time that all agree, so a future edit to either sealed module that silently
drifted the convention would break this module's import rather than propagate a
factor-of-``d`` error downstream.

**D-18, restated as code discipline.** This module reports numbers and nothing more: no
acceptance rule, no boolean verdict, and no key anywhere that folds the full-tensor and
reduced levels into one score (S1's never-collapse rule, ``02.6-SCREENING-RULE-02.md``). The
plan that runs it (02.6-14) prints exactly what it returns, and ``02.5-10`` is the only place
a bound is ever ratified against these numbers.
"""

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from torch.func import hessian, jacrev, vmap

from .chart_curvature import (
    CURVATURE_CONVENTION as _CHART_CURVATURE_CONVENTION,
    VMAP_CHUNK,
    _assert_float64,
    _pad_to_chunk,
)
from .decoder_curvature import (
    CURVATURE_CONVENTION as _DECODER_CURVATURE_CONVENTION,
    assert_c2_decoder,
    plain_decoder_map,
)

CURVATURE_CONVENTION = "trace"
"""``H = tr_g(II)``, the unnormalized ``g``-trace of the second fundamental form -- never the
averaged ``(1/d) tr_g(II)``. Declared here (rather than merely imported) so that a drift in
either sealed module's own convention constant breaks this module's import instead of
silently propagating a factor-of-``d`` error into a screening run."""

if CURVATURE_CONVENTION != _CHART_CURVATURE_CONVENTION:
    raise ValueError(
        f"derivative_bridge.CURVATURE_CONVENTION={CURVATURE_CONVENTION!r} disagrees with "
        f"chart_curvature.CURVATURE_CONVENTION={_CHART_CURVATURE_CONVENTION!r}. Two modules "
        f"computing the same mathematics must never silently diverge on which convention "
        f"they report under."
    )
if CURVATURE_CONVENTION != _DECODER_CURVATURE_CONVENTION:
    raise ValueError(
        f"derivative_bridge.CURVATURE_CONVENTION={CURVATURE_CONVENTION!r} disagrees with "
        f"decoder_curvature.CURVATURE_CONVENTION={_DECODER_CURVATURE_CONVENTION!r}. Two "
        f"modules computing the same mathematics must never silently diverge on which "
        f"convention they report under."
    )


# --- stencil and calibration constants ----------------------------------------------------

FD_STENCIL_CONVENTION = (
    "diagonal: H_ii = (f(z + h*e_i) - 2*f(z) + f(z - h*e_i)) / h**2 -- the standard second "
    "central difference of the map along one latent direction. off-diagonal (i != j): "
    "H_ij = (f(z + h*e_i + h*e_j) - f(z + h*e_i - h*e_j) - f(z - h*e_i + h*e_j) "
    "+ f(z - h*e_i - h*e_j)) / (4*h**2) -- the standard four-point mixed central difference "
    "along two latent directions, symmetric by construction (H_ji = H_ij, computed once and "
    "mirrored)."
)
"""Pins the exact stencil this module uses, so it cannot silently drift to a different
finite-difference convention (a forward difference, a five-point stencil, ...) without the
change being visible as a diff to this string."""

DEFAULT_FD_STEP = 1e-4
"""Measured minimum of the float64 central-difference error curve against this project's own
``_SphereDecoder`` fixture (``02.6-RESEARCH.md`` Pitfall 5: max abs error ``4.5e-8`` at this
step, versus ``4.5e-4`` one decade up and ``3.9e-2`` one decade down -- roughly four orders of
magnitude of degradation either side of the minimum). This is a measured STARTING POINT for
this project's own decoder shape and scale, not a universal constant: the optimum depends on
the local curvature scale of whatever decoder is actually being differentiated, so a caller
working on a differently-scaled decoder (assumption A-06 names the PU-regime ``d=40`` fits
specifically) is expected to call :func:`calibrate_fd_step` rather than trust this value
blind."""

MAX_FD_ROWS = 8192
"""Row cap for one batched call to ``decode_batch``, so the per-forward point chunk is
DERIVED from a documented budget rather than guessed. The stencil above needs
``1 + 2*d + 4*d*(d-1)/2`` decoder evaluations per input point (the ``1`` is the center point
``f(z)``, ``2*d`` is the diagonal's forward/backward pair per axis, ``4*d*(d-1)/2`` is the
four-point mixed difference over every unordered axis pair) -- ``9`` at the Swiss roll's
``d = 2``, ``3201`` at the PU control's ``d = 40``. At ``d = 40`` with a 768-dimensional
decoder output, evaluating 64 points in one shot would need roughly
``64 * 3201 ~= 205,000`` rows and, at float64, over a gigabyte of output held at once -- which
is why chunking the per-forward row count is not optional at that scale. ``8192`` keeps one
chunk's decoder output at ``8192 * 768 * 8 bytes ~= 50 MB``, comfortably bounded regardless of
how large a batch or latent dimension a caller supplies."""


def _chunked_eval(
    decode_batch: Callable[[torch.Tensor], torch.Tensor], points: torch.Tensor
) -> torch.Tensor:
    """Invoke ``decode_batch`` on chunks of at most :data:`MAX_FD_ROWS` rows, concatenating
    the results in the original row order. Bounds peak memory per call; does not change the
    result. The number of ``decode_batch`` calls this makes is
    ``ceil(points.shape[0] / MAX_FD_ROWS)``."""
    n = points.shape[0]
    parts = [
        decode_batch(points[start : start + MAX_FD_ROWS])
        for start in range(0, n, MAX_FD_ROWS)
    ]
    return torch.cat(parts, dim=0)


# --- finite-difference derivatives, the genuinely new half ---------------------------------


def finite_difference_jacobian(
    decode_batch: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor, h: float
) -> torch.Tensor:
    """Central-difference Jacobian of ``decode_batch`` at every row of ``z``, shape
    ``(batch, out_dim, d)``.

    ``decode_batch``: ``(m, d) -> (m, out_dim)``, the SAME map
    ``decoder_curvature.plain_decoder_map`` wraps -- ``model.decode``, never
    ``model.forward``. Uses the first-order central difference implied by
    :data:`FD_STENCIL_CONVENTION`'s diagonal formula:
    ``J[..., i] = (f(z + h*e_i) - f(z - h*e_i)) / (2*h)``. Every perturbed point across every
    axis, for the whole batch, is stacked into one tensor and issued to ``decode_batch`` in
    chunks of at most :data:`MAX_FD_ROWS` rows.
    """
    _assert_float64(decode_batch, z)
    if h <= 0:
        raise ValueError(f"finite_difference_jacobian: h must be positive; got {h}.")
    if z.ndim != 2:
        raise ValueError(
            f"finite_difference_jacobian: z must be (batch, d); got shape {tuple(z.shape)}."
        )
    batch, d = z.shape
    if batch == 0:
        raise ValueError("finite_difference_jacobian: z is empty; nothing to differentiate.")

    groups = []
    for i in range(d):
        ei = torch.zeros(d, dtype=z.dtype, device=z.device)
        ei[i] = h
        groups.append(z + ei)
        groups.append(z - ei)
    points = torch.cat(groups, dim=0)
    out = _chunked_eval(decode_batch, points)
    out_dim = out.shape[1]
    out = out.view(2 * d, batch, out_dim)

    J = torch.empty(batch, out_dim, d, dtype=z.dtype, device=z.device)
    for i in range(d):
        J[:, :, i] = (out[2 * i] - out[2 * i + 1]) / (2.0 * h)
    return J


def finite_difference_hessian(
    decode_batch: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    h: float = DEFAULT_FD_STEP,
) -> torch.Tensor:
    """Central-difference Hessian of ``decode_batch`` at every row of ``z``, shape
    ``(batch, out_dim, d, d)``, symmetric by construction (the upper triangle is computed
    once and mirrored onto the lower one, never independently re-evaluated).

    ``decode_batch``: ``(m, d) -> (m, out_dim)``, the SAME map
    ``decoder_curvature.plain_decoder_map`` wraps -- ``model.decode``, never
    ``model.forward``, so what is differentiated is the decoder's own image manifold and not
    the encoder-composed round trip. Implements :data:`FD_STENCIL_CONVENTION` exactly. Every
    perturbed point this stencil needs, across the whole batch, is built into one stacked
    tensor and issued to ``decode_batch`` in chunks of at most :data:`MAX_FD_ROWS` rows -- one
    ``decode_batch`` call per chunk, so the number of calls is
    ``ceil(batch * n_offsets / MAX_FD_ROWS)`` with ``n_offsets = 1 + 2*d + 4*d*(d-1)/2`` (see
    :data:`MAX_FD_ROWS`), never proportional to ``batch`` alone. Requires float64 input,
    checked by ``chart_curvature._assert_float64`` reused unchanged rather than by a new
    check.
    """
    _assert_float64(decode_batch, z)
    if h <= 0:
        raise ValueError(f"finite_difference_hessian: h must be positive; got {h}.")
    if z.ndim != 2:
        raise ValueError(
            f"finite_difference_hessian: z must be (batch, d); got shape {tuple(z.shape)}."
        )
    batch, d = z.shape
    if batch == 0:
        raise ValueError("finite_difference_hessian: z is empty; nothing to differentiate.")

    def _e(i: int) -> torch.Tensor:
        ei = torch.zeros(d, dtype=z.dtype, device=z.device)
        ei[i] = h
        return ei

    groups = [z]  # group 0: the center point, f(z)
    for i in range(d):
        ei = _e(i)
        groups.append(z + ei)
        groups.append(z - ei)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    for i, j in pairs:
        ei, ej = _e(i), _e(j)
        groups.append(z + ei + ej)
        groups.append(z + ei - ej)
        groups.append(z - ei + ej)
        groups.append(z - ei - ej)

    points = torch.cat(groups, dim=0)
    out = _chunked_eval(decode_batch, points)
    out_dim = out.shape[1]
    out = out.view(len(groups), batch, out_dim)

    f0 = out[0]
    H = torch.zeros(batch, out_dim, d, d, dtype=z.dtype, device=z.device)
    idx = 1
    for i in range(d):
        plus, minus = out[idx], out[idx + 1]
        idx += 2
        H[:, :, i, i] = (plus - 2.0 * f0 + minus) / (h * h)
    for i, j in pairs:
        pp, pm, mp, mm = out[idx], out[idx + 1], out[idx + 2], out[idx + 3]
        idx += 4
        val = (pp - pm - mp + mm) / (4.0 * h * h)
        H[:, :, i, j] = val
        H[:, :, j, i] = val
    return H


# --- the mirrored reduction: full derivatives -> mean curvature vector ---------------------


def reduce_to_H_vec(J: torch.Tensor, Hess: torch.Tensor) -> torch.Tensor:
    """Mean curvature vector ``H = tr_g(II)`` from a Jacobian ``(batch, out_dim, d)`` and a
    Hessian ``(batch, out_dim, d, d)`` of the SAME decoder map.

    This is a MIRROR of ``chart_curvature.chart_mean_curvature``'s body -- the trace-first-
    then-project ``d x d`` solve, never an explicit ``(out_dim, out_dim)`` normal projector
    and never a materialized second-fundamental-form tensor -- copied because that sealed
    file is never edited (T-02.6R-14), not re-derived from the algebra. Its only warrant is
    ``test_reduce_to_H_vec_pins_plain_decoder_curvature``, which asserts this reproduces
    ``decoder_curvature.plain_decoder_curvature``'s own ``H_vec`` on both an analytic fixture
    and a trained net -- the only thing standing between a mirror and a silent divergence
    from the reduction it mirrors.

        g    = J^T J                             pullback metric,    (batch, d, d)
        P_N  = I - J g^-1 J^T                    normal projector
        II   = P_N Hess                          second fundamental form
        H    = tr_g(II) = sum_jk g^jk II_jk      (batch, out_dim)
    """
    batch, out_dim, d = J.shape
    g = torch.einsum("boi,boj->bij", J, J)
    eye_d = torch.eye(d, dtype=g.dtype, device=g.device).expand(batch, d, d)
    g_inv = torch.linalg.solve(g, eye_d)

    raw = torch.einsum("bjk,bojk->bo", g_inv, Hess)
    alpha = torch.linalg.solve(
        g, torch.einsum("boi,bo->bi", J, raw).unsqueeze(-1)
    ).squeeze(-1)
    return raw - torch.einsum("boi,bi->bo", J, alpha)


# --- calibration: making assumption A-06 actionable -----------------------------------------

DEFAULT_CALIBRATION_STEPS: Sequence[float] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
"""The decade ladder ``02.6-RESEARCH.md`` Pitfall 5 measured directly against this project's
own ``_SphereDecoder`` fixture, reproducing the float64 central-difference error curve's
U-shape with its minimum at ``1e-4``."""


def calibrate_fd_step(
    decode_batch: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    steps: Sequence[float] = DEFAULT_CALIBRATION_STEPS,
) -> Dict[str, Any]:
    """Evaluate :func:`finite_difference_hessian` at each candidate step in ``steps`` against
    the autodiff Hessian of the SAME map (``torch.func.hessian``, wrapped one point at a time
    exactly as :func:`derivative_agreement` differentiates it), and return the per-step max
    abs error plus the step that minimizes it. Never mutates :data:`DEFAULT_FD_STEP` -- this
    is the function that makes assumption A-06 actionable rather than a warning: a caller
    working on a differently-scaled decoder calls this instead of trusting the default blind.
    """
    _assert_float64(decode_batch, z)
    if z.ndim != 2:
        raise ValueError(f"calibrate_fd_step: z must be (batch, d); got shape {tuple(z.shape)}.")
    if z.shape[0] == 0:
        raise ValueError("calibrate_fd_step: z is empty; nothing to calibrate against.")

    def decode_one(zz: torch.Tensor) -> torch.Tensor:
        return decode_batch(zz.unsqueeze(0)).squeeze(0)

    autodiff_hess = vmap(hessian(decode_one))(z)

    step_errors: Dict[float, float] = {}
    for h in steps:
        fd_hess = finite_difference_hessian(decode_batch, z, h=float(h))
        step_errors[float(h)] = float((fd_hess - autodiff_hess).abs().max().item())

    best_step = min(step_errors, key=step_errors.get)
    return {
        "step_errors": step_errors,
        "best_step": best_step,
        "best_error": step_errors[best_step],
    }


# --- the bridge's reporting unit -------------------------------------------------------------

_QUANTILE_MAX_ELEMENTS = 16_777_216
"""``2**24`` -- ``torch.quantile``'s own hard input-size cap (an ATen-level assert, not a
tunable option): above this many elements it raises
``RuntimeError: quantile() input tensor is too large`` rather than returning a value.

**POST-COMPLETION REPAIR (found 2026-08-11, during plan 02.6-14's Task 2 full PU-regime
run -- documented here since this reopens a module plan 02.6-10 had already delivered and
sealed).** The PU regime's full Hessian tensor at plan 02.6-14's ``BRIDGE_N_POINTS=32``,
``out_dim=768``, ``d=40`` is ``32 * 768 * 40 * 40 = 39,321,600`` elements -- 2.3x over this
cap. ``_p90`` below is the fix: it calls ``torch.quantile`` UNCHANGED (bit-for-bit identical
to the pre-repair behaviour) for any tensor at or below this cap, and falls back to
``numpy.quantile``'s own default ``method="linear"`` only above it -- the SAME
linear-interpolation formula ``torch.quantile``'s default computes
(``index = q * (n - 1)``, interpolated between the two nearest order statistics), pinned
equal below the cap by ``test_p90_matches_torch_quantile_below_the_cap`` rather than merely
asserted. None of plan 02.6-10's 13 tests exercised :func:`_agreement_stats` at PU
dimensionality, so this defect passed every one of that plan's acceptance criteria and
surfaced only at real scale -- ``test_agreement_stats_handles_tensor_above_quantile_cap``
below is the regression guard that would have caught it."""


def _p90(x: torch.Tensor) -> float:
    """The 90th percentile of ``x``'s flattened values. Delegates to ``torch.quantile``
    unchanged when ``x.numel() <= _QUANTILE_MAX_ELEMENTS`` (the common case, and every case
    the Swiss roll arm ever hits); above that cap -- which ``torch.quantile`` itself
    enforces and refuses to compute past -- falls back to ``numpy.quantile`` on the same
    flattened values, using its default ``method="linear"``, the identical interpolation
    formula. See :data:`_QUANTILE_MAX_ELEMENTS` for the defect this fixes and the PU-regime
    arithmetic that exceeds it."""
    flat = x.reshape(-1)
    if flat.numel() <= _QUANTILE_MAX_ELEMENTS:
        return float(torch.quantile(flat, 0.90).item())
    return float(np.quantile(flat.detach().cpu().numpy(), 0.90))


def _agreement_stats(reference: torch.Tensor, other: torch.Tensor) -> Dict[str, float]:
    """Max, median and 90th-percentile absolute difference between ``reference`` (the
    autodiff side) and ``other`` (the finite-difference side), and the same three relative to
    ``reference``'s own scale. Shared by the full-tensor and both reduced comparisons in
    :func:`derivative_agreement` so the two levels are computed identically and only differ
    in which tensors are handed in."""
    diff = (reference - other).abs().reshape(-1)
    scale = reference.abs().reshape(-1).clamp_min(1e-300)
    rel = diff / scale
    return {
        "max_abs": float(diff.max().item()),
        "median_abs": float(diff.median().item()),
        "p90_abs": _p90(diff),
        "max_abs_relative": float(rel.max().item()),
        "median_abs_relative": float(rel.median().item()),
        "p90_abs_relative": _p90(rel),
    }


def derivative_agreement(
    model: Any, z: torch.Tensor, h: Optional[float] = None
) -> Dict[str, Any]:
    """The bridge's reporting unit (D-16, D-17, D-18): autodiff-versus-finite-difference
    agreement on ``model``'s decoder second derivatives, reported at BOTH the full Hessian
    tensor level and the reduced mean-curvature level, under separate keys, never combined
    into one number (S1). Applies no acceptance rule and returns no boolean judgement of any
    kind -- ``02.5-10`` is the only place a bound is ever ratified against these numbers.

    ``z``: ``(batch, d)`` latent coordinates, float64. ``h``: the finite-difference step;
    when ``None`` (the default) :func:`calibrate_fd_step` is run against ``z`` first and its
    ``best_step`` is used, per assumption A-06's own recommendation not to trust
    :data:`DEFAULT_FD_STEP` blind.

    Both reductions below share the SAME autodiff Jacobian ``J`` for the pullback-metric
    solve -- one is reduced with the autodiff Hessian, the other with the finite-difference
    Hessian -- so any difference between the two reduced levels isolates the effect of
    swapping the Hessian source alone, uncontaminated by a second, independent source of
    Jacobian disagreement. D-16 is a question about second derivatives specifically.

    Returns a dict:

      ``"full_hessian_agreement"``               max/median/90th-percentile absolute
                                                   difference between the autodiff and
                                                   finite-difference Hessian TENSORS, and the
                                                   same three relative to the autodiff
                                                   tensor's own scale
      ``"reduced_mean_curvature_agreement"``      the same six statistics, separately, on
                                                   ``H_vec`` and on ``H_norm``
      ``"fd_step_used"``                          the step actually used (calibrated or
                                                   caller-supplied)
      ``"metric_condition_number"``               ``(batch,)`` -- ``cond(g)`` per point, the
                                                   same quantity the autodiff tracer already
                                                   reports
      ``"activation"``                            :func:`decoder_curvature.assert_c2_decoder`'s
                                                   return value
    """
    activation = assert_c2_decoder(model)
    _assert_float64(model, z)

    if z.ndim != 2:
        raise ValueError(
            f"derivative_agreement: z must be (batch, d); got shape {tuple(z.shape)}."
        )
    batch = z.shape[0]
    if batch == 0:
        raise ValueError("derivative_agreement: z is empty; nothing to differentiate.")

    decode_one = plain_decoder_map(model)

    # autodiff Jacobian and Hessian, chunked at chart_curvature.VMAP_CHUNK exactly as
    # decoder_curvature.plain_decoder_curvature -- reused, not re-derived.
    J_parts, Hess_parts = [], []
    for start in range(0, batch, VMAP_CHUNK):
        real = z[start : start + VMAP_CHUNK]
        n_real = real.shape[0]
        chunk = _pad_to_chunk(real)
        J_parts.append(vmap(jacrev(decode_one))(chunk)[:n_real].detach())
        Hess_parts.append(vmap(hessian(decode_one))(chunk)[:n_real].detach())
    J = torch.cat(J_parts, dim=0)
    Hess_autodiff = torch.cat(Hess_parts, dim=0)

    def decode_batch(zz: torch.Tensor) -> torch.Tensor:
        return model.decode(zz)

    if h is None:
        h_used = calibrate_fd_step(decode_batch, z)["best_step"]
    else:
        h_used = float(h)

    Hess_fd = finite_difference_hessian(decode_batch, z, h=h_used)

    g = torch.einsum("boi,boj->bij", J, J)
    metric_condition_number = torch.linalg.cond(g)

    H_vec_autodiff = reduce_to_H_vec(J, Hess_autodiff)
    H_vec_fd = reduce_to_H_vec(J, Hess_fd)
    H_norm_autodiff = torch.linalg.norm(H_vec_autodiff, dim=-1)
    H_norm_fd = torch.linalg.norm(H_vec_fd, dim=-1)

    return {
        "full_hessian_agreement": _agreement_stats(Hess_autodiff, Hess_fd),
        "reduced_mean_curvature_agreement": {
            "H_vec": _agreement_stats(H_vec_autodiff, H_vec_fd),
            "H_norm": _agreement_stats(H_norm_autodiff, H_norm_fd),
        },
        "fd_step_used": float(h_used),
        "metric_condition_number": metric_condition_number,
        "activation": activation,
    }
