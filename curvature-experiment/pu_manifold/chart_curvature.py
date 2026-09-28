"""
Phase 02.5 stage-2 Arm B: EXACT mean curvature through a CAE chart decoder, computed by
``torch.func`` autodiff rather than estimated statistically. Tensors in, tensors and dicts
out -- no file I/O, no cache handling; the runners under ``curvature-experiment/runners/`` own
paths and cache stems. Constants live in ``02.5-PREREGISTRATION-STAGE2.md``.

Like ``cae.py`` and unlike every other module in this package, this one imports ``torch``
at module level: differentiating a trained decoder genuinely needs it. For the same reason
``cae.py``, ``curvature.py`` and ``mknn.py`` are excluded from ``pu_manifold/__init__.py``'s
eager imports (so Phase-1-only callers do not need torch installed to import the package),
this module is deliberately NOT re-exported there either.

RESEARCH Open Question 1, resolved and recorded here rather than left implicit: this module
computes the same mathematics as ``archive/notebooks/pu_manifold/curvature.py``'s
``first_fundamental_form``, ``second_fundamental_form``, ``mean_curvature_vector`` and
``metric_condition_number`` stubs, and it does **not** fill, edit, or import them. Those four
stubs are each docstring-labelled "Implemented in Phase 3 (CURV-0N)" and ``REQUIREMENTS.md``
maps CURV-01..08 to Phase 3, so filling them here would silently deliver Phase 3 requirements
ahead of schedule under a phase-02.5 commit. The duplication is deliberate, is recorded in
``02.5-01-PLAN.md``'s ``<decisions_resolved_here>`` OQ-1, and is pinned from the other side by
``test_phase3_curvature_stubs_remain_unimplemented``, which asserts all four still raise.

Why this arm exists, in one paragraph, because it is easy to restate wrongly. Stage 1's
point-cloud estimator returned ``CURVATURE_VERDICT = FAIL`` for a measured structural reason:
at ``d = 20, n = 10^4, k = 30`` the k-NN ball radius is **90.6% of the manifold's own radius**
(11.5% at ``d = 2``), and ``r/R ~ (k/n)^(1/d)``, so halving it would need ~10^6 times more
data. No local neighbourhood exists in the point cloud at that intrinsic dimension. The
decoder arm escapes this by forming no neighbourhood at all -- ``r -> 0`` is achieved
analytically by autodiff, so ``r/R`` never enters its error. **That escape is statistical,
not computational**; the arm's advantage is not that a trace is cheaper than a full second
fundamental form. See ``02.5-NOTE-high-d-curvature-approaches.md`` Section 1 and
``02.5-NOTE-randomized-trace.md`` Addendum B.

What the arm pays for that escape, and what this module therefore also measures: ``H_F`` is
the exact curvature of the LEARNED manifold ``M_F = F(R^d)``, not of the data manifold. A
reconstruction objective asks ``F(E(x)) ~= x``; it never asks ``D^2 F ~= D^2 F_true``.
Reconstruction quality can therefore never validate a curvature estimate -- a decoder that
learns ``y = 0.7 a x^2`` where the truth is ``y = a x^2`` has tiny reconstruction error
wherever the sampled ``x`` sit near zero while its second derivative is ``1.4a`` instead of
``2a``, i.e. 30% curvature attenuation with no reconstruction signal at all. Both stage-1
gating statistics are rank-based and are exactly blind to this. :func:`curvature_fidelity_report`
is the countermeasure and it reports direction, magnitude and calibration SEPARATELY, never
collapsed into one scalar.

Curvature convention -- the single most expensive thing in this file to get wrong. Every
quantity here is the UNNORMALIZED trace ``H = tr_g(II)``, matching ``curvature_probe.py``'s
:data:`~pu_manifold.curvature_probe.CURVATURE_CONVENTION` and ``curvature.py``'s own stub
docstring ("g-trace of the second fundamental form"). The external source material circulating
on this topic uses the AVERAGED convention ``H = (1/d) tr_g(II)``; transcribing it verbatim
introduces a factor-of-``d`` = 20 error against every fixture, every 02.5 SUMMARY number, and
the sealed stage-1 gate. Under the trace convention the Laplace-Beltrami identity reads
``Delta_g F = H``, **not** ``Delta_g F = d H``. This codebase has already shipped and then
fixed exactly one factor-of-``d`` scale bug (``2*(d+2)/r2 -> 2*d/r2``, plan 02.5-01), which is
why the convention carries a regression guard
(``test_chart_curvature_uses_trace_convention_not_averaged``) rather than a comment.
"""

from typing import Any, Dict

import numpy as np
import torch

CURVATURE_CONVENTION = "trace"
"""``H = tr_g(II)``, the unnormalized ``g``-trace of the second fundamental form -- never the
averaged ``(1/d) tr_g(II)``. Deliberately equal to
``curvature_probe.CURVATURE_CONVENTION`` so the two arms of stage 2 are directly comparable
without a scale correction, and asserted equal by this phase's tests."""

VMAP_CHUNK = 32
"""The FIXED autodiff batch width. Every ``vmap``'d derivative in this module is taken over
exactly this many rows, with a short final chunk padded up to it and the padding discarded.

This is a reproducibility constant, not a tuning knob, and it is not caller-overridable on
purpose. Measured during plan 02.5-08: ``vmap(hessian(f))`` is **not** bit-reproducible across
differing batch widths -- the same row computed in a batch of 8 and in a batch of 24 differs by
~1.7e-18, because torch selects different batched-matmul kernels by batch size. Propagated
through the metric solve and the normal projection, that grew to ~5e-15 in ``H_norm``.

**Correction, verified independently after this module was written:** an earlier draft of this
docstring stated that ``jacrev`` was bit-identical under the same comparison and that "only the
doubly-nested Hessian transform moved." **That is not generally true, and must not be relied
on.** On a decoder-shaped map (``W2 @ silu(W1 z + b1) + b2``, ``chart_dim=20``, ``out_dim=768``)
``vmap(jacrev(f))`` at width 8 versus width 64 differs by ``1.07e-14`` -- the same order as
``vmap(hessian(f))`` on the identical construction. Whether ``jacrev`` moves is
construction-dependent, which is precisely why it cannot be assumed stable.

Consequence for anyone editing this module: the chunking around the ``jacrev`` call in
:func:`_chunked_jacobian` is **load-bearing and must not be removed as redundant**. Every
``vmap``'d derivative here is chunked, and that is deliberate rather than defensive uniformity.

Measured in the same session and load-bearing for this fix: at a FIXED width the result for a
given row is bit-identical regardless of which other rows share its chunk, and regardless of
whether the chunk is filled with real rows or with repeated padding. Fixing the width therefore
buys exact reproducibility of the whole field across any caller-side batching, which is what
makes ``test_chart_curvature_field_reassembles_in_row_order``'s ``torch.equal`` assertion a
genuine row-order check rather than a tolerance in disguise. The cost is at most one padded
chunk per call.

It also sets peak memory, since the Hessian dominates: ``VMAP_CHUNK * out_dim * chart_dim^2``
float64 entries. At the sealed 02.2 architecture (``out_dim = 768``, ``chart_dim = 20``) that is
``32 * 768 * 400 * 8 B = 78.6 MB``, with true peak a small multiple of it (``torch.func`` holds
intermediate ``jacfwd``/``jacrev`` buffers of comparable size)."""


# --- C2 smoothness guard (RESEARCH Pitfall 4, threat T-02.5-06) --------------------------

ZERO_SECOND_DERIVATIVE_ACTIVATIONS = frozenset(
    {
        "relu",
        "relu6",
        "leaky_relu",
        "leakyrelu",
        "prelu",
        "rrelu",
        "elu",
        "hardtanh",
        "hardshrink",
        "softshrink",
        "threshold",
    }
)
"""Activations whose second derivative is identically zero wherever autodiff evaluates it,
or which are C1-but-not-C2 at a kink. ``"elu"`` is included on the second ground: it is
smooth away from the origin but its second derivative jumps there, so it is not a C2 map and
must not be differentiated twice for curvature. ``cae.activation_module`` reaches ``"relu"``
only for the deliberate CAE-06 ReLU control fit; the other names are listed so that a future
decoder built outside ``cae.py`` cannot slip past this guard."""


def assert_c2_activation(model: Any) -> str:
    """Raise ``ValueError`` naming the activation unless ``model.activation`` is C2-smooth.

    RESEARCH Pitfall 4 and threat T-02.5-06. A piecewise-linear activation's second
    derivative is a sum of Dirac deltas at the kinks, which is numerically **exactly zero**
    everywhere autodiff evaluates it. Differentiating such a decoder does not fail: it
    returns a second fundamental form of exactly ``0.0`` at every point, which reads
    downstream as a perfectly flat manifold. That is a silent wrong answer, and it is the
    single most dangerous failure mode in this module.

    Pitfall 4's own stated mitigation is to confirm this at load time by checking the
    recorded attribute rather than trusting the cache stem name -- implemented exactly that
    way here. This function RAISES rather than warning, because a warning emitted inside a
    batch runner is a silent failure with extra steps.

    Returns the (lower-cased) activation name on success, so a caller can record it.
    """
    if not hasattr(model, "activation"):
        raise ValueError(
            "assert_c2_activation: model has no 'activation' attribute, so its smoothness "
            "cannot be confirmed. Refusing to differentiate an unknown decoder twice -- a "
            "zero-second-derivative activation would return an identically-zero second "
            "fundamental form rather than raising."
        )
    name = str(model.activation).lower()
    if name in ZERO_SECOND_DERIVATIVE_ACTIVATIONS:
        raise ValueError(
            f"assert_c2_activation: refusing to compute curvature through a decoder with "
            f"activation {name!r}. A piecewise-linear (or C1-but-not-C2) activation has a "
            f"second derivative that is a sum of Dirac deltas at its kinks, numerically zero "
            f"everywhere autodiff evaluates it, so the entire second fundamental form would "
            f"come back as exactly 0.0 and read as a flat manifold instead of raising. The "
            f"sealed 02.2 fits use 'silu' precisely so that this arm is possible; "
            f"activation {name!r} is reachable only for the CAE-06 ReLU control fit, which is "
            f"not a curvature-bearing model."
        )
    return name


# --- the map that is differentiated ------------------------------------------------------


def _assert_float64(model: Any, z_chart: torch.Tensor) -> None:
    """Refuse to run in float32. ``cae.py``'s fits persist as float64 arrays, and second
    derivatives are exactly where float32 noise shows up first -- a float32 Hessian of a
    three-hidden-layer MLP loses several digits before the normal projection even starts.
    Raising names the fix (``model.double()``) rather than silently degrading precision on a
    number that ends up in a verdict."""
    if z_chart.dtype != torch.float64:
        raise ValueError(
            f"chart curvature runs in float64 throughout; got z_chart.dtype={z_chart.dtype}. "
            f"Second derivatives are where float32 noise shows. Pass z_chart.double()."
        )
    params = getattr(model, "parameters", None)
    if callable(params):
        for p in params():
            if p.dtype != torch.float64:
                raise ValueError(
                    f"chart curvature runs in float64 throughout; the decoder carries "
                    f"{p.dtype} parameters. Call model.double() before differentiating -- "
                    f"the sealed 02.2 fits persist as float64 arrays, so this is a reload "
                    f"precision loss, not an inherent limit."
                )
            break


def _pad_to_chunk(rows: torch.Tensor) -> torch.Tensor:
    """Pad a short final chunk up to :data:`VMAP_CHUNK` by repeating its last row. The
    padding is discarded by the caller; see :data:`VMAP_CHUNK` for why a row's result does
    not depend on which rows share its chunk."""
    missing = VMAP_CHUNK - rows.shape[0]
    if missing <= 0:
        return rows
    return torch.cat([rows, rows[-1:].repeat(missing, 1)], dim=0)


# --- amplitude and orientation fidelity: the checks a rank statistic cannot make ---------


def curvature_fidelity_report(
    H_est: Any, H_true: Any, min_true_norm: float = 1e-12
) -> Dict[str, Any]:
    """Compare an estimated mean curvature FIELD against a known analytic one on three
    separate axes -- direction, magnitude, and calibration -- and deliberately never collapse
    them into a single score.

    ``H_est``, ``H_true``: ``(n, D)`` mean curvature vectors (numpy arrays or torch tensors),
    in the same ambient frame.

    **Why this function exists** (``02.5-NOTE-randomized-trace.md`` Addendum C and
    ``02.5-NOTE-high-d-curvature-approaches.md`` Section 2d). The decoder arm's curvature
    ``H_F`` is the exact curvature of the LEARNED manifold ``M_F = F(R^d)``, not of the data
    manifold. A decoder trained to reconstruct will happily regularize the bumps flatter than
    they are, producing a field that is smooth, well ordered, highly rank-correlated with the
    truth -- and systematically wrong in amplitude. Both stage-1 gating statistics
    (``spearman_rho`` and ``quantile_bin_concordance``) are rank-based and score such a
    decoder a perfect ``1.0``. The worked example, recorded because it is the sharpest form of
    the point: a decoder learning ``y = 0.7 a x^2`` where the truth is ``y = a x^2`` has tiny
    reconstruction error wherever the sampled ``x`` sit near zero, while its second derivative
    is ``1.4a`` instead of ``2a`` -- 30% curvature attenuation with no reconstruction signal at
    all. **Reconstruction quality can never validate a curvature estimate.**

    **Why three numbers and not one.** ``H`` is vector-valued, so amplitude attenuation and
    orientation error are DISTINCT failure modes with different remedies; a single scalar
    would let either hide behind the other.

      1. ``direction`` -- per-point cosine similarity between ``H_est`` and ``H_true``.
      2. ``magnitude`` -- the per-point ratio ``||H_est|| / ||H_true||``, reported as BOTH the
         median AND the coefficient of variation. Both are required and neither suffices:
         Section 1a measured the point-cloud estimator at ``d = 2`` giving median 0.905 at
         CV 0.250 ("a mild underestimate with modest scatter -- a correctable signature") and
         at ``d = 20`` giving CV 2.250 with a median that is not even monotone in ``d`` ("the
         scatter is 2.25x the mean ... there is no scale factor to calibrate out"). The pair
         separates *attenuated but calibratable* from *destroyed*; either number alone does
         not.
      3. ``calibration slope`` -- least-squares regression of ``||H_est||`` on ``||H_true||``,
         checking ``a ~= 1`` and ``b ~= 0``. A rank statistic cannot see ``a = 0.5``; this can.

    Points where ``||H_true|| <= min_true_norm`` carry no direction and no meaningful ratio
    (a flat point has no curvature to get right), so they are excluded from all three
    statistics and counted in ``"n_excluded"`` rather than silently contributing a division by
    something near zero. The CV uses the sample standard deviation (``ddof=1``).

    Gates nothing on its own: it produces the numbers that a pre-registered stage-2 rule reads.
    """
    est = np.asarray(
        H_est.detach().cpu().numpy() if isinstance(H_est, torch.Tensor) else H_est,
        dtype=np.float64,
    )
    true = np.asarray(
        H_true.detach().cpu().numpy() if isinstance(H_true, torch.Tensor) else H_true,
        dtype=np.float64,
    )
    if est.shape != true.shape or est.ndim != 2:
        raise ValueError(
            f"curvature_fidelity_report: H_est and H_true must be the same (n, D) shape; got "
            f"{est.shape} and {true.shape}. Comparing curvature fields expressed in different "
            f"ambient frames is a category error, not a tolerance question."
        )

    norm_est = np.linalg.norm(est, axis=-1)
    norm_true = np.linalg.norm(true, axis=-1)

    keep = (
        (norm_true > min_true_norm)
        & np.isfinite(norm_est)
        & np.isfinite(norm_true)
        & (norm_est > 0.0)
    )
    n_total = int(est.shape[0])
    n_kept = int(keep.sum())
    if n_kept < 2:
        raise ValueError(
            f"curvature_fidelity_report: only {n_kept} of {n_total} points have a usable "
            f"analytic curvature (||H_true|| > {min_true_norm}) and a finite estimate. A "
            f"fidelity report over fewer than two points is not a measurement."
        )

    e, t = est[keep], true[keep]
    ne, nt = norm_est[keep], norm_true[keep]

    cosine = np.einsum("nd,nd->n", e, t) / (ne * nt)
    ratio = ne / nt

    # calibration: ||H_est|| = a * ||H_true|| + b
    design = np.stack([nt, np.ones_like(nt)], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(design, ne, rcond=None)
    residual = ne - (slope * nt + intercept)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((ne - ne.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    mean_ratio = float(np.mean(ratio))
    cv = float("nan") if mean_ratio == 0.0 else float(np.std(ratio, ddof=1) / mean_ratio)

    return {
        # 1. direction
        "cosine_similarity": cosine,
        "median_cosine_similarity": float(np.median(cosine)),
        # 2. magnitude -- median AND CV, never one without the other
        "magnitude_ratio": ratio,
        "median_magnitude_ratio": float(np.median(ratio)),
        "mean_magnitude_ratio": mean_ratio,
        "magnitude_ratio_cv": cv,
        # 3. calibration
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "calibration_r2": r2,
        "n_points": n_kept,
        "n_excluded": n_total - n_kept,
        "curvature_convention": CURVATURE_CONVENTION,
    }
