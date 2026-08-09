"""
Phase 02.5 stage-2 Arm B: EXACT mean curvature through a CAE chart decoder, computed by
``torch.func`` autodiff rather than estimated statistically. Tensors in, tensors and dicts
out -- no file I/O, no cache handling; the runners under ``notebooks/diagnostics/`` own
paths and cache stems. Constants live in ``02.5-PREREGISTRATION-STAGE2.md``.

Like ``cae.py`` and unlike every other module in this package, this one imports ``torch``
at module level: differentiating a trained decoder genuinely needs it. For the same reason
``cae.py``, ``curvature.py`` and ``mknn.py`` are excluded from ``pu_manifold/__init__.py``'s
eager imports (so Phase-1-only callers do not need torch installed to import the package),
this module is deliberately NOT re-exported there either.

RESEARCH Open Question 1, resolved and recorded here rather than left implicit: this module
computes the same mathematics as ``notebooks/pu_manifold/curvature.py``'s
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

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from torch.func import hessian, jacrev, jvp, vmap

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
~1.7e-18, because torch selects different batched-matmul kernels by batch size. ``jacrev`` and
``linalg.solve``/``cond`` were bit-identical under the same comparison; only the doubly-nested
Hessian transform moved. Propagated through the metric solve and the normal projection, that
grew to ~5e-15 in ``H_norm``.

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


def chart_decoder_map(model: Any, chart_idx: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``decode_one(zc)``: a single ``(chart_dim,)`` chart coordinate to a single
    ``(out_dim,)`` ambient point, through chart ``chart_idx``'s own decoder and then the one
    shared embedding decoder (the two-hop decode, ``cae.py`` Pitfall 1).

    **Deviation from ``cae._decode_through_chart``, deliberate and flagged by
    ``02.5-PATTERNS.md``.** That helper takes ``z``, the *initial encoder's* output, and
    recomputes the chart coordinate internally via ``model.chart_coords(z)``. Wrapping it
    under ``jacrev``/``hessian`` would therefore differentiate through the
    chart-coordinate-producing encoder as well, and measure the second fundamental form of
    the encoder-composed map rather than of the chart. For curvature the chart coordinate
    **is** the local parameterization whose second fundamental form is being measured, so only
    the decoder half is differentiated here. ``cae._decode_through_chart`` is read for its
    composition and is deliberately not itself wrapped under a transform; ``cae.py`` is not
    edited.

    ``model`` need not be a full ``cae.ChartAutoEncoder``: only ``chart_decoders`` and
    ``embedding_decoder`` are consumed, so a duck-typed known-answer fixture works (02.2-05's
    precedent).
    """
    chart_decoder = model.chart_decoders[chart_idx]
    embedding_decoder = model.embedding_decoder

    def decode_one(zc: torch.Tensor) -> torch.Tensor:
        return embedding_decoder(chart_decoder(zc.unsqueeze(0))).squeeze(0)

    return decode_one


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


def _batched_eye(n: int, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(size, dtype=dtype, device=device).expand(n, size, size)


def _pad_to_chunk(rows: torch.Tensor) -> torch.Tensor:
    """Pad a short final chunk up to :data:`VMAP_CHUNK` by repeating its last row. The
    padding is discarded by the caller; see :data:`VMAP_CHUNK` for why a row's result does
    not depend on which rows share its chunk."""
    missing = VMAP_CHUNK - rows.shape[0]
    if missing <= 0:
        return rows
    return torch.cat([rows, rows[-1:].repeat(missing, 1)], dim=0)


def _chunked_jacobian(
    decode_one: Callable[[torch.Tensor], torch.Tensor], z_chart: torch.Tensor
) -> torch.Tensor:
    """``(batch, out_dim, chart_dim)`` decoder Jacobian, taken at the fixed autodiff width."""
    parts = []
    for start in range(0, z_chart.shape[0], VMAP_CHUNK):
        real = z_chart[start : start + VMAP_CHUNK]
        parts.append(vmap(jacrev(decode_one))(_pad_to_chunk(real))[: real.shape[0]].detach())
    return torch.cat(parts, dim=0)


# --- the gating computation: exact g-trace of the normal-projected Hessian ----------------


def chart_mean_curvature(
    model: Any, z_chart: torch.Tensor, chart_idx: int
) -> Dict[str, Any]:
    """Exact mean curvature vector of the manifold parameterized by chart ``chart_idx``'s
    decoder, at each chart coordinate in ``z_chart``.

    ``z_chart``: ``(batch, chart_dim)`` coordinates **within one chart's own coordinate
    space** -- ``model.chart_coords(z)[:, chart_idx, :]``, NOT the initial-encoder embedding
    ``z``. See :func:`chart_decoder_map` for why that distinction is the whole point.

    Returns a dict, not a bare tensor, because three separate things must be visible to the
    caller and the plan requires all of them: the curvature itself, the shapes the
    ``torch.func`` transform composition actually produced (Pitfall 5), and the conditioning
    of the pullback metric (threat T-02.5-20).

      ``"H_vec"``                    ``(batch, out_dim)``  mean curvature vectors, ``tr_g(II)``
      ``"H_norm"``                   ``(batch,)``          ``||H||``, the only reportable scalar
      ``"metric_condition_number"``  ``(batch,)``          ``cond(g)`` per point
      ``"jacobian_shape"``           tuple                 as produced, for the shape assertion
      ``"hessian_shape"``            tuple                 as produced, for the shape assertion
      ``"chart_idx"``, ``"activation"``                    provenance

    Mathematics, under this module's ``H = tr_g(II)`` trace convention:

        J    = D F(z)                            (out_dim, chart_dim)
        g    = J^T J                             (chart_dim, chart_dim) pullback metric
        P_N  = I - J g^-1 J^T                    normal projector
        II   = P_N D^2 F(z)                      second fundamental form
        H    = tr_g(II) = sum_jk g^jk II_jk      (out_dim,)

    **Implementation deviation from RESEARCH Pattern 4's illustrative snippet, deliberate.**
    That snippet materializes ``P_N`` as an explicit ``(out_dim, out_dim)`` matrix and forms
    the full ``II`` tensor before tracing. At the real ``out_dim = 768`` a batch of 32 such
    projectors is 151 MB of float64 and ``II`` is another 78 MB, both of which are pure waste.
    Since ``P_N`` acts only on the ambient index and the ``g``-trace acts only on the two
    chart indices, the two commute: ``tr_g(P_N Hess) = P_N tr_g(Hess)``. This function
    therefore traces first and projects second, and applies the projector by the
    ``chart_dim x chart_dim`` solve ``P_N a = a - J alpha`` with ``g alpha = J^T a`` -- never
    materializing a ``(D, D)`` matrix. That is an optimisation of the same mathematics, so it
    is proved rather than asserted: ``test_chart_curvature_dxd_solve_matches_explicit_projector``
    reimplements Pattern 4's snippet verbatim and requires agreement to float64 round-off.

    ``g_inv`` is obtained by ``torch.linalg.solve`` against a batched identity rather than by
    ``torch.linalg.inv``. The two are numerically near-identical here (``solve`` is an LU
    factorization plus triangular solves; ``inv`` is the same followed by an extra multiply),
    but ``solve`` is the better-conditioned formulation and costs nothing extra at
    ``chart_dim = 20``. ``cond(g)`` is returned alongside so that a near-singular pullback
    metric at a non-immersion point -- where a decoder's differential drops rank and the
    inverse metric blows up -- is visible in the artifact rather than silently inflating the
    curvature field (threat T-02.5-20).

    Runs in float64 throughout; :func:`_assert_float64` refuses anything else.
    """
    activation = assert_c2_activation(model)
    _assert_float64(model, z_chart)

    if z_chart.ndim != 2:
        raise ValueError(
            f"chart_mean_curvature: z_chart must be (batch, chart_dim); got shape "
            f"{tuple(z_chart.shape)}."
        )
    batch, chart_dim = z_chart.shape
    if batch == 0:
        raise ValueError("chart_mean_curvature: z_chart is empty; nothing to differentiate.")

    decode_one = chart_decoder_map(model, chart_idx)

    # One cheap forward pass establishes out_dim from the map itself rather than trusting an
    # attribute, so the shape assertions below compare against what the decoder actually
    # emits. Cross-checked against model.out_dim when that attribute exists.
    with torch.no_grad():
        probe = decode_one(z_chart[0])
    if probe.ndim != 1:
        raise ValueError(
            f"chart_mean_curvature: the chart decoder map must send a (chart_dim,) tensor to "
            f"a (out_dim,) tensor; it returned shape {tuple(probe.shape)}."
        )
    out_dim = int(probe.shape[0])
    declared = getattr(model, "out_dim", None)
    if declared is not None and int(declared) != out_dim:
        raise ValueError(
            f"chart_mean_curvature: model.out_dim is {int(declared)} but the decoder map "
            f"emits {out_dim} components. Refusing to proceed on a model whose declared "
            f"ambient dimension disagrees with its own output."
        )

    H_parts = []
    cond_parts = []
    for start in range(0, batch, VMAP_CHUNK):
        real = z_chart[start : start + VMAP_CHUNK]
        n_real = real.shape[0]
        # Pad a short final chunk up to the fixed width and discard the padding afterwards.
        # See VMAP_CHUNK: a row's result does not depend on which rows share its chunk, only
        # on the chunk's width, so this is exact rather than approximate.
        chunk = _pad_to_chunk(real)

        J = vmap(jacrev(decode_one))(chunk)
        if tuple(J.shape) != (VMAP_CHUNK, out_dim, chart_dim):
            raise ValueError(
                f"chart_mean_curvature: expected Jacobian of shape "
                f"{(VMAP_CHUNK, out_dim, chart_dim)}, got {tuple(J.shape)}. torch.func's "
                f"transform composition returned something other than a per-point Jacobian "
                f"(RESEARCH Pitfall 5)."
            )

        Hess = vmap(hessian(decode_one))(chunk)
        if tuple(Hess.shape) != (VMAP_CHUNK, out_dim, chart_dim, chart_dim):
            raise ValueError(
                f"chart_mean_curvature: expected Hessian of shape "
                f"{(VMAP_CHUNK, out_dim, chart_dim, chart_dim)}, got {tuple(Hess.shape)}. A "
                f"Jacobian-shaped result here is RESEARCH Pitfall 5's exact warning sign: "
                f"jacrev(jacrev(f)) and hessian(f) are not drop-in interchangeable under an "
                f"outer vmap, and the wrong composition order still runs."
            )

        g = torch.einsum("boi,boj->bij", J, J)
        eye_d = _batched_eye(VMAP_CHUNK, chart_dim, g.dtype, g.device)
        g_inv = torch.linalg.solve(g, eye_d)

        # g-trace first (the two operations commute; see the docstring), then the normal
        # projection via the chart_dim x chart_dim solve. No (out_dim, out_dim) matrix is
        # ever formed, and no full II tensor is ever materialized.
        raw = torch.einsum("bjk,bojk->bo", g_inv, Hess)
        alpha = torch.linalg.solve(
            g, torch.einsum("boi,bo->bi", J, raw).unsqueeze(-1)
        ).squeeze(-1)

        # detach: curvature is a measured value here, never an optimization objective. Left
        # attached, a 10,000-row field would retain the full autodiff graph of every chunk.
        H_parts.append((raw - torch.einsum("boi,bi->bo", J, alpha))[:n_real].detach())
        cond_parts.append(torch.linalg.cond(g)[:n_real].detach())

    H_vec = torch.cat(H_parts, dim=0)
    metric_cond = torch.cat(cond_parts, dim=0)

    return {
        "H_vec": H_vec,
        "H_norm": chart_mean_curvature_norm(H_vec),
        "metric_condition_number": metric_cond,
        "jacobian_shape": (batch, out_dim, chart_dim),
        "hessian_shape": (batch, out_dim, chart_dim, chart_dim),
        "chart_idx": int(chart_idx),
        "activation": activation,
        "curvature_convention": CURVATURE_CONVENTION,
    }


def chart_mean_curvature_norm(H: torch.Tensor) -> torch.Tensor:
    """``||H||`` per point: ``torch.linalg.norm(H, dim=-1)``.

    The norm of the mean curvature *vector* is the only reportable scalar, and this is not a
    stylistic preference. At codimension ``768 - 20`` there is no canonical normal direction,
    so any reduction of ``H`` to a signed scalar requires choosing one, and the sign of the
    result flips with that arbitrary choice. ``curvature.py``'s own module docstring and
    CURV-03 both mandate the norm for exactly this reason, and both record that Gaussian and
    principal curvature are category errors at this codimension rather than merely
    inconvenient.
    """
    return torch.linalg.norm(H, dim=-1)


def chart_curvature_field(
    model: Any, x: torch.Tensor, batch_size: int = 32
) -> Dict[str, Any]:
    """Per-point mean curvature over an ambient point cloud, each row measured in the chart
    the model itself assigns to it.

    ``x``: ``(n, in_dim)`` ambient rows. Encodes to ``z``, takes ``chart_coords(z)`` and
    ``chart_probs(z).argmax(dim=1)``, then for each chart index calls
    :func:`chart_mean_curvature` on exactly the rows assigned to that chart, using that
    chart's own coordinates, and reassembles the result in the original row order.

    Returns ``{"H_vec": (n, out_dim), "H_norm": (n,), "chart_assignment": (n,),
    "metric_condition_number": (n,), "n_charts_used": int, "batch_size": int,
    "curvature_convention": str}``.

    ``batch_size`` is the Python-loop granularity: how many rows are handed to
    :func:`chart_mean_curvature` per call. It is **numerically inert**. Peak memory is set by
    the module constant :data:`VMAP_CHUNK`, which fixes the autodiff batch width at 32 rows
    (78.6 MB for the Hessian at the sealed 02.2 architecture) no matter what ``batch_size``
    says; a larger ``batch_size`` only accumulates more assembled ``(rows, out_dim)`` output
    per call, which is two orders of magnitude smaller than the Hessian it never holds.

    Batching must never touch a value, and here it provably does not: every row's computation
    is independent (the einsums contract only over the ambient and chart indices, and
    ``torch.linalg.solve`` factorizes each point's ``g`` separately) and the autodiff width is
    fixed, so results are **bit-identical** across batch sizes.
    ``test_chart_curvature_field_reassembles_in_row_order`` pins that directly with
    ``torch.equal``. That test exists because a batching bug that reorders rows is invisible
    to every aggregate statistic this phase computes: Spearman and quantile-bin concordance
    would both simply drop, looking like a weak estimator rather than a bug. Without
    :data:`VMAP_CHUNK`'s fixed width the same assertion fails at ~5e-15 for a reason that has
    nothing to do with row order -- see that constant's docstring.
    """
    assert_c2_activation(model)
    _assert_float64(model, x.double() if x.dtype == torch.float64 else x)
    if batch_size < 1:
        raise ValueError(f"chart_curvature_field: batch_size must be >= 1; got {batch_size}.")

    with torch.no_grad():
        z = model.encode(x)
        z_charts = model.chart_coords(z)
        assignment = model.chart_probs(z).argmax(dim=1)

    n = x.shape[0]
    H_vec: Optional[torch.Tensor] = None
    H_norm = torch.empty(n, dtype=torch.float64)
    metric_cond = torch.empty(n, dtype=torch.float64)

    used = sorted(int(i) for i in torch.unique(assignment).tolist())
    for chart_idx in used:
        rows = torch.nonzero(assignment == chart_idx, as_tuple=False).squeeze(-1)
        coords = z_charts[rows, chart_idx, :]
        for start in range(0, rows.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            out = chart_mean_curvature(model, coords[sl], chart_idx)
            if H_vec is None:
                H_vec = torch.empty(n, out["H_vec"].shape[1], dtype=torch.float64)
            target = rows[sl]
            H_vec[target] = out["H_vec"]
            H_norm[target] = out["H_norm"]
            metric_cond[target] = out["metric_condition_number"]

    if H_vec is None:
        raise ValueError("chart_curvature_field: no rows to measure; x is empty.")

    return {
        "H_vec": H_vec,
        "H_norm": H_norm,
        "chart_assignment": assignment,
        "metric_condition_number": metric_cond,
        "n_charts_used": len(used),
        "batch_size": int(batch_size),
        "curvature_convention": CURVATURE_CONVENTION,
    }


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


# --- NON-GATING: randomized-probe convergence check on the exact path --------------------


def directional_second_derivative(
    f: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """``D^2 f(z)[v, v]`` for each row, by forward-over-forward autodiff:
    ``d/dt|_0 ( Df(z + t v) v )``. ``z``, ``v``: ``(batch, chart_dim)``; returns
    ``(batch, out_dim)``. Chunked at :data:`VMAP_CHUNK` for the same bit-reproducibility
    reason as :func:`chart_mean_curvature`.

    Note for anyone tempted to add antithetic sampling on top of this
    (``02.5-NOTE-randomized-trace.md`` Addendum A): ``B`` is a symmetric BILINEAR form, so
    ``B(-v, -v) = (-1)(-1) B(v, v) = B(v, v)`` **exactly**. The antithetic partner returns the
    identical value, not a negatively-correlated one; the pair is correlated at ``+1``, so
    averaging over ``{v, -v}`` has precisely the variance of the single sample ``v`` at twice
    the cost. Antithetic sampling helps estimators with an ODD-order dependence on the probe;
    a quadratic form is even. ``test_chart_curvature_antithetic_probes_are_exactly_redundant``
    pins this as bit-identity so the claim cannot rot into folklore.
    """
    if z.shape != v.shape or z.ndim != 2:
        raise ValueError(
            f"directional_second_derivative: z and v must share one (batch, chart_dim) shape; "
            f"got {tuple(z.shape)} and {tuple(v.shape)}."
        )

    def _one(zz: torch.Tensor, vv: torch.Tensor) -> torch.Tensor:
        def df(w: torch.Tensor) -> torch.Tensor:
            return jvp(f, (w,), (vv,))[1]

        return jvp(df, (zz,), (vv,))[1]

    batch = z.shape[0]
    parts = []
    for start in range(0, batch, VMAP_CHUNK):
        z_real, v_real = z[start : start + VMAP_CHUNK], v[start : start + VMAP_CHUNK]
        n_real = z_real.shape[0]
        parts.append(
            vmap(_one)(_pad_to_chunk(z_real), _pad_to_chunk(v_real))[:n_real].detach()
        )
    return torch.cat(parts, dim=0)


def randomized_trace_mean_curvature_nongating(
    model: Any, z_chart: torch.Tensor, chart_idx: int, n_probes: int, seed: int
) -> Dict[str, Any]:
    """A Hutchinson-style randomized estimate of the same ``H = tr_g(II)``.

    **NON-GATING, and the name says so at every call site on purpose.** This must never touch
    a gated number. ``02.5-NOTE-randomized-trace.md``'s "What 02.5-08 should do" demotes the
    randomized estimator from candidate to CONVERGENCE CHECK ON THE EXACT PATH: at
    ``d = 20`` the exact ``g``-trace is only 20 Hessian-vector products against ``K = 8``, a
    2.5x saving on a computation that was never the bottleneck, and the decoder arm's real
    advantage is statistical (it forms no neighbourhood, so ``r/R`` never enters its error),
    not computational. Its worth is the same as the sphere known-answer test's: agreement with
    the exact path is evidence the exact path is right.

    **Normalization -- the factor-of-``d`` trap, stated in full because the source material
    gets it the other way round.** With ``xi = g^{-1/2} eps`` and ``eps`` Rademacher,
    ``E[eps eps^T] = I`` so ``E[xi xi^T] = g^-1``, hence
    ``E[B(xi, xi)] = sum_jk g^jk II_jk = tr_g(II) = H``. Under this module's TRACE convention
    the estimator therefore carries **no ``1/d`` and no ``d``**. The alternative
    ``v = g^{-1/2} u`` with ``u`` uniform on ``S^{d-1}`` has ``E[u u^T] = I/d``, so under the
    trace convention *that* variant needs an EXPLICIT factor of ``d`` -- exactly inverted from
    the averaged-convention presentation, where Rademacher needs the explicit ``1/d`` and the
    sphere supplies it implicitly. Both pairings are internally correct; mixing any two of them
    costs a factor of ``d`` = 20. Rademacher is used here, and
    ``test_chart_curvature_randomized_trace_converges_to_exact`` is what proves the pairing.

    No antithetic path is offered; see :func:`directional_second_derivative` for why one would
    be exactly worthless here. Hutch++-style deflation is likewise not attempted: it estimates
    the scalar trace of a MATRIX, whereas ``II: T_zM x T_zM -> N_zM`` is VECTOR-valued, so at
    codimension greater than one there is no single scalar matrix whose trace it computes.
    Making it work would need a normal basis, and avoiding the construction of a normal basis
    at ``D = 768`` is precisely why the ``d x d`` solve exists.
    """
    assert_c2_activation(model)
    _assert_float64(model, z_chart)
    if n_probes < 1:
        raise ValueError(f"randomized_trace_mean_curvature_nongating: n_probes must be >= 1; got {n_probes}.")

    decode_one = chart_decoder_map(model, chart_idx)
    batch, chart_dim = z_chart.shape

    J = _chunked_jacobian(decode_one, z_chart)
    g = torch.einsum("boi,boj->bij", J, J)

    # g^{-1/2} by symmetric eigendecomposition: g is a Gram matrix, so eigh is the right
    # factorization and its eigenvalues are the conditioning diagnostic already reported by
    # chart_mean_curvature.
    evals, evecs = torch.linalg.eigh(g)
    if bool((evals <= 0).any()):
        raise ValueError(
            "randomized_trace_mean_curvature_nongating: the pullback metric is not positive "
            "definite at some point, so g^{-1/2} does not exist there. That is a "
            "non-immersion point; inspect chart_mean_curvature's metric_condition_number "
            "rather than regularizing it away."
        )
    g_inv_sqrt = torch.einsum("bij,bj,bkj->bik", evecs, evals.pow(-0.5), evecs)

    generator = torch.Generator().manual_seed(int(seed))
    acc = torch.zeros(batch, J.shape[1], dtype=z_chart.dtype)
    for _ in range(n_probes):
        eps = (
            torch.randint(0, 2, (batch, chart_dim), generator=generator, dtype=z_chart.dtype)
            * 2.0
            - 1.0
        )
        xi = torch.einsum("bij,bj->bi", g_inv_sqrt, eps)
        acc = acc + directional_second_derivative(decode_one, z_chart, xi)
    raw = acc / float(n_probes)

    alpha = torch.linalg.solve(g, torch.einsum("boi,bo->bi", J, raw).unsqueeze(-1)).squeeze(-1)
    H_vec = raw - torch.einsum("boi,bi->bo", J, alpha)

    return {
        "H_vec": H_vec,
        "H_norm": chart_mean_curvature_norm(H_vec),
        "n_probes": int(n_probes),
        "seed": int(seed),
        "probe_distribution": "rademacher",
        "gating": False,
        "curvature_convention": CURVATURE_CONVENTION,
    }


def randomized_trace_convergence_check(
    model: Any,
    z_chart: torch.Tensor,
    chart_idx: int,
    probe_counts: Sequence[int] = (4, 8, 16),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> Dict[str, Any]:
    """Report how fast :func:`randomized_trace_mean_curvature_nongating` converges on the
    exact :func:`chart_mean_curvature`, at each probe count, averaged over seeds.
    **Gates nothing** -- ``02.5-NOTE-randomized-trace.md`` asks for ``K = 4, 8, 16`` as
    recorded context and nothing more.

    Averaging over seeds is not cosmetic: at a single fixed seed the error sequence in ``K``
    is not monotone (measured during 02.5-08: ``K = 8`` landed worse than ``K = 4`` on one
    draw), which is the nature of a Monte-Carlo estimator rather than a defect. The
    ``1/sqrt(K)`` law is a statement about the average, so the average is what is reported.

    ``"mean_of_replicates_relative_error"`` pools every estimate computed, weighted by its
    probe count -- algebraically one estimator with ``sum_k K_k * n_seeds`` probes. Its error
    being far below the smallest single-run error is the unbiasedness evidence: it shows the
    spread at low ``K`` is variance around the right value, not a bias at the wrong scale.
    A mispaired probe normalization would be off by a constant factor of ``d`` and would NOT
    shrink with more probes, so this is the number that separates those two explanations.
    """
    exact = chart_mean_curvature(model, z_chart, chart_idx)["H_vec"]
    exact_norm = torch.linalg.norm(exact, dim=-1)

    median_rel: Dict[int, float] = {}
    pooled = torch.zeros_like(exact)
    pooled_weight = 0.0
    for n_probes in probe_counts:
        per_seed = []
        for seed in seeds:
            est = randomized_trace_mean_curvature_nongating(
                model, z_chart, chart_idx, n_probes, seed
            )["H_vec"]
            rel = torch.linalg.norm(est - exact, dim=-1) / exact_norm
            per_seed.append(float(rel.median()))
            pooled = pooled + est * float(n_probes)
            pooled_weight += float(n_probes)
        median_rel[int(n_probes)] = float(np.mean(per_seed))

    pooled_rel = torch.linalg.norm(pooled / pooled_weight - exact, dim=-1) / exact_norm

    return {
        "H_exact": exact,
        "median_relative_error": median_rel,
        "mean_of_replicates_relative_error": float(pooled_rel.median()),
        "probe_counts": tuple(int(k) for k in probe_counts),
        "seeds": tuple(int(s) for s in seeds),
        "gating": False,
        "curvature_convention": CURVATURE_CONVENTION,
    }
