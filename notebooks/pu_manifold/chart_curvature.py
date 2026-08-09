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

from typing import Any, Callable, Dict, Optional

import torch
from torch.func import hessian, jacrev, vmap

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
        if n_real < VMAP_CHUNK:
            # Pad up to the fixed width by repeating the last real row, then discard the
            # padding. See VMAP_CHUNK: a row's result does not depend on which rows share its
            # chunk, only on the chunk's width, so this is exact rather than approximate.
            chunk = torch.cat([real, real[-1:].repeat(VMAP_CHUNK - n_real, 1)], dim=0)
        else:
            chunk = real

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
