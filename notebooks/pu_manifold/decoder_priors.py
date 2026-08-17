"""
An opt-in, default-off isometry prior on the CAE chart decoder's Jacobian -- a training-time
regularizer that constrains the SAME map ``chart_curvature.chart_decoder_map`` differentiates,
which is exactly the object nothing in ``cae.train_cae``'s objective currently constrains.

**Why this module exists.** ``cae.train_cae`` regularizes ``model.chart_encoders`` through
``cae.lipschitz_penalty`` (``cae.py`` line 483). ``chart_curvature.chart_decoder_map`` composes
``model.chart_decoders[i]`` with ``model.embedding_decoder`` (``chart_curvature.py`` line 183).
Those two object sets share no parameter. ``cae.chart_loss`` constrains the decoder only through
reconstruction, a C0 quantity. Nothing in the training objective constrains the decoder's
derivatives at any order.

**The measured consequence.** On the corrected PU grid (``notebooks/.cache/03_curvature_field_pu.jsonl``,
15-record grid, 2026-08-15), ``cond(g)`` reaches ``4.886e7`` at ``n_charts=4`` -- against the
Swiss roll's ``1.4``-``122`` on the identical machinery. That destroys roughly seven digits of
float64 precision in the ``g^-1`` contraction inside ``H = sum_jk g^jk II_jk``
(``chart_curvature.chart_mean_curvature``): a near-singular pullback metric at an
effectively-non-immersion point, produced by a decoder whose derivatives were never asked to be
well-conditioned.

**What the prior is.** First order, and it constrains only the PARAMETERIZATION: it penalizes
``g = J^T J`` (the pullback metric, ``J`` the decoder's chart-space Jacobian) directly, never the
decoder's second derivatives and never the curvature the model is meant to measure.

**Why a curvature or Hessian-norm penalty must never be substituted for it -- the single most
important paragraph in this file.** Penalizing ``||D^2 F||`` or ``||H||`` biases the estimand: it
pushes the decoder toward flat surfaces and then measures flatness, making the curvature field a
function of the regularizer weight rather than of the data. The isometry prior is safe precisely
because ``H = tr_g(II)`` is parameterization-invariant -- a property of the decoder's IMAGE alone
-- so constraining the parameterization cannot move it. If a second-order term is ever added to
this file it must be the TANGENTIAL part ``P_T D^2 F`` only, which is pure Christoffel /
parameterization content; it must never be the normal part ``P_N D^2 F = II``, which is the
geometry itself.

**That second-order term now exists** (mode ``"christoffel"``, added 2026-08-17 on the
developer's instruction) and it obeys that rule exactly: :func:`christoffel_deviation` returns
``Gamma^k_ij = (g^{-1} J^T D^2 F)^k_ij``, the tangential part alone, and the normal part is
never formed. So the module DOES now differentiate the decoder twice, in exactly one place
(:func:`chart_decoder_hessian`), whose output exactly one consumer may read. The invariance
argument is unchanged: ``H = tr_g(II)`` is a property of the decoder's image, so constraining
the parameterization -- at first order via ``g``, or at second order via ``Gamma`` -- cannot
move it.

Like ``cae.py`` and ``chart_curvature.py``, this module imports ``torch`` at module level, so it
is deliberately NOT re-exported from ``pu_manifold/__init__.py`` (do not touch that file).
"""

from contextlib import contextmanager
from typing import Any, Dict

import torch
from torch.func import jacfwd, vmap

from . import cae
from . import chart_curvature

PRIOR_MODES = ("isometry", "conformal", "christoffel")
"""The legal values of the ``mode`` keyword on :func:`decoder_prior_active`.

FIRST ORDER, on the pullback metric ``g = J^T J``:

* ``"isometry"`` penalizes ``||g - I||_F^2`` directly.
* ``"conformal"`` penalizes only the DEVIATION FROM A UNIFORM SCALE, ``||g - c I||_F^2`` with
  ``c = tr(g)/d``.

Both were pre-declared before any spike number existed. ``cae.ChartEncoder`` ends in a sigmoid,
so chart coordinates live in the open unit cube ``(0,1)^d``: a decoder whose domain has extent 1
and whose image must span a manifold of extent ``L`` needs ``||J|| ~ L``, i.e. ``g ~ L^2 I``, not
``g = I``. The strict isometry penalty therefore contains a SCALE term that fights reconstruction
directly, while the measured pathology -- ``cond(g) = lambda_max / lambda_min`` -- is
scale-invariant and is exactly what the conformal variant targets. If the isometry arm fails its
mechanism check (``cond(g)`` does not fall as the weight rises), the conformal arm is the
pre-declared branch, not a post-hoc axis switch.

SECOND ORDER, on the Christoffel symbols:

* ``"christoffel"`` penalizes ``||Gamma||_F^2``, the TANGENTIAL part of ``D^2 F`` only. See
  :func:`christoffel_deviation`; it is listed in :data:`SECOND_ORDER_MODES` and is the only
  second-order quantity this module will ever admit.

:func:`metric_deviation` accepts the first-order modes ONLY -- it is a function of ``g`` alone
and structurally cannot compute a second-order quantity."""

FIRST_ORDER_MODES = ("isometry", "conformal")
"""Modes that are a function of ``g = J^T J`` alone, and so are computable by
:func:`metric_deviation`."""

SECOND_ORDER_MODES = ("christoffel",)
"""Modes that differentiate the decoder TWICE. Listed separately because they cost
``chart_dim`` times more than the first-order modes and because exactly one second-order
quantity is ever admissible here -- see :func:`christoffel_deviation`."""

CHRISTOFFEL_MAX_ROWS_PER_CHART = 8
"""Rows per chart per batch on which the second-order penalty is evaluated.

Not a tuning knob for the penalty's strength -- that is ``weight`` -- but a compute bound.
``vmap(jacfwd(jacfwd(decode_one)))`` costs roughly ``chart_dim`` times a Jacobian, so at
``chart_dim = 20`` the second-order term is ~20x the first-order prior, which itself roughly
tripled per-epoch cost. Subsampling rows keeps the term affordable; the penalty is a batch
mean either way, so a subsample estimates the same quantity with more variance rather than a
different quantity. Rows are taken as the first ``n`` of each chart's assignment, which is
already a shuffled minibatch order."""


def metric_deviation(g: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-point deviation of a batch of pullback metrics ``g`` -- ``(batch, d, d)`` -- from the
    target ``mode`` describes. Pure tensor function: no model, no dtype guard, so its answers are
    exactly known and independently checkable.

    ``mode == "isometry"``: ``||g - I||_F^2`` per point.

    ``mode == "conformal"``: ``c = trace(g) / d`` per point (the mean eigenvalue of a Gram
    matrix), then ``||g - c I||_F^2``. A uniformly scaled metric (``g = s I`` for any ``s``) has
    zero conformal deviation by construction -- that is the whole distinction between the two
    modes.

    Returns ``(batch,)``. Raises ``ValueError`` naming the offending string for any ``mode`` not
    in :data:`PRIOR_MODES` -- never a silent fall-through to a default.
    """
    if mode in SECOND_ORDER_MODES:
        raise ValueError(
            f"metric_deviation: mode {mode!r} is second-order and is a function of the "
            "decoder's Hessian, not of g alone -- this function cannot compute it. Use "
            "christoffel_deviation(J, Hess). Refusing rather than falling through to the "
            "conformal branch, which would silently compute a different quantity."
        )
    if mode not in FIRST_ORDER_MODES:
        raise ValueError(
            f"metric_deviation: unknown mode {mode!r}; must be one of {FIRST_ORDER_MODES}."
        )
    d = g.shape[-1]
    eye = torch.eye(d, dtype=g.dtype, device=g.device)
    if mode == "isometry":
        diff = g - eye
    else:  # "conformal"
        c = torch.diagonal(g, dim1=-2, dim2=-1).sum(dim=-1) / d
        diff = g - c.unsqueeze(-1).unsqueeze(-1) * eye
    return (diff**2).sum(dim=(-2, -1))


def chart_decoder_jacobian(model: Any, z_chart: torch.Tensor, chart_idx: int) -> torch.Tensor:
    """``(batch, out_dim, chart_dim)`` Jacobian of chart ``chart_idx``'s decoder map, taken at
    ``z_chart``, DIFFERENTIABLE with respect to the model's parameters -- this is a training-time
    quantity, not a measurement.

    Obtains the map via ``chart_curvature.chart_decoder_map(model, chart_idx)`` -- reusing the
    two-hop decode (chart decoder then the shared embedding decoder) rather than re-deriving it --
    and applies ``vmap(jacfwd(decode_one))``. Deliberately does NOT call
    ``chart_curvature._chunked_jacobian``: that helper ``.detach()``es its result, which is
    correct for measurement and fatal for a training term that must carry a live autograd graph
    back to ``chart_decoders`` and ``embedding_decoder``.

    Deliberately does NOT call ``chart_mean_curvature`` or anything else in ``chart_curvature``
    that runs ``_assert_float64`` -- the model is float32 during training, and that guard would
    (correctly, for its own purpose) refuse. This is a first-order quantity, so float32 is fine
    here.

    **Forward mode was verified to carry gradient back through both decoder stages** (measured
    directly during this plan's Task 1: ``vmap(jacfwd(decode_one))(z).sum().backward()``
    populates non-zero ``.grad`` on both a ``chart_decoders[i]`` parameter and an
    ``embedding_decoder`` parameter, for a real two-layer-linear composition). ``jacfwd`` is
    therefore used directly; the documented ``jacrev`` fallback (``out_dim`` is small on the
    Swiss roll, so reverse mode would not be expensive) was not needed.
    """
    decode_one = chart_curvature.chart_decoder_map(model, chart_idx)
    return vmap(jacfwd(decode_one))(z_chart)


def chart_decoder_hessian(model: Any, z_chart: torch.Tensor, chart_idx: int) -> torch.Tensor:
    """``(batch, out_dim, chart_dim, chart_dim)`` second derivative of chart ``chart_idx``'s
    decoder map at ``z_chart``, DIFFERENTIABLE with respect to the model's parameters.

    The training-time counterpart of the measurement path in ``chart_curvature``. It
    deliberately does NOT call ``chart_curvature._jacobian_hessian``: that path is built for
    measurement, runs ``_assert_float64`` (the model is float32 during training, and that guard
    would correctly refuse) and detaches, which is fatal for a term that must carry a live
    autograd graph back to ``chart_decoders`` and ``embedding_decoder``.

    ``vmap(jacfwd(jacfwd(decode_one)))`` rather than ``hessian`` = ``jacfwd(jacrev(...))``:
    forward-over-forward was already verified in this module to carry gradient back through
    both decoder stages (see :func:`chart_decoder_jacobian`), and reverse mode under ``vmap``
    inside a training loop is the path with more ways to go wrong.

    **This is the only place in this module that differentiates the decoder twice, and its
    output may only ever be consumed by :func:`christoffel_deviation`.** The module docstring's
    single most important paragraph governs: the normal part of this tensor IS the second
    fundamental form, and penalizing it would bias the estimand.
    """
    decode_one = chart_curvature.chart_decoder_map(model, chart_idx)
    return vmap(jacfwd(jacfwd(decode_one)))(z_chart)


def christoffel_deviation(J: torch.Tensor, Hess: torch.Tensor) -> torch.Tensor:
    """Per-point squared Frobenius norm of the Christoffel symbols, ``(batch,)``.

    ``J``: ``(batch, out_dim, chart_dim)``. ``Hess``: ``(batch, out_dim, chart_dim, chart_dim)``.

    **The decomposition this rests on, stated because the whole safety argument is here.** The
    decoder's second derivative splits orthogonally at each point into

        ``D^2 F  =  P_T D^2 F  +  P_N D^2 F``

    where ``P_T`` projects onto the tangent space spanned by ``J``'s columns and ``P_N`` onto its
    orthogonal complement. The NORMAL part is exactly the second fundamental form ``II``, whose
    ``g``-trace is the mean curvature this milestone measures. The TANGENTIAL part is exactly the
    Christoffel symbols ``Gamma^k_ij`` of the induced connection in these coordinates -- pure
    parameterization content, carrying no information about the decoder's image.

    This function returns only the tangential part, in the coordinate form

        ``Gamma^k_ij = (g^{-1} J^T D^2 F)^k_ij``,  ``g = J^T J``

    so a penalty built on it constrains HOW the chart parameterizes the surface and provably
    cannot move ``H = tr_g(II)``, which is invariant to reparameterization. A flat plane given
    wildly non-uniform coordinates has large ``Gamma`` and zero curvature; a sphere in normal
    coordinates at the origin has zero ``Gamma`` and large curvature. Both are exercised as
    known answers in this module's tests.

    ``torch.linalg.solve`` on ``g`` rather than an explicit inverse, matching
    ``chart_curvature.chart_mean_curvature``'s own treatment of the same matrix.
    """
    if J.ndim != 3:
        raise ValueError(f"christoffel_deviation: J must be (batch, out_dim, chart_dim); got {tuple(J.shape)}.")
    if Hess.ndim != 4:
        raise ValueError(
            f"christoffel_deviation: Hess must be (batch, out_dim, chart_dim, chart_dim); "
            f"got {tuple(Hess.shape)}."
        )
    batch, _, d = J.shape
    g = torch.einsum("boi,boj->bij", J, J)
    # J^T applied to the ambient index of the Hessian: (batch, d, d, d)
    jt_hess = torch.einsum("bok,boij->bkij", J, Hess)
    gamma = torch.linalg.solve(g, jt_hess.reshape(batch, d, d * d)).reshape(batch, d, d, d)
    return gamma.pow(2).sum(dim=(1, 2, 3))


def christoffel_penalty(
    model: Any,
    x: torch.Tensor,
    weight: float,
    max_rows_per_chart: int = CHRISTOFFEL_MAX_ROWS_PER_CHART,
) -> torch.Tensor:
    """Differentiable scalar: ``weight`` times the batch mean of :func:`christoffel_deviation`,
    evaluated at the coordinate and chart the model itself assigns each row.

    Structurally identical to :func:`isometry_penalty` -- same ``no_grad`` assignment step, same
    per-chart loop, same gradient path into ``chart_decoders`` and ``embedding_decoder`` only --
    and differs solely in computing a second-order quantity and in subsampling rows for cost
    (see :data:`CHRISTOFFEL_MAX_ROWS_PER_CHART`).

    This is the SECOND-ORDER prior. It is safe to use while measuring curvature precisely
    because it penalizes the tangential part of ``D^2 F`` and nothing else; a penalty on
    ``||D^2 F||`` or on ``||H||`` would bias the estimand and must never be substituted.
    """
    with torch.no_grad():
        z = model.encode(x)
        z_charts = model.chart_coords(z)
        assignment = model.chart_probs(z).argmax(dim=1)

    deviations = []
    used = sorted(int(i) for i in torch.unique(assignment).tolist())
    for chart_idx in used:
        rows = torch.nonzero(assignment == chart_idx, as_tuple=False).squeeze(-1)
        if max_rows_per_chart > 0:
            rows = rows[:max_rows_per_chart]
        coords = z_charts[rows, chart_idx, :]
        J = chart_decoder_jacobian(model, coords, chart_idx)
        Hess = chart_decoder_hessian(model, coords, chart_idx)
        deviations.append(christoffel_deviation(J, Hess))

    all_deviation = torch.cat(deviations, dim=0)
    return weight * all_deviation.mean()


def isometry_penalty(
    model: Any, x: torch.Tensor, weight: float, mode: str = "isometry"
) -> torch.Tensor:
    """Differentiable scalar tensor: ``weight`` times the batch mean of
    :func:`metric_deviation` over ``g = J^T J``, ``J`` the chart-decoder Jacobian of every row at
    the coordinate and chart the model itself assigns it.

    Encodes ``x``, takes ``model.chart_coords(z)`` and the argmax chart from
    ``model.chart_probs(z)`` -- the SAME assignment ``chart_curvature.chart_curvature_field``
    uses (``chart_curvature.py`` lines 522-525, mirrored here), so the prior constrains the
    decoder at the coordinates curvature will actually be measured at. That whole computation --
    encode, chart-coordinate, and the inherently non-differentiable argmax assignment -- is taken
    under ``torch.no_grad()``; the comment at the call site states this is deliberate. The
    gradient this function returns flows only through :func:`chart_decoder_jacobian`'s use of the
    (now-constant) coordinates as the Jacobian's evaluation point, into ``chart_decoders`` and
    ``embedding_decoder`` -- never into ``chart_encoders`` or ``chart_predictor``, which is the
    whole point: this prior constrains the decoder, not the encoder side ``cae.lipschitz_penalty``
    already regularizes.

    Loops over the charts actually present in the batch (mirroring
    ``chart_curvature.chart_curvature_field``'s own loop), forms ``g`` by
    ``torch.einsum("boi,boj->bij", J, J)`` -- the same contraction
    ``chart_curvature.chart_mean_curvature`` uses -- and returns ``weight`` times the batch mean
    of ``metric_deviation(g, mode)`` pooled across every chart's rows. Returns a scalar tensor the
    optimizer can trade off against reconstruction, exactly as ``cae.lipschitz_penalty`` does.
    """
    # Deliberately no_grad: the encode -> chart_coords -> argmax assignment is inherently
    # non-differentiable in its last step (argmax), and this prior constrains only the decoder
    # side, so the coordinates it evaluates the decoder Jacobian at are treated as constants.
    with torch.no_grad():
        z = model.encode(x)
        z_charts = model.chart_coords(z)
        assignment = model.chart_probs(z).argmax(dim=1)

    deviations = []
    used = sorted(int(i) for i in torch.unique(assignment).tolist())
    for chart_idx in used:
        rows = torch.nonzero(assignment == chart_idx, as_tuple=False).squeeze(-1)
        coords = z_charts[rows, chart_idx, :]
        J = chart_decoder_jacobian(model, coords, chart_idx)
        g = torch.einsum("boi,boj->bij", J, J)
        deviations.append(metric_deviation(g, mode))

    all_deviation = torch.cat(deviations, dim=0)
    return weight * all_deviation.mean()


@contextmanager
def decoder_prior_active(model: Any, weight: float, mode: str = "isometry"):
    """Scoped, restore-on-exit shim that installs the isometry prior into ``cae.train_cae``'s
    main loop WITHOUT editing ``cae.py``.

    ``cae.train_cae``'s main loop resolves ``chart_loss`` from ``cae``'s module globals at call
    time (``cae.py`` line 479, a bare-name call inside the function whose ``__globals__`` is
    ``cae.__dict__``). This context manager rebinds ``cae.chart_loss`` to a wrapper that calls the
    ORIGINAL ``chart_loss``, copies the returned dict, adds :func:`isometry_penalty` to ``total``
    only (leaving ``recon`` and ``xent`` at their base values so ``train_cae``'s history keeps
    reporting the reconstruction term unpolluted), and restores the original binding in a
    ``finally`` block -- including on exit by exception.

    This is chosen over forking ``train_cae`` into a duplicate loop: the loop that runs must be
    the one that will actually run, and ``cae.py`` is the sealed 02.2 architecture with
    RNG-order-dependent anchors -- a fork risks an unfaithful translation, a failure mode this
    project has already paid for once (see this module's own module docstring).

    At ``weight == 0.0`` this installs NOTHING AT ALL -- it yields immediately without touching
    ``cae.chart_loss`` -- so the zero-weight arm is structurally, not merely numerically, the
    untouched code path.
    """
    if mode not in PRIOR_MODES:
        raise ValueError(f"decoder_prior_active: mode must be one of {PRIOR_MODES}; got {mode!r}.")
    if weight == 0.0:
        yield
        return

    original_chart_loss = cae.chart_loss

    def _patched_chart_loss(
        x: torch.Tensor, y_charts: torch.Tensor, p: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        result = dict(original_chart_loss(x, y_charts, p))
        if mode in SECOND_ORDER_MODES:
            penalty = christoffel_penalty(model, x, weight)
        else:
            penalty = isometry_penalty(model, x, weight, mode)
        result["total"] = result["total"] + penalty
        return result

    cae.chart_loss = _patched_chart_loss
    try:
        yield
    finally:
        cae.chart_loss = original_chart_loss
