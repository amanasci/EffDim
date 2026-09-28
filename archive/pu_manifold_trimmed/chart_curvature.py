# Top-level definitions removed from notebooks/pu_manifold/chart_curvature.py in the paper-closure
# Stage 5: not reachable from any kept runner, generator or notebook. Verbatim, original order.


# --- removed from chart_curvature.py:64-72 ---
CURVATURE_MODES = ("reverse", "forward")
"""The two legal values of the ``mode`` keyword on :func:`chart_mean_curvature` and
:func:`chart_curvature_field`. ``"reverse"`` is the default and STAYS the default, before and
after the forward path's equivalence tests pass -- this is an explicit user instruction, not a
provisional state. The toggle exists so a forward-mode composition that turns out to be a bad
idea (an unimplemented ``vmap`` batching rule, a hidden numerical regression) can be abandoned
by simply never passing ``mode="forward"``, without touching a single line of the reverse path
every existing call site, the ``02.5-09`` notebook, and every sealed roll number depend on.
Flipping the default would silently change what every existing call site computes with."""


# --- removed from chart_curvature.py:183-209 ---
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


# --- removed from chart_curvature.py:236-237 ---
def _batched_eye(n: int, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(size, dtype=dtype, device=device).expand(n, size, size)


# --- removed from chart_curvature.py:250-258 ---
def _chunked_jacobian(
    decode_one: Callable[[torch.Tensor], torch.Tensor], z_chart: torch.Tensor
) -> torch.Tensor:
    """``(batch, out_dim, chart_dim)`` decoder Jacobian, taken at the fixed autodiff width."""
    parts = []
    for start in range(0, z_chart.shape[0], VMAP_CHUNK):
        real = z_chart[start : start + VMAP_CHUNK]
        parts.append(vmap(jacrev(decode_one))(_pad_to_chunk(real))[: real.shape[0]].detach())
    return torch.cat(parts, dim=0)


# --- removed from chart_curvature.py:261-261 ---
# --- the gating computation: exact g-trace of the normal-projected Hessian ----------------


# --- removed from chart_curvature.py:264-307 ---
def _jacobian_hessian(
    decode_one: Callable[[torch.Tensor], torch.Tensor], chunk: torch.Tensor, mode: str
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Dispatch the Jacobian/Hessian construction on ``mode``. This is the ONLY thing that
    branches on ``mode`` anywhere in this module -- everything downstream of the returned
    ``(J, Hess)`` pair (the ``g``-trace-first, ``d x d``-solve, normal-project block) is
    identical mathematics regardless of how the two tensors were produced, and stays
    byte-for-byte untouched by this function's existence.

    ``mode == "reverse"``: ``vmap(jacrev(decode_one))`` for the Jacobian and
    ``vmap(hessian(decode_one))`` -- ``hessian = jacfwd(jacrev(f))`` -- for the Hessian. Moved
    verbatim from this function's pre-03-05 inline call sites; this is the path every existing
    caller, the ``02.5-09`` notebook, and every sealed roll number depend on staying bit-exact.

    ``mode == "forward"``: ``vmap(jacfwd(decode_one))`` for the Jacobian (about ``d`` passes
    instead of reverse's ``D``), and ``vmap(jacfwd(jacfwd(decode_one)))`` for the Hessian.
    **Spike evidence (plan 03-05 Task 1):** ``vmap(jacfwd(jacfwd(decode_one)))`` was run
    against the real ``cae.ChartAutoEncoder`` chart-decoder architecture (``chart_dim=20,
    out_dim=768, hidden=[250,250,250], activation="silu"``, float64) and completed without
    raising, returning the expected ``(batch, out_dim, chart_dim, chart_dim)`` shape. This is
    therefore the PRIMARY forward-Hessian composition, not the documented
    ``jacfwd(jacrev(f))`` fallback -- the fallback was not needed. A single-chunk (32-row)
    Hessian at that architecture measured ~6.08s in reverse mode
    (``vmap(hessian(decode_one))``) versus ~0.26s in forward mode
    (``vmap(jacfwd(jacfwd(decode_one)))``), roughly a 23.6x wall-clock speedup -- well short of
    the ~38x operation-count ceiling (expected: PyTorch's forward-mode path is less optimized
    than its reverse path, and ``vmap`` over dual numbers carries its own constants) but a
    real, substantial, measured win. See the plan SUMMARY for the full spike transcript.

    Any other value: ``raise ValueError`` naming the offending string, in the
    refuse-and-name-the-fix style of :func:`_assert_float64` -- never a silent fall-through to
    a default.
    """
    if mode == "reverse":
        J = vmap(jacrev(decode_one))(chunk)
        Hess = vmap(hessian(decode_one))(chunk)
    elif mode == "forward":
        J = vmap(jacfwd(decode_one))(chunk)
        Hess = vmap(jacfwd(jacfwd(decode_one)))(chunk)
    else:
        raise ValueError(
            f"_jacobian_hessian: unknown mode {mode!r}; must be one of {CURVATURE_MODES}."
        )
    return J, Hess


# --- removed from chart_curvature.py:310-496 ---
def chart_mean_curvature(
    model: Any, z_chart: torch.Tensor, chart_idx: int, mode: str = "reverse"
) -> Dict[str, Any]:
    """Exact mean curvature vector of the manifold parameterized by chart ``chart_idx``'s
    decoder, at each chart coordinate in ``z_chart``.

    ``z_chart``: ``(batch, chart_dim)`` coordinates **within one chart's own coordinate
    space** -- ``model.chart_coords(z)[:, chart_idx, :]``, NOT the initial-encoder embedding
    ``z``. See :func:`chart_decoder_map` for why that distinction is the whole point.

    ``mode``: which differentiation path constructs the Jacobian and Hessian -- see
    :data:`CURVATURE_MODES` and :func:`_jacobian_hessian`. Defaults to, and stays defaulted
    to, ``"reverse"``: that is the path every existing call site, the ``02.5-09`` notebook,
    and every sealed roll number were measured against. ``"forward"`` is opt-in only, proved
    equal to ``"reverse"`` at float64 round-off by ``test_chart_curvature_forward_mode_matches_reverse_to_float64_round_off``,
    and only ever selected by an explicit caller. Only the Jacobian/Hessian construction
    branches on ``mode`` -- everything below it, including this docstring's mathematics, is
    identical for both.

    Returns a dict, not a bare tensor, because three separate things must be visible to the
    caller and the plan requires all of them: the curvature itself, the shapes the
    ``torch.func`` transform composition actually produced (Pitfall 5), and the conditioning
    of the pullback metric (threat T-02.5-20).

      ``"H_vec"``                    ``(batch, out_dim)``  mean curvature vectors, ``tr_g(II)``
      ``"H_norm"``                   ``(batch,)``          ``||H||``, the only reportable scalar
      ``"metric_condition_number"``  ``(batch,)``          ``cond(g)`` per point
      ``"lambda_min"``               ``(batch,)``          smallest eigenvalue of ``g`` (D-15, CURV-04)
      ``"lambda_max"``               ``(batch,)``          largest eigenvalue of ``g`` (D-15, CURV-04)
      ``"det_g"``                    ``(batch,)``          ``det(g)`` (D-15, CURV-04)
      ``"log10_det_g"``              ``(batch,)``          ``sum(log10(eig(g)))`` (D-15, CURV-04)
      ``"jacobian_shape"``           tuple                 as produced, for the shape assertion
      ``"hessian_shape"``            tuple                 as produced, for the shape assertion
      ``"chart_idx"``, ``"activation"``, ``"mode"``        provenance

    D-15 (CURV-04, reopened): ``metric_condition_number`` alone is scale-invariant and was
    measured ranking two uniformly-collapsed seeds (metric spectrum ``~1e-07`` everywhere)
    *ahead of* the only fit whose metric had a real absolute scale. The four fields above are
    derived from ONE extra ``torch.linalg.eigvalsh(g)`` decomposition per chunk, alongside the
    pre-existing metric-conditioning call below, which is retained byte-for-byte unchanged so
    the reverse path stays bit-identical (D-15's bit-identity requirement).

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
    lambda_min_parts = []
    lambda_max_parts = []
    det_g_parts = []
    log10_det_g_parts = []
    for start in range(0, batch, VMAP_CHUNK):
        real = z_chart[start : start + VMAP_CHUNK]
        n_real = real.shape[0]
        # Pad a short final chunk up to the fixed width and discard the padding afterwards.
        # See VMAP_CHUNK: a row's result does not depend on which rows share its chunk, only
        # on the chunk's width, so this is exact rather than approximate.
        chunk = _pad_to_chunk(real)

        J, Hess = _jacobian_hessian(decode_one, chunk, mode)
        if tuple(J.shape) != (VMAP_CHUNK, out_dim, chart_dim):
            raise ValueError(
                f"chart_mean_curvature: expected Jacobian of shape "
                f"{(VMAP_CHUNK, out_dim, chart_dim)}, got {tuple(J.shape)}. torch.func's "
                f"transform composition returned something other than a per-point Jacobian "
                f"(RESEARCH Pitfall 5)."
            )

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

        # D-15 / CURV-04: ONE extra eigendecomposition, reused for lambda_min, lambda_max,
        # det_g and log10_det_g -- never a second eigendecomposition beside this one, and
        # the cond_parts line immediately above is untouched (D-15's bit-identity
        # requirement: H_vec, H_norm and metric_condition_number must not move).
        eigs = torch.linalg.eigvalsh(g)  # (chunk, chart_dim), ascending; g is symmetric PSD
        lambda_min_parts.append(eigs[..., 0][:n_real].detach())
        lambda_max_parts.append(eigs[..., -1][:n_real].detach())
        det_g_parts.append(torch.linalg.det(g)[:n_real].detach())
        log10_det_g_parts.append(eigs.log10().sum(dim=-1)[:n_real].detach())

    H_vec = torch.cat(H_parts, dim=0)
    metric_cond = torch.cat(cond_parts, dim=0)
    lambda_min = torch.cat(lambda_min_parts, dim=0)
    lambda_max = torch.cat(lambda_max_parts, dim=0)
    det_g = torch.cat(det_g_parts, dim=0)
    log10_det_g = torch.cat(log10_det_g_parts, dim=0)

    return {
        "H_vec": H_vec,
        "H_norm": chart_mean_curvature_norm(H_vec),
        "metric_condition_number": metric_cond,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "det_g": det_g,
        "log10_det_g": log10_det_g,
        "jacobian_shape": (batch, out_dim, chart_dim),
        "hessian_shape": (batch, out_dim, chart_dim, chart_dim),
        "chart_idx": int(chart_idx),
        "activation": activation,
        "curvature_convention": CURVATURE_CONVENTION,
        "mode": mode,
    }


# --- removed from chart_curvature.py:499-510 ---
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


# --- removed from chart_curvature.py:513-605 ---
def chart_curvature_field(
    model: Any, x: torch.Tensor, batch_size: int = 32, mode: str = "reverse"
) -> Dict[str, Any]:
    """Per-point mean curvature over an ambient point cloud, each row measured in the chart
    the model itself assigns to it.

    ``x``: ``(n, in_dim)`` ambient rows. Encodes to ``z``, takes ``chart_coords(z)`` and
    ``chart_probs(z).argmax(dim=1)``, then for each chart index calls
    :func:`chart_mean_curvature` on exactly the rows assigned to that chart, using that
    chart's own coordinates, and reassembles the result in the original row order.

    ``mode``: threaded verbatim to every :func:`chart_mean_curvature` call, one per chart. See
    that function's docstring and :data:`CURVATURE_MODES` -- ``"reverse"`` is the default and
    stays the default.

    Returns ``{"H_vec": (n, out_dim), "H_norm": (n,), "chart_assignment": (n,),
    "metric_condition_number": (n,), "lambda_min": (n,), "lambda_max": (n,), "det_g": (n,),
    "log10_det_g": (n,), "n_charts_used": int, "batch_size": int,
    "curvature_convention": str}``. The four fields after ``metric_condition_number`` are
    D-15 / CURV-04's absolute-scale diagnostics -- see :func:`chart_mean_curvature`'s
    docstring for what they are and why ``metric_condition_number`` alone is insufficient.

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
    _assert_float64(model, x)
    if batch_size < 1:
        raise ValueError(f"chart_curvature_field: batch_size must be >= 1; got {batch_size}.")

    with torch.no_grad():
        z = model.encode(x)
        z_charts = model.chart_coords(z)
        assignment = model.chart_probs(z).argmax(dim=1)

    n = x.shape[0]
    H_vec: Optional[torch.Tensor] = None
    H_norm = torch.empty(n, dtype=torch.float64, device=x.device)
    metric_cond = torch.empty(n, dtype=torch.float64, device=x.device)
    lambda_min = torch.empty(n, dtype=torch.float64, device=x.device)
    lambda_max = torch.empty(n, dtype=torch.float64, device=x.device)
    det_g = torch.empty(n, dtype=torch.float64, device=x.device)
    log10_det_g = torch.empty(n, dtype=torch.float64, device=x.device)

    used = sorted(int(i) for i in torch.unique(assignment).tolist())
    for chart_idx in used:
        rows = torch.nonzero(assignment == chart_idx, as_tuple=False).squeeze(-1)
        coords = z_charts[rows, chart_idx, :]
        for start in range(0, rows.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            out = chart_mean_curvature(model, coords[sl], chart_idx, mode=mode)
            if H_vec is None:
                H_vec = torch.empty(n, out["H_vec"].shape[1], dtype=torch.float64, device=x.device)
            target = rows[sl]
            H_vec[target] = out["H_vec"]
            H_norm[target] = out["H_norm"]
            metric_cond[target] = out["metric_condition_number"]
            lambda_min[target] = out["lambda_min"]
            lambda_max[target] = out["lambda_max"]
            det_g[target] = out["det_g"]
            log10_det_g[target] = out["log10_det_g"]

    if H_vec is None:
        raise ValueError("chart_curvature_field: no rows to measure; x is empty.")

    return {
        "H_vec": H_vec,
        "H_norm": H_norm,
        "chart_assignment": assignment,
        "metric_condition_number": metric_cond,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "det_g": det_g,
        "log10_det_g": log10_det_g,
        "n_charts_used": len(used),
        "batch_size": int(batch_size),
        "curvature_convention": CURVATURE_CONVENTION,
    }


# --- removed from chart_curvature.py:726-726 ---
# --- NON-GATING: randomized-probe convergence check on the exact path --------------------


# --- removed from chart_curvature.py:729-766 ---
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


# --- removed from chart_curvature.py:769-852 ---
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

    # The generator stays CPU-only (its seed->draw mapping is independent of device, same
    # reasoning as cae.farthest_point_sample); only the drawn probe vector is moved to
    # z_chart's device before it meets any device-resident tensor.
    generator = torch.Generator().manual_seed(int(seed))
    acc = torch.zeros(batch, J.shape[1], dtype=z_chart.dtype, device=z_chart.device)
    for _ in range(n_probes):
        eps = (
            torch.randint(0, 2, (batch, chart_dim), generator=generator, dtype=z_chart.dtype)
            * 2.0
            - 1.0
        ).to(z_chart.device)
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


# --- removed from chart_curvature.py:855-907 ---
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
