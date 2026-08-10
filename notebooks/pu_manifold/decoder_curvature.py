"""
Phase 02.6 decoder-substrate screening: EXACT mean curvature through a decoder that has no
chart routing, computed by ``torch.func`` autodiff. This module is ``chart_curvature.py``
with the ``chart_decoders[chart_idx]`` two-hop composition removed -- a strict
simplification of already-reviewed code, not a new derivation. Where
``chart_curvature.chart_decoder_map`` composes a per-chart decoder and the shared embedding
decoder through a chart index, both free candidate substrates screened by Phase 02.6 (a
plain autoencoder, and a ``PlainAutoEncoder`` trained under ``topoae.train_topoae``) decode
through ONE smooth MLP end to end and have no chart index at all.

Tensors in, tensors and dicts out -- no file I/O, no cache handling; the runners under
``notebooks/diagnostics/`` own paths and cache stems.

Like ``cae.py``, ``chart_curvature.py`` and ``curvature_probe.py``'s torch-dependent
siblings, this module imports ``torch`` at module level. For the same reason those modules
are excluded from ``pu_manifold/__init__.py``'s eager imports (so Phase-1-only callers do
not need torch installed to import the package), this module is deliberately NOT
re-exported there either.

Curvature convention: every quantity here is the UNNORMALIZED trace ``H = tr_g(II)``,
identical to ``chart_curvature.CURVATURE_CONVENTION`` and ``curvature_probe.CURVATURE_CONVENTION``.
This module declares its own ``CURVATURE_CONVENTION`` constant (rather than merely
importing the sealed one) and asserts at import time that all three agree, so a future
edit to either sealed module that silently drifted the convention would break this module's
import rather than propagate a factor-of-``d`` error downstream.

**The critical finding this module exists to fix (see 02.6-01-PLAN.md
<critical_finding_from_planning>).** ``chart_curvature.assert_c2_activation`` requires a
``model.activation`` attribute. ``cae.ChartAutoEncoder.__init__`` sets ``self.activation``;
``cae.PlainAutoEncoder.__init__`` does NOT -- verified during planning: measured
``has .activation: False`` and the sealed guard raises on every ``PlainAutoEncoder``,
silu or otherwise. :func:`assert_c2_decoder` below is the fix: it delegates to the sealed
guard when an ``activation`` attribute is present (so CAE-shaped and duck-typed models keep
the sealed behaviour exactly), and otherwise inspects the decoder's own activation
submodules directly, reaching the same sealed ``ZERO_SECOND_DERIVATIVE_ACTIVATIONS``
frozenset by structural inspection rather than a recorded attribute.
"""

from typing import Any, Callable, Dict

import numpy as np
import torch
from torch.func import hessian, jacrev, vmap

from . import curvature_probe
from .chart_curvature import (
    CURVATURE_CONVENTION as _CHART_CURVATURE_CONVENTION,
    VMAP_CHUNK,
    ZERO_SECOND_DERIVATIVE_ACTIVATIONS,
    _assert_float64,
    _pad_to_chunk,
    assert_c2_activation,
)

CURVATURE_CONVENTION = "trace"
"""``H = tr_g(II)``, the unnormalized ``g``-trace of the second fundamental form -- never
the averaged ``(1/d) tr_g(II)``. Declared here (rather than merely imported) so that a
drift in either sealed module's own convention constant breaks this module's import instead
of silently propagating a factor-of-``d`` error into a decoder-substrate screening run."""

if CURVATURE_CONVENTION != _CHART_CURVATURE_CONVENTION:
    raise ValueError(
        f"decoder_curvature.CURVATURE_CONVENTION={CURVATURE_CONVENTION!r} disagrees with "
        f"chart_curvature.CURVATURE_CONVENTION={_CHART_CURVATURE_CONVENTION!r}. Two modules "
        f"computing the same mathematics must never silently diverge on which convention "
        f"they report under."
    )
if CURVATURE_CONVENTION != curvature_probe.CURVATURE_CONVENTION:
    raise ValueError(
        f"decoder_curvature.CURVATURE_CONVENTION={CURVATURE_CONVENTION!r} disagrees with "
        f"curvature_probe.CURVATURE_CONVENTION={curvature_probe.CURVATURE_CONVENTION!r}. Two "
        f"modules computing the same mathematics must never silently diverge on which "
        f"convention they report under."
    )


# --- C2 smoothness guard, reaching cae.PlainAutoEncoder's missing .activation ------------


def assert_c2_decoder(model: Any) -> str:
    """Raise ``ValueError`` unless ``model``'s decoder is built entirely from C2-smooth
    activations, and return a description of what was checked on success.

    Two branches:

    1. ``model`` has an ``.activation`` attribute (the ``cae.ChartAutoEncoder`` shape, and
       any duck-typed fixture that mimics it): delegate to the sealed
       ``chart_curvature.assert_c2_activation(model)`` and return its result UNCHANGED, so
       CAE-shaped models keep the sealed behaviour exactly -- this module never re-derives
       that check.
    2. ``model`` has no ``.activation`` attribute (``cae.PlainAutoEncoder``'s actual shape,
       verified during planning): inspect the decoder's own modules instead.
       ``getattr(model, "decoder", model)`` is required to expose ``.modules()`` -- raising
       ``ValueError`` naming the model type if it does not, the same "refuse to
       differentiate an uninspectable decoder twice" posture the sealed guard takes for a
       missing attribute. Every submodule that is not the container itself and not
       ``torch.nn.Linear`` contributes its lower-cased class name to a collected set; if
       any collected name is in the sealed ``ZERO_SECOND_DERIVATIVE_ACTIVATIONS``
       frozenset, raise ``ValueError`` naming the offending activation. On success, return
       the sorted, comma-joined collected names, or the literal ``"no-activation-modules"``
       when the set is empty (a decoder built only from linear layers, or a closed-form
       map with no ``nn.Module`` activation at all, is C2 by construction).
    """
    if hasattr(model, "activation"):
        return assert_c2_activation(model)

    decoder = getattr(model, "decoder", model)
    modules_fn = getattr(decoder, "modules", None)
    if not callable(modules_fn):
        raise ValueError(
            f"assert_c2_decoder: {type(model).__name__!r} has neither an 'activation' "
            f"attribute nor a decoder exposing '.modules()'. Refusing to differentiate an "
            f"uninspectable decoder twice -- a zero-second-derivative activation would "
            f"return an identically-zero second fundamental form rather than raising."
        )

    names = set()
    for m in modules_fn():
        if m is decoder or isinstance(m, torch.nn.Linear):
            continue
        names.add(type(m).__name__.lower())

    bad = names & ZERO_SECOND_DERIVATIVE_ACTIVATIONS
    if bad:
        offending = sorted(bad)[0]
        raise ValueError(
            f"assert_c2_decoder: refusing to compute curvature through a decoder with "
            f"activation {offending!r}. A piecewise-linear (or C1-but-not-C2) activation "
            f"has a second derivative that is a sum of Dirac deltas at its kinks, "
            f"numerically zero everywhere autodiff evaluates it, so the entire second "
            f"fundamental form would come back as exactly 0.0 and read as a flat manifold "
            f"instead of raising."
        )
    return ", ".join(sorted(names)) if names else "no-activation-modules"


# --- the map that is differentiated ------------------------------------------------------


def plain_decoder_map(model: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``decode_one(z)``: a single ``(latent_dim,)`` latent vector to a single
    ``(out_dim,)`` ambient point, via ``model.decode(z.unsqueeze(0)).squeeze(0)``.

    ``model.forward`` is deliberately never wrapped here. ``PlainAutoEncoder.forward``
    returns the dict ``{"z": ..., "y": ...}`` and re-derives ``z`` internally from ``x``,
    so differentiating it would measure the curvature of the encoder-composed round-trip
    map rather than of the decoder's own image manifold -- the same warning
    ``chart_curvature.chart_decoder_map`` carries about ``cae._decode_through_chart``.
    Only ``model.decode`` is ever called by the returned closure.
    """

    def decode_one(z: torch.Tensor) -> torch.Tensor:
        return model.decode(z.unsqueeze(0)).squeeze(0)

    return decode_one


# --- the gating computation: exact g-trace of the normal-projected Hessian ----------------


def plain_decoder_curvature(model: Any, z: torch.Tensor) -> Dict[str, Any]:
    """Exact mean curvature vector of the manifold parameterized by ``model``'s decoder
    alone, at each latent coordinate in ``z``.

    Mirrors ``chart_curvature.chart_mean_curvature``'s body exactly (A-03): the ``d x d``
    solve, trace-first-then-project form, never an explicit ``(out_dim, out_dim)``
    projector or a materialized second-fundamental-form tensor -- at the PU regime's
    ``out_dim = 768`` that would be 151 MB of float64 per chunk.

        J    = D F(z)                            (out_dim, latent_dim)
        g    = J^T J                             (latent_dim, latent_dim) pullback metric
        P_N  = I - J g^-1 J^T                    normal projector
        II   = P_N D^2 F(z)                      second fundamental form
        H    = tr_g(II) = sum_jk g^jk II_jk      (out_dim,)

    Guard order: :func:`assert_c2_decoder` first, then ``_assert_float64`` (imported
    unchanged from ``chart_curvature`` -- its message names ``z_chart`` because it is the
    sealed guard reused verbatim rather than forked; the fix it names, ``model.double()``,
    is correct here too).

    Returns a dict with the same keys ``chart_mean_curvature`` returns minus ``chart_idx``,
    which has no meaning for a decoder with no chart index:

      ``"H_vec"``                    ``(batch, out_dim)``  mean curvature vectors, ``tr_g(II)``
      ``"H_norm"``                   ``(batch,)``          ``||H||``
      ``"metric_condition_number"``  ``(batch,)``          ``cond(g)`` per point
      ``"jacobian_shape"``           tuple                 as produced, for the shape assertion
      ``"hessian_shape"``            tuple                 as produced, for the shape assertion
      ``"activation"``                                     :func:`assert_c2_decoder`'s return
      ``"curvature_convention"``                            this module's ``CURVATURE_CONVENTION``

    Runs in float64 throughout; ``_assert_float64`` refuses anything else.
    """
    activation = assert_c2_decoder(model)
    _assert_float64(model, z)

    if z.ndim != 2:
        raise ValueError(
            f"plain_decoder_curvature: z must be (batch, latent_dim); got shape "
            f"{tuple(z.shape)}."
        )
    batch, latent_dim = z.shape
    if batch == 0:
        raise ValueError("plain_decoder_curvature: z is empty; nothing to differentiate.")

    decode_one = plain_decoder_map(model)

    with torch.no_grad():
        probe = decode_one(z[0])
    if probe.ndim != 1:
        raise ValueError(
            f"plain_decoder_curvature: the decoder map must send a (latent_dim,) tensor to "
            f"a (out_dim,) tensor; it returned shape {tuple(probe.shape)}."
        )
    out_dim = int(probe.shape[0])

    H_parts = []
    cond_parts = []
    jacobian_shape = None
    hessian_shape = None
    for start in range(0, batch, VMAP_CHUNK):
        real = z[start : start + VMAP_CHUNK]
        n_real = real.shape[0]
        chunk = _pad_to_chunk(real)

        J = vmap(jacrev(decode_one))(chunk)
        if tuple(J.shape) != (VMAP_CHUNK, out_dim, latent_dim):
            raise ValueError(
                f"plain_decoder_curvature: expected Jacobian of shape "
                f"{(VMAP_CHUNK, out_dim, latent_dim)}, got {tuple(J.shape)}. torch.func's "
                f"transform composition returned something other than a per-point "
                f"Jacobian."
            )

        Hess = vmap(hessian(decode_one))(chunk)
        if tuple(Hess.shape) != (VMAP_CHUNK, out_dim, latent_dim, latent_dim):
            raise ValueError(
                f"plain_decoder_curvature: expected Hessian of shape "
                f"{(VMAP_CHUNK, out_dim, latent_dim, latent_dim)}, got "
                f"{tuple(Hess.shape)}. A Jacobian-shaped result here is the exact warning "
                f"sign that jacrev(jacrev(f)) and hessian(f) are not drop-in "
                f"interchangeable under an outer vmap."
            )
        jacobian_shape = (batch, out_dim, latent_dim)
        hessian_shape = (batch, out_dim, latent_dim, latent_dim)

        g = torch.einsum("boi,boj->bij", J, J)
        eye_d = torch.eye(latent_dim, dtype=g.dtype, device=g.device).expand(
            VMAP_CHUNK, latent_dim, latent_dim
        )
        g_inv = torch.linalg.solve(g, eye_d)

        raw = torch.einsum("bjk,bojk->bo", g_inv, Hess)
        alpha = torch.linalg.solve(
            g, torch.einsum("boi,bo->bi", J, raw).unsqueeze(-1)
        ).squeeze(-1)

        H_parts.append((raw - torch.einsum("boi,bi->bo", J, alpha))[:n_real].detach())
        cond_parts.append(torch.linalg.cond(g)[:n_real].detach())

    H_vec = torch.cat(H_parts, dim=0)
    metric_cond = torch.cat(cond_parts, dim=0)

    return {
        "H_vec": H_vec,
        "H_norm": torch.linalg.norm(H_vec, dim=-1),
        "metric_condition_number": metric_cond,
        "jacobian_shape": jacobian_shape,
        "hessian_shape": hessian_shape,
        "activation": activation,
        "curvature_convention": CURVATURE_CONVENTION,
    }


# --- analytic Swiss roll mean curvature VECTOR (direction, never a new magnitude) --------


def swiss_roll_analytic_H_vector(t: np.ndarray, global_std: float) -> np.ndarray:
    """Analytic mean curvature VECTOR of the preprocessed Swiss roll surface, ``(n, 3)``.

    Hosted here only because ``curvature_probe.py`` is Phase 02.5's sealed artifact and is
    never edited by this milestone; the sealed
    ``curvature_probe.swiss_roll_analytic_H_scaled`` remains the magnitude ground truth,
    and this function contributes a DIRECTION and never a new magnitude -- pinned by
    ``test_swiss_roll_analytic_H_vector_norm_pins_sealed_module``.

    Derivation: ``sklearn.datasets.make_swiss_roll`` parametrizes its surface as
    ``X(t, y) = (t*cos(t), y, t*sin(t))``. Holding ``y`` fixed traces out the planar
    Archimedean-spiral curve ``c(t) = (t*cos(t), t*sin(t))``; the ``y`` direction is
    exactly straight (zero curvature, the ruling direction), so the surface is a ruled
    generalized cylinder over that spiral and its mean curvature VECTOR is exactly the
    generating curve's own curvature vector, lying entirely in the ambient x-z plane with
    the y component identically zero.

    Plane-curve curvature vector: for ``c(t) = (x(t), z(t))`` with ``d1 = c'(t)`` and
    ``d2 = c''(t)``, the curvature vector is the component of ``d2`` orthogonal to ``d1``,
    normalized by ``speed^2 = |d1|^2``:
    ``k_vec = (d2 - (d2 . d1 / speed2) * d1) / speed2``. Here
    ``d1 = [cos(t) - t*sin(t), sin(t) + t*cos(t)]`` and
    ``d2 = [-2*sin(t) - t*cos(t), 2*cos(t) - t*sin(t)]``.

    ``k_vec``'s two components are placed into the returned array's column 0 (x) and
    column 2 (z); column 1 (y, the ruling direction) is exactly zero. Each component is
    scaled by ``global_std`` for the same reason
    ``curvature_probe.swiss_roll_analytic_H_scaled`` rescales the norm: CLAUDE.md's
    preprocessing shrinks ambient distances by ``1/global_std``, so curvature grows by
    ``global_std``.
    """
    t = np.asarray(t, dtype=np.float64)
    d1 = np.stack([np.cos(t) - t * np.sin(t), np.sin(t) + t * np.cos(t)], axis=1)
    d2 = np.stack([-2 * np.sin(t) - t * np.cos(t), 2 * np.cos(t) - t * np.sin(t)], axis=1)
    speed2 = np.sum(d1 * d1, axis=1)
    k_vec = (d2 - (np.sum(d2 * d1, axis=1) / speed2)[:, None] * d1) / speed2[:, None]

    n = t.shape[0]
    H_vec = np.zeros((n, 3), dtype=np.float64)
    H_vec[:, 0] = k_vec[:, 0] * global_std
    H_vec[:, 2] = k_vec[:, 1] * global_std
    return H_vec
