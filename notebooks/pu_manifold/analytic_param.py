"""
Phase 02.6 separating experiment (SC-2; D-12, D-13, D-14, D-15): closed-form Swiss roll
decoder and a net fit to that SAME closed form, both traced through the unchanged
``decoder_curvature.plain_decoder_curvature`` tracer.

Why this module exists. ``02.6-FINDINGS.md`` Section 4 found the halted phase's ranking axis
was a composite of three separable properties -- did the architecture learn the right
surface, are a trained net's second derivatives trustworthy, is pullback the right
approximator -- and never separated the middle one from the first. Section 8.3 names the
experiment that does: fit a net to the Swiss roll's own ANALYTIC parameterization, so the
surface is correct by construction, and measure what replacing the closed form with that net
costs the tracer's exactness. This module supplies the closed-form decoder (the exactness
floor), the net that stands in for it, and the exact intrinsic/ambient coordinate helpers
both need -- nothing here is a new derivation of ``decoder_curvature.py``'s tracer, which is
imported and reused byte-unchanged.

Tensors and arrays in, tensors, arrays and dicts out -- no file I/O and no
artifact-directory handling of any kind; the runners under ``notebooks/diagnostics/`` and the
Swiss roll notebook itself own paths.

Like ``cae.py``, ``chart_curvature.py``, ``curvature_probe.py`` and ``decoder_curvature.py``,
this module imports ``torch`` at module level. For the same reason those modules are excluded
from ``pu_manifold/__init__.py``'s eager imports (so Phase-1-only callers do not need torch
installed to import the package), this module is deliberately NOT re-exported there either.
"""

from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from . import cae

SWISS_ROLL_PARAM_CONVENTION = (
    "sklearn.datasets.make_swiss_roll's own parameterization: the ambient point is "
    "(t*cos(t), y, t*sin(t)), with t the raw Archimedean-spiral parameter (NOT arc length) "
    "and y the ruling coordinate. Every function in this module that takes a 't' argument "
    "means this raw spiral parameter unless its own docstring says otherwise."
)
"""Drift guard mirroring ``decoder_curvature.CURVATURE_CONVENTION``'s role: a module constant
stating, in one place, exactly which parameterization every function below assumes, so a
future edit cannot silently swap in arc length (or any other reparameterization) for one
function while leaving the rest on the raw spiral parameter."""


# --- exact intrinsic-coordinate helpers ---------------------------------------------------


def swiss_roll_arc_length(t: np.ndarray) -> np.ndarray:
    """Exact closed-form arc length of the Archimedean spiral ``c(t) = (t*cos(t), t*sin(t))``
    from ``0`` to ``t``, float64 throughout: ``0.5 * (t * sqrt(1 + t**2) + arcsinh(t))``.

    Derivation. The spiral's speed is
    ``sqrt((cos(t) - t*sin(t))**2 + (sin(t) + t*cos(t))**2) = sqrt(1 + t**2)`` -- the same
    ``speed2`` quantity ``decoder_curvature.swiss_roll_analytic_H_vector`` already computes,
    for a different purpose. The arc length is the integral of that speed from ``0`` to
    ``t``, which has the standard closed form returned here.

    Verified during planning against ``scipy.integrate.quad`` of ``sqrt(1 + u**2)`` from 0 to
    t, at ``t in {0.1, 1.0, 3.0, 7.0, 10.5}``: measured max abs deviation ``9.237e-14``.
    ``02.6-RESEARCH.md`` states a ``9e-14`` bound; that bound is BELOW the measured value and
    must not be adopted as this module's tolerance -- ``test_analytic_param.py`` pins
    ``< 1e-12``.
    """
    t = np.asarray(t, dtype=np.float64)
    return 0.5 * (t * np.sqrt(1.0 + t * t) + np.arcsinh(t))


def swiss_roll_intrinsic_plane(
    t: np.ndarray, y_scaled: np.ndarray, global_std: float
) -> np.ndarray:
    """The ``(n, 2)`` exact intrinsic-coordinate point cloud: column 0 is
    ``swiss_roll_arc_length(t) / global_std``, column 1 is ``y_scaled`` UNCHANGED.

    Both columns are lengths and both are divided by the SAME single global scalar the
    ambient cloud went through -- the opposite scaling direction from curvature (curvature
    grows by ``global_std`` under this same rescaling; see
    ``decoder_curvature.swiss_roll_analytic_H_vector``'s docstring).

    ``y_scaled`` is expected ALREADY CLAUDE.md-scaled: on the standard fixture it is
    literally ``curvature_probe.make_swiss_roll_fixture(...)["X"][:, 1]``, since that fixture
    centres and divides by one global scalar, so its ruling column is already
    ``(raw - mean) / global_std``. Do not divide it by ``global_std`` again here -- doing so
    would double-scale it.
    """
    t = np.asarray(t, dtype=np.float64)
    y_scaled = np.asarray(y_scaled, dtype=np.float64)
    s = swiss_roll_arc_length(t) / float(global_std)
    return np.stack([s, y_scaled], axis=1)


def swiss_roll_exact_ambient(
    t: np.ndarray, y_raw: np.ndarray, center: np.ndarray, global_std: float
) -> np.ndarray:
    """The ``(n, 3)`` exact ambient point at each ``(t, y_raw)`` pair, in the fixture's own
    centred-and-scaled units: ``((t*cos(t), y_raw, t*sin(t)) - center) / global_std``.

    This is what lets a dense held-out grid be built with exact targets and no resampling --
    every grid point's true ambient location is this closed form, never an interpolation.
    """
    t = np.asarray(t, dtype=np.float64)
    y_raw = np.asarray(y_raw, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    ambient = np.stack([t * np.cos(t), y_raw, t * np.sin(t)], axis=1)
    return (ambient - center[None, :]) / float(global_std)


# --- the exactness floor: a closed-form decoder with no parameters -----------------------


class AnalyticSwissRollDecoder(nn.Module):
    """The exactness FLOOR: the Swiss roll's own closed-form decoder, computed entirely in
    torch so it is twice differentiable by ``torch.func`` autodiff, holding no parameters and
    no activation submodule.

    ``decode(z)`` takes a ``(batch, 2)`` tensor whose column 0 is the RAW spiral parameter
    ``t`` (never arc length -- see ``SWISS_ROLL_PARAM_CONVENTION``) and whose column 1 is the
    already-scaled ruling coordinate ``y_scaled = (y_raw - center[1]) / global_std``. Because
    ``center`` and ``global_std`` are constructor-fixed and the centring/scaling is
    componentwise-additive-then-isotropic, ``y_scaled`` passes straight through to the
    output's column 1 unchanged -- no round trip through ``y_raw`` is needed:

        x_out = (t*cos(t) - center[0]) / global_std
        y_out = y_scaled
        z_out = (t*sin(t) - center[2]) / global_std

    This is algebraically identical to ``swiss_roll_exact_ambient(t, y_raw, center,
    global_std)`` for any ``y_raw`` with ``y_scaled = (y_raw - center[1]) / global_std``.

    ``forward`` returns the same value under the ``"y"`` key, matching this repo's
    dict-returning decoder convention (``PlainAutoEncoder.forward``, ``MlpDecoder.forward``).

    It holds no parameters and no ``nn.Module`` activation submodule, so
    ``decoder_curvature.assert_c2_decoder`` reaches its structural (no-``.activation``)
    branch and returns the literal ``"no-activation-modules"`` -- correct, since a
    closed-form map built from smooth elementary functions is C2 by construction.

    **Why it is legitimate to measure the floor in this (spiral-parameter, scaled-y)
    parameterization while D-12's trained net is fit on the arc-length parameterization
    (``swiss_roll_intrinsic_plane``'s column 0).** The mean curvature vector is an intrinsic
    property of the surface itself, invariant under any smooth reparameterization of the
    domain -- the same fact that lets ``decoder_curvature.swiss_roll_analytic_H_vector``
    serve as ground truth for both this floor (measured in spiral-parameter coordinates) and
    the trained net (measured in arc-length coordinates) without adjustment. A linear
    reparameterization of ``y`` (``y_scaled`` for ``y_raw``) and a monotone reparameterization
    of ``t`` (arc length for the spiral parameter) each change the Jacobian and Hessian of the
    decoder map, but the trace-first ``g``-normalized combination
    ``decoder_curvature.plain_decoder_curvature`` computes cancels that change exactly,
    leaving the same ambient curvature VECTOR at the corresponding surface point.
    """

    def __init__(self, center: np.ndarray, global_std: float):
        super().__init__()
        center_t = torch.as_tensor(np.asarray(center, dtype=np.float64), dtype=torch.float64)
        if tuple(center_t.shape) != (3,):
            raise ValueError(
                f"AnalyticSwissRollDecoder: center must be a length-3 vector; got shape "
                f"{tuple(center_t.shape)}."
            )
        self.register_buffer("center", center_t)
        self.global_std = float(global_std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != 2:
            raise ValueError(
                f"AnalyticSwissRollDecoder.decode: z must be (batch, 2) holding (t, "
                f"y_scaled); got shape {tuple(z.shape)}."
            )
        t = z[:, 0]
        y_scaled = z[:, 1]
        x_out = (t * torch.cos(t) - self.center[0]) / self.global_std
        z_out = (t * torch.sin(t) - self.center[2]) / self.global_std
        return torch.stack([x_out, y_scaled, z_out], dim=1)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"y": self.decode(z)}


# --- D-12's generously sized, deliberately not budget-matched net -------------------------


class AnalyticParamDecoder(nn.Module):
    """D-12's best-case net: fit by regression on the ANALYTIC Swiss roll parameterization
    ``(s, y_scaled) -> R^3`` (``s`` the arc-length intrinsic coordinate, per
    ``swiss_roll_intrinsic_plane``), so the surface it is asked to represent is correct by
    construction. Deliberately generously sized (``hidden=(256, 256, 256)`` default) and NOT
    budget-matched to any other substrate in this phase -- this is a best-case feasibility
    probe for whether a trained net's second derivatives survive at all, not a fair
    architecture comparison.

    Built from ``cae.mlp_stack`` exactly as ``cae.PlainAutoEncoder``'s decoder half and
    ``cae.MlpDecoder`` are, and sets an ``activation`` attribute (unlike
    ``AnalyticSwissRollDecoder``) so ``decoder_curvature.assert_c2_decoder`` takes its
    sealed-delegation branch (``chart_curvature.assert_c2_activation``) rather than its
    structural-inspection branch.

    ``forward`` returns the decoded ambient point under the ``"y"`` key -- exactly the dict
    shape ``cae.train_mlp_decoder`` (via ``cae._train_decoder_protocol``) expects from its
    ``forward_fn``, which is what makes that sealed training protocol usable on this class
    UNCHANGED, duck-typed the same way ``notebooks/diagnostics/decoder_substrate_screen_run.py``
    duck-types its own training entry points. No training loop is written in this module.
    """

    def __init__(self, hidden=(256, 256, 256), activation: str = "silu"):
        super().__init__()
        self.activation = activation
        self.net = cae.mlp_stack(2, hidden, 3, activation, out_activation=None)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"y": self.decode(z)}
