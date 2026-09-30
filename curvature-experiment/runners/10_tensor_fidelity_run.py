"""Tensor fidelity: the split runner's full probe-facing tensors against exact truth on the in-sphere fixture.

PURPOSE. ``09_instrument_adjudication_run.py`` validated only the trace of the second fundamental
form (``H_tan``). The paper's diagnostic uses the full contraction of II with the probe. This runner
estimates, exactly as the split runner does, ``pf_full = <w_N, II>``, its in-sphere part ``pf_tan``,
the label Hessian ``hess_y`` (local quadratic in tangent-projected coordinates) and the mismatch
``hess_y - pf_full``, and compares each with exact truth from the generator.

CHART INVARIANCE. The decoder's chart is not the generator's. Every (0,2) tensor T with chart
Jacobian J is compared through its ambient lift ``J ginv T ginv J^T`` (the tangent-plane tensor,
chart-free); inner products are formed as ``tr(M_a C M_b C^T)`` with ``M = ginv T ginv`` and
``C = J_a^T J_b``, never materialising D x D matrices.

TRUTH. float64 autodiff of the generator on CPU at each anchor's own latent: J, Hess, II by
``ppf.decoder_geometry(G, z)``; covariant label Hessian ``d^2 f - Gamma^k d_k f``.

DECISION RULES (fixed here, before any number is printed): on the noise-free small fixture,
median pf_full cosine >= TENSOR_COS_PASS and mismatch-norm Spearman >= MISMATCH_RHO_PASS.

NOT PRE-REGISTERED FOR THE PAPER, GATES NOTHING.

Usage:
    python curvature-experiment/runners/10_tensor_fidelity_run.py --mode small --n 4000 --noise 0 --seed 0 --threads 8
    python curvature-experiment/runners/10_tensor_fidelity_run.py --mode paper --noise 0.25 --seed 0 --threads 8 \\
        --device cuda --deterministic
"""

import importlib.util
import os
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_SPLIT_PATH = DIAGNOSTICS_ROOT / "09_physics_probe_facing_split_run.py"
_spec = importlib.util.spec_from_file_location("physics_probe_facing_split_run", _SPLIT_PATH)
_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_split)
ppf, adj, runner = _split.ppf, _split.adj, _split.runner
local_quadratics, split_columns = _split.local_quadratics, _split.split_columns

from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.func import hessian, jacrev, vmap  # noqa: E402

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

EXPERIMENT = "tensor-fidelity"
DEFAULT_RECORD_PATH = Path(os.environ.get("EFFDIM_CACHE_DIR") or NOTEBOOK_ROOT / ".cache") / "10_tensor_fidelity.jsonl"
TENSORS = ("pf_full", "pf_tan", "hess_y", "mismatch")

# Decision rules -- fixed in source before any run.
TENSOR_COS_PASS = 0.8
MISMATCH_RHO_PASS = 0.7
MISMATCH_FLOOR = 1e-6
"""Mismatch cosine is undefined where |mismatch_true| < MISMATCH_FLOOR * |hess_y_true| (label lam0)."""


def _M(T: np.ndarray, geo: Dict[str, np.ndarray]) -> np.ndarray:
    return np.einsum("bij,bjk,bkl->bil", geo["ginv"], T, geo["ginv"])


def ambient_inner(Ta: np.ndarray, geo_a: Dict[str, np.ndarray], Tb: np.ndarray, geo_b: Dict[str, np.ndarray]) -> np.ndarray:
    """<J_a M_a J_a^T, J_b M_b J_b^T>_F per anchor, M = ginv T ginv, via C = J_a^T J_b (d x d)."""
    C = np.einsum("bai,baj->bij", geo_a["J"], geo_b["J"])
    return np.einsum("bij,bjk,bkl,bil->b", _M(Ta, geo_a), C, _M(Tb, geo_b), C)


def tensor_cosine(Ta, geo_a, Tb, geo_b) -> np.ndarray:
    ab = ambient_inner(Ta, geo_a, Tb, geo_b)
    aa = ambient_inner(Ta, geo_a, Ta, geo_a); bb = ambient_inner(Tb, geo_b, Tb, geo_b)
    return ab / np.sqrt(np.maximum(aa * bb, 1e-300))


def relative_error(T_est, geo_est, T_true, geo_true, ref_norm=None) -> np.ndarray:
    """|lift(T_est) - lift(T_true)|_F / ref (default |lift(T_true)|_F). Formed from inner products, so the
    numerical floor is ~sqrt(machine eps) ~ 1e-8, far below any estimation error it is used to measure."""
    ee = ambient_inner(T_est, geo_est, T_est, geo_est)
    tt = ambient_inner(T_true, geo_true, T_true, geo_true)
    et = ambient_inner(T_est, geo_est, T_true, geo_true)
    diff = np.sqrt(np.maximum(ee + tt - 2 * et, 0.0))
    ref = np.sqrt(np.maximum(tt, 0.0)) if ref_norm is None else ref_norm
    return diff / np.maximum(ref, 1e-300)


def probe_facing_tensors(geo: Dict[str, np.ndarray], w: np.ndarray) -> Dict[str, np.ndarray]:
    """The split runner's pf_full / pf_tan formulas (split_columns), returning the tensors, not norms."""
    J, ginv, II, x = geo["J"], geo["ginv"], geo["II"], geo["image"]
    xhat = x / np.linalg.norm(x, axis=1, keepdims=True)
    wT = np.einsum("bai,a->bi", J, w)
    w_N = w[None, :] - np.einsum("bai,bij,bj->ba", J, ginv, wT)
    II_rad = np.einsum("baij,ba->bij", II, xhat)
    II_tan = II - np.einsum("ba,bij->baij", xhat, II_rad)
    return {"w_N": w_N, "pf_full": np.einsum("baij,ba->bij", II, w_N), "pf_tan": np.einsum("baij,ba->bij", II_tan, w_N)}


def truth_geometry(G: torch.nn.Module, z_anchor: np.ndarray) -> Dict[str, np.ndarray]:
    """Exact geometry of the generator at the anchors' own latents (float64, CPU)."""
    geo = ppf.decoder_geometry(G, torch.as_tensor(z_anchor, dtype=torch.float64))
    xhat = geo["image"] / np.linalg.norm(geo["image"], axis=1, keepdims=True)
    dev = float(np.max(np.abs(np.einsum("ba,ba->b", geo["H"], xhat) + G.d)))
    if not dev < 1e-8:
        raise AssertionError(f"exactness check failed: max|H_rad + d| = {dev:.3e} (expected < 1e-8)")
    geo["max_abs_H_rad_plus_d"] = dev
    return geo


def _norm(T, geo) -> np.ndarray:
    return np.sqrt(np.maximum(ambient_inner(T, geo, T, geo), 0.0))


def score_tensors(est: Dict[str, np.ndarray], truth: Dict[str, np.ndarray],
                  geo_est: Dict[str, np.ndarray], geo_true: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Per tensor: cosine p25/p50, relative-error p50 and valid-anchor count; NaN anchors excluded.
    The mismatch is normalised by |hess_y_true| and its cosine dropped where the true mismatch is ~0."""
    out: Dict[str, Any] = {}
    hess_ref = _norm(truth["hess_y"], geo_true)
    for t in TENSORS:
        cos = tensor_cosine(est[t], geo_est, truth[t], geo_true)
        rel = relative_error(est[t], geo_est, truth[t], geo_true, ref_norm=hess_ref if t == "mismatch" else None)
        if t == "mismatch":
            cos = np.where(_norm(truth[t], geo_true) < MISMATCH_FLOOR * hess_ref, np.nan, cos)
        ok = np.isfinite(cos)
        out[f"n_valid_{t}"] = int(ok.sum())
        out[f"cos_{t}_p25"] = float(np.percentile(cos[ok], 25)) if ok.any() else float("nan")
        out[f"cos_{t}_p50"] = float(np.median(cos[ok])) if ok.any() else float("nan")
        okr = np.isfinite(rel)
        out[f"relerr_{t}_p50"] = float(np.median(rel[okr])) if okr.any() else float("nan")
    return out


LAMBDAS = (0.0, 0.5, 1.0, 2.0)
BUMP_WIDTH = 0.8
LABEL_SEED = 20260930


def label_name(lam: float) -> str:
    return f"lam{lam:g}"


def _batched(f, zt: torch.Tensor, batch: int = 8192) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([f(zt[s:s + batch]).numpy() for s in range(0, zt.shape[0], batch)])


def make_labels(G: torch.nn.Module, z_all: np.ndarray) -> Dict[str, Any]:
    """lin, nonlin (the smoke fixture's shapes) and y_lambda = <w0, G(z)>/s_a + lambda * h(z)/s_h."""
    rng = np.random.default_rng(LABEL_SEED)
    d, D = G.d, G.D
    t = lambda v: torch.as_tensor(v, dtype=torch.float64)  # noqa: E731
    a1 = rng.standard_normal(d); a1 /= np.linalg.norm(a1)
    a2 = rng.standard_normal(d)
    w0 = rng.standard_normal(D); w0 /= np.linalg.norm(w0)
    centres = rng.standard_normal((2, d)) * 0.5
    a1t, a2t, w0t, ct = t(a1), t(a2), t(w0), t(centres)

    def lin(z): return z @ a1t
    def nonlin(z): return torch.sin(2 * (z @ a1t)) + (z @ a2t) ** 2
    def ambient(z): return G.decode(z) @ w0t
    def bump(z): return torch.exp(-((z[:, None, :] - ct[None, :, :]) ** 2).sum(-1) / (2 * BUMP_WIDTH ** 2)).sum(-1)

    zt = t(z_all)
    s_a = float(np.std(_batched(ambient, zt))); s_h = float(np.std(_batched(bump, zt)))
    f: Dict[str, Any] = {"lin": lin, "nonlin": nonlin}
    for lam in LAMBDAS:
        f[label_name(lam)] = (lambda lam_: (lambda z: ambient(z) / s_a + lam_ * bump(z) / s_h))(lam)
    y = {name: _batched(fn, zt) for name, fn in f.items()}
    return {"f": f, "y": y, "w0": w0, "scales": {"ambient": s_a, "bump": s_h}}


def covariant_hessian(f, geo: Dict[str, np.ndarray], z_anchor: np.ndarray) -> np.ndarray:
    """nabla^2 f = d^2 f - Gamma^k d_k f, Gamma^k_ij = ginv^{kl} J_l . d_i d_j G (float64 autodiff)."""
    zt = torch.as_tensor(z_anchor, dtype=torch.float64)
    f_one = lambda z1: f(z1.unsqueeze(0)).squeeze(0)  # noqa: E731
    grad = vmap(jacrev(f_one))(zt).detach().numpy()
    hess = vmap(hessian(f_one))(zt).detach().numpy()
    Gamma = np.einsum("bkl,bal,baij->bkij", geo["ginv"], geo["J"], geo["Hess"])
    return hess - np.einsum("bkij,bk->bij", Gamma, grad)
