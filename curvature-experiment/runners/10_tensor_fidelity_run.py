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
    python curvature-experiment/runners/10_tensor_fidelity_run.py --mode full --noise 0.25 --seed 0 --threads 8 \\
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

import argparse  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.func import hessian, jacrev, vmap  # noqa: E402

from sklearn.linear_model import Ridge  # noqa: E402

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


FIXTURE_SEED = 20260905


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_for(mode: str, n) -> Dict[str, Any]:
    if mode == "full":
        return dict(adj.FIXTURE)
    n = int(n) if n is not None else int(adj.SMOKE["n"])
    return {**adj.SMOKE, "n": n, "k": 128 if n <= 4000 else n // 32, "n_anchors": 64}


def run_config(cfg: Dict[str, Any], noise_frac: float, seed: int, max_epochs: int, device: str, mode: str,
               record_path: Path) -> list:
    d, D, n, k, n_anchors = cfg["d"], cfg["D"], cfg["n"], cfg["k"], cfg["n_anchors"]
    print(f"\n[config] mode={mode} d={d} D={D} n={n} k={k} anchors={n_anchors} noise={noise_frac} seed={seed}", flush=True)
    G = adj.InSphereGenerator(d, D, cfg["a"], cfg["bump_widths"], cfg["bump_amps"], seed=FIXTURE_SEED)
    z = adj.draw_latents(n, d, cfg["scale_choices"], cfg["scale_probs"], seed=FIXTURE_SEED + 1 + 1000 * seed)
    X0 = adj.generate_points(G, z)
    a = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)["anchor_idx"]
    panel = pcp.knn_panel(X0, a, k)
    if noise_frac > 0:
        # the adjudication runner's patch-noise protocol, at a chosen fraction of the median patch radius
        sigma = noise_frac * float(np.median(panel["distances"][:, -1])) / np.sqrt(D)
        X = X0 + np.random.default_rng(FIXTURE_SEED + 2 + 1000 * seed).standard_normal((n, D)) * sigma
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        panel = pcp.knn_panel(X, a, k)
    else:
        X = X0
    labels = make_labels(G, z)
    truth = truth_geometry(G, z[a])

    default_init = pcp.TORCH_INIT_SEED
    pcp.TORCH_INIT_SEED = int(default_init) + int(seed)
    try:
        fit = ppf.fit_decoder(X, d, D, max_epochs, device=device)
    finally:
        pcp.TORCH_INIT_SEED = default_init
    with torch.no_grad():
        z_hat = fit["model"].encode(fit["x64"][torch.as_tensor(a, dtype=torch.long, device=fit["x64"].device)])
    est = ppf.decoder_geometry(fit["curvature_model"], z_hat)
    print(f"[fit] var_explained={fit['var_explained']:.4f} in {fit['wallclock_fit_s']:.0f}s", flush=True)

    rows = []
    for name, y in labels["y"].items():
        ridge = Ridge(alpha=pcp.ALPHA_RIDGE).fit(X, y)
        w, b0 = ridge.coef_.astype(np.float64), float(ridge.intercept_)
        lq = local_quadratics(X, a, panel["indices"], est, {"y": y, "p": X @ w}, pcp.MIN_FINITE_NEIGHBOURS)
        pf_e, pf_t = probe_facing_tensors(est, w), probe_facing_tensors(truth, w)
        hess_e = lq["hess"]["y"]
        hess_t = covariant_hessian(labels["f"][name], truth, z[a])
        T_e = {"pf_full": pf_e["pf_full"], "pf_tan": pf_e["pf_tan"], "hess_y": hess_e, "mismatch": hess_e - pf_e["pf_full"]}
        T_t = {"pf_full": pf_t["pf_full"], "pf_tan": pf_t["pf_tan"], "hess_y": hess_t, "mismatch": hess_t - pf_t["pf_full"]}
        c_e = split_columns(est, w, b0, hess_e, lq["hess"]["p"], d)["cols"]
        c_t = split_columns(truth, w, b0, hess_t, pf_t["pf_full"], d)["cols"]
        row = {"experiment": EXPERIMENT, "row": "result", "mode": mode, "n": n, "k": k, "n_anchors": n_anchors,
               "noise_frac": float(noise_frac), "seed": int(seed), "label": name,
               "lam": next((lam for lam in LAMBDAS if label_name(lam) == name), None), "device": device,
               "var_explained": float(fit["var_explained"]), "max_abs_H_rad_plus_d": truth["max_abs_H_rad_plus_d"],
               "rho_mismatch": ppf._spearman(c_e["hess_mismatch_dec"], c_t["hess_mismatch_dec"]),
               "rho_align": ppf._spearman(c_e["align_cos_full"], c_t["align_cos_full"]),
               **score_tensors(T_e, T_t, est, truth), "timestamp": _utc_now()}
        ppf._append(row, record_path)
        rows.append(row)
        print(f"  {name:8s} cos pf_full {row['cos_pf_full_p50']:+.3f} pf_tan {row['cos_pf_tan_p50']:+.3f} "
              f"hess_y {row['cos_hess_y_p50']:+.3f} mismatch {row['cos_mismatch_p50']:+.3f} | "
              f"rho mismatch {row['rho_mismatch']:+.3f} align {row['rho_align']:+.3f}", flush=True)
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["small", "full"], required=True)
    p.add_argument("--n", type=int, default=None, help="small mode only; default 4000")
    p.add_argument("--noise", type=float, default=0.0, help="fraction of the median k-NN patch radius")
    p.add_argument("--seed", type=int, default=0, help="latent draw, noise and decoder init")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8")
    p.add_argument("--max-epochs", type=int, default=None, help="test override only; default pcp.MAX_EPOCHS")
    p.add_argument("--record-path", type=str, default=str(DEFAULT_RECORD_PATH))
    return p


def main() -> None:
    args = build_parser().parse_args()
    record_path = Path(args.record_path).resolve()
    for stem in ppf.PRODUCTION_STEMS:
        if record_path.name.startswith(stem):
            raise SystemExit(f"refusing to write to a Phase 9 production record path: {record_path}")
    assert runner._THREADS == args.threads, (runner._THREADS, args.threads)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available on this machine")
    gpu_name = torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None
    cfg = config_for(args.mode, args.n)
    max_epochs = args.max_epochs if args.max_epochs is not None else pcp.MAX_EPOCHS
    print(f"record -> {record_path}\nNOT PRE-REGISTERED FOR THE PAPER; GATES NOTHING.\n"
          f"DECISION RULE: noise-free small fixture, median pf_full cosine >= {TENSOR_COS_PASS} and "
          f"mismatch Spearman >= {MISMATCH_RHO_PASS}.")
    ppf._append({"experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": _utc_now(),
                 "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "threads": args.threads, "cfg": cfg,
                 "noise_frac": args.noise, "seed": args.seed, "max_epochs": max_epochs,
                 "numpy": np.__version__, "torch": torch.__version__, "python": sys.version.split()[0],
                 "device": args.device, "deterministic": args.deterministic, "gpu_name": gpu_name,
                 "cuda_version": torch.version.cuda}, record_path)
    t0 = time.monotonic()
    run_config(cfg, args.noise, args.seed, max_epochs, args.device, args.mode, record_path)
    print(f"DONE in {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
