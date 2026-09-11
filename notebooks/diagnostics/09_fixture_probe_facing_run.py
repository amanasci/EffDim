"""Probe-facing curvature on the sphere fixture: the local-residual theorem, term by term.

PURPOSE. Supplement 03 and Supplement 04 § 8 measured the sealed Phase 9 statistic (controlled
partial of ``||H_tan||`` against the local R^2 of a global ridge probe) on the adjudication fixture
with exact curvature, and found a curve in the density-curvature coupling: −0.29 at coupling +0.80,
a plateau near +0.22 for coupling below +0.15. Why an intrinsic-linear label reads POSITIVE at
zero coupling was left unresolved. This runner computes, per anchor and from exact geometry, the
quantities the second-order expansion of the residual actually contains, and asks which of them
carries the association.

THE EXPANSION. Global affine probe ``yhat = w.x + b0`` on a d-manifold M in R^D. At an anchor x0
with orthonormal tangent frame, second fundamental form II and normal part w_N of w:

    Hess_M(yhat) = <w_N, II>                  (restriction of a linear map to a submanifold)
    r = y - yhat,  E_patch[r^2] = c^2 + s^2 |delta|^2 + (s^4/4) [ (tr Delta)^2 + 2 |Delta|_F^2 ] + O(s^5)

with ``c`` the residual at the anchor, ``delta = grad_M y - grad_M yhat`` the gradient mismatch,
``Delta = Hess_M y - <w_N, II>`` the Hessian mismatch, ``s^2`` the isotropic second moment of the
patch, and all norms in the induced metric. Local R^2 = 1 - E[r^2] / Var_patch(y).

EXACT GEOMETRY. The fixture generator is ``G(z) = Q v(z)/|v(z)|`` with ``v`` supported on d+2
coordinates, so all geometry is computed in the unrotated R^(d+2) by central finite differences
(J, second derivatives), and ``w`` is rotated into that frame by ``Q^T``. Christoffel symbols come
from the tangential part of the second derivative, ``Gamma^k_ij = (g^{-1} J^T d_i d_j f)^k``, and
``Hess_M y = d^2 y - Gamma . dy`` for a label given in the chart coordinates z. Supplement 03
verified this finite-difference field against the sealed autodiff at rank 1.000000.

PER-ANCHOR COLUMNS (all exact): ``exact_H_tan`` (reference, the sealed verdict field),
``pf_curv`` = |<w_N, II>|_g, ``pf_trace_tan`` = <w_N, H_tan>, ``pf_trace_rad`` = -d <w, x0>,
``hess_mismatch`` = |Delta|_g, ``grad_mismatch`` = |delta|_g, ``bias_sq`` = c^2, ``pred_resid``
= the expansion's E[r^2] with s^2 = r_k^2/(d+2), ``pred_local_r2`` = 1 - pred_resid / pred_var.
Each is (i) Spearman-correlated with the measured local R^2 and with ``exact_H_tan``, and (ii)
run through the sealed three-control partial against the measured local R^2.

Samples are rebuilt bit-for-bit as ``09_fixture_probe_decodability_run.py`` builds them (same
pool, seeds, weighting, anchors, labels, probe), by importing that runner unchanged.

NOT PRE-REGISTERED, GATES NOTHING.

Usage:
    python notebooks/diagnostics/09_fixture_probe_facing_run.py --mode smoke --threads 8
    python notebooks/diagnostics/09_fixture_probe_facing_run.py --mode full --gammas -1,0,0.4,0.6,0.8,1 --threads 16
"""

import importlib.util
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_FIX_PATH = DIAGNOSTICS_ROOT / "09_fixture_probe_decodability_run.py"
_spec = importlib.util.spec_from_file_location("fixture_probe_decodability_run", _FIX_PATH)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)
adj, runner = fx.adj, fx.runner

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

EXPERIMENT = "fixture-probe-facing"
DEFAULT_RECORD_PATH = NOTEBOOK_ROOT / ".cache" / "09_fixture_probe_facing.jsonl"
COLUMNS = ("exact_H_tan", "pf_curv", "pf_trace_tan", "pf_trace_rad", "hess_mismatch", "grad_mismatch",
           "bias_sq", "pred_resid", "pred_local_r2")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- exact geometry at a small batch of latents ------------------------------------------------


def fd_geometry(G: Any, z: np.ndarray, h: float = fx.FD_STEP) -> Dict[str, np.ndarray]:
    """x0 (b,d+2), J (b,d+2,d), Hs (b,d+2,d,d), g, ginv, P_N-projected II, Christoffel Gamma (b,d,d,d)
    [Gamma[b,k,i,j] = Gamma^k_ij], H (b,d+2) = tr_g II, all in the unrotated R^(d+2)."""
    f = fx._unrotated_map(G)
    z = np.asarray(z, dtype=np.float64)
    b, d = z.shape
    E = np.eye(d)
    x0 = f(z)
    plus = [f(z + h * E[i]) for i in range(d)]
    minus = [f(z - h * E[i]) for i in range(d)]
    J = np.stack([(plus[i] - minus[i]) / (2 * h) for i in range(d)], -1)
    Hs = np.zeros((b, x0.shape[1], d, d))
    for i in range(d):
        Hs[:, :, i, i] = (plus[i] - 2 * x0 + minus[i]) / h ** 2
        for j in range(i + 1, d):
            v = (f(z + h * E[i] + h * E[j]) - f(z + h * E[i] - h * E[j])
                 - f(z - h * E[i] + h * E[j]) + f(z - h * E[i] - h * E[j])) / (4 * h * h)
            Hs[:, :, i, j] = v; Hs[:, :, j, i] = v
    g = np.einsum("bai,baj->bij", J, J)
    ginv = np.linalg.inv(g)
    Gamma = np.einsum("bkl,bal,baij->bkij", ginv, J, Hs)                   # tangential part
    II = Hs - np.einsum("bak,bkij->baij", J, Gamma)                          # normal part
    H = np.einsum("bij,baij->ba", ginv, II)
    return {"x0": x0, "J": J, "Hs": Hs, "g": g, "ginv": ginv, "Gamma": Gamma, "II": II, "H": H}


def label_derivatives(name: str, z: np.ndarray, x_rot: np.ndarray, geo: Dict[str, np.ndarray], Qt18: np.ndarray,
                      a1: np.ndarray, a2: np.ndarray, w_true: np.ndarray) -> Dict[str, np.ndarray]:
    """Chart-coordinate gradient (b,d) and Hessian (b,d,d) of each label at the anchors."""
    b, d = z.shape
    if name == "intrinsic_linear":
        return {"grad": np.tile(a1, (b, 1)), "hess": np.zeros((b, d, d))}
    if name == "nonlinear":
        u1, u2 = z @ a1, z @ a2
        grad = (2 * np.cos(2 * u1))[:, None] * a1 + (2 * u2)[:, None] * a2 - 0.5 * (u2[:, None] * a1 + u1[:, None] * a2)
        hess = (-4 * np.sin(2 * u1))[:, None, None] * np.outer(a1, a1)[None] + 2 * np.outer(a2, a2)[None] \
            - 0.5 * (np.outer(a1, a2) + np.outer(a2, a1))[None]
        return {"grad": grad, "hess": np.broadcast_to(hess, (b, d, d)).copy() if hess.shape[0] == 1 else hess}
    if name == "ambient_linear_null":
        c18 = Qt18 @ w_true                                                   # w_true in the unrotated frame
        grad = np.einsum("bai,a->bi", geo["J"], c18)
        hess = np.einsum("baij,a->bij", geo["Hs"], c18)                        # chart Hessian of c.x(z)
        return {"grad": grad, "hess": hess}
    raise ValueError(name)


def metric_norms(M: np.ndarray, ginv: np.ndarray) -> Dict[str, np.ndarray]:
    """Trace and Frobenius norm of a (b,d,d) covariant 2-tensor in the induced metric."""
    tr = np.einsum("bij,bji->b", ginv, M)
    fro2 = np.einsum("bij,bjk,bkl,bli->b", ginv, M, ginv, M)
    return {"tr": tr, "fro": np.sqrt(np.maximum(fro2, 0.0))}


def probe_facing_columns(geo: Dict[str, np.ndarray], w18: np.ndarray, b0: float, y_anchor: np.ndarray,
                         x18_anchor: np.ndarray, lab: Dict[str, np.ndarray], r_k: np.ndarray, d: int) -> Dict[str, np.ndarray]:
    J, ginv, II, Gamma, H, x0 = geo["J"], geo["ginv"], geo["II"], geo["Gamma"], geo["H"], geo["x0"]
    b = J.shape[0]
    # probe: gradient and Hessian on M (chart coordinates)
    w_T_coef = np.einsum("bai,a->bi", J, w18)                                   # covector J^T w
    w_N = w18[None, :] - np.einsum("bai,bij,bj->ba", J, ginv, w_T_coef)        # P_N w
    hess_probe = np.einsum("baij,ba->bij", II, w_N)                             # <w_N, II>
    # label: intrinsic Hessian = chart Hessian - Gamma . grad
    hess_label = lab["hess"] - np.einsum("bkij,bk->bij", Gamma, lab["grad"])
    delta = lab["grad"] - w_T_coef
    Delta = hess_label - hess_probe
    pf = metric_norms(hess_probe, ginv)
    mm = metric_norms(Delta, ginv)
    hl = metric_norms(hess_label, ginv)
    grad_mis2 = np.einsum("bi,bij,bj->b", delta, ginv, delta)
    grad_y2 = np.einsum("bi,bij,bj->b", lab["grad"], ginv, lab["grad"])
    H_rad = np.einsum("ba,ba->b", H, x0)
    H_tan = H - H_rad[:, None] * x0
    pf_trace_tan = np.einsum("ba,ba->b", w_N, H_tan)
    pf_trace_rad = H_rad * np.einsum("ba,ba->b", w_N, x0)                     # = -d <w, x0> on the sphere
    c = y_anchor - (x18_anchor @ w18 + b0)
    s2 = r_k ** 2 / (d + 2)
    pred_resid = c ** 2 + s2 * grad_mis2 + (s2 ** 2 / 4) * (mm["tr"] ** 2 + 2 * mm["fro"] ** 2)
    pred_var = s2 * grad_y2 + (s2 ** 2 / 4) * (hl["tr"] ** 2 + 2 * hl["fro"] ** 2)
    return {
        "exact_H_tan": np.linalg.norm(H_tan, axis=1), "pf_curv": pf["fro"], "pf_trace_tan": pf_trace_tan,
        "pf_trace_rad": pf_trace_rad, "hess_mismatch": mm["fro"], "grad_mismatch": np.sqrt(grad_mis2),
        "bias_sq": c ** 2, "pred_resid": pred_resid, "pred_local_r2": 1.0 - pred_resid / np.maximum(pred_var, 1e-300),
        "_s2": s2, "_H_rad": H_rad,
    }


# --- one gamma ---------------------------------------------------------------------------------


def run_gamma(gamma: float, cfg: Dict[str, Any], pool: Dict[str, Any], G: Any, args: argparse.Namespace,
              record_path: Path, n_perm: int) -> None:
    d, D, n, k, n_anchors = cfg["d"], cfg["D"], cfg["n"], cfg["k"], cfg["n_anchors"]
    print("\n" + "=" * 78 + f"\ngamma={gamma:+g}: n={n}, d={d}, D={D}, k={k}, anchors={n_anchors}\n" + "=" * 78, flush=True)
    rng = np.random.default_rng(args.seed + 100 + int(round(10 * gamma)))       # identical to the fixture runner
    sel = fx.weighted_subsample(pool["H_tan_norm"], gamma, n, rng)
    z = pool["z"][sel]
    X = adj.generate_points(G, z)
    split = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)
    a = split["anchor_idx"]
    panel = pcp.knn_panel(X, a, k)
    log_r = panel["log_knn_radius"]
    r_k = panel["distances"][:, -1]

    geo = fd_geometry(G, z[a])
    Q = G.Q.numpy()
    Qt18 = Q.T[: d + 2]                                                          # rows: unrotated coords
    x18 = geo["x0"]
    assert np.max(np.abs(x18 - (Qt18 @ X[a].T).T)) < 1e-9
    H_fd_norm = np.linalg.norm(geo["H"] - np.einsum("ba,ba->b", geo["H"], x18)[:, None] * x18, axis=1)
    print(f"geometry at anchors: FD ||H_tan|| vs pool FD rank {spearmanr(H_fd_norm, pool['H_tan_norm'][sel][a]).statistic:.6f}; "
          f"H_rad max|.+d| {np.max(np.abs(np.einsum('ba,ba->b', geo['H'], x18) + d)):.2e}; "
          f"coupling rho(||H_tan||, log r) = {spearmanr(H_fd_norm, log_r).statistic:+.3f}", flush=True)

    # labels exactly as the fixture runner draws them
    lrng = np.random.default_rng(args.seed + 7)
    a1 = lrng.standard_normal(d); a1 /= np.linalg.norm(a1)
    a2 = lrng.standard_normal(d); a2 /= np.linalg.norm(a2)
    w_true = lrng.standard_normal(D); w_true /= np.linalg.norm(w_true)
    labels = fx.make_labels(z, X, np.random.default_rng(args.seed + 7))

    for name in args.labels.split(","):
        y = labels[name]
        out = fx.probe_outcome(X, y, panel)
        r2, Z = out["r2"], out["Z"]
        ridge = Ridge(alpha=pcp.ALPHA_RIDGE).fit(X, y)
        w18 = Qt18 @ ridge.coef_
        lab = label_derivatives(name, z[a], X[a], geo, Qt18, a1, a2, w_true)
        cols = probe_facing_columns(geo, w18, float(ridge.intercept_), y[a], x18, lab, r_k, d)
        m = np.isfinite(r2)
        rho_r2 = {c: fx._spearman(cols[c][m], r2[m]) for c in COLUMNS}
        rho_H = {c: fx._spearman(cols[c], cols["exact_H_tan"]) for c in COLUMNS}
        rho_logr = {c: fx._spearman(cols[c], log_r) for c in COLUMNS}
        partials = {c: fx.partial_row(cols[c], r2, Z, n_perm) for c in COLUMNS}
        wn_frac = float(np.median(np.linalg.norm(w18[None, :] - np.einsum("bai,bij,bj->ba", geo["J"], geo["ginv"],
                        np.einsum("bai,a->bi", geo["J"], w18)), axis=1)) / np.linalg.norm(w18))
        print(f"\n[{name}] global OOF R2 {out['global_oof_r2']:.3f}; local R2 p05/p50 {np.nanpercentile(r2, 5):.3f}/{np.nanpercentile(r2, 50):.3f}; "
              f"|w_N|/|w| median {wn_frac:.3f}; theorem check rho(pred_local_r2, measured local R2) = {rho_r2['pred_local_r2']:+.3f}")
        print(f"{'column':15s} {'rho vs R2':>10s} {'partial':>8s} {'p':>8s} | {'rho vs ||H_tan||':>16s} {'rho vs log r':>12s} | {'p50':>10s}")
        for c in COLUMNS:
            pr = partials[c]
            print(f"{c:15s} {rho_r2[c]:+10.3f} {pr['partial']:+8.3f} {pr['p']:8.4f} | {rho_H[c]:+16.3f} {rho_logr[c]:+12.3f} | {np.nanmedian(cols[c]):10.4g}")
        fx._append({"experiment": EXPERIMENT, "row": "result", "mode": args.mode, "gamma": gamma, "label": name, "timestamp": _utc_now(),
                    "d": d, "D": D, "n": n, "k": k, "n_anchors": n_anchors, "global_oof_r2": out["global_oof_r2"],
                    "local_r2_p05_p50_p95": [float(v) for v in np.nanpercentile(r2, [5, 50, 95])],
                    "coupling_rho_H_tan_log_r": fx._spearman(cols["exact_H_tan"], log_r), "w_N_fraction_median": wn_frac,
                    "rho_vs_local_r2": rho_r2, "rho_vs_exact_H_tan": rho_H, "rho_vs_log_r": rho_logr,
                    "partials": {c: {k_: v for k_, v in partials[c].items() if k_ != "reason"} for c in COLUMNS},
                    "column_medians": {c: float(np.nanmedian(cols[c])) for c in COLUMNS},
                    "s2_p50": float(np.median(cols["_s2"]))}, record_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--gammas", type=str, default="-1,0,0.4,0.6,0.8,1")
    p.add_argument("--labels", type=str, default="intrinsic_linear,nonlinear,ambient_linear_null")
    p.add_argument("--pool-multiple", type=int, default=3)
    p.add_argument("--record-path", type=str, default=str(DEFAULT_RECORD_PATH))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--n-permutations", type=int, default=None)
    args = p.parse_args()
    assert runner._THREADS == args.threads, (runner._THREADS, args.threads)
    record_path = Path(args.record_path).resolve()
    fx._refuse_production_record(record_path)
    cfg = fx.SMOKE if args.mode == "smoke" else fx.FULL
    n_perm = args.n_permutations if args.n_permutations is not None else (200 if args.mode == "smoke" else 2000)
    gammas = [float(g) for g in args.gammas.split(",")]
    print(f"record -> {record_path}\nNOT PRE-REGISTERED; GATES NOTHING.\nmode={args.mode} gammas={gammas} n_perm={n_perm}")
    fx._append({"experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": _utc_now(),
                "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "threads": args.threads, "seed": args.seed, "gammas": gammas,
                "n_permutations": n_perm, "numpy": np.__version__, "python": sys.version.split()[0], "fd_step": fx.FD_STEP,
                "columns": COLUMNS, "pre_registered": False, "gates": "nothing"}, record_path)
    d, D = cfg["d"], cfg["D"]
    G = adj.InSphereGenerator(d, D, cfg["a"], cfg["bump_widths"], cfg["bump_amps"], seed=args.seed)
    z_pool = adj.draw_latents(args.pool_multiple * cfg["n"], d, cfg["scale_choices"], cfg["scale_probs"], seed=args.seed + 1)
    t0 = time.monotonic()
    pool = {"z": z_pool, **fx.fd_H_tan_norm(G, z_pool)}
    print(f"pool: {len(z_pool)} latents, FD ||H_tan|| in {time.monotonic() - t0:.1f}s", flush=True)
    for gamma in gammas:
        run_gamma(gamma, cfg, pool, G, args, record_path, n_perm)
    print("\nDONE")


if __name__ == "__main__":
    main()
