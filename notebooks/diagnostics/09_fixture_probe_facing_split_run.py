"""Probe-facing curvature on the sphere fixture, decomposed into in-sphere and radial parts, with alignment.

PURPOSE. ``09_fixture_probe_facing_run.py`` computed ``pf_curv = |<w_N, II>|_g`` from the full
second fundamental form of the in-sphere generator. On the unit sphere ``<x0, II_ij> = -g_ij``, so
``<w_N, II> = <w_N, II_tan> - (w.x0) g``; the second term is the prediction level times the metric.
This runner rebuilds the same samples, labels, probe and geometry bit-for-bit (by importing that
runner unchanged) and adds: ``pf_tan = |<w_N, II_tan>|_g``, ``pf_rad = sqrt(d) |w.x0|``, and the
alignment ``align_cos = cos_g(Hess_M y, <w_N, II>)`` (and its in-sphere variant), plus the signed
cross term ``<Hess_M y, <w_N, II>>_g`` that the residual expansion's ``|Delta|^2`` contains.

NOT PRE-REGISTERED, GATES NOTHING.

Usage:
    python notebooks/diagnostics/09_fixture_probe_facing_split_run.py --mode smoke --threads 8
    python notebooks/diagnostics/09_fixture_probe_facing_split_run.py --mode full --gammas -1,0.6 --threads 16
"""

import importlib.util
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_PF_PATH = DIAGNOSTICS_ROOT / "09_fixture_probe_facing_run.py"
_spec = importlib.util.spec_from_file_location("fixture_probe_facing_run", _PF_PATH)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)
fx, adj, runner = pf.fx, pf.adj, pf.runner

import argparse  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

EXPERIMENT = "fixture-probe-facing-split"
DEFAULT_RECORD_PATH = NOTEBOOK_ROOT / ".cache" / "09_fixture_probe_facing_split.jsonl"
COLUMNS = ("exact_H_tan", "pf_full", "pf_tan", "pf_rad", "hess_label", "hess_mismatch", "align_cos_full", "align_cos_tan",
           "cross_full", "cross_tan", "bias_sq")


def g_inner(A: np.ndarray, B: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    return np.einsum("bij,bjk,bkl,bli->b", ginv, A, ginv, B)


def split_columns(geo: Dict[str, np.ndarray], w18: np.ndarray, b0: float, y_anchor: np.ndarray, x18_anchor: np.ndarray,
                  lab: Dict[str, np.ndarray], d: int) -> Dict[str, Any]:
    J, ginv, II, Gamma, H, x0 = geo["J"], geo["ginv"], geo["II"], geo["Gamma"], geo["H"], geo["x0"]
    g = np.einsum("bai,baj->bij", J, J)
    w_T_coef = np.einsum("bai,a->bi", J, w18)
    w_N = w18[None, :] - np.einsum("bai,bij,bj->ba", J, ginv, w_T_coef)
    II_rad = np.einsum("baij,ba->bij", II, x0)
    II_tan = II - np.einsum("ba,bij->baij", x0, II_rad)
    hp_full = np.einsum("baij,ba->bij", II, w_N)
    hp_tan = np.einsum("baij,ba->bij", II_tan, w_N)
    w_rad = np.einsum("ba,ba->b", w_N, x0)
    hp_rad = w_rad[:, None, None] * II_rad
    hess_label = lab["hess"] - np.einsum("bkij,bk->bij", Gamma, lab["grad"])
    H_rad = np.einsum("ba,ba->b", H, x0)
    H_tan = H - H_rad[:, None] * x0
    nf = pf.metric_norms(hp_full, ginv)["fro"]; nt = pf.metric_norms(hp_tan, ginv)["fro"]; nr = pf.metric_norms(hp_rad, ginv)["fro"]
    ny = pf.metric_norms(hess_label, ginv)["fro"]
    cf = g_inner(hess_label, hp_full, ginv); ct = g_inner(hess_label, hp_tan, ginv)
    c = y_anchor - (x18_anchor @ w18 + b0)
    cols = {"exact_H_tan": np.linalg.norm(H_tan, axis=1), "pf_full": nf, "pf_tan": nt, "pf_rad": nr, "hess_label": ny,
            "hess_mismatch": pf.metric_norms(hess_label - hp_full, ginv)["fro"],
            "align_cos_full": cf / np.maximum(ny * nf, 1e-300), "align_cos_tan": ct / np.maximum(ny * nt, 1e-300),
            "cross_full": cf, "cross_tan": ct, "bias_sq": c ** 2}
    checks = {"II_rad_vs_minus_g_max_rel": float(np.max(np.abs(II_rad + g) / np.maximum(np.abs(g).max(axis=(1, 2))[:, None, None], 1e-300))),
              "H_rad_median": float(np.median(H_rad)), "w_N_fraction_median": float(np.median(np.linalg.norm(w_N, axis=1)) / np.linalg.norm(w18)),
              "w_rad_median_abs": float(np.median(np.abs(w_rad))), "rank_pf_full_vs_pf_tan": fx._spearman(nf, nt),
              "rank_pf_full_vs_pf_rad": fx._spearman(nf, nr)}
    return {"cols": cols, "checks": checks}


BUMP_FLIP = np.array([1.0, -1.0, -1.0, 1.0])


def bumpalt(G: Any, z: np.ndarray, beta: float) -> Dict[str, np.ndarray]:
    """Alignment-demo label piece beta*h_alt(z): the generator's Gaussian bumps with two amplitude signs
    flipped, so the label's Hessian matches the surface's in-sphere bending near two bumps and opposes it
    near the other two. Not linearly readable from the ambient coordinates. Analytic chart derivatives."""
    c = G.centres.numpy(); w = G.widths.numpy(); A = G.amps.numpy() * BUMP_FLIP[: len(w)]
    diff = z[:, None, :] - c[None, :, :]                                              # (b, m, d)
    e = np.exp(-(diff ** 2).sum(-1) / (2 * w ** 2)) * A                              # (b, m)
    val = beta * e.sum(-1)
    grad = beta * np.einsum("bm,bmi->bi", e, -diff / (w ** 2)[None, :, None])
    hess = beta * (np.einsum("bm,bmi,bmj->bij", e / (w ** 4)[None, :], diff, diff)
                   - np.einsum("bm,ij->bij", e / (w ** 2)[None, :], np.eye(z.shape[1])))
    return {"val": val, "grad": grad, "hess": hess}


def run_gamma(gamma: float, cfg: Dict[str, Any], pool: Dict[str, Any], G: Any, args: argparse.Namespace,
              record_path: Path, n_perm: int) -> None:
    d, D, n, k, n_anchors = cfg["d"], cfg["D"], cfg["n"], cfg["k"], cfg["n_anchors"]
    print("\n" + "=" * 78 + f"\ngamma={gamma:+g}: n={n}, d={d}, D={D}, k={k}, anchors={n_anchors}\n" + "=" * 78, flush=True)
    rng = np.random.default_rng(args.seed + 100 + int(round(10 * gamma)))
    sel = fx.weighted_subsample(pool["H_tan_norm"], gamma, n, rng)
    z = pool["z"][sel]
    X = adj.generate_points(G, z)
    split = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)
    a = split["anchor_idx"]
    panel = pcp.knn_panel(X, a, k)
    log_r = panel["log_knn_radius"]
    geo = pf.fd_geometry(G, z[a])
    Q = G.Q.numpy(); Qt18 = Q.T[: d + 2]; x18 = geo["x0"]
    H_fd_norm = np.linalg.norm(geo["H"] - np.einsum("ba,ba->b", geo["H"], x18)[:, None] * x18, axis=1)
    print(f"coupling rho(||H_tan||, log r) = {spearmanr(H_fd_norm, log_r).statistic:+.3f}", flush=True)
    lrng = np.random.default_rng(args.seed + 7)
    a1 = lrng.standard_normal(d); a1 /= np.linalg.norm(a1)
    a2 = lrng.standard_normal(d); a2 /= np.linalg.norm(a2)
    w_true = lrng.standard_normal(D); w_true /= np.linalg.norm(w_true)
    labels = fx.make_labels(z, X, np.random.default_rng(args.seed + 7))
    for name in args.labels.split(","):
        if name.startswith("bumpalt_"):
            beta = float(name.split("_", 1)[1])
            labels[name] = z @ a1 + bumpalt(G, z, beta)["val"]
        y = labels[name]
        out = fx.probe_outcome(X, y, panel)
        r2, Z = out["r2"], out["Z"]
        ridge = Ridge(alpha=pcp.ALPHA_RIDGE).fit(X, y)
        w18 = Qt18 @ ridge.coef_
        if name.startswith("bumpalt_"):
            ba = bumpalt(G, z[a], float(name.split("_", 1)[1]))
            lab = {"grad": ba["grad"] + a1[None, :], "hess": ba["hess"]}
        else:
            lab = pf.label_derivatives(name, z[a], X[a], geo, Qt18, a1, a2, w_true)
        sc = split_columns(geo, w18, float(ridge.intercept_), y[a], x18, lab, d)
        cols = sc["cols"]
        m = np.isfinite(r2)
        rho_r2 = {c: fx._spearman(cols[c][m], r2[m]) for c in COLUMNS}
        rho_H = {c: fx._spearman(cols[c], cols["exact_H_tan"]) for c in COLUMNS}
        rho_logr = {c: fx._spearman(cols[c], log_r) for c in COLUMNS}
        partials = {c: fx.partial_row(cols[c], r2, Z, n_perm) for c in COLUMNS}
        print(f"\n[{name}] global OOF R2 {out['global_oof_r2']:.3f}; checks {', '.join(f'{k_}={v:+.3g}' for k_, v in sc['checks'].items())}")
        print(f"{'column':15s} {'rho vs R2':>10s} {'partial':>8s} {'p':>8s} | {'rho vs ||H_tan||':>16s} {'rho vs log r':>12s} | {'p50':>10s}")
        for c in COLUMNS:
            pr = partials[c]
            print(f"{c:15s} {rho_r2[c]:+10.3f} {pr['partial']:+8.3f} {pr['p']:8.4f} | {rho_H[c]:+16.3f} {rho_logr[c]:+12.3f} | {np.nanmedian(cols[c]):10.4g}", flush=True)
        fx._append({"experiment": EXPERIMENT, "row": "result", "mode": args.mode, "gamma": gamma, "label": name, "timestamp": pf._utc_now(),
                    "d": d, "D": D, "n": n, "k": k, "n_anchors": n_anchors, "global_oof_r2": out["global_oof_r2"],
                    "coupling_rho_H_tan_log_r": fx._spearman(cols["exact_H_tan"], log_r), "checks": sc["checks"],
                    "rho_vs_local_r2": rho_r2, "rho_vs_exact_H_tan": rho_H, "rho_vs_log_r": rho_logr,
                    "partials": {c: {k_: v for k_, v in partials[c].items() if k_ != "reason"} for c in COLUMNS},
                    "column_medians": {c: float(np.nanmedian(cols[c])) for c in COLUMNS},
                    "align_cos_full_p25_p50_p75": [float(v) for v in np.nanpercentile(cols["align_cos_full"], [25, 50, 75])],
                    "align_cos_tan_p25_p50_p75": [float(v) for v in np.nanpercentile(cols["align_cos_tan"], [25, 50, 75])]}, record_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--gammas", type=str, default="-1,0.6")
    p.add_argument("--labels", type=str, default="intrinsic_linear,nonlinear")
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
    fx._append({"experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": pf._utc_now(),
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
