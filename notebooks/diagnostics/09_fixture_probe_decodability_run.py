"""Curvature versus density on a known surface: the Phase 9 probe pipeline on the sphere fixture.

PURPOSE. Notebooks 09.1 and 09.2 ran the sealed Phase 9 pipeline (OOF ridge probe, k-NN local
R^2, three controls, controlled partial Spearman, Freedman-Lane null) on the Swiss roll with
EXACT curvature and found (a) an intrinsic-linear label gives a negative curvature partial that
is stable when the density-curvature coupling is flipped, and (b) the ridge null (an
ambient-linear label at alpha=100) carries a partial that tracks the coupling sign for sign:
density leakage past the log-radius control, separable from curvature. The roll cannot go
further: it is a one-parameter surface, so curvature, density and position are the same
variable, and it lives in R^3 where neither production curvature instrument is in regime.

This runner repeats the design on the Phase 9 adjudication fixture (``InSphereGenerator`` from
``09_instrument_adjudication_run.py``, unchanged): unit sphere in R^768, d=16, n=86,471,
k=2048, 512 holdout anchors, the production regime. Three things the roll could not do:

  1. Set the density-curvature coupling by construction. A latent pool of ``pool_multiple * n``
     points is drawn from the fixture's scale mixture; for each ``gamma`` in ``--gammas`` the
     sample is a weighted subsample WITHOUT replacement with weights ``||H_tan||^gamma`` (gamma=0
     is a uniform subsample). Same surface, same generator, only the sampling changes.
  2. Score four curvature columns on the same anchors: exact pointwise ``||H_tan||`` (sealed
     autodiff of the generator at the anchor's own latent), exact PATCH-MEAN ``||H_tan||`` over
     the anchor's k neighbours, the decoder instrument (Amendment 01, frozen fit protocol) and
     the colleague's split-half quadratic ``K_H^cross`` (his code, unchanged). Pointwise-vs-patch
     tests the scale explanation; exact-vs-decoder tests the instrument explanation.
  3. Three label arms with known structure: ambient-linear (``y = w . x``, the ridge null),
     intrinsic-linear (``y = a . z``) and a smooth nonlinear function of z whose ridge fit is
     deliberately imperfect, so the "probe fits where data is dense" mechanism can appear.

EXACT CURVATURE FOR THE POOL. The sealed autodiff costs 0.14-0.5 s per point at 16 threads, too
slow for a 260k pool. Curvature is invariant under the fixed rotation Q, and the unrotated map
``f(z) = normalize([stereo(z); a*bumps(z)])`` lives in R^(d+2), so ``||H_tan||`` is computed for
the pool by batched central finite differences of f in R^(d+2) with an independent numpy
implementation of ``H = tr_g(II)``. At the 512 anchors the run compares that number with the
sealed autodiff value and records the agreement; the anchor column ``exact_point`` is the sealed
autodiff value, the pool weights and the ``exact_patch`` column use the finite-difference field.

NOT PRE-REGISTERED, GATES NOTHING. Diagnostic, additive to the record; no sealed constant is
reinterpreted and no Phase 9 verdict depends on it.

Usage:
    python notebooks/diagnostics/09_fixture_probe_decodability_run.py --mode smoke --skip-colleague
    python notebooks/diagnostics/09_fixture_probe_decodability_run.py --mode full --gammas -1,0,1 \\
        --colleague-root <root> --threads 16
    python notebooks/diagnostics/09_fixture_probe_decodability_run.py --mode full --skip-decoder --skip-colleague
"""

import importlib.util
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_ADJ_PATH = DIAGNOSTICS_ROOT / "09_instrument_adjudication_run.py"

# The adjudication runner loads the colleague runner, which loads the production runner, which
# applies the `--threads` cap before numpy/torch are imported. Same mechanism, called not copied.
_spec = importlib.util.spec_from_file_location("instrument_adjudication_run", _ADJ_PATH)
adj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adj)
colleague, runner = adj.colleague, adj.runner

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pu_manifold import decoder_curvature  # noqa: E402
from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

EXPERIMENT = "fixture-probe-decodability"
DEFAULT_RECORD_PATH = NOTEBOOK_ROOT / ".cache" / "09_fixture_probe_decodability.jsonl"
PRODUCTION_STEMS = ("09_physics_curvature", "09_colleague_estimator", "09_instrument_adjudication")

FULL = {**adj.FIXTURE}
SMOKE = {**adj.SMOKE}
FD_STEP = 1e-4
FD_CHUNK = 4096
LABEL_NAMES = ("ambient_linear_null", "intrinsic_linear", "nonlinear")
COLUMN_NAMES = ("exact_point", "exact_patch", "decoder_H_tan", "colleague_K_H_cross")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")


def _refuse_production_record(path: Path) -> None:
    for stem in PRODUCTION_STEMS:
        if path.name.startswith(stem):
            raise SystemExit(f"refusing to write to a Phase 9 production record path: {path}")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


# --- exact curvature of the unrotated map by finite differences ------------------------------


def _unrotated_map(G: Any) -> Any:
    """f(z) = normalize([stereo(z); a * bumps(z)]) in R^(d+2), numpy, batched. G's own buffers."""
    C = G.centres.numpy(); W = G.widths.numpy(); A = G.amps.numpy(); a = G.a

    def f(z: np.ndarray) -> np.ndarray:
        s = (z * z).sum(1, keepdims=True)
        st = np.concatenate([2 * z, s - 1], 1) / (1 + s)
        sq = ((z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        bump = (np.exp(-sq / (2 * W ** 2)) * A).sum(-1, keepdims=True)
        v = np.concatenate([st, a * bump], 1)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    return f


def fd_H_tan_norm(G: Any, z: np.ndarray, h: float = FD_STEP, chunk: int = FD_CHUNK) -> Dict[str, np.ndarray]:
    """||H_tan|| and H_rad of the fixture surface at every row of z, by central finite differences
    of the unrotated map and an independent numpy H = tr_g(II). Rotation-invariant, so equal to
    the sealed autodiff value on G up to finite-difference error (~1e-6 relative)."""
    f = _unrotated_map(G)
    z = np.asarray(z, dtype=np.float64)
    n, d = z.shape
    E = np.eye(d)
    H_tan_norm = np.empty(n); H_rad = np.empty(n)
    for start in range(0, n, chunk):
        zb = z[start:start + chunk]
        b = zb.shape[0]
        x0 = f(zb)                                                        # (b, d+2)
        plus = [f(zb + h * E[i]) for i in range(d)]
        minus = [f(zb - h * E[i]) for i in range(d)]
        J = np.stack([(plus[i] - minus[i]) / (2 * h) for i in range(d)], -1)   # (b, d+2, d)
        Hs = np.zeros((b, x0.shape[1], d, d))
        for i in range(d):
            Hs[:, :, i, i] = (plus[i] - 2 * x0 + minus[i]) / h ** 2
            for j in range(i + 1, d):
                v = (f(zb + h * E[i] + h * E[j]) - f(zb + h * E[i] - h * E[j])
                     - f(zb - h * E[i] + h * E[j]) + f(zb - h * E[i] - h * E[j])) / (4 * h * h)
                Hs[:, :, i, j] = v; Hs[:, :, j, i] = v
        g = np.einsum("bai,baj->bij", J, J)
        ginv = np.linalg.inv(g)
        tr = np.einsum("bij,baij->ba", ginv, Hs)                            # sum_ij g^ij d_i d_j f
        Hn = tr - np.einsum("bai,bi->ba", J, np.einsum("bij,bj->bi", ginv, np.einsum("bai,ba->bi", J, tr)))
        hr = np.einsum("ba,ba->b", Hn, x0)
        H_rad[start:start + b] = hr
        H_tan_norm[start:start + b] = np.linalg.norm(Hn - hr[:, None] * x0, axis=1)
    return {"H_tan_norm": H_tan_norm, "H_rad": H_rad}


# --- sampling, labels, columns ----------------------------------------------------------------


def weighted_subsample(H_pool: np.ndarray, gamma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices into the pool: n draws without replacement with p ∝ ||H_tan||^gamma, ||H_tan||
    clipped to its [p05, p95] band so a near-zero curvature cannot dominate the weights."""
    lo, hi = np.percentile(H_pool, [5, 95])
    w = np.clip(H_pool, lo, hi) ** gamma
    return np.sort(rng.choice(H_pool.shape[0], size=n, replace=False, p=w / w.sum()))


def make_labels(z: np.ndarray, X: np.ndarray, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    d, D = z.shape[1], X.shape[1]
    a1 = rng.standard_normal(d); a1 /= np.linalg.norm(a1)
    a2 = rng.standard_normal(d); a2 /= np.linalg.norm(a2)
    w = rng.standard_normal(D); w /= np.linalg.norm(w)
    u1, u2 = z @ a1, z @ a2
    return {
        "ambient_linear_null": X @ w,
        "intrinsic_linear": u1,
        "nonlinear": np.sin(2.0 * u1) + u2 ** 2 - 0.5 * u1 * u2,
    }


def probe_outcome(X: np.ndarray, y: np.ndarray, panel: Dict[str, np.ndarray]) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    y_hat = pcp.oof_ridge_predictions(X, y, pcp.ALPHA_RIDGE, pcp.N_OOF_FOLDS, pcp.OOF_FOLD_SEED)
    loc = pcp.local_r2_panel(y, y_hat, panel["indices"], pcp.MIN_FINITE_NEIGHBOURS)
    Z = np.column_stack([panel["log_knn_radius"], loc["local_label_variance"], loc["local_evaluation_count"]])
    global_r2 = 1.0 - float(np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"r2": loc["r2"], "Z": Z, "global_oof_r2": global_r2, "n_masked": int(loc["n_masked_anchors"])}


def partial_row(x: np.ndarray, r2: np.ndarray, Z: np.ndarray, n_perm: int) -> Dict[str, Any]:
    x = np.asarray(x, float)
    m = np.isfinite(x) & np.isfinite(r2) & np.all(np.isfinite(Z), axis=1)
    out: Dict[str, Any] = {"n_finite": int(m.sum()), "raw_rho": _spearman(x[m], r2[m])}
    try:
        fw = pcp.permutation_fwer({0: x[m]}, r2[m], Z[m], n_perm, pcp.PERMUTATION_SEED)
        out.update({"partial": fw["per_d"][0]["observed_rho"], "p": fw["per_d"][0]["p"],
                    "p_display": fw["per_d"][0]["p_display"], "undefined": False})
    except ValueError as e:
        out.update({"partial": float("nan"), "p": float("nan"), "p_display": "undefined", "undefined": True, "reason": str(e)})
    return out


# --- one gamma ---------------------------------------------------------------------------------


def run_gamma(gamma: float, cfg: Dict[str, Any], pool: Dict[str, Any], G: Any, args: argparse.Namespace,
              est: Optional[Dict[str, Any]], record_path: Path, max_epochs: int, n_perm: int) -> None:
    d, D, n, k, n_anchors = cfg["d"], cfg["D"], cfg["n"], cfg["k"], cfg["n_anchors"]
    print("\n" + "=" * 78 + f"\ngamma={gamma:+g}: weighted subsample of the pool, n={n}, d={d}, D={D}, k={k}, anchors={n_anchors}\n" + "=" * 78, flush=True)
    rng = np.random.default_rng(args.seed + 100 + int(round(10 * gamma)))
    sel = weighted_subsample(pool["H_tan_norm"], gamma, n, rng)
    z = pool["z"][sel]
    H_fd = pool["H_tan_norm"][sel]
    X = adj.generate_points(G, z)

    split = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)
    a = split["anchor_idx"]
    panel = pcp.knn_panel(X, a, k)
    log_r = panel["log_knn_radius"]
    rr = adj.measure_r_over_R(X, panel["distances"][:, -1])

    t0 = time.monotonic()
    truth = adj.exact_truth_at_anchors(G, z[a])
    t_truth = time.monotonic() - t0
    fd_vs_auto_rel = np.abs(H_fd[a] - truth["H_tan_norm"]) / np.maximum(truth["H_tan_norm"], 1e-12)
    print(f"truth at anchors {t_truth:.1f}s; FD vs autodiff ||H_tan||: max rel err {fd_vs_auto_rel.max():.2e}, "
          f"rank rho {_spearman(H_fd[a], truth['H_tan_norm']):.6f}; FD H_rad max|.+d| {np.max(np.abs(pool['H_rad'][sel][a] + d)):.2e}")

    columns: Dict[str, np.ndarray] = {
        "exact_point": truth["H_tan_norm"],
        "exact_patch": H_fd[panel["indices"]].mean(axis=1),
    }
    fit_info: Dict[str, Any] = {}
    if not args.skip_decoder:
        print(f"[decoder] fit_and_field_at_anchors d={d} in_dim={D} max_epochs={max_epochs} ...", flush=True)
        ours = runner.fit_and_field_at_anchors(
            X, d, a, in_dim=D, hidden=pcp.AE_HIDDEN, activation=pcp.AE_ACTIVATION, train_cfg=pcp.TRAIN_CFG,
            max_epochs=max_epochs, torch_init_seed=pcp.TORCH_INIT_SEED, split_seed=pcp.SPLIT_SEED,
            holdout_fraction=pcp.HOLDOUT_FRACTION,
        )
        columns["decoder_H_tan"] = np.linalg.norm(adj._tangential(ours["H_vec"], ours["image"]), axis=1)
        fit_info["decoder"] = {"var_explained": float(ours["var_explained"]), "wallclock_fit_s": float(ours["wallclock_fit_s"]),
                               "wallclock_field_s": float(ours["wallclock_field_s"]), "max_epochs": max_epochs,
                               "metric_condition_number_p50_p95": [float(v) for v in np.percentile(ours["metric_condition_number"], [50, 95])],
                               "fidelity_vs_truth": adj.score_ours(adj._tangential(ours["H_vec"], ours["image"]), truth["H_tan_vec"], log_r)}
        print(f"[decoder] var_explained={ours['var_explained']:.5f} fit {ours['wallclock_fit_s']:.0f}s; "
              f"rank vs truth {fit_info['decoder']['fidelity_vs_truth']['rank_spearman_rho']:.3f}", flush=True)
    if not args.skip_colleague:
        print(f"[colleague] k={k} d={d} n_splits={colleague.COLLEAGUE_N_SPLITS} ...", flush=True)
        his = adj.colleague_field(X, a, k, d, est, torch.device(args.device))
        columns["colleague_K_H_cross"] = his["K_H_cross"]
        fit_info["colleague"] = {"R_H_median": float(np.nanmedian(his["R_H"])), "wallclock_s": float(his["wallclock_s"]),
                                 "rank_vs_truth": _spearman(his["K_H_cross"], truth["H_tan_norm"] ** 2 / d ** 2)}
        print(f"[colleague] R_H median={fit_info['colleague']['R_H_median']:.3f} {his['wallclock_s']:.0f}s; "
              f"rank vs truth {fit_info['colleague']['rank_vs_truth']:.3f}", flush=True)

    coupling = {name: _spearman(col, log_r) for name, col in columns.items()}
    cross = {f"{p}_vs_{q}": _spearman(columns[p], columns[q]) for i, p in enumerate(columns) for q in list(columns)[i + 1:]}
    sample_row = {
        "experiment": EXPERIMENT, "row": "sample", "mode": args.mode, "gamma": gamma, "timestamp": _utc_now(),
        "d": d, "D": D, "n": n, "k": k, "n_anchors": n_anchors, "pool_size": int(pool["z"].shape[0]),
        **{f"regime_{k_}": v for k_, v in rr.items()},
        "truth_H_tan_norm_p05_p50_p95": [float(v) for v in np.percentile(truth["H_tan_norm"], [5, 50, 95])],
        "sample_H_tan_norm_fd_p05_p50_p95": [float(v) for v in np.percentile(H_fd, [5, 50, 95])],
        "fd_vs_autodiff_max_rel_err": float(fd_vs_auto_rel.max()), "fd_vs_autodiff_rank_rho": _spearman(H_fd[a], truth["H_tan_norm"]),
        "truth_max_abs_H_rad_plus_d": float(truth["max_abs_H_rad_plus_d"]), "wallclock_truth_s": t_truth,
        "coupling_rho_vs_log_knn_radius": coupling, "column_cross_rank_rho": cross, "fit_info": fit_info,
    }
    _append(sample_row, record_path)
    print("coupling rho(column, log r): " + "  ".join(f"{k_}={v:+.3f}" for k_, v in coupling.items()))
    print("cross-column rank rho:       " + "  ".join(f"{k_}={v:+.3f}" for k_, v in cross.items()))

    labels = make_labels(z, X, np.random.default_rng(args.seed + 7))
    hdr = f"{'label':22s} {'global R2':>9s} {'r2 p05/p50':>13s} | " + " | ".join(f"{c:>21s}" for c in columns)
    print("\n" + hdr)
    for name in LABEL_NAMES:
        out = probe_outcome(X, labels[name], panel)
        cells = []
        for col_name, col in columns.items():
            pr = partial_row(col, out["r2"], out["Z"], n_perm)
            _append({"experiment": EXPERIMENT, "row": "partial", "mode": args.mode, "gamma": gamma, "label": name,
                     "column": col_name, "timestamp": _utc_now(), "global_oof_r2": out["global_oof_r2"],
                     "local_r2_p05_p50_p95": [float(v) for v in np.nanpercentile(out["r2"], [5, 50, 95])],
                     "n_masked_anchors": out["n_masked"], "coupling_rho_vs_log_knn_radius": coupling[col_name], **pr}, record_path)
            cells.append(f"{pr['raw_rho']:+6.3f} {pr['partial']:+7.3f} p={pr['p']:.4f}" if not pr["undefined"] else f"{'undefined':>21s}")
        q = np.nanpercentile(out["r2"], [5, 50])
        print(f"{name:22s} {out['global_oof_r2']:9.3f} {q[0]:6.3f}/{q[1]:6.3f} | " + " | ".join(cells), flush=True)


# --- main ----------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--gammas", type=str, default="-1,0,1", help="comma-separated density exponents")
    p.add_argument("--pool-multiple", type=int, default=3, help="pool size as a multiple of n")
    p.add_argument("--colleague-root", type=str, default=None, help="read-only checkout at COLLEAGUE_COMMIT")
    p.add_argument("--skip-decoder", action="store_true")
    p.add_argument("--skip-colleague", action="store_true")
    p.add_argument("--record-path", type=str, default=str(DEFAULT_RECORD_PATH))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=20260905, help="fixture seed; the generator is the adjudication generator")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--n-permutations", type=int, default=None, help=f"default frozen N_PERMUTATIONS={pcp.N_PERMUTATIONS} (smoke: 200)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    assert runner._THREADS == args.threads, (runner._THREADS, args.threads)
    if not args.skip_colleague and args.colleague_root is None:
        raise SystemExit("--colleague-root is required unless --skip-colleague")
    record_path = Path(args.record_path).resolve()
    _refuse_production_record(record_path)
    cfg = SMOKE if args.mode == "smoke" else FULL
    max_epochs = args.max_epochs if args.max_epochs is not None else (adj.SMOKE_EPOCHS if args.mode == "smoke" else pcp.MAX_EPOCHS)
    n_perm = args.n_permutations if args.n_permutations is not None else (200 if args.mode == "smoke" else pcp.N_PERMUTATIONS)
    gammas = [float(g) for g in args.gammas.split(",")]
    print(f"record -> {record_path}\nNOT PRE-REGISTERED; GATES NOTHING.\nmode={args.mode} gammas={gammas} max_epochs={max_epochs} n_perm={n_perm}")

    est = None
    if not args.skip_colleague:
        est = colleague.load_colleague_estimator(args.colleague_root)
        print(f"colleague checkout HEAD={est['colleague_head']} (expected {colleague.COLLEAGUE_COMMIT}); topology shim={est['topology_is_shim']}")
    _append({"experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": _utc_now(),
             "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "colleague_head": est["colleague_head"] if est else None,
             "threads": args.threads, "device": args.device, "seed": args.seed, "gammas": gammas, "pool_multiple": args.pool_multiple,
             "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
             "decoder_image_projection": pcp.DECODER_IMAGE_PROJECTION, "curvature_convention": decoder_curvature.CURVATURE_CONVENTION,
             "fd_step": FD_STEP, "label_names": LABEL_NAMES, "pre_registered": False, "gates": "nothing"}, record_path)

    d, D = cfg["d"], cfg["D"]
    G = adj.InSphereGenerator(d, D, cfg["a"], cfg["bump_widths"], cfg["bump_amps"], seed=args.seed)
    n_pool = args.pool_multiple * cfg["n"]
    z_pool = adj.draw_latents(n_pool, d, cfg["scale_choices"], cfg["scale_probs"], seed=args.seed + 1)
    t0 = time.monotonic()
    fd = fd_H_tan_norm(G, z_pool)
    print(f"pool: {n_pool} latents, FD ||H_tan|| in {time.monotonic() - t0:.1f}s; p05/p50/p95 = "
          f"{np.percentile(fd['H_tan_norm'], [5, 50, 95]).round(4)}; max|H_rad + d| = {np.max(np.abs(fd['H_rad'] + d)):.2e}", flush=True)
    pool = {"z": z_pool, **fd}

    for gamma in gammas:
        run_gamma(gamma, cfg, pool, G, args, est, record_path, max_epochs, n_perm)
    print("\nDONE")


if __name__ == "__main__":
    main()
