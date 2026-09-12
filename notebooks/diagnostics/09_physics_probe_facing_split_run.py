"""Probe-facing curvature on the Physics anchors, decomposed, with the label's intrinsic Hessian from data.

PURPOSE. Supplement 06 computed ``pf_curv = |<w_N, II>|_g`` from the decoder's full second
fundamental form. On the unit sphere that II contains the sphere's own radial part,
``<x_hat, II_ij> = -g_ij``, so ``<w_N, II> = <w_N, II_tan> - (w.x_hat) g``: the second term is the
prediction level at the anchor times the metric, not the shape of the embedding manifold. This
runner (i) splits the probe-facing curvature into its in-sphere part ``pf_tan = |<w_N, II_tan>|_g``
and its radial part ``pf_rad = sqrt(d) |w.x_hat|`` and re-runs the partials for each, and (ii)
estimates the label's intrinsic Hessian ``Hess_M y`` per anchor from the data by a local quadratic
regression of ``y`` on the tangent-projected coordinates ``u = g^{-1} J^T (x - x0)`` of the anchor's
k neighbours (in those coordinates the fitted quadratic coefficient IS the covariant Hessian: the
Christoffel correction is absorbed because ``x - x0 = J u + II(u,u)/2 + ...`` and II is normal).
With ``Hess_M y`` in hand the residual expansion's variable ``Delta = Hess_M y - <w_N, II>`` and
the alignment ``cos_g(Hess_M y, <w_N, II>)`` are available on real data; the same regression
applied to the probe's own prediction ``w.x`` gives a data-side estimate of ``<w, II_data>`` that
checks the decoder's II in the probe direction against the point cloud itself.

Geometry at the anchors (J, D^2F, image, latent codes) is read from the npz Supplement 06 saved
under ``<output root>/probe-facing/``; the decoder is not refit. Everything else (embeddings,
labels, anchors, k-NN panel, out-of-fold probe, local R^2, controls, Freedman-Lane partial) is the
production pipeline's own call, reached through ``09_physics_probe_facing_run.py`` unchanged.

NOT PRE-REGISTERED, GATES NOTHING. Writes only to its own record.

Usage:
    python notebooks/diagnostics/09_physics_probe_facing_split_run.py --mode smoke --threads 8
    HF_HOME=... EFFDIM_09_OUTPUT_ROOT=... python notebooks/diagnostics/09_physics_probe_facing_split_run.py \\
        --mode physics --d-values 16,20 --threads 16
"""

import importlib.util
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_PPF_PATH = DIAGNOSTICS_ROOT / "09_physics_probe_facing_run.py"
_spec = importlib.util.spec_from_file_location("physics_probe_facing_run", _PPF_PATH)
ppf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppf)
adj, runner = ppf.adj, ppf.runner

import argparse  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

EXPERIMENT = "physics-probe-facing-split"
DEFAULT_RECORD_PATH = NOTEBOOK_ROOT / ".cache" / "09_physics_probe_facing_split.jsonl"
COLUMNS = ("H_tan_norm", "pf_full", "pf_tan", "pf_rad", "pf_trace_tan",
           "hess_label", "hess_mismatch_dec", "hess_mismatch_emp", "probe_emp",
           "align_cos_full", "align_cos_tan", "cross_full", "cross_tan", "bias_sq")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def geometry_from_arrays(J: np.ndarray, Hess: np.ndarray, image: np.ndarray) -> Dict[str, np.ndarray]:
    """Same algebra as ``ppf.decoder_geometry``, from stored arrays."""
    J = J.astype(np.float64); Hess = Hess.astype(np.float64); image = image.astype(np.float64)
    g = np.einsum("bai,baj->bij", J, J)
    ginv = np.linalg.inv(g)
    Gamma = np.einsum("bkl,bal,baij->bkij", ginv, J, Hess)
    II = Hess - np.einsum("bak,bkij->baij", J, Gamma)
    H = np.einsum("bij,baij->ba", ginv, II)
    return {"J": J, "Hess": Hess, "image": image, "g": g, "ginv": ginv, "II": II, "H": H}


def quad_design(u: np.ndarray) -> np.ndarray:
    """[1, u, u_i u_i / 2 (i), u_i u_j (i<j)] so the fitted coefficients are (c, grad, B_sym)."""
    n, d = u.shape
    iu, ju = np.triu_indices(d)
    quad = u[:, iu] * u[:, ju]
    quad[:, iu == ju] *= 0.5
    return np.concatenate([np.ones((n, 1)), u, quad], axis=1)


def unpack_sym(coef: np.ndarray, d: int) -> np.ndarray:
    iu, ju = np.triu_indices(d)
    B = np.zeros((d, d)); B[iu, ju] = coef; B[ju, iu] = coef
    return B


def local_quadratics(X: np.ndarray, x0_idx: np.ndarray, neigh: np.ndarray, geo: Dict[str, np.ndarray],
                     targets: Dict[str, np.ndarray], min_finite: int) -> Dict[str, Any]:
    """Per anchor: tangent coordinates of the neighbours, then one least-squares quadratic per target.
    Returns Hessians (b,d,d) per target, the quadratic-vs-linear R^2 gain, and the coordinate scale."""
    b, d = geo["J"].shape[0], geo["J"].shape[2]
    out = {name: np.full((b, d, d), np.nan) for name in targets}
    r2_lin = {name: np.full(b, np.nan) for name in targets}; r2_quad = {name: np.full(b, np.nan) for name in targets}
    u_scale = np.full(b, np.nan)
    for i in range(b):
        idx = neigh[i]
        dx = X[idx] - X[x0_idx[i]][None, :]
        u = dx @ geo["J"][i] @ geo["ginv"][i]                                    # (k, d) tangent-projected chart coords
        u_scale[i] = float(np.sqrt(np.einsum("ki,ij,kj->k", u, geo["g"][i], u).mean()))
        A = quad_design(u)
        for name, t in targets.items():
            y = t[idx]
            m = np.isfinite(y)
            if m.sum() < max(min_finite, A.shape[1] + 8):
                continue
            coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            out[name][i] = unpack_sym(coef[1 + d:], d)
            fit = A[m] @ coef
            sst = float(((y[m] - y[m].mean()) ** 2).sum())
            coef_l, *_ = np.linalg.lstsq(A[m, : 1 + d], y[m], rcond=None)
            r2_quad[name][i] = 1.0 - float(((y[m] - fit) ** 2).sum()) / max(sst, 1e-300)
            r2_lin[name][i] = 1.0 - float(((y[m] - A[m, : 1 + d] @ coef_l) ** 2).sum()) / max(sst, 1e-300)
    return {"hess": out, "r2_lin": r2_lin, "r2_quad": r2_quad, "u_scale": u_scale}


def g_inner(A: np.ndarray, B: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    return np.einsum("bij,bjk,bkl,bli->b", ginv, A, ginv, B)


def split_columns(geo: Dict[str, np.ndarray], w: np.ndarray, b0: float, hess_y: np.ndarray, probe_emp: np.ndarray,
                  d: int) -> Dict[str, np.ndarray]:
    J, ginv, g, II, H, x = geo["J"], geo["ginv"], geo["g"], geo["II"], geo["H"], geo["image"]
    xhat = x / np.linalg.norm(x, axis=1, keepdims=True)
    wT = np.einsum("bai,a->bi", J, w)
    w_N = w[None, :] - np.einsum("bai,bij,bj->ba", J, ginv, wT)
    II_rad = np.einsum("baij,ba->bij", II, xhat)                                  # <x_hat, II_ij>, expected -g_ij
    II_tan = II - np.einsum("ba,bij->baij", xhat, II_rad)
    pf_full = np.einsum("baij,ba->bij", II, w_N)
    pf_tan = np.einsum("baij,ba->bij", II_tan, w_N)
    w_rad = np.einsum("ba,ba->b", w_N, xhat)                                       # = w.x_hat (J^T x_hat = 0 on the sphere)
    pf_rad_tensor = w_rad[:, None, None] * II_rad
    H_rad = np.einsum("ba,ba->b", H, xhat)
    H_tan = H - H_rad[:, None] * xhat
    nf = ppf.metric_norms(pf_full, ginv)["fro"]; nt = ppf.metric_norms(pf_tan, ginv)["fro"]
    nr = ppf.metric_norms(pf_rad_tensor, ginv)["fro"]
    ny = ppf.metric_norms(hess_y, ginv)["fro"]; ne = ppf.metric_norms(probe_emp, ginv)["fro"]
    cross_full = g_inner(hess_y, pf_full, ginv); cross_tan = g_inner(hess_y, pf_tan, ginv)
    cols = {
        "H_tan_norm": np.linalg.norm(H_tan, axis=1),
        "pf_full": nf, "pf_tan": nt, "pf_rad": nr,
        "pf_trace_tan": np.einsum("ba,ba->b", w_N, H_tan),
        "hess_label": ny,
        "hess_mismatch_dec": ppf.metric_norms(hess_y - pf_full, ginv)["fro"],
        "hess_mismatch_emp": ppf.metric_norms(hess_y - probe_emp, ginv)["fro"],
        "probe_emp": ne,
        "align_cos_full": cross_full / np.maximum(ny * nf, 1e-300),
        "align_cos_tan": cross_tan / np.maximum(ny * nt, 1e-300),
        "cross_full": cross_full, "cross_tan": cross_tan,
    }
    checks = {
        "II_rad_vs_minus_g_max_rel": float(np.max(np.abs(II_rad + g) / np.maximum(np.abs(g).max(axis=(1, 2))[:, None, None], 1e-300))),
        "JT_xhat_max": float(np.max(np.abs(np.einsum("bai,ba->bi", J, xhat)))),
        "H_rad_median": float(np.median(H_rad)),
        "w_N_fraction_median": float(np.median(np.linalg.norm(w_N, axis=1)) / np.linalg.norm(w)),
        "w_rad_median_abs": float(np.median(np.abs(w_rad))),
        "cos_dec_vs_emp_probe_median": float(np.nanmedian(g_inner(pf_full, probe_emp, ginv) / np.maximum(nf * ne, 1e-300))),
        "cos_dec_tan_vs_emp_probe_median": float(np.nanmedian(g_inner(pf_tan, probe_emp, ginv) / np.maximum(nt * ne, 1e-300))),
        "rank_dec_vs_emp_probe": ppf._spearman(nf, ne), "rank_dec_tan_vs_emp_probe": ppf._spearman(nt, ne),
        "rank_pf_full_vs_pf_tan": ppf._spearman(nf, nt), "rank_pf_full_vs_pf_rad": ppf._spearman(nf, nr),
    }
    return {"cols": cols, "checks": checks}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "physics"], required=True)
    p.add_argument("--d-values", type=str, default="16,20")
    p.add_argument("--labels", type=str, default=",".join((ppf.pl.PRIMARY_LABEL,) + ppf.pl.SECONDARY_LABELS))
    p.add_argument("--geometry-root", type=str, default=None, help="dir holding 09_probe_facing_geometry_d{d}.npz (physics)")
    p.add_argument("--record-path", type=str, default=str(DEFAULT_RECORD_PATH))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260905, help="smoke fixture seed only")
    p.add_argument("--n-permutations", type=int, default=None)
    p.add_argument("--label-table", type=str, default=None,
                   help="parquet of the label columns (LABEL_REPO@LABEL_REVISION shards, column-projected, concatenated in "
                        "shard order) to read instead of streaming the shards over hf://; sha256 is recorded")
    args = p.parse_args()
    label_table_sha = None
    if args.label_table:
        import hashlib
        import pandas as pd
        label_table_sha = hashlib.sha256(open(args.label_table, "rb").read()).hexdigest()
        _frame = pd.read_parquet(args.label_table)
        _expected = ppf.pl.EXPECTED_N_PHYSICS_ROWS if hasattr(ppf.pl, "EXPECTED_N_PHYSICS_ROWS") else None

        def _load_label_table(columns, expected_rows=None):
            out = _frame[list(columns)].reset_index(drop=True)
            want = expected_rows if expected_rows is not None else _expected
            if want is not None and len(out) != want:
                raise RuntimeError(f"label table has {len(out)} rows, expected {want}")
            return out
        ppf.pl.load_label_table = _load_label_table
        print(f"label table <- {args.label_table} sha256 {label_table_sha[:16]} rows {len(_frame)}")
    assert runner._THREADS == args.threads, (runner._THREADS, args.threads)
    record_path = Path(args.record_path).resolve()
    for stem in ppf.PRODUCTION_STEMS:
        if record_path.name.startswith(stem):
            raise SystemExit(f"refusing to write to a Phase 9 production record path: {record_path}")
    n_perm = args.n_permutations if args.n_permutations is not None else (200 if args.mode == "smoke" else 2000)
    d_values = [int(v) for v in args.d_values.split(",")] if args.mode == "physics" else [adj.SMOKE["d"]]
    geometry_root = Path(args.geometry_root) if args.geometry_root else pcp.resolve_output_root() / "probe-facing"
    print(f"record -> {record_path}\nNOT PRE-REGISTERED; GATES NOTHING.\nmode={args.mode} d={d_values} n_perm={n_perm} geometry={geometry_root}")

    data = ppf.load_physics(args) if args.mode == "physics" else ppf.load_smoke(args)
    X, labels, in_dim = data["X"], data["labels"], data["in_dim"]
    n = X.shape[0]
    k = pcp.K_NEIGHBOURS if args.mode == "physics" else adj.SMOKE["k"]
    n_anchors = pcp.N_ANCHORS if args.mode == "physics" else adj.SMOKE["n_anchors"]
    ks = [kk for kk in ppf.MULTISCALE_KS if kk <= k]
    ppf._append({"experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": _utc_now(),
                 "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "threads": args.threads, "d_values": d_values,
                 "labels": list(labels), "n": n, "k": k, "n_anchors": n_anchors, "multiscale_ks": ks, "n_permutations": n_perm,
                 "geometry_root": str(geometry_root), "columns": COLUMNS, "label_table": args.label_table, "label_table_sha256": label_table_sha, "numpy": np.__version__, "torch": torch.__version__,
                 "python": sys.version.split()[0], "pre_registered": False, "gates": "nothing"}, record_path)

    split = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)
    a = split["anchor_idx"]
    t0 = time.monotonic()
    panel = pcp.knn_panel(X, a, k)
    log_r = panel["log_knn_radius"]
    log_r_multi = np.column_stack([np.log(panel["distances"][:, kk - 1]) for kk in ks])
    print(f"[knn] k={k} anchors={n_anchors} {time.monotonic() - t0:.1f}s", flush=True)

    per_label: Dict[str, Dict[str, Any]] = {}
    for name, y in labels.items():
        y = np.asarray(y, dtype=np.float64)
        y_hat = runner._oof_predictions_for_label(X, y, pcp.ALPHA_RIDGE, pcp.N_OOF_FOLDS, pcp.OOF_FOLD_SEED)
        loc = pcp.local_r2_panel(y, y_hat, panel["indices"], pcp.MIN_FINITE_NEIGHBOURS)
        fin = np.isfinite(y)
        ridge = Ridge(alpha=pcp.ALPHA_RIDGE).fit(X[fin], y[fin])
        per_label[name] = {"y": y, "r2": loc["r2"],
                           "Z_sealed": np.column_stack([log_r, loc["local_label_variance"], loc["local_evaluation_count"]]),
                           "Z_multi": np.column_stack([log_r_multi, loc["local_label_variance"], loc["local_evaluation_count"]]),
                           "w": ridge.coef_.astype(np.float64), "b0": float(ridge.intercept_),
                           "bias_sq": (y[a] - y_hat[a]) ** 2, "n_masked": int(loc["n_masked_anchors"]),
                           "global_oof_r2": 1.0 - float(np.nansum((y - y_hat) ** 2) / np.nansum((y - np.nanmean(y)) ** 2))}
        print(f"[probe] {name}: global OOF R2 {per_label[name]['global_oof_r2']:.3f}, masked anchors {loc['n_masked_anchors']}", flush=True)

    for d in d_values:
        print("\n" + "=" * 78 + f"\nd={d}\n" + "=" * 78, flush=True)
        if args.mode == "physics":
            path = geometry_root / f"09_probe_facing_geometry_d{d}.npz"
            z = np.load(path)
            assert np.array_equal(z["anchor_idx"], a), "anchor draw differs from the stored geometry"
            geo = geometry_from_arrays(z["J"], z["Hess"], z["image"])
            print(f"[geometry] loaded {path.name}", flush=True)
        else:
            fit = ppf.fit_decoder(X, d, in_dim, adj.SMOKE_EPOCHS)
            with torch.no_grad():
                z_anchor = fit["model"].encode(fit["x64"][torch.as_tensor(a, dtype=torch.long)])
            geo = ppf.decoder_geometry(fit["curvature_model"], z_anchor)
            print(f"[geometry] smoke decoder var_explained={fit['var_explained']:.4f}", flush=True)

        # local quadratics: label and the probe's own prediction, per label (the probe differs per label)
        targets: Dict[str, np.ndarray] = {}
        for name, L in per_label.items():
            targets[f"y:{name}"] = L["y"]
            targets[f"p:{name}"] = X @ L["w"]
        t0 = time.monotonic()
        lq = local_quadratics(X, a, panel["indices"], geo, targets, pcp.MIN_FINITE_NEIGHBOURS)
        print(f"[quadratics] {len(targets)} targets x {n_anchors} anchors in {time.monotonic() - t0:.0f}s; "
              f"u_scale p50 {np.nanmedian(lq['u_scale']):.3g}", flush=True)

        for name, L in per_label.items():
            sc = split_columns(geo, L["w"], L["b0"], lq["hess"][f"y:{name}"], lq["hess"][f"p:{name}"], d)
            cols = sc["cols"]; cols["bias_sq"] = L["bias_sq"]
            r2 = L["r2"]
            rows = {}
            print(f"\n[d={d}] {name}: checks {', '.join(f'{k_}={v:+.3g}' for k_, v in sc['checks'].items())}")
            print(f"   quadratic fit R2 gain (label) p50 {np.nanmedian(lq['r2_quad'][f'y:{name}'] - lq['r2_lin'][f'y:{name}']):.3f}; "
                  f"probe target quadratic R2 p50 {np.nanmedian(lq['r2_quad'][f'p:{name}']):.4f} (linear {np.nanmedian(lq['r2_lin'][f'p:{name}']):.4f})")
            print(f"{'column':18s} {'raw rho':>8s} {'sealed':>8s} {'p':>7s} {'multi':>8s} {'p':>7s} | {'vs H_tan':>9s} {'vs log r':>9s} {'vs pf_full':>10s} {'median':>10s}")
            for c in COLUMNS:
                x = cols[c]
                ps = ppf.partial_row(x, r2, L["Z_sealed"], n_perm); pm = ppf.partial_row(x, r2, L["Z_multi"], n_perm)
                rows[c] = {"raw_rho": ppf._spearman(x, r2), "sealed": ps, "multiscale": pm,
                           "rho_vs_H_tan_norm": ppf._spearman(x, cols["H_tan_norm"]), "rho_vs_log_r": ppf._spearman(x, log_r),
                           "rho_vs_pf_full": ppf._spearman(x, cols["pf_full"]), "median": float(np.nanmedian(x))}
                print(f"{c:18s} {rows[c]['raw_rho']:+8.3f} {ps['partial']:+8.3f} {ps['p']:7.4f} {pm['partial']:+8.3f} {pm['p']:7.4f} | "
                      f"{rows[c]['rho_vs_H_tan_norm']:+9.3f} {rows[c]['rho_vs_log_r']:+9.3f} {rows[c]['rho_vs_pf_full']:+10.3f} {rows[c]['median']:10.4g}", flush=True)
            ppf._append({"experiment": EXPERIMENT, "row": "result", "mode": args.mode, "d": d, "label": name, "timestamp": _utc_now(),
                         "global_oof_r2": L["global_oof_r2"], "n_masked_anchors": L["n_masked"],
                         "local_r2_p05_p50_p95": [float(v) for v in np.nanpercentile(r2, [5, 50, 95])],
                         "checks": sc["checks"], "u_scale_p50": float(np.nanmedian(lq["u_scale"])),
                         "label_quad_r2_gain_p50": float(np.nanmedian(lq["r2_quad"][f"y:{name}"] - lq["r2_lin"][f"y:{name}"])),
                         "probe_quad_r2_p50": float(np.nanmedian(lq["r2_quad"][f"p:{name}"])),
                         "probe_lin_r2_p50": float(np.nanmedian(lq["r2_lin"][f"p:{name}"])),
                         "align_cos_full_p25_p50_p75": [float(v) for v in np.nanpercentile(cols["align_cos_full"], [25, 50, 75])],
                         "align_cos_tan_p25_p50_p75": [float(v) for v in np.nanpercentile(cols["align_cos_tan"], [25, 50, 75])],
                         "columns": rows}, record_path)
    print("\nDONE")


if __name__ == "__main__":
    main()
