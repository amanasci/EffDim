"""Counterfactual normal-scaling test: does the probe's normal component help or hurt, patch by patch?

PURPOSE. The partials of Supplement 07/11 are rank correlations across anchors and cannot say whether
bending AWAY from the label's curvature hurts relative to a flat readout, because no anchor is flat.
This runner asks that question at each anchor by intervening on the readout instead of the manifold.
The fitted global ridge weight ``w`` is split at the anchor into a tangent part ``w_T = J g^-1 J^T w``,
a radial part ``w_rad = (w.x_hat) x_hat`` (the sphere term) and an in-sphere normal part
``w_S = w - w_T - w_rad``. Over the anchor's k-neighbourhood ``w_T.x`` is first order in the chart
coordinate and ``w_S.x`` is second order, ``(1/2) <w_S, II^S>(u,u)`` plus the off-manifold part of the
data. Scaling ``w(t) = w_T + w_rad + t w_S`` and refitting the local intercept makes the local sum of
squares an exact quadratic in ``t``: ``SS(t) = |e0 - t q|^2`` with ``e0`` the centred residual of the
``t = 0`` readout and ``q`` the centred ``w_S.x``. Hence ``t* = <e0,q>/<q,q>`` is the best scaling and the
fitted normal component (``t = 1``) beats the flat one (``t = 0``) iff ``2<e0,q> > <q,q>`` -- the data-side
analogue of the improvement condition ``2<Hess_M y, K> > |K|^2`` of the manuscript. Per anchor we record
``t*``, the change in local R^2 from ``t = 0`` to ``t = 1``, the same for the full normal part
(``w(t) = w_T + t w_N``), a random in-sphere normal direction as a null (matched on the contracted tensor's
metric norm ``|<v, II^S>|_g``, not on ``|v|``; the ``|v|``-matched version is kept for comparison), and the
decoder's own cross term ``<Hess_M y, <w_N, II^S>>_g`` so its predicted sign can be checked against
the counterfactual outcome. Predictions: ``t* > 0`` for most anchors; ``2<e0,q> > <q,q>`` more often
where the decoder cross term is positive than where it is negative; the random direction never helps.

Geometry (J, D^2F, image at the anchors) is read from the npz the split runner saved; the decoder is
not refit. Everything else reaches the production pipeline through the split runner unchanged.

NOT PRE-REGISTERED, GATES NOTHING. Writes only to its own record.

Usage:
    python notebooks/diagnostics/09_physics_normal_scaling_run.py --mode smoke --threads 8
    python notebooks/diagnostics/09_physics_normal_scaling_run.py --mode physics --d 16 --threads 16 \\
        --geometry-npz <.../09_probe_facing_geometry_d16.npz> --parquet-path <...> --embedding-column vit_base_galaxies \\
        --label-table <cached labels parquet> --record-path notebooks/.cache/09_physics_normal_scaling_vit_base.jsonl
"""

import importlib.util
import os
import sys
from pathlib import Path

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_PFS_PATH = DIAGNOSTICS_ROOT / "09_physics_probe_facing_split_run.py"
_spec = importlib.util.spec_from_file_location("physics_probe_facing_split_run", _PFS_PATH)
pfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pfs)
ppf, adj, runner = pfs.ppf, pfs.adj, pfs.runner

import argparse  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402

EXPERIMENT = "physics-normal-scaling"
DEFAULT_RECORD_PATH = Path(os.environ.get("EFFDIM_CACHE_DIR") or NOTEBOOK_ROOT / ".cache") / "09_physics_normal_scaling.jsonl"
T_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
VARIANTS = ("S", "full", "random_wnorm", "random_matched", "random_qmatched", "S_model", "S_proj", "full_model")


def scaling_at_anchor(Xn: np.ndarray, yn: np.ndarray, x0: np.ndarray, w: np.ndarray, J: np.ndarray, g: np.ndarray, ginv: np.ndarray,
                      II: np.ndarray, xhat: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    """Exact quadratic-in-t local sum of squares for the readout variants at one anchor.

    Data variants scale the normal part of the readout evaluated on the data (which includes the off-manifold part of
    x, first order in the reconstruction residual). Model variants replace the scaled term by the decoder's own
    second-order term (1/2) <w_S, II^S>(u,u) in the anchor's chart coordinates u = g^-1 J^T (x - x0): the quantity the
    residual expansion is about. S_proj also replaces the base by the model's first-order term plus the sphere term."""
    wT = J @ (ginv @ (J.T @ w))
    wN = w - wT
    w_rad = (wN @ xhat) * xhat
    wS = wN - w_rad
    r = rng.standard_normal(w.shape[0])
    r -= J @ (ginv @ (J.T @ r)); r -= (r @ xhat) * xhat
    r *= np.linalg.norm(wS) / max(np.linalg.norm(r), 1e-300)
    sst = float(((yn - yn.mean()) ** 2).sum())
    u = (Xn - x0[None, :]) @ J @ ginv                                            # chart coordinates, as local_quadratics
    II_tan = II - np.einsum("a,ij->aij", xhat, np.einsum("aij,a->ij", II, xhat))
    A_S = np.einsum("aij,a->ij", II_tan, wS); A_N = np.einsum("aij,a->ij", II, wN)
    qS = 0.5 * np.einsum("ki,ij,kj->k", u, A_S, u); qN = 0.5 * np.einsum("ki,ij,kj->k", u, A_N, u)
    # random control matched on what the theory says matters: the contracted tensor's metric Frobenius norm |<v, II^S>|_g,
    # not |v| (in codimension ~750 a random normal direction is nearly orthogonal to the <= d(d+1)/2 directions II^S spans)
    gnorm = lambda A: float(np.sqrt(max(np.einsum("ij,jk,kl,li->", ginv, A, ginv, A), 0.0)))
    A_r = np.einsum("aij,a->ij", II_tan, r)
    rand_ratio = gnorm(A_r) / max(gnorm(A_S), 1e-300)                            # |<r,II^S>|_g / |<w_S,II^S>|_g at equal |r| = |w_S|
    q_rm = 0.5 * np.einsum("ki,ij,kj->k", u, A_r / max(rand_ratio, 1e-300), u)     # rescaled so |<v,II^S>|_g = |<w_S,II^S>|_g
    # strongest finite-sample control: same centred quadratic amplitude on the actual neighbours, |q_v - mean|_2 = |q_S - mean|_2
    q_rq = q_rm * (np.linalg.norm(qS - qS.mean()) / max(np.linalg.norm(q_rm - q_rm.mean()), 1e-300))
    base_proj = u @ (J.T @ wT) - 0.5 * float(wN @ xhat) * np.einsum("ki,ij,kj->k", u, g, u)   # model first order + sphere term
    out: Dict[str, Any] = {"sst": sst, "wS_norm": float(np.linalg.norm(wS)), "wN_norm": float(np.linalg.norm(wN)), "rand_ratio": rand_ratio}
    for name, base, q in (("S", Xn @ (wT + w_rad), Xn @ wS), ("full", Xn @ wT, Xn @ wN), ("random_wnorm", Xn @ (wT + w_rad), Xn @ r),
                          ("random_matched", Xn @ (wT + w_rad), q_rm), ("random_qmatched", Xn @ (wT + w_rad), q_rq),
                          ("S_model", Xn @ (wT + w_rad), qS), ("S_proj", base_proj, qS), ("full_model", Xn @ wT, qN)):
        e0 = yn - base; e0 = e0 - e0.mean()
        q = q - q.mean()
        ee, eq, qq = float(e0 @ e0), float(e0 @ q), float(q @ q)
        out[name] = {"t_star": eq / max(qq, 1e-300), "dR2": (2 * eq - qq) / max(sst, 1e-300),
                     "eq": eq, "qq": qq,
                     "r2_curve": [1.0 - (ee - 2 * t * eq + t * t * qq) / max(sst, 1e-300) for t in T_GRID]}
    return out


def summarise(name: str, v: Dict[str, np.ndarray], cross_dec: np.ndarray, t_dec: np.ndarray, log_r: np.ndarray) -> Dict[str, Any]:
    m = np.isfinite(v["t_star"]) & np.isfinite(cross_dec)
    pos, neg = m & (cross_dec > 0), m & (cross_dec < 0)
    s = {"n": int(m.sum()), "frac_t_star_pos": float(np.mean(v["t_star"][m] > 0)),
         "frac_helps": float(np.mean(v["dR2"][m] > 0)), "dR2_p25_p50_p75": [float(x) for x in np.percentile(v["dR2"][m], [25, 50, 75])],
         "t_star_p25_p50_p75": [float(x) for x in np.percentile(v["t_star"][m], [25, 50, 75])],
         "r2_curve_median": [float(x) for x in np.nanmedian(v["r2_curve"][m], axis=0)],
         "n_dec_pos": int(pos.sum()), "n_dec_neg": int(neg.sum()),
         "frac_helps_dec_pos": float(np.mean(v["dR2"][pos] > 0)) if pos.any() else None,
         "frac_helps_dec_neg": float(np.mean(v["dR2"][neg] > 0)) if neg.any() else None,
         "dR2_median_dec_pos": float(np.median(v["dR2"][pos])) if pos.any() else None,
         "dR2_median_dec_neg": float(np.median(v["dR2"][neg])) if neg.any() else None,
         "frac_t_star_pos_dec_pos": float(np.mean(v["t_star"][pos] > 0)) if pos.any() else None,
         "frac_t_star_pos_dec_neg": float(np.mean(v["t_star"][neg] > 0)) if neg.any() else None,
         "sign_concordance_eq_vs_dec_cross": float(np.mean(np.sign(v["eq"][m]) == np.sign(cross_dec[m]))),
         "rho_t_star_vs_t_dec": float(spearmanr(v["t_star"][m], t_dec[m]).statistic) if not name.startswith("random") else None,
         "rho_dR2_vs_log_r": float(spearmanr(v["dR2"][m], log_r[m]).statistic),
         "rho_dR2_vs_dec_cross": float(spearmanr(v["dR2"][m], cross_dec[m]).statistic)}
    return s


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["smoke", "physics"], required=True)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--labels", type=str, default=",".join((ppf.pl.PRIMARY_LABEL,) + ppf.pl.SECONDARY_LABELS))
    p.add_argument("--geometry-npz", type=str, default=None, help="physics: npz with anchor_idx, J, Hess, image at the anchors")
    p.add_argument("--parquet-path", type=str, default=None)
    p.add_argument("--embedding-column", type=str, default=None)
    p.add_argument("--label-table", type=str, default=None)
    p.add_argument("--record-path", type=str, default=str(DEFAULT_RECORD_PATH))
    p.add_argument("--arrays-out", type=str, default=None, help="npz of the per-anchor arrays")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260915)
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    assert runner._THREADS == args.threads, (runner._THREADS, args.threads)
    record_path = Path(args.record_path).resolve()
    for stem in ppf.PRODUCTION_STEMS:
        if record_path.name.startswith(stem):
            raise SystemExit(f"refusing to write to a Phase 9 production record path: {record_path}")

    label_table_sha = None
    if args.mode == "physics":
        if args.parquet_path:  # same runner-level shims as the split runner (sealed loader hard-codes width 768)
            ppf.pl.PHYSICS_PARQUET_PATH = args.parquet_path
            if args.embedding_column:
                ppf.pl.PHYSICS_COLUMN = args.embedding_column

            def _load_embeddings(parquet_path=None, column=None, expected_rows=None, normalize=True):
                import pyarrow.parquet as pq
                path = parquet_path or ppf.pl.PHYSICS_PARQUET_PATH; col = column or ppf.pl.PHYSICS_COLUMN
                tbl = pq.read_table(path, columns=[col])
                raw = np.stack([np.asarray(v, dtype=np.float64) for v in tbl.column(col).to_pylist()])
                want = expected_rows or ppf.pl.EXPECTED_N_PHYSICS_ROWS
                if raw.shape[0] != want:
                    raise RuntimeError(f"{path}: {raw.shape[0]} rows, expected {want}")
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                X = raw / np.maximum(norms, 1e-12) if normalize else raw
                print(f"[shim] {path} -> {X.shape}, row norms {norms.min():.4f}-{norms.max():.4f}")
                return {"X": X, "n_rows": int(X.shape[0]), "n_features": int(X.shape[1])}
            ppf.pl.load_physics_embeddings = _load_embeddings
        if args.label_table:
            import pandas as pd
            label_table_sha = hashlib.sha256(open(args.label_table, "rb").read()).hexdigest()
            _frame = pd.read_parquet(args.label_table)

            def _load_label_table(columns, expected_rows=None):
                out = _frame[list(columns)].reset_index(drop=True)
                want = expected_rows if expected_rows is not None else ppf.pl.EXPECTED_N_PHYSICS_ROWS
                if len(out) != want:
                    raise RuntimeError(f"label table has {len(out)} rows, expected {want}")
                return out
            ppf.pl.load_label_table = _load_label_table
            print(f"label table <- {args.label_table} sha256 {label_table_sha[:16]} rows {len(_frame)}")

    print(f"record -> {record_path}\nNOT PRE-REGISTERED; GATES NOTHING.\nmode={args.mode} d={args.d}")
    data = ppf.load_physics(args) if args.mode == "physics" else ppf.load_smoke(args)
    X, labels = data["X"], data["labels"]
    n, in_dim = X.shape
    k = pcp.K_NEIGHBOURS if args.mode == "physics" else adj.SMOKE["k"]
    n_anchors = pcp.N_ANCHORS if args.mode == "physics" else adj.SMOKE["n_anchors"]
    d = args.d if args.mode == "physics" else adj.SMOKE["d"]
    split = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)
    a = split["anchor_idx"]
    panel = pcp.knn_panel(X, a, k)
    neigh, log_r = panel["indices"], panel["log_knn_radius"]

    if args.mode == "physics":
        z = np.load(args.geometry_npz)
        assert np.array_equal(z["anchor_idx"], a), "anchor draw differs from the stored geometry"
        geo = pfs.geometry_from_arrays(z["J"], z["Hess"], z["image"])
        geometry_sha = hashlib.sha256(open(args.geometry_npz, "rb").read()).hexdigest()
    else:
        fit = ppf.fit_decoder(X, d, in_dim, adj.SMOKE_EPOCHS)
        with torch.no_grad():
            z_anchor = fit["model"].encode(fit["x64"][torch.as_tensor(a, dtype=torch.long)])
        geo = ppf.decoder_geometry(fit["curvature_model"], z_anchor)
        geometry_sha = None
        print(f"[geometry] smoke decoder var_explained={fit['var_explained']:.4f}")
    image = geo["image"]
    xhat = image / np.linalg.norm(image, axis=1, keepdims=True)
    cos_img = np.einsum("ba,ba->b", xhat, X[a] / np.linalg.norm(X[a], axis=1, keepdims=True))
    print(f"[geometry] anchors={n_anchors} d={d} k={k}; cos(decoder image, data row) p05/p50 {np.percentile(cos_img, 5):.3f}/{np.median(cos_img):.3f}", flush=True)

    ppf._append({"experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": pfs._utc_now(),
                 "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "threads": args.threads, "d": d, "labels": list(labels), "n": n,
                 "in_dim": int(in_dim), "k": k, "n_anchors": n_anchors, "t_grid": list(T_GRID), "alpha": pcp.ALPHA_RIDGE,
                 "geometry_npz": args.geometry_npz, "geometry_sha256": geometry_sha, "parquet_path": args.parquet_path,
                 "embedding_column": args.embedding_column, "label_table": args.label_table, "label_table_sha256": label_table_sha,
                 "cos_image_data_p05_p50": [float(np.percentile(cos_img, 5)), float(np.median(cos_img))], "seed": args.seed,
                 "numpy": np.__version__, "python": sys.version.split()[0], "pre_registered": False, "gates": "nothing"}, record_path)

    arrays: Dict[str, np.ndarray] = {"anchor_idx": a, "log_r": log_r}
    for name, y in labels.items():
        y = np.asarray(y, dtype=np.float64)
        fin = np.isfinite(y)
        ridge = Ridge(alpha=pcp.ALPHA_RIDGE).fit(X[fin], y[fin])
        w, b0 = ridge.coef_.astype(np.float64), float(ridge.intercept_)
        t0 = time.monotonic()
        lq = pfs.local_quadratics(X, a, neigh, geo, {"y": y}, pcp.MIN_FINITE_NEIGHBOURS)
        hess_y = lq["hess"]["y"]
        # decoder-side columns: only the in-sphere cross term, its norm and the alignment are used (probe_emp slot is a dummy)
        cols = pfs.split_columns(geo, w, b0, hess_y, hess_y, d)["cols"]
        cross_dec, pf_tan, align = cols["cross_tan"], cols["pf_tan"], cols["align_cos_tan"]
        t_dec = cross_dec / np.maximum(pf_tan ** 2, 1e-300)                           # decoder-predicted best scaling
        rng = np.random.default_rng(args.seed)
        per = {v: {"t_star": np.full(n_anchors, np.nan), "dR2": np.full(n_anchors, np.nan), "eq": np.full(n_anchors, np.nan),
                   "qq": np.full(n_anchors, np.nan), "r2_curve": np.full((n_anchors, len(T_GRID)), np.nan)} for v in VARIANTS}
        wS_norm = np.full(n_anchors, np.nan); wN_norm = np.full(n_anchors, np.nan); rand_ratio = np.full(n_anchors, np.nan)
        for i in range(n_anchors):
            idx = neigh[i]; yn = y[idx]; m = np.isfinite(yn)
            if m.sum() < pcp.MIN_FINITE_NEIGHBOURS:
                continue
            sc = scaling_at_anchor(X[idx][m], yn[m], X[a[i]], w, geo["J"][i], geo["g"][i], geo["ginv"][i], geo["II"][i], xhat[i], rng)
            wS_norm[i], wN_norm[i], rand_ratio[i] = sc["wS_norm"], sc["wN_norm"], sc["rand_ratio"]
            for v in VARIANTS:
                for key in ("t_star", "dR2", "eq", "qq"):
                    per[v][key][i] = sc[v][key]
                per[v]["r2_curve"][i] = sc[v]["r2_curve"]
        summ = {v: summarise(v, per[v], cross_dec, t_dec, log_r) for v in VARIANTS}
        print(f"\n[{name}] global in-sample R2 {ridge.score(X[fin], y[fin]):.3f}; |w_S|/|w| p50 {np.nanmedian(wS_norm) / np.linalg.norm(w):.3f}; "
              f"|w_N|/|w| p50 {np.nanmedian(wN_norm) / np.linalg.norm(w):.3f}; random |<r,II>|/|<w_S,II>| at equal norm p50 {np.nanmedian(rand_ratio):.3f}; decoder cross term > 0 at {summ['S']['n_dec_pos']} anchors, < 0 at {summ['S']['n_dec_neg']}; {time.monotonic() - t0:.0f}s")
        print(f"{'variant':8s} {'t*>0':>6s} {'helps':>6s} {'dR2 p50':>9s} | {'helps|dec+':>10s} {'helps|dec-':>10s} {'dR2|dec+':>9s} {'dR2|dec-':>9s} | {'t*>0|dec+':>9s} {'t*>0|dec-':>9s} | {'sgn agree':>9s} {'rho t*':>7s} | median R2 curve over t=" + ",".join(f"{t:g}" for t in T_GRID))
        for v in VARIANTS:
            s = summ[v]
            f = lambda x: "   --" if x is None else f"{x:+.3f}" if isinstance(x, float) and abs(x) < 10 else str(x)
            print(f"{v:8s} {s['frac_t_star_pos']:6.2f} {s['frac_helps']:6.2f} {s['dR2_p25_p50_p75'][1]:+9.4f} | {f(s['frac_helps_dec_pos']):>10s} {f(s['frac_helps_dec_neg']):>10s} "
                  f"{f(s['dR2_median_dec_pos']):>9s} {f(s['dR2_median_dec_neg']):>9s} | {f(s['frac_t_star_pos_dec_pos']):>9s} {f(s['frac_t_star_pos_dec_neg']):>9s} | "
                  f"{s['sign_concordance_eq_vs_dec_cross']:9.2f} {f(s['rho_t_star_vs_t_dec']):>7s} | " + " ".join(f"{x:.3f}" for x in s["r2_curve_median"]), flush=True)
        ppf._append({"experiment": EXPERIMENT, "row": "result", "mode": args.mode, "d": d, "label": name, "timestamp": pfs._utc_now(),
                     "global_insample_r2": float(ridge.score(X[fin], y[fin])), "wS_over_w_p50": float(np.nanmedian(wS_norm) / np.linalg.norm(w)),
                     "wN_over_w_p50": float(np.nanmedian(wN_norm) / np.linalg.norm(w)),
                     "random_contracted_norm_ratio_p25_p50_p75": [float(v) for v in np.nanpercentile(rand_ratio, [25, 50, 75])],
                     "align_cos_tan_p50": float(np.nanmedian(align)), "variants": summ}, record_path)
        for v in VARIANTS:
            for key in ("t_star", "dR2", "eq", "qq", "r2_curve"):
                arrays[f"{name}:{v}:{key}"] = per[v][key]
        arrays[f"{name}:rand_ratio"] = rand_ratio; arrays[f"{name}:dec_cross_tan"] = cross_dec; arrays[f"{name}:dec_t_star"] = t_dec; arrays[f"{name}:align_cos_tan"] = align
    if args.arrays_out:
        Path(args.arrays_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.arrays_out, **arrays)
        print(f"arrays -> {args.arrays_out}")
    print("\nDONE")


if __name__ == "__main__":
    main()
