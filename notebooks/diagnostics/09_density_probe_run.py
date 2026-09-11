"""Density and the local linear probe: isolation experiments on the Physics rows.

NOT PRE-REGISTERED. GATES NOTHING. Post-hoc diagnostic on the sealed Phase 9 objects.

Question. Local out-of-fold R^2 of the global ridge probe covaries with k-NN radius on the real
data (rho -0.23 ours, -0.33 colleague), and on a known surface the curvature-vs-R^2 partial's
sign follows the sample's density-curvature coupling (09-SUPPLEMENT-03). Experiments 1 and 2
(local, on the cached anchor tables) showed a decile density control and matched-radius pairs
do NOT remove either instrument's sign. This runner asks where density enters the probe and
whether removing it changes the curvature partial. Four R^2 variants on the same 512 anchors:

  global          the sealed pipeline, reproduced (sanity: must match the anchor tables)
  weighted_full   global probe refit with inverse-density sample weights w ~ r_k30^d_est
  weighted_half   same with w ~ r_k30^(d_est/2)
  local           a ridge probe fit INSIDE each 2048-patch (5-fold OOF within the patch)
  fixed_radius    global probe scored in an epsilon-ball of one common radius, not fixed k

plus two control sets for every partial: the frozen three (log r at k=2048, label variance,
count) and a multi-scale set (log r at k in K_SCALES, label variance, count).

Curvature columns come from the sealed anchor tables: decoder ||H_tan|| per d from the
Amendment 01 tables, the colleague's K_H_cross per chart rank from the colleague tables. Nothing
is refit on the curvature side. Sealed modules are imported unchanged.

Usage:
    python notebooks/diagnostics/09_density_probe_run.py --mode smoke
    python notebooks/diagnostics/09_density_probe_run.py --mode full --threads 16 \\
        --amend01-root /path/to/phase9-out-amend01 --colleague-root /path/to/phase9-out \\
        --record-path /path/to/09_density_probe.jsonl
"""

import os
import sys
from pathlib import Path


def _flag_value_from_argv(flag, argv):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


_threads = _flag_value_from_argv("--threads", sys.argv) or "8"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(_threads))

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402
from pu_manifold import physics_labels as pl  # noqa: E402

EXPERIMENT = "09_density_probe"
DEFAULT_RECORD_PATH = NOTEBOOK_ROOT / ".cache" / "09_density_probe.jsonl"
K_SCALES = (16, 64, 256, 1024, 2048)
DENSITY_K = 30
D_EST = 20
WEIGHT_CLIP_PERCENTILES = (1.0, 99.0)
DECODER_D = (16, 20, 25, 32)
COLLEAGUE_D = (12, 16, 20)
LABELS = ("mag_r", "photo_z", "smooth_fraction", "stellar_mass")
MATCHED_PAIR_EPS = 0.02


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(row: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=NOTEBOOK_ROOT.parent, text=True).strip()
    except Exception:
        return None


def _sp(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


# --- probes ------------------------------------------------------------------------------------


def oof_ridge_sklearn(X: np.ndarray, y: np.ndarray, alpha: float, n_folds: int, fold_seed: int,
                      sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Same fold structure as pcp.oof_ridge_predictions (KFold shuffle, fold_seed), sklearn Ridge
    with optional sample weights. Rows with non-finite y get NaN, as in the sealed runner."""
    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(y)
    Xf, yf = X[finite], y[finite]
    wf = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)[finite]
    yhat_f = np.full(yf.shape[0], np.nan)
    for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=fold_seed).split(Xf):
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(Xf[tr], yf[tr], sample_weight=None if wf is None else wf[tr])
        yhat_f[te] = m.predict(Xf[te])
    out = np.full(y.shape[0], np.nan)
    out[finite] = yhat_f
    return out


def local_probe_r2(X: np.ndarray, y: np.ndarray, nbr_idx: np.ndarray, alpha: float, n_folds: int,
                   fold_seed: int, min_finite: int) -> Dict[str, np.ndarray]:
    """Ridge fit INSIDE each anchor's patch, scored out-of-fold within the patch."""
    n_anchors = nbr_idx.shape[0]
    r2 = np.full(n_anchors, np.nan)
    lv = np.full(n_anchors, np.nan)
    cnt = np.zeros(n_anchors, dtype=np.int64)
    for i in range(n_anchors):
        nb = nbr_idx[i]
        yn = y[nb]
        fin = np.isfinite(yn)
        cnt[i] = int(fin.sum())
        if cnt[i] < max(min_finite, n_folds + 1):
            continue
        Xn, yf = X[nb[fin]], yn[fin]
        lv[i] = float(np.var(yf))
        yhat = np.full(yf.shape[0], np.nan)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=fold_seed).split(Xn):
            m = Ridge(alpha=alpha, fit_intercept=True).fit(Xn[tr], yf[tr])
            yhat[te] = m.predict(Xn[te])
        sst = float(np.sum((yf - yf.mean()) ** 2))
        if sst > 0:
            r2[i] = 1.0 - float(np.sum((yf - yhat) ** 2)) / sst
    return {"r2": r2, "local_label_variance": lv, "local_evaluation_count": cnt}


def fixed_radius_r2(y: np.ndarray, y_hat: np.ndarray, nbr_lists: List[np.ndarray], min_finite: int) -> Dict[str, np.ndarray]:
    n_anchors = len(nbr_lists)
    r2 = np.full(n_anchors, np.nan)
    lv = np.full(n_anchors, np.nan)
    cnt = np.zeros(n_anchors, dtype=np.int64)
    for i, nb in enumerate(nbr_lists):
        yn, yh = y[nb], y_hat[nb]
        fin = np.isfinite(yn) & np.isfinite(yh)
        cnt[i] = int(fin.sum())
        if cnt[i] < min_finite:
            continue
        yf, yhf = yn[fin], yh[fin]
        lv[i] = float(np.var(yf))
        sst = float(np.sum((yf - yf.mean()) ** 2))
        if sst > 0:
            r2[i] = 1.0 - float(np.sum((yf - yhf) ** 2)) / sst
    return {"r2": r2, "local_label_variance": lv, "local_evaluation_count": cnt}


def row_density_radius(X: np.ndarray, k: int, threads: int) -> np.ndarray:
    """Distance to the k-th nearest neighbour for EVERY row (self excluded). faiss if present."""
    try:
        import faiss  # type: ignore

        faiss.omp_set_num_threads(int(threads))
        index = faiss.IndexFlatL2(X.shape[1])
        Xf = np.ascontiguousarray(X, dtype=np.float32)
        index.add(Xf)
        d2, _ = index.search(Xf, k + 1)
        return np.sqrt(np.maximum(d2[:, -1], 0.0)).astype(np.float64)
    except Exception:
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=int(threads)).fit(X)
        dist, _ = nn.kneighbors(X)
        return dist[:, -1]


def density_weights(r: np.ndarray, exponent: float) -> np.ndarray:
    w = np.power(np.maximum(r, 1e-12), exponent)
    lo, hi = np.percentile(w, WEIGHT_CLIP_PERCENTILES)
    w = np.clip(w, lo, hi)
    return w / w.mean()


# --- statistics --------------------------------------------------------------------------------


def partial_with_p(x: np.ndarray, y: np.ndarray, Z: np.ndarray, n_perm: int, rng: np.random.Generator) -> Dict[str, Any]:
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    xm, ym, Zm = x[m], y[m], Z[m]
    obs = pcp.controlled_partial(xm, ym, Zm)
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = pcp.freedman_lane_y(ym, Zm, rng)
        null[b] = pcp.controlled_partial(xm, yp, Zm)
    pv = pcp.p_value_from_null(obs, null)
    return {"partial": float(obs), "p": pv["p"], "p_display": pv["p_display"], "n": int(m.sum())}


def matched_pairs(curv: np.ndarray, r2: np.ndarray, logr: np.ndarray, eps: float, rng: np.random.Generator, n_perm: int) -> Dict[str, Any]:
    m = np.isfinite(curv) & np.isfinite(r2) & np.isfinite(logr)
    c, y, r = curv[m], r2[m], logr[m]
    order = np.argsort(r)
    c, y, r = c[order], y[order], r[order]
    gaps = np.diff(r)
    keep = gaps < eps
    dc = np.diff(c)[keep]
    dy = np.diff(y)[keep]
    ok = (dc != 0) & (dy != 0)
    if ok.sum() < 10:
        return {"n_pairs": int(ok.sum()), "tau": float("nan"), "p": float("nan")}
    dc, dy = dc[ok], dy[ok]
    tau = float(2 * np.mean(np.sign(dc) == np.sign(dy)) - 1)
    null = np.array([2 * np.mean(np.sign(dc * rng.choice([-1, 1], size=dc.size)) == np.sign(dy)) - 1 for _ in range(n_perm)])
    p = (1 + int(np.sum(np.abs(null) >= abs(tau)))) / (n_perm + 1)
    return {"n_pairs": int(ok.sum()), "tau": tau, "p": float(p)}


# --- data --------------------------------------------------------------------------------------


def load_real(args) -> Dict[str, Any]:
    pcp.assert_preregistered()
    pl.assert_preregistered()
    t0 = time.monotonic()
    emb = pl.load_physics_embeddings()
    X, n_rows = emb["X"], emb["n_rows"]
    print(f"[load] embeddings n={n_rows} D={X.shape[1]} {time.monotonic() - t0:.1f}s", flush=True)
    t0 = time.monotonic()
    table = pl.load_label_table(columns=list(pl.LABEL_COLUMN_MAP.values()))
    offset_perm = pl.shifted_pairing(n_rows, pl.ALIGNMENT_ASSUMED_OFFSET)
    y_by_label = {}
    for name in LABELS:
        y_by_label[name] = pl.canonical_label(table, name, pl.LABEL_COLUMN_MAP, pl.SENTINEL_VALUES)[offset_perm]
    print(f"[load] labels {time.monotonic() - t0:.1f}s", flush=True)
    idx = pcp.anchor_indices(n_rows=n_rows, split_seed=pcp.SPLIT_SEED, holdout_fraction=pcp.HOLDOUT_FRACTION,
                             n_anchors=pcp.N_ANCHORS, anchor_seed=pcp.ANCHOR_DRAW_SEED)
    anchor_idx = idx["anchor_idx"]

    curv: Dict[str, np.ndarray] = {}
    ref_r2: Dict[str, np.ndarray] = {}
    a_root = Path(args.amend01_root)
    for d in DECODER_D:
        z = np.load(a_root / f"09_anchor_table_d{d}_mag_r.npz")
        if not np.array_equal(z["anchor_idx"], anchor_idx):
            raise RuntimeError(f"anchor_idx mismatch against amend01 table d={d}")
        curv[f"decoder_d{d}"] = z["H_tan_norm"]
    for lab in LABELS:
        ref_r2[lab] = np.load(a_root / f"09_anchor_table_d16_{lab}.npz")["r2"]
    c_root = Path(args.colleague_root)
    for d in COLLEAGUE_D:
        z = np.load(c_root / f"09_colleague_anchor_table_d{d}.npz")
        if not np.array_equal(z["anchor_idx"], anchor_idx):
            raise RuntimeError(f"anchor_idx mismatch against colleague table d={d}")
        curv[f"colleague_d{d}"] = z["K_H_cross"]
    return {"X": X, "y": y_by_label, "anchor_idx": anchor_idx, "curv": curv, "ref_r2": ref_r2,
            "k": pcp.K_NEIGHBOURS, "alpha": pcp.ALPHA_RIDGE, "n_folds": pcp.N_OOF_FOLDS,
            "fold_seed": pcp.OOF_FOLD_SEED, "min_finite": pcp.MIN_FINITE_NEIGHBOURS, "k_scales": K_SCALES,
            "density_k": DENSITY_K}


def load_smoke(args) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    n, D, d = 3000, 48, 6
    z = rng.normal(size=(n, d)) * rng.choice([0.5, 1.0, 1.5], size=(n, 1))
    W = rng.normal(size=(d, D))
    X = np.tanh(z @ W) + 0.05 * rng.normal(size=(n, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    a = rng.normal(size=d)
    y_lin = z @ a
    y = {"mag_r": y_lin + 0.3 * rng.normal(size=n), "photo_z": np.sin(y_lin), "smooth_fraction": y_lin ** 2,
         "stellar_mass": np.where(rng.random(n) < 0.9, y_lin, np.nan)}
    anchor_idx = np.sort(rng.choice(n, 64, replace=False))
    curv = {f"decoder_d{d_}": rng.random(64) for d_ in DECODER_D}
    curv.update({f"colleague_d{d_}": rng.random(64) for d_ in COLLEAGUE_D})
    return {"X": X, "y": y, "anchor_idx": anchor_idx, "curv": curv, "ref_r2": {}, "k": 128, "alpha": 100.0,
            "n_folds": 5, "fold_seed": 1, "min_finite": 10, "k_scales": (8, 32, 128), "density_k": 10}


# --- main --------------------------------------------------------------------------------------


def run(args) -> None:
    record = Path(args.record_path)
    if record.exists() and args.mode == "full" and not args.append:
        raise SystemExit(f"record exists: {record}. Use --append or a new --record-path.")
    n_perm = args.n_permutations or (100 if args.mode == "smoke" else 2000)
    data = load_smoke(args) if args.mode == "smoke" else load_real(args)
    X, y_by_label, anchor_idx, curv = data["X"], data["y"], data["anchor_idx"], data["curv"]
    k, alpha, n_folds, fold_seed, min_finite = data["k"], data["alpha"], data["n_folds"], data["fold_seed"], data["min_finite"]
    k_scales = [s for s in data["k_scales"] if s <= k]
    rng = np.random.default_rng(20260910)
    threads = int(args.threads)

    _append({"experiment": EXPERIMENT, "row": "environment", "mode": args.mode, "timestamp": _utc_now(),
             "repo_head": _git_head(), "threads": threads, "n_permutations": n_perm, "k": k, "k_scales": k_scales,
             "density_k": data["density_k"], "d_est": D_EST, "weight_clip_percentiles": WEIGHT_CLIP_PERCENTILES,
             "n_rows": int(X.shape[0]), "n_anchors": int(anchor_idx.shape[0]), "note": "NOT PRE-REGISTERED; GATES NOTHING"}, record)

    # neighbourhoods at anchors, all scales from one query
    t0 = time.monotonic()
    knn = pcp.knn_panel(X, anchor_idx, k)
    logr = {s: np.log(knn["distances"][:, s - 1]) for s in k_scales}
    logr_k = knn["log_knn_radius"]
    print(f"[knn] k={k} {time.monotonic() - t0:.1f}s; scales {k_scales}", flush=True)

    # fixed-radius neighbourhoods
    t0 = time.monotonic()
    R0 = float(np.exp(np.median(logr_k)))
    nn_r = NearestNeighbors(radius=R0, n_jobs=threads).fit(X)
    _, r_idx = nn_r.radius_neighbors(X[anchor_idx])
    r_lists = [np.asarray(ix) for ix in r_idx]
    r_counts = np.array([ix.size for ix in r_lists])
    print(f"[radius] R0={R0:.5f} counts p05/p50/p95 = {np.percentile(r_counts, [5, 50, 95])} {time.monotonic() - t0:.1f}s", flush=True)

    # row-level density for weights
    t0 = time.monotonic()
    r_dens = row_density_radius(X, data["density_k"], threads)
    w_full = density_weights(r_dens, D_EST)
    w_half = density_weights(r_dens, D_EST / 2)
    print(f"[density] k={data['density_k']} row radius p05/p50/p95 = {np.percentile(r_dens, [5, 50, 95])}; "
          f"w_full p05/p95 = {np.percentile(w_full, [5, 95])} {time.monotonic() - t0:.1f}s", flush=True)
    _append({"experiment": EXPERIMENT, "row": "geometry", "R0": R0, "fixed_radius_count_pct": np.percentile(r_counts, [5, 25, 50, 75, 95]).tolist(),
             "row_density_radius_pct": np.percentile(r_dens, [5, 25, 50, 75, 95]).tolist(),
             "w_full_pct": np.percentile(w_full, [1, 5, 50, 95, 99]).tolist(), "w_half_pct": np.percentile(w_half, [1, 5, 50, 95, 99]).tolist(),
             "rho_anchor_logr_k_vs_row_density_radius": _sp(logr_k, np.log(r_dens[anchor_idx]))}, record)

    variants = ("global", "weighted_full", "weighted_half", "local", "fixed_radius")
    panels: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {v: {} for v in variants}

    for lab in LABELS:
        y = y_by_label[lab]
        t0 = time.monotonic()
        yhat_g = oof_ridge_sklearn(X, y, alpha, n_folds, fold_seed)
        yhat_wf = oof_ridge_sklearn(X, y, alpha, n_folds, fold_seed, sample_weight=w_full)
        yhat_wh = oof_ridge_sklearn(X, y, alpha, n_folds, fold_seed, sample_weight=w_half)
        pg = pcp.local_r2_panel(y, yhat_g, knn["indices"], min_finite)
        panels["global"][lab] = pg
        panels["weighted_full"][lab] = pcp.local_r2_panel(y, yhat_wf, knn["indices"], min_finite)
        panels["weighted_half"][lab] = pcp.local_r2_panel(y, yhat_wh, knn["indices"], min_finite)
        panels["local"][lab] = local_probe_r2(X, y, knn["indices"], alpha, n_folds, fold_seed, min_finite)
        panels["fixed_radius"][lab] = fixed_radius_r2(y, yhat_g, r_lists, min_finite)
        fin = np.isfinite(y)
        glob_r2 = {v: float(1 - np.nanmean((y[fin] - yh[fin]) ** 2) / np.var(y[fin])) for v, yh in
                   (("global", yhat_g), ("weighted_full", yhat_wf), ("weighted_half", yhat_wh))}
        ref = data["ref_r2"].get(lab)
        repro = None if ref is None else float(np.nanmax(np.abs(pg["r2"] - ref)))
        print(f"[probe] {lab}: global R2 {glob_r2['global']:.4f} wfull {glob_r2['weighted_full']:.4f} whalf {glob_r2['weighted_half']:.4f}; "
              f"repro max|dr2| vs sealed table = {repro}; {time.monotonic() - t0:.1f}s", flush=True)
        for v in variants:
            p = panels[v][lab]
            _append({"experiment": EXPERIMENT, "row": "variant_panel", "label": lab, "variant": v,
                     "global_oof_r2": glob_r2.get(v), "repro_max_abs_dr2_vs_sealed": repro if v == "global" else None,
                     "n_finite_anchors": int(np.isfinite(p["r2"]).sum()), "r2_pct": np.nanpercentile(p["r2"], [5, 25, 50, 75, 95]).tolist(),
                     "rho_r2_logr": {str(s): _sp(p["r2"], logr[s]) for s in k_scales},
                     "rho_r2_count": _sp(p["r2"], p["local_evaluation_count"].astype(float)),
                     "rho_r2_labelvar": _sp(p["r2"], p["local_label_variance"])}, record)

    # partials
    summary = []
    for cname, x in curv.items():
        for lab in LABELS:
            for v in variants:
                p = panels[v][lab]
                Z3 = np.column_stack([logr_k, p["local_label_variance"], p["local_evaluation_count"].astype(float)])
                Zm = np.column_stack([logr[s] for s in k_scales] + [p["local_label_variance"], p["local_evaluation_count"].astype(float)])
                raw = _sp(x, p["r2"])
                c3 = partial_with_p(x, p["r2"], Z3, n_perm, rng)
                cm = partial_with_p(x, p["r2"], Zm, n_perm, rng)
                mp = matched_pairs(x, p["r2"], logr_k, MATCHED_PAIR_EPS, rng, n_perm)
                row = {"experiment": EXPERIMENT, "row": "partial", "curvature": cname, "label": lab, "variant": v,
                       "rho_curv_logr": _sp(x, logr_k), "raw_rho": raw,
                       "partial_3control": c3["partial"], "p_3control": c3["p"], "p_3control_display": c3["p_display"],
                       "partial_multiscale": cm["partial"], "p_multiscale": cm["p"], "p_multiscale_display": cm["p_display"],
                       "matched_pairs_tau": mp["tau"], "matched_pairs_p": mp["p"], "matched_pairs_n": mp["n_pairs"], "n": c3["n"]}
                _append(row, record)
                summary.append(row)
        print(f"[partial] {cname} done", flush=True)

    print("\n" + "=" * 100)
    print("SUMMARY  rho(local R2, log r_k) per variant and label")
    for lab in LABELS:
        print(f"  {lab:15s} " + "  ".join(f"{v}={_sp(panels[v][lab]['r2'], logr_k):+.3f}" for v in variants))
    print("\nSUMMARY  curvature partials: variant | 3-control (p) | multiscale (p) | matched tau (p)")
    for cname in curv:
        for lab in LABELS:
            print(f"  {cname:14s} {lab:15s}")
            for r in summary:
                if r["curvature"] == cname and r["label"] == lab:
                    print(f"      {r['variant']:14s} raw={r['raw_rho']:+.3f} | {r['partial_3control']:+.3f} ({r['p_3control']:.4f}) "
                          f"| {r['partial_multiscale']:+.3f} ({r['p_multiscale']:.4f}) | tau={r['matched_pairs_tau']:+.3f} ({r['matched_pairs_p']:.4f})")
    _append({"experiment": EXPERIMENT, "row": "done", "timestamp": _utc_now()}, record)
    print(f"\nrecord -> {record}\nNOT PRE-REGISTERED; GATES NOTHING.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--amend01-root", type=str, default=None, help="dir with 09_anchor_table_d{d}_{label}.npz (Amendment 01)")
    p.add_argument("--colleague-root", type=str, default=None, help="dir with 09_colleague_anchor_table_d{d}.npz")
    p.add_argument("--record-path", type=str, default=str(DEFAULT_RECORD_PATH))
    p.add_argument("--n-permutations", type=int, default=None)
    p.add_argument("--append", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode == "full" and (args.amend01_root is None or args.colleague_root is None):
        raise SystemExit("--mode full needs --amend01-root and --colleague-root")
    if args.mode == "smoke" and args.record_path == str(DEFAULT_RECORD_PATH):
        args.record_path = str(NOTEBOOK_ROOT / ".cache" / "09_scratch_density_probe_smoke.jsonl")
        Path(args.record_path).unlink(missing_ok=True)
    run(args)


if __name__ == "__main__":
    main()
