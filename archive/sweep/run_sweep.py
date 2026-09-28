#!/usr/bin/env python3
"""
Cross-model validity-gate sweep over the UniverseTBD/pu-embeddings legacysurvey configs.

WHAT THIS MEASURES
------------------
For each vision foundation model, fit Isomap on the same 10,000 galaxies and compute the FULL
classical-MDS eigenspectrum of the double-centred geodesic distance matrix. Negative
eigenvalues measure how far the geodesic metric is from being Euclidean-embeddable -- i.e.
whether Isomap's final flattening step is valid at all for that model's embedding space.

    r = |lambda_min_neg| / lambda_max_pos     one dominant negative outlier?
    m = sum|lambda_neg| / sum|lambda|         how much total mass is negative?

Verdict is the worse of the two, strict less-than at every boundary:

    PASS      r < 0.10  AND  m < 0.05
    MARGINAL  r < 0.25  AND  m < 0.15
    FAIL      otherwise

A prior run on legacysurvey_dinov3_vitb16 measured r=0.052419, m=0.412071 -> FAIL. This sweep
asks whether that is universal across architectures or specific to DINOv3.

CONTRACT -- please do not change these
--------------------------------------
The thresholds, k, n_components, and the 10,000 row indices are PRE-REGISTERED (see
02-MODEL-SWEEP-PREREGISTRATION.md). They were fixed before any model was run, so that a
34-way comparison cannot be narrated after the fact. Changing any of them invalidates the
comparison. The script refuses to start if the shipped row-index file does not match its
expected hash.

Every model in MODELS is run and reported regardless of outcome. Nothing is dropped for being
inconvenient. Failures are recorded as failures, with their traceback.

USAGE
-----
    pip install -r requirements.txt
    python run_sweep.py --workers 8

    # resume after interruption (completed models are skipped)
    python run_sweep.py --workers 8

    # single model, e.g. to sanity-check the environment
    python run_sweep.py --only dinov3_vitb16

Results append to model_sweep_results.jsonl (one JSON object per model). Send that file back;
it is a few hundred KB and contains no embeddings.

RESOURCES
---------
~20 GB total download (only the legacysurvey column of each parquet is read; the paired hsc
column is skipped). Each model needs ~3.5 GB RAM at peak and ~2 minutes of CPU. Pick
--workers <= min(cores // 4, RAM_GB // 4). The script pins BLAS threads per worker to avoid
oversubscription.
"""
import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- PRE-REGISTERED CONSTANTS -- do not edit -------------------------------------------
R_MAX_PASS, M_MAX_PASS = 0.10, 0.05
R_MAX_MARGINAL, M_MAX_MARGINAL = 0.25, 0.15
K = 15
NCOMP = 18
N_ROWS = 10000
ROW_INDEX_SHA256 = "20b40cb5d4f57dc2d90214f61445c38648be57ba384d61b22d82bf11b8b0ca28"

# Control: the already-published result this sweep must reproduce before anything is trusted.
CONTROL_MODEL = "dinov3_vitb16"
CONTROL_R, CONTROL_M = 0.052419, 0.412071
CONTROL_TOL = 1e-5

REPO = "datasets/UniverseTBD/pu-embeddings"

MODELS = [
    "astropt_015M", "astropt_095M", "astropt_850M",
    "clip_base", "clip_large",
    "convnext_nano", "convnext_tiny", "convnext_base", "convnext_large",
    "dino_small", "dino_base", "dino_large", "dino_giant",
    "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vitl16",
    "dinov3_vith16plus", "dinov3_vit7b16",
    "ijepa_giant", "ijepa_huge",
    "llava_15_7b", "llava_15_13b",
    "paligemma_3b", "paligemma_10b", "paligemma_28b",
    "vit_base", "vit_large", "vit_huge",
    "vit-mae_base", "vit-mae_large", "vit-mae_huge",
    "vjepa_large", "vjepa_giant", "vjepa_huge",
]


def classify(r, m):
    if r < R_MAX_PASS and m < M_MAX_PASS:
        return "PASS"
    if r < R_MAX_MARGINAL and m < M_MAX_MARGINAL:
        return "MARGINAL"
    return "FAIL"


def load_row_indices():
    p = HERE / "row_indices_20260729.npy"
    if not p.exists():
        sys.exit(f"FATAL: {p} is missing. It pins the 10,000 galaxies every model is "
                 f"compared on and must ship with this script.")
    import numpy as np
    ri = np.load(p)
    got = hashlib.sha256(ri.tobytes()).hexdigest()
    if got != ROW_INDEX_SHA256:
        sys.exit(f"FATAL: row_indices hash mismatch.\n  expected {ROW_INDEX_SHA256}\n"
                 f"  got      {got}\nThe object population must be identical across models "
                 f"or the comparison is meaningless.")
    if ri.shape != (N_ROWS,):
        sys.exit(f"FATAL: row_indices shape {ri.shape}, expected ({N_ROWS},)")
    return ri


def run_one(model, row_indices, threads):
    """Fit one model end-to-end. Returns a result dict; never raises."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(threads)

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from scipy.linalg import eigvalsh
    from scipy.sparse.csgraph import connected_components
    from sklearn.manifold import Isomap
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph

    rec = {"model": model, "k": K, "n_components": NCOMP, "n_rows": N_ROWS,
           "threads": threads}
    t_all = time.perf_counter()
    try:
        # ---- load: ONLY the legacysurvey column ----
        t0 = time.perf_counter()
        path = f"{REPO}/legacysurvey/legacysurvey_{model}.parquet"
        col = f"{model}_legacysurvey"
        with HfFileSystem().open(path, "rb") as fh:
            tb = pq.ParquetFile(fh).read(columns=[col])
        arr = tb.column(0).combine_chunks()
        dim = arr.type.list_size
        flat = arr.flatten().to_numpy(zero_copy_only=False)
        # subset rows BEFORE widening to float64 so big models never materialize
        # the full 101725 x D array in double precision
        X = flat.reshape(-1, dim)[row_indices].astype(np.float64)
        del tb, arr, flat
        gc.collect()
        rec["load_seconds"] = round(time.perf_counter() - t0, 1)
        rec["ambient_dim"] = int(dim)

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if not (norms > 0).all():
            raise ValueError("zero-norm embedding row present")
        rec["raw_norm_mean"] = float(norms.mean())
        rec["raw_norm_std"] = float(norms.std())
        X /= norms  # L2 normalize, as the original analysis did

        # ---- connectivity: Isomap geodesics are undefined on a disconnected graph ----
        g = kneighbors_graph(X, n_neighbors=K, mode="distance", n_jobs=-1)
        ncomp, _ = connected_components(g, directed=False)
        del g
        gc.collect()
        rec["graph_n_components"] = int(ncomp)
        if ncomp != 1:
            rec["status"] = "skipped_disconnected"
            rec["total_seconds"] = round(time.perf_counter() - t_all, 1)
            return rec

        # ---- fit ----
        t0 = time.perf_counter()
        iso = Isomap(n_neighbors=K, n_components=NCOMP, eigen_solver="dense", n_jobs=-1)
        iso.fit(X)
        rec["fit_seconds"] = round(time.perf_counter() - t0, 1)

        # ---- full classical-MDS eigenspectrum, float64 end to end ----
        t0 = time.perf_counter()
        D2 = np.array(iso.dist_matrix_, dtype=np.float64, copy=True)
        del iso
        gc.collect()
        D2 **= 2
        D2 -= D2.mean(axis=1, keepdims=True)
        D2 -= D2.mean(axis=0, keepdims=True)
        D2 += D2.mean()
        D2 *= -0.5
        rec["symmetry_max"] = float(np.abs(D2 - D2.T).max())
        ev = eigvalsh(D2)
        del D2
        gc.collect()
        rec["eig_seconds"] = round(time.perf_counter() - t0, 1)

        if ev.shape != (N_ROWS,) or ev.dtype != np.float64:
            raise ValueError(f"spectrum shape/dtype wrong: {ev.shape} {ev.dtype}")

        neg, pos = ev[ev < 0], ev[ev > 0]
        if neg.size == 0:
            rec.update(r=0.0, m=0.0, n_positive=int(pos.size), n_negative=0,
                       lambda_max_pos=float(pos.max()), lambda_min_neg=0.0,
                       noise_floor=float(ev.size * np.finfo(np.float64).eps * pos.max()))
        else:
            rec.update(
                r=float(abs(neg.min()) / pos.max()),
                m=float(np.abs(neg).sum() / np.abs(ev).sum()),
                n_positive=int(pos.size), n_negative=int(neg.size),
                lambda_max_pos=float(pos.max()), lambda_min_neg=float(neg.min()),
                noise_floor=float(ev.size * np.finfo(np.float64).eps * pos.max()),
            )
        rec["verdict"] = classify(rec["r"], rec["m"])
        del ev
        gc.collect()

        # ---- intrinsic dimension (descriptive) ----
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(4000, len(X)), replace=False)
        d, _ = NearestNeighbors(n_neighbors=3, n_jobs=-1).fit(X).kneighbors(X[idx])
        r1, r2 = d[:, 1], d[:, 2]
        ok = (r1 > 0) & (r2 > r1)
        rec["twonn_dim"] = float(ok.sum() / np.log(r2[ok] / r1[ok]).sum())

        centers = rng.choice(len(X), size=600, replace=False)
        _, nbr = NearestNeighbors(n_neighbors=60, n_jobs=-1).fit(X).kneighbors(X[centers])
        dims = np.empty(600, dtype=np.int64)
        for i, row in enumerate(nbr):
            P = X[row] - X[row].mean(axis=0)
            e = np.linalg.svd(P, compute_uv=False) ** 2
            dims[i] = int(np.searchsorted(np.cumsum(e) / e.sum(), 0.90) + 1)
        rec["localpca_median"] = float(np.median(dims))
        rec["localpca_mean"] = float(dims.mean())
        rec["localpca_std"] = float(dims.std())

        rec["status"] = "ok"
        del X
        gc.collect()
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
        gc.collect()
    rec["total_seconds"] = round(time.perf_counter() - t_all, 1)
    return rec


def _worker(args):
    return run_one(*args)


def fmt(rec):
    if rec.get("status") == "ok":
        return (f"{rec['model']:24s} D={rec['ambient_dim']:5d}  r={rec['r']:.6f}  "
                f"m={rec['m']:.6f}  {rec['verdict']:8s}  "
                f"ID={rec['twonn_dim']:5.1f}/{rec['localpca_median']:4.0f}  "
                f"({rec['total_seconds']:.0f}s)")
    if rec.get("status") == "skipped_disconnected":
        return f"{rec['model']:24s} DISCONNECTED ({rec['graph_n_components']} components)"
    return f"{rec['model']:24s} ERROR  {rec.get('error','?')}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel models. Use <= min(cores//4, RAM_GB//4). Default 1.")
    ap.add_argument("--threads", type=int, default=4,
                    help="BLAS threads per worker (default 4)")
    ap.add_argument("--out", default=str(HERE / "model_sweep_results.jsonl"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these model names (must be in the pre-registered list)")
    ap.add_argument("--skip-control", action="store_true",
                    help="skip the reproduce-the-published-result check (not recommended)")
    args = ap.parse_args()

    row_indices = load_row_indices()
    print(f"row_indices verified: {N_ROWS} galaxies, sha256 ok")

    todo = list(MODELS)
    if args.only:
        bad = [m for m in args.only if m not in MODELS]
        if bad:
            sys.exit(f"FATAL: not in the pre-registered model list: {bad}\n"
                     f"Adding a model requires a new pre-registration.")
        todo = list(args.only)

    done = {}
    outp = Path(args.out)
    if outp.exists():
        for line in outp.read_text().splitlines():
            try:
                r = json.loads(line)
                done[r["model"]] = r
            except Exception:
                pass
    todo = [m for m in todo if m not in done]
    print(f"models: {len(MODELS)} pre-registered | {len(done)} already done | "
          f"{len(todo)} to run")

    # --- control check: must reproduce the published DINOv3 numbers ---
    if not args.skip_control and not args.only:
        ctrl = done.get(CONTROL_MODEL)
        if ctrl is None:
            print(f"\ncontrol check -- reproducing published {CONTROL_MODEL} result...")
            ctrl = run_one(CONTROL_MODEL, row_indices, args.threads)
            with outp.open("a") as f:
                f.write(json.dumps(ctrl) + "\n")
            done[CONTROL_MODEL] = ctrl
            todo = [m for m in todo if m != CONTROL_MODEL]
            print("  " + fmt(ctrl))
        if ctrl.get("status") != "ok":
            sys.exit("FATAL: control model failed to run. Fix the environment before trusting "
                     "any other result.")
        dr, dm = abs(ctrl["r"] - CONTROL_R), abs(ctrl["m"] - CONTROL_M)
        if dr > CONTROL_TOL or dm > CONTROL_TOL:
            sys.exit(f"FATAL: control mismatch.\n"
                     f"  expected r={CONTROL_R} m={CONTROL_M}\n"
                     f"  got      r={ctrl['r']:.6f} m={ctrl['m']:.6f}\n"
                     f"Environment differs from the reference run; results would not be "
                     f"comparable. Check library versions in requirements.txt.")
        print(f"  control OK -- reproduces r={CONTROL_R} m={CONTROL_M} "
              f"(delta {dr:.2e} / {dm:.2e})")

    if not todo:
        print("\nnothing to do.")
    else:
        print(f"\nrunning {len(todo)} models with {args.workers} worker(s), "
              f"{args.threads} BLAS threads each\n")
        t0 = time.perf_counter()
        payload = [(m, row_indices, args.threads) for m in todo]
        if args.workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers) as pool, outp.open("a") as f:
                for rec in pool.imap_unordered(_worker, payload):
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print("  " + fmt(rec), flush=True)
                    done[rec["model"]] = rec
        else:
            with outp.open("a") as f:
                for p in payload:
                    rec = _worker(p)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print("  " + fmt(rec), flush=True)
                    done[rec["model"]] = rec
        print(f"\nelapsed {(time.perf_counter()-t0)/60:.1f} min")

    # ---- summary: every pre-registered model, no dropping ----
    print("\n" + "=" * 100)
    print("FULL TABLE -- all pre-registered models, reported regardless of outcome")
    print("=" * 100)
    print(f"{'model':24s} {'D':>6s} {'r':>10s} {'m':>10s} {'verdict':>9s} "
          f"{'TwoNN':>7s} {'locPCA':>7s} {'neg':>6s}")
    ok = [done[m] for m in MODELS if m in done and done[m].get("status") == "ok"]
    for mname in MODELS:
        rec = done.get(mname)
        if rec is None:
            print(f"{mname:24s} {'--':>6s} {'not run':>10s}")
        elif rec.get("status") == "ok":
            print(f"{mname:24s} {rec['ambient_dim']:6d} {rec['r']:10.6f} {rec['m']:10.6f} "
                  f"{rec['verdict']:>9s} {rec['twonn_dim']:7.2f} "
                  f"{rec['localpca_median']:7.0f} {rec['n_negative']:6d}")
        else:
            print(f"{mname:24s} {'--':>6s} {rec.get('status'):>10s}  "
                  f"{rec.get('error','')[:40]}")

    if ok:
        verdicts = {}
        for r in ok:
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        print("\nverdict counts:", verdicts)
        passing = [r for r in ok if r["verdict"] in ("PASS", "MARGINAL")]
        accounted = [m for m in MODELS if m in done]
        complete = len(accounted) == len(MODELS)

        if passing:
            # Rule B fires as soon as ANY model clears FAIL -- it does not need the full sweep.
            print("\n*** RULE B -- not universal. Models clearing FAIL:")
            for r in passing:
                print(f"      {r['model']}  r={r['r']:.6f}  m={r['m']:.6f}  {r['verdict']}")
            print("    These are CANDIDATES only. Adopting one requires re-running the")
            print("    connectivity/plateau selection under it and a documented amendment.")
        elif complete:
            print("\n*** RULE A -- across the board. Every model returns FAIL.")
            print("    Classical MDS is an invalid description of Isomap geodesic geometry")
            print("    for this object population across all tested architectures.")
        else:
            # Rule A is a claim about ALL models and cannot be made from a partial sweep.
            missing = len(MODELS) - len(accounted)
            print(f"\n*** INCOMPLETE -- {len(ok)}/{len(MODELS)} models measured, "
                  f"{missing} not yet run.")
            print("    No rule fires. Rule A asserts something about EVERY model and cannot")
            print("    be concluded from a partial sweep. Re-run to continue; completed")
            print("    models are skipped automatically.")

        # pre-registered secondary analysis: does m simply track ambient dimension?
        try:
            from scipy.stats import spearmanr
            D = [r["ambient_dim"] for r in ok]
            M = [r["m"] for r in ok]
            ID = [r["twonn_dim"] for r in ok]
            rho, p = spearmanr(D, M)
            print(f"\nsecondary (pre-registered): spearman(m, ambient_dim) = "
                  f"{rho:+.3f}  p={p:.3g}  n={len(ok)}")
            rho2, p2 = spearmanr(ID, M)
            print(f"secondary (pre-registered): spearman(m, TwoNN dim)   = "
                  f"{rho2:+.3f}  p={p2:.3g}")
            print("  (descriptive only -- cannot override the primary rule)")
        except Exception as e:
            print(f"\n(secondary correlation skipped: {e})")

    print(f"\nresults -> {outp}")
    print("Send that JSONL file back. It contains no embeddings, only measurements.")


if __name__ == "__main__":
    main()
