"""Plan 08-07 Task 1 — is the ambient k-NN density field curvature-contaminated?

DIAGNOSTIC ONLY. Gates nothing. Touches no frozen constant, appends no row to
`notebooks/.cache/08_cka_alignment.jsonl`, reopens no sealed verdict.

The question. Every density control in this milestone -- Phase 7's partial Spearman, 07.1's
within-stratum permutation, Phase 8's density-stratified tertile split -- rests on a k-NN density
estimate taken in the AMBIENT 768-D LegacySurvey space. On a curved manifold the ambient chord
runs shorter than the geodesic, and more so where curvature is higher, so high `||H||` would read
as high density purely as a metric artifact. If that is what produces the measured
`spearman(density, ||H||) = +0.4281` at `d=20`, then every density control is partly removing real
curvature signal and the milestone's effect is understated.

The test. Recompute density in the GRAPH GEODESIC metric, which carries no such bias, and compare.
Three readings, in increasing directness:

  1. `spearman(density_geodesic, ||H||)` against the ambient field's own value.
  2. `spearman(r_geodesic / r_ambient, ||H||)` -- the ratio IS the local chord-shortening, so under
     the contamination hypothesis it must correlate with `||H||`. This is the decisive column.
  3. the partial `spearman(||H||, MKNN)` controlling on each density in turn and on both jointly.

Why the `DENSITY_FIELD_D = 20` exponent cannot matter here. `curvature_probe.local_density_weights`
computes `rho = k / (n * V_d * r^d)` at fixed `d`, so `log(rho) = const - d * log(r)`: density is a
strictly DECREASING monotone function of the radius. Every control in this milestone is rank-based
(partial Spearman on ranks; quantile strata), and rank statistics are invariant under monotone
transforms -- so every one of them is exactly equivalent to controlling for the raw k-NN radius, and
neither the assumed `d` nor the gamma-function ball volume can move any result. This runner
therefore uses `-r` directly wherever only ranks are needed, and records
`spearman(density_ambient, -r_ambient)` as the check that the equivalence holds numerically.

Usage:
    python notebooks/diagnostics/08_density_control_diagnostic_run.py
    python notebooks/diagnostics/08_density_control_diagnostic_run.py --threads 4
"""

import os
import sys


def _flag_value_from_argv(flag, argv):
    """Value passed for `flag`, accepting both `--flag value` and `--flag=value`. Copied from
    `07.1_density_stratified_null_run.py` (CR-03): a raw `flag in argv` scan misses the `=` form."""
    prefix = flag + "="
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


# Thread cap before numpy is pulled in, mirroring this directory's other runners.
_THREADS = _flag_value_from_argv("--threads", sys.argv)
if _THREADS is not None:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = str(int(_THREADS))

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pu_manifold import cache  # noqa: E402
from pu_manifold import crossmodal_curvature as cc  # noqa: E402
from pu_manifold import curvature_probe as cp  # noqa: E402
from pu_manifold import geodesic_graph as gg  # noqa: E402

RECORD_STEM = "08_density_control_diagnostic"
SUBSAMPLE_STEM = "subsample_20260729_a79b3460b838fd0a"
K_GRAPH = 15
"""Phase 2's frozen `k*` for the k-NN graph the geodesic metric is read off. Not a new choice --
`02-FINDINGS.md`'s own working graph, reused so the geodesic here is the milestone's geodesic."""

DIJKSTRA_CHUNK = 500
"""Dijkstra source chunk. Only a memory knob: `n x n` float64 at n=10,000 is 800 MB in one go."""


def _run_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def geodesic_knn_radius(X, k_density):
    """Distance from each point to its `k_density`-th nearest neighbour in the GRAPH GEODESIC
    metric -- shortest path over `gg.build_symmetric_knn_graph(X, K_GRAPH)`, not the ambient chord.

    Returns `(r_geodesic, component_readout)`. The graph was measured connected on this subsample
    (`n_components = 1`), but a later run must not assume it: any point left unreachable gets
    `inf`, and the caller drops it and reports how many were dropped.
    """
    n = X.shape[0]
    G = gg.build_symmetric_knn_graph(X, K_GRAPH)
    readout = gg.component_readout(G)
    r = np.full(n, np.inf)
    for s in range(0, n, DIJKSTRA_CHUNK):
        idx = np.arange(s, min(s + DIJKSTRA_CHUNK, n))
        D = dijkstra(G, directed=False, indices=idx)
        D.sort(axis=1)
        r[idx] = D[:, k_density]  # column 0 is the source itself, at distance 0
    return r, readout


def partial_on_ranks(x, y, controls):
    """Rank-space partial correlation: rank-transform `x`, `y` and every control column,
    residualise `x` and `y` against the controls (with intercept) by least squares, return the
    Pearson correlation of the residuals. Same construction as
    `cross_split_curvature.partial_spearman`, extended to accept SEVERAL controls -- which that
    sealed function already supports via its `(n, c)` argument, and which is used here to control
    on ambient and geodesic density jointly."""
    rx = rankdata(x)
    ry = rankdata(y)
    A = np.column_stack([np.ones(rx.shape[0])] + [rankdata(c) for c in controls])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=None, help="cap BLAS/OMP threads")
    ap.add_argument("--record-path", default=None,
                    help="override the JSONL destination (default: the frozen cache stem)")
    args = ap.parse_args()

    record_path = (args.record_path if args.record_path is not None
                   else str(cache.cache_path(RECORD_STEM, "jsonl")))
    t_start = time.time()

    sub = np.load(cache.cache_path(SUBSAMPLE_STEM, "npz"))
    X = sub["legacysurvey"]
    n = X.shape[0]
    print(f"loaded legacysurvey {X.shape} from {SUBSAMPLE_STEM}.npz", flush=True)

    # --- the ambient field, built exactly as every phase builds it -------------------------------
    w = cp.local_density_weights(X, cc.DENSITY_K, cc.DENSITY_FIELD_D)
    dens_ambient = 1.0 / w
    nbrs = NearestNeighbors(n_neighbors=cc.DENSITY_K + 1).fit(X)
    r_ambient = nbrs.kneighbors(X)[0][:, cc.DENSITY_K]

    # The monotone-equivalence check that licenses using -r as the rank proxy throughout.
    monotone_rho = float(spearmanr(dens_ambient, -r_ambient).statistic)
    print(f"[monotone check] spearman(density_ambient, -r_ambient) = {monotone_rho:.12f}",
          flush=True)

    # --- the geodesic field ---------------------------------------------------------------------
    t0 = time.time()
    r_geodesic, readout = geodesic_knn_radius(X, cc.DENSITY_K)
    ok = np.isfinite(r_geodesic) & (r_geodesic > 0)
    print(f"[geodesic] knn graph k={K_GRAPH}: n_components={readout['n_components']} "
          f"largest_size={readout['largest_size']} dropped_fraction={readout['dropped_fraction']:.6f}",
          flush=True)
    print(f"[geodesic] radii in {time.time() - t0:.0f}s; usable {int(ok.sum())}/{n}", flush=True)

    ratio = r_geodesic[ok] / r_ambient[ok]
    mknn = cc.per_point_mknn(sub["hsc"], sub["legacysurvey"], cc.HEADLINE_K)

    fields = np.load(cache.cache_path("07_crossmodal_curvature_fields", "npz"))
    run_commit = _run_commit()
    stamp = datetime.now(timezone.utc).isoformat()
    rows = []

    print(f"\n{'d':>4} {'rho(dens_amb,H)':>16} {'rho(dens_geo,H)':>16} {'rho(geo/amb,H)':>15} "
          f"{'raw':>10} {'|amb':>10} {'|geo':>10} {'|both':>10}", flush=True)
    print("-" * 96, flush=True)
    for d in cc.D_SWEEP:
        h = fields[f"h_norm_{d}"][ok]
        m = mknn[ok]
        row = dict(
            row_kind="per_d", d=int(d), gates_nothing=True,
            spearman_density_ambient_h=float(spearmanr(dens_ambient[ok], h).statistic),
            spearman_density_geodesic_h=float(spearmanr(-r_geodesic[ok], h).statistic),
            spearman_ratio_h=float(spearmanr(ratio, h).statistic),
            raw_spearman_h_mknn=float(spearmanr(h, m).statistic),
            partial_control_ambient=partial_on_ranks(h, m, [dens_ambient[ok]]),
            partial_control_geodesic=partial_on_ranks(h, m, [-r_geodesic[ok]]),
            partial_control_both=partial_on_ranks(h, m, [dens_ambient[ok], -r_geodesic[ok]]),
            n_used=int(ok.sum()), density_k=int(cc.DENSITY_K),
            density_field_d=int(cc.DENSITY_FIELD_D), k_graph=int(K_GRAPH),
            headline_k=int(cc.HEADLINE_K), run_commit=run_commit, timestamp=stamp,
        )
        rows.append(row)
        print(f"{d:>4} {row['spearman_density_ambient_h']:>16.6f} "
              f"{row['spearman_density_geodesic_h']:>16.6f} {row['spearman_ratio_h']:>15.6f} "
              f"{row['raw_spearman_h_mknn']:>10.6f} {row['partial_control_ambient']:>10.6f} "
              f"{row['partial_control_geodesic']:>10.6f} {row['partial_control_both']:>10.6f}",
              flush=True)

    q = [5, 50, 95]
    rows.append(dict(
        row_kind="summary", gates_nothing=True,
        spearman_density_ambient_vs_geodesic=float(spearmanr(dens_ambient[ok], -r_geodesic[ok]).statistic),
        spearman_density_ambient_vs_neg_r_ambient=monotone_rho,
        r_ambient_p05_p50_p95=[float(v) for v in np.percentile(r_ambient[ok], q)],
        r_geodesic_p05_p50_p95=[float(v) for v in np.percentile(r_geodesic[ok], q)],
        geodesic_over_ambient_p05_p50_p95=[float(v) for v in np.percentile(ratio, q)],
        spearman_density_ambient_mknn=float(spearmanr(dens_ambient[ok], mknn[ok]).statistic),
        n_components=int(readout["n_components"]), largest_size=int(readout["largest_size"]),
        n_used=int(ok.sum()), k_graph=int(K_GRAPH), run_commit=run_commit, timestamp=stamp,
        wallclock_s=round(time.time() - t_start, 2),
    ))

    with open(record_path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    s = rows[-1]
    print(f"\ngeodesic/ambient radius ratio p05/p50/p95 = "
          f"{s['geodesic_over_ambient_p05_p50_p95'][0]:.4f} / "
          f"{s['geodesic_over_ambient_p05_p50_p95'][1]:.4f} / "
          f"{s['geodesic_over_ambient_p05_p50_p95'][2]:.4f}")
    print(f"spearman(density_ambient, density_geodesic) = "
          f"{s['spearman_density_ambient_vs_geodesic']:.4f}")
    print(f"\nwrote {len(rows)} rows to {record_path}  ({s['wallclock_s']:.0f}s)")
    print("DENSITY CONTROL DIAGNOSTIC COMPLETE -- gates nothing, no frozen constant read as a "
          "gating value, no row appended to 08_cka_alignment.jsonl")


if __name__ == "__main__":
    main()
