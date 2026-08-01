"""
Fair re-scoring of the scale-free arms from stress_family_eval.py.

The distortion statistic median(|d2_rep - d2_geo| / d2_geo) penalizes absolute scale mismatch.
Three of the four arms evaluated do not preserve scale and never claimed to:

  - Laplacian eigenmaps: normalized eigenvector coordinates, arbitrary scale
  - LLE: same
  - non-metric MDS: preserves RANK ORDER only, so absolute scale is meaningless by construction

Scoring them raw measures the wrong thing. This re-fits an optimal isotropic scale s per arm,
minimizing the same statistic, and reports the best each method can achieve. Isomap and metric
SMACOF both target actual distances, so their raw scores were already fair and are carried over
unchanged as anchors.

This is a fairness correction to the measurement, not a revision of the statistic or of any
pre-registered rule: the statistic and the pair sample are unchanged, and the scale factor is
fitted per-arm to that arm's own advantage.
"""
import gc
import json
import os
import sys
import time

import numpy as np
import joblib
from scipy.optimize import minimize_scalar

sys.path.insert(0, "notebooks")
from pu_manifold import geometry_probes as gp  # noqa: E402

from sklearn.manifold import MDS, LocallyLinearEmbedding, SpectralEmbedding  # noqa: E402

CACHE = "notebooks/.cache"
FIT_KEY = "43cf438bc944c509"
PAIR_COUNT, PAIR_SEED = 200_000, 20260731
KREIN_BEST, ISOMAP_BEST, SMACOF_RAW = 0.065190, 0.079864, 0.079605
OUT = f"{CACHE}/stress_family_rescaled_{FIT_KEY}.json"
SEED, D_MAIN = 20260801, 18


def distortion_scaled(log_s, d2_rep, d2_geo):
    ok = d2_geo > 0
    return float(np.median(np.abs(np.exp(log_s) * d2_rep[ok] - d2_geo[ok]) / d2_geo[ok]))


def best_scale(coords, rows, cols, d2_geo):
    """Fit the isotropic scale that minimizes the statistic, to this arm's own advantage."""
    X = np.asarray(coords, dtype=np.float64)
    diff = X[rows] - X[cols]
    d2_rep = np.einsum("ij,ij->i", diff, diff)
    raw = distortion_scaled(0.0, d2_rep, d2_geo)
    # median ratio is a good starting point for a median-relative-error objective
    ok = (d2_geo > 0) & (d2_rep > 0)
    s0 = float(np.median(d2_geo[ok] / d2_rep[ok]))
    r = minimize_scalar(distortion_scaled, bracket=(np.log(s0) - 2, np.log(s0) + 2),
                        args=(d2_rep, d2_geo), method="brent",
                        options={"xtol": 1e-6, "maxiter": 200})
    return raw, float(np.exp(r.x)), float(r.fun)


iso = joblib.load(f"{CACHE}/isomap_{FIT_KEY}.joblib", mmap_mode="r")
D_mm = iso.dist_matrix_
n = D_mm.shape[0]
rows, cols = gp.draw_geo_pairs(np.random.default_rng(PAIR_SEED), n, PAIR_COUNT)
redrawn = np.array(D_mm[rows, cols], dtype=np.float64, copy=True)
cached = np.load(f"{CACHE}/mds_eigenspectrum_{FIT_KEY}.npz")["geo_pairs_r2"]
assert np.array_equal(redrawn, cached), "HALT: pair bit-identity failed"
print(f"pair bit-identity: True   n={n}")
d2_geo = redrawn ** 2
del redrawn, cached
gc.collect()

results = {"anchors": {"isomap_d18": ISOMAP_BEST, "krein_p40_q25": KREIN_BEST,
                       "metric_smacof_d18_raw": SMACOF_RAW},
           "note": "scale fitted per-arm to that arm's own advantage", "arms": {}}


def save():
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)


def run(name, fit_fn, note):
    print(f"\n--- {name} ---")
    t0 = time.perf_counter()
    Y = fit_fn()
    raw, s, best = best_scale(Y, rows, cols, d2_geo)
    dt = time.perf_counter() - t0
    flag = "  <-- BEATS KREIN" if best < KREIN_BEST else (
        "  (beats Isomap)" if best < ISOMAP_BEST else "")
    print(f"  raw={raw:.6f}  optimal_scale={s:.6g}  rescaled={best:.6f}  [{dt:.0f}s]{flag}")
    results["arms"][name] = {"raw": raw, "optimal_scale": s, "rescaled": best,
                             "seconds": round(dt, 1), "note": note}
    save()
    del Y
    gc.collect()


D_full = np.array(D_mm, dtype=np.float64, copy=True)
sigma = float(np.median(D_full[D_full > 0]))

run("laplacian_eigenmaps_d18",
    lambda: SpectralEmbedding(n_components=D_MAIN, affinity="precomputed",
                              random_state=SEED).fit_transform(
        np.exp(-(D_full ** 2) / (2.0 * sigma ** 2))),
    "normalized eigenvectors; no scale claim")

run("nonmetric_mds_d18",
    lambda: MDS(n_components=D_MAIN, metric=False, dissimilarity="precomputed", n_init=1,
                max_iter=150, random_state=SEED, n_jobs=-1,
                normalized_stress="auto").fit_transform(D_full),
    "rank-order only; absolute scale meaningless by construction")

del D_full
gc.collect()

LS = np.load(f"{CACHE}/subsample_20260729_a79b3460b838fd0a.npz")["legacysurvey"]
run("lle_standard_d18",
    lambda: LocallyLinearEmbedding(n_neighbors=15, n_components=D_MAIN, method="standard",
                                   random_state=SEED, n_jobs=-1).fit_transform(LS),
    "PSD-constrained by construction; no scale claim")

print("\n" + "=" * 78)
print("VERDICT (scale-free arms fitted to their own best scale)")
print("=" * 78)
table = {k: v["rescaled"] for k, v in results["arms"].items()}
table["metric SMACOF (d=18)"] = SMACOF_RAW
table["Isomap embedding (d=18)"] = ISOMAP_BEST
table["pseudo-Euclidean Krein (40,25)"] = KREIN_BEST
for k, v in sorted(table.items(), key=lambda kv: kv[1]):
    print(f"  {v:.6f}  {k}{' *' if v < KREIN_BEST else ''}")
w = min(table, key=table.get)
print(f"\nlowest: {w} at {table[w]:.6f}")
results["verdict"] = {"winner": w, "winner_distortion": table[w],
                      "beats_krein": bool(table[w] < KREIN_BEST), "table": table}
save()
print(f"\nwritten -> {OUT}")
