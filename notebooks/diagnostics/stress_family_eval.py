"""
Evaluate stress-optimizing and PSD-unconstrained sklearn representations against Isomap's
geodesic metric.

Motivation: every candidate measured so far in Phase 02.1 is either classical-MDS-family
(eigendecomposition of a double-centred Gram matrix, which is exactly where the negative
eigenvalues bite) or ambient-sphere (geomstats arms, which ignore the manifold). The methods
below optimize a stress functional directly, or build a graph Laplacian, and inherit NO
positive-semidefiniteness requirement. Non-metric MDS in particular is the textbook answer to
"these dissimilarities are not Euclidean" -- it preserves only rank order.

Judged on the SAME statistic over the SAME 200,000-pair sample as every prior candidate:

    DISTORTION = median(|d2_rep - d2_geo| / d2_geo)

Anchors (previously measured):
    Isomap embedding (d=18)          0.079864
    classical MDS (p=18, q=0)        0.079864
    pseudo-Euclidean Krein (40,25)   0.065190   <-- the bar

Results are written incrementally so a slow arm cannot lose the earlier ones.
"""
import gc
import json
import os
import sys
import time

import numpy as np
import joblib

sys.path.insert(0, "notebooks")
from pu_manifold import geometry_probes as gp  # noqa: E402

from sklearn.manifold import (  # noqa: E402
    MDS,
    LocallyLinearEmbedding,
    SpectralEmbedding,
    TSNE,
)

CACHE = "notebooks/.cache"
FIT_KEY = "43cf438bc944c509"
PAIR_COUNT = 200_000
PAIR_SEED = 20260731
KREIN_BEST = 0.065190
ISOMAP_BEST = 0.079864
OUT = f"{CACHE}/stress_family_eval_{FIT_KEY}.json"
SEED = 20260801
D_MAIN = 18  # matches Isomap's n_components, so arms are compared at equal budget


def distortion(d2_rep, d2_geo):
    ok = d2_geo > 0
    return float(np.median(np.abs(d2_rep[ok] - d2_geo[ok]) / d2_geo[ok]))


def score(coords, rows, cols, d2_geo):
    diff = np.asarray(coords, dtype=np.float64)[rows] - np.asarray(coords, dtype=np.float64)[cols]
    return distortion(np.einsum("ij,ij->i", diff, diff), d2_geo)


def save(results):
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, sort_keys=True)


print("=" * 78)
print("STRESS-FAMILY / PSD-UNCONSTRAINED EVALUATION vs ISOMAP GEODESIC METRIC")
print("=" * 78)

for p in (f"{CACHE}/isomap_{FIT_KEY}.joblib",
          f"{CACHE}/mds_eigenspectrum_{FIT_KEY}.npz",
          f"{CACHE}/subsample_20260729_a79b3460b838fd0a.npz"):
    if not os.path.exists(p):
        raise SystemExit(f"HALT: {p} missing; gitignored and NOT reproducible here. "
                         f"Regenerating would change provenance.")

iso = joblib.load(f"{CACHE}/isomap_{FIT_KEY}.joblib", mmap_mode="r")
D_geo_mm = iso.dist_matrix_
n = D_geo_mm.shape[0]

# Bit-identity guard: delegate to the ONE canonical implementation (02-PATTERNS.md idiom).
rows, cols = gp.draw_geo_pairs(np.random.default_rng(PAIR_SEED), n, PAIR_COUNT)
redrawn = np.array(D_geo_mm[rows, cols], dtype=np.float64, copy=True)
cached = np.load(f"{CACHE}/mds_eigenspectrum_{FIT_KEY}.npz")["geo_pairs_r2"]
if not np.array_equal(redrawn, cached):
    raise SystemExit("HALT: re-drawn pairs not bit-identical to cached geo_pairs_r2. Source of "
                     "truth is notebook 01 section 6.1's draw.")
print(f"pair bit-identity: True   n={n}")
d2_geo = redrawn ** 2
del redrawn, cached
gc.collect()

results = {"anchors": {"isomap_d18": ISOMAP_BEST, "krein_p40_q25": KREIN_BEST},
           "pair_identity_verified": True, "d_main": D_MAIN, "seed": SEED, "arms": {}}
save(results)


def record(name, val, seconds, note=""):
    flag = "  <-- BEATS KREIN" if val < KREIN_BEST else (
        "  (beats Isomap)" if val < ISOMAP_BEST else "")
    print(f"  {name:34s} distortion = {val:.6f}  [{seconds:.0f}s]{flag}")
    results["arms"][name] = {"distortion": val, "seconds": round(seconds, 1), "note": note}
    save(results)


# ---- fast arms first ----
print("\n--- Laplacian eigenmaps (SpectralEmbedding, precomputed affinity) ---")
try:
    t0 = time.perf_counter()
    # Heat kernel on geodesic distances -> affinity. sigma = median geodesic, a standard choice.
    D_full = np.array(D_geo_mm, dtype=np.float64, copy=True)
    sigma = float(np.median(D_full[D_full > 0]))
    A = np.exp(-(D_full ** 2) / (2.0 * sigma ** 2))
    del D_full
    gc.collect()
    se = SpectralEmbedding(n_components=D_MAIN, affinity="precomputed", random_state=SEED)
    Y = se.fit_transform(A)
    del A
    gc.collect()
    record("laplacian_eigenmaps_d18", score(Y, rows, cols, d2_geo), time.perf_counter() - t0,
           f"heat kernel, sigma=median geodesic={sigma:.4f}")
    del Y, se
    gc.collect()
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    results["arms"]["laplacian_eigenmaps_d18"] = {"error": f"{type(e).__name__}: {e}"}
    save(results)

print("\n--- LLE (standard), on raw embeddings, k=15 to match Isomap ---")
try:
    t0 = time.perf_counter()
    LS = np.load(f"{CACHE}/subsample_20260729_a79b3460b838fd0a.npz")["legacysurvey"]
    lle = LocallyLinearEmbedding(n_neighbors=15, n_components=D_MAIN, method="standard",
                                 random_state=SEED, n_jobs=-1)
    Y = lle.fit_transform(LS)
    record("lle_standard_d18", score(Y, rows, cols, d2_geo), time.perf_counter() - t0,
           "PSD-constrained by construction; cannot expose negative eigenvalues")
    del Y, lle
    gc.collect()
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    results["arms"]["lle_standard_d18"] = {"error": f"{type(e).__name__}: {e}"}
    save(results)

# ---- the headline arms: stress optimizers on the precomputed geodesic matrix ----
D_full = np.array(D_geo_mm, dtype=np.float64, copy=True)
print(f"\ngeodesic matrix materialized for SMACOF: {D_full.nbytes/1e9:.2f} GB")

for label, metric_flag, note in [
    ("nonmetric_mds_d18", False,
     "rank-order only; assumes NO embeddability -- the textbook answer to non-Euclidean input"),
    ("metric_smacof_d18", True,
     "stress majorization; no eigendecomposition, no PSD requirement"),
]:
    print(f"\n--- {label} (n_init=1, max_iter=150) ---")
    try:
        t0 = time.perf_counter()
        mds = MDS(n_components=D_MAIN, metric=metric_flag, dissimilarity="precomputed",
                  n_init=1, max_iter=150, random_state=SEED, n_jobs=-1,
                  normalized_stress="auto")
        Y = mds.fit_transform(D_full)
        record(label, score(Y, rows, cols, d2_geo), time.perf_counter() - t0,
               f"{note}; final stress={float(mds.stress_):.6g}")
        del Y, mds
        gc.collect()
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        results["arms"][label] = {"error": f"{type(e).__name__}: {e}"}
        save(results)

del D_full
gc.collect()

# ---- verdict ----
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
table = {k: v["distortion"] for k, v in results["arms"].items() if "distortion" in v}
table["Isomap embedding (d=18)"] = ISOMAP_BEST
table["pseudo-Euclidean Krein (40,25)"] = KREIN_BEST
for name, val in sorted(table.items(), key=lambda kv: kv[1]):
    mark = " *" if val < KREIN_BEST else ""
    print(f"  {val:.6f}  {name}{mark}")

winner = min(table, key=table.get)
beats = table[winner] < KREIN_BEST
print(f"\nlowest distortion: {winner} at {table[winner]:.6f}")
if not beats:
    print("Nothing beats the Krein (40,25) anchor. Legitimate, reportable outcome.")
results["verdict"] = {"winner": winner, "winner_distortion": table[winner],
                      "beats_krein": bool(beats), "table": table}
save(results)
print(f"\nwritten -> {OUT}")
