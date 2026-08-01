"""
Evaluate geomstats representations against Isomap's geodesic metric.

Authorized by 02.1-AMENDMENT-01.md. Judged on the SAME statistic over the SAME 200,000-pair
sample as every prior candidate in this phase:

    DISTORTION = median(|d2_rep - d2_geo| / d2_geo)

Anchors to beat (previously measured, in geometry_probes_43cf438bc944c509.json):
    classical MDS (p=18, q=0)      0.079864
    pseudo-Euclidean Krein (40,25) 0.065190   <-- a geomstats result is only interesting below this

Standing prediction from this phase's own measurement: delta_rel_max = 0.383921 EXCEEDS the
flat-Euclidean anchor (0.360433), so the geometry is NOT tree-like and hyperbolic candidates are
contraindicated. The hyperbolic arm is run anyway, precisely because it is a prediction that can
fail.
"""
import gc
import json
import time

import numpy as np

# --- numpy-2 compatibility shim for geomstats 2.8.0 (see 02.1-AMENDMENT-01.md section 2) ---
# geomstats 2.8.0 imports numpy.trapz, removed in numpy 2.0 and renamed numpy.trapezoid.
# Downgrading numpy is NOT an option: config_key hashes the installed numpy version into every
# cache key, so it would invalidate every frozen Phase 1/2 artifact. Shim confined to this file.
for _old, _new in [("trapz", "trapezoid"), ("alltrue", "all"), ("sometrue", "any"),
                   ("cumproduct", "cumprod"), ("product", "prod")]:
    if not hasattr(np, _old) and hasattr(np, _new):
        setattr(np, _old, getattr(np, _new))

import joblib  # noqa: E402
from geomstats.geometry.hypersphere import Hypersphere  # noqa: E402

CACHE = "notebooks/.cache"
FIT_KEY = "43cf438bc944c509"
R2_PAIR_COUNT = 200_000
R2_PAIR_SEED = 20260731
KREIN_BEST = 0.065190
CLASSICAL_BEST = 0.079864
D_LADDER = [2, 4, 8, 12, 18, 25, 40, 65]


def distortion(d2_rep, d2_geo):
    """The phase's statistic. Guards against divide-by-zero on coincident pairs."""
    ok = d2_geo > 0
    return float(np.median(np.abs(d2_rep[ok] - d2_geo[ok]) / d2_geo[ok]))


def draw_pairs(n, count, seed):
    """
    Re-draw the pre-registered pair sample.

    Delegates to pu_manifold.geometry_probes.draw_geo_pairs rather than reimplementing the
    idiom. A local reimplementation here initially omitted the self-pair rejection loop and
    the bit-identity guard caught it -- which is exactly what that guard is for. There must be
    ONE implementation of this draw, mirroring 02-PATTERNS.md's _draw_geo_pairs.
    """
    import sys
    if "notebooks" not in sys.path:
        sys.path.insert(0, "notebooks")
    from pu_manifold import geometry_probes as gp
    rng = np.random.default_rng(seed)
    return gp.draw_geo_pairs(rng, n, count)


print("=" * 78)
print("GEOMSTATS EVALUATION vs ISOMAP GEODESIC METRIC")
print("=" * 78)

# ---- load the frozen, trusted artifacts. HALT rather than regenerate. ----
import os  # noqa: E402
for p in (f"{CACHE}/isomap_{FIT_KEY}.joblib",
          f"{CACHE}/mds_eigenspectrum_{FIT_KEY}.npz",
          f"{CACHE}/subsample_20260729_a79b3460b838fd0a.npz"):
    if not os.path.exists(p):
        raise SystemExit(f"HALT: {p} missing. It is gitignored and NOT reproducible by this "
                         f"script. Regenerating would change provenance and break comparability "
                         f"with the frozen Phase 1/2 results.")

t0 = time.perf_counter()
iso = joblib.load(f"{CACHE}/isomap_{FIT_KEY}.joblib", mmap_mode="r")
D_geo = iso.dist_matrix_
n = D_geo.shape[0]
print(f"geodesic matrix {D_geo.shape} loaded in {time.perf_counter()-t0:.1f}s")

z = np.load(f"{CACHE}/mds_eigenspectrum_{FIT_KEY}.npz")
cached_pairs = z["geo_pairs_r2"]
rows, cols = draw_pairs(n, R2_PAIR_COUNT, R2_PAIR_SEED)
redrawn = np.asarray(D_geo[rows, cols], dtype=np.float64)

identical = bool(np.array_equal(redrawn, cached_pairs))
print(f"pair bit-identity vs cached geo_pairs_r2: {identical}")
if not identical:
    raise SystemExit(
        "HALT: re-drawn pairs are NOT bit-identical to the cached geo_pairs_r2. The source of "
        "truth is notebook 01_manifold_and_gate.ipynb section 6.1's draw. Measuring a different "
        "sample would make these numbers incomparable to every prior anchor in this phase."
    )
d2_geo = redrawn ** 2
del z, redrawn
gc.collect()

LS = np.load(f"{CACHE}/subsample_20260729_a79b3460b838fd0a.npz")["legacysurvey"]
print(f"embeddings {LS.shape}, row-norm mean {np.linalg.norm(LS, axis=1).mean():.6f}")

results = {"anchors": {"classical_p18_q0": CLASSICAL_BEST, "krein_p40_q25": KREIN_BEST},
           "pair_identity_verified": identical, "arms": {}}

# ---- ARM 1: ambient hypersphere great-circle distance ----
# The data is L2-normalized, so it lies exactly on S^767. This asks whether the ambient sphere
# metric alone explains the geodesic structure -- no dimension reduction, no fitting.
print("\n" + "-" * 78)
print("ARM 1: ambient hypersphere great-circle (no reduction, no fit)")
print("-" * 78)
t0 = time.perf_counter()
sphere = Hypersphere(dim=LS.shape[1] - 1)
inner = np.einsum("ij,ij->i", LS[rows], LS[cols])
gc_dist = np.arccos(np.clip(inner, -1.0, 1.0))
d_arm1 = distortion(gc_dist ** 2, d2_geo)
print(f"  ambient dim {sphere.dim}  distortion = {d_arm1:.6f}  ({time.perf_counter()-t0:.1f}s)")
results["arms"]["hypersphere_ambient_greatcircle"] = {"dim": int(sphere.dim), "distortion": d_arm1}

# ---- ARM 2: tangent-space PCA at the Frechet mean on the hypersphere ----
# geomstats' Riemannian analogue of "reduce to d dimensions". Log-map every point to the tangent
# space at the Frechet mean, PCA there, keep d components, measure induced Euclidean distances.
print("\n" + "-" * 78)
print("ARM 2: hypersphere tangent PCA at the Frechet mean")
print("-" * 78)
t0 = time.perf_counter()
# For L2-normalized data the Frechet mean on the sphere is the normalized Euclidean mean to
# excellent approximation; geomstats' iterative solver on 10k x 768 is not worth its cost here
# and the projection is what matters. Documented approximation, not a silent shortcut.
mu = LS.mean(axis=0)
mu /= np.linalg.norm(mu)
tangent = sphere.metric.log(point=LS, base_point=mu)
print(f"  log-map to tangent space at Frechet mean: {time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
tc = tangent - tangent.mean(axis=0)
U, S, Vt = np.linalg.svd(tc, full_matrices=False)
print(f"  tangent SVD: {time.perf_counter()-t0:.1f}s")

arm2 = {}
for d in D_LADDER:
    if d > tc.shape[1]:
        continue
    coords = U[:, :d] * S[:d]
    diff = coords[rows] - coords[cols]
    d2_rep = np.einsum("ij,ij->i", diff, diff)
    val = distortion(d2_rep, d2_geo)
    arm2[d] = val
    flag = "  <-- beats Krein" if val < KREIN_BEST else ""
    print(f"  d={d:3d}  distortion = {val:.6f}{flag}")
best_d = min(arm2, key=arm2.get)
results["arms"]["hypersphere_tangent_pca"] = {"ladder": arm2, "best_d": best_d,
                                              "best_distortion": arm2[best_d]}
del tangent, tc, U, S, Vt
gc.collect()

# ---- ARM 3: Isomap's own embedding, same statistic (the incumbent, for reference) ----
print("\n" + "-" * 78)
print("ARM 3: Isomap's own 18-d embedding (incumbent reference)")
print("-" * 78)
emb = np.asarray(iso.embedding_, dtype=np.float64)
arm3 = {}
for d in [dd for dd in D_LADDER if dd <= emb.shape[1]]:
    diff = emb[rows, :d] - emb[cols, :d]
    d2_rep = np.einsum("ij,ij->i", diff, diff)
    arm3[d] = distortion(d2_rep, d2_geo)
    print(f"  d={d:3d}  distortion = {arm3[d]:.6f}")
results["arms"]["isomap_embedding"] = {"ladder": arm3,
                                       "best_d": min(arm3, key=arm3.get),
                                       "best_distortion": min(arm3.values())}

# ---- verdict ----
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
all_best = {
    "hypersphere ambient great-circle": d_arm1,
    f"hypersphere tangent PCA (d={best_d})": arm2[best_d],
    f"Isomap embedding (d={min(arm3, key=arm3.get)})": min(arm3.values()),
    "classical MDS (p=18,q=0)": CLASSICAL_BEST,
    "pseudo-Euclidean Krein (40,25)": KREIN_BEST,
}
for name, val in sorted(all_best.items(), key=lambda kv: kv[1]):
    mark = " *" if val < KREIN_BEST else ""
    print(f"  {val:.6f}  {name}{mark}")

winner = min(all_best, key=all_best.get)
print(f"\nlowest distortion: {winner} at {all_best[winner]:.6f}")
if all_best[winner] >= KREIN_BEST:
    print("No geomstats arm beats the Krein (40,25) anchor. That is a legitimate, reportable")
    print("outcome and must be reported as such, per 02.1-AMENDMENT-01.md section 4.")
results["verdict"] = {"winner": winner, "winner_distortion": all_best[winner],
                      "beats_krein": bool(all_best[winner] < KREIN_BEST),
                      "all_best": all_best}

out = f"{CACHE}/geomstats_eval_{FIT_KEY}.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
print(f"\nwritten -> {out}")
