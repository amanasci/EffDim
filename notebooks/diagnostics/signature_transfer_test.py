"""
Does the pseudo-Euclidean advantage survive being handed to a decoder?

THE QUESTION. Phase 02.1 measured pseudo-Euclidean (p=40, q=25) coordinates at distortion
0.065190, beating the best flat-target result (0.079864) by 18.4%. Phase 3 would train a
C2-smooth decoder f: R^(p+q) -> R^768 on those coordinates and derive curvature from its
Jacobian.

But that 0.065190 is measured under the INDEFINITE form:

    d2_indef(i,j) = sum_a s_a * (X[i,a] - X[j,a])^2      with s_a = -1 for the q block

A decoder consumes the coordinate array. It does not consume the signature. Its loss, its
Jacobian, and the pullback metric J^T J all treat the latent as ordinary Euclidean R^(p+q):

    d2_eucl(i,j) = sum_a (X[i,a] - X[j,a])^2             all signs +1

If the advantage lives in the signs, a Euclidean-latent decoder cannot access it, and the
representation's measured superiority does not transfer to Phase 3.

THE TEST. Score the SAME coordinates both ways, on the same pair sample, same statistic.
No new fit. Deterministic. The comparison is exact, not an estimate.
"""
import gc
import sys

import numpy as np
import joblib

sys.path.insert(0, "notebooks")
from pu_manifold import geometry_probes as gp  # noqa: E402

CACHE = "notebooks/.cache"
FIT_KEY = "43cf438bc944c509"
PAIR_COUNT, PAIR_SEED = 200_000, 20260731
KREIN_BEST, FLAT_FLOOR = 0.065190, 0.079864
P_BEST, Q_BEST = 40, 25


def distortion(d2_rep, d2_geo):
    ok = d2_geo > 0
    return float(np.median(np.abs(d2_rep[ok] - d2_geo[ok]) / d2_geo[ok]))


iso = joblib.load(f"{CACHE}/isomap_{FIT_KEY}.joblib", mmap_mode="r")
D_mm = iso.dist_matrix_
n = D_mm.shape[0]
rows, cols = gp.draw_geo_pairs(np.random.default_rng(PAIR_SEED), n, PAIR_COUNT)
redrawn = np.array(D_mm[rows, cols], dtype=np.float64, copy=True)

z = np.load(f"{CACHE}/mds_eigenspectrum_{FIT_KEY}.npz")
assert np.array_equal(redrawn, z["geo_pairs_r2"]), "HALT: pair bit-identity failed"
print(f"pair bit-identity: True   n={n}")
d2_geo = redrawn ** 2
del redrawn
gc.collect()

# top-p positive block (cached by Phase 2) and bottom-q negative block (cached by plan 02.1-03)
ev_top, evec_top = z["eigvals_top"], z["eigvecs_top"]
b = np.load(f"{CACHE}/krein_bottom_{FIT_KEY}.npz")
ev_bot, evec_bot = b["eigvals_bottom"], b["eigvecs_bottom"]
print(f"positive block {ev_top.shape}, negative block {ev_bot.shape}")
print(f"  top eigenvalue  {ev_top.max():+.4f}   bottom eigenvalue {ev_bot.min():+.4f}")

# order: descending positive, then most-negative first
order_pos = np.argsort(ev_top)[::-1]
order_neg = np.argsort(ev_bot)

lam = np.concatenate([ev_top[order_pos][:P_BEST], ev_bot[order_neg][:Q_BEST]])
V = np.concatenate([evec_top[:, order_pos][:, :P_BEST], evec_bot[:, order_neg][:, :Q_BEST]], axis=1)
signs = np.sign(lam)
X = V * np.sqrt(np.abs(lam))[None, :]
print(f"\ncoordinates X: {X.shape}   signature = ({int((signs>0).sum())}, {int((signs<0).sum())})")

diff = X[rows] - X[cols]
sq = diff ** 2

d2_indef = (signs[None, :] * sq).sum(axis=1)      # what the phase measured
d2_eucl = sq.sum(axis=1)                          # what a decoder's latent actually is
d2_pos_only = sq[:, signs > 0].sum(axis=1)        # drop the q block entirely, p=40 classical

D_indef = distortion(d2_indef, d2_geo)
D_eucl = distortion(d2_eucl, d2_geo)
D_pos = distortion(d2_pos_only, d2_geo)

print("\n" + "=" * 78)
print("SAME COORDINATES, THREE READINGS")
print("=" * 78)
print(f"  indefinite form  (signature honored, q terms SUBTRACTED)   {D_indef:.6f}")
print(f"  Euclidean form   (signature discarded, all terms ADDED)    {D_eucl:.6f}")
print(f"  positive block only (p=40, q=0 -- classical MDS)           {D_pos:.6f}")
print(f"\n  flat-target floor measured across all methods             {FLAT_FLOOR:.6f}")

print("\n" + "=" * 78)
print("WHAT THIS MEANS FOR PHASE 3")
print("=" * 78)
transfers = D_eucl < FLAT_FLOOR
if transfers:
    print(f"  Euclidean reading ({D_eucl:.6f}) still beats the flat floor ({FLAT_FLOOR:.6f}).")
    print("  The advantage SURVIVES discarding the signature -> a Euclidean-latent decoder")
    print("  can access it, and the representation transfers to Phase 3.")
else:
    print(f"  Euclidean reading ({D_eucl:.6f}) is WORSE than the flat floor ({FLAT_FLOOR:.6f}).")
    print("  The advantage lives ENTIRELY in the signs. A decoder consuming these coordinates")
    print("  as ordinary R^65 vectors cannot access it. The measured superiority does NOT")
    print("  transfer to Phase 3 under the standard J^T J pullback.")

# How much negative structure is there, really?
neg_share = float(np.mean(d2_indef < 0))
print(f"\n  fraction of sampled pairs whose indefinite d2 is NEGATIVE: {neg_share:.4%}")
print("  (a genuinely pseudo-Euclidean pair has no real distance; a decoder trained on a")
print("   Euclidean loss over these coordinates has no way to represent that)")

sign_gap = D_eucl - D_indef
print(f"\n  distortion penalty for discarding the signature: {sign_gap:+.6f} "
      f"({100*sign_gap/D_indef:+.1f}%)")

import json  # noqa: E402
out = f"{CACHE}/signature_transfer_{FIT_KEY}.json"
with open(out, "w") as f:
    json.dump({"p": P_BEST, "q": Q_BEST,
               "distortion_indefinite": D_indef, "distortion_euclidean": D_eucl,
               "distortion_positive_block_only": D_pos, "flat_floor": FLAT_FLOOR,
               "advantage_transfers_to_euclidean_latent": bool(transfers),
               "negative_d2_pair_fraction": neg_share,
               "signature_discard_penalty": sign_gap}, f, indent=1, sort_keys=True)
print(f"\nwritten -> {out}")
