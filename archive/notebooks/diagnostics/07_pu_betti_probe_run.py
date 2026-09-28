"""beta_1 for the PU embedding cloud -- WITH the power controls that make the read meaningful.

**Why controls are not optional here.** PU's intrinsic dimension estimates cluster at 18-25.
Persistent homology at `n_ph` a few hundred in ~20 dimensions is desperately undersampled
(400^(1/20) ~ 1.35 samples per dimension), and Fasy et al.'s bootstrap band WIDENS as the cloud
gets sparser -- so `beta_1 = 0` is close to guaranteed whatever the truth is. A bare `beta_1 = 0`
on PU would therefore be uninterpretable: "no loop" and "no power to see a loop" produce the
identical number.

Three clouds through the IDENTICAL instrument at the same `n_ph`, `B`, `alpha`:

    positive control   S^1 x R^19 -- intrinsic dim 20, beta_1 = 1 BY CONSTRUCTION
    negative control   uniform ball in R^20 -- beta_1 = 0 by construction
    PU                 the real legacysurvey cloud

If the positive control returns `beta_1 >= 1` and the negative returns `0`, the instrument has
power at this `n` and `d` and PU's read means something. If the positive control ALSO returns
`0`, the instrument is blind here and PU's `0` is uninformative -- which is itself the finding,
and the one this run exists to be able to state.

`maxdim=1` throughout: beta_1 is what was asked for, and H2 is the expensive degree
(02.7-RESEARCH measured S^2 at n=500 timing out past 90s for maxdim=2, against ~0.19s for H1).

Both controls are embedded in D=768 and L2-normalized, matching PU's own preprocessing, so the
comparison is not confounded by ambient dimension or scale.
"""
import glob, json, sys, time
from pathlib import Path

NB = Path("/home/akagi/Documents/Projects/EffDim/notebooks")
sys.path.insert(0, str(NB))

import numpy as np
from pu_manifold import confidence_band, persistence_probe

N_PH_GRID = (400, 800)
B, ALPHA, N_DRAWS, SEED, D_AMB, D_INT = 10, 0.05, 3, 20260825, 768, 20
OUT = Path(__file__).with_name("pu_betti_probe.jsonl")


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def rotate_pad(X_local, D, rng):
    """Isometric embedding into R^D: random orthonormal frame, zero-pad. Preserves all
    intrinsic geometry and topology exactly."""
    d = X_local.shape[1]
    A = rng.standard_normal((D, d))
    Q, _ = np.linalg.qr(A)
    return X_local @ Q.T


def make_positive(n, rng):
    """S^1 x R^19: intrinsic dim 20, beta_1 = 1 by construction."""
    theta = rng.uniform(0, 2 * np.pi, n)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    rest = rng.uniform(-0.5, 0.5, size=(n, D_INT - 1))
    return l2(rotate_pad(np.concatenate([circle, rest], axis=1), D_AMB, rng))


def make_negative(n, rng):
    """Uniform ball in R^20: beta = (1, 0, 0) by construction."""
    v = rng.standard_normal((n, D_INT))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rng.random(n) ** (1.0 / D_INT)
    return l2(rotate_pad(v * r[:, None], D_AMB, rng))


def load_pu():
    cands = sorted(glob.glob(str(NB / ".cache" / "subsample_*.npz")))
    best, best_n = None, -1
    for c in cands:
        with np.load(c) as z:
            if "legacysurvey" in z.files and z["legacysurvey"].shape[0] > best_n:
                best, best_n = c, z["legacysurvey"].shape[0]
    with np.load(best) as z:
        return np.asarray(z["legacysurvey"], dtype=np.float64), Path(best).name


def betti01(X):
    """(beta_0, beta_1) under a per-degree Fasy bootstrap band. Euclidean metric."""
    D = persistence_probe.cloud_distance_matrix(X, prescale=False)[0]
    bands = confidence_band.bands_for_diagram(D, maxdim=1, B=B, alpha=ALPHA, seed=SEED)
    dgms = persistence_probe.persistence_diagram(D, maxdim=1)
    h0 = persistence_probe.finite_pairs(dgms[0])
    h1 = persistence_probe.finite_pairs(dgms[1])
    b0 = int(confidence_band.significant_bars(h0, bands[0]["band"]).sum()) + 1
    b1 = int(confidence_band.significant_bars(h1, bands[1]["band"]).sum())
    lives = (h1[:, 1] - h1[:, 0]) if h1.shape[0] else np.zeros(0)
    return {"beta_0": b0, "beta_1": b1, "h1_band": float(bands[1]["band"]),
            "h1_max_life": float(lives.max()) if lives.size else 0.0,
            "h1_n_bars": int(h1.shape[0])}


X_pu, pu_file = load_pu()
print(f"PU: {X_pu.shape} from {pu_file}", flush=True)
print(f"n_ph grid={N_PH_GRID} B={B} alpha={ALPHA} draws={N_DRAWS} maxdim=1 "
      f"D={D_AMB} d_intrinsic={D_INT}", flush=True)

results = {}
for N_PH in N_PH_GRID:
  print(f"\n===== n_ph = {N_PH} =====", flush=True)
  for name in ("positive_S1xR19", "negative_ball20", "PU_legacysurvey"):
    key = f"{name}@n{N_PH}"
    rows = []
    for draw in range(N_DRAWS):
        rng = np.random.default_rng(SEED + draw)
        if name == "positive_S1xR19":
            X = make_positive(N_PH, rng)
        elif name == "negative_ball20":
            X = make_negative(N_PH, rng)
        else:
            X = X_pu[rng.choice(X_pu.shape[0], N_PH, replace=False)]
        t = time.time()
        r = betti01(X)
        r.update({"draw": draw, "s": round(time.time() - t, 1)})
        rows.append(r)
        print(f"  {name:<18} draw {draw}: beta_0={r['beta_0']:<4} beta_1={r['beta_1']:<3} "
              f"(H1 bars={r['h1_n_bars']}, max life={r['h1_max_life']:.4f} vs "
              f"band={r['h1_band']:.4f})  {r['s']}s", flush=True)
    results[key] = rows
    OUT.write_text(json.dumps({"config": {"n_ph_grid": list(N_PH_GRID), "B": B,
                                          "alpha": ALPHA, "n_draws": N_DRAWS, "seed": SEED,
                                          "maxdim": 1, "D_ambient": D_AMB,
                                          "d_intrinsic": D_INT, "pu_file": pu_file},
                               "results": results}, indent=2) + "\n")

print()
for N_PH in N_PH_GRID:
    pos = [r["beta_1"] for r in results[f"positive_S1xR19@n{N_PH}"]]
    neg = [r["beta_1"] for r in results[f"negative_ball20@n{N_PH}"]]
    pu = [r["beta_1"] for r in results[f"PU_legacysurvey@n{N_PH}"]]
    print(f"  n_ph={N_PH}:  positive {pos} (truth 1)   negative {neg} (truth 0)   PU {pu}")
pos = [r["beta_1"] for r in results[f"positive_S1xR19@n{N_PH_GRID[-1]}"]]
neg = [r["beta_1"] for r in results[f"negative_ball20@n{N_PH_GRID[-1]}"]]
pu = [r["beta_1"] for r in results[f"PU_legacysurvey@n{N_PH_GRID[-1]}"]]
if all(p >= 1 for p in pos) and all(n == 0 for n in neg):
    print("\n  INSTRUMENT HAS POWER at this n and d. PU's read is meaningful.")
elif all(p == 0 for p in pos):
    print("\n  INSTRUMENT IS BLIND at this n and d -- the positive control's known loop was")
    print("  NOT recovered. PU's beta_1 = 0 is therefore UNINFORMATIVE, not evidence of")
    print("  trivial topology. Raising n_ph is the only lever that can change this.")
else:
    print("\n  MIXED. Neither reading is licensed; report the controls as measured.")
print("DONE", flush=True)
