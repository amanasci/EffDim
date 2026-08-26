"""PU-only beta_1 at n_ph=800 -- the sampling-density confirmation of the n=400 result.

The n=400 arm already answered the question with validated controls: positive S^1 x R^19
recovered beta_1 = 1 on 3/3 draws, negative ball^20 gave 0 on 3/3, PU gave 0 on 3/3. The n=800
arm then re-confirmed both CONTROLS behave identically (positive 1,1,1; negative 0,0) before it
was stopped for CPU contention, so control validity at n=800 is already established and is NOT
re-run here.

What is left, and the only thing this script does: PU itself at n=800. The concern it addresses
is sparsity -- at d~20, n=400 is ~1.35 samples per dimension, and if beta_1 = 0 were a sampling
artifact it should weaken as n grows. Doubling n is the direct test.

ripser/persim are effectively single-threaded, so this will not fight the torch sweeps.
"""
import glob
import json
import sys
import time
from pathlib import Path

NB = Path("/home/akagi/Documents/Projects/EffDim/notebooks")
sys.path.insert(0, str(NB))

import numpy as np

from pu_manifold import confidence_band, persistence_probe

N_PH, B, ALPHA, N_DRAWS, SEED = 800, 10, 0.05, 3, 20260825
OUT = Path(__file__).with_name("pu_betti_n800.jsonl")

cands = sorted(glob.glob(str(NB / ".cache" / "subsample_*.npz")))
best, best_n = None, -1
for c in cands:
    with np.load(c) as z:
        if "legacysurvey" in z.files and z["legacysurvey"].shape[0] > best_n:
            best, best_n = c, z["legacysurvey"].shape[0]
with np.load(best) as z:
    X_pu = np.asarray(z["legacysurvey"], dtype=np.float64)
print(f"PU {X_pu.shape} from {Path(best).name}; n_ph={N_PH} B={B} draws={N_DRAWS}\n", flush=True)

rows = []
for draw in range(N_DRAWS):
    rng = np.random.default_rng(SEED + draw)
    X = X_pu[rng.choice(X_pu.shape[0], N_PH, replace=False)]
    t = time.time()
    D = persistence_probe.cloud_distance_matrix(X, prescale=False)[0]
    bands = confidence_band.bands_for_diagram(D, maxdim=1, B=B, alpha=ALPHA, seed=SEED)
    dgms = persistence_probe.persistence_diagram(D, maxdim=1)
    h1 = persistence_probe.finite_pairs(dgms[1])
    b1 = int(confidence_band.significant_bars(h1, bands[1]["band"]).sum())
    lives = (h1[:, 1] - h1[:, 0]) if h1.shape[0] else np.zeros(0)
    row = {"draw": draw, "n_ph": N_PH, "beta_1": b1,
           "h1_band": float(bands[1]["band"]),
           "h1_max_life": float(lives.max()) if lives.size else 0.0,
           "h1_n_bars": int(h1.shape[0]), "s": round(time.time() - t, 1)}
    rows.append(row)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"  PU draw {draw}: beta_1={row['beta_1']}  (H1 bars={row['h1_n_bars']}, "
          f"max life={row['h1_max_life']:.4f} vs band={row['h1_band']:.4f})  {row['s']}s",
          flush=True)

b = [r["beta_1"] for r in rows]
print(f"\n  PU beta_1 at n_ph=800: {b}    (n_ph=400 gave [0, 0, 0])")
print("  controls at n=800 already confirmed: positive [1,1,1], negative [0,0]")
print("DONE", flush=True)
