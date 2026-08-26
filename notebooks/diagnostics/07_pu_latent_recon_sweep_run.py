"""Is PU's 98.2% reconstruction NOISE-limited or DIMENSION-limited? (reconstruction only)

Sets Phase 7 D7-01's latent dimension. The earlier version of this sweep also computed a full
curvature field per cell, which turned out to dominate the cost: measured on the d=20 PU fit, the
curvature step took 1457s against 374s for all 600 training epochs, and it scales as D*d*d -- so
six cells up to d=48 would have cost ~5 hours, essentially all of it on fields not needed to
answer this question.

Reconstruction alone answers it:

    noise-limited  -> reconstruction PLATEAUS once latent d passes the true dimension; the
                      leftover variance is unlearnable and d=20 was an adequate bottleneck
    dim-limited    -> reconstruction KEEPS CLIMBING past d=20, meaning the d=20 fit was
                      TRUNCATING PU, and every number taken from it -- including the ||H||
                      spread of 1.495 that the Phase 7 power argument rests on -- describes a
                      truncated approximation rather than PU

PU's intrinsic-dimension estimates on record cluster at 18-25 (local PCA 25.0, TwoNN 19.5,
Phase 1's eight geometric estimators 18), so the grid brackets that range and extends past it.

The curvature field is computed ONCE afterwards, at whichever d this sweep selects -- not per
cell.

NOT a reproduction of any sealed cell. Writes only to scratchpad.
"""
import glob
import json
import sys
import time
from pathlib import Path

NB = Path("/home/akagi/Documents/Projects/EffDim/notebooks")
sys.path.insert(0, str(NB))
sys.path.insert(0, str(NB / "diagnostics"))

import numpy as np
import torch

import curvature_field_pu_run as pu
from pu_manifold import cae

LATENT_GRID = (10, 15, 20, 25, 32, 48)
EPOCHS, SEED = 300, 20260825
OUT = Path(__file__).with_name("pu_latent_recon_sweep.jsonl")


def load_pu():
    cands = sorted(glob.glob(str(NB / ".cache" / "subsample_*.npz")))
    best, best_n = None, -1
    for c in cands:
        with np.load(c) as z:
            if "legacysurvey" in z.files and z["legacysurvey"].shape[0] > best_n:
                best, best_n = c, z["legacysurvey"].shape[0]
    with np.load(best) as z:
        return np.asarray(z["legacysurvey"], dtype=np.float64), Path(best).name


X, pu_file = load_pu()
D = X.shape[1]
x32 = torch.tensor(X, dtype=torch.float32)
x64 = torch.tensor(X, dtype=torch.float64)
tr, ho = pu._split(X.shape[0], pu.PU_SPLIT_SEED, pu.PU_HOLDOUT_FRACTION)
x_tr32 = x32[torch.as_tensor(tr, dtype=torch.long)]
x_ho = x64[torch.as_tensor(ho, dtype=torch.long)]
sig = float((torch.linalg.norm(x_ho, dim=1) ** 2).mean())
print(f"PU {X.shape} from {pu_file}; train={len(tr)} holdout={len(ho)}", flush=True)
print("intrinsic-dim estimates on record: local PCA 25.0, TwoNN 19.5, Phase 1 est. 18\n",
      flush=True)

rows = []
for d in LATENT_GRID:
    torch.manual_seed(0)
    model = cae.PlainAutoEncoder(in_dim=D, latent_dim=d, hidden=(250, 250, 250),
                                 activation="silu")
    t0 = time.time()
    model.train().float()
    cae.train_plain_ae(model, x_tr32, {
        "seed": SEED, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128,
        "max_epochs": EPOCHS, "early_stop_patience": EPOCHS + 1,
        "early_stop_min_delta": 1e-9, "lip_weight": 0.0, "fps_pretrain_epochs": 0,
        "wallclock_ceiling_s": float("inf")})
    model.eval().double()
    with torch.no_grad():
        rec = cae.reconstruction_stats(x_ho, model(x_ho)["y"])
    rel = rec["mse_total"] / sig
    row = {"latent_d": d, "var_explained": 1 - rel, "rel_sq_err": rel,
           "s": round(time.time() - t0, 1)}
    rows.append(row)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"  latent d={d:<3} recon={1 - rel:8.3%}   {row['s']}s", flush=True)

print(f"\n{'latent d':>10}{'recon':>11}{'gain vs prev':>15}")
for i, r in enumerate(rows):
    gain = "" if i == 0 else f"{r['var_explained'] - rows[i-1]['var_explained']:+.4%}"
    print(f"{r['latent_d']:>10}{r['var_explained']:>10.3%}{gain:>15}")

gains = [(rows[i + 1]["latent_d"], rows[i + 1]["var_explained"] - rows[i]["var_explained"])
         for i in range(len(rows) - 1)]
tail = [g for d, g in gains if d >= 25]
print()
if tail and max(tail) < 0.002:
    best_d = min(r["latent_d"] for r in rows if r["var_explained"] >= 0.98)
    print("  NOISE-LIMITED: reconstruction plateaus past d=20-25, so the 1.8% shortfall is")
    print("  unlearnable variance. The d=20 fit was an adequate bottleneck and the noise")
    print("  calibration matched the right failure mode.")
    print(f"  -> D7-01 latent dimension: {best_d}")
else:
    top = max(rows, key=lambda r: r["var_explained"])
    print("  DIMENSION-LIMITED: reconstruction keeps climbing past d=20, so the d=20 fit was")
    print("  TRUNCATING PU. Every number taken from it -- including the ||H|| spread of 1.495")
    print("  the Phase 7 power argument rests on -- describes a truncated approximation.")
    print(f"  -> D7-01 latent dimension: at least {top['latent_d']} "
          f"(recon {top['var_explained']:.3%})")
print("DONE", flush=True)
