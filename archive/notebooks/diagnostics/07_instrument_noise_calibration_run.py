"""What curvature fidelity does a decoder at PU's REALIZED reconstruction quality achieve?

The PU plain-AE fit converged at 98.207% variance explained with cond(g)=17.6. The clean
fixtures sat at 99.7-99.9% and gave rho up to +0.9745. The gap is 1.5 points of variance, and
whether that matters is exactly the question a threshold cannot answer -- so measure it.

Method: take the fixtures whose curvature is known, add isotropic Gaussian noise at increasing
levels, fit the identical PlainAutoEncoder, and record (reconstruction, rho, cosine, ratio) at
each level. The cell whose reconstruction lands nearest PU's 98.207% is the calibration point:
it says what fidelity to expect from a decoder fit that good, on a manifold whose answer is
known.

This converts "98.2% is below my arbitrary 99% bar" into a measured expectation. It CANNOT tell
us PU's actual fidelity -- PU has no ground truth -- but it bounds what is reasonable to hope
for, which is the decision step 3 needs.

Also records the ||H|| dynamic range per cell, because the PU fit's p95/p05 = 1.495 sits at the
level spike 003 measured for its CONSTANT-curvature negative control (bowl, 1.4x), and it is
worth knowing whether noise alone collapses the range that far on a fixture that genuinely varies.

NOT a reproduction of any sealed cell.
"""
import json, sys, time
from pathlib import Path

NB = Path("/home/akagi/Documents/Projects/EffDim/notebooks")
sys.path.insert(0, str(NB))

import numpy as np
import torch
from scipy.stats import spearmanr

from pu_manifold import cae, decoder_curvature
from pu_manifold import varying_ii_controls as vic

DIM, D, N, SEED, EPOCHS = 20, 768, 5000, 20260816, 400
NOISE = (0.0, 0.02, 0.05, 0.10, 0.20)
PU_RECON = 0.98207
OUT = Path(__file__).with_name("noise_calibration.jsonl")


def axes(He, Ht):
    he = np.linalg.norm(He, axis=1); ht = np.linalg.norm(Ht, axis=1)
    den = np.maximum(he * ht, 1e-30)
    return {"rho": float(spearmanr(he, ht).statistic),
            "median_cosine": float(np.median((He * Ht).sum(axis=1) / den)),
            "median_ratio": float(np.median(he / np.maximum(ht, 1e-12))),
            "est_spread_p95_p05": float(np.percentile(he, 95)
                                        / max(np.percentile(he, 5), 1e-30)),
            "true_spread_p95_p05": float(np.percentile(ht, 95)
                                         / max(np.percentile(ht, 5), 1e-30))}


rows = []
for fixture in ("cubic", "ridge"):
    fx = vic.FAMILIES[fixture](N, DIM, D, SEED)
    X0 = np.asarray(fx["X"], dtype=np.float64)
    H_true = np.asarray(fx["H_vec"], dtype=np.float64)
    scale = float(np.std(X0))
    for nz in NOISE:
        rng = np.random.default_rng(SEED)
        X = X0 + rng.standard_normal(X0.shape) * (nz * scale)
        x32 = torch.tensor(X, dtype=torch.float32); x64 = torch.tensor(X, dtype=torch.float64)
        sig = float((np.linalg.norm(X, axis=1) ** 2).mean())

        torch.manual_seed(0)
        model = cae.PlainAutoEncoder(in_dim=D, latent_dim=DIM, hidden=(250, 250, 250),
                                     activation="silu")
        t0 = time.time()
        model.train().float()
        cae.train_plain_ae(model, x32, {
            "seed": SEED, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128,
            "max_epochs": EPOCHS, "early_stop_patience": EPOCHS + 1,
            "early_stop_min_delta": 1e-9, "lip_weight": 0.0, "fps_pretrain_epochs": 0,
            "wallclock_ceiling_s": float("inf")})
        model.eval().double()
        with torch.no_grad():
            rec = cae.reconstruction_stats(x64, model(x64)["y"])
            z = model.encode(x64)
        rel = rec["mse_total"] / sig
        field = decoder_curvature.plain_decoder_curvature(model, z)
        a = axes(field["H_vec"].detach().cpu().numpy(), H_true)
        cond = float(np.median(field["metric_condition_number"].detach().cpu().numpy()))

        row = {"fixture": fixture, "noise": nz, "var_explained": 1 - rel,
               "cond_g_median": cond, **a, "s": round(time.time() - t0, 1)}
        rows.append(row); OUT.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"{fixture:<6} noise={nz:<5} recon={1-rel:8.3%} cond={cond:8.2f} "
              f"rho={a['rho']:+.4f} cos={a['median_cosine']:+.4f} "
              f"ratio={a['median_ratio']:.4f} est_spread={a['est_spread_p95_p05']:6.3f}",
              flush=True)

print(f"\n{'fixture':<8}{'noise':>7}{'recon':>10}{'rho':>10}{'cosine':>10}"
      f"{'ratio':>9}{'est spread':>12}")
for r in rows:
    print(f"{r['fixture']:<8}{r['noise']:>7}{r['var_explained']:>9.2%}{r['rho']:>+10.4f}"
          f"{r['median_cosine']:>+10.4f}{r['median_ratio']:>9.4f}"
          f"{r['est_spread_p95_p05']:>12.3f}")

print(f"\n  PU's realized fit: recon={PU_RECON:.3%}, cond(g)=17.57, est spread=1.495")
for f in ("cubic", "ridge"):
    sub = [r for r in rows if r["fixture"] == f]
    near = min(sub, key=lambda r: abs(r["var_explained"] - PU_RECON))
    print(f"  nearest {f:<6} cell (recon={near['var_explained']:.3%}, noise={near['noise']}): "
          f"rho={near['rho']:+.4f} cos={near['median_cosine']:+.4f} "
          f"ratio={near['median_ratio']:.4f} est_spread={near['est_spread_p95_p05']:.3f}")
print("DONE", flush=True)
