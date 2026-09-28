"""Does the plain-AE decoder's curvature win survive a different fixture and PU's real ambient D?

The single cubic/D=28 cell measured rho=+0.727, cosine=+0.907, magnitude ratio=0.939 -- beating
the training-free centroid estimator on all three axes at 99.8% reconstruction. Two reasons not
to trust that yet:

  1. spike 003 measured the BEST INSTRUMENT to be fixture-dependent -- `cubic` favours centroid's
     rank while `ridge` favours its direction -- so one fixture cannot select an instrument.
  2. PU's ambient dimension is 768, not 28. This project has already been burned once by
     acceptance criteria that passed at toy scale and failed at production dimensionality.

So: {cubic, ridge} x {D=28, D=768}, everything else fixed. The point-cloud arm is recomputed in
every cell rather than quoted, so both instruments are always measured on the identical cloud.

NOT a reproduction of any sealed cell.

Plan 08-07 Task 3 added `--d` and `--out`, additively and nothing else. `INSTRUMENT_FIDELITY_RANGE`
was measured on this script's `d=20` cells alone, and both Phase 7 (after density control) and
Phase 8 lose their signal at `d=32` -- so a dying instrument and a vanishing effect are not
distinguishable there without the same fixtures at the same `d`. `--d` defaults to 20, so an
invocation with no flag reproduces the sealed behaviour exactly. `K = 231` is NOT rescaled with
`d`: it is the sealed point-cloud neighbour count, and the decoder arm -- the one this sweep is
read for -- does not use it.
"""
import argparse, json, sys, time
from pathlib import Path

NB = Path("/home/akagi/Documents/Projects/EffDim/notebooks")
sys.path.insert(0, str(NB))

import numpy as np
import torch
from scipy.stats import spearmanr

from pu_manifold import cae, curvature_probe, decoder_curvature
from pu_manifold import varying_ii_controls as vic

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--d", type=int, default=20,
                 help="latent/intrinsic dimension (default 20, the sealed Phase 7 value)")
_ap.add_argument("--out", default=None,
                 help="output path (default: plain_decoder_sweep.jsonl beside this script)")
_args = _ap.parse_args()

DIM, N, SEED, K = _args.d, 5000, 20260816, 231
EPOCHS, TARGET_REL = 400, 0.002
OUT = Path(_args.out) if _args.out else Path(__file__).with_name("plain_decoder_sweep.jsonl")


def axes(H_est, H_true):
    he = np.linalg.norm(H_est, axis=1); ht = np.linalg.norm(H_true, axis=1)
    num = (H_est * H_true).sum(axis=1)
    den = np.maximum(he * ht, 1e-30)
    return {"rho": float(spearmanr(he, ht).statistic),
            "median_cosine": float(np.median(num / den)),
            "median_ratio": float(np.median(he / np.maximum(ht, 1e-12)))}


rows = []
for fixture in ("cubic", "ridge"):
    for D in (28, 768):
        fx = vic.FAMILIES[fixture](N, DIM, D, SEED)
        X = np.asarray(fx["X"], dtype=np.float64)
        H_true = np.asarray(fx["H_vec"], dtype=np.float64)
        x32 = torch.tensor(X, dtype=torch.float32); x64 = torch.tensor(X, dtype=torch.float64)
        sig = float((np.linalg.norm(X, axis=1) ** 2).mean())

        t0 = time.time()
        H_cloud = curvature_probe.centroid_mean_curvature(X, k=K, d=DIM)
        t_cloud = time.time() - t0
        a_cloud = axes(H_cloud, H_true)

        torch.manual_seed(0)
        model = cae.PlainAutoEncoder(in_dim=D, latent_dim=DIM, hidden=(250, 250, 250),
                                     activation="silu")
        cfg = {"seed": SEED, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128,
               "max_epochs": EPOCHS, "early_stop_patience": EPOCHS + 1,
               "early_stop_min_delta": 1e-9, "lip_weight": 0.0, "fps_pretrain_epochs": 0,
               "wallclock_ceiling_s": float("inf")}
        t1 = time.time()
        model.train().float(); cae.train_plain_ae(model, x32, cfg)
        t_train = time.time() - t1
        model.eval().double()
        with torch.no_grad():
            rec = cae.reconstruction_stats(x64, model(x64)["y"])
            z = model.encode(x64)
        rel = rec["mse_total"] / sig
        t2 = time.time()
        field = decoder_curvature.plain_decoder_curvature(model, z)
        t_curv = time.time() - t2
        H_dec = field["H_vec"].detach().cpu().numpy()
        a_dec = axes(H_dec, H_true)
        cond = float(np.median(field["metric_condition_number"].detach().cpu().numpy()))

        row = {"fixture": fixture, "d": DIM, "D": D, "n": N, "k": K,
               "ii_cv": fx["ii_variation"]["hess_fro_cv"],
               "var_explained": 1 - rel, "cond_g_median": cond,
               "point_cloud": a_cloud, "plain_decoder": a_dec,
               "t_cloud_s": round(t_cloud, 1), "t_train_s": round(t_train, 1),
               "t_curv_s": round(t_curv, 1)}
        rows.append(row)
        OUT.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"{fixture:<6} D={D:<4} recon={1-rel:7.3%} cond(g)={cond:9.3e} | "
              f"cloud rho={a_cloud['rho']:+.4f} cos={a_cloud['median_cosine']:+.4f} "
              f"ratio={a_cloud['median_ratio']:.4f} | "
              f"decoder rho={a_dec['rho']:+.4f} cos={a_dec['median_cosine']:+.4f} "
              f"ratio={a_dec['median_ratio']:.4f}", flush=True)

print()
print(f"{'fixture':<8}{'D':>6}{'recon':>10}{'cloud rho':>12}{'dec rho':>10}"
      f"{'cloud cos':>12}{'dec cos':>10}{'cloud ratio':>13}{'dec ratio':>11}")
for r in rows:
    print(f"{r['fixture']:<8}{r['D']:>6}{r['var_explained']:>9.2%}"
          f"{r['point_cloud']['rho']:>+12.4f}{r['plain_decoder']['rho']:>+10.4f}"
          f"{r['point_cloud']['median_cosine']:>+12.4f}{r['plain_decoder']['median_cosine']:>+10.4f}"
          f"{r['point_cloud']['median_ratio']:>13.4f}{r['plain_decoder']['median_ratio']:>11.4f}")
wins = sum(1 for r in rows if r['plain_decoder']['rho'] > r['point_cloud']['rho'])
print(f"\n  decoder beats cloud on rank in {wins} of {len(rows)} cells")
print("DONE", flush=True)
