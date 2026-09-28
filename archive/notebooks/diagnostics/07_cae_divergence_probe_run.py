"""Separate two explanations for the d=20 CAE reconstruction failure:

  (a) COMPUTE BUDGET  -- the curve is still descending at 300 epochs, so a GPU would fix it
  (b) CHART FRAGMENTATION -- the atlas splits a globally-graph manifold into several charts,
      each getting a fraction of the points, and no amount of compute repairs that

The saddle is globally a single graph over R^d, so n_charts=1 is the CORRECT atlas for it.
The sealed d=4 fits used 1 of 4 charts (all 10000 points in one) and scored rho=+0.989.
The sealed d=20 saddle fit used 4 of 4, split 792/1587/1467/6154, and scored rho=-0.0151.

Deviations from the sealed cell (this is NOT a reproduction of any sealed number):
    D 768 -> 28, n 10000 -> 3000, epoch_block 25 -> 100.
Both arms see identical data, so the COMPARISON is fair even though neither value is sealed.
Reconstruction follows the sealed convention: holdout split, model.reconstruct (argmax chart).
"""
import json, sys, time
from pathlib import Path

NB = Path("/home/akagi/Documents/Projects/EffDim/notebooks")
sys.path.insert(0, str(NB))
sys.path.insert(0, str(NB / "diagnostics"))

import numpy as np
import torch

import curvature_field_pu_run as pu
import synthetic_control_run as scr
from pu_manifold import cae, synthetic_controls

D, DIM, N, SEED = 28, 20, 3000, 20260816
BLOCK, TOTAL = 100, 600
OUT = Path(__file__).with_name("underfit_probe.jsonl")

fx = synthetic_controls.make_saddle_control(n=N, d=DIM, D=D, seed=SEED, domain_radius=2.0)
X = fx["X"]
x64 = torch.tensor(X, dtype=torch.float64)
x32 = torch.tensor(X, dtype=torch.float32)
train_idx, holdout_idx = pu._split(X.shape[0], pu.PU_SPLIT_SEED, pu.PU_HOLDOUT_FRACTION)
x_hold = x64[torch.as_tensor(holdout_idx, dtype=torch.long)]
x_tr32 = x32[torch.as_tensor(train_idx, dtype=torch.long)]
sig = float((torch.linalg.norm(x_hold, dim=1) ** 2).mean())   # E||x||^2 on the holdout

print(f"saddle d={DIM} D={D} n={N} train={len(train_idx)} hold={len(holdout_idx)} "
      f"E||x||^2={sig:.4f}", flush=True)

with OUT.open("w") as fh:
    for n_charts in (1, 4):
        torch.manual_seed(0)
        model = scr.build_matched_cae(
            n_charts=n_charts, chart_dim=DIM, embed_dim=40, in_dim=D,
            device=torch.device("cpu"),
        )
        done = 0
        while done < TOTAL:
            cfg = pu._converge_cfg(SEED, n_charts, BLOCK)
            if done > 0:
                cfg["fps_pretrain_epochs"] = 0
            model.train().float()
            t0 = time.time()
            fit = cae.train_cae(model, x_tr32, cfg)
            done += int(fit["epochs_run"])
            model.eval().double()
            with torch.no_grad():
                rec = cae.reconstruction_stats(x_hold, model.reconstruct(x_hold))
            rel = rec["mse_total"] / sig
            surv = cae.chart_survival(model, prune_tol=1e-3)
            row = {
                "n_charts": n_charts, "epochs": done,
                "mse_total": rec["mse_total"], "mse_per_dim": rec["mse_per_dim"],
                "rel_sq_err": rel, "var_explained": 1.0 - rel,
                "charts_surviving": surv["n_charts_surviving"],
                "block_s": round(time.time() - t0, 1),
            }
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"  nc={n_charts} ep={done:4d}  rel_err={rel:.5f}  "
                  f"var_expl={1-rel:8.2%}  charts={row['charts_surviving']}  "
                  f"{row['block_s']}s", flush=True)
print("DONE", flush=True)
