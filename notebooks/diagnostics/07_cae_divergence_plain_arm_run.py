"""Third arm for the underfit probe: the ACTUAL cae.PlainAutoEncoder (eq. 22), latent 20.

ChartAutoEncoder(n_charts=1) is NOT this. It is 12 hidden layers with two bottlenecks
(40 then 20) plus a Lipschitz penalty and an FPS pretrain stage; the plain AE is 6 hidden
layers with one bottleneck and none of that machinery. Same fixture, split, epochs and
holdout convention as underfit_probe.py so the three arms are directly comparable.
"""
import json, sys, time
from pathlib import Path

NB = Path("/home/akagi/Documents/Projects/EffDim/notebooks")
sys.path.insert(0, str(NB))
sys.path.insert(0, str(NB / "diagnostics"))

import torch
import curvature_field_pu_run as pu
from pu_manifold import cae, synthetic_controls

D, DIM, N, SEED = 28, 20, 3000, 20260816
BLOCK, TOTAL = 100, 600
OUT = Path(__file__).with_name("plain_arm.jsonl")

fx = synthetic_controls.make_saddle_control(n=N, d=DIM, D=D, seed=SEED, domain_radius=2.0)
x64 = torch.tensor(fx["X"], dtype=torch.float64)
x32 = torch.tensor(fx["X"], dtype=torch.float32)
train_idx, holdout_idx = pu._split(fx["X"].shape[0], pu.PU_SPLIT_SEED, pu.PU_HOLDOUT_FRACTION)
x_hold = x64[torch.as_tensor(holdout_idx, dtype=torch.long)]
x_tr32 = x32[torch.as_tensor(train_idx, dtype=torch.long)]
sig = float((torch.linalg.norm(x_hold, dim=1) ** 2).mean())
print(f"plain AE arm: saddle d={DIM} D={D} latent={DIM}  E||x||^2={sig:.4f}", flush=True)

torch.manual_seed(0)
model = cae.PlainAutoEncoder(in_dim=D, latent_dim=DIM, hidden=(250, 250, 250),
                             activation="silu")
done = 0
with OUT.open("w") as fh:
    while done < TOTAL:
        cfg = pu._converge_cfg(SEED, 1, BLOCK)
        cfg.pop("n_charts", None)
        model.train().float()
        t0 = time.time()
        fit = cae.train_plain_ae(model, x_tr32, cfg)
        done += int(fit["epochs_run"])
        model.eval().double()
        with torch.no_grad():
            rec = cae.reconstruction_stats(x_hold, model(x_hold)["y"])
        rel = rec["mse_total"] / sig
        row = {"arm": "plain_ae", "epochs": done, "mse_total": rec["mse_total"],
               "mse_per_dim": rec["mse_per_dim"], "rel_sq_err": rel,
               "var_explained": 1.0 - rel, "block_s": round(time.time() - t0, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print(f"  plain ep={done:4d}  rel_err={rel:.5f}  var_expl={1-rel:8.2%}  "
              f"{row['block_s']}s", flush=True)
print("DONE", flush=True)
