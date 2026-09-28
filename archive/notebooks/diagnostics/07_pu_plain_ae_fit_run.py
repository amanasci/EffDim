"""Step-3 precondition: does a PlainAutoEncoder actually fit PU at d=20, D=768, n=10000?

Everything the fixture sweep established rests on an antecedent that has NEVER been checked on
real data: the sweep's four cells all sat at 99.7-99.9% variance explained with cond(g) between
2.2 and 7.8. The flat-fixture result says a decoder FABRICATES curvature when it does not fit
(||H|| = 0.0073 against a truth of exactly 0, at 1.1e-06 reconstruction). So PU reconstruction
quality and PU cond(g) are the two numbers that decide whether the instrument transfers at all.

This run MEASURES those two numbers and nothing else. It computes no verdict, buckets no
residual, and touches no pre-registered constant. If reconstruction lands near the fixtures'
99.7%+ and cond(g) stays in single or low double digits, step 3 is licensed; if reconstruction
lands at 70% or cond(g) explodes, it is not, and that is the finding.

Fits on `legacysurvey` -- the column Phase 1 fit Isomap on and every curvature phase has used.
Holdout convention matches the sealed runners (pu._split at PU_HOLDOUT_FRACTION).

NOT a reproduction of any sealed cell. Writes only to scratchpad.
"""
import glob, json, sys, time
from pathlib import Path

NB = Path("/home/akagi/Documents/Projects/EffDim/notebooks")
sys.path.insert(0, str(NB)); sys.path.insert(0, str(NB / "diagnostics"))

import numpy as np
import torch

import curvature_field_pu_run as pu
from pu_manifold import cae, decoder_curvature

DIM, SEED = 20, 20260825
BLOCK, MAX_EPOCHS = 100, 600
OUT = Path(__file__).with_name("pu_plain_ae_fit.jsonl")


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
x32 = torch.tensor(X, dtype=torch.float32); x64 = torch.tensor(X, dtype=torch.float64)
tr, ho = pu._split(X.shape[0], pu.PU_SPLIT_SEED, pu.PU_HOLDOUT_FRACTION)
x_tr32 = x32[torch.as_tensor(tr, dtype=torch.long)]
x_ho = x64[torch.as_tensor(ho, dtype=torch.long)]
sig = float((torch.linalg.norm(x_ho, dim=1) ** 2).mean())
print(f"PU legacysurvey {X.shape} from {pu_file}", flush=True)
print(f"train={len(tr)} holdout={len(ho)}  E||x||^2={sig:.6f}  latent d={DIM}\n", flush=True)

torch.manual_seed(0)
model = cae.PlainAutoEncoder(in_dim=D, latent_dim=DIM, hidden=(250, 250, 250),
                             activation="silu")
cfg_base = {"seed": SEED, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128,
            "early_stop_patience": MAX_EPOCHS + 1, "early_stop_min_delta": 1e-9,
            "lip_weight": 0.0, "fps_pretrain_epochs": 0,
            "wallclock_ceiling_s": float("inf")}

rows, done = [], 0
while done < MAX_EPOCHS:
    model.train().float()
    t0 = time.time()
    cae.train_plain_ae(model, x_tr32, dict(cfg_base, max_epochs=BLOCK))
    done += BLOCK
    model.eval().double()
    with torch.no_grad():
        rec = cae.reconstruction_stats(x_ho, model(x_ho)["y"])
    rel = rec["mse_total"] / sig
    rows.append({"epochs": done, "rel_sq_err": rel, "var_explained": 1 - rel,
                 "block_s": round(time.time() - t0, 1)})
    print(f"  ep={done:4d}  rel_err={rel:.6f}  var_expl={1-rel:8.3%}  "
          f"{rows[-1]['block_s']}s", flush=True)

# --- the two numbers that decide step 3 -------------------------------------------------
model.eval().double()
with torch.no_grad():
    z = model.encode(x64)
t0 = time.time()
field = decoder_curvature.plain_decoder_curvature(model, z)
t_curv = time.time() - t0
cond = field["metric_condition_number"].detach().cpu().numpy()
h = np.linalg.norm(field["H_vec"].detach().cpu().numpy(), axis=1)

result = {"kind": "pu_plain_ae_precondition", "reproduces_sealed_cell": False,
          "pu_file": pu_file, "d": DIM, "D": D, "n": int(X.shape[0]),
          "epochs_run": done, "recon_curve": rows,
          "final_var_explained": rows[-1]["var_explained"],
          "cond_g": {"median": float(np.median(cond)), "p95": float(np.percentile(cond, 95)),
                     "max": float(cond.max())},
          "h_norm": {"median": float(np.median(h)), "p05": float(np.percentile(h, 5)),
                     "p95": float(np.percentile(h, 95)),
                     "spread_p95_p05": float(np.percentile(h, 95)
                                             / max(np.percentile(h, 5), 1e-30))},
          "curv_s": round(t_curv, 1), "activation": field["activation"],
          "fixture_reference": {"var_explained": "0.997-0.999", "cond_g_median": "2.2-7.8"}}
OUT.write_text(json.dumps(result, indent=2) + "\n")

print(f"\n  PU reconstruction : {rows[-1]['var_explained']:.3%}   "
      f"(fixtures were 99.7-99.9%)")
print(f"  PU cond(g) median : {np.median(cond):.4e}   (fixtures were 2.2-7.8)")
print(f"  PU ||H|| median   : {np.median(h):.4e}   p95/p05 = "
      f"{np.percentile(h,95)/max(np.percentile(h,5),1e-30):.3f}")
ok_recon = rows[-1]["var_explained"] >= 0.99
ok_cond = np.median(cond) < 1e3
if ok_recon and ok_cond:
    print("\n  PRECONDITIONS MET. The fixture result transfers; step 3 is licensed.")
else:
    print(f"\n  PRECONDITIONS NOT MET (recon_ok={ok_recon}, cond_ok={ok_cond}). The fixture")
    print("  result does NOT transfer as-is; step 3 would be bucketing on a field whose")
    print("  fidelity is unestablished -- the same defect Phases 5 and 6 carried.")
print("DONE", flush=True)
