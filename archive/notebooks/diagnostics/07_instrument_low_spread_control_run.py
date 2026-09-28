"""Can the decoder distinguish a genuinely LOW-spread field from a compressed high-spread one?

This is the control every calibration cell so far has been missing. All of them had a TRUE
||H|| spread of 28-34x, and the decoder was measured to compress dynamic range by 26-42% at
PU's reconstruction quality. PU reads 1.495. Two readings are currently indistinguishable:

    (a) PU's true spread really is ~1.5-2.5 and the decoder is reporting it roughly right
    (b) the decoder has a ||H|| variation FLOOR and reports ~1.5 whenever the truth is below
        it, in which case PU's number says nothing about PU

Nothing measured so far separates these, because no fixture with a genuinely low true spread
has ever been put through the decoder. Spike 003 built the knob -- `ridge`'s `phase` and
`frequency` tune the true spread continuously, and its own spread calibration swept 36x down to
1.1x -- but only ever ran it with the CENTROID estimator, never the decoder.

Method: tune `ridge` at d=20, D=768 across a range of TRUE spreads spanning PU's regime, fit the
identical PlainAutoEncoder, and plot estimated spread against true spread. If the estimate
tracks truth down into the low regime, PU's 1.495 is trustworthy. If the estimate flattens out
at some floor, PU's 1.495 is that floor and tells us nothing.

`phase = pi/2` centres `sin(w.x)` near its extremum so |sin| stays near 1 and the spread
collapses; `phase = 0` lets it sweep through zero and the spread explodes. That single knob
moves true spread over more than an order of magnitude at fixed d, D and n.

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
# phase near pi/2 -> |sin| pinned near 1 -> low spread; phase 0 -> sweeps zero -> high spread
CELLS = [("phase=1.5708 f=0.30", 1.5707963, 0.30),
         ("phase=1.5708 f=0.60", 1.5707963, 0.60),
         ("phase=1.4000 f=0.80", 1.4000000, 0.80),
         ("phase=1.0000 f=1.00", 1.0000000, 1.00),
         ("phase=0.0000 f=1.00", 0.0000000, 1.00)]
PU_EST_SPREAD, PU_RECON = 1.495201, 0.98207
OUT = Path(__file__).with_name("low_spread_control.jsonl")

rows = []
for label, phase, freq in CELLS:
    fx = vic.make_ridge_graph_control(N, DIM, D, SEED, amplitude=1.0,
                                      frequency=freq, phase=phase)
    X = np.asarray(fx["X"], dtype=np.float64)
    H_true = np.asarray(fx["H_vec"], dtype=np.float64)
    ht = np.linalg.norm(H_true, axis=1)
    true_spread = float(np.percentile(ht, 95) / max(np.percentile(ht, 5), 1e-30))

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
    He = field["H_vec"].detach().cpu().numpy()
    he = np.linalg.norm(He, axis=1)
    est_spread = float(np.percentile(he, 95) / max(np.percentile(he, 5), 1e-30))
    rho = float(spearmanr(he, ht).statistic)
    cond = float(np.median(field["metric_condition_number"].detach().cpu().numpy()))

    row = {"label": label, "phase": phase, "frequency": freq,
           "true_spread": true_spread, "est_spread": est_spread,
           "est_over_true": est_spread / true_spread, "rho": rho,
           "var_explained": 1 - rel, "cond_g_median": cond,
           "s": round(time.time() - t0, 1)}
    rows.append(row); OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{label:<22} true_spread={true_spread:8.3f}  est_spread={est_spread:8.3f}  "
          f"est/true={est_spread/true_spread:6.3f}  rho={rho:+.4f}  "
          f"recon={1-rel:7.3%}", flush=True)

print(f"\n{'cell':<22}{'true':>10}{'est':>10}{'est/true':>10}{'rho':>10}{'recon':>10}")
for r in rows:
    print(f"{r['label']:<22}{r['true_spread']:>10.3f}{r['est_spread']:>10.3f}"
          f"{r['est_over_true']:>10.3f}{r['rho']:>+10.4f}{r['var_explained']:>9.2%}")

lows = [r for r in rows if r["true_spread"] < 6.0]
print(f"\n  PU reads est_spread = {PU_EST_SPREAD:.3f} at recon {PU_RECON:.3%}")
if lows:
    lo = min(lows, key=lambda r: r["true_spread"])
    print(f"  lowest true-spread cell: true={lo['true_spread']:.3f} -> "
          f"est={lo['est_spread']:.3f} (est/true={lo['est_over_true']:.3f})")
    if lo["est_spread"] <= PU_EST_SPREAD * 1.35:
        print("\n  NO FLOOR DETECTED above PU's reading -- the decoder tracks a genuinely low")
        print("  spread down into PU's regime, so PU's 1.495 is a real measurement.")
    else:
        print(f"\n  FLOOR AT ~{lo['est_spread']:.2f} -- the decoder cannot report a spread")
        print("  below this even when the truth is lower. PU's 1.495 sits at/below that floor")
        print("  and is therefore UNINFORMATIVE about PU's real dynamic range.")
else:
    print("\n  NO LOW-SPREAD CELL ACHIEVED -- the knob did not reach PU's regime; inconclusive.")
print("DONE", flush=True)
