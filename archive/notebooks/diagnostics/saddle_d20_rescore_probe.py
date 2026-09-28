"""Does re-scoring at the reconstruction point move the ``d=20`` saddle's rank correlation?

**The question.** ``notebooks/.cache/03_synthetic_controls.jsonl`` records a sealed CAE fit on
the ``d=20`` saddle control scoring ``rho = -0.015107`` -- the number spike 002 quoted as
"sealed decoder rho (d=20 saddle)". That fit reconstructs at ``mse_per_dim = 1.6e-02``,
against ``5.1e-07`` for the same fixture at ``d=4`` (which scored ``rho = +0.989``). Four
orders of magnitude more reconstruction error sit underneath the collapse, and every ``rho``
in this milestone pairs ``H_est`` measured at ``F(z_chart(x_i))`` with ``H_true`` measured at
``x_i``. This probe asks how much of the ``d=20`` collapse is that mismatch.

**THIS DOES NOT REPRODUCE THE SEALED CELL, and no number it prints may be quoted as one.**
Retraining the sealed configuration costs about 2.6 hours (6254s train + 2994s curvature) and
the record stores only histogram summaries of ``H_est``, so re-scoring the sealed fit itself
is impossible without paying that again. The deviations here are deliberate and are printed
at the top of every run:

    ambient D   768 -> 28     the CAE's input layer changes, so this is a DIFFERENT model.
                              The fixture's intrinsic geometry is unchanged (zero-padding is
                              totally geodesic, per rotate_and_pad), but the network is not.
    n           10000 -> 3000
    epochs      300 -> configurable, default 150, and NOT in 25-epoch blocks

    unchanged:  chart_dim=20, embed_dim=40, n_charts=4, fixture seed 20260816, activation
                and widths from the PU runner via build_matched_cae.

What transfers from a probe like this is the SIGN and rough SIZE of the re-scoring delta at
``d=20`` under bad reconstruction, not the sealed cell's value.

**Reading the output.** ``delta`` is ``rho_at_recon - rho_at_input``. ``truth_rank_agreement``
is the guard: it is the rank correlation between the two truth vectors, and when it is low the
reconstruction has scrambled position so badly that ``rho_at_recon`` describes a surface that
is not the fixture. A large ``delta`` with a low ``truth_rank_agreement`` is not evidence that
the estimator was fine -- it is evidence that the decoder is not on the manifold.

    python notebooks/diagnostics/saddle_d20_rescore_probe.py --smoke
    python notebooks/diagnostics/saddle_d20_rescore_probe.py --epochs 150
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np
import torch

from pu_manifold import cae, chart_curvature, synthetic_controls
from pu_manifold import cross_split_curvature as csc
from pu_manifold import reconstruction_truth as rt

CACHE_DIR = NOTEBOOK_ROOT / ".cache"
DEFAULT_RECORD = CACHE_DIR / "03.2_saddle_d20_rescore_probe.jsonl"

# Sealed values, kept.
CHART_DIM = 20
EMBED_DIM = 40
N_CHARTS = 4
FIXTURE_SEED = 20260816
SEALED_RHO = -0.015106571347065712
SEALED_MSE_PER_DIM = 0.01624897910772578

# Deviations from the sealed cell.
AMBIENT_D = 28
N_POINTS = 3000

NO_EARLY_STOP = 10**9


def run_probe(epochs: int, n_points: int, ambient: int, seed: int) -> Dict[str, Any]:
    fx = synthetic_controls.make_saddle_control(
        n=n_points, d=CHART_DIM, D=ambient, seed=FIXTURE_SEED
    )
    X = np.asarray(fx["X"], dtype=np.float64)
    H_true_norm = np.asarray(fx["H_norm"], dtype=np.float64)

    x32 = torch.tensor(X, dtype=torch.float32)
    x64 = torch.tensor(X, dtype=torch.float64)

    torch.manual_seed(seed)
    model = cae.ChartAutoEncoder(
        in_dim=ambient, embed_dim=EMBED_DIM, chart_dim=CHART_DIM,
        n_charts=N_CHARTS, hidden=[64, 64, 64], activation="silu",
    )
    t0 = time.monotonic()
    fit = cae.train_cae(model, x32, {
        "seed": seed, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128,
        "max_epochs": epochs, "early_stop_patience": NO_EARLY_STOP,
        "early_stop_min_delta": 1e-4, "n_charts": N_CHARTS,
        "fps_pretrain_epochs": 20, "lip_weight": 1e-3, "lip_every_n_steps": 1,
    })
    train_s = time.monotonic() - t0
    model.eval()

    with torch.no_grad():
        recon32 = model.reconstruct(x32)
    recon = recon32.double().cpu().numpy()
    recon_stats = cae.reconstruction_stats(x64, recon32.double())

    t1 = time.monotonic()
    model.double()
    field = chart_curvature.chart_curvature_field(model, x64, mode="reverse")
    curv_s = time.monotonic() - t1
    h_est = field["H_norm"].detach().cpu().numpy()

    truth_recon = rt.saddle_truth_at(recon, fx, d=CHART_DIM, D=ambient, seed=FIXTURE_SEED)
    drift = rt.reconstruction_drift(X, recon)
    scores = rt.rescore(h_est, H_true_norm, truth_recon, drift=drift["drift"])

    return {
        "kind": "saddle_d20_rescore_probe",
        "reproduces_sealed_cell": False,
        "deviations": {
            "ambient_D": [768, ambient],
            "n_points": [10000, n_points],
            "max_epochs": [300, epochs],
            "epoch_block": [25, None],
        },
        "sealed_reference": {"rho": SEALED_RHO, "mse_per_dim": SEALED_MSE_PER_DIM},
        "chart_dim": CHART_DIM,
        "embed_dim": EMBED_DIM,
        "n_charts": N_CHARTS,
        "n_charts_used": int(field["n_charts_used"]),
        "fixture_seed": FIXTURE_SEED,
        "model_seed": seed,
        "epochs_run": int(fit["epochs_run"]),
        "reconstruction": {k: v for k, v in recon_stats.items() if not isinstance(v, list)},
        "scores": scores,
        "drift": {k: v for k, v in drift.items() if not isinstance(v, np.ndarray)},
        "cond_g_median": float(field["metric_condition_number"].median()),
        "train_s": train_s,
        "curv_s": curv_s,
        "curvature_convention": chart_curvature.CURVATURE_CONVENTION,
        "torch_version": torch.__version__,
    }


def _report(r: Dict[str, Any]) -> None:
    s, d = r["scores"], r["drift"]
    print()
    print("=" * 78)
    print("RESULT -- NOT a reproduction of the sealed cell")
    print("=" * 78)
    print(f"  reconstruction mse_per_dim   = {r['reconstruction']['mse_per_dim']:.4e}"
          f"   (sealed cell: {SEALED_MSE_PER_DIM:.4e})")
    print(f"  median relative drift        = {d['median_drift_relative']:.4f}")
    print(f"  charts used                  = {r['n_charts_used']} / {r['n_charts']}")
    print(f"  cond(g) median               = {r['cond_g_median']:.3e}")
    print()
    print(f"  rho at the INPUT point       = {s['rho_at_input']:+.4f}"
          f"   (sealed cell: {SEALED_RHO:+.4f})")
    print(f"  rho at the RECONSTRUCTION    = {s['rho_at_recon']:+.4f}")
    print(f"  delta                        = {s['delta']:+.4f}")
    print(f"  rho controlled for drift     = {s['rho_input_given_drift']:+.4f}")
    print()
    print(f"  truth rank agreement (GUARD) = {s['truth_rank_agreement']:.4f}")
    if s["truth_rank_agreement"] < 0.5:
        print("    LOW. The reconstruction has moved points far enough that the two truth")
        print("    vectors barely agree in rank. rho_at_recon is then describing a surface")
        print("    that is not the fixture, and the delta must NOT be read as 'the estimator")
        print("    was fine all along'. It says the decoder is off the manifold.")
    else:
        print("    Adequate -- the two truth vectors still order points similarly, so the")
        print("    delta is about where curvature was measured rather than about the decoder")
        print("    having left the manifold entirely.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--n-points", type=int, default=N_POINTS)
    p.add_argument("--ambient", type=int, default=AMBIENT_D)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="n=400, 10 epochs. Proves the path runs; measures nothing.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    print("=" * 78)
    print("d=20 saddle -- re-score at the reconstruction point")
    print("=" * 78)
    print("DEVIATIONS FROM THE SEALED CELL (this is not a reproduction):")
    print(f"  ambient D  768   -> {args.ambient}    (changes the CAE input layer)")
    print(f"  n          10000 -> {args.n_points}")
    print(f"  epochs     300   -> {args.epochs}   (and not in 25-epoch blocks)")
    print("  unchanged: chart_dim=20, embed_dim=40, n_charts=4, fixture seed 20260816")
    print()

    if args.smoke:
        print("SMOKE -- numbers meaningless, nothing written.\n")
        r = run_probe(epochs=10, n_points=400, ambient=args.ambient, seed=args.seed)
        _report(r)
        return

    r = run_probe(args.epochs, args.n_points, args.ambient, args.seed)
    _report(r)

    record_path = Path(args.record_path) if args.record_path else DEFAULT_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as fh:
        fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {record_path}")


if __name__ == "__main__":
    main()
