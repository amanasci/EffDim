"""D4-02 — point-cloud estimator vs CAE chart decoder, on a fixture that can tell them apart.

**Why this run decides something.** Phase 3 computed curvature by autodiff through a trained CAE
chart decoder and its sealed `d=20` saddle control scored `rho = -0.0151`. Spike 003 measured
`centroid_mean_curvature` on the point cloud at `rho = +0.6115` on `cubic` at the same `d`. Those
two numbers are **NOT a comparison** and must never be quoted as one: they were measured on
DIFFERENT FIXTURES, and the saddle has a constant second fundamental form so it returns ~0 for
any estimator regardless of merit.

This runner puts both instruments on the SAME fixture, at the same `d`, `D`, `n` and seed, on a
surface with a genuinely varying `II` where a good estimator is known to score ~+0.6. Whichever
wins here is the instrument Phase 4 should use (D4-02).

**The two arms.**

    point-cloud   `curvature_probe.centroid_mean_curvature(X, k, d)` -- no training, no decoder,
                  no pullback metric, so the `cond(g)` pathology cannot arise. One free
                  parameter (`k`).
    decoder       train a CAE on the same cloud, then `chart_curvature.chart_curvature_field`.
                  This is Phase 3's own path, and it is scored BOTH at the input point and at
                  the point the decoder actually placed (`reconstruction_truth`), because those
                  differ by the reconstruction error.

**Scored on all three fidelity axes, never collapsed**, because spike 003 measured them coming
apart at `d=20`: magnitude dies, direction survives, rank saturates. A single combined score
would hide exactly the signal D4-01 wants to use.

**This is not a reproduction of any sealed cell.** The sealed `d=20` controls ran at `D=768`,
`n=10000`, 300 epochs in 25-epoch blocks and cost ~2.6h each. This runs at reduced `D`, `n` and
epochs so the comparison is affordable; both arms see identical data, so the COMPARISON is fair
even though neither arm reproduces a sealed number. Deviations are printed and recorded.

    python notebooks/diagnostics/estimator_headtohead_run.py --smoke
    python notebooks/diagnostics/estimator_headtohead_run.py --fixture cubic --epochs 200
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import numpy as np
import torch
from scipy.stats import spearmanr

from pu_manifold import cae, chart_curvature, curvature_probe
from pu_manifold import varying_ii_controls as vic

DEFAULT_RECORD = NOTEBOOK_ROOT / ".cache" / "03.2_estimator_headtohead.jsonl"

CHART_DIM = 20
EMBED_DIM = 40
N_CHARTS = 4
FIXTURE_SEED = 20260816
NO_EARLY_STOP = 10**9


def _axes(H_est: np.ndarray, H_true: np.ndarray) -> Dict[str, float]:
    """The three axes, kept as three. Collapsing them would hide the direction signal."""
    he = np.linalg.norm(H_est, axis=-1)
    ht = np.linalg.norm(H_true, axis=-1)
    num = (H_est * H_true).sum(1)
    den = np.maximum(np.linalg.norm(H_est, axis=1) * np.linalg.norm(H_true, axis=1), 1e-12)
    return {
        "rho": float(spearmanr(he, ht).statistic),
        "median_cosine": float(np.median(num / den)),
        "median_ratio": float(np.median(he / np.maximum(ht, 1e-12))),
        "median_relative_error": float(curvature_probe.median_relative_error(he, ht)),
    }


def run(fixture: str, n: int, d: int, D: int, k: int, epochs: int,
        seed: int, model_seed: int) -> Dict[str, Any]:
    fx = vic.FAMILIES[fixture](n, d, D, seed)
    X = np.asarray(fx["X"], dtype=np.float64)
    H_true = np.asarray(fx["H_vec"], dtype=np.float64)

    # --- arm 1: point cloud, no training -------------------------------------------------
    t0 = time.monotonic()
    H_cloud = curvature_probe.centroid_mean_curvature(X, k=k, d=d)
    t_cloud = time.monotonic() - t0
    axes_cloud = _axes(H_cloud, H_true)

    # --- arm 2: CAE chart decoder --------------------------------------------------------
    x32 = torch.tensor(X, dtype=torch.float32)
    x64 = torch.tensor(X, dtype=torch.float64)
    torch.manual_seed(model_seed)
    model = cae.ChartAutoEncoder(
        in_dim=D, embed_dim=EMBED_DIM, chart_dim=d,
        n_charts=N_CHARTS, hidden=[64, 64, 64], activation="silu",
    )
    t1 = time.monotonic()
    fit = cae.train_cae(model, x32, {
        "seed": model_seed, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128,
        "max_epochs": epochs, "early_stop_patience": NO_EARLY_STOP,
        "early_stop_min_delta": 1e-4, "n_charts": N_CHARTS,
        "fps_pretrain_epochs": 20, "lip_weight": 1e-3, "lip_every_n_steps": 1,
    })
    t_train = time.monotonic() - t1
    model.eval()
    with torch.no_grad():
        recon = model.reconstruct(x32)
    recon_stats = cae.reconstruction_stats(x64, recon.double())

    t2 = time.monotonic()
    model.double()
    field = chart_curvature.chart_curvature_field(model, x64, mode="reverse")
    t_curv = time.monotonic() - t2
    H_dec = field["H_vec"].detach().cpu().numpy()
    axes_dec = _axes(H_dec, H_true)

    return {
        "kind": "estimator_headtohead",
        "reproduces_sealed_cell": False,
        "fixture": fixture,
        "ii_variation": fx["ii_variation"],
        "n": n, "d": d, "D": D, "k": k, "epochs": epochs,
        "fixture_seed": seed, "model_seed": model_seed,
        "point_cloud": axes_cloud,
        "decoder": axes_dec,
        "decoder_reconstruction": {kk: v for kk, v in recon_stats.items()
                                   if not isinstance(v, list)},
        "decoder_epochs_run": int(fit["epochs_run"]),
        "decoder_cond_g_median": float(field["metric_condition_number"].median()),
        "decoder_n_charts_used": int(field["n_charts_used"]),
        "t_cloud_s": t_cloud, "t_train_s": t_train, "t_curv_s": t_curv,
        "curvature_convention": chart_curvature.CURVATURE_CONVENTION,
    }


def _report(r: Dict[str, Any]) -> None:
    pc, dc = r["point_cloud"], r["decoder"]
    print()
    print("=" * 78)
    print(f"D4-02 HEAD-TO-HEAD -- fixture={r['fixture']}  d={r['d']}  D={r['D']}  "
          f"n={r['n']}  k={r['k']}")
    print("=" * 78)
    print(f"  fixture II variation (CV)   = {r['ii_variation']['hess_fro_cv']:.4f}"
          f"   (0 = unrankable by construction)")
    print(f"  decoder reconstruction      = {r['decoder_reconstruction']['mse_per_dim']:.4e}"
          f"  mse_per_dim")
    print(f"  decoder cond(g) median      = {r['decoder_cond_g_median']:.3e}")
    print(f"  charts used                 = {r['decoder_n_charts_used']} / {N_CHARTS}")
    print()
    print(f"  {'axis':>22} | {'point cloud':>13} | {'CAE decoder':>13}")
    print(f"  {'-'*22} | {'-'*13} | {'-'*13}")
    print(f"  {'rank (rho)':>22} | {pc['rho']:>+13.4f} | {dc['rho']:>+13.4f}")
    print(f"  {'direction (cosine)':>22} | {pc['median_cosine']:>+13.4f} | "
          f"{dc['median_cosine']:>+13.4f}")
    print(f"  {'magnitude (ratio)':>22} | {pc['median_ratio']:>13.4f} | "
          f"{dc['median_ratio']:>13.4f}")
    print(f"  {'magnitude (MRE)':>22} | {pc['median_relative_error']:>13.3f} | "
          f"{dc['median_relative_error']:>13.3f}")
    print()
    print(f"  cost: point cloud {r['t_cloud_s']:.0f}s   "
          f"decoder {r['t_train_s'] + r['t_curv_s']:.0f}s "
          f"({r['t_train_s']:.0f}s train + {r['t_curv_s']:.0f}s curvature)")
    print()

    d_rank = pc["rho"] - dc["rho"]
    d_dir = pc["median_cosine"] - dc["median_cosine"]
    if r["ii_variation"]["hess_fro_cv"] < 1e-9:
        print("  UNINFORMATIVE. This fixture has a constant second fundamental form, so it")
        print("  cannot rank for EITHER instrument and this comparison decides nothing.")
        print("  Re-run on `cubic` or `ridge`.")
    elif d_rank > 0.15 and d_dir > 0.05:
        print("  POINT CLOUD WINS on both rank and direction. D4-02 resolves to the")
        print("  point-cloud estimator, and Phase 03.1's metric regularization becomes")
        print("  OPTIONAL rather than blocking for Phase 4 -- the point-cloud route forms no")
        print("  pullback metric and cannot suffer the cond(g) pathology at all.")
    elif d_rank < -0.15 and d_dir < -0.05:
        print("  DECODER WINS on both rank and direction. D4-02 resolves to the decoder, and")
        print("  Phase 03.1's metric work stays on the critical path.")
    else:
        print("  SPLIT OR CLOSE. The two instruments do not separate cleanly on this cell.")
        print("  Report both; do NOT pick one on a margin this size. Vary k and epochs before")
        print("  concluding -- each arm has a free parameter and neither has been tuned.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fixture", type=str, default="cubic", choices=sorted(vic.FAMILIES))
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--d", type=int, default=CHART_DIM)
    p.add_argument("--ambient", type=int, default=28)
    p.add_argument("--k", type=int, default=231)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=FIXTURE_SEED)
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    a = build_arg_parser().parse_args()
    print("DEVIATIONS FROM THE SEALED d=20 CONTROLS (this is not a reproduction):")
    print(f"  ambient D  768   -> {a.ambient}   (changes the CAE input layer)")
    print(f"  n          10000 -> {a.n}")
    print(f"  epochs     300   -> {a.epochs}  (and not in 25-epoch blocks)")
    print("  Both arms see IDENTICAL data, so the COMPARISON is fair regardless.")

    if a.smoke:
        print("\nSMOKE: n=600, d=6, 10 epochs -- proves the path runs, measures nothing.")
        _report(run("cubic", 600, 6, 14, 30, 10, a.seed, a.model_seed))
        return

    r = run(a.fixture, a.n, a.d, a.ambient, a.k, a.epochs, a.seed, a.model_seed)
    _report(r)
    record_path = Path(a.record_path) if a.record_path else DEFAULT_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as fh:
        fh.write(json.dumps(r, default=float) + "\n")
    print(f"\nwrote {record_path}")


if __name__ == "__main__":
    main()
