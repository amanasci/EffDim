"""Multi-seed sweep of the split-half cross curvature statistic on the Swiss roll.

**What this answers.** ``notebooks/03.2_swiss_roll_cross_split_curvature_check.ipynb`` runs
the same machinery at ONE seed, inside CLAUDE.md's two-minute sanity-check budget. One seed
is not enough to say whether crossing helps: the first seed tried during development gave
``+0.115`` and looked like a clean win, and two further seeds gave ``-0.015`` and ``-0.030``.
This runner exists so the multi-seed number is on record rather than the lucky one.

**The statistic.** Ported from the ``curvature-experiments`` branch (fork point ``7b2401e``)
via :mod:`pu_manifold.cross_split_curvature`: two Chart Auto-Encoders trained on DISJOINT
halves of the same Swiss roll, each producing a mean-curvature field over all points, scored
by ``K_H_cross = <H^(A), H^(B)>`` and the reliability ratio ``R_H``. Every arm's field, and
the cross statistic, is rank-correlated against ``curvature_probe.swiss_roll_analytic_H_scaled``
-- a closed-form answer, which is the whole point.

**The comparison that matters is cross vs the MEAN of the two arms**, never cross vs the
better arm. Without ground truth there is no way to identify the better arm, so a
single-split user draws one at random; scoring against the better one flatters the baseline
in a way no real user could exploit.

**Early stopping is off by construction.** A development run let one arm stop at epoch 36
while the other ran 200, and the cross statistic then inherited the undertrained arm's error
instead of cancelling anything. ``03-08-SUPPLEMENT-03`` records the same effect on PU data.
Both arms always run the same fixed epoch budget here; ``--epochs`` sets it.

Writes one JSON record per (seed, epochs) cell to ``notebooks/.cache/``. Nothing here is a
gated milestone artifact and no verdict is written -- it reports numbers.

    python notebooks/diagnostics/cross_split_curvature_sweep_run.py --smoke
    python notebooks/diagnostics/cross_split_curvature_sweep_run.py --seeds 0 1 2 --epochs 400
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

from pu_manifold import cae, chart_curvature, curvature_probe, decoder_curvature
from pu_manifold import cross_split_curvature as csc
from pu_manifold import reconstruction_truth as rt

CACHE_DIR = NOTEBOOK_ROOT / ".cache"
DEFAULT_RECORD = CACHE_DIR / "03.2_cross_split_curvature_sweep.jsonl"

N_POINTS = 3000
CHART_DIM = 2
EMBED_DIM = 8
N_CHARTS = 4
HIDDEN = [64, 64]
BATCH = 128
NO_EARLY_STOP = 10**9

ARM_B_SEED_OFFSET = 1000
"""Arm B's torch seed is ``seed + ARM_B_SEED_OFFSET``. The arms are made independent by the
DISJOINT DATA SPLIT, not by this offset -- the offset only stops the two arms sharing an
initialisation, which would correlate their errors on top of the shared architecture they
already have. See :data:`pu_manifold.cross_split_curvature.INDEPENDENCE_MODES`."""

RELIABILITY_THRESHOLD = 0.5
"""Declared before any field was looked at. There is no principled value -- the source
branch's own cutoff was set for per-anchor split-half local quadratic fits on unit-normalised
ViT embeddings and does not transfer to a global chart auto-encoder. Recorded so the gate is
falsifiable rather than fitted after the fact."""


def _fit_arm(x_all32: torch.Tensor, idx: np.ndarray, seed: int, epochs: int):
    torch.manual_seed(seed)
    model = cae.ChartAutoEncoder(
        in_dim=3, embed_dim=EMBED_DIM, chart_dim=CHART_DIM,
        n_charts=N_CHARTS, hidden=HIDDEN, activation="silu",
    )
    fit = cae.train_cae(model, x_all32[idx], {
        "seed": seed, "lr": 1e-3, "weight_decay": 1e-4, "batch": BATCH,
        "max_epochs": epochs, "early_stop_patience": NO_EARLY_STOP,
        "early_stop_min_delta": 1e-4, "n_charts": N_CHARTS,
        "fps_pretrain_epochs": 20, "lip_weight": 1e-3, "lip_every_n_steps": 1,
    })
    model.eval()
    with torch.no_grad():
        y = model.reconstruct(x_all32)
        rel = float((torch.linalg.vector_norm(x_all32 - y, dim=-1)
                     / torch.linalg.vector_norm(x_all32, dim=-1)).mean())
    return model, rel, int(fit["epochs_run"]), y.double().cpu().numpy()


def run_cell(seed: int, epochs: int, n_points: int) -> Dict[str, Any]:
    t_start = time.monotonic()
    fx = curvature_probe.make_swiss_roll_fixture(n=n_points, seed=seed)
    X, t, global_std = fx["X"], fx["t"], fx["global_std"]
    H_true_norm = fx["H_norm"]
    H_true_vec = decoder_curvature.swiss_roll_analytic_H_vector(t, global_std)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_points)
    half = n_points // 2
    idx_A, idx_B = perm[:half], perm[half:]

    x_all32 = torch.tensor(X, dtype=torch.float32)
    x_all64 = torch.tensor(X, dtype=torch.float64)

    model_A, rel_A, ep_A, recon_A = _fit_arm(x_all32, idx_A, seed, epochs)
    model_B, rel_B, ep_B, recon_B = _fit_arm(x_all32, idx_B, seed + ARM_B_SEED_OFFSET, epochs)
    train_s = time.monotonic() - t_start

    t_curv = time.monotonic()
    model_A.double()
    model_B.double()
    field_A = chart_curvature.chart_curvature_field(model_A, x_all64, mode="reverse")
    field_B = chart_curvature.chart_curvature_field(model_B, x_all64, mode="reverse")
    curv_s = time.monotonic() - t_curv

    H_A = field_A["H_vec"].detach().cpu().numpy()
    H_B = field_B["H_vec"].detach().cpu().numpy()
    out = csc.cross_curvature_field(H_A, H_B, independence="disjoint_data")

    rho_A = float(spearmanr(out["norm_H_A"], H_true_norm).statistic)
    rho_B = float(spearmanr(out["norm_H_B"], H_true_norm).statistic)
    rho_cross = float(spearmanr(out["K_H_cross"], H_true_norm).statistic)
    rho_mean_arm = 0.5 * (rho_A + rho_B)

    def median_cos(H):
        num = (H * H_true_vec).sum(1)
        den = np.maximum(np.linalg.norm(H, axis=1) * np.linalg.norm(H_true_vec, axis=1), 1e-12)
        return float(np.median(num / den))

    rel_stats = csc.reliability_summary(
        out["R_H"], threshold=RELIABILITY_THRESHOLD, min_fraction=0.5
    )

    # Re-score against the truth AT THE POINT EACH DECODER ACTUALLY PLACED, not at the input
    # point. See reconstruction_truth's module docstring: H_est(i) is the curvature of the
    # learned manifold at F(z_chart(x_i)), which is the reconstruction, while H_true(i) is
    # the curvature of the true roll at x_i. They coincide only if reconstruction is exact.
    truth_recon_A = rt.swiss_roll_truth_at(recon_A, n=n_points, seed=seed)
    truth_recon_B = rt.swiss_roll_truth_at(recon_B, n=n_points, seed=seed)
    drift_A = rt.reconstruction_drift(X, recon_A)
    drift_B = rt.reconstruction_drift(X, recon_B)
    rescore_A = rt.rescore(out["norm_H_A"], H_true_norm, truth_recon_A, drift=drift_A["drift"])
    rescore_B = rt.rescore(out["norm_H_B"], H_true_norm, truth_recon_B, drift=drift_B["drift"])
    # For the cross statistic both arms drifted, so score it against the mean of the two
    # arms' re-evaluated truths -- neither arm's reconstruction alone is privileged.
    rescore_cross = rt.rescore(
        out["K_H_cross"], H_true_norm, 0.5 * (truth_recon_A + truth_recon_B)
    )

    return {
        "kind": "cross_split_swiss_roll",
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_run_A": ep_A,
        "epochs_run_B": ep_B,
        "n_points": n_points,
        "n_charts": N_CHARTS,
        "chart_dim": CHART_DIM,
        "rel_err_A": rel_A,
        "rel_err_B": rel_B,
        "rho_arm_A": rho_A,
        "rho_arm_B": rho_B,
        "rho_mean_arm": rho_mean_arm,
        "rho_cross": rho_cross,
        "gain_over_mean_arm": rho_cross - rho_mean_arm,
        "gain_over_best_arm": rho_cross - max(rho_A, rho_B),
        "median_cos_A": median_cos(H_A),
        "median_cos_B": median_cos(H_B),
        "reliability": rel_stats,
        "rescore_arm_A": rescore_A,
        "rescore_arm_B": rescore_B,
        "rescore_cross": rescore_cross,
        "drift_A": {k: v for k, v in drift_A.items() if not isinstance(v, np.ndarray)},
        "drift_B": {k: v for k, v in drift_B.items() if not isinstance(v, np.ndarray)},
        "cond_g_median_A": float(field_A["metric_condition_number"].median()),
        "cond_g_median_B": float(field_B["metric_condition_number"].median()),
        "train_s": train_s,
        "curv_s": curv_s,
        "curvature_convention": chart_curvature.CURVATURE_CONVENTION,
        "torch_version": torch.__version__,
    }


def _print_row(r: Dict[str, Any]) -> None:
    print(
        f"  seed {r['seed']:>3} ep {r['epochs_requested']:>4} | "
        f"A {r['rho_arm_A']:+.4f}  B {r['rho_arm_B']:+.4f}  "
        f"cross {r['rho_cross']:+.4f} | gain(mean) {r['gain_over_mean_arm']:+.4f} | "
        f"relerr {r['rel_err_A']:.3f}/{r['rel_err_B']:.3f} | "
        f"medR_H {r['reliability']['median_R_H']:.3f} "
        f"fneg {r['reliability']['fraction_negative']:.3f} "
        f"[{r['train_s'] + r['curv_s']:.0f}s]"
    )
    print(
        f"           re-scored at the reconstruction: "
        f"A {r['rescore_arm_A']['rho_at_input']:+.4f}->{r['rescore_arm_A']['rho_at_recon']:+.4f} "
        f"({r['rescore_arm_A']['delta']:+.4f})  "
        f"cross {r['rescore_cross']['rho_at_input']:+.4f}->"
        f"{r['rescore_cross']['rho_at_recon']:+.4f} "
        f"({r['rescore_cross']['delta']:+.4f}) | "
        f"drift {r['drift_A']['median_drift_relative']:.4f}/"
        f"{r['drift_B']['median_drift_relative']:.4f} | "
        f"truth-rank-agree {r['rescore_arm_A']['truth_rank_agreement']:.4f}"
    )


def summarize(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    gains_mean = np.array([r["gain_over_mean_arm"] for r in records])
    gains_best = np.array([r["gain_over_best_arm"] for r in records])
    med_R = np.array([r["reliability"]["median_R_H"] for r in records])
    rho_x = np.array([r["rho_cross"] for r in records])

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  cells                            = {len(records)}")
    print(f"  mean gain over a random arm      = {gains_mean.mean():+.4f}   "
          f"(min {gains_mean.min():+.4f}, max {gains_mean.max():+.4f})")
    print(f"  mean gain over the better arm    = {gains_best.mean():+.4f}")
    print(f"  cells where crossing helped      = {int((gains_mean > 0).sum())} / {len(records)}")
    print(f"  median R_H across cells          = {np.median(med_R):.4f}")
    print(f"  best cross rho                   = {rho_x.max():+.4f}")
    print()

    # The finding this runner exists to make visible, stated as a measurement.
    reliable_but_wrong = [
        r for r in records
        if r["reliability"]["median_R_H"] > 0.90 and r["rho_cross"] < 0.60
    ]
    if reliable_but_wrong:
        print("  RELIABLE BUT NOT CORRECT -- cells where the two arms agree closely")
        print("  (median R_H > 0.90) while the cross statistic misses the known answer")
        print("  (rho < 0.60). Split-half reliability certifies reproducibility, never")
        print("  correctness: a bias both arms share is perfectly reliable.")
        for r in reliable_but_wrong:
            print(f"    seed {r['seed']} ep {r['epochs_requested']}: "
                  f"median R_H {r['reliability']['median_R_H']:.4f}, "
                  f"rho_cross {r['rho_cross']:+.4f}")
    else:
        print("  No cell paired high arm agreement with a missed known answer in this run.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, nargs="+", default=[400])
    p.add_argument("--n-points", type=int, default=N_POINTS)
    p.add_argument("--record-path", type=str, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="One tiny cell (n=400, 15 epochs) to prove the path runs. The numbers "
                        "are meaningless and are not written to the record.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.smoke:
        print("SMOKE: n=400, 15 epochs, seed 0 -- proves the path runs, measures nothing.")
        r = run_cell(seed=0, epochs=15, n_points=400)
        _print_row(r)
        print("\nsmoke OK -- no record written")
        return

    record_path = Path(args.record_path) if args.record_path else DEFAULT_RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Split-half cross curvature statistic on the Swiss roll")
    print("=" * 78)
    print(f"record_path = {record_path}")
    print(f"seeds       = {args.seeds}")
    print(f"epochs      = {args.epochs}   (early stopping OFF -- both arms fixed budget)")
    print(f"n_points    = {args.n_points}   arms trained on disjoint halves")
    print()

    records: List[Dict[str, Any]] = []
    with record_path.open("a") as fh:
        for epochs in args.epochs:
            for seed in args.seeds:
                r = run_cell(seed=seed, epochs=epochs, n_points=args.n_points)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                records.append(r)
                _print_row(r)

    summarize(records)


if __name__ == "__main__":
    main()
