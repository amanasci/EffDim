"""Phase 02.6 decoder-substrate screening: the four-seed, four-axis screening sweep for both
free candidates (a plain autoencoder, and a TopoAE-trained ``cae.PlainAutoEncoder``).

**This script measures and prints; it applies no bar, emits no verdict, seals nothing and
writes no artifact.** It exists because SC-4 requires four seeds with the seed spread
reported, and CLAUDE.md caps a Swiss-roll sanity notebook at ~15 cells with no threshold
tables -- the same split Phase 02.4 made for its own lambda sweep
(``notebooks/diagnostics/topoae_lambda_sweep_run.py``). This runner shares its
configuration verbatim with the phase's two admission notebooks
(``notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb``, plan 02.6-03, and
``notebooks/02.6_swiss_roll_topoae_curvature_check.ipynb``, plan 02.6-04): a runner number
and a notebook number at the same seed are the same quantity by construction. Every
threshold comparison happens in ``02.6-FINDINGS.md`` (plan 02.6-06) against
``.planning/phases/02.6-decoder-substrate-screening/02.6-SCREENING-RULE.md`` -- never here.

Extends ``02.5-09-SUMMARY.md``'s own executor-side four-seed robustness check (run in a
scratchpad, gating nothing, leaving no re-runnable artifact) into a committed, re-runnable
script. That table measured the CAE's chart-decoder curvature Spearman running
``-0.0604 / -0.1444 / 0.8665 / 0.4250`` across four torch seeds at one configuration -- a
single-seed admission result for either free candidate in this phase would be a draw, not a
result.

Invoke:
    .venv/bin/python notebooks/diagnostics/decoder_substrate_screen_run.py --smoke
    .venv/bin/python notebooks/diagnostics/decoder_substrate_screen_run.py --candidate both
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import cae, chart_curvature, curvature_probe, decoder_curvature, topoae  # noqa: E402

# --- Swiss roll fixture / shared model constants, reused VERBATIM from the two admission
# notebooks (assumption A-06) -- a runner number and a notebook number at the same seed have
# to be the same quantity, or the phase has two incomparable sets of measurements.
FIXTURE_SEED = 20260807
N_POINTS = 3000
LATENT_DIM = 2
HIDDEN = (64, 64, 64)
K_BASELINE = 30
LAMBDA_TOPO = 0.1
HOLDOUT_FRACTION = 0.2

# early_stop_patience is DELIBERATELY ABSENT from CFG_COMMON below: both cae.train_plain_ae
# and topoae.train_topoae default it to max_epochs + 1 when the key is unset, so both arms
# run the identical full 150-epoch budget. Phase 02.4-05 reopened an early-stop asymmetry --
# train_topoae's plateau detector firing against a non-stationary warm-up/ramp objective,
# truncating every TopoAE fit at 15 of 40 epochs while its matched baseline ran the full 40,
# a confound T1/T2/T3's ratios could not tolerate -- and leaving this key unset here
# sidesteps that whole class of confound rather than re-deriving a stopping rule for it.
CFG_COMMON = dict(lr=3e-4, weight_decay=1e-4, batch=64, max_epochs=150)

DEFAULT_SEEDS = (0, 1, 2, 3)  # ratified D1, 02.6-SCREENING-RULE.md "The seed protocol"

TABLE_COLUMNS = (
    "torch seed",
    "recon rel err",
    "median cosine",
    "rho",
    "median ratio",
    "CV",
    "calib slope",
    "calib R2",
    "max cond",
)


def build_fixture(n: int, fixture_seed: int) -> dict:
    """The Swiss roll fixture, built once per invocation and shared across every seed and
    both candidates -- only the torch/split seed varies, matching 02.5-09's own seed
    protocol. Returns a dict with the fixture's ``X``, ``t``, ``H_norm``, ``global_std``,
    plus ``H_true`` (the analytic mean curvature VECTOR, needed for the direction axis) and
    the float32 ``x_all`` tensor every candidate trains from.

    Asserts the pin between ``H_true``'s row norms and the fixture's own sealed ``H_norm``
    is below ``1e-12`` and prints it once, so the ground truth is verified in every run
    rather than trusted.
    """
    fx = curvature_probe.make_swiss_roll_fixture(n=n, seed=fixture_seed)
    H_true = decoder_curvature.swiss_roll_analytic_H_vector(fx["t"], fx["global_std"])
    pin = float(np.max(np.abs(np.linalg.norm(H_true, axis=-1) - fx["H_norm"])))
    if not pin < 1e-12:
        raise ValueError(
            f"build_fixture: swiss_roll_analytic_H_vector's row norms disagree with the "
            f"fixture's own sealed H_norm by {pin:.3e}, exceeding the 1e-12 pin. The "
            "analytic direction and the sealed magnitude ground truth must agree before "
            "either is trusted."
        )
    print(
        f"  fixture pin (||H_true_vec|| vs curvature_probe's sealed H_norm): "
        f"max abs diff = {pin:.3e}  (< 1e-12)"
    )
    x_all = torch.tensor(fx["X"], dtype=torch.float32)
    return {
        "X": fx["X"],
        "t": fx["t"],
        "H_norm": fx["H_norm"],
        "global_std": fx["global_std"],
        "H_true": H_true,
        "x_all": x_all,
    }


def train_candidate(candidate: str, x_train: torch.Tensor, seed: int):
    """Trains one free candidate from scratch and returns ``(model.eval(), fit)``.

    Both candidates build the identical ``cae.PlainAutoEncoder(3, LATENT_DIM,
    hidden=HIDDEN, activation="silu")`` after ``torch.manual_seed(seed)`` -- the TopoAE
    arm is a PlainAutoEncoder trained under a different loss, never a different
    architecture. ``"plainae"`` trains under ``cae.train_plain_ae``; ``"topoae"`` trains
    under ``topoae.train_topoae`` with ``lambda_topo=LAMBDA_TOPO`` and a quarter-warmup/
    quarter-ramp schedule, matching Phase 02.4's own frozen schedule shape. Raises
    ``ValueError`` naming the received value for any other candidate string.
    """
    torch.manual_seed(seed)
    model = cae.PlainAutoEncoder(3, LATENT_DIM, hidden=HIDDEN, activation="silu")
    if candidate == "plainae":
        fit = cae.train_plain_ae(model, x_train, dict(seed=seed, **CFG_COMMON))
    elif candidate == "topoae":
        fit = topoae.train_topoae(
            model,
            x_train,
            dict(
                seed=seed,
                lambda_topo=LAMBDA_TOPO,
                warmup_frac=0.25,
                ramp_frac=0.25,
                **CFG_COMMON,
            ),
        )
    else:
        raise ValueError(
            f"train_candidate: unknown candidate {candidate!r}; expected 'plainae' or "
            "'topoae'."
        )
    return model.eval(), fit


def screen_seed(candidate: str, fx: dict, seed: int) -> dict:
    """One (candidate, seed) measurement: trains from scratch on an 80/20 split of the
    shared fixture, then measures held-out reconstruction and, separately, exact decoder
    curvature against the analytic answer on every point of the fixture.

    Split via ``np.random.default_rng(seed).permutation`` -- the same torch/split seed
    both selects the training draw and initialises the model, matching 02.5-09's own
    protocol (fixture held fixed, torch/split seed varied). Held-out reconstruction
    relative error follows the same per-point-ratio convention as the phase's own
    Swiss-roll sanity notebooks (``notebooks/02.2_swiss_roll_cae_check.ipynb`` and this
    phase's own admission notebooks, 02.6-03/04): the mean, over held-out points, of
    ``||x - y|| / ||x||`` -- computed alongside ``cae.reconstruction_stats`` rather than
    replacing it, so the aggregate ``mse_per_dim`` stays available for the per-seed
    console line even though the table itself reports the per-point ratio.

    The model is then cast to float64 (second derivatives are where float32 noise shows),
    encoded once over the WHOLE fixture (curvature is reported at every point, not just
    the held-out split), and differentiated through ``decoder_curvature.plain_decoder_curvature``
    -- ``model.decode`` only, never ``model.forward``, so the curvature measured is of the
    decoder's own image manifold and not the encoder-composed round-trip map.
    ``chart_curvature.curvature_fidelity_report`` and ``curvature_probe.spearman_gate_statistic``
    score the four axes, unchanged, never reimplemented.

    Returns one flat dict keyed exactly by ``TABLE_COLUMNS``, plus ``epochs_run``,
    ``early_stopped``, ``wallclock_s``, ``n_excluded`` and ``activation`` for provenance.
    Every value is a native Python float or int.
    """
    x_all = fx["x_all"]
    n = x_all.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * (1 - HOLDOUT_FRACTION))
    train_idx = perm[:n_train]
    holdout_idx = perm[n_train:]
    x_train = x_all[torch.from_numpy(train_idx)]
    x_holdout = x_all[torch.from_numpy(holdout_idx)]

    t0 = time.monotonic()
    model, fit = train_candidate(candidate, x_train, seed)
    wallclock_s = time.monotonic() - t0

    with torch.no_grad():
        y_holdout = model(x_holdout)["y"]
    recon_stats = cae.reconstruction_stats(x_holdout.double(), y_holdout.double())
    recon_rel_err = float(
        (
            torch.linalg.vector_norm(x_holdout - y_holdout, dim=-1)
            / torch.linalg.vector_norm(x_holdout, dim=-1)
        ).mean()
    )

    model.double()
    with torch.no_grad():
        z_all = model.encode(x_all.double())
    curv = decoder_curvature.plain_decoder_curvature(model, z_all)
    fidelity = chart_curvature.curvature_fidelity_report(curv["H_vec"], fx["H_true"])
    rho = curvature_probe.spearman_gate_statistic(
        curv["H_norm"].detach().cpu().numpy(), fx["H_norm"]
    )

    row = {
        "torch seed": int(seed),
        "recon rel err": recon_rel_err,
        "median cosine": float(fidelity["median_cosine_similarity"]),
        "rho": float(rho),
        "median ratio": float(fidelity["median_magnitude_ratio"]),
        "CV": float(fidelity["magnitude_ratio_cv"]),
        "calib slope": float(fidelity["calibration_slope"]),
        "calib R2": float(fidelity["calibration_r2"]),
        "max cond": float(curv["metric_condition_number"].max().item()),
        "epochs_run": int(fit["epochs_run"]),
        "early_stopped": bool(fit["early_stopped"]),
        "wallclock_s": float(wallclock_s),
        "n_excluded": int(fidelity["n_excluded"]),
        "activation": curv["activation"],
        "recon_mse_per_dim": float(recon_stats["mse_per_dim"]),
    }
    return row


def markdown_table(rows) -> str:
    """A GitHub-flavoured markdown table with ``TABLE_COLUMNS`` as the header, one row per
    seed, values formatted to four decimal places. No summary row and no derived column --
    the mean/median/SPREAD triple is printed separately by the caller."""
    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for col in TABLE_COLUMNS:
            v = row[col]
            cells.append(str(int(v)) if col == "torch seed" else f"{float(v):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", choices=("plainae", "topoae", "both"), default="both"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n", type=int, default=N_POINTS)
    parser.add_argument("--fixture-seed", type=int, default=FIXTURE_SEED)
    parser.add_argument("--max-epochs", type=int, default=CFG_COMMON["max_epochs"])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast path: --n 400 --max-epochs 4, a single seed. Exercises every code path "
            "-- fixture, both training loops, the shared curvature path, the four-axis "
            "report, the raw-point comparator, the table formatter -- in under 60 "
            "seconds. This is the runner's automated <verify>."
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        args.n = 400
        args.max_epochs = 4
        args.seeds = args.seeds[:1]

    CFG_COMMON["max_epochs"] = args.max_epochs

    print("=" * 78)
    print("Phase 02.6 decoder-substrate screening runner")
    print(
        "MEASURES AND PRINTS ONLY. Applies no bar, emits no verdict, writes no artifact -- "
        "the phase's bars are applied only in 02.6-FINDINGS.md."
    )
    print(
        f"candidate={args.candidate}  seeds={args.seeds}  n={args.n}  "
        f"fixture_seed={args.fixture_seed}  max_epochs={args.max_epochs}"
    )
    print("=" * 78)

    fx = build_fixture(args.n, args.fixture_seed)

    print()
    print(
        "Raw-point comparator (computed once, seed-independent -- its estimator involves "
        "no torch seed):"
    )
    H_raw = curvature_probe.centroid_mean_curvature(fx["X"], k=K_BASELINE, d=LATENT_DIM)
    raw_fidelity = chart_curvature.curvature_fidelity_report(H_raw, fx["H_true"])
    raw_rho = curvature_probe.spearman_gate_statistic(
        np.linalg.norm(H_raw, axis=-1), fx["H_norm"]
    )
    print(
        f"  raw-point (k={K_BASELINE}, d={LATENT_DIM}): rho={raw_rho:.4f}  "
        f"median cosine={raw_fidelity['median_cosine_similarity']:.4f}  "
        f"median ratio={raw_fidelity['median_magnitude_ratio']:.4f}  "
        f"CV={raw_fidelity['magnitude_ratio_cv']:.4f}  "
        f"calib slope={raw_fidelity['calibration_slope']:.4f}  "
        f"calib R2={raw_fidelity['calibration_r2']:.4f}"
    )
    print(
        "  seed-independent: this estimator's rho does not depend on any torch seed. "
        "Reference only, not a perfect answer -- per 02.6-SCREENING-RULE.md, the "
        "raw-point arm itself cleared only 3 of its own notebook's 4 sanity lines "
        "(02.5-09-SUMMARY.md, 'Reported, not absorbed')."
    )

    candidates = ["plainae", "topoae"] if args.candidate == "both" else [args.candidate]

    for candidate in candidates:
        print()
        print("=" * 78)
        print(f"CANDIDATE: {candidate}")
        print("=" * 78)
        rows = []
        for seed in args.seeds:
            print(f"  [seed={seed}] training...")
            try:
                row = screen_seed(candidate, fx, seed)
            except Exception:
                print(f"  [seed={seed}] EXCEPTION -- continuing to the next seed:")
                traceback.print_exc()
                continue
            rows.append(row)
            print(
                f"  [seed={seed}] recon_rel_err={row['recon rel err']:.4f}  "
                f"recon_mse_per_dim={row['recon_mse_per_dim']:.6f}  rho={row['rho']:.4f}  "
                f"median_cosine={row['median cosine']:.4f}  "
                f"calib_slope={row['calib slope']:.4f}  epochs_run={row['epochs_run']}  "
                f"early_stopped={row['early_stopped']}  "
                f"wallclock_s={row['wallclock_s']:.1f}  activation={row['activation']!r}"
            )

        print()
        print(markdown_table(rows))
        print()
        if rows:
            rhos = [r["rho"] for r in rows]
            finite_rhos = [r for r in rhos if np.isfinite(r)]
            if finite_rhos:
                spread = max(finite_rhos) - min(finite_rhos)
                print(
                    f"  rho: mean={np.mean(finite_rhos):.4f}  "
                    f"median={np.median(finite_rhos):.4f}  SPREAD={spread:.4f} (max - min)"
                )
            else:
                print("  rho: no finite values across seeds -- mean/median/SPREAD undefined")
        else:
            print("  No seed completed for this candidate -- no table, no summary statistics.")

    print()
    print("=" * 78)
    print(
        "MEASUREMENTS ONLY. This runner applies no bar and emits no verdict. The phase's "
        "screening bars live in "
        ".planning/phases/02.6-decoder-substrate-screening/02.6-SCREENING-RULE.md and are "
        "applied only in 02.6-FINDINGS.md -- never here."
    )
    print("=" * 78)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
