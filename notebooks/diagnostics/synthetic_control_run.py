"""
Phase 3 plan 10 (ROADMAP step 4): fit the SAME architecture under the SAME protocol to three
manifolds whose mean curvature is known in closed form -- flat plane, sphere, saddle -- and
state what the PU numbers can and cannot mean.

**Read this before quoting any number this runner prints.** A synthetic manifold sampled
cleanly and fitted under this protocol trains to a clean, unfragmented atlas. It has none of
the pathology the phase's gate override is worried about -- ``02.5-09``'s chart-count-driven
fragmentation, ``max cond(g)`` climbing from ``3.26`` to ``122.22``, second derivatives
banding along artificial chart seams. A control that passes therefore establishes only that
the decoder-curvature PIPELINE is correct on a manifold that is easy to fit. That is
necessary and NOT sufficient, and it can never be presented as evidence that the PU curvature
field is free of parameterization damage. The one failure mode the override worries about is
precisely the one this instrument is blind to.

What the controls CAN do is real, and it is why they are worth their wall clock:

* **Flat** -- analytic ``||H||`` is exactly zero at every point, so every value the pipeline
  reports there is entirely artifact and its scale is the pipeline's measured NOISE FLOOR,
  directly interpretable and directly comparable against the PU field's own ``||H||``.
* **Sphere** -- analytic ``||H|| = d / R`` under the trace convention, constant over the
  manifold, which makes the magnitude and calibration axes readable in a way a varying field
  does not.
* **Saddle** -- the only fixture with genuine trace-cancellation points, where positive and
  negative principal curvatures cancel in the trace. It is the only one that exercises the
  case ``cond(g)`` exists to disambiguate: small ``||H||`` because the curvature really
  cancels, versus small ``||H||`` because the pullback metric is near-singular.

Matched means matched. Every protocol and architecture constant is IMPORTED from
``curvature_field_pu_run`` rather than re-typed here, because a re-typed constant is how
matched quietly stops being matched. The epoch budget defaults to the CONVERGED protocol
(``CONVERGE_MAX_EPOCHS``, early stopping structurally inert) rather than the nine-cell grid's
40-epoch cap, because the PU field these controls calibrate was computed on the converged fit
-- calibrating against a 40-epoch control would compare the field to a differently-trained
model and call it a match.

Four axes, never collapsed into one score. At ``02.5-09``'s seed 0 the decoder scored cosine
``0.9706`` -- direction nearly right -- while its calibration slope was ``-0.1856`` at
R-squared ``0.0533``. A single agreement number is blind to exactly that split, so direction,
magnitude, calibration and rank stay four separate columns and no arithmetic anywhere below
combines them.

Invoke:
    .venv/bin/python notebooks/diagnostics/synthetic_control_run.py --dry-run
    .venv/bin/python notebooks/diagnostics/synthetic_control_run.py --smoke
    .venv/bin/python notebooks/diagnostics/synthetic_control_run.py --resume
    .venv/bin/python notebooks/diagnostics/synthetic_control_run.py --resume --fixture flat
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Matched protocol, imported and never re-typed -- see the module docstring.
import curvature_field_pu_run as pu  # noqa: E402
from pu_manifold import (  # noqa: E402
    cache,
    cae,
    chart_curvature,
    curvature_probe,
    decoder_curvature,
    synthetic_controls,
)

CONTROL_FIXTURES = ("flat", "sphere", "saddle")
"""The three known-geometry controls. Flat gives the noise floor, sphere gives a constant
analytic magnitude, saddle gives the mixed-sign trace-cancellation case."""

CONTROL_N = 10000
"""Matched to the PU cloud's own row count, so the controls are not quietly easier by being
smaller."""

CONTROL_FIXTURE_SEED = 20260816
"""Deliberately distinct from ``curvature_field_pu_run.PU_SPLIT_SEED`` (20260813). The
fixtures are constructed data, not a resample of PU; sharing the split seed would invite the
misreading that they are the same draw."""

SPHERE_RADIUS = 1.0
"""Analytic ``||H|| = d / R`` under the trace convention. At ``R = 1`` and ``d = 20`` that is
exactly 20 before the fixture's own ``global_std`` rescaling."""

SADDLE_DOMAIN_RADIUS = 2.0
"""``make_saddle_control``'s own default, restated here so the fixture call site is explicit."""

CONTROL_RECORD_STEM = "03_synthetic_controls"

CONTROL_EPOCH_BLOCK = 25
"""Epochs per ``train_cae`` call. **This is a documented deviation from the PU protocol, not
a tuning choice, and it exists because the matched protocol does not survive its own budget.**

Measured 2026-08-16 on the flat fixture at full scale (n=10000, d=20, D=768, n_charts=4):
a single continuous 300-epoch ``train_cae`` call diverges to non-finite weights at roughly
epoch 220 (~27,500 optimizer steps), surfacing as
``linalg.svd: input matrix contained non-finite values`` inside ``cae.lipschitz_penalty``'s
spectral-norm product. The identical configuration -- same data, same seed, same
hyperparameters -- trained in 25-epoch blocks completed all 300 epochs cleanly, with encoder,
decoder and embedding-decoder weight magnitudes stable throughout (enc 0.16 -> 0.48,
dec 0.22 -> 0.51, emb 0.16 -> 0.64) and reconstruction descending monotonically.

``cae.train_cae`` constructs a fresh optimizer per call, so blocking silently restarts Adam's
moment estimates every ``CONTROL_EPOCH_BLOCK`` epochs. That difference is the only systematic
one between the two runs, which points at accumulated optimizer state rather than weight
magnitude as the divergence mechanism -- a hypothesis consistent with the evidence and NOT a
confirmed cause; confirming it needs a continuous run with per-step instrumentation.

**Consequence for interpretation, which must travel with any control number:** the controls
are therefore NOT trained identically to the converged PU fit (one continuous 300-epoch call).
Architecture, data scale, optimizer settings, learning rate, batch, Lipschitz weight and epoch
budget all still match; the optimizer-state schedule does not. Set ``--epoch-block 0`` to run
continuously and reproduce the divergence."""

MIN_TRUE_NORM = 1e-12
"""``curvature_fidelity_report``'s own ``min_true_norm`` default, restated so this module can
ask the applicability question BEFORE calling it rather than catching its refusal. Used only
to decide whether a fixture has a non-degenerate analytic field at all -- never as a threshold
on any measured quantity."""

# Smoke sizes -- deliberately tiny, to prove the wiring, never a real measurement.
SMOKE_N = 400
SMOKE_D = 4
SMOKE_AMBIENT = 24
SMOKE_N_CHARTS = 2
SMOKE_EPOCHS = 2

DAMAGE_CAVEAT = """\
------------------------------------------------------------------------------------------
WHAT THESE CONTROLS CANNOT ESTABLISH -- read this with the numbers, not after them.

These fixtures are sampled cleanly and fit to a clean, unfragmented atlas. They therefore
NEVER reproduce the atlas fragmentation 02.5-09 measured on real data (chart-count-driven
fragmentation, max cond(g) climbing 3.26 -> 122.22, second derivatives banding along
artificial chart seams).

A control that passes establishes that the decoder-curvature PIPELINE is correct on a
manifold that is easy to fit. It is necessary and NOT sufficient. It CANNOT rule out
parameterization damage on PU, because the one failure mode the gate override worries about
is precisely the one a cleanly-training synthetic manifold cannot exhibit.

Phase 3 runs on a DELIBERATE OVERRIDE of a precondition no measurement in this milestone has
satisfied: every sealed verdict here is FAIL (02, 02.2, 02.4, 02.5 stage 1). Any answer below
about whether PU's curvature is real or artifact is CONDITIONED on that override and must
never be written as if the parameterization had been independently validated.
------------------------------------------------------------------------------------------"""


def _banner(msg: str) -> None:
    print("\n" + "=" * 90)
    print(msg)
    print("=" * 90)


def _default_record_path() -> Path:
    return cache.cache_path(CONTROL_RECORD_STEM, "jsonl")


def load_completed(record_path: Path) -> Dict[str, Dict[str, Any]]:
    """Every JSON-lines record on disk, keyed by ``config_id``. Same idiom as the PU runner."""
    completed: Dict[str, Dict[str, Any]] = {}
    if not record_path.exists():
        return completed
    with record_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completed[rec["config_id"]] = rec
    return completed


def append_record(record_path: Path, record: Dict[str, Any]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a") as f:
        f.write(json.dumps(pu._to_jsonable(record)) + "\n")


def build_fixture(
    name: str, n: int, d: int, D: int, seed: int
) -> Dict[str, np.ndarray]:
    """Construct one control. No geometry is written here -- every fixture comes from
    ``pu_manifold.synthetic_controls``, whose saddle arithmetic is itself cross-checked
    against an independent finite-difference computation by its own test suite."""
    if name == "flat":
        return synthetic_controls.make_flat_control(n=n, d=d, D=D, seed=seed)
    if name == "sphere":
        return synthetic_controls.make_sphere_control(n=n, d=d, D=D, seed=seed, R=SPHERE_RADIUS)
    if name == "saddle":
        return synthetic_controls.make_saddle_control(
            n=n, d=d, D=D, seed=seed, domain_radius=SADDLE_DOMAIN_RADIUS
        )
    raise ValueError(f"build_fixture: unknown fixture {name!r}; expected one of {CONTROL_FIXTURES}.")


def build_matched_cae(
    n_charts: int, chart_dim: int, embed_dim: int, in_dim: int, device: torch.device
) -> "cae.ChartAutoEncoder":
    """The PU architecture, with width and activation imported from the PU runner. Constructed
    then moved to device -- never ``device=`` in the constructor -- so ``torch.manual_seed``'s
    RNG consumption order matches the PU fits exactly (03-07-SUPPLEMENT-01 hard constraint 2)."""
    model = cae.ChartAutoEncoder(
        in_dim=in_dim,
        embed_dim=embed_dim,
        chart_dim=chart_dim,
        n_charts=n_charts,
        hidden=[pu.PU_HIDDEN_WIDTH, pu.PU_HIDDEN_WIDTH, pu.PU_HIDDEN_WIDTH],
        activation=pu.PU_ACTIVATION,
    )
    return model.to(device)


def train_blocked(
    model: Any,
    x_train32: torch.Tensor,
    seed: int,
    n_charts: int,
    max_epochs: int,
    block: int,
) -> Dict[str, Any]:
    """Train for ``max_epochs`` in ``block``-epoch segments, concatenating the histories.

    ``block <= 0`` runs a single continuous call -- the true PU protocol, which is known to
    diverge on the flat fixture at this budget (see :data:`CONTROL_EPOCH_BLOCK`). Kept as an
    option precisely so the divergence stays reproducible rather than becoming folklore.

    FPS pre-training runs only in the first segment: it is a one-off initialization stage
    (eq. 5, seeded per-chart), and repeating it every block would restart the atlas from its
    seed points over and over, which would be a far larger protocol change than the optimizer
    restart this function already carries."""
    if block <= 0:
        cfg = pu._converge_cfg(seed, n_charts, max_epochs)
        fit = cae.train_cae(model, x_train32, cfg)
        fit["epoch_block"] = 0
        fit["n_blocks"] = 1
        return fit

    history: List[Dict[str, Any]] = []
    total = 0
    early_stopped = False
    wallclock_truncated = False
    n_blocks = 0
    while total < max_epochs:
        this_block = min(block, max_epochs - total)
        cfg = pu._converge_cfg(seed, n_charts, this_block)
        if n_blocks > 0:
            cfg["fps_pretrain_epochs"] = 0
        fit = cae.train_cae(model, x_train32, cfg)
        history.extend(fit["history"])
        total += int(fit["epochs_run"])
        n_blocks += 1
        early_stopped = early_stopped or bool(fit["early_stopped"])
        wallclock_truncated = wallclock_truncated or bool(fit["wallclock_truncated"])
        if fit["early_stopped"] or fit["wallclock_truncated"]:
            break
    return {
        "history": history,
        "epochs_run": total,
        "early_stopped": early_stopped,
        "wallclock_truncated": wallclock_truncated,
        "cfg": pu._converge_cfg(seed, n_charts, max_epochs),
        "epoch_block": block,
        "n_blocks": n_blocks,
    }


def _fidelity_axes(H_est: np.ndarray, H_true: np.ndarray) -> Dict[str, Any]:
    """The four axes, kept as four. ``curvature_fidelity_report`` supplies direction,
    magnitude and calibration; ``spearman_gate_statistic`` supplies rank;
    ``median_relative_error`` is context. Nothing here combines them.

    **The flat control has no fidelity axes, and this is mathematics rather than a special
    case.** Its analytic ``H`` is the exactly-zero vector at every point, so cosine similarity
    against it is undefined, the magnitude ratio divides by zero, and a calibration regression
    on a constant-zero predictor has no slope. ``curvature_fidelity_report`` refuses such an
    input outright ("A fidelity report over fewer than two points is not a measurement"), and
    that refusal is correct. What the flat fixture measures instead is the pipeline's NOISE
    FLOOR: every value it reports is artifact, recorded under ``noise_floor``. Returning a
    not-applicable marker here keeps that distinction visible rather than manufacturing four
    numbers that would each be a division by zero wearing a float."""
    h_true_norm_all = np.linalg.norm(H_true, axis=-1)
    if not np.any(h_true_norm_all > MIN_TRUE_NORM):
        return {
            "applicable": False,
            "reason": (
                "analytic ||H|| is exactly zero at every point: cosine, magnitude ratio and "
                "calibration slope are all undefined against a zero field. This fixture "
                "measures the pipeline's noise floor instead -- see the noise_floor key."
            ),
            "curvature_convention": chart_curvature.CURVATURE_CONVENTION,
        }
    report = chart_curvature.curvature_fidelity_report(H_est, H_true)
    h_est_norm = np.linalg.norm(H_est, axis=-1)
    h_true_norm = np.linalg.norm(H_true, axis=-1)
    rho = curvature_probe.spearman_gate_statistic(h_est_norm, h_true_norm)
    mre = curvature_probe.median_relative_error(h_est_norm, h_true_norm)
    return {
        "applicable": True,
        "direction_median_cosine": report["median_cosine_similarity"],
        "magnitude_median_ratio": report["median_magnitude_ratio"],
        "magnitude_ratio_cv": report["magnitude_ratio_cv"],
        "calibration_slope": report["calibration_slope"],
        "calibration_intercept": report["calibration_intercept"],
        "calibration_r2": report["calibration_r2"],
        "rank_spearman_rho": float(rho),
        "median_relative_error": float(mre),
        "n_points": report["n_points"],
        "n_excluded": report["n_excluded"],
        "curvature_convention": report["curvature_convention"],
    }


def run_one_control(
    name: str,
    n: int,
    chart_dim: int,
    ambient: int,
    n_charts: int,
    embed_dim: int,
    max_epochs: int,
    seed: int,
    device: torch.device,
    epoch_block: int = CONTROL_EPOCH_BLOCK,
) -> Dict[str, Any]:
    fx = build_fixture(name, n=n, d=chart_dim, D=ambient, seed=seed)
    X = np.asarray(fx["X"], dtype=np.float64)
    H_true = np.asarray(fx["H_vec"], dtype=np.float64)
    h_true_norm = np.linalg.norm(H_true, axis=-1)

    x32 = torch.tensor(X, dtype=torch.float32).to(device)
    x64 = torch.tensor(X, dtype=torch.float64).to(device)

    # Same holdout convention as the PU runner, for the reconstruction figure only.
    train_idx, holdout_idx = pu._split(X.shape[0], pu.PU_SPLIT_SEED, pu.PU_HOLDOUT_FRACTION)
    x_train32 = x32[torch.as_tensor(train_idx, dtype=torch.long, device=device)]
    x_holdout64 = x64[torch.as_tensor(holdout_idx, dtype=torch.long, device=device)]

    torch.manual_seed(seed)
    model = build_matched_cae(n_charts, chart_dim, embed_dim, ambient, device)
    t0 = time.monotonic()
    fit = train_blocked(model, x_train32, seed, n_charts, max_epochs, epoch_block)
    train_wallclock_s = time.monotonic() - t0

    model.eval().double()

    # Before differentiating anything, confirm the decoder is C2 and record what was checked.
    c2_activation = decoder_curvature.assert_c2_decoder(model)

    with torch.no_grad():
        recon_holdout = model.reconstruct(x_holdout64)
    recon_stats = cae.reconstruction_stats(x_holdout64, recon_holdout)

    t1 = time.monotonic()
    field = chart_curvature.chart_curvature_field(model, x64, mode="reverse")
    curv_wallclock_s = time.monotonic() - t1

    H_est = field["H_vec"].detach().cpu().numpy().astype(np.float64)
    h_est_norm = field["H_norm"].detach().cpu().numpy().astype(np.float64)
    cond = field["metric_condition_number"].detach().cpu().numpy().astype(np.float64)
    assignment = field["chart_assignment"].detach().cpu().numpy()
    distinct, counts = np.unique(assignment, return_counts=True)

    axes = _fidelity_axes(H_est, H_true)

    # Same flagging policy as the PU field: the within-config percentile and nothing else.
    flag_threshold = float(np.percentile(cond, pu.COND_FLAG_PERCENTILE))
    flagged_mask = cond > flag_threshold

    rec: Dict[str, Any] = {
        "kind": "control_fit",
        "config_id": f"control_{name}_nc{n_charts}_d{chart_dim}_ep{max_epochs}",
        "fixture": name,
        "n": n,
        "chart_dim": chart_dim,
        "ambient_dim": ambient,
        "embed_dim": embed_dim,
        "n_charts": n_charts,
        "max_epochs": max_epochs,
        "fixture_seed": seed,
        "epochs_run": fit["epochs_run"],
        "early_stopped": fit["early_stopped"],
        "wallclock_truncated": fit["wallclock_truncated"],
        "epoch_block": fit["epoch_block"],
        "n_blocks": fit["n_blocks"],
        "protocol_deviation": (
            "trained in %d-epoch blocks; cae.train_cae builds a fresh optimizer per call, so "
            "Adam moment estimates restart each block. The continuous PU protocol diverges to "
            "non-finite weights at ~epoch 220 on the flat fixture. Architecture, data, lr, "
            "batch, Lipschitz weight and epoch budget all still match the PU fit; the "
            "optimizer-state schedule does not." % fit["epoch_block"]
        ) if fit["epoch_block"] > 0 else "none -- single continuous call, the true PU protocol",
        "c2_activation": c2_activation,
        "fidelity": axes,
        "h_norm_est": pu._dist_summary(h_est_norm, pu.FIELD_HIST_BINS),
        "h_norm_true": pu._dist_summary(h_true_norm, pu.FIELD_HIST_BINS),
        "cond": pu._dist_summary(cond, pu.FIELD_HIST_BINS),
        "flagged": {
            "percentile": pu.COND_FLAG_PERCENTILE,
            "threshold": flag_threshold,
            "count": int(flagged_mask.sum()),
            "fraction": float(flagged_mask.mean()),
        },
        "n_charts_used": int(field["n_charts_used"]),
        "per_chart_counts": {int(c): int(v) for c, v in zip(distinct.tolist(), counts.tolist())},
        "reconstruction": recon_stats,
        "train_wallclock_s": train_wallclock_s,
        "curv_wallclock_s": curv_wallclock_s,
        "device": str(device),
        "torch_version": torch.__version__,
    }

    if name == "flat":
        # The analytic answer is exactly zero everywhere, so every value here is artifact and
        # its scale IS the pipeline's noise floor. Recorded under its own key so no reader has
        # to infer that this distribution means something different from the other fixtures'.
        rec["noise_floor"] = {
            "analytic_h_norm": 0.0,
            "measured_median": rec["h_norm_est"]["median"],
            "measured_p95": rec["h_norm_est"]["p95"],
            "measured_max": rec["h_norm_est"]["max"],
            "note": "analytic ||H|| is exactly zero; every value measured here is artifact",
        }
    if name == "sphere":
        analytic = float(np.median(h_true_norm))
        measured = rec["h_norm_est"]["median"]
        rec["sphere_ratio"] = {
            "analytic_median_h_norm": analytic,
            "measured_median_h_norm": measured,
            "ratio_measured_over_analytic": (measured / analytic) if analytic != 0.0 else float("nan"),
            "R": SPHERE_RADIUS,
        }
    if name == "saddle":
        joint_counts, joint_h_edges, joint_cond_edges = np.histogram2d(
            h_est_norm,
            np.log10(np.maximum(cond, np.finfo(np.float64).tiny)),
            bins=pu.FIELD_JOINT_BINS,
        )
        rec["joint_hist"] = {
            "counts": [[int(c) for c in row] for row in joint_counts],
            "h_norm_edges": [float(e) for e in joint_h_edges],
            "log10_cond_edges": [float(e) for e in joint_cond_edges],
            "axes": "rows=||H|| bins, cols=log10(cond(g)) bins",
        }
        # The near-cancellation set: genuinely small TRUE curvature. Reported beside the
        # near-singular set so the two small-||H|| populations stay distinguishable.
        cancel_mask = h_true_norm <= np.percentile(h_true_norm, 10.0)
        rec["cancellation_vs_singular"] = {
            "cancellation_percentile": 10.0,
            "n_cancellation": int(cancel_mask.sum()),
            "cond_median_at_cancellation": float(np.median(cond[cancel_mask])),
            "cond_median_overall": rec["cond"]["median"],
            "h_est_median_at_cancellation": float(np.median(h_est_norm[cancel_mask])),
            "h_est_median_overall": rec["h_norm_est"]["median"],
        }
    return rec


def print_control_row(rec: Dict[str, Any]) -> None:
    f = rec["fidelity"]
    print(f"\n--- {rec['fixture'].upper()} (n={rec['n']}, d={rec['chart_dim']}, D={rec['ambient_dim']}, n_charts={rec['n_charts']})")
    print(f"  epochs_run={rec['epochs_run']}/{rec['max_epochs']}  early_stopped={rec['early_stopped']}  c2={rec['c2_activation']}")
    if not f.get("applicable", True):
        print("  four axes: NOT APPLICABLE")
        print(f"    {f['reason']}")
    else:
        print("  four axes, never combined:")
        print(f"    direction   median cosine        = {f['direction_median_cosine']:.6f}")
        print(f"    magnitude   median ratio         = {f['magnitude_median_ratio']:.6f}   CV = {f['magnitude_ratio_cv']:.6f}")
        print(f"    calibration slope={f['calibration_slope']:.6f} intercept={f['calibration_intercept']:.6e} R2={f['calibration_r2']:.6f}")
        print(f"    rank        spearman rho         = {f['rank_spearman_rho']:.6f}")
        print(f"  median_relative_error (context, non-gating) = {f['median_relative_error']:.6f}")
    est, tru, c = rec["h_norm_est"], rec["h_norm_true"], rec["cond"]
    print(f"  ||H|| est : median={est['median']:.6e} p95={est['p95']:.6e} max={est['max']:.6e}")
    print(f"  ||H|| true: median={tru['median']:.6e} p95={tru['p95']:.6e} max={tru['max']:.6e}")
    print(f"  cond(g)   : median={c['median']:.4e} p90={c['p90']:.4e} p99={c['p99']:.4e} max={c['max']:.4e}")
    print(f"  flagged at the within-config {rec['flagged']['percentile']:.0f}th percentile: {rec['flagged']['count']} points")
    print(f"  charts used: {rec['n_charts_used']} of {rec['n_charts']}   holdout mse_per_dim={rec['reconstruction']['mse_per_dim']:.6e}")
    if "noise_floor" in rec:
        nf = rec["noise_floor"]
        print(f"  NOISE FLOOR: analytic ||H|| is exactly 0; measured median={nf['measured_median']:.6e} p95={nf['measured_p95']:.6e} max={nf['measured_max']:.6e}")
        print("               every value on this fixture is artifact -- this is the scale PU's ||H|| must clear")
    if "sphere_ratio" in rec:
        sr = rec["sphere_ratio"]
        print(f"  SPHERE: analytic median ||H||={sr['analytic_median_h_norm']:.6e}  measured={sr['measured_median_h_norm']:.6e}  ratio={sr['ratio_measured_over_analytic']:.6f}")
    if "cancellation_vs_singular" in rec:
        cs = rec["cancellation_vs_singular"]
        print(f"  SADDLE cancellation set (lowest {cs['cancellation_percentile']:.0f}% true ||H||, n={cs['n_cancellation']}):")
        print(f"    cond(g) median there={cs['cond_median_at_cancellation']:.4e} vs overall {cs['cond_median_overall']:.4e}")
        print(f"    ||H|| est median there={cs['h_est_median_at_cancellation']:.6e} vs overall {cs['h_est_median_overall']:.6e}")
    print(f"  wall clock: train={rec['train_wallclock_s']:.1f}s curvature={rec['curv_wallclock_s']:.1f}s")


def print_dry_run_report(n_charts: int, max_epochs: int) -> None:
    _banner("synthetic_control_run.py -- DRY RUN")
    print(DAMAGE_CAVEAT)
    print(f"\nFixtures: {list(CONTROL_FIXTURES)}")
    print(f"  flat   -- analytic ||H|| exactly 0 everywhere; measures the pipeline's noise floor")
    print(f"  sphere -- analytic ||H|| = d / R, constant, R={SPHERE_RADIUS}")
    print(f"  saddle -- mixed-sign, the only fixture with genuine trace-cancellation points")
    print("\nMatched architecture and protocol, imported from curvature_field_pu_run:")
    print(f"  n                    = {CONTROL_N}")
    print(f"  chart_dim            = {pu.PU_CHART_DIM}")
    print(f"  ambient D            = {pu.AMBIENT_DIM}")
    print(f"  embed_dim            = {pu.PU_EMBED_DIM}")
    print(f"  n_charts             = {n_charts}   (the value plan 03-08's rule selected for PU)")
    print(f"  hidden               = [{pu.PU_HIDDEN_WIDTH}, {pu.PU_HIDDEN_WIDTH}, {pu.PU_HIDDEN_WIDTH}]")
    print(f"  activation           = {pu.PU_ACTIVATION}")
    print(f"  lr / weight_decay    = {pu.ADAM_LR} / {pu.WEIGHT_DECAY}")
    print(f"  batch                = {pu.BATCH}")
    print(f"  lip_weight           = {pu.LIP_WEIGHT}   every {pu.LIP_EVERY_N_STEPS} step(s)")
    print(f"  fps_pretrain_epochs  = {pu.FPS_PRETRAIN_EPOCHS}")
    print(f"  max_epochs           = {max_epochs}   (converged protocol: early stopping structurally inert)")
    print(f"  epoch_block          = {CONTROL_EPOCH_BLOCK}   PROTOCOL DEVIATION -- see below")
    print(f"  fixture seed         = {CONTROL_FIXTURE_SEED}   (distinct from PU_SPLIT_SEED={pu.PU_SPLIT_SEED})")
    print(f"  cond(g) flag policy  = within-config {pu.COND_FLAG_PERCENTILE:.0f}th percentile, same as the PU field")
    print()
    print("PROTOCOL DEVIATION, stated where the protocol is stated:")
    print("  A single continuous 300-epoch train_cae call DIVERGES to non-finite weights at")
    print("  ~epoch 220 on the flat fixture (linalg.svd on a NaN weight inside")
    print("  cae.lipschitz_penalty). The same configuration in 25-epoch blocks completes all")
    print("  300 epochs cleanly. train_cae builds a fresh optimizer per call, so blocking")
    print("  restarts Adam's moment estimates each block -- the only systematic difference")
    print("  between the two runs. Architecture, data, lr, batch, Lipschitz weight and epoch")
    print("  budget still match the PU fit; the optimizer-state schedule does NOT.")
    print("  --epoch-block 0 reproduces the divergence.")
    print("\nFour axes, reported as four columns and never combined into one score:")
    print("  direction (median cosine) | magnitude (median ratio AND CV) | calibration (slope, intercept, R2) | rank (spearman rho)")
    print(DAMAGE_CAVEAT)
    print("\n--dry-run: nothing executed, nothing written.")


def run_smoke(record_path: Path, device: torch.device) -> None:
    _banner(f"SMOKE -- tiny fixtures (n={SMOKE_N}, d={SMOKE_D}, D={SMOKE_AMBIENT}, {SMOKE_EPOCHS} epochs)")
    for name in CONTROL_FIXTURES:
        rec = run_one_control(
            name,
            n=SMOKE_N,
            chart_dim=SMOKE_D,
            ambient=SMOKE_AMBIENT,
            n_charts=SMOKE_N_CHARTS,
            embed_dim=2 * SMOKE_D,
            max_epochs=SMOKE_EPOCHS,
            seed=CONTROL_FIXTURE_SEED,
            device=device,
        )
        rec["kind"] = "smoke_control_fit"
        rec["config_id"] = f"smoke_{name}"
        append_record(record_path, rec)
        f = rec["fidelity"]
        if f.get("applicable", True):
            print(
                f"  {name}: cosine={f['direction_median_cosine']:.4f} "
                f"rho={f['rank_spearman_rho']:.4f} "
                f"mag_ratio={f['magnitude_median_ratio']:.4g} "
                f"||H||est_median={rec['h_norm_est']['median']:.4g} "
                f"charts_used={rec['n_charts_used']}"
            )
        else:
            print(
                f"  {name}: four axes NOT APPLICABLE (analytic ||H|| is exactly zero); "
                f"noise floor ||H||est_median={rec['h_norm_est']['median']:.4g} "
                f"max={rec['h_norm_est']['max']:.4g} charts_used={rec['n_charts_used']}"
            )
    print("SMOKE: exit 0.")


def run_controls(
    record_path: Path,
    resume: bool,
    fixtures: List[str],
    n_charts: int,
    max_epochs: int,
    device: torch.device,
    epoch_block: int = CONTROL_EPOCH_BLOCK,
) -> None:
    completed = load_completed(record_path) if resume else {}
    records: List[Dict[str, Any]] = []

    _banner("SYNTHETIC CONTROLS -- ROADMAP step 4")
    print(DAMAGE_CAVEAT)

    for name in fixtures:
        config_id = f"control_{name}_nc{n_charts}_d{pu.PU_CHART_DIM}_ep{max_epochs}"
        if resume and config_id in completed:
            print(f"  [skip, resumed] {config_id}")
            records.append(completed[config_id])
            continue
        print(f"\n[{config_id}] fitting...")
        t0 = time.monotonic()
        rec = run_one_control(
            name,
            n=CONTROL_N,
            chart_dim=pu.PU_CHART_DIM,
            ambient=pu.AMBIENT_DIM,
            n_charts=n_charts,
            embed_dim=pu.PU_EMBED_DIM,
            max_epochs=max_epochs,
            seed=CONTROL_FIXTURE_SEED,
            device=device,
            epoch_block=epoch_block,
        )
        append_record(record_path, rec)
        records.append(rec)
        print(f"[{config_id}] done in {time.monotonic() - t0:.1f}s")

    if not records:
        print("No control records.")
        return

    _banner("SYNTHETIC CONTROLS -- four-axis comparison, uncollapsed")
    for rec in records:
        print_control_row(rec)

    print()
    print(DAMAGE_CAVEAT)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Run all three fixtures at a deliberately tiny size (n={SMOKE_N}, d={SMOKE_D}, "
        f"D={SMOKE_AMBIENT}, {SMOKE_EPOCHS} epochs) to prove the wiring, and print a tally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixtures, the matched architecture and protocol, the four axes and "
        "the parameterization-damage caveat, then exit writing nothing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip any fixture already present in the record file rather than refitting it.",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default=None,
        help=f"Override the append-only record path (default: notebooks/.cache/"
        f"{CONTROL_RECORD_STEM}.jsonl).",
    )
    parser.add_argument(
        "--fixture",
        type=str,
        choices=list(CONTROL_FIXTURES),
        default=None,
        help="Run one fixture only. Omitted: all three.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=pu.CONVERGE_MAX_EPOCHS,
        help=f"Epoch budget (default: {pu.CONVERGE_MAX_EPOCHS}, the converged PU protocol the "
        "field being calibrated was computed under). Early stopping is structurally inert.",
    )
    parser.add_argument(
        "--n-charts",
        type=int,
        default=pu.CONVERGE_N_CHARTS,
        help=f"n_charts (default: {pu.CONVERGE_N_CHARTS}, the value plan 03-08's pre-declared "
        "rule selected for PU). Matched means matched.",
    )
    parser.add_argument(
        "--epoch-block",
        type=int,
        default=CONTROL_EPOCH_BLOCK,
        help=f"Epochs per train_cae call (default: {CONTROL_EPOCH_BLOCK}). 0 runs a single "
        "continuous call -- the true PU protocol, which diverges to non-finite weights at "
        "~epoch 220 on the flat fixture. See CONTROL_EPOCH_BLOCK's docstring.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' (default), 'cuda', or 'cuda:N'. GPU runs do NOT reproduce CPU runs "
        "bit-for-bit; every record carries its own 'device' field.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    record_path = Path(args.record_path) if args.record_path else _default_record_path()
    device = pu.resolve_device(args.device)

    _banner("Phase 3 Plan 10: synthetic control fits (ROADMAP step 4)")
    print(f"record_path={record_path}")
    print(f"device={device}  torch_version={torch.__version__}")

    if args.dry_run:
        print_dry_run_report(args.n_charts, args.epochs)
        return

    if args.smoke:
        run_smoke(record_path, device)
        return

    fixtures = [args.fixture] if args.fixture else list(CONTROL_FIXTURES)
    run_controls(
        record_path, args.resume, fixtures, args.n_charts, args.epochs, device, args.epoch_block
    )


if __name__ == "__main__":
    main()
