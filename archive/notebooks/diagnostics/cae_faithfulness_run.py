"""
Developer-directed work (see the phase 3 GPU-faithfulness handoff, not a numbered plan):
one self-contained experiment. Train a single, genuinely converged Chart Auto-Encoder on
the frozen PU subsample, then measure faithfulness strictly in the CAE paper's
(arXiv:1912.10094) own terms.

**This is NOT a comparison.** No plain autoencoder, no VAE, no protocol sweep, no A/B arm,
and no reference to any prior fit's constants. One model, trained properly, measured. The
"spread across seeds" reported at the end is the spread of THIS one config across its own
seeds -- never a comparison against a second model.

Convergence, not early stopping. ``train_cae`` early-stops on TOTAL loss (reconstruction +
cross-entropy + Lipschitz); a prior PU fit under a tight patience halted at epoch 7 --
a plateau in the penalty term, not in reconstruction. Here ``early_stop_patience`` is set
to ``max_epochs + 1`` (structurally inert: plateau_count can increment at most once per
epoch, so it can never reach ``max_epochs + 1`` within ``max_epochs`` epochs) AND, after
training, ``epochs_run == max_epochs`` is asserted explicitly -- belt and suspenders. If
that assertion ever fires, this runner refuses to report faithfulness numbers for an
unconverged model rather than reporting them anyway. The wallclock ceiling is disabled
(``float("inf")``) for the same reason.

Hyperparameters are the CAE paper's own stated experiment-section values: ``lr=3e-4``,
``batch=64``, Lipschitz penalty weight ``1e-2`` -- NOT the ``1e-3`` this milestone's other
PU runner (``curvature_field_pu_run.py``) carries after its own defect-driven realignment;
that value was chosen to fix a truncation defect under a tight patience, a problem this
runner does not have because patience is inert here by construction. ``chart_dim=20`` (the
PU intrinsic-dimension estimate, D-11) and ``embed_dim=40`` (``2d``, the paper's
Nash-Kuiper rationale) are this milestone's own established PU architecture constants,
both exposed as ``--chart-dim``/``--embed-dim`` for override. ``n_charts`` is
over-specified (default 32, the paper's own stated maximum) and left to the paper's own
a-posteriori pruning (``cae.chart_survival``) to decide the surviving count, rather than
chosen to match any prior sweep's winner.

Four faithfulness measurements, each in the paper's own terms, none collapsed into another:

1. **Sup-norm reconstruction error**, train and holdout separately -- the ``epsilon`` of
   Definition 1 and Theorems 1-3. This milestone has recorded only means before now; here
   mean, median, p95, p99, and max are all reported.
2. **Unfaithfulness and coverage** (eqs. 20-21) via ``cae.unfaithfulness_coverage``,
   unedited and imported, with ``n_distinct``/``n_samples`` alongside so coverage is never
   read without its own denominator.
3. **R_cycle** (eq. 8) -- the chart-transition residual. ``cae.r_cycle`` and
   ``cae.select_overlap_pairs`` already implement eq. 8 and the pre-registered T2
   overlap-pair selection rule (added for plan 02.2's T2 gate, exercised again by
   ``cae_evaluate_run.py``); this runner imports and calls them exactly as
   ``cae_evaluate_run.py`` does (grouped per (alpha, beta) pair, batched through
   ``r_cycle``), and reports the full distribution rather than only the mean.
   ``cae.py`` is not edited to build this.
4. **Chart survival vs. argmax occupancy**, side by side, never collapsed. Prior work found
   these two instruments disagree badly (16/16 charts nominally surviving by decoder
   weight-mass while the decoded image had collapsed onto 24 distinct points) -- reporting
   only one would hide exactly that disagreement.

Device support follows 03-07-SUPPLEMENT-01's pattern verbatim (construct-then-``.to(device)``,
never ``device=`` in a constructor; ``device``/``torch_version`` in every record; ``--resume``
refuses to continue a record file whose device disagrees).

Invoke:
    .venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --dry-run
    .venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --smoke
    .venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --resume
    .venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --device cuda --epochs 500 --resume

See also: .planning/phases/03-decoder-curvature-field/03-GPU-RUNBOOK-FAITHFULNESS.md
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import cache, cae  # noqa: E402

# =============================================================================================
# Constants -- architecture, protocol, split. All overridable from the CLI where the task
# brief calls for it (--chart-dim, --embed-dim, --n-charts, --epochs, --unfaithful-samples,
# --seeds, --device); everything else is a documented, fixed choice.
# =============================================================================================

AMBIENT_DIM = 768
HIDDEN_WIDTH = 250
ACTIVATION = "silu"

DEFAULT_CHART_DIM = 20
"""D-11: the PU intrinsic-dimension estimate. See curvature_field_pu_run.py's own docstring
for the full justification (TwoNN 19.5, local-PCA median 25.0, median-of-eight-estimators
18); this runner does not re-derive it, only reuses the established constant as a default."""
DEFAULT_EMBED_DIM = 40
"""``2d`` -- the paper's Nash-Kuiper rationale, this milestone's established PU value."""
DEFAULT_N_CHARTS = 32
"""The paper's own stated maximum. Over-specifying and letting a-posteriori pruning
(cae.chart_survival) decide the surviving count is the point -- NOT derived from any prior
sweep's winner (that would be exactly the kind of comparison this task explicitly forbids)."""

ADAM_LR = 3e-4
WEIGHT_DECAY = 1e-4
"""Not stated in the brief's hyperparameter list (lr, batch, Lipschitz weight); carried
from this milestone's established PU/roll protocol as a reasonable default, not tuned."""
BATCH = 64
LIP_WEIGHT = 1e-2
"""The CAE paper's own stated experiments-section value -- deliberately NOT the 1e-3
curvature_field_pu_run.py carries after its own truncation-defect fix (03-08-DEFECTS-01.md
defect 2); that fix targeted a tight-patience problem this runner does not have."""
LIP_EVERY_N_STEPS = 1
FPS_PRETRAIN_EPOCHS = 5
EARLY_STOP_MIN_DELTA = 1e-4
"""Moot by construction -- see DEFAULT_EPOCHS's early_stop_patience note -- kept only so
train_cae's effective_cfg echoes an explicit value rather than a silent default."""

DEFAULT_EPOCHS = 300
""""Generous" per the brief; matches the established MAX_EPOCHS precedent in this
milestone's Swiss roll protocol (swiss_roll_curvature_sweep_run.py's BASE_CFG). Exposed as
--epochs because the colleague running this has a GPU and can raise it."""

DEFAULT_UNFAITHFUL_SAMPLES = 1000
"""Matches this milestone's established PU value (cae_train_run.py's UNFAITHFUL_SAMPLES),
itself larger than the paper's own hundred-sample runs for a less noisy estimate."""

PRUNE_TOL = 1e-2
"""cae.chart_survival's pruning tolerance -- this milestone's established PU value
(cae_train_run.py's PRUNE_TOL), reused rather than re-derived."""
OVERLAP_P_MIN = 0.05
OVERLAP_MIN_POINTS = 200
"""T2 overlap-pair selection thresholds -- this milestone's established PU values
(cae_train_run.py's OVERLAP_P_MIN / OVERLAP_MIN_POINTS), reused rather than re-derived."""

PU_SPLIT_SEED = 20260813
PU_HOLDOUT_FRACTION = 0.2
"""Same split convention as curvature_field_pu_run.py -- one 8,000/2,000 split of the
frozen subsample, reused for comparability across this milestone's PU runners, not
re-derived here."""

DEFAULT_SEEDS: Tuple[int, ...] = (20260813, 20260814, 20260815)
"""The reported unit is the spread across these three seeds, never a single draw."""

SUBSAMPLE_STEM = "subsample_20260729_a79b3460b838fd0a"
"""The frozen Phase 1 10,000-row PU subsample every prior fit in this milestone was trained
and split against. Gitignored, irreproducible here -- see :func:`_load_subsample`."""

RECORD_STEM = "03_cae_faithfulness_pu"

# Smoke-mode sizes -- deliberately tiny, to prove the wiring, never a real measurement.
SMOKE_TRAIN_N = 200
SMOKE_HOLDOUT_N = 150
SMOKE_N_CHARTS = 2
SMOKE_MAX_EPOCHS = 3
SMOKE_UNFAITHFUL_SAMPLES = 50
SMOKE_OVERLAP_P_MIN = 0.02
SMOKE_OVERLAP_MIN_POINTS = 5
SMOKE_SEED = DEFAULT_SEEDS[0]


def resolve_device(device_str: str) -> torch.device:
    """Parse and validate a ``--device`` CLI argument ('cpu', 'cuda', or 'cuda:N') --
    identical contract to curvature_field_pu_run.py's helper (03-07-SUPPLEMENT-01),
    duplicated rather than imported across these two standalone scripts per this project's
    simplicity preference."""
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"--device={device_str!r} requested but torch.cuda.is_available() is False. "
            f"The installed torch build is CPU-only ({torch.__version__}). Install a CUDA "
            "build matching your driver's CUDA version, e.g.:\n"
            "  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "(see https://pytorch.org/get-started/locally/ for the correct cuXXX tag), then "
            "verify with:\n"
            '  .venv/bin/python -c "import torch; print(torch.cuda.is_available())"\n'
            "See .planning/phases/03-decoder-curvature-field/03-GPU-RUNBOOK-FAITHFULNESS.md "
            "for the full setup walkthrough."
        )
    return device


def _banner(msg: str) -> None:
    print()
    print("=" * 90)
    print(msg)
    print("=" * 90)


# =============================================================================================
# Data loading -- read-only, halts loudly rather than drawing a different subsample.
# =============================================================================================


def _load_subsample() -> np.ndarray:
    """The frozen 10,000-row ``legacysurvey`` array, read-only. Identical discipline to
    curvature_field_pu_run.py's ``_load_subsample`` -- never trains or draws a replacement,
    and checks the cache directory's entry count is unchanged by this read."""
    cache_dir = cache.CACHE_DIR
    n_before = sum(1 for _ in cache_dir.iterdir())
    path = cache.cache_path(SUBSAMPLE_STEM, "npz")
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"cae_faithfulness_run: required frozen Phase 1 PU subsample missing or empty: "
            f"{path}. This is the 10,000-row subsample every prior fit in this milestone was "
            f"trained and split against; it is gitignored and irreproducible here. Halting "
            f"rather than drawing a different subsample."
        )
    arrays = dict(np.load(path))
    n_after = sum(1 for _ in cache_dir.iterdir())
    if n_after != n_before:
        raise RuntimeError(
            f"cae_faithfulness_run: notebooks/.cache entry count changed ({n_before} -> "
            f"{n_after}) while loading the subsample -- this load must stay read-only."
        )
    return arrays["legacysurvey"]


def _split(n: int, split_seed: int, holdout_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_holdout = int(round(n * holdout_fraction))
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    return train_idx, holdout_idx


# =============================================================================================
# Model builder and the shared protocol cfg.
# =============================================================================================


def build_cae(
    n_charts: int, chart_dim: int, embed_dim: int, device: torch.device = torch.device("cpu")
) -> "cae.ChartAutoEncoder":
    """Constructed then moved to ``device`` AFTER construction, never via ``device=`` in the
    constructor -- so ``torch.manual_seed(seed)``'s RNG consumption order is unaffected
    (03-07-SUPPLEMENT-01's discipline)."""
    model = cae.ChartAutoEncoder(
        in_dim=AMBIENT_DIM,
        embed_dim=embed_dim,
        chart_dim=chart_dim,
        n_charts=n_charts,
        hidden=[HIDDEN_WIDTH, HIDDEN_WIDTH, HIDDEN_WIDTH],
        activation=ACTIVATION,
    )
    return model.to(device)


def _protocol_cfg(seed: int, n_charts: int, max_epochs: int) -> Dict[str, Any]:
    """No early stopping, no wallclock ceiling. ``early_stop_patience = max_epochs + 1`` is
    structurally inert: ``plateau_count`` increments at most once per epoch, so across
    ``max_epochs`` epochs it can reach at most ``max_epochs``, never ``max_epochs + 1`` --
    the patience threshold is mathematically unreachable, not merely set "high". See this
    module's own docstring for why the runner also asserts ``epochs_run == max_epochs``
    explicitly after training rather than relying on this alone."""
    return {
        "seed": seed,
        "lr": ADAM_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch": BATCH,
        "max_epochs": max_epochs,
        "early_stop_patience": max_epochs + 1,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "lip_weight": LIP_WEIGHT,
        "lip_every_n_steps": LIP_EVERY_N_STEPS,
        "fps_pretrain_epochs": FPS_PRETRAIN_EPOCHS,
        "wallclock_ceiling_s": float("inf"),
        "n_charts": n_charts,
    }


# =============================================================================================
# Faithfulness measurements.
# =============================================================================================


def _supnorm_stats(x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    """Definition 1 / Theorems 1-3's ``epsilon``: the sup-norm reconstruction error, as a
    full distribution rather than only its max -- mean, median, p95, p99, max of the
    per-row unsquared L2 reconstruction norm ``||x - y||``."""
    diff = (x.detach() - y.detach()).cpu().numpy().astype(np.float64)
    norms = np.linalg.norm(diff, axis=1)
    return {
        "mean": float(norms.mean()),
        "median": float(np.median(norms)),
        "p95": float(np.percentile(norms, 95)),
        "p99": float(np.percentile(norms, 99)),
        "max": float(norms.max()),
        "n": int(norms.shape[0]),
    }


def _occupancy_diagnostic(model: "cae.ChartAutoEncoder", z: torch.Tensor) -> Dict[str, Any]:
    """Argmax chart occupancy, computed independently of cae.chart_survival -- reused
    reasoning from curvature_field_pu_run.py: a decoder-weight-mass survival ratio and an
    argmax-occupancy count can disagree badly (16/16 "surviving" while occupancy had
    already fallen to 6/8 in prior measurement), so both are always reported, never one
    standing in for the other."""
    assignment = model.chart_probs(z).argmax(dim=1)
    distinct, counts = torch.unique(assignment, return_counts=True)
    per_chart = {int(c): int(n) for c, n in zip(distinct.tolist(), counts.tolist())}
    return {"n_distinct_charts": int(distinct.numel()), "per_chart_counts": per_chart}


def _r_cycle_diagnostic(
    model: "cae.ChartAutoEncoder",
    x_holdout: torch.Tensor,
    p_holdout_np: np.ndarray,
    surviving_indices: Sequence[int],
    p_min: float,
    min_points: int,
) -> Dict[str, Any]:
    """eq. 8's chart-transition cycle residual, on the T2 overlap set. Calls
    ``cae.select_overlap_pairs`` then ``cae.r_cycle`` exactly as ``cae_evaluate_run.py``
    already does (grouped per (alpha, beta) pair, batched through ``r_cycle``) -- neither
    function is reimplemented here, only imported and called; cae.py is not edited. Returns
    ``{"available": False, "reason": ...}`` rather than raising if the pre-registered
    overlap set is under-populated (``cae.select_overlap_pairs`` raises ``ValueError`` in
    that case) -- an unavailable R_cycle is itself informative and must not crash the run."""
    try:
        overlap = cae.select_overlap_pairs(p_holdout_np, p_min, min_points, surviving_indices)
    except ValueError as exc:
        return {"available": False, "reason": str(exc), "p_min": float(p_min), "min_points": int(min_points)}

    rows, alpha, beta = overlap["rows"], overlap["alpha"], overlap["beta"]
    pair_groups: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(rows)):
        key = (int(alpha[i]), int(beta[i]))
        pair_groups.setdefault(key, []).append(i)

    x_overlap = x_holdout[torch.from_numpy(rows).to(x_holdout.device)]
    vals = np.empty(len(rows), dtype=np.float64)
    with torch.no_grad():
        for (a, b), local_positions in pair_groups.items():
            idx_t = torch.tensor(local_positions, dtype=torch.long, device=x_holdout.device)
            xb = x_overlap[idx_t]
            r = cae.r_cycle(model, xb, a, b)
            vals[local_positions] = r.detach().cpu().numpy()

    return {
        "available": True,
        "n_pairs": int(len(rows)),
        "n_distinct_chart_pairs": int(len(pair_groups)),
        "p_min": float(p_min),
        "min_points": int(min_points),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(vals.max()),
    }


# =============================================================================================
# One seed's cell: train, assert convergence, measure.
# =============================================================================================


def run_seed_cell(
    seed: int,
    n_charts: int,
    chart_dim: int,
    embed_dim: int,
    epochs: int,
    unfaithful_samples: int,
    x_train32: torch.Tensor,
    x_holdout32: torch.Tensor,
    device: torch.device,
    overlap_p_min: float,
    overlap_min_points: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model = build_cae(n_charts, chart_dim, embed_dim, device=device)
    cfg = _protocol_cfg(seed, n_charts, epochs)

    t0 = time.monotonic()
    fit = cae.train_cae(model, x_train32, cfg)
    train_wallclock_s = time.monotonic() - t0

    if fit["epochs_run"] != epochs or fit["early_stopped"] or fit["wallclock_truncated"]:
        raise RuntimeError(
            f"cae_faithfulness_run: seed={seed} did not train to the full requested budget "
            f"(epochs_run={fit['epochs_run']} vs max_epochs={epochs}, "
            f"early_stopped={fit['early_stopped']}, wallclock_truncated="
            f"{fit['wallclock_truncated']}) -- refusing to report faithfulness numbers for "
            "an unconverged model. This should be structurally impossible given "
            "early_stop_patience=max_epochs+1 and wallclock_ceiling_s=inf; if it fires, "
            "something upstream of this runner has changed."
        )

    # Evaluated in float32, matching this milestone's established convention
    # (cae_evaluate_run.py's model_main/x_train_t stay float32 through unfaithfulness_coverage
    # and r_cycle -- neither needs float64, unlike chart_curvature's second-derivative work,
    # which _assert_float64 enforces elsewhere and is untouched by this runner). _supnorm_stats
    # upgrades to float64 internally, in numpy, only for the reported statistics themselves.
    model.eval()
    with torch.no_grad():
        recon_train = model.reconstruct(x_train32)
        recon_holdout = model.reconstruct(x_holdout32)
        z_holdout = model.encode(x_holdout32)
        p_holdout = model.chart_probs(z_holdout)

    supnorm_train = _supnorm_stats(x_train32, recon_train)
    supnorm_holdout = _supnorm_stats(x_holdout32, recon_holdout)

    survival = cae.chart_survival(model, PRUNE_TOL)
    occupancy = _occupancy_diagnostic(model, z_holdout)

    unfaithful = cae.unfaithfulness_coverage(
        model, x_train32, survival["surviving_indices"], unfaithful_samples, seed=seed
    )

    r_cycle_stats = _r_cycle_diagnostic(
        model,
        x_holdout32,
        p_holdout.detach().cpu().numpy(),
        survival["surviving_indices"],
        overlap_p_min,
        overlap_min_points,
    )

    return {
        "kind": "faithfulness_cell",
        "seed": seed,
        "n_charts": n_charts,
        "chart_dim": chart_dim,
        "embed_dim": embed_dim,
        "epochs": epochs,
        "epochs_run": fit["epochs_run"],
        "early_stopped": fit["early_stopped"],
        "wallclock_truncated": fit["wallclock_truncated"],
        "train_wallclock_s": train_wallclock_s,
        "cfg": fit["cfg"],
        "loss_curve": fit["history"],
        "n_train": int(x_train32.shape[0]),
        "n_holdout": int(x_holdout32.shape[0]),
        "split_seed": PU_SPLIT_SEED,
        "holdout_fraction": PU_HOLDOUT_FRACTION,
        "supnorm_reconstruction": {"train": supnorm_train, "holdout": supnorm_holdout},
        "unfaithfulness_coverage": unfaithful,
        "r_cycle": r_cycle_stats,
        "chart_survival": {
            "n_charts_initial": survival["n_charts_initial"],
            "n_charts_surviving": survival["n_charts_surviving"],
            "surviving_indices": survival["surviving_indices"],
            "pruned_indices": survival["pruned_indices"],
            "chart_logmass": survival["chart_logmass"].tolist(),
            "chart_mass_ratio": survival["chart_mass_ratio"].tolist(),
            "prune_tol": survival["prune_tol"],
        },
        "argmax_occupancy": occupancy,
        "device": str(device),
        "torch_version": torch.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================================
# Append-only JSONL record: resumability.
# =============================================================================================


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_completed(record_path: Path) -> Dict[str, Dict[str, Any]]:
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
        f.write(json.dumps(_to_jsonable(record)) + "\n")


def _default_record_path() -> Path:
    return cache.cache_path(RECORD_STEM, "jsonl")


def _config_id(seed: int, args: argparse.Namespace) -> str:
    return (
        f"seed{seed}_nc{args.n_charts}_cd{args.chart_dim}_ed{args.embed_dim}"
        f"_ep{args.epochs}_us{args.unfaithful_samples}"
    )


# =============================================================================================
# Smoke, dry-run, and the real multi-seed run.
# =============================================================================================


def run_smoke(
    x_train32: torch.Tensor, x_holdout32: torch.Tensor,
    record_path: Path, device: torch.device,
) -> None:
    _banner("SMOKE -- one deliberately tiny cell, proving the wiring end to end")
    x_train32_s = x_train32[:SMOKE_TRAIN_N]
    x_holdout32_s = x_holdout32[:SMOKE_HOLDOUT_N]

    rec = run_seed_cell(
        seed=SMOKE_SEED,
        n_charts=SMOKE_N_CHARTS,
        chart_dim=DEFAULT_CHART_DIM,
        embed_dim=DEFAULT_EMBED_DIM,
        epochs=SMOKE_MAX_EPOCHS,
        unfaithful_samples=SMOKE_UNFAITHFUL_SAMPLES,
        x_train32=x_train32_s,
        x_holdout32=x_holdout32_s,
        device=device,
        overlap_p_min=SMOKE_OVERLAP_P_MIN,
        overlap_min_points=SMOKE_OVERLAP_MIN_POINTS,
    )
    rec["kind"] = "smoke_cell"
    rec["config_id"] = "smoke"
    append_record(record_path, rec)

    print("SMOKE TALLY:")
    print(f"  train_wallclock_s={rec['train_wallclock_s']:.2f}  epochs_run={rec['epochs_run']}")
    sn_h = rec["supnorm_reconstruction"]["holdout"]
    print(f"  supnorm_reconstruction.holdout: mean={sn_h['mean']:.6g} p99={sn_h['p99']:.6g} max={sn_h['max']:.6g}")
    uf = rec["unfaithfulness_coverage"]
    print(
        f"  unfaithfulness={uf['unfaithfulness']:.6g} coverage={uf['coverage']:.4f} "
        f"n_distinct={uf['n_distinct']} n_samples={uf['n_samples']}"
    )
    rc = rec["r_cycle"]
    if rc["available"]:
        print(f"  r_cycle: n_pairs={rc['n_pairs']} mean={rc['mean']:.6g} max={rc['max']:.6g}")
    else:
        print(f"  r_cycle: unavailable ({rc['reason']})")
    cs = rec["chart_survival"]
    print(f"  chart_survival: {cs['n_charts_surviving']}/{cs['n_charts_initial']} surviving")
    occ = rec["argmax_occupancy"]
    print(f"  argmax_occupancy: n_distinct_charts={occ['n_distinct_charts']}")
    print("SMOKE: exit 0.")


def print_dry_run_report(args: argparse.Namespace, record_path: Path, device: torch.device) -> None:
    _banner("cae_faithfulness_run.py -- DRY RUN")
    print("NOT a comparison: one CAE config, trained once per seed, measured. No plain")
    print("autoencoder, no VAE, no sweep, no A/B arm.")
    print()
    print(f"record_path={record_path}")
    print(f"device={device}  torch_version={torch.__version__}")
    print()
    print("Architecture:")
    print(f"  chart_dim={args.chart_dim}  embed_dim={args.embed_dim}  n_charts={args.n_charts} "
          f"(over-specified; a-posteriori pruning via cae.chart_survival decides the count)")
    print(f"  hidden_width={HIDDEN_WIDTH}  activation={ACTIVATION}")
    print()
    print("Protocol (paper's own stated values where the brief names them):")
    print(f"  lr={ADAM_LR}  batch={BATCH}  lip_weight={LIP_WEIGHT}  weight_decay={WEIGHT_DECAY}")
    print(f"  max_epochs={args.epochs}  early_stop_patience={args.epochs + 1} (structurally inert)")
    print(f"  wallclock_ceiling_s=inf (disabled)  fps_pretrain_epochs={FPS_PRETRAIN_EPOCHS}")
    print(f"  asserted after training: epochs_run == max_epochs, else refuse to report")
    print()
    print("Data:")
    print(f"  subsample={SUBSAMPLE_STEM}  split_seed={PU_SPLIT_SEED}  "
          f"holdout_fraction={PU_HOLDOUT_FRACTION}")
    print()
    print(f"Seeds ({len(args.seeds)}): {list(args.seeds)}")
    print()
    print("Measurements per seed: sup-norm reconstruction (train+holdout, mean/median/p95/"
          "p99/max), unfaithfulness+coverage (n_samples="
          f"{args.unfaithful_samples}), R_cycle distribution (eq. 8, T2 overlap set, "
          f"p_min={OVERLAP_P_MIN}, min_points={OVERLAP_MIN_POINTS}), chart survival AND "
          "argmax occupancy (both, never collapsed), the full per-epoch loss curve.")
    print()
    print("--dry-run: nothing executed, nothing written.")


def print_seed_summary(records: List[Dict[str, Any]]) -> None:
    """The spread across seeds -- the reported unit, never a single draw."""
    _banner(f"SEED SUMMARY -- {len(records)} seed(s) on record")

    def _spread(vals: List[float]) -> str:
        a = np.asarray(vals, dtype=np.float64)
        return f"min={a.min():.6g} median={np.median(a):.6g} max={a.max():.6g}"

    sn_holdout_mean = [r["supnorm_reconstruction"]["holdout"]["mean"] for r in records]
    sn_holdout_p99 = [r["supnorm_reconstruction"]["holdout"]["p99"] for r in records]
    unfaithful = [r["unfaithfulness_coverage"]["unfaithfulness"] for r in records]
    coverage = [r["unfaithfulness_coverage"]["coverage"] for r in records]
    surviving = [r["chart_survival"]["n_charts_surviving"] for r in records]
    occ = [r["argmax_occupancy"]["n_distinct_charts"] for r in records]

    print(f"  supnorm_reconstruction.holdout.mean: {_spread(sn_holdout_mean)}")
    print(f"  supnorm_reconstruction.holdout.p99:  {_spread(sn_holdout_p99)}")
    print(f"  unfaithfulness:                      {_spread(unfaithful)}")
    print(f"  coverage:                             {_spread(coverage)}")
    print(f"  n_charts_surviving:                   {_spread([float(v) for v in surviving])}")
    print(f"  argmax n_distinct_charts:              {_spread([float(v) for v in occ])}")

    r_cycle_available = [r["r_cycle"] for r in records if r["r_cycle"]["available"]]
    if r_cycle_available:
        r_cycle_mean = [r["mean"] for r in r_cycle_available]
        print(
            f"  r_cycle.mean ({len(r_cycle_available)}/{len(records)} seeds with an "
            f"available overlap set): {_spread(r_cycle_mean)}"
        )
    else:
        print("  r_cycle: unavailable in every seed on record")


def run_all_seeds(
    args: argparse.Namespace,
    x_train32: torch.Tensor,
    x_holdout32: torch.Tensor,
    record_path: Path,
    device: torch.device,
) -> None:
    completed = load_completed(record_path) if args.resume else {}
    if args.resume:
        print(f"--resume: {len(completed)} record(s) already on disk, matched by config_id.")
        for config_id, rec in completed.items():
            rec_device = rec.get("device", "cpu")
            if rec_device != str(device):
                raise ValueError(
                    f"--resume: record_path={record_path} already contains cell "
                    f"{config_id!r} recorded at device={rec_device!r}, but this invocation "
                    f"requested device={str(device)!r} -- mixing devices within one run "
                    "mixes two different RNG draws. Use a distinct --record-path instead."
                )

    for seed in args.seeds:
        config_id = _config_id(seed, args)
        if args.resume and config_id in completed:
            print(f"  [skip, resumed] {config_id}")
            continue
        print(f"[{config_id}] fitting (max_epochs={args.epochs}, n_charts={args.n_charts})...")
        t0 = time.monotonic()
        rec = run_seed_cell(
            seed=seed,
            n_charts=args.n_charts,
            chart_dim=args.chart_dim,
            embed_dim=args.embed_dim,
            epochs=args.epochs,
            unfaithful_samples=args.unfaithful_samples,
            x_train32=x_train32,
            x_holdout32=x_holdout32,
            device=device,
            overlap_p_min=OVERLAP_P_MIN,
            overlap_min_points=OVERLAP_MIN_POINTS,
        )
        rec["config_id"] = config_id
        append_record(record_path, rec)
        print(f"[{config_id}] done in {time.monotonic() - t0:.1f}s (epochs_run={rec['epochs_run']})")

    completed_after = load_completed(record_path)
    wanted_ids = {_config_id(seed, args) for seed in args.seeds}
    present = [rec for cid, rec in completed_after.items() if cid in wanted_ids and rec.get("kind") == "faithfulness_cell"]
    if len(present) == len(args.seeds):
        print_seed_summary(present)
    else:
        print(f"{len(present)} of {len(args.seeds)} requested seeds present; summary printed once all are.")


# =============================================================================================
# main
# =============================================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Max training epochs, with no early stopping (default: {DEFAULT_EPOCHS}). "
        "The runner asserts epochs_run == this value after training.",
    )
    parser.add_argument(
        "--n-charts", type=int, default=DEFAULT_N_CHARTS,
        help=f"Initial chart count, over-specified (default: {DEFAULT_N_CHARTS}, the paper's "
        "own stated maximum); a-posteriori pruning decides the surviving count.",
    )
    parser.add_argument("--chart-dim", type=int, default=DEFAULT_CHART_DIM, help=f"Chart dimension (default: {DEFAULT_CHART_DIM}, D-11).")
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM, help=f"Embedding dimension (default: {DEFAULT_EMBED_DIM}).")
    parser.add_argument(
        "--unfaithful-samples", type=int, default=DEFAULT_UNFAITHFUL_SAMPLES,
        help=f"Sample count for eqs. 20-21 (default: {DEFAULT_UNFAITHFUL_SAMPLES}).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help=f"One CAE fit per seed; the spread across seeds is the reported unit "
        f"(default: {list(DEFAULT_SEEDS)}).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="'cpu' (default), 'cuda', or 'cuda:N'. Same three caveats as "
        "curvature_field_pu_run.py's --device (03-07-SUPPLEMENT-01): GPU runs are a "
        "different RNG draw, not a CPU reproduction; float64 (this runner's diagnostics) "
        "can be slower on a consumer GPU than on CPU; do not mix devices within one run.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip any seed's config_id already present in the record file.")
    parser.add_argument("--record-path", type=str, default=None, help=f"Override the record path (default: notebooks/.cache/{RECORD_STEM}.jsonl).")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned configuration; execute and write nothing.")
    parser.add_argument("--smoke", action="store_true", help="One deliberately tiny cell end to end, under a minute on CPU. This runner's automated <verify>.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    record_path = Path(args.record_path) if args.record_path else _default_record_path()
    device = resolve_device(args.device)

    _banner("cae_faithfulness_run.py -- CAE faithfulness, one model, measured in the paper's own terms")
    print(f"record_path={record_path}")
    print(f"device={device}  torch_version={torch.__version__}")
    if device.type == "cuda":
        print(
            "CAVEAT: GPU runs do NOT reproduce CPU runs bit-for-bit (CUDA RNG != CPU RNG); "
            "this is a different draw, recorded as such via each record's 'device' field."
        )

    if args.dry_run:
        print_dry_run_report(args, record_path, device)
        return

    x_full_np = _load_subsample()
    x_full32 = torch.tensor(x_full_np, dtype=torch.float32).to(device)
    train_idx, holdout_idx = _split(x_full_np.shape[0], PU_SPLIT_SEED, PU_HOLDOUT_FRACTION)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    holdout_idx_t = torch.as_tensor(holdout_idx, dtype=torch.long, device=device)
    x_train32 = x_full32[train_idx_t]
    x_holdout32 = x_full32[holdout_idx_t]

    if args.smoke:
        run_smoke(x_train32, x_holdout32, record_path, device)
        return

    run_all_seeds(args, x_train32, x_holdout32, record_path, device)


if __name__ == "__main__":
    main()
