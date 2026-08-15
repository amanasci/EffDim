"""
Phase 3 Plan 07: the PU sweep runner -- three ``n_charts`` values by three seeds, nine
fresh CAE fits at ``chart_dim = 20``, ``D = 768``, each carrying the four D-07 diagnostics.

Nothing here needs ground truth. PU has no analytic ``H``, so there is no gate on this
side -- only a diagnostics table to read. This plan (03-07) builds and proves the
instrument; plan 03-08 runs the grid.

Three things are settled here, in this file's own words, before any PU number exists.

**1. The working dimension.** ``chart_dim = 20``, NOT ``d_frozen = 5``. Justified against
three independent geometric estimates on the frozen Phase 1 subsample: TwoNN gives 19.5;
local PCA gives a median of 25.0 (mean 24.5, std 2.0, 5-95% range 21-28, no neighbourhood
above 29); a median of 18 across Phase 1's eight geometric estimators. Three estimates
cluster at 18-25. ``d_frozen = 5`` -- the residual-curve elbow -- is explicitly REJECTED,
not silently dropped: ``02-FINDINGS.md`` section 6.4 records that with 41% negative
eigenvalue mass the Tenenbaum residual curve saturates early, so its elbow measured the
flatness failure rather than the geometry, and it is the outlier against the other three
estimates. A module-level guard below raises ``ValueError`` naming D-11 if ``PU_CHART_DIM``
is ever anything other than 20, so no future edit can quietly reach 5 through this file.

**2. The gate override.** No PASS exists anywhere in this milestone (02, 02.2, 02.4, and
02.5 stage 1 are all FAIL, and Phase 2's stage is on hold with the CAE carried forward by
deliberate user override, not by a cleared gate). Phase 3 runs on that override, and the
consequence travels with every number this runner prints: a curvature field decoded from an
unvalidated parameterization conflates real curvature with parameterization damage, and the
synthetic control (``synthetic_controls.py``) provably cannot detect that confound from the
PU side.

**3. Fresh fits only, per D-10.** The sealed 02.2 PU fit (``fit_key = 43cf438bc944c509``)
plays NO part here -- no artifact of it is loaded, reused, or warm-started, and it is not
carried in this runner's own sweep as a labelled row. The architecture matches it
(``embed_dim = 40``, three hidden layers of width 250, SiLU) purely so the fits are
comparable in shape, but every weight this runner reports is trained here, under this
runner's own seeds. For the record, that sealed fit was ``D_CHART=20``, ``L_EMBED=40``,
``N_CHARTS_INIT=16`` with all 16 surviving pruning, SiLU, width 250, 8,000 train / 2,000
holdout at ``SPLIT_SEED=20260803``.

**The sweep set is chosen on PU's own terms.** ``PU_N_CHARTS_SWEEP = (4, 8, 16)`` spans two
octaves of atlas granularity and includes 16 for architectural continuity with the sealed
02.2 fit's ``N_CHARTS_INIT`` -- it is NOT derived from the Swiss roll. D-06 forbids the
roll's winning ``n_charts`` from selecting a PU hyperparameter, and nothing measured on a
2-D sheet bounds a ``d ~ 20`` manifold's atlas.

**The four D-07 diagnostics, computed separately, never collapsed.**

1. Metric conditioning -- max/median/p90/p99 of ``cond(g)`` plus its histogram, from
   ``chart_curvature.chart_curvature_field`` on the holdout rows.
2. Argmax chart occupancy -- ``torch.unique(model.chart_probs(z).argmax(dim=1))`` value
   counts on the holdout. This deliberately does NOT use ``cae.chart_survival``: that
   helper thresholds a ratio to the largest chart, so decoupled decay shrinks live and dead
   charts together and cancels -- a quick task measured it returning 8 of 8 in all 24 fits
   while argmax occupancy had already fallen to 6 of 8. Argmax occupancy is the same
   quantity ``chart_curvature_field`` computes internally as its own ``assignment``.
3. Held-out reconstruction -- ``cae.reconstruction_stats``' aggregate (``mse_per_dim``,
   ``mse_total``) AND its per-output-dimension distribution (``dim_mse_mean``,
   ``dim_mse_median``, ``dim_mse_p95``, ``dim_mse_max``), so a good average cannot hide a
   subset failure.
4. Persistent homology at H0 and H1 only -- ``persistence_probe.PH_MAXDIM = 1`` structurally
   excludes H2 and no code path here raises it. H2 is excluded deliberately: no power
   analysis exists for it anywhere in this milestone, so every ``beta_2 = 0`` would be an
   unbounded null, and it is the term that exhausted 02.7-07's compute budget. **PU has no
   intrinsic reference** -- there is no known parameterization of this cloud, which is the
   reason this milestone exists -- so ``references["ambient"]`` and
   ``references["intrinsic"]`` are BOTH set to the same ambient PU subsample. The four
   ``*|intrinsic|*`` columns are therefore degenerate duplicates of their ``*|ambient|*``
   counterparts and are excluded from selection. The PRIMARY read is
   ``latent|ambient|H0|bottleneck_norm`` and ``latent|ambient|H1|bottleneck_norm`` -- does
   the atlas's latent space preserve the topology of the data as given. The matching
   ``wasserstein_norm`` cells and the four ``decoder_image|ambient|*`` cells are reported as
   context (the decoder-image cells answer a reconstruction-quality question the separate
   reconstruction diagnostic already covers). The PH subsample size (300 rows) and the
   prescale policy (``prescale=True``) are REUSED verbatim from the values already
   established at PU/D=768 scale in this milestone -- ``template_benchmark_run.py``'s
   ``N_PH = 300`` (justified by ``ph_budget_calibration_run.py``'s measured cost envelope)
   and ``decoder_substrate_ph_screen_run.py``'s ``PRESCALE_CLOUDS = True`` -- rather than
   choosing new values here.

**The selection rule, declared before any grid cell runs.** Print the full 3x3 table --
every diagnostic's per-seed value and its across-seed median -- then (a) DISQUALIFY any
``n_charts`` whose median distinct-chart occupancy is below 2, since a single-chart atlas is
not an atlas; (b) among survivors, rank LEXICOGRAPHICALLY by median max ``cond(g)``
ascending, treating two configs as TIED on that axis when their medians are within a factor
of 2 of each other and passing the decision to the next axis; then by median holdout
``mse_per_dim`` ascending; then by median ``latent|ambient|H1|bottleneck_norm`` ascending;
(c) form NO weighted composite and print no single score. This matches this milestone's
consistent refusal to collapse multi-axis evidence -- ``curvature_fidelity_report``'s three
separate axes, ``persistence_probe``'s sixteen uncollapsed cells -- and a composite would be
a departure that would have to be named as one.

**D-12's escalation trigger is computed but never acted on here.** One matched
``cae.PlainAutoEncoder(768, PU_CHART_DIM, hidden=(250,250,250), activation="silu")`` control
is fit per seed under the same protocol -- ``PU_CHART_DIM`` (20), the CAE's own effective
bottleneck, NOT ``PU_EMBED_DIM`` (40): 03-08-DEFECTS-01.md defect 1 found the control
originally built at 40 solved a strictly easier reconstruction problem at double the CAE's
narrowest-point capacity. An OPTIONAL, separately-labelled 40-dim capacity-reference
control (``PU_CONTROL_CAPACITY_DIM``) is also fit per seed and reported on its own row, but
is never read by the trigger below. The trigger fires when the best ``d=20`` CAE config
loses to the MATCHED (20-dim) control on BOTH held-out reconstruction AND PH H0/H1
agreement. This runner computes and prints whether it fires; it does not escalate to a
``d`` sweep -- that is plan 03-08's decision. Known limitation, carried from ``02.5-09``:
reconstruction and topology do not predict curvature quality (seeds 0 and 3 differed by
0.29 percentage points of reconstruction error and 0.49 of Spearman), so this trigger is
EXPECTED to fire.

Invoke:
    .venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --dry-run
    .venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --smoke
    .venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --timing-probe --pu-n 200
    .venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --select-only
    .venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --resume
    .venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --device cuda --timing-probe
    .venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --device cuda --resume

**03-07-SUPPLEMENT-01: opt-in GPU support, three caveats stated here and in ``--device``'s
help text.** (1) GPU runs do NOT reproduce CPU runs bit-for-bit -- CUDA RNG differs from CPU
RNG, so ``torch.manual_seed(seed)`` draws a different model on a different device; a GPU cell
is a different draw, not a reproduction, which is why every record now carries a ``device``
field. (2) float64 throughput is hardware-dependent -- curvature is float64-only
(``chart_curvature._assert_float64`` is never relaxed), and float64 on a consumer GPU can run
at 1/32-1/64 of float32 (vs ~1/2 on a data-center GPU), so the curvature term may be SLOWER on
a consumer GPU than on CPU even though training (float32, the dominant nine-cell term at
~16,100-16,200s vs ~4,000-4,040s for curvature) should be faster. Run ``--timing-probe
--device cuda`` first and compare against a CPU probe before committing to the full grid. (3)
Do not mix devices within one grid -- all nine cells (plus the three D-12 controls) must run
on the same device, or the three-seed spread mixes two different draws; ``--resume`` refuses
to continue a record whose device disagrees with the one requested.
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import cache, cae, chart_curvature, persistence_probe  # noqa: E402

# =============================================================================================
# Constants -- the working dimension, the architecture, the protocol, the sweep.
# =============================================================================================

PU_CHART_DIM = 20
"""D-11: 20, never 5. See this module's own docstring section 1 for the full justification
(TwoNN 19.5, local-PCA median 25.0, median-of-eight-estimators 18) and the reason
``d_frozen = 5`` is rejected rather than inherited."""

PU_EMBED_DIM = 40
PU_CONTROL_CAPACITY_DIM = PU_EMBED_DIM
"""03-08-DEFECTS-01.md defect 1: the OPTIONAL, separately-labelled 40-dim capacity-reference
control (``config_id`` prefix ``control40_``). Genuinely informative as its own row (a plain
autoencoder at the CAE's total embed_dim capacity) but MUST NOT be read as the D-12 matched
comparison -- that control is built at ``PU_CHART_DIM`` (20), the CAE's actual bottleneck."""
PU_HIDDEN_WIDTH = 250
PU_ACTIVATION = "silu"
PU_N_CHARTS_SWEEP = (4, 8, 16)
"""Chosen on PU's own terms -- two octaves of atlas granularity, 16 for continuity with the
sealed 02.2 fit's N_CHARTS_INIT. NOT derived from the Swiss roll sweep (D-06)."""
PU_SEEDS = (20260813, 20260814, 20260815)
PU_SPLIT_SEED = 20260813
PU_HOLDOUT_FRACTION = 0.2

# Training protocol. 03-08-DEFECTS-01.md defect 2: MAX_EPOCHS=40/EARLY_STOP_PATIENCE=5/
# LIP_WEIGHT=1e-2 were carried across from 02.2's own sealed-fit protocol
# (notebooks/diagnostics/cae_train_run.py) "for comparability", not independently justified
# at PU scale -- and 02.2's own findings (02.2-06-SUMMARY.md) named "training budget/epochs,
# loss weighting, the Lipschitz penalty schedule" as unresolved candidate axes for a future
# iteration phase that never ran. Measured consequence at PU scale: `train_cae` early-stops
# on TOTAL loss (reconstruction + cross-entropy + the Lipschitz penalty), and with
# LIP_WEIGHT 10x heavier than the roll's own protocol, 5 of 9 grid cells stopped at
# epochs_run=7 -- a plateau in the penalty term, not in reconstruction, ended training.
# FIX: EARLY_STOP_PATIENCE and LIP_WEIGHT are realigned to the Swiss roll runner's own
# established protocol (300/25/1e-3 in swiss_roll_curvature_sweep_run.py's BASE_CFG) on
# exactly the two parameters that cause the truncation; MAX_EPOCHS is deliberately NOT
# raised to the roll's 300 -- see 03-08-SUPPLEMENT-02.md for the full reasoning and the
# expected wall-clock consequence. FPS_PRETRAIN_EPOCHS (5 vs the roll's 20) is left
# unchanged: the defect's own measured mechanism (total-loss early stopping under a
# tight patience and a heavy Lipschitz term) does not implicate the pre-training stage,
# and CLAUDE.md's keep-it-simple rule argues against widening the fix beyond the
# mechanism that was actually diagnosed.
ADAM_LR = 3e-4
WEIGHT_DECAY = 1e-4
BATCH = 64
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 25
EARLY_STOP_MIN_DELTA = 1e-4
LIP_WEIGHT = 1e-3
LIP_EVERY_N_STEPS = 1
FPS_PRETRAIN_EPOCHS = 5
WALLCLOCK_CEILING_S = 7200

SUBSAMPLE_STEM = "subsample_20260729_a79b3460b838fd0a"
"""The frozen Phase 1 10,000-row PU subsample every prior fit in this milestone was trained
and split against. Gitignored, irreproducible here -- see :func:`_load_subsample`."""

RECORD_STEM = "03_curvature_field_pu"

AMBIENT_DIM = 768

# PH subsample size and prescale policy at PU/D=768 scale -- REUSED, not re-derived.
# grep anchor: persistence_probe.PH_MAXDIM = 1 structurally excludes H2; no line below sets
# a maxdim above 1, and readout_matrix (which this file calls) is the only PH entry point.
N_PH_PU = 300
"""Matches ``template_benchmark_run.py``'s ``N_PH = 300``, the PH subsample size already
established and cost-justified at D=768 by ``ph_budget_calibration_run.py``."""
PRESCALE_PU = "median_distance"
"""03-08-DEFECTS-01.md defect 3 fix. Originally ``True`` (matching
``decoder_substrate_ph_screen_run.py``'s ``PRESCALE_CLOUDS = True``), which multiplies each
cloud by ``topoae.latent_unit_scale`` -- mean per-dimension variance 1, a scale that still
grows as ``sqrt(d)``, so comparing the 40-dim ``latent`` cloud against the 768-dim ``ambient``
reference (``sqrt(768/40) ~ 4.4x``) saturated by construction: every ``latent|ambient|*``
bottleneck cell measured in the nine-cell grid was pinned at exactly its saturation ceiling
(0.5, 12 of 12 records including controls), i.e. it measured ambient dimension, not topology.
``"median_distance"`` (``persistence_probe.cloud_distance_matrix``'s new opt-in mode) instead
fixes each cloud's own median pairwise distance to 1 -- a dimension-invariant distance-scale
statistic, verified against 03-08-DEFECTS-01.md's identical-structure-at-two-dimensions test
to give a non-saturated, near-zero bottleneck where the old default gave the saturation
ceiling exactly. ``persistence_probe``'s default (``True``/``False``) behaviour is unchanged
by this fix -- this is a PU-runner-local opt-in, not a change to any other call site."""

# Smoke-mode sizes -- deliberately tiny, to prove the wiring, never the real grid.
SMOKE_TRAIN_N = 400
SMOKE_HOLDOUT_N = 100
SMOKE_PH_N = 50
SMOKE_N_CHARTS = 2
SMOKE_MAX_EPOCHS = 2
SMOKE_SEED = PU_SEEDS[0]

# Timing-probe constants.
DEFAULT_TIMING_PROBE_PU_N = 200
TIMING_PROBE_TRAIN_STEPS = 20
WALLCLOCK_ENVELOPE_HOURS_MAX = 5.0
"""D-13's budget envelope (3-5 hours for the 9-fit grid). The halt-and-report branch compares
the projected nine-cell total against the UPPER end of that envelope."""
FORWARD_REVERSE_OPCOUNT_CEILING = 38.0
"""D-08's operation-count ceiling for the forward/reverse curvature speedup, against which
the first measured ratio at this scale (d=20, D=768) is compared."""

VMAP_CHUNK = chart_curvature.VMAP_CHUNK  # 32, imported rather than re-declared

if PU_CHART_DIM != 20:
    raise ValueError(
        "curvature_field_pu_run: PU_CHART_DIM must be 20 (D-11). d_frozen = 5 -- the "
        "residual-curve elbow from Phase 2 -- is explicitly rejected: 02-FINDINGS.md section "
        "6.4 records that at 41% negative eigenvalue mass the Tenenbaum residual curve "
        "saturates early, so the elbow measured the flatness failure, not the geometry, and "
        "it is the outlier against TwoNN's 19.5, local-PCA's median 25.0, and the "
        "median-of-eight-estimators' 18. No code path in this file may reach chart_dim=5."
    )


def resolve_device(device_str: str) -> torch.device:
    """Parse and validate a ``--device`` CLI argument ('cpu', 'cuda', or 'cuda:N'). Fails
    with an actionable message, naming the installed torch build, if ``cuda`` is requested
    but ``torch.cuda.is_available()`` is False -- 03-07-SUPPLEMENT-01's setup notes give the
    exact install command."""
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
            "See .planning/phases/03-decoder-curvature-field/03-07-SUPPLEMENT-01.md for the "
            "full setup walkthrough."
        )
    return device


# =============================================================================================
# Printable restatements -- used by --dry-run so the justification, override, PH-cell choice
# and selection rule are on the record before any PU number is measured, without re-parsing
# this module's own __doc__.
# =============================================================================================

DIMENSION_JUSTIFICATION_TEXT = f"""\
DIMENSION JUSTIFICATION (D-11): PU_CHART_DIM = {PU_CHART_DIM}, not d_frozen = 5.
  TwoNN estimate:            19.5
  local-PCA median:          25.0  (mean 24.5, std 2.0, 5-95% range 21-28)
  median of 8 geometric estimators (Phase 1): 18
  d_frozen = 5 (Tenenbaum residual-curve elbow) is REJECTED: at 41% negative eigenvalue
  mass the residual curve saturates early, so the elbow measured the flatness failure, not
  the geometry (02-FINDINGS.md section 6.4). It is the outlier against the other three."""

GATE_OVERRIDE_TEXT = """\
GATE OVERRIDE: no PASS exists anywhere in this milestone (02, 02.2, 02.4, 02.5 stage 1 are
  all FAIL; the Phase 2 stage is on hold, CAE carried forward by deliberate user override).
  Phase 3 runs on that override. Consequence: a curvature field decoded from an unvalidated
  parameterization conflates real curvature with parameterization damage, and the synthetic
  control cannot detect that confound from the PU side."""

FRESH_FITS_TEXT = """\
FRESH FITS ONLY (D-10): the sealed 02.2 PU fit (fit_key=43cf438bc944c509) plays no part --
  no artifact of it is loaded, reused, or warm-started, and it is not carried as a labelled
  sweep row. Every weight this runner reports is trained here, under this runner's own
  seeds."""

PH_CELL_CHOICE_TEXT = """\
PH CELL CHOICE: PU has no intrinsic reference -- references["ambient"] and
  references["intrinsic"] are both the ambient PU subsample; the four *|intrinsic|* cells
  are degenerate duplicates of *|ambient|* and are excluded from selection. PRIMARY cells:
  latent|ambient|H0|bottleneck_norm and latent|ambient|H1|bottleneck_norm. Matching
  wasserstein_norm cells and the four decoder_image|ambient|* cells are reported as context
  only. PH_MAXDIM=1 excludes H2 structurally; no code path here raises it."""

SELECTION_RULE_TEXT = """\
SELECTION RULE (declared before the grid runs):
  1. Print the full 3x3 table -- every diagnostic's per-seed value and its across-seed
     median.
  2. DISQUALIFY any n_charts whose median distinct-chart occupancy is below 2 -- a
     single-chart atlas is not an atlas.
  3. Among survivors, rank LEXICOGRAPHICALLY:
       (a) median max cond(g), ascending -- two configs are TIED on this axis when their
           medians are within a factor of 2 of each other, in which case the decision
           passes to the next axis;
       (b) median holdout mse_per_dim, ascending;
       (c) median latent|ambient|H1|bottleneck_norm, ascending.
  4. Form NO weighted composite and print no single score."""


def _banner(msg: str) -> None:
    print()
    print("=" * 90)
    print(msg)
    print("=" * 90)


# =============================================================================================
# Data loading -- read-only, halts loudly rather than drawing a different subsample.
# =============================================================================================


def _load_subsample() -> np.ndarray:
    """The frozen 10,000-row ``legacysurvey`` array, read-only. Raises ``FileNotFoundError``
    naming the exact path if the cache stem is absent, following ``derivative_bridge_run.py``'s
    ``load_pu_models`` discipline including its cache-entry-count-unchanged check -- this
    function never trains a replacement subsample."""
    cache_dir = cache.CACHE_DIR
    n_before = sum(1 for _ in cache_dir.iterdir())
    path = cache.cache_path(SUBSAMPLE_STEM, "npz")
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"curvature_field_pu_run: required frozen Phase 1 PU subsample missing or empty: "
            f"{path}. This is the 10,000-row subsample every prior fit in this milestone was "
            f"trained and split against; it is gitignored and irreproducible here. Halting "
            f"rather than drawing a different subsample."
        )
    arrays = dict(np.load(path))
    n_after = sum(1 for _ in cache_dir.iterdir())
    if n_after != n_before:
        raise RuntimeError(
            f"curvature_field_pu_run: notebooks/.cache entry count changed ({n_before} -> "
            f"{n_after}) while loading the subsample -- this load must stay read-only."
        )
    return arrays["legacysurvey"]


def _split(n: int, split_seed: int, holdout_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """One shared 8,000/2,000 split via ``np.random.default_rng(PU_SPLIT_SEED)``, so every
    config in the grid is compared on identical data."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_holdout = int(round(n * holdout_fraction))
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    return train_idx, holdout_idx


# =============================================================================================
# Model builders and the shared protocol cfg.
# =============================================================================================


def build_cae(n_charts: int, device: torch.device = torch.device("cpu")) -> "cae.ChartAutoEncoder":
    """Constructed exactly as before, then moved to ``device`` AFTER construction
    (``model.to(device)``) -- never by passing ``device=`` into the constructor -- so
    ``torch.manual_seed(seed)``'s RNG consumption order is unaffected (03-07-SUPPLEMENT-01,
    hard constraint 2). Default ``cpu``: zero behaviour change."""
    model = cae.ChartAutoEncoder(
        in_dim=AMBIENT_DIM,
        embed_dim=PU_EMBED_DIM,
        chart_dim=PU_CHART_DIM,
        n_charts=n_charts,
        hidden=[PU_HIDDEN_WIDTH, PU_HIDDEN_WIDTH, PU_HIDDEN_WIDTH],
        activation=PU_ACTIVATION,
    )
    return model.to(device)


def build_control(
    latent_dim: int, device: torch.device = torch.device("cpu")
) -> "cae.PlainAutoEncoder":
    """03-08-DEFECTS-01.md defect 1 fix: ``latent_dim`` is now an explicit, required
    argument rather than being hardcoded to ``PU_EMBED_DIM``. The D-12 escalation trigger's
    control MUST be built at ``latent_dim=PU_CHART_DIM`` (20) -- the CAE's own narrowest
    point (``encode -> z (embed_dim=40) -> chart_coords -> chart_dim=20 -> decode``) -- not
    at ``PU_EMBED_DIM`` (40), which was double the CAE's effective bottleneck and so solved
    a strictly easier reconstruction problem. ``PlainAutoEncoder``'s own docstring names the
    question this control exists to answer: "whether the atlas bought anything over one
    chart at matched capacity" -- at 40 vs 20, capacity was not matched. The 40-dim control
    is kept as an OPTIONAL, separately-labelled capacity reference (see
    :data:`PU_CONTROL_CAPACITY_DIM`), never as the D-12 comparison. Same
    construct-then-``.to(device)`` discipline as :func:`build_cae`."""
    model = cae.PlainAutoEncoder(
        AMBIENT_DIM, latent_dim, hidden=(PU_HIDDEN_WIDTH,) * 3, activation=PU_ACTIVATION
    )
    return model.to(device)


def _protocol_cfg(seed: int, n_charts: Optional[int], max_epochs: int = MAX_EPOCHS) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "seed": seed,
        "lr": ADAM_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch": BATCH,
        "max_epochs": max_epochs,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "lip_weight": LIP_WEIGHT,
        "lip_every_n_steps": LIP_EVERY_N_STEPS,
        "fps_pretrain_epochs": FPS_PRETRAIN_EPOCHS,
        "wallclock_ceiling_s": WALLCLOCK_CEILING_S,
    }
    if n_charts is not None:
        cfg["n_charts"] = n_charts
    return cfg


# =============================================================================================
# The four D-07 diagnostics.
# =============================================================================================


def _median(vals: List[float]) -> float:
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _cond_diagnostic(field: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnostic 1 -- metric conditioning. The distribution, not only the extreme."""
    cond = field["metric_condition_number"].detach().cpu().numpy().astype(np.float64)
    hist_counts, hist_edges = np.histogram(cond, bins=20)
    return {
        "max": float(cond.max()),
        "median": float(np.median(cond)),
        "p90": float(np.percentile(cond, 90)),
        "p99": float(np.percentile(cond, 99)),
        "hist_counts": [int(c) for c in hist_counts],
        "hist_edges": [float(e) for e in hist_edges],
    }


def _occupancy_diagnostic(model: Any, z: torch.Tensor) -> Dict[str, Any]:
    """Diagnostic 2 -- argmax chart occupancy. Deliberately NOT cae.chart_survival: that
    helper thresholds a ratio to the largest chart, so decoupled decay shrinks live and dead
    charts together and cancels -- a quick task measured it returning 8 of 8 in all 24 fits
    while argmax occupancy had already fallen to 6 of 8. This reads
    model.chart_probs(z).argmax(dim=1) value counts, the same quantity
    chart_curvature_field computes internally as its own assignment."""
    assignment = model.chart_probs(z).argmax(dim=1)
    distinct, counts = torch.unique(assignment, return_counts=True)
    per_chart = {int(c): int(n) for c, n in zip(distinct.tolist(), counts.tolist())}
    return {"n_distinct_charts": int(distinct.numel()), "per_chart_counts": per_chart}


SMOKE_PH_JITTER_STD = 1e-5
"""Smoke-only numerical stabilizer, never applied to a real grid or control cell.

Measured directly against the smoke config (400 rows, n_charts=2, max_epochs=2): that
deliberately tiny fit is so far from convergence that BOTH its argmax-chart reconstruction
and its own encoder embedding collapse to near-mean, near-constant output across the PH
slice (measured per-dimension variance ~1e-16 to 1e-18, i.e. genuinely indistinguishable
from float64 round-off, not a code bug -- a two-epoch fit on 400 rows through a
768<->250x3<->40<->20 architecture has had on the order of a dozen gradient steps, nowhere
near enough to escape its initialization). ``persistence_probe.cloud_distance_matrix``'s
prescale normalizer -- originally the global isotropic ``1 / sqrt(mean variance)``, now
``PRESCALE_PU``'s ``"median_distance"`` mode (03-08-DEFECTS-01.md defect 3) -- divides by a
scale statistic computed FROM the cloud itself, so on this exact pathology (a near-constant
cloud whose own variance and whose own pairwise distances both collapse toward the float64
noise floor) either normalizer divides by a near-zero denominator and amplifies round-off by
a similarly large factor, and ``persistence_diagram``'s own symmetry check (deliberately
strict, no silent auto-symmetrization) then correctly refuses a matrix that is no longer
symmetric to default tolerance. Real grid and control cells never hit this: they train
under a protocol realigned to avoid premature early stopping (03-08-DEFECTS-01.md defect 2)
and were verified not to collapse this way.
The fix applied ONLY here is a small seeded Gaussian perturbation, several orders of
magnitude above the float64 noise floor and several orders below the ambient data's own
scale, added to the latent and decoder-image clouds before the PH call -- restoring a
distance matrix ``cloud_distance_matrix`` can measure without changing what this smoke
cell is proving (that the wiring end to end -- fit, encode, reconstruct,
persistence_probe.readout_matrix -- executes and returns the expected sixteen cells)."""


def _ph_diagnostic(
    latent: np.ndarray,
    decoder_image: np.ndarray,
    ambient: np.ndarray,
    jitter_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Diagnostic 4 -- persistent homology at H0 and H1 only (persistence_probe.PH_MAXDIM=1
    structurally excludes H2; no line here raises it). PU has no intrinsic reference, so
    both references map to the same ambient cloud (see PH_CELL_CHOICE_TEXT).

    ``jitter_seed``: smoke-only, see :data:`SMOKE_PH_JITTER_STD`. ``None`` (the default,
    used by every real grid and control cell) applies no perturbation whatsoever."""
    if jitter_seed is not None:
        rng = np.random.default_rng(jitter_seed)
        latent = latent + rng.normal(scale=SMOKE_PH_JITTER_STD, size=latent.shape)
        decoder_image = decoder_image + rng.normal(scale=SMOKE_PH_JITTER_STD, size=decoder_image.shape)
    spaces = {"latent": latent, "decoder_image": decoder_image}
    references = {"ambient": ambient, "intrinsic": ambient}
    return persistence_probe.readout_matrix(spaces, references, prescale=PRESCALE_PU)


# =============================================================================================
# One CAE grid cell and one D-12 control cell.
# =============================================================================================


def _run_cae_cell(
    n_charts: int,
    seed: int,
    x_train32: torch.Tensor,
    x_holdout64: torch.Tensor,
    x_ph64: torch.Tensor,
    max_epochs: int,
    ph_jitter_seed: Optional[int] = None,
) -> Dict[str, Any]:
    # Device is derived from the already-placed input tensor rather than threaded as its
    # own parameter (main() moves x_train32/x_holdout64/x_ph64 to the target device once,
    # before any cell runs).
    device = x_train32.device
    torch.manual_seed(seed)
    model = build_cae(n_charts, device=device)
    cfg = _protocol_cfg(seed, n_charts, max_epochs=max_epochs)
    t0 = time.monotonic()
    fit = cae.train_cae(model, x_train32, cfg)
    train_wallclock_s = time.monotonic() - t0

    model.eval().double()
    with torch.no_grad():
        z_holdout = model.encode(x_holdout64)
        recon_holdout = model.reconstruct(x_holdout64)
        z_ph = model.encode(x_ph64)
        decoder_image_ph = model.reconstruct(x_ph64)

    recon_stats = cae.reconstruction_stats(x_holdout64, recon_holdout)
    occ = _occupancy_diagnostic(model, z_holdout)

    field = chart_curvature.chart_curvature_field(model, x_holdout64, mode="reverse")
    cond_stats = _cond_diagnostic(field)

    ph = _ph_diagnostic(
        z_ph.cpu().numpy(),
        decoder_image_ph.cpu().numpy(),
        x_ph64.cpu().numpy(),
        jitter_seed=ph_jitter_seed,
    )

    return {
        "kind": "grid_cell",
        "n_charts": n_charts,
        "seed": seed,
        "train_wallclock_s": train_wallclock_s,
        "epochs_run": fit["epochs_run"],
        "early_stopped": fit["early_stopped"],
        "wallclock_truncated": fit["wallclock_truncated"],
        "cond": cond_stats,
        "occupancy": occ,
        "reconstruction": recon_stats,
        "ph": ph,
        "device": str(device),
        "torch_version": torch.__version__,
    }


def _run_control_cell(
    seed: int,
    x_train32: torch.Tensor,
    x_holdout64: torch.Tensor,
    x_ph64: torch.Tensor,
    max_epochs: int,
    latent_dim: int,
) -> Dict[str, Any]:
    """One ``PlainAutoEncoder`` control at ``latent_dim``. Called at ``latent_dim=
    PU_CHART_DIM`` for D-12's matched control (computed but never acted on here) and,
    optionally, at ``latent_dim=PU_CONTROL_CAPACITY_DIM`` for the separately-labelled
    capacity reference -- see :func:`build_control` and 03-08-DEFECTS-01.md defect 1."""
    device = x_train32.device
    torch.manual_seed(seed)
    model = build_control(latent_dim, device=device)
    cfg = _protocol_cfg(seed, n_charts=None, max_epochs=max_epochs)
    t0 = time.monotonic()
    fit = cae.train_plain_ae(model, x_train32, cfg)
    train_wallclock_s = time.monotonic() - t0

    model.eval().double()
    with torch.no_grad():
        z_holdout = model.encode(x_holdout64)
        recon_holdout = model.decode(z_holdout)
        z_ph = model.encode(x_ph64)
        decoder_image_ph = model.decode(z_ph)

    recon_stats = cae.reconstruction_stats(x_holdout64, recon_holdout)
    ph = _ph_diagnostic(
        z_ph.cpu().numpy(), decoder_image_ph.cpu().numpy(), x_ph64.cpu().numpy()
    )

    return {
        "kind": "control_cell",
        "seed": seed,
        "latent_dim": latent_dim,
        "train_wallclock_s": train_wallclock_s,
        "epochs_run": fit["epochs_run"],
        "early_stopped": fit["early_stopped"],
        "reconstruction": recon_stats,
        "ph": ph,
        "device": str(device),
        "torch_version": torch.__version__,
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
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_completed(record_path: Path) -> Dict[str, Dict[str, Any]]:
    """Every JSON-lines record on disk, keyed by its own ``config_id``. Missing file -> empty
    dict (a fresh run, not an error)."""
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


def _grid_records(completed: Dict[str, Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for rec in completed.values():
        if rec.get("kind") == "grid_cell":
            out[(int(rec["n_charts"]), int(rec["seed"]))] = rec
    return out


def _control_records(
    completed: Dict[str, Dict[str, Any]], latent_dim: Optional[int] = None
) -> List[Dict[str, Any]]:
    """All ``control_cell`` records, optionally filtered to one ``latent_dim``.
    03-08-DEFECTS-01.md defect 1: :func:`print_d12_trigger` MUST pass
    ``latent_dim=PU_CHART_DIM`` so the escalation trigger only ever reads the matched
    (20-dim) control, never the 40-dim capacity reference."""
    records = [rec for rec in completed.values() if rec.get("kind") == "control_cell"]
    if latent_dim is None:
        return records
    return [rec for rec in records if int(rec.get("latent_dim", PU_CHART_DIM)) == latent_dim]


def _planned_cells() -> List[Tuple[int, int]]:
    return [(nc, seed) for nc in PU_N_CHARTS_SWEEP for seed in PU_SEEDS]


# =============================================================================================
# The selection rule -- lexicographic, tie-banded on axis 1, never a composite.
# =============================================================================================


def apply_selection_rule(grid_records: Dict[Tuple[int, int], Dict[str, Any]]) -> Dict[str, Any]:
    by_n_charts: Dict[int, List[Dict[str, Any]]] = {}
    for (n_charts, _seed), rec in grid_records.items():
        by_n_charts.setdefault(n_charts, []).append(rec)

    table: List[Dict[str, Any]] = []
    for n_charts in sorted(by_n_charts):
        cells = by_n_charts[n_charts]
        table.append(
            {
                "n_charts": n_charts,
                "n_seeds_present": len(cells),
                "median_occupancy": _median([c["occupancy"]["n_distinct_charts"] for c in cells]),
                "median_cond_max": _median([c["cond"]["max"] for c in cells]),
                "median_mse_per_dim": _median([c["reconstruction"]["mse_per_dim"] for c in cells]),
                "median_h0_bottleneck_norm": _median(
                    [c["ph"]["latent|ambient|H0|bottleneck_norm"] for c in cells]
                ),
                "median_h1_bottleneck_norm": _median(
                    [c["ph"]["latent|ambient|H1|bottleneck_norm"] for c in cells]
                ),
            }
        )

    print("Selection table (per n_charts, across-seed medians):")
    for row in table:
        print(
            f"  n_charts={row['n_charts']:>3} seeds_present={row['n_seeds_present']} "
            f"median_occupancy={row['median_occupancy']:.2f} "
            f"median_max_cond_g={row['median_cond_max']:.4g} "
            f"median_mse_per_dim={row['median_mse_per_dim']:.6g} "
            f"median_H0_bottleneck_norm={row['median_h0_bottleneck_norm']:.4g} "
            f"median_H1_bottleneck_norm={row['median_h1_bottleneck_norm']:.4g}"
        )

    disqualified = [r for r in table if r["median_occupancy"] < 2]
    survivors = [r for r in table if r["median_occupancy"] >= 2]
    for r in disqualified:
        print(
            f"  DISQUALIFIED n_charts={r['n_charts']}: median distinct-chart occupancy "
            f"{r['median_occupancy']:.2f} < 2 -- a single-chart atlas is not an atlas."
        )

    if not survivors:
        print("No surviving n_charts config; the selection rule yields no selection.")
        return {
            "selected_n_charts": None,
            "table": table,
            "disqualified": [r["n_charts"] for r in disqualified],
        }

    def _cmp(a: Dict[str, Any], b: Dict[str, Any]) -> int:
        ca, cb = a["median_cond_max"], b["median_cond_max"]
        lo, hi = min(ca, cb), max(ca, cb)
        tied_on_cond = (lo <= 0.0) or (hi / lo <= 2.0)
        if not tied_on_cond:
            return -1 if ca < cb else 1
        ma, mb = a["median_mse_per_dim"], b["median_mse_per_dim"]
        if ma != mb:
            return -1 if ma < mb else 1
        ha, hb = a["median_h1_bottleneck_norm"], b["median_h1_bottleneck_norm"]
        if ha != hb:
            return -1 if ha < hb else 1
        return 0

    ranked = sorted(survivors, key=functools.cmp_to_key(_cmp))
    print("Lexicographic ranking (ascending -- first is selected):")
    for r in ranked:
        print(f"  n_charts={r['n_charts']}")
    print("No weighted composite formed; no single score printed above this line.")

    return {
        "selected_n_charts": ranked[0]["n_charts"],
        "table": table,
        "disqualified": [r["n_charts"] for r in disqualified],
        "ranked": [r["n_charts"] for r in ranked],
    }


def print_d12_trigger(
    grid_records: Dict[Tuple[int, int], Dict[str, Any]],
    control_records: List[Dict[str, Any]],
    selection_result: Dict[str, Any],
) -> None:
    """D-12's escalation trigger -- computed and printed, never acted on here.

    03-08-DEFECTS-01.md defect 1: filters ``control_records`` down to ``latent_dim ==
    PU_CHART_DIM`` (the matched control) BEFORE computing anything, regardless of what the
    caller passed in -- so a caller that (by mistake or by including the optional 40-dim
    capacity reference) hands this function a mixed list can never silently pull an
    unmatched control into the D-12 comparison."""
    control_records = [
        rec for rec in control_records if int(rec.get("latent_dim", PU_CHART_DIM)) == PU_CHART_DIM
    ]
    selected = selection_result.get("selected_n_charts")
    if selected is None or not control_records:
        print(
            "D-12 trigger: no CAE selection or no matched (latent_dim=PU_CHART_DIM) control "
            "records available; not evaluated."
        )
        return

    best_cells = [rec for (nc, _seed), rec in grid_records.items() if nc == selected]
    best_mse = _median([c["reconstruction"]["mse_per_dim"] for c in best_cells])
    best_h0 = _median([c["ph"]["latent|ambient|H0|bottleneck_norm"] for c in best_cells])
    best_h1 = _median([c["ph"]["latent|ambient|H1|bottleneck_norm"] for c in best_cells])

    ctrl_mse = _median([c["reconstruction"]["mse_per_dim"] for c in control_records])
    ctrl_h0 = _median([c["ph"]["latent|ambient|H0|bottleneck_norm"] for c in control_records])
    ctrl_h1 = _median([c["ph"]["latent|ambient|H1|bottleneck_norm"] for c in control_records])

    loses_recon = best_mse > ctrl_mse
    loses_ph = (best_h0 > ctrl_h0) and (best_h1 > ctrl_h1)
    fires = loses_recon and loses_ph

    _banner("D-12 escalation trigger (computed, not acted on)")
    print(
        f"best d=20 CAE (n_charts={selected}) mse_per_dim={best_mse:.6g} vs control "
        f"mse_per_dim={ctrl_mse:.6g} -> loses_reconstruction={loses_recon}"
    )
    print(
        f"best H0/H1 bottleneck_norm=({best_h0:.4g},{best_h1:.4g}) vs control="
        f"({ctrl_h0:.4g},{ctrl_h1:.4g}) -> loses_ph_agreement={loses_ph}"
    )
    print(f"TRIGGER FIRES = {fires}")
    print(
        "No d-sweep escalation happens here -- that decision belongs to plan 03-08. Known "
        "limitation (02.5-09): reconstruction and topology do not predict curvature quality "
        "(seeds 0/3 differed by 0.29pp reconstruction, 0.49 Spearman); this trigger is "
        "expected to fire."
    )


# =============================================================================================
# Timing probe (Task 2) -- training and curvature, at the real scale, before any grid cell.
# =============================================================================================


def run_timing_probe(
    x_train32: torch.Tensor, x_full64: torch.Tensor, pu_n: int, record_path: Path
) -> Dict[str, Any]:
    # Device is derived from the already-placed input tensor (main() moves x_train32 and
    # x_full64 to the target device once, before this probe runs; main() already printed
    # the device banner).
    device = x_train32.device
    if device.type == "cuda":
        print(
            "CAVEAT: curvature is float64-only, and on a consumer GPU float64 throughput "
            "can run at 1/32-1/64 of float32 (vs ~1/2 on a data-center GPU), so the "
            "curvature term below may be SLOWER on this GPU than on CPU even though "
            "training (float32) should be faster. Compare against a CPU probe before "
            "committing to the full grid (03-07-SUPPLEMENT-01)."
        )

    _banner("TIMING PROBE -- training, n_charts=16 (the most expensive swept value)")
    cfg = _protocol_cfg(PU_SEEDS[0], 16)
    train_timing = cae.timing_probe(
        lambda: build_cae(16, device=device), x_train32, cfg, n_steps=TIMING_PROBE_TRAIN_STEPS
    )
    print(f"seconds_per_training_step={train_timing['seconds_per_step']:.4f}")
    print(f"projected_per_fit_training_wallclock_s={train_timing['projected_wallclock_s']:.1f}")
    print(f"training_exceeds_wallclock_ceiling={train_timing['exceeds_ceiling']}")

    _banner(f"TIMING PROBE -- curvature at chart_dim={PU_CHART_DIM}, out_dim={AMBIENT_DIM}, pu_n={pu_n}")
    model = build_cae(16, device=device)
    model.double()
    x_slice = x_full64[:pu_n]

    t0 = time.monotonic()
    chart_curvature.chart_curvature_field(model, x_slice, mode="reverse")
    reverse_wallclock_s = time.monotonic() - t0

    t0 = time.monotonic()
    chart_curvature.chart_curvature_field(model, x_slice, mode="forward")
    forward_wallclock_s = time.monotonic() - t0

    print(f"reverse_mode_curvature_wallclock_s={reverse_wallclock_s:.4f} (pu_n={pu_n})")
    print(f"forward_mode_curvature_wallclock_s={forward_wallclock_s:.4f} (pu_n={pu_n})")

    ratio = reverse_wallclock_s / forward_wallclock_s if forward_wallclock_s > 0 else float("inf")
    print(
        f"measured_forward_vs_reverse_ratio={ratio:.2f}x "
        f"(D-08 operation-count ceiling: ~{FORWARD_REVERSE_OPCOUNT_CEILING:.0f}x)"
    )

    reverse_per_row_s = reverse_wallclock_s / pu_n
    forward_per_row_s = forward_wallclock_s / pu_n
    reverse_projected_10000row_s = reverse_per_row_s * 10000
    forward_projected_10000row_s = forward_per_row_s * 10000
    print(f"reverse_projected_10000row_s={reverse_projected_10000row_s:.1f}")
    print(f"forward_projected_10000row_s={forward_projected_10000row_s:.1f}")

    peak_memory_bytes = VMAP_CHUNK * AMBIENT_DIM * (PU_CHART_DIM ** 2) * 8
    print(
        f"peak_memory_per_chunk_bytes={peak_memory_bytes} ({peak_memory_bytes / 1e6:.1f} MB) "
        f"-- fixed by VMAP_CHUNK={VMAP_CHUNK}, unchanged by the differentiation mode."
    )

    n_cells = len(PU_N_CHARTS_SWEEP) * len(PU_SEEDS)
    projected_training_total_s = train_timing["projected_wallclock_s"] * n_cells
    projected_curvature_total_s = reverse_per_row_s * HOLDOUT_N_DEFAULT * n_cells
    projected_total_s = projected_training_total_s + projected_curvature_total_s
    envelope_s = WALLCLOCK_ENVELOPE_HOURS_MAX * 3600.0

    print(f"projected_nine_cell_training_total_s={projected_training_total_s:.1f}")
    print(
        f"projected_nine_cell_curvature_total_s={projected_curvature_total_s:.1f} "
        f"(reverse mode, {HOLDOUT_N_DEFAULT}-row holdout per cell)"
    )
    print(
        f"projected_nine_cell_grand_total_s={projected_total_s:.1f} "
        f"({projected_total_s / 3600.0:.2f}h) vs {WALLCLOCK_ENVELOPE_HOURS_MAX:.0f}-hour "
        f"envelope={envelope_s:.0f}s"
    )

    over_budget = projected_total_s > envelope_s
    record = {
        "kind": "timing_probe",
        "config_id": "timing_probe",
        "seconds_per_training_step": train_timing["seconds_per_step"],
        "projected_per_fit_training_wallclock_s": train_timing["projected_wallclock_s"],
        "pu_n": pu_n,
        "reverse_mode_curvature_wallclock_s": reverse_wallclock_s,
        "forward_mode_curvature_wallclock_s": forward_wallclock_s,
        "measured_forward_vs_reverse_ratio": ratio,
        "reverse_projected_10000row_s": reverse_projected_10000row_s,
        "forward_projected_10000row_s": forward_projected_10000row_s,
        "peak_memory_per_chunk_bytes": peak_memory_bytes,
        "projected_nine_cell_training_total_s": projected_training_total_s,
        "projected_nine_cell_curvature_total_s": projected_curvature_total_s,
        "projected_nine_cell_grand_total_s": projected_total_s,
        "envelope_s": envelope_s,
        "over_budget": over_budget,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    append_record(record_path, record)

    if over_budget:
        _banner("OVER BUDGET -- projected nine-cell total exceeds the 5-hour envelope")
        dominant = "training" if projected_training_total_s >= projected_curvature_total_s else "curvature"
        print(f"Projected total {projected_total_s / 3600.0:.2f}h exceeds the 5h envelope.")
        print(f"Dominant term: {dominant}.")
        print("Options (a developer decision, not made here):")
        print("  1. narrow PU_N_CHARTS_SWEEP")
        print("  2. drop to two seeds")
        print("  3. accept a longer run")
        print("Grid not started.")
        sys.exit(1)

    return record


HOLDOUT_N_DEFAULT = int(round(10000 * PU_HOLDOUT_FRACTION))


# =============================================================================================
# Smoke and full-grid execution.
# =============================================================================================


def run_smoke(x_train32: torch.Tensor, x_holdout64: torch.Tensor, record_path: Path) -> None:
    _banner("SMOKE -- one deliberately tiny cell (400 rows, n_charts=2, max_epochs=2)")
    x_train_s = x_train32[:SMOKE_TRAIN_N]
    x_holdout_s = x_holdout64[:SMOKE_HOLDOUT_N]
    x_ph_s = x_holdout_s[:SMOKE_PH_N]

    rec = _run_cae_cell(
        SMOKE_N_CHARTS,
        SMOKE_SEED,
        x_train_s,
        x_holdout_s,
        x_ph_s,
        max_epochs=SMOKE_MAX_EPOCHS,
        ph_jitter_seed=SMOKE_SEED,
    )
    rec["kind"] = "smoke_cell"
    rec["config_id"] = "smoke"
    append_record(record_path, rec)

    print("SMOKE TALLY:")
    print(f"  train_wallclock_s={rec['train_wallclock_s']:.2f}")
    print(f"  cond.max={rec['cond']['max']:.4g}  cond.median={rec['cond']['median']:.4g}")
    print(f"  occupancy.n_distinct_charts={rec['occupancy']['n_distinct_charts']}")
    print(f"  reconstruction.mse_per_dim={rec['reconstruction']['mse_per_dim']:.6g}")
    print(f"  reconstruction.dim_mse_p95={rec['reconstruction']['dim_mse_p95']:.6g}")
    print(f"  ph.latent|ambient|H0|bottleneck_norm={rec['ph']['latent|ambient|H0|bottleneck_norm']:.4g}")
    print(f"  ph.latent|ambient|H1|bottleneck_norm={rec['ph']['latent|ambient|H1|bottleneck_norm']:.4g}")
    print("SMOKE: exit 0.")


def run_grid(
    x_train32: torch.Tensor,
    x_holdout64: torch.Tensor,
    x_ph64: torch.Tensor,
    record_path: Path,
    resume: bool,
    max_combos: Optional[int],
) -> None:
    completed = load_completed(record_path) if resume else {}
    if resume:
        print(f"--resume: {len(completed)} record(s) already on disk, matched by config_id.")
        # 03-07-SUPPLEMENT-01 caveat 3: never mix devices within one grid -- a resumed
        # record from a different device would mix two different RNG draws.
        device = x_train32.device
        for config_id, rec in completed.items():
            rec_device = rec.get("device", "cpu")
            if rec_device != str(device):
                raise ValueError(
                    f"--resume: record_path={record_path} already contains cell "
                    f"{config_id!r} recorded at device={rec_device!r}, but this invocation "
                    f"requested device={str(device)!r} -- mixing devices within one grid "
                    "mixes two different RNG draws (03-07-SUPPLEMENT-01 caveat 3). Use a "
                    "distinct --record-path instead."
                )

    n_run = 0

    def _budget_reached() -> bool:
        return max_combos is not None and n_run >= max_combos

    for n_charts, seed in _planned_cells():
        config_id = f"nc{n_charts}_seed{seed}"
        if resume and config_id in completed:
            print(f"  [skip, resumed] {config_id}")
            continue
        if _budget_reached():
            print(f"--max-combos={max_combos} reached; stopping this invocation.")
            return
        print(f"[{config_id}] fitting...")
        t0 = time.monotonic()
        rec = _run_cae_cell(n_charts, seed, x_train32, x_holdout64, x_ph64, max_epochs=MAX_EPOCHS)
        rec["config_id"] = config_id
        append_record(record_path, rec)
        n_run += 1
        print(f"[{config_id}] done in {time.monotonic() - t0:.1f}s")

    # 03-08-DEFECTS-01.md defect 1: two labelled control sets. "control_seed{seed}" is the
    # matched D-12 control (latent_dim=PU_CHART_DIM=20) -- the only one print_d12_trigger may
    # read. "control40_seed{seed}" is the OPTIONAL, separately-labelled 40-dim capacity
    # reference (latent_dim=PU_CONTROL_CAPACITY_DIM) -- informative on its own, never fed to
    # the D-12 trigger (print_d12_trigger filters it out even if passed in by mistake).
    control_sets = (
        ("control", PU_CHART_DIM),
        ("control40", PU_CONTROL_CAPACITY_DIM),
    )
    for prefix, latent_dim in control_sets:
        for seed in PU_SEEDS:
            config_id = f"{prefix}_seed{seed}"
            if resume and config_id in completed:
                print(f"  [skip, resumed] {config_id}")
                continue
            if _budget_reached():
                print(f"--max-combos={max_combos} reached; stopping this invocation.")
                return
            print(f"[{config_id}] fitting control (latent_dim={latent_dim})...")
            t0 = time.monotonic()
            rec = _run_control_cell(
                seed, x_train32, x_holdout64, x_ph64, max_epochs=MAX_EPOCHS, latent_dim=latent_dim
            )
            rec["config_id"] = config_id
            append_record(record_path, rec)
            n_run += 1
            print(f"[{config_id}] done in {time.monotonic() - t0:.1f}s")

    completed_after = load_completed(record_path)
    grid_records = _grid_records(completed_after)
    if len(grid_records) == len(_planned_cells()):
        result = apply_selection_rule(grid_records)
        matched_control_records = _control_records(completed_after, latent_dim=PU_CHART_DIM)
        if len(matched_control_records) == len(PU_SEEDS):
            print_d12_trigger(grid_records, matched_control_records, result)


# =============================================================================================
# --dry-run and --select-only reporting.
# =============================================================================================


def print_dry_run_report() -> None:
    _banner("curvature_field_pu_run.py -- DRY RUN")
    cells = _planned_cells()
    print(f"Planned grid: {len(cells)} cells (n_charts x seed):")
    for nc, seed in cells:
        print(f"  n_charts={nc} seed={seed}")
    print(
        f"Plus {len(PU_SEEDS)} matched D-12 control cell(s) (PlainAutoEncoder at "
        f"latent_dim=PU_CHART_DIM={PU_CHART_DIM}, one per seed) and {len(PU_SEEDS)} "
        f"OPTIONAL capacity-reference control cell(s) (latent_dim=PU_CONTROL_CAPACITY_DIM="
        f"{PU_CONTROL_CAPACITY_DIM}, never fed to the D-12 trigger)."
    )
    print()
    print(DIMENSION_JUSTIFICATION_TEXT)
    print()
    print(GATE_OVERRIDE_TEXT)
    print()
    print(FRESH_FITS_TEXT)
    print()
    print(PH_CELL_CHOICE_TEXT)
    print()
    print(SELECTION_RULE_TEXT)
    print()
    print("--dry-run: nothing executed, nothing written.")


def run_select_only(record_path: Path) -> None:
    completed = load_completed(record_path)
    grid_records = _grid_records(completed)
    planned = _planned_cells()
    print(f"{len(grid_records)} of {len(planned)} planned grid cells present in {record_path}")
    if not grid_records:
        print("No grid cells on record yet; nothing to select.")
        return
    result = apply_selection_rule(grid_records)
    print(f"Selected n_charts: {result['selected_n_charts']}")
    matched_control_records = _control_records(completed, latent_dim=PU_CHART_DIM)
    capacity_control_records = _control_records(completed, latent_dim=PU_CONTROL_CAPACITY_DIM)
    print(
        f"{len(matched_control_records)} of {len(PU_SEEDS)} planned matched "
        f"(latent_dim={PU_CHART_DIM}) D-12 control cells present."
    )
    print(
        f"{len(capacity_control_records)} of {len(PU_SEEDS)} planned capacity-reference "
        f"(latent_dim={PU_CONTROL_CAPACITY_DIM}) control cells present (informational only, "
        "never fed to the D-12 trigger)."
    )
    if matched_control_records:
        print_d12_trigger(grid_records, matched_control_records, result)


# =============================================================================================
# main
# =============================================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one deliberately tiny cell (400 rows, n_charts=2, max_epochs=2) to prove "
        "the wiring, and print a tally. This is the runner's automated <verify>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned nine-cell grid, the dimension justification, the gate "
        "override, the PH cell choice and the full selection rule, then exit writing "
        "nothing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip any (n_charts, seed) or control cell already present in the record file "
        "rather than recomputing it.",
    )
    parser.add_argument(
        "--record-path",
        type=str,
        default=None,
        help=f"Override the append-only record path (default: notebooks/.cache/"
        f"{RECORD_STEM}.jsonl).",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Run at most this many combinations this invocation, then stop.",
    )
    parser.add_argument(
        "--timing-probe",
        action="store_true",
        help="Measure both training and curvature wall clock at the real scale (d=20, "
        "D=768) before any grid cell runs, and halt-and-report if the projected nine-cell "
        "total exceeds the 5-hour envelope.",
    )
    parser.add_argument(
        "--pu-n",
        type=int,
        default=DEFAULT_TIMING_PROBE_PU_N,
        help="Row count for the --timing-probe curvature slice (default: "
        f"{DEFAULT_TIMING_PROBE_PU_N}).",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Read the JSONL record back, print the full table, apply the selection rule "
        "and name the selected n_charts, without running any fit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="03-07-SUPPLEMENT-01: 'cpu' (default, zero behaviour change), 'cuda', or "
        "'cuda:N'. Fails with an actionable message if cuda is requested but unavailable. "
        "THREE CAVEATS: (1) GPU runs do NOT reproduce CPU runs bit-for-bit -- CUDA RNG "
        "differs from CPU RNG, so a GPU cell is a different draw, not a reproduction; every "
        "record carries its own 'device' field for exactly this reason. (2) float64 "
        "throughput is hardware-dependent -- curvature is float64-only, and on a consumer "
        "GPU float64 can run at 1/32-1/64 of float32 (vs ~1/2 on a data-center GPU), so the "
        "curvature term may be SLOWER on a consumer GPU than on CPU even though training "
        "(the dominant nine-cell term) should be faster -- run --timing-probe --device cuda "
        "first and compare. (3) Do not mix devices within one grid -- every cell (plus the "
        "D-12 controls) must share one device, or the seed spread mixes two different "
        "draws; --resume refuses to continue a record whose device disagrees.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    record_path = Path(args.record_path) if args.record_path else _default_record_path()
    device = resolve_device(args.device)

    _banner("Phase 3 Plan 07: PU curvature-field sweep runner")
    print(f"record_path={record_path}")
    print(f"device={device}  torch_version={torch.__version__}")
    if device.type == "cuda":
        print(
            "CAVEAT: GPU runs do NOT reproduce CPU runs bit-for-bit (CUDA RNG != CPU RNG); "
            "this is a different draw, recorded as such via each record's 'device' field."
        )

    if args.dry_run:
        print_dry_run_report()
        return

    if args.select_only:
        run_select_only(record_path)
        return

    x_full64_np = _load_subsample()
    # Tensors are built on CPU first (torch.tensor from a numpy array always lands on
    # CPU), then moved to the target device in one place; index tensors are moved
    # explicitly rather than relying on numpy-array indexing to auto-place against a
    # non-CPU tensor.
    x_full32 = torch.tensor(x_full64_np, dtype=torch.float32).to(device)
    x_full64 = torch.tensor(x_full64_np, dtype=torch.float64).to(device)
    train_idx, holdout_idx = _split(x_full64_np.shape[0], PU_SPLIT_SEED, PU_HOLDOUT_FRACTION)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    holdout_idx_t = torch.as_tensor(holdout_idx, dtype=torch.long, device=device)
    x_train32 = x_full32[train_idx_t]
    x_holdout64 = x_full64[holdout_idx_t]
    x_ph64 = x_holdout64[:N_PH_PU]

    if args.timing_probe:
        run_timing_probe(x_train32, x_full64, args.pu_n, record_path)
        return

    if args.smoke:
        run_smoke(x_train32, x_holdout64, record_path)
        return

    run_grid(x_train32, x_holdout64, x_ph64, record_path, args.resume, args.max_combos)


if __name__ == "__main__":
    main()
