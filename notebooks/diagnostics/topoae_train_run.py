"""Phase 02.4's gated Topological Auto-Encoder (TopoAE) fit protocol over the frozen Phase 1
subsample at fit_key = 43cf438bc944c509 (the identical `legacysurvey` array Phase 2's audited
FAIL was measured on, and the same array Phase 02.2's Chart Auto-Encoder fits used).

Two preconditions run before any fit:

  0a. The Phase 1 subsample npz and the Phase 2 mds_eigenspectrum npz must both exist under
      notebooks/.cache/ (gitignored, irreproducible here) -- this script halts rather than
      regenerating either, because regenerating would change provenance and break
      comparability with the audited FAIL.
  0b. 02.4-PREREGISTRATION.md's adding commit must be a proven git ancestor of HEAD (re-resolved
      by `git log --diff-filter=A ... | tail -1` at every invocation, never hard-coded), so no
      threshold or constant below could have been chosen after a fit ran.

Every constant below is fixed in 02.4-PREREGISTRATION.md Sections 3-4 before any PU fit exists
-- copied here verbatim as named module-level values, never inlined as a magic number at a call
site. `MAX_EPOCHS` is the one exception: it is a *rule*, not a number (MAX_EPOCHS_CANDIDATES),
resolved by a real timing probe run inside this gated runner, after both preconditions pass.

Training itself reuses `pu_manifold.topoae.train_topoae` and `pu_manifold.cae.train_plain_ae`
unchanged -- this runner does not reimplement or "correct" either training loop. In particular,
`train_topoae`'s per-batch ambient/latent normalization (`d_x / d_x.max()` plus the
jointly-trained `latent_norm`) is the paper's own training convention and is NOT the same thing
as `AMBIENT_DIST_NORM = "none"` below, which scopes to the T1 gate only (02.4-PREREGISTRATION.md
Section 1, corrected by Erratum 1) -- this runner does not compute T1 at all; that is plan
02.4-07's evaluate runner.

`notebooks/pu_manifold/cae.py` is Phase 02.2's sealed artifact and is never edited by this
phase -- the matched-capacity `PlainAutoEncoder` baseline, `train_plain_ae`, and the
weights-to-arrays serialization are imported from it, never copied.

Invoke: PYTHONPATH=notebooks python notebooks/diagnostics/topoae_train_run.py [--runs ...]
"""

import argparse
import hashlib
import math
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch

from pu_manifold import cache
from pu_manifold import cae as cae_mod
from pu_manifold import topoae as topoae_mod

# --- pre-registered constants (02.4-PREREGISTRATION.md Section 3) -----------------------

LADDER = (8, 16, 20, 24, 32, 40)
D_PRIMARY = 20
SEEDS_PRIMARY = (20260806, 20260807, 20260808)

# --- pre-registered constants (02.4-PREREGISTRATION.md Section 4) -----------------------

FIT_KEY = "43cf438bc944c509"
SUBSAMPLE_STEM = "subsample_20260729_a79b3460b838fd0a"
MDS_EIGENSPECTRUM_STEM = "mds_eigenspectrum_43cf438bc944c509"
AMBIENT_DIM = 768
HIDDEN = (250, 250, 250)
ACTIVATION = "silu"  # C2-smooth throughout; the rectifier family must never appear here (DEC-02/CURV-01..03)
BATCH = 64
DROP_LAST = True
ADAM_LR = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4
WALLCLOCK_CEILING_S = 3600
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 20260806
LAMBDA_TOPO = 0.1
WARMUP_FRAC = 0.25
RAMP_FRAC = 0.25
RAMP_SHAPE = "linear"
AMBIENT_DIST_NORM = "none"  # T1-gate-only scope (Section 1); NOT read by train_topoae's training loop
LATENT_DIST_NORM = "global_isotropic_unit_variance"
T3_K_SWEEP = (5, 15, 30)
T3_K_GATE = 15
UNFAITHFUL_SAMPLES = 1000
UNFAITHFUL_SEED = 20260809
PAIR_SEED = 20260731

# `MAX_EPOCHS` is a rule, not a number (Section 4): the largest candidate below whose
# cae.timing_probe-style projection at D_PRIMARY lands under WALLCLOCK_CEILING_S. Resolved by
# STEP 2, after both preconditions pass -- no fit precedes the pre-registration.
MAX_EPOCHS_CANDIDATES = (40, 30, 20, 15, 10)

PREREG_PATH = (
    ".planning/phases/02.4-topological-auto-encoder-validity-test-inserted/02.4-PREREGISTRATION.md"
)

# --- fit registry order (02.4-PREREGISTRATION.md Section 7) -----------------------------
# The primary rung's first seed-matched TopoAE/baseline pair first, then the other two
# primary seeds, then the sensitivity rungs in ladder order 8, 16, 24, 32, 40 -- so a
# mid-run interruption leaves the verdict-critical fits complete.

_SENSITIVITY_RUNGS = tuple(d for d in LADDER if d != D_PRIMARY)

_PRIMARY_RUN_IDS = []
for _seed in SEEDS_PRIMARY:
    _PRIMARY_RUN_IDS.append(f"topoae_d{D_PRIMARY}_seed{_seed}")
    _PRIMARY_RUN_IDS.append(f"baseline_d{D_PRIMARY}_seed{_seed}")

_SENSITIVITY_RUN_IDS = []
for _d in _SENSITIVITY_RUNGS:
    _SENSITIVITY_RUN_IDS.append(f"topoae_d{_d}_seed{SEEDS_PRIMARY[0]}")
    _SENSITIVITY_RUN_IDS.append(f"baseline_d{_d}_seed{SEEDS_PRIMARY[0]}")

ALL_RUN_IDS = tuple(_PRIMARY_RUN_IDS + _SENSITIVITY_RUN_IDS)

_PRIMARY_FIRST_RUN_ID = f"topoae_d{D_PRIMARY}_seed{SEEDS_PRIMARY[0]}"

# Topological-loss-specific cfg keys a baseline fit must never carry -- excluded from the
# baseline-vs-TopoAE cfg match (there is nothing on the baseline side for them to match).
_TOPO_ONLY_KEYS = {"lambda_topo", "warmup_frac", "ramp_frac", "ramp_shape"}
_EXCLUDE_FROM_CFG_MATCH = _TOPO_ONLY_KEYS | {"kind"}


def _banner(msg: str) -> None:
    print()
    print("=" * 78)
    print(msg)
    print("=" * 78)


def run_and_cache(run_id: str, npz_stem: str, npz_cfg: dict, meta_stem: str, meta_cfg: dict, train_fn):
    """Runs train_fn() -> (npz_arrays, meta_dict) at most once, persisting through
    cache.npz_cache / cache.json_cache. A completed run is a cache hit on re-invocation:
    if both artifacts already match their recorded cfg, train_fn never executes -- the
    mechanism R2's bit-identical-reproducibility truth depends on."""
    box = {}

    def _compute():
        if "value" not in box:
            print(f"  [{run_id}] cache miss -- fitting...")
            t0 = time.monotonic()
            arrays, meta = train_fn()
            print(f"  [{run_id}] fit complete in {time.monotonic() - t0:.1f}s")
            box["value"] = (arrays, meta)
        return box["value"]

    arrays = cache.npz_cache(npz_stem, npz_cfg, lambda: _compute()[0])
    meta = cache.json_cache(meta_stem, meta_cfg, lambda: _compute()[1])
    cache_hit = "value" not in box
    if cache_hit:
        print(f"  [{run_id}] cache hit -- no fit performed")
    return arrays, meta, cache_hit


def _finalize_meta(fit: dict, seed: int, activation: str, **extra) -> dict:
    meta = dict(fit)
    meta.update(
        {
            "seed": seed,
            "activation": activation,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    meta.update(extra)
    return cae_mod.to_native(meta)


def _topoae_cfg(seed: int, d: int) -> dict:
    return {
        "fit_key": FIT_KEY,
        "kind": "topoae",
        "seed": seed,
        "d": d,
        "lr": ADAM_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch": BATCH,
        "drop_last": DROP_LAST,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "wallclock_ceiling_s": WALLCLOCK_CEILING_S,
        "lambda_topo": LAMBDA_TOPO,
        "warmup_frac": WARMUP_FRAC,
        "ramp_frac": RAMP_FRAC,
        "ramp_shape": RAMP_SHAPE,
        "activation": ACTIVATION,
    }


def _baseline_cfg(seed: int, d: int) -> dict:
    return {
        "fit_key": FIT_KEY,
        "kind": "baseline",
        "seed": seed,
        "d": d,
        "lr": ADAM_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch": BATCH,
        "drop_last": DROP_LAST,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "wallclock_ceiling_s": WALLCLOCK_CEILING_S,
        "activation": ACTIVATION,
    }


def _assert_baseline_matches_topoae_cfg(topoae_cfg: dict, baseline_cfg: dict) -> None:
    """Prohibition 3 enforced in code, not by care: raises ValueError naming the
    differing key if the baseline cfg diverges from its paired TopoAE cfg on any
    protocol key other than the topological-loss-specific keys."""
    topoae_keys = set(topoae_cfg) - _EXCLUDE_FROM_CFG_MATCH
    baseline_keys = set(baseline_cfg) - _EXCLUDE_FROM_CFG_MATCH
    missing_in_baseline = topoae_keys - baseline_keys
    if missing_in_baseline:
        raise ValueError(
            f"baseline cfg is missing key(s) {sorted(missing_in_baseline)} present in its "
            "paired TopoAE cfg -- MUST NOT give the PlainAutoEncoder baseline less capacity, "
            "shorter training, or a different protocol than the TopoAE"
        )
    for key in sorted(topoae_keys):
        if baseline_cfg[key] != topoae_cfg[key]:
            raise ValueError(
                f"baseline cfg key {key!r}={baseline_cfg[key]!r} does not match paired "
                f"TopoAE cfg value {topoae_cfg[key]!r} -- MUST NOT give the PlainAutoEncoder "
                "baseline less capacity, shorter training, or a different protocol than the "
                "TopoAE"
            )


def _build_model(d: int) -> "cae_mod.PlainAutoEncoder":
    return cae_mod.PlainAutoEncoder(AMBIENT_DIM, d, hidden=HIDDEN, activation=ACTIVATION)


# =============================================================================================
_banner(f"Phase 02.4 Topological Auto-Encoder training runner -- fit_key = {FIT_KEY}")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--runs",
    nargs="+",
    default=None,
    help=(
        "Registry ids to execute (defaults to every registry entry). Accepts either "
        "space-separated tokens or a single comma-separated token. Pass 'none' to run only "
        "the preconditions and the timing probe, executing no fits at all."
    ),
)
args = parser.parse_args()


def _flatten_runs(raw):
    if raw is None:
        return None
    out = []
    for token in raw:
        out.extend([r for r in token.split(",") if r])
    return out


_raw_runs = _flatten_runs(args.runs)
if _raw_runs is None:
    selected_runs = list(ALL_RUN_IDS)
elif _raw_runs == ["none"]:
    selected_runs = []
else:
    unknown = set(_raw_runs) - set(ALL_RUN_IDS)
    if unknown:
        raise SystemExit(f"Unknown run id(s): {sorted(unknown)}. Valid ids: {ALL_RUN_IDS}")
    # Canonical registry order, not the order the user supplied -- so a mid-run
    # interruption always leaves the verdict-critical fits complete regardless of how
    # --runs was invoked.
    selected_runs = [r for r in ALL_RUN_IDS if r in set(_raw_runs)]

print(f"  selected runs: {selected_runs if selected_runs else '(none -- precondition-only invocation)'}")

# =============================================================================================
_banner("STEP 0a -- precondition: required Phase 1/2 cache artifacts exist")

_required = [
    cache.cache_path(SUBSAMPLE_STEM, "npz"),
    cache.cache_path(MDS_EIGENSPECTRUM_STEM, "npz"),
]
for _path in _required:
    if not _path.exists() or _path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Required Phase 1/2 cache artifact missing or empty: {_path}. This script halts "
            "rather than regenerating it -- regenerating would change provenance and break "
            "comparability with the audited FAIL."
        )
    print(f"  present: {_path}")

# =============================================================================================
_banner("STEP 0b -- precondition: pre-registration commit is a proven git ancestor of HEAD")

_log_out = subprocess.run(
    ["git", "log", "--diff-filter=A", "--format=%H", "--", PREREG_PATH],
    capture_output=True,
    text=True,
    check=True,
)
_add_commits = [line for line in _log_out.stdout.strip().splitlines() if line]
if not _add_commits:
    raise RuntimeError(
        f"No commit found that added {PREREG_PATH} -- cannot prove the pre-registration "
        "precedes every fit."
    )
_prereg_sha = _add_commits[-1]

_ancestor_check = subprocess.run(["git", "merge-base", "--is-ancestor", _prereg_sha, "HEAD"])
if _ancestor_check.returncode != 0:
    raise RuntimeError(
        f"Ordering violated: pre-registration commit {_prereg_sha} is NOT an ancestor of "
        "HEAD. A fit must never run ahead of its own pre-registration -- halting before any "
        "fit begins."
    )
print(f"  pre-registration commit {_prereg_sha} confirmed an ancestor of HEAD -- proved, not asserted")

# =============================================================================================
_banner("STEP 1 -- data and holdout split")

_subsample = dict(np.load(cache.cache_path(SUBSAMPLE_STEM, "npz")))
X = _subsample["legacysurvey"]
if X.shape != (10_000, AMBIENT_DIM):
    raise ValueError(f"legacysurvey array has unexpected shape {X.shape}, expected (10000, {AMBIENT_DIM}).")

_split_rng = np.random.default_rng(SPLIT_SEED)
_perm = _split_rng.permutation(10_000)
_n_train = int(10_000 * (1 - HOLDOUT_FRACTION))
train_idx = _perm[:_n_train].astype(np.int64)
holdout_idx = _perm[_n_train:].astype(np.int64)

print(f"  train={train_idx.shape[0]}  holdout={holdout_idx.shape[0]}")
print(f"  sha256(train_idx)={hashlib.sha256(train_idx.tobytes()).hexdigest()}")
print(f"  sha256(holdout_idx)={hashlib.sha256(holdout_idx.tobytes()).hexdigest()}")

_split_cfg = {"fit_key": FIT_KEY, "split_seed": SPLIT_SEED, "holdout_fraction": HOLDOUT_FRACTION}
cache.npz_cache(
    f"topoae_split_{FIT_KEY}", _split_cfg, lambda: {"train_idx": train_idx, "holdout_idx": holdout_idx}
)
print(f"  split indices persisted to topoae_split_{FIT_KEY}.npz so the evaluation runner reproduces the same split")

x_all_t = torch.tensor(X, dtype=torch.float32)
x_train_t = x_all_t[torch.from_numpy(train_idx)]
x_holdout_t = x_all_t[torch.from_numpy(holdout_idx)]

# =============================================================================================
_banner("STEP 2 -- timing probe and the MAX_EPOCHS rule")


def _topoae_timing_probe(x_train: torch.Tensor, cfg: dict, n_steps: int):
    """Mirrors cae.timing_probe's shape but with this phase's per-batch loss -- a
    train_topoae-equivalent step including the topological term (persistence pairs +
    the MST-based loss), not chart_loss. Runs a fresh model/optimizer for n_steps real
    steps and returns (seconds_per_step, n_batches_per_epoch)."""
    model = _build_model(D_PRIMARY)
    latent_norm = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [latent_norm], lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    n = x_train.shape[0]
    batch_size = cfg["batch"]
    lambda_t = cfg["lambda_topo"]

    steps_done = 0
    start = time.monotonic()
    while steps_done < n_steps:
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            if steps_done >= n_steps:
                break
            idx = perm[i : i + batch_size]
            if idx.shape[0] < 2:
                continue
            xb = x_train[idx].double()
            z = model.encode(xb.float()).double()
            d_x = topoae_mod.pairwise_distances_f64(xb)
            d_z = topoae_mod.pairwise_distances_f64(z)
            d_x_norm = d_x / d_x.max()
            d_z_norm = d_z / latent_norm
            recon = ((xb - model.decode(z.float()).double()) ** 2).sum(-1).mean()
            topo = topoae_mod.topological_loss(d_x_norm, d_z_norm)["total"]
            total = recon + lambda_t * topo
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            steps_done += 1
    elapsed = time.monotonic() - start

    seconds_per_step = elapsed / n_steps
    n_batches_per_epoch = math.ceil(n / batch_size)
    return seconds_per_step, n_batches_per_epoch


_probe_cfg = {"lr": ADAM_LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH, "lambda_topo": LAMBDA_TOPO}


def _compute_probe():
    torch.manual_seed(SPLIT_SEED)
    seconds_per_step, n_batches_per_epoch = _topoae_timing_probe(x_train_t, _probe_cfg, n_steps=50)
    projections = {
        str(candidate): seconds_per_step * n_batches_per_epoch * candidate
        for candidate in MAX_EPOCHS_CANDIDATES
    }
    return {
        "seconds_per_step": seconds_per_step,
        "n_batches_per_epoch": n_batches_per_epoch,
        "projections": projections,
    }


_probe_artifact = cache.json_cache(
    f"topoae_timing_probe_{FIT_KEY}", {"fit_key": FIT_KEY, "d_primary": D_PRIMARY, "n_steps": 50}, _compute_probe
)

print(f"  seconds_per_step={_probe_artifact['seconds_per_step']:.4f}  n_batches_per_epoch={_probe_artifact['n_batches_per_epoch']}")
print("  MAX_EPOCHS_CANDIDATES projections (pre-registered rule: largest candidate under the ceiling):")
MAX_EPOCHS = None
for _candidate in MAX_EPOCHS_CANDIDATES:
    _proj = _probe_artifact["projections"][str(_candidate)]
    _fits = _proj < WALLCLOCK_CEILING_S
    print(f"    candidate={_candidate:>3}  projected_wallclock_s={_proj:>10.1f}  fits_under_ceiling={_fits}")
    if _fits and MAX_EPOCHS is None:
        MAX_EPOCHS = _candidate

if MAX_EPOCHS is None:
    raise RuntimeError(
        f"No MAX_EPOCHS_CANDIDATES value projects under WALLCLOCK_CEILING_S={WALLCLOCK_CEILING_S}s -- "
        f"refusing to silently fall back to the smallest candidate. candidates={MAX_EPOCHS_CANDIDATES}"
    )
print(f"  selected MAX_EPOCHS = {MAX_EPOCHS} (the pre-registered rule, applied -- not a choice made here)")

# =============================================================================================
_banner(f"STEP 3 -- fit registry ({len(selected_runs)} of {len(ALL_RUN_IDS)} selected)")


def _transfer_ratio_from_history(history: list, warmup_frac: float, ramp_frac: float, max_epochs: int) -> dict:
    """The lambda sweep's own transfer-diagnostic estimator
    (topoae_lambda_sweep_run.py's `_run_arm`), reproduced verbatim here so both sides of
    the Swiss-roll-vs-PU comparison share exactly one definition: the LAMBDA-WEIGHTED
    topological term divided by the reconstruction term, read at the first epoch after
    the ramp completes (`lambda_t == lambda_topo`, constant) -- never an unweighted,
    all-epoch average, which would let large un-regularized warm-up-epoch topo values
    (`lambda_t == 0`) dominate an estimate of the trained topo/recon balance.

    Falls back to the last recorded epoch, exactly as the sweep script's own comment
    documents, when training (early stop or wallclock) stopped before reaching the
    post-ramp epoch -- never fabricating a value. Returns the ratio plus which epoch and
    lambda_t it was actually read at, and whether that fallback fired, so a caller can
    tell a like-for-like comparison from a partial-ramp one without re-deriving it from
    the raw history."""
    warmup_epochs = math.floor(warmup_frac * max_epochs)
    ramp_epochs = math.floor(ramp_frac * max_epochs)
    post_ramp_epoch = warmup_epochs + ramp_epochs
    post_ramp_entries = [h for h in history if h["epoch"] == post_ramp_epoch]
    if post_ramp_entries:
        h0 = post_ramp_entries[0]
        reached_post_ramp_epoch = True
    else:
        h0 = history[-1]
        reached_post_ramp_epoch = False
    recon_term = h0["recon"]
    topo_term = h0["lambda_t"] * h0["topo"]
    ratio = topo_term / recon_term if recon_term != 0.0 else float("inf")
    return {
        "transfer_ratio": ratio,
        "transfer_ratio_epoch_used": h0["epoch"],
        "transfer_ratio_lambda_t_used": h0["lambda_t"],
        "transfer_ratio_post_ramp_epoch_target": post_ramp_epoch,
        "transfer_ratio_reached_post_ramp_epoch": reached_post_ramp_epoch,
    }


def _run_topoae(d: int, seed: int):
    cfg = _topoae_cfg(seed, d)

    def train_fn():
        torch.manual_seed(seed)
        model = _build_model(d)
        fit = topoae_mod.train_topoae(model, x_train_t, cfg)
        with torch.no_grad():
            z_all = model.encode(x_all_t)
            y_all = model.decode(z_all)

        arrays = cae_mod.state_dict_to_arrays(model.state_dict())
        arrays["z_all"] = z_all.numpy().astype(np.float64)
        arrays["y_holdout"] = y_all[torch.from_numpy(holdout_idx)].numpy().astype(np.float64)

        history = fit["history"]
        transfer_fields = _transfer_ratio_from_history(history, WARMUP_FRAC, RAMP_FRAC, MAX_EPOCHS)
        meta = _finalize_meta(
            fit,
            seed,
            ACTIVATION,
            d=d,
            max_epochs=MAX_EPOCHS,
            **transfer_fields,
        )
        return arrays, meta

    npz_stem = f"topoae_fit_{FIT_KEY}_seed{seed}_d{d}"
    meta_stem = f"topoae_fit_meta_{FIT_KEY}_seed{seed}_d{d}"
    arrays, meta, cache_hit = run_and_cache(f"topoae_d{d}_seed{seed}", npz_stem, cfg, meta_stem, cfg, train_fn)
    return arrays, meta, cache_hit, cfg


def _run_baseline(d: int, seed: int):
    topoae_cfg_for_match = _topoae_cfg(seed, d)
    cfg = _baseline_cfg(seed, d)
    _assert_baseline_matches_topoae_cfg(topoae_cfg_for_match, cfg)

    def train_fn():
        torch.manual_seed(seed)
        model = _build_model(d)
        fit = cae_mod.train_plain_ae(model, x_train_t, cfg)
        with torch.no_grad():
            y_holdout = model(x_holdout_t)["y"]

        arrays = cae_mod.state_dict_to_arrays(model.state_dict())
        arrays["y_holdout"] = y_holdout.numpy().astype(np.float64)
        meta = _finalize_meta(fit, seed, ACTIVATION, d=d, max_epochs=MAX_EPOCHS)
        return arrays, meta

    npz_stem = f"topoae_baseline_{FIT_KEY}_seed{seed}_d{d}"
    meta_stem = f"topoae_baseline_meta_{FIT_KEY}_seed{seed}_d{d}"
    arrays, meta, cache_hit = run_and_cache(f"baseline_d{d}_seed{seed}", npz_stem, cfg, meta_stem, cfg, train_fn)
    return arrays, meta, cache_hit, cfg


REGISTRY = {}
for _seed in SEEDS_PRIMARY:
    REGISTRY[f"topoae_d{D_PRIMARY}_seed{_seed}"] = (lambda seed=_seed: _run_topoae(D_PRIMARY, seed))
    REGISTRY[f"baseline_d{D_PRIMARY}_seed{_seed}"] = (lambda seed=_seed: _run_baseline(D_PRIMARY, seed))
for _d in _SENSITIVITY_RUNGS:
    REGISTRY[f"topoae_d{_d}_seed{SEEDS_PRIMARY[0]}"] = (lambda d=_d: _run_topoae(d, SEEDS_PRIMARY[0]))
    REGISTRY[f"baseline_d{_d}_seed{SEEDS_PRIMARY[0]}"] = (lambda d=_d: _run_baseline(d, SEEDS_PRIMARY[0]))

assert set(REGISTRY.keys()) == set(ALL_RUN_IDS), "registry / ALL_RUN_IDS drift"

results_summary = []
halted_for_nonconvergence = False
for run_id in selected_runs:
    _banner(f"running: {run_id}")
    _t0 = time.monotonic()
    arrays, meta, cache_hit, cfg_used = REGISTRY[run_id]()
    _elapsed = time.monotonic() - _t0
    _history = meta.get("history") or []
    row = {
        "id": run_id,
        "seed": meta.get("seed"),
        "d": meta.get("d"),
        "epochs_run": meta.get("epochs_run"),
        "wallclock_s": meta.get("wallclock_s"),
        "wallclock_truncated": meta.get("wallclock_truncated"),
        "early_stopped": meta.get("early_stopped"),
        "final_total_loss": _history[-1]["total"] if _history else None,
        "cache_hit": cache_hit,
        "this_invocation_s": _elapsed,
    }
    results_summary.append(row)
    print(
        f"  [{run_id}] seed={row['seed']} d={row['d']} epochs_run={row['epochs_run']} "
        f"wallclock_s={row['wallclock_s']:.1f} early_stopped={row['early_stopped']} "
        f"wallclock_truncated={row['wallclock_truncated']} final_total_loss={row['final_total_loss']} "
        f"cache_hit={cache_hit}"
    )

    if run_id == _PRIMARY_FIRST_RUN_ID:
        _wallclock_truncated = bool(meta.get("wallclock_truncated"))
        _early_stopped = bool(meta.get("early_stopped"))
        if _wallclock_truncated and not _early_stopped:
            _banner("D-19 CONVERGENCE CONTINGENCY FIRED -- primary rung did not converge")
            print(
                f"  The primary rung's first fit ({_PRIMARY_FIRST_RUN_ID}) finished with "
                "wallclock_truncated=True and early_stopped=False -- it hit the one-hour "
                "wall-clock ceiling still improving, rather than converging or plateauing."
            )
            print(
                "  Per 02.4-PREREGISTRATION.md Section 4's pre-registered D-19 convergence "
                "contingency, this run halts here and reports non-convergence at the primary "
                "rung rather than proceeding to the remaining rungs. Raising "
                "WALLCLOCK_CEILING_S to make this go away would be changing a pre-registered "
                "constant after a fit has run, which requires a fresh, separately-ratified "
                "pre-registration and a full re-run of all sixteen fits -- not an edit."
            )
            halted_for_nonconvergence = True
            break

# =============================================================================================
_banner("CLOSING TABLE")

if not results_summary:
    print("  (no runs selected -- precondition-only invocation)")
else:
    header = f"{'run_id':28s} {'epochs':>7s} {'wallclock_s':>12s} {'early_stop':>11s} {'truncated':>10s} {'cache_hit':>10s}"
    print(header)
    print("-" * len(header))
    for row in results_summary:
        print(
            f"{row['id']:28s} {row['epochs_run']:>7} {row['wallclock_s']:>12.1f} "
            f"{str(row['early_stopped']):>11s} {str(row['wallclock_truncated']):>10s} "
            f"{str(row['cache_hit']):>10s}"
        )

if halted_for_nonconvergence:
    _banner("RUN HALTED -- D-19 non-convergence at the primary rung (measured, not tuned around)")

print()
print("Done.")
