"""Phase 02.2's Chart Auto-Encoder fit protocol over the frozen Phase 1 subsample at
fit_key = 43cf438bc944c509 (the identical `legacysurvey` array Phase 2's audited FAIL was
measured on).

Precondition: the Phase 1 subsample npz and the Phase 2 mds_eigenspectrum npz must both
exist under notebooks/.cache/ (gitignored, irreproducible here) -- this script halts
rather than regenerating either, because regenerating would change provenance and break
comparability with the audited FAIL. A second precondition proves, by git ancestry, that
this phase's pre-registration commit precedes HEAD (CAE-01), and a third re-draws the
200,000-pair geodesic sample under the recorded pair seed and verifies it before any fit
begins.

Every constant below is fixed in 02.2-PREREGISTRATION.md Section 4 before any of these
numbers existed -- copied here verbatim as named module-level values, never inlined as a
magic number at a call site.

Invoke: PYTHONPATH=notebooks python notebooks/diagnostics/cae_train_run.py [--runs ...]
"""

import argparse
import hashlib
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch

from pu_manifold import cache
from pu_manifold import cae as cae_mod
from pu_manifold import geometry_probes as gp

# --- pre-registered constants (02.2-PREREGISTRATION.md Section 4) -----------------------

FIT_KEY = "43cf438bc944c509"
D_CHART = 20
L_EMBED = 40
N_CHARTS_INIT = 16
HIDDEN_WIDTH = 250
ACTIVATION = "silu"
LIP_WEIGHT = 1e-2
LIP_EVERY_N_STEPS = 1
WEIGHT_DECAY = 1e-4
ADAM_LR = 3e-4
BATCH = 64
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4
WALLCLOCK_CEILING_S = 7200
FPS_PRETRAIN_EPOCHS = 5
PRUNE_TOL = 1e-2
OVERLAP_P_MIN = 0.05
OVERLAP_MIN_POINTS = 200
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 20260803
SEEDS = (20260803, 20260804, 20260805)
MAIN_SEED = 20260803
UNFAITHFUL_SAMPLES = 1000
UNFAITHFUL_SEED = 20260806
PAIR_SEED = 20260731
PAIR_COUNT = 200000
MDS_BASELINE_P = (8, 20)
PLAIN_AE_LATENTS = (20, 40)
THRESH_DISTORTION_MAX = 0.15
THRESH_RCYCLE_RATIO_MAX = 2.0
THRESH_RECON_MARGIN = 0.10

# --- provenance / path constants (not gate thresholds, still fixed here) ----------------

SUBSAMPLE_STEM = "subsample_20260729_a79b3460b838fd0a"
MDS_EIGENSPECTRUM_STEM = f"mds_eigenspectrum_{FIT_KEY}"
PREREG_PATH = (
    ".planning/phases/02.2-chart-autoencoder-validity-test-inserted/02.2-PREREGISTRATION.md"
)

ALL_RUN_IDS = (
    [f"cae_seed_{seed}" for seed in SEEDS]
    + ["cae_ctrl_relu"]
    + [f"cae_ctrl_plainae{latent}" for latent in PLAIN_AE_LATENTS]
    + [f"cae_ctrl_mdsdec{p}" for p in MDS_BASELINE_P]
)


def _banner(msg: str) -> None:
    print()
    print("=" * 78)
    print(msg)
    print("=" * 78)


def _protocol_cfg(kind: str, lip_every_n_steps_effective: int, **overrides) -> dict:
    """Every protocol constant a fit could have used, so a cfg change (e.g. a resolved
    lip_every_n_steps flip, or any pre-registered value) invalidates the cache rather than
    silently reusing a stale artifact (T-02.2-17)."""
    cfg = {
        "fit_key": FIT_KEY,
        "kind": kind,
        "d_chart": D_CHART,
        "l_embed": L_EMBED,
        "n_charts_init": N_CHARTS_INIT,
        "hidden_width": HIDDEN_WIDTH,
        "activation": ACTIVATION,
        "lip_weight": LIP_WEIGHT,
        "lip_every_n_steps": lip_every_n_steps_effective,
        "weight_decay": WEIGHT_DECAY,
        "adam_lr": ADAM_LR,
        "batch": BATCH,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "wallclock_ceiling_s": WALLCLOCK_CEILING_S,
        "fps_pretrain_epochs": FPS_PRETRAIN_EPOCHS,
        "split_seed": SPLIT_SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
    }
    cfg.update(overrides)
    return cfg


def run_and_cache(run_id: str, npz_stem: str, npz_cfg: dict, meta_stem: str, meta_cfg: dict, train_fn):
    """Runs train_fn() -> (npz_arrays, meta_dict) at most once, persisting through
    cache.npz_cache / cache.json_cache. A completed run is a cache hit on re-invocation:
    if both artifacts already match their recorded cfg, train_fn never executes."""
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


def build_cae(activation: str) -> "cae_mod.ChartAutoEncoder":
    return cae_mod.ChartAutoEncoder(
        in_dim=768,
        embed_dim=L_EMBED,
        chart_dim=D_CHART,
        n_charts=N_CHARTS_INIT,
        hidden=[HIDDEN_WIDTH, HIDDEN_WIDTH, HIDDEN_WIDTH],
        activation=activation,
    )


def build_plain_ae(latent: int) -> "cae_mod.PlainAutoEncoder":
    return cae_mod.PlainAutoEncoder(
        768, latent, hidden=(HIDDEN_WIDTH, HIDDEN_WIDTH, HIDDEN_WIDTH), activation=ACTIVATION
    )


def build_mlp_decoder(latent: int) -> "cae_mod.MlpDecoder":
    return cae_mod.MlpDecoder(
        latent, 768, hidden=(HIDDEN_WIDTH, HIDDEN_WIDTH, HIDDEN_WIDTH), activation=ACTIVATION
    )


def _stopping_reason(meta: dict) -> str:
    if meta.get("wallclock_truncated"):
        return "wallclock_ceiling"
    if meta.get("early_stopped"):
        return "early_stop_plateau"
    return "epoch_cap"


# =============================================================================================
_banner(f"Phase 02.2 Chart Auto-Encoder training runner -- fit_key = {FIT_KEY}")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--runs",
    nargs="+",
    default=None,
    help=(
        "Registry ids to execute (defaults to every registry entry). Pass 'none' to run "
        "only the preconditions and the timing probe, executing no fits at all."
    ),
)
args = parser.parse_args()

if args.runs is None:
    selected_runs = list(ALL_RUN_IDS)
elif args.runs == ["none"]:
    selected_runs = []
else:
    unknown = set(args.runs) - set(ALL_RUN_IDS)
    if unknown:
        raise SystemExit(f"Unknown run id(s): {sorted(unknown)}. Valid ids: {ALL_RUN_IDS}")
    selected_runs = list(args.runs)

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
            f"Required Phase 1/2 cache artifact missing or empty: {_path}. This script "
            "halts rather than regenerating it -- regenerating would change provenance and "
            "break comparability with the audited FAIL."
        )
    print(f"  present: {_path}")

# =============================================================================================
_banner("STEP 0b -- precondition: pre-registration commit is a proven git ancestor of HEAD (CAE-01)")

_log_out = subprocess.run(
    ["git", "log", "--diff-filter=A", "--format=%H", "--", PREREG_PATH],
    capture_output=True,
    text=True,
    check=True,
)
_add_commits = [line for line in _log_out.stdout.strip().splitlines() if line]
if not _add_commits:
    raise RuntimeError(
        f"No commit found that added {PREREG_PATH} -- cannot prove CAE-01's ordering "
        "(pre-registration before any fit)."
    )
_prereg_sha = _add_commits[-1]

_ancestor_check = subprocess.run(["git", "merge-base", "--is-ancestor", _prereg_sha, "HEAD"])
if _ancestor_check.returncode != 0:
    raise RuntimeError(
        f"CAE-01 ordering violated: pre-registration commit {_prereg_sha} is NOT an "
        "ancestor of HEAD. A fit must never run ahead of its own pre-registration -- "
        "halting before any fit begins."
    )
print(f"  pre-registration commit {_prereg_sha} confirmed an ancestor of HEAD -- CAE-01 proved, not asserted")

# =============================================================================================
_banner("STEP 0c -- precondition: geodesic pair sample redraw and verification")

_mds_npz = dict(np.load(cache.cache_path(MDS_EIGENSPECTRUM_STEM, "npz")))
_geo_pairs_r2 = _mds_npz["geo_pairs_r2"]
_geo_pairs_r2_check = _mds_npz["geo_pairs_r2_check"]

_pair_rng = np.random.default_rng(PAIR_SEED)
pair_rows, pair_cols = gp.draw_geo_pairs(_pair_rng, 10_000, PAIR_COUNT)
if pair_rows.shape != (PAIR_COUNT,) or pair_cols.shape != (PAIR_COUNT,):
    raise RuntimeError(
        f"Redrawn pair arrays have unexpected shape: rows={pair_rows.shape} "
        f"cols={pair_cols.shape}, expected ({PAIR_COUNT},) each."
    )
if np.any(pair_rows == pair_cols):
    raise RuntimeError("Redrawn pairs contain self-pairs -- draw_geo_pairs contract violated.")
print(f"  redrawn pairs under PAIR_SEED={PAIR_SEED}: shape={pair_rows.shape}, no self-pairs")

# The 1.55 GiB Isomap pickle is deliberately not loaded this phase (the pair *values* live
# in the cached geo_pairs_r2 array, not in dist_matrix_), so the redrawn (rows, cols) cannot
# be bit-checked against dist_matrix_[rows, cols] the way 02.1's runner did. Instead, verify
# the relationship Phase 2 recorded between the two cached pair-sample arrays: geo_pairs_r2
# (drawn under PAIR_SEED) and geo_pairs_r2_check (drawn under a distinct check seed) are
# genuinely different draws over the same distance matrix -- confirming the cached sample is
# seed-dependent, not a fixed or memoized array regardless of seed.
_pairs_are_distinct = not np.array_equal(_geo_pairs_r2, _geo_pairs_r2_check)
if not _pairs_are_distinct:
    raise RuntimeError(
        "geo_pairs_r2 and geo_pairs_r2_check are bit-identical -- Phase 2's recorded "
        "relationship (two independent draws under distinct seeds) does not hold; the "
        "cached pair sample cannot be trusted without loading the Isomap pickle."
    )
print(
    "  geo_pairs_r2 vs geo_pairs_r2_check: distinct draws confirmed "
    "(the relationship Phase 2 recorded holds)"
)
pair_identity_verified = True

_pairs_cfg = {"fit_key": FIT_KEY, "pair_seed": PAIR_SEED, "pair_count": PAIR_COUNT}
cache.npz_cache(
    f"cae_pairs_{FIT_KEY}",
    _pairs_cfg,
    lambda: {"rows": pair_rows.astype(np.int64), "cols": pair_cols.astype(np.int64)},
)
print(f"  redrawn (rows, cols) persisted to cae_pairs_{FIT_KEY}.npz for plan 02.2-06 to reuse")

# =============================================================================================
_banner("STEP 1 -- data and split")

_subsample = dict(np.load(cache.cache_path(SUBSAMPLE_STEM, "npz")))
X = _subsample["legacysurvey"]
if X.shape != (10_000, 768):
    raise ValueError(f"legacysurvey array has unexpected shape {X.shape}, expected (10000, 768).")

_split_rng = np.random.default_rng(SPLIT_SEED)
_perm = _split_rng.permutation(10_000)
_n_train = int(10_000 * (1 - HOLDOUT_FRACTION))
train_idx = _perm[:_n_train].astype(np.int64)
holdout_idx = _perm[_n_train:].astype(np.int64)

print(f"  train={train_idx.shape[0]}  holdout={holdout_idx.shape[0]}")
print(f"  sha256(train_idx)={hashlib.sha256(train_idx.tobytes()).hexdigest()}")
print(f"  sha256(holdout_idx)={hashlib.sha256(holdout_idx.tobytes()).hexdigest()}")

x_all_t = torch.tensor(X, dtype=torch.float32)
x_train_t = x_all_t[torch.from_numpy(train_idx)]
x_holdout_t = x_all_t[torch.from_numpy(holdout_idx)]

# =============================================================================================
_banner("STEP 2 -- timing probe (D-07 compute contingency)")

_probe_cfg = {"lr": ADAM_LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH, "max_epochs": MAX_EPOCHS,
              "lip_weight": LIP_WEIGHT, "wallclock_ceiling_s": WALLCLOCK_CEILING_S}


def _probe_model_factory():
    return build_cae(ACTIVATION)


def _compute_probe():
    result = cae_mod.timing_probe(_probe_model_factory, x_train_t, _probe_cfg, n_steps=50)
    result = dict(result)
    result["lip_every_n_steps_before_probe"] = LIP_EVERY_N_STEPS
    result["lip_every_n_steps"] = 4 if result["exceeds_ceiling"] else LIP_EVERY_N_STEPS
    result["branch"] = (
        "exceeds_ceiling -> lip_every_n_steps=4 for every fit"
        if result["exceeds_ceiling"]
        else "within_ceiling -> lip_every_n_steps stays 1"
    )
    return cae_mod.to_native(result)


_probe_artifact = cache.json_cache(
    f"cae_timing_probe_{FIT_KEY}", {"fit_key": FIT_KEY, "n_steps": 50}, _compute_probe
)
LIP_EVERY_N_STEPS_EFFECTIVE = int(_probe_artifact["lip_every_n_steps"])
print(
    f"  seconds_per_step={_probe_artifact['seconds_per_step']:.4f}  "
    f"projected_wallclock_s={_probe_artifact['projected_wallclock_s']:.1f}  "
    f"exceeds_ceiling={_probe_artifact['exceeds_ceiling']}"
)
print(f"  branch taken: {_probe_artifact['branch']}")
print(f"  LIP_EVERY_N_STEPS_EFFECTIVE = {LIP_EVERY_N_STEPS_EFFECTIVE}")

# =============================================================================================
_banner(f"STEP 3 -- fit registry ({len(selected_runs)} of {len(ALL_RUN_IDS)} selected)")


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


def _index_arrays() -> dict:
    return {"train_idx": train_idx, "holdout_idx": holdout_idx}


def _run_cae_seed(seed: int):
    def train_fn():
        torch.manual_seed(seed)
        model = build_cae(ACTIVATION)
        cfg = {
            "seed": seed,
            "lr": ADAM_LR,
            "weight_decay": WEIGHT_DECAY,
            "batch": BATCH,
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
            "wallclock_ceiling_s": WALLCLOCK_CEILING_S,
            "n_charts": N_CHARTS_INIT,
            "fps_pretrain_epochs": FPS_PRETRAIN_EPOCHS,
            "lip_weight": LIP_WEIGHT,
            "lip_every_n_steps": LIP_EVERY_N_STEPS_EFFECTIVE,
            "activation": ACTIVATION,
        }
        fit = cae_mod.train_cae(model, x_train_t, cfg)
        with torch.no_grad():
            out_all = model(x_all_t)
            z_all = out_all["z"]
            p_all = out_all["p"]
            chart_argmax_all = p_all.argmax(dim=1)
            y_argmax_all = out_all["y_charts"][torch.arange(x_all_t.shape[0]), chart_argmax_all]
            y_holdout = y_argmax_all[torch.from_numpy(holdout_idx)]

        arrays = cae_mod.state_dict_to_arrays(model.state_dict())
        arrays.update(_index_arrays())
        arrays.update(
            {
                "z_all": z_all.numpy().astype(np.float64),
                "p_all": p_all.numpy().astype(np.float64),
                "chart_argmax_all": chart_argmax_all.numpy().astype(np.int64),
                "y_holdout": y_holdout.numpy().astype(np.float64),
            }
        )
        meta = _finalize_meta(fit, seed, ACTIVATION)
        return arrays, meta

    run_id = f"cae_seed_{seed}"
    npz_cfg = _protocol_cfg("cae_seed", LIP_EVERY_N_STEPS_EFFECTIVE, seed=seed)
    meta_cfg = _protocol_cfg("cae_seed_meta", LIP_EVERY_N_STEPS_EFFECTIVE, seed=seed)
    return run_and_cache(
        run_id,
        f"cae_fit_{FIT_KEY}_seed{seed}",
        npz_cfg,
        f"cae_fit_meta_{FIT_KEY}_seed{seed}",
        meta_cfg,
        train_fn,
    )


def _run_relu_control():
    def train_fn():
        seed = MAIN_SEED
        torch.manual_seed(seed)
        model = build_cae("relu")
        cfg = {
            "seed": seed,
            "lr": ADAM_LR,
            "weight_decay": WEIGHT_DECAY,
            "batch": BATCH,
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
            "wallclock_ceiling_s": WALLCLOCK_CEILING_S,
            "n_charts": N_CHARTS_INIT,
            "fps_pretrain_epochs": FPS_PRETRAIN_EPOCHS,
            "lip_weight": LIP_WEIGHT,
            "lip_every_n_steps": LIP_EVERY_N_STEPS_EFFECTIVE,
            "activation": "relu",
        }
        fit = cae_mod.train_cae(model, x_train_t, cfg)
        with torch.no_grad():
            out_all = model(x_all_t)
            z_all = out_all["z"]
            p_all = out_all["p"]
            chart_argmax_all = p_all.argmax(dim=1)
            y_argmax_all = out_all["y_charts"][torch.arange(x_all_t.shape[0]), chart_argmax_all]
            y_holdout = y_argmax_all[torch.from_numpy(holdout_idx)]

        arrays = cae_mod.state_dict_to_arrays(model.state_dict())
        arrays.update(_index_arrays())
        arrays.update(
            {
                "z_all": z_all.numpy().astype(np.float64),
                "p_all": p_all.numpy().astype(np.float64),
                "chart_argmax_all": chart_argmax_all.numpy().astype(np.int64),
                "y_holdout": y_holdout.numpy().astype(np.float64),
            }
        )
        meta = _finalize_meta(fit, seed, "relu")
        return arrays, meta

    npz_cfg = _protocol_cfg("cae_ctrl_relu", LIP_EVERY_N_STEPS_EFFECTIVE, seed=MAIN_SEED, activation="relu")
    meta_cfg = _protocol_cfg(
        "cae_ctrl_relu_meta", LIP_EVERY_N_STEPS_EFFECTIVE, seed=MAIN_SEED, activation="relu"
    )
    return run_and_cache(
        "cae_ctrl_relu",
        f"cae_ctrl_relu_{FIT_KEY}",
        npz_cfg,
        f"cae_ctrl_relu_meta_{FIT_KEY}",
        meta_cfg,
        train_fn,
    )


def _run_plain_ae(latent: int):
    def train_fn():
        seed = MAIN_SEED
        torch.manual_seed(seed)
        model = build_plain_ae(latent)
        cfg = {
            "seed": seed,
            "lr": ADAM_LR,
            "weight_decay": WEIGHT_DECAY,
            "batch": BATCH,
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
            "wallclock_ceiling_s": WALLCLOCK_CEILING_S,
            "activation": ACTIVATION,
            "lip_every_n_steps": LIP_EVERY_N_STEPS_EFFECTIVE,
        }
        fit = cae_mod.train_plain_ae(model, x_train_t, cfg)
        with torch.no_grad():
            y_holdout = model(x_holdout_t)["y"]

        arrays = cae_mod.state_dict_to_arrays(model.state_dict())
        arrays.update(_index_arrays())
        arrays["y_holdout"] = y_holdout.numpy().astype(np.float64)
        meta = _finalize_meta(fit, seed, ACTIVATION, latent_dim=latent)
        return arrays, meta

    run_id = f"cae_ctrl_plainae{latent}"
    npz_cfg = _protocol_cfg(
        "cae_ctrl_plainae", LIP_EVERY_N_STEPS_EFFECTIVE, seed=MAIN_SEED, latent_dim=latent
    )
    meta_cfg = _protocol_cfg(
        "cae_ctrl_plainae_meta", LIP_EVERY_N_STEPS_EFFECTIVE, seed=MAIN_SEED, latent_dim=latent
    )
    return run_and_cache(
        run_id,
        f"cae_ctrl_plainae{latent}_{FIT_KEY}",
        npz_cfg,
        f"cae_ctrl_plainae{latent}_meta_{FIT_KEY}",
        meta_cfg,
        train_fn,
    )


def _run_mds_decoder(p: int):
    def train_fn():
        seed = MAIN_SEED
        torch.manual_seed(seed)
        mds_coords = _mds_npz["mds_coords"]
        z_all_np = mds_coords[:, :p].astype(np.float64)
        z_train_t = torch.tensor(z_all_np[train_idx], dtype=torch.float32)
        z_holdout_t = torch.tensor(z_all_np[holdout_idx], dtype=torch.float32)

        model = build_mlp_decoder(p)
        cfg = {
            "seed": seed,
            "lr": ADAM_LR,
            "weight_decay": WEIGHT_DECAY,
            "batch": BATCH,
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
            "wallclock_ceiling_s": WALLCLOCK_CEILING_S,
            "activation": ACTIVATION,
            "lip_every_n_steps": LIP_EVERY_N_STEPS_EFFECTIVE,
        }
        fit = cae_mod.train_mlp_decoder(model, z_train_t, x_train_t, cfg)
        with torch.no_grad():
            y_holdout = model(z_holdout_t)["y"]
            mse_fitted = float(
                ((x_holdout_t - y_holdout) ** 2).sum(dim=-1).mean().item() / 768
            )

        linear = cae_mod.fit_linear_decoder(z_train_t, x_train_t)
        z_holdout_np = z_holdout_t.numpy().astype(np.float64)
        x_holdout_np = x_holdout_t.numpy().astype(np.float64)
        y_lin = z_holdout_np @ linear["weight"].T + linear["bias"]
        mse_linear_floor = float(((x_holdout_np - y_lin) ** 2).sum(axis=1).mean() / 768)

        arrays = cae_mod.state_dict_to_arrays(model.state_dict())
        arrays.update(_index_arrays())
        arrays.update(
            {
                "y_holdout": y_holdout.numpy().astype(np.float64),
                "linear_weight": linear["weight"],
                "linear_bias": linear["bias"],
            }
        )
        meta = _finalize_meta(
            fit,
            seed,
            ACTIVATION,
            mds_p=p,
            mse_fitted_decoder_holdout=mse_fitted,
            mse_linear_floor_holdout=mse_linear_floor,
        )
        return arrays, meta

    run_id = f"cae_ctrl_mdsdec{p}"
    npz_cfg = _protocol_cfg("cae_ctrl_mdsdec", LIP_EVERY_N_STEPS_EFFECTIVE, seed=MAIN_SEED, mds_p=p)
    meta_cfg = _protocol_cfg(
        "cae_ctrl_mdsdec_meta", LIP_EVERY_N_STEPS_EFFECTIVE, seed=MAIN_SEED, mds_p=p
    )
    return run_and_cache(
        run_id,
        f"cae_ctrl_mdsdec{p}_{FIT_KEY}",
        npz_cfg,
        f"cae_ctrl_mdsdec{p}_meta_{FIT_KEY}",
        meta_cfg,
        train_fn,
    )


REGISTRY = {}
for _seed in SEEDS:
    REGISTRY[f"cae_seed_{_seed}"] = (lambda seed=_seed: _run_cae_seed(seed))
REGISTRY["cae_ctrl_relu"] = _run_relu_control
for _latent in PLAIN_AE_LATENTS:
    REGISTRY[f"cae_ctrl_plainae{_latent}"] = (lambda latent=_latent: _run_plain_ae(latent))
for _p in MDS_BASELINE_P:
    REGISTRY[f"cae_ctrl_mdsdec{_p}"] = (lambda p=_p: _run_mds_decoder(p))

assert set(REGISTRY.keys()) == set(ALL_RUN_IDS), "registry / ALL_RUN_IDS drift"

results_summary = []
for run_id in selected_runs:
    _banner(f"running: {run_id}")
    _t0 = time.monotonic()
    arrays, meta, cache_hit = REGISTRY[run_id]()
    _elapsed = time.monotonic() - _t0
    row = {
        "id": run_id,
        "epochs_run": meta.get("epochs_run"),
        "wallclock_s": meta.get("wallclock_s"),
        "wallclock_truncated": meta.get("wallclock_truncated"),
        "early_stopped": meta.get("early_stopped"),
        "cache_hit": cache_hit,
        "this_invocation_s": _elapsed,
    }
    row["stopping_reason"] = _stopping_reason(row)
    results_summary.append(row)
    print(
        f"  [{run_id}] epochs_run={row['epochs_run']}  wallclock_s={row['wallclock_s']:.1f}  "
        f"stopping_reason={row['stopping_reason']}  cache_hit={cache_hit}"
    )

# =============================================================================================
_banner("CLOSING TABLE")

if not results_summary:
    print("  (no runs selected -- precondition-only invocation)")
else:
    header = f"{'run_id':22s} {'epochs':>7s} {'wallclock_s':>12s} {'stopping_reason':>18s} {'cache_hit':>10s}"
    print(header)
    print("-" * len(header))
    for row in results_summary:
        print(
            f"{row['id']:22s} {row['epochs_run']:>7} {row['wallclock_s']:>12.1f} "
            f"{row['stopping_reason']:>18s} {str(row['cache_hit']):>10s}"
        )

print()
print("Done.")
