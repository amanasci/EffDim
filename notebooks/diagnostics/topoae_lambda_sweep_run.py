"""Lambda selection sweep for Phase 02.4's Topological Auto-Encoder, run on the Swiss roll
fixture -- NOT on the PU embedding data. D-11 permits freezing lambda from a sweep on this
fixture precisely because the fixture is not the gate data; a sweep on PU data would be the
post-hoc metric-selection move the milestone's pre-registration discipline exists to
prevent.

**Binding selection rule, written before any lambda is measured (LAMBDA_SELECT_RULE below,
repeated here verbatim so it precedes every number this script prints):**

    The selected lambda is the largest value in LAMBDA_GRID whose Swiss-roll held-out
    mse_per_dim is strictly below 1.10 times the mse_per_dim of the zero-lambda fit.
    Maximise topological regularisation subject to not degrading reconstruction by more
    than ten percent against the unregularised fit at matched capacity. If no positive
    value satisfies it, the selection is the smallest positive grid value and the sweep
    reports that the constraint bound at the grid floor.

Task 2 (the plan that runs this script) must not adjust the grid, the rule, or any
constant to move the selection after the numbers are seen -- if the rule binds at the
floor, that is the measured result, reported as such.

**D-11's stated risk, restated:** lambda is tuned here on a 2-dimensional sheet embedded in
3 dimensions and will be applied to a ~20-dimensional manifold embedded in 768 dimensions.
This transfer is a stated risk, not a demonstrated property. To make a transfer failure
visible rather than silent, this sweep additionally records, for every arm, the ratio of
the (lambda-weighted) topological loss term to the reconstruction term at the end of
warm-up -- the concrete, independently-tunable quantity 02.4-RESEARCH.md Pitfall 2 warns
may not transfer. The PU training run reports the same ratio so an order-of-magnitude gap
between the two is visible on inspection rather than buried in an unexplained gate FAIL.

**A4 / D-13:** this runner deliberately carries NO git-ancestry precondition. It runs
before 02.4-PREREGISTRATION.md exists, and per D-13 that ordering is correct: the Swiss
roll fixture and its lambda sweep legitimately precede the pre-registration, because D-11's
own reasoning is that the fixture is not the gate data. Do not add a commit-ancestry proof
(the check `cae_train_run.py` runs against its own pre-registration) here -- that
discipline belongs to the PU fit runner (plan 02.4-05), not to this one.

Invoke: PYTHONPATH=notebooks python notebooks/diagnostics/topoae_lambda_sweep_run.py
"""

import time

import numpy as np
import torch
from sklearn.datasets import make_swiss_roll

from pu_manifold import cache
from pu_manifold import cae as cae_mod
from pu_manifold import topoae

# --- Swiss roll fixture constants -----------------------------------------------------------
# Shared verbatim with notebooks/02.4_swiss_roll_topoae_check.ipynb (D-12's shared-generator
# requirement) -- do not let the notebook diverge from these exact values.

SWISS_N = 3000
SWISS_SEED = 20260806
SWISS_NOISE = 0.0
SPLIT_SEED = 20260806
HOLDOUT_FRACTION = 0.2

# --- lambda grid and the binding selection rule ---------------------------------------------

LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)  # spans the paper's own best-run 0.43-0.50
                                                # range with roughly a decade either side

LAMBDA_SELECT_RULE = (
    "The selected lambda is the largest value in LAMBDA_GRID whose Swiss-roll held-out "
    "mse_per_dim is strictly below 1.10 times the mse_per_dim of the zero-lambda fit. "
    "Maximise topological regularisation subject to not degrading reconstruction by more "
    "than ten percent against the unregularised fit at matched capacity. If no positive "
    "value satisfies it, the selection is the smallest positive grid value and the sweep "
    "reports that the constraint bound at the grid floor."
)

# --- model / training constants --------------------------------------------------------------

SWEEP_LATENT = 2
SWEEP_HIDDEN = (64, 64, 64)
SWEEP_ACTIVATION = "silu"
SWEEP_BATCH = 64
SWEEP_MAX_EPOCHS = 300
SWEEP_LR = 3e-4
SWEEP_WEIGHT_DECAY = 1e-4
WARMUP_FRAC = 0.25
RAMP_FRAC = 0.25
SWEEP_SEED = 20260806


def _banner(msg: str) -> None:
    print()
    print("=" * 78)
    print(msg)
    print("=" * 78)


def _sweep_cfg() -> dict:
    """Every constant a sweep arm could have used, so a change to any of them
    invalidates the cache rather than silently reusing a stale table."""
    return {
        "swiss_n": SWISS_N,
        "swiss_seed": SWISS_SEED,
        "swiss_noise": SWISS_NOISE,
        "split_seed": SPLIT_SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "lambda_grid": list(LAMBDA_GRID),
        "sweep_latent": SWEEP_LATENT,
        "sweep_hidden": list(SWEEP_HIDDEN),
        "sweep_activation": SWEEP_ACTIVATION,
        "sweep_batch": SWEEP_BATCH,
        "sweep_max_epochs": SWEEP_MAX_EPOCHS,
        "sweep_lr": SWEEP_LR,
        "sweep_weight_decay": SWEEP_WEIGHT_DECAY,
        "warmup_frac": WARMUP_FRAC,
        "ramp_frac": RAMP_FRAC,
        "sweep_seed": SWEEP_SEED,
    }


# =================================================================================================
_banner("Phase 02.4 lambda selection sweep -- Swiss roll fixture only, not the gate data (D-11)")
print(LAMBDA_SELECT_RULE)

# =================================================================================================
_banner("STEP 1 -- Swiss roll fixture (shared verbatim with the sanity-check notebook)")

X_raw, t_param = make_swiss_roll(n_samples=SWISS_N, noise=SWISS_NOISE, random_state=SWISS_SEED)
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std()

_split_rng = np.random.default_rng(SPLIT_SEED)
_perm = _split_rng.permutation(SWISS_N)
_n_train = int(SWISS_N * (1 - HOLDOUT_FRACTION))
train_idx = _perm[:_n_train]
holdout_idx = _perm[_n_train:]

x_all = torch.tensor(X, dtype=torch.float32)
x_train = x_all[torch.from_numpy(train_idx)]
x_holdout = x_all[torch.from_numpy(holdout_idx)]

print(f"  X {X.shape}  train {x_train.shape[0]}  holdout {x_holdout.shape[0]}")

# =================================================================================================
_banner(f"STEP 2 -- sweep over LAMBDA_GRID = {LAMBDA_GRID}")


def _run_arm(lam: float) -> dict:
    torch.manual_seed(SWEEP_SEED)
    model = cae_mod.PlainAutoEncoder(3, SWEEP_LATENT, hidden=SWEEP_HIDDEN, activation=SWEEP_ACTIVATION)
    cfg = {
        "seed": SWEEP_SEED,
        "lr": SWEEP_LR,
        "weight_decay": SWEEP_WEIGHT_DECAY,
        "batch": SWEEP_BATCH,
        "max_epochs": SWEEP_MAX_EPOCHS,
        "lambda_topo": lam,
        "warmup_frac": WARMUP_FRAC,
        "ramp_frac": RAMP_FRAC,
    }
    t0 = time.monotonic()
    fit = topoae.train_topoae(model, x_train, cfg)
    wallclock_s = time.monotonic() - t0

    model.eval()
    with torch.no_grad():
        z_holdout = model.encode(x_holdout)
        y_holdout = model.decode(z_holdout)
    recon_stats = cae_mod.reconstruction_stats(x_holdout.double(), y_holdout.double())

    d_x = topoae.pairwise_distances_f64(x_holdout)
    z_scaled = z_holdout * topoae.latent_unit_scale(z_holdout)
    d_z = topoae.pairwise_distances_f64(z_scaled)
    loss = topoae.topological_loss(d_x, d_z)
    loss_x_to_z = float(loss["loss_x_to_z"].item())
    loss_z_to_x = float(loss["loss_z_to_x"].item())

    # Transfer diagnostic: ratio of the lambda-weighted topological term to the
    # reconstruction term, averaged over the first epoch after the ramp completes.
    warmup_epochs = int(WARMUP_FRAC * SWEEP_MAX_EPOCHS)
    ramp_epochs = int(RAMP_FRAC * SWEEP_MAX_EPOCHS)
    post_ramp_epoch = warmup_epochs + ramp_epochs  # first epoch at the constant lambda
    history = fit["history"]
    post_ramp_entries = [h for h in history if h["epoch"] == post_ramp_epoch]
    if post_ramp_entries:
        h0 = post_ramp_entries[0]
        recon_term = h0["recon"]
        topo_term = h0["lambda_t"] * h0["topo"]
    else:
        # Training stopped (early stop / wallclock) before reaching the post-ramp epoch;
        # fall back to the last recorded epoch rather than fabricate a value.
        h0 = history[-1]
        recon_term = h0["recon"]
        topo_term = h0["lambda_t"] * h0["topo"]
    transfer_ratio = topo_term / recon_term if recon_term != 0.0 else float("inf")

    return {
        "lambda": lam,
        "mse_per_dim": recon_stats["mse_per_dim"],
        "loss_x_to_z": loss_x_to_z,
        "loss_z_to_x": loss_z_to_x,
        "transfer_ratio": transfer_ratio,
        "wallclock_s": wallclock_s,
        "epochs_run": fit["epochs_run"],
    }


def _compute_sweep() -> dict:
    rows = []
    for lam in LAMBDA_GRID:
        print(f"  [lambda={lam}] training...")
        row = _run_arm(lam)
        print(
            f"  [lambda={lam}] mse_per_dim={row['mse_per_dim']:.6f}  "
            f"loss_x_to_z={row['loss_x_to_z']:.6f}  loss_z_to_x={row['loss_z_to_x']:.6f}  "
            f"transfer_ratio={row['transfer_ratio']:.6f}  wallclock_s={row['wallclock_s']:.1f}"
        )
        rows.append(row)

    baseline_mse = next(r["mse_per_dim"] for r in rows if r["lambda"] == 0.0)
    for row in rows:
        row["mse_ratio_vs_zero"] = row["mse_per_dim"] / baseline_mse

    # Apply the binding selection rule.
    candidates = [
        r for r in rows if r["lambda"] > 0.0 and r["mse_per_dim"] < 1.10 * baseline_mse
    ]
    if candidates:
        selected_row = max(candidates, key=lambda r: r["lambda"])
        selection_branch = "interior grid point satisfying the <=10% reconstruction-degradation bound"
    else:
        positive = [r for r in rows if r["lambda"] > 0.0]
        selected_row = min(positive, key=lambda r: r["lambda"])
        selection_branch = "grid floor -- no positive lambda satisfied the <=10% bound"

    return cae_mod.to_native(
        {
            "rows": rows,
            "baseline_mse_per_dim": baseline_mse,
            "selected_lambda": selected_row["lambda"],
            "selection_branch": selection_branch,
            "selection_rule": LAMBDA_SELECT_RULE,
        }
    )


_cfg_key = cache.config_key(_sweep_cfg())
result = cache.json_cache(f"topoae_lambda_sweep_{_cfg_key}", _sweep_cfg(), _compute_sweep)

# =================================================================================================
_banner("CLOSING TABLE")

header = (
    f"{'lambda':>8s} {'mse_per_dim':>13s} {'mse_ratio_v0':>13s} {'loss_x_to_z':>13s} "
    f"{'loss_z_to_x':>13s} {'transfer_ratio':>15s} {'wallclock_s':>12s}"
)
print(header)
print("-" * len(header))
for row in result["rows"]:
    print(
        f"{row['lambda']:>8.3g} {row['mse_per_dim']:>13.6f} {row['mse_ratio_vs_zero']:>13.4f} "
        f"{row['loss_x_to_z']:>13.6f} {row['loss_z_to_x']:>13.6f} {row['transfer_ratio']:>15.6f} "
        f"{row['wallclock_s']:>12.1f}"
    )

print()
print(f"  selected lambda = {result['selected_lambda']}")
print(f"  selection branch: {result['selection_branch']}")
print(f"  cache stem: topoae_lambda_sweep_{_cfg_key}")
print()
print("Done.")
