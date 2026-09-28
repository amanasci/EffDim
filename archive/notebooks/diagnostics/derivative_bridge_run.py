"""Phase 02.6 decoder-substrate screening: the derivative-usability bridge, run on BOTH
regimes (SC-5; D-16, D-17, D-18, D-19, D-21).

**This script measures and prints; it applies no bar of any kind, emits no verdict token,
seals nothing and writes no artifact.** ``notebooks/pu_manifold/derivative_bridge.py``
(plan 02.6-10) is the instrument: autodiff-versus-finite-difference agreement on a trained
decoder's own second derivatives, needing no analytic ground truth, reported at BOTH the
full Hessian tensor level and the reduced mean-curvature level under permanently separate
keys (S1's never-collapse rule, ``02.6-SCREENING-RULE-02.md``). This runner is the thing
that actually calls it, on the Swiss roll AND on PU, and prints both tables separately
(D-17) -- never merged, never averaged across regimes.

**Honest limit, travelling with every number this script prints (A-01,
02.6-CONTEXT.md):** autodiff-versus-finite-difference agreement establishes that the
derivative COMPUTATION is stable. A net that learned a smooth but WRONG surface passes it
cleanly. Nothing here demonstrates that any candidate's derivatives are right -- only that
they are computable.

**D-21, exercised only if reuse provably fails.** The PU arm reuses four already-cached,
already-trained artifact families read-only: the plain-AE control at ``d=40``
(``cae_ctrl_plainae40_*``), the TopoAE fit at ``d=40``
(``topoae_fit_*_amend01_seed20260806_d40``), three CAE fits with ``z_all`` cached directly
(``cae_fit_*_seed2026080{3,4,5}``), and the PU embedding subsample those fits were trained
against. :func:`load_pu_models` raises ``FileNotFoundError`` naming the missing path if any
of these is absent -- it never trains a replacement. D-21's amendment (new PU fits permitted
for this bridge arm only, rated costly) is exercised only by that raise being caught and a
new fit deliberately started elsewhere; this script never does that itself.

**Deviation from the plan's literal precondition text, recorded here as code comment and in
the plan's SUMMARY.** The precondition named ``subsample_20260801_*`` as the PU embedding
subsample. That stem exists on disk (10000 x 768, seed=20260801) but is not the subsample
any of the four cached fit families were actually trained or split against --
``cae_train_run.py``, ``topoae_train_run.py``, ``cae_evaluate_run.py`` and
``topoae_evaluate_run.py`` (this plan's own ``<read_first>`` targets) all use
``SUBSAMPLE_STEM = "subsample_20260729_a79b3460b838fd0a"`` (seed=20260729). Using the
20260801 draw would silently encode the wrong 10000 rows against ``cae_ctrl_plainae40``'s
cached ``holdout_idx`` -- a real bug, not a cosmetic mismatch. :data:`SUBSAMPLE_STEM`
below is the verified-correct stem, cross-checked against the four sealed evaluate/train
runners rather than the plan's literal glob.

Invoke:
    .venv/bin/python notebooks/diagnostics/derivative_bridge_run.py --smoke
    .venv/bin/python notebooks/diagnostics/derivative_bridge_run.py
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pu_manifold import cache, cae, chart_curvature, curvature_probe, derivative_bridge, topoae  # noqa: E402

# --- Swiss roll fixture / shared plain-AE + TopoAE model constants, carried forward
# VERBATIM from decoder_substrate_ph_screen_run.py (plan 02.6-11) -- a runner number here
# and a runner number there at the same seed share the identical fixture and the identical
# plain-AE/TopoAE fit protocol.
FIXTURE_SEED = 20260807
N_POINTS = 3000
LATENT_DIM = 2
HIDDEN = (64, 64, 64)
LAMBDA_TOPO = 0.1
HOLDOUT_FRACTION = 0.2

# CRITICAL, per this plan's own phase-context note: read from
# decoder_substrate_ph_screen_run.py's CFG_COMMON, NOT the plan text's stale
# lr=3e-4/max_epochs=150 config. That config left every plainae/topoae seed hitting its
# epoch ceiling still improving while cae converged -- exactly the confound
# 02.6-11-SUMMARY.md's two protocol-repair rounds fixed. This is ONE shared convergence
# rule for every roll-arm candidate; main() prints a WARNING (not a raise) for any fit that
# hits max_epochs without early stopping, since a ceiling hit here would be that same
# confound recurring on this axis.
CFG_COMMON = dict(
    lr=1e-3,
    weight_decay=1e-4,
    batch=64,
    max_epochs=800,
    early_stop_patience=25,
    early_stop_min_delta=1e-4,
)

DEFAULT_SEEDS = (0, 1, 2, 3)  # ratified seed protocol, 02.6-SCREENING-RULE-02.md

# The CAE's own architecture and configuration, reused VERBATIM from
# notebooks/02.2_swiss_roll_cae_check.ipynb via decoder_substrate_ph_screen_run.py -- not
# re-derived, not tuned for this bridge.
CAE_EMBED_DIM = 8
CAE_N_CHARTS = 8
CAE_HIDDEN = (64, 64)
CFG_CAE = dict(
    lr=1e-3,
    weight_decay=1e-4,
    batch=64,
    max_epochs=800,
    early_stop_patience=25,
    early_stop_min_delta=1e-4,
    fps_pretrain_epochs=20,
    lip_weight=1e-3,
    lip_every_n_steps=1,
)

# --- PU-regime cached artifact stems --------------------------------------------------------

PU_FIT_KEY = "43cf438bc944c509"
SUBSAMPLE_STEM = "subsample_20260729_a79b3460b838fd0a"  # see module docstring's Deviation note
PLAINAE40_STEM = f"cae_ctrl_plainae40_{PU_FIT_KEY}"
TOPOAE_D40_STEM = f"topoae_fit_{PU_FIT_KEY}_amend01_seed20260806_d40"
TOPOAE_SPLIT_STEM = f"topoae_split_{PU_FIT_KEY}"
CAE_PU_SEEDS = (20260803, 20260804, 20260805)
AMBIENT_DIM = 768
PU_EMBED_DIM = 40
PU_CHART_DIM = 20
PU_N_CHARTS_INIT = 16
PU_HIDDEN_WIDTH = 250
PU_ACTIVATION = "silu"

PU_SUBSAMPLE_SEED = 20260811
"""Seeds the bounded PU-regime evaluation subsample (see BRIDGE_N_POINTS below) --
distinct from every fit/split seed already in use, so this script's own reproducibility is
never accidentally coupled to a training seed."""

# --- the bridge's own constants (Task 1 requirement) ----------------------------------------

BRIDGE_N_POINTS = 32
"""The PU regime's bounded, seeded evaluation-point budget. The stencil
(derivative_bridge.FD_STENCIL_CONVENTION) costs ``1 + 2*d + 4*d*(d-1)/2`` decoder
evaluations per point: 9 at the Swiss roll's plain-AE/TopoAE d=2, 3201 at the PU plain-AE/
TopoAE control's d=40, 801 at the PU CAE's per-chart d=20. The roll arm's held-out split
(~600 points at the default N_POINTS=3000) is cheap enough to evaluate in full -- every
plainae/topoae bridge_one call below passes the WHOLE held-out set, never a further
subsample. The PU arm is not: 3201 evals/point x BRIDGE_N_POINTS x 7 calibration-ladder
steps is the reason this is a bounded, seeded draw rather than the full 2000-row holdout.
CAE chart groups partition this same budget across whichever charts are actually assigned
points by the model's own routing -- summing the group sizes never exceeds BRIDGE_N_POINTS,
so partitioning into charts costs nothing extra."""

FD_STEP_LADDER = derivative_bridge.DEFAULT_CALIBRATION_STEPS
"""The decade ladder handed to calibrate_fd_step -- imported, not re-derived, from the
sealed module's own measured U-shape (02.6-RESEARCH.md Pitfall 5)."""

TABLE_COLUMNS = (
    "candidate",
    "run_id",
    "chart_idx",
    "latent_dim",
    "n_points",
    "fd_step_used",
    "full_hess_max_abs",
    "full_hess_median_abs",
    "full_hess_p90_abs",
    "full_hess_max_abs_rel",
    "Hvec_max_abs",
    "Hvec_median_abs",
    "Hvec_p90_abs",
    "Hvec_max_abs_rel",
    "Hnorm_max_abs",
    "Hnorm_median_abs",
    "Hnorm_p90_abs",
    "Hnorm_max_abs_rel",
    "metric_cond_max",
    "activation",
    "wallclock_s",
)


class _ChartDecodeModelView:
    """Minimal duck-typed view over a CAE chart's two-hop decode, satisfying
    ``derivative_bridge.derivative_agreement``'s internal dispatch
    (``assert_c2_decoder``'s has-``.activation`` branch, ``plain_decoder_map``'s
    ``model.decode``, ``_assert_float64``'s ``model.parameters()``). Built OVER an
    already-pinned ``decode_batch`` callable (:func:`decode_closure`) rather than
    re-deriving the composition. plainae/topoae candidates need no wrapping at all -- a real
    ``cae.PlainAutoEncoder`` already satisfies this shape natively and is passed to
    ``derivative_agreement`` directly."""

    def __init__(self, decode_batch, activation, chart_decoder, embedding_decoder):
        self._decode_batch = decode_batch
        self.activation = activation
        self._chart_decoder = chart_decoder
        self._embedding_decoder = embedding_decoder

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode_batch(z)

    def parameters(self):
        return itertools.chain(
            self._chart_decoder.parameters(), self._embedding_decoder.parameters()
        )


# --- the five (six, counting this module's own constant block) required functions -------


def build_swiss_roll_models(seed: int, n: int = N_POINTS, max_epochs: int = None) -> dict:
    """Train plainae, topoae and cae from scratch on the shared Swiss roll fixture at one
    torch seed, mirroring ``decoder_substrate_ph_screen_run.py``'s ``train_candidate``
    exactly: the same architecture constants, the same repaired ``CFG_COMMON``/``CFG_CAE``
    convergence rule, the same 80/20 holdout split via
    ``np.random.default_rng(seed).permutation``.

    Returns each model evaluated and cast to float64, plus the latent points the bridge
    evaluates at: the FULL held-out split for plainae/topoae (the roll affords every
    held-out point at d=2 -- 9 decoder evaluations per point), and the CAE's global
    embedding of that same held-out split (chart-grouped by the caller via
    :func:`_cae_chart_groups`, since each chart needs its own decode map).

    Prints a WARNING, never raises, for any candidate that hits ``max_epochs`` without
    early stopping -- the shared convergence rule is repaired precisely so every fit ends
    by early stopping (02.6-11-SUMMARY.md "Protocol Repairs"), and a ceiling hit here would
    be that same confound recurring on this axis; it must be reported, not silently
    accepted (the plan's own instruction).
    """
    if max_epochs is not None:
        CFG_COMMON["max_epochs"] = max_epochs
        CFG_CAE["max_epochs"] = max_epochs

    fx = curvature_probe.make_swiss_roll_fixture(n=n, seed=FIXTURE_SEED)
    x_all = torch.tensor(fx["X"], dtype=torch.float32)
    n_total = x_all.shape[0]
    perm = np.random.default_rng(seed).permutation(n_total)
    n_train = int(n_total * (1 - HOLDOUT_FRACTION))
    train_idx, holdout_idx = perm[:n_train], perm[n_train:]
    x_train = x_all[torch.from_numpy(train_idx)]
    x_holdout = x_all[torch.from_numpy(holdout_idx)]

    torch.manual_seed(seed)
    plainae_model = cae.PlainAutoEncoder(3, LATENT_DIM, hidden=HIDDEN, activation="silu")
    plainae_fit = cae.train_plain_ae(plainae_model, x_train, dict(seed=seed, **CFG_COMMON))

    torch.manual_seed(seed)
    topoae_model = cae.PlainAutoEncoder(3, LATENT_DIM, hidden=HIDDEN, activation="silu")
    topoae_fit = topoae.train_topoae(
        topoae_model,
        x_train,
        dict(seed=seed, lambda_topo=LAMBDA_TOPO, warmup_frac=0.25, ramp_frac=0.25, **CFG_COMMON),
    )

    torch.manual_seed(seed)
    cae_model = cae.ChartAutoEncoder(
        in_dim=3,
        embed_dim=CAE_EMBED_DIM,
        chart_dim=LATENT_DIM,
        n_charts=CAE_N_CHARTS,
        hidden=CAE_HIDDEN,
        activation="silu",
    )
    cae_fit = cae.train_cae(cae_model, x_train, dict(seed=seed, n_charts=CAE_N_CHARTS, **CFG_CAE))

    for name, fit in (("plainae", plainae_fit), ("topoae", topoae_fit), ("cae", cae_fit)):
        if not fit["early_stopped"]:
            print(
                f"  WARNING: [roll seed={seed}] {name} hit max_epochs={fit['epochs_run']} "
                "without early stopping -- reported explicitly rather than accepted "
                "(see 02.6-11-SUMMARY.md 'Protocol Repairs' for why this matters)."
            )

    plainae_model.eval().double()
    topoae_model.eval().double()
    cae_model.eval().double()
    x_holdout64 = x_holdout.double()
    with torch.no_grad():
        plainae_z = plainae_model.encode(x_holdout64)
        topoae_z = topoae_model.encode(x_holdout64)
        cae_z_global = cae_model.encode(x_holdout64)

    return {
        "plainae": {"model": plainae_model, "fit": plainae_fit, "z_holdout": plainae_z},
        "topoae": {"model": topoae_model, "fit": topoae_fit, "z_holdout": topoae_z},
        "cae": {"model": cae_model, "fit": cae_fit, "z_holdout_global": cae_z_global},
    }


def load_pu_models() -> dict:
    """Rebuild each PU candidate from its cached fit, READ-ONLY (D-21). Returns
    ``{"plainae": {...}, "topoae": {...}, "cae": {seed: {...}}, "cae_holdout_idx": array}``.

    Every artifact opened is opened for reading only via ``np.load``/``cache.cache_path``
    (never ``cache.npz_cache``/``json_cache``, which write on a miss); the cache
    directory's entry count is asserted unchanged before and after this function runs
    (T-02.6R-24). Raises ``FileNotFoundError`` naming the missing path if any required
    family is absent -- D-21 permits a new PU fit only after THIS raises, never as a silent
    fallback taken by this function itself.

    The plain-AE control (``cae_ctrl_plainae40``) has no cached ``z_all``, so its holdout
    points are obtained by encoding the cached ambient subsample at its own cached
    ``holdout_idx`` -- the only PU family this function re-encodes rather than reading a
    cached latent directly. The TopoAE fit and all three CAE fits carry ``z_all`` cached
    directly and are never re-encoded. The CAE's holdout split is not cached per-seed;
    ``cae_train_run.py``'s own ``SPLIT_SEED=20260803`` is a FIXED constant shared by every
    fit under this fit_key regardless of the fit's own training seed (verified against
    source: ``cae_ctrl_plainae40``'s cached ``holdout_idx`` and ``split_seed=20260803``
    field), so that same cached array is reused as the canonical holdout for every CAE seed
    rather than re-deriving ``SPLIT_SEED`` here a second time.
    """
    cache_dir = cache.CACHE_DIR
    n_before = sum(1 for _ in cache_dir.iterdir())

    def _require(stem: str, ext: str) -> Path:
        p = cache.cache_path(stem, ext)
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(
                f"load_pu_models: required cached PU artifact missing or empty: {p}. D-21 "
                "permits a new PU fit only after reuse has provably failed -- this halts "
                "rather than training a replacement."
            )
        return p

    # --- plain-AE control at d=40 --------------------------------------------------------
    plainae_npz = dict(np.load(_require(PLAINAE40_STEM, "npz")))
    plainae_model = cae.PlainAutoEncoder(
        AMBIENT_DIM, PU_EMBED_DIM, hidden=(PU_HIDDEN_WIDTH,) * 3, activation=PU_ACTIVATION
    )
    plainae_model.load_state_dict(cae.arrays_to_state_dict(plainae_npz, plainae_model.state_dict()))
    plainae_model.eval().double()
    plainae_holdout_idx = plainae_npz["holdout_idx"]

    # --- TopoAE fit at d=40 ---------------------------------------------------------------
    topoae_npz = dict(np.load(_require(TOPOAE_D40_STEM, "npz")))
    topoae_model = cae.PlainAutoEncoder(
        AMBIENT_DIM, PU_EMBED_DIM, hidden=(PU_HIDDEN_WIDTH,) * 3, activation=PU_ACTIVATION
    )
    topoae_model.load_state_dict(cae.arrays_to_state_dict(topoae_npz, topoae_model.state_dict()))
    topoae_model.eval().double()
    topoae_z_all = topoae_npz["z_all"]  # (10000, 40) -- cached, never re-encoded
    topoae_holdout_idx = dict(np.load(_require(TOPOAE_SPLIT_STEM, "npz")))["holdout_idx"]

    # --- three CAE fits, sharing ONE fixed split (SPLIT_SEED=20260803) --------------------
    cae_fits = {}
    for seed in CAE_PU_SEEDS:
        cae_npz = dict(np.load(_require(f"cae_fit_{PU_FIT_KEY}_seed{seed}", "npz")))
        cae_model = cae.ChartAutoEncoder(
            in_dim=AMBIENT_DIM,
            embed_dim=PU_EMBED_DIM,
            chart_dim=PU_CHART_DIM,
            n_charts=PU_N_CHARTS_INIT,
            hidden=(PU_HIDDEN_WIDTH,) * 3,
            activation=PU_ACTIVATION,
        )
        cae_model.load_state_dict(cae.arrays_to_state_dict(cae_npz, cae_model.state_dict()))
        cae_model.eval().double()
        cae_fits[seed] = {"model": cae_model, "z_all": cae_npz["z_all"]}

    # --- re-encode plainae40's own cached holdout rows (no z_all cached for this control) -
    subsample = dict(np.load(_require(SUBSAMPLE_STEM, "npz")))
    x_holdout = torch.tensor(
        subsample["legacysurvey"][plainae_holdout_idx], dtype=torch.float64
    )
    with torch.no_grad():
        plainae_z_holdout = plainae_model.encode(x_holdout).numpy()

    n_after = sum(1 for _ in cache_dir.iterdir())
    if n_after != n_before:
        raise RuntimeError(
            f"load_pu_models: notebooks/.cache entry count changed ({n_before} -> "
            f"{n_after}) -- this function must never write to the cache directory it "
            "reads from (T-02.6R-24)."
        )
    print(f"  notebooks/.cache entry count unchanged: {n_before} (read-only confirmed)")

    return {
        "plainae": {"model": plainae_model, "z_holdout": plainae_z_holdout},
        "topoae": {"model": topoae_model, "z_all": topoae_z_all, "holdout_idx": topoae_holdout_idx},
        "cae": cae_fits,
        "cae_holdout_idx": plainae_holdout_idx,
    }


def decode_closure(model, chart_idx: int = None):
    """Return ``decode_batch``: ``(batch, d) -> (batch, out_dim)``, the batched
    ``.decode``-shaped map the bridge differentiates.

    For the two plain-autoencoder-family candidates (``chart_idx=None``) this is
    ``model.decode`` itself. For the CAE (``chart_idx`` given) it is the two-hop chart
    decode -- chart ``chart_idx``'s own decoder followed by the single shared embedding
    decoder, batched -- the same composition ``chart_curvature.chart_decoder_map`` builds
    one point at a time, and deliberately not ``cae.py``'s own encoder-recomputing chart
    decode helper (see ``chart_curvature.chart_decoder_map``'s own docstring for the exact
    deviation this mirrors), which would differentiate through the chart-coordinate-producing
    encoder as well and measure the wrong map.

    **Pinned against the sealed single-point map (T-02.6R-25) before returning**: evaluated
    at a handful of chart coordinates and asserted to agree with
    ``chart_curvature.chart_decoder_map`` to below ``1e-12``, so a batched rewrite cannot
    silently drift from the sealed composition it mirrors.
    """
    if chart_idx is None:
        return model.decode

    chart_decoder = model.chart_decoders[chart_idx]
    embedding_decoder = model.embedding_decoder

    def decode_batch(zc: torch.Tensor) -> torch.Tensor:
        return embedding_decoder(chart_decoder(zc))

    sealed_decode_one = chart_curvature.chart_decoder_map(model, chart_idx)
    probe = torch.randn(6, model.chart_dim, dtype=torch.float64)
    with torch.no_grad():
        batched_out = decode_batch(probe)
        sealed_out = torch.stack([sealed_decode_one(p) for p in probe], dim=0)
    max_diff = float((batched_out - sealed_out).abs().max().item())
    if not max_diff < 1e-12:
        raise ValueError(
            f"decode_closure: batched CAE chart={chart_idx} decode diverges from the "
            f"sealed chart_curvature.chart_decoder_map by {max_diff:.3e}, exceeding the "
            "1e-12 pin (T-02.6R-25). A batched rewrite must reproduce the sealed "
            "single-point composition exactly."
        )
    return decode_batch


def bridge_one(model, z: torch.Tensor, label: str, chart_idx: int = None) -> dict:
    """One bridge measurement (D-16, D-17, D-18). Calibrates the finite-difference step for
    THIS model at THESE points via ``derivative_bridge.calibrate_fd_step`` over
    :data:`FD_STEP_LADDER` (never the sealed module's own uncalibrated module-level default
    step, trusted blind -- A-06: the PU decoders' weight/activation scales have never been
    measured against the toy fixture that default was tuned on), prints the full ladder,
    then calls
    ``derivative_bridge.derivative_agreement`` at the calibrated step. Returns its dict plus
    ``label``, ``chart_idx``, ``latent_dim``, ``n_points``, ``wallclock_s`` and
    ``calibration_ladder``.
    """
    decode_batch = decode_closure(model, chart_idx)
    if chart_idx is None:
        agreement_model = model
    else:
        agreement_model = _ChartDecodeModelView(
            decode_batch, model.activation, model.chart_decoders[chart_idx], model.embedding_decoder
        )

    calib = derivative_bridge.calibrate_fd_step(decode_batch, z, steps=FD_STEP_LADDER)
    ladder_str = "  ".join(f"{h:.0e}->{e:.3e}" for h, e in calib["step_errors"].items())
    print(f"    [{label}] calibration ladder: {ladder_str}")
    print(f"    [{label}] best_step={calib['best_step']:.1e}  best_error={calib['best_error']:.3e}")

    t0 = time.monotonic()
    result = derivative_bridge.derivative_agreement(agreement_model, z, h=calib["best_step"])
    wallclock_s = time.monotonic() - t0

    row = dict(result)
    row["label"] = label
    row["chart_idx"] = chart_idx
    row["latent_dim"] = int(z.shape[1])
    row["n_points"] = int(z.shape[0])
    row["wallclock_s"] = wallclock_s
    row["calibration_ladder"] = calib["step_errors"]
    print(
        f"    [{label}] full_hessian max_abs={row['full_hessian_agreement']['max_abs']:.3e}  "
        f"H_vec max_abs={row['reduced_mean_curvature_agreement']['H_vec']['max_abs']:.3e}  "
        f"H_norm max_abs={row['reduced_mean_curvature_agreement']['H_norm']['max_abs']:.3e}  "
        f"n_points={row['n_points']}  wallclock_s={wallclock_s:.1f}"
    )
    return row


def markdown_table(rows) -> str:
    """A GitHub-flavoured markdown table over :data:`TABLE_COLUMNS` -- one row per bridge
    measurement (one per plainae/topoae seed, one per CAE (seed, surviving-in-sample chart)
    pair). Called once per regime by :func:`main`; the two regimes' tables are NEVER merged
    (D-17) and this function never sees rows from both at once."""
    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"
    lines = [header, sep]
    for row in rows:
        fh = row["full_hessian_agreement"]
        hv = row["reduced_mean_curvature_agreement"]["H_vec"]
        hn = row["reduced_mean_curvature_agreement"]["H_norm"]
        cond = row["metric_condition_number"]
        cond_max = float(cond.max().item()) if hasattr(cond, "max") else float(cond)
        cells = [
            row["candidate"],
            str(row["run_id"]),
            str(row["chart_idx"]) if row["chart_idx"] is not None else "-",
            str(row["latent_dim"]),
            str(row["n_points"]),
            f"{row['fd_step_used']:.1e}",
            f"{fh['max_abs']:.4e}",
            f"{fh['median_abs']:.4e}",
            f"{fh['p90_abs']:.4e}",
            f"{fh['max_abs_relative']:.4e}",
            f"{hv['max_abs']:.4e}",
            f"{hv['median_abs']:.4e}",
            f"{hv['p90_abs']:.4e}",
            f"{hv['max_abs_relative']:.4e}",
            f"{hn['max_abs']:.4e}",
            f"{hn['median_abs']:.4e}",
            f"{hn['p90_abs']:.4e}",
            f"{hn['max_abs_relative']:.4e}",
            f"{cond_max:.4e}",
            str(row["activation"]),
            f"{row['wallclock_s']:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n", type=int, default=N_POINTS)
    parser.add_argument("--max-epochs", type=int, default=CFG_COMMON["max_epochs"])
    parser.add_argument("--pu-n", type=int, default=BRIDGE_N_POINTS)
    parser.add_argument(
        "--pu-candidates",
        nargs="+",
        choices=("plainae", "topoae", "cae"),
        default=["plainae", "topoae", "cae"],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast path: one roll seed at n=300/max_epochs=4, one PU candidate (plainae) at "
            "pu_n=4. Exercises every code path -- roll training for all three candidates, "
            "the PU reload path, the pinned CAE chart closure, calibration, and both "
            "tables -- well under 180 seconds. This is the runner's automated <verify>."
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        args.seeds = args.seeds[:1]
        args.n = 300
        args.max_epochs = 4
        args.pu_n = 4
        args.pu_candidates = ["plainae"]

    CFG_COMMON["max_epochs"] = args.max_epochs
    CFG_CAE["max_epochs"] = args.max_epochs

    def _seeded_subsample_tensor(z_array: np.ndarray, n: int, seed: int) -> torch.Tensor:
        """A bounded, seeded subsample of at most ``n`` rows from ``z_array``, returned as
        a float64 tensor. See :data:`BRIDGE_N_POINTS` for the cost arithmetic this bound
        exists to respect. Nested here (rather than a module-level helper) so this
        runner's own required top-level function count stays exactly six."""
        n_avail = z_array.shape[0]
        n_take = min(n, n_avail)
        idx = np.random.default_rng(seed).choice(n_avail, size=n_take, replace=False)
        return torch.tensor(z_array[idx], dtype=torch.float64)

    def _seeded_subsample_rows(index_array: np.ndarray, n: int, seed: int) -> np.ndarray:
        """Same draw as ``_seeded_subsample_tensor`` above, but returning the SELECTED
        indices into ``index_array`` rather than the gathered rows -- used where the
        caller still needs the underlying dataset row numbers (the CAE arm, which groups
        by chart afterwards)."""
        n_avail = index_array.shape[0]
        n_take = min(n, n_avail)
        pick = np.random.default_rng(seed).choice(n_avail, size=n_take, replace=False)
        return index_array[pick]

    def _cae_chart_groups(model: "cae.ChartAutoEncoder", z_global: torch.Tensor):
        """Group ``z_global``'s rows (the CAE's GLOBAL ``embed_dim`` embedding, per
        ratified R1) by their own argmax chart -- mirrors
        ``cae.ChartAutoEncoder.reconstruct``'s own routing, never a different assignment
        rule. Returns ``{chart_idx: chart_coords_group}``, one entry per chart actually
        assigned at least one row, each value shaped ``(n_i, chart_dim)`` and float64.
        Charts assigned zero rows in this sample are simply absent from the returned
        dict -- not every chart the model has need appear."""
        with torch.no_grad():
            chart_idx_per_point = model.chart_probs(z_global).argmax(dim=1)
            z_charts = model.chart_coords(z_global)  # (n, n_charts, chart_dim)
        groups = {}
        for c in range(model.n_charts):
            mask = chart_idx_per_point == c
            count = int(mask.sum().item())
            if count > 0:
                groups[c] = z_charts[mask, c, :].detach()
        return groups

    print("=" * 78)
    print("Phase 02.6 derivative-usability bridge runner (SC-5; D-16, D-17, D-18, D-21)")
    print(
        "MEASURES AND PRINTS ONLY. Applies no bar of any kind, emits no boolean judgement, "
        "no ranking -- 02.5-10 is the only place a bound is ever ratified against these "
        "numbers."
    )
    print(
        f"roll seeds={args.seeds}  roll n={args.n}  roll max_epochs={args.max_epochs}  "
        f"pu_n={args.pu_n}  pu_candidates={args.pu_candidates}"
    )
    print("=" * 78)

    print()
    print(
        "HONEST LIMIT (A-01, 02.6-CONTEXT.md): this instrument establishes that the "
        "derivative COMPUTATION is stable -- autodiff and finite difference agree with "
        "each other -- not that the surface is right. A net that learned a smooth but "
        "WRONG surface passes it cleanly. Nothing below demonstrates that any candidate's "
        "derivatives are RIGHT, only that they are COMPUTABLE."
    )

    # ==================================================================================
    print()
    print("=" * 78)
    print("SWISS ROLL ARM")
    print("=" * 78)
    roll_rows = []
    for seed in args.seeds:
        print(f"\n[roll seed={seed}] training plainae, topoae, cae from scratch...")
        built = build_swiss_roll_models(seed, n=args.n, max_epochs=args.max_epochs)

        for cand in ("plainae", "topoae"):
            model = built[cand]["model"]
            z = built[cand]["z_holdout"]
            row = bridge_one(model, z, label=f"{cand}|roll|seed={seed}")
            row.update({"candidate": cand, "regime": "swiss_roll", "run_id": seed})
            roll_rows.append(row)

        cae_model = built["cae"]["model"]
        z_global = built["cae"]["z_holdout_global"]
        groups = _cae_chart_groups(cae_model, z_global)
        print(f"    [cae|roll|seed={seed}] {len(groups)} chart(s) assigned >=1 held-out point")
        for chart_idx, z_chart in groups.items():
            row = bridge_one(
                cae_model, z_chart, label=f"cae|roll|seed={seed}|chart={chart_idx}", chart_idx=chart_idx
            )
            row.update({"candidate": "cae", "regime": "swiss_roll", "run_id": seed})
            roll_rows.append(row)

    # ==================================================================================
    print()
    print("=" * 78)
    print("PU ARM (D-21: read-only reuse; raises rather than training a replacement)")
    print("=" * 78)
    pu_rows = []
    pu_models = load_pu_models()
    n_pu = args.pu_n
    print(
        f"  bounded seeded subsample: pu_n={n_pu} points per family, PU_SUBSAMPLE_SEED="
        f"{PU_SUBSAMPLE_SEED}"
    )
    print(
        "  PU seed availability: plainae40=1 cached fit, topoae_d40=1 cached fit "
        f"(seed20260806), cae={len(CAE_PU_SEEDS)} cached fits (seeds {CAE_PU_SEEDS}) -- "
        "the roll arm's four seeds per candidate is a construction choice (D-11); the PU "
        "arm's seed count is whatever the cache holds, stated here rather than left to "
        "infer."
    )

    if "plainae" in args.pu_candidates:
        z = _seeded_subsample_tensor(pu_models["plainae"]["z_holdout"], n_pu, PU_SUBSAMPLE_SEED)
        row = bridge_one(pu_models["plainae"]["model"], z, label="plainae|PU")
        row.update({"candidate": "plainae", "regime": "PU", "run_id": "plainae40"})
        pu_rows.append(row)

    if "topoae" in args.pu_candidates:
        topoae_z_holdout = pu_models["topoae"]["z_all"][pu_models["topoae"]["holdout_idx"]]
        z = _seeded_subsample_tensor(topoae_z_holdout, n_pu, PU_SUBSAMPLE_SEED)
        row = bridge_one(pu_models["topoae"]["model"], z, label="topoae|PU")
        row.update({"candidate": "topoae", "regime": "PU", "run_id": "topoae_d40_seed20260806"})
        pu_rows.append(row)

    if "cae" in args.pu_candidates:
        cae_holdout_idx = pu_models["cae_holdout_idx"]
        for seed, fit in pu_models["cae"].items():
            rows_idx = _seeded_subsample_rows(cae_holdout_idx, n_pu, PU_SUBSAMPLE_SEED)
            z_global = torch.tensor(fit["z_all"][rows_idx], dtype=torch.float64)
            groups = _cae_chart_groups(fit["model"], z_global)
            print(f"    [cae|PU|seed={seed}] {len(groups)} chart(s) assigned >=1 sampled point")
            for chart_idx, z_chart in groups.items():
                row = bridge_one(
                    fit["model"], z_chart, label=f"cae|PU|seed={seed}|chart={chart_idx}", chart_idx=chart_idx
                )
                row.update({"candidate": "cae", "regime": "PU", "run_id": f"cae_seed{seed}"})
                pu_rows.append(row)

    # ==================================================================================
    print()
    print("=" * 78)
    print("Swiss roll table (D-17: reported separately from PU, never merged):")
    print("=" * 78)
    print(markdown_table(roll_rows))

    print()
    print("=" * 78)
    print("PU table (D-17: reported separately from the Swiss roll, never merged):")
    print("=" * 78)
    print(markdown_table(pu_rows))

    print()
    print("=" * 78)
    print(
        "MEASUREMENTS ONLY. Applies no bar of any kind, emits no boolean judgement, no "
        "ranking. D-19: a substrate that tops the persistent-homology ranking but fails "
        "this bridge is not promoted -- 02.6 hands 02.5-10 both results and 02.5-10 "
        "decides what it can use."
    )
    print("=" * 78)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
