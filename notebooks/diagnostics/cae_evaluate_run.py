"""Phase 02.2's Chart Auto-Encoder evaluation runner.

Computes every pre-registered metric (T1 geodesic distortion, T2 chart-transition cycle
residual, T3 held-out reconstruction versus matched-capacity baselines) from the eight
fits plan 02.2-05 already trained and cached, applies the tested strict-less-than verdict
rule (``cae.verdict_from_metrics``, never re-derived inline here), and writes the phase's
terminal artifact: ``cae_verdict_{FIT_KEY}.json``.

Precondition: every cae_fit_*/cae_ctrl_* npz+meta pair plan 02.2-05 produced, plus
``geometry_probes_{FIT_KEY}.json`` (02.1's own measured artifact), must already exist
under notebooks/.cache/ (gitignored, irreproducible here) -- this script halts rather
than regenerating or refitting anything; there is no regeneration path. Importing
``cae_train_run`` re-runs that runner's own preconditions (CAE-01 ancestry, pair-sample
redraw/verification) and its fit registry -- a pure cache-hit read of the same eight
artifacts when everything is already trained, not a re-fit.

Invoke: PYTHONPATH=notebooks python notebooks/diagnostics/cae_evaluate_run.py
"""

import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import scipy
import torch

from pu_manifold import cache
from pu_manifold import cae as cae_mod
from pu_manifold import geometry_probes as gp

from cae_train_run import (
    ACTIVATION,
    D_CHART,
    HOLDOUT_FRACTION,
    L_EMBED,
    MAIN_SEED,
    MDS_BASELINE_P,
    MDS_EIGENSPECTRUM_STEM,
    OVERLAP_MIN_POINTS,
    OVERLAP_P_MIN,
    PAIR_COUNT,
    PAIR_SEED,
    PLAIN_AE_LATENTS,
    PREREG_PATH,
    PRUNE_TOL,
    SEEDS,
    SPLIT_SEED,
    SUBSAMPLE_STEM,
    THRESH_DISTORTION_MAX,
    THRESH_RCYCLE_RATIO_MAX,
    THRESH_RECON_MARGIN,
    UNFAITHFUL_SAMPLES,
    UNFAITHFUL_SEED,
    FIT_KEY,
    build_cae,
    build_mlp_decoder,
    build_plain_ae,
)

GEOMETRY_PROBES_STEM = f"geometry_probes_{FIT_KEY}"


def _banner(msg: str) -> None:
    print()
    print("=" * 78)
    print(msg)
    print("=" * 78)


def _load_npz(stem: str) -> dict:
    return dict(np.load(cache.cache_path(stem, "npz")))


def _load_json(stem: str) -> dict:
    return json.loads(cache.cache_path(stem, "json").read_text())


# =============================================================================================
_banner(f"Phase 02.2 Chart Auto-Encoder EVALUATION runner -- fit_key = {FIT_KEY}")

# =============================================================================================
_banner("STEP 0a -- precondition: every plan 02.2-05 fit artifact + geometry_probes json exist")

_required_stems = (
    [(f"cae_fit_{FIT_KEY}_seed{seed}", "npz") for seed in SEEDS]
    + [(f"cae_fit_meta_{FIT_KEY}_seed{seed}", "json") for seed in SEEDS]
    + [(f"cae_ctrl_relu_{FIT_KEY}", "npz"), (f"cae_ctrl_relu_meta_{FIT_KEY}", "json")]
    + [(f"cae_ctrl_plainae{latent}_{FIT_KEY}", "npz") for latent in PLAIN_AE_LATENTS]
    + [(f"cae_ctrl_plainae{latent}_meta_{FIT_KEY}", "json") for latent in PLAIN_AE_LATENTS]
    + [(f"cae_ctrl_mdsdec{p}_{FIT_KEY}", "npz") for p in MDS_BASELINE_P]
    + [(f"cae_ctrl_mdsdec{p}_meta_{FIT_KEY}", "json") for p in MDS_BASELINE_P]
    + [(f"cae_pairs_{FIT_KEY}", "npz")]
    + [(GEOMETRY_PROBES_STEM, "json")]
)
for _stem, _ext in _required_stems:
    _path = cache.cache_path(_stem, _ext)
    if not _path.exists() or _path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Required plan 02.2-05 (or 02.1) cache artifact missing or empty: {_path}. This "
            "script halts rather than regenerating or refitting it -- there is no "
            "regeneration path in this runner."
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
        "(pre-registration before any fit, before any evaluation)."
    )
_prereg_sha = _add_commits[-1]
_ancestor_check = subprocess.run(["git", "merge-base", "--is-ancestor", _prereg_sha, "HEAD"])
if _ancestor_check.returncode != 0:
    raise RuntimeError(
        f"CAE-01 ordering violated: pre-registration commit {_prereg_sha} is NOT an "
        "ancestor of HEAD. Halting before computing any metric."
    )
print(f"  pre-registration commit {_prereg_sha} confirmed an ancestor of HEAD")

# =============================================================================================
_banner("STEP 1 -- reload: rebuild each model, load fit-artifact arrays, assert split identity")

_seed_npz = {seed: _load_npz(f"cae_fit_{FIT_KEY}_seed{seed}") for seed in SEEDS}
_seed_meta = {seed: _load_json(f"cae_fit_meta_{FIT_KEY}_seed{seed}") for seed in SEEDS}
_relu_npz = _load_npz(f"cae_ctrl_relu_{FIT_KEY}")
_relu_meta = _load_json(f"cae_ctrl_relu_meta_{FIT_KEY}")
_plainae_npz = {latent: _load_npz(f"cae_ctrl_plainae{latent}_{FIT_KEY}") for latent in PLAIN_AE_LATENTS}
_plainae_meta = {
    latent: _load_json(f"cae_ctrl_plainae{latent}_meta_{FIT_KEY}") for latent in PLAIN_AE_LATENTS
}
_mdsdec_npz = {p: _load_npz(f"cae_ctrl_mdsdec{p}_{FIT_KEY}") for p in MDS_BASELINE_P}
_mdsdec_meta = {p: _load_json(f"cae_ctrl_mdsdec{p}_meta_{FIT_KEY}") for p in MDS_BASELINE_P}

# T-02.2-23 mitigation: assert the persisted train/holdout split index arrays are
# bit-identical across all eight loaded fits before computing anything.
_all_npz_for_split = (
    list(_seed_npz.values()) + [_relu_npz] + list(_plainae_npz.values()) + list(_mdsdec_npz.values())
)
_canonical_train_idx = _all_npz_for_split[0]["train_idx"]
_canonical_holdout_idx = _all_npz_for_split[0]["holdout_idx"]
for _npz in _all_npz_for_split[1:]:
    if not np.array_equal(_npz["train_idx"], _canonical_train_idx) or not np.array_equal(
        _npz["holdout_idx"], _canonical_holdout_idx
    ):
        raise RuntimeError(
            "Split index mismatch across the eight fit artifacts (T-02.2-23) -- a metric "
            "computed across mismatched splits would silently mean nothing. Halting."
        )
train_idx = _canonical_train_idx
holdout_idx = _canonical_holdout_idx
print(
    f"  split index arrays identical across all eight loaded runs: "
    f"train={train_idx.shape[0]} holdout={holdout_idx.shape[0]}"
)

_subsample = _load_npz(SUBSAMPLE_STEM)
X = _subsample["legacysurvey"]
x_all_t = torch.tensor(X, dtype=torch.float32)
x_train_t = x_all_t[torch.from_numpy(train_idx)]
x_holdout_t = x_all_t[torch.from_numpy(holdout_idx)]


def _rebuild_cae(npz: dict, activation: str) -> "cae_mod.ChartAutoEncoder":
    model = build_cae(activation)
    model.load_state_dict(cae_mod.arrays_to_state_dict(npz, model.state_dict()))
    model.eval()
    return model


def _rebuild_plain_ae(npz: dict, latent: int) -> "cae_mod.PlainAutoEncoder":
    model = build_plain_ae(latent)
    model.load_state_dict(cae_mod.arrays_to_state_dict(npz, model.state_dict()))
    model.eval()
    return model


def _rebuild_mlp_decoder(npz: dict, latent: int) -> "cae_mod.MlpDecoder":
    model = build_mlp_decoder(latent)
    model.load_state_dict(cae_mod.arrays_to_state_dict(npz, model.state_dict()))
    model.eval()
    return model


seed_models = {seed: _rebuild_cae(_seed_npz[seed], ACTIVATION) for seed in SEEDS}
relu_model = _rebuild_cae(_relu_npz, "relu")
plainae_models = {latent: _rebuild_plain_ae(_plainae_npz[latent], latent) for latent in PLAIN_AE_LATENTS}
mdsdec_models = {p: _rebuild_mlp_decoder(_mdsdec_npz[p], p) for p in MDS_BASELINE_P}
model_main = seed_models[MAIN_SEED]
print(
    f"  rebuilt {len(seed_models)} CAE seeds + 1 ReLU control + {len(plainae_models)} plain-AE "
    f"+ {len(mdsdec_models)} MDS-decoder models"
)

# =============================================================================================
_banner("STEP 2 -- chart survival (CAE-05), all three seeds; main seed feeds every downstream metric")

chart_survival_by_seed = {seed: cae_mod.chart_survival(seed_models[seed], PRUNE_TOL) for seed in SEEDS}
for _seed, _cs in chart_survival_by_seed.items():
    print(f"  seed={_seed}: {_cs['n_charts_surviving']}/{_cs['n_charts_initial']} surviving")

chart_survival_main = chart_survival_by_seed[MAIN_SEED]
surviving_indices_main = chart_survival_main["surviving_indices"]


def _survival_summary(cs: dict) -> dict:
    return {
        "n_charts_initial": cs["n_charts_initial"],
        "n_charts_surviving": cs["n_charts_surviving"],
        "surviving_indices": list(cs["surviving_indices"]),
        "pruned_indices": list(cs["pruned_indices"]),
        "chart_logmass": [float(v) for v in cs["chart_logmass"]],
        "chart_mass_ratio": [float(v) for v in cs["chart_mass_ratio"]],
        "prune_tol": cs["prune_tol"],
    }


chart_count_by_seed = {str(seed): _survival_summary(cs) for seed, cs in chart_survival_by_seed.items()}
chart_count_surviving = _survival_summary(chart_survival_main)

# =============================================================================================
_banner("STEP 3 -- gate T1: geodesic distortion on the main seed's global embedding")

_pairs_npz = _load_npz(f"cae_pairs_{FIT_KEY}")
pair_rows, pair_cols = _pairs_npz["rows"], _pairs_npz["cols"]
_mds_npz = _load_npz(MDS_EIGENSPECTRUM_STEM)
geo_pairs_r2 = _mds_npz["geo_pairs_r2"]

train_pair_mask = np.isin(pair_rows, train_idx) & np.isin(pair_cols, train_idx)
holdout_pair_mask = np.isin(pair_rows, holdout_idx) & np.isin(pair_cols, holdout_idx)

z_main = _seed_npz[MAIN_SEED]["z_all"]  # (10000, L_EMBED) -- the initial encoder's global embedding

t1_all = cae_mod.embedding_distortion(
    z=z_main,
    geo_pairs=geo_pairs_r2,
    rows=pair_rows,
    cols=pair_cols,
    train_mask=train_pair_mask,
    chart_dim=D_CHART,
    embed_dim=L_EMBED,
)
print(
    f"  T1 (all {t1_all['n_pairs']} pairs): median_abs_rel={t1_all['median_abs_rel']:.6f} "
    f"global_scale_factor={t1_all['global_scale_factor']:.6f}"
)

# Non-gating companion: same statistic restricted to holdout-only pairs, reusing the
# already-fit global scale factor (never refit on a holdout-only subset).
_d2_rep_all = ((z_main[pair_rows] - z_main[pair_cols]) ** 2).sum(axis=1)
_d2_geo_all = geo_pairs_r2.astype(np.float64) ** 2
t1_holdout_only = gp.distortion_stats(
    t1_all["global_scale_factor"] * _d2_rep_all[holdout_pair_mask], _d2_geo_all[holdout_pair_mask]
)
t1_holdout_only["n_pairs"] = int(holdout_pair_mask.sum())
print(
    f"  T1 (holdout-only {t1_holdout_only['n_pairs']} pairs, non-gating): "
    f"median_abs_rel={t1_holdout_only['median_abs_rel']:.6f}"
)

# =============================================================================================
_banner("STEP 4 -- gate T2: chart-transition cycle residual (eq. 8), holdout rows only")

p_all_main = _seed_npz[MAIN_SEED]["p_all"]  # (10000, N_CHARTS_INIT)
p_holdout_main = p_all_main[holdout_idx]  # (n_holdout, N_CHARTS_INIT), order matches x_holdout_t

overlap = cae_mod.select_overlap_pairs(
    p_holdout_main, OVERLAP_P_MIN, OVERLAP_MIN_POINTS, surviving_indices_main
)
overlap_rows, overlap_alpha, overlap_beta = overlap["rows"], overlap["alpha"], overlap["beta"]
print(f"  T2 overlap set: {len(overlap_rows)} qualifying holdout rows (>= {OVERLAP_MIN_POINTS} required)")

_pair_groups: dict = {}
for _i in range(len(overlap_rows)):
    _key = (int(overlap_alpha[_i]), int(overlap_beta[_i]))
    _pair_groups.setdefault(_key, []).append(_i)

r_cycle_vals = np.empty(len(overlap_rows), dtype=np.float64)
base_vals = np.empty(len(overlap_rows), dtype=np.float64)

with torch.no_grad():
    x_overlap = x_holdout_t[torch.from_numpy(overlap_rows)]
    for (_alpha, _beta), _local_positions in _pair_groups.items():
        _idx_t = torch.tensor(_local_positions, dtype=torch.long)
        _xb = x_overlap[_idx_t]

        _r = cae_mod.r_cycle(model_main, _xb, _alpha, _beta)
        r_cycle_vals[_local_positions] = _r.numpy()

        # matched base quantity: E_base(x) = 2 * ||x - yhat(x)||, twice the single-pass
        # argmax-chart reconstruction norm, matching eq. 8's two-term unsquared form
        _out = model_main(_xb)
        _chart_amax = _out["p"].argmax(dim=1)
        _y_amax = _out["y_charts"][torch.arange(_xb.shape[0]), _chart_amax]
        _base = 2.0 * torch.linalg.vector_norm(_xb - _y_amax, dim=-1)
        base_vals[_local_positions] = _base.numpy()

mean_r_cycle = float(r_cycle_vals.mean())
mean_base = float(base_vals.mean())
t2_ratio = mean_r_cycle / mean_base
print(f"  T2: mean(R_cycle)={mean_r_cycle:.6f}  mean(E_base)={mean_base:.6f}  ratio={t2_ratio:.6f}")

# =============================================================================================
_banner("STEP 5 -- gate T3: held-out reconstruction vs matched-capacity baselines")


def _recon(y_holdout_np: np.ndarray) -> dict:
    y_t = torch.tensor(np.asarray(y_holdout_np, dtype=np.float64), dtype=torch.float64)
    return cae_mod.reconstruction_stats(x_holdout_t.double(), y_t)


recon_cae_main = _recon(_seed_npz[MAIN_SEED]["y_holdout"])
recon_relu = _recon(_relu_npz["y_holdout"])
recon_plainae = {latent: _recon(_plainae_npz[latent]["y_holdout"]) for latent in PLAIN_AE_LATENTS}
recon_mdsdec = {p: _recon(_mdsdec_npz[p]["y_holdout"]) for p in MDS_BASELINE_P}

mse_cae = recon_cae_main["mse_per_dim"]
mse_plainae20 = recon_plainae[20]["mse_per_dim"]
mse_plainae40 = recon_plainae[40]["mse_per_dim"]
mse_mdsdec8 = recon_mdsdec[8]["mse_per_dim"]
mse_mdsdec20 = recon_mdsdec[20]["mse_per_dim"]

ratio_vs_plainae20 = mse_cae / mse_plainae20
ratio_vs_mdsdec8 = mse_cae / mse_mdsdec8
# metrics["recon_margin"] must compare via the same strict-less-than direction as T1/T2
# (verdict_from_metrics is uniform: passed = value < threshold). The pre-registration's own
# PASS condition is "mse_cae < (1 - THRESH_RECON_MARGIN) * mse_control" for BOTH controls;
# dividing both sides by mse_control gives "mse_cae/mse_control < (1 - THRESH_RECON_MARGIN)"
# -- algebraically identical to the pre-registered rule, just rearranged into value/threshold
# form. Taking the max ratio across both gating controls encodes the AND: PASS iff the worse
# (larger) of the two ratios still clears (1 - THRESH_RECON_MARGIN).
recon_margin_value = max(ratio_vs_plainae20, ratio_vs_mdsdec8)
recon_margin_threshold = 1.0 - THRESH_RECON_MARGIN

print(f"  mse_cae={mse_cae:.6e}  mse_plain_ae20={mse_plainae20:.6e}  mse_mds_dec8={mse_mdsdec8:.6e}")
print(
    f"  ratio_vs_plain_ae20={ratio_vs_plainae20:.6f}  ratio_vs_mds_dec8={ratio_vs_mdsdec8:.6f}  "
    f"gate value (max)={recon_margin_value:.6f}  threshold={recon_margin_threshold:.6f} "
    f"(derived from THRESH_RECON_MARGIN={THRESH_RECON_MARGIN})"
)

_DIM_KEYS = ("dim_mse_mean", "dim_mse_median", "dim_mse_p95", "dim_mse_max")

# =============================================================================================
_banner(
    "STEP 6 -- non-gating evidence: unfaithfulness/coverage, CAE-06 substitution cost, "
    "Krein comparison, wallclock"
)

uc = cae_mod.unfaithfulness_coverage(
    model_main, x_train_t, surviving_indices_main, UNFAITHFUL_SAMPLES, UNFAITHFUL_SEED
)
print(
    f"  unfaithfulness={uc['unfaithfulness']:.6f}  coverage={uc['coverage']:.6f}  "
    f"n_samples={uc['n_samples']}  n_distinct={uc['n_distinct']}"
)

activation_substitution = {
    "mse_silu_main": mse_cae,
    "mse_relu_control": recon_relu["mse_per_dim"],
    "delta_mse_relu_minus_silu": recon_relu["mse_per_dim"] - mse_cae,
    "sign_convention": (
        "positive delta_mse_relu_minus_silu means the pre-registered SiLU activation "
        "reconstructs BETTER than the ReLU control (lower held-out MSE) -- a benefit, not "
        "a cost. Negative means the SiLU substitution costs reconstruction quality relative "
        "to ReLU."
    ),
}
print(f"  CAE-06 activation substitution: delta(relu-silu)={activation_substitution['delta_mse_relu_minus_silu']:.6e}")

_geometry_probes = _load_json(GEOMETRY_PROBES_STEM)
_krein_q0 = {row["p"]: row for row in _geometry_probes["krein_ladder"] if row["q"] == 0}
krein_comparison_reported_not_gating = {
    "source": f"{GEOMETRY_PROBES_STEM}.json (02.1's own measured artifact, read not recomputed)",
    "classical_q0_p40_median_abs_rel": _krein_q0[40]["median_abs_rel"],
    "classical_q0_p8_median_abs_rel": _krein_q0[8]["median_abs_rel"],
    "working_dimension": _geometry_probes["working_dimension"],
    "gates_pass_fail": False,
    "note": (
        "D-01: reported for context only. This gate judges the CAE only against its own "
        "pre-registered T1/T2/T3 thresholds."
    ),
}

wallclock = {}
for _seed in SEEDS:
    _m = _seed_meta[_seed]
    wallclock[f"cae_seed_{_seed}"] = {
        "epochs_run": _m["epochs_run"],
        "wallclock_s": _m["wallclock_s"],
        "wallclock_truncated": _m["wallclock_truncated"],
        "early_stopped": _m["early_stopped"],
    }
wallclock["cae_ctrl_relu"] = {
    "epochs_run": _relu_meta["epochs_run"],
    "wallclock_s": _relu_meta["wallclock_s"],
    "wallclock_truncated": _relu_meta["wallclock_truncated"],
    "early_stopped": _relu_meta["early_stopped"],
}
for _latent in PLAIN_AE_LATENTS:
    _m = _plainae_meta[_latent]
    wallclock[f"cae_ctrl_plainae{_latent}"] = {
        "epochs_run": _m["epochs_run"],
        "wallclock_s": _m["wallclock_s"],
        "wallclock_truncated": _m["wallclock_truncated"],
        "early_stopped": _m["early_stopped"],
    }
for _p in MDS_BASELINE_P:
    _m = _mdsdec_meta[_p]
    wallclock[f"cae_ctrl_mdsdec{_p}"] = {
        "epochs_run": _m["epochs_run"],
        "wallclock_s": _m["wallclock_s"],
        "wallclock_truncated": _m["wallclock_truncated"],
        "early_stopped": _m["early_stopped"],
    }
_any_truncated = any(v["wallclock_truncated"] for v in wallclock.values())
print(f"  wallclock recorded for {len(wallclock)} runs; any_truncated={_any_truncated}")

# =============================================================================================
_banner("STEP 7 -- verdict: assemble metrics/thresholds, apply the tested strict-less-than rule")

metrics = {
    "distortion": t1_all["median_abs_rel"],
    "rcycle_ratio": t2_ratio,
    "recon_margin": recon_margin_value,
    "t1": {
        "median_abs_rel": t1_all["median_abs_rel"],
        "median_signed_rel": t1_all["median_signed_rel"],
        "p95_abs_rel": t1_all["p95_abs_rel"],
        "global_scale_factor": t1_all["global_scale_factor"],
        "n_pairs": t1_all["n_pairs"],
        "holdout_only_non_gating": t1_holdout_only,
    },
    "t2": {
        "mean_r_cycle": mean_r_cycle,
        "mean_base": mean_base,
        "ratio": t2_ratio,
        "n_qualify": int(len(overlap_rows)),
    },
    "t3": {
        "mse_cae": mse_cae,
        "mse_plain_ae20": mse_plainae20,
        "mse_plain_ae40": mse_plainae40,
        "mse_mds_dec8": mse_mdsdec8,
        "mse_mds_dec20": mse_mdsdec20,
        "mse_mds_dec8_linear_floor": _mdsdec_meta[8]["mse_linear_floor_holdout"],
        "mse_mds_dec20_linear_floor": _mdsdec_meta[20]["mse_linear_floor_holdout"],
        "margin_vs_plain_ae20": 1.0 - ratio_vs_plainae20,
        "margin_vs_mds_dec8": 1.0 - ratio_vs_mdsdec8,
        "recon_margin_threshold_fraction": THRESH_RECON_MARGIN,
        "recon_margin_gate_note": (
            "metrics.recon_margin is max(mse_cae/mse_plain_ae20, mse_cae/mse_mds_dec8); "
            "thresholds.recon_margin = 1 - THRESH_RECON_MARGIN. This is the pre-registration's "
            "own 'mse_cae < (1-margin)*mse_control' rule for both controls, divided through by "
            "mse_control on each side so the comparison direction stays strict-less-than, "
            "uniform with T1/T2 -- not a re-interpretation of the ratified rule."
        ),
        "per_dim": {
            "cae": {k: recon_cae_main[k] for k in _DIM_KEYS},
            "relu_control": {k: recon_relu[k] for k in _DIM_KEYS},
            "plain_ae20": {k: recon_plainae[20][k] for k in _DIM_KEYS},
            "plain_ae40": {k: recon_plainae[40][k] for k in _DIM_KEYS},
            "mds_dec8": {k: recon_mdsdec[8][k] for k in _DIM_KEYS},
            "mds_dec20": {k: recon_mdsdec[20][k] for k in _DIM_KEYS},
        },
    },
}

thresholds = {
    "distortion": THRESH_DISTORTION_MAX,
    "rcycle_ratio": THRESH_RCYCLE_RATIO_MAX,
    "recon_margin": recon_margin_threshold,
}

verdict, gate_detail = cae_mod.verdict_from_metrics(metrics, thresholds)
print(f"  CAE_VERDICT = {verdict}")
for _gate, _detail in gate_detail.items():
    print(f"    {_gate}: value={_detail['value']:.6f}  threshold={_detail['threshold']:.6f}  passed={_detail['passed']}")

versions = {
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "torch": torch.__version__,
    "python": platform.python_version(),
}

assumptions = [
    (
        "Theorem 2's compact-manifold-with-positive-reach hypothesis was never verified for "
        "this point cloud -- neither the reach tau nor the density ratio C = vol(M)/vol(B_1^d) "
        "has ever been estimated for the PU embedding manifold at n=10,000, so the theorem "
        "motivates the Chart Auto-Encoder architecture but predicts nothing about whether this "
        "specific fit passed or failed; that evidence is exclusively the three measured gate "
        "metrics above (02.2-PREREGISTRATION.md Section 2)."
    ),
    (
        "CAE-03 'matched capacity' is operationalised as the same hidden width (250), the same "
        "layer count per side (3), and the identical training protocol between the CAE and its "
        "plain-AE / MDS-decoder controls."
    ),
    (
        "CAE-04's overlap-pair rule pairs each qualifying holdout row with its two "
        "highest-probability surviving charts; the paper gives no rule for three or more "
        "charts clearing the cutoff simultaneously."
    ),
    (
        "CAE-06's substitution cost is measured from a single ReLU-activated control fit at "
        "MAIN_SEED; it is not itself averaged across seeds."
    ),
    (
        "CAE-01's ordering proof relies on 'git merge-base --is-ancestor' and the assumption "
        "that history is not rewritten after this evaluation."
    ),
]

remediation = [
    (
        "Adopt 02.1's Krein/pseudo-Euclidean retention (working dimension (p,q)=(8,0)) for "
        "Phase 3 instead of the Chart Auto-Encoder representation. Cost: re-specify Phase 3's "
        f"decoder/curvature plan against the Krein coordinates already measured in "
        f"{GEOMETRY_PROBES_STEM}.json; no new fit required."
    ),
    (
        "Iterate the Chart Auto-Encoder under a fresh, separately-ratified pre-registration -- "
        "a change to any threshold, architecture constant, or metric definition here is a "
        "documented amendment plus a full re-run, never a quiet edit of this sealed one. "
        "Cost: a new pre-registration cycle plus a fresh multi-hour fit protocol."
    ),
    (
        "Accept the documented outcome as the milestone's reported result. No further re-fit "
        f"is run; cae_verdict_{FIT_KEY}.json and 02.2-FINDINGS.md stand as the complete "
        "evidence trail, and Phase 3 is not planned in further detail until a human makes "
        "this call at this plan's phase-sealing checkpoint. This is a complete and legitimate "
        "milestone result, not a failure to be retried until it passes."
    ),
]

holdout = {
    "n": int(holdout_idx.shape[0]),
    "n_train": int(train_idx.shape[0]),
    "holdout_fraction": HOLDOUT_FRACTION,
    "split_seed": SPLIT_SEED,
    "t2_qualifying_points": int(len(overlap_rows)),
    "t2_min_points_required": OVERLAP_MIN_POINTS,
}

seeds = {
    "all_seeds": list(SEEDS),
    "main_seed": MAIN_SEED,
}

extra = {
    "chart_count_surviving": chart_count_surviving,
    "chart_count_by_seed": chart_count_by_seed,
    "unfaithfulness": uc["unfaithfulness"],
    "coverage": uc["coverage"],
    "krein_comparison_reported_not_gating": krein_comparison_reported_not_gating,
    "activation_substitution": activation_substitution,
    "holdout": holdout,
    "seeds": seeds,
    "wallclock": wallclock,
    "global_scale_factor": t1_all["global_scale_factor"],
    "versions": versions,
    "assumptions": assumptions,
    "remediation": remediation,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

written = cae_mod.write_cae_verdict(FIT_KEY, metrics, thresholds, verdict, extra=extra)
_verdict_path = cache.cache_path(f"cae_verdict_{FIT_KEY}", "json")
print(f"  written to: {_verdict_path}")

# =============================================================================================
_banner("STEP 8 -- Phase 3 handoff (PASS only) / halt (FAIL, no automatic substitution)")

if verdict == "PASS":
    handoff_payload = {
        "consumes": (
            f"cae_fit_{FIT_KEY}_seed{MAIN_SEED}.npz: the global embedding z_all, chart "
            "coordinates (derivable from z_all via the persisted 'chart_encoders.' state_dict "
            "keys), chart probabilities p_all, chart assignments chart_argmax_all, and the "
            "trained 'chart_decoders.'/'embedding_decoder.' state_dict"
        ),
        "global_embedding_key": "z_all",
        "chart_coords_key": "chart_coords_all",
        "chart_probs_key": "p_all",
        "chart_assignments_key": "chart_argmax_all",
        "decoder_state_keys": ["chart_decoders.", "embedding_decoder."],
        "surviving_charts": chart_count_surviving["surviving_indices"],
        "activation": ACTIVATION,
    }
    cae_mod.write_cae_handoff(FIT_KEY, verdict, handoff_payload)
    _handoff_path = cache.cache_path(f"cae_handoff_{FIT_KEY}", "json")
    print(f"  PASS -- handoff written to: {_handoff_path}")
else:
    print(
        "  FAIL -- halting. No handoff written. This runner never automatically adopts the "
        "02.1 Krein representation or any other substitute (D-02). See 02.2-FINDINGS.md for "
        "the complete finding and the options this leaves open for the user's decision."
    )

_required_verdict_keys = {
    "fit_key",
    "phase",
    "CAE_VERDICT",
    "metrics",
    "thresholds",
    "verdict_rule",
    "chart_count_surviving",
    "chart_count_by_seed",
    "unfaithfulness",
    "coverage",
    "krein_comparison_reported_not_gating",
    "activation_substitution",
    "holdout",
    "seeds",
    "wallclock",
    "global_scale_factor",
    "versions",
    "assumptions",
    "remediation",
    "timestamp",
}
_missing = _required_verdict_keys - set(written.keys())
assert not _missing, f"verdict artifact missing required keys: {_missing}"

print()
print(f"Done. CAE_VERDICT = {verdict}. All required verdict keys present.")
