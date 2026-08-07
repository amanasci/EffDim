"""Phase 02.4's Topological Auto-Encoder (TopoAE) evaluation runner -- the terminal artifact
of this milestone's Phase 02.4 gate.

Computes T1 (topological fidelity), T2 (reconstruction margin) and T3 (rank structure) --
exactly and only the three names in ``pu_manifold.topoae.GATING_METRICS`` -- on the held-out
set, across the whole eight-rung ladder, from the sixteen ``amend01``-tagged fits plans
02.4-05/06 already trained and independently verified. Applies the tested strict-less-than
verdict rule (``topoae.verdict_from_topoae_metrics``, delegating to ``cae.verdict_from_metrics``
unmodified, never re-derived inline here) and writes ``topoae_verdict_{FIT_KEY}.json`` on PASS
and on FAIL alike. Writes the Phase 3 handoff on PASS only; actively deletes any stale handoff
on FAIL (R6).

**Two artifact generations coexist on disk.** Only ``amend01``-tagged stems are ever read here.
The pre-amendment (buggy, ``epochs_run=15``, ``lambda_t`` stalled at half of ``LAMBDA_TOPO``)
fit artifacts are the deliberately preserved record of the stopping-rule defect
``02.4-PREREGISTRATION-AMENDMENT-01.md`` fixes -- they are never read, never deleted, and never
referenced by any stem this script constructs.

Precondition: the sixteen ``amend01``-tagged fit/baseline artifact pairs (npz + meta.json, each
with its own cache manifest) plan 02.4-05's reopened re-run produced and plan 02.4-06
independently verified must already exist under ``notebooks/.cache/`` (gitignored,
irreproducible here) -- this script halts rather than regenerating or refitting anything; there
is no regeneration path. A partial registry halts before computing anything, not after.

``notebooks/pu_manifold/cae.py`` and ``notebooks/pu_manifold/topoae.py`` are never edited by
this script -- every gate statistic, baseline-relative ratio, and artifact writer is imported
and called unmodified.

Invoke: PYTHONPATH=notebooks python notebooks/diagnostics/topoae_evaluate_run.py
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
from pu_manifold import topoae as topoae_mod

# --- pre-registered constants, copied verbatim (02.4-PREREGISTRATION.md Sections 3-4) --------
# Duplicated here rather than imported from topoae_train_run so the sixteen-fit presence
# assertion below (STEP 0a) runs BEFORE that module is imported: importing it re-executes its
# own fit registry, and a partial (missing-artifact) registry must halt here rather than let
# the import silently attempt to train the gap.

FIT_KEY = "43cf438bc944c509"
AMENDMENT_TAG = "amend01"
LADDER = (8, 16, 20, 24, 32, 40)
D_PRIMARY = 20
SEEDS_PRIMARY = (20260806, 20260807, 20260808)
SUBSAMPLE_STEM = "subsample_20260729_a79b3460b838fd0a"
MDS_EIGENSPECTRUM_STEM = "mds_eigenspectrum_43cf438bc944c509"

PREREG_PATH = (
    ".planning/phases/02.4-topological-auto-encoder-validity-test-inserted/02.4-PREREGISTRATION.md"
)
AMENDMENT_PATH = (
    ".planning/phases/02.4-topological-auto-encoder-validity-test-inserted/"
    "02.4-PREREGISTRATION-AMENDMENT-01.md"
)

# 02.4-PREREGISTRATION.md Section 2 -- copied verbatim, never re-derived or tuned toward a
# measured result.
THRESH_T1 = 0.90
THRESH_T2 = 1.00
THRESH_T3 = 0.90

_SENSITIVITY_RUNGS = tuple(d for d in LADDER if d != D_PRIMARY)
_ALL_RUNGS = [(D_PRIMARY, seed) for seed in SEEDS_PRIMARY] + [
    (d, SEEDS_PRIMARY[0]) for d in _SENSITIVITY_RUNGS
]


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
_banner(f"Phase 02.4 Topological Auto-Encoder EVALUATION runner -- fit_key = {FIT_KEY}")

# =============================================================================================
_banner("STEP 0a -- precondition: all sixteen amend01-tagged fit artifacts (+ manifests) present")

_required_stems = []
for _d, _seed in _ALL_RUNGS:
    for _kind in ("topoae_fit", "topoae_baseline"):
        _npz_stem = f"{_kind}_{FIT_KEY}_{AMENDMENT_TAG}_seed{_seed}_d{_d}"
        _meta_stem = f"{_kind}_meta_{FIT_KEY}_{AMENDMENT_TAG}_seed{_seed}_d{_d}"
        _required_stems.append((_npz_stem, "npz"))
        _required_stems.append((_npz_stem, "meta.json"))
        _required_stems.append((_meta_stem, "json"))
        _required_stems.append((_meta_stem, "meta.json"))
_required_stems.append((SUBSAMPLE_STEM, "npz"))
_required_stems.append((MDS_EIGENSPECTRUM_STEM, "npz"))
_required_stems.append((f"topoae_split_{FIT_KEY}", "npz"))
_required_stems.append((f"cae_pairs_{FIT_KEY}", "npz"))

_n_checked = 0
for _stem, _ext in _required_stems:
    _path = cache.cache_path(_stem, _ext)
    if not _path.exists() or _path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Required amend01-tagged fit artifact (or its manifest) missing or empty: {_path}. "
            "A partial fit registry must halt rather than produce a verdict from whatever "
            "happens to be on disk -- this script halts rather than regenerating or refitting "
            "anything; there is no regeneration path in this runner."
        )
    _n_checked += 1
print(
    f"  present: all {_n_checked} required artifacts/manifests across {len(_ALL_RUNGS)} rungs "
    "(8 TopoAE fits + 8 matched baselines, amend01-tagged only)"
)

# =============================================================================================
_banner("STEP 0b -- precondition: 02.4-PREREGISTRATION.md's adding commit is a proven git ancestor of HEAD")

_log_out = subprocess.run(
    ["git", "log", "--diff-filter=A", "--format=%H", "--", PREREG_PATH],
    capture_output=True,
    text=True,
    check=True,
)
_add_commits = [line for line in _log_out.stdout.strip().splitlines() if line]
if not _add_commits:
    raise RuntimeError(f"No commit found that added {PREREG_PATH} -- cannot prove ordering.")
_prereg_sha = _add_commits[-1]
_ancestor_check = subprocess.run(["git", "merge-base", "--is-ancestor", _prereg_sha, "HEAD"])
if _ancestor_check.returncode != 0:
    raise RuntimeError(
        f"Ordering violated: pre-registration commit {_prereg_sha} is NOT an ancestor of HEAD."
    )
print(f"  pre-registration commit {_prereg_sha} confirmed an ancestor of HEAD -- proved, not asserted")

# =============================================================================================
_banner("STEP 0c -- precondition: Amendment 1's adding commit is a proven git ancestor of HEAD")

_amend_log_out = subprocess.run(
    ["git", "log", "--diff-filter=A", "--format=%H", "--", AMENDMENT_PATH],
    capture_output=True,
    text=True,
    check=True,
)
_amend_add_commits = [line for line in _amend_log_out.stdout.strip().splitlines() if line]
if not _amend_add_commits:
    raise RuntimeError(f"No commit found that added {AMENDMENT_PATH} -- cannot prove ordering.")
_amend_sha = _amend_add_commits[-1]
_amend_ancestor_check = subprocess.run(["git", "merge-base", "--is-ancestor", _amend_sha, "HEAD"])
if _amend_ancestor_check.returncode != 0:
    raise RuntimeError(f"Ordering violated: Amendment 1 commit {_amend_sha} is NOT an ancestor of HEAD.")
print(f"  Amendment 1 commit {_amend_sha} confirmed an ancestor of HEAD -- proved, not asserted")

# =============================================================================================
_banner(
    "STEP 0d -- import topoae_train_run: re-verifies its own preconditions and reproduces the "
    "full sixteen-fit registry as pure cache hits (every artifact confirmed present above)"
)

from topoae_train_run import (  # noqa: E402
    ACTIVATION,
    AMBIENT_DIM,
    AMBIENT_DIST_NORM,
    BATCH,
    HIDDEN,
    LAMBDA_TOPO,
    MAX_EPOCHS,
    PAIR_SEED,
    RAMP_FRAC,
    RAMP_SHAPE,
    T3_K_GATE,
    T3_K_SWEEP,
    UNFAITHFUL_SAMPLES,
    UNFAITHFUL_SEED,
    WARMUP_FRAC,
)

print(f"  MAX_EPOCHS resolved by the training runner's own timing probe: {MAX_EPOCHS}")

# =============================================================================================
_banner("STEP 1 -- reload: cached holdout split, rebuild all sixteen models, encode+decode held-out rows")

_split_npz = _load_npz(f"topoae_split_{FIT_KEY}")
train_idx = _split_npz["train_idx"]
holdout_idx = _split_npz["holdout_idx"]
print(
    f"  loaded cached split: train={train_idx.shape[0]} holdout={holdout_idx.shape[0]} "
    "(the training runner's own persisted split -- never re-derived by drawing a fresh random "
    "reordering here)"
)

_subsample = _load_npz(SUBSAMPLE_STEM)
X = _subsample["legacysurvey"]
x_all_t = torch.tensor(X, dtype=torch.float32)
x_train_t = x_all_t[torch.from_numpy(train_idx)]
x_holdout_t = x_all_t[torch.from_numpy(holdout_idx)]


def _rebuild_model(npz: dict, d: int) -> "cae_mod.PlainAutoEncoder":
    model = cae_mod.PlainAutoEncoder(AMBIENT_DIM, d, hidden=HIDDEN, activation=ACTIVATION)
    model.load_state_dict(cae_mod.arrays_to_state_dict(npz, model.state_dict()))
    model.eval()
    return model


def _load_fit(kind: str, d: int, seed: int) -> dict:
    npz_stem = f"{kind}_{FIT_KEY}_{AMENDMENT_TAG}_seed{seed}_d{d}"
    meta_stem = f"{kind}_meta_{FIT_KEY}_{AMENDMENT_TAG}_seed{seed}_d{d}"
    npz = _load_npz(npz_stem)
    meta = _load_json(meta_stem)
    model = _rebuild_model(npz, d)
    with torch.no_grad():
        z_holdout = model.encode(x_holdout_t)
        y_holdout = model.decode(z_holdout)
    return {"model": model, "meta": meta, "z_holdout": z_holdout, "y_holdout": y_holdout, "d": d, "seed": seed}


fits = {}
for _d, _seed in _ALL_RUNGS:
    fits[(_d, _seed, "topoae")] = _load_fit("topoae_fit", _d, _seed)
    fits[(_d, _seed, "baseline")] = _load_fit("topoae_baseline", _d, _seed)
print(
    f"  rebuilt {len(fits)} models (8 TopoAE + 8 matched baselines) from their npz weights and "
    "encoded/decoded the held-out set under torch.no_grad()"
)

# =============================================================================================
_banner(
    "STEP 2 -- gate T1: topological_fidelity over the WHOLE held-out set in one pass (gating), "
    "plus the batch-wise average at the training batch size (reported, non-gating)"
)


def _batchwise_t1_average(model, latent_norm_final: float, batch_size: int) -> dict:
    """Batch-wise average of topological_loss's directional terms at the training batch
    size, normalized exactly as train_topoae normalizes each per-batch loss (d_x by its own
    batch max, d_z by the fit's own jointly-trained latent_norm) -- reported for consistency
    with how the loss was optimised, never gating (D-03 gates on the whole-set value only,
    computed separately above). Batches are contiguous, deterministic slices of the
    already-fixed-order held-out set -- no random reordering is drawn here."""
    n = x_holdout_t.shape[0]
    totals, dir_xz, dir_zx = [], [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x_holdout_t[i : i + batch_size]
            if xb.shape[0] < 2:
                continue
            z = model.encode(xb)
            d_x = topoae_mod.pairwise_distances_f64(xb)
            d_z = topoae_mod.pairwise_distances_f64(z)
            d_x_norm = d_x / d_x.max()
            d_z_norm = d_z / latent_norm_final
            loss = topoae_mod.topological_loss(d_x_norm, d_z_norm)
            totals.append(loss["total"].item())
            dir_xz.append(loss["loss_x_to_z"].item())
            dir_zx.append(loss["loss_z_to_x"].item())
    return {
        "n_batches": len(totals),
        "mean_total": float(np.mean(totals)) if totals else float("nan"),
        "mean_loss_x_to_z": float(np.mean(dir_xz)) if dir_xz else float("nan"),
        "mean_loss_z_to_x": float(np.mean(dir_zx)) if dir_zx else float("nan"),
    }


t1_raw = {}
for key, fit in fits.items():
    t1_raw[key] = topoae_mod.topological_fidelity(x_holdout_t, fit["z_holdout"])

batchwise_t1 = {}
for (_d, _seed) in _ALL_RUNGS:
    _fit = fits[(_d, _seed, "topoae")]
    batchwise_t1[(_d, _seed)] = _batchwise_t1_average(
        _fit["model"], _fit["meta"]["latent_norm_final"], BATCH
    )

for (_d, _seed) in _ALL_RUNGS:
    _w = t1_raw[(_d, _seed, "topoae")]["worse"]
    _bw = batchwise_t1[(_d, _seed)]["mean_total"]
    print(f"  T1 d={_d:>2} seed={_seed}: whole-set worse={_w:.6f}  batch-wise mean_total(non-gating)={_bw:.6f}")

# =============================================================================================
_banner("STEP 3 -- gate T2: reconstruction_stats on held-out data, model vs matched baseline")

t2_raw = {}
for key, fit in fits.items():
    t2_raw[key] = cae_mod.reconstruction_stats(x_holdout_t.double(), fit["y_holdout"].double())

for (_d, _seed) in _ALL_RUNGS:
    _m = t2_raw[(_d, _seed, "topoae")]["mse_per_dim"]
    _b = t2_raw[(_d, _seed, "baseline")]["mse_per_dim"]
    print(f"  T2 d={_d:>2} seed={_seed}: mse_per_dim topoae={_m:.6e}  baseline={_b:.6e}")

# =============================================================================================
_banner(
    f"STEP 4 -- gate T3: rank_structure at the full sweep T3_K_SWEEP={T3_K_SWEEP}, gating only "
    f"at T3_K_GATE={T3_K_GATE}"
)

t3_raw = {}
for key, fit in fits.items():
    t3_raw[key] = {k: topoae_mod.rank_structure(x_holdout_t, fit["z_holdout"], k) for k in T3_K_SWEEP}

for (_d, _seed) in _ALL_RUNGS:
    _m = t3_raw[(_d, _seed, "topoae")][T3_K_GATE]["gate_value"]
    _b = t3_raw[(_d, _seed, "baseline")][T3_K_GATE]["gate_value"]
    print(f"  T3 d={_d:>2} seed={_seed}: gate_value(k={T3_K_GATE}) topoae={_m:.6f}  baseline={_b:.6f}")

# =============================================================================================
_banner(
    "STEP 5 -- non-gating context block: geodesic distortion, unfaithfulness/coverage, "
    "batch-wise T1 average -- reported for comparability with 02.1/02.2, never gating "
    "(D-01, R4). T1 here is NOT comparable to 02.2's distortion figure -- a different "
    "normalization is a different statistic (D-05)."
)

_primary_seed0 = SEEDS_PRIMARY[0]
_primary_fit = fits[(D_PRIMARY, _primary_seed0, "topoae")]
_primary_model = _primary_fit["model"]
with torch.no_grad():
    _z_all_primary = _primary_model.encode(x_all_t)

_pairs_npz = _load_npz(f"cae_pairs_{FIT_KEY}")
pair_rows, pair_cols = _pairs_npz["rows"], _pairs_npz["cols"]
_mds_npz = _load_npz(MDS_EIGENSPECTRUM_STEM)
geo_pairs_r2 = _mds_npz["geo_pairs_r2"]
train_pair_mask = np.isin(pair_rows, train_idx) & np.isin(pair_cols, train_idx)

geodesic_distortion = cae_mod.embedding_distortion(
    z=_z_all_primary.detach().cpu().numpy().astype(np.float64),
    geo_pairs=geo_pairs_r2,
    rows=pair_rows,
    cols=pair_cols,
    train_mask=train_pair_mask,
    chart_dim=D_PRIMARY,
    embed_dim=D_PRIMARY,
)
print(
    f"  geodesic distortion (primary rung d={D_PRIMARY} seed={_primary_seed0}, non-gating): "
    f"median_abs_rel={geodesic_distortion['median_abs_rel']:.6f} "
    f"global_scale_factor={geodesic_distortion['global_scale_factor']:.6f}"
)

_uc_adapter = topoae_mod.unfaithfulness_adapter(_primary_model, _z_all_primary)
unfaithfulness_result = cae_mod.unfaithfulness_coverage(
    _uc_adapter, x_train_t, [0], UNFAITHFUL_SAMPLES, UNFAITHFUL_SEED
)
print(
    f"  unfaithfulness (non-gating): {unfaithfulness_result['unfaithfulness']:.6f}  "
    f"coverage: {unfaithfulness_result['coverage']:.6f}  "
    f"n_samples={unfaithfulness_result['n_samples']}  n_distinct={unfaithfulness_result['n_distinct']}"
)

context_metrics = {
    "geodesic_distortion": geodesic_distortion,
    "unfaithfulness": unfaithfulness_result["unfaithfulness"],
    "coverage": unfaithfulness_result["coverage"],
    "unfaithfulness_n_samples": unfaithfulness_result["n_samples"],
    "unfaithfulness_n_distinct": unfaithfulness_result["n_distinct"],
    "batchwise_t1_average_by_rung": {
        f"d{_d}_seed{_seed}": batchwise_t1[(_d, _seed)] for (_d, _seed) in _ALL_RUNGS
    },
    "batchwise_t1_average_primary_rung": batchwise_t1[(D_PRIMARY, _primary_seed0)],
    "note": (
        "Reported for comparability with 02.1's ~0.0796 wall and 02.2's CAE number, and never "
        "enters the verdict -- 02.1's wall was measured on a distance-preservation statistic "
        "this method never optimises, and ranking TopoAE on it would be the category error "
        "R4/D-01 forbids. D-05: this T1 number is not directly comparable to 02.2's distortion "
        "figure either -- a different latent normalization is a different statistic. Erratum 1 / "
        "Limitation 3: loss_x_to_z/loss_z_to_x above are scale-sensitive and not clean evidence "
        "on their own; the gate's baseline-relative ratio form is the trustworthy quantity."
    ),
}

# =============================================================================================
_banner(
    "STEP 6 -- reduce to the three gate values (D-20 worst-of-three-primary-seeds), assemble "
    "the T1/T2 frontier across the other five rungs (D-16: primary rung alone produces the "
    "verdict; a frontier point never promotes a different rung to primary)"
)

rung_table = []
for (_d, _seed) in _ALL_RUNGS:
    _t1_m, _t1_b = t1_raw[(_d, _seed, "topoae")], t1_raw[(_d, _seed, "baseline")]
    _t2_m, _t2_b = t2_raw[(_d, _seed, "topoae")], t2_raw[(_d, _seed, "baseline")]
    _t3_m, _t3_b = t3_raw[(_d, _seed, "topoae")], t3_raw[(_d, _seed, "baseline")]
    _t1_ratio = topoae_mod.t1_gate_value(_t1_m, _t1_b)
    _t2_ratio = topoae_mod.t2_gate_value(_t2_m, _t2_b)
    _t3_ratio = topoae_mod.t3_gate_value(_t3_m[T3_K_GATE], _t3_b[T3_K_GATE])
    rung_table.append(
        {
            "d": _d,
            "seed": _seed,
            "is_primary_rung": _d == D_PRIMARY,
            "t1_ratio": _t1_ratio,
            "t2_ratio": _t2_ratio,
            "t3_ratio": _t3_ratio,
            "t3_sweep_ratio": {
                k: topoae_mod.t3_gate_value(_t3_m[k], _t3_b[k]) for k in T3_K_SWEEP
            },
        }
    )

_phdr = f"{'d':>4s} {'seed':>10s} {'t1_ratio':>12s} {'t2_ratio':>12s} {'t3_ratio':>12s}"
print(_phdr)
print("-" * len(_phdr))
for _row in rung_table:
    print(
        f"{_row['d']:>4} {_row['seed']:>10} {_row['t1_ratio']:>12.6f} "
        f"{_row['t2_ratio']:>12.6f} {_row['t3_ratio']:>12.6f}"
    )

_primary_rows = [r for r in rung_table if r["is_primary_rung"]]
_sensitivity_rows = [r for r in rung_table if not r["is_primary_rung"]]


def _primary_reduction(ratio_key: str) -> dict:
    values = {r["seed"]: r[ratio_key] for r in _primary_rows}
    vals_list = list(values.values())
    worst_seed = max(values, key=lambda s: values[s])
    return {
        "per_seed": values,
        "worst_seed": worst_seed,
        "gate_value": max(vals_list),
        "mean": float(np.mean(vals_list)),
        "std": float(np.std(vals_list)),
        "min": float(np.min(vals_list)),
        "max": float(np.max(vals_list)),
    }


t1_reduction = _primary_reduction("t1_ratio")
t2_reduction = _primary_reduction("t2_ratio")
t3_reduction = _primary_reduction("t3_ratio")

print()
print(
    f"  T1 gate value (worst-of-3-seeds) = {t1_reduction['gate_value']:.6f}  "
    f"(mean={t1_reduction['mean']:.6f}  std={t1_reduction['std']:.6f}  worst_seed={t1_reduction['worst_seed']})"
)
print(
    f"  T2 gate value (worst-of-3-seeds) = {t2_reduction['gate_value']:.6f}  "
    f"(mean={t2_reduction['mean']:.6f}  std={t2_reduction['std']:.6f}  worst_seed={t2_reduction['worst_seed']})"
)
print(
    f"  T3 gate value (worst-of-3-seeds) = {t3_reduction['gate_value']:.6f}  "
    f"(mean={t3_reduction['mean']:.6f}  std={t3_reduction['std']:.6f}  worst_seed={t3_reduction['worst_seed']})"
)

frontier = {
    "rungs": [
        {"d": r["d"], "t1_ratio": r["t1_ratio"], "t2_ratio": r["t2_ratio"], "t3_ratio": r["t3_ratio"]}
        for r in _sensitivity_rows
    ],
    "note": (
        "D-16: reported as a measured T1/T2/T3 reconstruction-versus-topology frontier across "
        "the five sensitivity rungs (seed "
        f"{SEEDS_PRIMARY[0]} only). No frontier point promotes a different rung to primary -- "
        f"the verdict rests on D_PRIMARY={D_PRIMARY} alone, per D-16."
    ),
}

print()
print("  T1/T2/T3 frontier across the five sensitivity rungs (reported, never gating):")
_fhdr = f"{'d':>4s} {'t1_ratio':>12s} {'t2_ratio':>12s} {'t3_ratio':>12s}"
print(_fhdr)
print("-" * len(_fhdr))
for _r in frontier["rungs"]:
    print(f"{_r['d']:>4} {_r['t1_ratio']:>12.6f} {_r['t2_ratio']:>12.6f} {_r['t3_ratio']:>12.6f}")

metrics = {
    topoae_mod.GATING_METRICS[0]: t1_reduction["gate_value"],
    topoae_mod.GATING_METRICS[1]: t2_reduction["gate_value"],
    topoae_mod.GATING_METRICS[2]: t3_reduction["gate_value"],
    "t1": t1_reduction,
    "t2": t2_reduction,
    "t3": t3_reduction,
    "frontier": frontier,
    "per_rung_table": rung_table,
    "d09_note": (
        "D-09: T1 and T3 gate independently but may correlate (T1's MST measures the "
        "connectivity skeleton, T3 measures local neighbourhood ordering); a joint FAIL on "
        "both is read as one structural finding, not two independent ones."
    ),
}

thresholds = {
    topoae_mod.GATING_METRICS[0]: THRESH_T1,
    topoae_mod.GATING_METRICS[1]: THRESH_T2,
    topoae_mod.GATING_METRICS[2]: THRESH_T3,
}

# =============================================================================================
_banner("STEP 7 -- verdict: strict less-than via topoae.verdict_from_topoae_metrics (never re-derived inline)")

verdict, gate_detail = topoae_mod.verdict_from_topoae_metrics(metrics, thresholds)
print(f"  TOPOAE_VERDICT = {verdict}")
for _gate, _detail in gate_detail.items():
    print(
        f"    {_gate}: value={_detail['value']:.6f}  threshold={_detail['threshold']:.6f}  "
        f"passed={_detail['passed']}"
    )
if not gate_detail[topoae_mod.GATING_METRICS[0]]["passed"] and not gate_detail[topoae_mod.GATING_METRICS[2]]["passed"]:
    print(
        "  D-09: both T1 and T3 failed -- read as one structural finding (connectivity AND "
        "local-neighbourhood-ordering both broken relative to baseline), not two independent ones."
    )

versions = {
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "torch": torch.__version__,
    "python": platform.python_version(),
}

extra = {
    "per_rung_table": rung_table,
    "frontier": frontier,
    "primary_seed_reduction": {"t1_topo_fidelity": t1_reduction, "t2_recon_margin": t2_reduction, "t3_rank_structure": t3_reduction},
    "context_metrics": context_metrics,
    "seeds": {"primary": list(SEEDS_PRIMARY), "sensitivity_seed": SEEDS_PRIMARY[0]},
    "ladder": list(LADDER),
    "d_primary": D_PRIMARY,
    "resolved_max_epochs": MAX_EPOCHS,
    "activation": ACTIVATION,
    "ambient_dist_norm_t1_gate_only": AMBIENT_DIST_NORM,
    "versions": versions,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "verdict_rule_note": topoae_mod.VERDICT_RULE,
    "d09_note": metrics["d09_note"],
}

written = topoae_mod.write_topoae_verdict(FIT_KEY, metrics, thresholds, verdict, extra=extra)
_verdict_path = cache.cache_path(f"topoae_verdict_{FIT_KEY}", "json")
print(f"  written to: {_verdict_path}")

_required_verdict_keys = {
    "fit_key",
    "phase",
    "TOPOAE_VERDICT",
    "gating_metrics",
    "metrics",
    "thresholds",
    "gate_detail",
    "verdict_rule",
    "context_metrics",
}
_missing = _required_verdict_keys - set(written.keys())
assert not _missing, f"verdict artifact missing required keys: {_missing}"
assert tuple(written["gating_metrics"]) == topoae_mod.GATING_METRICS, written["gating_metrics"]

_written_json = json.loads(_verdict_path.read_text())
_recomputed = (
    "PASS"
    if all(
        float(_written_json["metrics"][g]) < float(_written_json["thresholds"][g])
        for g in _written_json["gating_metrics"]
    )
    else "FAIL"
)
assert _recomputed == _written_json["TOPOAE_VERDICT"], (_recomputed, _written_json["TOPOAE_VERDICT"])
print(
    f"  confirmed: stored verdict {_written_json['TOPOAE_VERDICT']!r} is reproducible from the "
    f"artifact's own recorded metrics/thresholds alone (recomputed={_recomputed!r}), without "
    "this runner"
)

# =============================================================================================
_banner(
    "STEP 7b -- gate-scope annotation (additive; coordinator-directed 2026-08-07 checkpoint "
    "response): each gate's scope (global vs local), so a downstream reader does not collapse "
    "the phase's most informative result into one FAIL string"
)

# The pre-registered gate output (TOPOAE_VERDICT, metrics, thresholds, gate_detail,
# verdict_rule, gating_metrics, fit_key, phase, context_metrics) is NEVER re-gated or altered
# here -- this step only ADDS two new top-level keys to the already-written artifact, after
# asserting every pre-registered field is byte-identical to what write_topoae_verdict produced
# above. T1 (whole-held-out-set MST / 0-dim persistence structure) and T2 (held-out
# reconstruction) are GLOBAL statistics; T3 (rank_structure at the gating neighbourhood size
# T3_K_GATE) is a LOCAL, k-neighbourhood statistic (sklearn.manifold.trustworthiness/
# continuity) -- the k it was actually gated at is read from the pre-registered constant, never
# assumed.
GATE_SCOPE = {
    "t1_topo_fidelity": "global",
    "t2_recon_margin": "global",
    "t3_rank_structure": "local",
}
assert set(GATE_SCOPE) == set(topoae_mod.GATING_METRICS), (GATE_SCOPE, topoae_mod.GATING_METRICS)

for _k in (
    "fit_key",
    "phase",
    "TOPOAE_VERDICT",
    "gating_metrics",
    "metrics",
    "thresholds",
    "gate_detail",
    "verdict_rule",
    "context_metrics",
):
    assert _written_json[_k] == written[_k], (
        f"gate-scope annotation step must not alter pre-registered field {_k!r} -- refusing to "
        "proceed"
    )

_global_failed = [n for n, s in GATE_SCOPE.items() if s == "global" and not _written_json["gate_detail"][n]["passed"]]
_local_passed = [n for n, s in GATE_SCOPE.items() if s == "local" and _written_json["gate_detail"][n]["passed"]]
scope_reading = (
    f"{len(_global_failed)} global-scoped gate(s) failed ({', '.join(_global_failed) or 'none'}: "
    "t1_topo_fidelity is the whole-set MST/0-dim persistence structure, t2_recon_margin is "
    f"held-out reconstruction) and {len(_local_passed)} local-scoped gate(s) passed "
    f"({', '.join(_local_passed) or 'none'}: t3_rank_structure is the k={T3_K_GATE}-neighbourhood "
    "rank-ordering statistic via sklearn.manifold.trustworthiness/continuity). "
    f"TOPOAE_VERDICT is {_written_json['TOPOAE_VERDICT']} because the pre-registered verdict "
    "rule requires all three gates to hold -- this scope annotation does not change that rule "
    "or its outcome. A downstream consumer evaluating a locally-scoped use (e.g. local curvature "
    "estimation, which depends on local fidelity, not global fidelity) must read gate_scope, not "
    "only TOPOAE_VERDICT."
)

if _written_json.get("gate_scope") == GATE_SCOPE and _written_json.get("scope_reading") == scope_reading:
    print("  gate_scope annotation already present and matches -- idempotent, no rewrite")
else:
    _written_json["gate_scope"] = GATE_SCOPE
    _written_json["gate_scope_t3_k"] = T3_K_GATE
    _written_json["scope_reading"] = scope_reading
    _verdict_path.write_text(json.dumps(_written_json, indent=2, sort_keys=True))
    print(f"  gate_scope written additively: {GATE_SCOPE}")
    print(f"  gate_scope_t3_k = {T3_K_GATE}")
    print(f"  scope_reading: {scope_reading}")

# =============================================================================================
_banner("STEP 8 -- PASS handoff / FAIL deletion of any stale handoff (R6)")

if verdict == "PASS":
    handoff_payload = {
        "primary_d": D_PRIMARY,
        "fit_artifact_keys": [
            f"topoae_fit_{FIT_KEY}_{AMENDMENT_TAG}_seed{seed}_d{D_PRIMARY}" for seed in SEEDS_PRIMARY
        ],
        "encoder_state_key": "encoder.",
        "gate_values": {name: metrics[name] for name in topoae_mod.GATING_METRICS},
        "thresholds": thresholds,
        "evidence_criteria": [
            "T1 topological_fidelity (whole held-out set, worse-of-two-directions) beats the "
            f"matched baseline by more than the pre-registered {THRESH_T1} margin at the "
            f"primary rung's worst seed",
            "T2 reconstruction_margin (mse_per_dim ratio) strictly beats the matched baseline",
            f"T3 rank_structure (1-min(trustworthiness,continuity) at k={T3_K_GATE}) beats the "
            f"matched baseline by more than the pre-registered {THRESH_T3} margin",
            "The T1/T2/T3 frontier across the sensitivity rungs (ladder d=8,16,24,32,40) shows "
            "where this representation sits on the reconstruction-versus-topology tradeoff",
            "Named limitations carried from 02.4-03/04/05 (absolute persistence correlation "
            "below the original 0.8 bound; the lambda selection rule's mis-specification; "
            "loss_x_to_z/loss_z_to_x scale-sensitivity) apply to any downstream reading",
        ],
        "reinstated_requirements": [
            "DEC-01", "DEC-02", "DEC-03", "DEC-04", "DEC-05",
            "CURV-01", "CURV-02", "CURV-03", "CURV-04", "CURV-05", "CURV-06", "CURV-07", "CURV-08",
        ],
        "activation": ACTIVATION,
    }
    written_handoff = topoae_mod.write_topoae_handoff(FIT_KEY, verdict, handoff_payload)
    _handoff_path = cache.cache_path(f"topoae_handoff_{FIT_KEY}", "json")
    print(f"  PASS -- handoff written to: {_handoff_path}")
    print(
        "  Next: plan 02.4-08's reconciliation runner --"
        "\n    PYTHONPATH=notebooks python notebooks/diagnostics/topoae_reconcile_run.py"
        "\n  This runner does not perform the reconciliation itself."
    )
else:
    _removed = topoae_mod.clear_stale_handoff(FIT_KEY)
    print(f"  FAIL -- halting. clear_stale_handoff removed_something={_removed}")
    print(
        "  No handoff written. This runner never automatically substitutes another "
        "representation. A FAIL is a complete, reportable outcome of this phase -- not an "
        "error to retry, downgrade, or suppress. Two phases of this milestone have already "
        "sealed on FAIL verdicts and both were legitimate results."
    )

print()
print(f"Done. TOPOAE_VERDICT = {verdict}. All required verdict keys present.")
