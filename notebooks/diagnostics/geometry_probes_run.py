"""
Phase 02.1 geometry-representation probes, run over the frozen Phase 2 cache
(fit_key=43cf438bc944c509). Diagnostic, not pre-registered, in the sense that it defines
no new gate -- 02.1-PREREGISTRATION.md's rules govern how these numbers are read; this
script only produces them. Every constant below was fixed in 02.1-PREREGISTRATION.md
before any of these numbers existed, named by the section it came from -- never invented
here.

Three questions:
  Q1. Correction blindness (GEOM-03). Do eigenvalue clipping and the Lingoes-type
      additive shift force zero negative mass on ANY input, curved or not?
  Q2. The pseudo-Euclidean (p, q) distortion ladder (GEOM-03, GEOM-05). How far does
      retaining negative eigenvalue directions reduce distortion below the q=0
      classical-MDS baseline, and what working dimension does that curve imply?
  Q3. Delta-hyperbolicity (GEOM-04). Does the real manifold's sampled Gromov delta sit
      near a tree anchor (single constant negative curvature defensible) or near a flat-
      Euclidean anchor (not defensible), at matched n?

Together these two probes (Q2, Q3) test the falsifier 02.1-PREREGISTRATION.md and
02.1-FORK.md both name: the coordinate-producing verdict is overturned only if BOTH the
delta reading rule refuses a single-curvature model AND the (p, q) ladder never drops
distortion materially below the q=0 baseline. This script reports the measured numbers
plainly; it does not adjudicate "materially below" -- that judgment is plan 02.1-04's.

Hard precondition: notebooks/.cache/isomap_43cf438bc944c509.joblib (~1.55 GiB) and
notebooks/.cache/mds_eigenspectrum_43cf438bc944c509.npz must already exist on disk --
notebooks/.cache/ is gitignored and neither artifact is reproducible by any plan in this
phase. This script halts (via assertion / FileNotFoundError) rather than regenerating
either; regenerating would silently change provenance and disconnect this phase's
evidence from the exact fit Phase 2 audited.

Invoke with: PYTHONPATH=notebooks python notebooks/diagnostics/geometry_probes_run.py
"""

import gc
import json
import resource
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.linalg import eigh, eigvalsh
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial.distance import pdist, squareform

from pu_manifold import geometry_probes as gp
from pu_manifold.cache import cache_path, json_cache, npz_cache

CACHE = "notebooks/.cache"
FIT_KEY = "43cf438bc944c509"

# --- pre-registered constants (02.1-PREREGISTRATION.md) ---------------------------------
# "## Working-Dimension Rule -- The (p, q) ladder"
P_LADDER = (1, 2, 3, 5, 8, 13, 18, 25, 40)
Q_LADDER = (0, 1, 2, 3, 5, 8, 13, 25, 40)
# "## Delta-Hyperbolicity Reading Rule -- Constants, fixed now"
DELTA_QUADRUPLES = 200_000
DELTA_SEED = 20260801
# "interface_from_prior_phase" / mds_eigenspectrum_{fit_key}.meta.json: the 200,000-pair
# geodesic sample Phase 2 drew and cached
PAIR_COUNT = 200_000
PAIR_SEED = 20260731
# Phase 2 published self-check values (gate_verdict_{fit_key}.json / 02-FINDINGS.md §3.4/3.6)
PUBLISHED_R = 0.0524192078526829
PUBLISHED_M = 0.4120712514841815
PUBLISHED_N_POSITIVE = 4971
PUBLISHED_N_NEGATIVE = 5029


def gate_stats(ev):
    """r and m exactly as Phase 2 defines them. Copied verbatim from
    notebooks/diagnostics/gate_diagnostics.py rather than imported, because importing
    that module would execute its own top-level diagnostic run as a side effect."""
    neg, pos = ev[ev < 0], ev[ev > 0]
    r = abs(neg.min()) / pos.max()
    m = np.abs(neg).sum() / np.abs(ev).sum()
    return r, m, int(pos.size), int(neg.size)


def spectrum_from_distmatrix(D):
    """Full classical-MDS double-centred matrix by mean-form double-centring, float64
    throughout. Copied verbatim from notebooks/diagnostics/gate_diagnostics.py's
    ``spectrum_from_distmatrix`` (minus its ``eigvalsh`` call, since here we want the
    matrix ``B`` itself for a subset eigensolve, not the full spectrum). ``copy=True``:
    ``np.asarray``/``np.array`` on a read-only memmap returns a *view*, not a copy, when
    the dtype already matches -- in-place arithmetic on a view then fails or corrupts."""
    D2 = np.array(D, dtype=np.float64, copy=True)
    D2 **= 2
    row = D2.mean(axis=1, keepdims=True)
    col = D2.mean(axis=0, keepdims=True)
    tot = D2.mean()
    D2 -= row
    D2 -= col
    D2 += tot
    D2 *= -0.5
    return D2


def _weighted_tree_geodesics(n, seed):
    """A random spanning tree with random positive edge weights; its shortest-path
    metric satisfies the four-point condition exactly (delta == 0 everywhere)."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.5, 5.0, size=(n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    mst = minimum_spanning_tree(W)
    return shortest_path(mst, method="D", directed=False)


def _flat_euclidean_geodesics(n, dim, seed):
    """n standard-normal points in R^dim under the Euclidean metric."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    return squareform(pdist(X))


def _to_native(obj):
    """Recursively cast numpy scalars to Python float/int/bool -- json_cache's
    ``json.dumps(..., sort_keys=True)`` cannot serialise numpy scalars."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _print_rss(label):
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"  [{label}] peak RSS so far: {kb / 1e6:.2f} GB")


print("=" * 78)
print(f"Phase 02.1 geometry probes -- fit_key = {FIT_KEY}")
print("=" * 78)

_gate_verdict = json.loads(Path(f"{CACHE}/gate_verdict_{FIT_KEY}.json").read_text())
D_FROZEN = _gate_verdict["d_frozen"]
D_PROVISIONAL = _gate_verdict["d_provisional"]
ELBOW_CRITERION = _gate_verdict["elbow_criterion"]
GATE_SPECTRUM = _gate_verdict["spectrum"]

# =========================================================================================
print()
print("=" * 78)
print("STEP 0 -- self-check: reproduce Phase 2's published r/m from the cached spectrum")
print("=" * 78)
_spectrum_npz = np.load(f"{CACHE}/mds_eigenspectrum_{FIT_KEY}.npz")
eigvals_all = _spectrum_npz["eigvals_all"]
eigvals_top = _spectrum_npz["eigvals_top"]
eigvecs_top = _spectrum_npz["eigvecs_top"]
geo_pairs_r2 = _spectrum_npz["geo_pairs_r2"]

r0, m0, n_pos, n_neg = gate_stats(eigvals_all)
print(f"  r={r0:.16f}  m={m0:.16f}  n_positive={n_pos}  n_negative={n_neg}")
assert abs(r0 - PUBLISHED_R) < 1e-9, f"r drifted from published value: {r0} != {PUBLISHED_R}"
assert abs(m0 - PUBLISHED_M) < 1e-9, f"m drifted from published value: {m0} != {PUBLISHED_M}"
assert n_pos == PUBLISHED_N_POSITIVE, f"n_positive drifted: {n_pos} != {PUBLISHED_N_POSITIVE}"
assert n_neg == PUBLISHED_N_NEGATIVE, f"n_negative drifted: {n_neg} != {PUBLISHED_N_NEGATIVE}"
print("  MATCH -- self-check passed. Nothing below this line runs on a mismatched artifact.")

# =========================================================================================
print()
print("=" * 78)
print("STEP 1 -- pair-sample identity: are the re-drawn 200,000 pairs the same Phase 2 used?")
print("=" * 78)
_isomap_path = cache_path(f"isomap_{FIT_KEY}", "joblib")
_expected_dist_matrix_bytes = 10_000 * 10_000 * 8
print(f"  loading {_isomap_path.name} with mmap_mode='r' "
      f"(dist_matrix_ ~{_expected_dist_matrix_bytes / 1e9:.2f} GB)")
_isomap_obj = joblib.load(_isomap_path, mmap_mode="r")
dist_matrix_ = _isomap_obj.dist_matrix_
_isomap_obj = None
gc.collect()
print(f"  dist_matrix_.shape={dist_matrix_.shape}  dtype={dist_matrix_.dtype}")

_pair_rng = np.random.default_rng(PAIR_SEED)
rows, cols = gp.draw_geo_pairs(_pair_rng, dist_matrix_.shape[0], PAIR_COUNT)
_redrawn = np.array(dist_matrix_[rows, cols], dtype=np.float64, copy=True)

pair_identity_verified = bool(np.array_equal(_redrawn, geo_pairs_r2))
if not pair_identity_verified:
    raise SystemExit(
        "HALT -- the 200,000 re-drawn pairs are NOT bit-identical to the cached "
        "geo_pairs_r2 array. pu_manifold.geometry_probes.draw_geo_pairs (mirroring "
        "02-PATTERNS.md's _draw_geo_pairs) does not match whatever produced geo_pairs_r2 "
        "under PAIR_SEED=20260731. The section 6 pair-draw cell of "
        "notebooks/01_manifold_and_gate.ipynb is the source of truth to mirror -- read it "
        "directly and compare against draw_geo_pairs before changing anything else. Every "
        "distortion number below this line depends on this identity and none of them can "
        "be trusted until it is fixed."
    )
print(f"  pair_identity_verified = {pair_identity_verified} "
      "(dist_matrix_[rows, cols] is bit-identical to cached geo_pairs_r2)")

# =========================================================================================
print()
print("=" * 78)
print("STEP 2 -- delta-hyperbolicity: real manifold vs. tree and flat-Euclidean anchors")
print("=" * 78)
t0 = time.perf_counter()
delta_real_n10000 = gp.sampled_delta_hyperbolicity(dist_matrix_, DELTA_QUADRUPLES, DELTA_SEED)
print(f"  real @ n=10000 ({time.perf_counter() - t0:.1f}s): "
      f"delta_rel_max={delta_real_n10000['delta_rel_max']:.6f}  "
      f"delta_rel_p95={delta_real_n10000['delta_rel_p95']:.6f}")

_anchor_rng = np.random.default_rng(DELTA_SEED)
_idx_2000 = _anchor_rng.choice(dist_matrix_.shape[0], size=2000, replace=False)
_real_n2000_D = np.array(dist_matrix_[np.ix_(_idx_2000, _idx_2000)], dtype=np.float64, copy=True)
delta_real_n2000 = gp.sampled_delta_hyperbolicity(_real_n2000_D, DELTA_QUADRUPLES, DELTA_SEED)
print(f"  real @ n=2000:      delta_rel_max={delta_real_n2000['delta_rel_max']:.6f}  "
      f"delta_rel_p95={delta_real_n2000['delta_rel_p95']:.6f}")

_tree_D = _weighted_tree_geodesics(2000, seed=DELTA_SEED)
delta_tree_n2000 = gp.sampled_delta_hyperbolicity(_tree_D, DELTA_QUADRUPLES, DELTA_SEED)
print(f"  tree @ n=2000:      delta_rel_max={delta_tree_n2000['delta_rel_max']:.6f}  "
      f"delta_rel_p95={delta_tree_n2000['delta_rel_p95']:.6f}")

_euclid_D = _flat_euclidean_geodesics(2000, 20, seed=DELTA_SEED)
delta_euclid_n2000 = gp.sampled_delta_hyperbolicity(_euclid_D, DELTA_QUADRUPLES, DELTA_SEED)
print(f"  flat R^20 @ n=2000: delta_rel_max={delta_euclid_n2000['delta_rel_max']:.6f}  "
      f"delta_rel_p95={delta_euclid_n2000['delta_rel_p95']:.6f}")

del _real_n2000_D, _tree_D, _euclid_D
gc.collect()

print()
print("  reading rule (02.1-PREREGISTRATION.md ## Delta-Hyperbolicity Reading Rule):")
print("  real (matched n=2000) <= tree anchor + 0.1 * (euclidean anchor - tree anchor)")
print("    => a single constant negative curvature is defensible, applied to delta_rel_max")
_delta_threshold = delta_tree_n2000["delta_rel_max"] + 0.1 * (
    delta_euclid_n2000["delta_rel_max"] - delta_tree_n2000["delta_rel_max"]
)
single_curvature_defensible = bool(delta_real_n2000["delta_rel_max"] <= _delta_threshold)
print(f"  tree anchor={delta_tree_n2000['delta_rel_max']:.6f}  "
      f"threshold={_delta_threshold:.6f}  "
      f"euclidean anchor={delta_euclid_n2000['delta_rel_max']:.6f}  "
      f"real@n=2000={delta_real_n2000['delta_rel_max']:.6f}")
print(f"  single_curvature_defensible = {single_curvature_defensible}")
print(f"  (falsifier condition (a) -- real value far from tree anchor -- is "
      f"{not single_curvature_defensible})")

# =========================================================================================
print()
print("=" * 78)
print("STEP 3 -- Krein bottom block: the 40 most-negative eigenpairs of B")
print("=" * 78)
_expected_B_bytes = 10_000 * 10_000 * 8
print(f"  rebuilding B (double-centred squared-distance matrix): "
      f"~{_expected_B_bytes / 1e9:.2f} GB; compute envelope budget ~4 GiB peak RSS")

t0 = time.perf_counter()
B = spectrum_from_distmatrix(dist_matrix_)
print(f"  B built in {time.perf_counter() - t0:.1f}s  shape={B.shape}  dtype={B.dtype}")
_print_rss("after building B")


def _compute_krein_bottom():
    t0 = time.perf_counter()
    vals, vecs = eigh(B, subset_by_index=[0, 39])
    print(f"  bottom-40 eigensolve: {time.perf_counter() - t0:.1f}s")
    return {
        "eigvals_bottom": vals.astype(np.float64),
        "eigvecs_bottom": vecs.astype(np.float64),
    }


_krein_bottom_cfg = {"fit_key": FIT_KEY, "block": "bottom", "block_width": 40}
_krein_bottom = npz_cache(f"krein_bottom_{FIT_KEY}", _krein_bottom_cfg, _compute_krein_bottom)
eigvals_bottom = _krein_bottom["eigvals_bottom"]
eigvecs_bottom = _krein_bottom["eigvecs_bottom"]
print(f"  eigvals_bottom.shape={eigvals_bottom.shape}  eigvecs_bottom.shape={eigvecs_bottom.shape}")
_print_rss("after bottom-block eigensolve")

_ref_bottom40 = np.sort(eigvals_all)[:40]
np.testing.assert_allclose(np.sort(eigvals_bottom), _ref_bottom40, rtol=1e-8)
print("  cross-check PASSED: bottom-40 eigenvalues agree with the 40 most-negative entries "
      "of Phase 2's eigvals_all to rtol=1e-8 -- this B is provably the same matrix the gate "
      "was computed from.")

del B, dist_matrix_
gc.collect()

# =========================================================================================
print()
print("=" * 78)
print(f"STEP 4 -- (p, q) distortion ladder: {len(P_LADDER)} x {len(Q_LADDER)} = "
      f"{len(P_LADDER) * len(Q_LADDER)} rungs")
print("=" * 78)

d2_geo = geo_pairs_r2.astype(np.float64) ** 2
_mass_capture_rows = gp.krein_mass_capture(eigvals_all, P_LADDER, Q_LADDER)
_mass_capture_by_pq = {(r["p"], r["q"]): r["mass_capture"] for r in _mass_capture_rows}

krein_ladder = []
for p in P_LADDER:
    ev_p, evec_p = eigvals_top[:p], eigvecs_top[:, :p]
    for q in Q_LADDER:
        ev_q, evec_q = eigvals_bottom[:q], eigvecs_bottom[:, :q]
        eigvals_sel = np.concatenate([ev_p, ev_q])
        eigvecs_sel = np.concatenate([evec_p, evec_q], axis=1)
        d2_rep = gp.pseudo_euclidean_sq_distances(eigvals_sel, eigvecs_sel, rows, cols)
        stats = gp.distortion_stats(d2_rep, d2_geo)
        krein_ladder.append({
            "p": int(p),
            "q": int(q),
            "median_abs_rel": stats["median_abs_rel"],
            "median_signed_rel": stats["median_signed_rel"],
            "p95_abs_rel": stats["p95_abs_rel"],
            "mass_capture": _mass_capture_by_pq[(p, q)],
            "n_negative_sq": int((d2_rep < 0).sum()),
        })

print(f"  {len(krein_ladder)} rungs computed")
_q0_rows = [row for row in krein_ladder if row["q"] == 0]
print(f"  q=0 (classical-MDS baseline) rows: {len(_q0_rows)}")
for row in sorted(_q0_rows, key=lambda r: r["p"]):
    print(f"    p={row['p']:3d} q=0    median_abs_rel={row['median_abs_rel']:.6f}  "
          f"mass_capture={row['mass_capture']:.6f}")

q0_best = min(r["median_abs_rel"] for r in _q0_rows)
full_ladder_best = min(r["median_abs_rel"] for r in krein_ladder)
reduction_fraction = 1.0 - (full_ladder_best / q0_best) if q0_best > 0 else 0.0
print()
print(f"  q=0 (classical-MDS) best median_abs_rel across P_LADDER: {q0_best:.6f}")
print(f"  full ladder best median_abs_rel across all {len(krein_ladder)} rungs: "
      f"{full_ladder_best:.6f}")
print(f"  reduction_fraction (full ladder best vs. q=0 best): {reduction_fraction:.6f}")
print("  (falsifier condition (b) is a judgment on this reduction, not a threshold fixed "
      "here -- reported plainly for plan 02.1-04, not adjudicated in this script.)")

print()
print("  working-dimension re-derivation (gate_verdict's elbow_criterion applied to the "
      "median_abs_rel-vs-dimension curve, in place of the Tenenbaum residual curve):")

_classical_x = np.array([r["p"] for r in sorted(_q0_rows, key=lambda r: r["p"])], dtype=np.float64)
_classical_y = np.array(
    [r["median_abs_rel"] for r in sorted(_q0_rows, key=lambda r: r["p"])], dtype=np.float64
)
classical_p = int(_classical_x[gp.kneedle_elbow(_classical_x, _classical_y)])

_by_total = {}
for row in krein_ladder:
    total = row["p"] + row["q"]
    if total not in _by_total or row["median_abs_rel"] < _by_total[total]["median_abs_rel"]:
        _by_total[total] = row
_frontier = sorted(_by_total.values(), key=lambda r: r["p"] + r["q"])
_frontier_x = np.array([r["p"] + r["q"] for r in _frontier], dtype=np.float64)
_frontier_y = np.array([r["median_abs_rel"] for r in _frontier], dtype=np.float64)
_frontier_elbow_idx = gp.kneedle_elbow(_frontier_x, _frontier_y)
krein_p = int(_frontier[_frontier_elbow_idx]["p"])
krein_q = int(_frontier[_frontier_elbow_idx]["q"])

print(f"  classical (q=0) elbow: p={classical_p}")
print(f"  pseudo-Euclidean best-per-total-dimension frontier elbow: "
      f"(p, q) = ({krein_p}, {krein_q})  total={krein_p + krein_q}")
print(f"  d_frozen (Phase 2, for reference; not adjudicated here) = {D_FROZEN}")

working_dimension = {
    "classical_p": classical_p,
    "krein_p": krein_p,
    "krein_q": krein_q,
    "criterion": ELBOW_CRITERION,
}

# =========================================================================================
print()
print("=" * 78)
print("STEP 5 -- correction blindness: real spectrum vs. two synthetic controls")
print("=" * 78)

cb_real = gp.correction_blindness(eigvals_all)

_flat_rng = np.random.default_rng(DELTA_SEED + 1)
_flat_X = _flat_rng.standard_normal((1000, 20))
_flat_D2 = squareform(pdist(_flat_X)) ** 2
_flat_B = -0.5 * (
    _flat_D2 - _flat_D2.mean(axis=1, keepdims=True) - _flat_D2.mean(axis=0, keepdims=True)
    + _flat_D2.mean()
)
_flat_eigvals = eigvalsh(_flat_B)
cb_synthetic_flat = gp.correction_blindness(_flat_eigvals)

_nonmetric_rng = np.random.default_rng(DELTA_SEED + 2)
_nonmetric_M = _nonmetric_rng.standard_normal((1000, 1000))
_nonmetric_M = (_nonmetric_M + _nonmetric_M.T) / 2
_nonmetric_eigvals = eigvalsh(_nonmetric_M)
cb_synthetic_nonmetric = gp.correction_blindness(_nonmetric_eigvals)

print(f"  {'':22s} {'m_before':>12s} {'m_after_clip':>14s} {'m_after_shift':>14s}")
for label, cb in (
    ("real (eigvals_all)", cb_real),
    ("synthetic_flat", cb_synthetic_flat),
    ("synthetic_nonmetric", cb_synthetic_nonmetric),
):
    print(f"  {label:22s} {cb['m_before']:>12.6f} {cb['m_after_clip']:>14.6f} "
          f"{cb['m_after_shift']:>14.6f}")
print()
print("  reading: m_after_clip and m_after_shift are 0.0 in every row above -- both "
      "corrections force a passing gate on any input by construction, so a passing "
      "corrected value is not evidence this data is flat (RESEARCH.md Pattern 1, Pitfall 2).")

# =========================================================================================
print()
print("=" * 78)
print("STEP 6 -- writing the measured artifact")
print("=" * 78)

artifact = _to_native({
    "fit_key": FIT_KEY,
    "r": r0,
    "m": m0,
    "n_positive": n_pos,
    "n_negative": n_neg,
    "pair_identity_verified": pair_identity_verified,
    "delta": {
        "real_n10000": delta_real_n10000,
        "quadruples": DELTA_QUADRUPLES,
        "seed": DELTA_SEED,
    },
    "delta_anchors": {
        "real_n10000": delta_real_n10000,
        "real_n2000": delta_real_n2000,
        "tree_n2000": delta_tree_n2000,
        "euclidean20d_n2000": delta_euclid_n2000,
    },
    "delta_reading_rule": {
        "threshold": _delta_threshold,
        "single_curvature_defensible": single_curvature_defensible,
    },
    "krein_ladder": krein_ladder,
    "working_dimension": working_dimension,
    "krein_falsifier_condition_b": {
        "q0_best_median_abs_rel": q0_best,
        "full_ladder_best_median_abs_rel": full_ladder_best,
        "reduction_fraction": reduction_fraction,
    },
    "correction_blindness": {
        "real": cb_real,
        "synthetic_flat": cb_synthetic_flat,
        "synthetic_nonmetric": cb_synthetic_nonmetric,
    },
    "d_frozen": D_FROZEN,
    "d_provisional": D_PROVISIONAL,
    "elbow_criterion": ELBOW_CRITERION,
    "gate_spectrum": GATE_SPECTRUM,
})

_artifact_cfg = {
    "fit_key": FIT_KEY,
    "p_ladder": list(P_LADDER),
    "q_ladder": list(Q_LADDER),
    "delta_quadruples": DELTA_QUADRUPLES,
    "delta_seed": DELTA_SEED,
    "pair_seed": PAIR_SEED,
    "pair_count": PAIR_COUNT,
}
written = json_cache(f"geometry_probes_{FIT_KEY}", _artifact_cfg, lambda: artifact)

_artifact_path = cache_path(f"geometry_probes_{FIT_KEY}", "json")
print(f"  written to: {_artifact_path}")
print()
for key, val in written.items():
    if isinstance(val, (dict, list)):
        print(f"  {key}: <{type(val).__name__}, {len(val)} entries>")
    else:
        print(f"  {key} = {val}")

_required_keys = {
    "fit_key", "r", "m", "n_positive", "n_negative", "pair_identity_verified",
    "delta", "delta_anchors", "krein_ladder", "working_dimension",
    "correction_blindness", "d_frozen", "d_provisional", "elbow_criterion",
}
_missing = _required_keys - set(written.keys())
assert not _missing, f"artifact missing required keys: {_missing}"

_print_rss("end of run")
print()
print("Done. All required keys present.")
