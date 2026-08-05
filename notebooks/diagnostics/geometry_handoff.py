"""
The machine-readable Phase 02.1 -> Phase 3 handoff. Phase 3 reads this artifact -- and
``.planning/phases/02.1-geometry-representation-research/02.1-RECOMMENDATION.md``, the prose it
is assembled from -- before running any expensive cell.

This script computes nothing new. It builds a flat dict entirely from decisions already argued in
``02.1-RECOMMENDATION.md`` (GEOM-03's retention judgment, GEOM-04's recommendation and rejected
alternatives, GEOM-05's working-dimension disposition) plus values read directly from four
artifacts already on disk: ``geometry_probes_43cf438bc944c509.json`` (plan 02.1-03's measured
evidence), ``signature_transfer_43cf438bc944c509.json`` (the measurement that fired the
falsifier), ``stress_family_rescaled_43cf438bc944c509.json`` (the flat-target wall), and
``gate_verdict_43cf438bc944c509.json`` (Phase 2's verdict, for provenance). It writes a fifth,
self-contained artifact through ``pu_manifold.cache.json_cache`` -- the same shape
``gate_verdict_{fit_key}.json`` uses: a flat dict from already-decided values, a ``json_cache``
write, a nested-aware formatted dump, and a required-keys assertion before the script declares
itself done.

REVISED 2026-08-05 for the graph-native branch. The original recommended pseudo-Euclidean/Krein
retention; ``02.1-AMENDMENT-01.md`` rejected it by user directive and ``02.1-AMENDMENT-02.md``
fired the pre-registered falsifier under an amended reading of condition (b). The handoff now
carries Ollivier-Ricci on the frozen k*=15 graph. Two shape changes follow from the branch switch
and are recorded as deviations in ``02.1-04-SUMMARY.md``:

  * ``requirements`` gains a fourth bucket, ``dropped``. Nine DEC/CURV requirements are
    inapplicable on this branch rather than modified; forcing them into ``rewrite`` would
    misreport nine dropped requirements as nine rewritten ones.
  * ``working_dimension`` carries ``applicable: false`` plus the branch's actual scale parameters.
    A per-edge curvature has no embedding dimension. The coordinate branch's re-derived (p, q) is
    retained under ``coordinate_branch_rederivation`` so rejecting Amendment 02 costs no re-work.

No package is installed and no expensive computation is run here -- this script only reads four
JSON files and writes a fifth.

Invoke with: PYTHONPATH=notebooks python notebooks/diagnostics/geometry_handoff.py
"""

import json
from pathlib import Path

from pu_manifold.cache import cache_path, json_cache

CACHE = "notebooks/.cache"
FIT_KEY = "43cf438bc944c509"

# =============================================================================================
print("=" * 78)
print(f"Phase 02.1 -> Phase 3 geometry handoff -- fit_key = {FIT_KEY}")
print("=" * 78)

_probes = json.loads(Path(f"{CACHE}/geometry_probes_{FIT_KEY}.json").read_text())
_gate = json.loads(Path(f"{CACHE}/gate_verdict_{FIT_KEY}.json").read_text())
_sigxfer = json.loads(Path(f"{CACHE}/signature_transfer_{FIT_KEY}.json").read_text())
_stress = json.loads(Path(f"{CACHE}/stress_family_rescaled_{FIT_KEY}.json").read_text())
print(
    f"  read geometry_probes / gate_verdict / signature_transfer / "
    f"stress_family_rescaled for {FIT_KEY}"
)

_wd = _probes["working_dimension"]
_krein_b = _probes["krein_falsifier_condition_b"]
_delta_rule = _probes["delta_reading_rule"]

# --- ## GEOM-03 -- Correction or Retention: unchanged by either amendment ------------------
# The stance is about never correcting the negativity away. The graph-native branch honours it
# more strictly than any coordinate representation: it never forms B, never eigendecomposes it,
# and never asks the metric to be PSD.
RETENTION_STANCE = "retain"

# --- ## GEOM-04 -- Recommendation: one representation, named ------------------------------
REPRESENTATION = {
    "id": "ollivier_ricci",
    "name": (
        "Ollivier-Ricci discrete curvature on the frozen k*=15 geodesic graph: per-edge "
        "kappa(x,y) = 1 - W1(m_x, m_y) / d(x,y), with m_x a lazy-random-walk measure on the "
        "neighbourhood of x, W1 the optimal-transport distance between neighbourhood measures, "
        "and d the cached graph geodesic distance"
    ),
    "branch": "graph_native",
    "cross_check": (
        "Forman-Ricci, combinatorial, retained as the cheap cross-check -- agreement raises "
        "confidence, disagreement localises where the validated geodesic weights carry "
        "information the combinatorics miss"
    ),
    "produces": "one scalar per graph edge; intrinsic Ricci, NOT an extrinsic mean-curvature vector",
}

# --- ## GEOM-05 -- Working Dimension: the question dissolves on this branch ----------------
WORKING_DIMENSION = {
    "applicable": False,
    "reason": (
        "Ollivier-Ricci assigns a scalar to each of the k*=15 graph's 150,000 edges. There is no "
        "coordinate vector, no target space, and therefore no dimension to freeze or sweep."
    ),
    "criterion": _wd["criterion"],
    "scale_parameters": {
        "graph": (
            "k* = 15, exactly 150,000 edges, codiag_k15_43cf438bc944c509.npz -- frozen and "
            "mandatory; rebuilding at k != 15 reopens the closed k-sensitivity question "
            "(02-REFIT-PREREGISTRATION.md Rule A)"
        ),
        "random_walk_laziness_alpha": (
            "the alpha in m_x = alpha*delta_x + (1-alpha)*uniform(N(x)); MUST be pre-registered "
            "before any curvature number is computed, with its sensitivity reported"
        ),
        "sinkhorn_epsilon": (
            "entropic regularisation strength; exact OT is intractable over 150,000 edges so the "
            "approximation is necessary, not chosen. Curvature distribution MUST be measured at "
            ">= 3 epsilon values; if the sign structure moves with epsilon the signal is "
            "contaminated and the result is not usable at this scale"
        ),
    },
    "coordinate_branch_rederivation": {
        "note": (
            "Retained so that rejecting 02.1-AMENDMENT-02.md costs no re-derivation. This is the "
            "answer GEOM-05 carries if the coordinate-producing verdict is reinstated."
        ),
        "classical_p": _wd["classical_p"],
        "krein_p": _wd["krein_p"],
        "krein_q": _wd["krein_q"],
    },
}

D_FROZEN_DISPOSITION = (
    f"Discarded as inapplicable -- NOT declared wrong. D_FROZEN={_gate['d_frozen']} "
    f"(Tenenbaum residual-curve elbow, 02-FINDINGS.md SS5) parameterises an embedding, and the "
    f"recommended branch produces no embedding. The GEOM-05 question dissolves here rather than "
    f"receiving a different numeric answer. The pre-registered prohibition forbids declaring "
    f"D_FROZEN wrong or inheriting it silently; neither happens. On the coordinate branch the "
    f"re-derivation under the identical kneedle criterion landed on (p,q)=("
    f"{_wd['krein_p']},{_wd['krein_q']}), corroborated by Phase 2's own eigenvalue-based residual "
    f"elbow of 8; that value is preserved under working_dimension.coordinate_branch_rederivation. "
    f"The intrinsic-dimension cluster (local PCA ~25, TwoNN ~19.5, estimator median 18) keeps a "
    f"role as a caveat on how well the k*=15 neighbourhoods resolve local structure -- Phase 3's "
    f"threats, not its dimension."
)

# --- ## GEOM-04 -- Evidence It Will Be Judged Against, in the order Phase 3 runs them ------
EVIDENCE_CRITERIA = [
    {
        "id": "fixture_before_data",
        "check": (
            "Synthetic weighted tree (known negative curvature) and near-regular expander (known "
            "near-zero/positive) at matched scale and edge count. No curvature number on the real "
            "graph is trusted until the estimator recovers known sign and rough magnitude on "
            "both."
        ),
        "status": "design-only today; building it is Phase 3's first task, not an assumed input",
    },
    {
        "id": "sinkhorn_bias",
        "check": (
            "Curvature distribution at >= 3 regularisation strengths epsilon, trend reported."
        ),
        "status": "untested; the regularisation bias is assumption 3 of the recommendation",
    },
    {
        "id": "ollivier_forman_agreement",
        "check": "Per-edge rank correlation between the metric and combinatorial curvatures.",
        "status": "untested",
    },
    {
        "id": "distribution_vs_negative_mass",
        "check": (
            "Curvature-distribution shape read against Phase 2's 5,029 negative eigenvalues "
            "carrying 41% of absolute mass. A predominantly negative distribution is "
            "independently consistent; a predominantly-positive or near-zero one is a genuine "
            "tension between two independent measurements of the same geometry and must be "
            "resolved before Phase 3 proceeds. NOTE: graph-native is scored on this axis and on "
            "whether it answers CURV-01..03's question -- NEVER on the distortion axis, which the "
            "pre-registration's Prohibited Moves calls a category error."
        ),
        "status": "untested",
    },
]

FALSIFIER = (
    "Pre-registered (02.1-PREREGISTRATION.md): the coordinate-producing verdict is overturned iff "
    "BOTH (a) real DELTA_REL sits far from the tree anchor -- indistinguishable from or closer to "
    "the flat-Euclidean anchor than the reading rule allows -- AND (b) the (p,q) ladder never "
    f"drops DISTORTION materially below the q=0 baseline at any rung up to 40. MEASURED: (a) "
    f"trips, delta_rel_max=0.383921 past the flat-Euclidean anchor 0.360433 against threshold "
    f"{_delta_rule['threshold']:.6f}; single_curvature_defensible="
    f"{_delta_rule['single_curvature_defensible']}. (b) under the ORIGINAL reading does NOT trip "
    f"-- the ladder drops to {_krein_b['full_ladder_best_median_abs_rel']:.6f} from a q=0 best of "
    f"{_krein_b['q0_best_median_abs_rel']:.6f}, a {_krein_b['reduction_fraction'] * 100:.1f}% "
    f"relative reduction. 02.1-AMENDMENT-02.md re-reads (b) to require the drop be realisable in "
    f"a form Phase 3's Euclidean-latent C2 decoder can consume; under that reading (b) TRIPS, "
    f"because signature_transfer measures the same coordinates at "
    f"{_sigxfer['distortion_euclidean']:.6f} with the signature discarded and "
    f"{_sigxfer['distortion_positive_block_only']:.6f} positive-block-only, both above the "
    f"{_sigxfer['flat_floor']:.6f} flat floor. BOTH trip -> the falsifier FIRES and the "
    "coordinate-producing verdict is overturned toward graph-native, which is the "
    "pre-registration's own stated consequence. The symmetric falsifier now governs this branch: "
    "graph-native is overturned by a coordinate candidate whose DISTORTION drops materially below "
    "the classical failure at a tractable working dimension, in a form Phase 3 can consume. Krein "
    "at (40,25) meets the first half and fails the second; a signature-aware decoder reopens it."
)

GENERALITY = {
    "status": "architecture-specific as stated, not yet shown to generalise",
    "reasoning": (
        "DINOv3 ViT-B/16 is the one variable never varied across every fit this milestone has "
        "produced (02-FINDINGS.md SS8); every number this handoff and 02.1-RECOMMENDATION.md cite "
        "was computed on that one architecture's embedding of this one astronomical population."
    ),
    "sweep_would_change": (
        "The packaged 35-model cross-architecture sweep (sweep/, "
        "02-MODEL-SWEEP-PREREGISTRATION.md, not yet run) bears directly: other architectures "
        "passing the Phase 2 gate cleanly would make this recommendation DINOv3-specific; a "
        "similar diffuse negative-mass pattern across architectures would support generalising "
        "and strengthen the LIB-02 promotion case."
    ),
}

# --- Phase 3 Re-specification Brief -> the DEC/CURV buckets --------------------------------
# Per 02.1-FORK.md's "Graph-native (not selected)" accounting, now selected. Four buckets: the
# fourth ("dropped") is the deviation this branch forces -- see the module docstring.
REQUIREMENTS = {
    "dropped": [
        "DEC-01", "DEC-02", "DEC-03", "DEC-04", "DEC-05",
        "CURV-01", "CURV-02", "CURV-04", "CURV-05",
    ],
    "rewrite": ["CURV-03", "CURV-06", "CURV-07"],
    "amend": ["CURV-08"],
    "unchanged": [],
}
REQUIREMENTS_RATIONALE = {
    "dropped": (
        "Nothing to decode, so no decoder: DEC-01..05 inapplicable. No decoder, so no "
        "Jacobian/Hessian path (CURV-01/02), no metric tensor to condition (CURV-04), no decoder "
        "second derivatives to verify (CURV-05)."
    ),
    "rewrite": (
        "CURV-03's extrinsic mean-curvature vector does not exist on this branch -- replaced by "
        "an intrinsic Ricci scalar per edge, a different mathematical object, not a renaming. "
        "CURV-06/07's synthetic control re-architects from flat-plane/sphere/saddle onto the "
        "tree-and-expander fixture."
    ),
    "amend": (
        "CURV-08 referent change: 'curvature only evaluated at or near the actual Isomap "
        "coordinates' becomes 'curvature only on the actual k*=15 graph's edges, never a "
        "resampled or reweighted graph'."
    ),
    "traceability": (
        "Adopting this recommendation requires a REQUIREMENTS.md traceability update as part of "
        "Phase 3 planning. This is the asymmetric consequence 02.1-FORK.md priced: the coordinate "
        "branch re-opens nothing, the graph-native branch re-opens all thirteen."
    ),
}

ALTERNATIVES_REJECTED = [
    {
        "candidate": "pseudo-Euclidean / Krein retention",
        "reason": (
            f"This document's own former recommendation. WON under the pre-registered criterion "
            f"({_krein_b['full_ladder_best_median_abs_rel']:.6f} at (40,25), "
            f"{_krein_b['reduction_fraction'] * 100:.1f}% below the q=0 best) and remains the only "
            f"measured arm below the ~0.0796 wall. Rejected first by user directive "
            f"(02.1-AMENDMENT-01.md SS1.3), then on independent measured grounds: the advantage is "
            f"signature-carried and does not transfer to a Euclidean latent "
            f"({_sigxfer['distortion_euclidean']:.6f}, signature_discard_penalty "
            f"{_sigxfer['signature_discard_penalty']:.6f}), so Phase 3's decoder cannot realise "
            f"it. Its elbow rung is (8,0) -- zero negative directions -- so the pre-registered "
            f"working dimension would have put Phase 3 on a flat representation regardless."
        ),
        "would_change": (
            "A Phase 3 decoder with an indefinite or signature-aware latent, or a Neuc-MDS-style "
            "optimised bilinear selection measurably better than top-p/top-q truncation. Closest "
            "loser by a wide margin; first thing to revisit if graph-native stalls."
        ),
    },
    {
        "candidate": "metric SMACOF",
        "reason": (
            f"Best decoder-consumable arm at "
            f"{_stress['anchors']['metric_smacof_d18_raw']:.6f}, statistically indistinguishable "
            f"from Isomap's own {_stress['anchors']['isomap_d18']:.6f}. Rejected because it IS the "
            f"wall: it reproduces by a structurally unrelated route (stress majorization, no "
            f"eigendecomposition, no PSD constraint) the classical result Phase 2 already gated as "
            f"FAIL. Adopting it carries the failed geometry into Phase 3 under a new name."
        ),
        "would_change": (
            "Nothing available -- its result is evidence about the target space, not the optimiser."
        ),
    },
    {
        "candidate": "hyperbolic (Poincare / Lorentz)",
        "reason": (
            f"Constant negative curvature contradicted by delta_rel_max=0.383921 sitting PAST the "
            f"flat-Euclidean anchor 0.360433, against threshold {_delta_rule['threshold']:.6f}. "
            f"This geometry is LESS tree-like than flat space under the pre-registered estimator; "
            f"single_curvature_defensible={_delta_rule['single_curvature_defensible']}."
        ),
        "would_change": "A fit, sample, or architecture whose delta lands near the tree anchor.",
    },
    {
        "candidate": "mixed-curvature product",
        "reason": (
            "Same delta evidence per constant-curvature factor, plus an unresolved "
            "factor/signature model-selection cost. Never in the pre-registered shortlist."
        ),
        "would_change": "A resolved signature plus a per-factor delta test.",
    },
    {
        "candidate": "ambient Riemannian via geomstats (hypersphere)",
        "reason": (
            "Measured and lost by an order of magnitude: great-circle 0.769597, tangent PCA "
            "0.780607 at best d=65. Not a verdict on the library -- both arms measure AMBIENT "
            "sphere geometry, ignoring the ~20-25-d manifold inside S^767. geomstats has no "
            "fit-to-a-geodesic-matrix primitive. Carries a known defect as a dependency: 2.8.0 "
            "imports numpy.trapz, removed in numpy 2.0, ImportError under the pinned 2.5.1."
        ),
        "would_change": (
            "Committing to a specific manifold and fitting points to it -- Phase 3 pipeline work "
            "this phase excludes."
        ),
    },
    {
        "candidate": "Laplacian eigenmaps / LLE / non-metric MDS",
        "reason": (
            f"Measured and lost even after re-scoring each with an optimal isotropic scale fitted "
            f"to its own advantage: "
            f"{_stress['verdict']['table']['laplacian_eigenmaps_d18']:.6f} / "
            f"{_stress['verdict']['table']['lle_standard_d18']:.6f} / "
            f"{_stress['verdict']['table']['nonmetric_mds_d18']:.6f}. Non-metric MDS is the "
            f"instructive one -- the textbook answer to non-Euclidean input, whose rank-order-only "
            f"objective discards exactly the metric information being measured."
        ),
        "would_change": "Nothing; these are structural properties of the objectives.",
    },
    {
        "candidate": "diffusion maps",
        "reason": (
            "Never measured, so recommending it would violate the evidence-strength prohibition. "
            "Not eliminated by delta. Most plausible COORDINATE-PRODUCING fallback if the "
            "graph-native branch stalls, but it is a flat-target method and the SMACOF result "
            "predicts the same wall."
        ),
        "would_change": (
            "Running its own probe (~1 dense-eigensolve class) and measuring its distortion curve."
        ),
    },
]

SOURCE_ARTIFACTS = {
    "recommendation": (
        ".planning/phases/02.1-geometry-representation-research/02.1-RECOMMENDATION.md"
    ),
    "preregistration": (
        ".planning/phases/02.1-geometry-representation-research/02.1-PREREGISTRATION.md"
    ),
    "fork": ".planning/phases/02.1-geometry-representation-research/02.1-FORK.md",
    "survey": ".planning/phases/02.1-geometry-representation-research/02.1-SURVEY.md",
    "amendment_01": (
        ".planning/phases/02.1-geometry-representation-research/02.1-AMENDMENT-01.md"
    ),
    "amendment_02": (
        ".planning/phases/02.1-geometry-representation-research/02.1-AMENDMENT-02.md"
    ),
    "phase_2_findings": (
        ".planning/phases/02-eigenspectrum-audit-validity-gate/02-FINDINGS.md"
    ),
    "gate_verdict": f"gate_verdict_{FIT_KEY}.json",
    "graph": f"codiag_k15_{FIT_KEY}.npz",
    "coordinate_branch_only": [
        f"mds_eigenspectrum_{FIT_KEY}.npz",
        f"krein_bottom_{FIT_KEY}.npz",
    ],
}

PROBE_ARTIFACT = {
    "primary": f"geometry_probes_{FIT_KEY}.json",
    "falsifier_deciding_measurement": f"signature_transfer_{FIT_KEY}.json",
    "losing_arms": [
        f"geomstats_eval_{FIT_KEY}.json",
        f"stress_family_eval_{FIT_KEY}.json",
        f"stress_family_rescaled_{FIT_KEY}.json",
    ],
    "note": (
        "Losing arms are kept so the coordinate branch does not have to be re-measured if this "
        "recommendation is revisited. Every arm verified pair_identity_verified=true against the "
        "same fixed 200,000-pair sample."
    ),
}

INSTALLS_REQUIRED = {
    "GraphRicciCurvature": "absent, SUS -- blocking checkpoint:human-verify, NOT pre-approved here",
    "POT": "absent, SUS -- blocking checkpoint:human-verify, NOT pre-approved here",
    "networkx": (
        "already present at 3.6.1 in .venv, arrived transitively. 02.1-FORK.md recorded it as "
        "'absent entirely'; that is now out of date and the branch's install cost is two "
        "packages, not three."
    ),
}

# =============================================================================================
_cfg = {
    "fit_key": FIT_KEY,
    "representation_id": REPRESENTATION["id"],
    "retention_stance": RETENTION_STANCE,
    "branch": REPRESENTATION["branch"],
    "working_dimension_applicable": WORKING_DIMENSION["applicable"],
    "amendments_applied": ["02.1-AMENDMENT-01", "02.1-AMENDMENT-02"],
}


def _build() -> dict:
    return {
        "phase": "02.1",
        "fit_key": FIT_KEY,
        "representation": REPRESENTATION,
        "retention_stance": RETENTION_STANCE,
        "working_dimension": WORKING_DIMENSION,
        "d_frozen_disposition": D_FROZEN_DISPOSITION,
        "evidence_criteria": EVIDENCE_CRITERIA,
        "falsifier": FALSIFIER,
        "generality": GENERALITY,
        "requirements": REQUIREMENTS,
        "requirements_rationale": REQUIREMENTS_RATIONALE,
        "alternatives_rejected": ALTERNATIVES_REJECTED,
        "installs_required": INSTALLS_REQUIRED,
        "source_artifacts": SOURCE_ARTIFACTS,
        "probe_artifact": PROBE_ARTIFACT,
    }


GEOMETRY_HANDOFF = json_cache(f"geometry_handoff_{FIT_KEY}", _cfg, _build)
print(f"  wrote {cache_path(f'geometry_handoff_{FIT_KEY}', 'json').name}")

# --- Formatted dump, nested-aware -----------------------------------------------------------
print(f"\n=== Phase 02.1 -> Phase 3 handoff ({len(GEOMETRY_HANDOFF)} top-level keys) ===")
for _top_key in sorted(GEOMETRY_HANDOFF):
    _top_val = GEOMETRY_HANDOFF[_top_key]
    if isinstance(_top_val, dict):
        print(f"| {_top_key}")
        for _k, _v in sorted(_top_val.items()):
            print(f"|   {_k:32s} = {str(_v)[:150]}")
    elif isinstance(_top_val, list):
        print(f"| {_top_key} ({len(_top_val)} entries)")
        for _i, _v in enumerate(_top_val):
            if isinstance(_v, dict):
                _label = _v.get("candidate") or _v.get("id") or f"[{_i}]"
                print(f"|   [{_i}] {_label}")
            else:
                print(f"|   [{_i}] {str(_v)[:150]}")
    else:
        print(f"| {_top_key:28s} = {str(_top_val)[:150]}")

# --- Required-keys contract, asserted before the script declares itself done. --------------
_required_keys = {
    "phase", "fit_key", "representation", "retention_stance", "working_dimension",
    "d_frozen_disposition", "evidence_criteria", "falsifier", "generality", "requirements",
    "source_artifacts", "probe_artifact", "alternatives_rejected",
}
_missing = _required_keys - set(GEOMETRY_HANDOFF.keys())
assert not _missing, f"handoff missing required keys: {_missing}"

assert GEOMETRY_HANDOFF["phase"] == "02.1" and GEOMETRY_HANDOFF["fit_key"] == FIT_KEY

# Four buckets now: "dropped" is the deviation the graph-native branch forces (module docstring).
_req = GEOMETRY_HANDOFF["requirements"]
assert set(_req.keys()) == {"unchanged", "amend", "rewrite", "dropped"}
_all_ids = (
    list(_req["unchanged"]) + list(_req["amend"]) + list(_req["rewrite"]) + list(_req["dropped"])
)
assert len(_all_ids) == len(set(_all_ids)) == 13, (
    f"expected all 13 DEC/CURV requirement IDs bucketed exactly once, got {len(_all_ids)} "
    f"entries, {len(set(_all_ids))} unique: {sorted(_all_ids)}"
)
_expected_ids = {f"DEC-0{i}" for i in range(1, 6)} | {f"CURV-0{i}" for i in range(1, 9)}
assert set(_all_ids) == _expected_ids, (
    f"bucketed ids do not match the DEC/CURV register: "
    f"missing {_expected_ids - set(_all_ids)}, unexpected {set(_all_ids) - _expected_ids}"
)

assert len(GEOMETRY_HANDOFF["alternatives_rejected"]) >= 5
assert all(
    set(a.keys()) >= {"candidate", "reason"} for a in GEOMETRY_HANDOFF["alternatives_rejected"]
)
assert len(GEOMETRY_HANDOFF["evidence_criteria"]) >= 2
assert GEOMETRY_HANDOFF["falsifier"].strip()
assert GEOMETRY_HANDOFF["working_dimension"]["criterion"].strip()

# The branch switch is the whole point of this revision -- assert it did not silently regress.
assert GEOMETRY_HANDOFF["representation"]["id"] == "ollivier_ricci"
assert GEOMETRY_HANDOFF["representation"]["branch"] == "graph_native"
assert GEOMETRY_HANDOFF["working_dimension"]["applicable"] is False
assert GEOMETRY_HANDOFF["d_frozen_disposition"].startswith("Discarded as inapplicable")
assert any(
    a["candidate"].startswith("pseudo-Euclidean")
    for a in GEOMETRY_HANDOFF["alternatives_rejected"]
), "the rejected former recommendation must appear in alternatives_rejected"

print(
    f"\nAll thirteen required keys present; {len(_all_ids)}/13 DEC+CURV requirement IDs bucketed "
    f"exactly once each across dropped/rewrite/amend/unchanged; "
    f"{len(GEOMETRY_HANDOFF['alternatives_rejected'])} rejected alternatives recorded (including "
    f"the former recommendation); falsifier and evidence criteria present; branch = "
    f"{GEOMETRY_HANDOFF['representation']['branch']}."
)
