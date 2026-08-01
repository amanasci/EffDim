"""
The machine-readable Phase 02.1 -> Phase 3 handoff. Phase 3 reads this artifact -- and
``.planning/phases/02.1-geometry-representation-research/02.1-RECOMMENDATION.md``, the prose it
is assembled from -- before running any expensive cell.

This script computes nothing new. It builds a flat dict entirely from decisions already argued in
``02.1-RECOMMENDATION.md`` (GEOM-03's retention judgment, GEOM-04's recommendation and rejected
alternatives, GEOM-05's working-dimension re-derivation and its relationship to ``D_FROZEN``) plus
values read directly from two artifacts already on disk:
``notebooks/.cache/geometry_probes_43cf438bc944c509.json`` (plan 02.1-03's measured evidence) and
``notebooks/.cache/gate_verdict_43cf438bc944c509.json`` (Phase 2's verdict, for provenance). It
writes a third, self-contained artifact through ``pu_manifold.cache.json_cache`` -- mirroring the
shape ``notebooks/01_manifold_and_gate.ipynb``'s ``phase1_handoff_{fit_key}.json`` (S5.3) and
``gate_verdict_{fit_key}.json`` (S6.7) both use: a flat dict from already-decided values, a
``json_cache`` write, a nested-aware formatted dump, and a required-keys assertion before the
script declares itself done.

No package is installed and no expensive computation is run here -- this script only reads two
JSON files and writes a third.

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
print(f"  read geometry_probes_{FIT_KEY}.json and gate_verdict_{FIT_KEY}.json")

_wd = _probes["working_dimension"]
_krein_falsifier_b = _probes["krein_falsifier_condition_b"]
_delta_rule = _probes["delta_reading_rule"]

# --- ## GEOM-03 -- Correction or Retention: the argued judgment, as a short literal --------
RETENTION_STANCE = "retain"

# --- ## GEOM-04 -- Recommendation: one representation, named ------------------------------
REPRESENTATION = {
    "id": "pseudo_euclidean_krein",
    "name": (
        "Pseudo-Euclidean / Krein-space retention: top-p positive and top-q most-negative "
        "eigenpairs of the double-centred geodesic Gram matrix, distances computed under "
        "the signed bilinear form sum_a sign(lambda_a) * (x_i,a - x_j,a)^2"
    ),
}

# --- ## GEOM-05 -- Working Dimension: the pair, never collapsed to one integer, plus the --
# --- alternative this document names explicitly, and the criterion it was derived under ---
WORKING_DIMENSION = {
    "p": _wd["krein_p"],
    "q": _wd["krein_q"],
    "classical_p": _wd["classical_p"],
    "criterion": _wd["criterion"],
    "alternative_higher_budget": {
        "p": 40,
        "q": 25,
        "median_abs_rel": _krein_falsifier_b["full_ladder_best_median_abs_rel"],
        "reduction_fraction_vs_q0_baseline": _krein_falsifier_b["reduction_fraction"],
        "note": (
            "Real, measured 18.4% relative distortion reduction over the (8,0) default's "
            "q=0 baseline, but past the kneedle elbow in the flat-tail diminishing-returns "
            "region; adopt only if Phase 3's own decoder-side evidence shows the extra "
            "dimensions are worth their cost (02.1-RECOMMENDATION.md GEOM-04 Evidence "
            "check (2))."
        ),
    },
}

D_FROZEN_DISPOSITION = (
    "Revised: D_FROZEN=5 (Tenenbaum residual-curve elbow, 02-FINDINGS.md SS5) is superseded "
    "by this phase's re-derivation of the identical pre-registered kneedle criterion applied "
    "to the pseudo-Euclidean/classical distortion-vs-dimension curve, which lands on 8 -- "
    "corroborated by Phase 2's own independent eigenvalue-based residual curve, whose elbow "
    "is also 8 (CURVE_DIVERGENCE_MAX=0.697664 against the Tenenbaum curve's elbow of 5 at "
    "d=5). Not discarded: 5 is not declared simply wrong, only superseded by a directly "
    "comparable re-derivation. Not inherited: the working dimension of record is (p,q)=(8,0), "
    "not 5. The 18-25 local intrinsic-dimension cluster (TwoNN, local PCA) answers a "
    "different question -- local tangent-space dimension, not this global diminishing-"
    "returns distortion budget -- and is not thereby refuted (02.1-RECOMMENDATION.md GEOM-05)."
)

# --- ## GEOM-04 -- Evidence It Will Be Judged Against: checks Phase 3 can actually run ----
EVIDENCE_CRITERIA = [
    (
        "Held-out decoder reconstruction error (DEC-03/DEC-04) at the (p,q)=(8,0) working "
        "dimension should be roughly consistent with the ~12.1% median relative "
        "squared-distance distortion measured directly on those coordinates (krein_ladder "
        "p=8,q=0 row: median_abs_rel=0.121024). A decoder reconstructing far worse than this "
        "baseline indicates the coordinate representation, not the decoder, is the bottleneck."
    ),
    (
        "Whether the (p,q)=(40,25) alternative meaningfully improves decoder-side "
        "reconstruction or curvature-field stability (CURV-04, CURV-05) over the (8,0) "
        "default. If Phase 3 finds no benefit at the higher dimension either, that undercuts "
        "this recommendation's reading of the measured 18.4% distortion reduction as real "
        "signal rather than sampling noise on the 200,000-pair distortion statistic."
    ),
    (
        "CURV-01..CURV-03's curvature field, computed on the retained coordinates, must pass "
        "the CURV-06/CURV-07 synthetic-control discrimination before being trusted on the "
        "real fit -- unaffected in kind by this phase's representation choice, but the "
        "coordinate domain those controls are matched against is now this phase's chosen "
        "representation, not Isomap's."
    ),
    (
        "The falsifier's status (below) should be revisited if Phase 3's own evidence does "
        "not corroborate the measured 18.4% distortion reduction as meaningful -- a dated, "
        "documented amendment to this handoff, not a quiet reversal."
    ),
]

FALSIFIER = (
    "02.1-PREREGISTRATION.md's Falsifier required BOTH (a) DELTA_REL sitting far from the "
    "tree anchor -- indistinguishable from, or closer to, the flat-Euclidean anchor than the "
    "reading rule's threshold allows -- AND (b) the pseudo-Euclidean ladder never dropping "
    "DISTORTION materially below the q=0 classical-MDS baseline at any rung, to overturn the "
    "coordinate-producing verdict. Measured: condition (a) TRIPPED, by a wide margin "
    f"(delta_rel_max={_probes['delta']['real_n10000']['delta_rel_max']:.6f} at n=10,000 "
    f"exceeds the flat-Euclidean anchor "
    f"{_probes['delta_anchors']['euclidean20d_n2000']['delta_rel_max']:.6f}; "
    f"single_curvature_defensible={_delta_rule['single_curvature_defensible']}). Condition (b) "
    "did NOT cleanly trip (a real, if modest, "
    f"{_krein_falsifier_b['reduction_fraction']:.4f} relative distortion reduction was "
    "measured). Because the falsifier requires both and only one tripped, it did NOT fire -- "
    "the coordinate-producing verdict stands, as ratified in 02.1-FORK.md. Condition (a)'s "
    "margin eliminates the hyperbolic/mixed-curvature branch specifically; condition (b)'s "
    "modest 18.4% is the quantitative basis for retention over the classical-MDS baseline "
    "that already failed Phase 2's gate."
)

GENERALITY = {
    "status": "architecture-specific as stated, not yet shown to generalise",
    "reasoning": (
        "DINOv3 ViT-B/16 is the one variable never varied across every fit this milestone "
        "has produced (02-FINDINGS.md SS8); every number this handoff and "
        "02.1-RECOMMENDATION.md cite was computed on that one architecture's embedding of "
        "this one astronomical population."
    ),
    "sweep_would_change": (
        "The packaged 35-model cross-architecture sweep (sweep/, "
        "02-MODEL-SWEEP-PREREGISTRATION.md) is prepared but not yet run. If it shows other "
        "architectures pass Phase 2's eigenspectrum gate cleanly, this recommendation is "
        "specific to DINOv3's embedding geometry rather than a general decoder-input "
        "strategy. If it shows a similar diffuse-negative-mass pattern across architectures, "
        "that supports the representation generalising and strengthens the case (deferred in "
        "REQUIREMENTS.md Future Requirements under LIB-02) for eventually promoting this "
        "representation strategy into src/effdim/."
    ),
}

# --- Requirements: all 13 DEC/CURV IDs, bucketed exactly once, per 02.1-FORK.md's ---------
# --- ## What Phase 3 Inherits table for the coordinate-producing branch this phase chose --
REQUIREMENTS = {
    "unchanged": [
        "DEC-02", "DEC-03", "DEC-04", "DEC-05",
        "CURV-01", "CURV-02", "CURV-03", "CURV-04", "CURV-05", "CURV-06", "CURV-07",
    ],
    "amend": ["DEC-01", "CURV-08"],
    "rewrite": [],
}

SOURCE_ARTIFACTS = [
    f"gate_verdict_{FIT_KEY}.json",
    f"mds_eigenspectrum_{FIT_KEY}.npz",
    f"krein_bottom_{FIT_KEY}.npz",
    f"geometry_probes_{FIT_KEY}.json",
]

ALTERNATIVES_REJECTED = [
    {
        "candidate": "Hyperbolic (Poincare ball / Lorentz hyperboloid)",
        "reason": (
            "Its single-constant-negative-curvature assumption is directly contradicted: "
            "delta_rel_max=0.386330 at n=2,000 exceeds the flat-Euclidean anchor 0.360433, "
            "not merely falls short of the tree anchor. Cost also avoided: Riemannian SGD "
            "at ~1+ full-Isomap-fit wall-clock class, geoopt 0.5.1 (SUS), a blocking "
            "checkpoint. What would change the answer: a fit/sample/architecture whose "
            "delta-hyperbolicity reading lands near the tree anchor instead."
        ),
    },
    {
        "candidate": "Mixed-curvature product manifold",
        "reason": (
            "Shares hyperbolic's per-factor single-curvature commitment, which the same "
            "delta-hyperbolicity evidence argues against for any hyperbolic factor. Adds an "
            "unresolved factor-count/signature model-selection problem this survey never "
            "resolved. Not on the pre-registered 3-item shortlist. What would change the "
            "answer: a resolved factor/signature choice plus a per-factor delta test."
        ),
    },
    {
        "candidate": "Diffusion maps",
        "reason": (
            "Its only actively-maintained package (datafold 2.0.2) is Supporting-tier and "
            "strictly heavier than the shortlist needed, so 02.1-PREREGISTRATION.md's "
            "Shortlist Rule excluded it from evidence-gathering -- zero distortion "
            "measurement exists for it, unlike Krein's measured 18.4% reduction. What would "
            "change the answer: running its own kernel-build-plus-eigensolve probe and "
            "measuring its distortion curve against the q=0 baseline."
        ),
    },
    {
        "candidate": "Ollivier-Ricci curvature",
        "reason": (
            "Graph-native per 02.1-FORK.md's ratified fork verdict: produces a per-edge "
            "scalar, no coordinate vector, cannot supply DEC-01's decoder input. Cost also "
            "avoided: three SUS-flagged packages behind blocking checkpoints, an unbuilt "
            "synthetic fixture, a Sinkhorn-approximate OT solve over 150,000 edges. The "
            "measured 18.4% distortion reduction is evidence against, not merely a failure "
            "to trigger, a move toward this branch (falsifier's symmetric framing)."
        ),
    },
    {
        "candidate": "Forman-Ricci curvature",
        "reason": (
            "Identical rejection to Ollivier-Ricci -- graph-native, same fork-test "
            "incompatibility, same DEC/CURV rewrite bucket per 02.1-FORK.md. Lower-cost "
            "variant of the same graph-native slot (no optimal-transport solve needed), but "
            "that cost advantage does not change the fork-test outcome and was not "
            "independently evaluated -- Ollivier-Ricci already covers this branch's profile."
        ),
    },
]

# =============================================================================================
_handoff_built = {
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
    "source_artifacts": SOURCE_ARTIFACTS,
    "probe_artifact": f"geometry_probes_{FIT_KEY}.json",
    "alternatives_rejected": ALTERNATIVES_REJECTED,
}

_handoff_cfg = {
    "fit_key": FIT_KEY,
    "recommendation_doc": "02.1-RECOMMENDATION.md",
    "elbow_criterion": _wd["criterion"],
}

GEOMETRY_HANDOFF = json_cache(f"geometry_handoff_{FIT_KEY}", _handoff_cfg, lambda: _handoff_built)

_handoff_path = cache_path(f"geometry_handoff_{FIT_KEY}", "json")
print(f"\n  written to: {_handoff_path}")

print()
print("=== geometry_handoff (formatted dump) ===")
for _top_key, _top_val in GEOMETRY_HANDOFF.items():
    if isinstance(_top_val, dict):
        print(f"| {_top_key}")
        for _sub_key, _sub_val in _top_val.items():
            print(f"|   {_sub_key:28s} = {_sub_val}")
    elif isinstance(_top_val, list):
        print(f"| {_top_key}")
        for _i, _item in enumerate(_top_val):
            print(f"|   [{_i}] {_item}")
    else:
        print(f"| {_top_key:28s} = {_top_val}")

# --- Required-keys contract, asserted before the script declares itself done. --------------
_required_keys = {
    "phase", "fit_key", "representation", "retention_stance", "working_dimension",
    "d_frozen_disposition", "evidence_criteria", "falsifier", "generality", "requirements",
    "source_artifacts", "probe_artifact", "alternatives_rejected",
}
_missing = _required_keys - set(GEOMETRY_HANDOFF.keys())
assert not _missing, f"handoff missing required keys: {_missing}"

assert GEOMETRY_HANDOFF["phase"] == "02.1" and GEOMETRY_HANDOFF["fit_key"] == FIT_KEY

_req = GEOMETRY_HANDOFF["requirements"]
assert set(_req.keys()) == {"unchanged", "amend", "rewrite"}
_all_ids = list(_req["unchanged"]) + list(_req["amend"]) + list(_req["rewrite"])
assert len(_all_ids) == len(set(_all_ids)) == 13, (
    f"expected all 13 DEC/CURV requirement IDs bucketed exactly once, got {len(_all_ids)} "
    f"entries, {len(set(_all_ids))} unique: {sorted(_all_ids)}"
)

assert len(GEOMETRY_HANDOFF["alternatives_rejected"]) >= 5
assert all(
    set(a.keys()) >= {"candidate", "reason"} for a in GEOMETRY_HANDOFF["alternatives_rejected"]
)
assert len(GEOMETRY_HANDOFF["evidence_criteria"]) >= 2
assert GEOMETRY_HANDOFF["falsifier"].strip()
assert GEOMETRY_HANDOFF["working_dimension"]["criterion"].strip()

print(
    "\nAll thirteen required keys present; 13/13 DEC+CURV requirement IDs bucketed exactly "
    "once each across unchanged/amend/rewrite; 5 rejected alternatives recorded; falsifier "
    "and evidence criteria present."
)
