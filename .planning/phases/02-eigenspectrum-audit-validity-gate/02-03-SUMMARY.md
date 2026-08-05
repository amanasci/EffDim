---
phase: 02-eigenspectrum-audit-validity-gate
plan: 03
subsystem: data
tags: [numpy, scipy, sklearn, isomap, classical-mds, gate, json-artifact, jupyter]

requires:
  - phase: 02-eigenspectrum-audit-validity-gate
    provides: "02-01: R_STAT=0.052419, M_STAT=0.412071, GATE_VERDICT=FAIL, _gate_classify, spectrum stats, mds_eigenspectrum_43cf438bc944c509.npz"
  - phase: 02-eigenspectrum-audit-validity-gate
    provides: "02-02: ELBOW_D=5, ELBOW_D_CHECK=5, ELBOW_D_EIGEN=8, CURVE_DIVERGENCE_MAX=0.697664, D_FROZEN=5, mds_residuals_43cf438bc944c509.npz"
provides:
  - "gate_verdict_43cf438bc944c509.json — 21-key self-contained terminal verdict artifact (verdict, r, m, four thresholds, verdict_rule, spectrum stats, elbow + criterion, d_frozen, sweep ceiling, fit_key, k_star, n_components, d_provisional, three Phase 1 flags, three remediations, artifact names, timestamp, versions)"
  - "_require_gate(record, *, context='') — downstream enforcement reading its rule from the artifact, exercised on three synthetic records every run"
  - "§6.8 copyable enforcement block — the entire Phase 2 to Phase 3 interface"
affects: [phase-02.1-geometry-representation, phase-02.2-chart-autoencoder-validity, phase-3-decoder-curvature]

tech-stack:
  added: []
  patterns:
    - "fit_key-keyed verdict filename + thresholds in the json_cache manifest: editing a pre-registered constant raises a manifest mismatch instead of silently re-verdicting under the same name"
    - "Artifact written BEFORE the halt fires, so the evidence trail survives the assertion"
    - "Three-way synthetic self-test keeps the never-fires-on-real-data FAIL branch from rotting"
    - "FAIL message reads remediation strings FROM the record — no re-typed threshold literals"

key-files:
  created:
    - "notebooks/.cache/gate_verdict_43cf438bc944c509.json (gitignored, 3,929 B)"
    - "notebooks/.cache/gate_verdict_43cf438bc944c509.meta.json (gitignored, 151 B)"
  modified:
    - "notebooks/01_manifold_and_gate.ipynb (107 -> 115 cells; appended §6.7-§6.9) — SUBSEQUENTLY DELETED by quick task 260801-ovf, commit 8958488; recoverable at a2ca11f"

key-decisions:
  - "Task 3 (blocking human-verify) approved 2026-08-05 on the surviving artifact rather than a fresh Restart-and-Run-All: the notebook the ten-step verification targets was deleted by quick task 260801-ovf on 2026-08-01 as superseded by notebook 02. Eight of ten steps re-verified against the artifact, the cache, and the committed notebook at a2ca11f; two are unrepeatable and are recorded as such below."
  - "Remediation option 3 accepted — the documented FAIL is the milestone's reported outcome for this fit. Phase 3 is not planned in detail against Isomap coordinates; Phases 02.1/02.2/02.3 exist because of this verdict."
  - "D-14 upheld: enforcement stays notebook prose, no pu_manifold/gate.py. D-16 self-containment is the mitigation and was judged directly by reading the json with no notebook present."

requirements-completed: [SPEC-06, SPEC-07]

coverage:
  - {id: D1, description: "Verdict written as machine-readable gate_verdict_{fit_key}.json keyed by fit_key, self-contained across all 21 top-level keys, internally consistent (stored verdict equals verdict re-derived from the artifact's own r/m/thresholds)", requirement: "SPEC-06", verification: [{kind: other, ref: "Artifact read in isolation 2026-08-05: 21/21 keys present; r=0.0524192078526829 < r_max_pass=0.1, m=0.4120712514841815 > m_max_marginal=0.15 -> FAIL, equals stored verdict; k_star=15/n_components=18/flags match the Phase 1 handoff; 1 <= d_frozen=5 <= 18", status: pass}], human_judgment: true}
  - {id: D2, description: "On FAIL the notebook halts with all three remediation options enumerated, the options also written into the artifact, and the branch exercised against synthetic PASS/MARGINAL/FAIL records on every run", requirement: "SPEC-07", verification: [{kind: other, ref: "Committed notebook outputs at a2ca11f: cell 108 GATE HALT RuntimeWarning carrying all three remediations verbatim; cell 110 three-way self-test — PASS/MARGINAL return d_frozen without raising, FAIL raises and its caught message carries all three remediation strings; GATE_HALTED=True, GATE_HALTED_CONFIRMED=True", status: pass}], human_judgment: false}
  - {id: D3, description: "Boundary cleanliness — no library or helper-package drift from this plan", requirement: "SPEC-06", verification: [{kind: command, ref: "git diff --quiet -- pyproject.toml src/effdim/ notebooks/pu_manifold/ (clean); python -m pytest notebooks/pu_manifold/tests/test_pu_manifold.py -q (14 passed)", status: pass}], human_judgment: false}

duration: ~20min execution (2026-07-31) + phase-sealing checkpoint held 2026-07-31 to 2026-08-05
completed: 2026-08-05
status: complete
---

# Phase 2 Plan 3: Verdict Artifact, Enforcement, and Phase Close-Out

**`GATE_VERDICT = FAIL` — `r = 0.052419`, `m = 0.412071`, `d_frozen = 5`, `fit_key = 43cf438bc944c509`.**

`r` clears its own bound (0.052419 < 0.10). `m` does not clear even the MARGINAL bound
(0.412071 vs 0.15) — 5,029 of 10,000 eigenvalues negative, carrying 41% of absolute eigenvalue
mass. The Isomap geodesic matrix is not adequately Euclidean-embeddable at this fit. This is the
phase's entire product, and it is a complete, legitimate, reportable milestone outcome.

## The verdict artifact, inline

`notebooks/.cache/` is gitignored (D-13's accepted cost). This is where the artifact survives in-repo.

```json
{
  "phase": 2,
  "verdict": "FAIL",
  "r": 0.0524192078526829,
  "m": 0.4120712514841815,
  "thresholds": {
    "r_max_pass": 0.1,
    "m_max_pass": 0.05,
    "r_max_marginal": 0.25,
    "m_max_marginal": 0.15
  },
  "verdict_rule": "PASS requires r strictly below r_max_pass AND m strictly below m_max_pass; otherwise MARGINAL requires r strictly below r_max_marginal AND m strictly below m_max_marginal; otherwise FAIL. Every comparison is strict less-than -- a value exactly at a threshold does not clear it. Both conditions are conjunctions over the same two statistics at looser and tighter bounds, so the returned verdict is already the worse of r's and m's individual readings by construction; no separate max() step is taken.",
  "spectrum": {
    "n_positive": 4971,
    "n_negative": 5029,
    "lambda_max_pos": 3230.8539634646067,
    "lambda_min_neg": -169.35880545251558,
    "noise_floor": 7.173936918879702e-09,
    "dropoff_index": 2,
    "dropoff_ratio": 2.444713943099398
  },
  "elbow": 5,
  "elbow_criterion": "Maximum-curvature (kneedle) elbow on the Tenenbaum residual-variance curve (1 - R^2 between geodesic and embedded pairwise distances): both axes are normalized to [0, 1] by their own range, and the elbow is the point of greatest perpendicular distance from the chord connecting the curve's first and last normalized point. Swept over d = 1..40 (K_EFF, bounded by D_SWEEP_MAX=40 and N_POSITIVE=4971); ties broken to the lower d (ELBOW_TIE_BREAK='lower').",
  "elbow_check": {
    "elbow_check_draw": 5,
    "elbow_eigen_curve": 8,
    "curve_divergence_max": 0.6976644052911366
  },
  "d_frozen": 5,
  "d_sweep_max": 40,
  "fit_key": "43cf438bc944c509",
  "k_star": 15,
  "n_components": 18,
  "d_provisional": 18,
  "flags": {
    "short_circuit_risk": false,
    "k_auto_extended": false,
    "n_components_no_headroom": true
  },
  "artifacts": {
    "spectrum_npz": "mds_eigenspectrum_43cf438bc944c509.npz",
    "residuals_npz": "mds_residuals_43cf438bc944c509.npz",
    "isomap_joblib": "isomap_43cf438bc944c509.joblib"
  },
  "timestamp": "2026-08-01T01:32:29.593658+00:00",
  "versions": {
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit_learn": "1.9.0",
    "python": "3.14.6"
  }
}
```

`remediation` is reproduced verbatim in its own section below.

## Artifact sizes

| Artifact | Size |
|---|---|
| `mds_eigenspectrum_43cf438bc944c509.npz` | 9,455.7 KiB |
| `mds_residuals_43cf438bc944c509.npz` | 2.6 KiB |
| `gate_verdict_43cf438bc944c509.json` | 3.8 KiB |
| `gate_verdict_43cf438bc944c509.meta.json` | 151 B |
| `isomap_43cf438bc944c509.joblib` (Phase 1, referenced) | 1.55 GiB |
| `notebooks/.cache/` total at §6.9 close-out (2026-08-01) | 6.351 GiB |
| `notebooks/.cache/` total now (post-02.2 CAE artifacts) | 6.8 GiB |

## Self-test outcomes

Three synthetic records through `_require_gate`, run on every notebook execution (T-02-16, the
anti-rot measure for a branch that never fires on real data):

| verdict | raised? | result |
|---|---|---|
| PASS | no | returns `d_frozen=5`, prints clear-to-proceed |
| MARGINAL | no | returns `d_frozen=5`, boxed print + `warnings.warn` (D-15's three channels) |
| FAIL | yes | raises; caught message asserted to contain all three remediation strings read from the record |

Then `_require_gate(GATE_VERDICT_RECORD)` on the real record: raised, as designed.
`GATE_HALTED = True`, `GATE_HALTED_CONFIRMED = True`.

The FAIL message is composed from the record's own `remediation` field — no threshold literal is
re-typed inside `_require_gate`, verified by automated check. That is what makes the §6.8 copyable
block safe to paste into a downstream notebook: it cannot drift from the artifact it reads.

## Task 3 outcome — the phase-sealing human-verify gate

**Approved 2026-08-05. Phase 2 is sealed with `GATE_VERDICT = FAIL` as its complete outcome.**

The gate was held open from 2026-07-31. During the hold, quick task `260801-ovf` deleted
`notebooks/01_manifold_and_gate.ipynb` (commit `8958488`, "superseded by notebook 02's
k-sensitivity refit"). Six of Task 3's ten verification steps targeted that notebook directly, so
the ten-step protocol as written could not be re-run. The gate was approved on the surviving
evidence instead. Recorded honestly, step by step:

| # | Step | Status |
|---|---|---|
| 1 | Clean Restart-and-Run-All | **Not repeatable.** Notebook deleted. The committed copy at `a2ca11f` carries executed outputs end to end, zero error cells — an execution record, not a fresh run |
| 2 | Pre-registration ordering | **Verified by git ancestry.** `057b084` pre-registers the k-sensitivity re-fit before any fit runs; `9e4b274` pre-registers the cross-model sweep before any non-DINOv3 fit; §6.0 gate constants land in `3401c0c`, ahead of the verdict in `aea04ff` |
| 3 | Spectrum length 10,000 / float64 / sklearn cross-check | **Verified against the npz.** `eigvals_all` shape `(10000,)` `float64`; `n_positive=4971` + `n_negative=5029` = 10,000 |
| 4 | Equivalence guard (manual double-centring vs sklearn) | **Committed record only** (§6.1 outputs at `a2ca11f`) |
| 5 | Gate statistics vs thresholds + nine boundary cases | **Re-derived independently 2026-08-05.** From the artifact's own `r`/`m`/`thresholds`: FAIL, equal to the stored verdict. Nine boundary cases: committed record only |
| 6 | Figures | **Committed record only** (§6.4 outputs at `a2ca11f`) |
| 7 | Elbow agreement + freeze branch + nesting check | **Verified from the artifact.** `elbow=5`, `elbow_check_draw=5`, `elbow_eigen_curve=8`, `curve_divergence_max=0.697664`, `1 <= d_frozen=5 <= n_components=18`. Nesting to 1.207e-14 carried from 02-02 |
| 8 | Artifact read in isolation (D-16 self-containment) | **Verified directly, and this is the load-bearing one.** The json was opened with no notebook present: 21/21 keys, verdict re-derivable from its own fields, `verdict_rule` and `elbow_criterion` actionable prose, all three remediations verbatim, Phase 1 handoff values (`k_star=15`, `n_components=18`, three flags) matching. D-16 holds |
| 9 | Enforcement self-test + copyable block in a scratch session | **Not repeatable.** Notebook deleted. The three-way self-test's committed output at `a2ca11f` shows all branches exercised (table above) |
| 10 | Boundary cleanliness | **Verified 2026-08-05.** `git diff --quiet -- pyproject.toml src/effdim/ notebooks/pu_manifold/` clean; `pytest notebooks/pu_manifold/tests/` 14 passed |

Steps 1 and 9 are the two that cannot be reproduced. Both are re-execution checks on a notebook
that no longer exists; neither bears on the verdict, which is independently reconstructible from
`gate_verdict_43cf438bc944c509.json`, `mds_eigenspectrum_43cf438bc944c509.npz`, and
`mds_residuals_43cf438bc944c509.npz`. The verdict has since been reproduced four separate ways
(k-sensitivity re-fit at k ∈ {5,10,30}, paired HSC survey, ~90% disjoint resample, unnormalized
re-fit) — see `02-FINDINGS.md`.

`git show a2ca11f:notebooks/01_manifold_and_gate.ipynb` restores the 115-cell notebook with all ten
§6.N subsections (§6.0–§6.9) and §0–§6 sequence intact if a future reader wants the full re-run.

## Remediation options (verbatim from the artifact)

1. **Re-fit at a different `n_neighbors` and re-run Section 4 onward.** `n_neighbors` is a `fit_key`
   field, so this moves away from the current `fit_key` (43cf438bc944c509) and regenerates
   `isomap_{fit_key}.joblib`, `mds_eigenspectrum_{fit_key}.npz`, `mds_residuals_{fit_key}.npz`, and
   this verdict under the new key rather than overwriting them. Cost: a fresh Isomap fit plus a
   fresh dense eigensolve of the 10,000x10,000 geodesic matrix (roughly 100 seconds each on this
   machine). The pre-registered k-sensitivity re-fit already tested k in {5, 10, 30} against this
   k*=15 incumbent and found FAIL at every k, with m(k) flat to slightly increasing rather than
   decreasing (`02-REFIT-PREREGISTRATION.md`, Rule A) — so this option is not expected to change the
   outcome without also revisiting that pre-registration.

2. **Resample with a new seed and re-derive `row_indices` from Section 1's subsample draw.** This
   changes `subsample_key` and therefore `fit_key` both, so every downstream artifact in this
   notebook is regenerated under a new key. Cost: re-streaming and re-aligning the 10,000-row
   subsample, plus a fresh Isomap fit and eigensolve. This tests sampling variance rather than
   neighbourhood scale, a different axis from the one the k-sensitivity re-fit already closed.

3. **Accept the documented FAIL as the milestone's reported outcome.** No further re-fit is run;
   `gate_verdict_43cf438bc944c509.json` and this notebook's halt record stand as the complete
   evidence trail, and Phase 3 (decoder/curvature) is not planned in detail until a human makes this
   call at the phase-sealing checkpoint. This is a complete and legitimate milestone result, not a
   failure to be retried until it passes.

**Option 3 was taken.** Options 1 and 2 were subsequently tested anyway and both returned FAIL —
the k-sensitivity re-fit closed axis 1, the disjoint resample closed axis 2.

## Commits

1. §6.7 (Task 1) — `aea04ff`.
2. §6.8–§6.9 (Task 2) — `a2ca11f`.
3. Hold recorded at the Task 3 gate — `ac1d7d6`.
4. Task 3 checkpoint — human approval 2026-08-05, sealed by this summary.

Notebook subsequently deleted by `8958488` (quick task `260801-ovf`), outside this plan's scope.

## Decisions / Deviations

- **Deviation (external).** `files_modified: [notebooks/01_manifold_and_gate.ipynb]` no longer exists
  in the working tree. The deletion was a deliberate user-directed cleanup during the checkpoint
  hold, not a failure of this plan. Task 3 was closed on surviving artifacts by explicit decision;
  the two unrepeatable steps are named above rather than papered over.
- D-14 upheld — no `pu_manifold/gate.py`, no new tests, no `__init__` changes, no requirements
  changes, no Phase 3 notebook. `pyproject.toml`, `src/effdim/`, `notebooks/pu_manifold/`
  byte-identical to pre-plan state.
- SPEC-06 and SPEC-07 carried edge-probe assumptions (phase-wide count: 8 edges = 2 authored +
  6 flagged). "Machine-readable" = fixed-key JSON parseable without the notebook — since validated
  in practice by Phases 02.1 and 02.2, which read `fit_key=43cf438bc944c509` provenance without the
  notebook present. "Halts" = `assert False` per Phase 1's idiom. The complete-outcome framing
  rested on judgment and was ratified at the Task 3 gate.

## What Phase 3 inherits

**Not the Isomap coordinates.** The gate that invalidated them is now sealed.

- `GATE_VERDICT = FAIL` against `fit_key=43cf438bc944c509`, terminal for this fit.
- `d_frozen = 5` is the dimension **of record**, not a recommendation. `02-FINDINGS.md` §6.4 flags it
  as suspect: three independent estimates cluster at 18–25 (local PCA ~25, TwoNN ~19.5, Phase 1's
  eight geometric estimators 18) while the residual-curve elbow alone says 5 — consistent with the
  Tenenbaum curve saturating early under 41% negative mass, i.e. measuring the failure rather than
  the geometry. **Do not inherit `d_frozen=5` downstream.**
- The §6.8 copyable enforcement block is the entire Phase 2 → Phase 3 interface: compose
  `cache_path` from the consumer's own `fit_key`, open the json, branch three ways, bind `D_FROZEN`
  from `d_frozen`. It is prose in a deleted notebook — reproduced at `a2ca11f`, and any consumer can
  rewrite it from the 21-key contract above.
- Per the ROADMAP's hard gate, Phase 3 is not planned in detail against this input. Phases 02.1
  (geometry representation research), 02.2 (chart auto-encoder validity test, `CAE_VERDICT=FAIL`),
  and 02.3 (CAE iteration, proposed) exist because of this verdict. Phase 3 now depends on 02.3
  reaching PASS.

## Self-Check: PASSED

FOUND: `gate_verdict_43cf438bc944c509.json` (3,929 B) + `.meta.json` (151 B) in `notebooks/.cache/`;
commits `aea04ff`, `a2ca11f`, `ac1d7d6`; notebook recoverable at `a2ca11f` (115 cells, §6.0–§6.9
complete). `pytest notebooks/pu_manifold/tests/` 14 passed. Boundary diff clean.

NOT FOUND (expected): `notebooks/01_manifold_and_gate.ipynb` in the working tree — deleted by
`8958488`, recorded above.

---
*Phase: 02-eigenspectrum-audit-validity-gate* · *Completed: 2026-08-05*
