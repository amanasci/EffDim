# Phase 2: Eigenspectrum Audit & Validity Gate - Discussion Log

> Audit trail only. Decisions live in CONTEXT.md; this preserves the alternatives considered.

**Date:** 2026-07-31 · Four gray areas presented; user selected all four.

## Gate verdict rule

- **Statistics:** ratio + mass (`r` + `m`, worst-of) ✓ over ratio-only (blind to a diffuse
  tail: 500 negatives at 0.01·λ_max pass on r) and a three-stat panel (third threshold,
  redundant with SPEC-04 elbow).
- **Cutoffs:** fixed moderate (PASS r<0.10 ∧ m<0.05; MARGINAL r<0.25 ∧ m<0.15) ✓ over fixed
  strict (r<0.05 ∧ m<0.02; MARGINAL r<0.15 ∧ m<0.08 — raises FAIL odds on a convention) and calibration on controls
  (most defensible; costs 2+ fits and a comparability argument).
- **Pre-registration:** mirror §4.0 (constants cell + cell-index assert + verbatim copy into
  the artifact) ✓ over a committed constants module and both.
- **Flags:** spectral-only verdict, flags as provenance ✓ over flag-downgrade (inert — both
  False) and composite verdict.

## Elbow, d, re-fit

- **Criterion:** kneedle ✓ over variance cutoff (arbitrary constant; requirement says
  "elbow") and eigengap (fragile on smooth decay). d-range must be pre-registered too.
- **Curve:** both curves, elbow from Tenenbaum R² ✓ over Tenenbaum-only and eigenvalue-only
  (flatters d when negatives present). Divergence is itself a signal.
- **Elbow ≤ 18:** d = elbow, slice the cached embedding (nesting makes it exact) ✓ over
  d=18 always (near-noise directions into the decoder) and elbow-with-floor (invites
  post-hoc adjustment).
- **Elbow > 18:** halt, human decides (cost + exact constant in the message) ✓ over
  auto-re-fit (silent minutes + ~1.55 GiB on a Run All) and pre-emptive n_components=40.

## Spectrum compute

- **Centring:** in-place mean form + equivalence assert vs literal J form ✓ over literal J
  matmul (two 10k³ GEMMs) and float32 (its ~1e-7 error lives where r/m are measured).
- **Eigensolve:** split (eigvalsh all values + eigh subset top-K) ✓ over one full eigh
  (800 MB of unread vectors) and reusing cached embedding_ (only 18 columns).
- **Cache:** npz via `npz_cache`, keyed fit_key + K ✓ over eigenvalues-only and no cache.
- **Memory:** mmap + free + print peak RSS ✓ over plain load and a hard RAM assert.

## Verdict artifact

- **Location:** keyed `notebooks/.cache/gate_verdict_{fit_key}.json` via json_cache ✓ over a
  committed unkeyed copy (re-fit overwrites it) and both (sync step, can disagree).
- **Enforcement:** inline in the downstream notebook's first cell ✓ — **the one place the
  user declined the recommendation** (recommended: `pu_manifold/gate.py` with unit-testable
  `require_gate()`). Accepted cost recorded in D-14: the rule lives in prose, consumers can
  drift; mitigation = D-16 schema. Planner must not silently swap gate.py back in.
- **MARGINAL:** proceed, caveat re-printed downstream ✓ over ACK_MARGINAL halt and
  MARGINAL-as-FAIL (contradicts roadmap).
- **Artifact contents:** self-contained schema ✓ over minimal + pointer and self-contained +
  synthetic FAIL-path test — the rejected test carried into D-16 as a planner note, since
  D-14 removed the module that would have made it testable.

## Claude's Discretion

No "you decide" answers. Six open sub-decisions (listed in CONTEXT.md): d-range/K ceiling;
pair-subsample scheme; D-09 tolerance + matrix size; SPEC-02 dropoff definition; §6 layout
and figures; output hygiene.

## Deferred Ideas

None. Three consequences recorded in CONTEXT.md instead: D-14 drift risk, D-16 planner note,
Phase 1 D-05 carry-forward (CURV-06 controls matched on S⁷⁶⁷, not a flat plane).
