# Phase 2: Eigenspectrum Audit & Validity Gate - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-31
**Phase:** 2-Eigenspectrum Audit & Validity Gate
**Areas discussed:** Gate verdict rule, Elbow/d/re-fit, Spectrum compute, Verdict artifact

Four gray areas were presented; the user selected all four.

---

## Gate verdict rule

### Q1 — Which statistics compose the PASS/MARGINAL/FAIL verdict?

| Option | Description | Selected |
|--------|-------------|----------|
| Ratio + mass | `r = |λ_min_neg|/λ_max_pos` plus `m = Σ|λ_neg|/Σ|λ|`; verdict = worst of the two. `r` catches one large negative eigenvalue, `m` catches a long diffuse tail `r` reads as clean | ✓ |
| Ratio only | `r` alone — the diagnostic PITFALLS.md prescribes. Blind spot: 500 small negatives at `0.01·λ_max` pass on `r` while carrying real non-Euclidean mass | |
| Three-stat panel | `r`, `m`, plus top-`d` positive mass captured. Couples the gate to the dimension used downstream; costs a third threshold and is partly redundant with the SPEC-04 elbow | |
| You decide | Planner picks and justifies | |

**User's choice:** Ratio + mass (recommended)

### Q2 — Where do the r and m cutoffs come from?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed, moderate | PASS `r<0.10 ∧ m<0.05`; MARGINAL `r<0.25 ∧ m<0.15`; else FAIL. Justified by the conventional reading that `r` under ~0.1 is negligible | ✓ |
| Fixed, strict | PASS `r<0.05 ∧ m<0.02`; MARGINAL `r<0.15 ∧ m<0.08`. Raises the chance the milestone terminates at FAIL, which the roadmap does allow as a complete outcome | |
| Calibrate on controls | Derive cutoffs from a known-Euclidean control (Gaussian cloud on S⁷⁶⁷) and a known-curved one at matched n and k. Most defensible; costs 2+ extra Isomap fits and an argument that `r`/`m` compare across n | |
| You decide | Planner picks numbers | |

**User's choice:** Fixed, moderate (recommended)
**Notes:** The strict variant was explicitly weighed against the FAIL risk for foundation-model
geodesic matrices, which routinely carry a visible negative tail.

### Q3 — How are the thresholds pre-registered so they provably precede the spectrum?

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror §4.0 | §6.0 constants cell + cell-index assertion that it runs before double-centring + thresholds copied verbatim into `gate_verdict.json` | ✓ |
| Committed constants | Thresholds in a versioned module committed before any spectrum cell runs; git history is the timestamp proof. Stronger against a quiet cell edit, but invisible to a reader of the notebook alone | |
| Both | Committed module imported by the §6.0 cell, plus assertion and verbatim copy. Costs a module and a separate commit step | |
| You decide | Planner picks | |

**User's choice:** Mirror §4.0 (recommended)

### Q4 — Do the Phase 1 provenance flags feed the verdict, or just ride along?

| Option | Description | Selected |
|--------|-------------|----------|
| Spectral only | Verdict is a pure function of `r` and `m`; flags, `k*`, `fit_key` recorded as provenance only | ✓ |
| Flags can downgrade | `short_circuit_risk` or `k_auto_extended` caps the result at MARGINAL. Honest about D-11's tension, but inert — both are False and a re-fit keeps `k*=15` | |
| Composite verdict | Flags as first-class verdict inputs, worst-of. Most machinery for conditions currently all False | |
| You decide | Planner picks | |

**User's choice:** Spectral only (recommended)

---

## Elbow, d, re-fit

### Q1 — What criterion locates the residual-variance elbow (SPEC-04)?

| Option | Description | Selected |
|--------|-------------|----------|
| Kneedle | Max-curvature knee via explicit implementation. Deterministic, one stated formula, no magic constant. Sensitive to the d-axis range, so that range must be pre-registered too | ✓ |
| Variance cutoff | Smallest `d` with residual below a pre-registered constant. Trivially reproducible; the constant is arbitrary and the requirement's word is "elbow" | |
| Eigengap | Largest relative drop between consecutive positive eigenvalues. Connects to SPEC-02's dropoff; fragile on a smoothly decaying spectrum | |
| You decide | Planner picks | |

**User's choice:** Kneedle (recommended)

### Q2 — How is the residual-variance curve itself computed?

| Option | Description | Selected |
|--------|-------------|----------|
| Both, elbow from R² | Tenenbaum `1 − R²(D_geodesic, D_embedded)` as the criterion curve, eigenvalue-based `1 − cumsum(λ_pos)/Σλ_pos` as a free cross-check; divergence is itself a non-Euclideanity signal. Forces eigenvectors out of the spectrum step | ✓ |
| Tenenbaum R² only | The canonical Isomap definition. Needs eigenvectors and a per-`d` distance recomputation | |
| Eigenvalue-based only | Free with a values-only eigvalsh, no eigenvectors. Not the Isomap paper's residual variance, and flatters `d` when negatives are present | |
| You decide | Planner picks | |

**User's choice:** Both, elbow from R² (recommended)

### Q3 — If the elbow lands at or below 18, what is the frozen d?

| Option | Description | Selected |
|--------|-------------|----------|
| d = elbow, slice | Freeze at the elbow, take columns `0..d-1` of the cached 18-d embedding — nested eigenvectors make the slice exact, no re-fit. Smaller decoder input, better-conditioned first fundamental form, tighter CURV-08 support | ✓ |
| d = 18 always | Keep the fit dimension, report the elbow as diagnostic only. Carries near-noise directions into decoder and curvature; frozen `d` would not follow from the SPEC-04 curve | |
| Elbow with floor | `d = max(elbow, floor)` to guard a degenerate spectrum. Another pre-registered constant that invites post-hoc adjustment | |
| You decide | Planner picks | |

**User's choice:** d = elbow, slice (recommended)

### Q4 — If the elbow exceeds 18, what does the notebook do?

| Option | Description | Selected |
|--------|-------------|----------|
| Halt, human decides | Stop with elbow value, required `n_components`, cost (fresh fit, minutes, ~1.55 GiB, new `fit_key`), and the exact constant to change. Matches D-11 and SPEC-07 posture | ✓ |
| Auto re-fit | Refit under the new `fit_key` and continue in one pass. A Run All then silently spends minutes and disk, and the fitted dimension becomes execution-order dependent | |
| Pre-emptive wide fit | Refit once at `n_components=40` up front. Removes the branch, pays the re-fit unconditionally, quietly reverses D-12's no-headroom choice | |
| You decide | Planner picks | |

**User's choice:** Halt, human decides (recommended)

---

## Spectrum compute

### Q1 — How is B = -0.5·J D² J formed at n = 10,000?

| Option | Description | Selected |
|--------|-------------|----------|
| Mean form, asserted | In-place row/column-mean centring; algebraically identical, no 10k³ GEMMs, ~one extra 800 MB array. Guarded by an equivalence assertion against the literal J form on a small random matrix | ✓ |
| Literal J matmul | The textbook form from PITFALLS.md. Nothing to justify; two dense 10,000³ matmuls and several GB peak | |
| float32 | Halves every array. Hazard: `r` and `m` measure the near-zero end of the spectrum, exactly where float32's ~1e-7 error can manufacture or erase the negative tail | |
| You decide | Planner picks | |

**User's choice:** Mean form, asserted (recommended)

### Q2 — How are eigenvalues and eigenvectors obtained?

| Option | Description | Selected |
|--------|-------------|----------|
| Split solve | `eigvalsh(B)` for all 10,000 values (the negative tail needs them), plus `eigh(B, subset_by_index=[n-K, n-1])` for the top-`K` vectors the R² curve needs. LAPACK syevr, deterministic, consistent with D-15 | ✓ |
| One full eigh | Single call, all values and vectors. Simplest; extra 800 MB resident and more wall clock for ~9,950 vectors nothing reads | |
| Values + reuse cached | eigvalsh plus the cached `embedding_` columns. Zero extra eigen-work, but only 18 columns exist so the residual curve cannot extend past `d=18` | |
| You decide | Planner picks solver and K | |

**User's choice:** Split solve (recommended)

### Q3 — Does the spectrum get its own cache artifact?

| Option | Description | Selected |
|--------|-------------|----------|
| npz cache | Via the existing `npz_cache`, keyed on `fit_key` + `K`: all eigenvalues, top-`K` vectors, both residual curves. Under 10 MB; makes §6 re-runs near-instant | ✓ |
| Eigenvalues only | Tiny artifact; recomputes the expensive subset eigh every pass | |
| No cache | Always provably derived from the live joblib; every Run All pays full centring and eigensolve | |
| You decide | Planner picks | |

**User's choice:** npz cache (recommended)

### Q4 — How does §6 handle the 1.55 GiB pickle and peak memory?

| Option | Description | Selected |
|--------|-------------|----------|
| mmap + free | `mmap_mode="r"`, extract `dist_matrix_`, drop the Isomap object before centring, print peak RSS. Cold runs within a few GB; warm runs skip the load | ✓ |
| Plain load | One line, no mmap caveats. Peak sits at 800 MB resident plus every centring intermediate | |
| mmap + budget assert | Adds a hard pre-flight RAM assertion. Costs a platform-dependent probe and a number needing revision | |
| You decide | Planner picks | |

**User's choice:** mmap + free (recommended)

---

## Verdict artifact

### Q1 — Where does gate_verdict.json live and how is it keyed?

| Option | Description | Selected |
|--------|-------------|----------|
| Keyed in .cache/ | `notebooks/.cache/gate_verdict_{fit_key}.json` via `json_cache`. Verdict inseparably bound to the fit audited; a re-fit yields a new key and a new file, so no stale PASS. Gitignored, so not repo-visible | ✓ |
| Committed, unkeyed | `notebooks/gate_verdict.json` in git, `fit_key` as a field. Repo-visible; one file a re-fit overwrites, mismatch caught only if a consumer checks | |
| Both | Keyed authoritative copy plus committed review copy. Costs a sync step; the two can disagree | |
| You decide | Planner picks | |

**User's choice:** Keyed in .cache/ (recommended)

### Q2 — How does Phase 3 check the gate before spending compute?

| Option | Description | Selected |
|--------|-------------|----------|
| New gate.py | `pu_manifold/gate.py` with verdict computation, artifact writer, and `require_gate()` raising on FAIL and on a missing file. Unit-testable alongside the existing 14 tests; extends D-02's package to five modules | |
| Inline in notebook 02 | First cell opens the JSON and asserts the verdict itself. No new module; the check is visible where it matters. The rule then lives in notebook prose, so later consumers re-implement it and can drift | ✓ |
| Helper in cache.py | Loader and check in the existing module, package stays at four. Muddies cache.py — a caching contract, not a policy gate | |
| You decide | Planner picks | |

**User's choice:** Inline in notebook 02 — **the one place the user declined the recommendation.**
**Notes:** Recorded in CONTEXT.md D-14 as a deliberate tension with the accepted cost named
explicitly (Phase 4 re-implements the rule and can drift), and the mitigation assigned to
D-16's self-contained schema. The planner must not silently swap `gate.py` back in.

### Q3 — What does MARGINAL mean operationally for Phase 3?

| Option | Description | Selected |
|--------|-------------|----------|
| Proceed, caveat rides | Proceeds per the roadmap's stated dependency, but verdict and `r`/`m` are re-printed at the top of notebook 02 and carried into every downstream artifact | ✓ |
| Explicit acknowledgement | Halts until a human sets `ACK_MARGINAL`. Stronger against a Run-All sailing past a borderline spectrum; manual step on every fresh clone | |
| Treat as FAIL | Only PASS proceeds. Contradicts the roadmap's Phase 3 dependency on "a PASS or MARGINAL gate verdict" | |
| You decide | Planner picks | |

**User's choice:** Proceed, caveat rides (recommended)

### Q4 — What does the artifact carry, and what does a FAIL halt say (SPEC-07)?

| Option | Description | Selected |
|--------|-------------|----------|
| Self-contained | Verdict, `r`, `m`, thresholds verbatim, elbow + criterion, frozen `d`, sweep range and `K`, `fit_key`, `k*`, Phase 1 flags, timestamp, versions; on FAIL the enumerated remediation list in both artifact and halt message | ✓ |
| Minimal + pointer | Verdict, stats, `d`, `fit_key`, pointer to §6. A future reader with the JSON alone cannot tell which thresholds produced the verdict | |
| Self-contained + halt test | The full schema plus a test exercising the FAIL path on a synthetic verdict. SPEC-07's halt is a branch that will never fire on real data — exactly the kind that rots untested | |
| You decide | Planner picks | |

**User's choice:** Self-contained (recommended)
**Notes:** The rejected halt test is carried into CONTEXT.md D-16 as a planner note rather than
dropped — D-14's inline check removed the module that would otherwise have made it testable.

---

## Claude's Discretion

The user selected a concrete option on every question — no "you decide" answers. Six items were
named as open sub-decisions during the discussion and never locked; they are listed in
CONTEXT.md under Claude's Discretion:

- The d-axis sweep range / `K` eigenvector ceiling (one number, pre-registered in §6.0)
- The point-pair subsampling scheme for the Tenenbaum R² residual at n=10,000
- The tolerance and matrix size for the D-09 mean-form equivalence assertion
- How SPEC-02's steep-dropoff location is defined and plotted, distinct from the elbow
- The §6 sub-section layout and figure set
- Cell-output hygiene for §6's figures, inheriting §0.4

The last two were offered as additional gray areas at the close of discussion; the user chose to
proceed to context rather than explore them.

## Deferred Ideas

None — discussion stayed within phase scope. No scope creep was raised.

Three items were recorded in CONTEXT.md as downstream consequences or accepted costs rather than
deferred ideas: D-14's drift risk, D-16's planner note on the untested FAIL path, and the
carried-forward Phase 1 D-05 consequence that Phase 3's CURV-06 controls must be matched on
S⁷⁶⁷ rather than a flat plane.
