---
phase: 02-eigenspectrum-audit-validity-gate
verified: 2026-08-05
verdict: PASS
requirements: [SPEC-01, SPEC-02, SPEC-03, SPEC-04, SPEC-05, SPEC-06, SPEC-07]
plans_complete: 3/3
method: inline (orchestrator, no verifier subagent)
---

# Phase 2 Verification — Eigenspectrum Audit & Validity Gate

**Verdict: PASS.** The phase delivered what it promised. Note the distinction that governs this
whole document: the *phase* passes verification because it correctly built and ran a gate; the
*gate* returned FAIL. Those are different objects. A gate that fires is a gate that works.

Verified inline by the orchestrator rather than by a `gsd-verifier` subagent, against the surviving
artifacts, the cache, the committed notebook at `a2ca11f`, and the three plan summaries.

## Success criteria

| # | Criterion (ROADMAP) | Requirement | Evidence | Status |
|---|---|---|---|---|
| 1 | Full classical-MDS eigenspectrum computed by manual double-centring of `isomap.dist_matrix_`, never from truncated `kernel_pca_.eigenvalues_` | SPEC-01 | `mds_eigenspectrum_43cf438bc944c509.npz` → `eigvals_all` shape `(10000,)` `float64` — the full n×n spectrum, not the 18-component truncation. In-place mean-form double-centring verified equal to the literal J-form on two 50×50 inputs; symmetry max deviation 1.421e-14 against a 2.132e-09 bound. Cross-checked against `kernel_pca_.eigenvalues_` at rtol=1e-8, worst diff 8.532e-15 | PASS |
| 2 | Leading eigenvalues confirmed large/positive with steep dropoff located; negative-eigenvalue magnitude reported against a stated, justified threshold | SPEC-02, SPEC-03 | Artifact `spectrum`: `lambda_max_pos=3230.85`, `dropoff_index=2`, `dropoff_ratio=2.4447`, `lambda_min_neg=-169.36`, `n_positive=4971`, `n_negative=5029`, `noise_floor=7.17e-09`. `m=0.412071` reported against pre-registered `m_max_pass=0.05` / `m_max_marginal=0.15`; `r=0.052419` against `r_max_pass=0.10` / `r_max_marginal=0.25`. Classifier boundaries asserted on nine synthetic cases before real values entered it | PASS |
| 3 | Residual-variance-vs-dimension curve's elbow identified by a stated criterion, not eyeballed | SPEC-04 | `elbow=5` with `elbow_criterion` carried in the artifact as full actionable prose (normalized-axes kneedle, max perpendicular distance from the endpoint chord, swept d=1..40, ties to lower d). Stability: `elbow_check_draw=5` on a disjoint 200,000-pair draw — exact agreement. `elbow_eigen_curve=8` reported alongside, never substituted as the freeze source | PASS |
| 4 | Chosen embedding dimension `d` frozen and recorded before any decoder is trained | SPEC-05 | `d_frozen=5`, frozen in 02-02 (`539dafa`) via the classical-MDS nesting slice, nesting verified to worst relative difference 1.207e-14; `1 <= 5 <= n_components=18` asserted. No decoder trained in this phase | PASS |
| 5 | PASS/MARGINAL/FAIL verdict written as a machine-readable artifact downstream notebooks check before running; on FAIL the notebook halts with remediation options enumerated | SPEC-06, SPEC-07 | `gate_verdict_43cf438bc944c509.json`, 21/21 keys, read in isolation with no notebook present and found self-contained: verdict re-derivable from its own `r`/`m`/`thresholds`, `verdict_rule` and `elbow_criterion` as actionable prose, three remediations verbatim, Phase 1 handoff values matching. FAIL halt fired (`GATE_HALTED=True`, `GATE_HALTED_CONFIRMED=True`) carrying all three remediations read from the record. Three-way synthetic self-test exercises PASS/MARGINAL/FAIL every run | PASS |

## Boundary and hygiene checks

| Check | Result |
|---|---|
| `git diff --quiet -- pyproject.toml src/effdim/ notebooks/pu_manifold/` | clean — no library drift, honouring the v1.1 "do not modify `src/effdim/`" constraint |
| `python -m pytest notebooks/pu_manifold/tests/test_pu_manifold.py -q` | 14 passed |
| Plans with SUMMARY.md | 3/3 |
| Requirements completed | SPEC-01..07, all seven |
| Pre-registration ordering (git ancestry, not assertion) | `057b084` pre-registers the k-sensitivity re-fit before any fit runs; `9e4b274` pre-registers the cross-model sweep before any non-DINOv3 fit; §6.0 gate constants (`3401c0c`) precede the verdict (`aea04ff`) |

## Gate outcome

`GATE_VERDICT = FAIL` against `fit_key=43cf438bc944c509`. `r=0.052419` clears its bound; `m=0.412071`
fails even the MARGINAL bound — 5,029 of 10,000 eigenvalues negative, carrying 41% of absolute
eigenvalue mass. The Isomap geodesic matrix is not adequately Euclidean-embeddable at this fit.

Reproduced four independent ways since (`02-FINDINGS.md`): k-sensitivity re-fit at k ∈ {5,10,30} — FAIL
at every k with m(k) flat-to-rising; paired HSC survey — m=0.4226; ~90% disjoint resample —
m=0.411948 with identical positive/negative counts; unnormalized re-fit — m moves 0.28%. Local
intrinsic dimension is stable and tight (TwoNN 19.5, local PCA median 25.0), so the cloud *is* a
manifold — one whose geodesic metric is strongly non-Euclidean.

Remediation option 3 accepted at the phase-sealing checkpoint: the documented FAIL is the
milestone's reported outcome for this fit.

## Caveats carried forward

- **`d_frozen = 5` is the dimension of record, not a recommendation.** `02-FINDINGS.md` §6.4 flags it
  suspect against three estimates clustering at 18–25. Do not inherit it downstream.
- **`n_components = 18` sat below the measured intrinsic dimension** — every fit this phase was
  dimension-starved. This does not move `r`/`m`, which derive from the full 10,000-value spectrum
  independently of `n_components`.
- **Two verification steps are unrepeatable.** `notebooks/01_manifold_and_gate.ipynb` was deleted by
  quick task `260801-ovf` (`8958488`) during the checkpoint hold. A fresh Restart-and-Run-All and a
  fresh scratch-session enforcement run cannot be performed. Both are re-execution checks; neither
  bears on the verdict, which is independently reconstructible from the cached npz/json. Notebook
  recoverable at `a2ca11f` (115 cells, §6.0–§6.9 intact). Full step-by-step accounting in
  `02-03-SUMMARY.md`.
- **Code review: not applicable.** Phase 2's entire source footprint across all six plan commits is
  one file, `notebooks/01_manifold_and_gate.ipynb`, deleted at HEAD. Nothing survives to review.
- **`02-SECURITY.md` not produced.** The ASVS-level-1 threat register (T-02-13 … T-02-19) lives in
  the plans with per-threat mitigations, verified through the artifact. Run `/gsd-secure-phase 2`
  if a formal security artifact is wanted.

## Phase 3 handoff

Not the Isomap coordinates — the gate that invalidated them is now sealed. The §6.8 copyable
enforcement block is the entire Phase 2 → Phase 3 interface: compose `cache_path` from the
consumer's own `fit_key`, open the json, branch three ways, bind `D_FROZEN` from `d_frozen`. Per the
ROADMAP's hard gate, Phase 3 is not planned in detail against this input; it now depends on Phase
02.3 reaching PASS.
