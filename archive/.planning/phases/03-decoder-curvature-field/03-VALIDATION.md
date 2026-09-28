---
phase: 3
slug: decoder-curvature-field
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-13
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Seeded by `/gsd-plan-phase 3` from `03-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing suite, 296+ tests as of 02.5-09, under `notebooks/pu_manifold/tests/`) |
| **Config file** | none — tests run by path, existing project convention |
| **Quick run command** | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_curvature_probe.py notebooks/pu_manifold/tests/test_derivative_bridge.py -x` |
| **Full suite command** | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/` |
| **Estimated runtime** | ~60s quick / ~300s full (02.5-09 recorded "286 passed" on the full suite) |

---

## Sampling Rate

- **After every task commit:** Run the quick command (the two edited modules' own test files)
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite green, **plus** the Step-1 Swiss-roll gate table printed and the D-15 floor decision recorded in the phase's own artifacts
- **Max feedback latency:** 60 seconds

---

## Phase Step → Evidence Map

The phase's four steps are staged so each gates or informs the next. The evidence needed to trust each differs in kind, not just degree. Source: `03-RESEARCH.md` § Validation Architecture.

| Step | Behavior | What would make it trustworthy | Sampling density |
|------|----------|-------------------------------|------------------|
| **1 — Swiss roll gate** | Median `rho_chart > 0.65` at best swept `n_charts`, ≥5 seeds | (a) sweep table reproduces `02.5-09`'s monotone-in-charts direction rather than noise; (b) `chart_curvature.py`'s shape/dtype/C2 guards still fire (unit tests, not just the sweep); (c) the raw-point `0.6712` context number reproduces unchanged — regression proof the D-08 edit did not touch the raw-point path | Full: every swept `n_charts` × every seed. No subsampling — this is the gate itself. D-15 forbids extra ceremony, not the sweep's own completeness. |
| **2 — PU field, one fit, one seed** | `‖H‖` + `cond(g)` per point, descriptive only | (a) shape assertions pass at real `D=768`, not just the roll's `D=3`; (b) `cond(g)` distribution reported, not only its extremes; (c) a wall-clock timing figure recorded — currently missing entirely at this scale | One fit is definitionally the whole step. Run the full unit suite before and after touching `chart_curvature.py`. |
| **3 — Seeds and sanity** | ≥3-seed spread, near-singular flags, finite/non-zero second derivatives, no extrapolation | (a) `derivative_bridge` output at PU scale with WR-01/02/03 fixed — both `full_hessian_agreement` and `reduced_mean_curvature_agreement`; (b) every evaluated point confirmed to be a chart-assigned coordinate via `chart_curvature_field`'s existing `assignment` machinery, never an off-manifold grid point | 3 seeds × 3 `n_charts` = 9 fits (D-13). Bridge on a representative subsample per config; re-derive the point budget for `d=20` — do not assume the `d=40` precedent transfers. |
| **4 — Synthetic control** | Same architecture/protocol on flat / sphere / saddle at matched `d=20, D=768` | (a) each fixture's own finite-difference cross-check passes, especially the new saddle (research Assumption A2); (b) fitted-decoder curvature compared to closed form via the **same** `curvature_fidelity_report` direction/magnitude/calibration split used for the roll, not a fresh ad-hoc comparison; (c) the parameterization-damage caveat printed alongside the numbers, not only in prose elsewhere | One fit per fixture type at the PU-matched config, minimum. If D-12's `d`-sweep escalation fires, re-fit the controls at the escalated `d` — do not leave them stale at `d=20`. |

---

## Per-Task Verification Map

Requirement IDs are **re-minted by the planner** (DEC-01..05 / CURV-01..08 are stale — written against Isomap coordinates and a global chart). Rows below are seeded from the research's Wave 0 gaps; the planner and `/gsd-validate-phase` fill task IDs and final requirement IDs.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | 0 | D-09 fwd/rev equivalence | — | Both autodiff paths agree to float64 round-off | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_curvature_probe.py -x -k equivalence` | ✅ (file exists) | ⬜ pending |
| TBD | TBD | 0 | D-14 / WR-01 | — | float64 guard receives the model, not a bound method; float32 model raises a friendly error | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_derivative_bridge.py -x -k float64` | ✅ (file exists) | ⬜ pending |
| TBD | TBD | 0 | D-14 / WR-02 | — | Relative-error columns robust against near-zero references | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_derivative_bridge.py -x -k relative` | ✅ (file exists) | ⬜ pending |
| TBD | TBD | 0 | D-14 / WR-03 | — | `calibrate_fd_step` chunks its autodiff Hessian (`z.shape[0] > VMAP_CHUNK`) | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_derivative_bridge.py -x -k chunk` | ✅ (file exists) | ⬜ pending |
| TBD | TBD | TBD | Step 4 synthetic controls | — | Flat exact-zero, sphere exact `d/R`, saddle FD cross-check | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_synthetic_controls.py -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | C2 guard on forward path | T-3-01 | D-08's toggle calls `assert_c2_activation` before differentiating, on **both** paths | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -x -k c2` | ✅ (file exists) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `notebooks/pu_manifold/tests/test_curvature_probe.py` — add D-09's forward/reverse equivalence test, mirroring `test_chart_curvature_dxd_solve_matches_explicit_projector`'s structure. Chart-curvature tests already live in this file; do **not** create a separate `test_chart_curvature.py`.
- [ ] `notebooks/pu_manifold/tests/test_derivative_bridge.py` — add WR-01/02/03 regression tests (float32-model-raises-friendly-error; near-zero-reference relative-error assertion; `z.shape[0] > VMAP_CHUNK` chunking assertion).
- [ ] `notebooks/pu_manifold/tests/test_synthetic_controls.py` — **new file**: flat-plane exact-zero test, sphere exact `d/R` test, saddle finite-difference cross-check (research Assumption A2 — the saddle is the only synthetic control with no existing tested analog).
- [ ] Framework install: **none needed** — pytest already in use.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Step-1 gate decision (median `rho_chart > 0.65`) | D-02, D-04, D-05a | The floor is a declared research bar, not a unit-test assertion. D-15 forbids verdict-JSON machinery — the sweep table is read by a human. | Run the Step-1 sweep runner; read the printed table; confirm the best-config median clears 0.65; confirm the multiple-comparisons caveat (D-04) is printed in the read-out. |
| CLAUDE.md Swiss roll sanity notebook | CLAUDE.md standing rule | Notebook must be committed executed with outputs; visual colour-band ordering in the x-z scatter is a human read. | Open `notebooks/03_swiss_roll_*_check.ipynb`; confirm ≤~15 cells, <2 min CPU, no `notebooks/.cache/` access, plots coloured by arc-length `t`, and 3–4 printed pass/fail lines. |
| Gate override + `n_charts` scope ruling recorded | D-05, domain § "precondition is deliberately overridden" | Both must appear in Phase 3's **own** artifacts, never inherited silently. This is a document-presence check with a judgment component. | Confirm the plan and the phase findings each state the 02.4 PASS override and its parameterization-damage consequence, and the D-05 ruling that opens `n_charts` across the 02.3 hold boundary. |
| Parameterization-damage caveat in the Step-4 read-out | Success criterion 5 | Requires the caveat to be stated *plainly alongside the numbers*, which is a prose-quality judgment. | Confirm the synthetic-control output states that a synthetic manifold passing never reproduces the fragmentation pathology the override carries. |

---

## Security Domain

ASVS L1, block on `high`. Finding, consistent with `02.6-REVIEW.md`'s prior assessment of this same codebase area: almost no standard ASVS category applies. No network surface, no untrusted input path, no persistence layer beyond gitignored local caches, no secrets.

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | No | No auth surface — local research notebooks/scripts only |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | **Partial** | Internal contract checks only (not user input): `_assert_float64` / `_assert_decode_batch_float64` dtype guards, `assert_c2_activation` / `assert_c2_decoder` activation-family guards, shape assertions on every `torch.func` composition. These catch programmer error and silent-wrong-answer, not adversaries. |
| V6 Cryptography | No | N/A |

**Threat patterns that do apply** — all data-integrity (silent wrong answer), not adversarial:

| Pattern | STRIDE | Mitigation |
|---------|--------|------------|
| ReLU-family decoder (zero 2nd derivative) silently zeroes `II` | Tampering (integrity) | `assert_c2_activation` / `assert_c2_decoder` **raise**, not warn — already implemented; D-08's forward path must call the guard too |
| Wrong `torch.func` transform composition | Tampering (integrity) | Shape assertions on every Jacobian/Hessian call |
| Undeclared `ripser`/`persim` dependency | Repudiation (result not reproducible from clean checkout) | Documented in `persistence_probe.py`'s import-guard error. **Inherited, not resolved, by this phase.** |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (`test_synthetic_controls.py`)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
