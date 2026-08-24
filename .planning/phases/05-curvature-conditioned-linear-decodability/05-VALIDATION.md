---
phase: 5
slug: curvature-conditioned-linear-decodability
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false  # Wave 0 gaps are plan 05-01 Tasks 1-3
created: 2026-08-24
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: `05-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest`, invoked via `.venv/bin/python -m pytest` (matches `test_region_partition.py`'s own invocation, per `04-VERIFICATION.md` Behavioral Spot-Checks) |
| **Config file** | none — no pytest config detected in `notebooks/pu_manifold/`; tests run by direct file path, one test file per module |
| **Quick run command** | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py -q` |
| **Full suite command** | `.venv/bin/python -m pytest notebooks/pu_manifold/tests -q` |
| **Estimated runtime** | ~5 s quick / ~90 s full suite (286+ tests) |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py -q`
- **After every plan wave:** Run `.venv/bin/python -m pytest notebooks/pu_manifold/tests -q`
- **Before `/gsd-verify-work`:** Full suite must be green, **and** `--selfcheck` mode must be green
  before any real PU number is computed (mirrors `region_partition_mknn_run.selfcheck()`'s role
  as the phase's automated implementation check)
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

> Task IDs are assigned by the planner; the Requirement column carries CONTEXT.md decision IDs
> because ROADMAP maps no REQ-IDs to this phase (`phase_req_ids: TBD`).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-T2 | 05-01 | 1 | D5-01 | — | N/A | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_fit_probe_shape_and_row_alignment -x` | ❌ W0 | ⬜ pending |
| 05-01-T2 | 05-01 | 1 | RESEARCH A3 | — | N/A | unit, known-answer | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_r2_matches_per_point_residual_aggregate -x` | ❌ W0 | ⬜ pending |
| 05-01-T2 | 05-01 | 1 | D5-04 / A1 | T-05-08 | N/A | unit, known-answer | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_pool_seeds_no_single_seed_dominates -x` | ❌ W0 | ⬜ pending |
| 05-01-T2 | 05-01 | 1 | D5-06 | — | N/A | unit (xfail until 05-04-T2) | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_curvature_convention_matches_sealed_modules -x` | ❌ W0 | ⬜ pending |
| 05-01-T2 | 05-01 | 1 | D5-07 / D5-09 | T-05-09 | N/A | unit, known-answer | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_bucket_assignment_known_answer -x` | ❌ W0 | ⬜ pending |
| 05-01-T2 | 05-01 | 1 | D5-08 | T-05-16 | N/A | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_size_matched_check_uses_test_split_counts -x` | ❌ W0 | ⬜ pending |
| 05-01-T2 | 05-01 | 1 | D5-10 | T-05-03 | guard raises rather than computes | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_assert_preregistered_raises_when_absent -x` | ❌ W0 | ⬜ pending |
| 05-01-T1 | 05-01 | 1 | Known-answer self-check | T-05-03 | synthetic only, never PU | integration, `--selfcheck` | `.venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --selfcheck` | ❌ W0 | ⬜ pending |
| 05-01-T3 | 05-01 | 1 | D5-03 | T-05-05 | correct curvature call site | integration, smoke | `.venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --mode field --smoke --smoke-n 64` | ❌ W0 | ⬜ pending |
| 05-01-T3 | 05-01 | 1 | D5-10 | T-05-03 | `--mode bucketed` raises pre-freeze | integration, negative | `.venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --mode bucketed` exits non-zero | ❌ W0 | ⬜ pending |
| 05-02-T1 | 05-02 | 2 | D5-03 | T-05-04, T-05-06 | per-seed cfg manifest | artifact assertion | inline `.venv/bin/python` check: three `05_curvature_field_seed*.npz` with `H_norm` (10000,) and `H_vec` (10000, 768), all finite | ❌ W0 | ⬜ pending |
| 05-02-T2 | 05-02 | 2 | D5-05 | T-05-07 | direction axis beside every rank statistic | artifact assertion | inline check on `05_inter_seed_diagnostics.json`: 3 pairwise Spearman, 3 pairwise direction, `r_over_R` null with reason | ❌ W0 | ⬜ pending |
| 05-03-T2 | 05-03 | 3 | D5-04 / D5-07 | T-05-08, T-05-09, T-05-10 | pooled field is not a raw mean; edges cut on all 10000 | artifact assertion | inline check on `05_curvature_field.npz`: tertile labels, ascending edges, full-field counts within 1, not close to `np.mean` | ❌ W0 | ⬜ pending |
| 05-03-T3 | 05-03 | 3 | D5-13 | — | disclosure, non-gating | artifact assertion | inline check on `05_density_diagnostics.json`: `spearman_density_pooled_h` with n=10000, `k_density`=30, `field_d`=20, both Phase 4 references | ❌ W0 | ⬜ pending |
| 05-04-T2 | 05-04 | 4 | D5-06 / D5-09 | T-05-11, T-05-13, T-05-14 | constants equal artifact; rule carries its caveats | integration assertion | inline check: `assert_preregistered()` passes, edges and pooling method equal the artifact, `VERDICT_RULE` carries all five required literals | ❌ W0 | ⬜ pending |
| 05-04-T3 | 05-04 | 4 | D5-09 / D5-11 / D5-12 | T-05-11 | record duplicates every constant | document assertion | inline check on `05-PREREGISTRATION.md`: all 28 constants named, rule quoted verbatim, gap literals present | ❌ W0 | ⬜ pending |
| 05-05-T1 | 05-05 | 5 | D5-02 / D5-10 | T-05-15, T-05-18 | one global fit; smoke writes nothing | integration, smoke | `.venv/bin/python notebooks/diagnostics/curvature_probe_decodability_run.py --mode bucketed --smoke` | ❌ W0 | ⬜ pending |
| 05-05-T2 | 05-05 | 5 | D5-01 / D5-07 / D5-08 | T-05-16, T-05-17 | realized-count size match; edges equal frozen | artifact assertion | inline check on `05_curvature_probe_decodability.jsonl`: N_BUCKETS bucket rows + 1 overall, verdict terminal, `size_match_n` at most smallest realized bucket_n | ❌ W0 | ⬜ pending |
| 05-05-T3 | 05-05 | 5 | RESEARCH A2 | — | N/A | artifact assertion | inline check: one `probe_conditioning` row with 40 singular values, condition number, three effective ranks, `alpha_at_grid_boundary` | ❌ W0 | ⬜ pending |
| 05-06-T1 | 05-06 | 6 | D5-07 / D5-08 | T-05-22 | notebook reads artifacts, recomputes nothing | notebook assertion | inline check: all code cells executed with outputs, required literals present, no recompute calls | ❌ W0 | ⬜ pending |
| 05-06-T2 | 05-06 | 6 | D5-09 / D5-11 / D5-12 | T-05-19, T-05-20, T-05-21 | ancestry proved, verdict quoted verbatim | document + git assertion | inline check: FINDINGS carries all gap literals and D5-01..D5-13, VERIFICATION carries the git commands, `git diff --quiet -- notebooks/pu_manifold/linear_probe.py` exits 0 | ❌ W0 | ⬜ pending |
| 05-06-T3 | 05-06 | 6 | D5-09 / D5-11 / D5-12 | T-05-21 | prose adequacy | manual (blocking human checkpoint) | see § Manual-Only Verifications | n/a | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Behavior detail per requirement**

| Req | Behavior under test |
|-----|---------------------|
| D5-01 | `W` predicts `legacysurvey` from `hsc`, row-aligned; shapes and row alignment asserted |
| D5-04 / A1 | Seed-pooling normalization behaves as documented — synthetic 3-seed fixture with one seed's field scaled 50×; assert pooled rank-order is **not** identical to that seed's rank-order alone |
| D5-06 | `CURVATURE_CONVENTION == "trace"` agreement assertion, mirroring `decoder_curvature.py`'s own import-time cross-check pattern |
| D5-07 / D5-09 | Bucket edges computed on the pooled field over all 10,000 points, applied correctly to the test-split subset |
| D5-08 | Size-matched subsampling check runs against **realized test-split** counts, not full-field counts (RESEARCH.md Pitfall 4) |
| D5-10 | `assert_preregistered()` raises rather than computes when constants or the frozen `‖H‖` artifact are absent |
| Self-check | Synthetic dataset with a planted exact linear relationship (`y = A @ x + b + tiny_noise`) and planted curvature ordering (fabricated `‖H‖` correlated with residual by construction) recovers the expected verdict |
| D5-05 | Pairwise inter-seed `spearman` diagnostics computed and reported; verdict does not branch on them |
| D5-13 | `spearman(density, ‖H‖)` re-measured on the decoder-side pooled field |

---

## Wave 0 Requirements

- [ ] `notebooks/pu_manifold/tests/test_linear_probe.py` — stubs covering D5-01, D5-04/A1, D5-06,
      D5-07/D5-09, D5-08, D5-10
- [ ] `--selfcheck` mode in `notebooks/diagnostics/curvature_probe_decodability_run.py` — the
      known-answer self-check with a planted linear relationship and planted curvature ordering;
      must assert the pre-registered verdict rule returns the expected outcome on this synthetic,
      analytically-known case **before any PU number is trusted**
- [ ] Both Wave 0 items are delivered by plan `05-01` (Task 1 the `--selfcheck` path, Task 2 the test file)
- [ ] Framework install: **not required** — `pytest` is already the project's test runner

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Pre-registration git-ancestry proof | D5-09 | Ordering between commits is a repository-history property, not a runtime one; no test process can assert its own commit's ancestry | `git merge-base --is-ancestor <freeze-commit> HEAD` exits 0, **and** `git diff <freeze-commit> HEAD -- notebooks/pu_manifold/linear_probe.py` is empty. Record both in `05-VERIFICATION.md`, per `04-VERIFICATION.md`'s template |
| Accepted-gaps narrative present and in the phase's own words | D5-11, D5-12 | Prose adequacy is a human judgement | Confirm `05-FINDINGS.md` states, without deferring to a cross-reference, that the `‖H‖` field has no demonstrated relationship to true curvature (sealed `d=20` row `rank_spearman_rho = -0.0151`) and that the CAE failed its own validity gate |
| Verdict rule admits "no detectable relationship" as a complete outcome | D5-09 | Requires reading the frozen `VERDICT_RULE` text for what it permits, not executing it | Read `VERDICT_RULE` in `linear_probe.py`; confirm a null result is a named terminal outcome, not a near-miss of the positive branch |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
