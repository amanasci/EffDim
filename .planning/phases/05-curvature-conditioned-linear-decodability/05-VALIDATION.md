---
phase: 5
slug: curvature-conditioned-linear-decodability
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| TBD | TBD | 0 | D5-01 | — | N/A | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_fit_probe_shape_and_row_alignment -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | D5-04 / A1 | — | N/A | unit, known-answer | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_pool_seeds_no_single_seed_dominates -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | D5-06 | — | N/A | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_curvature_convention_matches_sealed_modules -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | D5-07 / D5-09 | — | N/A | unit, known-answer | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_bucket_assignment_known_answer -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | D5-08 | — | N/A | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_size_matched_check_uses_test_split_counts -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | D5-10 | — | N/A | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py::test_assert_preregistered_raises_when_absent -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | Known-answer self-check | — | N/A | integration | `python notebooks/diagnostics/curvature_probe_decodability_run.py --selfcheck` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1+ | D5-05 | — | N/A | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py -q -k inter_seed` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1+ | D5-13 | — | N/A | unit | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py -q -k density` | ❌ W0 | ⬜ pending |

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
