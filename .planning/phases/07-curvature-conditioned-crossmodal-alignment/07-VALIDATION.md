---
phase: 7
slug: curvature-conditioned-crossmodal-alignment
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-25
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Seeded by `/gsd-plan-phase 7` from `07-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (project convention; `notebooks/pu_manifold/tests/` holds one test file per module) |
| **Config file** | Root `pyproject.toml` `[tool.pytest.ini_options]` sets `testpaths = ["tests"]` for `src/effdim/` only; `notebooks/pu_manifold/tests/` is invoked by explicit path |
| **Quick run command** | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py -q` |
| **Full suite command** | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` |
| **Estimated runtime** | ~5s quick, ~60s full suite |

**Hard separation.** The automated suite runs on small synthetic fixtures ONLY. It never loads the
real 10,000-point PU cloud and never trains a real decoder. The phase's scientific computation
(D7-01's `d ∈ {20,25,32}` sweep, ~2h wall-clock per `07-CONTEXT.md` §7) is a deliverable run, not
a test, and is excluded from every sampling gate below.

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py -q`
- **After every plan wave:** Run `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

*Populated after planning. Task IDs are assigned by `gsd-planner`; the requirement→test mapping
below is inherited from `07-RESEARCH.md` and is authoritative for which behaviors need coverage.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | D7-04 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py::test_per_point_mknn_mean_matches_mknn_score -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | D7-04 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py::test_per_point_mknn_known_answers -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | D7-06 | — | Raises on missing/malformed constant | unit | `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py::test_assert_preregistered_raises_on_unset_constants -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | D7-02 | — | N/A | unit/smoke | `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py::test_positive_control_recovers_planted_effect -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | TBD | D7-03 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cross_split_curvature.py -q` | ✅ pre-existing | ⬜ pending |
| TBD | TBD | TBD | D7-05 | T-7-01 | Runner writes only inside the cache root | unit | `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `notebooks/pu_manifold/tests/test_crossmodal_curvature.py` — new file, covers D7-02, D7-04, D7-06
- [ ] No new shared fixtures — existing `tests/` files already establish the small-synthetic-fixture pattern
- [ ] No framework install — pytest is already a dev dependency in `pyproject.toml`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Headline `spearman(‖H‖_i, MKNN_i)` at `d ∈ {20,25,32}` on the real 10,000-point PU cloud | D7-01 | ~2h wall-clock; must run serially with `OMP_NUM_THREADS` capped (`07-CONTEXT.md` §7). Too slow for per-commit sampling. | Run the phase runner in `--mode dsweep`, serially, one `d` at a time; record results to the cache; report all three `d` values. |
| Freeze-before-compute git ancestry proof | D7-06 | Requires inspecting real commit history against the run's recorded commit SHA | `git merge-base --is-ancestor <freeze-commit> HEAD` and confirm the freeze commit predates the first PU number. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
