---
phase: 7
slug: curvature-conditioned-crossmodal-alignment
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false  # test_crossmodal_curvature.py created in 07-01 T3 (wave 1)
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

*Populated by `/gsd-plan-phase 7` on 2026-08-25. The whole phase is serial — waves 1..5, one plan
per wave — because D7-06's freeze must precede every number, the `d`-sweep must not run concurrently
(§7's measured ~10x contention slowdown), and every plan touches `crossmodal_curvature.py`.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01 T2 | 07-01 | 1 | D7-06 | T-07-02 | Guard refuses to proceed while any constant is unset | unit | `.venv/bin/python -c "...cc.assert_preregistered()"` | ❌ W0 | ⬜ pending |
| 07-01 T3 | 07-01 | 1 | D7-06 | T-07-02 | Raises naming each absent/None/empty/malformed constant; strict-ancestor freeze proof | unit | `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py -q` | ❌ W0 | ⬜ pending |
| 07-02 T1 | 07-02 | 2 | D7-01, D7-04, D7-06 | T-07-01, T-07-04 | Cache containment on `--record-path`; thread caps before torch import; smoke writes nothing | smoke | `07_crossmodal_curvature_run.py --selfcheck && --mode smoke --smoke-rows 800` | ❌ W0 | ⬜ pending |
| 07-02 T2 | 07-02 | 2 | D7-04 | — | N/A | unit | `pytest ...::test_per_point_mknn_mean_matches_mknn_score -x` | ❌ W0 | ⬜ pending |
| 07-02 T2 | 07-02 | 2 | D7-04 | — | N/A | unit | `pytest ...::test_per_point_mknn_known_answers -x` | ❌ W0 | ⬜ pending |
| 07-02 T2 | 07-02 | 2 | D7-04 | — | Degenerate input raises rather than returning a partial array | unit | `pytest ...::test_per_point_mknn_guards -x` | ❌ W0 | ⬜ pending |
| 07-03 T1 | 07-03 | 3 | D7-02 | T-07-05 | Control is deterministic; constant/non-finite/short input raises | unit | `pytest ...-k positive_control` | ❌ W0 | ⬜ pending |
| 07-03 T2 | 07-03 | 3 | D7-03 | — | Diagnostics structurally cannot reach `apply_verdict` | unit | `pytest ...-k density` | ❌ W0 | ⬜ pending |
| 07-03 T2 | 07-03 | 3 | D7-03 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cross_split_curvature.py -q` | ✅ pre-existing | ⬜ pending |
| 07-03 T3 | 07-03 | 3 | D7-02, D7-05 | T-07-01 | `--field-npz` resolved through cache containment; refuses to invent a missing field | smoke | `07_crossmodal_curvature_run.py --selfcheck` | ❌ W0 | ⬜ pending |
| 07-04 T1 | 07-04 | 4 | D7-01, D7-06 | T-07-02, T-07-04 | Strict-ancestor freeze proof rejects the self-ancestor case; no concurrency primitives present | smoke | `--mode dsweep --smoke-rows 600 --max-epochs 2` at a scratch record path | ❌ W0 | ⬜ pending |
| 07-04 T3 | 07-04 | 4 | D7-01..D7-04, D7-06 | T-07-03, T-07-05 | Every record row stamps preregistration_commit and run_commit | manual-only | see Manual-Only Verifications | N/A | ⬜ pending |
| 07-05 T1 | 07-05 | 5 | D7-01 | T-07-06 | Notebook reads the record and recomputes nothing | unit | `python -c "...every code cell carries outputs"` | ❌ W0 | ⬜ pending |
| 07-05 T2 | 07-05 | 5 | D7-02, D7-03, D7-07 | T-07-07 | Fidelity quoted as a range, non-claims present, SHAs match the record | unit | `python -c "...findings ok"` | ❌ W0 | ⬜ pending |

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
| Headline `spearman(‖H‖_i, MKNN_i)` at `d ∈ {20,25,32}` on the real 10,000-point PU cloud | D7-01 | ~2h wall-clock; must run serially with `OMP_NUM_THREADS` capped (`07-CONTEXT.md` §7). Too slow for per-commit sampling. | ONE invocation of `07_crossmodal_curvature_run.py --mode dsweep --freeze-commit <sha> --threads 8`, which loops over `D_SWEEP` in-process. Never three invocations and never parallel wave tasks. Resumable with `--resume`. Report all three `d` values. |
| Freeze-before-compute git ancestry proof | D7-06 | Requires inspecting real commit history against the run's recorded commit SHA | BOTH `git merge-base --is-ancestor <freeze-commit> HEAD` exiting 0 AND `git rev-list --count <freeze-commit>..HEAD` being at least 1. The second is not redundant: a commit is its own ancestor, so `--is-ancestor` alone accepts a number produced in the freeze commit itself. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

**Sampling continuity check (planning-time):** no three consecutive tasks lack an `<automated>`
verify. The only manual-only item is 07-04 Task 3, the ~2h deliverable run, which is bracketed by
07-04 Task 1's reduced-scale automated invocation before it and 07-05 Task 1's automated
notebook-execution check after it.
