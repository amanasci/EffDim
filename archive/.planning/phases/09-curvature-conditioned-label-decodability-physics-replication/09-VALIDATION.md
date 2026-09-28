---
phase: 9
slug: curvature-conditioned-label-decodability-physics-replication
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-02
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest, existing suite at `notebooks/pu_manifold/tests/` (761+ tests per STATE.md, 2026-08-28) |
| **Config file** | none dedicated — invoked directly as `pytest tests/` from `notebooks/pu_manifold/` |
| **Quick run command** | `pytest tests/test_cross_split_curvature.py tests/test_linear_probe.py tests/test_density_stratified_null.py -q` (from `notebooks/pu_manifold/`) |
| **Full suite command** | `.venv/bin/python3 -m pytest notebooks/pu_manifold/tests/ -q` |
| **Estimated runtime** | quick: ~tens of seconds; full: several minutes (not exhaustively timed in 09-RESEARCH.md) |

---

## Sampling Rate

- **After every task commit:** Run the targeted new-module test file (`pytest tests/test_<new_module>.py -x`)
- **After every plan wave:** Run the three reused-module tests plus all new Phase 9 test files
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~60 seconds for the per-task command

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | D9-xx | T-9-01 / — | {filled by plan-phase / validate-phase from PLAN.md tasks} | unit | `{command}` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Seeded from `09-RESEARCH.md` § Validation Architecture (Phase Requirements → Test Map): D9-05..08 and D9-13/D9-16 → `tests/test_physics_labels.py` (new); D9-09 → `tests/test_cross_split_curvature.py` (exists); D9-11, D9-14, D9-17, D9-18 → `tests/test_physics_curvature_probe.py` (new). Rows are filled per task once PLAN.md files exist.*

---

## Wave 0 Requirements

- [ ] `notebooks/pu_manifold/tests/test_physics_labels.py` — stubs for D9-01, D9-05..08, D9-13, D9-16
- [ ] `notebooks/pu_manifold/tests/test_physics_curvature_probe.py` — stubs for D9-09..12, D9-14, D9-17, D9-18
- [ ] No framework install needed — pytest is already the project's test runner

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full-scale Physics run (four AE fits at 86,471 rows, real verdict) | D9-10, D9-12, D9-17 | Runs on a remote/colleague host, ~hours; not a unit-test target | Run the frozen runner in full mode on the execution host; return `notebooks/.cache/09_*.jsonl` and the reporting notebook outputs |
| Row-alignment proof at full scale | D9-05..08 | Needs the real downloaded data on the execution host | Run the alignment gate first; no Physics number without a PASS record |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
