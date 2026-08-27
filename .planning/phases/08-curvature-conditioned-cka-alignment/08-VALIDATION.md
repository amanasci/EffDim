---
phase: 8
slug: curvature-conditioned-cka-alignment
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-27
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `08-RESEARCH.md` § Validation Architecture. Requirement IDs are
> CONTEXT.md's locked decisions `D8-01..D8-24` (ROADMAP maps no REQ-IDs to this
> phase — the same arrangement Phase 7 used with `07-CONTEXT.md` §3).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already the project runner; `notebooks/pu_manifold/tests/` holds every prior phase's unit tests) |
| **Config file** | Root `pyproject.toml` `[tool.pytest.ini_options]` sets `testpaths = ["tests"]` — that targets the **library's** tests, not `notebooks/pu_manifold/tests/`. Invoke the notebooks path explicitly, as every prior phase does. |
| **Quick run command** | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_cka.py -x` |
| **Full suite command** | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` |
| **Estimated runtime** | quick ~10s (synthetic fixtures only, no PU data); full suite ~2-4 min |

---

## Sampling Rate

- **After every task commit:** `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_cka.py -x`
- **After every plan wave:** `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` — all prior phases' tests must stay green, because Phase 7/07.1's tests exercise the sealed modules Phase 8 imports read-only (D8-23's import-purity contract fails loudly here).
- **Before `/gsd-verify-work`:** full suite green, **plus** the three integration runs (positive control, negative control, sweep) recorded in the frozen JSONL.
- **Max feedback latency:** 15 seconds for the unit tier.

---

## Per-Task Verification Map

Task IDs are assigned by the planner; this table is keyed by locked decision and is
re-keyed to task IDs by `/gsd-validate-phase`.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | 1 | D8-02 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_unbiased_hsic_matches_reference -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | D8-02 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_unbiased_hsic_raises_below_n4 -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | D8-01 / D8-16 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_linear_cka_invariances -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | D8-01 / D8-16 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_rbf_cka_invariances -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | D8-16 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_noise_ladder_monotone -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | D8-03 / D8-04 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_sigma_is_frozen_global_constant -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | D8-06 / D8-08 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_tertile_within_stratum_split -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | D8-11 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_stratified_tertile_null_preserves_sizes -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | D8-15 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_seed_pooling_raises -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | D8-22 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_assert_preregistered_rejects_drift -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | D8-23 | — | Importing the Phase 8 module mutates no sealed module's globals under any import order | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_import_does_not_mutate_sealed_modules -x` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | D8-18 | — | N/A | integration | `.venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode positive-control` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | D8-19 | — | N/A | integration | `.venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode negative-control` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | D8-09 / D8-13 | — | N/A | integration | `.venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode sweep --freeze-commit <sha>` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Script and module paths above are the researcher's recommendation; the planner may
rename them. If it does, this table is re-keyed to the planner's names — the
behaviors and their tiers do not change.*

---

## Wave 0 Requirements

- [ ] `notebooks/pu_manifold/tests/test_cka.py` — covers D8-01, D8-02, D8-16 (estimator + invariance ladder; synthetic fixtures only, no PU data needed)
- [ ] In-file synthetic fixtures for the invariance ladder — orthogonal rotation, isotropic scaling, independent columns, additive-noise ladder. Small enough to need no shared `conftest.py`.
- [ ] Split/null coverage for D8-06 / D8-08 / D8-11 — its own `test_cka_split.py`, or folded into `test_cka.py` if the planner keeps split logic in one module.
- [ ] Import-purity test for D8-23 (must assert across import orders, not just one).
- [ ] Framework install: **none** — pytest, numpy 2.5.1, scipy 1.18.0 all verified present.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Freeze commit precedes every measured value | D8-22 | Git ancestry is a property of the repository's history, not of any process a test can run against itself. The precedent is `02.2`'s sealed FAIL and `06-PREREGISTRATION-AMENDMENT-01`. | `git merge-base --is-ancestor <freeze-sha> HEAD` must exit 0, and every run's JSONL must record `<freeze-sha>`; confirm no constant in the frozen block changed after that commit (`git log -p <freeze-sha>..HEAD -- <constants file>` is empty). |
| `08-FINDINGS.md` states the density-residualized-tertile semantics explicitly, not buried | D8-06 | Judgement about prominence in prose. | Read `08-FINDINGS.md`; the caveat must appear where the tertile definition is introduced, not in a footnote. |
| Verdict sentence cannot be written without `d=32`'s gap and the shuffled-`\|\|H\|\|` false-positive rate in the same sentence | D8-21 | Structural property of prose. | Read the verdict sentence in `08-FINDINGS.md`; both numbers must be in it. |
| The frozen unconditional reporting block prints every required row regardless of outcome, beside the headline and not in an appendix | D8-21 | Layout/prominence judgement. | Check `08-FINDINGS.md` prints: `d=32`'s gap, shuffled-`\|\|H\|\|` false-positive rate, planted-effect detection floor, realized `\|\|H\|\|` contrast per `S`, all three `sigma` rungs. |
| Human ratification of Phase 7 / 07.1 verdicts | (deferred) | Outstanding UAT item inherited from `07.1-FINDINGS.md`; **not** a Phase 8 task. Recorded here so it is not lost. | Developer sign-off on `07-FINDINGS.md` and `07.1-FINDINGS.md`. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s (unit tier)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
