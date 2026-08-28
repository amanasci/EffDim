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
| 08-01-T2 | 08-01 | 1 | D8-02 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_unbiased_hsic_matches_reference -x` | ✅ | ✅ green |
| 08-01-T2 | 08-01 | 1 | D8-02 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_unbiased_hsic_raises_below_n4 -x` | ✅ | ✅ green |
| 08-01-T2 | 08-01 | 1 | D8-01 / D8-16 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_linear_cka_invariances -x` | ✅ | ✅ green |
| 08-01-T2 | 08-01 | 1 | D8-01 / D8-16 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_rbf_cka_invariances -x` | ✅ | ✅ green |
| 08-01-T2 | 08-01 | 1 | D8-16 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_noise_ladder_monotone -x` | ✅ | ✅ green |
| 08-04-T3 | 08-04 | 1 | D8-03 / D8-04 | — | N/A | unit + manual | `pytest notebooks/pu_manifold/tests/test_cka.py::test_assert_preregistered_passes -x` (re-keyed: no dedicated `test_sigma_is_frozen_global_constant` was written; SIGMA_HSC/SIGMA_LEGACYSURVEY's full-precision equality to `08-03-SUMMARY.md`'s measured values is a direct literal-copy fact, cross-checked manually in `08-04-DECISION.md` and `08-04-SUMMARY.md`, not re-asserted by a separate unit test) | ✅ | ✅ green |
| 08-02-T1 | 08-02 | 1 | D8-06 / D8-08 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_tertile_within_stratum_split -x` | ✅ | ✅ green |
| 08-02-T2 | 08-02 | 1 | D8-11 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_stratified_tertile_null_preserves_sizes -x` | ✅ | ✅ green |
| 08-02-T3 | 08-02 | 1 | D8-15 | — | N/A | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_seed_pooling_raises -x` | ✅ | ✅ green |
| 08-04-T3 | 08-04 | 1 | D8-22 | T-08-18 / T-08-19 | Every pre-registered constant is filled and guarded; a drifted or reverted constant raises | unit | `pytest notebooks/pu_manifold/tests/test_cka.py::test_assert_preregistered_rejects_unset_constant -x` (re-keyed: parametrized over all 45 `_REQUIRED_CONSTANTS`, replacing the single hypothetical `test_assert_preregistered_rejects_drift` name) | ✅ | ✅ green (45 cases) |
| 08-03-T1 | 08-03 | 1 | D8-23 | T-08-18 | Importing the Phase 8 module mutates no sealed module's globals under any import order | unit | `pytest notebooks/pu_manifold/tests/test_cka_import_purity.py::test_import_cka_does_not_mutate_sealed_modules -x` (re-keyed per plan instruction) | ✅ | ✅ green |
| 08-05 | 08-05 | 2 | D8-18 | — | N/A | integration | `.venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode positive-control --freeze-commit 816863cae2209261470d1d041dcc4484a3056947` | ❌ (logic lands 08-05; pre-flight gates — `assert_preregistered()` + strict-ancestor — pass as of 08-04) | ⬜ pending |
| 08-05 | 08-05 | 2 | D8-19 | — | N/A | integration | `.venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode negative-control --freeze-commit 816863cae2209261470d1d041dcc4484a3056947` | ❌ (logic lands 08-05; pre-flight gates pass as of 08-04) | ⬜ pending |
| 08-05 | 08-05 | 2 | D8-09 / D8-13 | — | N/A | integration | `.venv/bin/python notebooks/diagnostics/08_cka_alignment_run.py --mode sweep --freeze-commit 816863cae2209261470d1d041dcc4484a3056947` | ❌ (logic lands 08-05; pre-flight gates pass as of 08-04) | ⬜ pending |

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
| Freeze commit precedes every measured value | D8-22 | Git ancestry is a property of the repository's history, not of any process a test can run against itself. The precedent is `02.2`'s sealed FAIL and `06-PREREGISTRATION-AMENDMENT-01`. | Freeze SHA (08-04, `notebooks/pu_manifold/cka.py`, exactly one file): `816863cae2209261470d1d041dcc4484a3056947`. `git merge-base --is-ancestor 816863cae2209261470d1d041dcc4484a3056947 HEAD` must exit 0, and every run's JSONL must record this SHA as `preregistration_commit`; confirm no constant in `_REQUIRED_CONSTANTS` changed after that commit: `git log -p 816863cae2209261470d1d041dcc4484a3056947..HEAD -- notebooks/pu_manifold/cka.py` must show no change to any of the 45 names. |
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
