---
phase: 4
slug: region-partitioning-regional-alignment-mknn
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-23
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.1.1 (pinned, `requirements-notebooks.txt`) |
| **Config file** | none — tests discovered by pytest's default `test_*.py` convention in `notebooks/pu_manifold/tests/` |
| **Quick run command** | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_curvature_probe.py notebooks/pu_manifold/tests/test_varying_ii_controls.py notebooks/pu_manifold/tests/test_cross_split_curvature.py -q` |
| **Full suite command** | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` |
| **Estimated runtime** | ~60 seconds (quick), full suite bounded by the sealed estimator fixtures |

---

## Sampling Rate

- **After every task commit:** Run the quick run command above (existing estimator/fixture tests only — no `test_mknn.py` exists, per D4-18)
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite green **plus** MKNN-02's global reproduction landing inside the verified 0.34%–2.25% range — that reproduction IS this phase's end-to-end check (D4-18)
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

*Task IDs are assigned by the planner. This map is seeded from the researcher's requirement→test map and must be completed against the final PLAN.md task list.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | REGN-01, REGN-02 | — | N/A | notebook (printed read-out) | N/A — density/curvature correlation is a reported number, not an assertion | N/A | ⬜ pending |
| TBD | TBD | TBD | REGN-03, REGN-04, REGN-06 | — | N/A | unit (round-trip) | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q -k "region_partition"` | ❌ W0 (new, optional — see Wave 0) | ⬜ pending |
| TBD | TBD | TBD | REGN-05 | — | N/A | notebook (printed) | N/A — region point counts printed | N/A | ⬜ pending |
| TBD | TBD | TBD | MKNN-01 | — | N/A | inline notebook sanity (identical embeddings → score 1.0) | manual notebook assertion cell — deliberately not a pytest file, per D4-18 | ❌ by decision (D4-18) | ⬜ pending |
| TBD | TBD | TBD | MKNN-02 | — | N/A | integration (notebook) | manual read-out against **0.34%–2.25%** (arXiv:2509.19453 Table 2, Legacy-vs-HSC column) | ✅ the runner IS the test (D4-18) | ⬜ pending |
| TBD | TBD | TBD | MKNN-03..07 | — | N/A | notebook | manual read-out — regional scores, region-scoped nulls, bootstrap CIs, pre-registered verdict | ✅ no gap (D4-18) | ⬜ pending |
| TBD | TBD | TBD | MKNN-08 | — | N/A | notebook | manual read-out — k-occurrence skewness from the same membership matrix | ✅ no gap | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

**None required for `mknn.py`'s new code**, per D4-18's locked decision not to add `tests/test_mknn.py`. Existing test files already cover every sealed function this phase reuses:

- [x] `notebooks/pu_manifold/tests/test_curvature_probe.py` — covers `centroid_mean_curvature`, `local_density_weights`
- [x] `notebooks/pu_manifold/tests/test_varying_ii_controls.py` — covers `make_ridge_graph_control` / `make_multinormal_ridge_control` (present and passing; **not exercised this phase**, per D4-10)
- [x] `notebooks/pu_manifold/tests/test_cross_split_curvature.py` — covers the `R_H` machinery D4-06/D4-07 extend
- [ ] *(optional, planner's call)* a round-trip test for the new region-partition / artifact-freeze helper — the only new test file this phase could add without contradicting D4-18, which is specific to `mknn.py`'s statistical functions, not to the partition-freezing helper. REGN-06's own frozen artifact is already the audit trail.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Density/curvature correlation reported before any region split is trusted | REGN-01, REGN-02 | The deliverable is a reported statistic and its interpretation, not a pass/fail threshold — pre-committing a threshold here would itself be a forking path | Read the printed correlation cell; confirm it appears **above** the partition cell in notebook order |
| Partition frozen before the first regional MKNN number exists | REGN-03, REGN-04, Ordering constraint | Cell-execution ordering is the artifact; no automated assertion can prove the human did not peek | Confirm the pre-registration cell (verdict rule + partition freeze) precedes every MKNN cell in notebook order, and that its output is committed |
| Global MKNN reproduction against the published range | MKNN-02 | Comparison against an external paper's Table 2 is a judgement about agreement, not an equality check | Read the printed global MKNN; compare against 0.34%–2.25% |
| Regional verdict, including "no detectable difference" | MKNN-06, MKNN-07 | The verdict rule is pre-registered prose applied to CIs; a valid outcome is a null result | Apply the pre-registered rule as written; do not amend it after seeing the numbers |
| Hubness caveat stated alongside results | MKNN-08 | A caveat is prose; the substantiating k-occurrence skewness is a printed number | Confirm both the statistic and the caveat text appear with the results |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or a recorded manual-only justification above
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Note on Nyquist compliance for this phase:** D4-18 deliberately declines pytest coverage for `mknn.py`'s statistical functions, and REGN-01/02/05 are reported statistics rather than assertions. A large share of this phase's verification is therefore manual-by-decision, recorded above rather than treated as a gap to close. `/gsd-validate-phase` should read the Manual-Only table as the justification, not flag it as missing coverage.

**Approval:** pending
