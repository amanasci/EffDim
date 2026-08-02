---
phase: 01-data-loading-manifold-reconstruction
plan: 04
subsystem: data-loading
tags: [isomap, scikit-learn, joblib, manifold-freeze, notebooks]
requires:
  - {phase: "01-data-loading-manifold-reconstruction (plan 03)", provides: "Confirmed k*=15 (Task 4 checkpoint, accept-candidate), SHORT_CIRCUIT_RISK=False, STABILITY_TABLE, PLATEAU_RUNS=[{k_values:[10,15,30],length:3}]"}
provides:
  - "01_manifold_and_gate.ipynb §5.1-5.2 -- K_STAR=15 frozen by the pre-registered plateau rule, cross-checked against the human-confirmed value, ANALYSIS_CFG['n_neighbors']=15 set, fit_key fully determined"
  - "notebooks/.cache/isomap_43cf438bc944c509.joblib -- the single canonical full Isomap fit at k*=15, carrying dist_matrix_/embedding_/nbrs_/kernel_pca_ (~1.55 GiB), the only artifact Phase 2's hand-rolled full spectrum can read"
  - "notebooks/.cache/phase1_handoff_43cf438bc944c509.json -- machine-readable Phase 1 to Phase 2 interface (14 top-level keys)"
  - "01_manifold_and_gate.ipynb §6 -- reserved terminal placeholder naming Phase 2's scope (SPEC-01..07)"
  - "Phase 1 sealed: all ten requirements (DATA-01..05, ISO-01..05) complete"
affects: [02]
tech-stack:
  added: []
  patterns:
    - "Machine-readable phase-boundary handoff: a small JSON carrying cache keys, artifact filenames, frozen values with full selection provenance (incl. thresholds a verdict was judged against), explicit flags -- built from already-bound state, never recomputed, rendered a second time as a printed table in committed output"
    - "Three-surface flag propagation (D-11): a qualifying condition a downstream gate must weigh is never left in one channel -- warnings.warn+boxed print, a JSON field, and a prose markdown cell that survives in committed output"
key-files:
  created: []
  modified: [notebooks/01_manifold_and_gate.ipynb]
key-decisions:
  - "Task 3 gate (blocking): approved. Coordinator independently re-verified the joblib artifact by loading it directly (n_neighbors=15, n_components=18, dist_matrix_ (10000,10000), embedding_ (10000,18)), the handoff JSON's 14 keys/flags/notes, the clean working tree, and the §2 norm histograms (band ~13.5-17.0, near-identical between modalities), before sealing Phase 1"
  - "fit_key == sweep_k15's key (43cf438bc944c509) is CORRECT cache-contract behaviour, not a collision: both the §4.2 stage-2 fit at k=15 and this plan's §5.2 canonical fit hash identically once ANALYSIS_CFG['n_neighbors']=15. Distinct files under distinct stems: sweep_k15_43cf438bc944c509.npz (~1.07 MB, no dist_matrix_) vs isomap_43cf438bc944c509.joblib (~1.55 GiB, carries dist_matrix_)"
  - "n_components_no_headroom=True is unconditional per D-12 (no headroom by construction). LIVE condition Phase 2 must treat as a real branch: if SPEC-04's elbow exceeds 18, a re-fit at a larger n_components is required. Because n_components is a fit_key field, the re-fit produces a new fit_key automatically -- no manual cache invalidation, no stale-pickle risk"
requirements-completed: [ISO-04, ISO-05]
coverage:
  - {id: D1, description: "§5.1: applies the §4.0 plateau rule to PLATEAU_RUNS (no re-derivation), selects the widest run [10,15,30], binds K_STAR=15, cross-checks against 01-03-SUMMARY.md and halts on divergence, re-evaluates SHORT_CIRCUIT_RISK=False against K_STAR, freezes ANALYSIS_CFG['n_neighbors']=15 / fit_key=43cf438bc944c509", requirement: "ISO-04", verification: [{kind: integration, ref: "§5.1 (committed 1535010)", status: pass}], human_judgment: false}
  - {id: D2, description: "§5.2: single canonical Isomap(n_neighbors=15, n_components=18, eigen_solver='dense') fit, cached as isomap_43cf438bc944c509.joblib (~1.55 GiB) carrying dist_matrix_(10000,10000)/embedding_(10000,18)/nbrs_/kernel_pca_; ISO-05 demonstrated: CACHE HIT on second call, a n_neighbors+1 mutation produces a different config_key; real fit_seconds=66.86", requirement: "ISO-05", verification: [{kind: integration, ref: "§5.2 (committed 1535010); coordinator independently re-loaded the joblib and confirmed shapes/params", status: pass}], human_judgment: false}
  - {id: D3, description: "§5.3-5.4: phase1_handoff_43cf438bc944c509.json written via json_cache -- all 14 top-level keys, k_star_selection incl. thresholds, flags (exactly 3), 5 notes_for_phase2 entries; above-30 flag surfaced in all 3 D-11 places; §5.4 close-out; terminal §6 placeholder names Phase 2's scope, no Phase 2 content", requirement: "ISO-05", verification: [{kind: integration, ref: "§5.3-5.4, §6 (committed f84f050); coordinator re-parsed the JSON and confirmed 14 keys, k_star=15, n_components=18, d_provisional=18, subsample_key=a79b3460b838fd0a", status: pass}], human_judgment: false}
  - {id: D4, description: "Task 3 checkpoint:human-verify (blocking) -- the complete Phase 1 deliverable presented for end-to-end sign-off: clean reproduction, cache-hit behaviour, pre-registration ordering, §4 tables, frozen k* match, handoff validity, flag surfacing, phase-boundary cleanliness, core untouched, output hygiene", verification: [{kind: manual_procedural, ref: "coordinator's independent re-verification: direct joblib load, direct JSON re-parse, working-tree cleanliness, fit_key/sweep-key coincidence analysis, visual inspection of the §2 histograms", status: pass}], human_judgment: true, rationale: "Sealing an entire phase -- and the irreversible k*-freeze Phases 2-4 all load from -- is an explicitly human-confirmed decision, not test-certifiable."}
duration: ~30min active work across two sessions
completed: 2026-07-31
status: complete
---

# Phase 1 Plan 4: Freeze k*, Cache the Fit, and the Phase 1-to-Phase 2 Handoff Summary

**`k* = 15` frozen, cached as `isomap_43cf438bc944c509.joblib` (~1.55 GiB, `dist_matrix_`); `phase1_handoff_43cf438bc944c509.json` written -- Phase 1 sealed, all ten requirements complete.**

## Frozen Values

| Field | Value |
|---|---|
| `k_star` | **15** |
| `n_components` | **18** |
| `d_provisional` | **18** |
| `fit_key` | **43cf438bc944c509** |
| `subsample_key` | **a79b3460b838fd0a** |
| `flags.short_circuit_risk` | **False** |
| `flags.k_auto_extended` | **False** |
| `flags.n_components_no_headroom` | **True** -- LIVE D-12 condition, see below |
| `fit_seconds` (measured, k*=15, n=10000, d=18) | **66.86** |
| `isomap_43cf438bc944c509.joblib` on-disk size | 1,664,401,892 bytes (~1.55 GiB) |
| `phase1_handoff_43cf438bc944c509.json` on-disk size | 3,560 bytes |
| `notebooks/.cache/` total size | ~1.7 GiB |

**`n_components_no_headroom: True`, deliberate prominence:** exact `ceil(median(...))` over the eight geometric keys (D-12), nothing added — a SPEC-04 elbow beyond 18 forces a re-fit (new `fit_key` automatically, since `n_components` is itself a key field). **`fit_key` coincidence with plan 03's sweep, not a bug:** matches `sweep_k15_43cf438bc944c509.npz`'s key because once `ANALYSIS_CFG["n_neighbors"]=15`, both cfg dicts became field-for-field identical — not the same file (npz ~1.07 MB no `dist_matrix_`, vs joblib ~1.55 GiB with it).

## Performance / Accomplishments

~30 min active work across two sessions (task commits `1535010`/`f84f050`, then a Task 3 blocking
checkpoint pause). Completed 2026-07-31. 3/3 tasks. 1 file modified, grown 74->90 cells. See the
Frozen Values table and `coverage` (frontmatter) for what each task produced. ISO-05 demonstrated
concretely: CACHE HIT 0.58-0.60s across three re-runs; a `n_neighbors+1` mutation produced
`config_key=53a54bf5917e48d0` without fitting. Cache-behaviour re-run completed in 36s vs. the
original ~103s cold run. Coordinator independently re-verified the joblib, handoff JSON,
working-tree cleanliness, fit_key coincidence, and §2 histograms. **Approved: Phase 1 sealed.**

## Commits / Files / Deviations / Known Limitations

1. §5.1-5.2 — `1535010`. 2. §5.3-5.4+§6 — `f84f050`. 3. Task 3 checkpoint — no commit. 32 new code
cells, zero error cells across three re-runs. No deviations — plan executed exactly as written, all
`<verify>` checks passed first execution. No issues: a third Restart-and-Run-All produced only a
timestamp diff, reverted via a targeted `git checkout --` (sanctioned exception). No external setup
needed — CPU-bound fit, no network. **Known limitation** (carried from `01-03-SUMMARY.md`,
`.planning/WINDOWS.md` id 1, not newly introduced): `STAGE2_K` gaps 5/5/15, k=8/k=20 never fit, so
the plateau run producing `K_STAR=15` is maximal in *index* space not *k* space — disclosed and
accepted at plan 03's gate, carried into Phase 2's gate-weighing narrative as context.

## Next Phase Readiness
Phase 1 sealed — all ten requirements complete. Phase 2 reads `phase1_handoff_43cf438bc944c509.json`
directly as its input contract. `§6` is Phase 2's append point; `§0`-`§5` never renumbered.
`isomap_43cf438bc944c509.joblib` is the one artifact carrying `dist_matrix_`;
`kernel_pca_.eigenvalues_` must not be substituted. **Live budgeting item:**
`n_components_no_headroom=True` means a SPEC-04 elbow beyond 18 forces a re-fit (~67s+ real work).
No blockers; core untouched.

---
*Phase: 01-data-loading-manifold-reconstruction* · *Completed: 2026-07-31*
## Self-Check: PASSED

Notebook (90 cells), `isomap_43cf438bc944c509.joblib` (1,664,401,892 bytes), and
`phase1_handoff_43cf438bc944c509.json` verified present on disk. Both task commits (`1535010`,
`f84f050`) present in `git log --oneline --all`. `git diff` clean. No missing items.
</content>
