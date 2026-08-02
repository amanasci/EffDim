---
phase: 01-data-loading-manifold-reconstruction
plan: 04
subsystem: data-loading
tags: [isomap, scikit-learn, joblib, manifold-freeze, notebooks]

# Dependency graph
requires:
  - phase: 01-data-loading-manifold-reconstruction (plan 03)
    provides: "Confirmed k*=15 (Task 4 checkpoint, accept-candidate), SHORT_CIRCUIT_RISK=False, STABILITY_TABLE, PLATEAU_RUNS=[{k_values:[10,15,30],length:3}]"
provides:
  - "notebooks/01_manifold_and_gate.ipynb §5.1-5.2 -- K_STAR=15 frozen by the pre-registered plateau rule, cross-checked against the human-confirmed value, ANALYSIS_CFG['n_neighbors']=15 set, fit_key fully determined"
  - "notebooks/.cache/isomap_43cf438bc944c509.joblib -- the single canonical full Isomap fit at k*=15, carrying dist_matrix_/embedding_/nbrs_/kernel_pca_ (~1.55 GiB), the only artifact in this phase Phase 2's hand-rolled full spectrum can read"
  - "notebooks/.cache/phase1_handoff_43cf438bc944c509.json -- the machine-readable Phase 1 to Phase 2 interface (14 top-level keys)"
  - "notebooks/01_manifold_and_gate.ipynb §6 -- reserved terminal placeholder naming Phase 2's scope (SPEC-01..07), the notebook's unambiguous append point"
  - "Phase 1 sealed: all ten requirements (DATA-01..05, ISO-01..05) complete"
affects: [02]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Machine-readable phase-boundary handoff: a small JSON (phase1_handoff_{fit_key}.json) carrying cache keys, artifact filenames, frozen values with full selection provenance (including the thresholds a verdict was judged against), and explicit flags -- built from already-bound notebook state, never recomputed, and rendered a second time as a printed table in committed cell output so the same content survives without executing anything"
    - "Three-surface flag propagation (D-11): a qualifying condition that a downstream phase's gate must weigh is never left in exactly one channel -- a warnings.warn + boxed print at the moment of computation, a machine-readable JSON field, and a prose markdown cell that survives in committed notebook output"

key-files:
  created: []
  modified:
    - notebooks/01_manifold_and_gate.ipynb

key-decisions:
  - "Task 3 gate (checkpoint:human-verify, gate=blocking): approved. Coordinator independently re-verified the isomap_43cf438bc944c509.joblib artifact by loading it directly (n_neighbors=15, n_components=18, dist_matrix_ (10000,10000), embedding_ (10000,18)), the phase1_handoff JSON's 14 keys/flags/notes, the clean working tree, and the §2 norm histograms (rendered correctly, left-skewed, near-identical HSC/LS distributions), before sealing Phase 1."
  - "fit_key == sweep_k15's key (43cf438bc944c509) is CORRECT cache-contract behaviour, not a collision or a bug: both the §4.2 stage-2 sweep fit at k=15 and this plan's §5.2 canonical fit were built from cfg dicts that hash identically once ANALYSIS_CFG['n_neighbors'] was set to 15 (the only field STAGE2_K's per-k fit_cfg and ANALYSIS_CFG differ on before that point). The two artifacts are NOT the same file, however: sweep_k15_43cf438bc944c509.npz (~1.07 MB slim record, no dist_matrix_) and isomap_43cf438bc944c509.joblib (~1.55 GiB, carries dist_matrix_) are distinct files under distinct stems (sweep_k{k}_... vs isomap_...) that happen to share the same 16-hex config_key suffix because their cfg dicts are identical."
  - "n_components_no_headroom=True is unconditional per D-12 (there is no headroom by construction -- N_COMPONENTS is the exact ceil(median(...)) with nothing added). This is a LIVE condition Phase 2 must treat as a real branch, not an edge case: if SPEC-04's residual-variance elbow exceeds 18, a re-fit at a larger n_components is required. Because n_components is itself a fit_key field, that re-fit produces a new fit_key automatically -- no manual cache invalidation, no risk of reusing a stale isomap_43cf438bc944c509.joblib."

requirements-completed: [ISO-04, ISO-05]

coverage:
  - id: D1
    description: "§5.1: applies the §4.0 pre-registered plateau rule to PLATEAU_RUNS (no re-derivation, no new threshold literal), selects the widest run [10,15,30], binds K_STAR=15 at the centre index, cross-checks against the human-confirmed 01-03-SUMMARY.md value and halts on divergence, re-evaluates SHORT_CIRCUIT_RISK=False against K_STAR (strict), and freezes ANALYSIS_CFG['n_neighbors']=15 / fit_key=43cf438bc944c509"
    requirement: "ISO-04"
    verification:
      - kind: integration
        ref: "notebooks/01_manifold_and_gate.ipynb §5.1 (executed cell outputs, committed 1535010); re-run via jupyter nbconvert --to notebook --execute --inplace notebooks/01_manifold_and_gate.ipynb"
        status: pass
    human_judgment: false
  - id: D2
    description: "§5.2: single canonical Isomap(n_neighbors=15, n_components=18, eigen_solver='dense') fit on the real 10,000-row LS array, cached as isomap_43cf438bc944c509.joblib (~1.55 GiB) carrying dist_matrix_ (10000,10000)/embedding_ (10000,18)/nbrs_/kernel_pca_; independently re-verified n_connected_components=1; ISO-05 demonstrated concretely -- a second joblib_cache call returns a bit-identical embedding_ (CACHE HIT) and a single n_neighbors+1 config change produces a different config_key; real fit_seconds=66.86 measured"
    requirement: "ISO-05"
    verification:
      - kind: integration
        ref: "notebooks/01_manifold_and_gate.ipynb §5.2 (executed cell outputs, committed 1535010); coordinator independently re-loaded isomap_43cf438bc944c509.joblib directly and confirmed n_neighbors=15, n_components=18, dist_matrix_ (10000,10000), embedding_ (10000,18)"
        status: pass
    human_judgment: false
  - id: D3
    description: "§5.3-5.4: phase1_handoff_43cf438bc944c509.json written via json_cache -- all 14 top-level keys (phase, fit_key, subsample_key, config, artifacts, k_star, k_star_selection [including thresholds], n_components, d_provisional, d_provisional_source, row_indices_sha256, alignment_check, flags [exactly 3], notes_for_phase2 [5 entries]); same content rendered as a printed table in committed output; the above-30 short-circuit-risk flag surfaced in all three D-11-required places (warn+print in §5.1, JSON flags.short_circuit_risk, prose markdown in §5.3); §5.4 close-out prints all frozen values plus total cache size; terminal §6 placeholder names Phase 2's scope with no Phase 2 content implemented"
    requirement: "ISO-05"
    verification:
      - kind: integration
        ref: "notebooks/01_manifold_and_gate.ipynb §5.3-5.4, §6 (executed cell outputs, committed f84f050); coordinator independently re-parsed phase1_handoff_43cf438bc944c509.json and confirmed 14 keys, k_star=15, n_components=18, d_provisional=18, subsample_key=a79b3460b838fd0a, flags, and 5 notes_for_phase2 entries"
        status: pass
    human_judgment: false
  - id: D4
    description: "Task 3 checkpoint:human-verify (gate=blocking) -- the complete Phase 1 deliverable (notebooks/01_manifold_and_gate.ipynb §0-§6 placeholder, pu_manifold package, four cache-artifact kinds) presented for end-to-end sign-off: clean Restart-and-Run-All reproduction, cache-hit behaviour on re-run, pre-registration ordering, §4 tables, frozen k* match, handoff artifact validity, short-circuit flag surfacing, phase-boundary cleanliness (no eigvalsh, no Phase 2 content), core untouched, output hygiene"
    verification:
      - kind: manual_procedural
        ref: "Coordinator's independent re-verification: direct joblib load of isomap_43cf438bc944c509.joblib, direct JSON re-parse of phase1_handoff_43cf438bc944c509.json, working-tree cleanliness check, fit_key/sweep-key coincidence analysis, and visual inspection of the §2 HSC/LS norm histograms (rendered correctly, left-skewed, band ~13.5-17.0, near-identical between modalities)"
        status: pass
    human_judgment: true
    rationale: "Sealing an entire phase -- and specifically the irreversible k*-freeze that Phases 2-4 all load from -- is explicitly a human-confirmed decision per this plan's own Task 3 design (gate=blocking), not something a test alone can certify as 'correct per intent.'"

# Metrics
duration: ~30min active work across two sessions (task commits + real end-to-end notebook execution, then a Task 3 checkpoint pause awaiting coordinator sign-off, then this closeout)
completed: 2026-07-31
status: complete
---

# Phase 1 Plan 4: Freeze k*, Cache the Fit, and the Phase 1-to-Phase 2 Handoff Summary

**`k* = 15` frozen by the pre-registered plateau rule and cross-checked against the human-confirmed value; the single canonical Isomap fit cached as `isomap_43cf438bc944c509.joblib` (~1.55 GiB, carrying `dist_matrix_`); `phase1_handoff_43cf438bc944c509.json` written as the concrete Phase 1 to Phase 2 interface -- Phase 1 is sealed, all ten requirements complete.**

## Frozen Values (unambiguous, machine-findable)

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

**`n_components_no_headroom: True` -- given prominence deliberately.** `N_COMPONENTS=18` is the exact `ceil(median(...))` over the eight geometric `compute_dim` keys (D-12), with nothing added. This is a **live condition Phase 2 planning must account for**, not an edge case buried in a flags dict: if Phase 2's SPEC-04 residual-variance elbow exceeds 18, `isomap_43cf438bc944c509.joblib` is insufficient and a **re-fit at a larger `n_components` is required**. Because `n_components` is itself a `fit_key` field, that re-fit produces a brand-new `fit_key` automatically -- no manual cache invalidation, no risk of silently reusing a stale pickle -- but it is real, budgeted work Phase 2's plan should anticipate rather than discover.

**`fit_key` coincidence with the plan 03 sweep artifact, not a bug.** `fit_key=43cf438bc944c509` is identical to plan 01-03's `sweep_k15_43cf438bc944c509.npz` key — correct cache-contract behaviour: once `ANALYSIS_CFG["n_neighbors"]` was set to `15` in §5.1, `ANALYSIS_CFG` and the stage-2 sweep's per-k `fit_cfg` for `k=15` became field-for-field identical D-14 config dicts, so `config_key` hashes identically by design. Not the same file: `sweep_k15_43cf438bc944c509.npz` (~1.07 MB slim record, no `dist_matrix_`, plan 03) and `isomap_43cf438bc944c509.joblib` (~1.55 GiB, carries `dist_matrix_`, this plan) are distinct files under distinct stems that merely share a 16-hex suffix.

## Performance

- **Duration:** ~30 min active work: two task commits (`1535010`, `f84f050`) in the first session (~1m20s apart, plus the real k*=15 Isomap fit's wall-clock time inside that window), then a Task 3 `checkpoint:human-verify` (gate `blocking`) pause awaiting coordinator sign-off, then this closeout.
- **Started:** 2026-07-30 (continuation from plan 03)
- **Completed:** 2026-07-31 (checkpoint approved same continuation)
- **Tasks:** 3/3 complete (2 code tasks + 1 human-confirmed verification gate)
- **Files modified:** 1 (`notebooks/01_manifold_and_gate.ipynb`, grown from 74 to 90 cells)

## Accomplishments

- **§5.1 (Task 1):** Applied the §4.0 pre-registered plateau rule to `PLATEAU_RUNS` exactly as written -- no new threshold literal, no re-derivation. Selected the widest run `[10, 15, 30]` (length 3, no tie), bound `K_STAR = 15` at the centre index, and **cross-checked it against the human-confirmed value from `01-03-SUMMARY.md`'s Task 4 checkpoint** (`K_STAR_CONFIRMED = 15`) -- match confirmed, no divergence halt triggered. Re-evaluated `SHORT_CIRCUIT_RISK = False` strictly against `K_STAR` (not `K_SMALLEST_CONNECTED`), bound `K_AUTO_EXTENDED = False`, and set `ANALYSIS_CFG["n_neighbors"] = 15`, fully determining `fit_key = 43cf438bc944c509`.
- **§5.2 (Task 1):** Fit the single canonical `Isomap(n_neighbors=15, n_components=18, eigen_solver="dense")` on the real 10,000-row `LS` array. **Real measured `fit_seconds = 66.86`** (closes the STACK.md wall-clock-benchmark gap flagged since plan 01). Cached as `isomap_43cf438bc944c509.joblib` (1,664,401,892 bytes, ~1.55 GiB), asserted `dist_matrix_.shape == (10000, 10000)` and `embedding_.shape == (10000, 18)`, confirmed `nbrs_`/`kernel_pca_` present, and independently re-verified `n_connected_components == 1` on the fitted estimator's own graph. Demonstrated ISO-05 concretely: a second `joblib_cache` call returned a bit-identical `embedding_` (`CACHE HIT` in 0.58-0.60s across three separate re-runs) and a single `n_neighbors + 1` config mutation produced a different `config_key` (`53a54bf5917e48d0`) without fitting the variant.
- **§5.3 (Task 2):** Built `PHASE1_HANDOFF` entirely from already-bound notebook state (no recomputation) and wrote `phase1_handoff_43cf438bc944c509.json` via `json_cache` -- all 14 top-level keys, the full D-14 `config` field set, both cache keys, `k_star_selection` (including `thresholds`, a copy of `PLATEAU_THRESH`, so the verdict carries the exact values it was judged against), the full `alignment_check`, exactly 3 `flags` (`short_circuit_risk=False`, `k_auto_extended=False`, `n_components_no_headroom=True`), and 5 `notes_for_phase2` entries (`dist_matrix_` availability + truncated-attribute prohibition, the SPEC-04 re-fit branch, the short-circuit-risk status, the S^767 sphere consequence for Phase 3's CURV-06, and the §6-append convention). Same content rendered as a printed table in committed output. **The above-30 short-circuit-risk flag surfaced in all three D-11-required places**: the `warnings.warn`/boxed print in §5.1 (dormant this run since the flag is `False`), `flags.short_circuit_risk` in the JSON, and a prose markdown cell in §5.3 stating the flag's value and that it travels to Phase 2 rather than being resolved here.
- **§5.4 (Task 2):** Reproducibility close-out printing `SEED`, both cache keys, `K_STAR`, `N_COMPONENTS`, `D_PROVISIONAL`, all three flags, `ROW_INDICES_SHA256`, and the total `notebooks/.cache/` size (~1.7 GiB); restated the §0.4 output-hygiene / commit-with-outputs policy as a close-out instruction.
- **Terminal §6 placeholder (Task 2):** The notebook's last cell names Phase 2's scope (`Eigenspectrum Audit and Validity Gate, SPEC-01 through SPEC-07`), states the `§0`-`§5` ownership and no-renumbering rule, and implements none of Phase 2's content -- confirmed mechanically as the final cell.
- **Task 3 (checkpoint:human-verify, gate `blocking`):** Ran the notebook end-to-end via `jupyter nbconvert --execute --inplace` three times total across this plan (once per code task plus one dedicated cache-behaviour re-run, which completed in 36s vs. the original ~103s cold run, confirming `CACHE HIT` at §1.3/§1.6/§5.2). Presented all ten `<how-to-verify>` items with evidence; **coordinator independently re-verified** by loading `isomap_43cf438bc944c509.joblib` directly, re-parsing the handoff JSON, confirming the working tree was genuinely clean after the verification re-run, explaining the `fit_key`/sweep-key coincidence as correct (not a collision), and visually inspecting the §2 norm histograms. **Approved: Phase 1 sealed.**

## Task Commits

Each code task was committed atomically; Task 3 is a checkpoint-only gate (no commit of its own beyond this closeout):

1. **Task 1: §5.1-5.2 -- freeze k*, cache the full Isomap fit (ISO-04, ISO-05)** -- `1535010` (feat)
2. **Task 2: §5.3-5.4 + §6 placeholder -- Phase 1 to Phase 2 handoff (ISO-05)** -- `f84f050` (feat)
3. **Task 3: checkpoint:human-verify (gate=blocking)** -- no commit (gate only); **approved** in this session, Phase 1 sealed.

**Plan metadata:** committed separately after this Summary (see final commit below).

## Files Created/Modified

- `notebooks/01_manifold_and_gate.ipynb` -- grown from 74 to 90 cells; `§5` header, `§5.1`-`§5.4`, and the terminal `§6` placeholder appended. Every cell carries real, committed execution outputs from an actual `Restart and Run All` (via `jupyter nbconvert --execute --inplace`) against the live cached dataset and the real k*=15 Isomap fit. All 32 new code cells carry non-null execution counts; zero error cells across three independent full re-runs.
- `notebooks/.cache/isomap_43cf438bc944c509.joblib` -- 1,664,401,892 bytes, sidecar `.meta.json` manifest. Gitignored.
- `notebooks/.cache/phase1_handoff_43cf438bc944c509.json` -- 3,560 bytes, sidecar `.meta.json` manifest. Gitignored.

## Decisions Made

- **Task 3 (approved, Phase 1 sealed):** coordinator independently re-verified the joblib artifact's attributes/shapes, the handoff JSON's full key/flag/note set, working-tree cleanliness, and the §2 histogram rendering before signing off.
- **`fit_key` == sweep_k15's key is correct, not a bug** (see Frozen Values section above for the full explanation) -- recorded here explicitly per the coordinator's request so a future reader does not mistake identical config hashes for a collision.
- **`n_components_no_headroom: True` carries forward as a live D-12 condition**, not a passive flag -- Phase 2 planning must budget for the possibility of a re-fit at a larger dimension if SPEC-04's elbow exceeds 18.

## Deviations from Plan

None -- plan executed exactly as written. All `<verify>` automated checks passed on first execution for both code tasks (cell-ordering assertions across all four pre-registration points, full token-presence checks, `eigvalsh`-absence, `pyproject.toml`/`src/effdim/` untouched, 14/14 pytest). The Task 3 checkpoint resolved with full approval; no remediation branch was needed (`K_STAR` matched the human-confirmed candidate exactly, `SHORT_CIRCUIT_RISK` stayed `False`).

## Issues Encountered

None this plan. A third full `Restart and Run All` was performed purely to demonstrate the ISO-05 cache-behaviour check (Task 3, item 2) beyond what the two task commits already showed; its resulting timestamp-only diff (no content change -- `git diff` confirmed byte-identical source across all 90 cells) was reverted with a targeted `git checkout -- notebooks/01_manifold_and_gate.ipynb` before staging, per the sanctioned single-file-restore exception, so the committed notebook stays exactly at Task 2's state. `git diff` on the notebook was re-confirmed clean immediately before this closeout -- no stray IDE/`ipykernel` corruption occurred this plan.

## Known Limitations

**Carried forward from `01-03-SUMMARY.md` (already an open entry, `.planning/WINDOWS.md` id 1, not newly introduced or re-acted-on here).** `STAGE2_K = [5, 10, 15, 30]` is unevenly spaced (gaps 5, 5, 15); the connected values `k=8` and `k=20` were never fit because `STAGE2_MAX_FITS=4` dropped them. The plateau run `[10, 15, 30]` that produced this plan's frozen `K_STAR=15` is therefore maximal in *index* space, not *k* space -- a documented gap in the plateau evidence's granularity, disclosed to and accepted by the coordinator at plan 03's Task 4 gate, not acted upon here per that explicit instruction. Phase 2's gate-weighing narrative should treat this as context for how coarse-grained the plateau evidence is, even though `SHORT_CIRCUIT_RISK` itself is clean (`False`).

## User Setup Required

None -- no external service configuration required. All work used the already-warm `notebooks/.cache/` artifacts from plans 01-03; the only new network-independent work was the real k*=15 Isomap fit itself (CPU-bound, no network access needed).

## Next Phase Readiness

- **Phase 1 is sealed. All ten requirements (DATA-01 through DATA-05, ISO-01 through ISO-05) are complete.**
- Phase 2's planner reads `notebooks/.cache/phase1_handoff_43cf438bc944c509.json` directly as its input contract: `k_star=15`, `n_components=18`, `d_provisional=18`, `fit_key=43cf438bc944c509`, `subsample_key=a79b3460b838fd0a`, all three flags, and 5 `notes_for_phase2` entries naming the exact branches Phase 2 must handle (the `dist_matrix_`-only spectrum source, the SPEC-04-elbow re-fit branch, the short-circuit-risk status, the S^767 sphere consequence for CURV-06, and the §6-append convention).
- `notebooks/01_manifold_and_gate.ipynb`'s terminal `§6` placeholder is Phase 2's unambiguous append point; `§0` through `§5` must never be renumbered.
- `isomap_43cf438bc944c509.joblib` is the one artifact in this milestone carrying `dist_matrix_` -- Phase 2's hand-rolled full classical-MDS spectrum reads it directly; `Isomap.kernel_pca_.eigenvalues_` must not be substituted (it is truncated to `n_components` by construction and cannot show a negative tail).
- **Live budgeting item for Phase 2 planning:** `n_components_no_headroom=True` means a SPEC-04 elbow beyond 18 forces a re-fit at a larger dimension. The re-fit is cheap to invalidate correctly (new `fit_key` automatically, no manual cache step) but is real wall-clock work (~67s+ per fit at this n) that should be anticipated, not discovered mid-phase.
- No blockers. `pyproject.toml` and `src/effdim/` remain byte-identical to their pre-plan state (verified via `git diff --quiet` after every task and at this closeout).

---
*Phase: 01-data-loading-manifold-reconstruction*
*Completed: 2026-07-31*

## Self-Check: PASSED

- `notebooks/01_manifold_and_gate.ipynb` verified present on disk (90 cells, §5.1-§5.4 + §6 placeholder).
- `notebooks/.cache/isomap_43cf438bc944c509.joblib` (1,664,401,892 bytes) verified present on disk.
- `notebooks/.cache/phase1_handoff_43cf438bc944c509.json` verified present on disk.
- Both task commits (`1535010`, `f84f050`) verified present in `git log --oneline --all`.
- `git diff` on the notebook re-confirmed clean immediately before this closeout (no stray `ipykernel`/IDE corruption).
- No missing items.
