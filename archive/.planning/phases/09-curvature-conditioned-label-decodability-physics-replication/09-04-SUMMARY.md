---
phase: 09-curvature-conditioned-label-decodability-physics-replication
plan: 04
subsystem: data-loading
tags: [huggingface, data-manifest, checkpoint, label-mapping, ratification]

# Dependency graph
requires:
  - phase: 09-03
    provides: "physics_labels.py's revision-pinned, column-projected loaders and 09_row_alignment_proof_run.py's --mode manifest CLI/record contract (row_kind key, comma-joined --candidate-columns), which this plan's Task 1 ran unmodified"
provides:
  - "09-DATA-MANIFEST.md: the full-scale (86,471-row, 16-shard) measurement of both HuggingFace datasets, per-column missingness for all 7 candidate columns compared against 09-RESEARCH.md's single-shard figures, the proposed mapping with evidence for and against every rejected alternative, and the developer's ratifying checkpoint answer transcribed verbatim"
  - "notebooks/.cache/09_data_manifest.jsonl: 8-row JSONL record (7 per-column + 1 summary), gitignored, carrying no statistic key -- the artifact 09-05's freeze constants are read from"
  - "The literal LABEL_COLUMN_MAP, SENTINEL_VALUES and ALIGNMENT_MARGIN_R2 values 09-05 must write into physics_labels.py's module constants, in the exact form it will use them"
  - "Assumption A1 (stellar_mass -> mass_med_photoz) resolved from [ASSUMED] to [RATIFIED 2026-09-03] by developer ruling, not independent confirmation of the colleague's build script"
affects: [09-05, 09-06, 09-07, 09-08, 09-09, 09-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single checkpoint carrying three one-way values (column mapping, sentinel set, alignment margin) that all enter the same freeze commit and share the same undo cost, rather than three separate checkpoints -- keeps developer attention on the one genuinely contested reading (09-CONTEXT.md's out-of-scope line) instead of gating already-settled arithmetic"
    - "Checkpoint ruling recorded as two distinct parts in the artifact: a verbatim quotation block of the developer's exact words (treated as data, never as executor instruction) followed by a separate planner-written paragraph stating what was literally applied -- the same separation 09-CONTEXT.md and 09-RESEARCH.md use elsewhere for developer rulings"

key-files:
  created: []
  modified:
    - .planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-DATA-MANIFEST.md

key-decisions:
  - "Developer ratified the proposed mapping, sentinel set and alignment margin unchanged (`ratify-as-proposed`), resolving 09-CONTEXT.md's out-of-scope line under the narrow reading: it strikes only the colleague's unresolved DESI cross-match associations (marked desi_label_alignment_unresolved, Proved=False), not mag_r_desi itself -- a photometry column, not one of those associations"
  - "LABEL_COLUMN_MAP frozen as {mag_r: mag_r_desi, photo_z: photo_z, smooth_fraction: smooth-or-featured_smooth_fraction, stellar_mass: mass_med_photoz}; SENTINEL_VALUES = (-99.0,); ALIGNMENT_MARGIN_R2 = 0.10 -- all three now literal for 09-05 to write, none filled by this plan"
  - "Assumption A1 marked [RATIFIED 2026-09-03] rather than silently cleared: the developer's ratify-as-proposed answer covers the mapping that includes stellar_mass -> mass_med_photoz with no amendment, but this is a ruling on the mapping choice, not an independent byte-for-byte confirmation of the colleague's absent build script"

requirements-completed: [D9-01, D9-05, D9-07, D9-16, D9-18]

coverage:
  - id: D1
    description: "Both datasets measured at full scale (all 16 label shards, the full embedding parquet); row counts recorded as exact integers against 86,471, resolving Assumption A2 on measurement"
    requirement: "D9-01"
    verification:
      - kind: other
        ref: "notebooks/.cache/09_data_manifest.jsonl summary row: n_rows_embeddings=86471, n_rows_labels=86471; 09-DATA-MANIFEST.md Section 2 states both equal 86,471"
        status: pass
    human_judgment: false
  - id: D2
    description: "Per-column missingness measured for all 7 candidates (4 chosen, 3 rejected), each compared against 09-RESEARCH.md's single-shard figure; mass_med_photoz's post-masking count (79,490) checked against the colleague's reported 79,490/86,471"
    requirement: "D9-16"
    verification:
      - kind: other
        ref: ".venv/bin/python -c \"import json; rows=[json.loads(l) for l in open('notebooks/.cache/09_data_manifest.jsonl')]; assert len(rows) >= 8\" -- passed, 8 rows, no r2/rho/p/passed key"
        status: pass
    human_judgment: false
  - id: D3
    description: "The developer ratified the raw-column mapping, sentinel set and alignment margin at a blocking checkpoint; the ruling is recorded verbatim beside a separate planner statement of what was applied"
    requirement: "D9-05"
    verification:
      - kind: other
        ref: ".venv/bin/python -c \"...assert 'Pending — see Task 2' not in t...assert 'ALIGNMENT_MARGIN_R2' in t and 'SENTINEL_VALUES' in t and 'LABEL_COLUMN_MAP' in t\" -- printed 'ruling recorded'"
        status: pass
    human_judgment: false
  - id: D4
    description: "No statistic exists anywhere in the tree; no gating constant is filled; no file under notebooks/ or src/effdim/ is modified"
    requirement: "D9-18"
    verification:
      - kind: other
        ref: "git status --porcelain notebooks/ src/effdim/ -- printed nothing after both tasks"
        status: pass
    human_judgment: false

# Metrics
duration: overnight checkpoint wait (Task 1: 2026-09-02T22:02Z; Task 2 resumed and completed 2026-09-03T04:52Z after the developer's reply)
completed: 2026-09-03
status: complete
---

# Phase 9 Plan 4: Full-Scale Data Manifest and Column-Mapping Ratification Summary

**Both datasets measured at full 86,471-row scale across all 16 label shards; the developer ratified the `mag_r_desi`/`mass_med_photoz` column mapping, the `-99.0` sentinel and the `0.10` alignment margin unchanged at a blocking checkpoint, resolving the DESI out-of-scope reading in favour of the colleague's own primary-result column.**

## Performance

- **Duration:** Task 1 committed 2026-09-02T22:02Z (previous session); this continuation resumed at Task 2's checkpoint after the developer's reply and completed at 2026-09-03T04:52Z
- **Started:** 2026-09-02T21:40:28Z (manifest run timestamp, Task 1)
- **Completed:** 2026-09-03T04:52:27Z
- **Tasks:** 2 completed (Task 1 in a prior session, Task 2 in this continuation)
- **Files modified:** 1 (`09-DATA-MANIFEST.md`, across both tasks)

## Accomplishments
- Ran `--mode manifest` once at full scale against all 16 `Smith42/galaxies@v2.0` shards and the `UniverseTBD/pu-embeddings` physics parquet; both row counts measured at exactly 86,471, resolving Assumption A2 on measurement
- Measured all 7 candidate raw columns' finite/sentinel/masked counts at full scale, each compared against `09-RESEARCH.md`'s single-shard figures; `mass_med_photoz`'s post-masking finite count (79,490/86,471) reproduced the colleague's own reported figure exactly
- Wrote the proposed mapping, its evidence, the three rejected alternatives' evidence, and the DESI out-of-scope reconciliation question into `09-DATA-MANIFEST.md`, then presented it to the developer at a blocking `checkpoint:decision`
- Developer answered `ratify-as-proposed` (2026-09-03 UTC); transcribed verbatim into Section 7 alongside a planner-written paragraph naming the literal `LABEL_COLUMN_MAP`, `SENTINEL_VALUES` and `ALIGNMENT_MARGIN_R2` values 09-05's freeze commit must write
- Assumption A1 (`stellar_mass` -> `mass_med_photoz`) updated from `[ASSUMED]` to `[RATIFIED 2026-09-03]`, not silently cleared

## Task Commits

1. **Task 1: Run the full-scale data manifest and write the evidence document** - `1ed6eee` (docs)
2. **Task 2: Ratify the raw-column mapping, the sentinel set and the alignment margin** - `acd4fad` (docs)

## Files Created/Modified
- `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-DATA-MANIFEST.md` - Task 1 wrote the full-scale measurement (row counts, per-column table, proposed mapping, reconciliation question, provenance framing, empty Ruling section); Task 2 filled the Ruling section with the developer's verbatim reply and the applied constants
- `notebooks/.cache/09_data_manifest.jsonl` - 8-row JSONL record written by Task 1's manifest run (gitignored, not committed)

## Decisions Made
- Developer ratified the proposed mapping, sentinel set and alignment margin unchanged (`ratify-as-proposed`)
- The DESI out-of-scope line in `09-CONTEXT.md` is read narrowly: it strikes only the colleague's unresolved DESI cross-match associations, not `mag_r_desi`
- `LABEL_COLUMN_MAP = {"mag_r": "mag_r_desi", "photo_z": "photo_z", "smooth_fraction": "smooth-or-featured_smooth_fraction", "stellar_mass": "mass_med_photoz"}`
- `SENTINEL_VALUES = (-99.0,)`
- `ALIGNMENT_MARGIN_R2 = 0.10`
- Assumption A1 marked `[RATIFIED 2026-09-03]`

## Checkpoint Record

**Type:** `checkpoint:decision`, `gate="blocking"`
**Question:** Which raw catalog column each canonical label resolves to, which sentinel values are masked, and what pre-registered margin the row-alignment proof must clear -- all one-way, all entering 09-05's freeze commit.
**Options presented:** `ratify-as-proposed`, `ratify-with-amendments`, `strike-desi-and-halt`
**Developer's reply (verbatim, 2026-09-03 UTC):**
> ratify-as-proposed

**Applied:** The proposed mapping, sentinel set and alignment margin, unchanged, exactly as laid out in `09-DATA-MANIFEST.md` Sections 4-5. Full text is in `09-DATA-MANIFEST.md` Section 7 alongside the developer's own words, per the plan's prohibition against silently applying a value without the verbatim record beside it.

## Deviations from Plan

None - both tasks executed exactly as written. Task 2's checkpoint was answered `ratify-as-proposed` with no amendment, so no architectural or value change was needed.

## Issues Encountered
None. Both tasks' automated verify steps passed on first run; `git status --porcelain notebooks/ src/effdim/` printed nothing after each task, confirming no gating constant was filled and no source file touched.

## User Setup Required
None for this plan. 09-05 (the freeze commit) and 09-06 onward (execution-host hand-off, still undecided per `STATE.md`) remain ahead.

## Next Phase Readiness
- 09-05 can write the freeze commit using the exact `LABEL_COLUMN_MAP`, `SENTINEL_VALUES` and `ALIGNMENT_MARGIN_R2` values recorded in `09-DATA-MANIFEST.md` Section 7
- Both `assert_preregistered()` functions still raise; no Physics number exists anywhere in the tree; `notebooks/.cache/09_row_alignment.jsonl` and `notebooks/.cache/09_physics_curvature.jsonl` do not exist
- No blockers

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Completed: 2026-09-03*

## Self-Check: PASSED

All 3 files found on disk (`09-DATA-MANIFEST.md`, `notebooks/.cache/09_data_manifest.jsonl`,
`09-04-SUMMARY.md`); both commits (`1ed6eee`, `acd4fad`) found in git history.
