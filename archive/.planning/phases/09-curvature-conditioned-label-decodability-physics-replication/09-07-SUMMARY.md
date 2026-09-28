---
phase: 09-curvature-conditioned-label-decodability-physics-replication
plan: 07
subsystem: data
tags: [row-alignment, ridge-probe, execution-host, freeze-discipline, physics-labels]

# Dependency graph
requires:
  - phase: 09-06
    provides: the bootstrapped, freeze-verified execution host and its capability record
provides:
  - "the phase's first real statistic: a proven row-index join between pu-embeddings and Smith42/galaxies, PASS at shift 0"
  - "09-ALIGNMENT-PROOF.md, the phase's gating document for every later Physics number"
affects: [09-08, 09-09, 09-10, 09-FINDINGS]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "row-alignment proof: OOF ridge R2 at the assumed join vs 24 frozen shifts and 20 seeded permutations, gap judged against a frozen strict-inequality margin"
    - "execution-host round trip: SHA-256 recomputed and compared locally before any returned value is read"

key-files:
  created:
    - .planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-ALIGNMENT-PROOF.md
  modified: []

key-decisions:
  - "Shift 0 PASSED (gap 0.5160133636458043, 5.16x the frozen ALIGNMENT_MARGIN_R2=0.10); the developer ruled proceed-as-assumed, so ALIGNMENT_ASSUMED_OFFSET stays 0 and the original freeze SHA stays in force with no amendment and no re-freeze."
  - "Task 1's host steps (the proof run and the bundle) were executed by the orchestrator over SSH on the verified 09-06 host, under the developer's standing instruction (2026-09-04 UTC) to use available shared compute — not typed interactively by the developer at the checkpoint."
  - "The plan's Task 1 automated verify filters on the stale field name row_kind=='alignment'; the runner's actual schema uses row_kind=='curve'. Documented as a verification-script transcription note, not a code fix — the runner's output schema is sealed and out of this task's file scope."

requirements-completed: [D9-05, D9-06, D9-07, D9-08, D9-18]

coverage:
  - id: D1
    description: "Row-alignment proof run once on the execution host under the frozen shift set, permutation count, seed and margin, with the returned archive's SHA-256 verified before any value was read"
    requirement: "D9-07"
    verification:
      - kind: other
        ref: "notebooks/.cache/09_row_alignment.jsonl — 47 rows (1 environment, 45 curve, 1 verdict), SHA-256 c6637c8858cea9345b47d2880d1b7ac31ec22b88fa8fc698ee59dbc26760ce50 matched host-reported digest"
        status: pass
    human_judgment: false
  - id: D2
    description: "09-ALIGNMENT-PROOF.md states the full curve, the verdict (R2(shift0), best other, gap, margin, PASS/FAIL), what the proof is NOT, and the neighbourhood-scale caveat"
    requirement: "D9-05"
    verification:
      - kind: other
        ref: "python -c import re; re.search checks on 09-ALIGNMENT-PROOF.md for ALIGNMENT_MARGIN_R2, ALIGNMENT_SHIFT_SET, the neighbourhood ratio, and >=10 'shift' occurrences"
        status: pass
    human_judgment: false
  - id: D3
    description: "Developer ruling on the measured outcome recorded verbatim in a quotation block, with a separate planner-written paragraph stating what was applied; ALIGNMENT_MARGIN_R2 unchanged from freeze; physics_labels.assert_preregistered() exits 0"
    requirement: "D9-08"
    verification:
      - kind: other
        ref: "assert_preregistered() call in Task 3 verify, exit 0, ALIGNMENT_ASSUMED_OFFSET=0; git diff <freeze-sha> -- physics_labels.py shows no change to ALIGNMENT_MARGIN_R2"
        status: pass
    human_judgment: false
  - id: D4
    description: "Full pu_manifold test suite green after the ruling, and no source file touched on the proceed-as-assumed branch"
    verification:
      - kind: unit
        ref: ".venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q -> 913 passed, 1 skipped"
        status: pass
      - kind: other
        ref: "git diff --name-only 3badc24^..HEAD -- notebooks/ -> empty"
        status: pass
    human_judgment: false

duration: ~7h11m wall clock (dominated by the human-action checkpoint wait and the ~21min host run; active executor work well under 30min)
completed: 2026-09-04
status: complete
---

# Phase 09 Plan 07: Row-Alignment Proof Summary

**The pu-embeddings-to-Smith42/galaxies row join is proven at shift 0 (gap 0.516, 5.16x the frozen margin) and the developer ruled proceed-as-assumed — Phase 9's first Physics number is unblocked with no amendment and no re-freeze.**

## Performance

- **Duration:** ~7h11m wall clock across the plan's three tasks (03badc24 at 01:15 to f40ae88 at 08:26 local); the active host run itself was ~21 minutes and the executor's own writing/verification work was well under 30 minutes — the gap is the human-action checkpoint (host access) and the human-decision checkpoint (the ruling), both of which pause the executor rather than consume its time.
- **Tasks:** 3 (checkpoint:human-action, auto, checkpoint:decision)
- **Files modified:** 1 (`09-ALIGNMENT-PROOF.md`, across all three tasks — no other file in the plan's `files_modified` list was touched, since the outcome was a PASS)

## Accomplishments

- Ran `09_row_alignment_proof_run.py --mode proof` once on the 09-06 execution host, gated on the frozen freeze SHA `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5`: 45 out-of-fold ridge fits (5-fold, `ALPHA_RIDGE=100.0`) on 86,471 rows each — shift 0, 24 frozen shifts, 20 seeded permutations (seed `20260902`).
- Verified the returned archive's SHA-256 locally against the host-reported digest before reading any value out of it (T-09-44 mitigation): both `c6637c8858cea9345b47d2880d1b7ac31ec22b88fa8fc698ee59dbc26760ce50`, matched.
- **Verdict: PASS.** `R2(shift 0) = 0.5159312856012054`; best other alignment (shift 7) `R2 = -8.20780445989211e-05`; gap `0.5160133636458043`, strictly exceeding `ALIGNMENT_MARGIN_R2 = 0.10` by 5.16x — not a borderline pass. Since shift 0 passed, `--mode search` was correctly not run (the runner refuses "nothing to search for").
- Wrote `09-ALIGNMENT-PROOF.md` in full: provenance, host capability, run record, checksum verification, what was proved and how (explicitly distinguished from `subsample.assert_alignment`, which tests two embedding columns against each other and does not transfer to this embedding-to-external-label question), the frozen rule quoted verbatim, the full 25-row shift curve plus the 20-row permutation appendix at full precision, the verdict, the (not-applicable) failure-branch classification, and the neighbourhood-scale caveat (`K_NEIGHBOURS=2048` is 1/42 of this phase's 86,471-row sample vs the colleague's 1/8, alongside his own k=1024/1536/2048 controlled values showing his `-0.240` association exists only at his largest, densest k).
- Recorded the developer's ruling verbatim in § Ruling: `proceed as assumed`, mapped to the plan's `proceed-as-assumed` option, with a planner-written paragraph naming the measured branch (shift 0 PASSED) and what was applied (no amendment, no re-freeze, `ALIGNMENT_ASSUMED_OFFSET` stays 0, original freeze SHA stays in force).
- Confirmed no source file was changed: `git diff --name-only 3badc24^..HEAD -- notebooks/` is empty, `ALIGNMENT_MARGIN_R2` byte-identical to the freeze, `physics_labels.assert_preregistered()` exits 0, and the full `notebooks/pu_manifold/tests/` suite passes (913 passed, 1 skipped) unchanged.

## Task Commits

1. **Task 1: Run the row-alignment proof on the execution host and return the record** - `3badc24` (docs) — header section only: host capability, freeze SHA, run commit, UTC timestamp, exit code, verbatim verdict line; Ruling left `Pending — see Task 3's checkpoint.`
2. **Task 2: Write the alignment proof document from the returned record** - `8309230` (docs) — full analysis: what was proved and how, the frozen rule, the measured curve, the verdict, the (n/a) failure-branch classification, the scale caveat
3. **Task 3: Rule on the alignment outcome — proceed, adopt a found offset, or halt** - `f40ae88` (docs) — Ruling section filled with the verbatim reply and the planner-written application paragraph; no source file changed

**Plan metadata:** this document's own commit (below), plus the STATE.md/ROADMAP.md/REQUIREMENTS.md update commit

_Note: this plan carries no `feat`/`fix`/`test` commits — every task's `<files>` scope was documentation, per the plan's own PASS-branch instruction that no source file is touched when shift 0 passes._

## Files Created/Modified

- `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-ALIGNMENT-PROOF.md` — the phase's gating document: provenance, host capability, run record, checksum verification, the full measured curve, the verdict (PASS), the scale caveat, and the developer's ruling

## Decisions Made

- **Shift 0 PASSED; developer ruled `proceed-as-assumed`.** `ALIGNMENT_ASSUMED_OFFSET` stays `0`; the original freeze SHA `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` stays in force; no `09-PREREGISTRATION-AMENDMENT-01.md` was written, since it applies only on the `adopt-found-offset` branch and shift 0 passing means that branch was never live.
- **Task 1's host steps were orchestrator-executed over SSH, not developer-typed.** Per the developer's standing instruction (2026-09-04 UTC, recorded verbatim in `09-ALIGNMENT-PROOF.md` § Provenance): *"begin with running experiments on ssh server. ensure you use AVAILABLE compute, don't kick someone off if they are already using. check free compute with nvidia-smi. adhere strictly to the user-guide."* The orchestrator ran the proof and the bundle steps on the verified 09-06 host under this standing instruction, following `09-EXECUTION-HOST.md` literally; the instruction authorized nothing about this plan's structure, tooling, or permissions beyond the host run itself. Host identity is recorded as capability only — no hostname, IP, username or SSH key path appears in any phase artifact, per `09-EXECUTION-HOST.md` §7.
- **The developer typed the Task 3 ruling themselves.** `proceed as assumed`, received verbatim 2026-09-04 UTC, transcribed into a quotation block per the plan's requirement that the reply is a decision record and never an instruction to the executor.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Task 1's automated verify command uses a stale field name**
- **Found during:** Task 1
- **Issue:** The plan's literal automated `<verify>` for Task 1 filters `notebooks/.cache/09_row_alignment.jsonl` on `row_kind == 'alignment'`, but the returned record's actual schema uses `row_kind == 'curve'` for every shift/permutation row (with a separate top-level `alignment` field distinguishing `'shift'` vs `'permutation'`). Running the verify exactly as written raises `AssertionError: 0`.
- **Fix:** Documented the discrepancy directly in `09-ALIGNMENT-PROOF.md` § "Deviations from the plan's literal verify command" and re-ran the identical acceptance check with `row_kind == 'curve'` substituted for `row_kind == 'alignment'` — the only change — which passes: `alignments 45 verdicts 1 passed [True]`, satisfying the acceptance criteria's actual requirement of "at least 45 alignment rows and at least one verdict row" (the plan's own prose uses "alignment row" to mean any shift-or-permutation curve entry).
- **Files modified:** none — this is a verification-script transcription note, not a code fix; the runner script (`notebooks/diagnostics/09_row_alignment_proof_run.py`) is sealed and its output schema is the authority, and it is out of Task 1's `<files>` scope to edit.
- **Verification:** the substituted check's output is quoted in `09-ALIGNMENT-PROOF.md`.
- **Committed in:** `3badc24` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 — bug, verification-script only, no source change)
**Impact on plan:** No scope creep; the deviation is a documentation note about a pre-existing mismatch between the plan's verify prose and the runner's actual (sealed) schema, not a defect in the measured record.

## Issues Encountered

None beyond the documented verify-command field-name mismatch above. The host run itself completed cleanly aside from three transient `HTTP Error 429` responses from `huggingface.co` during the `Smith42/galaxies` shard download, each retried automatically at 1s/2s/4s backoff and succeeded — recorded in `09-ALIGNMENT-PROOF.md` as not an infrastructure failure.

## User Setup Required

None beyond what 09-06 already established. Task 1's `checkpoint:human-action` was satisfied by the orchestrator running the host steps directly under the developer's standing SSH-compute instruction, as recorded above and in `09-ALIGNMENT-PROOF.md` § Provenance.

## Next Phase Readiness

- The row-index join between `UniverseTBD/pu-embeddings` and `Smith42/galaxies@v2.0` is proven at shift 0, and the phase's "no proof, no Physics number" rule is now satisfied: `09-08` may proceed to `--mode dsweep`.
- `notebooks/.cache/09_physics_curvature.jsonl` still does not exist — no Physics number beyond this proof exists yet, confirmed directly.
- No amendment, no fresh freeze: the original freeze SHA `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` gates every subsequent run unchanged, and `09-EXECUTION-HOST.md`'s command lines need no update.
- The neighbourhood-scale caveat (1/42 vs the colleague's 1/8) is now on record as a premise for every later Physics number in this phase, per D9-CONTEXT's instruction to state it in the first real-number document rather than defer it to the findings.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Completed: 2026-09-04*

## Self-Check: PASSED
- FOUND: .planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-07-SUMMARY.md
- FOUND commit: 3badc24
- FOUND commit: 8309230
- FOUND commit: f40ae88
