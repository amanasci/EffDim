---
phase: 09-curvature-conditioned-label-decodability-physics-replication
plan: 06
subsystem: infra
tags: [ssh-remote, bootstrap, artifact-bundling, cost-model, runbook, environment-reporting]

# Dependency graph
requires:
  - phase: 09-05
    provides: FREEZE_COMMIT_SHA wired into both runners, both assert_preregistered() passing
provides:
  - "--mode bundle, --print-cost-model, --output-root, --host-label on both Phase 9 runners"
  - "_describe_environment() reporting on both runners before any read or write"
  - "09-EXECUTION-HOST.md: fresh-clone bootstrap runbook, the full run sequence, cost table, artifact-return list, do-not-do list"
  - "A real execution host, bootstrapped and proven green on both smoke paths, with its capability recorded"
  - "A verified, extracted smoke bundle under the local output root (notebooks/.cache/)"
affects: [09-07, 09-08, 09-09, 09-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-thread cost model stated in core-hours, never in unqualified wall-clock minutes"
    - "Environment description printed before any read/write and embedded in every non-smoke record and every bundle"
    - "Host identity recorded as capability (core count, OS, Python/library versions) never as an address"

key-files:
  created:
    - .planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-EXECUTION-HOST.md
  modified:
    - notebooks/diagnostics/09_physics_curvature_run.py
    - notebooks/diagnostics/09_row_alignment_proof_run.py

key-decisions:
  - "--print-cost-model wired into the physics runner only (the D_SWEEP-bearing one); the alignment runner's fixed cost (45 OOF ridge fits, two gates) is documented directly in 09-EXECUTION-HOST.md's cost table instead of a duplicate flag -- the prior executor's discretion call to keep the addition minimal (recorded in 282e27f's commit message)."
  - "Cost model is stated in core-hours plus an implied wall-clock at a named thread count, not Phase 7's bare minutes -- the execution host's core count was unknown at planning time, so a bare figure would have been wrong on arrival."
  - "Training and curvature costs kept as two separate components rather than one scaled total, since Phase 9 evaluates curvature at 512 anchors instead of every row (D9-04), reversing which term dominates relative to Phase 7."
  - "The execution host was chosen and bootstrapped by the orchestrator over SSH, acting on the developer's explicit instruction (2026-09-04 UTC) to use available compute on the SSH server, check for free capacity with nvidia-smi before using it, and follow the host's own user guide strictly -- not typed interactively by the developer. Recorded as an evidence transcription, not as a directive that altered this plan's structure or tooling."
  - "The host's system python3 (3.10.12) could not satisfy the numpy/scipy pins (require >=3.12), so a Python 3.14.7 interpreter was provisioned via the host's own persistent-environment recipe (mamba) and Section 3's .venv was created from that interpreter instead of the bare system python3 -- every other runbook command still runs literally as written."

requirements-completed: [D9-01, D9-12, D9-18]

coverage:
  - id: D1
    description: "Both runners describe their environment (core count, thread cap, seven library versions, resolved cache/output paths, git HEAD, freeze SHA) before any read or write, printed and as a JSONL row on non-smoke records"
    requirement: "D9-01"
    verification:
      - kind: other
        ref: ".venv/bin/python notebooks/diagnostics/09_physics_curvature_run.py --print-cost-model --threads 8 (Task 1 acceptance criteria)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Per-thread cost model (--print-cost-model) prints per-d core-hour components and implied wall-clock at a named thread count, never an unqualified 'on this machine' figure"
    requirement: "D9-12"
    verification:
      - kind: other
        ref: "Task 1 acceptance criteria: header contains threads=8 and host core count; output contains 'core-hour', not 'on this machine'"
        status: pass
    human_judgment: false
  - id: D3
    description: "--mode bundle packs every 09_-prefixed artifact under the resolved output root into one checksummed, gzipped tar, exiting 0 even on a partial set"
    requirement: "D9-01"
    verification:
      - kind: other
        ref: "Task 1 acceptance criteria: archive path/size/SHA-256 printed; tar -tzf lists scratch tracer + environment member; EFFDIM_09_OUTPUT_ROOT honoured"
        status: pass
    human_judgment: false
  - id: D4
    description: "09-EXECUTION-HOST.md is a complete fresh-clone-to-green-smoke-bundle runbook: bootstrap, full run sequence with the literal freeze SHA, cost table, artifact-return list, do-not-do list, failure modes -- refers to the host generically throughout"
    requirement: "D9-18"
    verification:
      - kind: other
        ref: "Task 2's own automated verify script: exactly one 40-hex string (matches freeze SHA), both ancestry commands present, 'core-hour' present, 'on this machine' absent"
        status: pass
    human_judgment: false
  - id: D5
    description: "A real execution host was chosen, bootstrapped from a fresh clone with no file copied from the developer's machine, proven against the freeze gate, proven green on both smoke paths, and its capability recorded as a new 'Host as bootstrapped' section -- with no hostname, IP, username or key path written anywhere"
    requirement: "D9-18"
    verification:
      - kind: other
        ref: "Task 3's own automated verify script (section present, 64-hex archive digest present, 'core count' present) plus manual grep for '@|ssh |<ipv4>' patterns, both run in this session"
        status: pass
    human_judgment: true
    rationale: "Confirming the returned archive's SHA-256 genuinely matches the received file, and that no host-identifying string leaked into a committed document, are the two checks this plan treats as requiring a human-auditable trail rather than a purely mechanical pass -- recorded here for the next reviewer to re-check independently."

# Metrics
duration: ~25min (this session, Task 3 only; Tasks 1-2 were a prior session)
completed: 2026-09-04
status: complete
---

# Phase 9 Plan 6: Execution-host bootstrap and portability Summary

**Both Phase 9 runners can now bundle their own artifacts and state their cost per thread; the execution host is a real, bootstrapped machine (128 cores, Python 3.14.7, all pinned libraries matched) proven green on both smoke paths, with a checksummed smoke bundle verified and extracted locally.**

## Performance

- **Duration:** Tasks 1-2 ran in a prior session (2026-09-03); this session completed Task 3 in ~25 min.
- **Completed:** 2026-09-04T04:47:51Z
- **Tasks:** 3 (Task 1: auto, Task 2: auto, Task 3: checkpoint:human-action)
- **Files modified:** 3 (2 runner scripts, 1 new runbook document)

## Accomplishments

- `_describe_environment()`, `DSWEEP_COST_MODEL_CORE_HOURS`, `print_cost_model()` and `run_bundle()` added to `09_physics_curvature_run.py`; the bundling half added to `09_row_alignment_proof_run.py`. Both runners print their environment before touching data, state cost in portable core-hours, and can pack every `09_`-prefixed artifact into one checksummed archive.
- `09-EXECUTION-HOST.md` written: an 8-section runbook taking a fresh clone from `git clone` to a green smoke bundle, the full real-mode run sequence with the literal freeze SHA `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` written into every command line, a per-thread cost table, the artifact-return list naming which plan ingests each returned file, and a do-not-do list.
- The execution host was chosen (an SSH remote pod, following its own user guide, capacity-checked with `nvidia-smi` before use per the developer's instruction), bootstrapped from a fresh clone with no file copied from the developer's machine, proven against the freeze gate (`git merge-base --is-ancestor` exit 0, `git rev-list --count` = 5), and proven green on both smoke paths (`SMOKE PASS`, `ALIGNMENT SMOKE PASS`).
- The returned smoke bundle's SHA-256 (`20c6a8ba28f3b9b95ba9e01164520a3f5d33fdcc5f1949146fc5c3aeb99338cd`) was re-verified locally against the received file before extraction, matched, and the archive was extracted under the local resolved output root (`notebooks/.cache/`). A new "Host as bootstrapped" section was appended to `09-EXECUTION-HOST.md` recording the host's capability (never its address) and marking Section 3's bootstrap sequence verified.

## Task Commits

Each task was committed atomically:

1. **Task 1: Artifact bundling, environment reporting and a per-thread cost model on both runners** - `282e27f` (feat)
2. **Task 2: Write the execution-host runbook** - `ee992ba` (docs)
3. **Task 3: Choose the execution host, bootstrap it from a fresh clone, and return a green smoke bundle** - `7fa8128` (docs), fixed up by `4d3519f` (fix)

**Plan metadata:** (this commit, following)

## Files Created/Modified

- `notebooks/diagnostics/09_physics_curvature_run.py` - `_describe_environment()`, `DSWEEP_COST_MODEL_CORE_HOURS`, `print_cost_model(threads)`, `run_bundle(args)`; `--mode bundle`, `--print-cost-model`, `--host-label`, `--output-root` wired
- `notebooks/diagnostics/09_row_alignment_proof_run.py` - the bundling half of the same capability, sharing archive naming with the physics runner
- `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-EXECUTION-HOST.md` - created in Task 2 (8 sections), appended in Task 3 (§9 "Host as bootstrapped")

## Decisions Made

- **`--print-cost-model` wired into the physics runner only.** The alignment runner's cost (45 OOF ridge fits, two gates) is fixed and small; it is documented directly in the runbook's cost table rather than behind a duplicate flag. The prior executor's discretion call, recorded in `282e27f`'s own commit message.
- **Cost stated in core-hours plus a named thread count, never a bare wall-clock.** The execution host's core count was unknown at planning time; a bare minute figure from Phase 7's own 8-thread-cap hardware would have been meaningless on arrival.
- **Training and curvature costs kept as two separate components.** Phase 9 evaluates curvature at 512 anchors instead of every row (D9-04), which reverses which term dominates relative to Phase 7 — a single scaled total would have hidden that.
- **The checkpoint was answered by the orchestrator executing the host steps over SSH, on the developer's explicit instruction** (2026-09-04 UTC): *"begin with running experiments on ssh server. ensure you use AVAILABLE compute, don't kick someone off if they are already using. check free compute with nvidia-smi. adhere strictly to the user-guide."* This is recorded as an evidence transcription of what happened on the host, not as a directive that altered this plan's structure, tooling, or permissions.
- **Bootstrap deviation, recorded per Task 3's instructions:** the host's system `python3` (3.10.12) could not satisfy `numpy==2.5.1`/`scipy==1.18.0` (both require Python >= 3.12). Per the host's own persistent-environment recipe, a Python 3.14.7 interpreter was provisioned with `mamba` first, and Section 3's `.venv` was created from that interpreter rather than the bare system `python3`, so every other runbook command still runs literally as written via `.venv/bin/python`. No sealed module was touched; the host clone's `git status` is clean.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Host clone HEAD's full 40-hex SHA broke the runbook's "exactly one 40-hex string" invariant**
- **Found during:** Task 3, immediately after drafting the "Host as bootstrapped" section
- **Issue:** Task 2's own acceptance criterion requires `09-EXECUTION-HOST.md` to carry exactly one distinct 40-hex string (the freeze SHA). The new section recorded the host clone's full HEAD commit SHA (`ee992bac947f3469dfb0e607867901992f0b17de`), which is itself 40 hex characters, producing a second match and breaking the invariant.
- **Fix:** Shortened the recorded HEAD to its short form (`ee992ba`), consistent with the short form already used elsewhere in the same section (`environment.json`'s `git_describe_head`).
- **Files modified:** `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-EXECUTION-HOST.md`
- **Verification:** Re-ran both Task 2's and Task 3's automated verify scripts; exactly one 40-hex string remains (the freeze SHA), both acceptance checks pass.
- **Committed in:** `4d3519f` (separate fix commit, after `7fa8128`)

---

**Total deviations:** 1 auto-fixed (1 bug), plus one bootstrap-time environment deviation documented per the plan's own instructions (Python interpreter provisioning — not a defect in this plan's code, a fact about the host recorded because the plan requires it).
**Impact on plan:** Both are necessary corrections/records; no scope creep. No frozen constant touched at any point (`git diff --name-only 2f38585..HEAD -- notebooks/pu_manifold/` prints nothing; both `assert_preregistered()` calls still exit 0).

## Issues Encountered

- **Transcription discrepancy in the checkpoint's own resume-signal narrative, not in the underlying evidence.** The checkpoint's reported summary stated the alignment smoke's offset case as `passed=False`; the archive's actual record (`09_scratch_alignment.jsonl`, extracted and read directly in this session) shows `"passed": true` for both the aligned and offset cases, consistent with the printed `ALIGNMENT SMOKE PASS` banner and exit 0. This SUMMARY records the archive's actual value, not the narrative's. No action was taken on the strength of the narrative alone — the archive was the source of truth, per Task 3's own instruction to verify before reading anything out of it.
- **Stale banner wording, noted but not fixed (out of this plan's scope).** The alignment smoke banner still prints "Every gating constant in physics_labels/physics_curvature_probe is still UNSET" — left over from before the 09-05 freeze. Smoke mode reads no frozen constant, so this does not affect any result, but the wording is now inaccurate and worth cleaning up in a future plan that touches that banner. Not auto-fixed here: touching runner banner strings is outside Task 3's scope (a checkpoint task, not a code task) and the sealed-module discipline makes unscoped runner edits a Rule 4 (architectural/scope) question, not a Rule 1 bug fix.

## User Setup Required

None for this plan going forward — the one `user_setup` entry (choosing and bootstrapping the execution host) is exactly what Task 3 closed out. `HF_HOME` and `EFFDIM_09_OUTPUT_ROOT` are now set on the host's persistent disk (`/mnt/ssd-cluster/effdim/hf-cache`, `/mnt/ssd-cluster/effdim/phase9-out`); no further developer action is needed before 09-07 runs the real alignment proof there.

## Next Phase Readiness

- The execution host is proven: freeze gate holds, both smoke paths pass, environment recorded (128 cores, Python 3.14.7, torch 2.13.0+cpu, numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0, pyarrow 25.0.1, pandas 3.0.5, datasets 5.0.1).
- The cost model's calibrated estimate (per `--print-cost-model --threads 16` on the real host): sweep total 29.705 core-hours, ~1.86h wall-clock at 16 threads across all four `d` values — this is still Phase 7's *scaled estimate*, not a measured Phase 9 figure; 09-08 replaces it with the real Wave A timing.
- No Physics number exists yet: `09_row_alignment.jsonl` and `09_physics_curvature.jsonl` are absent both locally and in the returned bundle, as required.
- 09-07 is unblocked to run the real alignment proof (`--mode proof`) on this same host, following `09-EXECUTION-HOST.md` §4 step 1, using the same environment overrides already exported there.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Completed: 2026-09-04*

## Self-Check: PASSED

- FOUND: `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-06-SUMMARY.md`
- FOUND: `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-EXECUTION-HOST.md`
- FOUND: `notebooks/.cache/environment.json` (extracted from the returned archive)
- FOUND commit `282e27f` (Task 1)
- FOUND commit `ee992ba` (Task 2)
- FOUND commit `7fa8128` (Task 3)
- FOUND commit `4d3519f` (Task 3 fix-up, 40-hex-string invariant)
