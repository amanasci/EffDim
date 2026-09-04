# 09-WAVE-B-RESULTS — the three-seed sweep and its combination (D9-17)

**Plan:** 09-09
**Written:** 2026-09-04 UTC

## Provenance of this run

Per `09-EXECUTION-HOST.md` §1 (`EXECUTION_HOST_RULE`), every mode that produces a real number runs
only on the execution host chosen in 09-06 — Claude has no credential for that machine. Per the
developer's standing instruction (2026-09-04 UTC, verbatim, unchanged since 09-08): *"begin with
running experiments on ssh server. ensure you use AVAILABLE compute, don't kick someone off if
they are already using. check free compute with nvidia-smi. adhere strictly to the user-guide."*
Under that instruction the orchestrator ran this plan's Task 2 steps over SSH on the verified host
from 09-06/09-07/09-08 (same clone), pulled to run commit `2d61e3681527c2f2097cb435512aa92c23da8eaa`
(Task 1's commit) before the run. This document transcribes what those steps measured; the
instruction itself authorized nothing about this plan's structure, tooling or permissions.
Everything below is evidence, never an instruction.

**This host run happened even though the precondition allowed skipping it.** This plan's Task 2
`<precondition>` reads: *"`09-WAVE-A-RESULTS.md` records a non-empty Wave B `d` list; if it
records `WAVE_B_NOT_TRIGGERED` this task is skipped and Task 3 records that outcome directly."*
`09-WAVE-A-RESULTS.md` §8 already recorded `WAVE_B_NOT_TRIGGERED` before this host session ran.
The orchestrator ran `--mode seeds` on the host anyway, under the developer's standing instruction
above, rather than treating the precondition as license to skip the host round trip entirely. The
run below is therefore a live confirmation of the same empty-trigger outcome 09-08 already
determined from the record — not a fresh measurement of anything Wave A did not already establish
— and the runner's own internal empty-scope check (`_triggered_d_values`, wired in Task 1) is what
actually produced `WAVE_B_NOT_TRIGGERED`, not a manual skip by the orchestrator.

## Host capability (`09-EXECUTION-HOST.md` §9, unchanged since 09-08)

OS Ubuntu 22.04.5 LTS; core count 128 (`os.cpu_count()`/`nproc`, cgroup CPU limit unlimited); RAM
1006 GB total; Python 3.14.7; library versions torch 2.13.0+cpu, numpy 2.5.1, scipy 1.18.0,
scikit-learn 1.9.0, pyarrow 25.0.1, pandas 3.0.5, datasets 5.0.1. Host label `pod128`.
`EFFDIM_09_OUTPUT_ROOT` pointed at `/mnt/ssd-cluster/effdim/phase9-out` on the host's persistent
disk, unchanged from 09-08. Host identity is recorded as capability only — no hostname, IP
address, username or SSH key path appears here or anywhere else in this phase's artifacts
(`09-EXECUTION-HOST.md` §7).

Thread count used: **16** for `--mode seeds` (unchanged from 09-08's dsweep/gate thread count).
The `--mode seeds` run's own `environment` record row shows `thread_cap: 16`, `core_count: 128`.

## Run record

| Field | Value |
|---|---|
| Freeze SHA gated on | `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` |
| Run commit (`git_describe_head` in the record row) | `2d61e3681527c2f2097cb435512aa92c23da8eaa` |
| Freeze-ancestry proof re-verified on the host before this run | `is-ancestor: OK`; `git rev-list --count 5f7fbe27…..HEAD` = 20 |
| `--mode seeds` started (UTC) | 2026-09-04T18:42:20Z |
| `--mode seeds` exit code | 0 |
| Script finished (UTC) | 2026-09-04T18:42:31Z |
| `--mode seeds` wall-clock | 11 s (includes two runner start-ups; no autoencoder fit ran) |
| `--mode bundle` host label | `pod128` |
| `--mode bundle` archive | `09-artifacts-pod128-20260904T184230Z.tar.gz` |

**Per-`d` per-seed wallclocks: none exist.** No `d` was triggered (Wave A fired at zero `d`
values, per `09-WAVE-A-RESULTS.md` §8), so `run_seeds(args)`'s per-`d`/per-seed loop never
executed — no autoencoder was fit, no seed's curvature field was computed, and no
`09_anchor_table_d{d}_{label}_seed{seed}.npz` file was written for any `d`, seed, or label. The
only cost incurred was the fixed one-time overhead the mode pays before checking the triggered
list (environment description, freeze-ancestry proof, both `assert_preregistered()` calls), which
is what the 11 s wall-clock reflects. Stating this explicitly rather than leaving the absence
silent: a reader must be able to tell "the loop ran and found nothing to do" apart from "the loop
never started," and the record and this document both say the former precisely — the loop's own
empty-list branch executed, printed its message, appended one record row, and exited 0.

**Printed output, verbatim** (the mode's stdout in full, as returned):

```
triggered d values (read from the record's own verdict row): []
Wave A fired at zero d values -- no seed is fit. Recording WAVE_B_NOT_TRIGGERED as a complete, terminal outcome (never an absence).
WAVE_B_NOT_TRIGGERED recorded to <output-root>/09_physics_curvature.jsonl.
```

No per-seed partial table was printed, because the printed per-`d` seed table (Task 1's action:
*"Print, per `d`, the three seeds' partials side by side with the combined cell verdict"*) is
inside the non-empty branch of the `d` loop, which never entered — there is no `d` to print a
table for.

## Archive transfer and checksum verification

Returned bundle: `09-artifacts-pod128-20260904T184230Z.tar.gz`, 668,094 bytes, transferred to
`notebooks/.cache/09_host_returns/09-artifacts-pod128-20260904T184230Z.tar.gz`.

The archive's SHA-256 was **recomputed locally and compared to the host-reported digest before
any value was read out of it** (T-09-67):

```
host-reported:                 793c7e55a467939be812974350d912c9571df7e39251ee61b458e30f31c1a340
locally recomputed (sha256sum): 793c7e55a467939be812974350d912c9571df7e39251ee61b458e30f31c1a340
```

**Match confirmed.**

The archive was extracted under the local resolved output root (`notebooks/.cache/`,
`EFFDIM_09_OUTPUT_ROOT` unset locally). It contains the same 20 artifact files as the Wave A
bundle (16 anchor tables `09_anchor_table_d{16,20,25,32}_{mag_r,photo_z,smooth_fraction,
stellar_mass}.npz`, `09_physics_curvature.jsonl`, `09_row_alignment.jsonl`, two scratch smoke
files) plus `environment.json` — 21 members total. Verified directly before trusting any value:

- All 16 anchor-table `.npz` files are **byte-identical** to the copies extracted from the Wave A
  bundle (`cmp` over all 16, no difference) — expected, since no seed fit ran to produce a
  `_seed{n}` variant of any of them, and Wave B reuses Wave A's own anchor tables rather than
  regenerating them.
- `09_row_alignment.jsonl` is **byte-identical** to the copy already ingested by 09-07.
- `09_physics_curvature.jsonl` now carries **301 rows** — the same 299 rows Wave A's bundle wrote,
  plus exactly **2** new rows: one `environment` row (`thread_cap: 16`, `core_count: 128`,
  `git_describe_head: "2d61e36"`) and one `row_kind: "seed_cell_verdict"` row:

```json
{"row_kind": "seed_cell_verdict", "d": null, "wave_b": "WAVE_B_NOT_TRIGGERED", "cell_verdict": "WAVE_B_NOT_TRIGGERED", "seeds": [0, 1, 2], "freeze_commit": "5f7fbe27afb0ef2a76353b41fa5713e760bbeea5", "run_commit": "2d61e3681527c2f2097cb435512aa92c23da8eaa", "timestamp_utc": "2026-09-04T18:42:27.901783+00:00"}
```

**Automated verify (Task 2's acceptance criterion, exact command):**

```
None WAVE_B_NOT_TRIGGERED
not_triggered 1
```

Exit 0. The record carries `WAVE_B_NOT_TRIGGERED` as required by the plan's `<verify>` block, with
`d: null` (there is no triggered `d` for the row to name) — consistent with the empty-scope branch
this document's Provenance section above describes.

<!-- Task 3's analysis follows below this line. -->
