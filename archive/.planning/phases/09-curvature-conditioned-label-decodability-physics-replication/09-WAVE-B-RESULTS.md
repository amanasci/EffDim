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

## 1. The frozen rule, quoted

Every value below is the verbatim committed value in
`notebooks/pu_manifold/physics_curvature_probe.py` at freeze commit
`5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` (`09-PREREGISTRATION.md`):

```
SEED_HANDLING_RULE = "no_pooling_per_seed_verdicts"

TORCH_INIT_SEEDS_WAVE_B = (0, 1, 2)

SEED_VERDICT_COMBINATION_RULE = (
    "Wave B runs three torch init seeds (TORCH_INIT_SEEDS_WAVE_B) at every d where the Wave A "
    "per-d verdict fired. combine_seed_verdicts requires exactly three entries; unanimity across "
    "all three gives the shared per-d verdict, anything else gives 'SPLIT ACROSS SEEDS' -- never "
    "averaged and never upgraded by majority vote, per 05-03-DECISION.md's one-way ratification."
)

WAVE_B_TRIGGER_RULE = (
    "Wave B (the three-seed sweep, TORCH_INIT_SEEDS_WAVE_B) runs only at d values where the "
    "Wave A (single TORCH_INIT_SEED = 0) per-d verdict fired (PER_D_VERDICT_VALUES[0]); d values "
    "where Wave A did not fire are never re-run under Wave B."
)
```

`combine_seed_verdicts(seed_verdicts)` (the frozen function these rules govern, verbatim behavior):
raises `ValueError` unless given exactly three entries; returns the shared value when all three
agree (`len(set(verdicts)) == 1`); returns the literal string `"SPLIT ACROSS SEEDS"` on any other
split (1-of-3 or 2-of-3), never an average, never an upgrade. This is the same function pinned by
Task 1's `test_seed_cell_verdict_never_upgrades_a_split`.

Seeds are never pooled, and anything short of 3-of-3 unanimity is terminal — this inherits
`05-03-DECISION.md`'s one-way ratification (Phase 5, D5-04 superseded) that pooling seed statistics
into a single headline number is not a permitted combination anywhere downstream of that decision.
`SEED_VERDICT_COMBINATION_RULE` names `05-03-DECISION.md` directly in its own committed text, quoted
above.

## 2. Scope

**`WAVE_B_NOT_TRIGGERED`.** Applying `WAVE_B_TRIGGER_RULE` literally: Wave B runs only at `d`
values where Wave A's per-`d` verdict fired (`PER_D_VERDICT_VALUES[0]`, `"NEGATIVE AND CLEARS FWER
NULL"`). `09-WAVE-A-RESULTS.md` §8 applied `per_d_verdict` to all four `D_SWEEP` cells
(`d = 16, 20, 25, 32`) and every one returned `"DOES NOT CLEAR"` — `d=16` has the strongest
statistical signal in the sweep (`p_fwer < 9.999e-05`) but the wrong sign (`rho = +0.346967`, not
`< 0`); `d=32` has the correct sign (`rho = -0.003450`) but the weakest, furthest-from-clearing
`p_fwer` (`0.935506`); `d=20` and `d=25` clear neither condition. Zero of four cells fired.

No `d` value entered Wave B's scope. This host run (§ Run record above) independently confirms the
same conclusion from the runner's own live read of the Wave A record, not merely from re-reading
`09-WAVE-A-RESULTS.md`'s prose: `_triggered_d_values(record_path)` read the `verdict` record row
Wave A wrote, found an empty triggered list, and the mode's own empty-scope branch executed,
printed its message, and recorded `WAVE_B_NOT_TRIGGERED` as a terminal, complete outcome — exactly
the outcome the plan's own precondition anticipated.

**Seed stability has nothing to test at any `d`, and no seed agreement is claimed anywhere in this
document.** A reader must not mistake the record's `WAVE_B_NOT_TRIGGERED` row, or this document's
silence on any `d`'s three-seed table, for "seeds were run and happened to agree" — they were never
run, because no cell survived Wave A to be re-tested.

## 3. Per-`d` seed table

Not applicable. No `d` triggered Wave B, so no seed was fit, no controlled partial was recomputed
per seed, no Freedman-Lane null was drawn per seed, and no bootstrap band was computed per seed.
There is no table to report — an empty table would misstate the record as having three rows per
`d` when it has none.

## 4. Field disagreement diagnostic

Not applicable, for the same reason as §3: the pairwise Spearman between seeds' `H_tan_norm`
arrays at the anchors is only defined where three seed fields exist to compare, and none was
computed. No diagnostic value is reported here as a stand-in zero or placeholder — its absence is
the correct record of an untriggered wave, not a missing measurement.

## 5. What a split means, and what unanimity means

Stated for completeness even though this run has neither outcome to report: under
`SEED_VERDICT_COMBINATION_RULE`, a **split cell** (any 1-of-3 or 2-of-3 disagreement among
`TORCH_INIT_SEEDS_WAVE_B`) is a terminal non-supportive outcome — `combine_seed_verdicts` returns
`"SPLIT ACROSS SEEDS"` literally and that value is never averaged, never rounded up to a weaker
positive, and never treated as "leaning cleared." A **unanimous cell** is three independent fits
agreeing on sign and FWER clearance at one `d` — a statement about that `d` alone, not about the
sweep as a whole, and not strengthened by how many other cells did or did not agree. Neither
outcome occurs in this run: with zero cells triggered, `combine_seed_verdicts` was never called,
and the record's own `seed_cell_verdict` row carries the literal value `"WAVE_B_NOT_TRIGGERED"`,
which is neither of the two values `combine_seed_verdicts` can return — it is the mode's own
explicit third outcome for an empty scope, kept textually distinct from both `"SPLIT ACROSS
SEEDS"` and any per-`d` verdict string so a reader parsing the record cannot mistake "nothing to
run" for either a split or an agreement.

## 6. Carried into the phase verdict

The per-`d` cell verdicts 09-10 will read, in `D_SWEEP` order — unchanged from `09-WAVE-A-RESULTS.md`
§8, since Wave B did not run at any `d` and therefore altered nothing:

| `d` | Wave A per-`d` verdict | Wave B outcome |
|---:|---|---|
| 16 | `DOES NOT CLEAR` | not triggered — Wave A cell stands |
| 20 | `DOES NOT CLEAR` | not triggered — Wave A cell stands |
| 25 | `DOES NOT CLEAR` | not triggered — Wave A cell stands |
| 32 | `DOES NOT CLEAR` | not triggered — Wave A cell stands |

No phase verdict is written in this document. `09-08-SUMMARY.md`'s own "Next Phase Readiness"
section already states that 09-10 can write the phase verdict directly from Wave A's per-`d` cells
(`phase_verdict = "DOES NOT REPLICATE"` per `VERDICT_RULE`, since none of the four cells fired),
and this plan's own `<discretion_decisions>` makes that finalization 09-10's act, not this
document's.

