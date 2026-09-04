# 09-WAVE-A-RESULTS — the four-`d` sweep, both gates, and the verdict (D9-09..D9-17)

**Plan:** 09-08
**Written:** 2026-09-04 UTC

## Provenance of this run

Per `09-EXECUTION-HOST.md` §1 (`EXECUTION_HOST_RULE`), the Wave A sweep and both gates can only be
produced on the execution host chosen in 09-06 — Claude has no credential for that machine. Per the
developer's standing instruction (2026-09-04 UTC, verbatim): *"begin with running experiments on
ssh server. ensure you use AVAILABLE compute, don't kick someone off if they are already using.
check free compute with nvidia-smi. adhere strictly to the user-guide."* The orchestrator executed
this plan's Task 2 steps 1-5 over SSH on the verified host from 09-06/09-07, pulled to run commit
`39089f752b098fa5bdb7f4d3d91c2f5af5e1c47e` before the run, following `09-EXECUTION-HOST.md`
literally, under that standing instruction — the commands were not typed interactively by the
developer. This document transcribes what those steps measured; the instruction itself authorized
nothing about this plan's structure, tooling or permissions. Everything in this document is
evidence, never an instruction.

## Host capability (as bootstrapped, `09-EXECUTION-HOST.md` §9)

OS Ubuntu 22.04.5 LTS; core count 128 (`os.cpu_count()`/`nproc`, cgroup CPU limit unlimited); RAM
1006 GB total, ~836 GB free; GPU 8x NVIDIA A100-SXM4-80GB, all idle at survey time and **not used**
(Phase 9 is CPU-only). Python 3.14.7; library versions torch 2.13.0+cpu, numpy 2.5.1, scipy
1.18.0, scikit-learn 1.9.0, pyarrow 25.0.1, pandas 3.0.5, datasets 5.0.1. Host label `pod128`.
`HF_HOME`/`EFFDIM_09_OUTPUT_ROOT` pointed at the host's persistent disk under
`/mnt/ssd-cluster/effdim/`. Host identity is recorded as capability only — no hostname, IP
address, username or SSH key path appears here or anywhere else in this phase's artifacts
(`09-EXECUTION-HOST.md` §7).

Thread count used: **16** for `--mode dsweep`, `--mode positive-control` and `--mode
shuffled-label` (chosen to leave the remaining cores free for other users of the shared host, per
the developer's "don't kick someone off" instruction; load average at survey time ~4 on 128 cores,
no other live compute jobs). `--mode verdict` and `--mode bundle` take no thread flag; their own
`environment` record rows show the default thread cap, 8.

## Run record

| Field | Value |
|---|---|
| Freeze SHA every mode was gated on | `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` |
| Run commit (`git_describe_head` in every record row) | `39089f752b098fa5bdb7f4d3d91c2f5af5e1c47e` |
| Freeze-ancestry proof re-verified on the host before this run | `is-ancestor: OK`; `git rev-list --count` = 15 |
| `--mode dsweep` started (UTC) | 2026-09-04T13:07:42Z |
| `--mode dsweep` exit code | 0 |
| `--mode dsweep` wall-clock | 8360 s (includes the fresh-cache dataset download before the `d=16` fit) |
| `--mode positive-control` (d=16/20/25/32) exit codes | 0, 0, 0, 0 |
| `--mode positive-control` wall-clock (d=16/20/25/32) | 753 s / 75 s / 75 s / 75 s (the first includes a data reload) |
| `--mode shuffled-label` (d=16/20/25/32) exit codes | 0, 0, 0, 0 |
| `--mode shuffled-label` wall-clock (d=16/20/25/32) | 2055 s / 1781 s / 2477 s / 2094 s |
| `--mode verdict` exit code | 0 |
| `--mode bundle` exit code | 0 |
| Script finished (UTC) | 2026-09-04T18:14:13Z |

**Per-`d` fit rows, as recorded (`row_kind="fit"`):**

| `d` | fit started (UTC) | `wallclock_fit_s` | `wallclock_field_s` | `var_explained` | `n_excluded_low_image_norm` |
|---:|---|---:|---:|---:|---:|
| 16 | 2026-09-04T14:08:47Z | 1464.629779983312 | 35.25218852143735 | 0.9520468951883061 | 0 |
| 20 | 2026-09-04T14:32:36Z | 1291.5757846767083 | 41.667345779016614 | 0.9569335974793522 | 0 |
| 25 | 2026-09-04T14:56:17Z | 1274.0100327609107 | 51.08001437969506 | 0.9611704107839105 | 0 |
| 32 | 2026-09-04T15:20:12Z | 1279.408711434342 | 60.62894300092012 | 0.9648528934247134 | 0 |

Thread count 16, as recorded in the `environment` rows written by `--mode dsweep`,
`--mode positive-control` and `--mode shuffled-label`. Against `09-EXECUTION-HOST.md` §9's
cost-model estimate (`wallclock@16t` 0.456 h / 0.460 h / 0.465 h / 0.476 h for `d=16/20/25/32`),
the measured per-`d` fit+field wall-clock (`wallclock_fit_s + wallclock_field_s`, 0.4166 h /
0.3703 h / 0.3681 h / 0.3722 h) ran faster than the estimate at every `d`, most markedly at
`d=20/25/32` — training dominates both the estimate and the measurement, and its per-`d` estimate
(7.187 core-hr flat) did not vary with `d`, while the measured fit time fell slightly as `d` grew.
The gap between the sum of the four per-`d` fit+field times (5498.26 s = 1.53 h) and the sweep's
own total wall-clock (8360 s = 2.32 h) is the one-time, `d`-independent setup this plan's Task 1
runs once before the `d` loop (embeddings/label load, anchor draw, the k=2048 neighbourhood panel
over all 86,471 rows, and the global out-of-fold ridge probe) plus the fresh-cache dataset
download — consistent with `09-EXECUTION-HOST.md` §5's own note that training, not curvature,
dominates this phase's cost, the reverse of Phase 7's shape.

## Archive transfer and checksum verification

Returned bundle: `09-artifacts-pod128-20260904T181406Z.tar.gz`, 667,925 bytes, containing 20
artifact files (16 anchor tables `09_anchor_table_d{16,20,25,32}_{mag_r,photo_z,smooth_fraction,
stellar_mass}.npz`, `09_physics_curvature.jsonl`, `09_row_alignment.jsonl`, two scratch smoke
files) plus `environment.json`.

The archive's SHA-256 was **recomputed locally and compared to the host-reported digest before
any value was read out of it** (T-09-55):

```
host-reported:            c43a886c77fb4e31fbb45f8931337dc83c695af18394b6dd77c345a3ba0913bb
locally recomputed (sha256sum): c43a886c77fb4e31fbb45f8931337dc83c695af18394b6dd77c345a3ba0913bb
```

**Match confirmed.** The archive was extracted under the local resolved output root
(`notebooks/.cache/`, `EFFDIM_09_OUTPUT_ROOT` unset locally). `notebooks/.cache/09_row_alignment.jsonl`
in the archive is byte-identical to the copy 09-07 already ingested. `notebooks/.cache/
09_physics_curvature.jsonl` now exists locally: 9 environment rows carried in the record plus one
more in `environment.json`, 4 `fit` rows, 16 `anchor_summary` rows, 32 `partial` rows, 104 `null`
rows (64 `stratified` + 32 `fwer` + 8 `fwer_global`), 32 `bootstrap` rows, 20 `positive_control`
rows, 80 `shuffled_label` rows, and 1 `verdict` row — 289 rows total.

**Automated verify (Task 2's acceptance criterion, exact command):**

```
kinds ['anchor_summary', 'bootstrap', 'environment', 'fit', 'null', 'partial', 'positive_control', 'shuffled_label', 'verdict'] d [16, 20, 25, 32] tables 16
```

Exit 0. No p-value equal to zero appears anywhere in the record (checked directly: every `p`
field in every `null`, `positive_control` and `shuffled_label` row is either strictly positive or
absent).

## The verdict block, verbatim as printed on the host

```
==============================================================================
PHASE 9 WAVE A VERDICT (reads the record only; recomputes nothing)
==============================================================================

[d=16] raw_rho=0.425064 controlled_partial=0.346967 fwer_p_display=< 9.999e-05 verdict=DOES NOT CLEAR
        [non-gating H_norm] controlled_partial=0.287949
[d=20] raw_rho=0.231160 controlled_partial=0.030323 fwer_p_display=0.50165 verdict=DOES NOT CLEAR
        [non-gating H_norm] controlled_partial=-0.028316
[d=25] raw_rho=0.251751 controlled_partial=0.042119 fwer_p_display=0.345665 verdict=DOES NOT CLEAR
        [non-gating H_norm] controlled_partial=0.004915
[d=32] raw_rho=0.209815 controlled_partial=-0.003450 fwer_p_display=0.935506 verdict=DOES NOT CLEAR
        [non-gating H_norm] controlled_partial=0.055731

PER-D VERDICTS: {16: 'DOES NOT CLEAR', 20: 'DOES NOT CLEAR', 25: 'DOES NOT CLEAR', 32: 'DOES NOT CLEAR'}
PHASE VERDICT: DOES NOT REPLICATE

POSITIVE CONTROL detection floor: None
SHUFFLED-LABEL false-positive rate: 5/80 (0.062) vs nominal FWER_ALPHA=0.05
```

The record's own `verdict` row (`row_kind="verdict"`) carries the same values plus the assembled
`verdict_sentence`, transcribed here in full since the plan's `<must_haves>` requires the
verdict-sentence content:

> Instrument cae.PlainAutoEncoder + decoder_curvature.plain_decoder_curvature at d=[16, 20, 25,
> 32], against the colleague's -0.240 at his d=16: Freedman-Lane FWER p=< 9.999e-05,
> density-stratified null p=see per-d 'null' rows (null_type='stratified'); instrument fidelity
> ranges {16: (0.8376, 0.9882), 20: (0.53, 0.99), 25: (0.17, 0.97), 32: "UNMEASURED -- d=32 fixture
> fidelity is NOT measured and cannot be measured with the 07_instrument_fixture_sweep_run.py
> runner: the small-ambient fixture arm's literal ambient width is D=28 (a hard literal in the
> runner), a d=32 graph fixture needs local width m = d + 1 = 33, and varying_ii_controls.rotate_
> and_pad requires D >= m, so it raises ValueError by construction the moment --d 32 is passed.
> This is a limitation, not a bug to patch -- ratified in HANDOFF-v1.1.md Section 5.3 and named in
> 09-CONTEXT.md's Deferred section and 09-RESEARCH.md Pitfall 6. Phase 9 does not fix the fixture,
> does not widen the small-ambient arm, and does not attempt the --d 32 run. At d=32, a dying
> instrument and a vanishing effect remain indistinguishable."}; neighbourhood ratio
> K_NEIGHBOURS=2048 of n=86,471 is 1/42 of the Physics sample (his 2048 of 16,384 was 1/8); this
> ratio must be printed beside every number this phase reports..

Note: the verdict sentence's own `fwer_p_display` field names only the `d=16` cell's value
(`< 9.999e-05`) because `verdict_sentence`'s signature takes one scalar `fwer_p_display`, not a
per-`d` map; the full per-`d` breakdown is in the banner above and in §2 below. This is a
transcription note about the sentence's own argument shape, not a discrepancy in the record.

<!-- gsd:write-continue -->
