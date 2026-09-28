# 09-WAVE-A-RESULTS-AMENDMENT-01 — the four-`d` sweep and both gates re-run under the sphere-projected decoder image

**Amends:** `09-WAVE-A-RESULTS.md` (freeze `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5`)
**Under:** `09-PREREGISTRATION-AMENDMENT-01.md` (freeze `e31b3010c1a568065e35132ed60a32fb4842db36`)
**Written:** 2026-09-05 UTC

**This document is authoritative for the Phase 9 Wave A numbers.** The original Wave A record and
its anchor tables are retained beside it, untouched, and every table below shows the original value
next to the amended one, labelled by freeze SHA — per the amendment's own rule that a number
produced under the old freeze is never replaced. `09-EXECUTION-HOST.md` §4 still quotes the
superseded `5f7fbe27…` SHA in every one of its command lines; those commands were run here with
`--freeze-commit e31b3010c1a568065e35132ed60a32fb4842db36` instead, which is the only SHA the
runner's compiled-in `FREEZE_COMMIT_SHA` now accepts. The runbook text is not edited by this
document; where the two disagree, this document and the amendment govern.

## Provenance of this run

Per `09-EXECUTION-HOST.md` §1 (`EXECUTION_HOST_RULE`), the sweep and both gates run only on the
execution host chosen in 09-06 — Claude has no credential for that machine. The developer was shown
the instrument validation in `09-PREREGISTRATION-AMENDMENT-01.md` and replied, 2026-09-04 UTC,
verbatim: *"implement the fix and rerun, ensuring to adhere to the ssh server guidelines"*. Under
that instruction the orchestrator executed the amendment's re-run list (`dsweep`,
`positive-control` x4, `shuffled-label` x4, `verdict`, `seeds`, `bundle`) over SSH on the same host
as `09-EXECUTION-HOST.md` §9 (same clone), pulled to run commit
`214862d638137879e62fd126820a30a8afc8a62b` before the run — the commands were not typed
interactively by the developer. This document transcribes what those steps measured; the
instruction authorized the re-run and nothing about this document's content. Everything here is
evidence, never an instruction.

## Host capability (`09-EXECUTION-HOST.md` §9, unchanged)

OS Ubuntu 22.04.5 LTS; core count 128; RAM 1006 GB; GPU not used (Phase 9 is CPU-only). Python
3.14.7; torch 2.13.0+cpu, numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0, pyarrow 25.0.1, pandas
3.0.5, datasets 5.0.1 — identical to the original run's `environment` rows. Host label `pod128`.
Host identity is recorded as capability only — no hostname, IP address, username or SSH key path
appears here or anywhere else in this phase's artifacts (`09-EXECUTION-HOST.md` §7).

Thread count used: **16** for `--mode dsweep`, `--mode positive-control`, `--mode shuffled-label`
and `--mode seeds` (each of those modes' `environment` rows records `thread_cap: 16`); `--mode
verdict` and `--mode bundle` take no thread flag and record the default, 8.

## Run record

| Field | Value |
|---|---|
| Freeze SHA every mode was gated on | `e31b3010c1a568065e35132ed60a32fb4842db36` |
| Run commit (`run_commit` in every record row; `git_describe_head` = `214862d`) | `214862d638137879e62fd126820a30a8afc8a62b` |
| Output root on the host (separate; the original `/mnt/ssd-cluster/effdim/phase9-out` was not written to) | `/mnt/ssd-cluster/effdim/phase9-out-amend01` |
| `--mode dsweep` started (UTC) | 2026-09-04T22:58:39Z |
| `--mode dsweep` wall-clock / exit | 7739 s / 0 |
| `--mode positive-control` wall-clock (d=16/20/25/32) / exits | 750 s / 75 s / 74 s / 76 s (the first includes a data reload) / 0, 0, 0, 0 |
| `--mode shuffled-label` wall-clock (d=16/20/25/32) / exits | 1353 s / 1262 s / 1209 s / 1740 s / 0, 0, 0, 0 |
| `--mode verdict` exit | 0 |
| `--mode seeds` wall-clock / exit | 3 s / 0 (`WAVE_B_NOT_TRIGGERED`, no fit) |
| `--mode bundle` exit | 0 |
| Script finished (UTC) | 2026-09-05T02:58:01Z |

The step wall-clocks are consistent with the record's own timestamps: the sweep start plus the
sum of every step's wall-clock (14,281 s) lands at 2026-09-05T02:56:40Z, which is the `verdict`
row's own `timestamp_utc` to the second.

**Per-`d` fit rows, as recorded (`row_kind="fit"`), with the original run's values beside them:**

| `d` | fit row written (UTC) | `wallclock_fit_s` | `wallclock_field_s` | `var_explained` (amended) | `var_explained` (original) | `n_excluded_low_image_norm` |
|---:|---|---:|---:|---:|---:|---:|
| 16 | 2026-09-04T23:48:26Z | 1358.5354426503181 | 84.0060995304957 | 0.9520467501612097 | 0.9520468951883061 | 0 |
| 20 | 2026-09-05T00:11:40Z | 1197.0366138713434 | 101.82308888435364 | 0.9569335974793522 | 0.9569335974793522 | 0 |
| 25 | 2026-09-05T00:35:52Z | 1233.3839241219684 | 123.98108037002385 | 0.9611704107839105 | 0.9611704107839105 | 0 |
| 32 | 2026-09-05T01:00:51Z | 1246.0125791113824 | 158.11178336292505 | 0.9648528934247134 | 0.9648528934247134 | 0 |

`var_explained` agrees with the original run to four decimals at every `d` (0.9520 / 0.9569 /
0.9612 / 0.9649; bit-identical at `d=20/25/32`, a difference of 1.5e-7 at `d=16`). The autoencoder
fits reproduced; what the amendment changed is the curvature evaluation only, exactly as
`09-PREREGISTRATION-AMENDMENT-01.md` § 2 describes. `wallclock_field_s` roughly doubled at every
`d` relative to the original (35 / 42 / 51 / 61 s there) — the projection wrapper differentiates
`F/||F||` rather than `F`.

## Archive transfer and checksum verification

Returned bundle: `09-artifacts-pod128-20260905T025646Z.tar.gz`, 557,580 bytes, containing 16
anchor tables `09_anchor_table_d{16,20,25,32}_{mag_r,photo_z,smooth_fraction,stellar_mass}.npz`,
`09_physics_curvature.jsonl` and `environment.json` (18 files; no scratch or alignment file — the
alignment proof was not re-run, per the amendment).

SHA-256, host-reported and recomputed locally over the received file before any value was read
(T-09-55):

```
host-reported:                  1db27632adbe89ad2303c9c9230f0b776bba19f781b9f464e03c7d6e8f1e2271
locally recomputed (sha256sum): 1db27632adbe89ad2303c9c9230f0b776bba19f781b9f464e03c7d6e8f1e2271
```

**Match confirmed.** Extracted under a separate local root, `notebooks/.cache/09-amend01/`
(gitignored), so the original `notebooks/.cache/09_physics_curvature.jsonl`, its 16 anchor tables
and the archived Wave A / Wave B bundles stay byte-identical. The amended record holds 301 rows:
11 `environment`, 4 `fit`, 16 `anchor_summary`, 32 `partial`, 104 `null` (64 `stratified` + 32
`fwer` + 8 `fwer_global`), 32 `bootstrap`, 20 `positive_control`, 80 `shuffled_label`, 1
`verdict`, 1 `seed_cell_verdict` — the same shape as the original record after its own `seeds`
row. Every row carries `freeze_commit = e31b3010…`; every `fit` row carries
`decoder_image_projection = "sphere"`. No `p` field anywhere in the record equals zero.

## The verdict block, as carried by the amended record

Assembled from the amended record's `partial`, `null` (`fwer`) and `verdict` rows in the shape the
host banner prints (`--mode verdict` reads the record only and recomputes nothing):

```
==============================================================================
PHASE 9 WAVE A VERDICT (reads the record only; recomputes nothing)
==============================================================================

[d=16] raw_rho=0.413125 controlled_partial=0.328059 fwer_p_display=< 9.999e-05 verdict=DOES NOT CLEAR
        [non-gating H_norm] controlled_partial=0.328059
[d=20] raw_rho=0.233980 controlled_partial=0.016445 fwer_p_display=0.720128 verdict=DOES NOT CLEAR
        [non-gating H_norm] controlled_partial=0.016445
[d=25] raw_rho=0.255301 controlled_partial=0.030566 fwer_p_display=0.49585 verdict=DOES NOT CLEAR
        [non-gating H_norm] controlled_partial=0.030566
[d=32] raw_rho=0.220979 controlled_partial=-0.014720 fwer_p_display=0.736126 verdict=DOES NOT CLEAR
        [non-gating H_norm] controlled_partial=-0.014720

PER-D VERDICTS: {16: 'DOES NOT CLEAR', 20: 'DOES NOT CLEAR', 25: 'DOES NOT CLEAR', 32: 'DOES NOT CLEAR'}
PHASE VERDICT: DOES NOT REPLICATE

POSITIVE CONTROL detection floor: None
SHUFFLED-LABEL false-positive rate: 5/80 (0.062) vs nominal FWER_ALPHA=0.05
```

The `verdict` row's own fields, verbatim: `phase_verdict = "DOES NOT REPLICATE"`,
`per_d_verdicts = {"16": "DOES NOT CLEAR", "20": "DOES NOT CLEAR", "25": "DOES NOT CLEAR", "32":
"DOES NOT CLEAR"}`, `positive_control_detection_floor = null`,
`shuffled_label_false_positive_count = 5`, `shuffled_label_repeats = 80`. Its `verdict_sentence`
is character-for-character the sentence transcribed in `09-WAVE-A-RESULTS.md` ("Instrument
cae.PlainAutoEncoder + decoder_curvature.plain_decoder_curvature at d=[16, 20, 25, 32], against
the colleague's -0.240 at his d=16: Freedman-Lane FWER p=< 9.999e-05, … this ratio must be printed
beside every number this phase reports.."), since the sentence's arguments (instrument name,
`D_SWEEP`, the `d=16` FWER display, fidelity ranges, neighbourhood ratio) took the same values.

**The `H_norm` line equals the `H_tan_norm` line at every `d`.** Under the projection `H_rad` is
the constant `-d` at every anchor, so `H_norm = sqrt(H_tan_norm^2 + d^2)` is a monotone function
of `H_tan_norm` and the two fields have identical ranks (Spearman 1.000000 between them in every
amended anchor table). The non-gating `H_norm` comparison of `09-WAVE-A-RESULTS.md` § 4 therefore
has no content under this freeze; it is carried in the record for shape only.

## 1. Fit check — `H_rad = -d` under the projection

| `d` | `var_explained` | median `cond(g)` (amended) | median `cond(g)` (original) | `H_rad_median` | `H_rad_expected` | `H_rad_max_abs_dev` | `decoder_image_projection` | `H_rad` median, original run |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 16 | 0.9520467501612097 | 14.775124210562453 | 14.726070366715556 | -16.000000 | -16.0 | 2.49e-14 | `sphere` | -20.391490489648845 (-27.45%) |
| 20 | 0.9569335974793522 | 13.725461873363024 | 13.749418462007677 | -20.000000 | -20.0 | 3.55e-14 | `sphere` | -24.244205345752775 (-21.22%) |
| 25 | 0.9611704107839105 | 11.997368865987017 | 12.012144761657932 | -25.000000 | -25.0 | 3.55e-14 | `sphere` | -29.613265737402585 (-18.45%) |
| 32 | 0.9648528934247134 | 10.542289441198081 | 10.537474284699378 | -32.000000 | -32.0 | 4.97e-14 | `sphere` | -36.82735975422215 (-15.09%) |

The `H_rad` backstop that missed `-d` by 15-27% at every `d` in the original run
(`09-WAVE-A-RESULTS.md` § 3) now holds identically: the maximum over the 512 anchors of
`|H_rad + d|` is at floating-point round-off (2.5e-14 to 5.0e-14), re-measured from the returned
anchor tables as well as read from the `fit` rows. `09-08-PLAN.md`'s must-have (`H_rad` within
10% of `-d`) is satisfied at every `d`. `cond(g)` moves by less than 0.4% at every `d` and keeps
its monotone fall with `d`; `n_excluded_low_image_norm = 0` and `n_masked_anchors = 0` for every
`d` and label, as before.

## 2. Per-`d`, per-label partials — original beside amended, `H_tan_norm`

Full precision from the `partial` rows; `p` columns from the `null` rows (`fwer` = the cell's own
Freedman-Lane `p_display`; `S=10` / `S=20` = the density-stratified null's `p_display`);
`n_finite_anchors = 512` for every cell in both runs. "Original" is
`notebooks/.cache/09_physics_curvature.jsonl` under `5f7fbe27…`; "amended" is
`notebooks/.cache/09-amend01/09_physics_curvature.jsonl` under `e31b3010…`. Δ is amended minus
original controlled partial.

### `mag_r` (gating)

| `d` | raw `rho` orig | raw `rho` amend | controlled orig | controlled amend | Δ | FWER `p` orig | FWER `p` amend | strat `p` S=10/S=20 orig | strat `p` S=10/S=20 amend | bootstrap 95% amend |
|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| 16 | 0.425064 | 0.413125 | **0.346967** | **0.328059** | -0.0189 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | `< 2.000e-04` / `< 2.000e-04` | [0.246800, 0.412430] |
| 20 | 0.231160 | 0.233980 | **0.030323** | **0.016445** | -0.0139 | 0.501650 | 0.720128 | 0.479304 / 0.473905 | 0.700660 / 0.696261 | [-0.082609, 0.117653] |
| 25 | 0.251751 | 0.255301 | **0.042119** | **0.030566** | -0.0116 | 0.345665 | 0.495850 | 0.329334 / 0.333933 | 0.478104 / 0.493501 | [-0.067746, 0.132549] |
| 32 | 0.209815 | 0.220979 | **-0.003450** | **-0.014720** | -0.0113 | 0.935506 | 0.736126 | 0.931014 / 0.940812 | 0.731454 / 0.736253 | [-0.111129, 0.086985] |

The family-wise `fwer_global` envelope is `< 9.999e-05` for every label and field in the amended
record, as it was in the original; it is dominated by `d=16`'s own value and is not evidence for
any other cell (`09-WAVE-A-RESULTS.md` § 2).

### `photo_z` (non-gating)

| `d` | raw `rho` orig | raw `rho` amend | controlled orig | controlled amend | Δ | FWER `p` orig | FWER `p` amend | strat `p` S=10/S=20 amend | bootstrap 95% amend |
|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 16 | 0.172874 | 0.147277 | 0.366797 | 0.357525 | -0.0093 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.286157, 0.432685] |
| 20 | 0.149048 | 0.120047 | 0.314020 | 0.309166 | -0.0049 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.223295, 0.391939] |
| 25 | 0.155961 | 0.120305 | 0.377687 | 0.372807 | -0.0049 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.294805, 0.449459] |
| 32 | 0.197428 | 0.151229 | 0.417527 | 0.415289 | -0.0022 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.339882, 0.487944] |

### `smooth_fraction` (non-gating)

| `d` | raw `rho` orig | raw `rho` amend | controlled orig | controlled amend | Δ | FWER `p` orig | FWER `p` amend | strat `p` S=10/S=20 amend | bootstrap 95% amend |
|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 16 | 0.116237 | 0.096748 | 0.348011 | 0.340705 | -0.0073 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.254409, 0.420178] |
| 20 | 0.305862 | 0.271725 | 0.323578 | 0.326109 | +0.0025 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.240184, 0.408279] |
| 25 | 0.214073 | 0.172094 | 0.352684 | 0.343308 | -0.0094 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.259777, 0.426751] |
| 32 | 0.220063 | 0.173807 | 0.412762 | 0.417516 | +0.0048 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.343267, 0.491244] |

### `stellar_mass` (non-gating; `[RATIFIED 2026-09-03]` mapping)

| `d` | raw `rho` orig | raw `rho` amend | controlled orig | controlled amend | Δ | FWER `p` orig | FWER `p` amend | strat `p` S=10/S=20 orig | strat `p` S=10/S=20 amend | bootstrap 95% amend |
|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| 16 | 0.187777 | 0.185682 | 0.073530 | 0.070356 | -0.0032 | 0.099490 | 0.113889 | 0.087383 / 0.099180 | 0.107778 / 0.116377 | [-0.021077, 0.157619] |
| 20 | -0.027635 | -0.016773 | 0.131945 | 0.124384 | -0.0076 | 0.001999 | 0.004200 | 0.004199 / 0.003799 | 0.009398 / 0.010398 | [0.036209, 0.209211] |
| 25 | 0.104099 | 0.103895 | 0.227331 | 0.221580 | -0.0058 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | `< 2.000e-04` / `< 2.000e-04` | [0.131310, 0.303164] |
| 32 | 0.104656 | 0.114678 | 0.263450 | 0.260221 | -0.0032 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | `< 2.000e-04` / `< 2.000e-04` | [0.177510, 0.340686] |

Across all sixteen `H_tan_norm` cells the controlled partial moves by at most 0.019 in absolute
value (the `d=16` `mag_r` cell), no cell changes sign, and no cell crosses `FWER_ALPHA` in either
direction: the four cells that cleared FWER per label in the original (`mag_r` at `d=16`; `photo_z`
and `smooth_fraction` at every `d`; `stellar_mass` at `d=20/25/32`) clear it in the amended run,
and the cells that did not clear still do not.

## 3. Anchor-level agreement between the two fields

Recomputed locally with `scipy.stats.spearmanr` from the two sets of `mag_r` anchor tables (the
`anchor_idx` arrays are identical in the two runs at every `d`, as are the `r2` and
`log_knn_radius` columns — the probe and neighbourhood panel did not change):

| `d` | Spearman(`H_tan_norm` original, `H_tan_norm` amended) | Spearman(`H_norm` original, `H_tan_norm` amended) | Spearman(`H_norm` original, `H_tan_norm` original) |
|---:|---:|---:|---:|
| 16 | 0.997109 | 0.966149 | 0.970564 |
| 20 | 0.991896 | 0.952690 | 0.962781 |
| 25 | 0.992304 | 0.958365 | 0.960400 |
| 32 | 0.984160 | 0.949992 | 0.946985 |

The projected field is, at the anchor level, a near-reordering of the original `H_tan_norm` field
(`rho` 0.984-0.997, decreasing with `d`); the original `H_norm` field sits further from the
projected field (0.950-0.966) than the original `H_tan_norm` does.

`rho(H_tan_norm, log_knn_radius)` from the amended tables, beside `09-WAVE-A-RESULTS.md` § 6's
original column: `d=16` -0.599316 (original -0.561698), `d=20` -0.501870 (-0.436420), `d=25`
-0.619006 (-0.555967), `d=32` -0.596603 (-0.502702). Negative at every `d` under both freezes; his
own value at his `d=16` is `+0.765`.

## 4. The positive control

| `d` | target magnitude | target `rho` (planted) | achieved controlled partial (amended) | slope | cleared? | Freedman-Lane `p` | achieved, original run |
|---:|---:|---:|---:|---:|---|---|---:|
| 16 | 0.05/0.10/0.15/0.20/0.25 (all five collapse to one achieved value) | -0.05…-0.25 | 0.004545 | 1.8189894035458565e-12 (~0) | false (all five) | 0.919408 | 0.011552 |
| 20 | 0.05/0.10/0.15/0.20/0.25 | -0.05…-0.25 | 0.036505 | 2.0 (bracket ceiling) | false (all five) | 0.417858 | 0.053438 |
| 25 | 0.05/0.10/0.15/0.20/0.25 | -0.05…-0.25 | 0.024345 | 2.0 (bracket ceiling) | false (all five) | 0.588941 | 0.030884 |
| 32 | 0.05/0.10/0.15/0.20/0.25 | -0.05…-0.25 | -0.010313 | 2.0 (bracket ceiling) | false (all five) | 0.815818 | 0.001134 |

**Detection floor: `None` — no target of `POSITIVE_CONTROL_TARGET_RHOS` cleared at any `d`,
exactly as in the original run.** The slopes are the same two values as before — `~0` at `d=16`
(the direction test collapsing against the strongly positive real relation) and the bracket
ceiling `2.0` at `d=20/25/32` (the achievable statistic bounded by the near-zero real partial) —
so the structural mechanism recorded in `09-WAVE-A-RESULTS.md` § 5 and put to the developer in
`09-08-SUMMARY.md` is unchanged by this amendment. The amendment touched no positive-control
constant or plant mechanism (`09-PREREGISTRATION-AMENDMENT-01.md`, "What this amendment does not
change"), and the gate question remains open. On the frozen `09-EXECUTION-HOST.md` §8-literal
reading, the gate FAILED here as it did before.

## 5. The shuffled-label calibration

| `d` | repeats | false positives (`cleared=true`) (amended) | rate | false positives, original | nominal `FWER_ALPHA` |
|---:|---:|---:|---:|---:|---:|
| 16 | 20 | 1 | 0.05 | 1 | 0.05 |
| 20 | 20 | 1 | 0.05 | 1 | 0.05 |
| 25 | 20 | 2 | 0.10 | 2 | 0.05 |
| 32 | 20 | 1 | 0.05 | 1 | 0.05 |
| **pooled** | **80** | **5** | **0.0625** | **5** | **0.05** |

The same per-`d` counts and the same pooled 5/80 as the original run (the shuffle seeds are
frozen; the field they are tested against is the amended one). As before, at `n=20` per `d` a
single extra false positive moves the per-`d` rate by 5 points, so `d=25`'s 0.10 is reported as
measured, not adjusted.

## 6. Wave B

`--mode seeds` under the amended freeze read the amended record's own `verdict` row, found the
triggered `d` list empty (every per-`d` verdict `DOES NOT CLEAR`), fit no autoencoder and appended
one `seed_cell_verdict` row: `wave_b = "WAVE_B_NOT_TRIGGERED"`, `cell_verdict =
"WAVE_B_NOT_TRIGGERED"`, `seeds = [0, 1, 2]`, `d = null`. Wall-clock 3 s. Same terminal outcome
as `09-WAVE-B-RESULTS.md` records under the original freeze.

## 7. What the amendment established

Stated plainly, from the record and the tables above:

- **The sphere projection makes `H_rad = -d` exactly.** `H_rad_max_abs_dev` is 2.5e-14 to 5.0e-14
  over the 512 anchors at every `d`; the backstop that failed by 15-27% under the original freeze
  is now satisfied identically. The autoencoder fits themselves reproduced (`var_explained` equal
  to four decimals at every `d`), so the change is confined to the curvature evaluation.
- **The field ordering moved little.** Spearman between the original and the projected
  `H_tan_norm` is 0.997 / 0.992 / 0.992 / 0.984 at `d=16/20/25/32`.
- **Every partial moved by at most a few hundredths.** The largest change in any `H_tan_norm`
  controlled partial across sixteen `d`-label cells is -0.019 (`mag_r`, `d=16`: 0.346967 to
  0.328059). No sign changed. No cell moved across `FWER_ALPHA`.
- **The verdict is unchanged.** `DOES NOT REPLICATE`; `DOES NOT CLEAR` at every `d`; positive
  control detection floor `None`; shuffled-label 5/80; `WAVE_B_NOT_TRIGGERED`.

Therefore the positive `d=16` association under the autoencoder instrument (`+0.328`, FWER
`< 9.999e-05`, bootstrap [0.247, 0.412]) is not an artefact of the off-sphere decoder image the
amendment was raised to correct: with the image constrained to the sphere the data occupy, the
same cell gives the same sign, the same FWER clearance and a magnitude 0.019 lower. What the
amendment did resolve is the instrument-side ambiguity `09-WAVE-A-RESULTS.md` § 3 left open — a
poor-fit-vs-real-geometry question that the `H_rad` failure could not separate — and it resolved
it in favour of the original numbers being close to what the projected instrument measures.

The original Wave A record (`notebooks/.cache/09_physics_curvature.jsonl` and its anchor tables,
freeze `5f7fbe27…`) is retained beside this one and is not replaced. The positive-control gate's
structural failure and the colleague-sign question (`09-WAVE-A-RESULTS.md` § 5 and § 6) are not
touched by this amendment and remain as recorded there; the supplementary colleague-estimator run
is reported separately in `09-SUPPLEMENT-01-COLLEAGUE-ESTIMATOR.md`.

---
*Phase: 09-curvature-conditioned-label-decodability-physics-replication*
*Amendment 01 re-run — freeze `e31b3010c1a568065e35132ed60a32fb4842db36`, run commit `214862d638137879e62fd126820a30a8afc8a62b`*
