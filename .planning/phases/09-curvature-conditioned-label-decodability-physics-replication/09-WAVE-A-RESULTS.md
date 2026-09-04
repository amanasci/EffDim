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

## 1. The frozen rule, quoted

Every value below is the verbatim committed value in `notebooks/pu_manifold/physics_curvature_probe.py`
at freeze commit `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` (`09-PREREGISTRATION.md`):

```
VERDICT_RULE:

D9-10 VERDICT_RULE -- frozen in committed source before any Physics probe
number existed (D9-18).

"Replicates" at a given d in D_SWEEP means the controlled 3-control partial
(VERDICT_STATISTIC, on CURVATURE_FIELD_FOR_VERDICT = "H_tan_norm") is STRICTLY NEGATIVE
(rho < 0.0) AND clears its own Freedman-Lane rank-permutation null under the family-wise
envelope (the per-draw maximum absolute controlled partial across D_SWEEP) at FWER_ALPHA = 0.05,
using a strict < on p_fwer. No magnitude threshold. Magnitude is printed beside the colleague's
-0.240 with both bootstrap bands (his B=2000 paired anchor resamples; ours the same, N_BOOTSTRAP
= 2000).

Per-d cells are reported independently -- PER_D_VERDICT_VALUES[0] ("NEGATIVE AND CLEARS FWER
NULL") or PER_D_VERDICT_VALUES[1] ("DOES NOT CLEAR") -- with NO pooled headline number across d.
The phase verdict then aggregates the per-d cells: every d fired gives VERDICT_VALUES[0]
("REPLICATES AT EVERY d"), at least one but not all gives VERDICT_VALUES[1] ("REPLICATES AT
SUBSET OF d"), none gives VERDICT_VALUES[2] ("DOES NOT REPLICATE"). VERDICT_VALUES[3]
("HALTED - ALIGNMENT NOT PROVED") is reserved for the case where the D9-06/D9-07 row-alignment
proof itself never clears ALIGNMENT_MARGIN_R2 at any candidate offset -- not this run, since the
alignment proof PASSED (09-ALIGNMENT-PROOF.md).
```

```
WAVE_B_TRIGGER_RULE = (
    "Wave B (the three-seed sweep, TORCH_INIT_SEEDS_WAVE_B) runs only at d values where the "
    "Wave A (single TORCH_INIT_SEED = 0) per-d verdict fired (PER_D_VERDICT_VALUES[0]); d values "
    "where Wave A did not fire are never re-run under Wave B."
)
```

No magnitude threshold exists anywhere in `VERDICT_RULE`: a per-`d` cell fires purely on sign
(`rho < 0.0`, strict) and FWER clearance (`p_fwer < 0.05`, strict) — a controlled partial of
`-0.001` that clears its null fires exactly as hard as one of `-0.35`. Per-`d` cells are
independent by construction; `phase_verdict` aggregates them without ever computing a number
across `d`, and `WAVE_B_TRIGGER_RULE` reads only the per-`d` verdict map, never a magnitude.

## 2. Per-`d` table for `mag_r`

Full precision, `D_SWEEP` order. `n_bootstrap` = `N_BOOTSTRAP` = 2000 paired-anchor resamples for
every cell.

| `d` | field | raw `rho` | controlled partial | Freedman-Lane `p` (per-`d`) | Freedman-Lane `p` (family-wise) | stratified null `p`, S=10 | stratified null `p`, S=20 | paired bootstrap 95% band |
|---:|---|---:|---:|---|---|---|---|---|
| 16 | `H_tan_norm` (gating) | 0.425064 | **0.346967** | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` | `< 2.000e-04` | [0.265565, 0.428706] |
| 16 | `H_norm` (non-gating) | 0.380762 | 0.287949 | `< 9.999e-05` | `< 9.999e-05` | `< 2.000e-04` | `< 2.000e-04` | [0.200894, 0.376456] |
| 20 | `H_tan_norm` (gating) | 0.231160 | **0.030323** | 0.501650 | `< 9.999e-05` | 0.479304 | 0.473905 | [-0.071360, 0.131843] |
| 20 | `H_norm` (non-gating) | 0.178914 | -0.028316 | 0.529747 | `< 9.999e-05` | 0.514097 | 0.498100 | [-0.123326, 0.065528] |
| 25 | `H_tan_norm` (gating) | 0.251751 | **0.042119** | 0.345665 | `< 9.999e-05` | 0.329334 | 0.333933 | [-0.054442, 0.143788] |
| 25 | `H_norm` (non-gating) | 0.237670 | 0.004915 | 0.913809 | `< 9.999e-05` | 0.915617 | 0.914417 | [-0.089648, 0.102562] |
| 32 | `H_tan_norm` (gating) | 0.209815 | **-0.003450** | 0.935506 | `< 9.999e-05` | 0.931014 | 0.940812 | [-0.100984, 0.100657] |
| 32 | `H_norm` (non-gating) | 0.259529 | 0.055731 | 0.214279 | `< 9.999e-05` | 0.192761 | 0.201160 | [-0.036496, 0.152248] |

The family-wise Freedman-Lane `p` (the per-draw maximum absolute controlled partial across
`D_SWEEP`, a null construction, never a pooled headline statistic) sits at the permutation floor
(`< 9.999e-05`, `N_PERMUTATIONS = 10000`) for both fields — the family-wise envelope is dominated
by `d=16`'s own large observed value and clears trivially; it is not evidence for any other `d`
cell individually, and `per_d_verdict` reads only each cell's own `p_fwer` (the per-`d` column
above), never the envelope's.

Only `d=16` clears FWER (`p_fwer < 9.999e-05 < 0.05`), but its sign is **positive**
(`+0.346967`), not negative — `VERDICT_RULE` requires `rho < 0.0` AND `p_fwer < FWER_ALPHA`, both
strict, so `d=16` gets `DOES NOT CLEAR` despite having by far the strongest statistical signal in
the sweep. `d=20/25/32` neither clear FWER nor sit reliably on one side of zero (every bootstrap
band at `d=20/25/32` straddles zero).

## 3. Fit quality per `d`

| `d` | `var_explained` | median `cond(g)` | `H_rad` median | `-d` (expected) | deviation | low-image-norm exclusions | masked anchors (`mag_r`) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.9520468951883061 | 14.726070366715556 | -20.391490489648845 | -16 | -27.45% | 0 | 0 |
| 20 | 0.9569335974793522 | 13.749418462007677 | -24.244205345752775 | -20 | -21.22% | 0 | 0 |
| 25 | 0.9611704107839105 | 12.012144761657932 | -29.613265737402585 | -25 | -18.45% | 0 | 0 |
| 32 | 0.9648528934247134 | 10.537474284699378 | -36.82735975422215 | -32 | -15.09% | 0 | 0 |

`var_explained` is high at every `d` (95.2%-96.5%, rising monotonically with `d`) and `cond(g)`
falls monotonically (14.73 -> 10.54) — by both measures the autoencoder fit itself is not the weak
link. **The `H_rad` backstop truth does not hold at any `d`**: `09-08-PLAN.md`'s `must_haves`
states `H_rad` should sit within 10% of `-d`, and every cell here misses by 15-27%, worst at the
smallest `d` and shrinking (but never clearing 10%) as `d` grows. This is reported plainly per the
backstop's own instruction — a poor-fit-vs-real-geometry ambiguity is not resolved by this run,
and it is stated beside `var_explained` so a reader is not left to infer good geometry from good
reconstruction alone.

**Instrument-fidelity ranges** (rank-Spearman of the trained decoder's curvature against a known
ground truth, on a synthetic fixture, measured independently of this run — 09-FIXTURE-FIDELITY-D16.md
§3, `07-CONTEXT.md` §4/`HANDOFF-v1.1.md` §5.3, `09-FIXTURE-FIDELITY-D16.md` §5):

- `d=16`: `(0.8376, 0.9882)` — measured 2026-09-02.
- `d=20`: `(0.53, 0.99)`.
- `d=25`: `(0.17, 0.97)` — the widest range of the three, spanning from near-useless to
  near-perfect depending on fixture and ambient width.
- `d=32`: **unmeasured**. `INSTRUMENT_FIDELITY_D32_RULE`: the small-ambient fixture arm's
  literal ambient width is `D=28`, a `d=32` graph fixture needs local width `m = d + 1 = 33`, and
  `varying_ii_controls.rotate_and_pad` requires `D >= m`, so the fixture sweep raises `ValueError`
  by construction the moment `--d 32` is passed. Not a bug to patch — ratified in
  `HANDOFF-v1.1.md` §5.3. At `d=32`, a dying instrument and a vanishing effect remain
  indistinguishable by design.

Given the widest fidelity range at `d=25` runs from 0.17 to 0.97, and `d=32`'s is unmeasured
entirely, the `d=25` and `d=32` cells' near-zero, non-clearing controlled partials cannot be read
as evidence the underlying relationship is null — the instrument at those `d` values has, on the
independently-measured evidence, an uncertain-to-wide range of fidelity to begin with.

## 4. `H_tan` against `H_norm`

| `d` | controlled partial, `H_tan_norm` | controlled partial, `H_norm` | agree in sign? | magnitude ratio |
|---:|---:|---:|---|---|
| 16 | 0.346967 | 0.287949 | yes (both +) | 1.205x (`H_tan` larger) |
| 20 | 0.030323 | -0.028316 | **no** | n/a (sign disagreement) |
| 25 | 0.042119 | 0.004915 | yes (both +) | 8.57x (`H_tan` much larger) |
| 32 | -0.003450 | 0.055731 | **no** | n/a (sign disagreement) |

`08-DIAGNOSTICS.md` §2 measured, on Phase 8's own construction (a different label/probe setup
from this phase's own, quoted for the diagnostic comparison the plan asks for): the `||H||`- vs
`||H_tan||`-based partial "strengthens 1.12x" at `d=20` (-0.022580 -> -0.025253), "collapses 2.8x"
at `d=25` (-0.065909 -> -0.023256), and "sign flips" at `d=32` (-0.026858 -> +0.056385).

This phase's own data does **not** reproduce that exact pattern: here the sign disagreement shows
up at `d=20` and `d=32`, not at `d=25` (which agrees in sign both ways, just at very different
magnitudes, 8.57x apart), and the direction of the `d=32` disagreement is reversed relative to
`08-DIAGNOSTICS.md`'s own `d=32` measurement (there `H` is the negative one and `H_tan` flips
positive; here `H_tan` is the negative one and `H_norm` is positive). Both phases' instruments
agree only on the general finding that `H`-based and `H_tan`-based partials are not
interchangeable at every `d` — the specific `d` at which they diverge and the direction of the
divergence is not the same measurement reproduced twice. All four of this phase's `d=20/25/32`
values sit inside noise (none clears FWER), so "sign disagreement" here is a statement about
where zero-crossing noise happens to land, not a structural collapse of the kind `08-DIAGNOSTICS`
measured with `d=25`'s significant partial.

`H_tan` carries the verdict regardless of this comparison — `H_norm` is reported beside it and
never promoted, per `H_NORM_IS_NON_GATING`.

## 5. Both gates

### The positive control

| `d` | target magnitude | target `rho` (planted, curvature-side) | achieved controlled partial | slope | cleared? | Freedman-Lane `p` |
|---:|---:|---:|---:|---:|---|---|
| 16 | 0.05/0.10/0.15/0.20/0.25 (all five collapse to the same achieved value — see below) | -0.05...-0.25 | 0.011552 | 1.8189894035458565e-12 (~0) | false (all five) | 0.787421 |
| 20 | 0.05/0.10/0.15/0.20/0.25 | -0.05...-0.25 | 0.053438 | 2.0 (bracket ceiling) | false (all five) | 0.234877 |
| 25 | 0.05/0.10/0.15/0.20/0.25 | -0.05...-0.25 | 0.030884 | 2.0 (bracket ceiling) | false (all five) | 0.498050 |
| 32 | 0.05/0.10/0.15/0.20/0.25 | -0.05...-0.25 | 0.001134 | 2.0 (bracket ceiling) | false (all five) | 0.979702 |

**Detection floor: `None` — no target of `POSITIVE_CONTROL_TARGET_RHOS = (0.05, 0.10, 0.15, 0.20,
0.25)` cleared at any `d`.** Per `09-EXECUTION-HOST.md` §8's own failure-branch classification,
literally applied: *"This means the instrument ... is not sensitive enough to detect a real effect
of the planted size, regardless of what step 3's dsweep found ... do not proceed to trust any
`--mode dsweep` number as meaningful."*

**Beyond the §8-literal reading, the mechanism of this failure is structural, not a measurement of
instrument insensitivity — recorded here as evidence, not as a revision of the frozen verdict.**
`plant_curvature_positive_control` plants a spread-matched, binomially-noised rank-copy of the
real `H_tan_norm` field and bisects the slope of that plant against `controlled_partial(planted,
y, Z)` on the bracket `[0, 2]`, measuring the bisection direction once (slope 0 vs slope 2) before
searching. Two consequences follow directly from the mechanism, both checkable against the record
above:

1. **The achievable statistic is bounded by the real `h_real`-`y` partial, attenuated by the
   binomial plant noise.** At `d=20/25/32` the bisection ran to the bracket ceiling (slope 2.0)
   and the achieved value tracks the *unplanted* `H_tan_norm`-`mag_r` partial closely: achieved
   0.0534/0.0309/0.0011 versus the real partial 0.030323/0.042119/-0.003450 at the same `d` — the
   same order of magnitude, not the target magnitude the grid asks for (0.05-0.25). No target in
   that range is reachable at `d=20/25/32` unless the real relationship already carried an effect
   of comparable size, which it does not.
2. **The runner plants negative targets** (`target_rho` is `-0.05` through `-0.25` at every entry
   above, confirmed directly from the record's `positive_control` rows), matching Task 1's own
   deviation record, **while the real `d=16` relation is strongly positive** (`+0.346967`). The
   direction test at `d=16` measured `achieved_at_high >= achieved_at_low` as false (increasing
   real signal opposes the negative target direction), so the bisection collapsed toward slope
   `~0` and reports the pure-noise-floor partial (`0.011552`) rather than searching meaningfully
   at all.

Either mechanism alone would prevent any target in `POSITIVE_CONTROL_TARGET_RHOS` from clearing on
this data, independent of how sensitive the instrument itself is. **Whether to amend the gate's
plant direction or grid — which would require a sealed-module edit, a fresh freeze, and a re-run
of the gate modes only — is put to the developer in this plan's SUMMARY.md rather than decided
here.** The frozen verdict above (`DOES NOT REPLICATE`, per-`d` `DOES NOT CLEAR` at every `d`) is
not softened or re-derived by this finding; no new statistic is added to the record.

### The shuffled-label calibration

| `d` | repeats | false positives (`cleared=true`) | rate | nominal `FWER_ALPHA` |
|---:|---:|---:|---:|---:|
| 16 | 20 | 1 | 0.05 | 0.05 |
| 20 | 20 | 1 | 0.05 | 0.05 |
| 25 | 20 | 2 | 0.10 | 0.05 |
| 32 | 20 | 1 | 0.05 | 0.05 |
| **pooled** | **80** | **5** | **0.0625** | **0.05** |

The pooled false-positive rate (5/80 = 0.0625) sits modestly above the nominal 0.05 level; per-`d`
it ranges from exactly nominal (0.05 at `d=16/20/32`) to double nominal (0.10 at `d=25`, 2 of 20).
At this repeat count (`n=20` per `d`, `n=80` pooled) a single extra false positive moves the
per-`d` rate by 5 percentage points, so `d=25`'s 0.10 is not distinguishable from sampling noise
around a true 0.05 rate at this sample size; it is reported as measured, not adjusted.

**What a verdict means given both gates.** The positive control found no detectable floor at any
magnitude in `0.05`-`0.25` — for the reasons in the structural analysis above, this cannot be read
purely as an instrument-sensitivity statement, but on the frozen §8-literal reading it means a
cell whose observed magnitude sits below `0.25` is not, by this gate's own design, distinguishable
from a null instrument. Combined with a false-positive rate at or modestly above nominal, a cell
that *does* clear FWER (only `d=16`, and only in sign-disqualified direction) is not resting on an
unusually generous null; a cell that does not clear is weakened further by the gate's inability to
demonstrate it *could* detect a real effect of plausible size.

## 6. Beside his numbers

The colleague's own values, transcribed from `09-COLLEAGUE-REANALYSIS.md`, all at his chart rank
`d=16` unless noted: raw `rho(K_H_cross, r2_G)` = **-0.4124** (matches frozen), controlled (3
controls) = **-0.2405** (matches frozen), `rho(K_H_cross, log_knn_radius)` = **+0.765**. From his
parity table at other `d`: `d=12` (outside this phase's `D_SWEEP`, reported as non-comparable) raw
`-0.038`, controlled **+0.143**, `rho(K_H, log r)` `+0.495`; `d=20` raw `-0.392`, controlled
**-0.233**, `rho(K_H, log r)` `+0.711`.

This phase's own `mag_r` / `H_tan_norm` controlled partials, same `d` values where they overlap:
`d=16` **+0.346967** (his: -0.2405 — opposite sign), `d=20` **+0.030323** (his: -0.233 — opposite
sign, and this phase's value is two orders of magnitude smaller in absolute terms). Neither `d`
reproduces his sign.

**This phase's own `rho(H_tan_norm, log_knn_radius)`, computed directly from the returned anchor
tables, per `d` in `D_SWEEP` order:**

| `d` | `rho(H_tan_norm, log_knn_radius)` |
|---:|---:|
| 16 | -0.561698 |
| 20 | -0.436420 |
| 25 | -0.555967 |
| 32 | -0.502702 |

His own value at his `d=16` is **+0.765**; this phase's value at every one of its four `d` values
is **negative**, roughly -0.44 to -0.56 in magnitude — the opposite sign of his own, at every `d`,
not merely a different magnitude. If radius were doing "most of the work" on this phase's field
the way it appears to on his (his own `-0.216`/`-0.194` mean within-stratum rho after
density-stratifying shows the raw association attenuates substantially once radius is controlled,
consistent with radius carrying much of his raw `-0.4124`), the sign of that relationship would at
least match; here it runs the other direction, so this phase's field is not showing "the same
structure" his does with respect to neighbourhood density — it is a different structure, reported
plainly rather than glossed as equivalent.

**Neighbourhood ratio, restated:** `K_NEIGHBOURS=2048` of `n=86,471` Physics rows is **1/42** of
the sample; his own `k=2048` was drawn against a 16,384-row hash-selected subset of the same
86,471-row test set, **1/8** of it — his neighbourhoods are more than five times denser than this
phase's own, at the identical nominal `k`. Every comparison in this section runs across that
density-scale difference, not at matched density.

## 7. Secondary labels

Marked non-gating throughout (`SECONDARY_LABELS_ARE_NON_GATING = True`); each has its own nulls
and does not affect `per_d_verdict` or `phase_verdict`, which read `mag_r` only.

### `photo_z`

| `d` | field | raw `rho` | controlled partial | FWER `p` | stratified `p` (S=10/S=20) | bootstrap band | masked anchors |
|---:|---|---:|---:|---|---|---|---:|
| 16 | `H_tan_norm` | 0.172874 | 0.366797 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.295937, 0.441022] | 0 |
| 16 | `H_norm` | 0.175195 | 0.321931 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.246635, 0.398389] | 0 |
| 20 | `H_tan_norm` | 0.149048 | 0.314020 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.228033, 0.395687] | 0 |
| 20 | `H_norm` | 0.137392 | 0.268873 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.182797, 0.351229] | 0 |
| 25 | `H_tan_norm` | 0.155961 | 0.377687 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.300089, 0.455607] | 0 |
| 25 | `H_norm` | 0.145107 | 0.368353 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.290210, 0.441843] | 0 |
| 32 | `H_tan_norm` | 0.197428 | 0.417527 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.342428, 0.490312] | 0 |
| 32 | `H_norm` | 0.170779 | 0.389254 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.313673, 0.462694] | 0 |

`local_evaluation_count_constant = false` at every `d` for `photo_z` — its column has some
sentinel-masked coverage inside the 2048-neighbourhoods, so its three-control partial is not
reducible to a two-control one.

### `smooth_fraction`

| `d` | field | raw `rho` | controlled partial | FWER `p` | stratified `p` (S=10/S=20) | bootstrap band | masked anchors |
|---:|---|---:|---:|---|---|---|---:|
| 16 | `H_tan_norm` | 0.116237 | 0.348011 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.260228, 0.426746] | 0 |
| 16 | `H_norm` | 0.161863 | 0.309754 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.221933, 0.390207] | 0 |
| 20 | `H_tan_norm` | 0.305862 | 0.323578 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.235489, 0.407061] | 0 |
| 20 | `H_norm` | 0.347626 | 0.274311 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.190937, 0.358318] | 0 |
| 25 | `H_tan_norm` | 0.214073 | 0.352684 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.268180, 0.434819] | 0 |
| 25 | `H_norm` | 0.188671 | 0.309846 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.227337, 0.388102] | 0 |
| 32 | `H_tan_norm` | 0.220063 | 0.412762 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.337872, 0.487946] | 0 |
| 32 | `H_norm` | 0.207834 | 0.414264 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.341320, 0.483647] | 0 |

`local_evaluation_count_constant = true` at every `d` for `smooth_fraction` — fully-populated
inside every anchor's 2048-neighbourhood, so its three-control partial equals its two-control one.

### `stellar_mass`

The `stellar_mass -> mass_med_photoz` column mapping was originally tagged `[ASSUMED]`
(`09-RESEARCH.md` Assumption A1) and is now **`[RATIFIED 2026-09-03]`**, per the `09-04` Task 2
checkpoint (`09-DATA-MANIFEST.md` §7). Non-gating either way (D9-16).

| `d` | field | raw `rho` | controlled partial | FWER `p` | stratified `p` (S=10/S=20) | bootstrap band | masked anchors |
|---:|---|---:|---:|---|---|---|---:|
| 16 | `H_tan_norm` | 0.187777 | 0.073530 | 0.099490 | 0.087383 / 0.099180 | [-0.016789, 0.159129] | 0 |
| 16 | `H_norm` | 0.135906 | 0.054744 | 0.219678 | 0.210958 / 0.221556 | [-0.034041, 0.141256] | 0 |
| 20 | `H_tan_norm` | -0.027635 | 0.131945 | 0.001999 | 0.004199 / 0.003799 | [0.045455, 0.216173] | 0 |
| 20 | `H_norm` | -0.081992 | 0.103740 | 0.017098 | 0.030194 / 0.033593 | [0.018794, 0.188396] | 0 |
| 25 | `H_tan_norm` | 0.104099 | 0.227331 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.139741, 0.309730] | 0 |
| 25 | `H_norm` | 0.075700 | 0.215294 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.126124, 0.298948] | 0 |
| 32 | `H_tan_norm` | 0.104656 | 0.263450 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.181441, 0.343635] | 0 |
| 32 | `H_norm` | 0.141291 | 0.259745 | `< 9.999e-05` | `< 2.000e-04` / `< 2.000e-04` | [0.174734, 0.343378] | 0 |

`local_evaluation_count_constant = false` at every `d` for `stellar_mass`, consistent with its
~91.9% (rather than fully-populated) column coverage measured in `09-DATA-MANIFEST.md`. Unlike
`mag_r`, `stellar_mass`'s raw `rho` flips sign between `d=16` (positive) and `d=20` (negative)
while its controlled partial stays positive throughout — none of this is gating.

## 8. Per-`d` verdicts and the Wave B trigger

Applying `per_d_verdict(rho, p_fwer, FWER_ALPHA)` to each `mag_r`/`H_tan_norm` cell in §2 (strict
`rho < 0.0` AND strict `p_fwer < 0.05`):

| `d` | `rho` | `p_fwer` | `rho < 0`? | `p_fwer < 0.05`? | verdict |
|---:|---:|---:|---|---|---|
| 16 | 0.346967 | `< 9.999e-05` | no | yes | **DOES NOT CLEAR** |
| 20 | 0.030323 | 0.501650 | no | no | **DOES NOT CLEAR** |
| 25 | 0.042119 | 0.345665 | no | no | **DOES NOT CLEAR** |
| 32 | -0.003450 | 0.935506 | yes | no | **DOES NOT CLEAR** |

Every `d` cell returns `DOES NOT CLEAR`. `d=16` is the one cell worth naming explicitly: its
Freedman-Lane `p` clears the family-wise floor at `< 9.999e-05` with the strongest, most
statistically decisive signal in the entire sweep, and its raw and controlled partials are both
positive and far from zero — but `VERDICT_RULE` requires `rho < 0.0` (strictly negative) as one of
two independent AND conditions, and `d=16`'s sign is the wrong one. The frozen rule does not count
a strong, well-calibrated, wrong-signed effect as a fired cell; per-`d` verdicts read sign and
FWER clearance jointly, never magnitude or statistical significance alone. `d=32` is the sole cell
with the correct sign (`rho = -0.003450 < 0`), but its magnitude is the smallest in the sweep and
its `p_fwer` (0.935506) is the furthest from clearing of all four.

**Applying `WAVE_B_TRIGGER_RULE` literally: Wave B runs only at `d` values where the Wave A per-`d`
verdict fired (`PER_D_VERDICT_VALUES[0]`, "NEGATIVE AND CLEARS FWER NULL"). Zero of the four `d`
cells fired. Wave B scope: `WAVE_B_NOT_TRIGGERED`.** 09-09's three-seed sweep does not run at any
`d`; per `09-EXECUTION-HOST.md` §4 step 6, "if Wave A fired at zero `d` values, skip this step
entirely; there is nothing for it to run."

No phase verdict is written in this plan — Wave B is not triggered, so there is nothing further
for 09-10 to resolve before the phase verdict can be finalized there, but that finalization is
09-10's own act per this plan's `<discretion_decisions>`, not this document's.

