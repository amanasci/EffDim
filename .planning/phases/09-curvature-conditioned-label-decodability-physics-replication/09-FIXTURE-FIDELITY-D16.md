# Instrument fixture fidelity at `d=16`

## Scope

This document is a statement about the plain-autoencoder decoder curvature instrument's
performance on **analytic fixtures with a known-in-closed-form answer**, never about the Physics
data. No Physics result — no `d=16` verdict, no partial correlation, no fit-quality number from
`09_physics_curvature_run.py` — may be defended by quoting a number from this document. This
document exists so that when a `d=16` Physics verdict is read, the instrument that produced it has
a known-answer number attached; it is not itself evidence about galaxies.

## 1. What was measured and why

`d=16` is the one value in Phase 9's `D_SWEEP = (16, 20, 25, 32)` that matches the colleague's
chart rank directly (`09-COLLEAGUE-REANALYSIS.md`'s 512-anchor, `k=2048`, chart-rank-16 record),
so it carries the phase's most-read comparison — and it is the one `d` in this entire milestone at
which the plain-autoencoder decoder curvature field has never been scored against an analytic
answer. `07-CONTEXT.md`'s `INSTRUMENT_FIDELITY_RANGE = (0.53, 0.99)` was measured at `d=20` only;
plan 08-07 added `d=25`. Without a `d=16` measurement, a `d=16` verdict would be exactly the
uninterpretable case `09-RESEARCH.md` Pitfall 2 warns about — and the colleague's own instrument
(split-half `R_H` only, see §6 below) is unvalidated against a known answer at `d=16` too, so
neither side would have a known-answer number to read the verdict beside.

The measurement ran `notebooks/diagnostics/07_instrument_fixture_sweep_run.py --d 16 --out
notebooks/.cache/09_fixture_fidelity_d16.jsonl` unmodified — the existing `--d`/`--out` flags plan
08-07 added, invoked with no code change to the runner, the fixture module
(`varying_ii_controls.py`), or any other sealed file. It exited 0 with a final `DONE` line and
produced four cells: `{cubic, ridge}` fixtures crossed with ambient width `{D=28, D=768}`, `n=5000`
points, `k=231` for the point-cloud arm, on machine seed `20260816`. The point-cloud arm
(`centroid_mean_curvature`) is recomputed per cell rather than quoted, so both instruments are
always measured on the identical cloud; it is the matched training-free baseline, not the
instrument this document is about.

## 2. Measured table

All values below are exactly as the runner's JSONL record stores them (the stdout summary rounds
to 4 places; this table carries the record's own precision for `rho`, `median_cosine`, and
`median_ratio`).

| fixture | `D` | `n` | `var_explained` | `cond(g)` median | `ii_cv` | arm | `rho` | `median_cosine` | `median_ratio` |
|---|---|---|---|---|---|---|---|---|---|
| cubic | 28 | 5000 | 0.9989458270763205 | 2.7936153119538227 | 0.11440970121338638 | point-cloud | 0.6449142866605714 | 0.7884043036064365 | 0.027156354576359745 |
| cubic | 28 | 5000 | 0.9989458270763205 | 2.7936153119538227 | 0.11440970121338638 | plain-decoder | 0.9422766741710669 | 0.9932794885913858 | 0.9610396277738668 |
| cubic | 768 | 5000 | 0.9985989197885459 | 3.932020251317966 | 0.11440970121338638 | point-cloud | 0.6449142866605714 | 0.7884043036064665 | 0.027156354576359856 |
| cubic | 768 | 5000 | 0.9985989197885459 | 3.932020251317966 | 0.11440970121338638 | plain-decoder | 0.8376129732485188 | 0.9719196818369312 | 0.9972002312554775 |
| ridge | 28 | 5000 | 0.9994405376046311 | 1.9975955288513783 | 0.48850984933786046 | point-cloud | 0.43396982091079284 | 0.9270656277226932 | 0.026447075096237217 |
| ridge | 28 | 5000 | 0.9994405376046311 | 1.9975955288513783 | 0.48850984933786046 | plain-decoder | 0.987230725569229 | 0.9995659766206267 | 0.9923055974100696 |
| ridge | 768 | 5000 | 0.9993710455021906 | 2.247765801568611 | 0.48850984933786046 | point-cloud | 0.43396982091079284 | 0.9270656277226512 | 0.026447075096247163 |
| ridge | 768 | 5000 | 0.9993710455021906 | 2.247765801568611 | 0.48850984933786046 | plain-decoder | 0.9881738890309554 | 0.9994735427208619 | 0.9853165903890537 |

The decoder arm beats the point-cloud arm on rank `rho` in all four cells (`decoder beats cloud on
rank in 4 of 4 cells`, per the runner's own stdout summary) — at `d=16` the trained decoder is
never the worse instrument of the two, unlike at `d=20`/`d=25` where the ranking is fixture- and
cell-dependent.

## 3. The `d=16` fidelity range

The decoder-arm rank Spearman across the four cells is `0.9423` (cubic, D=28), `0.8376` (cubic,
D=768), `0.9872` (ridge, D=28), `0.9882` (ridge, D=768). The minimum and maximum define the
`d=16` fidelity range, in the same ordered-pair form as `07-CONTEXT.md`'s
`INSTRUMENT_FIDELITY_RANGE`:

**`INSTRUMENT_FIDELITY_RANGE_D16 = (0.8376, 0.9882)`**

This is the value plan 09-05 freezes into `physics_curvature_probe.INSTRUMENT_FIDELITY_RANGE_D16`
(currently the empty placeholder `()` in the module, per 09-01's frozen-constants block) and the
value plan 09-10 quotes beside the `d=16` verdict sentence. Both endpoints clear `0.53`, the floor
of the `d=20` range, and the floor of this `d=16` range (`0.8376`, at `cubic`/`D=768`) is itself
higher than the `d=20` ceiling's neighbour cells and far above the `d=25` floor of `0.1713` — at
`d=16` this instrument has never scored below `0.84` on either fixture at either ambient width.

## 4. Spread-for-spread comparison

Per the project skill's standing requirement to compare fixtures spread-for-spread rather than
name-for-name, and to report the direction axis (`median_cosine`) beside every rank number:

| fixture | `ii_cv` (dynamic range) | decoder `rho` (D=28 / D=768) | decoder `median_cosine` (D=28 / D=768) |
|---|---|---|---|
| cubic | 0.1144 | 0.9423 / 0.8376 | 0.9933 / 0.9719 |
| ridge | 0.4885 | 0.9872 / 0.9882 | 0.9996 / 0.9995 |

`ridge`'s Hessian-Frobenius coefficient of variation (`ii_cv = 0.4885`) is more than 4x `cubic`'s
(`0.1144`) — `ridge` has substantially more dynamic range in ground-truth curvature magnitude
across its points than `cubic` does. Despite that gap, `ridge` scores the *higher* decoder rho at
both ambient widths (0.9872/0.9882 vs 0.9423/0.8376), which is the opposite of the naive
expectation that a fixture with more spread is easier to rank correctly. The direction axis moves
in the same order: `ridge`'s median cosine (0.9995-0.9996) sits closer to a perfect 1.0 than
`cubic`'s (0.9719-0.9933) at both widths, so the two fixtures agree on which one the instrument
handles better on both the rank axis and the direction axis simultaneously, even though `ridge`
has the wider ground-truth dynamic range. The one place the two fixtures disagree is ambient
width: `cubic` degrades D=28 to D=768 (`rho` 0.9423 to 0.8376, cosine 0.9933 to 0.9719), while
`ridge` is flat across the same widths (`rho` 0.9872 to 0.9882, cosine 0.9996 to 0.9995) — the
ambient-width sensitivity is a `cubic`-specific effect at `d=16`, not a property shared by both
fixtures.

## 5. Beside the other `d` values

| `d` | fidelity range | source |
|---|---|---|
| 16 | (0.8376, 0.9882) | this document, measured 2026-09-02 |
| 20 | (0.53, 0.99) | `07-CONTEXT.md` `INSTRUMENT_FIDELITY_RANGE`, `HANDOFF-v1.1.md` §5.3 |
| 25 | (0.17, 0.97) | `HANDOFF-v1.1.md` §5.3, plan 08-07 |
| 32 | unmeasured | see below |

`d=16` sits entirely above both of the other two measured ranges' floors: `d=20`'s reported floor
is `0.53` and `d=25`'s is `0.17`, while `d=16`'s own floor is `0.8376`. The instrument's known-
answer performance is not monotone in `d` across this sweep — `d=25` has the lowest floor of the
three measured points, not `d=32`, which remains entirely unmeasured (below).

**`d=32` fixture fidelity is NOT measured here and cannot be measured with this runner.** The
small-ambient fixture arm's literal ambient width is `D=28` (a hard literal in the runner, not a
parameter); a `d=32` graph fixture needs local width `m = d + 1 = 33`; and `varying_ii_controls
.rotate_and_pad` requires `D >= m`, so it raises `ValueError: rotate_and_pad: D=28 must be >= local
width m=33` by construction the moment `--d 32` is passed. This is not a bug to patch — it is a
limitation already discovered and ratified in `HANDOFF-v1.1.md` §5.3 ("`d=32` has no fidelity
measurement... The developer ratified recording that as a fixture-design finding and deferring the
measurement") and named explicitly in the Deferred section of `09-CONTEXT.md` and Pitfall 6 of
`09-RESEARCH.md`. Phase 9 does not fix the fixture, does not widen the small-ambient arm, and does
not attempt the `--d 32` run. No invocation of the fixture sweep runner with `--d 32` appears
anywhere in this document or elsewhere in Phase 9's artifacts. At `d=32`, a dying instrument and a
vanishing effect remain indistinguishable, exactly as `HANDOFF-v1.1.md` states.

## 6. Why a reliability score is not a substitute

`06-FINDINGS.md` measured, on the Swiss roll where the true curvature is known in closed form, a
near-perfect split-half reliability `R_H = 0.990` coexisting with a mediocre true-answer accuracy
`rho = 0.469` against ground truth. Reliability and correctness are different quantities: split-
half agreement can be high while the two halves share a common bias that a known-answer check would
catch and a reliability check cannot, by construction, ever see (both halves inherit the identical
bias, so they still agree with each other).

The colleague's own instrument at `d=16` carries exactly this gap. `09-COLLEAGUE-REANALYSIS.md`
reports his split-half `R_H` at `d=16` medians `0.514`, with 42% of his 512 anchors scoring
`R_H < 0.5` and none above `0.7` — a reliability score that is itself unimpressive by his own
(blind-spot-prone) metric, and one that, even if it had scored `R_H` near 1.0, would still say
nothing about whether his instrument agrees with an analytic answer at `d=16`. He has run no
known-answer check on his own instrument at any `d`.

Neither side, therefore, has a known-answer number at `D=768, k=2048, d=16` for the instrument
that actually produced the colleague's chart — his `R_H = 0.514` median is a reliability score, not
a fidelity score, and prior to this document the plain-autoencoder decoder used on the Physics side
had no fixture-fidelity measurement at `d=16` either. This document is the only known-answer
evidence either side has at `d=16`, and it is a statement about a fixture, not about the fitted
label-decodability effect. §3's range is what a `d=16` verdict should be read beside; a reliability
score, from either instrument, is not a substitute for it.

## 7. Measured wallclock

Per-cell timings from the JSONL record (point-cloud fit `t_cloud_s`, decoder training
`t_train_s`, curvature-field derivation `t_curv_s`), run single-process on a 12th Gen Intel
Core i7-1280P (20 logical threads available, `nproc=20`), no explicit thread cap set (the machine
was otherwise idle during this run):

| fixture | `D` | `t_cloud_s` | `t_train_s` | `t_curv_s` | cell total |
|---|---|---|---|---|---|
| cubic | 28 | 2.2 | 178.9 | 16.7 | 197.8 |
| cubic | 768 | 202.0 | 126.4 | 578.8 | 907.2 |
| ridge | 28 | 2.1 | 72.1 | 16.6 | 90.8 |
| ridge | 768 | 195.9 | 124.2 | 577.9 | 898.0 |

Total wallclock across all four cells: **2093.8 s (~34.9 minutes)**. The two `D=768` cells are the
ones relevant to a 5,000-row, ambient-768 cost estimate: AE training took 126.4 s and 124.2 s, and
curvature-field derivation (the `torch.func` Jacobian/Hessian pass) took 578.8 s and 577.9 s —
together roughly 705-707 s of compute per 5,000-row, `D=768` cell, dominated by the curvature-field
derivation step rather than training itself.
