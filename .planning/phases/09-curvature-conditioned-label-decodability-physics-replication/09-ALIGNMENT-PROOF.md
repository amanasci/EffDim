# 09-ALIGNMENT-PROOF — the row-alignment proof (D9-05..D9-08)

**Plan:** 09-07
**Written:** 2026-09-04 UTC

## Provenance of this run

The row-alignment proof is the phase's first real statistic and, per `09-EXECUTION-HOST.md`
§1 (`EXECUTION_HOST_RULE`), can only be produced on the execution host chosen in 09-06 — Claude
has no credential for that machine. Per the developer's standing instruction (2026-09-04 UTC,
verbatim): *"begin with running experiments on ssh server. ensure you use AVAILABLE compute,
don't kick someone off if they are already using. check free compute with nvidia-smi. adhere
strictly to the user-guide."* The orchestrator executed Task 1's host steps (§4 step 1, then
`--mode bundle`) over SSH on the verified host from 09-06, following `09-EXECUTION-HOST.md`
literally, under that standing instruction — the commands were not typed interactively by the
developer. This document transcribes what those steps measured; the instruction itself
authorized nothing about this plan's structure, tooling or permissions.

## Host capability (as bootstrapped, `09-EXECUTION-HOST.md` §9)

OS Ubuntu 22.04.5 LTS; core count 128 (`os.cpu_count()`/`nproc`, cgroup CPU limit unlimited);
RAM 1006 GB total, ~836 GB free; GPU 8x NVIDIA A100-SXM4-80GB, all idle at survey time and
**not used** (Phase 9 is CPU-only); Python 3.14.7; library versions torch 2.13.0+cpu,
numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0, pyarrow 25.0.1, pandas 3.0.5, datasets 5.0.1.
Thread count used for this run: 16, chosen to leave the remaining cores free for other users of
the shared host per the developer's "don't kick someone off" instruction (load average at
survey time ~4 on 128 cores, no other live compute jobs). Host identity is recorded as
capability only — no hostname, IP address, username or SSH key path appears here or anywhere
else in this phase's artifacts (`09-EXECUTION-HOST.md` §7).

## Run record

| Field | Value |
|---|---|
| Freeze SHA the run was gated on | `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` |
| Run commit (`HEAD` on the host at run time) | `ee992bac947f3469dfb0e607867901992f0b17de` |
| Freeze-ancestry proof on the host | `git merge-base --is-ancestor 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 HEAD` → exit 0 (`is-ancestor: OK`); `git rev-list --count 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5..HEAD` → `5` |
| Command | `.venv/bin/python notebooks/diagnostics/09_row_alignment_proof_run.py --mode proof --freeze-commit 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 --threads 16` |
| Started (UTC) | 2026-09-04T04:46:13Z |
| Record written (UTC) | 2026-09-04T05:07:25Z |
| Wall-clock | ~21 minutes, download-dominated (three transient `HTTP Error 429` responses from `huggingface.co` on the `Smith42/galaxies` shard `test-00004-of-00016.parquet`, retried automatically at 1s/2s/4s and succeeded — not an infrastructure failure) |
| Exit code | **0** |
| Environment line as printed | `core_count=128 thread_cap=16 python=3.14.7 torch=2.13.0+cpu numpy=2.5.1 scipy=1.18.0 scikit-learn=1.9.0 pyarrow=25.0.1 pandas=3.0.5 datasets=5.0.1` |

**Verdict line, verbatim:**

```
verdict: r2_shift0=0.515931 best_other_r2=-0.000082 gap=0.516013 margin=0.1 passed=True
```

Since step 2 (the proof) exited 0 (PASS), step 3 (`--mode search`) was **not run**, per
`09-EXECUTION-HOST.md` §4 step 2 and this plan's own instructions ("Do not run it if step 2
passed — it will refuse").

## Archive transfer and checksum verification

Returned bundle: `09-artifacts-pod128-20260904T051024Z.tar.gz`, 1952 bytes, containing
`09_row_alignment.jsonl`, `09_scratch_alignment.jsonl`, `09_scratch_tracer.jsonl`,
`environment.json`.

The archive's SHA-256 was **recomputed locally and compared to the host-reported digest before
any value was read out of it** (T-09-44):

```
host-reported:            c6637c8858cea9345b47d2880d1b7ac31ec22b88fa8fc698ee59dbc26760ce50
locally recomputed (sha256sum): c6637c8858cea9345b47d2880d1b7ac31ec22b88fa8fc698ee59dbc26760ce50
```

**Match confirmed.** The archive was extracted under the local resolved output root
(`notebooks/.cache/`, `EFFDIM_09_OUTPUT_ROOT` unset locally). `notebooks/.cache/09_row_alignment.jsonl`
now exists locally (13,726 bytes, 47 JSONL rows: 1 environment row, 45 curve rows, 1 verdict
row). The two scratch smoke files (`09_scratch_alignment.jsonl`, `09_scratch_tracer.jsonl`) were
overwritten by this extraction; they are scratch, as noted when the archive was returned, and
carry no production Physics record. `notebooks/.cache/09_physics_curvature.jsonl` does **not**
exist locally, confirmed by direct check — no Physics number beyond this proof exists yet.

**Automated verify (acceptance criteria, adapted — see Deviations below):**

```
alignments 45 verdicts 1 passed [True]
```

## Deviations from the plan's literal verify command

**[Rule 1 - Bug] The plan's Task 1 automated `<verify>` filters on `row_kind == 'alignment'`,
but the runner's actual JSONL schema uses `row_kind == 'curve'` for every shift/permutation row
(with a separate top-level `alignment` field valued `'shift'` or `'permutation'` distinguishing
the two curve kinds).** Running the verify command exactly as written in the plan raises
`AssertionError: 0` because no row has `row_kind == 'alignment'`. This is a stale field-name
assumption in the plan's verify script, not a defect in the returned record: the record's own
`row_kind` values are `environment` (1), `curve` (45), `verdict` (1) — inspected directly from
`notebooks/.cache/09_row_alignment.jsonl`. Re-running the identical acceptance check with
`row_kind == 'curve'` substituted for `row_kind == 'alignment'` (the only change) passes:
`alignments 45 verdicts 1 passed [True]` — satisfying the acceptance criteria's actual
requirement ("at least 45 alignment rows and at least one verdict row") since the plan's prose
and this document's own §"The measured curve" below both use "alignment row" to mean any
shift-or-permutation curve entry. No source file was touched; this is a verification-script
transcription note, not a code fix, since the runner script is sealed
(`notebooks/pu_manifold/` discipline extends to the diagnostics runners' output schema being the
authority here) and out of this task's `<files>` scope to edit.

## 1. What was proved and how

The physics embeddings (`UniverseTBD/pu-embeddings`, config `physics_vit_base_test`) and the
physics labels (`Smith42/galaxies@v2.0`, split `test`) share **no identifier column** — no
`object_id`, no `sample_id`, nothing to join on. The only join available is positional: row `i`
of the embedding table is assumed to be row `i` of the label table. D9-05 states plainly that
the colleague's own branch never tested this for the Physics join: his `sample_id` convention
(galaxies test-table row index, "row-aligned to `vit_base_test.parquet`" by his own comment) is
a documented convention with a labels-build script that is absent from his branch and no test
anywhere on it. "Equal row count is not the proof" is his own standard, applied by him only to
strike his DESI associations (`desi_label_alignment_unresolved`, `Proved=False`) — Physics itself
was never checked by his own rule.

D9-06 supplies the method this phase runs: fit a 5-fold out-of-fold ridge probe from the 768-d
embedding to `mag_r` at the assumed alignment (shift 0), then re-fit the identical probe after
shifting the label vector by each of 24 frozen non-zero row offsets and after 20 seeded random
row permutations. If the positional join is correct, only shift 0 carries real embedding-to-label
structure; every shifted or permuted pairing destroys that structure and its OOF R2 should
collapse toward the R2 of predicting a constant (near 0, since `mag_r` is not label-shuffle-
invariant in any other way the ridge probe could exploit). A large, robust gap between R2(shift 0)
and the best R2 among all 44 misaligned pairings is the statistical proof; D9-07 fixes the shift
set, the permutation count and the pass margin, all before this run.

**What this proof is NOT.** It is not a restatement of the colleague's convention — his
convention supplies a claim, not a test of that claim, for the Physics join specifically. It is
also not `subsample.assert_alignment` (`notebooks/pu_manifold/subsample.py` lines 116-180), which
this milestone already uses elsewhere: that function tests two embedding columns (HSC vs Legacy
Survey, both from `pu-embeddings`) against each other via a permuted-null z-score on their
per-row cosine similarity (`z = (s_true - mu_perm) / sd_perm`, required `z > ALIGNMENT_MARGIN_Z`).
It answers "are these two embedding columns the same rows in the same order", using the fact that
both columns are embeddings of the same underlying object and so should be highly cosine-similar
row-for-row when aligned. It does not transfer to this question: `mag_r` is an external scalar
label from a different table with a different schema, not a second embedding, so there is no
cosine similarity to test and no reason a correctly-aligned label vector would look more
"similar" to the embedding under any row-independent statistic. The row-alignment proof this
document reports is a purpose-built test for exactly this join, not a re-application of that
existing check.

## 2. The frozen rule, quoted

Every value below is the verbatim committed value in `notebooks/pu_manifold/physics_labels.py`
at freeze commit `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5` (`09-PREREGISTRATION.md`):

```
ALIGNMENT_SHIFT_SET = (
    -1000, -100, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000,
)
```

```
ALIGNMENT_N_PERMUTATIONS = 20
ALIGNMENT_PERMUTATION_SEED = 20260902
ALIGNMENT_MARGIN_R2 = 0.10
```

```
ALIGNMENT_PASS_RULE = (
    "passed is True iff gap = r2_shift0 - best_other_r2 is STRICTLY greater than "
    "ALIGNMENT_MARGIN_R2; a gap exactly equal to the margin FAILS."
)
```

The comparison is a strict `>` on the gap — a gap exactly equal to `ALIGNMENT_MARGIN_R2 = 0.10`
FAILS, not passes. `ALIGNMENT_LABEL = "mag_r"` is the sole label the curve is run against
(D9-06); `mag_r` is `PRIMARY_LABEL`, the gating label, mapped to the raw column `mag_r_desi` per
the `09-04` ratification.

## 3. The measured curve

45 out-of-fold ridge fits (`ALPHA_RIDGE = 100.0`, 5-fold OOF) on 86,471 rows each: shift 0, the 24
frozen shifts (in `ALIGNMENT_SHIFT_SET` order), and 20 seeded permutations. Every alignment —
including the `+/-1000` shifts, the widest — scored exactly 86,471 finite pairs
(`n_finite = 86471` on every row, D9-07 adjacency confirmed against real data).

**Shift curve, frozen order, shift 0 first:**

| shift | r2 | n_finite |
|---:|---:|---:|
| 0 | 0.515931 | 86471 |
| -1000 | -0.000379574 | 86471 |
| -100 | -0.000269473 | 86471 |
| -10 | -0.000282517 | 86471 |
| -9 | -0.000122996 | 86471 |
| -8 | -0.000392456 | 86471 |
| -7 | -0.000339864 | 86471 |
| -6 | -0.000628738 | 86471 |
| -5 | -0.000383625 | 86471 |
| -4 | -0.000174089 | 86471 |
| -3 | -0.000292646 | 86471 |
| -2 | -0.000316872 | 86471 |
| -1 | -0.000349167 | 86471 |
| 1 | -0.000187667 | 86471 |
| 2 | -0.000357190 | 86471 |
| 3 | -0.000199375 | 86471 |
| 4 | -0.000322108 | 86471 |
| 5 | -0.000237974 | 86471 |
| 6 | -0.000229943 | 86471 |
| 7 | -0.0000820780 | 86471 |
| 8 | -0.000486142 | 86471 |
| 9 | -0.000337187 | 86471 |
| 10 | -0.000278615 | 86471 |
| 100 | -0.000284877 | 86471 |
| 1000 | -0.000102329 | 86471 |

Shift 0's R2 (0.515931) is roughly three orders of magnitude above every one of the 24 shifted
R2 values, which cluster tightly between -0.000628738 (shift -6) and -0.0000820780 (shift 7, the
best of the 24) — all effectively zero relative to shift 0, consistent with a correct positional
join and no detectable structure at any misalignment.

**Permutations — summary:** minimum -0.000585684, median -0.000309451, maximum -0.0000328360
(20 draws, seed 20260902). Full appendix:

| draw | r2 | n_finite |
|---:|---:|---:|
| 0 | -0.000428158 | 86471 |
| 1 | -0.000155980 | 86471 |
| 2 | -0.000323908 | 86471 |
| 3 | -0.000330268 | 86471 |
| 4 | -0.000267189 | 86471 |
| 5 | -0.000299787 | 86471 |
| 6 | -0.000465970 | 86471 |
| 7 | -0.000177798 | 86471 |
| 8 | -0.0000328360 | 86471 |
| 9 | -0.000425824 | 86471 |
| 10 | -0.000304515 | 86471 |
| 11 | -0.000186625 | 86471 |
| 12 | -0.000316400 | 86471 |
| 13 | -0.000355733 | 86471 |
| 14 | -0.000298289 | 86471 |
| 15 | -0.000286307 | 86471 |
| 16 | -0.000314386 | 86471 |
| 17 | -0.000585684 | 86471 |
| 18 | -0.000422662 | 86471 |
| 19 | -0.000137172 | 86471 |

The permutation R2 range sits inside the same near-zero band as the shift curve, giving no
indication that any of the 44 alternative pairings carries embedding-to-label structure the
positional join does.

## 4. The verdict

Only one run of `--mode proof` was needed (shift 0 passed on the first attempt), so there is a
single verdict row to transcribe:

| run_commit | timestamp_utc | R2(shift 0) | best other alignment | best other R2 | gap | margin | PASS/FAIL |
|---|---|---:|---|---:|---:|---:|---|
| `ee992bac947f3469dfb0e607867901992f0b17de` | 2026-09-04T05:07:25Z | 0.5159312856012054 | shift 7 | -8.20780445989211e-05 | 0.5160133636458043 | 0.10 | **PASS** |

`gap = 0.5160133636458043` strictly exceeds `ALIGNMENT_MARGIN_R2 = 0.10` by a wide margin
(5.16x the margin itself) — this is not a borderline pass. Verdict: **PASS**, exit code 0.

## 5. Failure-branch classification

**Not applicable.** Shift 0 PASSED on the frozen margin; `--mode search` was correctly not run
(the runner itself refuses to run search after a passing proof — "nothing to search for"), and
no `CANDIDATE OFFSET` / `AMBIGUOUS` / `NO ALIGNMENT FOUND` classification exists for this run.
Per D9-08 and the frozen `ALIGNMENT_SEARCH_RULE`, the SEARCH branch and its adoption machinery
are reserved for the case where shift 0 fails; they do not apply here.

## 6. Scale caveat

`K_NEIGHBOURS = 2048` of `n = 86,471` Physics rows is **1/42** of the sample — this phase's
frozen `NEIGHBOURHOOD_RATIO_RULE` (`physics_curvature_probe.py`): *"K_NEIGHBOURS=2048 of
n=86,471 is 1/42 of the Physics sample (his 2048 of 16,384 was 1/8); this ratio must be printed
beside every number this phase reports."* The colleague's own `k=2048` was drawn against a
16,384-row hash-selected subset of the same 86,471-row Physics test set — **1/8** of it, more
than five times denser a neighbourhood than this phase's own.

This document is the first in Phase 9 to carry a real number, so the ratio is stated here as a
premise every later number inherits, not as a footnote deferred to the findings. His own scale
table (`09-COLLEAGUE-REANALYSIS.md`) makes the ratio load-bearing rather than cosmetic: his
controlled 3-control partial at `d=16` was `-0.027` at `k=1024`, `-0.080` (p=0.37) at `k=1536`,
and `-0.2405` only at his largest, `k=2048` — his association exists **only at his largest k**,
one row in a monotone-in-k progression, not a value that holds across scale. A phase running at
`k=2048` but at 1/42 sample density rather than 1/8 is running a materially different
neighbourhood-scale experiment from the one that produced his `-0.240`; any comparison between
this phase's later curvature-decodability numbers and his own must carry this difference
explicitly, not treat the two `k=2048` values as the same measurement at a different rho.

## 7. Ruling

Pending — see Task 3's checkpoint.
