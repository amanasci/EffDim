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

## Ruling

Pending — see Task 3's checkpoint.
