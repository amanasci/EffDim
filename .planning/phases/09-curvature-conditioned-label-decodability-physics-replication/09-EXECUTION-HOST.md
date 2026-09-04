# Phase 9 Execution Host Runbook

**Written:** 2026-09-03 (plan 09-06, Task 2)
**Freeze commit:** `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5`

This document is a runbook a developer follows, by hand, on a machine Claude cannot reach. It
takes a fresh, empty machine from `git clone` to a green smoke bundle, then lists every real run
in order with the literal freeze commit written into each command line, then lists what comes
back and how it is transferred, then states what must never be done on the host.

Refer to the host generically throughout this document. No hostname, IP address, username or SSH
key path appears anywhere below, and none should be added later — see Section 7.

## 1. What this is and why

Phase 9's real numbers are not produced on the developer's own machine. `09-CONTEXT.md`'s
Specifics section (amended 2026-09-02) states the execution host is either an SSH remote server
or the colleague's own machine — undecided at planning time. Claude has no credential for either,
so the fresh-clone bootstrap below and every real-number run after it are executed by the
developer, by hand, and the resulting artifacts are transferred back as files.

The two runner scripts encode this directly: `physics_curvature_probe.EXECUTION_HOST_RULE`
states "No real number is produced on the developer's machine... `--mode smoke` and `--mode
manifest` are the only modes that run locally; every mode that produces a real number is gated
on `_strict_ancestor_or_exit`'s strict-ancestor freeze proof." Concretely:

- **`--mode smoke`** (both runners) — a pure in-memory exercise on synthetic arrays. No network
  read, no `--freeze-commit` required, no Physics number. Safe anywhere, including the
  developer's own machine.
- **`--mode manifest`** (the row-alignment runner) — already run for real in plan 09-04, before
  the freeze existed. Not re-run by this runbook.
- **Every other mode** — `proof`, `search`, `dsweep`, `positive-control`, `shuffled-label`,
  `verdict`, `seeds`, `bundle` — is gated on `_strict_ancestor_or_exit`, which refuses to run
  without `--freeze-commit` naming this document's freeze SHA, resolved through `git rev-parse`
  and checked as a STRICT ancestor of `HEAD` (`git merge-base --is-ancestor` AND
  `git rev-list --count` at least 1 — a commit is its own ancestor, so the count check is not
  redundant). This is what makes it safe to hand this document to a machine nobody has audited:
  the gate itself refuses to compute anything without proof that the clone descends from the
  frozen commit, not merely contains it.

## 2. Host requirements

- **Network reachability to `huggingface.co`** — both datasets (`UniverseTBD/pu-embeddings` and
  `Smith42/galaxies`) are read from there. No other network endpoint is required.
- **A Python environment** with `torch`, `numpy`, `scipy`, `scikit-learn`, `pyarrow`, `pandas` and
  `datasets` installed — see Section 3 for the exact install command. `huggingface_hub` arrives
  transitively via `datasets`.
- **Roughly 1 GB free disk** for the HuggingFace cache (the physics embedding parquet alone is
  ~245 MB) plus a few MB for the JSONL records and anchor-table `.npz` files this phase writes.
- **CPU only — no GPU required.** `07-CONTEXT.md` Section 7's own cost model, which Section 5
  below scales from, was measured CPU-only.
- **The host's core count is unknown at planning time.** Every cost figure in Section 5 is stated
  per thread, never as an unqualified wall-clock; `--print-cost-model` prints the assumed thread
  count and the host's own `os.cpu_count()` side by side so a mismatch is visible immediately.

## 3. Fresh-clone bootstrap

Every step below must work with no file copied from the developer's own machine — no cached
HuggingFace download, no `.venv`, no absolute path.

```bash
# 1. Clone the pushed branch, single-branch, into an empty directory.
git clone --single-branch --branch fixture-validity-audit \
  https://github.com/amanasci/EffDim.git
cd EffDim

# 2. Verify the freeze commit is present and is a STRICT ancestor of HEAD -- both checks, not
#    either alone (a commit is its own ancestor, so is-ancestor alone would pass even if HEAD
#    were the freeze commit itself).
git merge-base --is-ancestor 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 HEAD \
  && echo "is-ancestor: OK" || echo "is-ancestor: FAILED -- stop here, do not proceed"
git rev-list --count 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5..HEAD
# ^ must print an integer >= 1. A 0 means HEAD IS the freeze commit -- stop, do not proceed.

# 3. Create and populate the Python environment (the exact dependency set 09-RESEARCH.md's
#    Standard Stack names, pinned in the repo's own requirements file -- no separate list to
#    keep in sync).
python3 -m venv .venv
.venv/bin/pip install -r notebooks/requirements-notebooks.txt

# 4. Optional environment overrides -- unset means today's default behaviour, unchanged.
#    export HF_HOME=/some/larger/disk/huggingface-cache
#    export EFFDIM_09_OUTPUT_ROOT=/some/scratch/disk/effdim-phase9

# 5. See the cost model before running anything real, naming the host's own thread count.
.venv/bin/python notebooks/diagnostics/09_physics_curvature_run.py \
  --print-cost-model --threads <n>

# 6. Smoke-test both runners. Both must end with their PASS banner.
.venv/bin/python notebooks/diagnostics/09_physics_curvature_run.py \
  --mode smoke --record-path notebooks/.cache/09_scratch_tracer.jsonl
.venv/bin/python notebooks/diagnostics/09_row_alignment_proof_run.py \
  --mode smoke --record-path notebooks/.cache/09_scratch_alignment.jsonl
```

If step 2's ancestry check fails, stop — do not proceed to step 3 or beyond, and report the
failure back rather than attempting a repair on the host (Section 8).

**Verified against a real host: see Section 9, "Host as bootstrapped."**

## 4. The run sequence

Every command below carries the literal freeze commit `5f7fbe27afb0ef2a76353b41fa5713e760bbeea5`
— the host never has to resolve it itself, and `_strict_ancestor_or_exit` checks the supplied
value against the runner's own compiled-in `FREEZE_COMMIT_SHA` by exact string equality, so a
typo or an abbreviation is rejected rather than silently accepted. Run these **serially**, never
concurrently — `07_crossmodal_curvature_run.py`'s own precedent measured three concurrent torch
jobs driving load up roughly 10x on a 20-core machine; both runners cap threads via
`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`NUMEXPR_NUM_THREADS` but that caps a single job's own
threads, not how many jobs run at once.

1. **The alignment proof (D9-06/D9-07).**
   ```bash
   .venv/bin/python notebooks/diagnostics/09_row_alignment_proof_run.py \
     --mode proof --freeze-commit 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 --threads <n>
   ```
   Writes `09_row_alignment.jsonl` (one row per shift/permutation curve entry, then one verdict
   row). Dominated by network I/O (both datasets download in full on this run, since a fresh
   clone has no cache); Section 5 quotes the measured download wallclock. A FAIL here (shift 0
   does not exceed every shifted R2 by `ALIGNMENT_MARGIN_R2 = 0.10`) is a **real terminal
   outcome**, not an error — report it exactly as printed, do not retry with different arguments.

2. **The conditional search (D9-08), only if step 1 FAILED.**
   ```bash
   .venv/bin/python notebooks/diagnostics/09_row_alignment_proof_run.py \
     --mode search --freeze-commit 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 --threads <n>
   ```
   Classifies which shifted alignments (if any) clear the margin: exactly one is a CANDIDATE
   OFFSET, two or more is AMBIGUOUS, zero is NO ALIGNMENT FOUND. Both AMBIGUOUS and NO ALIGNMENT
   FOUND halt the phase here — adopting an offset is a separate, blocking developer decision
   plus a fresh freeze, never made by this runner. Do not run this step if step 1 PASSED; the
   runner itself refuses (`--mode search` exits 2 naming "nothing to search for" when the proof
   record's verdict already passed).

3. **The Wave A sweep (D9-12).**
   ```bash
   .venv/bin/python notebooks/diagnostics/09_physics_curvature_run.py \
     --mode dsweep --freeze-commit 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 --threads <n>
   ```
   One serial in-process loop over `D_SWEEP = (16, 20, 25, 32)`: fit the autoencoder on the full
   86,471-row Physics sample, evaluate curvature at the 512 frozen anchors, decompose radial vs.
   tangential, run the 3-control partial and its Freedman-Lane null. Writes
   `09_physics_curvature.jsonl` plus one `.npz` anchor table per `d`. Section 5's cost table is
   this step's estimate; a FAIL at a given `d` is per-`d` and does not stop the sweep from
   continuing to the next `d`.

4. **The two gates.**
   ```bash
   .venv/bin/python notebooks/diagnostics/09_physics_curvature_run.py \
     --mode positive-control --freeze-commit 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 --threads <n>
   .venv/bin/python notebooks/diagnostics/09_physics_curvature_run.py \
     --mode shuffled-label --freeze-commit 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 --threads <n>
   ```
   `positive-control` plants a target correlation at each of `POSITIVE_CONTROL_TARGET_RHOS =
   (0.05, 0.1, 0.15, 0.2, 0.25)` and checks the controlled-partial statistic recovers it — an
   instrument-sanity gate, not a Physics result. `shuffled-label` repeats a global label shuffle
   `SHUFFLED_LABEL_REPEATS = 20` times and checks the resulting statistic behaves like a null —
   a null-calibration gate. Both reuse the `H_tan_norm` field `--mode dsweep` already computed
   (no autoencoder re-fit), so their added cost is small next to step 3's. A FAIL on either gate
   means the *instrument*, not Physics, failed — report it and stop; do not read a `--mode
   dsweep` number as meaningful until both gates pass.

5. **The verdict print.**
   ```bash
   .venv/bin/python notebooks/diagnostics/09_physics_curvature_run.py \
     --mode verdict --freeze-commit 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5
   ```
   Reads the already-written `09_physics_curvature.jsonl` and prints the per-`d` verdicts plus
   the aggregated phase verdict (`VERDICT_RULE`, transcribed in full in `09-PREREGISTRATION.md`).
   Computes nothing new; touches no network.

6. **The conditional Wave B (D9-17), only at `d` values where Wave A's per-`d` verdict fired.**
   ```bash
   .venv/bin/python notebooks/diagnostics/09_physics_curvature_run.py \
     --mode seeds --freeze-commit 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 --threads <n>
   ```
   `WAVE_B_TRIGGER_RULE`: runs the three-seed sweep (`TORCH_INIT_SEEDS_WAVE_B = (0, 1, 2)`) only
   at `d` values where Wave A's per-`d` verdict fired; `d` values where Wave A did not fire are
   never re-run. Unanimity across all three seeds keeps the shared per-`d` verdict; anything else
   reports `SPLIT ACROSS SEEDS` (`SEED_VERDICT_COMBINATION_RULE`) — never averaged, never
   upgraded by majority vote. If Wave A fired at zero `d` values, skip this step entirely; there
   is nothing for it to run.

## 5. Cost table

Per-`d` figures in **core-hours**, split into training and curvature components so the reader
can see which dominates (`print_cost_model`'s own header names the thread count and the host's
own core count at run time; the table below is this document's static snapshot at three
illustrative thread counts). Derived from Phase 7's measured `DSWEEP_COST_MODEL_MINUTES`
(`07-CONTEXT.md` Section 7, an 8-thread cap, 10,000 rows, curvature evaluated at every row),
scaled two ways: training scales linearly by rows (86,471 Physics rows over Phase 7's 10,000,
~8.65x); curvature scales by the anchor-evaluation ratio (512 anchors over Phase 7's every-row
evaluation, 0.0512x) — D9-04's single biggest cost difference from Phase 7, and the reason
training, not curvature, now dominates (the reverse of Phase 7's own shape). `d=16` has no entry
in Phase 7's own table; its relative-cost multiplier is derived from Phase 7's own stated `D*d^2`
scaling law, consistent with the measured ratios Phase 7 recorded for `d=25`/`d=32`.

**These are ESTIMATES. Plan 09-08 replaces this table with the measured figure from the real
Wave A run.**

| `d` | training core-hr | curvature core-hr | total core-hr | wall-clock, 4 threads | 8 threads | 16 threads |
|---|---|---|---|---|---|---|
| 16 | 7.187 | 0.106 | 7.293 | 1.82 h | 0.91 h | 0.46 h |
| 20 | 7.187 | 0.166 | 7.352 | 1.84 h | 0.92 h | 0.46 h |
| 25 | 7.187 | 0.259 | 7.446 | 1.86 h | 0.93 h | 0.47 h |
| 32 | 7.187 | 0.424 | 7.611 | 1.90 h | 0.95 h | 0.48 h |
| **sweep total** | **28.749** | **0.955** | **29.705** | **7.43 h** | **3.71 h** | **1.86 h** |

- **The alignment proof** (step 1) runs 45 out-of-fold ridge fits on the full 86,471-row, 768-
  feature embedding — a closed-form linear solve, not an iterative training loop, and not
  separately measured; expected to be small next to this step's own dominant cost, the dataset
  download. **Quoted download wallclock:** `09-DATA-MANIFEST.md`'s own full-scale `--mode
  manifest` run took 25m47s wall-clock on a fresh (uncached) HuggingFace read of both datasets
  (`09-03-SUMMARY.md`) — the figure a fresh clone with no prior cache should expect for step 1's
  own first HuggingFace read, since this host has never run any Phase 9 command before.
- **The two gates** (step 4) reuse the already-computed `H_tan_norm` field rather than re-fitting
  the autoencoder; their added cost is a handful of controlled-partial/permutation-null
  recomputations across `POSITIVE_CONTROL_TARGET_RHOS` (5 targets) and `SHUFFLED_LABEL_REPEATS`
  (20 repeats) — expected in the minutes, not hours, and not separately measured yet.
- **Wave B** (step 6), if triggered, costs up to 3x a single-`d` sweep entry for each `d` where
  Wave A fired (the three seeds of `TORCH_INIT_SEEDS_WAVE_B`), not 3x the whole table above —
  only the fired `d` values are re-run.

## 6. Artifact return

Run the bundle mode once every step above that is going to run has finished (or once the run is
interrupted and the partial evidence needs to come back — `--mode bundle` exits 0 either way):

```bash
.venv/bin/python notebooks/diagnostics/09_physics_curvature_run.py \
  --mode bundle --host-label <a short name>
```

This prints the archive's path, its byte size, and its SHA-256. **Transfer the printed archive
back and report its SHA-256 alongside it** — the digest is re-checked locally against the
received file before anything is read out of it (T-09-36); a truncated or altered transfer is
caught here, before a number is ever read.

The archive (`09-artifacts-<host-label>-<UTC-stamp>.tar.gz`) contains every `09_`-prefixed file
under the resolved output root, plus an `environment.json` member recording the host's core
count, thread cap, Python and library versions, the resolved HuggingFace cache directory and
output root, `git describe` of `HEAD`, and the freeze SHA. What each contained file is for:

| Returned file | Ingested by |
|---|---|
| `09_row_alignment.jsonl` | 09-07 (the alignment proof's curve and verdict) |
| `09_physics_curvature.jsonl` | 09-08 and 09-09 (the dsweep/gate/verdict/seed rows) |
| `09_physics_curvature.npz` (per-`d` anchor tables) | 09-08 and 09-09 |
| `09_data_manifest.jsonl` | already ingested by 09-05 (present in the bundle only because it still lives under the output root; no re-ingestion needed) |
| `environment.json` (inside the archive) | every SUMMARY from 09-06 onward, so a reported wall-clock always carries the hardware that produced it (T-09-41) |

`--mode bundle` is safe to run more than once — its own archive name never collides with the
`09_`-prefix glob it bundles (a hyphen after `09`, never an underscore), so re-running it after
more artifacts exist just produces a second, later-stamped archive.

## 7. What must NOT be done on the host

- **Do not edit any file under `notebooks/pu_manifold/`.** That is the sealed, frozen module
  tree; `09-PREREGISTRATION.md`'s closing rule makes any edit after a Physics number exists a
  pre-registration BREACH, remediable only by a fresh freeze commit, a numbered amendment
  document, and a complete re-run of everything the changed value touched.
- **Do not pass a `--freeze-commit` other than the one in this document.**
  `_strict_ancestor_or_exit` checks the resolved value against the runner's own compiled-in
  `FREEZE_COMMIT_SHA` by exact string equality and rejects anything else, including a genuine
  earlier ancestor — this is enforced by the code, not merely requested here.
- **Do not re-run a mode with different constants to get a different answer.** Every constant
  this phase reads is frozen in the clone itself; there is nothing to vary without editing a
  sealed module, which the first bullet above already forbids.
- **Do not create or commit anything on the host beyond the artifacts listed in Section 6.** The
  host runs a read-only checkout of a pushed branch. The artifacts come back as files, carried by
  hand, never as commits made on the host and pushed from it.
- **Do not write any hostname, IP address, username or SSH key path into any file this phase
  produces.** The recorded host identity is limited to core count, OS, Python version and library
  versions (Section 8 of this document, once appended) — capability, never an address.

## 8. If something goes wrong

**The three gates:**

- **The alignment proof (step 1) FAILs.** Shift 0's out-of-fold R2 did not exceed every shifted
  R2 by `ALIGNMENT_MARGIN_R2`. This is a real terminal outcome about whether the two datasets are
  positionally aligned at all, not a bug — run step 2 (`--mode search`) next, and report both
  outcomes together.
- **The positive-control gate fails** (the controlled-partial statistic does not recover the
  planted correlation at the target rhos). This means the *instrument* — the autoencoder plus
  curvature field plus controlled-partial statistic, as wired for Phase 9 — is not sensitive
  enough to detect a real effect of the planted size, regardless of what step 3's dsweep found.
  Report the full stdout; do not proceed to trust any `--mode dsweep` number as meaningful.
- **The shuffled-label gate fails** (the null repeats behave unlike a well-calibrated null). This
  means the Freedman-Lane FWER control itself is miscalibrated for this data — the `p_fwer`
  values step 3 wrote cannot be trusted as stated. Report the full stdout; do not proceed to read
  a verdict off step 3's numbers.

**The two loader failures:**

- **Row-count mismatch.** Either dataset returns a row count other than the frozen
  `EXPECTED_N_PHYSICS_ROWS = 86471`. This usually means the pinned revision changed upstream
  between planning and this run — report the exact counts observed; do not proceed on a
  mismatched row count, since the positional join between embeddings and labels depends on both
  sides having exactly this many rows in the same order.
- **Missing column at the pinned revision.** A candidate label column is absent from
  `Smith42/galaxies`, revision `v2.0`. `09-RESEARCH.md` Pitfall 1 records that the public default
  revision (`main`) carries only `image` and `dr8_id`, no catalog columns — if this error
  appears, first confirm the clone is genuinely requesting revision `v2.0` and not silently
  falling back to `main` before reporting it as a real upstream change.

**In every case above: return the full stdout of the failing step, together with whatever
partial bundle exists (Section 6's `--mode bundle` exits 0 on a partial set precisely for this
reason), rather than attempting a repair on the host.** A repair on an unaudited host is itself a
new source of the exact kind of undocumented drift this freeze discipline exists to prevent.

## 9. Host as bootstrapped

**Recorded 2026-09-04 UTC.** Section 3's fresh-clone bootstrap was executed on a real execution
host and is now **verified**: the freeze proof held, both smoke modes passed, and a bundle
transferred back cleanly. Host identity is recorded as capability only — core count, OS, Python
and library versions, thread count — never as an address, per Section 7.

Per the developer's instruction (2026-09-04 UTC): *"begin with running experiments on ssh
server. ensure you use AVAILABLE compute, don't kick someone off if they are already using.
check free compute with nvidia-smi. adhere strictly to the user-guide."* The bootstrap steps
below were executed over SSH by the orchestrator acting on that instruction, following the
host's own user guide, not typed interactively by the developer.

**Host capability (measured, not assumed):**

- **OS:** Ubuntu 22.04.5 LTS
- **Core count:** `os.cpu_count()` / `nproc` report 128. cgroup CPU limit unlimited (cgroup v2
  `cpu.max` = `-1 100000` at survey time). RAM: 1006 GB total, ~836 GB free.
- **GPU:** 8x NVIDIA A100-SXM4-80GB, all idle (0 MiB used, 0% util) at survey time — **not
  used**; Phase 9 is CPU-only. Load average at survey: ~4 on 128 cores, no other live compute
  jobs (only defunct zombie processes). Thread count chosen for the cost model and smoke runs:
  **16**, leaving the remaining cores free for other users of the shared host, per the
  developer's "don't kick someone off" instruction.
- **Python:** 3.14.7
- **Library versions:** torch 2.13.0+cpu, numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0,
  pyarrow 25.0.1, pandas 3.0.5, datasets 5.0.1. (The local development machine differs only in
  pyarrow 25.0.0 — not a Phase 9-relevant discrepancy.)

**Bootstrap deviation from Section 3, step 3 (must be recorded):** the host's system `python3`
is 3.10.12. The pins `numpy==2.5.1` and `scipy==1.18.0` require Python >= 3.12 per their PyPI
`requires_python` metadata, so `python3 -m venv .venv` as literally written cannot satisfy
`requirements-notebooks.txt` on this host. Per the host's own persistent-environment recipe in
its user guide, a Python 3.14.7 interpreter was provisioned first
(`mamba create -p <persistent-root>/env python=3.14 pip`), and Section 3's `.venv` was then
created **from that interpreter** (`<persistent-root>/env/bin/python -m venv .venv`) so every
other command in Section 3 and Section 4 runs literally as written, via `.venv/bin/python`. The
dependency install took 36 minutes wall-clock. No sealed module (`notebooks/pu_manifold/`) was
touched by this deviation; `git status` in the host clone is clean.

**Clone.** `git clone --single-branch --branch fixture-validity-audit` over HTTPS into a
persistent directory on the host. HEAD at clone time: `ee992bac947f3469dfb0e607867901992f0b17de`.
Freeze proof on the host, both checks:
- `git merge-base --is-ancestor 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5 HEAD` → exit 0
  (`is-ancestor: OK`)
- `git rev-list --count 5f7fbe27afb0ef2a76353b41fa5713e760bbeea5..HEAD` → `5`

**Environment overrides used** (Section 3, step 4), both pointed at the host's persistent disk:
`HF_HOME=/mnt/ssd-cluster/effdim/hf-cache`, `EFFDIM_09_OUTPUT_ROOT=/mnt/ssd-cluster/effdim/phase9-out`.
The host's home directory is ephemeral and wiped on unannounced restarts; nothing this phase
needs lives there.

**Cost-model output** (`--print-cost-model --threads 16`), verbatim:

```
Phase 9 cost model -- CORE-HOURS, portable across hosts. threads=16 host_core_count=128
   d    training core-hr   curvature core-hr   total core-hr    wallclock@16t (hr)
  16               7.187               0.106           7.293                 0.456
  20               7.187               0.166           7.352                 0.460
  25               7.187               0.259           7.446                 0.465
  32               7.187               0.424           7.611                 0.476
```

**Physics smoke** (`--mode smoke`, 12.3 s wall-clock): all seven stages PASS — `ae_fit` 0.9893
(> 0.7), `radial_decomposition` -1.8809, `knn` 64, `oof_ridge` 2000, `local_r2` 128,
`controlled_partial` -0.0014, `permutation_fwer` 0.99005. Banner: **`SMOKE PASS`**, exit 0.

**Alignment smoke** (`--mode smoke`): aligned case `argmax_shift=0`, `r2_shift0=0.9949`,
`passed=true`; offset case `injected_offset=5`, `clearing_alignments=[5]`, `passed=true`
(the archive's own record — both smoke cases individually report `passed: true`, consistent
with the overall banner). Banner: **`ALIGNMENT SMOKE PASS`**, exit 0. One wording note, not a
defect: the smoke-mode banner also printed "Every gating constant in
physics_labels/physics_curvature_probe is still UNSET" — stale text left over from before the
freeze; smoke mode reads no frozen constant, so this does not affect the result, but it should
be cleaned up in a future plan touching that banner.

**Returned bundle.** Archive `09-artifacts-pod128-20260904T044351Z.tar.gz`, 883 bytes.
SHA-256 (as reported by the host and re-verified locally over the received file — both match):
`20c6a8ba28f3b9b95ba9e01164520a3f5d33fdcc5f1949146fc5c3aeb99338cd`. Contents: `09_scratch_alignment.jsonl`,
`09_scratch_tracer.jsonl`, `environment.json`. Extracted locally under the resolved local output
root (`notebooks/.cache/`, unset `EFFDIM_09_OUTPUT_ROOT` on the local machine). Every extracted
record's `mode` value, where present, is `smoke`; no `verdict`, `phase_verdict` or
data-derived `passed` key appears anywhere in the archive. No production Physics file
(`09_row_alignment.jsonl`, `09_physics_curvature.jsonl`) exists locally or in the archive.

**Section 3's "Fresh-clone bootstrap" is now verified against this host**, with the one
recorded deviation above (Python interpreter provisioning) and no other departure from the
literal steps.
