---
status: complete
phase: 03-decoder-curvature-field
created: 2026-08-14
supplements: |
  notebooks/pu_manifold/cae.py, notebooks/pu_manifold/chart_curvature.py,
  notebooks/diagnostics/curvature_field_pu_run.py (03-07-PLAN.md's runner),
  notebooks/diagnostics/swiss_roll_curvature_sweep_run.py (03-01-PLAN.md's runner),
  notebooks/pu_manifold/tests/test_curvature_probe.py
trigger: |
  03-07's timing probe measured the nine-cell PU grid at ~5.6-5.7h against D-13's 5-hour
  envelope (training ~16,100-16,200s dominates the total; curvature ~4,000-4,040s in reverse
  mode, 2,000-row holdout per cell). The developer has GPU access and chose to add opt-in
  device support rather than narrow the sweep, drop seeds, or accept the overrun -- narrowing
  or dropping seeds would both have damaged the pre-declared 3x3 design (03-07-SUMMARY.md);
  this option preserves it.
commits:
  - 3aeb27d feat(03-07-supplement): thread optional CUDA device support through cae.py and chart_curvature.py
  - 036b762 feat(03-07-supplement): add --device to swiss_roll_curvature_sweep_run.py
  - 0166baa feat(03-07-supplement): add --device to curvature_field_pu_run.py, device-aware timing probe
  - 0286001 test(03-07-supplement): add CPU/CUDA device-parity test for chart curvature
---

# Phase 3 Plan 07 -- Supplement 1: opt-in CPU/GPU device support

**Developer-directed work, sits between plans 03-07 and 03-08. Not a numbered plan.** Same
artifact convention as `03-02-AMENDMENT-01.md`: this document is committed alongside the code
it describes, states exactly what was built and why, and every hard constraint below carries
the actual verification evidence, not just a claim that it was checked.

## 1. Why

`03-07`'s timing probe (`--timing-probe --pu-n 200`) measured the nine-cell PU grid at
**~5.6-5.7 hours**, over D-13's 5-hour envelope. Training dominates
(~16,100-16,200s across the nine cells) with curvature a smaller but real term
(~4,000-4,040s, reverse mode, 2,000-row holdout per cell). `03-07-SUMMARY.md` named three
options and left the choice to the developer: narrow `PU_N_CHARTS_SWEEP`, drop to two seeds,
or accept the longer run. Narrowing the sweep or dropping seeds would both change the
pre-declared 3x3 design this milestone has been careful to state before any PU number exists
(D-07, D-12); adding GPU support preserves the design and moves the bottleneck instead of
cutting it.

## 2. Hard constraints and how each was verified

**1. Default device is `"cpu"`. Zero behaviour change on CPU.**
Every new `device` parameter defaults to `torch.device("cpu")`. Every internal tensor-creation
site now derives its device from an existing input tensor, module parameter, or the resolved
CLI device rather than defaulting to CPU implicitly -- `.to(torch.device("cpu"))` is a no-op on
an already-CPU tensor, so the CPU path is byte-for-byte what it was before this supplement.

**2. RNG consumption order on CPU is unchanged.** The highest risk in this task. Every model
constructor call (`cae.ChartAutoEncoder(...)`, `cae.PlainAutoEncoder(...)`) is followed by
`.to(device)` **after** construction, never by passing `device=` into a layer constructor or
initializer -- `build_cae`, `build_control`, and both runners' `run_cell`/model-construction
sites all follow this discipline explicitly (see each function's docstring/inline comment).
`torch.manual_seed(seed)` therefore consumes CPU RNG in exactly the order it always did.

Internal tensor-creation sites that use their own local RNG stream independent of
`torch.manual_seed` (`farthest_point_sample`'s and `unfaithfulness_coverage`'s
`torch.Generator()`, and `chart_curvature.randomized_trace_mean_curvature_nongating`'s probe
generator) are left as CPU-only generators on purpose -- their seed-to-draw mapping never
depended on device, and moving the *generator* to a device would be an unforced, unnecessary
RNG change. Only the *drawn values* are moved to the working tensor's device afterward, via an
explicit `.to(device)` call at each site.

**3. The anchor reproduces.** Verified three times across this supplement's four commits (once
before any change, once after `cae.py`/`chart_curvature.py`'s device plumbing, once after the
Swiss roll runner's `--device` flag), using a scratch `--record-path` to force fresh
computation without touching or resuming into the real cache:

```
.venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py \
  --n-charts 8 --seeds 0 --max-combos 1 --record-path <scratch>.jsonl
```

All three runs: `rho_chart = -0.06041003026778113`, exact match to the value stated in this
supplement's brief, verified via `assert rec['rho_chart'] == -0.06041003026778113` against the
JSONL record's full-precision float each time. It did not change at all.

**4. `_assert_float64` is untouched.** No line in `chart_curvature.py` was changed inside
`_assert_float64`; curvature still refuses anything but float64 on every device, with no
float32 curvature path added and no device-conditional relaxation. `model.double()` runs
identically before curvature regardless of `device` -- casting dtype and moving device are two
separate, independently-applied operations in every call site this supplement touches.

**5. All 286 existing tests pass unchanged.** `.venv/bin/python -m pytest
notebooks/pu_manifold/tests/ -q` was run after every commit in this supplement (five times
total, including the pre-change baseline) and stayed at **286 passed** with no test modified,
relaxed, or skipped. The one new test added by this supplement
(`test_chart_curvature_cpu_cuda_agree_to_float64_tolerance`) reports as `1 skipped` on this
CPU-only machine -- final count `286 passed, 1 skipped`. No existing bit-identity
(`torch.equal`) test was touched or made device-parametrized; the reverse-mode golden-array
test (`test_chart_curvature_reverse_mode_is_bit_identical_to_sealed_baseline`) stays exactly
as it was, CPU-scoped, unchanged.

## 3. What was built

**Device plumbing (`notebooks/pu_manifold/cae.py`, `notebooks/pu_manifold/chart_curvature.py`).**
Every internal tensor-creation site that previously defaulted to CPU now derives its device
from an existing tensor or module parameter: `ChartAutoEncoder.reconstruct`'s index tensor,
`_chart_encoder_spectral_product`'s accumulator, `farthest_point_sample`'s distance buffer and
cross-tensor indexing, `fps_pretrain_loss`'s index tensor, `train_cae`'s minibatch-permutation
and FPS seed-chart-index tensors, `timing_probe`'s permutation, `_train_decoder_protocol`'s
permutation, `unfaithfulness_coverage`'s sample/coordinate/decode buffers,
`arrays_to_state_dict`'s reconstructed tensors, and `chart_curvature_field`'s output buffers
and `randomized_trace_mean_curvature_nongating`'s accumulator and probe vector. No model
constructor gained a `device=` parameter; every model is moved with `.to(device)` strictly
after construction (constraint 2).

**`--device` on both runners** (`curvature_field_pu_run.py`, `swiss_roll_curvature_sweep_run.py`).
Accepts `cpu` (default), `cuda`, or `cuda:N`. A shared `resolve_device()` helper (duplicated,
not imported across the two standalone scripts, per this project's simplicity preference)
validates the string and raises a `SystemExit` naming the installed CPU-only torch build and
the exact fix if `cuda` is requested but `torch.cuda.is_available()` is `False`.

**`device` and `torch_version` recorded in every cache record.** `grid_cell`, `control_cell`,
`smoke_cell` (all via `_run_cae_cell`/`_run_control_cell`), and `timing_probe` records in
`curvature_field_pu_run.py`; every per-cell record in `swiss_roll_curvature_sweep_run.py`. A
GPU-run cell and a CPU-run cell are distinguishable after the fact by reading these two fields.

**`--timing-probe` is device-aware.** `run_timing_probe` derives `device` from the already-
placed `x_train32` tensor, builds its `n_charts=16` model on that device via
`build_cae(16, device=device)`, and prints a float64-throughput caveat when `device.type ==
"cuda"` before either timed section runs. Training and curvature wall-clock are still reported
as separate terms, exactly as before this supplement -- verified end to end on this CPU-only
machine (`--timing-probe --pu-n 200` exits 0, prints both terms, and records a `timing_probe`
JSONL line with `device` and `torch_version` fields).

**Device-mixing guard.** Both runners' `--resume` path now raises, naming the record path and
the two disagreeing device strings, if a previously-recorded cell's `device` field does not
match the currently-requested device -- caveat 3 below is enforced in code, not only stated in
prose.

**Device parity test.** `notebooks/pu_manifold/tests/test_curvature_probe.py::test_chart_curvature_cpu_cuda_agree_to_float64_tolerance`,
`@pytest.mark.skipif(not torch.cuda.is_available(), ...)`. Deep-copies a CPU-trained model's
exact weights onto a CUDA copy (so both devices differentiate literally the same parameters,
side-stepping the CUDA-RNG-differs-from-CPU-RNG problem entirely) and asserts the two devices'
`chart_mean_curvature` output agrees to `rtol=1e-6, atol=1e-8` -- not bit-identity, which is
unachievable across devices (see caveat 1 below).

## 4. Setup notes for the colleague running the PU grid on GPU

**1. Install a CUDA-enabled torch build.** The install captured while writing this supplement
was CPU-only:

```
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# -> 2.13.0+cpu None False
```

Install a matching CUDA build for your driver (pick the `cuXXX` tag matching your installed
CUDA driver version -- see <https://pytorch.org/get-started/locally/> for the current
selector):

```
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**2. Verify the install:**

```
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect: True for the third value
```

**3. Run the timing probe on GPU first, before committing to the full grid** (caveat 2 below
is exactly why this step is not optional):

```
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --timing-probe --pu-n 200 --device cuda
```

Compare its printed `projected_nine_cell_grand_total_s` against this CPU machine's measured
~5.6-5.7h before deciding to run the full grid on GPU.

**4. Invoke the real grid with `--device cuda`:**

```
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --device cuda --resume
```

**5. Resuming after an interruption.** The runner is already resumable via its JSONL cache
(`--resume`, matched by `config_id`) -- this was true before this supplement and is unchanged.
The one new behaviour: `--resume` now refuses to continue a record file whose recorded `device`
disagrees with the one you pass on the command line (caveat 3), so if you started on `cpu` and
want to switch to `cuda`, use a distinct `--record-path` rather than resuming the same file
across devices.

The Swiss roll runner (`swiss_roll_curvature_sweep_run.py`) takes the identical `--device`
flag and the identical resume-device-guard, should it ever need to be re-run on GPU.

## 5. The three caveats -- stated here, and in both runners' `--device` help text / module docstring

**1. GPU runs do not reproduce CPU runs bit-for-bit.** CUDA RNG differs from CPU RNG, so
`torch.manual_seed(seed)` yields a different model initialization per device. A grid run on
GPU is a *different draw*, not a reproduction of any CPU run -- this is exactly why the
`device` field now lives in every cache record: a GPU-run cell and a CPU-run cell are
distinguishable after the fact, and must never be treated as replicates of each other.

**2. float64 throughput is hardware-dependent.** Curvature is float64-only throughout this
milestone (constraint 4; `_assert_float64` is never relaxed). Data-center GPUs (A100/H100) run
float64 at roughly 1/2 of float32 throughput; consumer GeForce cards run it at roughly 1/32 to
1/64. Since curvature never runs in anything but float64, **the curvature term may be SLOWER
on a consumer GPU than on this CPU machine**, even though training (float32, the dominant term
at ~16,100s vs ~4,000s here) should speed up regardless of GPU tier. This is why setup note 3
above is a required step, not a suggestion: run `--timing-probe --device cuda` and compare
against the CPU numbers in this document before committing to the multi-hour full grid.

**3. Do not mix devices within one grid.** All nine cells (plus the three D-12 control cells)
must run on the same device, or the three-seed spread mixes two different RNG draws into one
supposedly-comparable table. Enforced in code (section 3 above): `--resume` raises rather than
silently continuing a mixed-device record file.

## 6. Verification transcript (this machine, CPU-only)

```
# baseline, before any change in this supplement
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
  -> 286 passed, 19 warnings in 50.72s

.venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py \
  --n-charts 8 --seeds 0 --max-combos 1 --record-path <scratch>/anchor_baseline.jsonl
  -> rho_chart = -0.06041003026778113

# after cae.py / chart_curvature.py device plumbing (commit 3aeb27d)
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
  -> 286 passed, 19 warnings in 50.49s
.venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py \
  --n-charts 8 --seeds 0 --max-combos 1 --record-path <scratch>/anchor_after.jsonl
  -> rho_chart = -0.06041003026778113  (ANCHOR MATCHES EXACTLY)

# after swiss_roll_curvature_sweep_run.py --device (commit 036b762)
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
  -> 286 passed, 19 warnings in 51.33s
.venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py \
  --device cuda --dry-run
  -> SystemExit, actionable CUDA-unavailable message (exit 1)
.venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py \
  --n-charts 8 --seeds 0 --max-combos 1 --record-path <scratch>/anchor_after2.jsonl
  -> rho_chart = -0.06041003026778113, device=cpu, torch_version=2.13.0+cpu (ANCHOR MATCHES EXACTLY)

# after curvature_field_pu_run.py --device + timing-probe device-awareness (commit 0166baa)
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --device cuda --dry-run
  -> SystemExit, actionable CUDA-unavailable message (exit 1)
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --smoke --record-path <scratch>
  -> exit 0, SMOKE TALLY printed, record carries device=cpu torch_version=2.13.0+cpu
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --select-only --record-path <scratch>
  -> exit 0, "0 of 9 planned grid cells present" (smoke_cell correctly excluded from grid count)
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --timing-probe --pu-n 200 --record-path <scratch>
  -> exit 1 (OVER BUDGET, as before this supplement): projected_nine_cell_grand_total_s=19524.8
     (5.42h) vs 5-hour envelope=18000s; measured_forward_vs_reverse_ratio=21.63x; both terms
     (training seconds_per_training_step=0.3316, curvature reverse=51.12s/forward=2.36s at
     pu_n=200) printed as separate labelled lines; timing_probe JSONL record carries
     device=cpu, torch_version=2.13.0+cpu
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
  -> 286 passed, 19 warnings in 52.10s

# after the device-parity test (commit 0286001)
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
  -> 286 passed, 1 skipped, 19 warnings in 52.20s

# final confirmation, all four commits applied
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
  -> 286 passed, 1 skipped, 19 warnings in 51.45s
.venv/bin/python notebooks/diagnostics/swiss_roll_curvature_sweep_run.py \
  --n-charts 8 --seeds 0 --max-combos 1 --record-path <scratch>/anchor_final.jsonl
  -> rho_chart = -0.06041003026778113  (FINAL ANCHOR CONFIRMATION: MATCHES EXACTLY)
```

## 7. What this supplement does not do

No CUDA path was exercised on real hardware -- this machine has no GPU
(`torch.cuda.is_available()` is `False`, `torch.version.cuda` is `None`). The CUDA branches
(the `resolve_device` success path, `build_cae`/`build_control`'s `.to("cuda")`, the device
parity test's assertions) are written and guarded but unexercised here; the parity test exists
specifically so the colleague's first GPU run has an automated check rather than only a visual
one. This supplement does not narrow `PU_N_CHARTS_SWEEP`, drop seeds, or otherwise change
03-07's pre-declared 3x3 grid design -- 03-08 still runs the full nine-cell grid (plus three
D-12 controls) exactly as planned, now with the option of running it on either device.

## 8. State

`03-08` is unblocked and can now run the real grid on CPU (as originally planned, ~5.6-5.7h)
or, following the setup notes above, on the colleague's GPU. No PU number has been measured on
the real grid by this supplement -- every check above ran `--dry-run`, `--smoke`, `--select-
only`, `--timing-probe`, or the Swiss roll anchor cell, never a full grid cell.
