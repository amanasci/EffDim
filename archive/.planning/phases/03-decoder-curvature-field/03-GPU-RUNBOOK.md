# GPU Run Handoff — Phase 3 PU curvature-field grid

**Audience:** the coding agent on the GPU machine.
**Task:** run a 9-cell PyTorch sweep (plus 3 control cells), send back one JSONL file.
**You are not asked to interpret the results, change the design, or fix unfavourable numbers.**

---

## 0. What this is, in one paragraph

A chart auto-encoder is fitted to a frozen 10,000-row astronomical embedding at 768 dimensions,
then its decoder is differentiated twice to get a mean-curvature field. The grid is
`n_charts ∈ {4, 8, 16}` × `seed ∈ {20260813, 20260814, 20260815}` — nine independent fits.
On CPU the projection is ~5.6–5.7 hours, dominated by training (~16,100s) rather than curvature
(~4,000s). The hope is that a GPU makes this materially cheaper. **Whether it does is an open
question you are being asked to measure, not assume** — see §4.

---

## 1. Prerequisites — read this before anything else

### 1.1 The data file must be transferred manually. The repo alone is not enough.

The run depends on a frozen subsample that is **gitignored and irreproducible**. Cloning the repo
does not get it. Two files must be copied into `notebooks/.cache/` on the GPU machine:

```
notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz         (123 MB — required)
notebooks/.cache/subsample_20260729_a79b3460b838fd0a.meta.json   (164 B  — required)
```

Do **not** substitute, regenerate, or re-draw this subsample. Every prior fit in this milestone was
trained and split against it, and the runner deliberately raises `FileNotFoundError` and halts
rather than drawing a different one. If the file is missing, stop and ask for it.

The rest of `notebooks/.cache/` (~7.4 GB) is **not** needed. Only these two files.

### 1.2 Verify the file arrived intact

```bash
cd <repo root>
ls -l notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz   # expect ~123121268 bytes
.venv/bin/python -c "
import numpy as np
a = np.load('notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz')['legacysurvey']
print(a.shape, a.dtype)   # expect (10000, 768)
"
```

Stop if the shape is not `(10000, 768)`.

---

## 2. Environment

```bash
python -m venv .venv                      # if not already present
.venv/bin/pip install -r requirements.txt # or the project's usual install
```

Then replace the CPU-only torch with a CUDA build matching the machine's driver:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Pick the correct `cuXXX` tag for the hardware — see https://pytorch.org/get-started/locally/.
Verify:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` must print `True`. If it prints `False`, the wrong wheel is installed —
fix that before continuing. Everything below assumes `.venv/bin/python`.

---

## 3. Verification gates — run in order, stop at the first failure

Do not skip these. They take about two minutes total and they are the only evidence the CUDA path
works; it was written on a machine with no GPU and has never been executed on real hardware.

```bash
# Gate 1 — full test suite. Expect "286 passed, 1 skipped" on CPU.
# On a CUDA machine the skipped test now RUNS; expect 287 passed.
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q

# Gate 2 — the CPU/CUDA curvature parity test specifically. This is the important one.
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q -k cuda_agree -v

# Gate 3 — the runner's own dry run. Writes nothing; prints the planned grid and selection rule.
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --dry-run --device cuda

# Gate 4 — one deliberately tiny cell, end to end on the GPU.
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --smoke --device cuda
```

**If Gate 2 fails, stop and report the failure output.** It means CPU and CUDA disagree beyond
float64 round-off, which invalidates the GPU run. Do not adjust the tolerance to make it pass.

---

## 4. Timing probe — REPORT BACK BEFORE RUNNING THE GRID

```bash
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --timing-probe --device cuda
```

This measures real training and curvature wall clock at full scale (`d=20, D=768`) and projects the
nine-cell total. It takes a few minutes, not hours.

**Send these numbers back before starting the full grid.** Specifically:

- projected training total (seconds)
- projected curvature total (seconds)
- projected grand total (hours)
- whether it printed `OVER BUDGET`
- the GPU model from `torch.cuda.get_device_name(0)`

### Why this gate exists

Curvature runs in **float64** and cannot be changed to float32 — second derivatives are exactly
where float32 noise appears first, and the guard enforcing this is load-bearing. float64 throughput
is extremely hardware-dependent:

| Hardware | float64 rate vs float32 |
|---|---|
| A100 / H100 (data-center) | ~1/2 |
| GeForce / RTX (consumer) | 1/32 to 1/64 |

On a consumer card the **curvature term may run slower on GPU than on CPU**, even though training
(float32, and the dominant term) speeds up. The probe tells us whether this is a win. If it is not,
the answer is to run on CPU, not to weaken float64.

The runner exits non-zero with `OVER BUDGET` if the projection exceeds 5 hours. That is a report,
not an error to work around.

---

## 5. The grid run

Only after the probe numbers have been sent back and the go-ahead given:

```bash
nohup .venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py \
      --device cuda --resume > pu_grid.log 2>&1 &
```

Use `nohup`/`tmux`/`screen` — this is a multi-hour job and must survive a disconnected SSH session.

Progress:

```bash
tail -f pu_grid.log
wc -l notebooks/.cache/03_curvature_field_pu.jsonl   # one line per completed cell
```

Expect **12 records total**: 9 grid cells (3 `n_charts` × 3 seeds) plus 3 D-12 control cells.

### If it is interrupted

Nothing is lost. Every cell is appended to the JSONL as it completes. Re-run the exact same command
— `--resume` skips completed cells and continues. Do not delete the JSONL to "start clean".

---

## 6. Hard rules

Violating any of these makes the result unusable. They are not stylistic.

1. **Do not mix devices within one grid.** Every cell, including the 3 controls, must run on the
   same device. CUDA RNG differs from CPU RNG, so a GPU cell and a CPU cell are different draws, and
   the reported unit is the spread across the three seeds — mixing devices contaminates it. The
   `--resume` guard refuses to continue a record whose device disagrees; do not override it.

2. **Do not change `PU_N_CHARTS_SWEEP = (4, 8, 16)` or `PU_SEEDS` (three seeds).** These were fixed
   in writing before any number from this grid existed, deliberately, so that the selection rule
   could not be tuned to the result. Narrowing the sweep or dropping a seed to save time invalidates
   the design. If the run is too expensive, report that — do not shrink it.

3. **Do not weaken float64 or relax any tolerance** to make something pass or run faster.

4. **Do not modify `notebooks/pu_manifold/cae.py`.** It is a sealed architecture with reproduction
   anchors depending on its exact RNG consumption order. A one-line change there can silently
   invalidate the whole milestone.

5. **Do not edit the JSONL record**, delete rows, or hand-correct values.

6. **Do not interpret or act on the results.** No number in this grid passes or fails anything —
   there is a table to read and a configuration to select, and the selection happens on our side by
   a rule that was written down in advance. If a cell produces something that looks wrong (a
   `cond(g)` in the tens of thousands, a near-zero curvature field, a diverged fit), that is
   **data, not a bug** — report it, do not fix it. One such conditioning blow-up has already been
   observed and is expected to be informative.

7. **If something genuinely breaks, stop and report** with the traceback and the log. Do not work
   around it.

---

## 7. What to send back

1. `notebooks/.cache/03_curvature_field_pu.jsonl` — the whole file, unedited. This is the deliverable.
2. `pu_grid.log` — the full stdout/stderr.
3. The timing-probe numbers from §4 (if not already sent).
4. `torch.cuda.get_device_name(0)`, `torch.__version__`, driver version (`nvidia-smi` header line).
5. Total wall clock, and a note of any interruption or resume.

Every record already carries its own `device` and `torch_version` field, so provenance travels with
the data.

---

## 8. Quick reference

```bash
# verify data
.venv/bin/python -c "import numpy as np; print(np.load('notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz')['legacysurvey'].shape)"

# verify cuda
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# gates
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q -k cuda_agree -v
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --dry-run  --device cuda
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --smoke    --device cuda

# probe (REPORT BACK BEFORE §5)
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --timing-probe --device cuda

# grid
nohup .venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --device cuda --resume > pu_grid.log 2>&1 &

# progress / resume after interruption
wc -l notebooks/.cache/03_curvature_field_pu.jsonl        # expect 12 when complete
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --device cuda --resume
```

Full flag documentation, including the three device caveats in the runner's own words:

```bash
.venv/bin/python notebooks/diagnostics/curvature_field_pu_run.py --help
```

Background on the device support itself: `.planning/phases/03-decoder-curvature-field/03-07-SUPPLEMENT-01.md`.
