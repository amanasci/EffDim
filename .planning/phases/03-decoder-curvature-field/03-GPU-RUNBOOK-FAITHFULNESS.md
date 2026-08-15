# GPU Run Handoff — CAE faithfulness on PU

**Audience:** the coding agent on the GPU machine.
**Task:** train one well-converged Chart Auto-Encoder on a frozen astronomical embedding and
measure its faithfulness in the CAE paper's own terms. Send back one JSONL file.
**You are not asked to interpret the results, change the design, or fix unfavourable numbers.**

---

## 0. What this is, in one paragraph

A single Chart Auto-Encoder (arXiv:1912.10094) is fitted to a frozen 10,000-row astronomical
embedding at 768 dimensions, `chart_dim=20`, `embed_dim=40`, `n_charts=32` (over-specified,
pruned a posteriori), trained for a generous, fixed epoch budget with early stopping made
structurally inert — this is deliberately **not** the sweep in `03-GPU-RUNBOOK.md`, and it is
**not a comparison**: no plain autoencoder, no VAE, no A/B arm. One model, trained once per
seed (three seeds by default), measured against the paper's own faithfulness definitions:
sup-norm reconstruction error, unfaithfulness/coverage, the chart-transition cycle residual
`R_cycle`, and chart survival vs. argmax occupancy. The runner is
`notebooks/diagnostics/cae_faithfulness_run.py`.

---

## 1. Prerequisites — read this before anything else

### 1.1 The data file must be transferred manually. The repo alone is not enough.

The run depends on a frozen subsample that is **gitignored and irreproducible**. Cloning the
repo does not get it. Two files must be copied into `notebooks/.cache/` on the GPU machine:

```
notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz         (123 MB — required)
notebooks/.cache/subsample_20260729_a79b3460b838fd0a.meta.json   (164 B  — required)
```

Do **not** substitute, regenerate, or re-draw this subsample. Every prior fit in this milestone
was trained and split against it, and the runner deliberately raises `FileNotFoundError` and
halts rather than drawing a different one. If the file is missing, stop and ask for it.

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

`torch.cuda.is_available()` must print `True`. If it prints `False`, the wrong wheel is
installed — fix that before continuing. The install captured while writing this runbook was
CPU-only (`2.13.0+cpu`, `torch.cuda.is_available()` False); this machine has no GPU, so the
CUDA path below has never been exercised on real hardware. Everything below assumes
`.venv/bin/python`.

---

## 3. Verification gates — run in order, stop at the first failure

```bash
# Gate 1 -- full test suite. Expect "302 passed, 1 skipped" on CPU.
# On a CUDA machine the skipped device-parity test now RUNS; expect 303 passed.
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q

# Gate 2 -- the runner's own dry run. Writes nothing; prints the planned config.
.venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --dry-run --device cuda

# Gate 3 -- one deliberately tiny cell, end to end on the GPU. Under a minute.
.venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --smoke --device cuda
```

**If Gate 1 regresses below 302 passed (or 303 with CUDA), stop and report the failure output**
rather than working around it — `cae.py` must never be edited by this task, and a test
regression here is a signal something upstream changed, not a signal to relax a test.

**If Gate 3 (`--smoke`) fails, stop and report the traceback.** Do not weaken the assertion
that fires it (`epochs_run == max_epochs`) to make it pass — that assertion exists to prevent
this exact runner from ever reporting faithfulness numbers for an unconverged model.

---

## 4. The real run

There is no timing probe for this runner (unlike `03-GPU-RUNBOOK.md`'s sweep) — one model,
trained once per seed, is a much smaller compute budget than a nine-cell grid. Still, run one
seed first and time it before committing to all three:

```bash
.venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py \
      --device cuda --seeds 20260813 --epochs 300 \
      --record-path notebooks/.cache/03_cae_faithfulness_pu_probe.jsonl
```

If that single seed's wall clock is comfortable, raise `--epochs` if you judge it warranted
(the GPU has headroom this CPU machine doesn't) and run the full three-seed default:

```bash
nohup .venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py \
      --device cuda --resume > cae_faithfulness.log 2>&1 &
```

Use `nohup`/`tmux`/`screen` — a 300-epoch fit at `n_charts=32` is not instant, and this must
survive a disconnected SSH session.

Progress:

```bash
tail -f cae_faithfulness.log
wc -l notebooks/.cache/03_cae_faithfulness_pu.jsonl   # one line per completed seed
```

Expect **3 records** (one per default seed) once complete, plus whichever probe record you
made in the step above (in its own, separately-named file — do not mix it into the real one).

### If it is interrupted

Nothing is lost. Every seed is appended to the JSONL as it completes. Re-run the exact same
command — `--resume` skips completed seeds and continues. Do not delete the JSONL to "start
clean".

---

## 5. Hard rules

Violating any of these makes the result unusable. They are not stylistic.

1. **Do not reduce the epoch budget or re-enable early stopping to save time.** The whole point
   of this runner is a genuinely converged model — a prior PU fit under a tight patience halted
   at epoch 7 on a plateau in the Lipschitz penalty term, not in reconstruction, and that
   silently produced faithfulness numbers for an unconverged model. `--epochs` may be *raised*
   if the GPU has headroom; it must never be lowered to make the run finish faster, and
   `early_stop_patience`/the wallclock ceiling must never be reintroduced.

2. **Do not modify `notebooks/pu_manifold/cae.py`.** It is a sealed architecture with
   reproduction anchors depending on its exact RNG consumption order. A one-line change there
   can silently invalidate other fits across this milestone. This runner calls
   `cae.r_cycle`/`cae.select_overlap_pairs`/`cae.unfaithfulness_coverage`/`cae.chart_survival`
   exactly as written — if one of them appears to misbehave, report it, do not patch it.

3. **Do not edit the JSONL** — no hand-correcting a value, no deleting a row, no re-ordering.

4. **Do not mix devices within one record file.** `--resume` refuses to continue a record whose
   recorded `device` disagrees with the one requested — do not override that guard. If you
   need to compare a CPU probe against a GPU run, use two distinct `--record-path` values.

5. **Do not tune anything to improve a number.** This is not a comparison and there is no
   target to beat — `n_charts`, `chart_dim`, `embed_dim`, `lr`, `batch`, and the Lipschitz
   weight are the CAE paper's own stated values (or this milestone's own established PU
   constants where the paper doesn't specify). If a metric looks bad, that is the finding.

6. **Report anomalies as data, not bugs.** A high `R_cycle`, a collapsed `argmax_occupancy`
   (few distinct charts despite many "surviving" by decoder weight-mass), a poor coverage
   number — all of these are exactly the kind of disagreement this runner exists to surface.
   Do not adjust a threshold, add a fallback, or silently retry to make a number look better.

7. **If the run is too expensive, report that** — the projected wall clock, what dominated it
   (training is very likely the dominant term here, unlike the curvature-field grid) — rather
   than shrinking `--epochs`, `--n-charts`, or `--seeds` to fit a budget.

8. **If something genuinely breaks, stop and report** with the traceback and the log. Do not
   work around it.

---

## 6. What to send back

1. `notebooks/.cache/03_cae_faithfulness_pu.jsonl` — the whole file, unedited. This is the
   deliverable. Every record carries its own per-epoch loss curve (`loss_curve`), the four
   faithfulness measurements, and `device`/`torch_version` for provenance.
2. `cae_faithfulness.log` — the full stdout/stderr, including the printed seed-summary spread
   at the end (min/median/max across seeds for each headline number).
3. `torch.cuda.get_device_name(0)`, `torch.__version__`, driver version (`nvidia-smi` header
   line).
4. Total wall clock per seed and in aggregate, and a note of any interruption or resume.
5. If you looked at the loss curves yourselves (they're in the JSONL), whether the total-loss
   curve visibly flattened by the end of training — a one-line eyeball note is useful, but do
   not act on it (e.g. do not decide to cut a seed short because it "looked converged already";
   the point of the fixed budget and the `epochs_run == max_epochs` assertion is that this
   judgement is made structurally, not by eye, mid-run).

---

## 7. Quick reference

```bash
# verify data
.venv/bin/python -c "import numpy as np; print(np.load('notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz')['legacysurvey'].shape)"

# verify cuda
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# gates
.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q
.venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --dry-run --device cuda
.venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --smoke    --device cuda

# one-seed probe, own record file, before committing to all three
.venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --device cuda --seeds 20260813 \
      --record-path notebooks/.cache/03_cae_faithfulness_pu_probe.jsonl

# real run, all three default seeds
nohup .venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --device cuda --resume \
      > cae_faithfulness.log 2>&1 &

# progress / resume after interruption
wc -l notebooks/.cache/03_cae_faithfulness_pu.jsonl        # expect 3 when complete
.venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --device cuda --resume
```

Full flag documentation, including `--epochs`, `--n-charts`, `--chart-dim`, `--embed-dim`,
`--unfaithful-samples`, and `--seeds`:

```bash
.venv/bin/python notebooks/diagnostics/cae_faithfulness_run.py --help
```

Background on the device support itself:
`.planning/phases/03-decoder-curvature-field/03-07-SUPPLEMENT-01.md`. This runner follows its
device-threading pattern (construct-then-`.to(device)`, `device`/`torch_version` in every
record, `--resume` device-mismatch guard) verbatim but was not built by that supplement.
