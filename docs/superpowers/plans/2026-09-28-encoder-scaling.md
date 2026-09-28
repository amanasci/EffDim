# Encoder Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the paper's per-encoder battery (9 jobs) on all 31 `physics` encoders of the PU release on the pod GPUs, and turn the records into committed tables, figures and a computed report.

**Architecture:** The existing split runner gains `--device`/`--deterministic` (default CPU path unchanged, proven by the equivalence gate). A YAML manifest lists the 31 encoders; `sweep/jobs.py` expands manifest × battery into job specs; `sweep/run_queue.py` runs them one per GPU with done-markers so pod restarts only lose in-flight jobs; `sweep/aggregate.py` reads records the same way `paper/generate/appendix_gen.py` does and writes outputs to `curvature-experiment/results/scaling/`.

**Tech Stack:** Python 3.14 venv, PyTorch (CUDA on the pod), numpy/scipy/scikit-learn, pyarrow, PyYAML, matplotlib, pytest; ssh/tmux/rsync to the EleutherAI pod `universetbd-0`.

**Spec:** `docs/superpowers/specs/2026-09-28-encoder-scaling-design.md`

## Global Constraints

- Repo: `/home/akagi/Documents/Projects/EffDim`, branch `encoder-scaling` (`$R` below). Python: `PY=$R/.venv/bin/python`.
- Default CPU behaviour of every existing runner must stay byte-identical: after any runner change, `CLOSURE_WORK=$HOME/.cache/effdim-closure $R/docs/superpowers/harness/gate.sh $R <label>` must print `GATE PASS`.
- `paper/latex/main.tex` must not change. Never import or run `paper/generate/appendix_gen.py` outside a sandbox copy (it rewrites main.tex when run).
- Never write into the record store `/home/akagi/Documents/Projects/EffDim/notebooks/.cache` (also reachable via the `curvature-experiment/.cache` symlink) except the new subdirectory `curvature-experiment/.cache/scaling/` created in Task 9.
- PU snapshot: `bc081f8a5db4767edcd958653d96efde9137de0b`. 31 encoders, 86,471 rows each, parquet `physics/<name>_test.parquet`, column `<name>_galaxies`.
- Protocol per encoder: d=16 unless stated; labels mag_r, photo_z, smooth_fraction, stellar_mass; 512 anchors, k=2048, α=100 (runner defaults).
- GPU runs always pass `--device cuda --deterministic`; deterministic mode = `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Pod rules (from `docs/remote-compute/eleutherai-pod-user-guide.md` and `CLAUDE.md`), binding for every remote step:
  - Read the local guide in full before the first SSH command of a session; then check the remote `/root/user-guide.md` sha256 equals the local file's; if not, re-fetch and re-read before continuing.
  - Everything persistent under `/mnt/ssd-cluster/EffDim`; nothing we need in `/root`.
  - Long jobs only inside `tmux`.
  - No concurrent or unbounded `find`/`ls -R` on `/mnt`; direct paths, `-maxdepth`, `timeout 30`.
  - Never write to `/mnt/datasets` or `/mnt/ssd-1..4`; never touch other users' files in `/root`.
  - Check `nvidia-smi` and use only GPUs with no other processes; verify the effective cgroup CPU limit (`cat /sys/fs/cgroup/cpu.max`) before choosing `--threads`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **Pod restart mid-job** — a job killed after writing part of its record must be re-run from scratch, never marked done. Pinned by `test_partial_record_is_moved_aside_and_rerun` (Task 4).
2. **A job whose runner exits 0 but writes no result rows** (e.g. all labels masked) — must not get a done-marker. Pinned by `test_exit_zero_without_result_rows_is_not_done` (Task 4).
3. **`--device cuda` on a machine without CUDA** — must stop with a clear message, not silently fall back to CPU. Pinned by `test_cuda_without_gpu_refuses` (Task 1).
4. **Encoder parquet with the wrong row count or column** — must fail loudly before any compute. Pinned by `test_load_physics_rejects_wrong_row_count` (Task 1, via the existing loader's expected-rows check with the parquet override).
5. **Aggregation on a partially complete sweep** — tables must show `--` for missing encoders/jobs and the report must count only completed ones, not crash. Pinned by `test_aggregate_with_missing_jobs` (Task 5).

---

## File structure

```
curvature-experiment/
  runners/09_physics_probe_facing_run.py        MODIFY  fit_decoder(..., device="cpu")
  runners/09_physics_probe_facing_split_run.py  MODIFY  --device, --deterministic, env-row fields
  encoders.yaml                                 CREATE  31-encoder manifest
  sweep/__init__.py                             CREATE  empty
  sweep/manifest.py                             CREATE  load + validate the manifest
  sweep/jobs.py                                 CREATE  manifest x battery -> JobSpec list; --list
  sweep/run_queue.py                            CREATE  GPU scheduler with done-markers
  sweep/extract.py                              CREATE  record readers (ported from appendix_gen)
  sweep/aggregate.py                            CREATE  tables, figures, SCALING_REPORT.md
  sweep/setup_pod.sh                            CREATE  idempotent pod environment
  sweep/POD_RUNBOOK.md                          CREATE  exact pod commands incl. restart recovery
  tests/test_device_flags.py                    CREATE
  tests/test_sweep_manifest.py                  CREATE
  tests/test_sweep_jobs.py                      CREATE
  tests/test_sweep_queue.py                     CREATE
  tests/test_sweep_extract.py                   CREATE
  tests/test_sweep_aggregate.py                 CREATE
  results/scaling/                              CREATE (Task 9) tables, figures, report, SHA256SUMS
  requirements.txt                              MODIFY  pin PyYAML
docs/superpowers/harness/compare.py             MODIFY  ignore new environment metadata keys
```

Deviation from the spec, recorded: the spec lists `09_physics_normal_scaling_run.py` for `--device`. In physics mode it reads job 1's stored geometry and fits nothing (it fits only in smoke mode), so it needs no GPU path; it is left unchanged.

---

### Task 1: GPU and deterministic options for the split runner

**Files:**
- Modify: `curvature-experiment/runners/09_physics_probe_facing_run.py:100-118` (`fit_decoder`)
- Modify: `curvature-experiment/runners/09_physics_probe_facing_split_run.py` (`build_parser` ~169-189, `main` ~192-306)
- Modify: `docs/superpowers/harness/compare.py` (VOLATILE regex)
- Test: `curvature-experiment/tests/test_device_flags.py`

**Interfaces:**
- Produces: `ppf.fit_decoder(X, d, in_dim, max_epochs, device="cpu") -> dict` (same keys as today; `x64`, `model`, `curvature_model` live on `device`). Split runner flags `--device` (str, default `"cpu"`) and `--deterministic` (store_true). Environment row keys added: `device`, `deterministic`, `gpu_name` (str or None), `cuda_version` (str or None).

- [ ] **Step 1: Write the failing tests** — `curvature-experiment/tests/test_device_flags.py`:

```python
"""--device / --deterministic on the split runner; default CPU path unchanged."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

RUNNERS = Path(__file__).resolve().parents[1] / "runners"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), RUNNERS / name)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def test_split_parser_device_defaults():
    split = _load("09_physics_probe_facing_split_run.py")
    args = split.build_parser().parse_args(["--mode", "smoke"])
    assert args.device == "cpu" and args.deterministic is False
    args = split.build_parser().parse_args(["--mode", "physics", "--device", "cuda", "--deterministic"])
    assert args.device == "cuda" and args.deterministic is True


def test_fit_decoder_cpu_device_matches_default():
    ppf = _load("09_physics_probe_facing_run.py")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    a = ppf.fit_decoder(X, 2, 12, 3)
    b = ppf.fit_decoder(X, 2, 12, 3, device="cpu")
    for pa, pb in zip(a["model"].state_dict().values(), b["model"].state_dict().values()):
        assert torch.equal(pa, pb)
    assert a["var_explained"] == b["var_explained"]


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU refusal")
def test_cuda_without_gpu_refuses(tmp_path, monkeypatch):
    split = _load("09_physics_probe_facing_split_run.py")
    monkeypatch.setattr(sys, "argv", ["x", "--mode", "smoke", "--device", "cuda", "--threads", "8",
                                      "--record-path", str(tmp_path / "r.jsonl")])
    with pytest.raises(SystemExit, match="CUDA is not available"):
        split.main()


def test_load_physics_rejects_wrong_row_count(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ppf = _load("09_physics_probe_facing_run.py")
    p = tmp_path / "e.parquet"
    pq.write_table(pa.table({"m_galaxies": [[0.1, 0.2, 0.3]] * 5}), p)
    with pytest.raises(Exception, match="86471|rows"):
        ppf.pl.load_physics_embeddings(parquet_path=str(p), column="m_galaxies")
```

- [ ] **Step 2: Run to confirm failure**

Run: `cd $R/curvature-experiment && EFFDIM_CACHE_DIR=$(mktemp -d) $PY -m pytest -q -p no:cacheprovider tests/test_device_flags.py`
Expected: `test_split_parser_device_defaults` fails (`unrecognized arguments: --device`), `test_fit_decoder_cpu_device_matches_default` fails (`unexpected keyword argument 'device'`), `test_cuda_without_gpu_refuses` fails. `test_load_physics_rejects_wrong_row_count` may already pass (existing check) — keep it as a guard. If it fails because `load_physics_embeddings` has a different keyword name, read its signature (`physics_labels.py:549`) and adjust only the test's keyword.

- [ ] **Step 3: Implement `fit_decoder(..., device="cpu")`** in `09_physics_probe_facing_run.py`. Replace the function body so the CPU path executes exactly the same statements as today when `device == "cpu"`:

```python
def fit_decoder(X: np.ndarray, d: int, in_dim: int, max_epochs: int, device: str = "cpu") -> Dict[str, Any]:
    """`fit_and_field_at_anchors`'s fit steps, returning the model so the Hessian can be taken.
    ``device`` other than "cpu" moves the model and tensors there; the CPU path is unchanged."""
    torch.manual_seed(pcp.TORCH_INIT_SEED)
    model = cae.PlainAutoEncoder(in_dim=in_dim, latent_dim=d, hidden=pcp.AE_HIDDEN, activation=pcp.AE_ACTIVATION)
    train_idx, holdout_idx = crossmodal_curvature.split_indices(X.shape[0], pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION)
    x32 = torch.tensor(X, dtype=torch.float32)
    x64 = torch.tensor(X, dtype=torch.float64)
    if device != "cpu":
        model = model.to(device); x32 = x32.to(device); x64 = x64.to(device)
    cfg = dict(pcp.TRAIN_CFG); cfg["max_epochs"] = max_epochs
    t0 = time.monotonic()
    cae.train_plain_ae(model, x32[torch.as_tensor(train_idx, dtype=torch.long, device=x32.device)], cfg)
    wall = time.monotonic() - t0
    model.eval().double()
    x_hold = x64[torch.as_tensor(holdout_idx, dtype=torch.long, device=x64.device)]
    with torch.no_grad():
        y_hold = model(x_hold)["y"]
    recon = cae.reconstruction_stats(x_hold, y_hold)
    var_explained = 1.0 - recon["mse_total"] / float((torch.linalg.norm(x_hold, dim=1) ** 2).mean())
    curvature_model = runner.SphereProjectedDecoder(model).eval() if pcp.DECODER_IMAGE_PROJECTION == "sphere" else model
    return {"model": model, "curvature_model": curvature_model, "x64": x64, "var_explained": float(var_explained), "wallclock_fit_s": wall}
```
(`torch.as_tensor(..., device=cpu-tensor.device)` on CPU produces the identical index tensor as before.) `decoder_geometry` needs no change: it builds chunks from `z`, which is already on the device, and converts every output with `.detach().cpu().numpy()`.

- [ ] **Step 4: Implement the runner flags** in `09_physics_probe_facing_split_run.py`:
  - In `build_parser()`, after the `--threads` line:
    ```python
    p.add_argument("--device", type=str, default="cpu", help="torch device for the decoder fit and geometry (e.g. cuda)")
    p.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8")
    ```
  - At the top of `main()` right after `args = p.parse_args()` (before any data load or torch op):
    ```python
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available on this machine")
    gpu_name = torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None
    ```
    Add `import os` at the top if missing (with the other stdlib imports, `# noqa: E402` like its neighbours).
  - In the environment-row dict, append: `"device": args.device, "deterministic": args.deterministic, "gpu_name": gpu_name, "cuda_version": torch.version.cuda,`.
  - Pass `device=args.device` to both `ppf.fit_decoder(...)` calls (physics refit branch and smoke branch). The `fit["x64"][torch.as_tensor(a, dtype=torch.long)]` indexing lines become `fit["x64"][torch.as_tensor(a, dtype=torch.long, device=fit["x64"].device)]`.

- [ ] **Step 5: Tolerate the new environment metadata in the equivalence comparer.** In `docs/superpowers/harness/compare.py`, extend `VOLATILE` with `|^device$|^deterministic$|^gpu_name$|^cuda_version$` (environment metadata, not results). Then confirm nothing else is newly ignored:
  ```bash
  cd $R && CLOSURE_WORK=$HOME/.cache/effdim-closure $PY - <<'PYEOF'
  import json, os, sys; sys.path.insert(0, "docs/superpowers/harness"); import compare as c
  from pathlib import Path
  drop = set()
  for f in (Path(os.environ["CLOSURE_WORK"]) / "baseline/smoke").glob("*.jsonl"):
      for l in f.read_text().splitlines():
          def w(o):
              if isinstance(o, dict):
                  for k, v in o.items():
                      (drop.add(k) if (c.VOLATILE.search(k) or c.COLLEAGUE_KEY.search(k)) else w(v))
              elif isinstance(o, list): [w(x) for x in o]
          w(json.loads(l))
  print(sorted(drop))
  PYEOF
  ```
  The printed set must contain only timestamps/wallclock/version/path/root keys, colleague keys, and `device` (already present in `09_physics_probe_facing_run` environment rows).

- [ ] **Step 6: Tests pass**

Run: `cd $R/curvature-experiment && EFFDIM_CACHE_DIR=$(mktemp -d) $PY -m pytest -q -p no:cacheprovider tests`
Expected: all pass (208 existing + 4 new; the CUDA refusal test runs because this machine has no GPU).

- [ ] **Step 7: Gate** — the default CPU path must be unchanged:

Run: `CLOSURE_WORK=$HOME/.cache/effdim-closure $R/docs/superpowers/harness/gate.sh $R scaling-t1` (~10 min)
Expected: `GATE PASS (scaling-t1)`. Also `git -C $R diff --exit-code -- paper/latex/main.tex`.

- [ ] **Step 8: Commit**

```bash
cd $R && git add curvature-experiment/runners/09_physics_probe_facing_run.py curvature-experiment/runners/09_physics_probe_facing_split_run.py curvature-experiment/tests/test_device_flags.py docs/superpowers/harness/compare.py
git commit -m "feat(runners): --device and --deterministic for the split runner's decoder fit

Default CPU path unchanged (equivalence gate passes). Environment rows record device,
deterministic, gpu_name and cuda_version; the comparer treats them as metadata.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Encoder manifest

**Files:**
- Create: `curvature-experiment/encoders.yaml`, `curvature-experiment/sweep/__init__.py` (empty), `curvature-experiment/sweep/manifest.py`
- Modify: `curvature-experiment/requirements.txt` (add `PyYAML==6.0.3`)
- Test: `curvature-experiment/tests/test_sweep_manifest.py`

**Interfaces:**
- Produces: `sweep.manifest.Encoder` (frozen dataclass: `name: str, family: str, dim: int, params: int, params_source: str, in_paper: bool`, properties `parquet_file -> str` = `f"physics/{name}_test.parquet"`, `column -> str` = `f"{name}_galaxies"`); `sweep.manifest.Manifest` (frozen: `snapshot: str, repo: str, n_rows: int, label_table: str, encoders: tuple[Encoder, ...]`); `sweep.manifest.load_manifest(path: str | Path = DEFAULT_PATH) -> Manifest`; `DEFAULT_PATH` = `curvature-experiment/encoders.yaml`.

- [ ] **Step 1: Write the failing test** — `curvature-experiment/tests/test_sweep_manifest.py`:

```python
from pathlib import Path

import pytest
import yaml

from sweep.manifest import load_manifest

PAPER = {"vit_base", "vit_large", "clip_base", "convnext_base", "dinov3_vitb16"}


def test_manifest_has_31_unique_encoders():
    m = load_manifest()
    names = [e.name for e in m.encoders]
    assert len(names) == 31 and len(set(names)) == 31
    assert {e.name for e in m.encoders if e.in_paper} == PAPER
    assert m.snapshot == "bc081f8a5db4767edcd958653d96efde9137de0b"
    assert m.n_rows == 86471


def test_every_encoder_has_dim_params_and_source():
    for e in load_manifest().encoders:
        assert e.dim > 0 and e.params > 0, e.name
        assert e.params_source.startswith("https://"), e.name
        assert e.parquet_file == f"physics/{e.name}_test.parquet"
        assert e.column == f"{e.name}_galaxies"


def test_manifest_rejects_duplicate_names(tmp_path):
    src = yaml.safe_load(Path(load_manifest.__globals__["DEFAULT_PATH"]).read_text())
    src["encoders"].append(dict(src["encoders"][0]))
    p = tmp_path / "m.yaml"; p.write_text(yaml.safe_dump(src))
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(p)
```

- [ ] **Step 2: Run to confirm failure** — `cd $R/curvature-experiment && $PY -m pytest -q -p no:cacheprovider tests/test_sweep_manifest.py` → `ModuleNotFoundError: No module named 'sweep'`.

- [ ] **Step 3: Write `sweep/manifest.py`**:

```python
"""The 31-encoder manifest for the encoder-scaling sweep (encoders.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "encoders.yaml"


@dataclass(frozen=True)
class Encoder:
    name: str
    family: str
    dim: int
    params: int
    params_source: str
    in_paper: bool

    @property
    def parquet_file(self) -> str:
        return f"physics/{self.name}_test.parquet"

    @property
    def column(self) -> str:
        return f"{self.name}_galaxies"


@dataclass(frozen=True)
class Manifest:
    repo: str
    snapshot: str
    n_rows: int
    label_table: str
    encoders: Tuple[Encoder, ...]


def load_manifest(path: Union[str, Path] = DEFAULT_PATH) -> Manifest:
    src = yaml.safe_load(Path(path).read_text())
    encs = tuple(Encoder(**e) for e in src["encoders"])
    names = [e.name for e in encs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate encoder names in {path}: {dup}")
    return Manifest(repo=src["repo"], snapshot=src["snapshot"], n_rows=int(src["n_rows"]),
                    label_table=src["label_table"], encoders=encs)
```

- [ ] **Step 4: Write `encoders.yaml`.** Top-level keys: `repo: UniverseTBD/pu-embeddings`, `snapshot: bc081f8a5db4767edcd958653d96efde9137de0b`, `n_rows: 86471`, `label_table:` the pod path of the label parquet — copy it exactly from `curvature-experiment/REPRODUCE.md` §2 (Inputs; `labels_Smith42_galaxies_v2.0_test.parquet` under `/mnt/ssd-cluster/effdim/…`). Then `encoders:` one mapping per encoder with `name, family, dim, params, params_source, in_paper`, in this order and with these dims (from the parquet footers):

  astropt_015M 384, astropt_095M 768, astropt_850M 2048 (family AstroPT); clip_base 512, clip_large 768 (CLIP); convnext_nano 640, convnext_tiny 768, convnext_base 1024, convnext_large 1536 (ConvNeXt); dinov3_vits16 384, dinov3_vits16plus 384, dinov3_vitb16 768, dinov3_vitl16 1024, dinov3_vith16plus 1280, dinov3_vit7b16 4096 (DINOv3); vit_base 768, vit_large 1024, vit_huge 1280 (ViT); vit-mae_base 768, vit-mae_large 1024, vit-mae_huge 1280 (ViT-MAE); ijepa_huge 1280, ijepa_giant 1408 (I-JEPA); vjepa_large 1024, vjepa_huge 1280, vjepa_giant 1408 (V-JEPA); llava_15_7b 4096, llava_15_13b 5120 (LLaVA-1.5); paligemma_3b 2304, paligemma_10b 3584, paligemma_28b 4608 (PaliGemma).

  `in_paper: true` only for vit_base, vit_large, clip_base, convnext_base, dinov3_vitb16. `params` = total parameter count of the vision model the PU release used, as an integer, and `params_source` = the `https://` URL (model card or paper) the number comes from. Find each on the Hugging Face model card named in the PU release's model list (the PU `README.md` / the `UniverseTBD` org), or the model's paper. If a card gives only a rounded figure (e.g. "7B"), use it (7000000000) and keep the URL. Do not invent numbers: if no source can be found for a model, stop and report NEEDS_CONTEXT with the model name.

- [ ] **Step 5: Pin PyYAML** — add `PyYAML==6.0.3` to `curvature-experiment/requirements.txt` next to the other pins, with a one-line comment `# sweep/manifest.py (encoders.yaml)`.

- [ ] **Step 6: Tests pass** — `cd $R/curvature-experiment && $PY -m pytest -q -p no:cacheprovider tests/test_sweep_manifest.py` → 3 passed.

- [ ] **Step 7: Commit** — `git add curvature-experiment/encoders.yaml curvature-experiment/sweep curvature-experiment/tests/test_sweep_manifest.py curvature-experiment/requirements.txt` and commit `feat(sweep): 31-encoder manifest with sourced parameter counts` + trailer.

---

### Task 3: Job expansion

**Files:**
- Create: `curvature-experiment/sweep/jobs.py`
- Test: `curvature-experiment/tests/test_sweep_jobs.py`

**Interfaces:**
- Consumes: `sweep.manifest.load_manifest`, `Encoder`, `Manifest`.
- Produces:
  - `JOB_SUFFIXES: tuple[str, ...]` = `("main_xfit", "main", "seed1", "seed2", "w400", "alpha1", "d20", "cf", "thin")`.
  - `@dataclass(frozen=True) JobSpec: id: str, encoder: str, suffix: str, argv: tuple[str, ...], outputs: tuple[str, ...], deps: tuple[str, ...], kind: str` where `kind ∈ {"split", "cf", "thin"}`; `outputs[0]` is the primary output (jsonl for split/cf, npz for thin).
  - `Layout` (frozen dataclass): `root: Path` with properties `records`, `geometry`, `arrays`, `done`, `logs`, `hf_parquet(encoder) -> Path` where `hf_parquet` = `root/"hf"/encoder.parquet_file`.
  - `build_jobs(manifest: Manifest, layout: Layout, python: str, runners_dir: Path, threads: int) -> list[JobSpec]`.
  - CLI: `python -m sweep.jobs --root <dir> [--list]` prints one line per job: `id deps -> outputs[0]`.

- [ ] **Step 1: Write the failing test** — `curvature-experiment/tests/test_sweep_jobs.py`:

```python
from pathlib import Path

from sweep.jobs import JOB_SUFFIXES, Layout, build_jobs
from sweep.manifest import load_manifest

RUN = Path("/runners")


def _jobs(tmp_path):
    return build_jobs(load_manifest(), Layout(tmp_path), "/py", RUN, threads=3)


def test_279_jobs_unique_ids(tmp_path):
    jobs = _jobs(tmp_path)
    assert len(jobs) == 31 * 9
    assert len({j.id for j in jobs}) == len(jobs)
    assert {j.suffix for j in jobs} == set(JOB_SUFFIXES)


def test_dependencies(tmp_path):
    by = {j.id: j for j in _jobs(tmp_path)}
    assert by["vit_base__cf"].deps == ("vit_base__main_xfit",)
    assert by["vit_base__thin"].deps == ("vit_base__cf",)
    assert all(by[f"vit_base__{s}"].deps == () for s in JOB_SUFFIXES[:7])


def test_split_argv_flags(tmp_path):
    by = {j.id: j for j in _jobs(tmp_path)}
    a = by["clip_base__main_xfit"].argv
    assert a[:2] == ("/py", str(RUN / "09_physics_probe_facing_split_run.py"))
    for flag in ("--mode", "--fit-seed", "--hessian-xfit", "--geometry-out", "--parquet-path",
                 "--embedding-column", "--label-table", "--device", "--deterministic", "--record-path", "--threads"):
        assert flag in a, flag
    assert a[a.index("--embedding-column") + 1] == "clip_base_galaxies"
    assert a[a.index("--device") + 1] == "cuda"
    assert "--hessian-xfit" not in by["clip_base__main"].argv
    assert by["clip_base__seed1"].argv[by["clip_base__seed1"].argv.index("--fit-seed") + 1] == "1"
    assert "400,400,400" in by["clip_base__w400"].argv
    assert by["clip_base__alpha1"].argv[by["clip_base__alpha1"].argv.index("--alpha") + 1] == "1"
    d20 = by["clip_base__d20"].argv
    assert d20[d20.index("--d-values") + 1] == "20"
    for s in JOB_SUFFIXES[:7]:
        v = by[f"clip_base__{s}"].argv
        assert v[v.index("--d-values") + 1] == ("20" if s == "d20" else "16")


def test_cf_and_thin_chain_paths(tmp_path):
    by = {j.id: j for j in _jobs(tmp_path)}
    lay = Layout(tmp_path)
    geo = lay.geometry / "clip_base" / "09_probe_facing_geometry_d16_seed0.npz"
    cf = by["clip_base__cf"].argv
    assert cf[cf.index("--geometry-npz") + 1] == str(geo)
    arrays = cf[cf.index("--arrays-out") + 1]
    th = by["clip_base__thin"].argv
    assert th[th.index("--arrays-npz") + 1] == arrays
    assert "--device" not in th and "--device" not in cf


def test_record_names_avoid_production_stems(tmp_path):
    for j in _jobs(tmp_path):
        assert not Path(j.outputs[0]).name.startswith(("09_physics_curvature", "09_instrument_adjudication"))
```

- [ ] **Step 2: Run to confirm failure** — `$PY -m pytest -q -p no:cacheprovider tests/test_sweep_jobs.py` → `ImportError`.

- [ ] **Step 3: Write `sweep/jobs.py`**:

```python
"""Expand the encoder manifest x the per-encoder battery into job specs for run_queue."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sweep.manifest import Encoder, Manifest, load_manifest

JOB_SUFFIXES = ("main_xfit", "main", "seed1", "seed2", "w400", "alpha1", "d20", "cf", "thin")
SPLIT = "09_physics_probe_facing_split_run.py"
CF = "09_physics_normal_scaling_run.py"
THIN = "09_physics_normal_scaling_thin_run.py"
# per split job: (fit seed, extra flags, d)
SPLIT_JOBS = {
    "main_xfit": (0, ("--hessian-xfit",), 16),
    "main": (0, (), 16),
    "seed1": (1, (), 16),
    "seed2": (2, (), 16),
    "w400": (0, ("--hidden", "400,400,400"), 16),
    "alpha1": (0, ("--alpha", "1"), 16),
    "d20": (0, (), 20),
}


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def records(self) -> Path: return Path(self.root) / "records"
    @property
    def geometry(self) -> Path: return Path(self.root) / "geometry"
    @property
    def arrays(self) -> Path: return Path(self.root) / "arrays"
    @property
    def done(self) -> Path: return Path(self.root) / "done"
    @property
    def logs(self) -> Path: return Path(self.root) / "logs"

    def hf_parquet(self, enc: Encoder) -> Path:
        return Path(self.root) / "hf" / enc.parquet_file


@dataclass(frozen=True)
class JobSpec:
    id: str
    encoder: str
    suffix: str
    argv: Tuple[str, ...]
    outputs: Tuple[str, ...]
    deps: Tuple[str, ...]
    kind: str


def _common(enc: Encoder, m: Manifest, lay: Layout, threads: int) -> Tuple[str, ...]:
    return ("--parquet-path", str(lay.hf_parquet(enc)), "--embedding-column", enc.column,
            "--label-table", m.label_table, "--threads", str(threads))


def build_jobs(m: Manifest, lay: Layout, python: str, runners_dir: Path, threads: int) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for enc in m.encoders:
        rec = lambda s: str(lay.records / f"scaling__{enc.name}__{s}.jsonl")
        geo_dir = lay.geometry / enc.name
        for s, (seed, extra, d) in SPLIT_JOBS.items():
            argv = (python, str(runners_dir / SPLIT), "--mode", "physics", "--d-values", str(d),
                    "--fit-seed", str(seed), *extra, *_common(enc, m, lay, threads),
                    "--device", "cuda", "--deterministic", "--record-path", rec(s))
            outs: Tuple[str, ...] = (rec(s),)
            if s == "main_xfit":
                argv = argv + ("--geometry-out", str(geo_dir))
                outs = outs + (str(geo_dir / "09_probe_facing_geometry_d16_seed0.npz"),)
            jobs.append(JobSpec(f"{enc.name}__{s}", enc.name, s, argv, outs, (), "split"))
        arrays = str(lay.arrays / f"scaling__{enc.name}__cf.npz")
        cf_argv = (python, str(runners_dir / CF), "--mode", "physics", "--d", "16",
                   "--geometry-npz", str(geo_dir / "09_probe_facing_geometry_d16_seed0.npz"),
                   *_common(enc, m, lay, threads), "--arrays-out", arrays, "--record-path", rec("cf"))
        jobs.append(JobSpec(f"{enc.name}__cf", enc.name, "cf", cf_argv, (rec("cf"), arrays), (f"{enc.name}__main_xfit",), "cf"))
        thin_out = str(lay.arrays / f"scaling__{enc.name}__thin.npz")
        th_argv = (python, str(runners_dir / THIN), "--parquet-path", str(lay.hf_parquet(enc)),
                   "--embedding-column", enc.column, "--arrays-npz", arrays, "--out", thin_out, "--threads", str(threads))
        jobs.append(JobSpec(f"{enc.name}__thin", enc.name, "thin", th_argv, (thin_out,), (f"{enc.name}__cf",), "thin"))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--runners", default=str(Path(__file__).resolve().parents[1] / "runners"))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    for j in build_jobs(load_manifest(), Layout(Path(a.root)), a.python, Path(a.runners), a.threads):
        print(f"{j.id} {','.join(j.deps) or '-'} -> {j.outputs[0]}")


if __name__ == "__main__":
    main()
```
Before finalising, check each flag against the runners' real parsers (`build_parser()` in the split and normal-scaling runners, the thin runner's parser): `--d` (normal-scaling), `--arrays-out`, `--arrays-npz`, `--out`, `--threads` exist as used above. If the thin runner takes no `--label-table`, keep it out (as written). If a flag name differs, use the runner's name and update the test.

- [ ] **Step 4: Tests pass** — `$PY -m pytest -q -p no:cacheprovider tests/test_sweep_jobs.py` → 5 passed. Also `cd $R/curvature-experiment && $PY -m sweep.jobs --root /tmp/x --list | head -3` prints three lines.

- [ ] **Step 5: Commit** — `feat(sweep): expand manifest x battery into 279 job specs` + trailer.

---

### Task 4: Resumable GPU job queue

**Files:**
- Create: `curvature-experiment/sweep/run_queue.py`
- Test: `curvature-experiment/tests/test_sweep_queue.py`

**Interfaces:**
- Consumes: `JobSpec`, `Layout`, `build_jobs`, `load_manifest`.
- Produces:
  - `validate_outputs(job: JobSpec) -> tuple[bool, str]` — split/cf: primary jsonl parses, has an `environment` row and ≥1 row whose `row` is not in `{"environment", "fit"}`; every other listed output exists and is non-empty; thin: npz loads and has key `overlap`.
  - `is_done(job, layout) -> bool` — `done/<id>.done` exists, and its recorded sha256 of `outputs[0]` matches the file on disk.
  - `mark_done(job, layout) -> None` — writes `done/<id>.done` as JSON `{"id", "outputs": {path: sha256}, "finished": iso-time}`.
  - `quarantine(job, layout) -> None` — moves existing outputs of a not-done job to `<path>.partial.<unix-time>`.
  - `run_queue(jobs, layout, gpus: list[str], poll_s: float = 5.0, launcher=subprocess.Popen) -> dict` returning `{"done": [...], "failed": [...], "skipped": [...]}`; runs each ready job with env `CUDA_VISIBLE_DEVICES=<gpu>`, at most one job per GPU, stdout+stderr to `logs/<id>.log`; a job whose runner exits non-zero or fails validation goes to `failed` (not retried in the same invocation), and its dependents are not started.
  - CLI: `python -m sweep.run_queue --root <dir> --gpus 0,1,2 --threads N [--only <substring>] [--dry-run]`.

- [ ] **Step 1: Write the failing tests** — `curvature-experiment/tests/test_sweep_queue.py`. The fake launcher runs a tiny Python snippet instead of a runner:

```python
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sweep.jobs import JobSpec, Layout
from sweep.run_queue import is_done, mark_done, quarantine, run_queue, validate_outputs


def _split_job(lay, jid, body_rows, deps=(), exit_code=0):
    rec = lay.records / f"{jid}.jsonl"
    rows = [{"row": "environment"}] + [{"row": r} for r in body_rows]
    code = ("import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); "
            f"p.write_text(''.join(json.dumps(r)+'\\n' for r in {rows!r})); sys.exit({exit_code})")
    return JobSpec(jid, "enc", jid, (sys.executable, "-c", code, str(rec)), (str(rec),), tuple(deps), "split")


def test_success_marks_done_and_second_run_skips(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])
    r1 = run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert r1["done"] == ["a__main"] and is_done(j, lay)
    r2 = run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert r2["skipped"] == ["a__main"]


def test_exit_zero_without_result_rows_is_not_done(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", [])
    r = run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert r["failed"] == ["a__main"] and not is_done(j, lay)


def test_failed_dependency_blocks_dependents(tmp_path):
    lay = Layout(tmp_path)
    a = _split_job(lay, "a__main_xfit", ["result"], exit_code=1)
    b = _split_job(lay, "a__cf", ["result"], deps=["a__main_xfit"])
    r = run_queue([a, b], lay, gpus=["0", "1"], poll_s=0.05)
    assert r["failed"] == ["a__main_xfit"] and "a__cf" not in r["done"]


def test_partial_record_is_moved_aside_and_rerun(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])
    rec = Path(j.outputs[0]); rec.parent.mkdir(parents=True)
    rec.write_text('{"row": "environment"}\n')           # killed mid-run: no done marker
    r = run_queue([j], lay, gpus=["0"], poll_s=0.05)
    assert r["done"] == ["a__main"]
    assert list(rec.parent.glob("a__main.jsonl.partial.*"))


def test_done_marker_invalid_if_output_changed(tmp_path):
    lay = Layout(tmp_path)
    j = _split_job(lay, "a__main", ["result"])
    run_queue([j], lay, gpus=["0"], poll_s=0.05)
    Path(j.outputs[0]).write_text('{"row": "environment"}\n')
    assert not is_done(j, lay)


def test_one_job_per_gpu(tmp_path):
    lay = Layout(tmp_path)
    seen = []

    def launcher(argv, env=None, stdout=None, stderr=None):
        seen.append(env["CUDA_VISIBLE_DEVICES"])
        return subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr)

    jobs = [_split_job(lay, f"e{i}__main", ["result"]) for i in range(4)]
    run_queue(jobs, lay, gpus=["3", "5"], poll_s=0.05, launcher=launcher)
    assert set(seen) == {"3", "5"} and len(seen) == 4


def test_thin_validation_requires_overlap(tmp_path):
    lay = Layout(tmp_path)
    out = tmp_path / "t.npz"
    np.savez(out, other=np.zeros(2))
    j = JobSpec("a__thin", "a", "thin", (sys.executable, "-c", "pass"), (str(out),), (), "thin")
    ok, why = validate_outputs(j)
    assert not ok and "overlap" in why
```

- [ ] **Step 2: Run to confirm failure** — `ImportError`.

- [ ] **Step 3: Write `sweep/run_queue.py`**:

```python
"""Run sweep jobs on free GPUs, one job per GPU, with done-markers so a pod restart only loses
the jobs that were in flight. Idempotent: rerunning skips jobs whose done-marker still matches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from sweep.jobs import JobSpec, Layout, build_jobs
from sweep.manifest import load_manifest


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_outputs(job: JobSpec) -> Tuple[bool, str]:
    primary = Path(job.outputs[0])
    if not primary.exists() or primary.stat().st_size == 0:
        return False, f"missing or empty {primary}"
    if job.kind == "thin":
        with np.load(primary) as z:
            if "overlap" not in z.files:
                return False, f"{primary} has no 'overlap' array"
    else:
        try:
            rows = [json.loads(l) for l in primary.read_text().splitlines() if l.strip()]
        except json.JSONDecodeError as e:
            return False, f"{primary} is not valid jsonl: {e}"
        kinds = [r.get("row") for r in rows]
        if "environment" not in kinds:
            return False, f"{primary} has no environment row"
        if not any(k not in ("environment", "fit") for k in kinds):
            return False, f"{primary} has no result rows"
    for extra in job.outputs[1:]:
        p = Path(extra)
        if not p.exists() or p.stat().st_size == 0:
            return False, f"missing or empty {p}"
    return True, "ok"


def _marker(job: JobSpec, lay: Layout) -> Path:
    return lay.done / f"{job.id}.done"


def is_done(job: JobSpec, lay: Layout) -> bool:
    m = _marker(job, lay)
    if not m.exists():
        return False
    rec = json.loads(m.read_text())
    for path, sha in rec["outputs"].items():
        p = Path(path)
        if not p.exists() or _sha256(p) != sha:
            return False
    return True


def mark_done(job: JobSpec, lay: Layout) -> None:
    lay.done.mkdir(parents=True, exist_ok=True)
    rec = {"id": job.id, "outputs": {o: _sha256(Path(o)) for o in job.outputs},
           "finished": datetime.now(timezone.utc).isoformat()}
    _marker(job, lay).write_text(json.dumps(rec, indent=1))


def quarantine(job: JobSpec, lay: Layout) -> None:
    stamp = int(time.time())
    for o in job.outputs:
        p = Path(o)
        if p.exists():
            p.rename(p.with_name(f"{p.name}.partial.{stamp}"))


def run_queue(jobs: List[JobSpec], lay: Layout, gpus: List[str], poll_s: float = 5.0,
              launcher: Callable = subprocess.Popen) -> Dict[str, List[str]]:
    lay.logs.mkdir(parents=True, exist_ok=True)
    status: Dict[str, str] = {}
    out: Dict[str, List[str]] = {"done": [], "failed": [], "skipped": []}
    for j in jobs:
        if is_done(j, lay):
            status[j.id] = "done"; out["skipped"].append(j.id)
    pending = [j for j in jobs if j.id not in status]
    running: Dict[str, Tuple[JobSpec, subprocess.Popen, object]] = {}   # gpu -> (job, proc, logfile)
    while pending or running:
        for gpu, (j, proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close(); del running[gpu]
            ok, why = (validate_outputs(j) if proc.returncode == 0 else (False, f"exit {proc.returncode}"))
            if ok:
                mark_done(j, lay); status[j.id] = "done"; out["done"].append(j.id)
            else:
                status[j.id] = "failed"; out["failed"].append(j.id)
                print(f"[queue] FAILED {j.id}: {why}", flush=True)
        blocked = [j for j in pending if any(status.get(d) == "failed" for d in j.deps)]
        for j in blocked:
            pending.remove(j); status[j.id] = "blocked"
            print(f"[queue] blocked {j.id} (dependency failed)", flush=True)
        free = [g for g in gpus if g not in running]
        for j in [j for j in pending if all(status.get(d) == "done" for d in j.deps)]:
            if not free:
                break
            gpu = free.pop(0)
            quarantine(j, lay)
            for o in j.outputs:
                Path(o).parent.mkdir(parents=True, exist_ok=True)
            log = open(lay.logs / f"{j.id}.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            running[gpu] = (j, launcher(list(j.argv), env=env, stdout=log, stderr=subprocess.STDOUT), log)
            pending.remove(j)
            print(f"[queue] start {j.id} on GPU {gpu}", flush=True)
        if running:
            time.sleep(poll_s)
        elif pending and not free:
            time.sleep(poll_s)
        elif pending:
            # nothing running and nothing ready: remaining jobs wait on blocked/failed deps
            for j in pending:
                status[j.id] = "blocked"
            pending = []
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gpus", required=True, help="comma-separated GPU indices that are free, e.g. 0,1,2")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--runners", default=str(Path(__file__).resolve().parents[1] / "runners"))
    ap.add_argument("--only", default=None, help="run only jobs whose id contains this substring")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    lay = Layout(Path(a.root))
    jobs = build_jobs(load_manifest(), lay, a.python, Path(a.runners), a.threads)
    if a.only:
        wanted = {j.id for j in jobs if a.only in j.id}
        needed = set(wanted)
        by = {j.id: j for j in jobs}
        for jid in list(wanted):
            stack = list(by[jid].deps)
            while stack:
                d = stack.pop(); needed.add(d); stack.extend(by[d].deps)
        jobs = [j for j in jobs if j.id in needed]
    if a.dry_run:
        for j in jobs:
            print(("DONE " if is_done(j, lay) else "TODO ") + j.id)
        return
    res = run_queue(jobs, lay, [g.strip() for g in a.gpus.split(",") if g.strip()])
    print(json.dumps({k: len(v) for k, v in res.items()}), flush=True)
    if res["failed"]:
        print("failed: " + ", ".join(res["failed"]), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Tests pass** — `$PY -m pytest -q -p no:cacheprovider tests/test_sweep_queue.py` → 7 passed.

- [ ] **Step 5: Commit** — `feat(sweep): resumable one-job-per-GPU queue with done-markers` + trailer.

---

### Task 5: Record extraction and aggregation

**Files:**
- Create: `curvature-experiment/sweep/extract.py`, `curvature-experiment/sweep/aggregate.py`
- Test: `curvature-experiment/tests/test_sweep_extract.py`, `curvature-experiment/tests/test_sweep_aggregate.py`

**Interfaces:**
- Consumes: `load_manifest`, `Layout`, `JOB_SUFFIXES`.
- Produces (`sweep/extract.py`):
  - `LABELS = ("mag_r", "photo_z", "smooth_fraction", "stellar_mass")`
  - `read_rows(path) -> list[dict]` (empty list if the file is missing)
  - `split_cells(rows) -> dict[(label, column), {"partial": float, "p": float}]` from `row == "result"` records' `columns[column]["multiscale"]` (the same access as `appendix_gen.py:34` and `:195`)
  - `xfit_cells(rows) -> dict[(label, column), {"fitA_scoreB": {...}, "fitB_scoreA": {...}}]` from `row == "xfit"` records' `columns[column]` (as `appendix_gen.py:62,224`)
  - `global_r2(rows) -> dict[label, float]` (`global_oof_r2` of result rows, as `appendix_gen.py:192`)
  - `var_explained(rows) -> float | None` (from the `row == "fit"` record)
  - `cf_summary(npz_path) -> dict[label, dict]` — per label and variant `S_model` / `random_qmatched`: `help`, `hurt`, `d_r2_plus`, `d_r2_minus`, and for `S_model` also `t_star`, computed exactly as `appendix_gen.py` lines of section "E: counterfactual normal scaling" (`m = isfinite(eq); dp = cv[m,4]-cv[m,2]; dm = cv[m,0]-cv[m,2]; ts = eq[m]/max(qq[m],1e-300)`; help = mean(dp>0), hurt = mean(dm<0), medians of dp, dm, ts)
  - `sign_test(cf_npz, thin_npz, thr=0.05) -> dict[label, {"n", "help", "hurt", "p_help", "p_hurt"}]` — the min-degree greedy independent set `_indep` and binomial tests exactly as `appendix_gen.py` "thinned-anchor sign tests" block.
- Produces (`sweep/aggregate.py`): `aggregate(manifest, records_dir: Path, arrays_dir: Path, out_dir: Path) -> None` writing `tab_scaling_main.tex`, `tab_scaling_xfit.tex`, `tab_scaling_cf.tex`, `tab_scaling_robust.tex`, `fig_scaling_partials.{pdf,png}`, `fig_scaling_cf.{pdf,png}`, `fig_scaling_robust.{pdf,png}`, `SCALING_REPORT.md`; CLI `python -m sweep.aggregate --records <dir> --arrays <dir> --out <dir>`.

- [ ] **Step 1: Port the extraction and pin it against the paper's own records.** Write `tests/test_sweep_extract.py` first. It checks that the new readers reproduce cells printed in the committed manuscript, using the paper's CPU records in the record store (read-only; skips if absent). This is a correctness test of the extraction code, not a platform comparison:

```python
import os
import re
from pathlib import Path

import pytest

from sweep.extract import cf_summary, read_rows, split_cells

REC = Path(os.environ.get("EFFDIM_RECORDS", "/home/akagi/Documents/Projects/EffDim/notebooks/.cache"))
MAIN_TEX = Path(__file__).resolve().parents[2] / "paper" / "latex" / "main.tex"
needs = pytest.mark.skipif(not (REC / "09_physics_probe_facing_split_clip_base.jsonl").exists(), reason="paper records absent")


def _fmt(v):
    s = f"${v['partial']:+.2f}" + ("^{*}" if v["p"] > 0.05 else "") + "$"
    return s


@needs
def test_split_cells_match_tab_xenc():
    cells = split_cells(read_rows(REC / "09_physics_probe_facing_split_clip_base.jsonl"))
    tex = MAIN_TEX.read_text()
    block = tex[tex.index(r"\label{tab:xenc}") - 6000: tex.index(r"\label{tab:xenc}")]
    for lab in ("mag_r", "photo_z"):
        s = _fmt(cells[(lab, "hess_mismatch_emp")])
        assert s in block, (lab, s)


@needs
def test_cf_summary_matches_tab_cf():
    z = REC / "09_physics_normal_scaling_clip_base_d16.npz"
    cf = cf_summary(z)
    tex = MAIN_TEX.read_text()
    block = tex[tex.index(r"\label{tab:cf}") - 8000: tex.index(r"\label{tab:cf}")]
    v = cf["mag_r"]["S_model"]
    assert f"{v['help']:.2f}" in block and f"{v['t_star']:.1f}" in block
```

Run: `$PY -m pytest -q -p no:cacheprovider tests/test_sweep_extract.py` → ImportError (then, after Step 2, PASS).

- [ ] **Step 2: Write `sweep/extract.py`** by porting the field access and computations from `paper/generate/appendix_gen.py` (read it; do not import or run it). Code:

```python
"""Record readers for the scaling sweep, ported from paper/generate/appendix_gen.py so the
sweep tables read records exactly the way the paper's appendix does."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest

LABELS = ("mag_r", "photo_z", "smooth_fraction", "stellar_mass")


def read_rows(path) -> List[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def split_cells(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    return {(r["label"], c): v["multiscale"] for r in rows if r.get("row") == "result"
            for c, v in r["columns"].items() if isinstance(v, dict) and "multiscale" in v}


def xfit_cells(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    return {(r["label"], c): v for r in rows if r.get("row") == "xfit" for c, v in r["columns"].items()}


def global_r2(rows: List[dict]) -> Dict[str, float]:
    return {r["label"]: r["global_oof_r2"] for r in rows if r.get("row") == "result" and "global_oof_r2" in r}


def var_explained(rows: List[dict]):
    fits = [r["var_explained"] for r in rows if r.get("row") == "fit"]
    return fits[0] if fits else None


def cf_summary(npz_path) -> Dict[str, Dict[str, dict]]:
    z = np.load(npz_path)
    out: Dict[str, Dict[str, dict]] = {}
    for lab in LABELS:
        if f"{lab}:S_model:eq" not in z.files:
            continue
        out[lab] = {}
        for var in ("S_model", "random_qmatched"):
            eq, qq, cv = z[f"{lab}:{var}:eq"], z[f"{lab}:{var}:qq"], z[f"{lab}:{var}:r2_curve"]
            m = np.isfinite(eq); dp = cv[m, 4] - cv[m, 2]; dm = cv[m, 0] - cv[m, 2]
            v = {"help": float(np.mean(dp > 0)), "hurt": float(np.mean(dm < 0)),
                 "d_r2_plus": float(np.median(dp)), "d_r2_minus": float(np.median(dm))}
            if var == "S_model":
                v["t_star"] = float(np.median(eq[m] / np.maximum(qq[m], 1e-300)))
            out[lab][var] = v
    return out


def _indep(ov: np.ndarray, thr: float) -> np.ndarray:
    A = ov > thr; np.fill_diagonal(A, False); alive = np.ones(ov.shape[0], bool); keep = np.zeros(ov.shape[0], bool)
    while alive.any():
        deg = (A & alive[None, :]).sum(1); deg[~alive] = 10**9
        i = int(np.argmin(deg)); keep[i] = True; alive[i] = False; alive[A[i]] = False
    return keep


def sign_test(cf_npz, thin_npz, thr: float = 0.05) -> Dict[str, dict]:
    z = np.load(cf_npz); keep = _indep(np.load(thin_npz)["overlap"].astype(float), thr)
    out = {}
    for lab in LABELS:
        if f"{lab}:S_model:eq" not in z.files:
            continue
        eq, cv = z[f"{lab}:S_model:eq"], z[f"{lab}:S_model:r2_curve"]
        m = np.isfinite(eq) & keep
        dp = cv[m, 4] - cv[m, 2]; dm = cv[m, 0] - cv[m, 2]
        n = int(m.sum()); kh = int((dp > 0).sum()); ku = int((dm < 0).sum())
        out[lab] = {"n": n, "help": kh / n if n else float("nan"), "hurt": ku / n if n else float("nan"),
                    "p_help": binomtest(kh, n, 0.5, alternative="greater").pvalue if n else float("nan"),
                    "p_hurt": binomtest(ku, n, 0.5, alternative="greater").pvalue if n else float("nan")}
    return out
```
Before trusting `sign_test`, compare it line by line with appendix_gen's thinned block (`for enc, d, stem in runs:` after `thin = {...}`): the mask it applies (`isfinite` and `keep`), and whether it indexes `r2_curve` columns 4/2/0 identically. Port any difference verbatim and keep the test green.

- [ ] **Step 3: Write the aggregation tests** — `tests/test_sweep_aggregate.py`, with synthetic records (two encoders, one with missing jobs):

```python
import json
from pathlib import Path

import numpy as np

from sweep.aggregate import aggregate
from sweep.extract import LABELS
from sweep.manifest import load_manifest

COLS = ("hess_mismatch_emp", "align_cos_tan", "hess_mismatch_dec", "pf_rad")


def _split(path, partial, p=0.001):
    rows = [{"row": "environment", "device": "cuda"}, {"row": "fit", "var_explained": 0.95}]
    for lab in LABELS:
        rows.append({"row": "result", "label": lab, "global_oof_r2": 0.5,
                     "columns": {c: {"multiscale": {"partial": partial, "p": p}} for c in COLS}})
        rows.append({"row": "xfit", "label": lab, "hessian_split_half_cos_p50": 0.9,
                     "columns": {c: {"fitA_scoreB": {"partial": partial, "p": p}, "fitB_scoreA": {"partial": partial, "p": p}} for c in COLS}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _cf(path, n=40, seed=0):
    rng = np.random.default_rng(seed); d = {}
    for lab in LABELS:
        for var in ("S_model", "random_qmatched"):
            cv = rng.normal(size=(n, 5)); cv[:, 4] = cv[:, 2] + 0.1; cv[:, 0] = cv[:, 2] - 0.1
            d[f"{lab}:{var}:eq"] = rng.uniform(1, 2, n); d[f"{lab}:{var}:qq"] = np.ones(n); d[f"{lab}:{var}:r2_curve"] = cv
    path.parent.mkdir(parents=True, exist_ok=True); np.savez(path, **d)


def _thin(path, n=40):
    np.savez(path, overlap=np.zeros((n, n)))


def _fixture(tmp_path, complete=("vit_base", "clip_base"), partial=("dinov3_vitb16",)):
    rec, arr = tmp_path / "records", tmp_path / "arrays"
    for e in complete:
        for s in ("main_xfit", "main", "seed1", "seed2", "w400", "alpha1", "d20"):
            _split(rec / f"scaling__{e}__{s}.jsonl", -0.3 if s != "seed2" else 0.1)
        _cf(arr / f"scaling__{e}__cf.npz"); _thin(arr / f"scaling__{e}__thin.npz")
    for e in partial:
        _split(rec / f"scaling__{e}__main_xfit.jsonl", -0.2)
    return rec, arr


def test_aggregate_writes_all_outputs(tmp_path):
    rec, arr = _fixture(tmp_path)
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out)
    for f in ("tab_scaling_main.tex", "tab_scaling_xfit.tex", "tab_scaling_cf.tex", "tab_scaling_robust.tex",
              "fig_scaling_partials.pdf", "fig_scaling_cf.png", "fig_scaling_robust.pdf", "SCALING_REPORT.md"):
        assert (out / f).exists() and (out / f).stat().st_size > 0, f


def test_aggregate_with_missing_jobs(tmp_path):
    rec, arr = _fixture(tmp_path)
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out)
    main = (out / "tab_scaling_main.tex").read_text()
    assert "--" in main                                   # encoders with no records
    rep = (out / "SCALING_REPORT.md").read_text()
    assert "3 of 31 encoders have records" in rep
    assert "2 of 31 encoders complete all 9 jobs" in rep


def test_robust_counts_sign_changes(tmp_path):
    rec, arr = _fixture(tmp_path)
    out = tmp_path / "out"
    aggregate(load_manifest(), rec, arr, out)
    rob = (out / "tab_scaling_robust.tex").read_text()
    assert "vit_base" in rob.replace("\\_", "_") and "1" in rob   # seed2 flips the sign once


def test_aggregate_is_deterministic(tmp_path):
    rec, arr = _fixture(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    aggregate(load_manifest(), rec, arr, a); aggregate(load_manifest(), rec, arr, b)
    for f in ("tab_scaling_main.tex", "tab_scaling_cf.tex", "SCALING_REPORT.md", "fig_scaling_cf.png"):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
```
Run → ImportError.

- [ ] **Step 4: Write `sweep/aggregate.py`.** Required behaviour (each point is exercised by the tests above or the Task 9 regeneration test):
  - `cell(v)` formats exactly as `appendix_gen.py`'s `cell`: `"--"` for None/NaN partial, else `f"${partial:+.2f}" + ("^{*}" if p > 0.05 else "") + "$"`.
  - Encoder order: manifest order grouped by `family`, sorted within family by `params`. Display name: the manifest `name` with `_` escaped as `\_`.
  - `tab_scaling_main.tex`: one `table*` with rows = encoders, columns per label = (mismatch `hess_mismatch_emp`, alignment `align_cos_tan`) from `scaling__<e>__main.jsonl`, plus a `var.\ expl.` column from its fit row; `--` when missing. Caption states: rank-partial Spearman with local R² under the multi-scale density control; d=16; decoder seed 0; GPU (device from the environment rows); `^{*}` not significant at 0.05.
  - `tab_scaling_xfit.tex`: from `scaling__<e>__main_xfit.jsonl` xfit rows, `hess_mismatch_dec` fitA_scoreB / fitB_scoreA and `align_cos_tan` both halves, as `tab:xencx`.
  - `tab_scaling_cf.tex`: from `cf_summary(arrays/scaling__<e>__cf.npz)`: for each encoder and label, help, hurt, ΔR²(+1), ΔR²(−1), t* for `S_model`, then help, hurt for `random_qmatched`; plus a final column with the sign-test `p_help` upper bound from `sign_test(...)` formatted with appendix_gen's `_ptex` (port it verbatim).
  - `tab_scaling_robust.tex`: per encoder and label, for mismatch and alignment: min–max of the partial over the 6 split variants (main, seed1, seed2, w400, alpha1, d20) and the number of variants whose sign differs from `main`'s.
  - Figures with `matplotlib` using the `Agg` backend, `svg.hashsalt` fixed and PDF metadata `{"CreationDate": None, "ModDate": None}` and PNG metadata `{"Software": None}` so output bytes are deterministic; x = log10(params); `fig_scaling_partials`: 2×4 panels (mismatch/alignment × label), points coloured by family, the 5 paper encoders ringed; `fig_scaling_cf`: help fraction (S_model) vs log params per label, with random_qmatched as hollow markers and a line at 0.5; `fig_scaling_robust`: per encoder, a vertical min–max bar for each label's mismatch partial.
  - `SCALING_REPORT.md`: first lines exactly `N of 31 encoders have records` and `M of 31 encoders complete all 9 jobs`; then, per paper claim, counts over encoders with that record: (a) mismatch partial negative and significant, per label; (b) alignment partial sign and significance, per label, reported as counts of negative-significant / positive-significant / non-significant; (c) counterfactual help > hurt-of-random and help > 0.5 per label; (d) sign-test p_help < 0.05 per label; then a bullet list of encoders that break each claim. All numbers computed; no literals.
  - CLI `main()` with `--records --arrays --out`, defaulting to `curvature-experiment/.cache/scaling/{records,arrays}` and `curvature-experiment/results/scaling`.

- [ ] **Step 5: Tests pass** — `EFFDIM_CACHE_DIR=$(mktemp -d) $PY -m pytest -q -p no:cacheprovider tests/test_sweep_extract.py tests/test_sweep_aggregate.py` → all pass (extract tests run because the paper records are present locally).

- [ ] **Step 6: Commit** — `feat(sweep): record extraction (ported from appendix_gen) and scaling aggregation` + trailer.

---

### Task 6: Pod setup script and runbook

**Files:**
- Create: `curvature-experiment/sweep/setup_pod.sh`, `curvature-experiment/sweep/POD_RUNBOOK.md`

**Interfaces:**
- Produces: `setup_pod.sh` — idempotent; creates `/mnt/ssd-cluster/EffDim/{repo,venv,sweep-out,hf-cache}` as needed; clones or fast-forwards the repo at branch `encoder-scaling` into `/mnt/ssd-cluster/EffDim/repo`; creates the venv at `/mnt/ssd-cluster/EffDim/venv` (if absent) and installs `curvature-experiment/requirements.txt` minus the `torch` pin, then the pod's CUDA torch wheel matching the torch version pinned in requirements (`pip install torch==<pinned> --index-url https://download.pytorch.org/whl/cu121`); downloads the 31 parquets (only missing ones) to `sweep-out/hf/physics/` via `huggingface_hub.hf_hub_download(repo_id, filename, repo_type="dataset", revision=<snapshot>, local_dir=sweep-out/hf, cache_dir=/mnt/ssd-cluster/EffDim/hf-cache)`; prints `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` and `cat /sys/fs/cgroup/cpu.max`.

- [ ] **Step 1: Write `setup_pod.sh`**:

```bash
#!/usr/bin/env bash
# Idempotent environment setup for the encoder-scaling sweep on the EleutherAI pod.
# Everything lives under /mnt/ssd-cluster/EffDim (the pod's /root is wiped on restart).
set -euo pipefail
BASE=/mnt/ssd-cluster/EffDim
REPO=$BASE/repo; VENV=$BASE/venv; OUT=$BASE/sweep-out; HF=$BASE/hf-cache
BRANCH=${BRANCH:-encoder-scaling}
SNAPSHOT=bc081f8a5db4767edcd958653d96efde9137de0b
mkdir -p "$BASE" "$OUT" "$HF"
if [ -d "$REPO/.git" ]; then git -C "$REPO" fetch -q origin "$BRANCH" && git -C "$REPO" checkout -q "$BRANCH" && git -C "$REPO" merge -q --ff-only "origin/$BRANCH"
else git clone -q --branch "$BRANCH" https://github.com/amanasci/EffDim.git "$REPO"; fi
if [ ! -x "$VENV/bin/python" ]; then python3 -m venv "$VENV"; fi
TORCH_PIN=$(grep -E '^torch==' "$REPO/curvature-experiment/requirements.txt" | sed 's/+cpu//')
grep -vE '^torch==' "$REPO/curvature-experiment/requirements.txt" > "$BASE/requirements-gpu.txt"
"$VENV/bin/pip" install -q -r "$BASE/requirements-gpu.txt"
"$VENV/bin/pip" install -q "$TORCH_PIN" --index-url https://download.pytorch.org/whl/cu121
export HF_HOME=$HF
"$VENV/bin/python" - "$REPO" "$OUT" "$SNAPSHOT" <<'EOF'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, out, snap = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(repo / "curvature-experiment"))
from sweep.manifest import load_manifest
m = load_manifest()
assert m.snapshot == snap, (m.snapshot, snap)
for e in m.encoders:
    dest = out / "hf" / e.parquet_file
    if dest.exists() and dest.stat().st_size > 0:
        continue
    hf_hub_download(m.repo, e.parquet_file, repo_type="dataset", revision=snap, local_dir=str(out / "hf"))
    print("downloaded", e.parquet_file, flush=True)
if not Path(m.label_table).exists():
    raise SystemExit(f"label table missing on the pod: {m.label_table}")
print("parquets and label table present")
EOF
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo "cpu.max not readable"
```
Check the torch pin in `requirements.txt` first (`grep torch curvature-experiment/requirements.txt`). If it reads `torch==2.13.0+cpu`, the `sed` above strips `+cpu`. If the PyTorch index has no cu121 wheel for that version, use the CUDA index the torch release notes list for it (cu124/cu126) and record the choice in the runbook.

- [ ] **Step 2: Write `POD_RUNBOOK.md`** with, verbatim, these sections:
  1. *Before any SSH:* read `docs/remote-compute/eleutherai-pod-user-guide.md`; then `ssh root@216.153.49.26 'sha256sum /root/user-guide.md'` and compare with `sha256sum docs/remote-compute/eleutherai-pod-user-guide.md`; stop and re-read if different.
  2. *Setup / after a restart:* `ssh root@216.153.49.26`, `tmux new -As effdim`, `bash /mnt/ssd-cluster/EffDim/repo/curvature-experiment/sweep/setup_pod.sh` (first time, the repo is not on the pod yet: `mkdir -p /mnt/ssd-cluster/EffDim && git clone --branch encoder-scaling https://github.com/amanasci/EffDim.git /mnt/ssd-cluster/EffDim/repo`; if the pod cannot clone it (private repo, no credentials), push it from this machine instead: `rsync -a --exclude .venv --exclude 'notebooks/.cache' --exclude 'curvature-experiment/.cache' /home/akagi/Documents/Projects/EffDim/ root@216.153.49.26:/mnt/ssd-cluster/EffDim/repo/` and set `BRANCH=` empty handling in `setup_pod.sh` so it skips the git fetch when `$REPO/.git` has no `origin` access — write whichever path worked into the runbook).
  3. *Choose GPUs and threads:* from `nvidia-smi`, the GPU indices with no processes (`nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv` to see busy ones); threads = floor(cgroup CPUs / number of GPUs used), at least 2.
  4. *Run:* inside tmux, `cd /mnt/ssd-cluster/EffDim/repo/curvature-experiment && /mnt/ssd-cluster/EffDim/venv/bin/python -m sweep.run_queue --root /mnt/ssd-cluster/EffDim/sweep-out --gpus <list> --threads <n> [--only <substr>] 2>&1 | tee -a /mnt/ssd-cluster/EffDim/sweep-out/queue.log`; detach with Ctrl+B D.
  5. *Progress:* `python -m sweep.run_queue ... --dry-run | sort | uniq -c -w4`; `tail -n 20 sweep-out/logs/<id>.log`.
  6. *Restart recovery:* re-run §2, then §4 unchanged; done jobs are skipped, partial outputs are moved aside.
  7. *Fetch results:* from this machine: `rsync -a root@216.153.49.26:/mnt/ssd-cluster/EffDim/sweep-out/{records,arrays,done} curvature-experiment/.cache/scaling/` (geometry npz stays on the pod).
  8. *Rules:* the Global Constraints' pod bullets, copied.

- [ ] **Step 3: Check the script parses** — `bash -n curvature-experiment/sweep/setup_pod.sh` → no output. `shellcheck` if installed.

- [ ] **Step 4: Commit** — `feat(sweep): pod setup script and runbook` + trailer. Then push the branch so the pod can clone it: `git -C $R push -u origin encoder-scaling`.

---

### Task 7: Pod bring-up, determinism check and memory probe

This task runs remote commands on the shared pod. Follow `POD_RUNBOOK.md` exactly; every Global Constraints pod rule applies.

**Files:** none changed locally unless a fix is needed (then TDD in the relevant task's files, gate, commit, push, `setup_pod.sh` again).

- [ ] **Step 1: Guide check** — read the local guide; compare sha256 with the remote `/root/user-guide.md` (runbook §1). Record both hashes in the task report.

- [ ] **Step 2: Setup** — runbook §2 inside `tmux new -As effdim`. Expected final lines: `parquets and label table present`, `torch <ver> cuda <ver> available True`, a GPU table, the cgroup CPU limit. Record them.

- [ ] **Step 3: GPU smoke determinism** — on one free GPU, twice, in tmux:

```bash
cd /mnt/ssd-cluster/EffDim/repo/curvature-experiment
for i in 1 2; do CUDA_VISIBLE_DEVICES=<gpu> EFFDIM_09_OUTPUT_ROOT=/mnt/ssd-cluster/EffDim/sweep-out/smoke/out$i /mnt/ssd-cluster/EffDim/venv/bin/python runners/09_physics_probe_facing_split_run.py --mode smoke --threads <n> --device cuda --deterministic --record-path /mnt/ssd-cluster/EffDim/sweep-out/smoke/run$i.jsonl; done
/mnt/ssd-cluster/EffDim/venv/bin/python - <<'EOF'
import json
a=[json.loads(l) for l in open('/mnt/ssd-cluster/EffDim/sweep-out/smoke/run1.jsonl')]
b=[json.loads(l) for l in open('/mnt/ssd-cluster/EffDim/sweep-out/smoke/run2.jsonl')]
strip=lambda r:{k:v for k,v in r.items() if 'timestamp' not in k and 'wallclock' not in k and k not in ('record_path','geometry_root')}
print("IDENTICAL" if [strip(r) for r in a]==[strip(r) for r in b] else "DIFFERENT")
EOF
```
Expected: `IDENTICAL`, and the environment rows show `device: cuda`, `deterministic: true`. If `use_deterministic_algorithms` raises on an op, stop and report BLOCKED with the op name (spec: never drop the flag silently).

- [ ] **Step 4: Memory probe on the largest encoder** — run `llava_15_13b__main_xfit` alone: `python -m sweep.run_queue --root ... --gpus <gpu> --threads <n> --only llava_15_13b__main_xfit`. Watch `nvidia-smi` for peak memory. Expected: job done. If CUDA OOM: stop and report BLOCKED with the log tail (the fix — a smaller geometry chunk on GPU — is a design change).

- [ ] **Step 5: Report** — task report with the hashes, setup output, determinism result, peak GPU memory and wall time for the probe job.

---

### Task 8: One encoder end to end (vit_base)

- [ ] **Step 1:** runbook §4 with `--only vit_base` on as many free GPUs as available (7 independent jobs then cf then thin). Expected: `{"done": 9, "failed": 0, ...}`.
- [ ] **Step 2: Fetch and aggregate locally** — runbook §7 into `curvature-experiment/.cache/scaling/`; then `cd $R/curvature-experiment && $PY -m sweep.aggregate --out /tmp/scaling-vitb`. Expected: all outputs written; `SCALING_REPORT.md` starts `1 of 31 encoders have records` / `1 of 31 encoders complete all 9 jobs`.
- [ ] **Step 3: Sanity read** — in the report, list for vit_base: global R² per label, var. expl., the mismatch/alignment partials, the cf help fractions. Flag anything non-finite or any var. expl. below 0.9. No comparison with the paper's CPU numbers is required or made (spec).
- [ ] **Step 4: Timing** — per-job wall time from `logs/*.log` and done-marker times; estimate the full-sweep duration for the GPU count available; record it.

---

### Task 9: Full sweep, results and final checks

- [ ] **Step 1: Run all encoders** — runbook §4 without `--only`, in tmux, on the free GPUs. Monitor with runbook §5 at sensible intervals (the sweep takes hours). On a pod restart, runbook §6.
- [ ] **Step 2: Failures** — for each failed job, read its log tail. A reproducible code failure: fix with TDD in the owning task's files, rerun the gate if a runner changed, commit, push, `setup_pod.sh`, rerun the queue (done jobs are skipped). Environmental failures (GPU taken, pod restart): rerun. Stop and report if the same job fails twice for a non-environmental reason.
- [ ] **Step 3: Fetch** — runbook §7. Then write `curvature-experiment/results/scaling/SHA256SUMS` = `cd curvature-experiment/.cache/scaling && find records arrays -type f -name 'scaling__*' | sort | xargs sha256sum`.
- [ ] **Step 4: Aggregate** — `$PY -m sweep.aggregate` (defaults write to `curvature-experiment/results/scaling/`). Expected: report begins `31 of 31 encoders have records` / `31 of 31 encoders complete all 9 jobs`.
- [ ] **Step 5: Regeneration test** — add to `tests/test_sweep_aggregate.py`:

```python
import os
import filecmp
import pytest

SC = Path(__file__).resolve().parents[1] / ".cache" / "scaling"
RES = Path(__file__).resolve().parents[1] / "results" / "scaling"


@pytest.mark.skipif(not (SC / "records").exists(), reason="scaling records absent")
def test_committed_results_regenerate(tmp_path):
    aggregate(load_manifest(), SC / "records", SC / "arrays", tmp_path)
    for f in sorted(p.name for p in RES.iterdir() if p.suffix in (".tex", ".md", ".png", ".pdf")):
        assert filecmp.cmp(RES / f, tmp_path / f, shallow=False), f
```
Run it → PASS.
- [ ] **Step 6: Final checks** — `EFFDIM_CACHE_DIR=$(mktemp -d) $PY -m pytest -q -p no:cacheprovider curvature-experiment/tests`; `cd paper && EFFDIM_CACHE_DIR=/home/akagi/Documents/Projects/EffDim/notebooks/.cache $PY -m pytest -q -p no:cacheprovider tests`; the CPU gate (`gate.sh $R scaling-final` → GATE PASS); `git diff --exit-code linear-probes-curvature -- paper/latex/main.tex`.
- [ ] **Step 7: Commit** — `results/scaling/` (tables, figures, report, SHA256SUMS) and the regeneration test: `feat(results): encoder scaling across 31 PU galaxy encoders` + trailer; push.
