# Paper Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the EffDim repo to the code that supports "Linear Probes on Curved Latent Spaces" (ML4PS 2026), with the manuscript in `paper/` and its supporting code in `curvature-experiment/`, everything else moved verbatim to `archive/`, and prove that no kept number changed.

**Architecture:** Stage-by-stage, in-place restructure on branch `simplify/paper-closure`. Before any code moves, a golden baseline is captured (seeded smoke runs of every kept runner, the kept test suite, and every generator run against the real record cache). After each stage a single gate script re-runs all of it and requires bit-identical records, byte-identical `main.tex` and table output, and pixel-identical figures. One commit per stage.

**Tech Stack:** Python 3 (repo venv `/home/akagi/Documents/Projects/EffDim/.venv`), PyTorch (CPU), numpy/scipy/scikit-learn, pytest, jupyter nbconvert, git, XeLaTeX/pdflatex/bibtex.

**Spec:** `docs/superpowers/specs/2026-09-23-paper-closure-design.md` (read it first; this plan argues from it).

## Global Constraints

- Work only in the worktree `/home/akagi/Documents/Projects/EffDim/.claude/worktrees/paper-closure` (branch `simplify/paper-closure`). Referred to below as `$WT`.
- **Never modify** the main checkout `/home/akagi/Documents/Projects/EffDim` (branch `fixture-validity-audit`, holds the user's uncommitted submitted paper). Read-only copies from it are allowed.
- **Never write** into `/home/akagi/Documents/Projects/EffDim/notebooks/.cache` (8.7G of records). Read-only.
- Nothing is deleted: every file or function that leaves the closure goes to `archive/` verbatim.
- No renames of kept public symbols. No change to numerics, defaults, dtypes, RNG call order or seeds. No reformatting of lines you are not otherwise changing.
- The `effdim` package stays untouched: `src/effdim`, `tests/`, `benchmarks/`, `pyproject.toml`, `.github/`, `mkdocs.yml`, `docs/*.md`, `docs/tutorials`, `docs/javascripts`, `README.md` (only one pointer line added), `MANIFEST.in`, `LICENSE`.
- `paper/latex/main.tex` must stay byte-identical to the Stage 0 snapshot for the whole plan.
- Runner file names keep their original `09_*` names.
- After Task 9, dependency is one-way: `paper/` finds records only through `paper/records.py`; nothing in `curvature-experiment/` references `paper/`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Python for everything: `PY=/home/akagi/Documents/Projects/EffDim/.venv/bin/python`.
- Harness: `H=$WT/docs/superpowers/harness` (scripts, committed). Work data: `export CLOSURE_WORK=$HOME/.cache/effdim-closure` (baseline, gate runs, colleague checkout; not committed).
- A gate failure means **stop and report**, never "adjust the baseline". Two exceptions, both with proof: widening the colleague filter in Task 4 (shown to drop only colleague material), and replacing the baseline `main.tex` in Task 10 Step 6 (shown byte-equal to the committed paper).

## Review Focus

1. **Code reached only in full physics mode** (`--fit-seed`, `--hidden`, `--geometry-out`, `--parquet-path`, `--embedding-column`, `--label-table`) is not run by smoke. A reasonable reviewer expects those paths to still work. The finder keeps anything named from a kept file, so they survive. Task 6 adds a test that builds each paper invocation's argparse namespace with `--help`-free parsing and asserts every attribute each paper invocation reads exists.
2. **Monkeypatched attributes** (`pl.load_physics_embeddings`, `pl.load_label_table`, `pcp.ALPHA_RIDGE`, `pcp.TORCH_INIT_SEED`, `pcp.AE_HIDDEN`) look unused to static analysis. Expected: they survive trimming. Task 7 adds `test_monkeypatch_targets_exist`.
3. **Running from a different working directory.** After the move, generators and runners should not depend on cwd. Expected: `paper/generate/appendix_gen.py` run from `/tmp` writes `paper/latex/main.tex`. Task 9 adds that check.
4. **Cache resolution.** Expected: `EFFDIM_CACHE_DIR` set means that directory is used; unset means `curvature-experiment/.cache`. Task 9 adds `test_cache_dir_env_override`.
5. **The thin runner has no smoke mode.** Expected: `--help` still parses and its `pcp` imports resolve after trimming. The harness runs `--help` at every gate (`run_smoke.sh` last line).

---

## File structure (end state)

```
paper/                  manuscript + record -> LaTeX/figure scripts
  README.md, records.py
  latex/      main.tex references.bib neurips_2026.sty neurips_2026.tex checklist.tex check.sh COMPLIANCE.md
    figures/  fig1_intervention.{pdf,png} figF_mean_curvature.{pdf,png} fig1_probe_facing.{pdf,png}
              make_fig1.py make_fig_intervention.py
  generate/   table_main_gen.py appendix_gen.py
  tests/      test_generators.py
curvature-experiment/   supporting code that produces the records
  README.md, REPRODUCE.md, requirements.txt, conftest.py
  runners/    09_physics_probe_facing_split_run.py 09_physics_normal_scaling_run.py
              09_physics_normal_scaling_thin_run.py 09_physics_probe_facing_run.py
              09_fixture_probe_facing_run.py 09_fixture_probe_facing_split_run.py
              09_fixture_probe_decodability_run.py 09_instrument_adjudication_run.py
              09_physics_curvature_run.py 09_row_alignment_proof_run.py
  pu_manifold/ __init__ cache subsample physics_curvature_probe physics_labels cae geometry_probes
              chart_curvature decoder_curvature curvature_probe crossmodal_curvature
              cross_split_curvature density_stratified_null linear_probe [mknn if still used]
  tests/      tests for the above
  notebooks/  02.6_swiss_roll_plainae_curvature_check.ipynb 09.1_swiss_roll_probe_decodability_check.ipynb
              09.2_swiss_roll_density_decoupling_check.ipynb
archive/      README.md + everything else, original relative paths; pu_manifold_trimmed/ for removed functions
docs/superpowers/harness/  gate.sh run_smoke.sh run_generators.sh compare.py compare_generators.py unused_symbols.py
```

---

### Task 1: Snapshot the submitted paper (Stage 0)

**Files:**
- Modify (copy over from main checkout): `docs/latex/ml4ps/{main.tex,references.bib,COMPLIANCE.md,appendix_gen.py,preview_times_metric.pdf}`, `docs/latex/ml4ps/figures/{make_fig1.py,fig1_probe_facing.pdf,fig1_probe_facing.png}`, `HANDOFF-v1.1.md`, `.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/novelty-review-2026-09-15/README.md`
- Create (copy): `docs/latex/ml4ps/figures/{make_fig_intervention.py,fig1_intervention.pdf,fig1_intervention.png,figF_mean_curvature.pdf,figF_mean_curvature.png}`, `docs/latex/ml4ps/preview_cm_fallback.pdf`

**Interfaces:**
- Produces: a commit whose `docs/latex/ml4ps/main.tex` is the submitted paper; later tasks compare against it.

- [ ] **Step 1: Confirm the source still holds the expected uncommitted state**

```bash
cd /home/akagi/Documents/Projects/EffDim && git rev-parse HEAD && git status -s
```
Expected: HEAD `78dced8...`; 16 paths: 8 ` M` and 6 `??` under `docs/latex/ml4ps/`, plus ` M HANDOFF-v1.1.md` and ` M` on the novelty-review README. If it differs, stop and ask.

- [ ] **Step 2: Copy the files**

```bash
SRC=/home/akagi/Documents/Projects/EffDim; WT=/home/akagi/Documents/Projects/EffDim/.claude/worktrees/paper-closure
cd $SRC
for f in $(git status -s --untracked-files=all | awk '{print $2}'); do mkdir -p "$WT/$(dirname $f)"; cp -p "$f" "$WT/$f"; done
cd $WT && git status -s
```
Expected: the same 16 paths, now as changes in `$WT`.

- [ ] **Step 3: Verify byte equality**

```bash
cd /home/akagi/Documents/Projects/EffDim && for f in $(git status -s --untracked-files=all | awk '{print $2}'); do cmp "$f" "$WT/$f" || echo "MISMATCH $f"; done; echo checked
```
Expected: only `checked`.

- [ ] **Step 4: Commit**

```bash
cd $WT && git add -A docs/latex/ml4ps HANDOFF-v1.1.md .planning/phases/09-curvature-conditioned-label-decodability-physics-replication/novelty-review-2026-09-15/README.md
git commit -m "docs(ml4ps): snapshot the submitted paper from the fixture-validity-audit working tree

Copies the uncommitted and untracked paper files from the main checkout
(fixture-validity-audit @ 78dced8, working tree as of 2026-09-23): main.tex,
references.bib, COMPLIANCE.md, appendix_gen.py, make_fig1.py,
make_fig_intervention.py and the figure and preview PDFs/PNGs, plus the
HANDOFF-v1.1.md and novelty-review README edits. Byte-identical copies.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Capture the baseline (Stage 1)

**Files:**
- Create (not committed): `$CLOSURE_WORK/colleague/` (colleague checkout), `$CLOSURE_WORK/baseline/{smoke,gen,pytest.log}`
- Harness already committed in `docs/superpowers/harness/` (commit it in Step 6 if not yet committed).

**Interfaces:**
- Produces: `$CLOSURE_WORK/baseline/smoke/*.jsonl|*.npz`, `$CLOSURE_WORK/baseline/gen/{main.tex,table_main.txt,*.png}`; `gate.sh <tree> <label>` compares against them.

- [ ] **Step 1: Extract the colleague checkout** (only the baseline needs it; the adjudication runner requires `--colleague-root` until Task 4)

```bash
export CLOSURE_WORK=$HOME/.cache/effdim-closure; mkdir -p $CLOSURE_WORK/colleague
git -C /home/akagi/Documents/Projects/EffDim archive 97efb2eb6cd7dec7f2c568f53c534752ff3c32c8 | tar -x -C $CLOSURE_WORK/colleague
ls $CLOSURE_WORK/colleague/experiments | head -3
```
Expected: a non-empty listing.

- [ ] **Step 2: Baseline smoke runs** (~4.5 min)

```bash
cd $WT && $H/run_smoke.sh $WT/notebooks/diagnostics $CLOSURE_WORK/baseline/smoke $CLOSURE_WORK/colleague && cat $CLOSURE_WORK/baseline/smoke/summary.txt
```
Expected: 10 lines, every one containing ` rc=0 `.

- [ ] **Step 3: Baseline generators** (~1 min)

```bash
$H/run_generators.sh $WT $CLOSURE_WORK/baseline/gen && ls $CLOSURE_WORK/baseline/gen
```
Expected: `main.tex table_main.txt fig1_intervention.png fig1_probe_facing.png figF_mean_curvature.png` plus logs. Then:
```bash
diff $WT/docs/latex/ml4ps/main.tex $CLOSURE_WORK/baseline/gen/main.tex | wc -l
```
Expected: `116` (the known Table 1 clobber, fixed in Task 10). Record the number in the task report.

- [ ] **Step 4: Baseline tests** (~100 s)

```bash
cd $WT/notebooks && $PY -m pytest -q -p no:cacheprovider pu_manifold/tests/test_pu_manifold.py pu_manifold/tests/test_cae.py pu_manifold/tests/test_geometry_probes.py pu_manifold/tests/test_decoder_curvature.py pu_manifold/tests/test_curvature_probe.py pu_manifold/tests/test_crossmodal_curvature.py pu_manifold/tests/test_cross_split_curvature.py pu_manifold/tests/test_density_stratified_null.py pu_manifold/tests/test_linear_probe.py pu_manifold/tests/test_physics_curvature_probe.py pu_manifold/tests/test_physics_labels.py pu_manifold/tests/test_instrument_adjudication_run.py pu_manifold/tests/test_colleague_estimator_run.py 2>&1 | tee $CLOSURE_WORK/baseline/pytest.log | tail -1
```
Expected: `566 passed, 8 skipped` (dry-run on 2026-09-23).

- [ ] **Step 5: Self-check that the gate passes on the unchanged tree**

```bash
$H/gate.sh $WT stage1-selfcheck
```
Expected: last line `GATE PASS (stage1-selfcheck)`. The pytest line runs *all* current tests; failures in tests outside the kept list are acceptable here only if they also fail on the main checkout. Record any.

- [ ] **Step 6: Commit the harness** (skip if `git log -- docs/superpowers/harness` already shows it)

```bash
cd $WT && git add docs/superpowers/harness docs/superpowers/plans && git commit -m "chore(harness): golden-output gate for the paper closure

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Archive whole files outside the closure (Stage 2)

**Files:** `git mv` only. Every target keeps its relative path under `archive/`.

**Interfaces:**
- Consumes: baseline from Task 2.
- Produces: `notebooks/` containing only the closure (plus chain members removed in Tasks 4-5).

- [ ] **Step 1: Write the move script** to `$CLOSURE_WORK/stage2_moves.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/akagi/Documents/Projects/EffDim/.claude/worktrees/paper-closure
mvto() { mkdir -p "archive/$(dirname "$1")"; git mv "$1" "archive/$1"; }
# notebooks: keep 02.6 plain-AE curvature, 09.1, 09.2, requirements
for f in $(git ls-files 'notebooks/*.ipynb'); do
  case "$f" in notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb|notebooks/09.1_swiss_roll_probe_decodability_check.ipynb|notebooks/09.2_swiss_roll_density_decoupling_check.ipynb) ;; *) mvto "$f";; esac
done
# runners: keep the closure plus chain members that Tasks 4-5 remove
KEEP_RUN="09_physics_probe_facing_split_run.py 09_physics_normal_scaling_run.py 09_physics_normal_scaling_thin_run.py 09_physics_probe_facing_run.py 09_fixture_probe_facing_run.py 09_fixture_probe_facing_split_run.py 09_fixture_probe_decodability_run.py 09_instrument_adjudication_run.py 09_row_alignment_proof_run.py 09_physics_curvature_run.py 09_colleague_estimator_run.py synthetic_control_run.py curvature_field_pu_run.py derivative_bridge_run.py"
for f in $(git ls-files notebooks/diagnostics); do
  b=${f#notebooks/diagnostics/}
  case "$b" in colleague_shims/*) continue;; esac
  [[ " $KEEP_RUN " == *" $b "* ]] || mvto "$f"
done
# modules: keep closure plus chain members (mknn, synthetic_controls, derivative_bridge, persistence_probe, topoae)
KEEP_MOD="__init__.py cache.py subsample.py physics_curvature_probe.py physics_labels.py cae.py geometry_probes.py chart_curvature.py decoder_curvature.py curvature_probe.py crossmodal_curvature.py cross_split_curvature.py density_stratified_null.py linear_probe.py mknn.py synthetic_controls.py derivative_bridge.py persistence_probe.py topoae.py"
for f in $(git ls-files 'notebooks/pu_manifold/*.py' | grep -v /tests/); do
  [[ " $KEEP_MOD " == *" ${f#notebooks/pu_manifold/} "* ]] || mvto "$f"
done
KEEP_TEST="test_pu_manifold.py test_cae.py test_geometry_probes.py test_decoder_curvature.py test_curvature_probe.py test_crossmodal_curvature.py test_cross_split_curvature.py test_density_stratified_null.py test_linear_probe.py test_physics_curvature_probe.py test_physics_labels.py test_instrument_adjudication_run.py test_colleague_estimator_run.py test_physics_import_purity.py test_synthetic_controls.py test_derivative_bridge.py test_topoae.py test_persistence_probe.py"
for f in $(git ls-files notebooks/pu_manifold/tests); do
  [[ " $KEEP_TEST " == *" ${f#notebooks/pu_manifold/tests/} "* ]] || mvto "$f"
done
# non-notebook areas
for f in .planning sweep docs/latex/main.tex docs/latex/main.pdf docs/latex/references.bib HANDOFF-v1.1.md TODO.md PYPI_SETUP.md; do mvto "$f"; done
```

- [ ] **Step 2: Run it and inspect**

```bash
bash $CLOSURE_WORK/stage2_moves.sh && cd $WT && git status -s | grep -c '^R' && ls notebooks notebooks/diagnostics notebooks/pu_manifold notebooks/pu_manifold/tests
```
Expected: `notebooks/` holds 3 notebooks + `requirements-notebooks.txt`; `diagnostics/` holds the 14 runners + `colleague_shims/`; `pu_manifold/` holds 19 modules + `tests/` with 18 tests.

- [ ] **Step 3: Trim `test_physics_import_purity.py`** — it imports `pointcloud_probe` and `cka` (now archived). Open it, remove only the test functions/parametrize entries that name `pointcloud_probe` or `cka`, and append the removed code verbatim to `archive/notebooks/pu_manifold/tests/test_physics_import_purity_removed.py` with header:

```python
# Removed from notebooks/pu_manifold/tests/test_physics_import_purity.py in the paper-closure
# Stage 2: these cases import pointcloud_probe / cka, which are archived. Verbatim.
```
Run: `cd $WT/notebooks && $PY -m pytest -q -p no:cacheprovider pu_manifold/tests/test_physics_import_purity.py`
Expected: PASS.

- [ ] **Step 4: Gate**

```bash
$H/gate.sh $WT stage2
```
Expected: `GATE PASS (stage2)`. Any failure: stop, report the gate output.

- [ ] **Step 5: Commit**

```bash
cd $WT && git add -A && git commit -m "refactor(closure): archive notebooks, runners, modules and records outside the paper closure

Stage 2 of docs/superpowers/specs/2026-09-23-paper-closure-design.md. git mv only
(plus two cases dropped from test_physics_import_purity, archived verbatim).
Gate: smoke records, generators and tests identical to baseline.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Cut the colleague path (Stage 3)

**Files:**
- Modify: `notebooks/diagnostics/09_instrument_adjudication_run.py`, `09_physics_probe_facing_run.py`, `09_fixture_probe_decodability_run.py`, and any other kept runner where `grep -n colleague` hits (`09_physics_probe_facing_split_run.py`, `09_physics_normal_scaling_run.py`, `09_fixture_probe_facing_run.py`, `09_fixture_probe_facing_split_run.py`)
- Modify: `notebooks/pu_manifold/tests/test_instrument_adjudication_run.py`
- Move to archive: `notebooks/diagnostics/09_colleague_estimator_run.py`, `notebooks/diagnostics/colleague_shims/`, `notebooks/pu_manifold/tests/test_colleague_estimator_run.py`
- Create: `archive/notebooks/diagnostics/colleague_removed/<runner>.py` (removed code per runner, verbatim)

**Interfaces:**
- Produces: kept runners no longer accept `--colleague-root` / `--skip-colleague`; module-level name `runner` in `09_instrument_adjudication_run.py` still refers to the loaded `09_physics_curvature_run` module (downstream runners use `adj.runner`).

- [ ] **Step 1: Change the test first.** In `test_instrument_adjudication_run.py`, remove the skip condition on `EFFDIM_COLLEAGUE_ROOT` and any `--colleague-root` in the invocation, so the smoke test always runs. Run it:

```bash
cd $WT/notebooks && $PY -m pytest -q -p no:cacheprovider pu_manifold/tests/test_instrument_adjudication_run.py
```
Expected: FAIL (argparse: `--colleague-root` is required).

- [ ] **Step 2: Re-point the adjudication loader.** Replace lines 69-81 of `09_instrument_adjudication_run.py` (the block that loads `09_colleague_estimator_run.py` and sets `colleague`/`runner`) with:

```python
DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_RUNNER_PATH = DIAGNOSTICS_ROOT / "09_physics_curvature_run.py"

# Load the production runner FIRST, before numpy/torch are imported anywhere in this process:
# its module-level code applies the `--threads` cap (OMP/MKL/NUMEXPR env vars, then
# `torch.set_num_threads`) from `sys.argv`, and puts `notebooks/` and `notebooks/diagnostics/`
# on `sys.path` for the imports below.
_spec = importlib.util.spec_from_file_location("physics_curvature_run", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
```
(The module name `"physics_curvature_run"` is the one the colleague runner used; keep it.)

- [ ] **Step 3: Remove colleague code in each runner.** Rules, applied per file:
  - Delete `--colleague-root` and `--skip-colleague` arguments, and every branch guarded by them.
  - Delete functions whose only purpose is the colleague estimator (`colleague_BS_at_anchors`, `colleague_probe_facing`, `colleague_field`, and any function that calls `colleague.*`).
  - Delete record keys and rows produced only from colleague output (for example `COL_COLUMNS = ("K_H_cross_col", "pf_curv_col", "K_w_dir_col")` and the `colleague_K_H_cross` partial rows, `colleague_head`, `colleague_commit_expected`, `colleague_wallclock_s`).
  - Keep everything that computes `ours_*` / `decoder_*` / exact-truth quantities, in the same order.
  - Append every deleted block verbatim to `archive/notebooks/diagnostics/colleague_removed/<runner file name>`, in file order, each preceded by `# --- removed from <file>:<first line>-<last line> ---`.
  - Downstream references: `adj.colleague` / `fx.adj.colleague` uses in the fixture and physics runners go with the colleague code.

After editing, this must print nothing:
```bash
cd $WT/notebooks/diagnostics && grep -n -i "colleague" 09_*.py | grep -v "^09_physics_curvature_run.py"
```
Leftover mentions inside docstrings/comments that describe removed behaviour: delete the sentence; if a docstring becomes wrong, fix it to describe the remaining behaviour in one line.

- [ ] **Step 4: Archive the colleague files**

```bash
cd $WT && mkdir -p archive/notebooks/diagnostics archive/notebooks/pu_manifold/tests
git mv notebooks/diagnostics/09_colleague_estimator_run.py archive/notebooks/diagnostics/
git mv notebooks/diagnostics/colleague_shims archive/notebooks/diagnostics/
git mv notebooks/pu_manifold/tests/test_colleague_estimator_run.py archive/notebooks/pu_manifold/tests/
```

- [ ] **Step 5: Test passes**

```bash
cd $WT/notebooks && $PY -m pytest -q -p no:cacheprovider pu_manifold/tests/test_instrument_adjudication_run.py
```
Expected: PASS (previously skipped without a colleague checkout; now runs).

- [ ] **Step 6: Gate**

```bash
$H/gate.sh $WT stage3
```
Expected: `GATE PASS (stage3)`. The comparer already drops keys matching `colleague|(^|_)col($|_)` and rows whose `row/instrument/estimator/method/arm/column` value names the colleague. If the gate fails **only** on colleague material the filter misses, you may widen `COLLEAGUE_KEY` or the row fields in `$H/compare.py`, then prove the widening drops nothing else:
```bash
$PY - <<'EOF'
import json, sys; sys.path.insert(0, "docs/superpowers/harness"); import compare as c
from pathlib import Path
import os; B = Path(os.environ["CLOSURE_WORK"]) / "baseline/smoke"
drop = set()
for f in B.glob("*.jsonl"):
    for l in f.read_text().splitlines():
        r = json.loads(l)
        def w(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    (drop.add(k) if (c.VOLATILE.search(k) or c.COLLEAGUE_KEY.search(k)) else w(v))
            elif isinstance(o, list): [w(x) for x in o]
        w(r)
print(sorted(drop))
EOF
```
Expected: only volatile keys (`timestamp`, `wallclock_*`, `python`, `torch`, `numpy`, `repo_head`, `*_root`, `parquet_path`) and colleague keys. Include the printed list in the commit message. Any non-colleague numeric drift: stop and report.

- [ ] **Step 7: Commit**

```bash
cd $WT && git add -A && git commit -m "refactor(closure): remove the colleague estimator path

Stage 3. 09_instrument_adjudication_run now loads 09_physics_curvature_run directly
(same module name, same load-before-numpy ordering). --colleague-root/--skip-colleague,
colleague columns and rows removed from the kept runners; removed code archived under
archive/notebooks/diagnostics/colleague_removed/. The adjudication smoke test now runs
without a colleague checkout. Gate: non-colleague records identical to baseline.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Cut the `_fidelity_axes` import chain (Stage 4)

**Files:**
- Modify: `notebooks/diagnostics/09_instrument_adjudication_run.py`
- Move to archive: `notebooks/diagnostics/{synthetic_control_run,curvature_field_pu_run,derivative_bridge_run}.py`, `notebooks/pu_manifold/{synthetic_controls,derivative_bridge,persistence_probe,topoae}.py`, `notebooks/pu_manifold/tests/{test_synthetic_controls,test_derivative_bridge,test_topoae,test_persistence_probe}.py`

**Interfaces:**
- Produces: `_fidelity_axes(H_est: np.ndarray, H_true: np.ndarray) -> Dict[str, Any]` defined in `09_instrument_adjudication_run.py`, byte-identical body to `synthetic_control_run.py:289`.

- [ ] **Step 1: Find everything `_fidelity_axes` needs**

```bash
cd $WT/notebooks/diagnostics && $PY - <<'EOF'
import ast
src = open("synthetic_control_run.py").read(); t = ast.parse(src)
defs = {n.name: n for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
consts = {tg.id: n for n in t.body if isinstance(n, ast.Assign) for tg in n.targets if isinstance(tg, ast.Name)}
need, todo = set(), ["_fidelity_axes"]
while todo:
    k = todo.pop(); need.add(k); node = defs.get(k) or consts.get(k)
    for x in ast.walk(node):
        if isinstance(x, ast.Name) and (x.id in defs or x.id in consts) and x.id not in need: todo.append(x.id)
for k in sorted(need, key=lambda k: (defs.get(k) or consts.get(k)).lineno):
    n = defs.get(k) or consts.get(k); print(k, n.lineno, n.end_lineno)
EOF
grep -n "^import\|^from" synthetic_control_run.py
```
Record the list (expected at least `_fidelity_axes` and `MIN_TRUE_NORM`). Also note which modules those definitions use (`chart_curvature`, `curvature_probe`, `np`, …).

- [ ] **Step 2: Copy them verbatim** into `09_instrument_adjudication_run.py`, directly after its imports, under the comment:

```python
# --- scorer copied verbatim from synthetic_control_run.py (archived); definitions unchanged ---
```
Add any import the copied code needs that the file lacks (for example `from pu_manifold import chart_curvature`). Then replace every `scr._fidelity_axes` with `_fidelity_axes` (and `scr.<X>` for any other copied name), and delete the line `import synthetic_control_run as scr  # noqa: E402  -- sealed scorer, unmodified`.

- [ ] **Step 3: Nothing else uses the chain**

```bash
cd $WT/notebooks && grep -rn -E "synthetic_control_run|curvature_field_pu_run|derivative_bridge_run|synthetic_controls|derivative_bridge|persistence_probe|topoae" --include=*.py diagnostics pu_manifold | grep -v -E "^(diagnostics/(synthetic_control_run|curvature_field_pu_run|derivative_bridge_run)|pu_manifold/(synthetic_controls|derivative_bridge|persistence_probe|topoae)|pu_manifold/tests/test_(synthetic_controls|derivative_bridge|topoae|persistence_probe))\.py"
```
Expected: no import lines (docstring mentions are fine; list them in the report). If a kept file imports one, stop and report.

- [ ] **Step 4: Archive the chain**

```bash
cd $WT && for f in notebooks/diagnostics/synthetic_control_run.py notebooks/diagnostics/curvature_field_pu_run.py notebooks/diagnostics/derivative_bridge_run.py notebooks/pu_manifold/synthetic_controls.py notebooks/pu_manifold/derivative_bridge.py notebooks/pu_manifold/persistence_probe.py notebooks/pu_manifold/topoae.py notebooks/pu_manifold/tests/test_synthetic_controls.py notebooks/pu_manifold/tests/test_derivative_bridge.py notebooks/pu_manifold/tests/test_topoae.py notebooks/pu_manifold/tests/test_persistence_probe.py; do mkdir -p archive/$(dirname $f); git mv $f archive/$f; done
```

- [ ] **Step 5: Gate**

```bash
$H/gate.sh $WT stage4
```
Expected: `GATE PASS (stage4)`.

- [ ] **Step 6: Commit**

```bash
cd $WT && git add -A && git commit -m "refactor(closure): inline _fidelity_axes and archive the synthetic-control chain

Stage 4. 09_instrument_adjudication_run imported synthetic_control_run for one scorer,
which pulled in curvature_field_pu_run, derivative_bridge_run, synthetic_controls,
derivative_bridge, persistence_probe and topoae. The scorer and its helpers are copied
verbatim; the chain and its tests are archived. Gate: identical to baseline.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Trim `09_physics_curvature_run.py` to what the kept runners use (Stage 5a)

**Files:**
- Modify: `notebooks/diagnostics/09_physics_curvature_run.py` (1843 lines; ~1200 expected unused)
- Modify/Move: `notebooks/pu_manifold/tests/test_physics_curvature_probe.py` (its runner tests at ~line 573 load this file)
- Create: `archive/notebooks/diagnostics/trimmed/09_physics_curvature_run.py`, `notebooks/pu_manifold/tests/test_paper_invocations.py`

**Interfaces:**
- Consumes: kept runners reference `runner._THREADS`, `runner.fit_and_field_at_anchors`, `runner.SphereProjectedDecoder`, `runner._oof_predictions_for_label`.
- Produces: the same four names, unchanged.

- [ ] **Step 1: Write the invocation test** `notebooks/pu_manifold/tests/test_paper_invocations.py` (Review Focus 1). It parses each paper invocation (from `REPRODUCE.md`'s source, the Phase-09 supplements) with the runner's own parser, and checks the runner module exposes every `runner.*` name the kept runners use:

```python
"""Paper invocations still parse, and the production-runner names the kept runners use exist.

Guards trimming: full physics mode is not exercised by smoke, so this pins the CLI surface and
the cross-runner attributes that physics mode reads.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

RUNNERS = Path(__file__).resolve().parents[2] / "diagnostics"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), RUNNERS / name)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


PAPER_ARGS = {
    "09_physics_probe_facing_split_run.py": [
        ["--mode", "physics", "--d-values", "16,20", "--label-table", "x.parquet"],
        ["--mode", "physics", "--fit-seed", "1"],
        ["--mode", "physics", "--fit-seed", "0", "--hidden", "400,400,400"],
        ["--mode", "physics", "--hessian-xfit"],
        ["--mode", "physics", "--alpha", "1"],
        ["--mode", "physics", "--fit-seed", "0", "--hessian-xfit", "--parquet-path", "p.parquet",
         "--embedding-column", "clip_base_galaxies", "--geometry-out", "g.npz"],
    ],
    "09_physics_normal_scaling_run.py": [
        ["--mode", "physics", "--geometry-npz", "g.npz", "--arrays-out", "a.npz"],
    ],
    "09_physics_normal_scaling_thin_run.py": [
        ["--parquet-path", "p.parquet", "--embedding-column", "c", "--arrays-npz", "a.npz", "--out", "o.npz"],
    ],
    "09_fixture_probe_facing_run.py": [["--mode", "full"]],
    "09_fixture_probe_facing_split_run.py": [["--mode", "full"]],
    "09_fixture_probe_decodability_run.py": [["--mode", "full", "--gammas", "-1,0,1"], ["--mode", "full", "--gammas", "0.4,0.6,0.8"]],
    "09_instrument_adjudication_run.py": [["--mode", "sphere-fixture", "--noise", "0"]],
}


@pytest.mark.parametrize("runner_file,argv", [(r, a) for r, v in PAPER_ARGS.items() for a in v])
def test_paper_invocation_parses(runner_file, argv):
    mod = _load(runner_file)
    parser = mod.build_parser() if hasattr(mod, "build_parser") else None
    if parser is None:
        pytest.skip(f"{runner_file} builds its parser inside main(); covered by the gate's smoke run")
    parser.parse_args(argv)


def test_production_runner_names_used_by_kept_runners_exist():
    adj = _load("09_instrument_adjudication_run.py")
    for name in ("_THREADS", "fit_and_field_at_anchors", "SphereProjectedDecoder", "_oof_predictions_for_label"):
        assert hasattr(adj.runner, name), name
```
Before relying on the parse test, check how each runner builds its parser: `grep -n "def build_parser\|def _parser\|def parse_args\|ArgumentParser(" $WT/notebooks/diagnostics/09_*.py`. If a runner builds it inside `main()`, extract it into `def build_parser() -> argparse.ArgumentParser:` in that runner, with no other change (a pure extract-function), so the test can parse without running. List every runner you extracted from in the report.

Run: `cd $WT/notebooks && $PY -m pytest -q -p no:cacheprovider pu_manifold/tests/test_paper_invocations.py`
Expected: PASS on the untrimmed code (this is a regression guard; it must pass before and after).

- [ ] **Step 2: List unused symbols**

```bash
cd $WT/notebooks && D=diagnostics P=pu_manifold
ENTRY="09_physics_probe_facing_split_run 09_physics_normal_scaling_run 09_physics_normal_scaling_thin_run 09_physics_probe_facing_run 09_fixture_probe_facing_run 09_fixture_probe_facing_split_run 09_fixture_probe_decodability_run 09_instrument_adjudication_run 09_row_alignment_proof_run"
MODS=$(ls $P/*.py | xargs -n1 basename | sed 's/\.py$//')
$PY $H/unused_symbols.py --ignore-name main \
  $(for r in $ENTRY; do echo --entry $D/$r.py; done) \
  --generator ../docs/latex/ml4ps/table_main_gen.py --generator ../docs/latex/ml4ps/appendix_gen.py \
  --generator ../docs/latex/ml4ps/figures/make_fig1.py --generator ../docs/latex/ml4ps/figures/make_fig_intervention.py \
  $(for n in 02.6_swiss_roll_plainae_curvature_check 09.1_swiss_roll_probe_decodability_check 09.2_swiss_roll_density_decoupling_check; do echo --notebook $n.ipynb; done) \
  --library $D/09_physics_curvature_run.py $(for m in $MODS; do echo --library $P/$m.py; done) > $CLOSURE_WORK/unused_stage5.txt
sed -n '/09_physics_curvature_run/,/^==/p' $CLOSURE_WORK/unused_stage5.txt
```
Expected: a list of ~20+ unused top-level symbols in `09_physics_curvature_run.py` (its `main`, CLI modes, verdict/bundle/preregistration machinery).

- [ ] **Step 3: Move them.** For each listed symbol in `09_physics_curvature_run.py`, cut the definition (including its decorators and the comment block immediately above it that documents only it) and append it verbatim to `archive/notebooks/diagnostics/trimmed/09_physics_curvature_run.py`, in original order, each preceded by `# --- removed from 09_physics_curvature_run.py:<start>-<end> ---`. Header of that archive file:

```python
# Top-level definitions removed from notebooks/diagnostics/09_physics_curvature_run.py in the
# paper-closure Stage 5: not reachable from any kept runner, generator or notebook
# (docs/superpowers/harness/unused_symbols.py). Verbatim, original order. Not importable.
```
Also remove the `if __name__ == "__main__":` block (this file is no longer run directly; append it to the archive file too) and imports that become unused (check each with `grep -c`). Keep the module-level thread-cap and `sys.path` code exactly.

- [ ] **Step 4: Tests that targeted removed code move too.** In `test_physics_curvature_probe.py`, find tests that load `09_physics_curvature_run.py` (`_RUNNER_PATH`, ~line 573) and call a removed symbol. Move those test functions (and fixtures only they use) verbatim to `archive/notebooks/pu_manifold/tests/trimmed/test_physics_curvature_probe.py`. Re-run the fixed-point: `$PY $H/unused_symbols.py ...` (Step 2 command) must list nothing more for `09_physics_curvature_run.py`.

- [ ] **Step 5: Tests pass**

```bash
cd $WT/notebooks && $PY -m pytest -q -p no:cacheprovider pu_manifold/tests
```
Expected: all pass.

- [ ] **Step 6: Gate**

```bash
$H/gate.sh $WT stage5a
```
Expected: `GATE PASS (stage5a)`.

- [ ] **Step 7: Commit**

```bash
cd $WT && git add -A && git commit -m "refactor(closure): trim 09_physics_curvature_run to the names the paper runners use

Stage 5a. Unreachable top-level definitions (CLI modes, verdict and preregistration
machinery, __main__) archived verbatim under archive/notebooks/diagnostics/trimmed/.
Adds test_paper_invocations.py pinning the paper CLI surface and the production-runner
names. Gate: identical to baseline.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Trim unreachable functions in `pu_manifold` (Stage 5b)

**Files:**
- Modify: each kept `notebooks/pu_manifold/*.py` with unused symbols (dry-run 2026-09-23: `cae` ~420 lines incl. `ChartAutoEncoder`, `curvature_probe` ~1150, `linear_probe` ~390, `density_stratified_null` ~360, `crossmodal_curvature` ~350, `chart_curvature` ~180, `geometry_probes` ~160, `cross_split_curvature` ~140, `mknn` ~80, `physics_curvature_probe` ~40)
- Modify: matching tests in `notebooks/pu_manifold/tests/`
- Create: `archive/pu_manifold_trimmed/<module>.py`, `archive/pu_manifold_trimmed/tests/<test_module>.py`, test `notebooks/pu_manifold/tests/test_monkeypatch_targets.py`

**Interfaces:**
- Produces: every kept module keeps its surviving public names with unchanged signatures.

- [ ] **Step 1: Write the monkeypatch guard** `notebooks/pu_manifold/tests/test_monkeypatch_targets.py` (Review Focus 2):

```python
"""Attributes the paper runners overwrite at runtime must survive trimming."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402
from pu_manifold import physics_labels as pl  # noqa: E402


def test_monkeypatch_targets_exist():
    for mod, name in ((pl, "load_physics_embeddings"), (pl, "load_label_table"),
                      (pcp, "ALPHA_RIDGE"), (pcp, "TORCH_INIT_SEED"), (pcp, "AE_HIDDEN")):
        assert hasattr(mod, name), f"{mod.__name__}.{name}"
```
Also grep the runners for any other patched attribute and add it: `grep -n -E "^\s*(pcp|pl|cae|decoder_curvature|chart_curvature)\.[A-Za-z_]+ *=" $WT/notebooks/diagnostics/*.py`.
Run: `$PY -m pytest -q -p no:cacheprovider pu_manifold/tests/test_monkeypatch_targets.py` — Expected: PASS.

- [ ] **Step 2: Re-run the finder** (same command as Task 6 Step 2) and save to `$CLOSURE_WORK/unused_stage5b.txt`. Work module by module in the order listed under Files. For each module:
  1. Cut each listed symbol (with decorators and its own leading comment) and append verbatim to `archive/pu_manifold_trimmed/<module>.py`, headed:
     ```python
     # Top-level definitions removed from notebooks/pu_manifold/<module>.py in the paper-closure
     # Stage 5: not reachable from any kept runner, generator or notebook. Verbatim, original order.
     ```
     each block preceded by `# --- removed from <module>.py:<start>-<end> ---`.
  2. Remove imports that became unused in that module.
  3. In the module's test file, move tests that call only removed symbols to `archive/pu_manifold_trimmed/tests/<test file>` verbatim. A test that calls both kept and removed symbols: move it (it tests removed behaviour) and, if the kept symbol then has no test at all, say so in the report.
  4. Run that module's tests: `$PY -m pytest -q -p no:cacheprovider pu_manifold/tests/test_<module>.py` — Expected: PASS.
  5. `mknn`: if the finder lists every symbol in it and `grep -n "mknn" pu_manifold/*.py diagnostics/*.py` shows only a lazy import in `crossmodal_curvature` inside a removed function, archive the whole file with `git mv` instead.

- [ ] **Step 3: Fixed point.** Re-run the finder; repeat Step 2 until it reports `TOTAL unused lines: 0` for every `pu_manifold` file.

- [ ] **Step 4: Swiss roll notebooks still import cleanly** (full execution is in Task 12):

```bash
cd $WT/notebooks && for nb in 02.6_swiss_roll_plainae_curvature_check 09.1_swiss_roll_probe_decodability_check 09.2_swiss_roll_density_decoupling_check; do $PY - "$nb.ipynb" <<'EOF'
import json, re, sys
src = "\n".join("".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"] if c["cell_type"] == "code")
import importlib
mods = {"cae", "chart_curvature", "curvature_probe", "decoder_curvature", "physics_curvature_probe"}
for alias, mod in (("pcp", "physics_curvature_probe"),) + tuple((m, m) for m in mods):
    m = importlib.import_module(f"pu_manifold.{mod}")
    for name in set(re.findall(rf"\b{alias}\.([A-Za-z_]\w*)", src)) - {"__file__"}:
        assert hasattr(m, name), f"{sys.argv[1]}: {mod}.{name} missing"
print("ok", sys.argv[1])
EOF
done
```
Expected: three `ok` lines.

- [ ] **Step 5: Gate**

```bash
$H/gate.sh $WT stage5b
```
Expected: `GATE PASS (stage5b)`.

- [ ] **Step 6: Commit** (one commit for the stage; the message lists per-module lines removed from `$CLOSURE_WORK/unused_stage5b.txt`)

```bash
cd $WT && git add -A && git commit -m "refactor(closure): trim pu_manifold to functions reachable from the paper

Stage 5b. Removed definitions archived verbatim under archive/pu_manifold_trimmed/
with their tests. Adds test_monkeypatch_targets.py. Gate: identical to baseline.
<per-module line counts here>

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Remove dead parameters and branches (Stage 6)

**Files:**
- Modify: `docs/latex/ml4ps/appendix_gen.py` (the `INCLUDE_RELATIVE_II = False` block), and any kept file found in Step 1.
- Create: `archive/closure_dead_branches.md` (verbatim removed blocks with file:line).

- [ ] **Step 1: Find candidates**

```bash
cd $WT && grep -n -E "^[A-Z_]+ *= *(False|True|None)\b" notebooks/diagnostics/*.py notebooks/pu_manifold/*.py docs/latex/ml4ps/*.py docs/latex/ml4ps/figures/*.py
grep -n -E "add_argument\(\"--adjudication-mode|choices=\[\"smoke\", \"swiss-roll\"" notebooks/diagnostics/09_instrument_adjudication_run.py
```
A branch is dead only when the flag is a module constant that nothing reassigns (`grep -n "<NAME> *=" -r notebooks docs/latex`) and no kept invocation can change it. Also dead: the adjudication runner's `--mode swiss-roll` path (no paper invocation uses it; smoke and sphere-fixture are used) — remove the choice, its dispatch branch and functions only it calls (re-run the finder after to catch them).

- [ ] **Step 2: Remove each dead branch**; append each removed block verbatim to `archive/closure_dead_branches.md` under a heading `## <file>:<start>-<end> (<reason>)` in a fenced code block.

- [ ] **Step 3: Gate**

```bash
$H/gate.sh $WT stage6
```
Expected: `GATE PASS (stage6)` (generator output identical: the removed `INCLUDE_RELATIVE_II` block never ran).

- [ ] **Step 4: Commit**

```bash
cd $WT && git add -A && git commit -m "refactor(closure): drop dead flags and the unused adjudication swiss-roll mode

Stage 6. Removed blocks archived verbatim in archive/closure_dead_branches.md.
Gate: identical to baseline.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Move to the `paper/` + `curvature-experiment/` layout (Stage 7)

**Files:**
- Move: see Step 2 map.
- Modify: `curvature-experiment/pu_manifold/cache.py:23`, runner `DEFAULT_RECORD_PATH` lines (8 files, e.g. `09_fixture_probe_facing_run.py:70`), test path lines (`parents[2]`, `parents[3]`, `"diagnostics"`), generator paths (`appendix_gen.py:3` and its `p = "docs/latex/ml4ps/main.tex"`, `table_main_gen.py:3`, `make_fig1.py:5,66,88`, `make_fig_intervention.py:9,15,41`), notebook import cell (3 notebooks), `.gitignore`.
- Create: `curvature-experiment/conftest.py`, `curvature-experiment/tests/test_cache_resolution.py`, `paper/records.py`, `curvature-experiment/tests/test_no_paper_dependency.py`.

**Interfaces:**
- Produces: `cache.CACHE_DIR` = `Path(os.environ["EFFDIM_CACHE_DIR"]).resolve()` if set and non-empty, else `curvature-experiment/.cache`.
- Produces: `paper/records.py` exposing `RECORDS: pathlib.Path` (same resolution: `EFFDIM_CACHE_DIR`, else `<repo>/curvature-experiment/.cache`) and `PAPER: pathlib.Path` (the `paper/` directory). This is the only place `paper/` knows where records live; generators and figure scripts import it. Nothing under `curvature-experiment/` references `paper/`.

- [ ] **Step 1: Write the layout tests** (not run until Step 5)

`notebooks/pu_manifold/tests/test_cache_resolution.py` (Review Focus 4; moves with the other tests in Step 2):
```python
"""CACHE_DIR honours EFFDIM_CACHE_DIR and otherwise sits next to pu_manifold."""
import importlib
import sys
from pathlib import Path

PKG_PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_PARENT))


def _reload_cache():
    import pu_manifold.cache as c
    return importlib.reload(c)


def test_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFDIM_CACHE_DIR", str(tmp_path / "records"))
    assert _reload_cache().CACHE_DIR == (tmp_path / "records").resolve()
    monkeypatch.delenv("EFFDIM_CACHE_DIR")
    _reload_cache()


def test_cache_dir_default(monkeypatch):
    monkeypatch.delenv("EFFDIM_CACHE_DIR", raising=False)
    assert _reload_cache().CACHE_DIR == (PKG_PARENT / ".cache").resolve()
```

`notebooks/pu_manifold/tests/test_no_paper_dependency.py` (the one-way dependency rule):
```python
"""The experiment code never reaches into paper/: the manuscript depends on the records, not the reverse."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_code_does_not_reference_paper_dir():
    hits = [f"{p.relative_to(ROOT)}:{i}" for p in ROOT.rglob("*.py") if ".cache" not in p.parts
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if "paper/" in line or '"paper"' in line or "'paper'" in line]
    assert hits == [], hits
```
(`parents[1]` is correct once the files live at `curvature-experiment/tests/`.)

- [ ] **Step 2: Move files**

```bash
cd $WT && E=curvature-experiment
mkdir -p paper/latex/figures paper/generate paper/tests $E/runners $E/pu_manifold $E/tests $E/notebooks
for f in main.tex references.bib neurips_2026.sty neurips_2026.tex checklist.tex check.sh COMPLIANCE.md preview_cm_fallback.pdf preview_times_metric.pdf; do git mv docs/latex/ml4ps/$f paper/latex/$f; done
git mv docs/latex/ml4ps/figures/* paper/latex/figures/
git mv docs/latex/ml4ps/table_main_gen.py docs/latex/ml4ps/appendix_gen.py paper/generate/
git ls-files docs/latex/ml4ps   # expect empty; if not, list leftovers in the report and move them to paper/latex/
for f in $(git ls-files notebooks/diagnostics); do git mv $f $E/runners/$(basename $f); done
for f in $(git ls-files 'notebooks/pu_manifold/*.py' | grep -v /tests/); do git mv $f $E/pu_manifold/; done
git mv notebooks/pu_manifold/tests/* $E/tests/
git mv notebooks/*.ipynb $E/notebooks/
git mv notebooks/requirements-notebooks.txt $E/requirements.txt
git ls-files notebooks   # expect empty
```

- [ ] **Step 3: Rewrite paths — experiment side**
  - `curvature-experiment/pu_manifold/cache.py:23`:
    ```python
    CACHE_DIR = Path(os.environ.get("EFFDIM_CACHE_DIR") or Path(__file__).resolve().parents[1] / ".cache").resolve()
    ```
    (add `import os` if missing). `parents[1]` is `curvature-experiment/`.
  - Each runner's `DEFAULT_RECORD_PATH = NOTEBOOK_ROOT / ".cache" / "<name>.jsonl"` becomes
    `DEFAULT_RECORD_PATH = Path(os.environ.get("EFFDIM_CACHE_DIR") or NOTEBOOK_ROOT / ".cache") / "<name>.jsonl"` (add `import os` where missing). `NOTEBOOK_ROOT` (= `Path(__file__).resolve().parents[1]`) is now `curvature-experiment/`; no other runner path changes are needed because `runners/` and `pu_manifold/` are still siblings.
  - Tests: in every `curvature-experiment/tests/*.py` **except** the two written in Step 1, apply in this order: `parents[2]` → `parents[1]`, then `parents[3]` → `parents[2]` (reverse order would double-shift), then `/ "diagnostics"` → `/ "runners"`. Check with `grep -n "parents\[\|diagnostics" curvature-experiment/tests/*.py`.
  - `curvature-experiment/conftest.py`:
    ```python
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ```
  - Notebooks: in each of the 3 notebooks, in the first code cell containing `import sys`, add after it one line: `sys.path.insert(0, str(__import__("pathlib").Path.cwd().parent))`. Edit with a small `json` script, not by hand, so outputs are untouched; `git diff --stat` must show a 1-line source change per notebook.
  - `.gitignore`: add `curvature-experiment/.cache/`.

- [ ] **Step 4: Rewrite paths — paper side**
  - Create `paper/records.py`:
    ```python
    """Where the paper's generators find the experiment records. The only link from paper/ to the code."""
    import os
    from pathlib import Path

    PAPER = Path(__file__).resolve().parent
    RECORDS = Path(os.environ.get("EFFDIM_CACHE_DIR") or PAPER.parent / "curvature-experiment" / ".cache").resolve()
    ```
  - In `paper/generate/{appendix_gen,table_main_gen}.py` and `paper/latex/figures/{make_fig1,make_fig_intervention}.py`: replace `C = "notebooks/.cache/"` (and `table_main_gen.py`'s default record path) with
    ```python
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # paper/generate/*.py; use parents[2] in paper/latex/figures/
    from records import PAPER, RECORDS  # noqa: E402
    C = str(RECORDS) + "/"
    ```
    Replace output paths: `"docs/latex/ml4ps/main.tex"` → `str(PAPER / "latex" / "main.tex")`; `f"docs/latex/ml4ps/figures/<name>.{ext}"` → `str(Path(__file__).resolve().parent / f"<name>.{ext}")`; `table_main_gen.py`'s default record → `C + "09_physics_probe_facing_split.jsonl"`.

- [ ] **Step 5: Link the records, run tests**

```bash
ln -sfn /home/akagi/Documents/Projects/EffDim/notebooks/.cache $WT/curvature-experiment/.cache && git -C $WT status -s curvature-experiment/.cache
cd $WT/curvature-experiment && $PY -m pytest -q -p no:cacheprovider tests
```
Expected: `status` prints nothing (ignored); all tests pass, including `test_cache_resolution.py` and `test_no_paper_dependency.py`.

- [ ] **Step 6: cwd independence** (Review Focus 3)

```bash
SB=$(mktemp -d); rsync -a --exclude .git --exclude curvature-experiment/.cache $WT/ $SB/; ln -s /home/akagi/Documents/Projects/EffDim/notebooks/.cache $SB/curvature-experiment/.cache
cd /tmp && $PY $SB/paper/generate/appendix_gen.py && $PY $SB/paper/generate/table_main_gen.py > /dev/null && diff $WT/paper/latex/main.tex $SB/paper/latex/main.tex | wc -l; rm -rf $SB
```
Expected: `116` (the known pre-Task-10 Table 1 clobber) and nothing written under `/tmp` except the sandbox.

- [ ] **Step 7: Update the harness for the new layout.** In `$H/gate.sh`, replace the layout-detection `if` with:
```bash
if [ -d "$T/curvature-experiment/runners" ]; then R=$T/curvature-experiment/runners; TD=$T/curvature-experiment; TP=tests
else R=$T/notebooks/diagnostics; TD=$T/notebooks; TP=pu_manifold/tests; fi
```
and after the pytest block add:
```bash
if [ -d "$T/paper/tests" ]; then ( cd "$T/paper" && "$PY" -m pytest -q -p no:cacheprovider tests ) > "$OUT/pytest_paper.log" 2>&1 || fail=1; tail -1 "$OUT/pytest_paper.log"; fi
```
In `$H/run_generators.sh`, replace the link and layout lines with:
```bash
mkdir -p "$SB/notebooks" "$SB/curvature-experiment"; ln -sfn "$REC" "$SB/notebooks/.cache"; ln -sfn "$REC" "$SB/curvature-experiment/.cache"
```
and `--exclude paper/.cache` with `--exclude curvature-experiment/.cache`. (Its `if [ -d paper/generate ]` branch already matches the new paper side.)

- [ ] **Step 8: Gate**

```bash
$H/gate.sh $WT stage7
```
Expected: `GATE PASS (stage7)`.

- [ ] **Step 9: Commit**

```bash
cd $WT && git add -A && git commit -m "refactor(closure): split the manuscript (paper/) from its supporting code (curvature-experiment/)

Stage 7. paper/{latex,generate,tests,records.py}; curvature-experiment/{runners,pu_manifold,
tests,notebooks,requirements.txt}. Records resolve from EFFDIM_CACHE_DIR, else
curvature-experiment/.cache; paper/records.py is the only link from the manuscript to the
code, and test_no_paper_dependency.py keeps the dependency one-way. Generators no longer
depend on the working directory. Gate: identical to baseline.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Make `appendix_gen.py` reproduce `main.tex` exactly (Stage 8)

**Files:**
- Modify: `paper/generate/appendix_gen.py`, `paper/generate/table_main_gen.py`
- Create: `paper/tests/test_generators.py`

**Interfaces:**
- Produces: `table_main_gen.main_rows(records: list[dict]) -> list[str]` (the LaTeX rows it prints today, one per label, plus the `% checks:` lines separately via `check_lines(records) -> list[str]`); `table_main_gen.py` run as a script prints exactly what it prints today.

- [ ] **Step 1: Write the failing test** `paper/tests/test_generators.py`:

```python
"""appendix_gen.py regenerates main.tex byte-for-byte from the records (needs the record cache)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("EFFDIM_CACHE_DIR") or PAPER.parent / "curvature-experiment" / ".cache")
needs_records = pytest.mark.skipif(not (CACHE / "09_physics_probe_facing_split.jsonl").exists(),
                                   reason="record cache not available")


@needs_records
def test_appendix_gen_reproduces_main_tex(tmp_path):
    sandbox = tmp_path / "paper"
    shutil.copytree(PAPER / "generate", sandbox / "generate")
    shutil.copytree(PAPER / "latex", sandbox / "latex")
    shutil.copy(PAPER / "records.py", sandbox / "records.py")
    env = dict(os.environ, EFFDIM_CACHE_DIR=str(CACHE))
    subprocess.run([sys.executable, str(sandbox / "generate" / "appendix_gen.py")], check=True, env=env, cwd=tmp_path)
    assert (sandbox / "latex" / "main.tex").read_bytes() == (PAPER / "latex" / "main.tex").read_bytes()
```
Run: `cd $WT/paper && $PY -m pytest -q -p no:cacheprovider tests/test_generators.py`
Expected: FAIL (bytes differ: generated appendix lacks `tab:real` and the hand-edited Appendix C prose).

- [ ] **Step 2: See the exact difference**

```bash
SB=$(mktemp -d); cp -r $WT/paper/generate $WT/paper/latex $WT/paper/records.py $SB/; EFFDIM_CACHE_DIR=$WT/curvature-experiment/.cache $PY $SB/generate/appendix_gen.py; diff $SB/latex/main.tex $WT/paper/latex/main.tex > $CLOSURE_WORK/appendix_diff.txt; wc -l $CLOSURE_WORK/appendix_diff.txt; rm -rf $SB
```
Expected: 116 lines. Every difference lies between `% BEGIN APPENDIX AUTOGEN` and `% END APPENDIX AUTOGEN`.

- [ ] **Step 3: Refactor `table_main_gen.py`** into `main_rows(records)` and `check_lines(records)` plus an `if __name__ == "__main__":` that prints exactly what it prints today. Verify: `$PY paper/generate/table_main_gen.py | cmp - $CLOSURE_WORK/baseline/gen/table_main.txt` → no output.

- [ ] **Step 4: Emit the hand-edited Appendix C from the generator.** In `appendix_gen.py`, at the point where the Appendix C output starts (the `# --- C` / cross-encoder section), emit, in the order they appear in `main.tex`: the hand-written text blocks as raw-string literals copied from the snapshot `main.tex` (section heading, "Main results." paragraph, the `tab:real` table wrapper and caption, "Across encoders" prose, and edited captions), and the `tab:real` body rows from `table_main_gen.main_rows(rows(C + "09_physics_probe_facing_split.jsonl"))` (import it with `sys.path.insert(0, str(Path(__file__).resolve().parent))`). Numbers must come from records; only prose and table scaffolding are literals. Where the hand-edited captions replaced generated captions, the literal replaces the generated string. Iterate with Step 2's command until the diff is empty.

- [ ] **Step 5: Test passes**

```bash
cd $WT/paper && $PY -m pytest -q -p no:cacheprovider tests/test_generators.py
```
Expected: PASS.

- [ ] **Step 6: Re-baseline the generator output, once, with proof.** The generator output legitimately changes in this stage (it now equals `main.tex`). Replace the baseline main.tex only after showing it equals the committed paper:

```bash
$H/run_generators.sh $WT $CLOSURE_WORK/runs/stage8-gen && cmp $CLOSURE_WORK/runs/stage8-gen/main.tex $WT/paper/latex/main.tex && cp $CLOSURE_WORK/runs/stage8-gen/main.tex $CLOSURE_WORK/baseline/gen/main.tex && echo rebaselined
```
Expected: `rebaselined`. Then `$H/gate.sh $WT stage8` → `GATE PASS (stage8)`.

- [ ] **Step 7: Commit**

```bash
cd $WT && git add -A && git commit -m "fix(ml4ps): appendix_gen regenerates main.tex byte-for-byte

Stage 8. The generator now emits Table 1 (tab:real, rows from table_main_gen.main_rows)
and the hand-edited Appendix C prose and captions, so re-running it no longer deletes
Table 1. test_generators.py pins the round trip.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Documentation (Stage 9)

**Files:**
- Create: `paper/README.md`, `curvature-experiment/README.md`, `curvature-experiment/REPRODUCE.md`, `archive/README.md`
- Modify: `README.md` (two lines), `CLAUDE.md` (paths), `curvature-experiment/requirements.txt` (trim), `paper/latex/COMPLIANCE.md` (path mentions only)

- [ ] **Step 1: `paper/README.md`** — sections, in order:
  1. One paragraph: what the paper claims; this folder is the manuscript plus the scripts that turn records into its tables and figures; the code that produces the records is in `../curvature-experiment/`.
  2. **Map table** with columns `Paper element | Generator (paper/) | Records | Runner (curvature-experiment/runners/) | Modules (curvature-experiment/pu_manifold/)`. Rows (from the spec's closure trace): Table 1 & Table E; main-text checks; tab:ablate (App. A); tab:sens (App. B); tab:xenc/tab:xencx (App. C); tab:cf and intervention numbers (App. D); sign tests (App. D); Figure 1; figF panels a/b; validation numbers 0.999/1.00/0.94; App. E known-surface numbers; "86,471". Names must be ones that exist after Task 7 (check with `ls curvature-experiment/runners curvature-experiment/pu_manifold`).
  3. **Regenerating**: `python paper/generate/appendix_gen.py`, `python paper/latex/figures/make_fig_intervention.py`, `python paper/latex/figures/make_fig1.py`, `bash paper/latex/check.sh`; records location via `EFFDIM_CACHE_DIR` or the `curvature-experiment/.cache` symlink; `pytest paper/tests`.

- [ ] **Step 2: `curvature-experiment/README.md`** — sections:
  1. One paragraph: the instrument (plain auto-encoder, sphere-projected decoder, II by automatic differentiation, K = ⟨w_N, II⟩ vs Hess_M y) and what the code computes, no results.
  2. **Layout**: `runners/` (one line per runner: what it computes and which record it writes), `pu_manifold/` (one line per module), `tests/`, `notebooks/` (Swiss roll checks, per CLAUDE.md).
  3. **Checks**: `pytest curvature-experiment/tests`; smoke runs (`python curvature-experiment/runners/<runner> --mode smoke`); the equivalence gate `docs/superpowers/harness/gate.sh`.
  4. Pointer to `REPRODUCE.md`.

- [ ] **Step 3: `curvature-experiment/REPRODUCE.md`** — sections:
  1. **Environment**: `pip install -r curvature-experiment/requirements.txt`; records location (`EFFDIM_CACHE_DIR` or `curvature-experiment/.cache` symlink to the 8.7G record store).
  2. **Inputs**: HF `UniverseTBD/pu-embeddings` snapshot (full hash: copy from `paper/latex/COMPLIANCE.md`, or from `ls ~/.cache/huggingface/hub/datasets--UniverseTBD--pu-embeddings/snapshots/`; it begins `bc081f8a`; note `physics_labels.py` does not pin the revision), `Smith42/galaxies@v2.0` labels, the label parquet `labels_Smith42_galaxies_v2.0_test.parquet` (sha256 and pod path copied from `paper/latex/COMPLIANCE.md`; no script in the repo builds it), ViT-B geometry `09_probe_facing_geometry_d{16,20}.npz` (sha256 from COMPLIANCE.md; pod only), frozen subsample `subsample_20260729_a79b3460b838fd0a.npz`.
  3. **Invocations**, one subsection per record file listed in the spec's closure, each with the exact command line copied from the archived supplement that recorded it (`archive/.planning/phases/09-curvature-conditioned-label-decodability-physics-replication/09-SUPPLEMENT-{02,03,05,06,07,09,11,12}-*.md`) and a `Source:` line naming that file and section. Rewrite paths to the new layout (`notebooks/diagnostics/` → `curvature-experiment/runners/`) and drop colleague flags; say so once at the top of the section.
  4. **Record hashes**: copy the sha256 table from `paper/latex/COMPLIANCE.md`.
  5. **Caveats**: (a) records from 2026-09-12 to 09-15 carry `repo_head 71914dd` although the flags they used were added in later commits (79589d3, c54a2eb, 139950f, 9ec69d7…28d3ee6) — the field does not pin the code version; (b) Appendix E "mean-bending trace −0.24 to +0.27 across five encoders" is not produced by this repo: source `origin/curvature-experiments@dabe5e2:outputs/geometry/curvature_program_synthesis/CURVATURE_PROGRAM_SUMMARY.md` §6; (c) two mismatch definitions: `hess_mismatch_emp` (main tables) and `hess_mismatch_dec` (cross-fitted columns, `paper/generate/appendix_gen.py`); (d) the colleague estimator comparison was removed from this code; its code is under `archive/`; (e) smoke mode runs the same code paths at small n and is what the equivalence gate checks; full physics mode needs the pod.

- [ ] **Step 4: `archive/README.md`** — what is archived and why (one paragraph), then a table `Archived path | Original path | Reason` with one row per top-level group: `archive/notebooks/` (runners, modules, tests, notebooks outside the closure), `archive/notebooks/diagnostics/colleague_removed/`, `archive/notebooks/diagnostics/trimmed/`, `archive/pu_manifold_trimmed/`, `archive/closure_dead_branches.md`, `archive/.planning/`, `archive/sweep/`, `archive/docs/latex/`, `archive/HANDOFF-v1.1.md`, `archive/TODO.md`, `archive/PYPI_SETUP.md`. State that archived code is not importable as-is (paths changed) and that the pre-closure tree is `fixture-validity-audit@78dced8` + the Stage 0 snapshot commit.

- [ ] **Step 5: Root pointers**
  - `README.md`: add after the title:
    ```markdown
    The ML4PS 2026 paper "Linear Probes on Curved Latent Spaces" is in [paper/](paper/README.md);
    the code that produces its results is in [curvature-experiment/](curvature-experiment/README.md).
    ```
  - `CLAUDE.md`: replace `notebooks/pu_manifold/` → `curvature-experiment/pu_manifold/`, `notebooks/.cache/` → `curvature-experiment/.cache/` (records; `EFFDIM_CACHE_DIR` overrides), `notebooks/02.2_swiss_roll_cae_check.ipynb` → `curvature-experiment/notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb`, `notebooks/<phase>_swiss_roll_<model>_check.ipynb` → `curvature-experiment/notebooks/<phase>_swiss_roll_<model>_check.ipynb`, `notebooks/diagnostics/` → `curvature-experiment/runners/`; `cae.PlainAutoEncoder` stays. Leave the remote-compute path unchanged. `grep -n "notebooks/" CLAUDE.md` afterwards: every hit must be a `curvature-experiment/notebooks/` path.
  - `paper/latex/COMPLIANCE.md`: rewrite repo paths only (`notebooks/diagnostics/` → `curvature-experiment/runners/`, `notebooks/.cache/` → `curvature-experiment/.cache/`, `docs/latex/ml4ps/` → `paper/latex/`, generator paths → `paper/generate/`); no other edits.

- [ ] **Step 6: Trim `curvature-experiment/requirements.txt`** to packages imported by `curvature-experiment/` or `paper/` code:

```bash
cd $WT && grep -rhoE "^\s*(import|from) [a-zA-Z_][a-zA-Z0-9_]*" --include=*.py curvature-experiment paper | awk '{print $2}' | sort -u
```
Keep a pinned line for each third-party package in that list, plus `pytest` and `nbconvert`/`jupyter` for the tests and notebooks. Remove lines for packages nothing imports (for example `persim`, `ripser` if only archived modules used them). Verify: `$PY -c "import torch, numpy, scipy, sklearn, pandas, pyarrow, datasets, matplotlib"`.

- [ ] **Step 7: Commit**

```bash
cd $WT && git add -A && git commit -m "docs: paper and curvature-experiment READMEs, REPRODUCE provenance, archive index

Stage 9.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Final verification

**Files:** none changed unless a check fails (then stop and report).

- [ ] **Step 1: Full gate**

```bash
$H/gate.sh $WT final
```
Expected: `GATE PASS (final)`.

- [ ] **Step 2: Swiss roll notebooks re-execute** (each < 2 min on CPU)

```bash
cd $WT/curvature-experiment/notebooks && for nb in 02.6_swiss_roll_plainae_curvature_check 09.1_swiss_roll_probe_decodability_check 09.2_swiss_roll_density_decoupling_check; do
  $PY -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --output $CLOSURE_WORK/$nb.executed.ipynb $nb.ipynb && \
  $PY - $nb.ipynb $CLOSURE_WORK/$nb.executed.ipynb <<'EOF'
import json, re, sys
def lines(p):
    out = []
    for c in json.load(open(p))["cells"]:
        for o in c.get("outputs", []):
            out += [l for l in "".join(o.get("text", "")).splitlines() if re.search(r"\b(PASS|FAIL)\b", l)]
    return out
a, b = lines(sys.argv[1]), lines(sys.argv[2])
print(sys.argv[1], "committed:", a); print(sys.argv[1], "executed :", b)
sys.exit(0 if a == b and a else 1)
EOF
done
```
Expected: for each notebook the committed and executed PASS/FAIL lines are identical and non-empty. Do not commit the executed copies.

- [ ] **Step 3: Paper builds**

```bash
bash $WT/paper/latex/check.sh; git -C $WT status -s paper/latex
```
Expected: `LaTeX errors: 0`, spill `0`. `check.sh` rewrites `preview_*.pdf`; restore them with `git -C $WT checkout -- paper/latex/preview_times_metric.pdf paper/latex/preview_cm_fallback.pdf` so the tree stays clean.

- [ ] **Step 4: effdim package unaffected**

```bash
cd $WT && $PY -m pytest -q -p no:cacheprovider tests && git diff --stat fixture-validity-audit -- src tests benchmarks pyproject.toml .github mkdocs.yml MANIFEST.in LICENSE docs/index.md docs/api.md docs/theory.md docs/deployment.md docs/tutorials docs/javascripts
```
Expected: pytest passes; the diff is empty.

- [ ] **Step 5: Size report** for the final summary

```bash
cd $WT && echo "kept python lines:" && git ls-files 'paper/*.py' 'curvature-experiment/*.py' | xargs wc -l | tail -1 && echo "archived python lines:" && git ls-files 'archive/*.py' | xargs wc -l | tail -1 && echo "files archived:" && git ls-files archive | wc -l
git log --oneline fixture-validity-audit..HEAD
```
Report: kept Python lines under `paper/` and `curvature-experiment/`, number of files archived, commits.

- [ ] **Step 6: Whole-branch review** — request review (superpowers:requesting-code-review) of `fixture-validity-audit..simplify/paper-closure`, focusing on: any non-move change in Stages 2-7, the colleague removal, the appendix generator, and `REPRODUCE.md` accuracy against the archived supplements.
