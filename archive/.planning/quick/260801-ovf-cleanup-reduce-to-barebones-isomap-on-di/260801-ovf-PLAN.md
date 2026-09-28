---
phase: quick-260801-ovf
plan: 01
type: execute
wave: 1
depends_on: []
autonomous: true
requirements: [D-01, D-02, D-03, D-04, D-05, D-06]
files_modified:
  - notebooks/01_manifold_and_gate.ipynb
  - notebooks/02_k_sensitivity_refit.ipynb
  - notebooks/diagnostics/gate_diagnostics.py
  - notebooks/diagnostics/geometry_handoff.py
  - notebooks/diagnostics/geomstats_eval.py
  - notebooks/diagnostics/hsc_crosscheck.py
  - notebooks/diagnostics/model_sweep.py
  - notebooks/diagnostics/signature_transfer_test.py
  - notebooks/diagnostics/stress_family_eval.py
  - notebooks/diagnostics/stress_family_rescale.py
  - notebooks/diagnostics/geometry_probes_run.py
  - notebooks/diagnostics/seed_crosscheck.py
  - notebooks/pu_manifold/__init__.py
  - notebooks/pu_manifold/cache.py
  - notebooks/pu_manifold/subsample.py
  - notebooks/pu_manifold/geometry_probes.py
  - notebooks/requirements-notebooks.txt
  - .planning/STATE.md
  - .planning/PROJECT.md
  - .planning/ROADMAP.md
  - .planning/REQUIREMENTS.md
  - .planning/WINDOWS.md
  - .planning/phases/01-data-loading-manifold-reconstruction/*.md
  - .planning/phases/02-eigenspectrum-audit-validity-gate/*.md
  - .planning/phases/02.1-geometry-representation-research/*.md

must_haves:
  truths:
    - "notebooks/ holds exactly one .ipynb file, 02_k_sensitivity_refit.ipynb (D-01)"
    - "notebooks/diagnostics/ holds exactly two scripts: seed_crosscheck.py and geometry_probes_run.py (D-02)"
    - "02_k_sensitivity_refit.ipynb reads no notebook off disk except itself, and k=15 runs through _process_k on the same path as k in {5,10,30} (D-01)"
    - "geometry_probes_run.py runs from a clean checkout: its four provenance values are module-level literals, not a cache file read (D-03)"
    - "pu_manifold/curvature.py and pu_manifold/mknn.py still exist and still export their Phase 3/4 stubs (D-04)"
    - "python -m pytest notebooks/pu_manifold/tests/ passes 32 tests, unchanged, at every task boundary"
    - "Every .planning/ document in the D-05 set is shorter and retains every number, threshold, verdict, file path, commit SHA and citation it had before (D-05)"
    - "No surviving file in notebooks/ names a file this plan deleted (D-02, D-06)"
  artifacts:
    - notebooks/02_k_sensitivity_refit.ipynb
    - notebooks/diagnostics/seed_crosscheck.py
    - notebooks/diagnostics/geometry_probes_run.py
    - notebooks/pu_manifold/geometry_probes.py
    - notebooks/pu_manifold/curvature.py
    - notebooks/pu_manifold/mknn.py
    - notebooks/pu_manifold/tests/test_pu_manifold.py
    - notebooks/pu_manifold/tests/test_geometry_probes.py
  key_links:
    - "notebook 02 cell 11 `_spectrum_arrays` -> `mds_eigenspectrum_{fit_key}.npz` -> `geometry_probes_run.py:154-158`. The probe run reads `eigvecs_top` AND `geo_pairs_r2` out of that npz. The current `_spectrum_arrays` emits NEITHER. Routing k=15 through it without adding those two keys silently truncates the incumbent artifact and breaks the probe run at a KeyError. This is the single most likely breakage in the plan."
    - "`FIT_KEY_INCUMBENT = \"43cf438bc944c509\"` must still reconstruct from `config_key(ANALYSIS_CFG_BASE | {\"n_neighbors\": 15})` (cell 6 asserts this). Nothing in this plan may touch ANALYSIS_CFG_BASE's fields."
    - "`_spectrum_arrays`'s cfg dict is the npz sidecar manifest. If it gains dependence on the pair sample it must gain `r2_pair_count`/`r2_pair_seed`, or a stale artifact will be accepted as a hit."
---

<objective>
Reduce this repo to the barebones Isomap-on-DINOv3 experiment: one notebook, two diagnostics
scripts, the `pu_manifold` package, and terse planning docs. Delete the superseded notebook 01
and eight superseded diagnostics scripts, cut every dangling reference they leave behind, and
strip LLM-flavoured verbosity from what survives.

Purpose: the guiding rule is the user's — "keep it simple, stupid". Nice-to-have is not
necessary; necessary stays.

Output: `notebooks/` reduced to 1 notebook + 2 diagnostics + 6 modules + 2 test files;
`.planning/` rewritten terser with every number preserved.

**No tracer task.** This plan is pure subtraction from an already-built, already-proven
artifact set. There is no new layer to wire, so a thin vertical slice would carry no
information the existing test suite does not already carry (`--no-tracer` rationale).

**Decision ID mapping.** CONTEXT.md numbers its locked decisions `D1`..`D6`; this plan cites
them as `D-01`..`D-06` respectively. They are the same decisions.

**Nothing in this plan may be run.** The notebooks need a ~93 GB HF dataset stream, a
multi-GiB gitignored cache, and tens of minutes per Isomap fit. Every `<verify>` below is a
static check — JSON parse, `ast.parse`, `py_compile`, `grep`, and the existing fast unit
tests. Do not execute a notebook. Do not fit an Isomap.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/quick/260801-ovf-cleanup-reduce-to-barebones-isomap-on-di/260801-ovf-CONTEXT.md
@.planning/STATE.md

Read on demand, not upfront:
- `notebooks/02_k_sensitivity_refit.ipynb` (27 cells; Tasks 2 and 4)
- `notebooks/diagnostics/geometry_probes_run.py` (485 lines; Task 3)
- `notebooks/.cache/gate_verdict_43cf438bc944c509.json` (Task 3 source of truth — gitignored
  but present locally)
</context>

<hard_boundaries>
Out of scope, do not open, do not edit: `src/effdim/**`, `tests/**`, `benchmarks/**`,
`docs/**`, `sweep/**`, `mkdocs.yml`, `pyproject.toml`, `README.md`, `PYPI_SETUP.md`,
`MANIFEST.in`, `LICENSE`, `TODO.md`, `.planning/research/**`.

Leave every Rust-rewrite / Rust-extension sentence exactly as written. Known locations:
`.planning/STATE.md:185-186` (Pending Todos) and `.planning/ROADMAP.md:388-389` (Backlog
note). `TODO.md:7` is out of scope entirely.

`notebooks/.cache/` is gitignored. Never `git add` a cache artifact. Never delete the user's
local cache — Task 3 reads from it.
</hard_boundaries>

<notebook_editing_protocol>
`.ipynb` files are JSON. Never hand-edit raw notebook JSON with string replacement, and never
run `Edit` against a `.ipynb` path — a stray escape or a broken `source` array silently
corrupts the file.

Edit notebooks one of two ways only:

1. `NotebookEdit`, addressing a cell by index; or
2. a throwaway Python script under the scratchpad that does
   `json.load` -> mutate `cells[i]["source"]` -> `json.dump(..., indent=1)` and writes the
   file back.

`source` must stay a **list of strings, each line ending in `\n` except the last**. Do not
collapse it to a single string. Preserve each cell's `id`, `metadata`, `cell_type`, and (for
code cells) `outputs` / `execution_count` keys.

After every notebook write, run the structural gate in `<verify>` before moving on.
</notebook_editing_protocol>

<tasks>

<task type="auto">
  <name>Task 1: Delete notebook 01 and the eight superseded diagnostics scripts</name>
  <files>
    notebooks/01_manifold_and_gate.ipynb,
    notebooks/diagnostics/gate_diagnostics.py,
    notebooks/diagnostics/hsc_crosscheck.py,
    notebooks/diagnostics/model_sweep.py,
    notebooks/diagnostics/geomstats_eval.py,
    notebooks/diagnostics/stress_family_eval.py,
    notebooks/diagnostics/stress_family_rescale.py,
    notebooks/diagnostics/signature_transfer_test.py,
    notebooks/diagnostics/geometry_handoff.py
  </files>
  <action>
Per D-01 and D-02, `git rm` all nine listed files. Nothing else in this task.

Keep, untouched in this task: `notebooks/diagnostics/seed_crosscheck.py` (the seed-variation
arm the user explicitly asked to preserve), `notebooks/diagnostics/geometry_probes_run.py`,
`notebooks/02_k_sensitivity_refit.ipynb`, and all of `notebooks/pu_manifold/`.

Per D-04 the Phase 3/4 stubs `pu_manifold/curvature.py` and `pu_manifold/mknn.py` stay as
scaffolding, along with their `__init__.py` exports — do not delete, do not gut.

This deletion is safe on the import graph: it has been verified that no surviving Python
module imports any of the nine. The references they leave behind are prose only, and Tasks 2,
3 and 4 close them. Do not attempt to fix prose here — keep this commit a pure deletion.

Do not touch `notebooks/.cache/`. Several deleted scripts wrote artifacts there; those files
are gitignored and Task 3 still reads one of them.
  </action>
  <verify>
    <automated><![CDATA[
test "$(ls notebooks/*.ipynb | wc -l)" -eq 1 &&
test -f notebooks/02_k_sensitivity_refit.ipynb &&
test "$(ls notebooks/diagnostics/*.py | wc -l)" -eq 2 &&
test -f notebooks/diagnostics/seed_crosscheck.py &&
test -f notebooks/diagnostics/geometry_probes_run.py &&
test -f notebooks/pu_manifold/curvature.py &&
test -f notebooks/pu_manifold/mknn.py &&
python -m pytest notebooks/pu_manifold/tests/ -q 2>&1 | tail -1 | grep -q "32 passed" &&
echo TASK1_OK
    ]]></automated>
  </verify>
  <done>
`notebooks/` contains one notebook. `notebooks/diagnostics/` contains exactly
`seed_crosscheck.py` and `geometry_probes_run.py`. Both `pu_manifold` stubs still exist.
The 32 unit tests still pass. One commit, deletions only.
  </done>
</task>

<task type="auto">
  <name>Task 2: Make notebook 02 standalone — drop both couplings to the deleted notebook</name>
  <files>notebooks/02_k_sensitivity_refit.ipynb</files>
  <read_first>
Read cells 4, 11, 13 and 14 of `notebooks/02_k_sensitivity_refit.ipynb` before editing. Also
read `notebooks/diagnostics/geometry_probes_run.py:154-158` and `:300-305` — they are the
downstream consumer whose contract this task must not break.
  </read_first>
  <action>
Per D-01, remove notebook 02's two runtime couplings to the deleted notebook so it becomes the
single, self-contained entry point. Follow `<notebook_editing_protocol>`.

**Cell 4 (code, §2 pre-registered constants).** Keep every constant assignment and every
pre-registration assertion: the four thresholds, `D_SWEEP_MAX`, `R2_PAIR_COUNT`, the two pair
seeds, `SYMMETRY_RTOL`, `K_REFIT`/`K_INCUMBENT`/`K_ALL`, `N_COMPONENTS`, the two published
incumbent statistics, `FIT_KEY_INCUMBENT`, `REFIT_PREREG`, and the three asserts on `K_REFIT`.
Delete everything from the `# --- Machine-verified threshold identity ...` banner down to the
`print("R2_PAIR_SEED / R2_PAIR_SEED_CHECK expressions match ...")` line: the `import re`, the
four `_nb01*` bindings, the `_declared_here` dict, and both verification loops. Keep the three
closing `print` lines that report `K_REFIT` / `K_INCUMBENT` / `N_COMPONENTS`. Reword the
leading comment so its authority is the committed pre-registration
(`.planning/phases/02-eigenspectrum-audit-validity-gate/02-REFIT-PREREGISTRATION.md` §4.3,
commit `057b084`) rather than a notebook that no longer exists — the thresholds are still not
revisable here, only the machine check against a second file goes away. Confirm `re` is used
nowhere else in the notebook before dropping the import; `_json` IS still used by cell 12, so
leave the cell 2 imports alone.

**Cell 11 (code, §4 shared machinery).** Delete `_INCUMBENT_SPECTRUM_CFG` and
`_refuse_incumbent_recompute` outright. In `_process_k`, delete the
`if k == K_INCUMBENT: ... else:` branch around the spectrum step and call
`_spectrum_arrays(k, fit_key_k)` unconditionally, so k=15 takes the identical path to
k in {5,10,30}.

**Cell 11, `_spectrum_arrays` — the load-bearing part of this task.** Routing k=15 through
`_spectrum_arrays` makes it the writer of `mds_eigenspectrum_43cf438bc944c509.npz`, and
`npz_cache` overwrites on a cfg-manifest mismatch rather than raising. The existing artifact
carries nine arrays; `_spectrum_arrays` currently returns six, and is missing two that
`geometry_probes_run.py` reads directly. Extend it so the npz it writes is a superset of what
downstream consumers need:

- Return `eigvecs_top` — the descending-order counterpart of the `eigvals_top` reversal, i.e.
  reverse the column order of the array `scipy.linalg.eigh(..., subset_by_index=...)` already
  produces, and stop deleting it before the return.
- Return `geo_pairs_r2` — sample it as `dist_matrix[R2_PAIR_ROWS, R2_PAIR_COLS]` while
  `dist_matrix` is still mmap'd and indexable, i.e. before the `np.array(..., copy=True)`
  line, matching what `_codiag_arrays` already does.
- Because the returned arrays now depend on the pair sample, add `r2_pair_count` and
  `r2_pair_seed` to `_spectrum_arrays`'s cfg dict. Without that the sidecar manifest no longer
  describes the artifact it keys.
- `mds_coords` and `geo_pairs_r2_check` are in the old artifact but have no surviving consumer
  — do not reintroduce them.

Keep, in `_spectrum_arrays`: the shape/dtype asserts, the chunk-wise symmetry check against
`SYMMETRY_RTOL`, and the `copy=True` comment explaining why `np.asarray` on a read-only memmap
returns a view. Those are real guards, not ceremony. Reattribute the two comments that credit
"01 §6.1" to the notebook itself.

**Cell 13 (markdown, §5 header).** Rewrite. Its entire premise — the incumbent is not re-fit,
its spectrum is read back from plan 02-01's audited npz, the compute callback raises — is now
false. State what is actually true: k=15 is fit and its spectrum computed by this notebook on
the same path as every other k, its `r`/`m` are regression-checked against the published
`0.052419` / `0.412071`, and `LONG_EDGE_TAU` (the 99th percentile of the k=15 graph's edge
lengths) is established here as the fixed reference for every `LONG_EDGE_FRACTION(k)`.

**Cell 14 (code, §5 baseline).** Keep the two regression asserts against
`R_STAT_INCUMBENT_PUBLISHED` / `M_STAT_INCUMBENT_PUBLISHED` unchanged — with the incumbent now
re-fit rather than read back, they are the strongest correctness guard in the notebook. Rework
the pair-sample block: `npz_cache(..., _refuse_incumbent_recompute)` is gone, so read
`geo_pairs_r2` from the spectrum npz that `_process_k` just produced. The comparison against
`_codiag_arrays`'s `geo_pairs_r2` is still worth keeping — the two arrays reach the same
numbers through two independent code paths over the same fit — so keep the `np.array_equal`
assert and reword its message to describe that cross-path check instead of identity with a
deleted notebook. Keep the whole `LONG_EDGE_TAU` block and the `del`/`gc.collect()` teardown.

Do not touch `ANALYSIS_CFG_BASE` (cell 6) or any of its fields — `FIT_KEY_INCUMBENT` must
still reconstruct to `43cf438bc944c509`. Do not touch cell 12's ordering self-assertion; it
reads this notebook's own JSON, which is legitimate. Do not port notebook 01's elbow /
residual-variance analysis into 02 — D-01 rules that out explicitly.

Leave prose sweeps in cells 0/1/3/5/7/8/9/15/17/19 to Task 4.
  </action>
  <verify>
    <automated><![CDATA[
python - <<'PY'
import json, ast, sys
p = "notebooks/02_k_sensitivity_refit.ipynb"
nb = json.load(open(p))
assert "nbformat" in nb and "cells" in nb, "top-level notebook keys missing"
code = []
for i, c in enumerate(nb["cells"]):
    assert c["cell_type"] in ("code", "markdown"), (i, c["cell_type"])
    assert isinstance(c["source"], list), f"cell {i}: source is not a list of lines"
    s = "".join(c["source"])
    if c["cell_type"] == "code":
        ast.parse(s)              # every code cell must still parse
        code.append(s)
joined = "\n".join(code)
for tok in ("_refuse_incumbent_recompute", "_INCUMBENT_SPECTRUM_CFG", "_nb01"):
    assert tok not in joined, f"{tok} still present in a code cell"
assert '"eigvecs_top"' in joined, "_spectrum_arrays no longer emits eigvecs_top"
assert '"geo_pairs_r2"' in joined, "_spectrum_arrays no longer emits geo_pairs_r2"
assert '"r2_pair_seed"' in joined, "_spectrum_arrays cfg missing r2_pair_seed"
assert "R_STAT_INCUMBENT_PUBLISHED" in joined and "M_STAT_INCUMBENT_PUBLISHED" in joined
assert 'FIT_KEY_INCUMBENT = "43cf438bc944c509"' in joined
print("TASK2_OK", len(nb["cells"]), "cells")
PY
    ]]></automated>
  </verify>
  <done>
Notebook 02 parses as JSON, every code cell parses as Python, and no code cell mentions the
refusal helper, the incumbent-only cfg, or the deleted notebook. `_spectrum_arrays` emits
`eigvecs_top` and `geo_pairs_r2` and keys its manifest on the pair sample. `_process_k` has a
single spectrum path for all four k. The published-statistics regression asserts and the
frozen incumbent fit key are intact.
  </done>
</task>

<task type="auto">
  <name>Task 3: Inline geometry_probes_run.py's provenance literals; reword the dangling comments</name>
  <files>notebooks/diagnostics/geometry_probes_run.py, notebooks/pu_manifold/geometry_probes.py</files>
  <read_first>
`notebooks/diagnostics/geometry_probes_run.py` lines 60-90, 139-150, 335-375, 440-460.
`notebooks/pu_manifold/geometry_probes.py` lines 275-325.
`notebooks/.cache/gate_verdict_43cf438bc944c509.json` — the verbatim source for the prose
string below.
  </read_first>
  <action>
Per D-03, cut `geometry_probes_run.py`'s dependency on the cached verdict file, whose only
producer was the deleted notebook. Replace the four-line block at lines 143-147 (the
`json.loads(Path(...).read_text())` read and the four derived bindings) with four module-level
literal assignments, placed next to the other published-constant block around line 64:

- `D_FROZEN = 5`
- `D_PROVISIONAL = 18`
- `ELBOW_CRITERION` — the prose string, **copied verbatim** from the `elbow_criterion` key of
  `notebooks/.cache/gate_verdict_43cf438bc944c509.json`. That cached file is the only verbatim
  source; the copy quoted in `pu_manifold/geometry_probes.py` is abbreviated with an ellipsis
  and must not be used. Do not paraphrase, do not re-wrap in a way that changes the string
  content, do not drop the trailing `ELBOW_TIE_BREAK='lower'` clause.
- `GATE_SPECTRUM` — a dict with exactly these six keys and values: `dropoff_index` 2,
  `dropoff_ratio` 2.444713943099398, `lambda_max_pos` 3230.8539634646067, `lambda_min_neg`
  -169.35880545251558, `n_negative` 5029, `n_positive` 4971, `noise_floor`
  7.173936918879702e-09.

This is safe because all four are pure provenance: printed at line 363 ("for reference; not
adjudicated here") and copied into the output artifact at lines 369 and 449-452. Nothing
computes from them. Leave lines 363, 369 and 449-452 unchanged — the names they read are the
same names. Net effect: the script becomes runnable from a clean checkout instead of depending
on a gitignored artifact. Add a short source note above the literals citing
`02-FINDINGS.md`; do not add a paragraph.

Then close the three dangling prose references to a script Task 1 deleted (D-02):

- `geometry_probes_run.py:72-74` (`gate_stats` docstring) and `:83-84`
  (`spectrum_from_distmatrix` docstring) each say the body was copied verbatim from a
  now-deleted module and explain why it was copied rather than imported. Both rationales are
  dead. Replace with a one-line statement that this is Phase 2's definition of `r`/`m`
  (respectively, Phase 2's mean-form double-centring), citing `02-FINDINGS.md`. Keep the
  `copy=True` sentence in `spectrum_from_distmatrix` — that one documents a real memmap trap.
- `geometry_probes_run.py:189-198` — the `SystemExit` HALT message names the deleted notebook
  as "the source of truth to mirror". Point it at
  `.planning/phases/02-eigenspectrum-audit-validity-gate/02-PATTERNS.md`'s `_draw_geo_pairs`
  instead, which is the surviving record of that draw, and cut the message to about a third of
  its length. Keep the raise itself — a pair-sample mismatch invalidates every distortion
  number below it.
- `pu_manifold/geometry_probes.py:283-285` — the `m_before` description cites the deleted
  module's `gate_stats`. Keep the formula (`sum(abs(negative)) / sum(abs(all))`), drop the
  attribution to the deleted file.

Do not change any numeric value, any function signature, or any control flow in either file.
Do not touch `pu_manifold/tests/test_geometry_probes.py`.
  </action>
  <verify>
    <automated><![CDATA[
python -m py_compile notebooks/diagnostics/geometry_probes_run.py notebooks/pu_manifold/geometry_probes.py &&
python - <<'PY'
import ast, json, sys
src = open("notebooks/diagnostics/geometry_probes_run.py").read()
tree = ast.parse(src)
top = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        try:
            top[node.targets[0].id] = ast.literal_eval(node.value)
        except ValueError:
            pass
assert top.get("D_FROZEN") == 5, top.get("D_FROZEN")
assert top.get("D_PROVISIONAL") == 18, top.get("D_PROVISIONAL")
ref = json.load(open("notebooks/.cache/gate_verdict_43cf438bc944c509.json"))
assert top.get("ELBOW_CRITERION") == ref["elbow_criterion"], "ELBOW_CRITERION is not verbatim"
assert top.get("GATE_SPECTRUM") == ref["spectrum"], "GATE_SPECTRUM does not match the record"
code_lines = [l for l in src.splitlines() if not l.lstrip().startswith("#")]
assert not any("read_text()" in l and "verdict" in l for l in code_lines), "verdict file read remains"
print("TASK3_OK")
PY
python -m pytest notebooks/pu_manifold/tests/ -q 2>&1 | tail -1 | grep -q "32 passed" &&
echo TASK3_TESTS_OK
    ]]></automated>
  </verify>
  <done>
`geometry_probes_run.py` compiles, defines `D_FROZEN=5`, `D_PROVISIONAL=18`, a byte-identical
`ELBOW_CRITERION`, and a `GATE_SPECTRUM` dict equal to the recorded spectrum, with no file read
for any of them. Neither file names a deleted script. The 32 unit tests still pass.
  </done>
</task>

<task type="auto">
  <name>Task 4: De-verbose the surviving notebook and modules</name>
  <files>
    notebooks/02_k_sensitivity_refit.ipynb,
    notebooks/pu_manifold/__init__.py,
    notebooks/pu_manifold/cache.py,
    notebooks/pu_manifold/subsample.py,
    notebooks/pu_manifold/geometry_probes.py,
    notebooks/diagnostics/seed_crosscheck.py,
    notebooks/diagnostics/geometry_probes_run.py,
    notebooks/requirements-notebooks.txt
  </files>
  <action>
Per D-06, strip over-explanatory commentary from what survives. Cut: multi-paragraph docstrings
that restate the obvious, comments narrating what the next line does, self-congratulatory
framing, restated context, and hedging. Keep: assertions guarding a real correctness property
(row alignment, cache-hit identity, pre-registration ordering, shape/dtype checks), and any
comment documenting a non-obvious trap (the read-only-memmap `copy=True` note, the
`eigen_solver="dense"` determinism note, the `n_components` truncation note).

Follow `<notebook_editing_protocol>` for the notebook. Change no numeric value, no threshold,
no seed, no function signature, no control flow anywhere in this task.

**Notebook 02 markdown.** Cells 0, 1, 3, 5, 7, 9, 13, 15, 17, 19 carry the bulk of the
verbosity. Specifically:
- Cell 0 — keep the pre-registration provenance (the file path and commit `057b084`), the
  binding "what this analysis may not do" list, and the H1/H2 statement. Cut the "Why a
  separate notebook" paragraph entirely: it exists to justify not editing a notebook that no
  longer exists.
- Cell 3 — cut the "machine-verified against that notebook's own source" paragraph; Task 2
  deleted the machinery it describes.
- Cells 15, 17 and 19 (the §6/§7/§8 headers for k=5/10/30) repeat the same
  double-centring paragraph three times verbatim. Reduce each to a one-line header; the
  mechanism is already stated once at §4.
- Cells 1, 5, 7, 9, 13 — trim to what a reader needs, and reattribute anything currently
  credited to a "§N.M" section of the deleted notebook to the committed planning record
  (`02-REFIT-PREREGISTRATION.md`, `02-FINDINGS.md`, `02-PATTERNS.md`) or to this notebook's own
  §4.

**Notebook 02 code cells.** Same treatment for comments and docstrings only. Cell 8's
`_gate_classify` docstring and cell 11's function docstrings are the main targets. Keep cell
8's eight-case classifier boundary self-test — it is a real test of a real function, not
ceremony. Keep the `_rss_line` reporting: memory is a live constraint on a 10,000-point dense
fit.

**Modules.** `pu_manifold/__init__.py` opens with a 25-line docstring; its first paragraph and
its closing "D-01's three-notebook filenames" paragraph both reference the deleted notebook.
Cut it to a few lines naming the four modules and why `curvature`/`mknn` are not imported at
module level (torch must not become an import-time requirement). Do not change the export
list. Give `cache.py`, `subsample.py` and `geometry_probes.py` the same pass — most functions
there carry a full Parameters/Returns block restating a two-argument signature; collapse those
to a one-line summary and keep only the parameter notes that carry non-obvious information.
`seed_crosscheck.py`'s 17-line module docstring can state its purpose in about five lines.
Sweep `geometry_probes_run.py`'s remaining narration too — Task 3 only touched three specific
sites.

**`requirements-notebooks.txt`.** The comment block around lines 20-28 justifies the pins by
reference to the deleted notebook's reproducibility header. Reword to cite the pinned versions
themselves. Change no pin, add no pin, remove no pin.

After this task, no file anywhere under `notebooks/` may name a file Task 1 deleted.
<!-- planner-discipline-allow: 01_manifold_and_gate -->
<!-- planner-discipline-allow: gate_diagnostics -->
  </action>
  <verify>
    <automated><![CDATA[
python - <<'PY'
import json, ast
p = "notebooks/02_k_sensitivity_refit.ipynb"
nb = json.load(open(p))
assert "nbformat" in nb and "cells" in nb
for i, c in enumerate(nb["cells"]):
    assert isinstance(c["source"], list), f"cell {i}: source is not a list of lines"
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print("NB_STRUCTURE_OK", len(nb["cells"]), "cells")
PY
DELETED='01_manifold_and_gate|gate_diagnostics|hsc_crosscheck|model_sweep|geomstats_eval|stress_family_eval|stress_family_rescale|signature_transfer_test|geometry_handoff'
HITS=$(grep -rEIl "$DELETED" notebooks/ --exclude-dir=.cache --exclude-dir=__pycache__ | wc -l)
test "$HITS" -eq 0 || { echo "FAIL: dangling references in:"; grep -rEIl "$DELETED" notebooks/ --exclude-dir=.cache --exclude-dir=__pycache__; exit 1; }
python -m py_compile notebooks/pu_manifold/*.py notebooks/diagnostics/*.py &&
python -m pytest notebooks/pu_manifold/tests/ -q 2>&1 | tail -1 | grep -q "32 passed" &&
echo TASK4_OK
    ]]></automated>
  </verify>
  <done>
Notebook 02 is still valid JSON with list-of-lines `source` and every code cell parsing. No
file under `notebooks/` (excluding the gitignored cache) names any of the nine deleted files.
All modules compile and the 32 unit tests pass. Every pin in
`requirements-notebooks.txt` is byte-identical to before.
  </done>
</task>

<task type="auto">
  <name>Task 5: Compress the planning documents</name>
  <files>.planning/STATE.md, .planning/PROJECT.md, .planning/ROADMAP.md, .planning/REQUIREMENTS.md, .planning/WINDOWS.md, .planning/phases/01-data-loading-manifold-reconstruction/*.md, .planning/phases/02-eigenspectrum-audit-validity-gate/*.md, .planning/phases/02.1-geometry-representation-research/*.md</files>
  <action>
Per D-05, rewrite every planning document terser in place. 42 files, ~16,700 lines total.

**Preserve exactly, in every file:** every number, threshold, statistic, verdict, decision,
file path, cache key, `fit_key`, commit SHA, arXiv ID, and citation. Every table row and every
table cell value. All YAML frontmatter — same keys, same values, same schema. `[CITED]` vs
`[VERIFIED]` labels stay as-is; do not upgrade a claim's status while rewriting it.

**Cut:** LLM-flavoured prose, hedging, self-congratulation, restated context the reader already
has from the file above it, ceremonial framing, and any paragraph that says the same thing a
table two lines down says. Prefer a table row over a paragraph. Prefer one sentence over three.

**Do not delete any file.** This is a rewrite in place, not a purge. Do not renumber, do not
merge files, do not change any filename.

**Leave every Rust-rewrite sentence alone** — `.planning/STATE.md` Pending Todos and
`.planning/ROADMAP.md`'s Backlog note. Copy them through verbatim.

**Out of scope for this task:** `.planning/research/**` (predates this branch),
`.planning/quick/**` (this task's own directory), and everything in `<hard_boundaries>`.

**Before starting, commit `02-PATTERNS.md` as-is.**
`.planning/phases/02-eigenspectrum-audit-validity-gate/02-PATTERNS.md` is currently untracked.
The value-preservation gate below compares each file against `git show HEAD:<file>`, so an
untracked file is silently skipped and gets no protection. Commit it unchanged first
(`docs(260801-ovf): track 02-PATTERNS.md before compression`), then compress it in batch B3
like every other file.

**Work in four batches, committing after each.** The batches are independent — a fresh
executor can resume at any unstarted batch from this list. Within a batch, handle one file at
a time: read it, rewrite it, move on. Do not hold more than one file's full text at a time.

| Batch | Scope | Files | Lines |
|-------|-------|-------|-------|
| B1 | `.planning/` root: `STATE.md`, `PROJECT.md`, `ROADMAP.md`, `REQUIREMENTS.md`, `WINDOWS.md` | 5 | ~1,150 |
| B2 | `.planning/phases/01-data-loading-manifold-reconstruction/` | 13 | ~4,680 |
| B3 | `.planning/phases/02-eigenspectrum-audit-validity-gate/` | 10 | ~4,770 |
| B4 | `.planning/phases/02.1-geometry-representation-research/` | 15 | ~4,660 |

Commit message per batch: `docs(260801-ovf): compress planning docs (batch N/4)`.

Target roughly a 40-60% line reduction where prose dominates (`PLAN.md`, `CONTEXT.md`,
`REVIEW.md`, `DISCUSSION-LOG.md`, `SURVEY.md`, `RECOMMENDATION.md`) and much less where tables
and recorded numbers dominate (`FINDINGS.md`, `SUMMARY.md`, `PREREGISTRATION.md`,
`VALIDATION.md`). A file that is already almost entirely numbers should barely shrink — that
is the correct outcome, not a failure. Never hit a line-count target by dropping a number.

This is the context-dominant task in the plan. If context pressure builds, commit the finished
batches and record the remaining batch letters in the SUMMARY as explicit follow-up. Never
leave a batch half-written across a commit boundary.
  </action>
  <verify>
    <automated><![CDATA[
python - <<'PY'
import glob, re, subprocess, sys
files = (glob.glob(".planning/*.md")
         + glob.glob(".planning/phases/01-*/*.md")
         + glob.glob(".planning/phases/02-*/*.md")
         + glob.glob(".planning/phases/02.1-*/*.md"))
files = [f for f in files if "/quick/" not in f and "/research/" not in f]
NUM = re.compile(r"\d+\.\d{3,}|\b[0-9a-f]{7,40}\b|arXiv:\S+")
missing, shrunk = [], 0
before_total = after_total = 0
for f in sorted(files):
    old = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True)
    if old.returncode != 0:
        print("FAIL -- untracked, so unprotected by this gate:", f); sys.exit(1)
    new = open(f).read()
    before_total += old.stdout.count("\n"); after_total += new.count("\n")
    if new.count("\n") < old.stdout.count("\n"):
        shrunk += 1
    lost = sorted(set(NUM.findall(old.stdout)) - set(NUM.findall(new)))
    if lost:
        missing.append((f, lost[:8]))
if missing:
    print("FAIL -- values dropped during compression:")
    for f, lost in missing:
        print(" ", f, lost)
    sys.exit(1)
print(f"lines {before_total} -> {after_total}; {shrunk}/{len(files)} files shrank")
assert after_total < before_total, "no net compression"
print("TASK5_VALUES_OK")
PY
python - <<'PY'
import glob, sys, re
bad = []
for f in (glob.glob(".planning/*.md") + glob.glob(".planning/phases/0*/*.md")):
    if "/quick/" in f or "/research/" in f:
        continue
    t = open(f).read()
    if t.startswith("---\n"):
        end = t.find("\n---\n", 4)
        if end == -1:
            bad.append(f)
if bad:
    print("FAIL -- unterminated frontmatter:", bad); sys.exit(1)
print("TASK5_FRONTMATTER_OK")
PY
grep -q "Rust" .planning/STATE.md && grep -q "Rust" .planning/ROADMAP.md && echo TASK5_RUST_PRESERVED
    ]]></automated>
  </verify>
  <done>
All four batches committed. Every file that existed still exists. The corpus is measurably
shorter overall. No high-precision number, commit SHA, or arXiv ID present in a file's previous
version is absent from its new version. Every frontmatter block still terminates. The
Rust-rewrite sentences survive in both `STATE.md` and `ROADMAP.md`.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| git history -> working tree | The only asset at risk is the scientific record. No network, no user input, no package install, no new dependency, no executable surface change. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-ovf-01 | Tampering | Task 5 doc compression | high | mitigate | The Task 5 verify diffs every high-precision number, commit SHA and arXiv ID against `git show HEAD:<file>` and fails on any loss. Compression cannot silently rewrite a measured result. |
| T-ovf-02 | Tampering | Task 2 `_spectrum_arrays` -> `mds_eigenspectrum_*.npz` | high | mitigate | `npz_cache` overwrites on a manifest mismatch rather than raising, so a narrowed return dict would silently truncate the incumbent artifact and break `geometry_probes_run.py`. Task 2's verify asserts `eigvecs_top`, `geo_pairs_r2` and the widened cfg are all present. |
| T-ovf-03 | Tampering | `.ipynb` JSON edits | medium | mitigate | `<notebook_editing_protocol>` forbids raw string replacement on notebook JSON; Tasks 2 and 4 both gate on a JSON parse plus `ast.parse` of every code cell plus a list-of-lines `source` check. |
| T-ovf-04 | Denial of service | deleted `notebooks/.cache/` artifacts | low | accept | The cache is gitignored and locally reproducible; the plan reads from it (Task 3) and never writes or deletes it. |
| T-ovf-SC | Tampering | package installs | n/a | n/a | No package-manager install occurs in this plan. No `requirements-notebooks.txt` pin is added, removed or changed. |
</threat_model>

<verification>
After all five tasks, from a clean tree:

```
ls notebooks/*.ipynb                       # exactly 02_k_sensitivity_refit.ipynb
ls notebooks/diagnostics/*.py              # exactly seed_crosscheck.py, geometry_probes_run.py
python -m py_compile notebooks/pu_manifold/*.py notebooks/diagnostics/*.py
python -m pytest notebooks/pu_manifold/tests/ -q      # 32 passed
git status --short                         # no notebooks/.cache/ entry staged
git diff --stat main...HEAD -- src/ tests/ benchmarks/ docs/ sweep/ pyproject.toml TODO.md
                                           # empty: nothing out of scope was touched
```

Do not run a notebook. Do not fit an Isomap. The full experiment is not reproducible in this
session and no verification step depends on it.
</verification>

<success_criteria>
- `notebooks/` holds 1 notebook, 2 diagnostics scripts, 6 modules and 2 test files.
- Notebook 02 is self-contained: no notebook read except its own JSON, k=15 on the same
  `_process_k` path as k in {5,10,30}, `mds_eigenspectrum_*.npz` still carrying `eigvecs_top`
  and `geo_pairs_r2`.
- `geometry_probes_run.py` runs from a clean checkout — no gitignored-artifact read.
- No file under `notebooks/` names any of the nine deleted files.
- 32 unit tests pass at every task boundary.
- `.planning/` is measurably shorter with zero numbers, SHAs or citations lost.
- Nothing in `<hard_boundaries>` was modified.
</success_criteria>

<output>
Create `.planning/quick/260801-ovf-cleanup-reduce-to-barebones-isomap-on-di/260801-ovf-SUMMARY.md`
when done. Record: files deleted, per-file line-count deltas for `.planning/`, and any
Task 5 batch left unfinished.
</output>
