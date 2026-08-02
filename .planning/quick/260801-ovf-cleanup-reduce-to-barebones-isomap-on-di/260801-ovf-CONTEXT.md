# Quick Task 260801-ovf: Reduce to barebones Isomap-on-DINO experiment — Context

**Gathered:** 2026-08-01
**Status:** Ready for planning

<domain>
## Task Boundary

Clean up this repo: strip LLM-flavoured prose, cut verbosity, delete smoke tests, and reduce
the work to the barebones experiment — Isomap on DINOv3 embeddings. No overengineering.
Guiding rule, stated by the user: **"keep it simple, stupid" — even if something is nice to
have, if it isn't necessary, remove it.** Keep the parts that vary k and seed. Afterwards,
reduce verbosity in the GSD planning documents.

### In scope

Only what the `isomap-curvature` branch added, and within that only:

- `notebooks/*.ipynb`
- `notebooks/pu_manifold/**`
- `notebooks/diagnostics/**`
- `.planning/**` (verbosity reduction)

### Explicitly OUT of scope — do not touch

- `src/effdim/**`, `tests/**`, `benchmarks/**`, `docs/**`, `mkdocs.yml`, `pyproject.toml`,
  `README.md`, `PYPI_SETUP.md`, `MANIFEST.in`, `LICENSE`
- `sweep/**` — added by this branch but the user scoped cleanup to `notebooks/`
- Any GSD artifact text concerning the **Rust rewrite / Rust extension** (e.g. the stale Rust
  reference in `TODO.md` and the STATE.md "Pending Todos" entry). Leave that text alone.

</domain>

<decisions>
## Implementation Decisions

These were settled with the user before planning. They are **locked** — do not revisit,
do not re-litigate, do not widen.

### D1 — Delete `notebooks/01_manifold_and_gate.ipynb`; make notebook 02 standalone

`01_manifold_and_gate.ipynb` (115 cells, 7410 lines) is deleted outright. It is superseded:

| § | Content | Why it goes |
|---|---|---|
| §0 | env header, `%pip install` | boilerplate |
| §1 | 14-cell smoke-config self-test | the "smoke tests" the user flagged |
| §2 | norm histograms | result is one number (raw norms 16.029 ± 0.504, cv 3.1%) |
| §3 | `compute_dim` panel → `n_components=18` | STATE.md records 18 as *below* the measured intrinsic dim; also the experiment's only dependency on `src/effdim/` |
| §4 | connectivity scan + Procrustes plateau → `k*=15` | superseded by notebook 02, which varies the same k against the real gate statistic instead of a stability proxy |
| §5 | freeze k*, Phase 3 handoff JSON | Phase 3 was invalidated by the gate FAIL |
| §6 | eigenspectrum gate; elbow → `D_FROZEN=5` | gate stat is computed per-k in 02; STATE.md explicitly flags `D_FROZEN=5` as SUSPECT, do not inherit |

**The elbow / residual-variance analysis is NOT ported into 02.** The user chose this
deliberately over the port-it variant.

`notebooks/02_k_sensitivity_refit.ipynb` becomes the single entry point and must be made
fully self-contained. Two couplings to 01 must be removed:

1. **Cell 4** reads `01_manifold_and_gate.ipynb` off disk and regex-greps its §6.0 cell to
   machine-verify threshold identity (`_nb01_path`, `_nb01_src`, `_nb01_prereg`, the
   `_declared_here` loop). Delete that whole verification block; keep the literal threshold
   constants.
2. **Cell 11** defines `_refuse_incumbent_recompute()` and `_INCUMBENT_SPECTRUM_CFG` so that
   k=15 must be a cache hit on `mds_eigenspectrum_43cf438bc944c509.npz` (written by 01 §6.2).
   Delete the refusal. k=15 must run through the same `_process_k` path as k ∈ {5,10,30}.

After the change 02 fits all four k itself and the artifact chain is preserved: `_process_k`
writes `isomap_{fit_key}.joblib` and `mds_eigenspectrum_{fit_key}.npz` under the same keys,
and `FIT_KEY_INCUMBENT = "43cf438bc944c509"` still reconstructs from
`config_key(ANALYSIS_CFG_BASE | {"n_neighbors": 15})`.

**Known accepted cost:** re-fitting k=15 is a dense 10,000-point Isomap fit — tens of minutes
of wall clock — and it means the existing gitignored `isomap_43cf438bc944c509.joblib` is
regenerated rather than inherited. The user accepted this.

### D2 — `notebooks/diagnostics/`: keep seed + the Phase 02.1 geometry research

**KEEP:**
- `seed_crosscheck.py` (seed variation — explicitly requested)
- `geometry_probes_run.py`
- `pu_manifold/geometry_probes.py`
- `pu_manifold/tests/test_geometry_probes.py`

**DELETE:**
- `gate_diagnostics.py`
- `hsc_crosscheck.py`
- `model_sweep.py`
- `geomstats_eval.py`
- `stress_family_eval.py`
- `stress_family_rescale.py`
- `signature_transfer_test.py`
- `geometry_handoff.py`

Verified: **no Python code imports any deleted script.** The only references are prose
comments — `pu_manifold/geometry_probes.py:284` and `geometry_probes_run.py:73,83` both cite
`gate_diagnostics.py`. Those comments must be reworded so they do not point at a deleted file.

### D3 — `geometry_probes_run.py` loses its `gate_verdict` file dependency

`geometry_probes_run.py:143` reads `notebooks/.cache/gate_verdict_{FIT_KEY}.json`, whose sole
producer is notebook 01 cell 108. Since 01 is deleted, inline the four consumed fields as
module-level literals and drop the file read:

```python
D_FROZEN = 5
D_PROVISIONAL = 18
ELBOW_CRITERION = "<the prose string, verbatim from the Phase 2 record>"
GATE_SPECTRUM = {
    "dropoff_index": 2, "dropoff_ratio": 2.444713943099398,
    "lambda_max_pos": 3230.8539634646067, "lambda_min_neg": -169.35880545251558,
    "n_negative": 5029, "n_positive": 4971, "noise_floor": 7.173936918879702e-09,
}
```

This is safe because the four fields are **pure provenance** — printed at line 363 ("for
reference; not adjudicated here") and copied into the output artifact at lines 369 and
449–452. `ELBOW_CRITERION` is a prose string, never executed. No computation depends on any
of them. Net effect: the script becomes runnable from a clean checkout instead of depending
on a gitignored artifact.

### D4 — Keep the Phase 3/4 stubs

`pu_manifold/curvature.py` and `pu_manifold/mknn.py` stay as scaffolding for a future
Phase 3 respec, along with their `__init__.py` exports. Do not delete, do not gut.

### D5 — Compress every `.planning/` document

Terse rewrite across the board: `STATE.md`, `PROJECT.md`, `ROADMAP.md`, `REQUIREMENTS.md`,
**and** all phase `PLAN` / `SUMMARY` / `FINDINGS` / `CONTEXT` / `PREREGISTRATION` / `SURVEY` /
`RECOMMENDATION` / `VALIDATION` / `AMENDMENT` / `PATTERNS` / `DISCUSSION-LOG` / `REVIEW` files
under `.planning/phases/01-*`, `.planning/phases/02-*`, `.planning/phases/02.1-*`.

Compression rules:
- Cut LLM-flavoured prose, hedging, self-congratulation, restated context, and ceremonial
  framing. Preserve every **number, threshold, decision, verdict, file path, commit SHA, and
  citation** exactly.
- Do not delete any planning file. This is a rewrite-in-place, not a purge.
- Frontmatter stays valid and unchanged in schema.
- **Leave Rust-rewrite text alone** (see Task Boundary).

### D6 — Prose cleanup in kept code

In the kept notebook and kept modules, strip over-explanatory LLM commentary: multi-paragraph
docstrings that restate the obvious, comments narrating what the next line does, defensive
assertions whose only purpose is ceremony. Keep assertions that guard a real correctness
property (row alignment, cache-hit identity, pre-registration ordering, shape/dtype checks).

### Claude's Discretion

- Exact wording of compressed planning docs.
- Which specific ceremonial assertions in notebook 02 are ceremony vs. real guards.
- How to reword the two dangling `gate_diagnostics.py` comment references.

</decisions>

<specifics>
## Specific Ideas

- Current state: `01_manifold_and_gate.ipynb` 115 cells / 7410 lines;
  `02_k_sensitivity_refit.ipynb` 27 cells / 1731 lines; `notebooks/diagnostics/` 9 scripts
  totalling ~1900 lines; `notebooks/pu_manifold/` 6 modules + 2 test files.
- After this task `notebooks/` should hold: one notebook, four kept diagnostics/probe files,
  and the `pu_manifold` package (cache, subsample, geometry_probes, curvature stub, mknn stub).
- `notebooks/.cache/` is gitignored; do not commit cache artifacts, do not delete the user's
  local cache.

</specifics>

<canonical_refs>
## Canonical References

- `.planning/STATE.md` — records `GATE_VERDICT=FAIL` (R_STAT=0.052419, M_STAT=0.412071), the
  k ∈ {5,10,15,30} FAIL table, and the standing warning that `D_FROZEN=5` is suspect.
- `.planning/phases/02-eigenspectrum-audit-validity-gate/02-FINDINGS.md` — the gate result of
  record.
- `.planning/PROJECT.md` — milestone scope, Key Decisions table.

</canonical_refs>
