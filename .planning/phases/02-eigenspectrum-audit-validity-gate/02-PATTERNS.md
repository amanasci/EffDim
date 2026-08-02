# Phase 2: Eigenspectrum Audit & Validity Gate - Pattern Map

**Mapped:** 2026-07-31 · **Files analyzed:** 1 (all work = new §6+ cells in one existing notebook)
· **Analogs found:** 6/6, all sections of `notebooks/01_manifold_and_gate.ipynb`

## Scope note

No new source file is created — only two gitignored runtime artifacts (`spectrum_{key}.npz`,
`gate_verdict_{fit_key}.json`) and appended notebook cells. `src/effdim/`, `pyproject.toml`,
`notebooks/pu_manifold/*.py` not modified (D-14 declines gate.py). The analog source is prior
sections of the same notebook — D-01/D-03/D-08/D-11/D-13 all say "mirror the Phase 1 mechanism."

## File Classification

| New/Modified | Role | Closest Analog | Match |
|---|---|---|---|
| §6.0 pre-registration cell | config | §4.0 (cells 60-61) | exact |
| §6.x spectrum computation | transform | §5.2 k* fit cell (79) + PITFALLS.md Pitfall 3 | role-match |
| §6.x spectrum npz cache | cache I/O | §4.2 stage-2 sweep npz cache (67) | exact |
| §6.x Tenenbaum R² pair subsampling | transform | §4.0 `_draw_geo_pairs` (61) | exact |
| §6.x kneedle elbow + d-freeze branch | gate logic | §4.1 `K_CEILING` halt for/else (63) | exact |
| §6.x verdict write | artifact writer | §5.3 `phase1_handoff` build (84) | exact |
| §6.x D-09 equivalence guard | inline test | §1.7 alignment negative control (37-40) | role-match |
| Phase 3 first-cell enforcement | guard | none (future file) | no analog |

## Pattern Assignments

- **§6.0 pre-registration** — copy §4.0's shape: constants declared once (`R_MAX_PASS=0.10`,
  `M_MAX_PASS=0.05`, `R_MAX_MARGINAL=0.25`, `M_MAX_MARGINAL=0.15`, `D_SWEEP_MAX`,
  `R2_PAIR_COUNT`, `R2_PAIR_SEED=SEED+2` per the SEED+1 convention), echoed by a print block;
  cell-ordering marker variable in the consuming cell (§4.3's `STAGE2_SWEEP` idiom);
  compute-and-assert-shape-immediately idiom for the pair draw.
- **Spectrum computation** — copy §5.2's closure shape: print expected memory before compute,
  timed `_compute_spectrum()` closure (mmap load per D-12, drop the Isomap object, in-place
  mean-form centring, `eigvalsh` + `eigh(subset_by_index)`), peak-RSS print, cache-hit-vs-fresh
  branching. The numerical recipe itself has no in-repo analog — source is PITFALLS.md Pitfall 3.
- **Spectrum npz cache** — §4.2's `npz_cache(stem, cfg, closure-returning-dict)` template; cfg
  must carry `fit_key` + the K ceiling (D-11). No parallel caching path.
- **Pair subsampling** — reuse `_draw_geo_pairs(rng, n_rows_total, count)` from §4.0 verbatim
  with a fresh seed (SEED+2) and the separate pre-registered `R2_PAIR_COUNT`; keep the
  self-pair-rejection loop and post-draw shape asserts.
- **Elbow branch / halt** — D-08 is a single `if elbow <= 18: freeze else: assert False,
  "<observed value, cost, exact constant to edit, remediation>"` — prefer §4.2's plain
  assert-with-message (cell 66) over §4.1's for/else scaffolding (two branches, not a ladder).
  Companion markdown states why the branch exists (cell 64's pattern).
- **Verdict write** — §5.3's shape: flat dict from already-bound state, `json_cache`, formatted
  nested-aware dump, required-keys asserts. Contents per D-16 (thresholds verbatim, flags copied
  straight from `PHASE1_HANDOFF["flags"]`, remediation list embedded on FAIL). Cfg keyed on
  fit_key + thresholds + ceiling so the manifest binds verdict to its rules. New read pattern:
  `json.load(open(cache_path(f"phase1_handoff_{fit_key}", "json")))` — no prior cell reads a
  json_cache back.
- **D-09 equivalence guard** — §1.7's deliberate-small-self-test pattern: 50×50 symmetric
  random D², compute B by mean form and by literal `-0.5·J D² J`,
  `np.testing.assert_allclose`; tolerance (~1e-10 default) stated in prose as a discretion item.

## Shared Patterns

- **Cache contract:** `pu_manifold.cache` (`config_key`/`cache_path`/`npz_cache`/`json_cache`)
  — the only sanctioned mechanism, imported as §4.2/§5.3 already do.
- **Pre-registration + cell-index assertion:** §4.0/§4.3 — declare once, print, never
  re-declare.
- **Documented halt with enumerated remediation:** §4.1 for/else + §4.2 plain assert — both
  D-08's halt and SPEC-07's FAIL message take this shape.
- **Self-contained JSON artifact:** §5.3 — thresholds verbatim, flags as booleans, formatted
  dump + required-keys assert on close.
- **Timed-compute-with-cache-hit branch:** §5.2 — populate a seconds dict only on real
  compute; print "already cached" otherwise.

## No Analog Found

| Section | Reason |
|---|---|
| Double-centring + split eigensolve recipe | No prior dense eigendecomposition in repo; spec-to-implement from PITFALLS.md Pitfall 3 (D-09/D-10 give the exact recipe) |
| Kneedle elbow finder | No existing elbow code anywhere; D-05 requires from-scratch deterministic implementation |
| Phase 3 first-cell enforcement | Notebook 03 doesn't exist; forward contract only (D-14/D-16) |

## Metadata

Scope: the notebook (90 cells), `pu_manifold/cache.py` (full), `tests/test_pu_manifold.py`
(convention scan), `phase1_handoff_43cf438bc944c509.json` (field shapes). `src/effdim/` not
searched — CONTEXT.md states it is not called by this phase. Extracted 2026-07-31.
