# Phase 7: Curvature-Conditioned Crossmodal Alignment - Research

**Researched:** 2026-08-25
**Domain:** Statistical instrumentation over existing PU-manifold code (curvature fields, MKNN,
pre-registration/freeze machinery) — no new modeling technique, no new library.
**Confidence:** HIGH for code-level API facts (all read directly from source in this session);
MEDIUM for the positive-control design and tie-handling recommendation (methodological judgment
applied to measured facts, not itself independently verified against a second source).

## Summary

Phase 7 is an *instrumentation and statistics* phase, not a modeling phase. Every heavy artifact
it needs already exists as working code: `cae.PlainAutoEncoder` / `cae.train_plain_ae` (trains the
decoder), `decoder_curvature.plain_decoder_curvature` (the validated curvature instrument),
`mknn.py` (the crossmodal alignment metric), and two precedent modules — `linear_probe.py`
(Phase 5) and `pointcloud_probe.py` (Phase 6) — that between them are the exact template for the
freeze-then-compute pattern D7-06 requires. The `notebooks/diagnostics/07_*.py` scripts named in
`07-CONTEXT.md` §9 already measured the instrument's validity range, PU's reconstruction curve,
and PU's topology; **those scripts are exploratory/spike scripts, not pre-registered
infrastructure** — they write to `notebooks/diagnostics/*.jsonl` (not `notebooks/.cache/`), carry
no `assert_preregistered()` guard, and their numbers are correctly already quoted verbatim in
`07-CONTEXT.md`. They should be read as informational inputs (what `d` to sweep, what the
instrument's honest rho range is) — **not treated as the frozen D7-01..D7-04 computation itself**,
which must be a new pre-registered module written from scratch, mirroring `pointcloud_probe.py`'s
relationship to `linear_probe.py` (inherit-by-re-declaration, add only what changes).

The one genuine code gap is D7-04: `mknn.mknn_score` computes the per-point array
`(A & B).sum(axis=1) / k` internally but only returns its mean (`.mean()` at the last line). The
per-point array is not exposed anywhere. **Do not edit `mknn.py`.** The fix is a new function in
the new Phase 7 module that reuses `mknn._membership_matrix` (leading-underscore, but a plain
Python function — freely importable) and composes it exactly as `mknn_score` does, returning the
array instead of the mean. `curvature_probe.permutation_null` is already generic over any
`statistic_fn(x, y) -> float` and defaults to `spearmanr(x, y).statistic` under
`permutation_type="pairings"` — it is a direct, zero-edit fit for D7-04's headline
significance test. `cross_split_curvature.partial_spearman(x, y, controls)` already implements
exactly the "partial Spearman on ranks via least-squares residualization" D7-03 needs for the
density partial, and is directly importable.

**Primary recommendation:** Write one new module (e.g. `notebooks/pu_manifold/crossmodal_curvature.py`)
following `pointcloud_probe.py`'s inherit-by-re-declaration pattern against `linear_probe.py`'s
freeze idiom, containing (a) a new `per_point_mknn(z1, z2, k)` built on `mknn._membership_matrix`,
(b) the frozen pre-registration block (constants + `assert_preregistered()` + `VERDICT_RULE`) for
D7-01/02/03/04, and (c) thin wrappers around `curvature_probe.permutation_null` and
`cross_split_curvature.partial_spearman` for significance and the density partial. Reuse
`cae.PlainAutoEncoder`, `cae.train_plain_ae`, and `decoder_curvature.plain_decoder_curvature`
unchanged for the `d ∈ {20, 25, 32}` fits; reuse `region_partition_mknn_run.py`'s `load_pu_pair()`
pattern for loading the row-aligned `hsc`/`legacysurvey` 10,000-row arrays. Run the three `d`
fits **strictly serially** (the cost model in `07-CONTEXT.md` §7 measured a ~10x slowdown from
three concurrent torch jobs on this machine) with `torch.set_num_threads` / `OMP_NUM_THREADS`
capped, something no prior runner in this codebase does and which must be added new.

<user_constraints>
## User Constraints (from CONTEXT.md)

`07-CONTEXT.md` carries no `## Decisions` / `## Claude's Discretion` / `## Deferred Ideas`
section shape — it is a self-contained evidence dossier written directly by the user/developer
in one session, structured as locked decisions D7-01 through D7-07 (§3), plus supporting
measurement sections §1-§9. It is reproduced here in full because the phase task instructions
name it as authoritative and self-contained; nothing below should be treated as reopened or
softened.

### Locked Decisions (07-CONTEXT.md §3, verbatim)

- **D7-01 — the curvature field, from the validated instrument.**
  `cae.PlainAutoEncoder(in_dim=768, latent_dim=d, hidden=(250,250,250), activation="silu")`,
  trained with `cae.train_plain_ae`, curvature via
  `decoder_curvature.plain_decoder_curvature(model, model.encode(x))` — which differentiates
  `model.decode` ALONE, never the encoder-composed round trip. Run the headline correlation at
  `d ∈ {20, 25, 32}` and report all three (§5 explains why a single `d` cannot be defended). Same
  answer at each `d` ⇒ the truncation question is moot for the conclusion, which is stronger than
  picking one.

- **D7-02 — the positive control, and it is not optional.** Plant a curvature–MKNN relationship
  **at PU's realized `‖H‖` dynamic range** and show the test recovers it. Phase 6's existing
  selfcheck does NOT serve: it planted `rng.random(n)`, a ~20x-spread field, against PU's
  order-2x. Without this, a null cannot be distinguished from an underpowered test, and a null is
  the likely outcome.

- **D7-03 — density and hubness, reported and gating nothing.** `spearman(density, ‖H‖)`, the
  density partial on the headline correlation, and `mknn.hubness_skewness`. MKNN is a k-NN
  statistic and therefore mechanically density-sensitive; this is exactly how Phase 4's result
  became uninterpretable (§6).

- **D7-04 — per-point, not per-region.** `mknn.mknn_score` computes `(A & B).sum(axis=1) / k`, a
  per-point array, then averages it away. Retain it: 10,000 paired observations instead of 2-3
  buckets. Spearman is scale-free, so this also sidesteps the low-dynamic-range problem that
  makes tertile bucketing underpowered here. Headline statistic: `spearman(‖H‖_i, MKNN_i)` over
  all points.

- **D7-05 — additive only.** `linear_probe.py` (Phase 5) and `pointcloud_probe.py` (Phase 6) are
  sealed; import, never edit. `src/effdim/` untouched. New constants live in a new module.

- **D7-06 — freeze before any number.** Constants and the verdict/reporting rule committed in
  source before the runner can produce a PU number, with an `assert_preregistered()` guard and
  git ancestry as the proof, exactly as Phases 5 and 6 established.

- **D7-07 — CKA is out of scope.** Not implemented anywhere in the codebase. MKNN is the source
  paper's headline probe and `notebooks/pu_manifold/mknn.py` is complete (`mknn_score`,
  `permutation_null`, `bootstrap_ci`, `chance_floor`, `hubness_skewness`). Adding CKA is a
  separate decision, not a Phase 7 task.

### What Phase 7 will NOT claim (07-CONTEXT.md §8, verbatim)

- That the field measures true curvature. No ground truth for PU exists; the analytic validation
  gives a range (`+0.53` to `+0.99`), not a point estimate.
- That a null means no relationship exists, absent D7-02's power evidence.
- Anything about CKA (D7-07), or about MKNN at the source paper's `n=101,725` — this milestone
  works at `n=10,000`, where the `k/n` chance floor is ~10x higher (Phase 4's D4-19).
- Any reinterpretation of Phases 2, 02.x, 3, 03.1, 4, 5 or 6.

### Deferred / Out of Scope (nothing new; carried from REQUIREMENTS.md's project-wide table)

- Alternative alignment metrics (CKA, mutual information) — explicitly out of scope project-wide.
- Correcting for hubness in MKNN — flagged as a caveat only, project-wide (MKNN-08 precedent).
- Model-size ladder / intramodal MKNN — deferred to SCALE-01, not in v1.1.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| D7-01 | Curvature field from `cae.PlainAutoEncoder` + `decoder_curvature.plain_decoder_curvature`, swept at `d ∈ {20,25,32}` | Exact signatures confirmed below (Standard Stack, Code Examples). Cost model confirmed serial-only (Common Pitfalls 1). |
| D7-02 | Positive control planting a curvature-MKNN relationship at PU's ~1.5x `‖H‖` spread | Design pattern in Architecture Patterns / Pattern 3; no ready-made helper exists, must be hand-built per this phase |
| D7-03 | Density/hubness diagnostics, density partial reported not gating | `cross_split_curvature.partial_spearman` is a direct, unedited fit (Don't Hand-Roll); `mknn.hubness_skewness` and `curvature_probe.local_density_weights` exist unchanged |
| D7-04 | Per-point `spearman(‖H‖_i, MKNN_i)` over all 10,000 points | `mknn.mknn_score` does NOT expose the per-point array (confirmed by reading source) — new wrapper required, built on `mknn._membership_matrix` (Don't Hand-Roll / Code Examples) |
| D7-05 | Additive-only; `linear_probe.py`/`pointcloud_probe.py` sealed | Confirmed both modules' sealed status and content; new module must re-declare, never import-and-mutate |
| D7-06 | Freeze-before-compute with `assert_preregistered()` + git ancestry | Exact template extracted from `linear_probe.py` and `pointcloud_probe.py` (Architecture Patterns / Pattern 1) |
| D7-07 | CKA out of scope | No action; confirmed `grep`-verified nowhere in codebase |
</phase_requirements>

## Architectural Responsibility Map

This is a single-tier, offline, notebook/script research pipeline — there is no browser, server,
API, or database tier. The map below uses the pipeline's own internal stages instead.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Data loading (row-aligned hsc/legacysurvey) | Cached artifact (`notebooks/.cache/subsample_*.npz`) | — | Frozen since Phase 1; read-only for Phase 7 |
| Decoder training (3x, one per `d`) | New Phase 7 runner script | `cae.py` (imported, unchanged) | Compute-heavy; owns the `d`-sweep loop and serialization discipline |
| Curvature field computation | New Phase 7 runner script | `decoder_curvature.py` (imported, unchanged) | Dominant cost (~24-62 min/`d`); must run serially per §7's cost model |
| Per-point MKNN | New Phase 7 module (`per_point_mknn`) | `mknn.py` (`_membership_matrix`, imported unchanged) | The one genuine code gap D7-04 identifies |
| Significance (permutation) | New Phase 7 module, wrapping `curvature_probe.permutation_null` | — | Generic, already handles arbitrary paired statistics under ties |
| Density partial | New Phase 7 module, wrapping `cross_split_curvature.partial_spearman` | — | Already rank-based, already handles covariate control |
| Pre-registration / freeze | New Phase 7 module's own constants block + `assert_preregistered()` | `linear_probe.py` / `pointcloud_probe.py` (pattern reference only, never imported for constants) | D7-05 forbids editing either sealed module |
| Positive control (D7-02) | New Phase 7 module/runner | — | No existing helper; must be built fresh per this phase (see Pattern 3) |
| Verdict / reporting | New Phase 7 module's `VERDICT_RULE` + a runner script | — | Mirrors `region_partition_mknn_run.py`'s `apply_verdict` shape |

## Standard Stack

### Core (all already installed and pinned; no new package needed)

| Library | Version (measured, this env) | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | already used throughout `cae.py`/`decoder_curvature.py` | CAE training + `torch.func` autodiff for curvature | Sealed instrument; do not swap |
| `scipy` | **1.18.0** `[VERIFIED: .venv python -c "import scipy; print(scipy.__version__)"]` | `spearmanr`, `bootstrap`, `permutation_test`, `skew` — all already used by `mknn.py`/`curvature_probe.py` | Already the project's exclusive stats-testing library; `bootstrap`'s `paired=True` kwarg (confirmed present in this scipy version's signature) is the correct primitive for a paired-array bootstrap CI on a Spearman statistic |
| `scikit-learn` | already used (`NearestNeighbors`, `RidgeCV`) | k-NN membership matrices, ridge regression elsewhere | Unchanged from prior phases |
| `numpy` | already used | Everything else | Unchanged |

**No `pingouin` or other partial-correlation package is installed or needed**
`[VERIFIED: .venv python -c "import pingouin" raised ModuleNotFoundError]`. The project already
carries a hand-verified partial-Spearman implementation
(`cross_split_curvature.partial_spearman`) that does exactly what D7-03 needs — installing a new
dependency would duplicate existing, already-reviewed code.

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| none new | — | — | This phase needs zero new third-party packages |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cross_split_curvature.partial_spearman` | `pingouin.partial_corr(method="spearman")` | Not installed; would need a new dependency and DATA-05-style notebook-cell install for a one-function need already met in-repo |
| `curvature_probe.permutation_null`'s permutation-based p-value | `scipy.stats.spearmanr`'s built-in asymptotic p-value | The asymptotic p-value assumes an approximately continuous, tie-light distribution; MKNN's massive ties (see Common Pitfalls 2) make the permutation route the defensible one, and it is already implemented, generic, and scipy-1.18-pinned |

### Package Legitimacy Audit

Not applicable — no external packages are installed by this phase. All required functionality
exists in already-installed, already-audited dependencies (`torch`, `scipy`, `scikit-learn`,
`numpy`).

## Architecture Patterns

### System Architecture Diagram

```
                    notebooks/.cache/subsample_*.npz  (frozen, read-only, Phase 1)
                         |
                         v
              load_pu_pair()-style loader            <- copy pattern from
              (hsc[10000,768], legacysurvey[10000,768])   region_partition_mknn_run.py
                    |                    |
                    |                    +----------------------------+
                    v                                                 |
        for d in (20, 25, 32):  [STRICTLY SERIAL, OMP_NUM_THREADS capped]
          cae.PlainAutoEncoder(768, d, (250,250,250), "silu")
          cae.train_plain_ae(model, x_train32, cfg)  -----> trained model
          model.encode(x64) -> z                     -----> latent coords
          decoder_curvature.plain_decoder_curvature(model, z)
                    |
                    v
          H_norm_d (10,000,)  = ||H_vec|| per point, this d's field
                    |                                                 |
                    v                                                 v
     +-------------------------------+          NEW: per_point_mknn(hsc, legacysurvey, k)
     |  D7-04 headline statistic:    |<---------  (built on mknn._membership_matrix,
     |  spearman(H_norm_d, MKNN_i)   |             D7-04's per-point (A & B).sum(axis=1)/k)
     |  via curvature_probe.         |
     |  permutation_null(..., stat)  |
     +-------------------------------+
                    |
        +-----------+------------+-------------------------------+
        v                        v                                v
  D7-03 density diagnostics  D7-02 positive control          D7-03 hubness
  curvature_probe.           (plant relationship at PU's     mknn.hubness_skewness
  local_density_weights +    realized ~1.5x H spread,        (already computed at
  cross_split_curvature.     rerun the SAME pipeline,         zero extra cost inside
  partial_spearman(H, MKNN,  confirm recovery)                _membership_matrix)
  controls=density)
        |
        v
  New pre-registered module's VERDICT_RULE, applied mechanically
  (mirrors linear_probe.apply_verdict_rule / pointcloud_probe.VERDICT_RULE shape)
        |
        v
  notebooks/07_*.ipynb  (committed with outputs, reads the runner's cached record)
```

### Recommended Project Structure

```
notebooks/pu_manifold/
├── crossmodal_curvature.py       # NEW — Phase 7's own module (name is the planner's choice;
│                                 #   pointcloud_probe.py is the naming precedent)
│                                 #   - pre-registration constants block (D7-06)
│                                 #   - assert_preregistered()
│                                 #   - per_point_mknn(z1, z2, k)          (D7-04 gap-fill)
│                                 #   - density/hubness wrappers          (D7-03)
│                                 #   - positive-control planting helper  (D7-02)
│                                 #   - VERDICT_RULE + apply_verdict-style function
├── mknn.py                        # UNCHANGED, imported only (mknn_score, hubness_skewness,
│                                  #   chance_floor, and the private _membership_matrix)
├── cae.py                         # UNCHANGED, imported only
├── decoder_curvature.py           # UNCHANGED, imported only
├── curvature_probe.py             # UNCHANGED, imported only (permutation_null,
│                                  #   local_density_weights)
├── cross_split_curvature.py       # UNCHANGED, imported only (partial_spearman)
├── linear_probe.py                # SEALED (D7-05) — read for pattern only, not imported for
│                                  #   Phase 7 constants
├── pointcloud_probe.py            # SEALED (D7-05) — read for pattern only
└── tests/
    └── test_crossmodal_curvature.py   # NEW — mirrors test_pointcloud_probe.py's shape

notebooks/diagnostics/
└── 07_crossmodal_curvature_run.py    # NEW runner — mirrors region_partition_mknn_run.py's
                                       #   --mode flag structure and load_pu_pair() pattern;
                                       #   distinct from the existing 07_*.py SPIKE scripts,
                                       #   which stay as-is (informational, not re-run for the
                                       #   frozen number)
```

### Pattern 1: The freeze-then-compute template (D7-06), extracted verbatim from Phase 5/6

Both `linear_probe.py` and `pointcloud_probe.py` share one shape. Copy it exactly for Phase 7's
new module:

1. A top-of-file comment block stating the constants are FROZEN, committed before any real number
   exists, and that a later edit is a recorded pre-registration BREACH, not a silent fix.
2. A flat block of `UPPER_CASE` constants — every constant a plain literal, no computed defaults,
   no function call producing a "convenient" default (`pointcloud_probe.py`'s own stated
   discipline: "a default is how a pre-registered value gets inherited by accident instead of by
   an explicit call-site choice").
3. One `VERDICT_RULE` (or `SEED_VERDICT_COMBINATION_RULE`) as a triple-quoted string literal,
   containing the mechanical decision rule IN PROSE, including every caveat that must travel with
   any verdict produced under it (Phase 5/6 both embed their inherited-gap caveats directly in
   this string, not only in surrounding prose).
4. `assert_preregistered() -> None`: raises `RuntimeError` naming the FIRST failing check, one
   check per constant, checking type/shape/non-emptiness/tuple-ordering — never silently
   defaulting. `pointcloud_probe.assert_preregistered` is the simpler of the two (a
   `_REQUIRED_CONSTANTS` tuple + one generic "is any of these absent/None/empty" loop) and is the
   better template for a phase, like 7, with no per-seed branching.
5. All downstream compute functions (`fit_probe`, `bucket_edges_from_field`, ... — for Phase 7:
   `per_point_mknn`, the positive-control planting helper, the density-partial wrapper) are PURE:
   no file I/O, no defaults on pre-registered parameters, everything the caller already has in
   memory. The RUNNER (a separate `notebooks/diagnostics/07_*_run.py` script) owns all file I/O,
   caching, and the git-ancestry proof.
6. Git ancestry as freeze proof: Phase 5/6's commits show the constants committed in one commit
   (`32dabe3` for Phase 5, referenced by exact SHA in `pointcloud_probe.py`'s own docstring for
   Phase 5's inheritance), checked at runtime via `git merge-base --is-ancestor` in the relevant
   plan's acceptance criteria (per STATE.md's Phase 5 record: "`05-06` proves exactly 3 commits on
   `linear_probe.py` with the freeze most recent via `git merge-base --is-ancestor`"). Phase 7's
   plan should specify the equivalent check against ITS OWN new module, not against
   `linear_probe.py`/`pointcloud_probe.py` (D7-05 forbids touching those at all).

### Pattern 2: The per-point MKNN gap-fill (D7-04)

`mknn.py`'s exact final line of `mknn_score`:

```python
# Source: notebooks/pu_manifold/mknn.py, lines 47-59 (read directly this session)
def mknn_score(z1: Any, z2: Any, k: Any) -> Any:
    ...
    A = _membership_matrix(z1, k)
    B = _membership_matrix(z2, k)
    return float(((A & B).sum(axis=1) / k).mean())   # <-- the per-point array is discarded here
```

`_membership_matrix(Z, k)` (also read directly, lines 21-44) is a plain top-level function — not
a class method, not name-mangled — importable as `mknn._membership_matrix`. It always builds
`NearestNeighbors(n_neighbors=k + 1, algorithm="brute")`, matching the project's fixed convention.
Never editing `mknn.py`, the new module should add:

```python
# NEW function, in the Phase 7 module — composes mknn._membership_matrix exactly as
# mknn.mknn_score does, but returns the array mknn_score discards.
from . import mknn

def per_point_mknn(z1: Any, z2: Any, k: Any) -> np.ndarray:
    """(A & B).sum(axis=1) / k -- the per-point array mknn.mknn_score computes internally
    and averages away. Composes mknn._membership_matrix unchanged; k+1-neighbour convention,
    self-excluded, is identical to mknn_score's own."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    if z1.shape[0] != z2.shape[0]:
        raise ValueError(...)  # mirror mknn_score's own guard text
    A = mknn._membership_matrix(z1, k)
    B = mknn._membership_matrix(z2, k)
    return (A & B).sum(axis=1) / k
```

A regression test should assert `per_point_mknn(z1, z2, k).mean() == mknn.mknn_score(z1, z2, k)`
on a small fixture, pinning the two functions' agreement so a future edit to either cannot
silently diverge them (the exact discipline `decoder_curvature.py` already applies to
`CURVATURE_CONVENTION` across three modules).

### Pattern 3: Positive control at PU's realized spread (D7-02) — no ready-made helper exists

Phase 6's own `pointcloud_probe.py` selfcheck used `rng.random(n)`, an unstructured
~20x-spread synthetic field, and D7-02 explicitly rejects that as underpowered evidence for PU's
~1.5x-spread regime. No function in this codebase currently plants a curvature-MKNN relationship
at a *specified* dynamic range — this must be authored new for Phase 7. Design pattern
(informed by `curvature_probe.permutation_null`'s existing statistic-generic shape and the
measured PU field's own spread, `‖H‖` p95/p05 ≈ 1.495 per `07-CONTEXT.md` §5):

1. Take PU's own real `d=20` `‖H‖` field (already the exact spread D7-02 must plant *at*) as the
   independent variable `H_real`.
2. Construct a synthetic per-point `MKNN_planted` array whose values are `j/k` for integer
   `j in [0, k]` (matching MKNN's real discretization — see Common Pitfalls 2), where `j` is drawn
   with a probability that increases monotonically with `H_real`'s rank — e.g. rank-transform
   `H_real` to `[0, 1]`, then draw `j ~ Binomial(k, p_i)` where `p_i` is a logistic or linear
   function of the rank, tuned so the resulting `spearman(H_real, MKNN_planted)` lands at a
   pre-declared target effect size (e.g. `rho ≈ 0.05, 0.10, 0.20` — a small grid, not one point,
   so the control reports a detectable-effect-size *curve* rather than a single pass/fail).
3. Run the IDENTICAL pipeline (the same `per_point_mknn`-shaped statistic function fed to
   `curvature_probe.permutation_null`) on `(H_real, MKNN_planted)` and report at which planted
   effect size the permutation test's `clears_null` first turns `True`, at the SAME `n=10,000` and
   the SAME permutation/bootstrap parameters the real headline test will use.
4. This must be pre-registered (the target effect-size grid, the planting mechanism, the sample
   size) BEFORE the real `spearman(‖H‖_i, MKNN_i)` number is computed, per D7-06 — otherwise the
   control could be tuned post hoc to make an observed null look either well- or under-powered.

This is a genuinely new piece of statistical engineering for this phase; flag it to the planner as
the one task carrying real design risk, not a reuse-and-wire task like the rest of D7-01/03/04.

### Anti-Patterns to Avoid

- **Editing `mknn.py` to add a `return_per_point` flag to `mknn_score`.** D7-05 forbids editing
  sealed modules, and while `mknn.py` is not explicitly named alongside `linear_probe.py` /
  `pointcloud_probe.py` in D7-05's sentence, it carries the same "additive only" spirit CLAUDE.md
  states project-wide, and Phase 4's own `region_partition_mknn_run.py` and Phase 6's
  `pointcloud_probe.py` both treat `mknn.py` as a stable, unedited dependency. Add a new function
  in the new module instead (Pattern 2).
- **Re-running the `07_*.py` spike scripts as if they were the frozen computation.** They write to
  `notebooks/diagnostics/*.jsonl` directly (not `notebooks/.cache/`), have no
  `assert_preregistered()` guard, and their own docstrings say "NOT a reproduction of any sealed
  cell. Writes only to scratchpad." Their numbers are already correctly quoted into
  `07-CONTEXT.md` as informational context (what `d` to use, the instrument's honest rho range);
  re-running them does not satisfy D7-06's freeze-before-compute requirement.
- **Running the three `d ∈ {20, 25, 32}` fits concurrently (e.g. as three parallel plan-execution
  waves).** `07-CONTEXT.md` §7 explicitly measured three concurrent torch jobs driving system
  load to 44 and causing "roughly a 10x slowdown" on this 20-core machine. The three-`d` sweep is
  ~2 hours SERIAL; running it as concurrent wave tasks would not save wall-clock time and risks
  much worse total time from thread contention. The plan must schedule these as one sequential
  task/script, not parallel tasks.
- **Recomputing MKNN three times, once per `d`.** MKNN depends only on the frozen `hsc`/
  `legacysurvey` embeddings and `k`, never on the decoder's latent dimension. Compute
  `per_point_mknn` ONCE and reuse it against all three `d`'s `‖H‖` fields.
- **Trusting `scipy.stats.spearmanr`'s built-in asymptotic p-value on this data.** See Common
  Pitfalls 2 below — the massive tie structure in MKNN's `j/k` values makes the permutation route
  (already implemented, `curvature_probe.permutation_null`) the defensible choice.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Partial Spearman controlling for density (D7-03) | A new residualize-and-correlate routine | `cross_split_curvature.partial_spearman(x, y, controls)` (already in-repo, read this session, lines 232-288) | Rank-transforms x/y/controls, residualizes by least squares with intercept, returns Pearson on residuals — exactly the "partial correlation in rank space" definition D7-03 needs, already reviewed code (ported from `curvature-experiments` branch and adapted) |
| Permutation-based significance for a Spearman statistic under massive ties (D7-04) | A hand-rolled shuffle-and-recompute loop | `curvature_probe.permutation_null(x, y, n_resamples, seed, quantile, statistic_fn=None)` (already generic; default `statistic_fn` is exactly `spearmanr(x,y).statistic`) | Already uses `scipy.stats.permutation_test(..., permutation_type="pairings")` correctly (both arrays independently repermuted per resample, verified against this scipy version per the function's own docstring); reusing it means Phase 7's significance test is bit-for-bit the same machinery Phase 02.5 already validated |
| Bootstrap CI on a paired Spearman statistic | A manual resample-and-recompute loop | `scipy.stats.bootstrap((x, y), statistic_fn, paired=True, ...)` (`paired=True` confirmed present in this scipy 1.18.0's signature) | scipy's own paired-bootstrap primitive; avoids re-deriving the correct resampling unit (pairs, not independent columns) |
| k-NN membership matrix construction | A new `NearestNeighbors` wrapper | `mknn._membership_matrix` (import, do not edit) | Already the fixed, three-times-repeated convention across `mknn.py` and `curvature_probe.py`; recomputing it differently for Phase 7 risks a subtly different `k+1`/self-exclusion convention |
| Local density weighting for D7-03's density diagnostic | A new k-NN density estimator | `curvature_probe.local_density_weights(X, k_density, d)` (already used identically by Phase 4's `run_partition`) | Already the exact function Phase 4 used for REGN-01/REGN-02; reusing it keeps Phase 7's density number comparable to Phase 4's |

**Key insight:** Phase 7 needs almost no new numerical code — the gap is entirely in
*composition and freezing discipline* (D7-04's missing per-point exposure, D7-06's freeze
machinery, D7-02's positive-control design), not in new mathematics. Any plan that proposes
writing a new Spearman-with-permutation-test routine, a new bootstrap-CI routine, or a new
partial-correlation routine from scratch is duplicating already-reviewed code and should be
rejected at plan-review.

## Common Pitfalls

### Pitfall 1: Concurrent `d`-sweep torch jobs silently blowing the ~2-hour cost budget

**What goes wrong:** A plan structures the `d ∈ {20, 25, 32}` fits as three independently
launchable tasks (e.g. three parallel wave tasks in a GSD plan), reasoning that they are
data-independent and therefore parallelizable.

**Why it happens:** They ARE logically independent (`d=25`'s fit does not depend on `d=20`'s
output), so parallelization looks free. But `07-CONTEXT.md` §7 states directly: "three concurrent
torch jobs on this 20-core machine drove load to 44 and cost roughly a 10x slowdown." Torch's
default CPU threading contends across processes.

**How to avoid:** Schedule the `d`-sweep as ONE sequential script/task, looping over
`(20, 25, 32)` in-process, with `torch.set_num_threads(N)` (or `OMP_NUM_THREADS` set in the
environment before the torch import) capped to a sane fraction of available cores. Do not let a
GSD wave structure imply concurrency here.

**Warning signs:** A plan wave diagram showing three separate `d`-sweep tasks in the same wave;
absence of any explicit `OMP_NUM_THREADS`/`torch.set_num_threads` call in the new runner (grep of
existing scripts, done this session, found ZERO prior use of `OMP_NUM_THREADS` anywhere in
`notebooks/` — this is new territory for the codebase, not a copy-paste from a prior runner).

### Pitfall 2: Treating MKNN's per-point array as a continuous variable for significance testing

**What goes wrong:** `per_point_mknn(hsc, legacysurvey, k)` returns values `j / k` for integer
`j ∈ {0, ..., k}`. At the pre-registered `k=10` (the milestone's convention per prior phases'
`k` grids `(5, 10, 20, 50)`), there are only **11 distinct possible values** across all 10,000
points. `scipy.stats.spearmanr`'s reported p-value uses (depending on scipy version and tie
count) either an exact permutation distribution for small n or a t-distribution approximation
that assumes an approximately continuous underlying variable — with 10,000 points collapsed into
11 buckets, that asymptotic approximation's validity is doubtful, though `spearmanr`'s rho
POINT ESTIMATE itself (using average-rank tie handling) remains a valid descriptive statistic.

**Why it happens:** MKNN is inherently discrete by construction (`(A & B).sum(axis=1) / k` where
the intersection count is an integer 0..k) — this is not a bug in the estimator, it is the
statistic's nature.

**How to avoid:** Use `curvature_probe.permutation_null`'s permutation-based p-value (exact under
the null of no association, valid regardless of tie structure, since it repeatedly recomputes the
SAME `spearmanr` statistic under random re-pairing rather than relying on a parametric
approximation) as the significance route, not `spearmanr`'s own bundled p-value. For the
Spearman rho point estimate itself, `spearmanr`'s default average-rank tie handling is fine and
standard. State the number of distinct MKNN values explicitly in the phase's write-up (mirroring
`05-RESEARCH.md` Pitfall 2 / the `linear_probe.py` docstring's own precedent of stating distinct-
value counts explicitly rather than letting a reader assume continuity) — this is exactly the
"05-02-SUMMARY.md's wrong distinct-value claim" class of error already once made and corrected in
this project (see `linear_probe.py`'s "Correction to 05-02-SUMMARY.md" docstring paragraph); do
not repeat it for MKNN.

**Warning signs:** A plan that reports a p-value from `scipy.stats.spearmanr(...).pvalue` directly
without cross-checking it against a permutation-based p-value; a plan that does not state the
number of distinct MKNN values achieved at its chosen `k`.

### Pitfall 3: Reading the instrument-validation scripts' numbers as if they were Phase 7's own frozen output

**What goes wrong:** A plan cites `07_instrument_fixture_sweep_run.py`'s `rho = +0.9745` (ridge,
`D=768`) or `07_pu_plain_ae_fit_run.py`'s `cond(g)` median as though these were computed under
Phase 7's own pre-registration, satisfying D7-06.

**Why it happens:** The numbers ARE real, already measured, and already correctly quoted in
`07-CONTEXT.md` — but they were produced by scratch scripts (writing to
`notebooks/diagnostics/*.jsonl`, not `notebooks/.cache/`) run interactively before any Phase 7
module existed. They establish WHICH instrument to use and WHAT `d` to sweep — they are not
themselves subject to, or evidence of, the freeze-before-compute discipline D7-06 requires for the
headline PU number.

**How to avoid:** Treat every number in `07-CONTEXT.md` §4-§5 as fixed, already-established
INPUT to Phase 7's design (informing what to build), never as output the phase gets credit for
having produced under its own pre-registration. The phase's own frozen module must independently
compute D7-01's field at each `d` inside its own runner, using the exact same `cae.py` /
`decoder_curvature.py` calls, but through Phase 7's own pre-registered, git-ancestry-proven code
path.

**Warning signs:** A plan task that says "reuse the fields already computed by
`07_pu_plain_ae_fit_run.py`" instead of "recompute the `d=20/25/32` fields via the new frozen
module's own runner."

### Pitfall 4: Assuming the Swiss Roll sanity check (CLAUDE.md) applies fresh to Phase 7

**What goes wrong:** A plan schedules a new `notebooks/07_swiss_roll_*.ipynb` under CLAUDE.md's
"every new manifold-learning model" rule.

**Why it happens:** CLAUDE.md's rule is broad ("chart auto-encoders, ... decoder
parameterizations, curvature estimators, and anything else in that family"), and Phase 7 does use
a curvature estimator.

**How to avoid:** Phase 7 introduces NO new model or estimator — it reuses `cae.PlainAutoEncoder`
+ `decoder_curvature.plain_decoder_curvature` verbatim, and this EXACT combination already has a
Swiss Roll sanity check on record: `notebooks/02.6_swiss_roll_plainae_curvature_check.ipynb`
`[VERIFIED: file exists, found by directory listing this session]`. Additionally,
`decoder_curvature.plain_decoder_curvature` was independently validated against four analytic
ground-truth fixtures at `d=20`, `D=768` this same week (`07-CONTEXT.md` §4) — a stronger check
than the Swiss Roll gives, since it has a known closed-form answer at the SAME `d`/`D` PU actually
uses. The planner should record this explicitly (which existing notebook satisfies the CLAUDE.md
gate, and why the §4 analytic-fixture sweep is a superset of what a fresh Swiss Roll run would
show) rather than either skip the question silently or schedule redundant work.

**Warning signs:** A plan task titled "Swiss Roll check for Phase 7's curvature instrument" with
no citation of `02.6_swiss_roll_plainae_curvature_check.ipynb`.

## Code Examples

### Loading the row-aligned PU pair (copy from Phase 4's runner)

```python
# Source: notebooks/diagnostics/region_partition_mknn_run.py, lines 42-70 (read this session)
def load_pu_pair(column_a="hsc", column_b="legacysurvey"):
    cands = sorted(glob.glob(str(NOTEBOOK_ROOT / ".cache" / "subsample_*.npz")))
    if not cands:
        raise FileNotFoundError("no subsample_*.npz in notebooks/.cache/")
    best, best_n = None, -1
    for c in cands:
        with np.load(c) as z:
            if column_a in z.files and column_b in z.files and z[column_a].shape[0] > best_n:
                best, best_n = c, z[column_a].shape[0]
    with np.load(best) as z:
        Xa = np.asarray(z[column_a], dtype=np.float64)
        Xb = np.asarray(z[column_b], dtype=np.float64)
    return Xa, Xb, best
```

### The D7-01 decoder + curvature call (exact signatures, confirmed by reading source)

```python
# cae.PlainAutoEncoder.__init__(self, in_dim, latent_dim, hidden=(250,250,250), activation="silu")
# cae.train_plain_ae(model, x_train, cfg) -> {"history", "epochs_run", "wallclock_s",
#     "wallclock_truncated", "early_stopped", "cfg"}  (identical shape to train_cae)
# decoder_curvature.plain_decoder_curvature(model, z) -> {
#     "H_vec": (batch, out_dim), "H_norm": (batch,), "metric_condition_number": (batch,),
#     "jacobian_shape", "hessian_shape", "activation", "curvature_convention" }
model = cae.PlainAutoEncoder(in_dim=768, latent_dim=d, hidden=(250, 250, 250), activation="silu")
cae.train_plain_ae(model, x_train32, cfg)          # cfg needs at least: seed, lr, weight_decay,
                                                     # batch, max_epochs (see 07_pu_plain_ae_fit_run.py
                                                     # for a working cfg_base)
model.eval().double()
with torch.no_grad():
    z = model.encode(x64)                           # x64: (10000, 768) float64
field = decoder_curvature.plain_decoder_curvature(model, z)
H_norm = field["H_norm"].detach().cpu().numpy()     # (10000,)
```

### The generic permutation-null call for D7-04's significance test

```python
# Source: notebooks/pu_manifold/curvature_probe.py, lines 1021-1147 (read this session)
# statistic_fn=None reproduces spearmanr(x, y).statistic under permutation_type="pairings"
result = curvature_probe.permutation_null(
    H_norm, mknn_per_point, n_resamples=N_PERMUTATIONS, seed=SEED, quantile=NULL_QUANTILE
)
# result: {"observed_rho", "null_quantile", "null_threshold", "null_mean", "null_std",
#          "n_resamples", "seed", "clears_null"}
```

### The partial-Spearman call for D7-03's density diagnostic

```python
# Source: notebooks/pu_manifold/cross_split_curvature.py, lines 232-288 (read this session)
raw_rho = cross_split_curvature.partial_spearman(H_norm, mknn_per_point, controls=None)
density_controlled_rho = cross_split_curvature.partial_spearman(
    H_norm, mknn_per_point, controls=density  # density: (10000,) from curvature_probe.local_density_weights
)
```

## State of the Art

Not applicable in the usual sense (no external library/API surface to track) — the "state of the
art" here is entirely intra-project: which of this codebase's own modules are the current,
correct instrument. That state is:

| Superseded approach | Current approach | When changed | Impact |
|--------------------|-------------------|---------------|--------|
| Bucketing curvature into 2-3 regions/tertiles then comparing means (Phases 4, 5, 6) | Per-point Spearman over all 10,000 points (D7-04) | 2026-08-25, this phase | 10,000 paired observations instead of 2-3 buckets; sidesteps PU's near-constant `‖H‖` field problem |
| Ridge-regression residual as the outcome variable (Phases 5, 6) | MKNN as the outcome variable (D7-04) | 2026-08-25, this phase | Matches the source paper's (arXiv:2509.19453) actual headline probe; Phases 5/6 answered a different question |
| Curvature-direction partitioning (Phase 4) | Curvature-magnitude Spearman (D7-04) | 2026-08-25, this phase | Avoids the density-confounded direction axis (`spearman(density, direction) = +0.8208`) that made Phase 4's HOLDS uninterpretable |
| Phase 6's `rng.random(n)` selfcheck (~20x spread) as the sole power evidence | A spread-matched (~1.5x) positive control (D7-02) | 2026-08-25, this phase | The old selfcheck cannot license a null on PU's actual, much narrower dynamic range |

**Deprecated/outdated:** None of Phases 4/5/6's sealed verdicts are deprecated or reopened — they
remain valid answers to the (different) questions they asked. Only their outcome variable/axis
choice is superseded FOR THE PURPOSE of answering Phase 7's specific research question.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The positive-control design in Pattern 3 (rank-based logistic/linear planting of `MKNN_planted` at a target Spearman rho, swept across a small effect-size grid) is a reasonable, defensible instantiation of D7-02's requirement | Architecture Patterns / Pattern 3 | This is original design work for this research doc, not verified against a second source or prior project precedent; the planner/developer should treat the exact planting mechanism as open for their own judgment, not as settled |
| A2 | `k=10` is the right MKNN `k` to use for the headline per-point statistic (used for the tie-count illustration in Pitfall 2) | Common Pitfalls 2 | Prior phases used a grid `{5, 10, 20, 50}` with `HEADLINE_K=20` (Phase 4); if Phase 7 adopts a different headline `k`, the exact tie count changes (at `k=20` there are 21 distinct values, still massive ties relative to n=10,000) but the qualitative pitfall and its permutation-based fix are unaffected |
| A3 | `mknn._membership_matrix` being a plain, unmangled top-level function (not a class method or name-mangled with `__`) makes it safely importable from a new module without violating D7-05's "additive only" spirit | Architecture Patterns / Pattern 2 | Low risk — confirmed directly by reading `mknn.py`'s source this session; the function has no leading-underscore access restriction beyond Python's naming convention |

**If this table is empty:** N/A — see entries above. All entries are genuinely assumption-level
(design judgment or scope interpretation), not unverified facts — every code-level claim in this
document (function signatures, return shapes, sealed-module status, file existence) was confirmed
by directly reading the relevant source file in this session and is tagged accordingly.

## Open Questions

1. **What exact `k` should the headline per-point MKNN statistic use?**
   - What we know: prior phases used the grid `{5, 10, 20, 50}` with `HEADLINE_K = 20`
     (`region_partition.py`). `07-CONTEXT.md` does not explicitly pin a `k` for D7-04.
   - What's unclear: whether Phase 7 should inherit `HEADLINE_K = 20` for continuity with Phase 4,
     or choose independently since the outcome variable itself has changed.
   - Recommendation: inherit `HEADLINE_K = 20` for cross-phase comparability (it is already the
     project's established headline `k`), and report the sensitivity grid `{5, 10, 50}` exactly as
     Phase 4 did, non-gating. This should be an explicit pre-registered constant in the new
     module, decided BEFORE any real number is computed (D7-06).

2. **Does the positive control (D7-02) need its own frozen pre-registration entry distinct from
   the headline test's, or can it share the same freeze commit?**
   - What we know: D7-06 requires freeze-before-compute for "the runner" generally; Phase 5/6's
     precedent freezes everything (including selfcheck-adjacent constants) in one commit.
   - What's unclear: whether the effect-size grid and planting mechanism for D7-02 must be
     pre-registered with the SAME rigor as the headline statistic, given it operates on synthetic
     rather than real MKNN values.
   - Recommendation: freeze it in the same commit/module as everything else — cheaper to over-
     freeze than to have a reviewer later ask why the power analysis wasn't pre-registered
     alongside the test it licenses.

3. **Should `d=20/25/32`'s three fields be trained from three different seeds, or one seed
   reused across `d`?**
   - What we know: Phase 5's decoder fields at fixed `d=20` were measured to be seed-UNSTABLE
     (mutually anti-correlated across three seeds; `linear_probe.py` docstring (b)). `07-CONTEXT.md`
     D7-01 does not mention seeds at all for the `d`-sweep, only that `d` itself sweeps.
   - What's unclear: whether "same answer at each `d`" (D7-01's stated success condition) is
     robust to the seed-instability Phase 5 already measured, or whether a single seed per `d`
     risks reproducing Phase 5's instability under a different guise (three different `d`s instead
     of three different seeds of the same `d`).
   - Recommendation: use ONE fixed seed across the `d`-sweep (simplest, matches D7-01's literal
     text), but the plan should explicitly name this as an accepted limitation inherited from
     Phase 5's known seed-instability finding, not silently assume single-seed stability.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `torch` | D7-01 (CAE training, curvature autodiff) | Yes (used by every prior `07_*.py` script this session) | Not independently re-checked this session; already exercised successfully by `07_pu_plain_ae_fit_run.py` | — |
| `scipy` | D7-03, D7-04 (spearmanr, bootstrap, permutation_test) | Yes | `[VERIFIED: 1.18.0, checked this session]` | — |
| `scikit-learn` | `mknn._membership_matrix` (`NearestNeighbors`) | Yes | Already used throughout `mknn.py`, unversioned check this session | — |
| `notebooks/.cache/subsample_*.npz` | D7-01, D7-04 data loading | Yes (multiple `07_*.py` scripts this session successfully loaded it via `glob`) | Frozen since Phase 1 | — |
| `git merge-base --is-ancestor` | D7-06 freeze proof | Yes (standard git, already used by Phase 5/6's own plans) | — | — |
| `pingouin` | Not required (see Don't Hand-Roll) | No | — | Not needed; `cross_split_curvature.partial_spearman` covers the requirement without a new dependency |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** `pingouin` — not installed, but not needed; in-repo
`cross_split_curvature.partial_spearman` is the fallback and is preferred anyway (already
reviewed code, avoids DATA-05-style new notebook-cell installs).

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already the project convention; `notebooks/pu_manifold/tests/` holds 22 existing test files, one per module, e.g. `test_pointcloud_probe.py`, `test_region_partition.py`) |
| Config file | Root `pyproject.toml`'s `[tool.pytest.ini_options]` sets `testpaths = ["tests"]` for `src/effdim/`'s OWN tests; `notebooks/pu_manifold/tests/` is invoked by explicit path (confirmed convention: every prior phase's plan runs `python -m pytest notebooks/pu_manifold/tests/test_<module>.py -q`) |
| Quick run command | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py -q` (new file, name TBD by planner — `test_crossmodal_curvature.py` matches the module-name convention if the module is named `crossmodal_curvature.py`) |
| Full suite command | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/ -q` (22 existing files + the new one; prior phases report counts like "289 passed, 1 skipped" as their full-suite gate) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| D7-04 | `per_point_mknn(z1, z2, k).mean() == mknn.mknn_score(z1, z2, k)` on a small fixture | unit | `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py::test_per_point_mknn_mean_matches_mknn_score -x` | ❌ Wave 0 |
| D7-04 | Known-answer MKNN cases (identical pair scores 1.0, independent Gaussian clouds land near chance floor) — mirror `mknn.py`'s own `selfcheck()` shape in `region_partition_mknn_run.py` | unit | `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py::test_per_point_mknn_known_answers -x` | ❌ Wave 0 |
| D7-06 | `assert_preregistered()` raises `RuntimeError` on any missing/malformed constant, mirroring `pointcloud_probe.assert_preregistered`'s test shape | unit | `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py::test_assert_preregistered_raises_on_unset_constants -x` | ❌ Wave 0 |
| D7-03 | `cross_split_curvature.partial_spearman` regression-pinned against a hand-computed value on a tiny synthetic case (already exists in `test_cross_split_curvature.py` presumably — verify, do not re-derive) | unit | `pytest notebooks/pu_manifold/tests/test_cross_split_curvature.py -q` | ✅ (pre-existing) |
| D7-02 | Positive control recovers a planted effect at the pre-registered target rho, on a synthetic (not PU) fixture, fast enough for CI | unit/smoke | `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py::test_positive_control_recovers_planted_effect -x` | ❌ Wave 0 |
| D7-01 | Full PU-scale run (3 `d` values, real 10,000-point data, ~2 hours) | manual-only (too slow for automated CI-style sampling) | `.venv/bin/python notebooks/diagnostics/07_crossmodal_curvature_run.py --mode dsweep` (real run, not a test) | N/A — this is the phase's actual deliverable computation, not a test |

### Sampling Rate

- **Per task commit:** `pytest notebooks/pu_manifold/tests/test_crossmodal_curvature.py -q` (fast,
  synthetic-fixture only — never touches the real 10,000-point PU data or trains a real decoder)
- **Per wave merge:** `pytest notebooks/pu_manifold/tests/ -q` (full suite, ~seconds, matches prior
  phases' gate)
- **Phase gate:** Full suite green before `/gsd-verify-work`; separately, the real `d`-sweep run
  (D7-01, ~2 hours wall-clock) is NOT part of the automated test suite — it is the phase's actual
  scientific deliverable, run once (or a small number of times) manually/via a long-running
  background task, with its own `notebooks/07_*.ipynb` committed with outputs per CLAUDE.md.

### Wave 0 Gaps

- [ ] `notebooks/pu_manifold/tests/test_crossmodal_curvature.py` — new file, covers D7-02, D7-04, D7-06
- [ ] No new shared fixtures needed — existing `tests/` files already establish the small-synthetic-
      fixture pattern (`rng.normal(size=(400, 16))`-style, per `region_partition_mknn_run.py`'s
      `selfcheck()` and the existing `tests/test_mknn.py`... **note:** no `tests/test_mknn.py` was
      found in the tests directory listing this session (`mknn.py` is covered only by
      `region_partition_mknn_run.py --selfcheck`, per `04-18` declining a dedicated test file per
      that phase's own D4-18) — Phase 7's new module should decide independently whether it wants
      a `--selfcheck` runner flag (matching `mknn.py`'s own precedent) or a proper `tests/`
      pytest file (matching `linear_probe.py`/`pointcloud_probe.py`'s precedent); the latter is
      recommended since D7-06's freeze discipline is closer to Phase 5/6's than to Phase 4's MKNN
      module.
- [ ] Framework install: none — pytest is already a project dev dependency (`pyproject.toml`
      `[project.optional-dependencies] dev = ["pytest", ...]`)

## Security Domain

`security_enforcement` is absent from `.planning/config.json`, so it defaults to enabled; the
section is included per protocol, but its content is necessarily thin — this is the same
conclusion Phase 02.6's own code review reached and stated explicitly ("this code has no network
surface, no auth, no user-input path and no persistence layer") and nothing about Phase 7 changes
that posture.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth surface anywhere in this notebook-only research pipeline |
| V3 Session Management | No | No sessions |
| V4 Access Control | No | No multi-user access model |
| V5 Input Validation | Partial | The new module's functions should raise on malformed input (shape mismatches, non-finite values, wrong dtypes) — exactly the pattern every existing `pu_manifold` module already follows (`mknn.py`, `linear_probe.py`, `pointcloud_probe.py` all guard their inputs this way); this is a code-quality/correctness practice here, not a security boundary, since there is no untrusted external input — all data originates from the project's own frozen, local `.npz` cache |
| V6 Cryptography | No | No cryptography used or needed |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed/adversarial `--record-path` CLI argument (path traversal into an unintended write location) | Tampering | `cache.py`'s `_assert_inside_cache` containment guard already exists and is used by every `npz_cache`/`json_cache` call; the new runner should route through `cache.py`'s helpers for any cached artifact rather than writing raw paths, matching every existing `07_*.py`/Phase-4-6 runner's convention |
| None else applicable | — | This is a local, single-user, offline research script with no network I/O, no user accounts, and no persisted secrets |

## Sources

### Primary (HIGH confidence — all read directly from source files in this session)

- `notebooks/pu_manifold/mknn.py` (full file) — `mknn_score`, `permutation_null`, `bootstrap_ci`,
  `hubness_skewness`, `chance_floor`, `_membership_matrix` exact signatures and return shapes
- `notebooks/pu_manifold/decoder_curvature.py` (full file) — `plain_decoder_curvature`,
  `assert_c2_decoder`, `plain_decoder_map` exact signatures, return dict keys, cost/batching notes
- `notebooks/pu_manifold/cae.py` (full file) — `PlainAutoEncoder`, `train_plain_ae`,
  `reconstruction_stats`, `_train_decoder_protocol`
- `notebooks/pu_manifold/linear_probe.py` (full file) — the freeze template, `assert_preregistered`
  shape, `VERDICT_RULE` string conventions
- `notebooks/pu_manifold/pointcloud_probe.py` (full file) — the "inherit by re-declaration" template
  a new Phase 7 module should follow against Phase 5's precedent
- `notebooks/pu_manifold/region_partition.py` (partial, lines 1-120) — pre-registration idiom,
  `assert_preregistered` pattern, `VERDICT_RULE` shape
- `notebooks/pu_manifold/cross_split_curvature.py` (relevant sections) — `partial_spearman` full
  implementation
- `notebooks/pu_manifold/curvature_probe.py` (function list + `permutation_null` full
  implementation, lines 1021-1147)
- `notebooks/pu_manifold/subsample.py` (full file) — `load_subsample`, alignment guarantees, cache
  key structure
- `notebooks/diagnostics/region_partition_mknn_run.py` (full file) — `load_pu_pair`,
  `run_global_cell`, `run_regional_cell`, `apply_verdict`, `selfcheck` — the exact runner-script
  template to mirror
- `notebooks/diagnostics/curvature_field_pu_run.py` (partial, lines 440-480) — `_load_subsample`,
  `_split`, `PU_SPLIT_SEED`, `PU_HOLDOUT_FRACTION`
- `notebooks/diagnostics/07_pu_plain_ae_fit_run.py`, `07_pu_latent_recon_sweep_run.py` (full files)
  — confirmed these are spike/scratch scripts, not sealed infrastructure
- `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-CONTEXT.md` (full file) — the
  authoritative, self-contained decision/evidence dossier for this phase
- `.planning/REQUIREMENTS.md`, `.planning/STATE.md` (relevant sections) — project-wide requirement
  set and decision history
- `.claude/skills/spike-findings-effdim/` (loaded via Skill tool) — curvature-estimator validation
  protocol, the d=20 saddle-fixture open question (informs but does not gate Phase 7, since Phase 7
  uses `decoder_curvature.plain_decoder_curvature`, not the local-polynomial teacher the spike
  tested)
- Direct shell verification this session: `scipy.__version__ == "1.18.0"`; `pingouin` not
  installed; `inspect.signature(scipy.stats.bootstrap)` confirms `paired=False` default kwarg
  exists; `git log` confirms no Phase 7 module yet exists in `notebooks/pu_manifold/`; directory
  listing confirms `02.6_swiss_roll_plainae_curvature_check.ipynb` exists

### Secondary (MEDIUM confidence)

- The exact positive-control planting mechanism in Pattern 3 (Binomial-draw-on-logistic-rank) is
  original design reasoning applied to measured facts (PU's ~1.5x spread, MKNN's `j/k`
  discretization), not itself drawn from or verified against an external source — flagged in the
  Assumptions Log (A1).

### Tertiary (LOW confidence)

- None — every quantitative claim in this document about `07-CONTEXT.md`'s own measured numbers
  (rho ranges, cost timings, reconstruction percentages) is a direct citation of that file, which
  the task instructions designate as authoritative and non-reopenable; this research did not
  re-derive or re-verify those numbers independently (that would violate the phase's own "no
  sealed verdict reopened" discipline, and `07-CONTEXT.md` is explicitly the sealed evidence base
  for this phase, not a claim needing further checking).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages, all versions/availability directly checked this session
- Architecture (freeze pattern, data loading, module structure): HIGH — extracted directly from
  three full source files (`linear_probe.py`, `pointcloud_probe.py`, `region_partition_mknn_run.py`)
  that are the literal precedent this phase is instructed to follow
- D7-04 gap-fill (`per_point_mknn`): HIGH — the gap and its fix are both directly demonstrable from
  `mknn.py`'s source, not inferred
- D7-02 positive-control design: MEDIUM — the requirement is clear (D7-02's text) but the specific
  planting mechanism is original design work, not itself verified against a second source
- Pitfalls (concurrency cost, MKNN ties): HIGH (concurrency, directly quoted from `07-CONTEXT.md`
  §7's own measurement) and MEDIUM (tie-handling recommendation — the discreteness fact is
  HIGH-confidence arithmetic, `k=10` implies 11 distinct values; the recommended fix, using the
  existing permutation machinery, is a defensible but not independently-benchmarked methodological
  choice)

**Research date:** 2026-08-25
**Valid until:** This phase's own code does not change upstream — validity is bounded by whether
`mknn.py`, `cae.py`, `decoder_curvature.py`, `linear_probe.py`, `pointcloud_probe.py`,
`cross_split_curvature.py`, or `curvature_probe.py` are edited before Phase 7 is planned/executed.
Since D7-05 forbids editing the sealed ones and the others are stable, treat this research as valid
until Phase 7 completes (no natural 30-day decay applies to a fixed, non-networked codebase
snapshot).
