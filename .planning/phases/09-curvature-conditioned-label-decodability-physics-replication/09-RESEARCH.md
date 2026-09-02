# Phase 9: Curvature-Conditioned Label Decodability (Physics Replication) - Research

**Researched:** 2026-09-02
**Domain:** Instrument replication (decoder curvature vs. a published nested-PCA quadratic-chart
curvature estimator) on real astronomical embeddings; row-alignment proof for an id-less join;
partial-Spearman rank statistics with permutation/bootstrap inference.
**Confidence:** MEDIUM — the statistical machinery is HIGH confidence (verified directly against
sealed source and the colleague's branch); the data-loading path has one HIGH-confidence
correction to CONTEXT.md's own assumption (the label source needs a non-default HF revision) and
one still-open column-mapping choice that needs a plan-time or execution-time decision.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Sample, neighbourhood and anchor scale**
- D9-01 — Full 86,471 Physics rows. `UniverseTBD/pu-embeddings` config `physics_vit_base_test`
  (768-D, single column `vit_base_galaxies`), the whole test split. No subsample.
- D9-02 — Local `R^2` neighbourhood `k = 2048` only. No `k` grid. State the `n`-ratio (1/42 vs
  his 1/8) next to the number.
- D9-03 — 512 anchors, seeded uniform draw, matching his `n = 512`.
- D9-04 — Anchors are drawn from the AE holdout rows only (~17k at Phase 7's
  `HOLDOUT_FRACTION = 0.2`). A deliberate departure from Phase 7's
  `FIELD_EVALUATED_ON = all_10000_rows_including_the_8000_training_rows`. Neighbourhoods (k-NN
  over all 86,471 rows) and the OOF probe folds are independent of the AE split. Reversibility:
  costly.

**Row-alignment proof**
- D9-05 — The colleague's standard is a principle, not a method. His Physics join is a documented
  convention with no test on the branch. Phase 9 supplies the method.
- D9-06 — Method: statistical shifted-row check. Fit the 5-fold OOF ridge probe embedding →
  `mag_r` at shift 0, then at each alignment in a frozen shift set. Aligned data gives
  `R^2(shift 0)` far above every shifted `R^2` (≈0).
- D9-07 — Shift set, frozen before download: `mag_r` only; row shifts `±1..±10`, `±100`, `±1000`,
  plus 20 seeded random permutations. Pass rule: `R^2(shift 0)` exceeds the max over every
  shifted/permuted alignment by a pre-registered margin (margin value: planner).
- D9-08 — On failure, SEARCH for the true offset. Recorded as a post-hoc, data-chosen step: its
  own amendment document and freeze commit; `09-FINDINGS.md` states which offset was used and
  that it was found rather than assumed. Reversibility: one-way.

**Replication verdict rule**
- D9-09 — The verdict statistic is his exact 3-control rank-partial Spearman:
  `rho(curvature, local OOF R^2 | log_knn_radius, local_label_variance, local_evaluation_count)`,
  ranks residualized on ranks, as `inference.py`'s `associate`/`control_matrix`. Raw `rho` and
  07.1's within-density-stratum permutation partial are reported beside it, non-gating. Both
  nulls reported unconditionally.
- D9-10 — "Replicates" = controlled partial is NEGATIVE and clears its own Freedman–Lane rank
  permutation null with FWER (max-|rho| envelope) across `d`, at one or more `d`. No magnitude
  threshold. Magnitude printed beside his `-0.240` with both bootstrap bands (his B=2000 paired
  anchor resamples; ours the same). Reversibility: one-way.
- D9-11 — `||H_tan||` (sphere-tangential mean curvature) carries the verdict; `||H||` is in the
  same table, non-gating. Decomposition machinery: `08_radial_curvature_decomposition_run.py`.
  Reversibility: one-way.
- D9-12 — `D_SWEEP = (16, 20, 25, 32)`. Phase 9 declares its own sweep constant in its own module;
  Phase 7's frozen `D_SWEEP` is not edited. Fit-quality read-out (`var_explained`, `cond(g)`)
  required at every `d` including 16. Fixture fidelity at `d=16` currently unmeasured; whether to
  measure it is Claude's discretion. His `d=12` `+0.143` lies outside the sweep, non-comparable.

**Probe and control construction**
- D9-13 — Ridge `alpha = 100`, fixed. No grid, no selection. Five-fold OOF: fold `f` predicted
  from weights fit on the other four. Local `R^2`, MSE, SST per anchor over its 2048 neighbours
  with finite `y`, `ŷ`, uniform weights, exactly §10. `linear_probe.fit_probe` / `predict_probe`
  reused (sealed; import only).
- D9-14 — Positive control: curvature-side rank-plant on the pattern of
  `crossmodal_curvature.plant_positive_control`. Real local `R^2` kept; synthetic curvature array
  spread-matched to the realized `||H_tan||` range, planted at a grid of target `rho` by
  bisection, pushed through the identical 3-control partial and null. Reports the smallest
  cleared target.
- D9-15 — Shuffled-label calibration: shuffle the label vector across rows, run the entire
  pipeline, read the false-positive rate. Shape is Claude's discretion.
- D9-16 — Secondary labels `photo_z`, `smooth_fraction`, `stellar_mass` reported, non-gating.
  Same pipeline per label, own nulls; `mag_r` alone decides. `stellar_mass` needs a missing-value
  mask (his record: 79,490 of 86,471 labelled). `sfr` excluded as underpowered.
- D9-17 — Seeds, two waves. Wave A: single `TORCH_INIT_SEED` across all four `d`. Wave B,
  conditional: three seeds at every `d` where wave-A verdict fired; unanimity 3-of-3 or
  `SPLIT ACROSS SEEDS`; seeds never pooled. Reversibility: one-way.

**Inherited, non-negotiable**
- D9-18 — Freeze before any number (D7-06/D8-22). Every constant committed in one freeze commit,
  git-ancestry-proved to precede every measured value. Additive only; no sealed module mutated on
  import; `src/effdim/` untouched. Verdict sentence names instrument and `d` beside his, prints
  the per-`d` table. Report `p < 1/(B+1)`, never `p = 0`.

### Claude's Discretion
- The alignment margin's numeric value (D9-07) and how a found offset is ratified (D9-08).
- Whether to measure fixture fidelity at `d=16` before the Physics run (D9-12).
- Shuffled-label calibration shape: repeats, labels shuffled globally or local `R^2` shuffled
  across anchors (D9-15).
- Permutation count (his `B = 10^4`) and bootstrap count (his `B = 2000`) — inherit or reduce
  with a measured cost table, the 08-PREREGISTRATION-AMENDMENT-01 pattern.
- The OOF fold seed, the anchor-draw seed, the density/radius `k` for controls (his
  `log_knn_radius` is the radius of the same 2048-neighbourhood).
- Positive-control target-`rho` grid values.
- Module naming, runner layout, wave decomposition, runtime budget.
- How 07.1's stratified null attaches: strata on `log_knn_radius` rank versus
  `curvature_probe.local_density_weights` — either defensible; state which.

### Deferred Ideas (OUT OF SCOPE)
- Per-anchor instrument comparison against his `K_H` (needs his `selection.npz`, not on branch).
- `k` sensitivity grid on our data.
- `R^2`-side positive control.
- Re-embedding galaxies with the ViT-B checkpoint as an alignment proof.
- Fixture fidelity at `d=32`.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| D9-01 | Full 86,471-row Physics sample, no subsample | Data section: confirmed 86,471 rows, single column `vit_base_galaxies`, 245 MB parquet |
| D9-02 | k=2048 fixed local-R² neighbourhood | Code Examples: k-NN query pattern; cost model shows this is cheap (query-only, not full graph) |
| D9-03 | 512 seeded anchors | `subsample.draw_row_indices` pattern reusable for seeded uniform draw |
| D9-04 | Anchors from AE holdout rows only | `crossmodal_curvature.split_indices(n, SPLIT_SEED, HOLDOUT_FRACTION)` verified; `round(86471*0.2)=17294` holdout rows |
| D9-05/06/07/08 | Row-alignment statistical proof | Architecture Patterns §Row-Alignment Proof; Common Pitfalls §1 (labels are NOT on the public `main` revision) |
| D9-09 | 3-control rank-partial Spearman | `cross_split_curvature.partial_spearman` verified to implement this exact statistic already, generalized to multi-column controls |
| D9-10 | Verdict rule: negative + clears FWER null at ≥1 d | Architecture Patterns §Verdict Rule Template; existing `apply_verdict`/`VERDICT_VALUES` pattern in `crossmodal_curvature.py` |
| D9-11 | `‖H_tan‖` carries verdict | Code Examples §Radial/Tangential Decomposition, verified exact formula from `08_radial_curvature_decomposition_run.py` |
| D9-12 | D_SWEEP=(16,20,25,32), fit-quality read-out | Cost model; fixture-fidelity note (`d=16` measurable with existing runner, `d=32` still blocked) |
| D9-13 | Fixed-alpha OOF ridge | `linear_probe.fit_probe`'s `RidgeCV(alphas=alpha_grid)` — pass a one-element tuple `(100.0,)` for "fixed, no selection" |
| D9-14 | Curvature-side positive control | `crossmodal_curvature.plant_positive_control` mechanism documented; needs adaptation (target statistic differs) |
| D9-15 | Shuffled-label calibration | Common Pitfalls; Discretion — shape recommendation given |
| D9-16 | Secondary labels, non-gating | Data section: verified raw-column candidates and measured missingness per label |
| D9-17 | Two-wave, unanimity, never pool | `05-03-DECISION.md` precedent; `apply_seed_verdict` pattern in `density_stratified_null.py` |
| D9-18 | Freeze-before-number discipline | Architecture Patterns §Freeze Discipline; verified `assert_preregistered`/ancestor-gate pattern in two prior phases |
</phase_requirements>

## Summary

Phase 9 replicates a specific published finding using this milestone's own curvature instrument
in place of the colleague's nested-PCA quadratic-chart estimator. The statistical machinery this
phase needs — the rank-partial-Spearman-with-controls statistic, a Freedman–Lane permutation
null, a within-stratum permutation null, seeded OOF ridge fitting, and the freeze/verdict/
git-ancestry discipline — already exists in `notebooks/pu_manifold/` almost verbatim, built by
Phases 5, 7 and 07.1 for structurally identical problems. `cross_split_curvature.partial_spearman`
in particular already computes the exact 3-control statistic D9-09 asks for (rank-transform every
column, residualize by least squares, Pearson the residuals) — it should be reused directly rather
than reimplemented from the colleague's `inference.py`. The main new engineering is a Physics data
loader (no existing code loads this config), the row-alignment statistical proof (D9-05..08, novel
to this phase), the 5-fold explicit OOF wrapper around `fit_probe`/`predict_probe` (the sealed
functions are single-fit, not OOF), and a curvature-side positive-control adaptation targeting the
partial-Spearman statistic rather than a raw Spearman.

One finding changes what CONTEXT.md assumed about the data path and must reach the planner before
any download happens: the public `Smith42/galaxies` dataset's default (`main`) revision test split
carries only an image and a `dr8_id` — **no** `mag_r`, `photo_z`, `smooth_fraction` or
`stellar_mass` columns exist there. The full photometric/morphological catalog only exists under
the `v2.0` git revision of that same HF repo (`load_dataset("Smith42/galaxies", revision="v2.0",
split="test")`), which is ~7.8 GB across 16 parquet shards (dominated by image bytes; the label
columns alone are a small fraction of that and can be fetched with column projection without
downloading images). This confirms and extends what D9-05 already anticipated — the colleague's
own labels-build script is genuinely absent from his branch — and it also resolves the exact
raw-column mapping empirically: measured missingness on a 5,405-row shard shows `mag_r_desi` is
the fully-populated DESI photometry column (`mag_r` itself is 93% missing, an NSA cross-match
artifact), `mass_med_photoz` is 93% populated and the best `stellar_mass` candidate (`elpetro_mass_
log` is 93% missing, matching `mag_r`'s missing set), and `total_sfr_median` is only 8.7% populated
— independently corroborating the colleague's own "sfr excluded as underpowered" note. See Common
Pitfalls §1 and the Data section for the full evidence and the one remaining open call (whether
`mass_med_photoz` or another mass column is the intended `stellar_mass`).

**Primary recommendation:** Reuse `cross_split_curvature.partial_spearman` for the verdict
statistic, wrap `linear_probe.fit_probe`/`predict_probe` in a hand-written 5-fold OOF loop (do not
rely on `RidgeCV`'s internal CV for the OOF split — pass a one-element `alpha_grid=(100.0,)` to
pin `alpha=100` with no selection, but split folds manually so "fold f predicted from weights fit
on the other four" is literal), load `Smith42/galaxies` at `revision="v2.0"` (never `main`), and
build the row-alignment statistical proof (D9-06/07) as the very first executable artifact, before
any label value is used for anything else.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Physics embedding load (`pu-embeddings`) | Data/Storage (HF dataset cache) | — | Read-only external parquet, cached under `~/.cache/huggingface` |
| Label catalog load (`Smith42/galaxies`) | Data/Storage (HF dataset cache) | — | Separate repo, separate revision; joined by row-index convention only |
| Row-alignment proof | Notebook/analysis (`pu_manifold` module + diagnostics runner) | — | Pure numpy/scipy computation over cached arrays |
| AE fit + curvature field | Notebook/analysis (`cae.py`, `decoder_curvature.py`) | — | Sealed instrument, CPU-bound torch training, no service boundary |
| OOF ridge probe + local R² | Notebook/analysis (`linear_probe.py` wrapped) | — | In-process sklearn fit, no service boundary |
| Partial-Spearman + nulls | Notebook/analysis (`cross_split_curvature.py`, `density_stratified_null.py`) | — | Pure statistics over the 512×N anchor table |
| Freeze/verdict/record | Notebook/analysis (new `pu_manifold` module, `notebooks/.cache/09_*.jsonl`) | — | Mirrors Phase 7/8's runner shape exactly |

There is no browser, server, or CDN tier in this milestone — everything is a single-process,
CPU-bound analysis pipeline reading from and writing to `notebooks/.cache/`. This map exists only
to confirm that fact explicitly: no capability in this phase should be assigned to a tier that
does not exist in this project.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `datasets` (HF) | 5.0.1 (already in `.venv`) [VERIFIED: `.venv/bin/python3 -c "import datasets"`] | Loads `UniverseTBD/pu-embeddings` and `Smith42/galaxies` parquet configs | Already the project's sole loader for `pu-embeddings`, per `subsample.py` |
| `huggingface_hub` | 1.25.1 (already in `.venv`) [VERIFIED] | Revision/file introspection (`dataset_info`, `list_repo_refs`) before a full download | Needed to confirm `v2.0` revision and file sizes before committing to a download plan |
| `torch` | already in `.venv` (Phase 7/8 dependency) | `cae.PlainAutoEncoder`, `train_plain_ae` | Sealed instrument, no substitute |
| `scikit-learn` | already in `.venv` (Phase 5 dependency, `RidgeCV`, `NearestNeighbors`) | OOF ridge probe (`linear_probe.fit_probe`), k-NN queries for the 2048-neighbourhood | Sealed via `linear_probe.py`; k-NN is new but uses the same library already vendored |
| `scipy` | already in `.venv` | `rankdata`, `spearmanr`, permutation machinery | Used throughout `pu_manifold` |
| `pyarrow` | already in `.venv` (transitive via `datasets`) | Column-projected parquet reads (see Data section) to avoid downloading image bytes | Confirmed installed and working via `pq.read_table(url, columns=[...])` over `hf://` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandas` | already in `.venv` | Anchor table assembly, per-`d`/per-label wide tables mirroring the colleague's `wide` DataFrame shape | Any table with one row per anchor, columns per `d` or per label |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual 5-fold OOF loop around `fit_probe` | `sklearn.model_selection.cross_val_predict` | `cross_val_predict` with a plain `Ridge(alpha=100)` estimator does exactly "fold f predicted from weights fit on the other four" in one call and avoids reusing `RidgeCV`'s internal-CV machinery for a use it was not built for; **this is arguably simpler than wrapping `fit_probe`** and is worth the planner's explicit consideration even though D9-13 names `fit_probe`/`predict_probe` as "the existing implementations to reuse" — either is defensible, but `cross_val_predict(Ridge(alpha=100.0), X, y, cv=5)` is fewer lines and cannot silently drift from the "predicted from the other four folds" contract |
| Re-deriving the row-alignment proof from scratch | Reusing `subsample.assert_alignment`'s permuted-null z-score pattern (built for a different alignment question — pu-embeddings' own internal hsc/legacysurvey pairing) | The existing `assert_alignment` proves alignment between two *embedding* columns via a structural + permutation check; it is not a template for embedding→external-label alignment (D9-06 is a different statistic, an OOF R² comparison across row shifts). Do not import it hoping it transfers — it doesn't test the same thing. |

**Installation:** No new packages need installing; `datasets`, `huggingface_hub`, `torch`,
`scikit-learn`, `scipy`, `pyarrow`, `pandas` are all already present in `.venv` per direct
`import` checks against `.venv/bin/python3` on 2026-09-02.

**Version verification:** [VERIFIED] via `.venv/bin/python3 -c "import datasets; print(datasets.
__version__)"` → `5.0.1`; `import huggingface_hub; print(huggingface_hub.__version__)` → `1.25.1`.
No `pyproject.toml`/`requirements` change needed (CLAUDE.md: notebook-only milestone).

## Package Legitimacy Audit

No new external packages are introduced by this phase. Every library used is already installed in
`.venv` and was verified present by direct `import` in this research session (see Standard Stack).
The Package Legitimacy Gate does not apply — nothing new is being installed.

**Packages removed due to [SLOP] verdict:** none — no packages evaluated.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
UniverseTBD/pu-embeddings              Smith42/galaxies @ revision v2.0
  config physics_vit_base_test           split=test (NOT the default "main" revision)
  86,471 rows, col vit_base_galaxies     86,471 rows, ~150 catalog columns + image
        │ (768-D embedding)                    │ (mag_r_desi, photo_z, smooth-or-featured_*,
        │                                       │  mass_med_photoz, total_sfr_median, ...)
        ▼                                       ▼
  ┌─────────────────────────┐         ┌──────────────────────────┐
  │ Physics loader (NEW)     │         │ Label loader (NEW)        │
  │ column-projected parquet │         │ column-projected parquet  │
  │ read, no image bytes     │         │ read, no image bytes      │
  └────────────┬─────────────┘         └────────────┬──────────────┘
               │  row index = "sample_id" convention (assumed, then PROVEN)
               ▼                                     ▼
      ┌─────────────────────────────────────────────────────┐
      │  D9-05..08 Row-alignment statistical proof           │
      │  OOF ridge R²(shift 0) vs R²(shift k) for k in       │
      │  frozen shift set -- gates everything downstream     │
      └───────────────────────┬───────────────────────────────┘
                               │ PASS (assumed shift) or FOUND (adopted offset)
                               ▼
      ┌───────────────────────────────────────────────────────────────┐
      │  AE fit (cae.PlainAutoEncoder / train_plain_ae) per d in       │
      │  D_SWEEP=(16,20,25,32), holdout split via split_indices        │
      └───────────┬───────────────────────────────┬────────────────────┘
                  │ z = model.encode(x)            │ (train/holdout split, D9-04)
                  ▼                                 │
      ┌────────────────────────────┐                │
      │ decoder_curvature.          │                │ anchors = 512 seeded draw
      │ plain_decoder_curvature      │                │ FROM HOLDOUT ROWS ONLY
      │ at the 512 anchor z's only  │                ▼
      └──────────────┬──────────────┘      ┌──────────────────────────┐
                     │ H_vec, cond(g)        │ k=2048 NN query per      │
                     ▼                       │ anchor over ALL 86,471   │
      ┌────────────────────────────┐         │ rows (independent of     │
      │ 08_radial_curvature_        │         │ AE split)                │
      │ decomposition-style split:  │         └─────────────┬─────────────┘
      │ H_rad = <H,u>, H_tan =      │                       │
      │ H - H_rad*u  (u = decoder   │                       ▼
      │ image / its norm)           │         ┌──────────────────────────┐
      └──────────────┬───────────────┘         │ 5-fold OOF ridge probe   │
                     │ ‖H‖, ‖H_tan‖ per anchor  │ (alpha=100 fixed) over   │
                     │                          │ ALL 86,471 rows -> ŷ     │
                     │                          │ then per-anchor local    │
                     │                          │ R²/MSE/SST over its      │
                     │                          │ 2048 neighbours          │
                     │                          └─────────────┬─────────────┘
                     └─────────────┬──────────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────────┐
                  │ cross_split_curvature.partial_spearman   │
                  │ (curvature, local_R2 | log_knn_radius,   │
                  │  local_label_variance, local_eval_count) │
                  └─────────────┬─────────────────────────────┘
                                │
              ┌─────────────────┼──────────────────────┐
              ▼                 ▼                       ▼
     Freedman-Lane FWER   07.1-style stratified   Positive control +
     null (colleague's     permutation null        shuffled-label
     inference.py pattern) (radius-rank strata)     calibration
              │                 │                       │
              └────────┬────────┴───────────┬────────────┘
                       ▼                     ▼
              per-d verdict table    09-FINDINGS.md + notebook
              (D9-10 rule, two-wave  (caveat-bearing verdict,
              seed unanimity)        D8-21 pattern)
```

### Recommended Project Structure
```
notebooks/pu_manifold/
├── physics_labels.py        # NEW: column-projected loaders for both HF datasets, row-
│                             #   alignment proof functions, canonical-label column mapping
├── physics_curvature_probe.py  # NEW: Phase 9's own frozen constants block (D_SWEEP, alpha,
│                             #   shift set, anchor/OOF seeds), assert_preregistered(),
│                             #   verdict rule, positive-control adaptation, OOF wrapper
notebooks/diagnostics/
├── 09_row_alignment_proof_run.py   # NEW: D9-05..08, runs FIRST, gates everything else
├── 09_physics_curvature_run.py     # NEW: --mode smoke|dsweep|positive-control, mirrors
│                                   #   07_crossmodal_curvature_run.py's shape exactly
notebooks/
├── 09_physics_replication_report.ipynb   # reporting notebook, committed with outputs
.planning/phases/09-.../
├── 09-PREREGISTRATION.md    # frozen constants, git-ancestry-proved (D9-18)
├── 09-FINDINGS.md
```

### Pattern 1: Freeze-before-any-number, git-ancestry-proved
**What:** A module-level constants block committed in one commit, with a strict-ancestor gate
(`git merge-base --is-ancestor` AND `git rev-list --count >= 1`) checked by every code path that
produces a number.
**When to use:** Always, for this phase (D9-18) — this is the fourth phase in this milestone to
use the identical pattern (Phase 7, 07.1, Phase 8 all did).
**Example (verified, from `crossmodal_curvature.py`):**
```python
# Source: notebooks/pu_manifold/crossmodal_curvature.py:300-340 (assert_preregistered)
def assert_preregistered() -> None:
    g = globals()
    missing = []
    for name in _REQUIRED_CONSTANTS:
        if name not in g:
            missing.append(f"{name} (absent)")
            continue
        value = g[name]
        if value is None:
            missing.append(f"{name} (None)")
        elif isinstance(value, (tuple, list)) and len(value) == 0:
            missing.append(f"{name} (empty sequence)")
    if missing:
        raise RuntimeError(
            "...Phase 7 is not frozen -- the following pre-registered constants are unset: "
            + ", ".join(missing)
        )
```
```python
# Source: notebooks/diagnostics/07_crossmodal_curvature_run.py:407-459 (_strict_ancestor_or_exit)
is_ancestor = subprocess.run(
    ["git", "merge-base", "--is-ancestor", freeze_commit, "HEAD"], cwd=..., capture_output=True
)
count = int(subprocess.run(
    ["git", "rev-list", "--count", f"{freeze_commit}..HEAD"], cwd=..., capture_output=True
).stdout.strip())
if is_ancestor.returncode != 0 or count < 1:
    raise SystemExit("ERROR: --freeze-commit is not a STRICT git ancestor of HEAD.")
```

### Pattern 2: The exact verdict statistic already exists — reuse it, don't reimplement
**What:** `cross_split_curvature.partial_spearman(x, y, controls=None)` rank-transforms `x`, `y`
and every column of `controls`, residualizes `x` and `y` against the rank-transformed controls
(with intercept) by least squares, and returns the Pearson correlation of the residuals.
**When to use:** For D9-09's exact statistic. `controls` accepts an `(n, c)` array, so passing
`np.column_stack([log_knn_radius, local_label_variance, local_evaluation_count])` reproduces the
colleague's `associate`/`control_matrix` 3-control partial in one call — [VERIFIED: this function's
docstring names the colleague's own `report.py` quote verbatim and states the express purpose is
comparability with his `-0.240`/`-0.412` pair].
**Example:**
```python
# Source: notebooks/pu_manifold/cross_split_curvature.py:232-262 (partial_spearman)
from scipy.stats import rankdata
def partial_spearman(x, y, controls=None):
    rx, ry = rankdata(np.asarray(x, dtype=np.float64).ravel()), rankdata(np.asarray(y, dtype=np.float64).ravel())
    if controls is None:
        return float(np.corrcoef(rx, ry)[0, 1])
    Z = np.column_stack([rankdata(controls[:, j]) for j in range(controls.shape[1])])
    A = np.column_stack([np.ones(len(rx)), Z])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])
```
Do **not** re-derive the colleague's `partial()` closure from `09-COLLEAGUE-REANALYSIS.md`'s
reproduction script for the phase's own production statistic — that script exists only to
reproduce his frozen numbers for the record, and this sealed house function already generalizes
it (multi-column `controls`, same math, tested).

### Pattern 3: Radial/sphere-tangential curvature decomposition
**What:** Because `subsample.l2_normalize` puts every embedding row on the unit sphere, the mean
curvature vector of a `d`-dimensional submanifold of that sphere carries an exact `-d` radial
component under this milestone's `H = tr_g(II)` convention — a constant that says nothing about
manifold shape but enters `||H||` in full. `08-DIAGNOSTICS.md` §2 measured removing it collapsing
a partial 2.8x at `d=25` and flipping its sign at `d=32`, which is exactly why D9-11 requires
`||H_tan||` to carry the verdict.
**When to use:** Compute both `||H||` and `||H_tan||` for every anchor, every `d`; report both,
`||H_tan||` gates.
**Example (verified, exact formula in production use today):**
```python
# Source: notebooks/diagnostics/08_radial_curvature_decomposition_run.py:114-127 (decompose)
def decompose(H_vec, image):
    """image = model.decode(z), the decoder's point on the manifold (NOT L2-renormalized)."""
    img_norm = np.linalg.norm(image, axis=1)
    u = image / img_norm[:, None]                       # outward radial direction
    H_rad = np.einsum("ij,ij->i", H_vec, u)              # signed, ~ -d for a good fit
    H_tan = H_vec - H_rad[:, None] * u                   # residual after removing the radial part
    return H_rad, np.linalg.norm(H_tan, axis=1), img_norm
```
This requires keeping `H_vec` (the full curvature vector, not just its norm) and the decoder
image `model.decode(z)` — `decoder_curvature.plain_decoder_curvature` returns `H_vec` under key
`"H_vec"`; the norm alone (what most callers keep) is insufficient for this decomposition.

### Pattern 4: Freedman–Lane rank-space permutation (colleague's construction, reproduce exactly)
**What:** Under the null that curvature carries no information about the outcome beyond the
controls, permute the *residual* of the outcome's ranks on the controls' ranks, not the raw
outcome — this preserves the control relationship under the null rather than destroying it.
**When to use:** For the controlled-partial permutation null in D9-09/D9-10, to match his
methodology exactly (§11 of `METHODS_FOR_PAPER.md`, [CITED: `origin/curvature-experiments:paper/
curvature_neurreps/audit_outputs/submission_validation/METHODS_FOR_PAPER.md`]).
**Example (verified from his branch):**
```python
# Source: origin/curvature-experiments:experiments/geometry/physics_curvature_probe_rank_sweep/inference.py:58-73
def freedman_lane_y(y, Z, rng):
    from scipy.stats import rankdata
    m = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    y2 = y.copy()
    yr = rankdata(y[m]).astype(np.float64)
    Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(int(m.sum())), Zr])
    fit = A @ np.linalg.lstsq(A, yr, rcond=None)[0]
    resid = yr - fit
    y2[m] = fit + rng.permutation(resid)          # permute residuals, add back the fit
    return y2
```
FWER across `d`: take `max_d |controlled_rho_permuted(d)|` per permutation draw, compare the
observed `max_d |controlled_rho_observed(d)|` against that envelope's distribution — [VERIFIED,
`permutation_curves`'s `tmax`/`p_global` logic, same source file lines 76-130].

### Pattern 5: Within-density(radius)-stratum permutation null (07.1's house construction)
**What:** `density_strata(field, n_strata)` assigns equal-count quantile bins by stable-sort rank;
a permutation loop then shuffles both the curvature array and the outcome array *independently
within each stratum*, preserving the marginal relationship with the stratifying field while
breaking the curvature-outcome link.
**When to use:** The 07.1-style stratified null D9-09 requires beside the colleague's Freedman–
Lane null. The **existing** `density_stratified_null.stratified_partial_null` is built for a
*single*-control partial (it calls `cross_split_curvature.partial_spearman(h, m, controls=
density)` with one column). Phase 9 needs the 3-control partial as its observed statistic, so this
sealed function cannot be called unmodified for the observed-statistic computation — but its
*stratification and permutation loop* (`density_strata`, the per-stratum independent-permutation
pattern) is directly reusable: write a **new, phase-9-owned function** that reuses
`density_strata` for binning (on `log_knn_radius` rank per the discretion note) and passes the
full 3-column `controls` matrix to `cross_split_curvature.partial_spearman` inside the loop. Do
not edit `density_stratified_null.py` to generalize it — additive only (D9-18).
**Example (verified structure to mirror, not to call unmodified):**
```python
# Source: notebooks/pu_manifold/density_stratified_null.py:466-482 (density_strata)
def density_strata(density, n_strata):
    order = np.argsort(np.asarray(density, dtype=np.float64).ravel(), kind="stable")
    n = order.shape[0]; bin_size = n // n_strata
    strata = np.empty(n, dtype=int)
    for i in range(n_strata):
        lo, hi = i * bin_size, (i + 1) * bin_size if i < n_strata - 1 else n
        strata[order[lo:hi]] = i
    return strata
```

### Pattern 6: Seeded, sealed-function-reusing OOF ridge (new wrapper needed)
**What:** `linear_probe.fit_probe(X_train, Y_train, alpha_grid, alpha_per_target, fit_intercept)`
wraps `sklearn.linear_model.RidgeCV` — it is a single-fit function, not an OOF loop. D9-13's
"fold f predicted from weights fit on the other four" needs an explicit 5-fold wrapper.
**When to use:** For the OOF ridge probe. Two defensible implementations:
1. **Reuse `fit_probe`/`predict_probe` literally** (matches D9-13's naming): call `fit_probe(X[
   train_idx], Y[train_idx].reshape(-1,1), alpha_grid=(100.0,), alpha_per_target=False,
   fit_intercept=True)` per fold, `predict_probe(fit, X[test_idx])` for the held-out predictions.
   A one-element `alpha_grid` makes `RidgeCV` degenerate to a fixed-alpha fit (no real selection
   happens with one candidate) — this satisfies "no grid, no selection" while still routing
   through the sealed function.
2. **Simpler, considered in Standard Stack's Alternatives table:** `sklearn.model_selection.
   cross_val_predict(Ridge(alpha=100.0), X, y, cv=KFold(5, shuffle=True, random_state=SEED))` in
   one call. Both are correct; (1) satisfies D9-13's literal text more closely, (2) is fewer lines
   and cannot silently diverge from "predicted from the other four folds only." Flag this choice
   for the planner rather than picking silently, since D9-13 names `fit_probe`/`predict_probe`
   explicitly as "the existing implementations to reuse."
**Example:**
```python
# Sealed signature, verified: notebooks/pu_manifold/linear_probe.py:412-460
def fit_probe(X_train, Y_train, alpha_grid, alpha_per_target, fit_intercept):
    """Wraps sklearn.linear_model.RidgeCV(alphas=alpha_grid, alpha_per_target=alpha_per_target,
    fit_intercept=fit_intercept). X_train, Y_train must both be 2-D."""
def predict_probe(fit, X):
    """fit["estimator"].predict(X)"""
```

### Anti-Patterns to Avoid
- **Loading `Smith42/galaxies` at its default revision.** [VERIFIED, see Common Pitfalls §1] The
  `main` revision's test split has only `image` and `dr8_id` — no catalog columns exist to load.
  Every label-loading call must pass `revision="v2.0"` explicitly.
- **Downloading the full `Smith42/galaxies` test split as images+catalog.** At `v2.0` this is
  ~7.8 GB across 16 shards. Use `pyarrow.parquet.read_table(url, columns=[...])` (or `datasets`'
  column-selection equivalent) to fetch only the catalog columns needed — [VERIFIED: a
  column-projected read of 8 columns from one shard completed in ~43s without downloading image
  bytes].
- **Trusting `mag_r` as the raw column for the primary label.** [VERIFIED] The raw `mag_r` column
  is 93% missing in a sampled shard (an NSA-crossmatch artifact); `mag_r_desi` is 100% populated
  and is almost certainly what the colleague's own `mag_r_desi`-keyed labels.npz actually stored.
- **Computing curvature over the full 86,471-row cloud.** Only the 512 anchor points need
  curvature evaluated (`decoder_curvature.plain_decoder_curvature(model, z_anchors)`) — this is
  "seconds per `d`" per CONTEXT.md's own cost note; do not accidentally evaluate curvature at
  every row the way Phase 7's `FIELD_EVALUATED_ON` convention did (that convention is explicitly
  NOT inherited, D9-04).
- **Computing a full k-NN graph over all 86,471 rows.** D9-02's k=2048 neighbourhood is only ever
  needed *from* the 512 anchor points *into* the full 86,471-row pool — a k-NN *query* from 512
  points, not an all-pairs graph. `sklearn.neighbors.NearestNeighbors(n_neighbors=2048).fit(X_all)
  .kneighbors(X_anchors)` is the right shape; building a symmetric graph is unnecessary work.
- **Reimplementing the colleague's `associate`/`control_matrix`/`partial()` from scratch.** The
  sealed `cross_split_curvature.partial_spearman` already generalizes it. Reimplementing risks a
  silent numerical divergence from both his numbers and this milestone's own sealed statistic.
- **Editing `crossmodal_curvature.py`'s `D_SWEEP` to add 16.** D9-12 explicitly forbids this —
  Phase 9 declares its own sweep constant in its own module.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rank-partial correlation with arbitrary control count | A custom `partial()` closure | `cross_split_curvature.partial_spearman(x, y, controls=Z)` | Already implements the exact statistic, tested, and its docstring explicitly targets comparability with the colleague's `-0.240` |
| Freedman–Lane permutation | A from-scratch residual-permute loop | Adapt the colleague's own `freedman_lane_y` (his branch, verified above) — it's ~15 lines and correct; port it into Phase 9's own module rather than re-deriving the algorithm | Getting Freedman–Lane's residual-then-permute order wrong is a documented class of statistical bug; his implementation is already validated against his own frozen numbers |
| Git-ancestry freeze proof | A custom "is this commit before that one" check | `git merge-base --is-ancestor` AND `git rev-list --count >= 1`, exactly as `_strict_ancestor_or_exit` in `07_crossmodal_curvature_run.py` | A commit is its own ancestor — `--is-ancestor` alone under-constrains; this exact bug class is why the milestone's convention pairs both checks |
| Density/radius stratification | A custom quantile-binning function | `density_stratified_null.density_strata(field, n_strata)` | Stable-sort tie handling is a specified, tested requirement (`STRATIFICATION_RULE`), easy to get wrong with a naive `pd.qcut` |
| Radial/tangential curvature split | Re-deriving the sphere-projection algebra | `08_radial_curvature_decomposition_run.py`'s `decompose()` (11 lines, verified above) | Already measured and validated on this exact pipeline; re-deriving risks a sign error in `H_rad` |

**Key insight:** This phase's statistical surface area is almost entirely covered by code the
milestone already built for structurally identical problems (Phases 5, 7, 07.1). The genuinely
new work is the data-loading path (a Physics-specific loader, novel to this phase) and the
row-alignment proof (D9-05..08, explicitly novel — the colleague never built one). Everything
downstream of "the labels are correctly aligned to the embeddings" is largely assembly of
existing, tested primitives.

## Data Section — Verified Against Live Sources

### `UniverseTBD/pu-embeddings`, config `physics_vit_base_test`
[VERIFIED: `pyarrow.parquet.ParquetFile` schema read against `hf://datasets/UniverseTBD/pu-
embeddings/physics/vit_base_test.parquet`, 2026-09-02]
- 86,471 rows, single column `vit_base_galaxies: fixed_size_list<element: float>[768]`.
- File size 245,131,130 bytes (~245 MB) — a full download is cheap, unlike the label side.
- No `object_id`/row-id column, matching CONTEXT.md's "pu-embeddings carries no ids" note and
  `subsample.py`'s existing convention for the other pu-embeddings configs (positional join only).
- Not yet cached locally as of this research session (`~/.cache/huggingface/datasets/UniverseTBD
  ___pu-embeddings/` only holds `legacysurvey_dinov3_vitb16`).

### `Smith42/galaxies` — revision matters, and CONTEXT.md's assumption needs correcting
[VERIFIED, 2026-09-02, via `huggingface_hub.HfApi` and direct `pyarrow.parquet` schema/data reads]

- **`main` revision, `test` split** (2 shards, 86,471 rows total, ~9.8 GB): schema is
  `image: struct<bytes, path>`, `dr8_id: string`. **No label columns exist here at all.** This is
  the revision `load_dataset("Smith42/galaxies", split="test")` returns by default, and it cannot
  supply `mag_r`, `photo_z`, `smooth_fraction`, or `stellar_mass`.
- **`v2.0` revision, `test` split** (16 shards, confirmed row count not fully tallied — the
  colleague's own record states 86,471; each of the 16 shards sampled ranges ~5,300-5,450 rows,
  consistent with `~86,471 / 16 ≈ 5,404` average): schema carries ~150 columns including the full
  Galaxy Zoo DESI morphology vote fractions, multiple photometry systems, and multiple redshift/
  mass/SFR catalogs. **This is the revision that must be loaded**, via
  `datasets.load_dataset("Smith42/galaxies", revision="v2.0", split="test")` or (cheaper) direct
  column-projected `pyarrow.parquet.read_table(url, columns=[...])` per shard, avoiding the
  ~7.8 GB of image bytes entirely.
- Both revisions exist as named git refs on the HF repo (`branches: ['main', 'v2.0']`,
  `tags: []` — `v2.0` is a **branch**, not a tag, so `revision="v2.0"` is the correct
  `load_dataset`/`hf://` syntax).

### Raw-column-to-canonical-label mapping — measured, not assumed
[VERIFIED: missingness measured directly on `v2.0` test shard 0, n=5,405 rows, 2026-09-02]

| Canonical label (D9 name) | Candidate raw column | Valid / n=5405 | Recommendation |
|---|---|---|---|
| `mag_r` (primary) | `mag_r_desi` | 5405 / 5405 (100%) | **Use this.** |
| `mag_r` (primary) | `mag_r` (raw NSA-style) | 381 / 5405 (7%) | Do not use — this is almost certainly why the colleague's own npz key is `mag_r_desi`, not `mag_r` |
| `photo_z` | `photo_z` | 5021 / 5405 (93%) | Direct match, column name is unambiguous |
| `smooth_fraction` | `smooth-or-featured_smooth_fraction` | 5405 / 5405 (100%) | Direct semantic match to colleague's CANONICAL dict meaning ("Galaxy Zoo smooth-or-featured smooth vote fraction") |
| `stellar_mass` | `mass_med_photoz` | 5025 / 5405 (93%) | **Best candidate** — scale (values ~9-11) matches log-stellar-mass; missingness (~7%) is in the same order as the colleague's reported 79,490/86,471 (~8% missing) |
| `stellar_mass` | `elpetro_mass_log` | 381 / 5405 (7%) | Reject — same near-total-missing pattern as raw `mag_r`, an NSA-crossmatch subset far too small |
| `sfr` (excluded per D9-16, informational only) | `total_sfr_median` | 471 / 5405 (8.7%) | Not used this phase, but its extreme sparsity independently corroborates the colleague's own "sfr excluded as underpowered" note |

**Open call for the plan or execution phase:** `mass_med_photoz` is the strongest evidence-based
candidate for `stellar_mass` but is not a certainty — the colleague's exact labels.npz-build
script is absent from his branch (confirmed: `git grep` across the branch finds only *consumers*
of `vit_base_test_labels.npz`, never a generator), so there is no way to confirm his exact mapping
byte-for-byte. This is a genuinely new convention Phase 9 must document as its own (mirroring
D9-05's "the colleague's standard is a principle, not a method" framing) rather than presented as
a reproduction of his exact pipeline. Tag: **[ASSUMED]** pending either (a) a full-table
missingness check confirming the ~79,490/86,471 ratio at production scale, or (b) explicit
developer sign-off on the column choice at plan or execution time.

**Full-scale verification still needed before freeze:** the missingness/dtype findings above are
from ONE of 16 shards (~6.25% sample, n=5,405). A cheap full-scale check (project the same ~7
columns across all 16 shards) should run once during D9-06's data-loading step, which needs the
full label vector anyway — no extra download cost, just don't discard the intermediate counts.

## Common Pitfalls

### Pitfall 1: Assuming `Smith42/galaxies`'s default revision carries the labels
**What goes wrong:** `load_dataset("Smith42/galaxies", split="test")` (no `revision=`) returns
`image` + `dr8_id` only — every label lookup (`mag_r`, `photo_z`, etc.) raises a `KeyError`, or
worse, silently returns nothing if the loader is written defensively.
**Why it happens:** CONTEXT.md's canonical refs describe the dataset by name without naming a
revision, because the colleague's own code reads from a **locally cached parquet path**
(`data_hf/physics/vit_base_test_labels.npz`) that was itself built from some unknown, unrecorded
revision — the revision requirement is invisible from his code, only from a live HF query.
**How to avoid:** Always pass `revision="v2.0"` explicitly when loading `Smith42/galaxies` for
this phase; assert the loaded schema contains `mag_r_desi` (or whichever column D9-06's plan
settles on) before proceeding, so a future re-run against a hypothetical `v3.0` fails loudly
rather than silently loading an empty label set again.
**Warning signs:** Any `KeyError` naming a label column while loading `Smith42/galaxies`; a
"labelled row count" that reads exactly 0 or exactly `n` (both anomalous for a photometric survey).

### Pitfall 2: Reliability gate blind spot — his own instrument is unvalidated at D=768, k=2048
**What goes wrong:** Assuming the colleague's `-0.240` is itself a trustworthy ground truth to
match against, when his own reliability gate (`R_H`, split-half correlation) cannot detect a
shared bias between the two halves — `06-FINDINGS.md` already measured `R_H = 0.990` alongside a
true `rho = 0.469` against known ground truth on the Swiss roll, i.e., a near-perfect reliability
score coexisting with a mediocre accuracy score. His own `R_H` at d=16 medians 0.514 with 42% of
anchors below 0.5 — his instrument is not confidently reliable even by his own (blind-spot-prone)
metric.
**Why it happens:** Split-half reliability measures internal consistency, not correctness against
a known answer; the only known-answer check either side has run is Phase 7's `INSTRUMENT_
FIDELITY_RANGE = (0.53, 0.99)` at `d=20` on analytic fixtures, none of it at `d=16`.
**How to avoid:** Report both instruments' known limitations side by side in `09-FINDINGS.md`,
never present a sign-and-null match as proof either estimator is "correct" — only that they agree
or disagree on this one dataset.
**Warning signs:** A verdict sentence that reads "confirms his finding" rather than "reproduces
the same sign under a different, differently-validated instrument."

### Pitfall 3: Radial term dominating a spurious partial (already measured once)
**What goes wrong:** Reporting `||H||` as the headline curvature quantity, when `08-DIAGNOSTICS.
md` §2 already measured the sphere-radial term (`-d`, constant, carries no shape information)
collapsing a partial correlation 2.8x at `d=25` and flipping its sign at `d=32` when swapped for
`||H_tan||`.
**Why it happens:** `l2_normalize` places every embedding on the unit sphere, so `H = tr_g(II)`
always contains a `-d` radial component baked in by the normalization convention alone, unrelated
to manifold shape.
**How to avoid:** D9-11 already mandates `||H_tan||` carries the verdict — implement the
decomposition (Pattern 3 above) before computing any partial correlation, not as a post-hoc check.
**Warning signs:** A `d`-sweep where `spearman(||H||, ||H_tan||)` is high (>0.9, rank-agreement)
but the two partials disagree in sign or by more than 2x — Phase 8 saw exactly this and it is the
entire reason D9-11 exists.

### Pitfall 4: `RidgeCV`'s internal CV silently substituting for the required OOF structure
**What goes wrong:** Calling `fit_probe(X, Y, alpha_grid=(100.0,), ...)` once on the *entire*
86,471-row dataset and treating its predictions as "OOF" — `RidgeCV` fits on the training data it
is given and predicts on that same data; it does not by itself produce held-out predictions for
every row the way a genuine 5-fold OOF loop does.
**Why it happens:** `fit_probe`'s docstring name ("CV") plus its `RidgeCV` backing invites the
assumption that calling it once already delivers cross-validated, held-out predictions.
**How to avoid:** Explicitly split `KFold(n_splits=5, ...)`, call `fit_probe` on 4/5 of the rows,
`predict_probe` on the remaining 1/5, and only assemble the full-length OOF prediction array after
all 5 folds — never call `fit_probe` once on the full set and call it "OOF."
**Warning signs:** An OOF R² implausibly close to the probe's own training R² (near-zero gap is
suspicious for a 768-D-to-1-D ridge regression with real held-out folds).

### Pitfall 5: The positive control targets the wrong statistic if copied literally
**What goes wrong:** `crossmodal_curvature.plant_positive_control` plants a curvature-MKNN
relationship and validates it through `two_tailed_permutation_null` on a *raw* Spearman rho — but
D9-14 needs the plant validated through the identical **3-control partial** and its Freedman–Lane
null, not the raw statistic Phase 7's function was built for.
**Why it happens:** D9-14 explicitly says "on the pattern of `plant_positive_control`" — the
bisection-to-a-target-rho mechanism transfers, but the machinery it feeds does not: Phase 7's
function calls `two_tailed_permutation_null` internally, which is the wrong null for this phase.
**How to avoid:** Reuse the bisection *mechanism* (rank-transform, bisect a slope to hit a target
achieved rho, `_planted_array`'s binomial discretization) but re-target it at the achieved
**controlled partial** rho (via `cross_split_curvature.partial_spearman`) and push the result
through the Freedman–Lane null (Pattern 4), not `two_tailed_permutation_null`.
**Warning signs:** A "positive control cleared at target rho X" claim that was validated against
raw Spearman rather than the controlled partial the headline verdict actually uses.

### Pitfall 6: Fixture fidelity at `d=32` cannot be measured — do not attempt it, and say so plainly
**What goes wrong:** Trying to run `07_instrument_fixture_sweep_run.py --d 32` to fill the D9-12
fit-quality gap for `d=32`.
**Why it happens:** The small-ambient fixture arm's literal ambient width is `D=28`; graph
fixtures need local width `m = d+1 = 33` at `d=32`, and `rotate_and_pad` requires `D >= m` — this
raises `ValueError` by construction, already discovered and ratified as a limitation in
`HANDOFF-v1.1.md` §5.3 and named explicitly in the Deferred section of `09-CONTEXT.md`.
**How to avoid:** Do not attempt `d=32` fixture fidelity in this phase (Deferred, explicitly out
of scope). `d=16`, by contrast, IS measurable with the same runner unmodified (`m = 17 <= 28`) —
see Environment Availability below.
**Warning signs:** A plan task that tries to "fix" the `d=32` fixture-fidelity gap — this is
scoped out, not a bug to patch.

## Code Examples

### Column-projected parquet read (avoid downloading unneeded image bytes)
```python
# Verified working, 2026-09-02, ~43s wall for 8 columns / 5,405 rows from one v2.0 shard
import pyarrow.parquet as pq
url = "hf://datasets/Smith42/galaxies@v2.0/data/test-00000-of-00016.parquet"
table = pq.read_table(url, columns=["dr8_id", "mag_r_desi", "photo_z",
                                     "smooth-or-featured_smooth_fraction", "mass_med_photoz"])
```

### k-NN query FROM 512 anchors INTO the full 86,471-row pool (not a full graph)
```python
# Only the anchors need neighbourhoods -- do not build an n x n graph.
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2048).fit(X_all)          # X_all: (86471, 768)
distances, indices = nbrs.kneighbors(X_all[anchor_idx])        # anchor_idx: (512,)
log_knn_radius = np.log(distances[:, -1])                      # radius = farthest of the 2048
```

### Explicit 5-fold OOF, not a single `RidgeCV` call
```python
from sklearn.model_selection import KFold
from pu_manifold import linear_probe

oof_pred = np.full(n, np.nan)
kf = KFold(n_splits=5, shuffle=True, random_state=OOF_FOLD_SEED)
for train_idx, test_idx in kf.split(X):
    fit = linear_probe.fit_probe(
        X[train_idx], y[train_idx].reshape(-1, 1),
        alpha_grid=(100.0,), alpha_per_target=False, fit_intercept=True,
    )
    oof_pred[test_idx] = linear_probe.predict_probe(fit, X[test_idx]).ravel()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Colleague's nested-PCA quadratic-chart curvature estimator (`K_H_cross`) | This milestone's plain-autoencoder decoder curvature (`plain_decoder_curvature`) | Phase 9 (this phase) | Different instrument, same outcome/controls — the entire point of the replication |
| Phase 7's `FIELD_EVALUATED_ON = all_10000_rows_including_the_8000_training_rows` | Anchors restricted to AE holdout rows only (D9-04) | Phase 9 | Curvature measured only where the decoder never trained, closing a leakage concern Phase 7 did not address |
| Single global `D_SWEEP` per phase | Phase 9 declares its own `D_SWEEP=(16,20,25,32)`, does not edit Phase 7's | Phase 9 (D9-12) | Keeps Phase 7's frozen record untouched while adding a same-`d` (`d=16`) comparison point |

**Deprecated/outdated:**
- Treating `Smith42/galaxies`'s default revision as sufficient for any label-bearing task on this
  dataset — it is not, and this is not documented anywhere in this repo prior to this research.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `mass_med_photoz` is the correct raw column for the canonical `stellar_mass` label | Data Section | A wrong mass proxy changes D9-16's secondary-label numbers only (non-gating per D9-16), but would misdescribe the label in `09-FINDINGS.md`; low overall risk since `mag_r` (primary, gating) is unaffected |
| A2 | `Smith42/galaxies`'s `v2.0` test split has exactly 86,471 rows (only directly confirmed for the 16-shard row-count sum being consistent with, not equal to, the colleague's stated total; individual shard counts summed were not fully tallied in this session) | Data Section | If the true row count differs from `pu-embeddings`' 86,471, the positional-alignment assumption underlying D9-05..08 fails structurally before the statistical proof even runs — the proof itself would surface this (a shape mismatch), so risk is caught early, not silently |
| A3 | The `cross_val_predict`/`fit_probe`-wrapper choice for the OOF loop (Architecture Patterns, Pattern 6) has no numerical difference between the two implementations | Architecture Patterns | Low — both route through `sklearn.linear_model.Ridge`/`RidgeCV` at a fixed `alpha=100`; any difference would be floating-point noise, not a methodology difference |

**If this table is empty:** N/A — three assumptions recorded above; A1 is the one most likely to
need explicit developer confirmation before the freeze commit (D9-18), since it silently sets the
content of a reported-but-non-gating label.

## Open Questions

1. **Which raw column is `stellar_mass`?**
   - What we know: `mass_med_photoz` matches the value scale (log stellar mass, ~9-11) and has
     ~93% coverage, in the same order as the colleague's reported 79,490/86,471 (~92%) labelled
     rate; `elpetro_mass_log` is a poor match (93% missing).
   - What's unclear: whether the colleague's exact npz used `mass_med_photoz`, some combination
     with `mass_inf_photoz`/`mass_sup_photoz`, or a different column entirely — his build script
     is genuinely absent from the branch.
   - Recommendation: adopt `mass_med_photoz`, document it explicitly as Phase 9's own convention
     (not a byte-for-byte reproduction of his), and confirm the missing-row count matches
     ~79,490/86,471 at full scale as a sanity check before freezing.

2. **`fit_probe`/`predict_probe` OOF wrapper vs. `cross_val_predict`** — see Architecture Patterns
   Pattern 6. D9-13 names the sealed functions explicitly; `cross_val_predict` is simpler and
   arguably lower-risk. Recommend the planner pick one explicitly at plan time rather than leaving
   it to execution-time interpretation, since D9-13's wording could support either.

3. **How does `photo_z`'s own missingness (~7% in the sampled shard) interact with row-alignment
   proof (D9-06)?** The row-alignment proof is defined on `mag_r` only (D9-07), so this doesn't
   block the gating proof, but any downstream `photo_z` local-R² anchors with missing labels in
   their 2048-neighbourhood need the same finite-`y`/finite-`ŷ` masking §10 already specifies —
   confirm the existing local-R² formula's "finite `y`, `ŷ`" clause is applied per label, not just
   for `mag_r`.

## Environment Availability

**Execution host caveat (added to CONTEXT.md concurrently with this research, 2026-09-02):**
Phase 9 will NOT run its real numbers on the machine this research was performed on — compute is
either an SSH remote server or the colleague's box, undecided at planning time. Every row below
was verified on the *research* machine (`.venv` present, network reachable) and is offered as
evidence that the *approach* works, not as proof the *execution host* has these exact versions
pre-installed. The practical consequence for the plan: no hard-coded absolute paths (the existing
`cache.CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"` pattern already satisfies
this — it is relative to the repo, not a machine — so no new code needed there, only discipline
not to introduce a machine-specific path), the HF cache directory should be left at its
library-default (`~/.cache/huggingface`, overridable via `HF_HOME` if the execution host needs a
different disk), a `--mode smoke` dry run should be exercised on the research/dev machine before
handoff (this is already every runner's convention, per Pattern 1), and the runner's own
`DSWEEP_COST_MODEL_MINUTES`-style banner should print per-thread-count cost rather than an
absolute "on this machine" wall-clock, since the execution host's core count is unknown at
planning time.

| Dependency | Required By | Available (research machine) | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `datasets` | Loading both HF datasets | ✓ | 5.0.1 | — |
| `huggingface_hub` | Revision/schema introspection | ✓ | 1.25.1 | — |
| `torch`, `scikit-learn`, `scipy`, `pandas`, `pyarrow` | AE fit, OOF ridge, stats, tables | ✓ | already vendored | — |
| Network access to `huggingface.co` | Downloading `physics_vit_base_test` (~245 MB) and `Smith42/galaxies@v2.0` label columns (small, column-projected) | ✓ (verified live in this session) | — | — |
| `notebooks/diagnostics/07_instrument_fixture_sweep_run.py --d 16` | Optional fixture-fidelity measurement at `d=16` (D9-12 discretion) | ✓ runner already supports arbitrary `--d`, and `d=16`'s required local width `m=17 <= D=28` (unlike `d=32`'s `m=33 > 28`) | — | — |
| CUDA / GPU | Not required — this milestone runs CPU-only (`07-CONTEXT.md` §7's cost model is CPU-measured) | n/a | — | CPU (already the project's default; `--device` flag exists but unexercised) |
| Execution host itself (SSH remote / colleague's box) | Every real (non-smoke) number | **UNKNOWN at research time** — undecided per CONTEXT.md | — | Plan must not assume the research machine's `.venv`, cache contents, or core count transfer; a fresh-clone bootstrap step (dependency install + smoke-mode verification) belongs in the plan's first wave |

**Missing dependencies with no fallback:** the execution host's own environment is unverified and
cannot be verified until it is chosen — the plan should include an explicit "verify the target
host has `torch`/`datasets`/`scikit-learn` and can reach `huggingface.co`" step before any real
`d`-sweep fit, mirroring what this research session did on the research machine.

**Missing dependencies with fallback:** none identified beyond the host-selection question above.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest, existing suite at `notebooks/pu_manifold/tests/` (761+ tests as of the last recorded full run in STATE.md, 2026-08-28) |
| Config file | none dedicated — invoked directly as `pytest tests/` from `notebooks/pu_manifold/` |
| Quick run command | `pytest tests/test_cross_split_curvature.py tests/test_linear_probe.py tests/test_density_stratified_null.py -q` (targets the three modules this phase reuses/extends) |
| Full suite command | `.venv/bin/python3 -m pytest notebooks/pu_manifold/tests/ -q` (a full run took several minutes of CPU in this research session and was not exhaustively timed — budget accordingly, run once before the freeze commit and once before the final verdict, not per-task) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| D9-05..08 | Row-alignment proof: `R²(shift 0)` exceeds every shifted `R²` by the margin | unit + integration | `pytest notebooks/pu_manifold/tests/test_physics_labels.py -x` | ❌ Wave 0 (new module) |
| D9-09 | 3-control rank-partial Spearman matches `partial_spearman`'s existing tested behavior for multi-column controls | unit | `pytest notebooks/pu_manifold/tests/test_cross_split_curvature.py -x` | ✅ (already covers the general multi-control case) |
| D9-11 | `||H_tan||` decomposition matches `08_radial_curvature_decomposition_run.py`'s formula, `H_rad ≈ -d` | unit | `pytest notebooks/pu_manifold/tests/test_physics_curvature_probe.py -x` (new) | ❌ Wave 0 |
| D9-13 | OOF wrapper produces one prediction per row, no row predicted by a model trained on itself | unit | `pytest notebooks/pu_manifold/tests/test_physics_labels.py -k oof -x` | ❌ Wave 0 |
| D9-14 | Positive control recovers a planted target through the controlled-partial null, not the raw-rho null | unit | `pytest notebooks/pu_manifold/tests/test_physics_curvature_probe.py -k positive_control -x` | ❌ Wave 0 |
| D9-18 | `assert_preregistered()` raises before any constant exists; strict-ancestor gate rejects a non-strict commit | unit | `pytest notebooks/pu_manifold/tests/test_physics_curvature_probe.py -k preregistered -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** targeted module test file (`pytest tests/test_<new_module>.py -x`)
- **Per wave merge:** the three reused-module tests (`test_cross_split_curvature.py`,
  `test_linear_probe.py`, `test_density_stratified_null.py`) plus all new Phase 9 test files
- **Phase gate:** full `notebooks/pu_manifold/tests/` suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `notebooks/pu_manifold/tests/test_physics_labels.py` — covers D9-01, D9-05..08, D9-13, D9-16
- [ ] `notebooks/pu_manifold/tests/test_physics_curvature_probe.py` — covers D9-09..12, D9-14,
  D9-17, D9-18 (frozen constants, verdict rule, positive control, seed-unanimity combination)
- [ ] No framework install needed — pytest is already the project's test runner

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth surface anywhere in this notebook-only pipeline |
| V3 Session Management | No | No sessions |
| V4 Access Control | No | No multi-user access boundary |
| V5 Input Validation | Yes (narrow) | Data-shape/finiteness guards already present throughout `pu_manifold` (`fit_probe` raises on non-2D or non-finite input, `plant_positive_control` raises on non-finite/constant input) — new Phase 9 code should follow the same guard-first convention, not because of an adversarial input model but because malformed astronomical data (NaN-heavy columns, sentinel values like `stellar_mass`'s `-99.0`) is the realistic failure mode |
| V6 Cryptography | No | No secrets, no crypto surface; HF downloads are public, unauthenticated dataset reads |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via a cache-stem string | Tampering | `cache._assert_inside_cache`'s containment guard, already in place project-wide and inherited automatically by any `cache.npz_cache`/`cache.cache_path` call this phase makes |
| Sentinel-value contamination (`stellar_mass`'s documented `-99.0` missing-value code) silently entering a mean/variance/ridge fit as a valid numeric value | Tampering (data integrity, not adversarial) | Explicit `y[y == -99.0] = np.nan` masking before any statistic, mirrored from the colleague's own `load_catalog_label` pattern (verified above), applied consistently to whichever label uses a sentinel |

No network-facing surface, no user-input path, no persistence layer beyond the project's own
`notebooks/.cache/` — this phase's security profile mirrors every prior phase's own review
finding ("no network surface, no auth, no user-input path, no persistence layer" — `02.6-REVIEW.md`
STATE.md excerpt) with the sole realistic addition of untrusted-shape external data from two
public HF datasets, addressed by the finiteness/shape guards already conventional in this codebase.

## Sources

### Primary (HIGH confidence)
- `notebooks/pu_manifold/cross_split_curvature.py` — read directly, `partial_spearman` verified line-by-line
- `notebooks/pu_manifold/crossmodal_curvature.py` — read directly, constants block, `assert_preregistered`, `plant_positive_control`, `split_indices`, `VERDICT_VALUES`, `apply_verdict`
- `notebooks/pu_manifold/density_stratified_null.py` — read directly, `density_strata`, `stratified_partial_null`
- `notebooks/pu_manifold/linear_probe.py` — read directly, `fit_probe`, `predict_probe`, `per_point_residuals`
- `notebooks/pu_manifold/subsample.py` — read directly, `load_subsample`, `l2_normalize`, `draw_row_indices`
- `notebooks/pu_manifold/cae.py` — read directly, `PlainAutoEncoder`, `train_plain_ae`, `reconstruction_stats`
- `notebooks/pu_manifold/decoder_curvature.py` — read directly, `plain_decoder_curvature` signature confirmed
- `notebooks/diagnostics/07_crossmodal_curvature_run.py` — read directly, runner shape, `_strict_ancestor_or_exit`, `FREEZE_COMMIT_SHA` pattern, `DSWEEP_COST_MODEL_MINUTES`
- `notebooks/diagnostics/08_radial_curvature_decomposition_run.py` — read directly, `decompose()` formula, `fit_and_decompose()`
- `origin/curvature-experiments:experiments/geometry/physics_curvature_probe_rank_sweep/inference.py` — read directly via `git show`, `associate`, `control_matrix`, `freedman_lane_y`, `permutation_curves`, `paired_bootstrap_curves` all confirmed present at the cited paths
- `origin/curvature-experiments:paper/curvature_neurreps/audit_outputs/submission_validation/METHODS_FOR_PAPER.md` — read directly via `git show`, §9/§10/§11 confirmed
- `origin/curvature-experiments:experiments/geometry/physics_adaptive_dataset_curvature_probe/inventory.py` — read directly, `CANONICAL` label dict, `phys_labels` raw-key list, `vit_base_test_labels.npz` reference
- Live HF Hub queries (`huggingface_hub.HfApi.dataset_info`, `list_repo_refs`) and live `pyarrow.parquet` schema/data reads against `UniverseTBD/pu-embeddings` and `Smith42/galaxies` (both revisions) — all commands and outputs captured in this research session, 2026-09-02

### Secondary (MEDIUM confidence)
- `.planning/phases/07-curvature-conditioned-crossmodal-alignment/07-CONTEXT.md` §4/§5/§7 — cost
  model and fixture-fidelity numbers, cited as previously-measured facts, not re-verified this
  session (no new fit was run)
- `HANDOFF-v1.1.md` §5 — post-freeze diagnostics summary, cited for the radial-term and fidelity
  findings, consistent with the primary-source `08_radial_curvature_decomposition_run.py` code

### Tertiary (LOW confidence)
- `mass_med_photoz` as the correct `stellar_mass` raw column (Assumption A1) — strong circumstantial
  evidence (missingness rate, value scale) but not confirmed against the colleague's actual
  (absent) build script

## Metadata

**Confidence breakdown:**
- Standard stack / statistical machinery reuse: HIGH — every function cited was read directly from
  source in this session, not recalled from training data
- Data loading path: MEDIUM-HIGH — the `pu-embeddings` side is fully verified; the `Smith42/
  galaxies` label side required a corrective finding (the revision issue) not present anywhere in
  CONTEXT.md's own assumptions, and one column-mapping choice (`stellar_mass`) remains a strong
  recommendation rather than a certainty
- Architecture patterns / freeze discipline: HIGH — this is the fourth phase to use this exact
  pattern in this milestone, all four instances directly inspected
- Cost model: MEDIUM — inherited from Phase 7's measured numbers, scaled arithmetically to 86,471
  rows and four `d` values, not independently re-measured (a real AE fit at this scale was not run
  in this research session)

**Research date:** 2026-09-02
**Valid until:** ~14 days for the data-loading findings (HF dataset revisions can change without
notice; re-verify `Smith42/galaxies@v2.0`'s schema and row count immediately before the freeze
commit if more than a few days pass) — the code-reuse findings are stable for the life of the
milestone (they depend only on this repo's own sealed modules, which do not change).
