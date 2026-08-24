# Phase 5: Curvature-Conditioned Linear Decodability - Research

**Researched:** 2026-08-24
**Domain:** Crossmodal linear probing (ridge/OLS regression) conditioned on a decoder-side
differential-geometry field, inside a notebook-scoped pre-registration discipline.
**Confidence:** MEDIUM — the statistical design is HIGH confidence (standard, well-understood
regression machinery); the curvature-field extraction path required a direct code correction
against CONTEXT.md's own description (see Pitfall 1), which is now HIGH confidence because it
is grounded in the sealed module's own docstring and Phase 3's actual call site.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D5-01:** The probe predicts **the other modality**: a linear map `W : hsc[i] (768) ->
  legacysurvey[i] (768)` on frozen embeddings. The cached PU subsample carries no label
  column (`hsc`, `legacysurvey`, `hsc_norms`, `ls_norms`, `row_indices` only), so there is
  no external response variable available without sourcing and row-aligning new data. The
  other-modality target needs no new data and is the linear analogue of exactly what Phase 4
  measured with MKNN, which makes the two phases' results directly comparable.

- **D5-02:** **Fit globally, evaluate per-region.** One `W` is fit on a held-out training
  split drawn from the whole manifold; per-point residuals are computed on the test split and
  then bucketed by `||H||`. One model everywhere, so any bucket-to-bucket difference is a
  property of the data's local decodability rather than of fitting different models to
  different amounts of data. **Explicitly rejected:** per-region independent fits as the
  headline (reintroduces Phase 4's sample-size artifact).

- **D5-03:** The split field is **decoder-side** mean curvature — autodiff through the CAE
  chart decoder — not Phase 4's point-cloud `centroid_mean_curvature`.

- **D5-04:** **Pool the three cached CAE seeds into one averaged `||H||` field**
  (`03_converged_cae_pu_nc4_seed2026081{3,4,5}.pt`); the pooled field is the verdict field.
  One-way: changing the pooling rule after freeze invalidates the pre-registration.

- **D5-05:** **Inter-seed agreement is measured and reported as a diagnostic, and does not
  change the verdict.** Report pairwise `spearman` between the three seeds' `||H||` fields and
  between each seed and the pooled field.

- **D5-06:** `CURVATURE_CONVENTION = "trace"` — `H = tr_g(II)` unnormalized, a unit `d`-sphere
  giving `||H|| = d`. Non-negotiable.

- **D5-07:** Split on **`||H||` magnitude**, not Phase 4's direction sign. Bucket edges
  pre-registered (D5-09) before any PU probe number exists. A continuous
  `spearman(||H||_i, residual_i)` over all test points is reported alongside the bucketed
  comparison.

- **D5-08:** **Bucket sizes must be reported, and the bucketed comparison must be checked
  against a size-matched version.**

- **D5-09:** **Full pre-registration freeze, Phase 4 discipline.** Bucket edges, the probe
  protocol, the train/test split rule, the seed-pooling rule, the scoring metric, the seed,
  and the full `VERDICT_RULE` text frozen as named constants in committed source, plus a
  committed `05-PREREGISTRATION.md`, before any PU probe number exists. Git ancestry must be
  provable.

- **D5-10:** The runner must **refuse to compute a bucketed probe number** unless the
  pre-registration constants and the frozen `||H||` field artifact both already exist — a
  hard guard that raises rather than computing.

- **D5-11:** **Phase 5 runs with no known-answer anchor, and this is a deliberate choice.**
  The sealed `d=20` decoder row is `rank_spearman_rho = -0.015106571347065712`. A Swiss roll /
  low-`d` anchor stage was offered and declined. `05-FINDINGS.md` must state, in the phase's
  own words, that any relationship measured cannot be attributed to curvature by anything in
  this phase.

- **D5-12:** The CAE underlying the decoder **failed its own validity gate**
  (`CAE_VERDICT = FAIL`, Phase 02.2); Phase 3 runs on a deliberate override; Phase 03.1 found
  the metric repaired by `scale` but the ordering only partially and non-seed-consistently
  moved. Every Phase 5 number inherits that chain.

- **D5-13:** **The density confound is expected to be weaker here than in Phase 4, and this
  must be verified rather than assumed.** Re-measure `spearman(density, ||H||)` on the
  **decoder-side pooled field** — Phase 4's `-0.0273` was measured on the point-cloud field
  and does not automatically transfer.

### Claude's Discretion

- Train/test split fraction and cross-validation scheme (subject to being frozen at D5-09).
- Residual metric details: per-point squared error vs cosine vs normalized residual — planner
  chooses, then freezes. Whichever is chosen, `R^2` and a per-point residual must both be
  derivable so the bucketed and continuous versions share one underlying quantity.
- Number and placement of `||H||` buckets (tertiles vs quartiles), frozen at D5-09.
- Whether the probe is fit on raw or re-normalized embeddings, given both modalities are
  already L2-normalized upstream.

### Deferred Ideas (OUT OF SCOPE)

- Swiss roll / low-`d` anchor for the probe methodology — offered and declined (D5-11).
- Resolving the saddle-control fixture question (whether `rho = -0.015` is the decoder's
  failure or the fixture's) — its own phase, blocks nothing here.
- An external astrophysical label as the probe target — requires sourcing labels and row-
  alignment proof against `row_indices` first.
- Per-region independent probe fits at matched `n` as a sensitivity analysis — rejected as
  headline at D5-02, addable later as sensitivity-only without disturbing pre-registration.

</user_constraints>

<phase_requirements>
## Phase Requirements

No milestone-level REQ-IDs were minted for Phase 5 in `ROADMAP.md` (`Requirements: TBD`,
matching Phase 02.5's precedent of having no REQ-IDs and instead treating its CONTEXT.md
decisions as the de-facto requirement set). The following table uses `05-CONTEXT.md`'s
**D5-01..D5-13** as that de-facto requirement set, each mapped to the research support that
enables its implementation.

| ID | Description | Research Support |
|----|-------------|------------------|
| D5-01 | Probe `hsc -> legacysurvey`, both 768-d, from `load_pu_pair`'s resolved npz | § Code Examples 1; § Architecture Patterns (data loading) |
| D5-02 | Fit `W` globally on train split, bucket residuals on test split by `||H||` | § Standard Stack; § Architecture Patterns (probe fit); § Common Pitfalls 4 |
| D5-03 | Decoder-side `||H||` via CAE chart decoder autodiff | § Common Pitfalls 1 (critical correction to CONTEXT.md's own module citation); § Code Examples 2 |
| D5-04 | Pool three seeds' `||H||` fields into one averaged field | § Common Pitfalls 2 (52x scale mismatch across seeds — raw averaging is dominated by 2 of 3 seeds); recommendation in same section |
| D5-05 | Report inter-seed Spearman diagnostics, non-gating | § Code Examples 2 (reuse `spearmanr` pattern already used in `region_partition_mknn_run._spearman_report`) |
| D5-06 | `CURVATURE_CONVENTION = "trace"` | § Code Examples 2 (already asserted at import time by `decoder_curvature.py`/`chart_curvature.py`) |
| D5-07 | Split on magnitude; continuous Spearman alongside bucketed | § Residual Metric research answer; § Bucket Count research answer |
| D5-08 | Report bucket `n`; size-matched check | § Bucket Count research answer; direct citation of Phase 4's own `04-FINDINGS.md` §7 recommendation |
| D5-09 | Full pre-registration freeze, git-ancestry-provable | § Architecture Patterns (pre-registration template, verbatim from Phase 4) |
| D5-10 | Runner refuses to compute without pre-registration + frozen field | § Code Examples 3; § Validation Architecture (guard tests) |
| D5-11 | No known-answer anchor; state consequence honestly | § Assumptions Log; carried verbatim, not re-litigated |
| D5-12 | CAE gate FAIL inheritance stated | § Assumptions Log; carried verbatim, not re-litigated |
| D5-13 | Re-measure density confound on decoder-side pooled field | § Code Examples (density weights reuse); Phase 4's exact call pattern |

</phase_requirements>

<claude_md_constraints>
## Project Constraints (from CLAUDE.md)

- **Additive only.** Never delete or rewrite existing notebooks or runner scripts. Phase 5
  creates `notebooks/pu_manifold/linear_probe.py` and
  `notebooks/diagnostics/curvature_probe_decodability_run.py`, plus a new notebook. It reads
  Phase 4's runner for pattern and extends nothing in it.
- **Do not modify `src/effdim/`** during the v1.1 milestone.
- **Notebooks are committed with their outputs, executed end to end.**
- **KEEP THINGS SIMPLE FIRST.** Do not pre-build every edge case; add complexity only as
  problems are found.
- **Swiss roll sanity check rule.** Required for every *new* manifold-learning or
  representation-learning model that maps data to a lower-dimensional representation and
  back, or claims to recover manifold structure. **Determination for Phase 5 (see § Swiss
  Roll Rule Applicability below): the rule does NOT trigger for the linear probe itself.**
  The curvature *estimator* the field is read from is already covered by
  `notebooks/03_swiss_roll_chart_curvature_field_check.ipynb` (see that section for the
  reasoning, which corrects an imprecision in how `05-CONTEXT.md`'s `<code_context>` cites
  the estimator module).

</claude_md_constraints>

## Summary

Phase 5 fits one linear map `W: hsc(768) -> legacysurvey(768)` on frozen PU embeddings,
scores held-out per-point residuals, buckets those residuals by a decoder-side mean-curvature
field pooled from three CAE seeds, and tests — under a rule frozen before any PU number
exists — whether decodability degrades where curvature is higher. The regression itself is
textbook (ridge/OLS via scikit-learn 1.9.0, already pinned) and needs no new dependency and no
package-legitimacy audit. The scientific-conduct machinery — pre-registration with git-
ancestry proof, a runner that refuses to compute without a frozen artifact, JSONL provenance
rows — is a direct, well-documented template from Phase 4's `region_partition.py` /
`region_partition_mknn_run.py`, reproduced here as a concrete pattern rather than summarized.

**The single most important correction this research makes to `05-CONTEXT.md`:** the module
named in D5-03 and `<code_context>` (`decoder_curvature.py`'s `plain_decoder_curvature`) is
**not** the function that computes curvature through the CAE's per-chart decoder. Its own
docstring states it is `chart_curvature.py` "with the `chart_decoders[chart_idx]` two-hop
composition removed" — built for **decoders with no chart index at all** (the plain
autoencoder and TopoAE substrates Phase 02.6 screened). The function Phase 3 actually used to
decode curvature through the CAE, and that Phase 5 must use, is
`chart_curvature.chart_curvature_field(model, x, mode="reverse")`, which internally handles
chart assignment (`model.chart_probs(z).argmax(dim=1)`) and returns one row-order-aligned
`H_norm` per point. This resolves research question 7 (chart-assignment) completely: a single
pooled `||H||` per point **is** well-defined, because `chart_curvature_field` already performs
the assignment and reassembly — Phase 5's own code does not need to route by chart at all, it
only needs to call this one function per seed's loaded model.

**Second major finding, load-bearing for D5-04/D5-05:** Phase 3's own three-seed spread
(`03-09-SUMMARY.md`) measured `||H||` medians of **1,359.0 / 51,437.9 / 70,794.1** across
seeds 20260813/14/15 — a 52x range, with two of the three seeds' fields piecewise-constant
(only 3-4 distinct values, one per surviving chart) on metrics whose spectrum has collapsed to
`~1e-07`. **Naively averaging the three raw `||H||` fields would produce a pooled field
dominated ~99% by seeds 14 and 15's near-constant, chart-indexed values** (proportional
contribution ≈1.1% / 41.6% / 57.3% by raw magnitude), effectively erasing seed 13's real
per-point variation. Pooling must normalize each seed's field before averaging (recommended:
per-seed rank/percentile transform, or division by each seed's own median) — this needs to be
an explicit, frozen choice at D5-09, not left to a raw `np.mean` across seeds.

**Primary recommendation:** fit ridge regression (`sklearn.linear_model.RidgeCV`, single
shared alpha, LOOCV-selected on the training split only) rather than plain OLS — not because
`n_train >> parameters` makes OLS literally underdetermined (it does not: 768 output
regressions each have ~769 parameters against thousands of training rows), but because the
manifold's own established intrinsic dimensionality (~18-25, established across Phases 1-4)
means the 768-d ambient design matrix is severely rank-deficient in effective terms, so OLS
along near-null singular directions is numerically unstable. Score with per-point squared L2
residual and an aggregate variance-weighted `R^2` sharing that same residual as numerator —
this satisfies CONTEXT.md's "one underlying quantity" constraint directly via
`sklearn.metrics.r2_score(..., multioutput="variance_weighted")`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| PU pair loading (`hsc`, `legacysurvey`) | Data/cache tier (`pu_manifold/cache.py`, resolved-npz glob pattern) | — | Read-only, already-frozen artifact; no new loading logic needed beyond reusing `load_pu_pair`'s resolution rule |
| Decoder-side curvature field extraction | Estimator tier (`pu_manifold/chart_curvature.py`, sealed Phase 3 module) | Model tier (`pu_manifold/cae.py`, sealed checkpoint) | Sealed, tested, imported unchanged; Phase 5 writes no new estimator |
| Seed pooling / normalization | New module tier (`pu_manifold/linear_probe.py`) | — | New logic; must be written, tested, and frozen at D5-09 |
| Linear probe fit (ridge/OLS) | New module tier (`pu_manifold/linear_probe.py`) | scikit-learn (`sklearn.linear_model`) | Standard library call, no hand-rolled solver |
| Residual scoring, bucketing, verdict | New module tier (`pu_manifold/linear_probe.py`) + runner tier (`diagnostics/curvature_probe_decodability_run.py`) | scikit-learn (`sklearn.metrics.r2_score`), scipy (`spearmanr`, `bootstrap`) | Mirrors Phase 4's `mknn.py` / `region_partition_mknn_run.py` split of pure functions vs orchestration |
| Pre-registration guard | Runner tier (`--mode` dispatch, `assert_preregistered()`) | — | Direct copy of Phase 4's `region_partition.assert_preregistered()` pattern |
| Provenance / JSONL persistence | Cache tier (`pu_manifold/cache.py`, manual JSONL append) | — | Reuse `cache.py`'s containment-checked path helpers; JSONL append is hand-rolled in every prior runner (no `cache.py` JSONL helper exists — this is the one place Phase 5 replicates an existing pattern rather than importing a function) |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `scikit-learn` | `1.9.0` (pinned, `[VERIFIED: notebooks/requirements-notebooks.txt]`) | `Ridge`/`RidgeCV` for the linear map, `r2_score` for the aggregate metric | Already the project's regression/metrics library (used by no prior Phase 5 predecessor module directly, but is the pinned dependency every notebook installs from `requirements-notebooks.txt`) |
| `numpy` | `2.5.1` (pinned, `[VERIFIED: notebooks/requirements-notebooks.txt]`) | Array ops, `np.percentile` for bucket edges | Already universal in `pu_manifold/` |
| `scipy` | `1.18.0` (pinned, `[VERIFIED: notebooks/requirements-notebooks.txt]`) | `spearmanr` (continuous statistic), `bootstrap` (per-bucket CI), `permutation_test` if a null is added | `region_partition_mknn_run.py` and `mknn.py` already use exactly these three scipy entry points for the analogous Phase 4 statistics |
| `torch` | `2.13.0+cpu` (pinned, `[VERIFIED: notebooks/requirements-notebooks.txt]`) | Loading and running the sealed CAE checkpoints through `chart_curvature.chart_curvature_field` | Already a pinned dependency for every torch-touching `pu_manifold` module |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `sklearn.linear_model.RidgeCV` | (part of scikit-learn 1.9.0) | Chooses ridge alpha via efficient LOOCV on the training split alone | When freezing a regularization-selection *rule* rather than a hand-picked constant (see § Train/Test Split research answer) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `RidgeCV` (sklearn) | Plain `np.linalg.lstsq` (OLS, no regularization) | Simpler, but ill-conditioned given the manifold's low effective dimensionality (~18-25) inside a 768-d ambient space; coefficients along near-null singular directions would be dominated by noise, inflating test-set residual variance unrelated to curvature |
| `RidgeCV` (sklearn) | Manual k-fold CV loop over a hand-picked alpha grid | Redundant — `RidgeCV`'s default generalized cross-validation (efficient LOOCV via one SVD) is exactly this, already implemented, already tested by scikit-learn's own suite |
| `sklearn.metrics.r2_score(multioutput="variance_weighted")` | Per-output-dimension `R^2` averaged uniformly (`multioutput="uniform_average"`, sklearn's default) | Uniform averaging treats a near-zero-variance output dimension identically to a high-variance one, so it does **not** share the same SS_res/SS_tot numerator as the per-point residual `r_i = sum_j (y_ij - yhat_ij)^2` — breaks CONTEXT.md's "one underlying quantity, shared denominator" constraint |

**Installation:** No new packages. Every library above is already pinned in
`notebooks/requirements-notebooks.txt` and imported by sealed `pu_manifold/` modules.

**Version verification:** `[VERIFIED: notebooks/requirements-notebooks.txt]` — read directly
(lines 50-52, 47): `numpy==2.5.1`, `scipy==1.18.0`, `scikit-learn==1.9.0`, `torch==2.13.0+cpu`.
These are the exact versions the Phase 1 tracer ran under, per that file's own comment, and
every subsequent phase (2 through 4, 02.x, 03.x) has run against them unchanged.

## Package Legitimacy Audit

**Not applicable — Phase 5 installs no external packages.** Every library used
(`scikit-learn`, `numpy`, `scipy`, `torch`) is already pinned in
`notebooks/requirements-notebooks.txt` and imported by sealed `pu_manifold/` modules from
Phases 1-4. No `npm view` / `pip index versions` / package-legitimacy check is needed because
no new registry dependency is introduced. This matches Phase 4's own `COVERAGE.md` posture
("no external API integration ... computes with numpy, scipy and scikit-learn (all already
pinned ... and already imported by sealed modules)").

**Packages removed due to [SLOP] verdict:** none — no new packages proposed.
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
                     notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz
                     (resolved by load_pu_pair()'s "most rows, tie -> lexicographically
                      first sorted(glob) path" rule -- VERIFIED to resolve to this exact
                      file today)
                                    |
                       hsc (10000,768) legacysurvey (10000,768)
                       both already L2-normalized (norm == 1.0 to
                       float64 rounding -- VERIFIED against the cached npz)
                                    |
                +-------------------+-------------------+
                |                                        |
        [TRAIN/TEST SPLIT]                    [DECODER-SIDE CURVATURE FIELD]
        random split, frozen seed                          |
        (D5-09 constant)                    for seed in (20260813, 20260814, 20260815):
                |                                model = load_converged_model(n_charts=4, seed)
        train_idx, test_idx                      H_norm_seed = chart_curvature.chart_curvature_field(
                |                                    model, legacysurvey_all.double(), mode="reverse"
                |                                )["H_norm"]
                |                                          |
        [RIDGE FIT: hsc[train] -> legacysurvey[train]]    normalize each seed's field
        RidgeCV(alpha grid, LOOCV on train only)          (per-seed rank/median -- NOT raw
                |                                          average, see Pitfall 2)
        W, b (frozen coefficients)                        |
                |                                pooled_H_norm = mean(normalized seed fields)
        [PREDICT test: yhat = W @ hsc[test] + b]           |
                |                                inter-seed Spearman diagnostics (D5-05,
        [PER-POINT RESIDUAL]                    non-gating, reported alongside)
        r_i = ||legacysurvey[test_i] - yhat_i||^2          |
                |                                          |
                +-------------------+---------------------+
                                    |
                    [BUCKET test points by pooled_H_norm]
                    edges = quantile(pooled_H_norm, over ALL 10,000 pts)  (D5-09 frozen)
                    each test point's bucket = which edge-interval its own
                    pooled_H_norm falls in
                                    |
                +-------------------+-------------------+
                |                                        |
    [BUCKETED COMPARISON]                    [CONTINUOUS COMPARISON]
    per-bucket R^2 / mean residual,          spearman(pooled_H_norm[test], r_i)
    D5-08 size-matched re-check                          |
                +-------------------+-------------------+
                                    |
                          [VERDICT_RULE, frozen pre-registration]
                          HOLDS / NO DETECTABLE DIFFERENCE
                                    |
                     05-PREREGISTRATION.md, 05-FINDINGS.md,
                     05-VERIFICATION.md (git-ancestry proof)
```

### Recommended Project Structure

```
notebooks/
├── pu_manifold/
│   └── linear_probe.py            # NEW -- pure functions: fit, score, pool, bucket, verdict
│                                    # (no file I/O, mirrors curvature_probe.py's own posture)
├── diagnostics/
│   └── curvature_probe_decodability_run.py   # NEW -- orchestration, JSONL, --mode dispatch,
│                                               # assert_preregistered() guard (mirrors
│                                               # region_partition_mknn_run.py exactly)
├── 05_curvature_conditioned_linear_decodability.ipynb   # NEW -- executed notebook, committed
                                                            # with outputs
```

### Pattern 1: Pre-registration freeze with git-ancestry proof (Phase 4's exact template)

**What:** every free parameter — split fraction, ridge-selection rule, seed-pooling
normalization, bucket count/edges rule, `VERDICT_RULE` full text, `SEED` — lives as a
module-level constant in a sealed `notebooks/pu_manifold/` module (`linear_probe.py`), with an
`assert_preregistered()` guard the runner calls before computing anything. A committed
`05-PREREGISTRATION.md` restates every constant verbatim.

**When to use:** every constant that could be tuned after seeing a PU number.

**Example (verbatim structural pattern from the sealed `region_partition.py`, lines 48-136,
adapted names for Phase 5):**

```python
# --- Pre-registration (D5-09) -------------------------------------------------------
#
# PRE-REGISTERED: every constant below, and VERDICT_RULE's full text, were ratified at
# this plan's blocking decision checkpoint BEFORE any PU probe number existed. Amending
# any of these after a probe number has been computed invalidates the phase.

TRAIN_FRACTION = 0.7          # -> frozen value; see research answer for rationale
SPLIT_SEED = <frozen>
N_BUCKETS = 3                 # tertiles; see research answer
RIDGE_ALPHA_GRID = tuple(np.logspace(-2, 4, 13))   # frozen grid, RidgeCV selects within it
POOLING_METHOD = "per_seed_median_normalize"        # frozen normalization rule (D5-04)
SEED_STEMS = (20260813, 20260814, 20260815)

VERDICT_RULE = """... full text, frozen ..."""


def assert_preregistered() -> None:
    """Raise RuntimeError unless every constant above is present and well-formed. Called
    at the top of the runner's --mode bucketed branch (D5-10) so the bucketed path fails
    loudly rather than computing anything when the pre-registration is absent."""
    ...
```

**Verification precedent (what `05-VERIFICATION.md` must reproduce, from
`04-VERIFICATION.md`'s own Critical Focus section):**

```bash
git merge-base --is-ancestor <freeze-commit> <first-probe-number-commit>
git diff <freeze-commit> HEAD -- notebooks/pu_manifold/linear_probe.py   # must be empty
```

### Pattern 2: Runner that refuses to compute without the guard (D5-10)

**What:** the `--mode` dispatch in `region_partition_mknn_run.py`'s `main()` (lines 795-808)
calls `region_partition.assert_preregistered()` and checks a frozen npz artifact's existence
**before** touching any regional data. Phase 5's runner must do the identical check before its
bucketed-probe branch.

**Example (verbatim from the sealed runner, showing the exact guard shape to replicate):**

```python
if a.mode == "regional":
    # D4-11/REGN-04: fail loudly rather than compute anything when the pre-registration
    # or the frozen partition artifact is absent. This guard must run BEFORE any regional
    # cell is computed -- it is what makes the pre-registration commit's ordering
    # enforceable, not merely documented.
    region_partition.assert_preregistered()
    partition_artifact = cache.cache_path("04_region_partition", "npz")
    if not partition_artifact.exists():
        raise FileNotFoundError(
            f"--mode regional requires the frozen partition artifact at "
            f"{partition_artifact}, which does not exist. Run --mode partition first ..."
        )
```

Phase 5's analogue: `--mode bucketed` calls `linear_probe.assert_preregistered()` and checks
for a frozen `05_curvature_field.npz` (the pooled `||H||` array) before computing any
bucketed residual number.

### Pattern 3: Loading a sealed CAE checkpoint and extracting decoder-side curvature

**What:** `curvature_field_pu_run.load_converged_model` (lines 1510-1530) rebuilds the model
architecture from named constants, loads `state_dict`, validates the checkpoint's own recorded
`n_charts`/`seed` against the requested values, and **never retrains** — a missing checkpoint
is a raised `FileNotFoundError`, never a silent fallback to training.

**Example (verbatim, the exact call Phase 5 must reuse or closely mirror):**

```python
# Source: notebooks/diagnostics/curvature_field_pu_run.py lines 1510-1530, 487-500
def build_cae(n_charts: int, device=torch.device("cpu")):
    model = cae.ChartAutoEncoder(
        in_dim=768, embed_dim=40, chart_dim=20, n_charts=n_charts,
        hidden=[250, 250, 250], activation="silu",
    )
    return model.to(device)

def load_converged_model(n_charts, seed, device):
    ckpt_path = cache.cache_path(f"03_converged_cae_pu_nc{n_charts}_seed{seed}", "pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if int(ckpt["n_charts"]) != n_charts or int(ckpt["seed"]) != seed:
        raise ValueError(f"{ckpt_path} carries a mismatched n_charts/seed.")
    model = build_cae(n_charts, device=device).double()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

# The curvature field itself -- NOT decoder_curvature.plain_decoder_curvature (see Pitfall 1)
from pu_manifold import chart_curvature
field = chart_curvature.chart_curvature_field(model, x64, mode="reverse")
H_norm = field["H_norm"].detach().cpu().numpy().astype(np.float64)   # (10000,)
```

All three sealed checkpoints are `n_charts=4` (`03_converged_cae_pu_nc4_seed2026081{3,4,5}.pt`,
`[VERIFIED: notebooks/.cache/ directory listing]`), so `build_cae(4, ...)` is the correct
architecture call for every seed.

### Anti-Patterns to Avoid

- **Calling `decoder_curvature.plain_decoder_curvature(model, z)` on the CAE.** This is the
  function CONTEXT.md's D5-03/`<code_context>` names, but it is built for models with **no
  chart index** — see Pitfall 1. Calling it on a `ChartAutoEncoder` instance would either
  raise (no bare `.decode(z)` matching the expected single-hop signature — `ChartAutoEncoder`
  has no `.decode` method, only `.reconstruct`/chart-routed decode paths) or silently compute
  the wrong quantity if a compatibility shim were added. Use `chart_curvature.chart_curvature_field`.
- **Averaging the three seeds' raw `||H||` fields with `np.mean`.** Given the measured 52x
  median range (1,359 / 51,438 / 70,794), a naive average is ~99% seed-14/15 by weight. See
  Pitfall 2.
- **Fitting OLS/ridge on the training split, then scoring with a different distance (e.g.
  cosine) than the fit objective minimized.** Creates a train/eval-metric mismatch; see the
  Residual Metric research answer.
- **Computing bucket edges on the test split alone "to guarantee equal counts."** CONTEXT.md's
  discretion item names "tertiles vs quartiles of the pooled `||H||` field over 10,000
  points" — edges belong to the full pooled field, not the post-split test subset. See the
  Bucket Count research answer for the consequence this has for D5-08.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Ridge regularization strength selection | A manual k-fold CV loop over a hand-picked alpha grid | `sklearn.linear_model.RidgeCV(alphas=..., store_cv_values=False)` | Implements efficient LOOCV via one SVD; already tested by scikit-learn's own suite; freezing the *grid* and *selection rule* (not a specific alpha) satisfies D5-09 without inventing new machinery |
| Aggregate multi-output `R^2` | A hand-rolled `1 - SS_res/SS_tot` loop over 768 columns | `sklearn.metrics.r2_score(Y_true, Y_pred, multioutput="variance_weighted")` | Exact aggregate form; avoids the uniform-average pitfall (see Alternatives Considered) |
| Per-bucket bootstrap confidence interval on mean residual | A manual resampling loop | `scipy.stats.bootstrap((per_point_residuals,), np.mean, method="percentile", n_resamples=..., confidence_level=..., rng=...)` | Identical call shape already used by `mknn.bootstrap_ci` (lines 124-167) for the analogous Phase 4 statistic — same library, same method, same seeding discipline |
| Equal-frequency quantile bucket assignment | A `np.quantile`-edge-based bucketer (edge ties silently break bin balance) | Rank-based partition via `np.argsort` + `np.array_split`, the exact pattern already implemented (but private, `_`-prefixed) in `curvature_probe._quantile_bin_labels` (lines 432-446) | Guarantees well-defined bin membership with no edge-tie ambiguity; the existing private helper is the reference implementation to replicate (not import — it is module-internal to `curvature_probe.py`, a different phase's sealed module) |
| Continuous rank-correlation statistic | A hand-rolled Spearman computation | `scipy.stats.spearmanr` | Already the project's exclusive Spearman entry point (`curvature_probe.spearman_gate_statistic`, `region_partition_mknn_run._spearman_report`, `03-*` fidelity axes all use it) |

**Key insight:** every statistical primitive Phase 5 needs (ridge selection, aggregate `R^2`,
bootstrap CI, rank correlation, equal-frequency binning) already has either a library call or
an existing in-repo private-helper pattern used by a sealed sibling module for the structurally
identical Phase 4 problem. There is no genuinely novel statistical machinery in this phase —
the risk surface is entirely in getting the curvature-field extraction call right (Pitfall 1)
and the seed-pooling normalization right (Pitfall 2), not in the regression itself.

## Runtime State Inventory

Not applicable — Phase 5 is a greenfield addition (new module, new runner, new notebook), not
a rename/refactor/migration phase.

## Common Pitfalls

### Pitfall 1: Calling the wrong decoder-curvature function (CONTEXT.md's own citation is imprecise)

**What goes wrong:** `05-CONTEXT.md`'s D5-03 and `<code_context>` name
`notebooks/pu_manifold/decoder_curvature.py`'s `plain_decoder_curvature(model, z)` as "the
decoder-side autodiff path" for "the CAE chart decoder." Read literally, a planner would wire
the new runner to call `decoder_curvature.plain_decoder_curvature`.

**Why it happens:** `decoder_curvature.py` genuinely is a decoder-side autodiff curvature
path, and it genuinely was written for this milestone's decoder-curvature work — but its own
module docstring (lines 1-18) states it explicitly: *"this module is `chart_curvature.py` with
the `chart_decoders[chart_idx]` two-hop composition removed ... both free candidate substrates
screened by Phase 02.6 (a plain autoencoder ... and a `PlainAutoEncoder` trained under
`topoae.train_topoae`) decode through ONE smooth MLP end to end and have no chart index at
all."* It was built for the Phase 02.6 screening candidates (plain AE, TopoAE), which have a
single `.decode(z)` call with no chart routing — not for the CAE. `plain_decoder_map`'s own
docstring (line 152) calls `model.decode(z.unsqueeze(0))` directly — `ChartAutoEncoder` has no
bare `.decode` method matching that signature; its decode path is chart-routed
(`chart_decoders[i]` + shared `embedding_decoder`).

**How to avoid:** use `chart_curvature.chart_curvature_field(model, x, mode="reverse")`
(`notebooks/pu_manifold/chart_curvature.py` lines 513-611) — the function Phase 3's own
`curvature_field_pu_run.py` calls at line 1627 (`chart_curvature.chart_curvature_field(model,
x64, mode="reverse")`) to produce the exact field `03-09-SUMMARY.md` reports. It takes
**ambient** `x` (768-d rows), not a latent `z`, and internally handles encoding, chart
assignment (`model.chart_probs(z).argmax(dim=1)`), per-chart curvature computation, and
row-order reassembly. `05-PREREGISTRATION.md` should name this exact function and cite this
correction explicitly, since a planner or implementer following D5-03's literal text would
otherwise wire the wrong call.

**Warning signs:** any code path that calls `.decode(` directly on a `ChartAutoEncoder`
instance, or that receives an `AttributeError` on `model.decode`, or that imports
`plain_decoder_curvature` and passes it a chart-routed model.

### Pitfall 2: Naive averaging of the three seeds' `||H||` fields is dominated by 2 of 3 seeds

**What goes wrong:** D5-04 says "pool the three cached CAE seeds into one averaged `||H||`
field" without specifying a normalization. A literal `np.mean(np.stack([H1, H2, H3]), axis=0)`
on the raw fields produces a pooled field that is ~99% determined by seeds 20260814/20260815.

**Why it happens:** `03-09-SUMMARY.md` (§ "COMPLETE — and the completed spread invalidates
this plan's headline") measured `||H||` medians of **1,359.0 (seed 20260813) / 51,437.9 (seed
20260814) / 70,794.1 (seed 20260815)** — a 52x range — with seeds 14 and 15's fields
**piecewise-constant** (4 and 3 distinct values respectively, one per surviving chart) on
metrics whose entire spectrum has collapsed to `~1e-07` (`det(g) ~ 1e-162`). By raw magnitude,
a naive average weights the three seeds at approximately 1.1% / 41.6% / 57.3%. Because seeds
14/15 are each essentially a 4-valued (chart-indexed) step function, the pooled field's
per-point ordering would be almost entirely determined by **which chart each point is assigned
to** in seeds 14/15 — not by any real within-chart curvature variation from seed 13, the only
seed whose field actually varies continuously per point.

**How to avoid:** normalize each seed's field before averaging — e.g. divide each seed's
`H_norm` by its own median (or apply a per-seed rank/percentile transform to `[0, 1]`) before
taking the mean across seeds. This is a genuine design choice with scientific consequences and
must be an explicit, named, frozen constant at D5-09 (e.g. `POOLING_METHOD =
"per_seed_median_normalize"`), not left as an implicit `np.mean` default. D5-05's inter-seed
Spearman diagnostic is exactly the check that would have surfaced this problem after the fact
— running it *before* deciding the pooling method, not only as a post-hoc report, is strongly
recommended even though D5-05 as written frames it as non-gating.

**Warning signs:** the pooled field's per-point values only take a handful of distinct values
(a symptom that 2 of 3 chart-indexed piecewise-constant seeds are dominating); the pooled
field's percentile distribution closely matching seed 14's or seed 15's shape rather than a
genuine blend of all three.

### Pitfall 3: Treating `n_train >> parameters` as proof OLS needs no regularization

**What goes wrong:** the naive parameter count (768 × 768 = 589,824 total entries in `W`) makes
the fit look underdetermined at `n=10,000`, prompting either (a) an incorrect conclusion that
regularization is *mandatory* for well-posedness, or (b) the opposite incorrect conclusion
that because each of the 768 per-output regressions has only 769 parameters against thousands
of training rows, OLS is safely overdetermined and no regularization is needed at all.

**Why it happens:** the 589,824-parameter framing conflates the full matrix `W` with the
per-output regression structure. Each output dimension `j` is its own OLS problem,
`y_j = X w_j + b_j`, with `X` shared across all 768 problems and `769` parameters per problem —
at a 70/30 split this is `7,000` training rows against `769` parameters, nominally
overdetermined by ~9x. **But** every measurement in this milestone (Phase 1's `compute_dim`
panel, Phase 2's frozen `d`, the GEOM-05 re-derivation, `D-11`'s TwoNN/local-PCA/geometric
cluster at 18-25) establishes that the *effective* dimensionality of these 768-d embeddings is
roughly 18-25, not 768. The design matrix `X` (the 7,000 x 768 training `hsc` block) is
therefore severely rank-deficient in effective (not literal) terms — most of its singular
spectrum sits at noise level, and OLS coefficients along those near-null directions are
dominated by noise amplification, not signal.

**How to avoid:** use ridge regression with a small, LOOCV-selected penalty (`RidgeCV`) rather
than plain OLS — not because the system is literally underdetermined, but because it is
poorly conditioned. Freeze the *selection rule* (alpha grid + LOOCV-on-train-only), not a
single hand-picked alpha value, at D5-09.

**Warning signs:** plain OLS producing `W` with very large coefficient norms, or held-out
residuals blowing up relative to training residuals (classic overfitting-to-noise-directions
signature); `np.linalg.cond(X_train)` (or the singular value spectrum of `X_train`) showing a
sharp drop after the first ~20-25 singular values, matching the established intrinsic
dimensionality.

### Pitfall 4: Bucketing on the full pooled field silently breaks D5-08's equal-count premise

**What goes wrong:** the discretion item frames bucketing as "tertiles vs quartiles of the
pooled `||H||` field over 10,000 points" — a phrasing that reads as if equal-frequency binning
automatically produces equal-sized groups for the bucketed residual comparison. It does, **at
the full-10,000-point level**. But the bucketed *residual* comparison only ever scores the
**test split** (per D5-02, residuals are computed on held-out test points, not all 10,000). If
bucket edges are computed on the full pooled field and then applied to only the test subset,
the resulting per-bucket test counts are a random subsample of each (exactly equal) full-field
bucket — subject to ordinary sampling variance, not guaranteed equal.

**Why it happens:** equal-frequency binning is size-matched at the population level it was
computed on; it says nothing about size-matching at any other subset level unless that subset
was itself stratified by bucket during sampling (which a plain random train/test split does
not do).

**How to avoid:** report the realized test-split `n` per bucket explicitly (D5-08 already
requires this) and run the size-matched subsampling check against those **realized test-split
counts**, not against the full-field counts. This is precisely the lesson Phase 4's own
`04-FINDINGS.md` §7 records as its "fifth item, raised by this closure plan itself": *"a
size-matched or chance-floor-normalized regional MKNN statistic ... was not on
`04-CONTEXT.md`'s Deferred list because the region-size artifact was not identified until this
closure plan's mandatory verification."* Phase 5 should build the size-matched check into the
pre-registered protocol from the start rather than discovering the need for it after computing
the headline number, exactly as Phase 4 now recommends for any future phase re-running a
similar comparison.

**Warning signs:** the four (or three) bucket test-set counts differing by more than a few
percent from `test_n / n_buckets`.

## Code Examples

Verified patterns from this repository's own sealed modules:

### 1. Resolving the same 10,000-point subsample Phase 4 used

```python
# Source: notebooks/diagnostics/region_partition_mknn_run.py lines 42-70 (verbatim)
def load_pu_pair(column_a="hsc", column_b="legacysurvey"):
    """Both columns from the SAME resolved `subsample_*.npz`, plus the resolved path.
    Keeps only files carrying both columns, selects the one with the most rows; on a
    tie keeps the lexicographically first path (mirrors the existing strictly-greater-than
    comparison over a sorted(glob))."""
    cands = sorted(glob.glob(str(NOTEBOOK_ROOT / ".cache" / "subsample_*.npz")))
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

`[VERIFIED: notebooks/.cache/ directory contents, checked directly this session]` — this
resolution rule, run today, resolves to `subsample_20260729_a79b3460b838fd0a.npz` — the
**identical file** `04-FINDINGS.md` §2 names as its "resolved subsample file." Three
`subsample_*.npz` candidates exist in cache; two carry 10,000 rows
(`subsample_20260729_a79b3460b838fd0a.npz` and `subsample_20260801_1f03afec9d0b8e38.npz`) and
tie on row count, so the lexicographic tiebreak (not the row count) is what decides the
resolution — confirmed by direct read: both `hsc` and `legacysurvey` arrays in the resolved
file have `np.linalg.norm(row) == 1.0` to float64 rounding for every row (already
L2-normalized), and `hsc_norms`/`ls_norms` carry the pre-normalization original norms
(~15-17), unused by the probe itself but present in the npz.

### 2. Decoder-side curvature field, per seed (the corrected call — see Pitfall 1)

```python
# Source: notebooks/diagnostics/curvature_field_pu_run.py lines 1510-1530, 1619-1628
# and notebooks/pu_manifold/chart_curvature.py lines 513-611 (chart_curvature_field)
from pu_manifold import cache, chart_curvature

def build_cae(n_charts, device=torch.device("cpu")):
    model = cae.ChartAutoEncoder(
        in_dim=768, embed_dim=40, chart_dim=20, n_charts=n_charts,
        hidden=[250, 250, 250], activation="silu",
    )
    return model.to(device)

def load_converged_model(n_charts, seed, device):
    ckpt_path = cache.cache_path(f"03_converged_cae_pu_nc{n_charts}_seed{seed}", "pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No converged checkpoint at {ckpt_path}.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_cae(n_charts, device=device).double()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

x64 = torch.tensor(legacysurvey_all, dtype=torch.float64)   # (10000, 768)
seed_fields = {}
for seed in (20260813, 20260814, 20260815):
    model, _ = load_converged_model(n_charts=4, seed=seed, device=torch.device("cpu"))
    field = chart_curvature.chart_curvature_field(model, x64, mode="reverse")
    seed_fields[seed] = field["H_norm"].detach().cpu().numpy().astype(np.float64)

# D5-05: inter-seed diagnostics -- reuse the exact spearmanr pattern already used for
# REGN-02 in region_partition_mknn_run._spearman_report (lines 125-136)
from scipy.stats import spearmanr
for a, b in itertools.combinations(seed_fields, 2):
    rho, p = spearmanr(seed_fields[a], seed_fields[b])
    print(f"inter-seed spearman({a}, {b}) = {rho:+.4f}  p={p:.4g}")
```

`03-09-SUMMARY.md`'s measured full-10,000-row wallclock for **one seed's** field via this
exact call was **3,129.5 seconds (~52 minutes) on CPU** (`[VERIFIED:
03-09-SUMMARY.md`'s `key-decisions`/`duration` frontmatter, "field 3129.5s"]`). For three
seeds run sequentially, this is roughly **2.6 hours of CPU wall-clock** for field extraction
alone — the single largest runtime cost in Phase 5, and worth budgeting explicitly into the
plan's wave structure (e.g. one wave/task per seed, each independently cacheable via a
per-seed npz keyed by `cache.npz_cache`, so a re-run never repeats a seed already computed).

### 3. The pre-registration guard (D5-10)

```python
# Source: notebooks/diagnostics/region_partition_mknn_run.py lines 795-808 (verbatim shape)
if a.mode == "bucketed":
    linear_probe.assert_preregistered()
    field_artifact = cache.cache_path("05_curvature_field", "npz")
    if not field_artifact.exists():
        raise FileNotFoundError(
            f"--mode bucketed requires the frozen pooled curvature field at "
            f"{field_artifact}, which does not exist. Run --mode field first."
        )
    ...
```

### 4. Bootstrap CI on a per-bucket statistic (reuse `mknn.bootstrap_ci`'s exact shape)

```python
# Source: notebooks/pu_manifold/mknn.py lines 124-167, generalized to any per-point array
from scipy.stats import bootstrap

def bucket_residual_ci(residuals_in_bucket, n_resamples, seed, confidence_level):
    rng = np.random.default_rng(seed)
    result = bootstrap(
        (residuals_in_bucket,), np.mean, method="percentile",
        n_resamples=n_resamples, confidence_level=confidence_level, rng=rng,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)
```

## State of the Art

Not applicable in the "libraries changed recently" sense — the regression machinery
(`sklearn.linear_model.Ridge`/`RidgeCV`) has been stable API for many releases and the pinned
`scikit-learn==1.9.0` supports every function this phase needs, including `alpha_per_target`
on `RidgeCV` (available since scikit-learn 0.24, well before the pinned 1.9.0).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Per-seed normalization (median-divide or rank-transform) is the right fix for the seed-pooling scale mismatch found in Pitfall 2 | Common Pitfalls 2; Summary | If wrong, the pooled field could still be dominated by a subset of seeds under a different failure mode; the planner should treat the *specific* normalization method as a discretion item to ratify at the pre-registration checkpoint, not as settled by this research |
| A2 | Ridge (not plain OLS) is the right default given the manifold's established ~18-25 effective dimensionality inside 768 ambient dims | Common Pitfalls 3; Summary; Standard Stack | If the embeddings' training-split design matrix turns out well-conditioned in practice (e.g. because `hsc` retains more effective rank than the milestone's Isomap-based dimension estimates, which were computed on a different processing path), ridge with a near-zero selected alpha degrades gracefully to OLS anyway — low risk, but the *reason given* for using ridge should be checked against the training split's own measured condition number/singular spectrum before being asserted as fact in `05-PREREGISTRATION.md` |
| A3 | `r2_score(..., multioutput="variance_weighted")` is mathematically equivalent to the aggregate Frobenius-form `1 - sum(r_i)/sum(||y_i - ybar||^2)` this document recommends | Standard Stack; Don't Hand-Roll | This is a documented sklearn behavior (`[CITED: scikit-learn's own r2_score docstring describes variance_weighted as "scores are weighted by the variances of each individual output"]`) rather than independently re-derived and unit-tested in this session; the planner should have the implementation verify this equivalence with a small numeric test before relying on it, since a subtle mismatch (e.g. ddof convention) would silently break the "one underlying quantity, shared denominator" constraint |
| A4 | 70/30 train/test split with tertile bucketing leaves comfortably-sized test buckets (~1,000/bucket) | Bucket Count / Train-Test Split research answers | This is a recommendation, not a locked fact; the planner should confirm the realized post-split, post-bucket counts once the split is actually run, per Pitfall 4's own point that pre-split equal-frequency binning does not guarantee post-split equal counts |

**Everything else in this document (the Pitfall 1 correction, the Pitfall 2 seed-scale
measurement, the `load_pu_pair` resolution proof, the 3,129.5s field-extraction timing, the
already-L2-normalized embedding proof, the pinned dependency versions) is `[VERIFIED]` against
this session's own direct reads of the repository, not `[ASSUMED]`.**

## Open Questions

1. **Exact seed-pooling normalization method (median-divide vs percentile-rank vs z-score).**
   - What we know: raw averaging is dominated by 2 of 3 seeds (Pitfall 2); some normalization
     is necessary.
   - What's unclear: which specific normalization the planner/user wants frozen. Median-divide
     preserves the raw field's shape within each seed (a multiplicative rescale) while
     percentile-rank discards magnitude information entirely (every seed becomes uniform on
     `[0,1]` before averaging) — these produce genuinely different pooled fields, not just
     differently-scaled versions of the same one, because seeds 14/15 are piecewise-constant
     (only 3-4 distinct values) while seed 13 varies continuously.
   - Recommendation: median-divide is the more conservative choice (it does not discard within-
     seed relative structure), but the planner should surface this explicitly as a discussion
     point in `05-PREREGISTRATION.md` rather than silently picking one, given how much D5-04's
     verdict field depends on it.

2. **Whether ridge's selected alpha, once computed, should be reported per-seed-pooling-choice
   or fixed once and reused.**
   - What we know: `RidgeCV`'s LOOCV selection happens once, on the training split, independent
     of the curvature field entirely (the probe fit does not use `||H||` as a feature).
   - What's unclear: nothing structurally — this is actually resolved by construction, since
     the probe fit (`hsc -> legacysurvey`) and the curvature-field computation are fully
     independent pipelines that only meet at the bucketing step. Flagged here only so the
     planner does not accidentally introduce a dependency between them.
   - Recommendation: keep the two pipelines independent in the module's function signatures
     (`fit_probe(X_train, Y_train) -> W, b, alpha` takes no curvature argument;
     `bucket_residuals(residuals, pooled_H_norm, edges) -> ...` takes no probe argument).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `numpy` | Array ops throughout | ✓ (`[VERIFIED]`, imported successfully this session) | 2.5.1 (pinned) | — |
| `scipy` | `spearmanr`, `bootstrap` | ✓ (`[VERIFIED]`, imported successfully this session) | 1.18.0 (pinned) | — |
| `scikit-learn` | `Ridge`/`RidgeCV`, `r2_score` | ✓ (pinned in `requirements-notebooks.txt`; not directly imported this session, but every other sealed `pu_manifold` module imports it without issue) | 1.9.0 (pinned) | — |
| `torch` (CPU build) | Loading CAE checkpoints, `chart_curvature_field` | ✓ (`[VERIFIED]`, used this session to inspect module contents; the three checkpoint files load successfully per Phase 3's own sealed runs) | 2.13.0+cpu (pinned) | — |
| `notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz` | The 10,000-point PU pair | ✓ (`[VERIFIED]`, read directly this session) | — | — |
| `notebooks/.cache/03_converged_cae_pu_nc4_seed2026081{3,4,5}.pt` | The three CAE checkpoints | ✓ (`[VERIFIED]`, listed directly this session) | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none — every dependency this phase needs is already
present and pinned.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `pytest`, invoked via `.venv/bin/python -m pytest` (matching `test_region_partition.py`'s own invocation, `04-VERIFICATION.md`'s Behavioral Spot-Checks) |
| Config file | none detected in `notebooks/pu_manifold/` — tests run by direct file path, one test file per module (`region_partition.py` -> `tests/test_region_partition.py`) |
| Quick run command | `.venv/bin/python -m pytest notebooks/pu_manifold/tests/test_linear_probe.py -q` |
| Full suite command | `.venv/bin/python -m pytest notebooks/pu_manifold/tests -q` |

### Phase Requirement -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| D5-01 | `W` predicts `legacysurvey` from `hsc`, row-aligned | unit | `pytest tests/test_linear_probe.py::test_fit_probe_shape_and_row_alignment -x` | ❌ Wave 0 |
| D5-04/A1 | Seed pooling normalization behaves as documented (median-divide does not let one seed dominate) | unit, known-answer | `pytest tests/test_linear_probe.py::test_pool_seeds_no_single_seed_dominates -x` | ❌ Wave 0 — construct a synthetic 3-seed fixture with one seed's field scaled 50x, assert the pooled rank-order is not identical to that seed's rank-order alone |
| D5-06 | `CURVATURE_CONVENTION` agreement assertion (mirrors `decoder_curvature.py`'s own import-time cross-check pattern) | unit | `pytest tests/test_linear_probe.py::test_curvature_convention_matches_sealed_modules -x` | ❌ Wave 0 |
| D5-07/D5-09 | Bucket edges computed on pooled field over all 10,000 points, applied correctly to test-split subset | unit, known-answer | `pytest tests/test_linear_probe.py::test_bucket_assignment_known_answer -x` | ❌ Wave 0 |
| D5-08 | Size-matched subsampling check runs against realized test-split counts, not full-field counts (Pitfall 4) | unit | `pytest tests/test_linear_probe.py::test_size_matched_check_uses_test_split_counts -x` | ❌ Wave 0 |
| D5-10 | `assert_preregistered()` raises when constants/artifact absent | unit | `pytest tests/test_linear_probe.py::test_assert_preregistered_raises_when_absent -x` | ❌ Wave 0 |
| Known-answer self-check | A synthetic dataset with a planted exact linear relationship (`y = A @ x + b + tiny_noise`) and planted curvature ordering (fabricated `||H||` correlated with residual by construction) recovers the expected verdict | integration, `--selfcheck` | `python notebooks/diagnostics/curvature_probe_decodability_run.py --selfcheck` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest notebooks/pu_manifold/tests/test_linear_probe.py -q`
- **Per wave merge:** `pytest notebooks/pu_manifold/tests -q` (full suite, matching the
  project-wide convention of running the whole `pu_manifold/tests` directory before a phase
  gate — `04-VERIFICATION.md`'s spot-check ran `test_region_partition.py` in isolation but the
  project's broader pattern, e.g. `03-07-SUPPLEMENT-01.md`, runs the full 286+-test suite)
- **Phase gate:** full suite green before `/gsd-verify-work`; `--selfcheck` mode green before
  any real PU number is computed (mirrors `region_partition_mknn_run.selfcheck()`'s role as
  "the phase's automated implementation check" when D4-18 declined a dedicated test file for
  `mknn.py`)

### Wave 0 Gaps

- [ ] `notebooks/pu_manifold/tests/test_linear_probe.py` — covers D5-01, D5-04/A1, D5-06,
      D5-07/D5-09, D5-08, D5-10 (all listed above)
- [ ] `--selfcheck` mode in `curvature_probe_decodability_run.py` — the known-answer self-check
      with a planted linear relationship and planted curvature ordering; must assert the
      pre-registered verdict rule correctly returns the expected outcome on this synthetic,
      analytically-known case before any PU number is trusted
- [ ] Framework install: none — `pytest` is already the project's test runner and does not
      need installation

## Security Domain

`security_enforcement` is absent from `.planning/config.json` (defaults to enabled), so this
section is included per the standing rule. Applying the same reasoning `03.1-FINDINGS.md`'s
code review and Phase 4's `COVERAGE.md` both already used for this codebase: **this phase has
no network surface, no authentication, no user-input path, and no persistence layer beyond the
gitignored local cache.** It reads a locally-generated `.npz`, torch checkpoint `.pt` files
already sealed by Phase 3, computes in-process, and writes JSONL/npz through `cache.py`'s
`_assert_inside_cache` containment guard — identical posture to every prior phase's notebook
work in this milestone.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | no auth surface anywhere in this milestone |
| V3 Session Management | no | no sessions |
| V4 Access Control | no | single-user local notebook execution |
| V5 Input Validation | partial | `cache.py`'s `_assert_inside_cache` already guards every path this phase's cache writes touch; `linear_probe.py`'s public functions should validate array shapes/finiteness the same way `region_partition.region_partition` does (raises `ValueError` naming the offending argument on non-finite/wrong-shape input) — not a new pattern, a reuse of the existing one |
| V6 Cryptography | no | no cryptographic operation anywhere in this phase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via a caller-supplied cache stem | Tampering | Already mitigated project-wide by `cache._assert_inside_cache`; Phase 5's runner must route every cache write through `cache.cache_path`/`cache.npz_cache`/`cache.json_cache`, never construct a raw `Path` itself, matching every prior phase's discipline |
| Pickle deserialization via `joblib.load`/`torch.load` on an untrusted checkpoint | Tampering | Not a new risk this phase introduces — `torch.load(..., weights_only=False)` is already used by the sealed `curvature_field_pu_run.load_converged_model`, on checkpoints this project itself generated and that live only in the gitignored local cache, never from an external or untrusted source; Phase 5 reuses the identical loading call and inherits the identical (already-accepted) risk posture, not a new one |

## Sources

### Primary (HIGH confidence — direct repository reads this session)

- `notebooks/diagnostics/region_partition_mknn_run.py` — `load_pu_pair`, `--mode` dispatch,
  `selfcheck()`, JSONL append pattern (read in full, lines 1-245 and 643-941)
- `notebooks/pu_manifold/region_partition.py` — pre-registration constant block,
  `assert_preregistered()`, `region_partition()` (read in full)
- `notebooks/pu_manifold/decoder_curvature.py` — read in full; the module docstring and
  `plain_decoder_map` are the direct evidence for Pitfall 1
- `notebooks/pu_manifold/chart_curvature.py` — `chart_curvature_field` (lines 513-611), the
  function Phase 5 must call instead
- `notebooks/diagnostics/curvature_field_pu_run.py` — `load_converged_model` (1510-1530),
  `build_cae` (487-500), `_field_record` (1619-1680), `PU_CHART_DIM`/`PU_EMBED_DIM` constants
- `notebooks/pu_manifold/curvature_probe.py` — `local_density_weights`,
  `centroid_mean_curvature`, `_quantile_bin_labels` (read lines 1-987)
- `notebooks/pu_manifold/mknn.py` — `permutation_null`, `bootstrap_ci` (read in full)
- `notebooks/pu_manifold/cache.py` — cache-path/manifest conventions (read in full)
- `.planning/phases/04-region-partitioning-regional-alignment-mknn/04-PREREGISTRATION.md`,
  `04-FINDINGS.md`, `04-VERIFICATION.md` — the pre-registration template, the region-size
  artifact lesson (Pitfall 4), the checkpoint-ratification-under-standing-authorization pattern
- `.planning/phases/03-decoder-curvature-field/03-09-SUMMARY.md` — the 52x seed-spread
  measurement (Pitfall 2) and the 3,129.5s field-extraction timing
- `notebooks/.cache/subsample_20260729_a79b3460b838fd0a.npz` — read directly with `numpy.load`
  this session to confirm L2-normalization and the resolution rule's output
- `notebooks/requirements-notebooks.txt` — pinned dependency versions
- `.planning/phases/05-curvature-conditioned-linear-decodability/05-CONTEXT.md` — the phase
  boundary and locked decisions this research is scoped against
- `.claude/skills/spike-findings-effdim/SKILL.md` — the `d=20` decoder row, the direction-axis
  disclosure requirement, the `r/R` wall (context for D5-11's no-anchor posture)

### Secondary (MEDIUM confidence)

- `sklearn.metrics.r2_score`'s `multioutput="variance_weighted"` semantics — cited from the
  library's own documented behavior, not independently re-derived/tested this session (see
  Assumption A3)

### Tertiary (LOW confidence)

- None — every claim in this document is either directly verified against the repository this
  session or explicitly logged in the Assumptions table above.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies, versions read directly from
  `requirements-notebooks.txt`
- Architecture (curvature-field extraction path): HIGH — corrected against the sealed module's
  own docstring and Phase 3's actual call site, not inferred
- Architecture (seed pooling): MEDIUM — the *problem* (scale dominance) is HIGH confidence
  (directly measured in `03-09-SUMMARY.md`), the *specific fix* (median-divide) is a
  recommendation, not a locked fact (see Open Question 1)
- Pitfalls: HIGH — all four are grounded in direct code/data reads or a direct citation of
  Phase 4's own closing findings, not speculation

**Research date:** 2026-08-24
**Valid until:** stable for this milestone's duration — the sealed checkpoints, the frozen
subsample, and the pinned dependency versions do not change without a new phase deliberately
retraining or re-pinning; no external time pressure (no API version, no library deprecation)
applies to this research.
