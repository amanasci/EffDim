# Architecture Research: PU Manifold Curvature Notebook (v1.1)

**Domain:** Research notebook pipeline — manifold reconstruction, differential geometry, representational-alignment statistics, built on top of an existing installable Python library (EffDim)
**Researched:** 2026-07-29
**Confidence:** HIGH (sklearn API facts verified against current official docs; notebook/caching patterns are established research-engineering practice, not exotic)

## Standard Architecture

### System Overview

Three notebooks, two cache boundaries, one hard gate. The gate sits after the slow, non-iterated steps (1–3) and before the fast, heavily-iterated steps (4–7):

```
┌───────────────────────────────────────────────────────────────────────┐
│  01_manifold_and_gate.ipynb            (slow, run rarely)             │
├───────────────────────────────────────────────────────────────────────┤
│  stream+subsample(seed) → hsc, legacysurvey, row_indices (10000×768)  │
│         ↓                                                             │
│  effdim.compute_dim(legacysurvey) → candidate n_components   [A]      │
│         ↓                                                             │
│  Isomap(eigen_solver="dense").fit(legacysurvey)                       │
│         ↓                                                             │
│  full-spectrum audit: eigvalsh(B) on doubly-centred dist_matrix_      │
│         ↓                                                             │
│  ┌────────────────────────── GATE ───────────────────────────┐       │
│  │ PASS / MARGINAL → cache & continue   |   FAIL → stop, log  │       │
│  └──────────────────────────────────────────────────────────┘       │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │  notebooks/.cache/*.npz, *.joblib
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  02_decoder_and_curvature.ipynb        (fast, run many times)         │
├───────────────────────────────────────────────────────────────────────┤
│  load embedding_(10000,k), legacysurvey(10000,768) from cache         │
│  train MLP decoder f: R^k → R^768 (C2-smooth activation, torch seed)  │
│  torch.func.jacrev / hessian → first & second fundamental forms       │
│  → H_vec (10000,768), H_norm (10000,)                                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │  notebooks/.cache/curvature_*.npz
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│  03_regional_alignment.ipynb           (fast, run many times)         │
├───────────────────────────────────────────────────────────────────────┤
│  load H_norm + hsc/legacysurvey (same row_indices as notebook 01)     │
│  quantile-partition by ||H|| → high_mask, low_mask                    │
│  MKNN(hsc[mask], legacysurvey[mask]) + permutation null + bootstrap   │
│  → per-region alignment table, plots, written verdict                 │
└───────────────────────────────────────────────────────────────────────┘

[A] = EffDim integration point (genuine — see Integration Points)
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|-------------------------|
| `notebooks/01_manifold_and_gate.ipynb` | Stream+subsample, EffDim pre-audit, Isomap fit, full-spectrum eigenvalue audit, **gate decision** | Thin orchestration cells calling helpers in `notebooks/pu_manifold/` |
| `notebooks/02_decoder_and_curvature.ipynb` | Decoder training, curvature field via `torch.func`, iterated tuning loop | Same |
| `notebooks/03_regional_alignment.ipynb` | Quantile partition, MKNN + permutation null + bootstrap, final report cells | Same |
| `notebooks/pu_manifold/` (new, local, not installed) | Shared plumbing: subsampling, cache read/write, curvature math, MKNN stats | Plain `.py` modules imported by the notebooks |
| `notebooks/.cache/` (gitignored) | Checkpoint store for every expensive or iterated artifact | `.npz` for arrays, `.joblib` for fitted sklearn objects, `.pt`/`.json` for torch runs |
| `src/effdim/` (existing, **untouched**) | `compute_dim` panel — used as an input to notebook 01 only | No changes this milestone |

## Recommended Project Structure

```
notebooks/
├── 01_manifold_and_gate.ipynb        # steps 1–3 + GATE (slow, run rarely)
├── 02_decoder_and_curvature.ipynb    # steps 4–5 (fast, run often)
├── 03_regional_alignment.ipynb       # steps 6–7 (fast, run often)
├── pu_manifold/                      # local helper package, notebook-only
│   ├── __init__.py
│   ├── subsample.py                  # seeded streaming subsample, row-alignment asserts
│   ├── cache.py                      # config-hash-keyed npz/joblib load/save helpers
│   ├── curvature.py                  # torch.func fundamental-form / mean-curvature helpers
│   └── mknn.py                       # MKNN, permutation null, bootstrap CI
├── requirements-notebooks.txt        # torch, datasets, matplotlib pinned versions (NOT pyproject.toml)
└── .cache/                           # gitignored — see Caching Strategy
```

### Structure Rationale

- **Three notebooks, not one:** the real seam is cost/iteration, not "one notebook per numbered step." Steps 1–3 pay a one-time cost that must never be silently repeated (dataset streaming + an n=10,000 Isomap fit — dense eigendecomposition of a 10,000×10,000 matrix is on the order of tens of seconds to a few minutes even with multithreaded LAPACK). Steps 4–5 (decoder + curvature) form one tight iteration loop — you retrain, immediately inspect the curvature field, adjust architecture, repeat — so they stay together. Steps 6–7 (partition + MKNN stats) form a second, distinct iteration loop with its own knobs (quantile cutoffs, permutation count, bootstrap resamples) that don't require retraining the decoder, so they get their own notebook and their own cache boundary.
- **Three notebooks, not seven (one per step):** splitting decoder/curvature apart, or partition/MKNN apart, adds file-switching friction without adding a real cache boundary — curvature computation on an already-trained decoder is seconds, not minutes, so there's nothing expensive to protect by separating them further.
- **`pu_manifold/` helper package:** the row-alignment invariant (hsc/legacysurvey must never be sampled, sorted, or masked independently) is the single easiest thing to break by copy-pasting slightly different code into three notebooks. Putting subsampling, cache I/O, curvature math, and MKNN stats in one small, plain, importable module means there is exactly one implementation of each invariant-sensitive operation, and it shows up as an ordinary, diffable `.py` file in code review — not buried in notebook JSON. This module is **not** part of the installable `effdim` package and is never referenced from `src/effdim/` or `pyproject.toml`.
- **`.cache/` under `notebooks/`, not repo root:** keeps all generated, non-reviewable artifacts scoped to the one directory that already deals with heavy/ephemeral data, and mirrors the existing convention that `notebooks/` streams and derives rather than stores.

## Architectural Patterns

### Pattern 1: Config-hash-keyed checkpointing

**What:** Every expensive or iterated artifact is saved to `notebooks/.cache/` under a filename that embeds the run's seed and a short hash of its hyperparameters, computed once at the top of each notebook. Loading is "hash exists → load; else → compute → save."
**When to use:** Steps 1–3 (subsample, Isomap fit, full eigenspectrum) always; steps 4–5 (decoder, curvature) per architecture/hyperparameter variant during tuning.
**Trade-offs:** Slightly more code than "just re-run the cell," but the hash key makes stale-cache bugs self-evident (a config change produces a new filename rather than silently reusing an old array) instead of relying on notebook authors to remember to rename files by hand.

**Example:**
```python
# notebooks/pu_manifold/cache.py
import hashlib, json
from pathlib import Path
import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

def config_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

def npz_cache(name: str, cfg: dict, compute_fn):
    path = CACHE_DIR / f"{name}_{config_key(cfg)}.npz"
    if path.exists():
        return dict(np.load(path))
    arrays = compute_fn()
    np.savez(path, **arrays)
    return arrays
```

### Pattern 2: Gate as a first-class, branching notebook artifact

**What:** The negative-eigenvalue audit in notebook 01 is not a passive diagnostic plot — it is a decision cell that writes a small `gate_verdict.json` (`{"status": "PASS"|"MARGINAL"|"FAIL", "n_negative_eigs": ..., "negative_eig_ratio": ..., "elbow_k": ...}`) to the cache, and every downstream notebook checks it before running expensive cells.
**When to use:** Any pipeline where a later, expensive stage's validity depends on an earlier stage's structural assumption (here: Euclidean-embeddability).
**Trade-offs:** A small amount of ceremony (an explicit verdict object instead of "eyeball the plot and decide"), in exchange for making the FAIL branch a legitimate, documented, reproducible outcome rather than something a reader has to reconstruct from scattered markdown prose. See **Suggested Build Order** for what each branch does.

**Example:**
```python
# in notebook 02/03, first cell
verdict = json.loads((CACHE_DIR / "gate_verdict.json").read_text())
assert verdict["status"] in ("PASS", "MARGINAL"), (
    f"Isomap gate FAILED ({verdict}); do not train the decoder on this embedding. "
    "See notebook 01's Gate Failed section for remediation options."
)
```

### Pattern 3: Full-spectrum audit independent of the embedding routine

**What:** Do not read the negative-eigenvalue signal off `Isomap.kernel_pca_.eigenvalues_` (or the newer `sklearn.manifold.ClassicalMDS.eigenvalues_`). Both are constructed with `n_components` set to the small target embedding dimension (e.g. 2–15) — they return only the top-`n_components` algebraically-largest eigenvalues of the double-centred Gram matrix, which by construction **cannot show the negative tail**. Instead, use `isomap.dist_matrix_` (the full n×n geodesic distance matrix, which Isomap retains as a public fitted attribute) to build the classical-MDS Gram matrix directly and get the *entire* eigenvalue spectrum:

```python
import numpy as np
from scipy.linalg import eigvalsh

D2 = isomap.dist_matrix_.astype(np.float64) ** 2
n = D2.shape[0]
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ D2 @ J
eigs = eigvalsh(B)[::-1]   # full spectrum, descending, includes the negative tail
```
**When to use:** Any classical-MDS / Isomap validity audit that needs to detect non-Euclidean-ness, not just produce an embedding.
**Trade-offs:** `eigvalsh` on a dense 10,000×10,000 symmetric matrix is itself a real, one-time cost (LAPACK `syevd`, roughly comparable in order of magnitude to the Isomap fit itself) — cache the resulting ~80KB eigenvalue array immediately (Pattern 1) so this is paid exactly once per subsample seed, never during steps 4–7 iteration.

### Pattern 4: Row-alignment as an explicit, asserted invariant

**What:** `hsc` and `legacysurvey` arrays (and every mask/subset derived from them) share one canonical row order established once, at subsample time, and never independently re-sorted, re-sampled, or re-filtered. Every function that touches both arrays takes and returns them together (or takes a shared index/mask), never as two independently-obtained objects.
**When to use:** Throughout — this is the single correctness invariant the entire pipeline depends on, because the dataset has no `object_id` join key.
**Trade-offs:** None real; the discipline costs nothing and the failure mode it prevents (silently wrong MKNN numbers with no error raised) is invisible without it.

## Data Flow

### Step-by-step shapes

| Step | Input | Output | Shape |
|------|-------|--------|-------|
| 1. Stream + subsample | HF streaming iterator over 101,725 rows | `hsc`, `legacysurvey`, `row_indices` | `(10000,768) f32` ×2, `(10000,) i64` |
| 2. `compute_dim` pre-audit [EffDim] | `legacysurvey` | ED panel dict | scalars |
| 2. Isomap fit | `legacysurvey`, `n_components=k` | `embedding_`, `dist_matrix_` | `(10000,k) f32`, `(10000,10000) f32/f64` |
| 3. Full-spectrum audit | `dist_matrix_` | `eigs` (full), `gate_verdict` | `(10000,) f64`, small JSON |
| 4. Decoder training | `embedding_` (X), `legacysurvey` (Y) | decoder weights | torch state_dict |
| 5. Curvature | decoder, `embedding_` | `H_vec`, `H_norm` | `(10000,768) f32`, `(10000,) f32` |
| 6. Partition | `H_norm` | `high_mask`, `low_mask` | `(10000,) bool` ×2 |
| 7. MKNN per region | `hsc[mask]`, `legacysurvey[mask]` | alignment stats | scalars + null distributions |

### Where alignment can silently break

- **Two independent samples of the same size.** Sampling `hsc` and `legacysurvey` via two separate calls into the stream (even with the same seed) is not the same as sampling one shared `row_indices` set and reading both columns off the *same* selected rows. Always select rows once, read both columns from each selected row.
- **Any post-hoc sort/shuffle on one array only.** If `legacysurvey` gets reordered for a nearest-neighbour index build (e.g. FAISS index construction, or a library that returns permuted results) but `hsc` doesn't get the same permutation applied, the pairing is silently destroyed with no error raised — there is no `object_id` to catch it.
- **Deriving two separate pandas objects instead of one.** Keep `hsc`/`legacysurvey`/`row_indices` as columns of one bundle (one dict, one `.npz`, or one dataframe) so any mask or `.iloc` naturally applies to all three together, rather than as three independently-manipulable variables.
- **Boolean masking is safe; independent re-indexing is not.** `hsc[mask]` and `legacysurvey[mask]` with the *same* `mask` array preserve relative row order and pairing. The risk is entirely in operations that touch only one of the two arrays.
- **Recommended guard:** an assertion cell immediately after subsampling and after every mask operation: `assert hsc.shape[0] == legacysurvey.shape[0] == row_indices.shape[0]`, plus (cheap, worth adding) re-fetching a handful of the sampled `row_indices` from the stream at the very end of notebook 01 and diff-checking against the cached arrays, to catch a drift bug before it propagates into two more notebooks.

## Caching Strategy

Directory: `notebooks/.cache/` (gitignored — see Gitignore below). Format choice is deliberate: **npz for arrays, joblib for fitted sklearn/torch objects, JSON for small scalar metadata** — not parquet. The payloads here are fixed-shape, homogeneous-dtype float arrays (embeddings, curvature fields, eigenvalues); parquet's value proposition (columnar heterogeneous schemas, nullable columns, predicate pushdown on partitions larger than memory) doesn't apply, and it would add a `pyarrow` dependency for no benefit. npz is numpy-native, requires nothing beyond what's already needed, and loads with zero schema inference.

Concrete artifacts, in pipeline order:

| File | Contents | Written by | Approx. size |
|------|----------|------------|--------------|
| `subsample_{seed}.npz` | `hsc`, `legacysurvey`, `row_indices` | Notebook 01 | ~60 MB |
| `effdim_panel_{seed}.json` | `compute_dim(legacysurvey)` result dict | Notebook 01 | <1 KB |
| `isomap_{seed}_{cfg}.joblib` | fitted `Isomap` object (includes `dist_matrix_`, `embedding_`, `nbrs_`, `kernel_pca_`) | Notebook 01 | ~800 MB–1 GB (dominated by `dist_matrix_`) |
| `mds_eigenspectrum_{seed}_{cfg}.npz` | full-spectrum `eigs` (descending) | Notebook 01 | ~80 KB |
| `gate_verdict_{seed}_{cfg}.json` | PASS/MARGINAL/FAIL + supporting numbers | Notebook 01 | <1 KB |
| `decoder_{run_id}.pt` + `decoder_{run_id}.json` | torch `state_dict` + architecture/hyperparam config | Notebook 02 | a few MB |
| `curvature_{run_id}.npz` | `H_norm` (always), `H_vec` (optional, larger) | Notebook 02 | ~40 KB (`H_norm` only) to ~30 MB (with `H_vec`) |
| `mknn_results_{run_id}.json` | per-region observed MKNN, null distribution summary, bootstrap CI | Notebook 03 | <10 KB |

Notes:
- The `isomap_*.joblib` file is the one genuinely large artifact (~1 GB). That is expected and acceptable for a gitignored local cache — it is exactly what stands between "re-run a multi-minute dense eigendecomposition" and "load in a second." Flag it explicitly so the artifact doesn't get mistaken for a bug or synced somewhere it shouldn't be (cloud-synced home directories, etc.).
- `H_vec` (full curvature vectors, 10000×768) does not need to be retained by default — only `H_norm` is required for partitioning in step 6. Cache it only when actively debugging the curvature computation itself.
- `run_id` in notebooks 02/03 should itself be the config hash (Pattern 1), not a manually incremented counter — this is what lets a stale/mismatched cache be detected structurally rather than by convention.

## Integration Points

### `effdim.compute_dim` — genuine primary use, secondary use is real but weaker

**Primary (genuine): estimate `n_components` before calling Isomap.** Isomap requires an explicit `n_components`, and guessing 2 or 3 arbitrarily is exactly the problem EffDim exists to avoid. Running `compute_dim(legacysurvey)` on the raw 10,000×768 array before fitting Isomap and using the estimator panel (particularly the geometric estimators — TwoNN, MLE, ESS, TLE, GMST, DANCo, MiND — since they are intrinsic/nonlinear-aware, unlike the ambient-linear spectral panel) to pick a data-informed `n_components` is a direct, natural application of the library to its own stated purpose. This should be the first substantive analysis cell in notebook 01, before `Isomap.fit`.

**Secondary (genuine, but read the caveat): compare the panel against the eigenspectrum elbow.** Once notebook 01 has both `compute_dim`'s panel and the full MDS eigenspectrum from Pattern 3, comparing them costs nothing extra and materially strengthens the gate's PASS/MARGINAL/FAIL narrative — agreement between an estimator that never saw the geodesic structure and one that's built entirely from it is stronger corroborating evidence than the eigenspectrum shape alone. **Caveat, stated honestly:** this comparison is not apples-to-apples across the whole panel. The Isomap/MDS eigenspectrum operates on *geodesic* distances, so methodologically it sits closer to the geometric estimators (local, nonlinear-aware) than to the spectral estimators (global, linear/PCA-based — participation ratio, stable rank, PCA-95%, cumulative eigenvalue ratio). Expect the spectral panel to read noticeably *higher* than both the geometric panel and the Isomap elbow whenever the manifold is meaningfully curved — a curved low-dimensional manifold can still need many linear components to represent faithfully in a 768-d ambient space. That disagreement is not a bug in the comparison; it is, incidentally, direct supporting evidence for this milestone's own premise (linear/spectral dimension estimates inflate under curvature) and worth calling out explicitly in the notebook's narrative rather than treated as noise.

**Explicitly forced, and out of scope: per-region `compute_dim` on the high-/low-curvature subsets.** It is technically easy (n≈2,500–5,000 points per region is well within `compute_dim`'s range) but has no role in this milestone's actual gate or hypothesis — the milestone's regional test is MKNN alignment, not a third, orthogonal ED comparison. Adding it would dilute the notebook's narrative without a driving question. If curiosity demands it, it belongs in an optional appendix cell, clearly labeled as exploratory, not as part of the required pipeline.

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|----------------|-------|
| `notebooks/*.ipynb` ↔ `src/effdim` | `import effdim; effdim.compute_dim(...)` | Read-only use of the installed package; **no changes to `src/effdim/` this milestone** |
| `notebooks/*.ipynb` ↔ `notebooks/pu_manifold/` | plain Python import (`sys.path`-relative or notebook run from `notebooks/`) | New, notebook-scoped helper package; never imported from `src/effdim/` |
| Notebook 01 ↔ Notebook 02 | `notebooks/.cache/*.npz`, `*.joblib` | One-directional; 02 never writes into 01's cache namespace |
| Notebook 02 ↔ Notebook 03 | `notebooks/.cache/curvature_*.npz` | Same pattern |
| Notebook 03 ↔ Notebook 01's cache | `subsample_{seed}.npz` (for `hsc`/`legacysurvey`/`row_indices`) | Notebook 03 depends on caches from *both* 01 and 02 — this is fine; caches are keyed by pipeline stage, not by notebook |

## Determinism and Reproducibility

| Source of nondeterminism | Where it enters | How to pin it |
|---|---|---|
| Subsample selection | Step 1 | Single `numpy.random.default_rng(seed)` used once to pick `row_indices`; seed recorded in a markdown+code cell at the top of notebook 01 and embedded in every downstream cache filename |
| Isomap eigensolver | Step 2 | `Isomap` has **no `random_state` parameter**. `eigen_solver="auto"`/`"arpack"` uses ARPACK's Lanczos iteration with a random starting vector unless pinned, which risks run-to-run sign/axis differences on top of the inherent sign ambiguity of any eigendecomposition. At n=10,000, `eigen_solver="dense"` is fully deterministic (LAPACK, no RNG) and computationally feasible (one-time cost, then cached) — pin it explicitly rather than leaving `"auto"`, whose selection logic is not part of the documented public contract and could differ across sklearn versions |
| Embedding axis sign/rotation | Step 2 (inherent to any eigendecomposition-based embedding, even fully deterministic ones) | Not a bug to fix — document it. Downstream steps (curvature magnitude `‖H‖`, MKNN) are invariant to a global sign flip or rotation of the coordinate chart, so this doesn't affect final results, but it's worth one line in the notebook so a reader isn't confused by axis-flipped plots across reruns |
| Torch decoder init + training | Steps 4–5 | `torch.manual_seed(seed)` before model construction and training; if training ever moves to GPU, additionally note (not necessarily enforce) that `torch.use_deterministic_algorithms(True)` and disabling cuDNN benchmarking are needed for bit-exact reruns — CPU training (plausible at this scale: 10,000 points, a small MLP) sidesteps this class of nondeterminism entirely |
| Permutation null / bootstrap CI (step 7) | Notebook 03 | Its own seeded `default_rng`, recorded and cached alongside the results JSON — this is the one place nondeterminism is the *point* (many permutation draws), so record the seed and the draw count, not attempt bit-exact reproducibility of individual draws |

**Recording pattern:** each notebook's first code cell prints/records a small reproducibility header — seed(s) used, `sklearn`/`numpy`/`torch`/`faiss` versions, and (via `!git rev-parse --short HEAD` or `subprocess`) the repo commit — and the same information is embedded in the cache filename via the config-hash pattern, so a cached artifact and the notebook state that produced it can never silently drift apart.

## Suggested Build Order

The build order follows the gate, not the step numbering — notebook 02/03 work is simply undefined until notebook 01 resolves.

1. **Notebook 01, steps 1–2:** subsample + row-alignment asserts, `compute_dim` pre-audit, Isomap fit (`eigen_solver="dense"`, `n_components` from the EffDim panel). Cache both.
2. **Notebook 01, step 3 — the gate:** full-spectrum audit (Pattern 3), elbow-vs-panel comparison, explicit `gate_verdict.json`.
   - **PASS** (large positive eigenvalues, steep dropoff, negligible negative tail) → proceed to notebook 02.
   - **MARGINAL** (some negative eigenvalues present but small relative to the leading positive eigenvalues) → proceed, but the caveat travels downstream: the magnitude of non-Euclidean-ness gets carried into notebook 03's final write-up rather than silently dropped once the gate is passed.
   - **FAIL** (negative eigenvalues comparable to or exceeding the positive spectrum, no clean elbow) → **stop here.** Notebooks 02/03 are not run against this subsample/config. Notebook 01 ends with a written verdict section enumerating remediation options for a human to choose (increase `n_neighbors` for better graph connectivity, try a different model/config from the `pu-embeddings` family, fall back to a non-metric MDS variant, restrict to a denser sub-region) — these are research judgment calls a notebook should surface, not auto-retry in a loop. A documented FAIL is a legitimate, complete v1.1 deliverable in its own right, consistent with PROJECT.md's existing acceptance that a null MKNN result is a valid, reportable outcome — the same standard applies to the gate.
3. **Notebook 02 (only if PASS/MARGINAL):** decoder architecture/hyperparameter iteration, curvature computation, cache each run by config hash. This is the "tune freely" loop — every iteration loads from notebook 01's cache and never re-streams or re-fits Isomap.
4. **Notebook 03 (only once a notebook-02 run is chosen as final):** partition, MKNN + permutation null + bootstrap, per-region report, tying the final write-up back to the gate's PASS/MARGINAL caveat if applicable.

Within GSD terms, this maps naturally to phase/plan granularity: notebook 01 is one plan with a hard verification gate (the PASS/MARGINAL/FAIL cell *is* the plan's UAT criterion), notebooks 02 and 03 are separate plans that explicitly depend on 01's gate outcome and should not be planned in detail until 01's result is known.

## Anti-Patterns

### Anti-Pattern 1: One monolithic notebook, "Restart & Run All" as the iteration loop

**What people do:** Put steps 1–7 in a single notebook and re-run the whole thing (or scroll up and re-run from the top) whenever a decoder hyperparameter changes.
**Why it's wrong:** Repays the dataset stream + n=10,000 Isomap fit (minutes) on every architecture tweak, and accumulates stale kernel state (leftover torch tensors, large in-memory arrays) across a long-lived session, which is its own source of hard-to-reproduce bugs.
**Do this instead:** Cache the boundary after step 3 (Pattern 1) and iterate only in notebook 02/03, which load from cache in a fresh kernel each time.

### Anti-Pattern 2: Reading the gate off `Isomap.kernel_pca_.eigenvalues_`

**What people do:** Plot `isomap.kernel_pca_.eigenvalues_` (or `sklearn.manifold.ClassicalMDS.eigenvalues_`) and look for negative values there.
**Why it's wrong:** Both are constructed with `n_components` fixed to the small target embedding dimension — they return only the top-k algebraically-largest eigenvalues by design, so the negative tail is structurally invisible no matter how bad the non-Euclidean-ness actually is. A "clean" plot from this attribute proves nothing about non-Euclidean-ness.
**Do this instead:** Recompute the full spectrum from `isomap.dist_matrix_` via `scipy.linalg.eigvalsh` (Pattern 3), cache it once, and audit the full array.

### Anti-Pattern 3: Sampling `hsc` and `legacysurvey` independently

**What people do:** Two separate `.shuffle(seed=X).select(range(10000))`-style calls, or two separate reservoir samples, one per column.
**Why it's wrong:** With no `object_id`, there is nothing to catch the resulting misalignment — MKNN numbers come out wrong silently, with no exception raised anywhere.
**Do this instead:** Select one shared `row_indices` set from a single pass over the stream; read both columns off the same selected rows (Pattern 4).

### Anti-Pattern 4: Promoting the curvature operator into `src/effdim/` mid-milestone

**What people do:** Notice the curvature math is reusable and start moving `notebooks/pu_manifold/curvature.py` into `src/effdim/` "while we're at it."
**Why it's wrong:** Explicitly out of scope for v1.1 per PROJECT.md — promotion needs unit tests against known-curvature surfaces (a sphere, a saddle) as its own milestone, and pulling torch into `src/effdim/` (even optionally) contradicts the "core stays installable without a deep-learning stack" constraint unless done deliberately as an opt-in extra.
**Do this instead:** Keep it in `notebooks/pu_manifold/`. If the method proves out, that's a decision for the *next* milestone, made with the benefit of this milestone's results.

## Gitignore and Notebook Output Hygiene

**Add to root `.gitignore`:**
```
# Notebook artifacts
.ipynb_checkpoints/
notebooks/.cache/
```
(`.ipynb_checkpoints/` is currently missing from `.gitignore` and will otherwise leak Jupyter's autosave directories into the repo the first time a notebook is opened.)

**Never gitignore, and never commit into the repo tree:** raw dataset shards or parquet files from `UniverseTBD/pu-embeddings` — consistent with PROJECT.md's existing "notebooks stream from source" constraint. Point the HF `datasets` cache at its normal out-of-repo default (`~/.cache/huggingface`); don't redirect `HF_HOME`/`HF_DATASETS_CACHE` into the repo tree, which would create exactly the temptation `.gitignore` then has to defend against.

**Notebook outputs — strip iteration notebooks, keep final deliverable outputs, opinionated recommendation:**
- Notebook 02 (decoder/curvature tuning) will be re-run dozens of times during architecture search; its intermediate outputs (loss curves from abandoned configs) are not meaningful deliverable content and would otherwise churn the diff on every commit. Clear its outputs before committing (`jupyter nbconvert --clear-output --inplace notebooks/02_decoder_and_curvature.ipynb`).
- Notebooks 01 and 03 are the actual deliverable artifacts a reader opens on GitHub without executing anything (this matches the existing ROADMAP Phase 3 success criterion: "a reader can open a notebook and reproduce an analysis"). Once each is stable, do a deliberate `Restart & Run All` and commit **with outputs intact** — a fully-stripped gate notebook or fully-stripped final-results notebook is a materially worse deliverable than one showing its own eigenspectrum plot and MKNN table.
- This repo has no pre-commit framework or `.gitattributes` filter configured today, and the project is otherwise low-ceremony (no CI enforcement mentioned as a requirement). A manual `nbconvert --clear-output` step before committing notebook 02 is the right amount of process for now. If output-diff noise becomes a real problem later (e.g. multiple contributors iterating in parallel), `nbstripout --install` as a repo-scoped git filter is the standard upgrade path — worth flagging as an option, not adopting pre-emptively for a milestone this size.

## Sources

- [sklearn.manifold.Isomap — official docs](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html) (HIGH — confirms `dist_matrix_`, `kernel_pca_`, `embedding_`, `nbrs_` fitted attributes; no `random_state` parameter; `eigen_solver` options `auto`/`arpack`/`dense`)
- [scikit-learn/scikit-learn `sklearn/manifold/_isomap.py` source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_isomap.py) (HIGH — confirms `kernel_pca_` is constructed with `n_components` equal to the target embedding dimension, i.e. its `eigenvalues_` is truncated and cannot show a negative tail; confirms `dist_matrix_` computed via `shortest_path` and `embedding_` via `kernel_pca_.fit_transform(-0.5 * dist_matrix_**2)`)
- [sklearn.manifold.ClassicalMDS — official docs](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.ClassicalMDS.html) (HIGH — confirms this newer estimator has the same `n_components`-truncated `eigenvalues_` limitation, so it does not obviate the need for a manual full-spectrum `eigvalsh` audit)
- [GitHub issue #31246, scikit-learn — "Faster Eigen Decomposition for Isomap & KernelPCA"](https://github.com/scikit-learn/scikit-learn/issues/31246) (MEDIUM — corroborates that Isomap's Gram matrix is not PSD in general and that randomized eigensolvers are unsuited to it, reinforcing why `eigen_solver="dense"` is the right deterministic choice at n=10,000)
- Local repo inspection: `/home/akagi/Documents/Projects/EffDim/src/effdim/api.py`, `.gitignore`, `.planning/PROJECT.md`, `.planning/ROADMAP.md` (HIGH — primary source for existing package boundaries, current milestone constraints, and repo conventions)

---
*Architecture research for: EffDim v1.1 "PU Manifold Curvature" notebook pipeline*
*Researched: 2026-07-29*
