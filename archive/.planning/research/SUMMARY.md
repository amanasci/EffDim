# Project Research Summary

**Project:** EffDim v1.1 "PU Manifold Curvature"
**Domain:** Applied differential-geometry analysis notebook — manifold reconstruction (Isomap) → learned smooth decoder → analytic curvature (torch.func) → curvature-stratified representational-alignment probe (MKNN)
**Researched:** 2026-07-29
**Confidence:** MEDIUM-HIGH (stack facts and sklearn internals verified against primary sources; the falsification protocol for decoder-curvature and the density-confound analysis are original methodological synthesis grounded in standard practice, not lifted from a single citable source)

## Executive Summary

This milestone is not a conventional software feature — it is a six-to-seven-step scientific pipeline that reconstructs the geometric shape of a foundation-model embedding manifold and asks whether crossmodal representational alignment (MKNN, per Chechik et al. 2010, as used in the origin paper arXiv:2509.19453) varies with local curvature. Every research thread converges on the same structural finding: **the pipeline has more ways to silently produce a plausible-looking but wrong number than ways to fail loudly.** sklearn's `Isomap` bridges disconnected graphs without raising, truncates its own eigenvalue attribute so the diagnostic it's needed for is structurally unanswerable from the obvious API, and neural-network curvature is fundamentally ambiguous between "real manifold shape" and "decoder interpolation artifact" until proven otherwise by a synthetic control. The stack itself is small and low-risk (torch 2.13 CPU, datasets 5.0.1, matplotlib — all notebook-only, `%pip install`-able, nothing touches `pyproject.toml`), so implementation risk concentrates almost entirely in correctness of the geometry/statistics, not in tooling.

The recommended approach is a three-notebook structure with one hard, artifact-producing gate: notebook 01 (subsample → Isomap → full eigenspectrum audit → PASS/MARGINAL/FAIL verdict) must complete and pass before notebook 02 (decoder + curvature, iterated freely against a cache) or notebook 03 (regional MKNN, the payoff) are touched. This mirrors the single most important ordering constraint surfaced by every research file independently: upstream structural validity (Euclidean-embeddability of the geodesic distances, decoder generalization, density-confound magnitude) must be established using upstream-only diagnostics *before* anyone looks at the downstream MKNN number, or the whole exercise degenerates into a garden-of-forking-paths hunt for a "nice" result on a headline effect (Legacy Survey crossmodal MKNN, 0.4–2% in the origin paper) that has very little statistical headroom to begin with.

The key risks, in descending order of how badly they invalidate the result if missed: (1) reading the negative-eigenvalue diagnostic off `Isomap.kernel_pca_.eigenvalues_`, which is truncated to `n_components` and structurally cannot show what it's being asked to show — this must be recomputed from `isomap.dist_matrix_` by hand; (2) reporting curvature as if it were the familiar scalar Gaussian curvature, when at codimension 768−d it is a *vector* in the normal space and the only legitimate reportable scalar is `‖H‖`; (3) treating decoder curvature as ground truth without a falsification protocol, since any smooth interpolant's second derivative is unconstrained between sparse training points — the synthetic control manifold (flat/sphere/saddle, same architecture, same d and n=768) is what turns "is this the data's curvature or the decoder's" from an assumption into a tested claim; (4) conflating "high curvature" with "sparse/undersampled," which can mechanically produce a spurious curvature→alignment relationship through at least three independent channels (decoder extrapolation, geodesic-distance inflation, and MKNN's own sensitivity to local density/hubness) that all push in the same direction; and (5) the Python-floor mismatch (real floor 3.11, `pyproject.toml` says ≥3.8) which is notebook-scoped and requires no packaging change but must be stated explicitly in notebook 01 so a reader on an old kernel isn't confused by a hard, unhelpful pip failure.

## Key Findings

### Recommended Stack

Everything new in this milestone is notebook-only and installs via `%pip install`, never touching `pyproject.toml`: `torch==2.13.0` (CPU wheel, from `download.pytorch.org/whl/cpu` to avoid pulling multi-GB CUDA runtime deps), `datasets==5.0.1` (per-config parquet routing means only the ~553 MiB `legacysurvey_dinov3_vitb16.parquet` file is ever fetched, streaming or not — `load_dataset(...).shuffle(seed=...).select(range(10000))` gives a true uniform sample with no buffer-size tuning), and `matplotlib==3.11.1` (static curvature-field scatter plots, `mpl_toolkits.mplot3d` included). `torch.func.jacrev`/`jacfwd`/`hessian`/`vmap` are stable since torch 2.0 — no `functorch` import needed. `faiss.IndexFlatL2` (already a core `effdim` dependency) is preferred over `sklearn.neighbors.NearestNeighbors` for the MKNN bootstrap's repeated k-NN calls. TDA libraries (`gudhi`/`ripser`), UMAP, and heavy DL-infra packages (`pytorch-lightning`, `accelerate`) are explicitly out — the curvature computation is analytic (Jacobian/Hessian via `torch.func`), not topological, and the decoder is a hand-written training loop on a small MLP.

**Core technologies:**
- `torch==2.13.0` (CPU): MLP decoder + `torch.func` curvature autodiff — zero GPU requirement, notebook-scoped only
- `datasets==5.0.1`: pulls exactly the `legacysurvey_dinov3_vitb16` config, no accidental full-repo download
- `matplotlib==3.11.1`: curvature-field and eigenspectrum plots, already includes 3-D scatter support
- `scipy.linalg.eigvalsh` (already core): the only correct way to get the full classical-MDS eigenspectrum including the negative tail

**Practical floor:** Python 3.11 is the real minimum for this notebook (`torch` CPU wheels stop at cp310–cp312, and `scikit-learn` 1.9.0 and `faiss-cpu` 1.14.3 both require ≥3.10/3.11) — one full minor version above `pyproject.toml`'s declared `>=3.8`. This is a pre-existing drift in core deps, not something this milestone introduces, but the notebook must state its own floor explicitly since it inherits it.

### Expected Features

This is a linear analysis pipeline, not a feature set with alternatives — "table stakes" means *the result is not scientifically credible without it*.

**Must have (P1, table stakes):**
- Connectivity check (`scipy.sparse.csgraph.connected_components == 1`) before trusting any Isomap fit — sklearn silently bridges disconnected graphs otherwise
- `n_neighbors` short-circuit sensitivity sweep (5/8/10/15/20/30), not a single fit-and-move-on
- Full classical-MDS eigenspectrum (not `kernel_pca_.eigenvalues_`) + `|λ_min|/λ_max` non-Euclideanity ratio + residual-variance-vs-dimension elbow — this *is* the milestone's stated Step 2/3 deliverable
- C²-smooth decoder, held-out train/val split, reconstruction metric (relative L2 + per-dim R²) reported before any curvature is trusted
- Mean curvature vector norm `‖H‖` via exact `torch.func` autodiff (never finite differences, never Gaussian curvature/principal curvatures — those are category errors at this codimension)
- Density-checked quantile split into high/low curvature regions (never a fixed absolute threshold, never 1-D k-means)
- Per-region MKNN vs a region-specific permutation null (regions differ in n, so each needs its own null — this correctness requirement stands regardless of P2 scope), with bootstrap CIs

**Should have (P2, cheap add-ons within v1.1 scope, per user's scope decisions):**
- k-sensitivity curve (k = 5, 10, 20, 50) for the regional MKNN comparison — IS in v1.1
- Reporting a null-subtracted "excess MKNN" statistic — explicitly OUT of v1.1 reporting scope (the per-region null itself is still required, just not subtracted as a headline statistic)

**Defer (explicitly out of v1.1):**
- Intramodal MKNN across a model-size ladder (needs a second model size — deferred)
- Promoting the curvature operator into `src/effdim/` (needs its own unit-test milestone against known-curvature synthetic surfaces)
- Comparing MKNN against other alignment metrics (CKA, mutual information)

### Architecture Approach

Three notebooks, two cache boundaries, one hard gate — the seam is cost/iteration, not step-count. `notebooks/01_manifold_and_gate.ipynb` (slow, run rarely: stream+subsample, Isomap fit at `eigen_solver="dense"` for determinism, full-spectrum audit, writes a `gate_verdict.json` with PASS/MARGINAL/FAIL) must complete before `notebooks/02_decoder_and_curvature.ipynb` (fast, iterated freely: decoder training + curvature, loads from 01's cache, never re-streams or re-fits Isomap) or `notebooks/03_regional_alignment.ipynb` (fast, iterated freely: quantile partition, MKNN + null + bootstrap, final report) run. A shared `notebooks/pu_manifold/` local helper package (never imported from `src/effdim/`, never installed) holds the one canonical implementation of subsampling, cache I/O, curvature math, and MKNN stats — this is deliberately where the row-alignment invariant (hsc/legacysurvey share one canonical row order, established once, never independently re-sorted or re-sampled) lives as code, not convention. Caching is config-hash-keyed (`notebooks/.cache/`, gitignored): npz for arrays, joblib for fitted sklearn/torch objects, JSON for scalar metadata — the `isomap_*.joblib` file (~800 MB–1 GB, dominated by `dist_matrix_`) is the one genuinely large artifact and is expected.

**Major components:**
1. `notebooks/01_manifold_and_gate.ipynb` — subsample, EffDim pre-audit (`compute_dim` on raw embeddings to pick Isomap's `n_components` rather than guessing), Isomap fit, full-spectrum audit, gate decision (a first-class branching artifact, not a passive plot)
2. `notebooks/02_decoder_and_curvature.ipynb` — decoder training with held-out validation, `torch.func` Jacobian/Hessian curvature field, synthetic-control-manifold falsification test
3. `notebooks/03_regional_alignment.ipynb` — density-checked quantile partition, per-region MKNN + permutation null + bootstrap CI, k-sensitivity curve, final written verdict

### Critical Pitfalls

1. **`Isomap.kernel_pca_.eigenvalues_` is truncated to `n_components` and cannot reveal the negative eigenvalue tail** — recompute the full spectrum by hand from `isomap.dist_matrix_` (double-center, `scipy.linalg.eigvalsh`). Reading the truncated attribute and reporting "no negative eigenvalues found" is a false negative by construction, not a finding.
2. **sklearn silently bridges disconnected k-NN graphs** (`_fix_connected_components`), fabricating edges. Must independently verify `connected_components == 1` before trusting the fit — a warning in notebook output is not a substitute for an assertion.
3. **Curvature is a property of the fitted decoder until proven otherwise** — an overfit decoder manufactures large `‖H‖` in sparse gaps; an underfit one erases real curvature. The mandated defense is the synthetic control manifold (flat/sphere/saddle, same decoder architecture, matched d and n=768), which doubles as the specificity/false-positive-rate check: if the control produces comparable-magnitude, similarly separable curvature to the real data, the whole real-data signal is decoder noise.
4. **The density confound is the single most dangerous risk in the pipeline** — sparse/edge regions get mechanically higher apparent curvature (decoder underconstrained), inflated geodesic distances (finite-sample zigzag), *and* noisier/lower MKNN (k-NN-based metrics are density- and hubness-sensitive) through independent channels that all point the same direction, which can manufacture a "high curvature → low alignment" result with zero true representational content. Must be measured (Spearman correlation of `‖H‖` vs. local density) and controlled for before the headline MKNN-by-region comparison is treated as interpretable.
5. **Statistical headroom is thin** (Legacy Survey crossmodal MKNN is 0.4–2% in the origin paper) — the curvature quantile split, `n_neighbors`, decoder architecture/seed, and k must all be frozen using upstream-only diagnostics *before* anyone looks at regional MKNN, or post-hoc tuning to make the result "look cleaner" invalidates any reported significance. A null regional result is a plausible, legitimate, explicitly-anticipated v1.1 outcome.

## Implications for Roadmap

Based on research, suggested phase structure — **this directly maps to the three-notebook architecture, and the gate is the single hardest ordering constraint in the whole roadmap.**

### Phase 1: Manifold Reconstruction & Validity Gate
**Rationale:** Everything downstream (decoder input dimension, curvature shapes, MKNN region definitions) is a function of the Isomap fit and its dimension `d`. This is also the slowest, least-iterated step (n=10,000 dense geodesic matrix, one-time cost of minutes, ~1 GB persisted) — it must be cached and gated before any downstream iteration begins, per the architecture research's central "config-hash-keyed checkpoint + gate" pattern.
**Delivers:** Row-aligned `hsc`/`legacysurvey`/`row_indices` subsample cache; connectivity + short-circuit `n_neighbors` sweep table; `compute_dim` pre-audit informing Isomap's `n_components`; Isomap fit at `eigen_solver="dense"`; full classical-MDS eigenspectrum computed by hand from `dist_matrix_` (not `kernel_pca_.eigenvalues_`); negative-eigenvalue ratio + residual-variance elbow; explicit `gate_verdict.json` (PASS/MARGINAL/FAIL) as a first-class artifact.
**Addresses:** FEATURES.md P1 items — connectivity check, short-circuit sensitivity, full eigenspectrum + negative-eigenvalue statistic, residual-variance elbow, frozen `d`.
**Avoids:** Pitfalls 1 (silent graph bridging), 2 (short-circuit edges), 3 (truncated eigenvalue attribute), 4 (Euclidean/cosine metric mismatch — norm-histogram check belongs here, at data-loading time), 5 (subsampling density inflation — seed discipline set here, stability re-run validated later).
**Hard gate:** A FAIL verdict stops the milestone here — a documented FAIL (with remediation options enumerated for human judgment) is itself a legitimate, complete deliverable, consistent with the project's existing acceptance of a null MKNN result as valid. Phase 2 work should not be planned in detail until Phase 1's gate outcome is known.

### Phase 2: Decoder, Curvature Field & Falsification Protocol
**Rationale:** Fast, heavily-iterated work (architecture/hyperparameter tuning) that only becomes well-defined once Phase 1's `d` and cached Isomap coordinates exist. This phase must build its own validity gate — curvature is only meaningful conditional on decoder validity — before Phase 3 touches it.
**Delivers:** C²-smooth MLP decoder trained with held-out split, reconstruction metrics (relative L2, per-dim R²); `torch.func` Jacobian/Hessian → first/second fundamental forms → mean curvature vector field, reported as `‖H‖` only (never Gaussian curvature/principal curvatures); metric-tensor conditioning check (`cond(g)`, near-singular points flagged/excluded); synthetic control manifold (flat plane, sphere, saddle at matched d, n=768) run through the identical pipeline as the falsification test.
**Uses:** `torch==2.13.0` CPU, `torch.func` (jacrev/hessian/vmap), decoder architecture per PROJECT.md's C²-smooth-activation decision.
**Implements:** `notebooks/02_decoder_and_curvature.ipynb` + `pu_manifold/curvature.py`.
**Must not skip:** held-out-error-vs-curvature correlation check, activation/Hessian sanity check (assert non-zero, finite second derivatives away from training nodes), and the synthetic-control comparison — these gate whether Phase 3 is even meaningful to run.

### Phase 3: Density-Confound Diagnostics & Regional MKNN
**Rationale:** The payoff step and the one with the least statistical headroom (0.4–2% baseline signal) — must be run last, with all upstream hyperparameters (n_neighbors, d, decoder config, curvature quantile threshold) already frozen by Phases 1–2's own diagnostics, never adjusted to make this phase's result look cleaner.
**Delivers:** Local-density proxy per point + Spearman correlation against `‖H‖` (reported explicitly, before any regional split is trusted); density-checked quantile partition into high/low curvature regions; per-region MKNN vs. region-specific permutation null (each region gets its own null — this correctness requirement holds regardless of the P2 excess-MKNN reporting descope); bootstrap CIs; k-sensitivity curve (k=5,10,20,50); pre-registered (not post-hoc-tuned) quantile threshold; honest reporting of a null result if that's what the data shows.
**Delivers explicitly NOT:** a null-subtracted "excess MKNN" reporting statistic (descoped by user decision) — raw per-region MKNN + its own null + CI is the reported quantity.
**Implements:** `notebooks/03_regional_alignment.ipynb` + `pu_manifold/mknn.py`.

### Phase Ordering Rationale

- **The gate after Phase 1 is not optional and not soft.** Every research file independently converges on this: STACK.md flags `dist_matrix_` vs. truncated `kernel_pca_.eigenvalues_` as an implementation detail with roadmap consequences; ARCHITECTURE.md makes it an explicit branching artifact (`gate_verdict.json`) with a documented FAIL path; PITFALLS.md ranks the truncated-eigenvalue mistake as Pitfall 3, addressed in "the eigenspectrum-audit phase — this is the phase's core deliverable." A phase plan for Phase 1 should treat the gate cell as its UAT/verification criterion, not a diagnostic aside.
- **Phase 2's falsification protocol must complete before Phase 3 starts, not run in parallel with it.** Curvature validity (Pitfall 6) is a precondition for Phase 3's headline comparison being interpretable at all — if the synthetic control fails (comparable curvature to real data), Phase 3 should not proceed with the current decoder.
- **Hyperparameter freezing must happen using upstream-only diagnostics.** This is the single most consequential sequencing point for scientific validity: n_neighbors, d, decoder architecture, and the curvature quantile threshold must all be locked in by Phases 1–2's own diagnostics (connectivity, short-circuit stability, residual-variance elbow, held-out reconstruction quality, capacity-sweep stability) *before* Phase 3 computes a single regional MKNN number. Any phase plan for Phase 3 should open with "pre-specify the split, then compute MKNN" as an explicit ordering constraint, not an implementation detail left to judgment mid-phase.
- **Notebooks 02/03 iterate against Phase 1's cache and never re-fit Isomap.** This is both an architecture pattern (config-hash-keyed checkpointing) and a pitfall-avoidance measure (Anti-Pattern 1: "Restart & Run All" as the iteration loop wastes minutes per tweak and risks stale kernel state).

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (decoder + curvature):** the mean-curvature-in-high-codimension math (first/second fundamental form, `‖H‖` derivation, `torch.func` batching via `vmap`) is dense and easy to get subtly wrong on tensor shapes/index conventions — a `--research-phase 2` pass focused on `torch.func` batched Hessian patterns and the exact linear-algebra steps would reduce implementation risk.
- **Phase 3 (regional MKNN):** the density-confound control methodology (partial regression, density-matched stratification, density-matched permutation null) is original synthesis, not a documented off-the-shelf recipe — worth a focused research pass on partial-correlation/stratification implementation patterns in Python before planning this phase's tasks in detail.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Isomap + gate):** sklearn `Isomap` internals, classical-MDS double-centering, and connectivity checks are all well-documented, verified against primary sklearn source in ARCHITECTURE.md and PITFALLS.md — implementation is mechanical once the plan encodes the correct API calls (`dist_matrix_`, `eigvalsh`, `connected_components`).

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Every version read directly from PyPI JSON API / official docs on the research date; sklearn truncation behavior cross-checked against `main` branch source |
| Features | MEDIUM | MKNN definition and origin-paper numbers are HIGH (read directly from arXiv:2509.19453 PDF); differential-geometry practitioner conventions (eigenspectrum diagnostics, curvature conventions) are web-sourced and cross-checked across multiple independent sources, not single-primary-source-verified |
| Architecture | HIGH | sklearn API facts (fitted attributes, `eigen_solver` behavior, no `random_state`) verified against official docs and source; notebook/caching structure is established research-engineering practice, cross-checked against local repo conventions |
| Pitfalls | MEDIUM | Isomap mechanics (graph bridging, short-circuiting, truncated eigenvalues) verified against sklearn source/docs (MEDIUM-HIGH); the six-part decoder-curvature falsification protocol and the density-confound analysis are original methodological synthesis for this specific pipeline, grounded in standard ML/statistics practice but not lifted from a single citable source — treat as domain-expert reasoning, not documented consensus |

**Overall confidence:** MEDIUM-HIGH — tooling and library-internals facts are solidly verified; the parts of the research that are genuinely novel to this milestone (the falsification protocol, the density-confound controls) are sound synthesis but should be treated as the plan's own methodology to defend, not citations to lean on.

### Gaps to Address

- **No published benchmark for Isomap `.fit()` wall-clock time at exactly n=10,000, d≈2–10** — STACK.md's "a few minutes, not hours" estimate is derived from documented complexity bounds, not measured. Address by timing the actual Phase 1 fit early and adjusting expectations/iteration cadence accordingly; not a blocker, just budget slack in the first phase plan.
- **No single agreed hard cutoff for "how large is too large" on the `|λ_min|/λ_max` non-Euclideanity ratio** — literature gives only rules-of-thumb ranges. The PASS/MARGINAL/FAIL gate thresholds will need to be chosen and justified explicitly in Phase 1's plan (or discuss-phase), not assumed to exist as an external standard.
- **No canonical density-confound control recipe in the representational-alignment literature** — Pitfall 7's four-part control strategy (correlation check, centroid-distance check, partial regression, density-matched stratification/null) is original synthesis. Phase 3 planning should treat this as the phase's core methodological deliverable and allow room to iterate on which control(s) are most tractable to implement well, rather than assuming a single standard approach exists to look up.
- **Huh et al. 2024 (Platonic Representation Hypothesis) full-text k-value conventions were not independently confirmed** beyond what arXiv:2509.19453 itself states (WebFetch on the arXiv abstract returned metadata only) — the k=10 default is grounded via the origin paper's own reported null-baseline numbers (`k/n` consistency), which is sufficient for this milestone's purposes, but a Phase 3 planner should not assume k=10 is a field-wide documented standard beyond that indirect derivation.

## Sources

### Primary (HIGH confidence)
- `pypi.org/pypi/<package>/json` for torch, datasets, scikit-learn, faiss-cpu, huggingface_hub, hf_xet, matplotlib — official PyPI registry, 2026-07-29
- `raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/manifold/_isomap.py` — confirms `KernelPCA(n_components=self.n_components, ...)` truncation and `dist_matrix_`/`kernel_pca_`/`embedding_`/`nbrs_` fitted attributes
- `scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html`, `.../ClassicalMDS.html` — official docs
- `huggingface.co/api/datasets/UniverseTBD/pu-embeddings` and HTTP HEAD against the resolved parquet URL — confirms per-config data routing and exact file size (~553 MiB)
- arXiv:2509.19453 (Duraphe, Smith, Sourav & Wu, "The Platonic Universe: Do Foundation Models See the Same Sky?") — read directly, primary source for MKNN formula and origin-paper numbers
- Local repo: `src/effdim/api.py`, `.gitignore`, `.planning/PROJECT.md`, `.planning/ROADMAP.md`

### Secondary (MEDIUM confidence)
- GitHub PRs #21915, #20531, issue #31246 (scikit-learn) — `_fix_connected_components` mechanism, non-PSD Gram matrix behavior
- Web search synthesis, cross-checked across multiple sources: Isomap short-circuit edges, negative-eigenvalue non-Euclideanity diagnostics, subsampling/landmark geodesic-distance inflation, MutualNN/CKNNA-family density/hubness sensitivity
- `huggingface.co/docs/hub/en/xet/using-xet-storage` — hf_xet auto-install behavior

### Tertiary (LOW confidence)
- The six-part decoder-curvature falsification protocol (PITFALLS.md, Pitfall 6) — original synthesis, no single canonical source found; grounded in standard train/val, ablation, negative-control, seed-ensemble ML practice applied to differential-geometric quantities
- arXiv:2606.06329 abstract-level reference on high-dimensional mean curvature computation — confirms this is an active research area, not independently verified beyond abstract

---
*Research completed: 2026-07-29*
*Ready for roadmap: yes*
