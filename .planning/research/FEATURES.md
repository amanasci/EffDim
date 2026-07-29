# Feature Research

**Domain:** Applied manifold-geometry analysis notebook (Isomap reconstruction → smooth decoder → mean-curvature field → curvature-stratified representational-alignment probe)
**Researched:** 2026-07-29
**Confidence:** MEDIUM (web-sourced differential-geometry and manifold-learning practice, cross-checked across multiple independent sources; MKNN definition and origin-paper numbers are HIGH confidence, read directly from arXiv:2509.19453)

This is not a conventional "product" feature landscape — it is a six-step analysis pipeline. "Table stakes" means *the analysis is not scientifically credible without this*; "differentiators" means *goes beyond what arXiv:2509.19453 did*; "anti-features" means *tempting shortcuts that would silently produce a wrong or misleading result*.

---

## Feature Landscape

### Table Stakes (Analysis Is Not Credible Without These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Connectivity check before fitting Isomap (single connected component in the k-NN graph) | If the neighbourhood graph is disconnected, shortest-path (geodesic) distances between components are undefined/infinite; sklearn's `Isomap` will warn and silently patch disconnected components by adding edges between nearest components, which corrupts exactly the large-scale distances the eigenspectrum diagnostic depends on. Must be checked, not just trusted to sklearn's auto-fix. | LOW | Check with `scipy.sparse.csgraph.connected_components` on the k-NN graph before calling `Isomap.fit`; increase `n_neighbors` if >1 component. sklearn `_fix_connected_components` will patch silently otherwise — do not rely on this. |
| Short-circuit sensitivity check on `n_neighbors` | A single "short-circuit" edge — one k-NN edge that jumps across two geometrically distant but locally nearby manifold sheets (common in noisy/curved/high-dimensional embeddings) — can corrupt many entries of the geodesic distance matrix and produce a qualitatively different, wrong embedding. This is Isomap's best-documented, most consequential failure mode. | LOW–MED | Practical check: refit at 2–3 nearby `n_neighbors` values (e.g. 8, 10, 15) and confirm the eigenspectrum and embedding are qualitatively stable. Do not tune `n_neighbors` post hoc to make the spectrum look "nicer" — that is p-hacking the diagnostic. |
| Report the classical-MDS eigenspectrum, not just the embedding | Isomap silently discards negative eigenvalues and only keeps the top-d positive ones for the returned embedding; if you never look at the full spectrum you cannot tell whether the discarded structure was noise or a large, load-bearing non-Euclidean component. | LOW | `sklearn.manifold.Isomap` exposes `kernel_pca_.eigenvalues_` (fit with `kernel_pca_` — note: recent sklearn versions only retain the top `n_components` eigenpairs by default unless you request more, so extracting the *full* spectrum may require constructing the doubly-centred matrix by hand from `dist_matrix_`, or using `KernelPCA(n_components=None)`/`eigen_solver="dense"` directly on the double-centred `-0.5 * J D^2 J` matrix). |
| Report the "relative magnitude of the most negative eigenvalue" statistic | This is the standard scalar diagnostic for classical-MDS non-Euclideanity: `|λ_min| / λ_max` (or `|λ_min| / Σ|λ_i|`). A near-zero value says the geodesic dissimilarities are effectively Euclidean-embeddable in the chosen dimension; a large value (rules of thumb in the literature range from "a few percent" as a caution flag to "same order of magnitude as λ_max" as an outright validity failure) says the Isomap embedding is not a faithful Euclidean representation of the geodesic distances. | LOW | Report it as a single number per fit, not just eyeballed off a plot. Combine with the residual-variance-vs-dimension curve (see below) — they diagnose different failure modes and should not be conflated. |
| Residual-variance-vs-dimension curve with an explicit elbow/knee criterion | Tenenbaum et al.'s own diagnostic (residual variance = `1 − ρ²(geodesic distances, embedded Euclidean distances)`), used to justify the choice of embedding dimension `d`. Reporting a single `d` without this curve is an unjustified/unverifiable choice. | LOW–MED | Compute residual variance for `d = 1..~20`, plot, and state the elbow explicitly (e.g. via `kneedle`/`kneed` algorithm or simple second-difference maximum) rather than "it looked like an elbow around d≈X." |
| A stated, justified dimension `d` for Isomap's target embedding before doing anything downstream | Everything downstream (decoder input dimension, Jacobian/Hessian shapes, curvature) is a function of `d`. Choosing `d` post hoc to make later results (e.g. curvature) look interesting is a degrees-of-freedom leak. | LOW | Fix `d` from the residual-variance elbow *before* training the decoder; if `d` is later revisited, document why and rerun the full pipeline, don't cherry-pick. |
| Decoder trained and evaluated with a held-out split | A decoder trained and evaluated on the same 10k points cannot distinguish "good parameterization of the manifold" from "memorized the training points," which directly undermines the curvature step (curvature of an overfit/memorizing decoder is dominated by interpolation noise, not manifold shape). | LOW–MED | Standard train/val/test split (e.g. 80/10/10) on the Isomap coordinates → embedding pairs; report reconstruction error on held-out points, not just train loss. |
| Standard reconstruction-quality metric, reported per-dimension and aggregate | "The decoder works" is not falsifiable without a number. Community-standard choices for this exact use case (mapping a low-d coordinate to a high-d embedding vector) are relative L2 reconstruction error (`‖x̂−x‖ / ‖x‖`, averaged over held-out points) and per-dimension R², both of which are the direct high-dimensional-regression analogues of what Tenenbaum's residual variance does for distances. Cosine similarity is a weaker/partial metric here since it discards norm information, which matters for embeddings). | LOW | Report both a global relative-L2 (or global R²) number and a per-dimension R² distribution (min/median/mean) — a good aggregate number can hide catastrophic failure on a subset of the 768 output dimensions. |
| C²-smooth decoder activation, not ReLU/piecewise-linear | Mean curvature requires the second derivative of the parameterization; ReLU-family networks are piecewise linear with a.e.-zero (and everywhere-undefined-at-kinks) second derivative, so `II ≡ 0` a.e. and any curvature computed on them is degenerate/meaningless, not "flat." | LOW | Already a project decision (see PROJECT.md) — use `tanh`, `GELU`, `SiLU`/swish, or similar C²-smooth activations throughout the decoder. Confirm no ReLU/LeakyReLU/hardtanh anywhere in the forward path, including any residual/skip blocks. |
| Correct, codimension-honest mean curvature vector definition | n=768 ≫ d (Isomap target dimension, likely single or low digits). Mean curvature here is fundamentally a **vector**, not a scalar, unless d = n−1 (a hypersurface), which this problem is nowhere close to. Reporting "the mean curvature" as if it were the familiar 2-surface-in-3-space scalar is a category error. | MED | See dedicated math section below. The reportable scalar is `‖H‖`, the Euclidean norm of the mean curvature vector in the normal bundle — state this explicitly every time a "curvature value" per point is shown. |
| Curvature computed via exact autodiff Jacobian + Hessian, not finite differences | Finite-difference second derivatives are numerically unstable (error scales as ~1/h² vs O(h) roundoff) and require re-deriving a step size per region of the manifold; for a decoder that is already a differentiable `torch` module, exact reverse-mode/forward-mode autodiff (`torch.func.jacrev`/`jacfwd`, `torch.func.hessian`, or `vmap`-batched Jacobian-of-Jacobian) is strictly better and is already the project's stated approach. | LOW–MED (given torch.func) | `torch.func.hessian` computes the full `d×d×n` (or `n×d×d`) tensor of second partials per point; batch with `vmap` over the 10k points rather than looping in Python. |
| Curvature reported relative to a null/baseline scale | A raw `‖H‖` number in embedding-space units is not interpretable on its own — it needs to be compared either against curvature on a control (e.g. curvature of a decoder fit to a known-flat/known-curvature synthetic manifold of the same `d,n`) or at minimum reported in units normalized by local scale (e.g. `‖H‖ · (local injectivity radius or reach)` or z-scored across the point cloud) so "high" and "low" are meaningful, not just min/max of an arbitrary distribution. | MED | Cheapest version: z-score / rank-normalize `‖H‖` over the 10k points before doing anything with "high vs low." |
| Region split reported alongside a density/coverage sanity check | If "high curvature" region is simply where Isomap coordinates are sparse (undersampled), the "curvature" signal may be a decoder-interpolation artifact (extrapolation instability) rather than true manifold shape, and any MKNN difference between regions is confounded with local sample density rather than geometry. | MED | Report local point density (e.g. k-NN distance in Isomap coordinate space) per region alongside curvature; flag/exclude points in low-density outlier tails before curvature-based splitting. |
| MKNN computed exactly as `MKNN(z1,z2) = k⁻¹|N_k(z1) ∩ N_k(z2)|` (Chechik et al. 2010), matching the origin paper | This is the entire point of comparison to arXiv:2509.19453 — using a different alignment metric (even a superficially similar one, e.g. CKA) would make "does alignment vary with curvature" incomparable to the paper's headline crossmodal numbers. | LOW | k-NN sets computed independently in each embedding space (HSC embedding space and Legacy Survey embedding space), same object rows via the dataset's row alignment, same `k` used across regions being compared. |
| MKNN with a permutation null baseline, computed *per region*, not just once globally | The origin paper's null is a random permutation of one embedding set along the sample axis, giving `π(HSC)` scores of ~0.03–0.05% (consistent with the closed-form expectation `E[MKNN | random] ≈ k/n` under a fully random hypergeometric null). Regions differ in `n` (point count) and possibly in local density, so the *same* raw MKNN score means different things relative to *different* region-specific null expectations — the null must be recomputed per region, not reused from the global one. | LOW–MED | Permute within each region's index set (not globally then subset) so the null respects each region's local `n` and any local density structure. |
| Confidence intervals on regional MKNN (not point estimates) | A single MKNN number per region cannot show whether a high-vs-low-curvature difference is real or sampling noise, especially since curvature-based splitting can produce unequal-`n` regions. | LOW–MED | Bootstrap over the object index within each region (resample with replacement, recompute MKNN, repeat ~1000×) to get percentile or BCa CIs; this is already a stated project decision. |

### Differentiators (Goes Beyond arXiv:2509.19453)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Explicit non-Euclideanity audit of the Isomap embedding before trusting it | The origin paper never validates that its embedding spaces are geometrically well-formed before running MKNN on them (it uses the *raw* foundation-model embeddings, not an Isomap reconstruction, so this question didn't even arise for them). Quantifying and reporting the negative-eigenvalue statistic is a genuinely new validity check this milestone introduces. | LOW (given table-stakes work above) | This is the milestone's first novel contribution over the paper — frame it as "we checked something the original methodology had no need to check." |
| Analytic, closed-form differentiable manifold parameterization (the decoder) | The paper works entirely in raw embedding space; it never asks "what is the shape of the manifold the embeddings live on," only "how much do k-NN sets overlap." Building an explicit smooth chart is new machinery, not a replication. | HIGH | This is the core technical novelty of v1.1; complexity is dominated by decoder training stability (see Pitfalls-adjacent notes above on held-out reconstruction quality) and correct curvature-tensor bookkeeping. |
| Curvature-stratified (rather than global) representational alignment | The paper reports one crossmodal MKNN number per model pair; asking "does alignment vary across the manifold, and does it correlate with local geometric flatness/curvature" is the actual novel scientific question of this milestone (per PROJECT.md's stated goal) and has no analogue in the original paper. | MED (given curvature field + MKNN already built) | This is the payoff step — frame results as either "alignment concentrates in flat regions" (interesting, supports a geometric refinement of PRH) or "no significant regional difference" (also reportable, and per PROJECT.md's own risk framing, plausible given Legacy Survey crossmodal alignment is already the paper's weakest signal at 0.4–2%). |
| Reporting a null-corrected, per-region *excess* MKNN (observed − permutation-null) rather than raw MKNN | Raw MKNN mixes true alignment signal with a baseline noise floor that itself depends on `k` and region size `n`; subtracting the region-specific null isolates the interesting quantity and makes cross-region comparison honest. | LOW (once regional null exists) | `excess_MKNN = MKNN_observed − mean(MKNN_permuted)`, with the bootstrap CI computed on the excess directly (paired bootstrap) rather than propagating two separate CIs. |
| k-sensitivity curve for the MKNN comparison (not a single k) | Chechik/Huh/the origin paper all use a single, largely unstated/unmotivated k (this analysis's grounding search suggests k=10 is the field's de facto default and is numerically consistent with the origin paper's own reported permutation-null values, `E[MKNN\|null] ≈ k/n`). Showing that the high-vs-low-curvature finding (or null result) is stable across k=5,10,20,50 is stronger evidence than a single-k result and something the origin paper does not do. | LOW–MED | Cheap to add once MKNN machinery exists — just loop k and replot. Strengthens (or appropriately weakens) any headline claim. |

### Anti-Features (Tempting But Wrong or Misleading Here)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| Reporting a single scalar "Gaussian curvature" or "the curvature" of the decoder | Feels like the familiar, intuitive notion of curvature from 2D-surface intuition (a sphere's curvature, a saddle's curvature); many curvature-computation tutorials online default to this framing. | Gaussian/sectional curvature in codimension > 1 is not a single well-defined scalar without an extra choice: `K^ν = det(II^ν)/det(I)` depends on a choice of unit normal direction `ν` in the (n−d)-dimensional normal space, and different `ν` give different, even oppositely-signed, values at the same point. There is no canonical choice when the normal space has dimension > 1 (here it has dimension 768−d ≫ 1). Reporting a single number under this name silently hides an arbitrary/unstated normal-direction choice. | Report `‖H‖`, the norm of the mean curvature vector (the trace/average of `II` over the normal bundle, no direction choice required) — the well-defined, codimension-agnostic scalar. If a Gaussian-curvature-like quantity is wanted, report it explicitly as "generalized Gaussian curvature, averaged over the normal sphere" and say so, don't call it "the curvature." |
| Using the decoder's per-point "principal curvatures" (eigenvalues of a single shape operator) as if this generalizes directly from the d=2,codim=1 case | Principal curvatures are eigenvalues of *the* shape operator, which itself requires a single normal direction / single second-order form; natural in codim 1, ill-posed as a single spectrum in high codimension for the same reason as Gaussian curvature above. | Same root problem as above — there are `d` eigen-directions but `(n−d)`-many possible shape operators (one per normal direction), not one. | Either fix `‖H‖` as the reported scalar, or if per-direction detail is wanted, report the full second-fundamental-form tensor's norm (Frobenius norm of `II`, sometimes called the "total curvature" or used to build "scalar curvature"-like invariants) with an explicit statement that this is not the classical Gaussian curvature. |
| Choosing `n_neighbors`/`d`/curvature-quantile thresholds after seeing the MKNN result, to make the headline finding cleaner | Natural temptation once a "nice" result is in view — small tweaks to any of these upstream knobs can flip a marginal region-difference result. | This is a garden-of-forking-paths / multiple-comparisons problem: each of the 4–5 pipeline knobs (n_neighbors, d, decoder architecture/training seed, curvature quantile cutoff, k for MKNN) is a researcher degree of freedom; tuning any of them post hoc to the downstream MKNN result invalidates the reported significance/CIs. | Freeze upstream hyperparameters using upstream-only diagnostics (connectivity, short-circuit stability, residual-variance elbow, held-out reconstruction quality) *before* looking at MKNN by region; if later revisited, rerun the entire pipeline and report both the original and revised result. |
| A fixed absolute threshold (e.g. `‖H‖ > 1.0`) for "high curvature," reused across reruns/configs | Feels more "principled" or reproducible than a quantile because it's a fixed number. | `‖H‖`'s scale depends on the decoder's parameterization, training run, and Isomap's arbitrary global scale factor for the embedding; a fixed absolute threshold is not comparable across reruns or even across random seeds of the same decoder, and silently changes what fraction of the manifold counts as "high curvature" if the curvature distribution's scale drifts. | Use a quantile split (e.g. top/bottom tercile or quartile of `‖H‖`, or a z-scored/rank-normalized cutoff) computed fresh each run — comparable in spirit to the paper's own use of relative, distribution-based statistics (MKNN as a *percentage*, not an absolute count). |
| Naive quantile split on raw `‖H‖` without checking it isn't just a proxy for local sample density | Simplest possible region-splitting rule; tempting because it requires no extra code. | Decoder curvature can spuriously spike in undersampled/extrapolated regions of Isomap coordinate space purely from interpolation instability (the decoder is only well-constrained where training data is dense), so a naive high/low split may really be splitting "well-sampled" vs "poorly-sampled" rather than "flat" vs "curved" — and MKNN differences would then reflect a data-density artifact, not a geometric one. | Report local density (k-NN distance in Isomap coordinates) alongside curvature per point; either control for density (e.g. partial correlation of MKNN with curvature given density) or restrict the analysis to a density-trimmed core of the point cloud before quantile-splitting on curvature. |
| Clustering (e.g. k-means) on `‖H‖` values to define "high" vs "low" regions | Feels more data-driven/less arbitrary than a fixed quantile cut. | 1-D k-means on a scalar field is just a fancier, less transparent threshold-finder (it will find natural breakpoints in the *marginal distribution* of curvature values, not spatial regions of the manifold) and doesn't address the sampling-density confound above; it also adds an unnecessary extra hyperparameter (number of clusters) without benefit over a quantile split for a scalar field. | Stick to quantile splitting on the (density-checked) scalar field; reserve clustering for cases where you need spatially contiguous regions in the *coordinate* space, which is a different problem from splitting by a scalar field value. |
| Treating Isomap's returned `d`-dimensional embedding as automatically valid because sklearn ran without erroring | sklearn does not raise on non-Euclidean geodesic input — it just discards negative eigencomponents and returns the top-d positive ones, silently. | This is precisely the failure mode Step 2 of the analysis exists to catch; trusting "it ran" as validation defeats the purpose of the milestone's own second step. | Always perform the negative-eigenvalue and residual-variance diagnostics regardless of whether Isomap raised any warning. |
| Extending the decoder's domain far beyond the convex hull / support of the Isomap training coordinates to "smooth out" or densely grid the curvature field | Tempting for making a nice dense curvature heatmap/visualization. | Neural decoders are unreliable extrapolators; curvature computed off-manifold (outside the region actually populated by real Isomap coordinates) reflects the decoder's arbitrary extrapolation behavior, not the data manifold's geometry, and can dominate a naive min/max-based color scale. | Only evaluate/report curvature at (or very near, via small local perturbation for finite-sample smoothing if ever needed) the actual 10k Isomap-embedded points, or at most a fine grid restricted to their convex hull / density support. |
| Reusing a single global k for MKNN without checking region-size sensitivity | Simpler to implement, matches the origin paper's single-k choice at first glance. | If curvature-based regions have very different `n` (plausible, since curvature and density can correlate — see above), a fixed absolute k represents a different *fraction* of each region's neighbor graph, subtly changing what MKNN measures across regions. | Either use a k proportional to region size, or explicitly confirm/report absolute region sizes and k so the comparison's caveats are visible; the k-sensitivity curve differentiator above also mitigates this. |

---

## Feature Dependencies

```
[Connectivity + short-circuit checks on k-NN graph]
    └──requires──> [Isomap fit] (upstream gate, not downstream)

[Isomap fit]
    └──requires──> [Connectivity + short-circuit checks]
    └──produces──> [doubly-centred squared-geodesic matrix, dist_matrix_]

[Full classical-MDS eigenspectrum + negative-eigenvalue statistic]
    └──requires──> [Isomap fit] (needs dist_matrix_, not just the truncated embedding)

[Residual-variance-vs-dimension curve + elbow]
    └──requires──> [Isomap fit at multiple candidate d, or geodesic distances computed once + re-embedded at multiple d]
    └──determines──> [chosen embedding dimension d, frozen before decoder training]

[Decoder training]
    └──requires──> [chosen d] (input dimension)
    └──requires──> [C2-smooth activation] (else curvature step degenerates)
    └──produces──> [held-out reconstruction metric: relative L2 / per-dim R2]

[Mean curvature vector field ‖H‖]
    └──requires──> [Decoder training] (need a differentiable f: R^d -> R^768)
    └──requires──> [first fundamental form g_ij, second fundamental form II_ij via Jacobian/Hessian]
    └──requires──> [held-out reconstruction quality already validated] (curvature of a bad decoder is meaningless)

[Density check alongside curvature]
    └──requires──> [Mean curvature vector field] and [Isomap coordinates] (need both to check confound)

[Quantile-based high/low curvature region split]
    └──requires──> [Mean curvature vector field ‖H‖]
    └──requires──> [Density check] (to rule out density-confound before trusting the split)

[Per-region MKNN + permutation null + bootstrap CI]
    └──requires──> [Region split]
    └──requires──> [row-aligned crossmodal embeddings] (already available: HSC vs Legacy Survey columns)

[Regional excess-MKNN differentiator]
    └──requires──> [Per-region MKNN + permutation null]

[k-sensitivity curve differentiator]
    └──enhances──> [Per-region MKNN] (not a hard dependency, can be added after a single-k result exists)
```

### Dependency Notes

- **Eigenspectrum diagnostics require the full `dist_matrix_`, not the truncated embedding** — this dictates an implementation detail: don't rely solely on `Isomap.embedding_`; keep access to `Isomap.dist_matrix_` (or build the doubly-centred Gram matrix by hand) so the full spectrum, including negative eigenvalues, is inspectable.
- **Dimension `d` must be frozen before decoder training** — this is a hard ordering constraint for defensible results (see Anti-Features: post-hoc tuning).
- **C²-smooth activation is a hard precondition for the curvature step**, not just a nice-to-have — with a non-smooth activation the entire curvature computation is either zero-almost-everywhere or undefined at kinks, making Steps 4–6 meaningless regardless of how well steps 1–3 went.
- **Density check enhances (and should gate) the region split** — it is not strictly required to *produce* a split, but skipping it means any MKNN-vs-curvature finding cannot be defended against the density-confound anti-feature.
- **Regional permutation null must be recomputed per region**, not reused globally, because region `n` and local structure differ — this is a correctness dependency, not just a nice-to-have.

---

## MVP Definition

### Launch With (v1.1 — matches PROJECT.md's Active requirements)

- [ ] Connectivity + short-circuit stability check before trusting any Isomap fit — cheap insurance against Isomap's best-documented failure mode
- [ ] Full eigenspectrum extraction + negative-eigenvalue statistic + residual-variance/elbow curve — this *is* PROJECT.md's stated Step 2 requirement
- [ ] Decoder with C²-smooth activation, trained with a held-out split, reconstruction metric reported (relative L2 + per-dim R²)
- [ ] Mean curvature vector field via `torch.func` Jacobian/Hessian on the decoder, reported as `‖H‖` (not Gaussian curvature, not principal curvatures)
- [ ] Density-checked quantile split into high/low curvature regions
- [ ] Per-region MKNN (HSC vs Legacy Survey) vs region-specific permutation null, with bootstrap CIs

### Add After Validation (natural v1.1 stretch, still notebook-only)

- [ ] Regional excess-MKNN (null-subtracted) reporting — cheap once the null exists
- [ ] k-sensitivity curve for the regional MKNN comparison — cheap once MKNN machinery exists, strengthens any claim

### Future Consideration (explicitly out of v1.1 per PROJECT.md)

- [ ] Intramodal MKNN across a model-size ladder (the paper's stronger 28–56% signal) — deferred, needs a second model size
- [ ] Promoting the curvature operator into `src/effdim/` as a library feature — deferred milestone decision, needs unit tests against known-curvature synthetic surfaces first
- [ ] Comparing MKNN against other alignment metrics (CKA, mutual information) mentioned as the origin paper's own stated future work — out of scope, would require reimplementing metrics the paper didn't even use itself

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Connectivity/short-circuit checks | HIGH (gates everything) | LOW | P1 |
| Eigenspectrum + negative-eigenvalue stat + residual-variance elbow | HIGH (is the stated milestone Step 2) | LOW–MED | P1 |
| C²-smooth decoder + held-out reconstruction metric | HIGH (gates curvature step) | MED | P1 |
| Mean curvature vector `‖H‖` via autodiff | HIGH (is the stated milestone Step 4) | MED–HIGH | P1 |
| Density-checked quantile region split | HIGH (defends Step 5 against the sampling-density anti-feature) | LOW–MED | P1 |
| Regional MKNN + null + bootstrap CI | HIGH (is the stated milestone Step 6, the payoff) | MED | P1 |
| Regional excess-MKNN (null-subtracted) | MEDIUM (sharper, but P1 already gives a defensible result) | LOW | P2 |
| k-sensitivity curve | MEDIUM (robustness check, strengthens claim) | LOW | P2 |
| Model-size ladder / intramodal MKNN | LOW for v1.1 (explicitly deferred) | HIGH (second model) | P3 |
| Promote curvature operator into `src/effdim/` | LOW for v1.1 (explicitly out of scope) | HIGH (needs test suite) | P3 |

**Priority key:**
- P1: Required for a defensible v1.1 result (matches PROJECT.md Active requirements)
- P2: Cheap add-ons that materially strengthen the headline finding, natural stretch within v1.1
- P3: Explicitly deferred per PROJECT.md Deferred/Out of Scope

---

## Mathematical Reference (for the downstream plan/implementation)

### 1. Isomap eigenspectrum and negative eigenvalues

Isomap computes pairwise geodesic distances `D` (shortest paths over a k-NN graph), then applies classical MDS: double-center the squared distance matrix,

```
B = -1/2 · J D^(2) J,   J = I - (1/n) 11ᵀ
```

and eigendecompose `B = V Λ Vᵀ`. If `D` were an exact Euclidean distance matrix, `B` is guaranteed PSD (all `λ_i ≥ 0`). Geodesic shortest-path distances over a k-NN graph are **not guaranteed to be exactly Euclidean-embeddable** — they can violate the conditions for a valid Euclidean distance matrix (e.g. from short-circuit edges, curvature of the true manifold, or graph-approximation error in the geodesics) — so `B` need not be PSD, and negative eigenvalues appear. **Large** negative eigenvalues (relative to the leading positive eigenvalues) indicate that a non-negligible fraction of the geodesic structure is non-Euclidean, i.e. the low-dimensional Euclidean embedding Isomap returns (which uses only the top-`d` positive eigenpairs and silently discards the negative part) is not a faithful representation of the input geodesic distances. Standard practitioner diagnostics:
- **Relative magnitude of the most negative eigenvalue**: `|λ_min| / λ_max` (or normalized by `Σ|λ_i|`). No single universally agreed hard cutoff exists in the literature searched; treat values that are a sizable fraction (order tens of percent) of `λ_max`, or comparable in magnitude to the eigenvalues you intend to keep for the embedding, as an explicit red flag requiring investigation (rerun with different `n_neighbors`, inspect for short-circuits, or apply a Cailliez-type correction) rather than proceeding uncritically.
- **Residual variance vs. dimension** (Tenenbaum et al. 2000): `R(d) = 1 - ρ²(D_geodesic, D_embedding(d))`, with `ρ` the linear correlation coefficient over all pairwise distances. Plot `R(d)` for increasing `d` and pick the elbow — the `d` past which `R(d)` stops decreasing appreciably.
- These two diagnostics catch different failure modes: the eigenvalue-sign statistic flags *non-Euclideanity of the geodesic structure itself*; the residual-variance curve flags *how many Euclidean dimensions are needed* to capture the (positive, Euclidean-consistent) part of that structure. Report both.

### 2. Decoder / inverse mapping

The "preimage problem" (mapping a nonparametric embedding coordinate back to ambient space) is less studied than the forward out-of-sample-extension problem (which has closed-form Nyström-style solutions because it only needs to extend eigenvectors). Standard approaches, in increasing order of what this milestone needs:
1. **RBF/kernel interpolation of the inverse map** — fit radial basis functions from embedding coordinates to ambient vectors; simple, but not naturally differentiable to high order everywhere and doesn't scale as gracefully to n=768 outputs.
2. **Parametric regression decoder (this milestone's choice)** — train a neural network `f_θ: R^d → R^768` by regression on (Isomap coordinate, original embedding) pairs. Equivalent to fixing the "encoder" (Isomap) and only learning the "decoder" half of an autoencoder. This is the right choice here specifically *because* it yields a single closed-form differentiable map, which the curvature computation (Step 4) needs.
- **Expected/standard reconstruction metrics**: relative L2 error `‖x̂ - x‖₂ / ‖x‖₂` averaged over held-out points (the direct analogue, in ambient-vector space, of Isomap's own residual-variance diagnostic in distance space), and per-output-dimension R² (catches cases where reconstruction is good on average but fails badly on a subset of the 768 embedding dimensions). Cosine similarity alone is a weaker choice for this use case since it is invariant to the reconstructed vector's norm, discarding information that matters for a metric-embedding reconstruction task.

### 3. Mean curvature in high codimension — precise formulation

Let `f: U ⊆ R^d → R^n` be a smooth immersion (the decoder), with `n = 768 ≫ d`. At a point `u ∈ U`:

- **First fundamental form** (metric induced on the parameter domain):
  `g_ij(u) = ⟨∂_i f(u), ∂_j f(u)⟩`, i.e. `g = J^T J` where `J = Df(u) ∈ R^{n×d}` is the Jacobian. `g` is a `d×d` symmetric positive-definite matrix (assuming `f` is a genuine immersion, i.e. `J` has full column rank `d`).

- **Second fundamental form** (normal component of the second derivative):
  `II_ij(u) = (I_n − P_T) ∂_i∂_j f(u)`, where `∂_i∂_j f(u) ∈ R^n` is the (i,j) entry of the ambient Hessian tensor of `f`, and `P_T = J g^{-1} J^T` is the orthogonal projector onto the `d`-dimensional tangent space `T_u = \mathrm{range}(J) ⊂ R^n`. `(I_n − P_T)` projects onto the `(n−d)`-dimensional **normal space**. `II_ij` is therefore itself a vector in `R^n` (lying in the normal space) for each `(i,j)` — the second fundamental form is a normal-bundle-valued symmetric bilinear form on the tangent space, not a scalar or a single matrix.

- **Mean curvature vector**:
  `H(u) = (1/d) · g^{ij} II_ij(u)` (Einstein summation over `i,j`, `g^{ij}` = entries of `g^{-1}`) — i.e. the trace of `II` with respect to the induced metric `g`, averaged over the `d` tangent directions. `H(u) ∈ R^n` and lies in the normal space at `u`. **This is a vector, well-defined for any codimension `n − d`, with no arbitrary choice required** — this is precisely why it, and not Gaussian/sectional curvature, is the right quantity to report here.

- **What is conventionally reported as "the" curvature scalar**: `‖H(u)‖`, the Euclidean norm of the mean curvature vector, computed pointwise over the 10k Isomap-embedded points. This is the standard reduction to a single scalar per point for visualization/region-splitting purposes and is the quantity referenced throughout this document as "curvature."

- **Common alternatives people conflate with mean curvature, and why they differ here**:
  - *Gaussian curvature* (product of principal curvatures, in the classical `d=2, n=3` hypersurface case) generalizes to codimension `> 1` only via a per-normal-direction shape operator, `K^ν(u) = \det(II^ν) / \det(g)` for a chosen unit normal `ν` in the `(n−d)`-dimensional normal space — this depends on an essentially arbitrary choice of `ν` (or requires averaging `K^ν` over the unit normal sphere to get a codimension-invariant "generalized Gaussian curvature," a different and less standard quantity). There is no single canonical Gaussian curvature once `n − d > 1`, which is emphatically the regime here (`n − d = 768 − d`, `d` small).
  - *Scalar curvature* and *sectional curvature* are intrinsic Riemannian-geometry quantities computable from `g` alone (via Christoffel symbols and the Gauss equation) and, again, only reduce to a clean single scalar in special (e.g. hypersurface) cases in the extrinsic picture; conflating them with the extrinsic mean curvature vector norm is a common but incorrect simplification in applied write-ups.
  - *Principal curvatures* are eigenvalues of a single shape operator, which — same as Gaussian curvature above — requires fixing one normal direction; in codimension `> 1` there is a family of shape operators (one per normal direction), not one eigenvalue spectrum.
  - **Bottom line for this milestone**: report `‖H‖`, state explicitly that it is the norm of a vector-valued curvature, and do not use "Gaussian curvature" or "principal curvature" language for it.

- **Computation**: with `torch.func`, `J = jacrev(f)(u)` gives the `n×d` Jacobian directly; the full ambient Hessian tensor (`n×d×d`) is obtained via `jacrev(jacrev(f))(u)` or `hessian(f)(u)` per output dimension, batched over the point cloud with `vmap`. `g`, `P_T`, and `II` follow by the linear-algebra steps above; `H` is then a `(N, n)` array (one normal-space vector per point) and `‖H‖` an `(N,)` array of scalars.

### 4. Region partitioning by a scalar field

Defensible practice: **quantile split** on the (density-checked) scalar field `‖H‖` — e.g. top/bottom tercile or quartile — rather than a fixed absolute threshold (not comparable across reruns/seeds, since `‖H‖`'s scale is arbitrary/decoder-dependent) or unsupervised clustering on the 1-D value distribution (finds breakpoints in the marginal distribution, not spatial regions, and doesn't address the sampling-density confound). The key defensibility risk specific to this pipeline is that a neural decoder's curvature can spuriously spike in undersampled or extrapolated regions of Isomap coordinate space (interpolation/extrapolation instability, not true manifold shape) — so a curvature-based split should always be reported alongside a local-density check (e.g. k-NN distance within Isomap coordinate space) to rule out "high curvature" being a relabeling of "sparsely sampled."

### 5. MKNN metric

`MKNN(z1, z2) = k⁻¹ |N_k(z1) ∩ N_k(z2)|` (Chechik et al. 2010; adopted verbatim by Huh et al. 2024's Platonic Representation Hypothesis paper and by the origin paper, Duraphe/Smith/Sourav/Wu 2025, arXiv:2509.19453). `N_k` is computed independently in each embedding space (k-nearest neighbours by, typically, cosine or Euclidean distance on L2-normalized vectors), for the *same* object across the two representations (here: HSC vs Legacy Survey embeddings of the same row-aligned object).
- **Choice of k**: not rigidly standardized in the literature searched; this milestone's grounding derivation (from the origin paper's own reported numbers) is consistent with **k=10**: for their DESI-vs-HSC-sized null test (n≈18,600 galaxies), the closed-form fully-random expectation `E[MKNN | null] ≈ k/n` gives `10/18600 ≈ 0.054%`, matching their reported permutation-null values of 0.03–0.05% closely. k=10 is a reasonable, literature-consistent default; report a k-sensitivity curve (k=5,10,20,50) as a differentiator rather than committing to one value alone.
- **Null baseline**: random permutation of one embedding set's row order (breaking the true object correspondence while preserving each set's internal k-NN structure) — exactly the origin paper's `π(HSC)` construction. Under this null, overlap counts follow (approximately, for large n) a hypergeometric distribution with mean `k²/n` when both neighbor sets are drawn independently and uniformly at random — the simpler `k/n` approximation used above is standard shorthand and matches the paper's own reported figures to within their reporting precision.
- **Confidence intervals**: bootstrap over the object index (resample with replacement within the comparison set, recompute MKNN, repeat ~1000×) for percentile or BCa intervals — standard nonparametric practice for a statistic (a mean overlap fraction) without a simple closed-form finite-sample variance; not used in the origin paper itself (which reports point estimates and a binomial sign test across model-size comparisons instead), making per-region bootstrap CIs a genuine differentiator of this milestone.
- **Known confound to flag, not necessarily solve, in v1.1**: MKNN-family metrics are sensitive to "hubness" in high-dimensional embedding spaces (a small number of points appearing disproportionately often as nearest neighbours to many others), which can distort local neighbourhood overlap independent of true representational alignment; this is a known open issue for k-NN-based alignment metrics in the broader representational-alignment literature and worth a caveat in any write-up, even if out of scope to fully correct for in v1.1.

## Sources

- arXiv:2509.19453 (Duraphe, Smith, Sourav & Wu, "The Platonic Universe: Do Foundation Models See the Same Sky?", NeurIPS 2025 ML4PS) — read directly (PDF), HIGH confidence, primary source for MKNN formula, origin-paper null-baseline numbers, dataset/model details.
- Web search: Isomap classical-MDS negative eigenvalues / non-Euclidean geodesic distances (Cailliez correction, non-metric MDS as remedies) — MEDIUM confidence, cross-checked across multiple independent results.
- Web search: Isomap residual variance / elbow criterion (Tenenbaum et al. 2000 original diagnostic) — MEDIUM confidence.
- Web search: out-of-sample extension vs. preimage/inverse-mapping problem for nonlinear DR (RBF interpolation, parametric decoder approaches) — MEDIUM confidence.
- Web search + arXiv:1312.2554 ("Gaussian curvature in codimension > 1") — MEDIUM confidence; grounds the anti-feature that Gaussian curvature is not canonically well-defined once normal-space dimension exceeds 1.
- Web search: mean curvature vector / second fundamental form for immersions of arbitrary codimension (mean curvature flow literature) — MEDIUM confidence.
- Web search + arXiv:2606.06329 ("Efficient Mean Curvature Computation on High-Dimensional Data Manifolds") — MEDIUM confidence (abstract-level detail only; confirms this is an active, recent research area with the same O(codimension) computational concerns).
- Web search: sklearn Isomap disconnected-graph-component handling (`_fix_connected_components`, silent warning-then-patch behavior) — MEDIUM confidence, cross-checked against scikit-learn GitHub PRs.
- Web search: Isomap short-circuit-edge failure mode — MEDIUM confidence, cross-checked across multiple independent sources.
- Web search: manifold-learning subsampling/sample-density effects on geodesic distance estimation consistency — MEDIUM confidence.
- Web search: quantile vs. fixed-threshold vs. density-adaptive binning of a scalar field over a point cloud — LOW–MEDIUM confidence (general data-binning literature, not manifold-curvature-specific; reasoning extrapolated to this use case).
- Web search: MKNN / mutual-nearest-neighbour metric hubness and local-density confounds in representation-alignment literature — MEDIUM confidence.
- Chechik, Sharma, Shalit & Bengio (2010), "Large Scale Online Learning of Image Similarity Through Ranking" (OASIS), JMLR 11 — identified via web search as the MKNN metric's origin, cited directly by arXiv:2509.19453.
- Huh et al. (2024), "Position: The Platonic Representation Hypothesis," ICML — identified as the paper arXiv:2509.19453 follows methodologically for MKNN; full-text k-value/null-baseline details not independently confirmed beyond what arXiv:2509.19453 itself states (WebFetch on the arXiv abstract page returned metadata only, not full paper body) — flagged as a gap below.

---
*Feature research for: applied manifold-curvature analysis notebook (EffDim v1.1 PU Manifold Curvature)*
*Researched: 2026-07-29*
