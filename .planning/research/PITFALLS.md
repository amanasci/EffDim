# Pitfalls Research

**Domain:** Manifold-geometry analysis of high-dimensional neural embeddings (Isomap → learned decoder → curvature → regional representational-alignment comparison)
**Researched:** 2026-07-29
**Confidence:** MEDIUM — Isomap/MDS mechanics and metric-choice facts verified against scikit-learn source and docs (MEDIUM/verified); decoder-curvature falsification protocol and confound analysis are standard differential-geometry/statistics reasoning applied to this specific pipeline, not lifted from a single citable source (treat as domain-expert synthesis, not a documented consensus)

## Critical Pitfalls

### Pitfall 1: Silent Isomap graph bridging masks a broken manifold assumption

**What goes wrong:**
`sklearn.manifold.Isomap` does **not** raise when the `n_neighbors` graph is disconnected. When `n_connected_components > 1` it emits a warning and silently calls an internal graph-completion routine (`_fix_connected_components`) that bridges components with artificial long edges between the nearest cross-component point pair. The notebook runs to completion, produces an embedding, an eigenspectrum, and eventually a curvature field — all built on a geodesic distance matrix containing fabricated edges that do not correspond to any real path along the data.

**Why it happens:**
The embedding vectors are DINOv3 CLS-token embeddings of astronomical images pooled from two different surveys (HSC, Legacy Survey) with different depth/seeing/instrument characteristics; there is no guarantee the resulting point cloud in 768-d is a single connected, uniformly-sampled manifold rather than several loosely related clusters (e.g., by image quality, source type, redshift). 10k points sampled from 101,725 only makes disconnection more likely (fewer points per unit density → more likely to leave gaps at `n_neighbors≈10-20`).

**How to avoid:**
- Explicitly check `isomap.nbrs_.n_samples_fit_` against the connected-component count of the fitted neighbor graph *before* trusting anything downstream: run `scipy.sparse.csgraph.connected_components` on `isomap.nbrs_.kneighbors_graph(mode="connectivity")` and assert `n_components == 1`, printing the count and the size of each component.
- If disconnected, do not rely on the silent bridge. Sweep `n_neighbors` upward until the graph is a single component *before* fixing the value used for the eigenspectrum audit and curvature pipeline, and report the smallest `n_neighbors` that achieves connectivity as a diagnostic, not just the value that "worked."
- If connectivity requires an implausibly large `n_neighbors` (e.g., >> the value that keeps short-circuiting in check per Pitfall 2), treat that as evidence the point cloud is not a single manifold, and say so rather than forcing Isomap to run.

**Warning signs:**
`ConvergenceWarning`/UserWarning mentioning "connected components" in notebook output; `isomap.dist_matrix_` containing a small number of anomalously large entries equal to a bridging distance repeated across many row/column pairs (a bridged component shows up as a near-constant large distance from every one of its members to every member of the other component).

**Phase to address:**
Isomap-fitting phase (pipeline step 2), before any eigenspectrum audit.

---

### Pitfall 2: Short-circuit edges from k too large (or the data isn't a manifold at all)

**What goes wrong:**
If `n_neighbors` is too large relative to local manifold curvature, or a subset of points are noisy/off-manifold outliers, the k-NN graph gains edges that jump between regions that should be geodesically far apart. Because Isomap's embedding comes from an all-pairs shortest-path computation, a single short-circuit edge can corrupt many entries of the geodesic distance matrix (Dijkstra will route many shortest paths through the shortcut), producing a globally distorted, drastically different low-dimensional embedding — one that looks fine but is quantitatively wrong. This is the classic Isomap failure mode and is a documented "tip" in the scikit-learn manifold-learning guide.

**Why it happens:**
`n_neighbors` trades off two failure modes (too small → disconnection, per Pitfall 1; too large → short-circuiting) and there is no principled a-priori value for CLS-token embeddings of noisy astronomical imagery, which likely mixes on-manifold structure with instrument/redshift/quality-driven outliers.

**How to avoid:**
- Never fit at a single `n_neighbors` and move on. Sweep a range (e.g., 5, 8, 10, 15, 20, 30) and for each value record: number of connected components, reconstruction error (`isomap.reconstruction_error()`), and the shape of the top-20 eigenvalues from the audit in Pitfall 3.
- Stability check: for a stable `n_neighbors`, the *rank ordering* of pairwise geodesic distances for a fixed random subset of point pairs should change smoothly as `n_neighbors` is nudged by ±1-2; a sharp discontinuity in geodesic distances or in the elbow location of the eigenspectrum as `n_neighbors` changes by a small amount is the signature of a short-circuit edge switching on/off.
- Leave-one-out sensitivity: refit with a handful of points removed from dense clusters; if the eigenspectrum elbow or the embedding of far-away points moves substantially, a short-circuit through those points is likely.
- Report the chosen `n_neighbors` and the sweep, not just the final number — the roadmap should treat this as a diagnostic artifact, not a hyperparameter picked once and hidden.

**Warning signs:**
Reconstruction error non-monotonic or with a sharp kink across the `n_neighbors` sweep; a small number of points whose geodesic distance to most other points barely changes between two very different `n_neighbors` values while everything else shifts substantially (a "hub" acting as a shortcut).

**Phase to address:**
Isomap-fitting phase (pipeline step 2) and eigenspectrum-audit phase (step 3) jointly — the sweep is the connective tissue between them.

---

### Pitfall 3: Trusting `Isomap.kernel_pca_.eigenvalues_` for the negative-eigenvalue audit

**What goes wrong:**
`sklearn.manifold.Isomap` stores its internal `KernelPCA` fit in `isomap.kernel_pca_`, and `isomap.kernel_pca_.eigenvalues_` is truncated to `n_components` (the number of output dimensions requested from Isomap). The pipeline's stated audit goal — "large positive eigenvalues with a steep dropoff, and explicit detection of large negative eigenvalues" — is unanswerable from this attribute alone: negative eigenvalues, by construction, are never among the top `n_components` positive ones that `KernelPCA` retains, so `eigenvalues_` cannot show a negative tail even when one exists.

**Why it happens:** `KernelPCA` (which Isomap uses internally) is written to return the top-k eigenpairs for embedding purposes, not to expose the full spectrum for diagnostic purposes. It is an easy, silent mistake to read `isomap.kernel_pca_.eigenvalues_`, see all-positive values, and conclude the geodesic distance matrix is Euclidean-embeddable when the truncation itself guarantees that appearance.

**How to avoid:**
- Recompute the full eigenspectrum manually: take `isomap.dist_matrix_` (the N×N geodesic distance matrix), apply double-centering (`B = -0.5 * J @ (D**2) @ J` where `J = I - (1/n) * ones(n,n)`), and get the complete eigenvalue set via `scipy.linalg.eigvalsh(B)` (symmetric eigensolver, returns all N eigenvalues including negative ones, ascending order).
- Report: (a) the top-20 positive eigenvalues and their cumulative-variance dropoff, (b) the most negative eigenvalue(s) and their magnitude relative to the largest positive eigenvalue (a ratio `|λ_min_neg| / λ_max_pos` is the standard non-Euclideanity diagnostic — large ratios indicate severe non-Euclidean structure, e.g. from short-circuiting or a fundamentally non-manifold point cloud).
- Do this audit *before* deciding on the Isomap output dimensionality used for the decoder, since a bad ratio should change the `n_neighbors`/dimensionality decision, not just get logged and ignored.

**Warning signs:** Audit code that reads `.kernel_pca_.eigenvalues_` and reports "no negative eigenvalues found" without ever having computed the full spectrum — this is a false negative by construction, not a real finding.

**Phase to address:** Eigenspectrum-audit phase (pipeline step 3) — this is the phase's core deliverable and must not be built on the truncated attribute.

---

### Pitfall 4: Euclidean Isomap on embeddings conventionally compared by cosine similarity

**What goes wrong:**
Isomap's default metric is Minkowski p=2 (Euclidean). For L2-normalized vectors, squared Euclidean distance and cosine similarity are monotonically related (`‖a−b‖² = 2(1 − cos(a,b))`), so neighbor *rankings* are identical either way and the choice is cosmetic. But if the `dinov3_vitb16_legacysurvey` embeddings stored in the HF dataset are **not** pre-normalized (raw CLS-token output, with meaningful norm variation e.g. from image brightness/contrast/exposure differences across the survey), Euclidean and cosine neighborhoods diverge — Euclidean distance is sensitive to vector magnitude, cosine is not — and Isomap would be reconstructing a manifold defined by a notion of "closeness" that has nothing to do with the angular/cosine geometry conventionally used to compare foundation-model embeddings (including, plausibly, inside DINOv3's own training objective and inside the origin paper's own MKNN computation, which is typically run on cosine-normalized features).

**Why it happens:** It is easy to hand a raw `(N, 768)` float array straight to `Isomap()` without checking `np.linalg.norm(X, axis=1)` first — the code runs without error either way, and the resulting embedding "looks like" a reasonable 2D/3D manifold regardless of which metric was used, so nothing visibly signals the mismatch.

**How to avoid:**
- Before fitting, compute and histogram `np.linalg.norm(X, axis=1)` for both `_legacysurvey` and `_hsc` columns. If norms vary by more than a few percent across the sample, Euclidean and cosine neighborhoods will differ materially.
- Default to L2-normalizing the embeddings onto the unit hypersphere before Isomap (`X / np.linalg.norm(X, axis=1, keepdims=True)`), which makes the Euclidean base metric equivalent to cosine and reconciles Isomap's manifold with the geometry MKNN implicitly uses (MKNN itself should also be computed on the same normalization convention — check this in the same phase, since a mismatch between the two would reintroduce the same problem one level up).
- Alternatively, pass `metric="cosine"` directly to `Isomap(...)` (scikit-learn's `Isomap` forwards its `metric` parameter to the internal `NearestNeighbors` and uses the same metric to compute geodesic edge weights) — but note cosine "distance" (`1 − cos`) is not a true metric in the strict sense used for MDS's Euclidean-embeddability assumptions in the same way L2-normalized Euclidean distance is, so normalizing-then-using-Euclidean is the more defensible choice for compatibility with the classical-MDS eigenspectrum audit in Pitfall 3.
- Record which choice was made and why in the notebook's own documentation, since this decision changes essentially everything downstream (neighbor graph, geodesic distances, eigenspectrum, decoder target, curvature).

**Warning signs:** Norm histogram with visible spread (coefficient of variation > ~5%) rather than a tight spike near 1; Isomap neighbor graph and a separately-computed cosine k-NN graph (`sklearn.neighbors.NearestNeighbors(metric="cosine")`) disagreeing on more than a small fraction of each point's neighbor set when checked directly.

**Phase to address:** Data-loading / subsampling phase (pipeline step 1), decided and locked in before Isomap fitting (step 2) — this is a preprocessing decision, not something to patch after the fact.

---

### Pitfall 5: Subsampling 10k of 101,725 systematically inflates geodesic distances and is not obviously comparable across draws

**What goes wrong:**
Isomap's graph shortest-path distances converge to true manifold geodesics only as sampling density increases (in the limit, neighborhood radius → 0 requires N → ∞). With finite, and especially *subsampled*, data, shortest paths zigzag along the discrete point cloud rather than following a smooth geodesic, producing systematic inflation of estimated geodesic distance relative to the true (unknown) manifold distance — an effect documented in the manifold-learning literature as worsening precisely when landmark/subsampling acceleration schemes are used. Two consequences for this pipeline: (1) the eigenspectrum shape (Pitfall 3), the decoder's target coordinates, and the curvature field are all functions of *this specific 10k draw's* density pattern, not of some draw-independent "true" curvature of the full 101,725-point manifold; (2) local density is not uniform across a random 10k subsample of an already non-uniformly-sampled astronomical survey, so the zigzag-inflation effect itself varies by region — sparser regions get more geodesic-distance inflation, which (per Pitfall 7) is exactly the same regions likely to be flagged as "high curvature."

**Why it happens:** 10k was chosen for a concrete, defensible engineering reason (keeps the dense geodesic matrix at ~800 MB so Isomap stays exact rather than landmark-approximated — see PROJECT.md Key Decisions), but that choice was not driven by, and should not be conflated with, geometric fidelity to the full manifold.

**How to avoid:**
- Never report curvature results from a single subsample draw without a stability check. Repeat the full pipeline (Isomap → decoder → curvature) on at least 3-5 independent random 10k draws (different seeds) from the 101,725 rows, and report whether the high/low curvature partition and the resulting MKNN comparison are stable across draws (e.g., Jaccard overlap of the "high-curvature" point sets across draws, restricted to points common to multiple subsamples where possible via the row index).
- As a secondary check, run the same pipeline at two subsample sizes (e.g., 5k and 15k, memory permitting: 15k dense float64 matrix is ~1.8 GB) and check whether the eigenspectrum elbow location and the qualitative high/low curvature split are consistent — if the split reorganizes substantially with sample size, curvature is measuring subsampling density, not manifold shape.
- Explicitly state in the writeup that "curvature" in this pipeline is a property of *this specific finite sample's* graph-geodesic reconstruction, not a population-level geometric invariant — this is a correct and defensible framing as long as it is stated, not implied to be more than it is.

**Warning signs:** High/low curvature region assignment changes substantially (e.g., > 20-30% of points reassigned) between two random subsamples of the same size; eigenspectrum elbow location shifts by more than 1-2 components between subsample sizes.

**Phase to address:** Subsampling phase (pipeline step 1) sets the seed/config discipline; the stability re-run is a validation task that belongs at the end of the curvature phase (step 5/6), before the MKNN comparison (step 7) is treated as final.

---

### Pitfall 6: Curvature is a property of the fitted decoder function, not necessarily of the data manifold

**What goes wrong:**
This is the central methodological risk in the pipeline. The mean curvature field is derived analytically from the decoder `f: R^d → R^768` via `torch.func` Jacobian/Hessian — but `f` is a neural network fit to a finite set of (Isomap-coordinate, embedding) training pairs, and *any* smooth interpolant through those points has curvature everywhere between the training points that is unconstrained by the data. An overfit decoder (too much capacity, too little regularization, trained too long) will wiggle between sparse training points and manufacture large `‖H‖` values that have nothing to do with the geometry of the underlying data manifold — they are an artifact of how the network chose to interpolate the gaps. An underfit decoder does the opposite: it smooths everything into a low-order (often near-linear) map and reports near-zero curvature everywhere, silently erasing real structure. Because sparse regions are *both* where overfitting is least constrained *and* (per Pitfall 5) where geodesic-distance inflation is worst, "high curvature" in this pipeline is disproportionately likely to mean "sparse/underconstrained region," not "genuinely bent region of the manifold" — this is the confound formalized in Pitfall 7.

**Why it happens:** Curvature requires second derivatives, and second derivatives are the *most* sensitive property of a fitted function to overfitting — first derivatives (tangent directions) are comparatively well-constrained by nearby training points, but curvature can differ wildly between two decoders that both achieve near-identical training loss, if they differ in how they interpolate between points. There is no way to "read off" curvature from data directly; it only exists conditional on a choice of smooth parameterization, and the pipeline's parameterization is a learned one.

**How to avoid — concrete falsification protocol (all of the following, not just one):**
1. **Train/validation split with held-out curvature audit.** Split the 10k Isomap-coordinate → embedding pairs into train/val (e.g., 85/15). Fit the decoder on train only. Compute held-out reconstruction error (`‖f(u_val) − x_val‖`) per validation point, and correlate it (Spearman) against `‖H‖` estimated at nearby training points. **If held-out error correlates strongly with local curvature (say |r| > 0.4), that is direct evidence the "curvature" signal is co-located with poor generalization, i.e., overfitting, not real geometry** — this is the single most important diagnostic and should gate whether the curvature results are reported at all.
2. **Capacity sweep.** Fit decoders across a range of capacities (e.g., hidden width 32/64/128/256, depth 2/3/4, with and without weight decay / dropout). For each, compute the high/low curvature quantile partition. **Report the partition's stability (e.g., pairwise Jaccard overlap of the "high-curvature" point set) across capacities.** A real geometric signal should be roughly stable as capacity increases past some minimum; a signal that reshuffles freely with capacity is a fitting artifact, not a property of the data.
3. **Synthetic-flat-manifold negative control.** Construct a synthetic dataset with the *same* Isomap-coordinate distribution (same 10k points in `R^d`, same density pattern) but a known analytically-flat embedding (e.g., a random linear map `R^d → R^768`, or the actual PCA/linear reconstruction of the real Isomap coordinates, which is curvature-zero by construction). Run the identical decoder-training + curvature pipeline on this synthetic target. **The pipeline should report near-zero `‖H‖` everywhere on this control; if it instead reports a comparable-magnitude, similarly quantile-separable curvature field on data that is curvature-free by construction, the entire curvature signal is decoder noise and the real-data results cannot be trusted.** This is a true specificity/false-positive-rate check and should be run before any real-data curvature numbers are reported.
4. **Independent non-learned curvature baseline.** Compute a local, non-learned curvature proxy directly in ambient space — e.g., local PCA on each point's k-nearest-neighbor ambient-space neighborhood (fit a local quadratic/tangent-plane patch, curvature from the residual normal-direction variance) — and correlate it against the decoder-derived `‖H‖`. **If the two are not meaningfully correlated (Spearman r near zero), distrust the decoder-based curvature** — the local-PCA baseline is far less flexible and much less prone to overfitting-driven artifacts, so it is a reasonable (if noisier) sanity check, not a replacement.
5. **Seed-ensemble stability.** Retrain the decoder from several random initializations (same architecture, same data, different seeds). Compute per-point `‖H‖` for each seed and check cross-seed agreement (Spearman correlation of per-point `‖H‖`, and Jaccard overlap of the high-curvature quantile set). **Curvature that is not reproducible across training seeds at fixed architecture and data is measuring optimization noise, not geometry**, and must not be reported as a stable finding.
6. **Activation and Hessian sanity check.** ReLU has an identically-zero second derivative almost everywhere (undefined/Dirac-delta at kinks), so a ReLU decoder would report `H ≡ 0` almost everywhere by construction — this is already correctly avoided by the C2-smooth-activation decision in PROJECT.md, but must be *verified*, not assumed: after fitting, sample `torch.func.hessian` output at a batch of points away from training nodes and assert (a) values are non-zero and (b) no NaN/Inf (some smooth activations, e.g. unscaled `softplus`, can saturate and produce numerically-zero second derivatives at large pre-activations, silently reproducing the ReLU failure mode).

**Warning signs:** High-curvature quantile membership changes substantially across the capacity sweep or the seed ensemble; held-out reconstruction error and `‖H‖` are visibly correlated on a scatter plot; the synthetic-flat control produces a curvature distribution with a similar spread/shape to the real-data one.

**Phase to address:** Decoder-training phase (pipeline step 4) must build in the train/val split, capacity sweep, and activation sanity check as part of the training procedure itself, not as an afterthought. The synthetic-flat-manifold control and the non-learned local-PCA baseline belong in the curvature-computation phase (step 5), run and reported *before* the high/low quantile split (step 6) is treated as meaningful.

---

### Pitfall 7: The density confound in regional MKNN — high/low curvature regions may just be sparse/dense regions

**What goes wrong:**
This is the second central risk, and it compounds directly with Pitfall 6. Even if the decoder-curvature falsification protocol above is satisfied, there remains a mechanical, non-representational reason why a low-curvature-vs-high-curvature MKNN comparison could show a difference: **local sampling density is very plausibly correlated with both `‖H‖` and with MKNN's own statistical behavior, through entirely separate mechanisms, in the same direction, which would produce a spurious "high curvature → low alignment" result that has nothing to do with representational convergence.**

The mechanisms, concretely:
1. **Density → apparent curvature (upstream, via Pitfall 6).** Sparse regions leave the decoder underconstrained between training points, mechanically inflating `‖H‖` there independent of true manifold shape.
2. **Density → geodesic-distance inflation (via Pitfall 5).** Sparse regions have more zigzag-inflated Isomap geodesic distances, which can also distort the local decoder-coordinate geometry.
3. **Density → MKNN statistical noise, independent of both of the above.** MKNN is itself a k-NN-based statistic. In a sparse region, "the k nearest neighbors" of a point are farther away in absolute embedding-space terms and less stable — small, essentially meaningless numerical differences between the *same* object's HSC and Legacy-Survey embeddings become more likely to reorder which points fall inside vs. outside the k-NN set, mechanically **lowering** MKNN in sparse regions even when the two modalities are, in every meaningful sense, equally well aligned there. This is a variance-inflation-from-sparsity artifact operating on the metric itself, not on the underlying representations. It is documented in adjacent literature that neighborhood-overlap alignment metrics (the MutualNN/CKNNA family MKNN belongs to) are highly sensitive to local density and "hubness" effects — a small number of high/low-degree points can dominate the score.
4. **Boundary/edge effects.** Points near the boundary of the sampled 10k point cloud (i.e., far from the Isomap-coordinate centroid) have one-sided, asymmetric neighborhoods. This inflates decoder extrapolation error (hence apparent curvature, compounding #1) *and* independently degrades k-NN-based metrics near the edge of any finite sample, for reasons unrelated to representation quality.

If all four mechanisms push in the same direction — sparse/edge regions get both nominally "higher curvature" *and* lower, noisier MKNN — then the pipeline's central comparison (does MKNN differ by curvature region?) can produce a significant-looking result that is entirely explained by density and sample-boundary effects, with zero contribution from actual representational geometry. Given the origin paper's crossmodal Legacy Survey MKNN is only 0.4-2% to begin with (Pitfall 8), the headroom for a genuine curvature effect is tiny relative to the plausible size of this confound.

**Why it happens:** Curvature-region definition (by `‖H‖` quantile) and density are never independently randomized — they are both functions of the same underlying point positions — so any naive comparison of MKNN across curvature quantiles is, by construction, also a comparison across whatever density pattern happens to correlate with curvature in this particular fitted decoder on this particular subsample.

**How to avoid — concrete diagnostics and controls (required, not optional):**
1. **Measure the confound directly before doing anything else.** Compute a local-density proxy per point — e.g., mean distance to its k-th nearest neighbor in Isomap-coordinate space (or in ambient 768-d space) — and compute Spearman correlation against `‖H‖`. **Report this number explicitly.** If `|r|` is large (a reasonable red-flag threshold: > 0.3–0.5), the curvature-quantile split is substantially a density split in disguise, and the headline comparison must not be reported without the controls below.
2. **Compute and report distance-from-centroid per point** (distance from each point's Isomap coordinate to the coordinate-space centroid of the 10k sample). Check whether high-curvature points cluster near the boundary (e.g., compare the distance-from-centroid distribution of the high-curvature quantile vs. the full sample via a Kolmogorov-Smirnov test). If they do, boundary artifacts (mechanism 4) are likely driving part of the curvature signal, and boundary points should be trimmed (e.g., exclude the outermost decile by centroid distance) before defining regions, with results reported both with and without trimming.
3. **Partial-correlation / multiple-regression control.** Fit `MKNN_i ~ curvature_i + density_i + centroid_distance_i` (per-point or per-small-bin MKNN, or a permutation-based per-region equivalent) and report curvature's *partial* effect after controlling for density and boundary distance. **If density and/or centroid-distance absorb most of the explanatory power and curvature's partial coefficient is not significant, the "curvature causes lower alignment" story does not hold** — the correct conclusion is "sparse/edge regions have both nominally higher curvature and noisier MKNN," which is a different (and much weaker) finding than the headline claim.
4. **Density-matched region comparison as the primary robustness check.** Rather than (or in addition to) a raw `‖H‖`-quantile split, stratify points into density bins first, then split by curvature *within* each density stratum, and compare MKNN across curvature levels only within matched-density strata. **If the curvature effect disappears or reverses under density matching, the un-matched result was confounded** and must be reported as such, not suppressed.
5. **Size- and density-matched permutation null.** The planned permutation null (pipeline step 7) must draw permutation regions matched in **size** to the actual high/low curvature groups (not merely a global permutation), and a second, density-matched null variant should also be reported: permute region *labels* only among points of similar local density, so the null itself reflects the same density-driven MKNN noise floor that the real comparison is subject to. If the real curvature-region gap is not distinguishable from this density-matched null, the effect is not attributable to curvature.

**Warning signs:** `‖H‖` and local density are visibly anti/correlated in a scatter plot; the high-curvature quantile's centroid-distance distribution is visibly shifted toward the boundary relative to the full sample; the curvature effect on MKNN shrinks toward zero (or reverses sign) once density-matching or partial regression is applied.

**Phase to address:** This is a dedicated validation sub-phase that must sit between curvature computation (step 5/6) and the final MKNN comparison (step 7) — it should not be folded silently into step 7's headline number. The density/centroid-distance diagnostics (points 1-2) can be computed as soon as curvature is available (step 5); the partial-regression and density-matched controls (points 3-5) are prerequisites for treating step 7's result as interpretable.

---

### Pitfall 8: Statistical validity — near-zero headroom, quantile-threshold selection after seeing the data, and multiple comparisons

**What goes wrong:**
The origin paper (arXiv:2509.19453) reports crossmodal MKNN for Legacy Survey at only 0.4-2%, against a permutation-null baseline that is itself close to zero by construction (`k/n` for random overlap). This leaves very little headroom for a genuine high-vs-low-curvature difference to be both real and detectable with ~10k points split into two regions (so ~1,250-2,500 points per region at a top/bottom-quartile split, fewer at more extreme splits). Two further risks compound this: (a) if the curvature quantile threshold (top/bottom X%) is chosen, adjusted, or swept *after* looking at how the MKNN comparison comes out, this is textbook researcher-degrees-of-freedom / data-dredging, and a "significant" split can be found by trying several thresholds even under the null; (b) reporting several cut points (median split, quartiles, deciles) as if they were independent confirmatory findings inflates the effective false-positive rate.

**Why it happens:** Quantile thresholds feel like a free, harmless hyperparameter, and it is natural to try a few and see which "looks cleanest" — but with an effect size this small relative to noise, that practice can manufacture an apparently robust finding from pure sampling variation.

**How to avoid:**
- **Pre-specify the primary comparison before computing MKNN.** Fix one curvature split (e.g., top 25% vs bottom 25% by `‖H‖`, or top/bottom halves) based on diagnostics from Pitfalls 6-7 alone (decoder validity, density-confound magnitude) — never based on which split maximizes the MKNN gap. Write the chosen split into the notebook/config *before* running the MKNN cell.
- **Power/headroom check before running the comparison.** Using the permutation null's own empirical standard deviation at the actual region size (~1,250-2,500 points), estimate whether a plausible effect (e.g., MKNN doubling from 1% to 2%, matching the upper end of the paper's reported range) would be statistically distinguishable from noise at this n. If a simulation/analytic check shows the comparison is underpowered for any effect smaller than, say, a 3-5x change, state this limitation explicitly *before* reporting a null or marginal result as "no effect" — an underpowered null is not evidence of no effect.
- **Bootstrap CIs on both regions, require non-overlap or a formal two-sample test on the bootstrap difference distribution** (already planned per PROJECT.md) rather than eyeballing point estimates; report the CI width alongside the point estimate so readers can judge headroom for themselves.
- **Treat any non-primary cut point as explicitly exploratory**, labeled as such, with either a Bonferroni-style correction for the number of cuts tried or a clear statement that secondary cuts are hypothesis-generating only.
- **Report honestly if the result is null or ambiguous.** Given the small headroom, a null regional result is plausible and explicitly anticipated in PROJECT.md's Key Decisions — the correct scientific behavior is to report it as such, not to keep adjusting thresholds/k until something crosses a significance-looking line.

**Warning signs:** The chosen quantile split is only decided/finalized after an MKNN-comparison cell has already been run once; multiple splits are computed and only the "best-looking" one appears in the final notebook; CI on the MKNN difference includes zero but the writeup states a directional conclusion anyway.

**Phase to address:** MKNN-comparison phase (pipeline step 7) — the pre-registration of the split and the power check must happen at the start of this phase, before the first MKNN number is computed, and the phase's exit criteria should require the CI/permutation-null comparison to be reported regardless of outcome.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|-----------------|------------------|
| Fitting Isomap once at a "reasonable-looking" `n_neighbors` without a sweep | Saves compute time, faster to a plot | Undetected short-circuit or disconnection silently invalidates everything downstream | Never for the reported result; fine for a first exploratory pass explicitly labeled as such |
| Reading `isomap.kernel_pca_.eigenvalues_` for the negative-eigenvalue audit | Zero extra code | Structurally cannot detect the negative tail (truncated attribute) — false confidence | Never |
| Training the decoder once, at one capacity, and reading off curvature | Fast iteration | No way to distinguish real curvature from decoder-fitting artifact (Pitfall 6) | Only for early prototyping plots explicitly marked "not yet validated" |
| Splitting curvature regions by a single unmatched `‖H‖` quantile | Simple, fast | Confounded by density/boundary effects (Pitfall 7); result is not interpretable as "curvature causes alignment differences" | Only as a first-pass visualization, never as the reported finding |
| Caching the 10k subsample / Isomap fit / decoder weights to disk between notebook sessions without a config fingerprint | Avoids re-running expensive cells | Silent staleness when upstream parameters change (Pitfall 10) | Acceptable only if the cache filename or an adjacent manifest encodes every parameter that affects it |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|-----------------|-------------------|
| `datasets` streaming of `UniverseTBD/pu-embeddings` | Subsampling the two paired columns via separate `.shuffle()`/`.select()` calls, silently breaking row alignment since there is no `object_id` to catch the mismatch | Select a single fixed set of row indices once (`rng.choice(101725, 10000, replace=False)` with a logged seed) and apply that same index array to both `_hsc` and `_legacysurvey` views/slices of the same underlying table — never shuffle the two columns independently |
| `sklearn.manifold.Isomap` | Treating a UserWarning about connected components as noise to ignore | Fail loudly: assert `n_connected_components == 1` after fit, or explicitly document and sweep the bridging behavior (Pitfall 1) |
| `torch.func.hessian`/`jacrev` on a 768-d output decoder | Computing the full `(768, d, d)` Hessian without checking for NaN/Inf from activation saturation | Assert finiteness of Jacobian/Hessian outputs on a validation batch immediately after training, before computing curvature at scale |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|-----------------|
| Dense N×N geodesic distance matrix (`isomap.dist_matrix_`, float64) | Notebook kernel OOM-kills or swaps heavily during/after `.fit()` | Confirm subsample size before fitting: 10k × 10k × 8 bytes ≈ 800 MB for the matrix alone, but peak transient memory during fit (neighbor graph + shortest-path buffers + KernelPCA working arrays) can run 2-4x that; budget ≥ 3-4 GB free RAM for a 10k fit, and never attempt Isomap on the full 101,725 rows (101725² × 8 bytes ≈ 82.8 GB dense — must use landmark/Nyström Isomap or an entirely different approach if the full dataset is ever needed) | Scales O(N²) in memory; becomes unworkable well before 101,725 |
| Per-point `torch.func.hessian` over all 10k points without batching/vmap | Curvature computation runs far longer than expected, or exhausts GPU/CPU memory | Use `torch.func.vmap` over batches of points rather than a Python loop; checkpoint intermediate `‖H‖` results to disk incrementally so an OOM mid-run doesn't lose all completed work | Becomes noticeable well under 10k points if done point-by-point in eager Python |

## Security Mistakes

Not directly applicable — this is a local research notebook against a public, unauthenticated HF dataset with no user data or credentials involved. The only relevant note: pin `datasets`/`huggingface_hub` versions and avoid executing arbitrary dataset-provided code (not applicable here since this is a plain parquet-backed dataset, not one with a custom loading script).

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-------------------|
| Reporting a single curvature plot / MKNN bar chart with no uncertainty or diagnostics | Reader cannot tell a real finding from noise or an artifact | Always pair the headline MKNN-by-region result with: the decoder validation diagnostics (Pitfall 6), the density-confound correlation (Pitfall 7), and the bootstrap CI / permutation null (Pitfall 8) in the same figure or an adjacent diagnostic panel |
| Silently dropping points excluded for numerical-conditioning reasons (Pitfall 9) without reporting how many/where | Reader cannot judge whether the reported regions are representative | Report the count and rough distribution (e.g., in curvature-magnitude and Isomap-coordinate space) of any excluded points |

## "Looks Done But Isn't" Checklist

- [ ] **Isomap fit:** Often missing a connected-components check — verify `scipy.sparse.csgraph.connected_components` on the fitted neighbor graph reports exactly 1 component, or that the `n_neighbors` sweep and its rationale are documented.
- [ ] **Eigenspectrum audit:** Often computed from the truncated `kernel_pca_.eigenvalues_` — verify the full spectrum was recomputed via manual double-centering of `isomap.dist_matrix_` and `scipy.linalg.eigvalsh`, and that the negative-eigenvalue-ratio diagnostic is reported as a number, not just "looked fine."
- [ ] **Decoder training:** Often trained once at one capacity with only a training-loss curve shown — verify a train/val split, a capacity sweep, and the synthetic-flat-manifold negative control (Pitfall 6) were all run and their results (not just the final decoder) are part of the notebook output.
- [ ] **Curvature field:** Often reported as `‖H‖` without checking the metric tensor's condition number — verify `cond(g)` was computed per point and any near-singular points were flagged/excluded (Pitfall 9) before the quantile split.
- [ ] **Curvature-region split:** Often a single unmatched quantile split — verify the density/centroid-distance confound diagnostics (Pitfall 7) were computed and reported alongside the split, not just the split itself.
- [ ] **MKNN comparison:** Often reported as a single point estimate — verify bootstrap CIs, a size-matched permutation null, and a density-matched permutation null (Pitfall 8) are all present, and that the quantile threshold was fixed before the comparison was first run.
- [ ] **Notebook reproducibility:** Often "worked once" in an out-of-order execution — verify a clean Restart & Run All reproduces the reported numbers, with all seeds fixed (subsample selection, torch training, Isomap's `eigen_solver` if randomized).

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|----------------|------------------|
| Discovered late that `n_neighbors` was in the short-circuit regime | MEDIUM | Re-run the `n_neighbors` sweep diagnostics (Pitfall 2), pick a validated value, and re-run the full pipeline from Isomap onward — earlier decoder/curvature results must be discarded, not patched |
| Discovered late that curvature correlates strongly with held-out decoder error (overfitting) | MEDIUM-HIGH | Reduce decoder capacity or add regularization, re-run the capacity sweep and synthetic-flat-manifold control (Pitfall 6) until the diagnostic clears, then redo curvature and everything downstream |
| Discovered late that curvature strongly confounds with density | HIGH | Re-run the MKNN comparison using the density-matched stratification and partial-regression control (Pitfall 7) — this does not require redoing Isomap/decoder, only the region-definition and comparison steps, so cost is lower than it looks provided curvature values themselves are trusted |
| Discovered late that the reported eigenspectrum audit used the truncated `kernel_pca_.eigenvalues_` | LOW | Recompute the full spectrum from the already-saved `isomap.dist_matrix_` via double-centering + `eigvalsh` — no re-fitting needed if `dist_matrix_` was cached |
| Discovered late that HSC/Legacy row alignment was broken by independent shuffling | HIGH | The entire MKNN comparison is invalid; must re-subsample from source with a single shared index array and rerun the crossmodal step (steps 1 and 7) — earlier curvature analysis (which only used one modality) is unaffected |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-------------------|----------------|
| 1. Silent Isomap graph bridging | Isomap-fitting phase (step 2) | `connected_components(...) == 1` asserted and logged before proceeding |
| 2. Short-circuit edges / bad `n_neighbors` | Isomap-fitting + eigenspectrum-audit phases (steps 2-3) | `n_neighbors` sweep table (reconstruction error, component count, eigenspectrum elbow) present in notebook |
| 3. Truncated eigenvalue attribute hides negative tail | Eigenspectrum-audit phase (step 3) | Full spectrum computed via manual double-centering + `eigvalsh`; negative-eigenvalue ratio reported as a number |
| 4. Euclidean vs cosine metric mismatch | Data-loading/subsampling phase (step 1) | Norm histogram reported; normalization or `metric` choice documented and consistent with MKNN's own metric convention |
| 5. Subsampling density inflates geodesic distances | Subsampling phase (step 1) sets seed discipline; validated at end of curvature phase (step 5/6) | Multi-seed subsample stability check (Jaccard overlap of high-curvature sets) reported |
| 6. Curvature is a property of the fitted decoder | Decoder-training phase (step 4) builds controls; curvature-computation phase (step 5) runs the negative control | Held-out error vs. curvature correlation, capacity-sweep stability, synthetic-flat-manifold control, non-learned baseline correlation, and seed-ensemble stability all reported |
| 7. Density confound in regional MKNN | Dedicated validation sub-phase between curvature (step 5/6) and MKNN comparison (step 7) | Density/centroid-distance correlation with `‖H‖` reported; partial-regression and density-matched-null results reported alongside the headline MKNN gap |
| 8. Statistical validity / pre-registration | MKNN-comparison phase (step 7) | Quantile split fixed before first MKNN run; power/headroom check documented; bootstrap CIs and size-and-density-matched permutation nulls reported |
| 9. Numerical conditioning of the metric tensor | Curvature-computation phase (step 5) | `cond(g)` computed per point; near-singular points flagged/excluded with a reported count before the quantile split (step 6) |
| 10. Notebook memory blowup / non-determinism / stale cache | Applies across all phases; enforced at notebook-infrastructure level | Peak-memory budget checked before full-dataset temptation; all seeds fixed; cache files keyed by a config fingerprint/manifest; Restart & Run All reproduces reported numbers |

### Pitfall 9 (referenced above, detailed here): Numerical conditioning of the metric tensor g_ij

**What goes wrong:** The first fundamental form is `g = JᵀJ` where `J` is the decoder's `(768, d)` Jacobian at a point (`d` = Isomap output dimensionality). Mean curvature requires `g⁻¹`. If the decoder's local Jacobian is close to rank-deficient — e.g., two nearby Isomap coordinates map to nearly the same point in 768-d, or the decoder has locally collapsed a dimension — `g` becomes near-singular and `g⁻¹` (hence `‖H‖`) blows up numerically without corresponding to any real geometric feature. This is distinct from Pitfall 6 (overfitting-driven curvature) — it is a pure numerical-conditioning failure that can occur even in a well-validated decoder, at isolated points.

**How to avoid:** Compute `cond(g)` (ratio of largest to smallest singular value of `g`, e.g., via `torch.linalg.cond` or SVD of `J`) at every point alongside `‖H‖`. Set a threshold (e.g., `cond(g) > 1e6`) above which points are flagged and excluded from the quantile split and MKNN comparison, with the count and location of excluded points reported (not silently dropped). Also cross-check against the eigenspectrum audit (Pitfall 3): if a given Isomap output dimension has a near-zero eigenvalue, that coordinate carries little real signal and is a likely source of Jacobian collapse — consider dropping that dimension from `d` rather than patching the symptom downstream.

**Warning signs:** A small number of points with `‖H‖` orders of magnitude larger than the rest of the distribution (heavy-tailed outliers in an otherwise smooth-looking curvature histogram); those same points showing very high `cond(g)`.

**Phase to address:** Curvature-computation phase (step 5), before the quantile split (step 6).

### Pitfall 10 (referenced above, detailed here): Notebook-specific silent failures

**What goes wrong — three distinct failure modes:**
1. **Memory blowup.** A dense 10k×10k float64 geodesic matrix is ~800 MB; combined with neighbor-graph and KernelPCA working memory, peak usage during `.fit()` can reach several GB. The full 101,725-row dataset would require an ~83 GB dense matrix — categorically infeasible without landmark/Nyström approximation, and must never be attempted directly even "just to see."
2. **Non-determinism.** Sources of unseeded randomness compound silently: the initial row subsampling from the HF dataset, Isomap's `eigen_solver` (if `'arpack'` or `'randomized'` rather than `'dense'` or `'auto'` with a fixed seed), and torch decoder training (weight init, any stochastic data ordering). Re-running the notebook can shift curvature values and region membership enough to change conclusions, without any code change.
3. **Stale cached intermediates.** Long-running steps (Isomap fit, decoder training) are natural candidates for `pickle`/`torch.save` caching between sessions. If the upstream config changes (different `n_neighbors`, different subsample seed, different decoder architecture) but the cache filename doesn't change, later cells silently consume stale data that no longer matches the code that appears to have produced it — a classic notebook hidden-state bug, worsened here because a pickled `Isomap` or `state_dict` does not self-describe its fitting hyperparameters when reloaded.

**How to avoid:**
- Compute and print expected peak memory for the chosen subsample size before fitting; never scale up without recomputing this.
- Fix every seed explicitly at the top of the notebook (subsample RNG, `torch.manual_seed`, Isomap's `random_state` if using the randomized eigensolver) and verify a full Restart & Run All reproduces reported numbers bit-for-bit or within documented tolerance.
- Key every cached artifact's filename (or an adjacent JSON manifest) to a hash/fingerprint of the exact parameters that produced it (subsample seed + size, `n_neighbors`, decoder architecture + training config); on load, assert the manifest matches the current notebook config before trusting the cache, and regenerate on mismatch rather than silently reusing.

**Warning signs:** Numbers change between re-runs of the same notebook without code changes; a cached `.pkl`/`.pt` file loads successfully but its shape/parameter count doesn't match what the current code would produce; memory usage climbing unexpectedly during Isomap `.fit()` on a machine with limited RAM.

**Phase to address:** Cuts across all phases; should be enforced as a standing notebook-infrastructure discipline from the subsampling phase (step 1) onward, with an explicit Restart & Run All check as an exit criterion for the milestone as a whole.

## Sources

- [scikit-learn manifold learning user guide](https://scikit-learn.org/stable/modules/manifold.html) — computational complexity, general Isomap tips including short-circuiting note (MEDIUM confidence, official docs)
- [scikit-learn `_isomap.py` source, GitHub](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_isomap.py) — disconnected-component warn-and-bridge behavior, `metric` parameter forwarding including `'cosine'`/`'precomputed'` support, `kernel_pca_.eigenvalues_` truncation (MEDIUM confidence, verified against source)
- [scikit-learn PR #21915 "FIX Isomap with precomputed distances and disconnected graph"](https://github.com/scikit-learn/scikit-learn/pull/21915) and [PR #20531](https://github.com/scikit-learn/scikit-learn/pull/20531) — `_fix_connected_components` mechanism (MEDIUM confidence)
- Web search synthesis on short-circuit edges / noisy off-manifold points in Isomap (MEDIUM confidence, corroborated across multiple independent sources)
- Web search synthesis on classical-MDS negative eigenvalues as a non-Euclidean-embeddability diagnostic, and the positive/negative eigenvalue-ratio heuristic (MEDIUM confidence)
- Web search synthesis on subsampling/landmark acceleration worsening "spurious geodesic curvature" from finite-sample zigzagging (MEDIUM confidence)
- Web search synthesis on MutualNN/CKNNA-family neighborhood-overlap alignment metrics' documented sensitivity to perturbation, hubness, and outliers (MEDIUM confidence) — direct support for the density-confound mechanism in Pitfall 7
- Web search on decoder/autoencoder curvature-from-Hessian approaches and adjacent literature flagging decoder-overfitting artifacts (LOW confidence — no single canonical falsification protocol found in the literature; the six-part protocol in Pitfall 6 is original methodological synthesis for this pipeline, grounded in standard ML validation practice (train/val split, ablation/capacity sweep, negative controls, seed ensembles) applied to differential-geometric quantities)
- Standard differential geometry (immersion Jacobian/Hessian, first/second fundamental forms, mean curvature vector, metric-tensor conditioning) — textbook-level domain knowledge, not independently web-verified for this write-up but mathematically standard
- `/home/akagi/Documents/Projects/EffDim/.planning/PROJECT.md` — pipeline scope, dataset structure, and Key Decisions (subsample size rationale, decoder activation choice, single-model-single-milestone scope) used to ground pitfall applicability

---
*Pitfalls research for: manifold-curvature analysis of PU foundation-model embeddings (EffDim v1.1)*
*Researched: 2026-07-29*
