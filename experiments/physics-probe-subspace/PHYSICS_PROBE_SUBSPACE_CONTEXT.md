# PHYSICS PROBE SUBSPACE CONTEXT

Detailed summary of the research conversation leading to the proposal of the **Physics Probe Subspace** alignment experiment. This methodology builds upon the findings in the `sae-shared-basis` geometry research (where unsupervised low-rank projections degrade $\text{mKNN}$ and affine Ridge maps on SAE codes succeed) and proposes a mathematically rigorous supervised alternative.

---

## 1. Background: The Failure Modes of Unsupervised Projections
Our prior empirical work (detailed in `experiments/SAE-shared-basis/CONTEXT.md`) revealed the following geometric realities about foundation model embeddings (e.g., ViT-B ↔ DINOv3 on `Smith42/galaxies`):
* **The "Soft Shell" Problem:** Embeddings form a ~10D linear core surrounded by a thick, soft shell extending to median $d \sim 87$ (for 95% variance). This soft shell is not random noise; it contains fine-grained, distributed semantic features essential for determining local neighborhood ranking (e.g., $\text{mKNN}@10$).
* **Truncation Discards Signal:** Unsupervised dimensionality reduction (PCA, standard autoencoder bottlenecks) indiscriminately truncates this soft shell. This explains why PCA-40 $\text{mKNN}$ drops below full ambient cosine $\text{mKNN}$.
* **Arbitrary Scalings ("Swaying"):** Different models assign arbitrary variance scalings to the exact same semantic concepts. Ambient cosine distance is highly distorted by these arbitrary hierarchies.
* **Why Blind Whitening Fails:** Attempting to neutralize scalings via ZCA/PCA whitening in raw dense space ($D=768$) fails catastrophically because dividing by $\lambda^{-1/2}$ forces low-variance noise directions to have unit variance, exploding background noise by $>30\times$.

---

## 2. The Solution: Supervised Physical Subspaces
Instead of relying on unsupervised variance hierarchies (which intermingle physical signal with model-private architectural noise), we construct a **Task-Anchored Subspace** spanned by the normal vectors of supervised linear probes.

### How it solves the noise problem:
1. Train $M$ linear regression probes on ground-truth astronomical properties. Each probe yields a normal vector $w_m \in \mathbb{R}^D$ pointing in the direction of steepest physical variation.
2. The span of these vectors creates an $M$-dimensional physical subspace.
3. Projecting the 768D galaxy embeddings onto this $M$-dimensional subspace mathematically annihilates the $(768-M)$ dimensions of unaligned background noise, forcing distance calculations to evaluate nearest neighbors strictly on actionable, physically meaningful semantics.

### Why we need $M = 50\text{–}100$ Probes:
A small suite of 3–12 probes (e.g., redshift, mass, basic morphology) only projects galaxies onto coarse macroscopic predictions. In a 12D space, thousands of galaxies map to nearly identical coordinates, destroying the visual granularity required for $\text{mKNN}@10$.
To match the intrinsic dimensionality of the data's soft shell ($d \sim 87$) while remaining noise-free, we must expand the target suite to $50\text{–}100$ targets:
* **Multi-band photometry & color gradients** ($u, g, r, i, z, W1\dots$)
* **Structural fits** (Sérsic indices, half-light radii, ellipticities)
* **Full Galaxy Zoo vote fractions** (detailed arm winding, bar strength, merging flags)

---

## 3. Alignment Strategy: Task-Anchored Orthogonal Procrustes
Standard unconstrained regression ($X_A W \approx X_B$) aligns raw data points, risking overfitting to residual high-variance background noise. To construct a cleaner mapping, we map the spaces by **aligning the probe normal vectors themselves**.

### Method:
1. Extract normal vector matrices $W_A \in \mathbb{R}^{D_A \times M}$ and $W_B \in \mathbb{R}^{D_B \times M}$.
2. (Optional) Apply **Accuracy Weighting**: Scale each normal vector pair by the geometric mean of their cross-validated probe accuracy ($\sqrt{R_A^2 \times R_B^2}$) so that noisy, poorly learned probes are discounted.
3. Solve for the alignment mapping $T$ using **Orthogonal Procrustes Analysis**:
   * Compute the cross-covariance of the weighted normal vectors: $M = W_A W_B^\top$.
   * Take the SVD: $M = U \Sigma V^\top$.
   * The optimal rigid rotation is $R = U V^\top$.
4. **Why Orthogonal Procrustes?** A pure rotation ($R^\top R = I$) perfectly preserves $100\%$ of Model A's internal geometry (all pairwise distances and angles remain identical). It simply rotates Model A's coordinate axes until its physical semantic vectors perfectly align with Model B's physical semantic vectors.

### Why projecting down is critical:
If we apply the $768 \times 768$ Procrustes rotation $R$ directly to the raw 768D galaxy vectors without projecting down, the $(768-M)$ null-space dimensions are rotated arbitrarily but remain inside the vectors, continuing to inject noise into the cosine similarity. Projecting down explicitly acts as a filter to delete that unaligned noise.

---

## 4. Unifying with Curvature: A Joint Experiment
Our prior curvature research revealed that higher local projector variance (Tyagi bootstrap PCA curvature) correlates with worse local linear probe $R^2$. This means that in high-curvature regions, global linear normal vectors diverge from local tangent spaces.

**The Proposed Joint Experiment:**
1. Compute the local curvature proxy $\tilde{V}_d(p)$ for all test galaxies.
2. Stratify the dataset into curvature quartiles ($Q_1$: flat $\to$ $Q_4$: highly curved).
3. Evaluate whether our normal-vector Procrustes alignment map yields exceptionally high $\text{mKNN}$ agreement in the flat $Q_1$ regime, but systematically breaks down in $Q_4$.
4. **Scientific Impact:** This tests the fundamental hypothesis that cross-model representation misalignment is driven explicitly by local manifold curvature distorting semantic normal vectors.

---

## 5. Formal Nomenclature & Literature Strategy
When documenting or publishing this approach, we link it to established mathematical literature:
* *Probe Subspace:* **Sufficient Dimension Reduction**, **Supervised Principal Component Analysis**.
* *Normal Vector Mapping:* **Orthogonal Procrustes Problem**, **Concept Space Alignment**.
* *Curvature Drift:* **Riemannian Manifold Learning**, **Tangent Space Drift**.

To avoid pitfalls (such as collinearity of physical properties causing ill-conditioned projections), we use SVD/QR decomposition on the weight matrix $W$ to form an orthonormal basis before projection.
