# Theory & Estimators

EffDim implements a variety of estimators for "effective dimensionality" (ED). These can be broadly categorized into **Spectral Estimators**, which operate on the eigenvalues (spectrum) of the data's covariance/correlation matrix, and **Geometric Estimators**, which operate on the distances between data points.

## Mathematical Foundation

### Eigenvalue Computation

For a data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ with $n$ samples and $p$ features:

1. **Centering**: The data is centered by subtracting the mean: $\tilde{\mathbf{X}} = \mathbf{X} - \frac{1}{n}\mathbf{1}\mathbf{1}^T\mathbf{X}$

2. **SVD Decomposition**: $\tilde{\mathbf{X}} = \mathbf{U}\mathbf{S}\mathbf{V}^T$ where $\mathbf{S}$ contains singular values $s_1 \ge s_2 \ge \dots \ge s_{\min(n,p)} \ge 0$

3. **Eigenvalue Calculation**: The eigenvalues of the sample covariance matrix are:
   $$ \lambda_i = \frac{s_i^2}{n-1} $$
   
   This follows from $\mathbf{C} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}} = \frac{1}{n-1}\mathbf{V}\mathbf{S}^2\mathbf{V}^T$

4. **Normalization**: The normalized spectrum (probability distribution) is:
   $$ p_i = \frac{\lambda_i}{\sum_{j=1}^{D} \lambda_j} $$
   
   where $D = \min(n-1, p)$ is the maximum number of non-zero eigenvalues.

**Note**: The implementation uses $(n-1)$ normalization (sample covariance, unbiased estimator) rather than $n$ (population covariance).

## Spectral Estimators

These methods rely on the spectrum $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$ of the covariance matrix. We define the normalized spectrum as $p_i = \frac{\lambda_i}{\sum_j \lambda_j}$, which can be treated as a probability distribution.

### PCA Explained Variance

The classic approach used in Principal Component Analysis. It defines the effective dimension as the number of components required to explain a certain fraction (threshold) of the total variance.

**Formula:**
$$ ED_{PCA}(\tau) = \min \left\{ k \mid \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^D \lambda_j} \ge \tau \right\} $$

where $\tau$ is the threshold (default 0.95).

**Implementation Details:**
- Uses `np.searchsorted` for efficient binary search
- Returns integer number of components
- Common thresholds: 0.90, 0.95, 0.99

**Interpretation:**
- Direct measure of intrinsic dimensionality
- Easy to interpret: "95% of variance explained by $k$ dimensions"
- Depends on threshold choice

### Participation Ratio (PR)

Widely used in physics and neuroscience to quantify the "spread" of the spectrum. If the variance is equally distributed across $N$ dimensions, $PR=N$. If it is concentrated in 1 dimension, $PR=1$.

**Formula:**
$$ PR = \frac{\left(\sum_{i=1}^D \lambda_i\right)^2}{\sum_{i=1}^D \lambda_i^2} = \frac{1}{\sum_{i=1}^D p_i^2} $$

**Mathematical Equivalence:**
The Participation Ratio is equivalent to the Rényi Effective Dimension with $\alpha = 2$:
$$ PR = \exp\left(\frac{1}{1-2} \ln \sum_i p_i^2\right) = \left(\sum_i p_i^2\right)^{-1} = \frac{1}{\sum_i p_i^2} $$

**Implementation Details:**
- Works with eigenvalues $\lambda_i$ directly (scale-invariant)
- Protects against division by zero
- Continuous (non-integer) output

**Interpretation:**
- $PR = D$ when all eigenvalues are equal (maximum spread)
- $PR = 1$ when one eigenvalue dominates (minimum spread)
- Robust to small eigenvalues (inverse quadratic weighting)

### Shannon Effective Dimension

Based on the Shannon entropy of the spectral distribution. It corresponds to the exponential of the entropy.

**Formula:**
$$ H = - \sum_{i=1}^D p_i \ln p_i $$
$$ ED_{Shannon} = \exp(H) $$

**Implementation Details:**
- Filters zero probabilities before computing $\ln p_i$ to avoid $\ln(0)$
- Returns exponential of entropy (not entropy itself)
- Also known as "Effective Rank" or "Perplexity"

**Interpretation:**
- $ED_{Shannon} = D$ when all eigenvalues are equal
- $ED_{Shannon} = 1$ when one eigenvalue dominates
- Represents the "number of equally likely states"
- More sensitive to tail of distribution than PR
- Related to information content of the spectrum

**Connection to Information Theory:**
The Shannon ED can be interpreted as the effective number of "independent" dimensions, where each dimension contributes equally to the total variance in an information-theoretic sense.

### Rényi Effective Dimension (Alpha-Entropy)

A generalization of the Shannon dimension using Rényi entropy of order $\alpha$.

**Formula:**
$$ H_\alpha = \frac{1}{1-\alpha} \ln \left(\sum_{i=1}^D p_i^\alpha\right) $$
$$ ED_{\alpha} = \exp(H_\alpha) = \left(\sum_{i=1}^D p_i^\alpha\right)^{\frac{1}{1-\alpha}} $$

**Special Cases:**
*   $\alpha \to 1$: Converges to Shannon Effective Dimension (via L'Hôpital's rule)
*   $\alpha = 2$: $ED_2 = \left(\sum_i p_i^2\right)^{-1} = PR$ (Participation Ratio)
*   $\alpha = 0$: Counts non-zero eigenvalues (algebraic rank)
*   $\alpha \to \infty$: $ED_\infty = 1/\max_i p_i$ (inverse of largest probability)

**Implementation Details:**
- Validates $\alpha > 0$ and $\alpha \neq 1$
- For $\alpha = 1$, use Shannon ED instead
- Protects against zero denominator

**Interpretation:**
- $\alpha$ controls sensitivity to rare vs common components
- Small $\alpha$ (e.g., 0.5): More weight to rare eigenvalues
- Large $\alpha$ (e.g., 5): More weight to dominant eigenvalues
- Provides a one-parameter family interpolating between different notions of dimension

**When to Use Which Alpha:**
- $\alpha = 2$ (PR): Balanced measure, equivalent to inverse Simpson index
- $\alpha = 3, 4, 5$: Emphasize dominant components
- For general analysis, $\alpha = 2$ (PR) is most common

### Effective Rank

Often used in matrix completion and low-rank approximation contexts. EffDim implements this as an alias for Shannon Effective Dimension.

### Geometric Mean Dimension

A dimension proxy based on the ratio of the arithmetic mean to the geometric mean of the spectrum.

**Formula:**
$$ d_{GM} = \frac{\text{AM}(\lambda)}{\text{GM}(\lambda)} = \frac{\frac{1}{D'} \sum_{i=1}^{D'} \lambda_i}{\left(\prod_{i=1}^{D'} \lambda_i\right)^{1/D'}} $$

where $D'$ is the number of positive eigenvalues, $\lambda_i > 0$.

**Equivalent Form:**
$$ d_{GM} = \frac{\frac{1}{D'} \sum_{i=1}^{D'} \lambda_i}{\exp\left(\frac{1}{D'} \sum_{i=1}^{D'} \ln \lambda_i\right)} $$

**Implementation Details:**
- Filters zero/negative eigenvalues before computation
- Uses $\ln$ for numerical stability in geometric mean
- Scale-invariant (gives same result for eigenvalues or probabilities)
- Returns 0.0 if no positive eigenvalues

**Interpretation:**
- $d_{GM} = 1$ when all positive eigenvalues are equal
- $d_{GM} > 1$ indicates spread in the spectrum
- By AM-GM inequality: $d_{GM} \ge 1$ always
- Larger values indicate more inequality in eigenvalue distribution
- Related to condition number of the covariance matrix

**Note on Scale Invariance:**
Because $\text{AM}(c\lambda) = c \cdot \text{AM}(\lambda)$ and $\text{GM}(c\lambda) = c \cdot \text{GM}(\lambda)$, the ratio is unchanged by scaling. Therefore, using eigenvalues $\lambda_i$ or normalized probabilities $p_i$ yields the same result.

### Stable Rank

A stable alternative to the algebraic rank, often used in high-dimensional probability. It is robust to small perturbations of the singular values.

$$ R_{stable} = \frac{\sum_i \lambda_i}{\max_i \lambda_i} $$

where $\lambda_i$ are the eigenvalues (variances).

### Numerical Rank (Epsilon-Rank)

The number of singular values greater than a specific threshold $\epsilon$.

$$ rank_\epsilon(A) = | \{ \sigma_i \mid \sigma_i > \epsilon \} | $$

If $\epsilon$ is not provided, it defaults to a value based on the machine precision and the largest singular value.

### Cumulative Eigenvalue Ratio (CER)

A weighted sum of the normalized spectrum, giving more weight to earlier components.

$$ CER = \sum_{i=1}^D w_i p_i $$

where weights decrease linearly from 1 to 0.

## Geometric Estimators

These methods estimate the intrinsic dimension (ID) of the data manifold based on local neighborhoods, without relying on global projections like PCA.

### kNN Intrinsic Dimension (MLE)

The Maximum Likelihood Estimator proposed by Levina and Bickel (2005). It estimates dimension by examining the ratio of distances to the $k$-th nearest neighbor.

**Reference:** Levina, E., & Bickel, P. J. (2005). Maximum likelihood estimation of intrinsic dimension. *Advances in Neural Information Processing Systems* 17 (NIPS 2004).

**Formula:**
For each point $x_i$, let $r_1(x_i) \le r_2(x_i) \le \dots \le r_k(x_i)$ be the distances to its $k$ nearest neighbors. The local dimension estimate is:

$$ \hat{d}_k(x_i) = \left[ \frac{1}{k-1} \sum_{j=1}^{k-1} \ln \frac{r_k(x_i)}{r_j(x_i)} \right]^{-1} = \frac{k-1}{\sum_{j=1}^{k-1} \ln r_k(x_i) - \ln r_j(x_i)} $$

The global estimate is the average over all points:
$$ \hat{d}_k = \frac{1}{n} \sum_{i=1}^n \hat{d}_k(x_i) $$

**Implementation Details:**
- Uses FAISS library for efficient kNN search with L2 distance
- Adds $\epsilon = 10^{-10}$ to distances to prevent $\ln(0)$
- Uses `np.errstate` to handle potential numerical issues
- Default $k = 10$ neighbors
- Complexity: $O(n^2)$ naive, $O(n \log n)$ with FAISS indexing

**Numerical Stability:**
- $\ln(r_k/r_j) = \ln r_k - \ln r_j$ computed in log-space for stability
- Epsilon added to denominator to prevent division by zero
- Handles duplicate points (distance = 0) gracefully

**When to Use:**
- Data lies on or near a low-dimensional manifold
- Sufficient sample density (at least $k$ neighbors per point)
- Metric (Euclidean) distance is meaningful

**Choosing k:**
- Too small $k$: High variance, sensitive to noise
- Too large $k$: Bias if curvature is significant
- Typical range: $k \in [5, 20]$
- Rule of thumb: $k \approx \sqrt{n}$ for small datasets

### Two-NN

A robust estimator proposed by Facco et al. (2017) that relies only on the distances to the first two nearest neighbors. It is less sensitive to density variations and curvature than standard kNN.

**Reference:** Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*, 7(1), 12140. DOI: 10.1038/s41598-017-11873-y

**Theory:**
Assumes data is uniformly distributed on a $d$-dimensional manifold. The ratio $\mu_i = r_2(x_i) / r_1(x_i)$ follows a distribution with CDF:
$$ F(\mu) = 1 - \mu^{-d} $$

Taking logarithms:
$$ -\ln(1 - F(\mu)) = d \cdot \ln(\mu) $$

**Formula:**
1. For each point $x_i$, compute $\mu_i = r_2(x_i) / r_1(x_i)$
2. Sort the $\mu_i$ values: $\mu_{(1)} \le \mu_{(2)} \le \dots \le \mu_{(n)}$
3. Use empirical CDF: $F(\mu_{(i)}) = i/n$ for $i = 1, \dots, n-1$ (drop last to avoid $\ln(0)$)
4. Fit linear regression through origin:
   $$ y_i = d \cdot x_i $$
   where $x_i = \ln(\mu_{(i)})$ and $y_i = -\ln(1 - i/n)$
5. Estimate: $\hat{d} = \frac{\sum_i x_i y_i}{\sum_i x_i^2}$

**Implementation Details:**
- Uses FAISS for kNN search (k=2)
- Adds $\epsilon = 10^{-10}$ to distances
- Drops last point to avoid $F=1$ causing $\ln(0)$
- Linear regression uses dot product formula (no intercept)
- Complexity: $O(n \log n)$ with FAISS

**Numerical Stability:**
- Epsilon prevents division by zero in $\mu = r_2/r_1$
- Dropping last point prevents $\ln(1-1) = \ln(0)$
- Checks for zero variance in $\mu$ values

**Advantages over MLE:**
- More robust to density variations
- Less sensitive to manifold curvature
- Requires only 2 neighbors (works with smaller samples)
- No parameter tuning required

**When to Use:**
- Preferred for datasets with non-uniform density
- When sample size is limited
- When you want parameter-free estimation

### DANCo (Dimensionality from Angle and Norm Concentration)

Exploits the concentration of angles between nearest neighbor vectors to estimate intrinsic dimension.

**Reference:** Ceruti, C., et al. (2012). DANCo: Dimensionality from Angle and Norm Concentration. *arXiv:1206.3881*.

**Theory:**
In a $d$-dimensional space, the angles between random vectors concentrate around $\pi/2$ as $d$ increases. Specifically, for two random unit vectors in $\mathbb{R}^d$, the expected value of $\cos^2(\theta) \approx 1/d$.

**Formula:**
1. For each point $x_i$, compute vectors $v_{ij} = x_j - x_i$ to its $k$ nearest neighbors.
2. Normalize to unit vectors: $\hat{v}_{ij} = v_{ij} / \|v_{ij}\|$.
3. Compute pairwise cosines: $\cos(\theta_{jl}) = \hat{v}_{ij} \cdot \hat{v}_{il}$.
4. Estimate: $\hat{d} = 1 / \overline{\cos^2(\theta)}$ where the average is over all points and all pairs.

**Implementation Details:**
- Uses FAISS for efficient kNN search
- Adds $\epsilon = 10^{-10}$ to norms to prevent division by zero
- Default $k = 10$ neighbors
- Uses `np.einsum` for efficient pairwise cosine computation

**When to Use:**
- When angle-based estimation is preferred over distance-based
- For data with complex geometric structure

### MiND-MLi (Minimum Distance — Single Neighbor)

Maximum Likelihood Estimator based on the distribution of nearest neighbor distances only.

**Reference:** Rozza, A., et al. (2012). Novel high intrinsic dimensionality estimators. *Machine Learning*, 89(1-2), 37-65.

**Formula:**
1. For each point $x_i$, compute the nearest neighbor distance $r_1(x_i)$.
2. Let $r_{\max} = \max_i r_1(x_i)$.
3. Estimate: $\hat{d} = n / \sum_{i=1}^n \ln(r_{\max} / r_1(x_i))$.

**Implementation Details:**
- Uses only the single nearest neighbor distance
- Returns 0.0 if all distances are equal (degenerate case)
- Requires at least 3 points

**When to Use:**
- When minimal neighborhood information is available
- For quick estimates with low computational cost

### MiND-MLk (Minimum Distance — k Neighbors)

Extension of the Levina-Bickel MLE using the median of per-point estimates for robustness against outliers.

**Reference:** Rozza, A., et al. (2012). Novel high intrinsic dimensionality estimators. *Machine Learning*, 89(1-2), 37-65.

**Formula:**
Same per-point estimates as kNN MLE:
$$ \hat{d}_k(x_i) = \frac{k-1}{\sum_{j=1}^{k-1} \ln r_k(x_i) - \ln r_j(x_i)} $$

Global estimate uses the **median** instead of the mean:
$$ \hat{d}_k = \text{median}\left(\hat{d}_k(x_1), \dots, \hat{d}_k(x_n)\right) $$

**Advantages over standard MLE:**
- More robust to outliers and density inhomogeneities
- The median is less sensitive to extreme per-point estimates

### ESS (Expected Simplex Skewness)

Estimates dimension by analyzing the skewness of local simplices formed by nearest neighbors.

**Reference:** Johnsson, K., et al. (2015). Low bias local intrinsic dimension estimation from expected simplex skewness. *IEEE Trans. PAMI*, 37(1), 196-202.

**Theory:**
For each point, the $k$ nearest neighbors form a simplex. The "skewness" is measured as the squared norm of the mean of unit direction vectors. In $d$ dimensions, the expected squared norm of the mean of $k$ random unit vectors is approximately $1/(k \cdot d)$.

**Formula:**
1. For each point $x_i$, compute unit vectors $\hat{v}_{ij}$ to its $k$ neighbors.
2. Compute centroid: $\bar{v}_i = \frac{1}{k}\sum_{j=1}^k \hat{v}_{ij}$.
3. Compute skewness: $S_i = \|\bar{v}_i\|^2$.
4. Average: $\bar{S} = \frac{1}{n}\sum_{i=1}^n S_i$.
5. Estimate: $\hat{d} = 1 / (k \cdot \bar{S})$.

**When to Use:**
- When low-bias estimation is important
- For manifolds with moderate curvature

### TLE (Tight Localities Estimator)

Maximizes likelihood on scale-normalized distances, making it more robust to density variations.

**Reference:** Amsaleg, L., et al. (2019). Intrinsic dimensionality estimation within tight localities. *SDM 2019*.

**Theory:**
For each point, the distances to $k$ neighbors are normalized by the $k$-th neighbor distance: $u_j = r_j / r_k$. Under a $d$-dimensional uniform distribution, each $u_j$ follows a $\text{Beta}(d, 1)$ distribution with PDF $p(u) = d \cdot u^{d-1}$.

**Formula:**
$$ \hat{d}_i = \frac{-(k-1)}{\sum_{j=1}^{k-1} \ln u_j} = \frac{k-1}{\sum_{j=1}^{k-1} \ln(r_k / r_j)} $$

The global estimate is $\hat{d} = \frac{1}{n}\sum_{i=1}^n \hat{d}_i$.

**Implementation Details:**
- Mathematically equivalent to the Levina-Bickel MLE
- The per-point normalization by $r_k$ provides scale invariance

### GMST (Geodesic Minimum Spanning Tree)

Estimates dimension from how the total length of the Minimum Spanning Tree (MST) scales with the number of points.

**Reference:** Costa, J. A., & Hero, A. O. (2004). Geodesic entropic graphs for dimension and entropy estimation in manifold learning. *IEEE Trans. Signal Processing*, 52(8), 2210-2221.

**Theory:**
For $n$ points sampled uniformly from a $d$-dimensional manifold ($d > 1$), the total MST edge weight scales as:
$$ L_{\text{MST}} \propto n^{(d-1)/d} $$

Taking logarithms: $\ln L_{\text{MST}} = \alpha \cdot \ln n + c$, where $\alpha = (d-1)/d$, giving $d = 1/(1-\alpha)$.

**Formula:**
1. Take subsamples of sizes $n_1, n_2, \dots$ from the data.
2. For each subsample, compute the MST and its total edge weight $L_i$.
3. Fit linear regression: $\ln L_i = \alpha \cdot \ln n_i + c$.
4. Estimate: $\hat{d} = 1 / (1 - \alpha)$.

**Geodesic Mode:**
When `geodesic=True`, distances are computed along the data manifold using shortest paths on a $k$-NN graph, rather than straight-line Euclidean distances.

**Implementation Details:**
- Uses `scipy.sparse.csgraph.minimum_spanning_tree` for MST computation
- Uses `scipy.spatial.distance.pdist` for Euclidean distances
- Uses `sklearn.neighbors.kneighbors_graph` + `scipy.sparse.csgraph.shortest_path` for geodesic distances
- Subsamples at sizes $[n/8, n/4, n/2, n]$ with a fixed random seed for reproducibility
- Requires at least 10 points

**When to Use:**
- When data lies on a curved manifold (use geodesic mode)
- When graph-based analysis is preferred
- For datasets where local methods may be biased

---

## Assumptions and Limitations

### General Assumptions

**Data Requirements:**
- Data should be centered (automatically handled in implementation)
- Sufficient sample size: $n \gg d$ for reliable spectral estimates
- Sufficient sample size: $n \ge 2k$ for geometric estimates
- No missing values (NaN/Inf not handled)

**Spectral Estimators:**
- Assume linear (global) structure
- Best for: Gaussian or approximately Gaussian data
- May overestimate dimension for highly curved manifolds
- Sensitive to outliers (affect eigenvalue spectrum)
- Assume data lies in Euclidean space (not on a non-linear manifold)

**Geometric Estimators:**
- Assume local manifold structure
- Best for: Data on or near low-dimensional manifolds
- Require sufficient local sample density
- Assume metric (Euclidean) distance is meaningful
- May fail for highly sparse or non-uniformly sampled data

### Known Limitations

**Computational Complexity:**
- SVD: $O(\min(n^2p, np^2))$ for full SVD, $O(npk)$ for randomized SVD
- MLE: $O(n^2)$ for exact kNN, $O(n \log n)$ with FAISS indexing
- Two-NN: $O(n \log n)$ with FAISS

**Sample Size Dependencies:**
- Spectral methods: Eigenvalues stabilize when $n \ge 5d$ (rule of thumb)
- MLE: Requires at least $k+1$ points, converges when $n \ge 100$
- Two-NN: Requires at least 3 points, more robust for small $n$ than MLE
- Small $n$: High variance in all estimates

**High-Dimensional Data ($p \gg n$):**
- Only $\min(n-1, p)$ eigenvalues are non-zero
- Covariance matrix is rank-deficient
- Spectral methods still work but are limited by $n$
- Consider using randomized SVD for efficiency

**Numerical Stability:**
- Very small eigenvalues ($\lambda < 10^{-10}$) may be unstable
- Very large condition numbers may cause issues
- Log of very small values protected by epsilon addition
- Division by zero protected throughout

### When to Use Which Estimator

**Use PCA Explained Variance when:**
- You need an interpretable, threshold-based dimension
- Data is approximately Gaussian
- You want to choose how much information to retain

**Use Participation Ratio when:**
- You want a continuous measure
- You need mathematical equivalence to Rényi-2
- Comparing across different datasets

**Use Shannon ED when:**
- You want an information-theoretic measure
- You need sensitivity to the full spectrum (including tail)
- Comparing to entropy-based methods

**Use MLE when:**
- Data lies on a non-linear manifold
- You have sufficient sample density
- You can tune the $k$ parameter

**Use Two-NN when:**
- You want a robust, parameter-free geometric method
- Sample size is limited
- Data has non-uniform density

### References

**Spectral Methods:**
- Roy, O., & Vetterli, M. (2007). The effective rank: A measure of effective dimensionality. *EUSIPCO*.

**Geometric Methods:**
- Levina, E., & Bickel, P. J. (2005). Maximum likelihood estimation of intrinsic dimension. *NIPS 2004*.
- Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*, 7(1), 12140. DOI: 10.1038/s41598-017-11873-y
- Ceruti, C., et al. (2012). DANCo: Dimensionality from Angle and Norm Concentration. *arXiv:1206.3881*.
- Rozza, A., et al. (2012). Novel high intrinsic dimensionality estimators. *Machine Learning*, 89(1-2), 37-65.
- Johnsson, K., et al. (2015). Low bias local intrinsic dimension estimation from expected simplex skewness. *IEEE Trans. PAMI*, 37(1), 196-202.
- Amsaleg, L., et al. (2019). Intrinsic dimensionality estimation within tight localities. *SDM 2019*.
- Costa, J. A., & Hero, A. O. (2004). Geodesic entropic graphs for dimension and entropy estimation in manifold learning. *IEEE Trans. Signal Processing*, 52(8), 2210-2221.

**Survey:**
- Camastra, F., & Staiano, A. (2016). Intrinsic dimension estimation: Advances and open problems. *Information Sciences*, 328, 26-41.
