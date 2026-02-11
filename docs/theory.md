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

### DANCo

**Dimensionality from Angle and Norm Concentration**. This method jointly exploits the statistics of the norms of vectors to nearest neighbors and the angles between them. High-dimensional data exhibits specific concentration of measure properties for both angles and norms. DANCo estimates $d$ by minimizing the KL-divergence between the empirical distributions and the theoretical distributions derived for a d-dimensional ball.

### MiND (Maximum Likelihood on Minimum Distances)

A family of estimators based on the statistics of nearest neighbor distances.
*   **MiND-MLi**: Uses the distribution of the distance to the nearest neighbor ($r_1$).
*   **MiND-MLk**: Uses the joint distribution of distances to the first $k$ neighbors.

### ESS (Expected Simplex Skewness)

Estimates dimension by analyzing the "skewness" (volume) of the simplex formed by a point and its neighbors. In high dimensions, random simplices tend to be regular (perfectly "skewed"). The estimator compares the empirical volumes of local simplices to theoretical expected volumes.

### TLE (Tight Localities Estimator)

Estimates dimension by maximizing the likelihood of distances within small, "tight" neighborhoods. It is designed to be robust to scale variations.

### GMST (Geodesic Minimum Spanning Tree)

Estimates dimension based on the scaling law of the length of the Minimum Spanning Tree (MST) of a graph constructed from the data.
$$ L(N) \propto N^{1 - 1/d} $$
where $L(N)$ is the length of the MST on $N$ points. The dimension $d$ is estimated from the slope of $\log L(N)$ vs $\log N$ using subsampling. The graph can be constructed using Euclidean distances or Geodesic distances (approximated by k-NN graph paths).

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

**Survey:**
- Camastra, F., & Staiano, A. (2016). Intrinsic dimension estimation: Advances and open problems. *Information Sciences*, 328, 26-41.
