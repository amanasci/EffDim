# Theory & Estimators

EffDim implements a variety of estimators for "effective dimensionality" (ED). These can be broadly categorized into **Spectral Estimators**, which operate on the eigenvalues (spectrum) of the data's covariance/correlation matrix, and **Geometric Estimators**, which operate on the distances between data points.

## Spectral Estimators

These methods rely on the spectrum $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$ of the covariance matrix (or the squared singular values of the data matrix). We define the normalized spectrum as $p_i = \frac{\lambda_i}{\sum_j \lambda_j}$, which can be treated as a probability distribution.

### PCA Explained Variance

The classic approach used in Principal Component Analysis. It defines the effective dimension as the number of components required to explain a certain fraction (threshold) of the total variance.

$$ ED_{PCA}(x) = \min \{ k \mid \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^D \lambda_j} \ge x \} $$

where $x$ is the threshold (default 0.95).

### Participation Ratio (PR)

Widely used in physics and neuroscience to quantify the "spread" of the spectrum. If the variance is equally distributed across $N$ dimensions, $PR=N$. If it is concentrated in 1 dimension, $PR=1$.

$$ PR = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2} = \frac{1}{\sum_i p_i^2} $$

### Shannon Effective Dimension

Based on the Shannon entropy of the spectral distribution. It corresponds to the exponential of the entropy.

$$ H = - \sum_i p_i \ln p_i $$
$$ ED_{Shannon} = \exp(H) $$

### Rényi Effective Dimension (Alpha-Entropy)

A generalization of the Shannon dimension using Rényi entropy of order $\alpha$.

$$ H_\alpha = \frac{1}{1-\alpha} \ln (\sum_i p_i^\alpha) $$
$$ ED_{\alpha} = \exp(H_\alpha) $$

* For $\alpha \to 1$, this converges to Shannon Effective Dimension.
* For $\alpha = 2$, this is equivalent to the Participation Ratio.

### Effective Rank

Often used in matrix completion and low-rank approximation contexts. EffDim implements this as an alias for Shannon Effective Dimension.

### Geometric Mean Dimension

A dimension proxy based on the ratio of the arithmetic mean to the geometric mean of the spectrum.

$$ d \approx \frac{\frac{1}{D} \sum \lambda_i}{(\prod \lambda_i)^{1/D}} $$

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

$$ \hat{d}_k(x_i) = \left[ \frac{1}{k-1} \sum_{j=1}^{k-1} \ln \frac{r_k(x_i)}{r_j(x_i)} \right]^{-1} $$

where $r_j(x_i)$ is the distance from $x_i$ to its $j$-th nearest neighbor. The final estimate is the average over all points $x_i$.

### Two-NN

A robust estimator proposed by Facco et al. (2017) that relies only on the distances to the first two nearest neighbors. It is less sensitive to density variations and curvature than standard kNN.

It assumes that the ratio of distances $\mu_i = \frac{r_2(x_i)}{r_1(x_i)}$ follows a Pareto distribution depending on the intrinsic dimension $d$.

### DANCo

**Dimensionality from Angle and Norm Concentration**. This method jointly exploits the statistics of the norms of vectors to nearest neighbors and the angles between them. High-dimensional data exhibits specific concentration of measure properties for both angles and norms. DANCo estimates $d$ by minimizing the KL-divergence between the empirical distributions and the theoretical distributions derived for a d-dimensional ball.

### MiND (Maximum Likelihood on Minimum Distances)

A family of estimators based on the statistics of nearest neighbor distances.

* **MiND-MLi**: Uses the distribution of the distance to the nearest neighbor ($r_1$).
* **MiND-MLk**: Uses the joint distribution of distances to the first $k$ neighbors.

### ESS (Expected Simplex Skewness)

Estimates dimension by analyzing the "skewness" (volume) of the simplex formed by a point and its neighbors. In high dimensions, random simplices tend to be regular (perfectly "skewed"). The estimator compares the empirical volumes of local simplices to theoretical expected volumes.

### TLE (Tight Localities Estimator)

Estimates dimension by maximizing the likelihood of distances within small, "tight" neighborhoods. It is designed to be robust to scale variations.

### GMST (Geodesic Minimum Spanning Tree)

Estimates dimension based on the scaling law of the length of the Minimum Spanning Tree (MST) of a graph constructed from the data.
$$ L(N) \propto N^{1 - 1/d} $$
where $L(N)$ is the length of the MST on $N$ points. The dimension $d$ is estimated from the slope of $\log L(N)$ vs $\log N$ using subsampling. The graph can be constructed using Euclidean distances or Geodesic distances (approximated by k-NN graph paths).
