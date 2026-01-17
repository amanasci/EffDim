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

*   For $\alpha \to 1$, this converges to Shannon Effective Dimension.
*   For $\alpha = 2$, this is equivalent to the Participation Ratio.

### Effective Rank

Often used in matrix completion and low-rank approximation contexts. For a covariance matrix, it is often defined identically to the Shannon Effective Dimension or related trace-norm metrics. EffDim implements it as an alias for Shannon Effective Dimension currently.

### Geometric Mean Dimension

Based on the ratio of the arithmetic mean to the geometric mean of the spectrum.

## Geometric Estimators

These methods estimate the intrinsic dimension (ID) of the data manifold based on local neighborhoods, without relying on global projections like PCA.

### kNN Intrinsic Dimension (MLE)

The Maximum Likelihood Estimator proposed by Levina and Bickel (2005). It estimates dimension by examining the ratio of distances to the $k$-th nearest neighbor.

$$ \hat{d}_k(x_i) = \left[ \frac{1}{k-1} \sum_{j=1}^{k-1} \ln \frac{r_k(x_i)}{r_j(x_i)} \right]^{-1} $$

where $r_j(x_i)$ is the distance from $x_i$ to its $j$-th nearest neighbor. The final estimate is the average over all points $x_i$.

### Two-NN

A robust estimator proposed by Facco et al. (2017) that relies only on the distances to the first two nearest neighbors. It is less sensitive to density variations and curvature than standard kNN.

It assumes that the ratio of distances $\mu_i = \frac{r_2(x_i)}{r_1(x_i)}$ follows a Pareto distribution depending on the intrinsic dimension $d$.

$$ d \approx \frac{\ln(1-F(\mu))}{-\ln(\mu)} $$
