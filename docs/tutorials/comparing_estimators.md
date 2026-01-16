# Comparing Estimators

Different fields use different definitions of "effective dimension". This tutorial highlights the differences.

## PCA vs Participation Ratio

*   **PCA** relies on a hard threshold (e.g., 95% variance). It answers "how many axes do I need to keep?".
*   **Participation Ratio (PR)** is a "soft" count. It answers "how spread out is the variance?".

Consider a spectrum where eigenvalues decay slowly: $\lambda_i = 1/i$.

```python
import numpy as np
import effdim
import matplotlib.pyplot as plt

# Simulate a slow decay spectrum directly
# (We pass a diagonal matrix to simulate uncorrelated data with specific variances)
D = 50
lambdas = 1.0 / np.arange(1, D+1)
# Create a covariance matrix
cov = np.diag(lambdas)

# effdim accepts covariance matrices directly if we assume these are eigenvalues
# But currently `compute` expects data (N,D) and calls `adapters.get_singular_values`.
# If we pass (D, D), it might treat it as N=D samples.
# Ideally, we construct data that has this spectrum.
# X = U * S * V.T.
# Let's create data X (N=1000, D=50) with singular values s_i = sqrt(lambda_i * (N-1))

N = 1000
s = np.sqrt(lambdas * (N - 1))
# Random orthogonal matrix U (N x D)
U, _ = np.linalg.qr(np.random.randn(N, D))
# Identity V (since we don't care about rotation)
X = U @ np.diag(s)

pca_95 = effdim.compute(X, method='pca', threshold=0.95)
pr = effdim.compute(X, method='pr')

print(f"PCA (95%): {pca_95}")
print(f"Participation Ratio: {pr:.2f}")
```

In heavy-tailed distributions, PCA might suggest a very high dimension (to capture the tail), whereas PR might suggest a lower dimension because the mass is concentrated at the start.

## Shannon vs Rényi

Shannon Entropy weights probabilities logarithmically. Rényi entropy (with $\alpha=2$, which relates to PR) weights higher probabilities more heavily.

*   **Shannon** is sensitive to the entire distribution.
*   **PR (Rényi-2)** is more dominated by the largest eigenvalues.

If you have a dataset with many small noise directions, Shannon dimension might be higher than PR.
