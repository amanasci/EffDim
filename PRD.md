# EffDim: Product Requirement Document

## 1. Executive Summary
**EffDim** is a unified, research-oriented Python library designed to compute "effective dimensionality" (ED) across diverse data modalities. It aims to standardize the fragmented landscape of ED metrics found in statistics, physics, information theory, and machine learning into a single, cohesive interface.

## 2. Problem Statement
The concept of "effective dimension" is ubiquitous but fragmented. 
- **Inconsistency**: Different fields (e.g., neuroscience vs. cosmology) use different terms for the same mathematical concepts (e.g., Participation Ratio vs. Inverse Simpson Concentration).
- **Implementation Overhead**: Researchers re-implement the same estimators (PCA thresholding, participation ratio) repeatedly, often with subtle bugs or lack of optimization.
- **Modality Silos**: Tools are often specific to one data type (e.g., only for graphs, or only for time series), making cross-modal analysis difficult.

## 3. Goals & Philosophy
- **Unified Interface**: A single entry point for all ED estimators.
- **First-Class Documentation**: Every estimator must include mathematical definitions, references, and intuition.
- **Modality Agnostic**: The library handles the abstraction of "spectral representation" so users can pass raw data, covariances, or graphs seamlessly.
- **Minimal Dependencies**: Core functionality depends only on `numpy` and `scipy`.

## 4. User Stories
- **As a Machine Learning Researcher**, I want to track the effective dimension of hidden layer representations during training to understand collapse.
- **As a Neuroscientist**, I want to compare the dimensionality of neural activity (time series) with the structural connectivity (graph) of the same region.
- **As a Data Scientist**, I want to quickly estimate the intrinsic dimension of a dataset to set hyperparameters for dimensionality reduction (e.g., UMAP n_components).

## 5. Functional Requirements (v0.1)

### 5.1 Supported Input Data
The library must support converting various inputs into a spectral representation (eigenvalues/singular values):
| Input Type | Conversion Method | Note |
|---|---|---|
| (N, D) Array | Correlations/Covariance | Classic PCA approach |
| (N, D) Array | Singular Value Decomposition | Direct usage of singular values |
| (N, N) Matrix | Direct Eigendecomposition | For pre-computed kernels/Covariances |
| (N, N) Graph | Laplacian Spectrum | For network analysis |

### 5.2 Core Estimators
The library will implement the following estimators. All estimators return a `float`.

#### Spectral Estimators (Based on eigenvalues $\lambda_i$)
1.  **PCA Explained Variance**:
    - *Def*: Number of components needed to explain $\ge x\%$ variance.
    - *Params*: `threshold` (default 0.95).
2.  **Participation Ratio (PR)**:
    - *Def*: $PR = (\sum \lambda_i)^2 / \sum \lambda_i^2$.
    - *Context*: widely used in physics and neuroscience.
3.  **Shannon Effective Dimension**:
    - *Def*: $\exp(H)$ where $H = -\sum p_i \ln p_i$ and $p_i = \lambda_i / \sum \lambda_k$.
4.  **Rényi Effective Dimension (Alpha-Entropy)**:
    - *Def*: Generalization of Shannon for $\alpha \neq 1$.
5.  **Effective Rank (Trace-Norm)**:
    - *Def*: Metric for matrix approximations (Roy & Vetterli).
6.  **Geometric Mean Dimension**:
    - *Def*: Based on arithmetic vs geometric mean of singular values.

#### Geometric Estimators (Geometry-based)
1.  **kNN Intrinsic Dimension (MLE)**:
    - *Def*: Levina-Bickel Maximum Likelihood Estimator based on nearest neighbor distances.
    - *Note*: Does not use eigenvalues; operates on distance matrices.
2.  **Two-NN**:
    - *Def*: robust estimator using only 2 nearest neighbors.

### 5.3 API Specification (Draft)

#### Functional API
Simple, stateless usage for quick analysis.
```python
import effdim

# Compute using a specific method
ed = effdim.compute(data, method='participation_ratio')

# Compute using a specific method with parameters
ed = effdim.compute(data, method='pca', threshold=0.99)

# Compute multiple metrics at once
report = effdim.analyze(data, methods=['pca', 'shannon', 'rank'])
```

#### Object-Oriented API
For advanced usage, caching, or pipeline integration.
```python
from effdim import ParticipationRatio, PCAEstimator

est = ParticipationRatio()
ed = est.fit_transform(data)
```

## 6. Non-Functional Requirements
- **Performance**: 
    - Eigendecomposition is $O(N^3)$. For v0.1, exact computation is acceptable for $N < 10,000$.
    - Input validation must be strict (check for NaNs, Infs).
- **Usability**:
    - Descriptive error messages when inputs are ill-conditioned.
    - All mathematical functions must link to their definition in the docstring.

## 7. Roadmap

### Phase 1: Foundation (v0.1)
- [ ] Skeleton project structure.
- [ ] Implementation of Input Adapters (Covariance, SVD).
- [ ] Core spectral estimators (PR, Shannon, PCA).
- [ ] Basic tests and CI.

### Phase 2: Geometry & Robustness (v0.2)
- [ ] kNN-based estimators.
- [ ] Handling of very large matrices (Randomized SVD support).
- [ ] "Streaming" dimension estimation.

### Phase 3: Advanced Features (v0.3)
- [ ] Fractal dimension (Box-counting) - *tentative*.
- [ ] Local Intrinsic Dimension (LID) maps.
