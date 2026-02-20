# Agent Tasks for Future Work

This document outlines future development tasks for the `EffDim` library. These tasks are well-suited for autonomous AI agents to pick up and implement.

## 1. Implement Missing Geometric Estimators

Several advanced geometric estimators need to be implemented in `src/effdim/geometry.py` and integrated into the `compute_dim` API in `src/effdim/api.py`. 

### Target Estimators:
- **DANCo (Dimensionality from Angle and Norm Concentration):** Exploits statistics of norms of vectors to nearest neighbors and angles between them. 
- **MiND (Maximum Likelihood on Minimum Distances):**
    - **MiND-MLi:** Uses the distribution of distance to the nearest neighbor ($r_1$).
    - **MiND-MLk:** Uses the joint distribution of distances to the first $k$ neighbors.
- **ESS (Expected Simplex Skewness):** Estimates dimension by analyzing the volume/skewness of local simplices.
- **TLE (Tight Localities Estimator):** Maximizes likelihood on scale-normalized distances.
- **GMST (Geodesic Minimum Spanning Tree):** Estimates dimension from the scaling of the MST length with sample size (supporting both Euclidean and Geodesic modes).

**Requirements:**
- Add thorough unit tests for each new estimator in the `tests/` directory.
- Update `docs/theory.md` and `docs/api.md` to reflect the newly added methods.
- Update `docs/tutorials/advanced_geometric_analysis.md` (recreate if necessary) to showcase these new methods.

## 2. Enhance Testing and CI/CD
- Expand the current test suite to rigorously validate estimates against known dimensionalities (e.g., random noise should approximate $D$, manifolds like Swiss Roll should approximate their intrinsic dimension).
- Ensure CI pipeline executes tests for both the standard Python implementation and the compiled Rust extension across different target platforms.
