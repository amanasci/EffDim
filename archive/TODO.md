# Agent Tasks for Future Work

This document outlines future development tasks for the `EffDim` library. These tasks are well-suited for autonomous AI agents to pick up and implement.

## 1. Enhance Testing and CI/CD
- Expand the current test suite to rigorously validate estimates against known dimensionalities (e.g., random noise should approximate $D$, manifolds like Swiss Roll should approximate their intrinsic dimension).
- Ensure CI pipeline executes tests for both the standard Python implementation and the compiled Rust extension across different target platforms.
