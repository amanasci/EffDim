# Cross-model Hessian mismatch

Replication of the readout-curvature method: does
\(\|\mathcal H_y-B_w\|_g\) predict held-out probe MSE, and does
alignment \(\cos_g(\mathcal H_y,B_w)\) predict lower error?

Five encoders, four targets, \(d=16\), \(k=2048\). No manuscript edits.
Paper PDF was not in-repo; implementation follows the written spec and
reuses frozen decoder / probe / split artifacts.
