# METHODS — QLCA audit

Read-only inputs: frozen `physics_quadratic_label_chart_alignment` tables and NDC `H_vectors`.
No writes into original experiment or output trees. Original decision label is not edited.

## Rank
Numerical rank uses `max(m,n) * machine_eps * s_max` (numpy `matrix_rank` default).
Energy ranks are cumulative squared singular values at 90/95/99%.
Frozen BS retains `min(r_99, 48)` left singular vectors of ambient `B^S`.
Reachable Hessian fraction is `||P_row(B) γ||² / ||γ||²`.

## Truncated BS
Geometry-only ranks (90/95/99% energy and the frozen 99%+cap-48 rule). Same outer folds, train-only scalar RMS, nested block ridge, and evaluation objects as the original comparison. Ranks are never chosen using labels.

## Alignment nulls
Primary: Haar right-singular frames preserving Σ (n=2000). Secondary: radius/K_H/rank-binned pairing of γ_i with spectrum_j. Split-half A vs B. Stability subset uses the frozen cosine threshold 0.5, not tuned on A_B.

## Shuffle
False-positive safety: one-sided median Δ_Q>0. Null calibration: |median Δ_Q| small. Synthetic path is the original fixed-α estimator; real-design path is nested-CV L vs UQ on frozen coordinates.
