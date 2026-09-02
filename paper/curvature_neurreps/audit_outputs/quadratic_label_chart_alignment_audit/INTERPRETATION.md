# INTERPRETATION

Original mechanical decision (unchanged): `quadratic_chart_link_unresolved`

Audit interpretation: `geometry_regularized_quadratic_decoding`

## Constraint
- algebraic numerical rank (median): 136.0
- energy r_95 (median): 90.0
- rank used by frozen BS (median): 48.0 (fraction of 136: 0.35294117647058826)
- genuinely constrained by energy rank: False
- implementation cap only: True

## Alignment
Haar/spectrum-preserving null survived: True
p_MC (median A_B vs Haar): 0.0004997501249375312

## Shuffle
False-positive safety (synthetic, one-sided dQ>0): True
Null calibration (synthetic): False
Real nested-CV false-positive safety: True

## Mediation
Partial correlation of K_H with dMSE_G->P rose after conditioning on Delta_Q. Quadratic decoding does **not** explain the previous patch/global adaptation result.

## Paper-level wording (if the audit interpretation is a quadratic-chart class)

> At the frozen chart rank and neighbourhood scale, the physical label exhibits held-out quadratic structure in local chart coordinates. The predictive importance of this structure increases with sphere-normal mean-curvature energy, and the label Hessian preferentially aligns with high-energy sphere-normal bending modes.
