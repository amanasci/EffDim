# FORMULAS

## D-full (historical, unnormalized)

J = DF, g = J^T J, P_T = J g^{-1} J^T
II^E = (I - P_T) D²F
H^E = g^{ab} II^E_ab    (no 1/d)
F is raw model.decode.

Also reported: H^E / d.

## D-residual

F̃ = F / ||F|| differentiated as a map (not a post-hoc projection of D²F).
B^S = (I - xx^T - P_T) D²F̃
H^S = g^{ab} B^S_ab
Scal = d(d-1) + ||H^S||² - ||B^S||_g²

## Q

Frozen nested_pca_frame + fit_quad + split-half.
K_H^cross = <H_A^S, H_B^S>  (production unpacked / Hessian pair as stored)
K_dir^cross unchanged, no clamping of negative cross-products.
Scal_cross = d(d-1) + <H_A^S, H_B^S> - <B_A^S, B_B^S>_g

## Oracles

T2: same Q neighbourhood indices, clean f(z), exact latents, true tangent.
T3: Sobol uniform latent ball, same radius, ≤2048 points.
