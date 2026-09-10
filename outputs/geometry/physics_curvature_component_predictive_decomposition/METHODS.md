# Methods — curvature-component predictive decomposition

This analysis is an **exploratory decomposition** motivated by completed
full-curvature and QLCA results. It is **not** prospectively preregistered.

## Frozen reuse

Five encoders (ViT-B, DINOv3, CLIP, ConvNeXt-B, ViT-L), 512 hash-stable
anchors, $k=2048$, $d=16$. Same object identities, neighbours,
A/B geometry splits, global/patch predictions, probe folds, evaluation
objects, physical target (`mag_r_desi`), and controls
(`log_knn_radius`, `local_label_variance`, `local_evaluation_count`).
ViT-B QLCA reuses the same charts, $Q$, $B^S$, folds, held-out L/UQ/BS,
and foldwise Hessians $\Gamma$. Alignment is by `sample_id`.

Prior trees are read-only. New writes go only to
`experiments/geometry/physics_curvature_component_predictive_decomposition/`
and `outputs/geometry/physics_curvature_component_predictive_decomposition/`.

## Geometry

The sphere-normal residual of the fitted quadratic chart is $B^S=Q-Q_T-Q_R$.
In orthonormal PCA chart coordinates,

$$
B^S = B^H + \mathring B,\qquad B^H_{ab}=\delta_{ab}H,\qquad H=\tfrac1d\mathrm{tr}B^S.
$$

Split-cross statistics (signed, unclamped):

$$
K_H^{\mathrm{cross}}=\langle H_A,H_B\rangle,\qquad
K_{\mathrm{TF}}^{\mathrm{cross}}=\frac{2}{d(d+2)}\langle\mathring B_A,\mathring B_B\rangle_F,
$$

and $K_{\mathrm{dir}}^{\mathrm{cross}}=K_H^{\mathrm{cross}}+K_{\mathrm{TF}}^{\mathrm{cross}}$.
`K_aniso_cross` from the completed reconciliation **is** $K_{\mathrm{TF}}^{\mathrm{cross}}$.

Within-split descriptive energies $E_H=\|H\|^2$,
$E_{\mathrm{TF}}=\frac{2}{d(d+2)}\|\mathring B\|_F^2$,
$F_H=E_H/E_{\mathrm{dir}}$, and
$C_{\mathrm{trace}}=\|\mathrm{tr}B^S\|^2/(d\|B^S\|_F^2)$
are **not** substituted for split-cross association statistics.

## Inference

Controlled Spearman uses the frozen rank-space Freedman–Lane `associate()`
from `physics_curvature_probe_rank_sweep`. Component-conditional tests add
the other component to the control matrix. Do **not** condition on $K_{\mathrm{dir}}$.

Primary cross-model outcome: global OOF MSE. $R_G^2$ is the sign-reversed
parity endpoint. Patch MSE and $\Delta_{\mathrm{adapt}}$ are secondary and
kept distinct from quadratic label gain $\Delta_Q$.

Encoders share anchors: joint-anchor bootstrap (2,000) and Monte Carlo
permutations (10,000). Equal-weight and Fisher-$z$ means. Holm correction
within the two-component primary family (unique $K_H$ and unique $K_{\mathrm{TF}}$
vs global MSE). This is a diagnostic family, not a confirmatory analysis.

## ViT-B Hessian and probes

$\Gamma=\Gamma_H+\mathring\Gamma$ in the same orthonormal chart.
A component is not interpreted if foldwise cosine $<0.5$.

Alignments $A_B$, $A_H$, $A_{\mathrm{TF}}$ use the QLCA Frobenius-preserving
136-vector convention. Nulls: 2,000 Haar / random-$\gamma$ draws per
anchor, A/B geometry splits, and the induced-energy cross term
$2\gamma^\top B_H^{\mathrm{flat}\top}B_{\mathrm{TF}}^{\mathrm{flat}}\gamma$.

Held-out models on frozen coordinates and folds:

- L: tangent linear
- IQ: isotropic quadratic (1-D trace of $\Gamma$)
- TQ: traceless quadratic (135-D)
- UQ-v2: full quadratic **with** $\alpha_Q=\infty$ omit-quadratic candidate
- BSH / BSTF / BS: chart-constrained, cap-48 SVD (90/95/99% energy ranks reported)

IQ+TQ span check: $n_{\mathrm{TF}}=135$, span error
$9.102292307133257e-15$. Frozen QLCA UQ point estimates are retained for
parity and are not replaced by UQ-v2.

## Decision

Exactly one label is assigned by the frozen rule in `decision.py`
(`rules_version=1`). Labels are not retuned after seeing results.
