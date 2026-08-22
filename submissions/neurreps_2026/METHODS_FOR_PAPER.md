# METHODS_FOR_PAPER

Extracted from the implementation used for the frozen ViT-B / physics analysis. Not reconstructed from memory.

## 1. Local coordinates and centring

File: `experiments/geometry/physics_activation_atlas/full_curvature_audit.py` (`pca_tangent_gpu`, `fit_nested_fixed_tangent_gpu`).

Embeddings are row-ℓ₂ unit-normalised in the multimodel prepare stage (`load_model_X`). At an anchor \(x_0\), the neighbourhood \(X_{\mathrm{loc}}\) is sphere-centred:

- \(x_0 \leftarrow x_0/\|x_0\|\)
- \(\Delta X \leftarrow (X_{\mathrm{loc}}-x_0) - ((X_{\mathrm{loc}}-x_0)x_0)\,x_0^\top\) (radial component removed before PCA)
- nested tangent basis \(J\) from PCA/SVD of \(\Delta X\), then `sphere_project_basis`: \(J \leftarrow \mathrm{orth}((I-x_0 x_0^\top)J)\).

Local coordinates: \(U = (X-x_0)J\). Coordinate RMS whitening: \(s_a = \sqrt{\mathrm{mean}(U_{\mathrm{fit},a}^2)}\) (`fit_nested_fixed_tangent_gpu`).

Identity test: after `sphere_project_basis`, \(x_0^\top J \approx 0\) and \(J^\top J \approx I\).

## 2. Nested coordinate basis

File: `experiments/geometry/physics_activation_atlas/nested_dimension_curvature.py` (`nested_pca_frame`, `_fit_rank`).

A single nested PCA frame \(J_{:,1:d_{\max}}\) is computed; rank-\(d\) charts use \(J_{:,1:d}\). Split-half curvature uses three random splits of the \(k\)-neighbourhood; each half is itself split 80/20 for ridge selection (`_half_fit_indices`).

## 3. Quadratic-map convention and factors of 1/2

File: `experiments/geometry/physics_activation_atlas/quadratic.py` (`quadratic_features`); packing in `confirmatory_object_curvature.py` (`unpack_BS_symmetric`).

Features are \(\phi_{ab}=u_a u_b\) for \(a\le b\) (no \(\sqrt{2}\) and no \(1/2\) in the feature map). Off-diagonal packed coefficients store \(2B_{ab}\), so unpacking divides off-diagonals by 2. The decode residual is \(B_{\mathrm{flat}}\phi\), which equals \(\sum_a B_{aa}u_a^2 + 2\sum_{a<b}B_{ab}u_a u_b\).

QPD uses a different, metric-aware map `phi2` with off-diagonal \(\sqrt{2}\,u_a u_b\) (`physics_quadratic_predictive_dimension/algebra.py`). Curvature \(K_H\) does **not** use `phi2`.

## 4. Regularisation and fitting

File: `full_curvature_audit.py` `fit_nested_fixed_tangent_gpu`.

Two-stage ridge on quadratic features \(\Phi\): (i) tangential warp \(A\) selected by validation MSE after row-ℓ₂ decode; (ii) sphere-normal \(B^S\) on residuals after the warp, again λ-selected on validation. Decode is row-ℓ₂ renormalised. Forced-sphere radial residual is formed by scaling targets by \(\|L\|\) before the tangential/normal residual (radius normalisation).

## 5. Metric whitening

Coordinate scales \(s_a\) as above. Downstream SVD analyses rescale \(B^S\) columns by \(s_a s_b\) (`sphere_normal_quadratic.py` `fit_config_nested`). Production \(K_H\) uses packed \(B^S\) after the sphere-normal projection, without a further ambient metric.

## 6. Tangent and sphere-normal projectors

File: `sphere_normal_quadratic.py`.

- `sphere_project_basis(x0,J)`: \(J\leftarrow\mathrm{orth}((I-x_0 x_0^\top)J)\).
- `normal_projector_apply(V,x0,J)`: \(P_{N,S}V = V - QQ^\top V\) where \(Q=\mathrm{qr}[x_0\,J]\).
- After fitting, \(B^S \leftarrow B^S - Q(Q^\top B^S)\) so columns of \(B^S\) lie in \(\mathrm{im}\,P_{N,S}\).

Identities: \(P_{N,S}^2=P_{N,S}\), \(P_{N,S}x_0=0\), \(P_{N,S}J=0\).

## 7. Subtraction of the forced sphere-radial component

In `fit_nested_fixed_tangent_gpu`, linear decode \(L = x_0 + U J^\top\) is used as the sphere-radial carrier. Residuals for \(B^S\) are projected orthogonal to \(\mathrm{span}(x_0,J)\) before the ridge solve, so the fitted second fundamental form is sphere-normal rather than ambient.

## 8. \(B_S\), \(H_S\), \(K_H\), radius normalisation

Files: `effdim_curvature_metrics.py` (`decompose_tensors`, `cross_metric_pair`); production scalar in `_fit_rank`.

Unpack \(B^S\) to a symmetric \(d\times d\) matrix per ambient coordinate. Mean curvature vector \(H_a = d^{-1}\sum_i B^S_{a,ii}\). Split-half production statistic:

\[
K_H^{\mathrm{cross}} = \langle H^{(A)}, H^{(B)}\rangle
\]

(`cross_metric_pair`: `khx = dot(H_A, H_B)`). Reliability \(R_H = 2\langle H_A,H_B\rangle / (\|H_A\|^2+\|H_B\|^2)\) (`tensor_agreement`). Radius normalisation is the row-ℓ₂ decode above, not a separate scalar \(r\) in the \(K_H\) formula.

Note: `metric_scalars()["K_H"]` is \(\|H\|\), used in identity tests. The frozen discovery curve uses \(K_H^{\mathrm{cross}}\), not \(\|H\|\).

## 9. OOF probe construction

File: `multimodel_graph_prior_quadratic.py` `stage_global_probes`.

Five-fold ridge (\(\alpha=100\)) from embeddings to labels. Fold \(f\) is predicted from weights fit on all other folds. Artifact: `global_probes/oof_predictions/vit_base_mag_r_desi.npz` key `oof`.

## 10. Exact local \(R^2\)

File: `global_probe_curvature_alignment.py` `local_r2_fixed_predictions` / `weighted_r2`.

On neighbours with finite \(y\) and \(\hat y\), uniform weights \(w_i=1/n\):

\[
R^2_{\mathrm{local}} = 1 - \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}.
\]

This is `mag_r_desi_local_oof_r2`. Catalog `mag_r_desi` is `mag_r_desi_catalog_value` and is never substituted.

## 11. Controls, bootstrap, permutation

File: `physics_curvature_probe_rank_sweep/inference.py`.

Controls: `log_knn_radius`, `local_label_variance`, `local_evaluation_count`. Partial Spearman on ranks. Permutation: \(B=10^4\); raw shuffles \(y\); controlled uses rank-space Freedman–Lane. FWER from the max-\(|\rho|\) envelope across ranks. Bootstrap: \(B=2000\) paired anchor resamples; simultaneous bands from the 95th percentile of \(\max_d|\hat\rho_d^\ast-\hat\rho_d|\). Never report \(p=0\); use \(p<1/(B+1)\).

## 12. Dimensionality-range construction

Held-out linear/quadratic NMSE and pooled \(R^2_L\) from QPD (`aggregate_risk_curves.csv`, \(k=2048\)). Variance crossings: \(\tau=0.80\to d=12\), \(\tau=0.85\to d=20\) (post hoc). Predeclared chart positions for scale mapping: lower \(d=12\), middle \(d=16\), upper \(d=20\). These are geometry-only and are not chosen by maximising correlation. Rank analysis over \(d=8,\ldots,20\) is exploratory / not preregistered. Linear charts continue to improve beyond this family (\(d_L^{\mathrm{plat}}=115\) in the adaptive interval); the paper reports a plausible finite-scale range, not an intrinsic dimension.
