# METHODS — known curvature point/patch fixture audit

## Estimands and scales

Two instruments are scored against *matched* geometric targets.

- **Estimator D** (learned decoder) estimates the *pointwise* sphere-normal second
  fundamental form of a globally fitted surface
  \(\widetilde F(z)=F(z)/\|F(z)\|\). Its proper target is **T1**.
- **Estimator Q** (frozen local quadratic) estimates a *finite-patch* quadratic
  representation of the observed cloud. Its proper target at neighbourhood
  radius \(r\) is **T2** (uniform volume) or **T3** (sampling-weighted). T1 is
  only the *asymptotic* target of Q as \(r\to 0\) with adequate density.

They are compared directly only after aligning contraction (mean vs full
\(B^S\)) *and* spatial scale.

## Common geometry

- Intrinsic dimension \(d=16\), ambient \(D=768\), unit-normalized.
- Frozen ambient rotation, seed 20260907, SHA256[:16] `251f756d4adcce56`.
- Float64 for all truth calculations and differential geometry.
- Hash-stable anchors by `sha256(seed:sample_id)`.
- Identical observed points and anchors for D and Q within every condition.

## Sphere-normal tensors

\[
J=\partial G,\quad g=J^\top J,\quad P_T=Jg^{-1}J^\top,\quad
P_{N,S}=I-GG^\top-P_T,\quad
B^S_{ab}=P_{N,S}\partial_{ab}G.
\]

\[
H^S=\frac1d g^{ab}B^S_{ab},\qquad
K_{\mathrm{dir}}=\frac{2\|B^S\|_F^2+\|\mathrm{tr}B^S\|^2}{d(d+2)}
\]

after metric whitening. Split \(B^S=gH^S+\mathring B^S\).

**Packing.** T1 \(B^S\) is the Hessian. Frozen Q stores \(\Phi=u_a u_b\)
coefficients \(S\); \(\mathrm{Hess}=2\,\mathrm{unpack}(S)\). Production
unpacked scalars equal one-quarter of the Hessian \(K_{\mathrm{dir}}\)
cross statistic. Tables labelled Hessian use the user formula; tables labelled
unpacked retain the production scalar for continuity with real-data papers.
Negative split-cross values are not clamped.

\(C_\rho=\mathrm{sign}(K_{\mathrm{dir}}^{\mathrm{cross}})\,
\rho\,\sqrt{|K_{\mathrm{dir}}^{\mathrm{cross}}|}\)
on Hessian tensors.

## Fixtures

- F0 great \(S^{16}\subset S^{767}\): \(B^S=0\).
- F1 latitude \(c=0.6\): pure mean curvature \(\kappa=c/r\).
- F2 minimal Clifford \(r^2=s^2=1/2\): \(H^S=0\), \(\mathring B^S\neq 0\).
- F3 nonminimal Clifford \(r^2=0.7\), \(s^2=0.3\).
- F4 low-frequency bumped sphere, widths [0.8, 0.95, 0.7, 1.05].
- F5 high-frequency bumped sphere, widths [0.22, 0.18, 0.25, 0.2].
  Bump parameters were frozen before estimator scoring.

Independent truth: analytic (F0–F3), torch autodiff, and central finite
differences, plus orthogonality, radial identity, latent- and ambient-rotation
invariance. A fixture is used only if these agree within frozen tolerances.

## Estimators

**D.** `cae.PlainAutoEncoder` \(768\to 16\to 768\), hidden (250, 250, 250), SiLU,
600 epochs, `TRAIN_CFG` from the colleague protocol
(lr \(10^{-3}\), weight decay \(10^{-4}\), batch 128). Train/holdout split
seed 20260813, holdout fraction 0.2. Anchors are taken from the holdout only.
Seeds (0, 1, 2) on primary conditions. Full \(B^S\) is retained (not
only the trace).

**Q.** Exact frozen path: `nested_pca_frame` + `fit_quad` (ridge grid
\([10^{-4},\ldots,3]\), A/B splits, sphere-radial removal, frozen packing).

## Finite-patch oracles

T2: population-optimal quadratic under uniform manifold-volume weights
\(\sqrt{\det g}\) over the ambient ball of radius \(r\), Sobol latent
candidates. T3: the same fit with the known fixture sampling density.
Inverse-density-weighted Q is a secondary path against T2.

## Suites

A clean geometry; B sampling; C noise; D density–noise; E scale sweep
on \(n=16384\) only. Bounded fixture check: colleague-scale n=86471 and Suite F (fixed radius / adaptive k / inverse-density weights) were not run.

Primary: \(n=16384\), \(k=2048\), 512 holdout anchors.
