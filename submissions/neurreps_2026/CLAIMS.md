# CLAIMS

Decision label: `claim_supported_but_scale_dependent`

Permitted (with the scale caveat): at the frozen neighbourhood $k=2048$, rank-conditioned sphere-normal mean curvature is associated with worse local OOF linear-probe performance at the middle and upper evaluated chart positions. The association survives direct-error and denominator checks, but its magnitude attenuates substantially at intermediate neighbourhood scales.

Forbidden: `submission_claim_supported`; exact intrinsic dimension; causality; intrinsic Riemannian curvature of an unknown manifold; scale-invariant or predictive “tracking”; cross-dataset generality; independent replication; DESI label associations.

Primary target: `mag_r_desi_local_oof_r2` (never `mag_r_desi_catalog_value`).

Primary frozen analysis: $n=512$ anchors, $k=2048$.
Controlled Spearman: $d=12$ $+0.143$; $d=16$ $-0.240$; $d=20$ $-0.233$; $\rho_{20}-\rho_{12}=-0.376$.
Raw Spearman at $d=16$: $-0.412$.

Scale comparisons: $n=128$ hash-selected anchors. $k=512$ fails $R_H$ reliability and is not confirmatory.
