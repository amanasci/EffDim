# Methods: adaptive per-dataset curvature–physics probe

## Estimand

The unit of analysis is a **dataset**, not a dataset pair. For each eligible
dataset $j$ the geometry-only interval $D_j$ is chosen from embeddings, then
frozen. Only afterwards are registered physics labels loaded.

For every $d\in D_j$ the rank-conditioned statistic $K_{{H,j}}^{{(d)}}$ is the
frozen nested split-half inner product $\langle H_A,H_B\rangle$ of the
sphere-normal mean-curvature vector after removing $Q_R$ and whitening. The
maximizing rank is **not** an intrinsic dimension.

## Discovery vs confirmation

Discovery is the completed ViT-B / `mag_r_desi` rank sweep on Smith42/galaxies.
It is plotted as a reference curve and **excluded** from confirmatory aggregate
$p$-values.

## Inventory

Eligible catalogs are the Smith42 registries (`galaxies`, `desi_hsc_crossmatched`,
`jwst_hsc_crossmatched`, `legacysurvey_hsc_crossmatched`, `sdss_hsc_crossmatched`)
plus documented adapters. Labels are taken only from registered columns, never
from every numeric field. Orientation is taken from documented photometry /
redshift semantics and is never reversed from observed $\rho$.

Primary encoder family: ViT-B imaging (`vit_base_galaxies` or `vit_base_hsc`).
Other encoders are listed in the inventory and are not mixed into the primary
replication.

## Neighbourhood scale

Primary $k$ is the largest preset in {256,512,768,1024,1536,2048} with
$k \le 0.125 n$. This rule is sample-size only.

## Geometry range

Held-out linear $R^2_L(d)$ uses the complete spherical-log ambient energy.
Thresholds tau in {0.70,0.75,0.80,0.825,0.85,0.875,0.90,0.95} are not
extrapolated. Quadratic screening is coarse (fixed-coordinate) then refined
with the frozen closest-point map from `physics_quadratic_predictive_dimension`.
$D_j$ follows the predeclared min/max formula. If $d_{90}$ is not reached the
interval is marked right-truncated.

## Inference

Within each dataset×label, one label permutation is reused across all
$d\in D_j$ ($B\ge 10000$ unless smoke). Report $p<1/(B+1)$ on zero exceedances.
Paired-anchor bootstrap $B\ge 2000$ unless smoke. Same-object labels share one
object permutation. Independent datasets enter a joint maximum statistic.
Discovery is excluded from that family.

Shared-core confounders: `log_knn_radius`, `local_label_variance`,
`local_evaluation_count`. Dataset-specific extras (DESI `ZERR`/`EBV`) are
secondary only and are not searched.

## Cross-dataset axes

Absolute rank $d\mapsto\rho$ and variance $\tau=R^2_L(d)\mapsto\rho$.
Interpolation on $\tau$ is only between observed reliable ranks.

Smoke=False. Seed=0.
