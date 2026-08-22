# Audit methods

The completed adaptive run and all listed geometry trees are **read-only**.
Corrected tables are written only under `outputs/geometry/physics_adaptive_dataset_curvature_probe_audit`.

## Estimands

The frozen ViT-B / `mag_r_desi` discovery curve is

$$\\rho_d=\\rho_{\\mathrm{Spearman}}\\bigl(K_H^{(d)},\\mathrm{local\\_r2}\\bigr)$$

where `local_r2` is the out-of-fold local ridge-probe $R^2$ for target
`mag_r_desi` in `local_probe_fields.parquet`. It is **not** the catalog
magnitude.

The adaptive run estimated the different quantity

$$\\rho_d=\\rho_{\\mathrm{Spearman}}\\bigl(K_H^{(d)},\\mathrm{catalog\\ mag\\_r\\_desi}\\bigr).$$

Three curves are reported and never aliased:

1. Raw association (stated \(y\)).
2. Frozen discovery-control association (`local_r2` + `local_probe_fields` controls).
3. Harmonized catalog-control association (catalog \(y\) + the same control names).

Curve 3 is not discovery parity.

## Parity

For \(d\\in\\{12,16,20\\}\) the audit compares embeddings, anchors, \(k=2048\)
neighbours, \(K_H\), and both \(y\) vectors on the original sets, the
intersection, and the frozen discovery order. Factorial Spearman
correlations are computed from the per-anchor tables without calling either
experiment's high-level inference.

## DESI

Alignment requires object IDs, a source-row manifest, or a reproducible
reconstruction of embedding order. Equal catalog and embedding row counts
are recorded and are **not** treated as proof. No correlation-maximizing
permutation is searched.

## Inference

Permutations: raw shuffle of \(y\); controlled rank-space Freedman–Lane.
Same-object physics labels share one object permutation. DESI, when
computed, is an independent sample and is excluded from scientific
conclusions while alignment is unproven.

Global corrections (confirmatory family, discovery `mag_r_desi` excluded):

- Unstudentized \(\\max|\\rho|\) (the previous global statistic).
- Westfall–Young min-\(p\) on within-label curve-level permutation \(p\)-values.
- Studentized \(\\max|T|\) with \(T=(\\rho-\\mu_0)/\\sigma_0\).

Zero exceedances are reported as \(p<1/(B+1)\), never \(p=0\). Monte Carlo
intervals use a Clopper–Pearson exceedance interval mapped through
\((e+1)/(B+1)\).

The global “any association” test is **not** a test of a common dimensional
transition.

## Sample size

The inferential unit is the curvature anchor. A label is underpowered if
valid labelled anchors \(< 64\) (frozen \(|\\rho|=0.35\), \(\\alpha=0.05\),
nominal 80% power floor).

## Scale

Scale sensitivity is deferred until discovery-quantity parity and every
included label join are proved. This audit writes `AUDIT_COMPLETE.json`
and does **not** write a scientific `COMPLETE.json`.

Thresholds: `{"alpha": 0.05, "min_valid_anchors": 64, "n_perm": 10000, "n_boot": 2000, "r_h_fail": 0.2, "kh_exact_atol": 1e-12, "kh_tol": 1e-08}`
