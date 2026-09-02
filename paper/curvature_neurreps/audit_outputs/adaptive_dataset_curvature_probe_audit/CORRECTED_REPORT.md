# Corrected report: adaptive dataset curvature–physics audit

**Audit label:** `probe_label_alignment_failure`

The previous label `dataset_specific_curvature_probe_associations` is
**suspended**. This file does not replace the 12.2-hour run; that tree is
unchanged.

## Root cause

First divergence: **probe / label quantity**, not geometry.

The frozen discovery curve correlates \(K_H^{(d)}\) with `local_r2` of the
`mag_r_desi` probe field. The adaptive run correlated the **same** \(K_H\)
with catalog `mag_r_desi`. Those \(y\) vectors are different quantities
(Spearman ≈ -0.214981).

Assigned causes: probe_label_alignment_failure, control_specification_change, inference_or_summary_bug, desi_alignment_unproven, multiple_testing_calibration_problem.

Conditional repair: reuse existing per-anchor \(K_H\) (exact match at
\(d=12,16,20\)). Do not refit geometry. Do not launch a 12-hour rerun.
DESI label associations are removed from scientific conclusions.

## Old-versus-new \(K_H\) parity

- d=12: Pearson=1, Spearman=1, max|Δ|=0, exact_rate=1, identical=True
- d=16: Pearson=1, Spearman=1, max|Δ|=0, exact_rate=1, identical=True
- d=20: Pearson=1, Spearman=1, max|Δ|=0, exact_rate=1, identical=True

A monotone rescaling is not required: the reused ranks are identical.

## Old-versus-new label parity

- Frozen \(y\): `local_r2` (range roughly 0.04–0.44).
- Adaptive \(y\): catalog `mag_r_desi` (range roughly 15–19).
- Pearson=-0.143402, Spearman=-0.214981.
- Physics `sample_id` is the galaxies test-table row. `vit_base_test_labels.npz`
  is row-aligned to the parquet; `selection.npz` indexes both. Equal row
  count is not the proof.

## Anchor and neighbourhood parity

- Shared anchors: 512 / 512, Jaccard=1.
- Same set, different order (adaptive `adcp:` hash of the same 512).
- Neighbours: both use `vit_base_kmax2048.npz` at \(k=2048\), compared after
  aligning on `sample_id`. Agreement=True.

## Factorial raw correlations

- d=12: oldK-oldy=-0.0384256  oldK-newy=-0.000821161  newK-oldy=-0.0384256  newK-newy=-0.000821161  follows=labels
- d=16: oldK-oldy=-0.41243  oldK-newy=0.0535206  newK-oldy=-0.41243  newK-newy=0.0535206  follows=labels
- d=20: oldK-oldy=-0.392251  oldK-newy=0.0275248  newK-oldy=-0.392251  newK-newy=0.0275248  follows=labels

The raw-\\(\\rho\\) disagreement follows **labels**, not \(K_H\) or anchors.

## Frozen versus harmonized controls

| d | raw local_r2 | frozen-control local_r2 | raw catalog | harmonized-control catalog |
|---:|-------------:|------------------------:|------------:|---------------------------:|
| 12 | -0.0384256 | 0.14299 | -0.000821161 | 0.0438967 |
| 16 | -0.41243 | -0.240484 | 0.0535206 | 0.154706 |
| 20 | -0.392251 | -0.233325 | 0.0275248 | 0.182425 |

Frozen published values: d=12 raw -0.038426 ctl 0.14299;
d=16 raw -0.41243 ctl -0.240484;
d=20 raw -0.392251 ctl -0.233325.

The d=12 sign change under frozen controls (+0.143 vs raw −0.038) is a
property of the **discovery** control model, not a reason to prefer
harmonized controls.

## Corrected \(\\Delta^{85-80}\)

- Frozen discovery-control (local_r2, \(d_{85}=20\), \(d_{80}=12\)):
  -0.376315
  \(= -0.233325 - 0.14299\).
- Harmonized catalog-control (not discovery parity):
  0.138528.

There is **one** independent magnitude catalog (DESI), and its label join
is unproven. No leave-one-dataset-out stability and no cross-dataset
meta-analysis are reported as replications.

## DESI alignment

Status: `desi_label_alignment_unresolved`. Proved=False.
Embedding columns are vision vectors only. Catalog `desi_object_id` has no
partner in the embedding parquet. Equal \(n=20465\) is not proof.
DESI geometry is retained. DESI curvature–label associations are **not**
scientific results.

## Anchor-level sample sizes

```
       dataset_id           label  full_dataset_rows  valid_labelled_rows  total_curvature_anchors  valid_labelled_anchors  controlled_analysis_anchors  n_controls  residual_df               missingness  underpowered  scientific                            note
 physics_vit_base      mag_r_desi              86471                86471                      512                     512                          512           3          508   0/512 anchors unlabeled         False        True                             NaN
 physics_vit_base         photo_z              86471                80035                      512                     477                          477           3          473  35/512 anchors unlabeled         False        True                             NaN
 physics_vit_base smooth_fraction              86471                86471                      512                     512                          512           3          508   0/512 anchors unlabeled         False        True                             NaN
 physics_vit_base    stellar_mass              86471                79490                      512                     476                          476           3          472  36/512 anchors unlabeled         False        True                             NaN
 physics_vit_base             sfr              86471                 7306                      512                      45                           45           3           41 467/512 anchors unlabeled          True        True                             NaN
desi_vit_base_hsc          spec_z              20465                20465                      512                     512                          512           3          508   0/512 anchors unlabeled         False       False desi_label_alignment_unresolved
desi_vit_base_hsc           mag_r              20465                20465                      512                     512                          512           3          508   0/512 anchors unlabeled         False       False desi_label_alignment_unresolved
```

`sfr` has **45** valid labelled anchors, not 1,340. It is underpowered
under the frozen \(n<64\) rule.

## Global multiple-testing (scientific family: physics catalog labels, discovery excluded, DESI excluded)

Unstudentized max-|ρ|: p=0.00889911 (0.00889911), CI=[0.00716295, 0.0109297], T=0.52108.

Westfall–Young min-p: p=0.00049995 (0.00049995), CI=[0.000208976, 0.00112373].

Studentized max-T: p=9.999e-05 (<9.999e-05), CI=[9.999e-05, 0.000468773], T=5.58558.

Curves surviving WY (α=0.05):
```
      dataset_id           label  curve_p  wy_adjusted_p
physics_vit_base         photo_z 0.000100       0.000500
physics_vit_base smooth_fraction 0.001500       0.006099
physics_vit_base    stellar_mass 0.000200       0.000900
physics_vit_base             sfr 0.008899       0.035196
```

This is a test of **any association anywhere** in the confirmatory physics
family. It is not a test of a common dimensional transition.

As-published family (includes unaligned DESI) is in the CSV files and is
not used for scientific claims.

## Transition-specific inference

- Any-association: see global tests above.
- Magnitude-transition replication: frozen \(\\Delta^{85-80}=-0.376315\)
  (discovery reference). No proven independent magnitude replicate.
- Redshift: physics `photo_z` vs DESI `spec_z` is a **post hoc** observation.
  DESI `spec_z` is not a scientific result. No signed heterogeneous-label
  transition statistic is formed.

## Reliability sensitivity

High-rank peaks (`smooth_fraction` near d=43, DESI `mag_r` near d=36) sit
where median \(R_H\) is declining. Cutoffs 0.2 (frozen), 0.4, 0.5, 0.6 are
reported in `high_rank_reliability_sensitivity.csv`. Cutoffs were not
chosen to preserve significance. Ranks remain in the figures and are
marked weak when \(R_H\) is below a cutoff.

## Scale analysis

**Not completed.** The prior `COMPLETE.json` left `predeclared_pending`
rows. Discovery-quantity parity is now understood, but DESI joins are
unproven, so scale refits are deferred. No scientific `COMPLETE.json`.

## Corrected scientific label

None. Suspended: `dataset_specific_curvature_probe_associations`.
Audit outcome: `probe_label_alignment_failure`.

## Runtime, tests, paths

- Audit runtime: 4161 s
- Tests: 18
- Output: `/home/angus/platonic-universe/outputs/geometry/physics_adaptive_dataset_curvature_probe_audit`
- Read-only sources: adaptive run, rank sweep, nested curvature, multimodel, QPD
