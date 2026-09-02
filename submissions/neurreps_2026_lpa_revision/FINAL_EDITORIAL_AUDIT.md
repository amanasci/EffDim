# FINAL_EDITORIAL_AUDIT

## 1. Final title

Rank- and Scale-Conditioned Curvature Is Associated with Local Linear Decodability in Vision Representations

## 2. Final abstract

Geometric summaries of neural activations depend on the rank and neighbourhood scale of the local chart used to compute them. We evaluate nested regularised quadratic charts of a ViT-B galaxy representation, define sphere-normal mean curvature \(K_H^{\mathrm{cross}}\) from a split-half second fundamental form, and select chart ranks from held-out reconstruction rather than a single eigengap. At the frozen neighbourhood \(k=2048\) (\(n=512\) anchors), greater curvature is associated with poorer local out-of-fold linear decodability of \(r\)-band magnitude (`mag_r_desi_local_oof_r2`) at the middle and upper evaluated ranks: controlled Spearman \(\rho=-0.240\) at \(d=16\) and \(\rho=-0.233\) at \(d=20\). Direct-error associations have the opposite sign (MAE \(\rho=+0.251\), MSE \(\rho=+0.227\)), while local SST/target variance is essentially unassociated (\(\rho\approx-0.025\)). On \(n=128\) hash-selected anchors the middle/upper direction remains negative at \(k=1024\) and \(k=1536\), but magnitudes attenuate substantially; \(k=512\) fails curvature reliability and is not confirmatory. The association is therefore supported at the frozen scale and scale-dependent. Scope: one encoder, one astronomical dataset, one probe target, an exploratory rank sweep, and no independent replication.

## 3. Exact page allocation

Compiled `main.pdf` and `main_submission.pdf` are 7 pages:

| Pages | Content |
|---|---|
| 1–4 | Main text (title, abstract, §§1–4, Figures 1–2) |
| 5 | References |
| 6–7 | Appendix |

Official Extended Abstract limit: 4 pages excluding references and appendices. Findings is the unlimited track. This allocation complies.

## 4. Material textual changes

- Title no longer uses “tracks”; it states rank- and scale-conditioning and association.
- Abstract rewritten after the fixed decision label; includes method, frozen \(k=2048\) numbers, direct-error/SST contrast, \(n=128\) attenuation, \(k=512\) reliability failure, and scope.
- Dimensionality language is “evaluated chart range \(d=12\)–\(20\)” (~80–85% explained variance). Linear reconstruction is still falling at \(d=20\). No intrinsic-dimension claim.
- Primary vs scale sample sizes are distinguished (\(n=512\) vs \(n=128\)).
- Direct-error audit is in the main result section, not only the appendix.
- \(k=512\) is reported as unreliable, not as confirmatory opposite-sign evidence.
- Contributions rewritten as method / statistic / frozen-scale association / audit.
- Catalog-substitution history moved to a brief appendix note.
- Editors line set to “NeurReps 2026 organisers” (not “List of editors’ names”).
- Review watermark retained (official template requires it until camera-ready).
- Two methodological citations added after verification: Alain & Bengio (2017, arXiv:1610.01644) for linear probes; Donoho & Grimes (PNAS 2003) for local Hessian/quadratic charts.
- Figures redrawn from frozen artifacts: Fig. 1 annotates that linear NMSE is still falling; Fig. 2 overlays \(n=128\) scale points and marks \(k=512\) unreliable; right panel is OOF MSE at predeclared \(d=16\).

## 5. Scientific-number consistency checks

Authoritative sources: `outputs/geometry/physics_curvature_probe_submission_validation/` (`metric_associations.csv`, `metric_bootstrap.csv`, `metric_permutation.csv`, `scale_sensitivity.csv`, `scale_reliability.csv`, `parity_correlations.csv`, `leakage_report.json`, `decision.json`).

All main-text rounded values match those artifacts to the stated rounding. No silent substitution of catalog magnitude.

| Quantity | Artifact | Paper |
|---|---|---|
| ctl \(\rho_{12}\) | \(+0.142990\) | \(+0.143\) |
| ctl \(\rho_{16}\) | \(-0.240484\) | \(-0.240\) |
| ctl \(\rho_{20}\) | \(-0.233325\) | \(-0.233\) |
| \(\rho_{20}-\rho_{12}\) | \(-0.376315\) | \(-0.376\) |
| raw \(\rho_{16}\) | \(-0.412430\) | \(-0.412\) |
| MAE ctl \(\rho_{16}\) | \(+0.250721\) | \(+0.251\) |
| MSE/SSE ctl \(\rho_{16}\) | \(+0.227048\) | \(+0.227\) |
| SST ctl \(\rho_{16}\) | \(-0.024557\) | \(-0.025\) |
| \(R^2\) CI \(d=16\) | \([-0.322515,-0.155741]\) | \([-0.323,-0.156]\) |
| MAE CI \(d=16\) | \([0.153662,0.332814]\) | \([0.154,0.333]\) |
| MSE CI \(d=16\) | \([0.122757,0.317103]\) | \([0.123,0.317]\) |
| FWER \(d=12\) | \(0.0126\) | \(0.013\) |
| FWER \(d=16,20\) | \(0\) in CSV | \(p<10^{-4}=1/(B+1)\) |
| SST FWER \(d=16\) | \(1.0\) | \(1.00\) |
| outer-half \(R^2\) \(d=16\) | \(-0.246463\) | \(-0.246\) |
| outer-half MSE \(d=16\) | \(+0.237510\) | \(+0.238\) |
| shuffle ctl \(d=16\) | \(-0.059499\) | \(\approx-0.059\) |
| common-128 ctl \(\rho_{16}\) | \(-0.170636\) | \(-0.171\) |
| \(k=1024\) ctl \(\rho_{16}\) | \(-0.026643\) | \(-0.027\) |
| \(k=1536\) ctl \(\rho_{16}\) | \(-0.080468\) | \(-0.080\) |
| \(k=512\) \(R_H\) med \(d=16\) | \(-0.180888\) | \(\approx-0.19\) (table uses \(-0.19\)) |
| linear \(R^2_L\) \(d=12,20\) | \(0.803683\), \(0.851077\) | \(\approx 0.80\), \(0.85\) |

Disagreement / non-use: `METHODS_FOR_PAPER.md` mentions \(d_L^{\mathrm{plat}}=115\) from an adaptive interval. The QPD `aggregate_risk_curves.csv` used for Figure 1 only includes \(d\le 20\). The paper therefore states only that linear NMSE is still falling at \(d=20\), not a numerical plateau at 115.

NMSE is algebraically \(1-R^2_{\mathrm{local}}\) (controlled \(\rho_{16}=+0.240484\)) and is labelled as such.

## 6. Figure-to-artifact provenance

Figures were regenerated in `submissions/neurreps_2026/figures/` from frozen files. Validation output trees were not modified.

**Figure 1** (`figure1_dimension.pdf`)

- `outputs/geometry/physics_quadratic_predictive_dimension/aggregate_risk_curves.csv` (read-only copy for plotting; \(k=2048\) linear/quadratic NMSE)
- `outputs/geometry/physics_curvature_probe_rank_sweep/variance_explained.csv` (\(R^2_L\) and increments)
- Shaded evaluated range \(d=12\)–\(20\); no intrinsic-dimension line; annotation that linear NMSE is still falling.

**Figure 2** (`figure2_curvature_probe.pdf`)

- Left: `metric_associations.csv` + `metric_bootstrap.csv` (`ctl_sim_lo/hi` simultaneous bands) for \(n=512\), \(k=2048\); `scale_sensitivity.csv` for \(n=128\) markers (`refit_scale_cache` at \(k=1024,1536,512\); `common_anchor_k2048_subset` at \(k=2048\)); \(k=512\) plotted as unreliable.
- Right: `probe_metrics_full.csv` joined to \(K_H^{\mathrm{cross}}\) at \(d=16\) from `per_anchor_rank_curve.parquet` (rank-sweep, read-only export of the \(d=16\) columns).
- No catalog magnitude; no DESI.

There is no `ARTIFACT_MANIFEST.json` inside the validation tree with relative figure sources; the validation `ARTIFACT_MANIFEST.json` contains host-absolute paths and was not copied into the submission.

## 7. Anonymity and PDF QA

- Official `mlabstract` jmlr template.
- No author block; PDF Author and Title metadata empty.
- Strings `angus`, `/home/angus`, `gmail`, `openreview`, `PlatonicUniverse` absent from `main.pdf`.
- Review watermark present (official sample: remove only for camera-ready).
- Editors: “NeurReps 2026 organisers” (not the template placeholder).
- No overfull boxes in the final `main.log`.
- No undefined citations.
- Fonts: embedded Type 1 + CID TrueType (figures); no Type 3.
- Modular and flattened PDFs both 7 pages with the same page-start headers.

## 8. Remaining author actions

1. Confirm that “NeurReps 2026 organisers” is an acceptable editors line, or replace with the official editors list when announced.
2. If OpenReview requires a single `.tex` file, upload `main_submission.tex` (plus `references.bib`, `jmlr.cls`, figures).
3. Keep the review watermark for submission; remove it only for camera-ready, as the official template states.
4. Do not upgrade the decision label to `submission_claim_supported`.
5. This agent did not submit to OpenReview.

## 9. Recommendation

**ready_to_submit**

Reasons: the scientific decision is `claim_supported_but_scale_dependent` and is stated in the title, abstract, results, and conclusion; numbers match the validation artifacts; the main body is 4 pages; the package is double-blind; figures omit catalog/DESI; scale dependence is in the main text and Figure 2; \(k=512\) is not used as confirmatory evidence.

Not ready would have been required if the paper still implied a scale-invariant effect, an intrinsic dimension, or used catalog magnitude as the primary target. Those issues were corrected.
