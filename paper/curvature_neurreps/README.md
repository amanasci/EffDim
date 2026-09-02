# Curvature / photometric-decoding paper bundle

Self-contained notes, copied audit reports, and draft TeX for the curvature–decodability line.

**Read first:** repository-root [`CONTEXT.md`](../../CONTEXT.md).

Original code trees under `experiments/geometry/` and venue packages under `submissions/` are the live sources. This directory is an archive of notes plus **copied** JSON/CSV/MD (and some PDFs) from the science host. Do not treat it as a place to rewrite frozen experiment packages.

## Layout

| Path | Contents |
|------|----------|
| `submission_neurreps_2026/` | Copy of NeurReps extended abstract |
| `submission_neurreps_2026_lpa_revision/` | Copy + LPA appendix |
| `drafts/` | Earlier working TeX / figures |
| `working_notes/` | Older CONTEXT excerpt, claim notes |
| `figure_scripts/` | Plot scripts |
| `audit_outputs/submission_validation/` | Frozen CPRS / validation reports |
| `audit_outputs/adaptive_dataset_curvature_probe_audit/` | Adaptive-dataset audit |
| `audit_outputs/local_probe_adaptation_audit/` | LPA audit |
| `audit_outputs/quadratic_label_chart_alignment/` | QLCA reports + `anchor_risks.csv` |
| `audit_outputs/quadratic_label_chart_alignment_audit/` | Rank / Haar / shuffle audit |
| `audit_outputs/multilabel_chart_screen/` | Global OOF multi-label screen |
| `audit_outputs/multilabel_chart_screen_quadratic/` | L/UQ/alignment multi-label screen |

**Venue TeX to compile:** `submissions/neurreps_2026/` and `submissions/ml4ps_2026/` (not the copies here).

## Decision / interpretation labels (workflow metadata)

These are **not** sentences for a paper abstract.

- Submission validation: `claim_supported_but_scale_dependent`
- LPA: `include_as_exploratory_appendix`
- QLCA mechanical: `quadratic_chart_link_unresolved` (synthetic gate; not a ViT-B science fail)
- QLCA audit interpretation: `geometry_regularized_quadratic_decoding`
