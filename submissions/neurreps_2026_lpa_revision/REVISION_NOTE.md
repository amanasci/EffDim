# LPA revision note

Package: `submissions/neurreps_2026_lpa_revision/`

Derived from `submissions/neurreps_2026/` after the full local-probe-adaptation audit
(`outputs/geometry/physics_local_probe_adaptation_audit/`, host run finished 2026-08-23).

## Changes vs base submission

1. **Main text** — one limitations sentence pointing to Appendix `app:lpa`.
2. **Appendix** — new section *Exploratory local probe adaptation (post hoc)* with audit numbers.
3. **CLAIMS.md** — LPA permitted only as relative adaptation in the appendix.

## Audit gate

- Final label: `curvature_predicts_relative_local_adaptation`
- Manuscript action: `include_as_exploratory_appendix` (not main-result)
- Mean $\Delta\mathrm{MSE}_{G\to P}\approx-0.10$ disclosed
- Direction-adaptation interpretation rejected

## Build

```bash
cd submissions/neurreps_2026_lpa_revision
latexmk -pdf -interaction=nonstopmode main.tex
# Main body must remain ≤ 4 pages (refs + appendix outside limit).
```
