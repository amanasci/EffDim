# REPRODUCIBILITY

```
cd ~/platonic-universe && source .venv/bin/activate && PYTHONPATH=experiments \
  python experiments/geometry/run_curvature_probe_submission_validation.py
```

n_perm=10000, n_boot=2000, seed=0, k=2048.
No geometry refit. Reads frozen QPD, rank-sweep, multimodel, and nested-curvature artifacts.
Smoke: add `--smoke`.

Official NeurReps 2026 Extended Abstract limit: 4 pages excluding references and appendices.
The Findings track is the one with no page limit. Do not upload a Findings-format preprint to this track.

Compile the anonymous extended abstract (from this directory):

```
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Single-file source (official template request):

```
latexpand --empty-comments main.tex > main_submission.tex
latexmk -pdf -jobname=main_submission -interaction=nonstopmode -halt-on-error main_submission.tex
```
