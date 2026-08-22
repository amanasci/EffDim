# REPRODUCIBILITY

```
cd ~/platonic-universe && source .venv/bin/activate && PYTHONPATH=experiments \
  python experiments/geometry/run_curvature_probe_submission_validation.py
```

n_perm=10000, n_boot=2000, seed=0, k=2048.
No geometry refit. Reads frozen QPD, rank-sweep, multimodel, and nested-curvature artifacts.
Smoke: add `--smoke`.
