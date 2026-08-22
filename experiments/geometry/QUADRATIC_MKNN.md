# Per-patch quadratic mKNN

Focused experiment: does **local quadratic flattening** recover more cross-model
mKNN than **tangent/PCA flattening** (and ambient) on Physics
`vit_base` / `dinov3` / `clip_base`?

Reuses `NestedChart` / `fit_nested_chart` from
`physics_activation_atlas/sphere_normal_quadratic.py` (no geometry redesign).

| Paper name | Implementation |
|---|---|
| \(Q_T\) | `NestedChart.A_flat` |
| \(Q_R\) | forced sphere radial via `decode_R` = `Normalize(x0 + J u)` |
| \(B^S\) | `NestedChart.BS_flat` |
| Primary distance | inverse-coordinate flat \(\|u^\star\|_2\) |

## Run

```bash
export PLATONIC_ROOT=~/platonic-universe
export PYTHONPATH=experiments
source "$PLATONIC_ROOT/.venv/bin/activate"

# Dense activation charts (original)
python experiments/geometry/run_quadratic_mknn.py \
  --space dense \
  --output-dir outputs/geometry/quadratic_mknn/smoke

# TopK SAE code charts (+ optional IDF)
python experiments/geometry/run_quadratic_mknn.py \
  --space sae_idf \
  --output-dir outputs/geometry/quadratic_mknn/smoke_sae_idf \
  --n-anchors 96 --chart-scales 256,512 --retrieval-ks 5,10,20

# Component / geodesic / random-B^S follow-up
python experiments/geometry/run_quadratic_mknn.py \
  --output-dir outputs/geometry/quadratic_mknn/smoke_phase2 \
  --phase2

# Shared patch neighbourhoods (reference = vit_base)
python experiments/geometry/run_quadratic_mknn.py \
  --patch-mode shared \
  --output-dir outputs/geometry/quadratic_mknn/smoke_shared
```

Outputs land under `$PLATONIC_ROOT/outputs/geometry/quadratic_mknn/<tag>/`
(`quadratic_mknn_report.md`, mKNN tables, heatmaps, patch diagnostics).

## Critical comparisons

\[
\Delta_{\mathrm{quad}}(K,k)=\mathrm{mKNN}_{\mathrm{flatQ}}-\mathrm{mKNN}_{T}
\]

Candidate retrieval is **ambient-global** then locally reranked (not restricted
to the chart’s \(K\) neighbours).
