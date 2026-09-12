# Curvature handoff example configs

Read-only examples. Paths use environment variables, never a machine-specific home directory.

| Variable | Meaning |
|---|---|
| `PLATONIC_ROOT` | Repository root containing `outputs/`, `experiments/` |
| `PLATONIC_OUTPUTS` | Geometry output root (default `$PLATONIC_ROOT/outputs/geometry`) |
| `PLATONIC_EMBEDDINGS` | Frozen embedding / graph store (not in this bundle) |
| `PLATONIC_CHECKPOINTS` | Autoencoder checkpoints (not in this bundle) |

These YAML files do **not** launch jobs. They document the frozen protocol so a later authorized run can be reconstructed.

**Expensive if re-run later (do not launch from this handoff):**

- any Q chart refit (`k=2048`, 512 anchors)
- decoder autodiff on real embeddings
- Freedman–Lane permutations (`n_perm=10000`) or bootstrap (`n_boot=2000`)
- cross-model loops
