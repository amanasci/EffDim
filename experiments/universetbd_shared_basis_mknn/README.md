# UniverseTBD SAE shared-basis mKNN (k-scaling)

Apply the **Ridge affine SAE shared-basis + IDF** method across UniverseTBD /
Platonic Universe embedding pairs and score mutual kNN at

\[
k \in \{10, 20, 50, 100\}.
\]

Dense baseline follows the Platonic Universe paper protocol
([arXiv:2509.19453](https://arxiv.org/pdf/2509.19453) Table 2): ambient cosine
MKNN on the **full catalog**. Learned methods are fit on a train split and
scored as held-out queries with full-catalog galleries.

No new SAE training: reuses existing TopK checkpoints under `$PLATONIC_ROOT/outputs/sae/`.

For protocol details and the `full_paper_protocol` numbers, see [`CONTEXT.md`](CONTEXT.md).

## Layout

| File | Role |
|---|---|
| `run_universetbd_shared_basis_mknn_ks.py` | **Main runner** (smoke / full, multi-k, report + figures) |
| `sae_affine_basis_mknn_gpu.py` | Ridge affine map, SAE encode, IDF, knn/mknn helpers |
| `compatible_pairs.yaml` | 34 named pairs (physics / jwst / desi / legacy / cosmosweb) |
| `_common.py` | Parquet loaders, `PLATONIC_ROOT` resolution |
| `sae/sae_model.py` | Vendored `TopKSAE` |

## Prerequisites

- CUDA GPU (recommended ≥16 GB for `n=16384` full-catalog knn)
- Row-aligned embedding parquets under `$PLATONIC_ROOT/data_hf/`
- TopK SAE dirs under `$PLATONIC_ROOT/outputs/sae/<parquet_stem>/<column>/<tag>/`
  (`model.pt`, `config.json`, `scaler_stats.npz`)
- Python env with [`requirements.txt`](requirements.txt)

```bash
export PLATONIC_ROOT=/path/to/platonic-universe
source "$PLATONIC_ROOT/.venv/bin/activate"   # or any env with requirements.txt
pip install -r experiments/universetbd_shared_basis_mknn/requirements.txt
```

## Run

From the EffDim repo root (or any checkout of `sae-shared-basis`):

```bash
# Smoke: 5 representative pairs
python experiments/universetbd_shared_basis_mknn/run_universetbd_shared_basis_mknn_ks.py \
  --mode smoke --run-tag smoke_paper_v2

# Full: all includable pairs from compatible_pairs.yaml
python experiments/universetbd_shared_basis_mknn/run_universetbd_shared_basis_mknn_ks.py \
  --mode full --run-tag full_paper_protocol

# Restrict by survey or named pairs
python experiments/universetbd_shared_basis_mknn/run_universetbd_shared_basis_mknn_ks.py \
  --mode full --surveys jwst,legacy --run-tag jwst_legacy_only

python experiments/universetbd_shared_basis_mknn/run_universetbd_shared_basis_mknn_ks.py \
  --pairs jwst_cross_vit,legacy_cross_vit --run-tag paper_pairs
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--ks` | `10,20,50,100` | Neighbourhood sizes |
| `--max-n` | `16384` | Cap catalog size (JWST uses full ~1496 via yaml `default_max_n: 0`) |
| `--test-size` | `0.2` | Holdout fraction for map / IDF fit |
| `--alpha` | `1.0` | Ridge penalty |
| `--platonic-root` | `$PLATONIC_ROOT` or `~/platonic-universe` | Data / SAE tree |
| `--out-dir` | `outputs/universetbd_shared_basis_mknn_ks` | Under `PLATONIC_ROOT` |

## Outputs

`$PLATONIC_ROOT/outputs/universetbd_shared_basis_mknn_ks/<run-tag>/`

- `mknn_by_k.parquet` — per pair / method / k
- `pair_fit_metrics.parquet` — Ridge map test MSE / R² / cosine
- `aggregate_summary.csv`, `report.md`, `figures/mknn_vs_k_*.png`
- `pair_manifest.csv`, `config.json`

### Methods in the table

| method | Protocol |
|---|---|
| `dense_cosine` | Paper: full-catalog ambient cosine MKNN |
| `dense_cosine_heldout` | Same embeddings; average MKNN only on test queries (full gallery) |
| `sae_*` / `shared_*` | Train-fit codes/maps/IDF; held-out queries, full gallery |
| `shared_best_basis_idf` | max of the two shared-IDF directions |
| `dense_cosine_test_subset` | Diagnostic only (inflated; not the baseline) |

**Primary lift:** `shared_best_basis_idf − dense_cosine_heldout`.  
**Paper absolute baseline:** `dense_cosine`.

## Optional single-pair Ridge check

```bash
python experiments/universetbd_shared_basis_mknn/sae_affine_basis_mknn_gpu.py \
  --parquet1 data_hf/physics/vit_base_test.parquet --col1 vit_base_galaxies \
  --parquet2 data_hf/physics/dinov3_vitb16_test.parquet --col2 dinov3_vitb16_galaxies \
  --sae1 outputs/sae/vit_base_test/vit_base_galaxies/F2048_k20_seed0 \
  --sae2 outputs/sae/dinov3_vitb16_test/dinov3_vitb16_galaxies/F2048_k20_seed0 \
  --max-n 16384
```
