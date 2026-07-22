# SAE shared-basis experiments

Cross-model alignment by fitting an **affine map between TopK SAE codes**, so one model’s codes are expressed in the other’s SAE feature basis:

\[
C_{\text{basis}} \approx C_{\text{other}} W + b
\]

Then score held-out **mKNN** between true basis codes and the mapped codes (plus dense / SAE-IDF baselines).

The result that worked well on Physics ViT↔DINOv3 was **Ridge** (not Lasso): ~**0.22** mKNN vs ~0.13 dense / ~0.17 SAE-IDF.

For the full research arc (geometry → topology → SAE shared basis), see [`CONTEXT.md`](CONTEXT.md).

| Experiment | Script | Role |
|---|---|---|
| `ridge` | `sae_affine_basis_mknn_gpu.py` | **Primary** — dense Ridge \(W,b\) |
| `lasso` | `sae_affine_lasso_basis_mknn_gpu.py` | Sparse / L1 maps (weaker here) |
| `eigenbasis` | `sae_lasso_eigenbasis_mknn_gpu.py` | Singular charts of \(W\) + controls |

## Data

Primary images / metadata:

- [Smith42/galaxies](https://huggingface.co/datasets/Smith42/galaxies) (DESI Legacy Survey galaxy cutouts; use revision `v2.0` where applicable)

Related embedding / project resources:

- [Smith42/galaxies_embeddings](https://huggingface.co/datasets/Smith42/galaxies_embeddings) (AstroPT embeddings aligned to the same rows)
- [UniverseTBD/platonic-universe](https://github.com/UniverseTBD/platonic-universe) (cross-model embedding export / mKNN tooling used to produce the ViT / DINOv3 / … parquets under `data_hf/`)

These runners expect **row-aligned** embedding parquets (same index = same object), typically under `$PLATONIC_ROOT/data_hf/`, plus pretrained TopK SAE checkpoints under `$PLATONIC_ROOT/outputs/sae/`.

## Recommended system

| | Minimum (Ridge smoke, `max-n` ≤ 2k) | Recommended (full Ridge / suite, `max-n` = 16k) |
|---|---|---|
| GPU | CUDA GPU, ≥ 8 GB VRAM | CUDA GPU, **≥ 16–24 GB** VRAM (e.g. RTX 3090 / 4090 / A5000-class) |
| Driver / runtime | Recent NVIDIA driver + CUDA-capable PyTorch | Same |
| CPU / RAM | 8+ cores, ≥ 32 GB RAM | 16+ cores, **≥ 64 GB** RAM (parquet load + sklearn Ridge on \(F{=}2048\) codes) |
| Disk | ≥ 20 GB free for one pair + SAE | ≥ 100 GB if keeping multiple surveys / SAE runs |
| OS | Linux x86_64 | Linux x86_64 |

Notes:

- Scripts default to `--device cuda`. CPU is not supported for the GPU runners.
- Ridge at \(n{=}16384\), \(F{=}2048\) is dominated by code encoding + a dense multi-output Ridge; Lasso / eigenbasis (FISTA + rank sweeps) need more wall-clock time on the same GPU.
- Set `PLATONIC_ROOT` to your data/outputs tree (or pass `--platonic-root`).

## Requirements

- Python deps: see [`requirements.txt`](requirements.txt) (`torch`, `pyarrow`, `scikit-learn`, `PyYAML`, …).
- Vendored `sae/sae_model.py` ships with this folder so a clean EffDim checkout can import TopKSAE.
- Cross-matched embeddings must have **equal row counts** (positional alignment). Length mismatch errors unless `--allow-truncate`.

### Method notes (review fixes)

- Singular charts (A/C) apply SVD of `W_std` in **standardized** code/embedding space.
- Local Ridge (C) fits on 70% of each ball and evaluates mKNN on the held-out 30%.
- Lasso active-set metrics use **positive** TopK predictions (not `|y|`).
- λ / “Best” selection uses a **validation** split; test is for reporting.
- Optional `--ridge-ref` is only used when meta (n/cols/seed) matches.
- Symmetrized eig of `W` skipped when feature dims differ (non-square `W`).

## Quick start (CLI)

From the EffDim repo root (this worktree or a clone of the `sae-shared-basis` branch), with `PLATONIC_ROOT` pointing at your platonic-universe-style data tree:

```bash
export PLATONIC_ROOT=/path/to/platonic-universe   # data_hf/ + outputs/sae/
source "$PLATONIC_ROOT/.venv/bin/activate"        # or any env with requirements.txt

# list named cross-matched pairs
python experiments/SAE-shared-basis/run_shared_basis.py list

# check parquet + SAE paths
python experiments/SAE-shared-basis/run_shared_basis.py doctor --dataset physics_vit_dino

# run the Ridge shared-basis experiment (recommended)
python experiments/SAE-shared-basis/run_shared_basis.py run \
  --dataset physics_vit_dino \
  --experiment ridge
```

Outputs land under:

```text
$PLATONIC_ROOT/outputs/sae_shared_basis/ridge_<dataset>_n<N>_<sae_tag>/
  results.json
  results.md
```

## Choosing a dataset

Named presets live in [`datasets.yaml`](datasets.yaml). The important one:

```bash
python experiments/SAE-shared-basis/run_shared_basis.py run \
  --dataset physics_vit_dino --experiment ridge --max-n 16384
```

Custom cross-matched pair (same parquet twice is fine for two-column JWST/DESI files):

```bash
python experiments/SAE-shared-basis/run_shared_basis.py run \
  --experiment ridge \
  --parquet1 data_hf/jwst/jwst_vit_base.parquet --col1 vit_base_jwst \
  --parquet2 data_hf/jwst/jwst_dinov3_vitb16.parquet --col2 dinov3_vitb16_jwst \
  --sae1 outputs/sae/.../F2048_k32_seed0 \
  --sae2 outputs/sae/.../F2048_k32_seed0 \
  --max-n 0
```

If you omit `--sae1`/`--sae2`, the CLI infers:

```text
outputs/sae/<parquet_stem>/<column>/<sae_tag>
```

with `--sae-tag` defaulting to the dataset’s tag (usually `F2048_k64_seed0`).

## Experiments

### Ridge (recommended)

```bash
python experiments/SAE-shared-basis/run_shared_basis.py run \
  --dataset physics_vit_dino --experiment ridge \
  --alpha 1.0 --test-size 0.3 --k 10 --seed 0
```

Direct script equivalent:

```bash
python experiments/SAE-shared-basis/sae_affine_basis_mknn_gpu.py \
  --parquet1 ... --col1 ... --parquet2 ... --col2 ... \
  --sae1 ... --sae2 ... \
  --max-n 16384 --output-dir outputs/sae_affine_basis/my_run
```

### Lasso / eigenbasis follow-ups

```bash
python experiments/SAE-shared-basis/run_shared_basis.py run \
  --dataset physics_vit_dino --experiment lasso

python experiments/SAE-shared-basis/run_shared_basis.py run \
  --dataset physics_vit_dino --experiment eigenbasis --skip-c
```

Forward extra script flags after `--`:

```bash
python experiments/SAE-shared-basis/run_shared_basis.py run \
  --dataset physics_vit_dino --experiment eigenbasis -- \
  --ranks 64 128 256 --fista-steps 400
```

## Interpreting results

In `results.md` / `results.json`:

- **Code prediction** — cosine / Jaccard of mapped codes vs true basis codes on the test split (R² can look awful on sparse TopK targets; cosine/Jaccard are more informative).
- **mKNN table** — compare `shared_*_basis_idf` to `dense_cosine` and `sae_idf_cosine`.
- Best Physics reference: `shared_dino_basis_idf ≈ 0.22`.

## Adding a dataset

Edit `datasets.yaml`:

```yaml
my_pair:
  description: "..."
  parquet1: data_hf/.../a.parquet
  col1: embedding_col_a
  parquet2: data_hf/.../b.parquet   # may equal parquet1
  col2: embedding_col_b
  sae_tag: F2048_k64_seed0
  # or explicit:
  # sae1: outputs/sae/...
  # sae2: outputs/sae/...
  default_max_n: 16384
```

Then `list` / `doctor` / `run --dataset my_pair`.
