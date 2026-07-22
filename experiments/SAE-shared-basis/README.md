# SAE shared-basis experiments

Cross-model alignment by fitting an **affine map between TopK SAE codes**, so one model’s codes are expressed in the other’s SAE feature basis:

\[
C_{\text{basis}} \approx C_{\text{other}} W + b
\]

Then score held-out **mKNN** between true basis codes and the mapped codes (plus dense / SAE-IDF baselines).

The result that worked well on Physics ViT↔DINOv3 was **Ridge** (not Lasso): ~**0.22** mKNN vs ~0.13 dense / ~0.17 SAE-IDF.

| Experiment | Script | Role |
|---|---|---|
| `ridge` | `sae_affine_basis_mknn_gpu.py` | **Primary** — dense Ridge \(W,b\) |
| `lasso` | `sae_affine_lasso_basis_mknn_gpu.py` | Sparse / L1 maps (weaker here) |
| `eigenbasis` | `sae_lasso_eigenbasis_mknn_gpu.py` | Singular charts of \(W\) + controls |

## Requirements

- Python deps: see [`requirements.txt`](requirements.txt) (`torch`, `pyarrow`, `scikit-learn`, `PyYAML`, …).
- Vendored `sae/sae_model.py` ships with this folder so a clean EffDim checkout can import TopKSAE.
- Data/checkpoints live under `PLATONIC_ROOT` (default `~/platonic-universe` or `/home/angus/platonic-universe`). Override with `--platonic-root` / env `PLATONIC_ROOT`.
- Cross-matched embeddings must have **equal row counts** (positional alignment). Length mismatch errors unless `--allow-truncate`.
- Pretrained TopK SAEs for both sides under `outputs/sae/...`.

### Review fixes (methods)

- Singular charts (A/C) apply SVD of `W_std` in **standardized** code/embedding space.
- Local Ridge (C) fits on 70% of each ball and evaluates mKNN on the held-out 30%.
- Lasso active-set metrics use **positive** TopK predictions (not `|y|`).
- λ / “Best” selection uses a **validation** split; test is for reporting.
- Optional `--ridge-ref` is only used when meta (n/cols/seed) matches.
- Symmetrized eig of `W` skipped when feature dims differ (non-square `W`).

## Quick start (CLI)

From this worktree (or with paths adjusted):

```bash
ssh -F /dev/null -i ~/.ssh/id_ed25519_cursor -o IdentitiesOnly=yes angus@100.97.36.119
source ~/platonic-universe/.venv/bin/activate
cd /path/to/EffDim-worktrees/SAE-shared-basis   # or sync this folder onto the host

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

## Syncing this folder to the GPU host

The runners expect SAE helpers at `~/platonic-universe/experiments/sae/`. Sync the EffDim worktree scripts as needed:

```bash
rsync -e 'ssh -F /dev/null -i ~/.ssh/id_ed25519_cursor -o IdentitiesOnly=yes' -av \
  experiments/SAE-shared-basis/ \
  angus@100.97.36.119:~/platonic-universe/experiments/SAE-shared-basis/
```

Then on the host:

```bash
cd ~/platonic-universe
source .venv/bin/activate
python experiments/SAE-shared-basis/run_shared_basis.py run --dataset physics_vit_dino
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
