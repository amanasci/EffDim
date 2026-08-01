# Physics Probe Subspace Experiment

## Overview

This experiment tests the hypothesis that foundation model embeddings can be aligned without discarding fine-grained semantic signal by projecting them onto a **Task-Anchored Subspace**. 

Unlike blind whitening (which explodes low-variance background noise) or Autoencoders/PCA (which discard the "soft shell" where physics semantics live), this method explicitly constructs a subspace spanned by the normal vectors of $M$ linear probes trained on ground-truth astronomical properties.

By extracting $M = 50\text{–}100$ independent properties, we span the $d \approx 87$ intrinsic dimensionality of the soft shell, annihilating the unaligned noise dimensions while retaining the geometry necessary for dense retrieval (mKNN).

## Usage

### 1. Run the Probing Pipeline
This script streams labels from `Smith42/galaxies` (v2.0), trains linear probes on the L2-normalized embeddings, orthogonalizes the weight vectors via QR decomposition, and computes mKNN overlap in the resulting projected subspace.

```bash
conda activate effdim
python experiments/physics-probe-subspace/probe_basis_mknn.py \
  --max-n 16384 \
  --device cuda \
  --probes independent \
  --k 10
```

**Arguments:**
- `--probes`: Which set of properties to probe. Options: `independent` (~38 properties), `all` (~44 properties), `default11` (the original small set).

### 2. Analysis
Analyze the outputs (probe $R^2$ and subspace principal angles):

```bash
python experiments/physics-probe-subspace/analyse_probes.py \
  --results outputs/probe_basis/results.json
```

## Structure
- `_common.py`: Shared utilities for streaming labels and computing derived properties.
- `probe_basis_mknn.py`: Main execution pipeline.
- `analyse_probes.py`: Analysis plotting.
- `datasets.yaml`: Dataset configuration.
