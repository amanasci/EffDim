#!/bin/bash


# Load necessary modules (uncomment and adjust depending on your HPC's module system)
# module load miniconda3
# module load cuda/11.8

# Initialize conda and activate the environment
eval "$(conda shell.bash hook)"
conda activate effdim

# Ensure outputs go to the right place
export PLATONIC_ROOT=$HOME/platonic-universe
mkdir -p logs

echo "Starting Physics Probe Subspace pipeline..."
python experiments/physics-probe-subspace/probe_basis_mknn.py \
    --max-n 16384 \
    --device cuda \
    --probes independent \
    --k 10 \
    --output-dir $PLATONIC_ROOT/outputs/probe_basis

echo "Generating analysis charts..."
python experiments/physics-probe-subspace/analyse_probes.py \
    --results $PLATONIC_ROOT/outputs/probe_basis/results.json

echo "Done!"
