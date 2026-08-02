#!/bin/bash
#SBATCH --job-name=curvature_probe
#SBATCH --output=logs/curvature_probe_%j.log
#SBATCH --error=logs/curvature_probe_%j.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# Minimum GPU class: V100 16 GB or better (A100 40 GB preferred for SAE training speed)
# To restrict to A100:  #SBATCH --constraint=a100
# To restrict to V100:  #SBATCH --constraint=v100
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via environment or edit here
# ---------------------------------------------------------------------------
PLATONIC_ROOT="${PLATONIC_ROOT:-$HOME/platonic-universe}"
MAX_N="${MAX_N:-16384}"
PROBES="${PROBES:-independent}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

# SAE hyperparameters
SAE_FEATURE_DIM="${SAE_FEATURE_DIM:-2048}"
SAE_TOPK="${SAE_TOPK:-64}"
SAE_EPOCHS="${SAE_EPOCHS:-50}"

# SAE curvature neighborhood
K_CURV="${K_CURV:-50}"

# Multi-scale PCA curvature neighborhoods
K_SMALL="${K_SMALL:-30}"
K_LARGE="${K_LARGE:-200}"

# Output roots (relative to PLATONIC_ROOT)
SAE_OUT="outputs/sae_curvature_probe"
MS_OUT="outputs/multiscale_curvature_probe"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
# Load necessary modules — uncomment and adjust for your HPC
# module load cuda/12.1
# module load miniconda3

eval "$(conda shell.bash hook)"
conda activate effdim

mkdir -p logs "${PLATONIC_ROOT}/${SAE_OUT}" "${PLATONIC_ROOT}/${MS_OUT}"

echo "============================================================"
echo " Curvature ↔ Probe Correlation Experiments"
echo " PLATONIC_ROOT : ${PLATONIC_ROOT}"
echo " MAX_N         : ${MAX_N}"
echo " PROBES        : ${PROBES}"
echo " Device        : ${DEVICE}"
echo " SLURM_JOB_ID  : ${SLURM_JOB_ID:-local}"
echo " Started       : $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Experiment 1: SAE-based curvature
# ---------------------------------------------------------------------------
echo ""
echo ">>> [1/2] SAE Curvature Experiment"
python experiments/physics-probe-subspace/sae_curvature_probe.py \
    --platonic-root      "${PLATONIC_ROOT}" \
    --model-a            vit_base \
    --model-b            dinov3_vitb16 \
    --dataset            physics \
    --max-n              "${MAX_N}" \
    --probes             "${PROBES}" \
    --k-curv             "${K_CURV}" \
    --sae-feature-dim    "${SAE_FEATURE_DIM}" \
    --sae-topk           "${SAE_TOPK}" \
    --sae-epochs         "${SAE_EPOCHS}" \
    --device             "${DEVICE}" \
    --seed               "${SEED}" \
    --output-dir         "${PLATONIC_ROOT}/${SAE_OUT}"

echo ""
echo "SAE experiment complete. Results at: ${PLATONIC_ROOT}/${SAE_OUT}/results.md"

# ---------------------------------------------------------------------------
# Experiment 2: Multi-scale PCA residual curvature
# ---------------------------------------------------------------------------
echo ""
echo ">>> [2/2] Multi-Scale PCA Residual Curvature Experiment"
python experiments/physics-probe-subspace/multiscale_curvature_probe.py \
    --platonic-root  "${PLATONIC_ROOT}" \
    --model-a        vit_base \
    --model-b        dinov3_vitb16 \
    --dataset        physics \
    --max-n          "${MAX_N}" \
    --probes         "${PROBES}" \
    --k-small        "${K_SMALL}" \
    --k-large        "${K_LARGE}" \
    --seed           "${SEED}" \
    --output-dir     "${PLATONIC_ROOT}/${MS_OUT}"

echo ""
echo "Multi-scale experiment complete. Results at: ${PLATONIC_ROOT}/${MS_OUT}/results.md"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " All experiments finished: $(date)"
echo "============================================================"
