#!/bin/bash
# QM9 Diffusion Pre-training Launch Script
# Phase 1: Trains the mixed continuous-discrete denoising diffusion model on QM9
#
# Usage:
#   Fresh start:    ./run_train_qm9.sh
#   Resume:         ./run_train_qm9.sh --resume

set -e

cd /home/tianyajun/MARL_for_COFs

# Activate conda environment
source /home/tianyajun/anaconda3/bin/activate env_cof

# Set GPU (A6000 — check nvidia-smi for availability)
export CUDA_VISIBLE_DEVICES=5

# Create directories
mkdir -p symmcd_diffusion/logs symmcd_diffusion/checkpoints

# Parse args
RESUME=""
EXTRA_ARGS=""
if [ "${1}" = "--resume" ]; then
    RESUME="--resume symmcd_diffusion/checkpoints/qm9_latest.pt"
    shift
fi

# Launch training with unbuffered output
python -u symmcd_diffusion/train_qm9.py \
    --epochs 500 \
    --batch_size 64 \
    --lr 1e-4 \
    --grad_accum 2 \
    --use_amp \
    --data_root ./data/qm9 \
    ${RESUME} \
    ${EXTRA_ARGS} \
    2>&1 | tee symmcd_diffusion/logs/train_qm9_$(date +%Y%m%d_%H%M%S).log
