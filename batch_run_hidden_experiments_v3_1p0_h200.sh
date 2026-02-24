#!/bin/bash
# =============================================================================
# batch_run_hidden_experiments_v3_1p0_h200.sh — V3 experiments: 1p0B on H200
#
# All models initialized from the 4M pretrained LightningDiT-1p0B checkpoint.
# Scales the best hidden-token approach from V2 (MSE 0.2) to the 1p0B model.
#
# Usage:
#   bash batch_run_hidden_experiments_v3_1p0_h200.sh [num_chains]
#
# Experiments:
#   1. Standard fine-tune (no hidden) for 200K steps            — 4x H200
#   2. Base H8, shared t-emb, MSE 0.2  (best V2 config)        — 4x H200
#   3. Base H8, shared t-emb, MSE 0.1, no cosine               — 4x H200
# =============================================================================

set -euo pipefail

NUM_CHAINS=${1:-8}

echo "============================================="
echo "  V3 Experiments: 1p0B on H200"
echo "  Chains per experiment: ${NUM_CHAINS}"
echo "============================================="
echo ""

# 1. Standard fine-tune (no hidden) — 4x H200
echo ">>> V3-1: Standard 1p0B fine-tune, no hidden (2x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_1p0_h200/v3_finetune_no_hidden.yaml "${NUM_CHAINS}" 2
echo ""

# # 2. Base H8, shared t-emb, MSE 0.2 (best from V2)
# echo ">>> V3-2: 1p0B H8, shared t-emb, MSE 0.2 (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_1p0_h200/v3_base_mse02.yaml "${NUM_CHAINS}" 6
# echo ""

# # 3. Base H8, shared t-emb, MSE-only 0.1
# echo ">>> V3-3: 1p0B H8, shared t-emb, MSE 0.1, no cosine (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_1p0_h200/v3_base_mse01.yaml "${NUM_CHAINS}" 4
# echo ""

echo "============================================="
echo "  All V3 (1p0B) experiments submitted!"
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
