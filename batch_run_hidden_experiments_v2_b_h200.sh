#!/bin/bash
# =============================================================================
# batch_run_hidden_experiments_v2_b_h200.sh — V2 experiments on H200 cluster
#
# All models initialized from the 400K fine-tuned LightningDiT-B checkpoint.
# Changes from V1: per-token adaLN conditioning, hidden cosine loss support.
#
# Usage:
#   bash batch_run_hidden_experiments_v2_b_h200.sh [num_chains]
#
# Experiments:
#   1. Standard fine-tune (no hidden) for 200K more steps         — 1x H200
#   2. Base H8, shared t-emb, MSE 0.2                            — 2x H200
#   3. Base H16, shared t-emb, MSE 0.2                           — 2x H200
#   4. Separate patch embedder, shared t-emb, MSE 0.2            — 2x H200
#   5. Base H8, shared t-emb, cosine-only 0.2                    — 2x H200
#   6. Base H8, shared t-emb, MSE 0.2 + cosine 0.2              — 2x H200
#   7. Base H8, shared t-emb, MSE 0.1 + cosine 0.1              — 2x H200
#   8. Base H8, non-shared t-emb, MSE 0.1 + cosine 0.1          — 2x H200
# =============================================================================

set -euo pipefail

NUM_CHAINS=${1:-5}

echo "============================================="
echo "  V2 Experiments on H200"
echo "  Chains per experiment: ${NUM_CHAINS}"
echo "============================================="
echo ""

# # 1. Standard fine-tune (no hidden) — 1x H200
# echo ">>> V2-1: Standard fine-tune, no hidden (1x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/v2_finetune_no_hidden.yaml "${NUM_CHAINS}" 1
# echo ""

# # 2. Base H8, shared t-emb, MSE 0.2
# echo ">>> V2-2: Base H8, shared t-emb, MSE 0.2 (2x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/v2_base_mse02.yaml "${NUM_CHAINS}" 2
# echo ""

# # 3. Base H16, shared t-emb, MSE 0.2
# echo ">>> V2-3: Base H16, shared t-emb, MSE 0.2 (2x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/v2_base_h16_mse02.yaml "${NUM_CHAINS}" 2
# echo ""

# # 4. Separate patch embedder, shared t-emb, MSE 0.2
# echo ">>> V2-4: Separate patch embedder, MSE 0.2 (2x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/v2_sep_embedder_mse02.yaml "${NUM_CHAINS}" 2
# echo ""

# # 5. Base H8, shared t-emb, cosine-only 0.2
# echo ">>> V2-5: Cosine-only 0.2 (2x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/v2_cos02.yaml "${NUM_CHAINS}" 2
# echo ""

# # 6. Base H8, shared t-emb, MSE 0.2 + cosine 0.2
# echo ">>> V2-6: MSE 0.2 + cosine 0.2 (2x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/v2_mse02_cos02.yaml "${NUM_CHAINS}" 2
# echo ""

# # 7. Base H8, shared t-emb, MSE 0.1 + cosine 0.1
# echo ">>> V2-7: MSE 0.1 + cosine 0.1 (2x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/v2_mse01_cos01.yaml "${NUM_CHAINS}" 2
# echo ""

# # 8. Base H8, non-shared t-emb, MSE 0.1 + cosine 0.1
# echo ">>> V2-8: Non-shared t-emb, MSE 0.1 + cosine 0.1 (2x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/v2_nonshr_temb_mse01_cos01.yaml "${NUM_CHAINS}" 2
# echo ""

# 9. Base H8, shared t-emb, MSE 0.1 + cosine 0.1, same timestep (2x H200)
echo ">>> V2-9: Same timestep MSE 0.1 + cosine 0.1 (2x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/v2_mse01_cos01_same_t.yaml "${NUM_CHAINS}" 2
echo ""

echo "============================================="
echo "  All V2 experiments submitted!"
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
