#!/bin/bash
# =============================================================================
# batch_run_hidden_experiments_b_h200.sh — Launch B-size experiments on H200 cluster
#
# Usage:
#   bash batch_run_hidden_experiments_b_h200.sh [num_chains]
#
# Experiments (B-size, H200):
#   1B. HiddenLightningDiT-B/1 H8 from scratch        — 4x H200, bs=256, no accum
#   2B. Standard LightningDiT-B/1 fine-tune from 120K  — 2x H200, bs=256, no accum
#   3B. HiddenLightningDiT-B/1 H8 from 120K pretrained — 4x H200, bs=256, no accum
#
# All experiments run with NO gradient accumulation on H200.
# =============================================================================

set -euo pipefail

NUM_CHAINS=8 # ${1:-12}

echo "============================================="
echo "  Launching B-size experiments on H200"
echo "  Chains per experiment: ${NUM_CHAINS}"
echo "============================================="
echo ""

# Experiment 1B: Hidden from scratch (2x H200)
echo ">>> Experiment 1B: Hidden from scratch (2x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/exp1_hidden_scratch.yaml "${NUM_CHAINS}" 2
echo ""

# # Experiment 2B: Standard fine-tune (1x H200)
# echo ">>> Experiment 2B: Standard fine-tune (1x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/exp2_standard_finetune.yaml "${NUM_CHAINS}" 1
# echo ""

# Experiment 3B: Hidden from pretrained (2x H200)
echo ">>> Experiment 3B: Hidden from pretrained (2x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained.yaml "${NUM_CHAINS}" 2
echo ""

# Experiment 3B: Hidden from pretrained (2x H200)
echo ">>> Experiment 3B: Hidden from pretrained (2x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_h1.yaml "${NUM_CHAINS}" 2
echo ""

# Experiment 3B: Hidden from pretrained (2x H200)
echo ">>> Experiment 3B: Hidden from pretrained (2x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_h16.yaml "${NUM_CHAINS}" 2
echo ""

# Experiment 3B: Hidden from pretrained (2x H200)
echo ">>> Experiment 3B: Hidden from pretrained (2x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_separate_embedder.yaml "${NUM_CHAINS}" 2
echo ""

# Experiment 3B: Hidden from pretrained (2x H200)
echo ">>> Experiment 3B: Hidden from pretrained (2x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_hidden_pos_encoding.yaml "${NUM_CHAINS}" 2
echo ""

#Experiment 3B: Hidden from pretrained (2x H200)
echo ">>> Experiment 3B: Hidden from pretrained (2x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_weak_h_loss.yaml "${NUM_CHAINS}" 2
echo ""



echo "============================================="
echo "  All B-size H200 experiments submitted!"
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
