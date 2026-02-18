#!/bin/bash
# =============================================================================
# batch_run_hidden_experiments_b_h200.sh — Launch B-size experiments on DAIS cluster
#
# Usage:
#   bash batch_run_hidden_experiments_xl_dais.sh [num_chains]
#
# Experiments (B-size, DAIS):
#   1B. HiddenLightningDiT-B/1 H8 from scratch        — 4x DAIS, bs=256, no accum
#   2B. Standard LightningDiT-B/1 fine-tune from 120K  — 2x DAIS, bs=256, no accum
#   3B. HiddenLightningDiT-B/1 H8 from 120K pretrained — 4x DAIS, bs=256, no accum
#
# All experiments run with NO gradient accumulation on DAIS.
# =============================================================================

set -euo pipefail

NUM_CHAINS=12 # ${1:-12}

echo "============================================="
echo "  Launching XL-size experiments on DAIS"
echo "  Chains per experiment: ${NUM_CHAINS}"
echo "============================================="
echo ""

# Experiment 1B: Hidden from scratch (2x DAIS)
# echo ">>> Experiment 1B: Hidden from scratch (2x DAIS)"
# bash run_train_slurm_b200.sh configs/sfd/hidden_xl_dais/exp1_hidden_scratch.yaml "${NUM_CHAINS}" 2
# echo ""

# Experiment 2B: Standard fine-tune (1x DAIS)
echo ">>> Experiment 2B: Standard fine-tune (1x DAIS)"
bash run_train_slurm_b200.sh configs/sfd/hidden_xl_dais/exp2_standard_finetune.yaml "${NUM_CHAINS}" 1
echo ""

# Experiment 3B: Hidden from pretrained (2x DAIS)
echo ">>> Experiment 3B: Hidden from pretrained (2x DAIS)"
bash run_train_slurm_b200.sh configs/sfd/hidden_xl_dais/exp3_hidden_from_pretrained.yaml "${NUM_CHAINS}" 2
echo ""

# # Experiment 3B: Hidden from pretrained (2x DAIS)
# echo ">>> Experiment 3B: Hidden from pretrained (2x DAIS)"
# bash run_train_slurm_b200.sh configs/sfd/hidden_xl_dais/exp3_hidden_from_pretrained_h1.yaml "${NUM_CHAINS}" 2
# echo ""

# # Experiment 3B: Hidden from pretrained (2x DAIS)
# echo ">>> Experiment 3B: Hidden from pretrained (2x DAIS)"
# bash run_train_slurm_b200.sh configs/sfd/hidden_xl_dais/exp3_hidden_from_pretrained_h16.yaml "${NUM_CHAINS}" 2
# echo ""

# # Experiment 3B: Hidden from pretrained (2x DAIS)
# echo ">>> Experiment 3B: Hidden from pretrained (2x DAIS)"
# bash run_train_slurm_b200.sh configs/sfd/hidden_xl_dais/exp3_hidden_from_pretrained_separate_embedder.yaml "${NUM_CHAINS}" 2
# echo ""

# # Experiment 3B: Hidden from pretrained (2x DAIS)
# echo ">>> Experiment 3B: Hidden from pretrained (2x DAIS)"
# bash run_train_slurm_b200.sh configs/sfd/hidden_xl_dais/exp3_hidden_from_pretrained_hidden_pos_encoding.yaml "${NUM_CHAINS}" 2
# echo ""

# Experiment 3B: Hidden from pretrained (2x DAIS)
# echo ">>> Experiment 3B: Hidden from pretrained (2x DAIS)"
# bash run_train_slurm_b200.sh configs/sfd/hidden_xl_dais/exp3_hidden_from_pretrained_weak_h_loss.yaml "${NUM_CHAINS}" 2
# echo ""



echo "============================================="
echo "  All XL-size DAIS experiments submitted!"
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
