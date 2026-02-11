#!/bin/bash
# =============================================================================
# batch_run_hidden_experiments.sh — Launch all 3 hidden token experiments
#
# Usage:
#   bash batch_run_hidden_experiments.sh [num_chains]
#
# Arguments:
#   num_chains — number of chained 4h SLURM jobs per experiment (default: 6 ≈ 24h)
#
# Experiments:
#   1. HiddenLightningDiT-XL/1 H8 from scratch (400K steps)
#   2. Standard LightningDiT-XL/1 fine-tune from 400K ckpt (+400K steps)
#   3. HiddenLightningDiT-XL/1 H8 initialized from 400K ckpt (+400K steps)
# =============================================================================

set -euo pipefail

# 6 chains are 24h 
NUM_CHAINS=12   #${1:-6}

echo "============================================="
echo "  Launching all hidden token experiments"
echo "  Chains per experiment: ${NUM_CHAINS}"
echo "============================================="
echo ""

# Experiment 1: Hidden from scratch
echo ">>> Experiment 1: Hidden from scratch"
bash run_train_slurm.sh configs/sfd/hidden_xl/exp1_hidden_scratch.yaml "${NUM_CHAINS}"
echo ""

# Experiment 2: Standard fine-tune (baseline comparison)
echo ">>> Experiment 2: Standard fine-tune"
bash run_train_slurm.sh configs/sfd/hidden_xl/exp2_standard_finetune.yaml "${NUM_CHAINS}"
echo ""

# Experiment 3: Hidden from pretrained
echo ">>> Experiment 3: Hidden from pretrained"
bash run_train_slurm.sh configs/sfd/hidden_xl/exp3_hidden_from_pretrained.yaml "${NUM_CHAINS}"
echo ""

echo "============================================="
echo "  All experiments submitted!"
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
