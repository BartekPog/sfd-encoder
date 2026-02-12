#!/bin/bash
# =============================================================================
# batch_run_hidden_experiments_b.sh — Launch all 3 hidden token experiments (B size)
#
# Usage:
#   bash batch_run_hidden_experiments_b.sh [num_chains]
#
# Arguments:
#   num_chains — number of chained 4h SLURM jobs per experiment (default: 12)
#
# Experiments (B-size variants):
#   1B. HiddenLightningDiT-B/1 H8 from scratch (400K steps)
#   2B. Standard LightningDiT-B/1 fine-tune from 120K ckpt (+400K steps)
#   3B. HiddenLightningDiT-B/1 H8 initialized from 120K ckpt (+400K steps)
#
# B model is ~5x smaller than XL → less grad accumulation needed:
#   Hidden (3 passes): 42/GPU, accum=2 → 252 effective batch
#   Standard (1 pass):  84/GPU, accum=1 → 252 effective batch (no accum overhead)
# =============================================================================

set -euo pipefail

NUM_CHAINS=${1:-12}

echo "============================================="
echo "  Launching all hidden token experiments (B)"
echo "  Chains per experiment: ${NUM_CHAINS}"
echo "============================================="
echo ""

# Experiment 1B: Hidden from scratch
echo ">>> Experiment 1B: Hidden from scratch (B)"
bash run_train_slurm.sh configs/sfd/hidden_b/exp1_hidden_scratch.yaml "${NUM_CHAINS}"
echo ""

# Experiment 2B: Standard fine-tune (baseline comparison)
echo ">>> Experiment 2B: Standard fine-tune (B)"
bash run_train_slurm.sh configs/sfd/hidden_b/exp2_standard_finetune.yaml "${NUM_CHAINS}"
echo ""

# Experiment 3B: Hidden from pretrained
echo ">>> Experiment 3B: Hidden from pretrained (B)"
bash run_train_slurm.sh configs/sfd/hidden_b/exp3_hidden_from_pretrained.yaml "${NUM_CHAINS}"
echo ""

echo "============================================="
echo "  All B-size experiments submitted!"
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
