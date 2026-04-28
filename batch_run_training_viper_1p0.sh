#!/bin/bash
# =============================================================================
# batch_run_training_viper.sh — Submit the two target hdrop0p1_sync_from20k
# training runs to SLURM on Viper-GPU (MI300A).
#
# Both configs fine-tune 20k steps → 40k starting from the matched parent
# checkpoint (weight_init inside the YAML). Each run uses
# run_train_slurm_viper.sh which:
#   - asks SLURM for MI300A APUs with the correct constraint / memory caps,
#   - sets ROCm + wandb-offline + XFORMERS_DISABLED env vars,
#   - launches accelerate with a bash -c wrapper so \$SLURM_NODEID expands
#     per-remote-task (otherwise multi-node rendezvous times out).
#
# Usage:
#   bash batch_run_training_viper.sh [num_chains] [num_apus]
#
# Arguments:
#   num_chains  — chained SLURM jobs per experiment (default: 6)
#   num_apus    — APUs per experiment (default: 4 = 2 nodes × 2 APUs)
#
# Examples:
#   bash batch_run_training_viper.sh           # 6 chains, 4 APUs per exp
#   bash batch_run_training_viper.sh 3 2       # 3 chains, 2 APUs per exp
# =============================================================================

set -euo pipefail

NUM_CHAINS=${1:-6}
NUM_APUS=${2:-8}

CONFIGS=(
    configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5_ft_cfgrepg.yaml
)

echo "============================================="
echo "  sfd-encoder training on Viper-GPU (MI300A)"
echo "  Experiments:    ${#CONFIGS[@]}"
echo "  Chains/exp:     ${NUM_CHAINS}"
echo "  APUs/exp:       ${NUM_APUS}"
echo "============================================="
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    EXP_NAME=$(basename "${CONFIG}" .yaml)
    echo ">>> ${EXP_NAME}"
    bash run_train_slurm_viper.sh "${CONFIG}" "${NUM_CHAINS}" "${NUM_APUS}"
    echo ""
done

echo ""
echo "All experiments submitted."
echo "Monitor with: squeue -u \$USER"


 