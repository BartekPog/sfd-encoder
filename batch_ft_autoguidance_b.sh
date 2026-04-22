#!/bin/bash
# =============================================================================
# batch_ft_autoguidance_b.sh
#
# 100K-step FT of the B-sized autoguidance model on our distribution, using
# LR warmup 8K (analogous to 1p0_finetune_no_hidden_warm8k).
# Init from outputs/train/sfd_autoguidance_b/checkpoints/0070000.pt (EMA only —
# train.py handles 'ema'-only init by copying EMA into model weights).
# =============================================================================

set -euo pipefail

NUM_CHAINS=${1:-5}
NUM_GPUS=${NUM_GPUS:-1}

echo "============================================="
echo "  B autoguidance FT (100K, warm8k) on our distribution"
echo "  Chains: ${NUM_CHAINS} | GPUs: ${NUM_GPUS}"
echo "============================================="

bash run_train_slurm_h200.sh configs/sfd/autoguidance_b/finetune_no_hidden_warm8k.yaml "${NUM_CHAINS}" "${NUM_GPUS}"

echo "Monitor with: squeue -u \$USER"
