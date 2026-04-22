#!/bin/bash
#SBATCH --job-name diag_ft_drift
#SBATCH --output job_outputs/diagnose_ft_drift.o%J
#SBATCH --time 00-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --gres gpu:h200:1

set -euo pipefail

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Diagnose FT drift: num_batches=32 batch_size=16"
echo "  config:   configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden_warm8k.yaml"
echo "  src_ckpt: outputs/train/sfd_1p0/checkpoints/4000000_full.pt"
echo "  ft_ckpt:  outputs/train/1p0_finetune_no_hidden_warm8k/checkpoints/0060000.pt"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

STATS_FLAG=""
if [ -n "" ]; then
    STATS_FLAG="--stats_dir_override "
fi

python diagnose_ft_drift.py \
    --config     configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden_warm8k.yaml \
    --src_ckpt   outputs/train/sfd_1p0/checkpoints/4000000_full.pt \
    --ft_ckpt    outputs/train/1p0_finetune_no_hidden_warm8k/checkpoints/0060000.pt \
    --num_batches 32 \
    --batch_size  16 ${STATS_FLAG}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
