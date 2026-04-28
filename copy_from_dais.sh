#!/usr/bin/env bash
set -euo pipefail

SRC=/dais/fs/scratch/bpogodzi/hidden-diffusion/sfd-encoder
DST=/viper/ptmp2/bpogodzi/hidden-diffusion/sfd-encoder

PARENT_RUNS=(
    v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5
    v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2
)

echo "==> Copying parent 20k checkpoints"
for run in "${PARENT_RUNS[@]}"; do
    mkdir -p "$DST/outputs/train/$run/checkpoints"
    rsync -ah --info=progress2 \
        "$SRC/outputs/train/$run/checkpoints/0020000.pt" \
        "$DST/outputs/train/$run/checkpoints/0020000.pt"
done

echo "==> Copying FID result files (txt + json)"
rsync -ah --prune-empty-dirs \
    --include='*/' \
    --include='fid_result.txt' \
    --include='fid_result.json' \
    --exclude='*' \
    "$SRC/outputs/inference/" \
    "$DST/outputs/inference/"

echo "==> Done"
