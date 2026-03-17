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

NUM_CHAINS=${1:-6}

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

# # 9. Base H8, shared t-emb, MSE 0.1 + cosine 0.1, same timestep (2x H200)
# echo ">>> V2-9: Same timestep MSE 0.1 + cosine 0.1 (2x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200/v2_mse01_cos01_same_t.yaml "${NUM_CHAINS}" 2
# echo ""

# ---- V4: from v2_finetune_no_hidden/1540000.pt ----
# echo ">>> V4-1: Base H8, MSE 0.2 — from ft-1540k (2x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_base_mse02.yaml "${NUM_CHAINS}" 1
# echo ""

# echo ">>> V4-2: Base H16, MSE 0.2 — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_base_h16_mse02.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-3: MSE 0.1 + cosine 0.01, same_t — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_same_t.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-4: MSE 0.1 + cosine 0.01 — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-5: Base H16, MSE 0.2, merged — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_base_h16_mse02_merged.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-6: MSE 0.1 + cosine 0.01, merged — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_merged.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-7: Base H8, MSE 0.2, noisy_img_encode — from ft-1540k (2x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_base_mse02_noisy_enc.yaml "${NUM_CHAINS}" 1
# echo ""

# echo ">>> V4-8: MSE 0.1 + cosine 0.01, merged + noisy_img_encode — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_merged_noisy_enc.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-9: MSE 0.1 + cosine 0.01, noisy_img_encode (3-pass) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">> V4-10: MSE 0.02 + cosine 0.005, merged + noisy_img_encode — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_merged_noisy_enc.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-11: MSE 0.1 + cosine 0.01, merged + noisy_enc + curriculum — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_merged_noisy_enc_curriculum.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-12: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-13: MSE 0.02 + cosine 0.005, 3-pass + curriculum — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_curriculum.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-14: MSE 0.02 + cosine 0.005, 3-pass + strong_shift curriculum — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_curriculum_strong_shift.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-15: MSE 0.02 + cosine 0.005, merged + curriculum — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_merged_curriculum.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-16: MSE 0.02 + cosine 0.005, merged + noisy_enc + curriculum — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_merged_noisy_enc_curriculum.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-17: MSE 0.02 + cosine 0.005, 3-pass + curriculum 5 to 0.5 at 40K — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_curriculum_5_05_40k.yaml "${NUM_CHAINS}" 2
# echo ""



# echo ">>> V4-18: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum + hgd_scale=4 — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_hgd_scale_4.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-19: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum + hgd_scale=2 — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_hgd_scale_2.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-20: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum + hgs_scale=1.5 — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_hgs_scale_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-21: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum + hgs_scale=2 — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_hgs_scale_2.yaml "${NUM_CHAINS}" 2
# echo ""

echo ">>> V4-22: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum + encode_mode_emb — from ft-1540k (4x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_enc_mode_emb.yaml "${NUM_CHAINS}" 2
echo ""

# echo ">>> V4-23: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum + encode_mode_emb + hgd_scale=4 — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_enc_mode_hgd_scale_4.yaml "${NUM_CHAINS}" 2
# echo "" CANCELLED FOR NOW

echo ">>> V4-24: MSE 0.02 + cosine 0.005, 3-pass + curriculum + encode_mode_emb (no noisy enc) — from ft-1540k (4x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_curriculum_enc_mode_emb.yaml "${NUM_CHAINS}" 2
echo ""

echo ">>> V4-25: MSE 0.02 + cosine 0.005, merged + noisy_enc — from ft-1540k (4x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_merged_noisy_enc.yaml "${NUM_CHAINS}" 2
echo ""

echo ">>> V4-26: MSE 0.02 + cosine 0.005, merged + noisy_enc + curriculum — from ft-1540k (4x H200)"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_merged_noisy_enc_curriculum.yaml "${NUM_CHAINS}" 2
echo ""

echo "============================================="
echo "  All V2/V4 experiments submitted!"
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="

