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

# echo ">>> V4-22: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum + encode_mode_emb — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_enc_mode_emb.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-23: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum + encode_mode_emb + hgd_scale=4 — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_enc_mode_hgd_scale_4.yaml "${NUM_CHAINS}" 2
# echo "" CANCELLED FOR NOW

# echo ">>> V4-24: MSE 0.02 + cosine 0.005, 3-pass + curriculum + encode_mode_emb (no noisy enc) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_curriculum_enc_mode_emb.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-25: MSE 0.02 + cosine 0.005, merged + noisy_enc — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_merged_noisy_enc.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-26: MSE 0.02 + cosine 0.005, merged + noisy_enc + curriculum — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_merged_noisy_enc_curriculum.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-27: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum + hidden_guidance_1.5 — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_cfg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-28: MSE 0.1 + cosine 0.01, 3-pass + noisy_enc + curriculum + hgd_scale_4 + NO hidden denoise loss — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_hgd_scale_4_no_hloss.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-29: MSE 0.1 + cosine 0.01, 3-pass + clean enc + curriculum + hgd_scale_4 — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_curriculum_hgd_scale_4.yaml "${NUM_CHAINS}" 2
# echo ""

# ---- V4.1: consistent hidden flow (Pass 1 & 3 share noise) ----

# echo ">>> V4.1-1: MSE 0.1 + cosine 0.01, curriculum + hgd_scale_4 + repg_1.5 (consistent flow) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_1_mse01_cos001_noisy_enc_curriculum_hgd_scale_4_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4.1-2: MSE 0.1 + cosine 0.01, curriculum + repg_1.5 (consistent flow rerun) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_1_mse01_cos001_noisy_enc_curriculum_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4.1-3: MSE 0.1 + cosine 0.01, base curriculum (consistent flow rerun) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_1_mse01_cos001_noisy_enc_curriculum.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4.1-4: cosine 0.1 only, curriculum + repg_1.5 (consistent flow) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_1_cos01_noisy_enc_curriculum_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4.1-5: MSE 0.1 only, curriculum + repg_1.5 (consistent flow) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_1_mse01_noisy_enc_curriculum_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4.1-6: weak MSE 0.01 only, curriculum + repg_1.5 (consistent flow) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_1_mse001_noisy_enc_curriculum_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4.1-7: MSE 0.1 + cosine 0.01, slow curriculum (60K warmup) + repg_1.5 (consistent flow) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_1_mse01_cos001_noisy_enc_curriculum_slow_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4.1-8: NO hidden loss (weights=0), curriculum + hgd_scale_4 (consistent flow, correct no-hloss) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_1_noisy_enc_curriculum_hgd_scale_4_no_hloss.yaml "${NUM_CHAINS}" 2
# echo ""

# ---- V4 R3: Flow reuse for Pass 3, Resample for Pass 2 ----

# echo ">>> V4-R3-1: MSE 0.1 + cosine 0.01, curriculum (reuse p3, resample p2) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_r3.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-R3-2: NO hidden loss, curriculum + hgd_scale_4 (reuse p3, resample p2) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_noisy_enc_curriculum_hgd_scale_4_no_hloss_r3.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-R3-3: MSE 0.1 + cosine 0.01, curriculum + hgd_scale_4 + repg_1.5 (reuse p3, resample p2) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_hgd_scale_4_repg_1p5_r3.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-R3-4: MSE 0.1 only, curriculum + repg_1.5 (reuse p3, resample p2) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_noisy_enc_curriculum_repg_1p5_r3.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-R3-5: MSE 0.1 + cosine 0.01, curriculum + repg_1.5 (reuse p3, resample p2) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_repg_1p5_r3.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-R3-6: weak MSE 0.01 only, curriculum + repg_1.5 (reuse p3, resample p2) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_curriculum_repg_1p5_r3.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-R3-7: NO hidden loss, curriculum + repg_1.5 (reuse p3, resample p2, zero pass 3 compute) — from ft-1540k (4x H200)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_noisy_enc_curriculum_repg_1p5_no_hloss_r3.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> Separate Encoder, NO dropout, NO noisy_enc, default config base"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_clean_enc_curriculum_cfg_1p5_sep_enc.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> Separate Encoder, NO dropout, default config base"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_cfg_1p5_sep_enc.yaml "${NUM_CHAINS}" 2
# echo ""



# echo "============================================="
# echo "  E-Series: Encoding Ablation Experiments"
# echo "============================================="

# echo ">>> E1: Clean encoding, drop 30% to pure noise, no Pass 3, no rep guidance"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/e1_clean_enc_drop03_no_p3.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> E2: Noisy encoding, drop 30% to pure noise, no Pass 3, rep guidance 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/e2_noisy_enc_drop03_no_p3.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> E3: Noisy encoding, drop 30% to pure noise, WITH Pass 3, rep guidance 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/e3_noisy_enc_drop03_p3.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> E4: Noisy encoding, drop 30%, WITH Pass 3, SEPARATE parameter encoder"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/e4_noisy_enc_drop03_p3_sep_enc.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> E4.1: Clean encoding, drop 30%, WITH Pass 3, SEPARATE parameter encoder"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/e4_clean_enc_drop03_p3_sep_enc.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: MSE 0.001 weak only, curriculum + repg_1.5 (resample p2 and p3)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_curriculum_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: MSE 0.001 weak only, curriculum + hgd_scale=4 + repg_1.5 (resample p2 and p3)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_curriculum_hgd_scale_4_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: MSE 0.0001 ultra weak only, curriculum + repg_1.5 (resample p2 and p3)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_curriculum_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: NO hidden loss, curriculum + repg_1.5 (resample p2 and p3)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_noisy_enc_curriculum_repg_1p5_no_hloss.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: MSE 0.01 + COS 0.01, curriculum + repg_1.5 (resample p2 and p3)"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse001_cos001_noisy_enc_curriculum_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: No Curriculum, MSE 0.01, Shift 1.0"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: No Curriculum, MSE 0.01, Shift 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: No Curriculum, MSE 0.001, Shift 1.0"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: No Curriculum, MSE 0.001, Shift 1.5 MERGED PASSES"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5_merged.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: No Curriculum, No Hidden Loss, Shift 1.0, repg 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_noisy_enc_nocurr_shift1_repg_1p5_no_hloss.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: No Curriculum, MSE 0.0001, Shift 1.5, repg 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> V4-NEW: No Curriculum, MSE 0.0001, Shift 0.5, repg 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift0p5_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# ---- T-shift sweep (higher shifts) ----

# echo ">>> T-shift 2.0: No Curriculum, MSE 0.0001, Shift 2.0, repg 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift2_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> T-shift 3.0: No Curriculum, MSE 0.0001, Shift 3.0, repg 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift3_repg_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# # ---- Repg + HGD interaction search ----

# echo ">>> Repg 1.2 (no HGD): No Curriculum, MSE 0.0001, Shift 1.5, repg 1.2"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p2.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> Repg 2.0 (no HGD): No Curriculum, MSE 0.0001, Shift 1.5, repg 2.0"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_2.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> Repg 1.5 + HGD 1.5: No Curriculum, MSE 0.0001, Shift 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> Repg 1.5 + HGD 2.0: No Curriculum, MSE 0.0001, Shift 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> Repg 2.0 + HGD 1.5: No Curriculum, MSE 0.0001, Shift 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_2_hgd_1p5.yaml "${NUM_CHAINS}" 2
# echo ""

# ---- HGD-only (no training-time repg) — for test-time guidance compatibility ----

# echo ">>> No repg + HGD 2.0: No Curriculum, MSE 0.0001, Shift 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_2.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> No repg + HGD 4.0: No Curriculum, MSE 0.0001, Shift 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_4.yaml "${NUM_CHAINS}" 2
# echo ""

# echo ">>> No repg + HGD 5.0: No Curriculum, MSE 0.0001, Shift 1.5"
# bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5.yaml "${NUM_CHAINS}" 2
# echo ""

# ---- Hidden-dropout + sync class-dropout FT (continue from 20K → +40K = 60K total) ----
# Tests fix for OOD-unconditional / class-leak via hidden tokens. Both branches eval'd
# (training-repg may interact differently with hidden dropout than no-repg).

echo ">>> Repg 1.5 + HGD 2.0 + hidden_dropout 0.1 + sync class drop — FT from 20K"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2_hdrop0p1_sync_from20k.yaml "${NUM_CHAINS}" 2
echo ""

echo ">>> No repg + HGD 5.0 + hidden_dropout 0.1 + sync class drop — FT from 20K"
bash run_train_slurm_h200.sh configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5_hdrop0p1_sync_from20k.yaml "${NUM_CHAINS}" 2
echo ""

echo "============================================="
echo "  All V2/V4/V4.1 experiments submitted!"
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="



