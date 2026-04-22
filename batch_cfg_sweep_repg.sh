#!/bin/bash
# =============================================================================
# batch_cfg_sweep_repg.sh
#
# CFG sweep on the B-sized repg-trained model to check whether CFG becomes
# useless (as observed in the XXL model trained with repg 1.5).
#
# Model: v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5
#
# Existing results at t_fix=0.88, ckpt=60K:
#   cfg=1.0 baseline:        FID ~4.76
#   cfg=1.1 noise_h=false:   FID ~3.81
#   cfg=1.1 noise_h=true:    FID ~4.00
#
# Sweep design:
#   CFG values: 1.3, 1.5, 2.0, 2.5, 3.0
#   cfg_noise_hidden: true, false
#   All at t_fix=0.88, ckpt=60K
#
# Total: 5 cfg × 2 noise_h = 10 jobs
# =============================================================================

set -euo pipefail

CKPT_STEP=60000
T_FIX=0.88
export TIME="00-04:00:00"
export PER_PROC_BATCH_SIZE=1024

CONFIG_DIR="configs/sfd/hidden_b_h200_from_ft"

submit_job() {
    local config=$1 cfg=$2 noiseh=$3 label=$4
    local config_path="${CONFIG_DIR}/${config}.yaml"

    echo "  Submitting: ${label} (cfg=${cfg}, noise_h=${noiseh})"

    EXPERIMENTS_OVERRIDE="${config_path}|${config}" \
    CFG_SCALE="${cfg}" \
    HIDDEN_REP_GUIDANCE="1.0" \
    CFG_NOISE_HIDDEN="${noiseh}" \
    REGROUND_FIXED_ENC_NOISE=true \
    REGROUND_REUSE_ENCODE_FOR_REPG=false \
    T_FIX_VALUES_OVERRIDE="${T_FIX}" \
        bash batch_run_inference_reground_b_h200.sh "${CKPT_STEP}" 2>&1 | grep -E "submitted|SKIP"
}

REPG_MODEL="v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5"

echo "============================================="
echo "  CFG sweep on repg-trained B model"
echo "  Model: ${REPG_MODEL}"
echo "  t_fix=${T_FIX}, ckpt=${CKPT_STEP}"
echo "============================================="
echo ""

SUBMITTED=0

short=$(echo "${REPG_MODEL}" | sed 's/v4_mse0001_noisy_enc_nocurr_//')
echo "=== ${short} ==="

for cfg in 1.3 1.5 2.0 2.5 3.0; do
    for noiseh in false true; do
        submit_job "${REPG_MODEL}" "${cfg}" "${noiseh}" "${short}_cfg${cfg}_noiseh${noiseh}"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} CFG sweep jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
