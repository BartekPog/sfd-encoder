#!/bin/bash
# =============================================================================
# batch_reground_tfix_sweep_repg1p2.sh
#
# Reground t_fix sweep for repg_1p2 model, CFG=1.3, no rep guidance.
# Tests with and without --cfg_noise_hidden (noising hidden tokens for the
# CFG negative pass).
#
# Model: v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p2
# CFG: 1.3
# t_fix values: 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
# cfg_noise_hidden: true, false
# Ckpt: 60K
#
# Total: 6 × 2 = 12 jobs
# =============================================================================

set -euo pipefail

CKPT_STEP=60000
export TIME="00-04:00:00"
export PER_PROC_BATCH_SIZE=1024

MODEL="v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p2"
CONFIG_DIR="configs/sfd/hidden_b_h200_from_ft"
CONFIG_PATH="${CONFIG_DIR}/${MODEL}.yaml"

CFG=1.3
# T_FIX_VALUES=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
T_FIX_VALUES=(0.1 0.2 0.3)

echo "============================================="
echo "  Reground t_fix sweep — ${MODEL}"
echo "  CFG=${CFG}, no repg"
echo "  t_fix: ${T_FIX_VALUES[*]}"
echo "  cfg_noise_hidden: false, true"
echo "  ckpt=${CKPT_STEP}"
echo "============================================="
echo ""

SUBMITTED=0

for t_fix in "${T_FIX_VALUES[@]}"; do
    for noiseh in false true; do
        echo "  Submitting: t_fix=${t_fix}, cfg_noise_hidden=${noiseh}"

        EXPERIMENTS_OVERRIDE="${CONFIG_PATH}|${MODEL}" \
        CFG_SCALE="${CFG}" \
        HIDDEN_REP_GUIDANCE="1.0" \
        CFG_NOISE_HIDDEN="${noiseh}" \
        REGROUND_FIXED_ENC_NOISE=true \
        T_FIX_VALUES_OVERRIDE="${t_fix}" \
            bash batch_run_inference_reground_b_h200.sh "${CKPT_STEP}" 2>&1 | grep -E "submitted|SKIP"

        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} reground t_fix sweep jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
