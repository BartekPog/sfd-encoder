#!/bin/bash
# =============================================================================
# batch_cfg_sweep_linear_repg1p2.sh
#
# Linear-schedule CFG sweep for repg_1p2 model (no rep guidance at inference).
#
# Model: v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p2
# CFG values: 1.2, 1.3, 1.4, 1.5, 1.6
# Schedule: linear + sphere clamp
# No hidden rep guidance (repg off)
# Ckpt: 60K
#
# Total: 5 jobs
# =============================================================================

set -euo pipefail

CKPT_STEP=60000
export TIME="00-04:00:00"
export PER_PROC_BATCH_SIZE=1024

MODEL="v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2"
CONFIG_DIR="configs/sfd/hidden_b_h200_from_ft"
CONFIG_PATH="${CONFIG_DIR}/${MODEL}.yaml"

CFG_VALUES=(1.2 1.3 1.4 1.5 1.6)

echo "============================================="
echo "  Linear CFG sweep — ${MODEL}"
echo "  CFGs: ${CFG_VALUES[*]}"
echo "  ckpt=${CKPT_STEP}, no repg"
echo "============================================="
echo ""

SUBMITTED=0

for cfg in "${CFG_VALUES[@]}"; do
    echo "  Submitting: cfg=${cfg}"

    EXPERIMENTS_OVERRIDE="${CONFIG_PATH}|${MODEL}" \
    CFG_SCALE="${cfg}" \
    HIDDEN_REP_GUIDANCE="1.0" \
        bash batch_run_inference_linear_hidden_b_h200_sphereclamp.sh "${CKPT_STEP}" 2>&1 | grep -E "submitted|SKIP"

    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} linear CFG sweep jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
