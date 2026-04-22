#!/bin/bash
# =============================================================================
# batch_cfg_sweep_hgd_only.sh
#
# CFG sweep on HGD-only B models (no training repg).
#
# From the guidance eval we know:
#   - CFG helps all HGD-only models (the more HGD, the better baseline)
#   - cfg_noise_hidden=true is better than false for HGD-only models
#   - no_repg_hgd_4 @ CFG 1.5 gave 3.53 — not saturated
#
# Sweep design:
#   Models: no_repg_hgd_4, no_repg_hgd_5
#   CFG values: 1.3, 1.5, 2.0, 2.5, 3.0
#   cfg_noise_hidden: true, false
#   All at t_fix=0.88, ckpt=60K
#
# We already have (skip these):
#   hgd_4: cfg 1.1 {true,false}, cfg 1.5 {false}
#   hgd_5: cfg 1.1 {true,false}
#
# New runs: 2 models × 5 cfg × 2 noise_h = 20
#   minus hgd_4 cfg 1.5 false (already done) = 19 jobs
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

NOREPG_HGD4="v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_4"
NOREPG_HGD5="v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5"

echo "============================================="
echo "  CFG sweep on HGD-only B models"
echo "  t_fix=${T_FIX}, ckpt=${CKPT_STEP}"
echo "============================================="
echo ""

SUBMITTED=0

for model in "${NOREPG_HGD4}" "${NOREPG_HGD5}"; do
    short=$(echo "${model}" | sed 's/v4_mse0001_noisy_enc_nocurr_//')
    echo ""
    echo "=== ${short} ==="

    for cfg in 1.3 1.5 2.0 2.5 3.0; do
        for noiseh in false true; do
            # Skip hgd_4 cfg 1.5 false — already done
            if [ "${model}" = "${NOREPG_HGD4}" ] && [ "${cfg}" = "1.5" ] && [ "${noiseh}" = "false" ]; then
                echo "  SKIP (already done): ${short}_cfg${cfg}_noiseh=${noiseh}"
                continue
            fi
            submit_job "${model}" "${cfg}" "${noiseh}" "${short}_cfg${cfg}_noiseh${noiseh}"
            SUBMITTED=$((SUBMITTED + 1))
        done
    done
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} CFG sweep jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
