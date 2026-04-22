#!/bin/bash
# =============================================================================
# batch_t_fix_sweep_hgd5.sh
#
# Check whether t_fix=0.88 is still optimal under guidance for the best
# performing no-repg model (v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5).
#
# Guidance: CFG=1.3, cfg_noise_hidden=true (applies only to reground runs)
# Sweep:    t_fix ∈ {0.70, 0.75, 0.80, 0.85, 0.90} (reground)
#           + 1 linear-schedule run
# Overlay:  hidden_rep_guidance ∈ {1.0, 1.1, 1.3}
#
# Total: (5 t_fix + 1 linear) × 3 repg = 18 jobs
# All at ckpt=60000, Euler 100 steps.
# =============================================================================

set -euo pipefail

CKPT_STEP=60000
export TIME="00-04:00:00"
export PER_PROC_BATCH_SIZE=1024

CONFIG_DIR="configs/sfd/hidden_b_h200_from_ft"
MODEL="v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5"
CONFIG_PATH="${CONFIG_DIR}/${MODEL}.yaml"
CFG=1.3
CFG_NOISE_HIDDEN=true

T_FIX_VALUES=(0.70 0.75 0.80 0.85 0.90)
REPG_VALUES=(1.0 1.1 1.3)

submit_reground() {
    local tfix=$1 repg=$2
    echo "  Reground: t_fix=${tfix}, cfg=${CFG}, noise_h=${CFG_NOISE_HIDDEN}, repg=${repg}"
    EXPERIMENTS_OVERRIDE="${CONFIG_PATH}|${MODEL}" \
    CFG_SCALE="${CFG}" \
    HIDDEN_REP_GUIDANCE="${repg}" \
    CFG_NOISE_HIDDEN="${CFG_NOISE_HIDDEN}" \
    REGROUND_FIXED_ENC_NOISE=true \
    REGROUND_REUSE_ENCODE_FOR_REPG=false \
    T_FIX_VALUES_OVERRIDE="${tfix}" \
        bash batch_run_inference_reground_b_h200.sh "${CKPT_STEP}" 2>&1 | grep -E "submitted|SKIP"
}

submit_linear() {
    local repg=$1
    echo "  Linear: cfg=${CFG}, repg=${repg} (cfg_noise_hidden not applied in linear path)"
    EXPERIMENTS_OVERRIDE="${CONFIG_PATH}|${MODEL}" \
    CFG_SCALE="${CFG}" \
    HIDDEN_REP_GUIDANCE="${repg}" \
        bash batch_run_inference_linear_hidden_b_h200_sphereclamp.sh "${CKPT_STEP}" 2>&1 | grep -E "submitted|SKIP"
}

echo "============================================="
echo "  t_fix sweep under guidance"
echo "  Model: ${MODEL} @ ${CKPT_STEP}"
echo "  CFG=${CFG}, cfg_noise_hidden=${CFG_NOISE_HIDDEN} (reground only)"
echo "  t_fix: ${T_FIX_VALUES[*]}"
echo "  repg:  ${REPG_VALUES[*]}"
echo "============================================="
echo ""

SUBMITTED=0

for repg in "${REPG_VALUES[@]}"; do
    echo "=== repg=${repg} ==="
    for tfix in "${T_FIX_VALUES[@]}"; do
        submit_reground "${tfix}" "${repg}"
        SUBMITTED=$((SUBMITTED + 1))
    done
    submit_linear "${repg}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
