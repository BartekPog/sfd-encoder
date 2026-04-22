#!/bin/bash
# =============================================================================
# batch_guidance_eval_b.sh
#
# Evaluate whether models trained WITHOUT representation guidance (HGD-only)
# respond better to inference-time guidance than models trained WITH repg.
#
# Core question: can HGD-only + test-time guidance beat repg-trained models?
#
# Models (all shift 1.5, ckpt 60K):
#   WITH training repg:
#     1. repg_1p5              — reference (repg 1.5, no HGD)
#     2. repg_1p5_hgd_2        — repg 1.5 + HGD 2.0
#   WITHOUT training repg (HGD only):
#     3. no_repg_hgd_2         — HGD 2.0
#     4. no_repg_hgd_4         — HGD 4.0
#     5. no_repg_hgd_5         — HGD 5.0
#
# Guidance configs at t_fix=0.88:
#   a. No guidance (baseline)
#   b. CFG 1.1, cfg_noise_hidden=false
#   c. CFG 1.1, cfg_noise_hidden=true
#   d. repg 1.1
#   e. CFG 1.5 (no_repg_hgd_4 only — probe stronger CFG)
#
# Total: 5×4 + 1 = 21 jobs
# =============================================================================

set -euo pipefail

CKPT_STEP=60000
T_FIX=0.88
export TIME="00-04:00:00"
export PER_PROC_BATCH_SIZE=1024

CONFIG_DIR="configs/sfd/hidden_b_h200_from_ft"

submit_job() {
    # Args: CONFIG_BASENAME  CFG_SCALE  HIDDEN_REP_GUIDANCE  CFG_NOISE_HIDDEN  label
    local config=$1 cfg=$2 repg=$3 noiseh=$4 label=$5
    local config_path="${CONFIG_DIR}/${config}.yaml"

    echo "  Submitting: ${label} (cfg=${cfg}, repg=${repg}, noise_h=${noiseh})"

    EXPERIMENTS_OVERRIDE="${config_path}|${config}" \
    CFG_SCALE="${cfg}" \
    HIDDEN_REP_GUIDANCE="${repg}" \
    CFG_NOISE_HIDDEN="${noiseh}" \
    REGROUND_FIXED_ENC_NOISE=true \
    REGROUND_REUSE_ENCODE_FOR_REPG=false \
    T_FIX_VALUES_OVERRIDE="${T_FIX}" \
        bash batch_run_inference_reground_b_h200.sh "${CKPT_STEP}" 2>&1 | grep -E "submitted|SKIP"
}

echo "============================================="
echo "  B-model guidance evaluation"
echo "  t_fix=${T_FIX}, ckpt=${CKPT_STEP}, BS=${PER_PROC_BATCH_SIZE}"
echo "============================================="
echo ""

# ---- Models WITH training repg ----

REPG_REF="v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5"
REPG_HGD2="v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2"

echo "=== Models WITH training repg ==="
for model in "${REPG_REF}" "${REPG_HGD2}"; do
    short=$(echo "${model}" | sed 's/v4_mse0001_noisy_enc_nocurr_//')
    echo ""
    echo "--- ${short} ---"
    submit_job "${model}" 1.0  1.0  false "${short}_baseline"
    submit_job "${model}" 1.1  1.0  false "${short}_cfg1.1"
    submit_job "${model}" 1.1  1.0  true  "${short}_cfg1.1_noiseh"
    submit_job "${model}" 1.0  1.1  false "${short}_repg1.1"
done

# ---- Models WITHOUT training repg (HGD only) ----

NOREPG_HGD2="v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_2"
NOREPG_HGD4="v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_4"
NOREPG_HGD5="v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5"

echo ""
echo "=== Models WITHOUT training repg (HGD only) ==="
for model in "${NOREPG_HGD2}" "${NOREPG_HGD4}" "${NOREPG_HGD5}"; do
    short=$(echo "${model}" | sed 's/v4_mse0001_noisy_enc_nocurr_//')
    echo ""
    echo "--- ${short} ---"
    submit_job "${model}" 1.0  1.0  false "${short}_baseline"
    submit_job "${model}" 1.1  1.0  false "${short}_cfg1.1"
    submit_job "${model}" 1.1  1.0  true  "${short}_cfg1.1_noiseh"
    submit_job "${model}" 1.0  1.1  false "${short}_repg1.1"
done

# ---- Extra: stronger CFG on no_repg_hgd_4 ----
echo ""
echo "--- Extra: no_repg_hgd_4 with stronger CFG ---"
submit_job "${NOREPG_HGD4}" 1.5  1.0  false "norepg_hgd4_cfg1.5"

echo ""
echo "============================================="
echo "  21 jobs submitted."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
