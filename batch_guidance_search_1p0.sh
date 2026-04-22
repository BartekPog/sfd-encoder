#!/bin/bash
# =============================================================================
# batch_guidance_search_1p0.sh
#
# Two-stage parameter search for guidance on the 1p0B hidden model.
# Fixed: t_fix=0.6, Euler 100 steps, BS=2048, checkpoint=80000.
#
# Stage 1 (run first — 17 jobs, all independent):
#   A. CFG sweep at repg=1.0, cfg_noise_hidden=false (hidden kept in neg pass):
#      - CFG 1.05, 1.1, 1.15, 1.2, 1.3
#   B. CFG sweep at repg=1.0, cfg_noise_hidden=true (fully noised neg pass):
#      - CFG 1.05, 1.1, 1.15, 1.2, 1.3
#   C. Repg=1.1 × FP mode variants (CFG off):
#      - merged FP, fresh noise
#      - merged FP, fixed noise
#      - separate FP, fixed noise
#   D. CFG sweep with AG-Hidden20K (hidden model at 20K as autoguidance):
#      - CFG 1.1, 1.2, 1.3
#   + baseline (no guidance)
#
# Stage 2 (run after reviewing stage 1):
#   Best CFG settings × best repg × FP modes
#   Invoke with: bash batch_guidance_search_1p0.sh stage2
#
# Usage:
#   bash batch_guidance_search_1p0.sh          # runs stage 1
#   bash batch_guidance_search_1p0.sh stage2   # runs stage 2 (edit bests below first)
# =============================================================================

set -euo pipefail

STAGE=${1:-stage1}

# ---- Shared settings ----
CKPT_STEP=80000
T_FIX=0.6
export TIME="00-20:00:00"
export PER_PROC_BATCH_SIZE=2048

submit_job() {
    # Args: CFG_SCALE  HIDDEN_REP_GUIDANCE  REGROUND_REUSE_ENCODE_FOR_REPG  REGROUND_FIXED_ENC_NOISE  CFG_NOISE_HIDDEN  AUTOGUIDANCE_CONFIG  label
    local cfg=$1 repg=$2 reuse=$3 fixenc=$4 noiseh=$5 agcfg=$6 label=$7

    echo "  Submitting: ${label} (cfg=${cfg}, repg=${repg}, reuse=${reuse}, fixenc=${fixenc}, noise_h=${noiseh}, ag=${agcfg:-none})"

    CFG_SCALE="${cfg}" \
    HIDDEN_REP_GUIDANCE="${repg}" \
    REGROUND_REUSE_ENCODE_FOR_REPG="${reuse}" \
    REGROUND_FIXED_ENC_NOISE="${fixenc}" \
    CFG_NOISE_HIDDEN="${noiseh}" \
    AUTOGUIDANCE_CONFIG="${agcfg}" \
    T_FIX_VALUES_OVERRIDE="${T_FIX}" \
        bash batch_run_inference_reground_1p0_h200.sh "${CKPT_STEP}" 2>&1 | grep -E "submitted|SKIP"
}

if [ "${STAGE}" = "stage1" ]; then
    echo "============================================="
    echo "  STAGE 1: CFG sweep + repg FP-mode test"
    echo "  t_fix=${T_FIX}, ckpt=${CKPT_STEP}, BS=${PER_PROC_BATCH_SIZE}"
    echo "============================================="
    echo ""

    # --- A. CFG sweep, cfg_noise_hidden=false (hidden kept in negative pass) ---
    echo ""
    echo "--- A. CFG sweep (cfg_noise_hidden=false) ---"
    for cfg in 1.05 1.15 1.2 1.3; do
        submit_job ${cfg}  1.0  false true false ""  "cfg${cfg}"
    done

    # --- B. CFG sweep, cfg_noise_hidden=true (fully noised negative pass) ---
    echo ""
    echo "--- B. CFG sweep (cfg_noise_hidden=true) ---"
    for cfg in 1.05 1.15 1.2 1.3; do
        submit_job ${cfg}  1.0  false true true ""  "cfg${cfg}_noiseh"
    done

    # --- C. Repg=1.1, FP mode variants ---
    echo ""
    echo "--- C. Repg=1.1 × FP modes ---"
    # Merged FP, fresh noise each step
    submit_job 1.0  1.1  true  false false ""  "repg1.1_merged_fresh"
    # Merged FP, fixed encode noise across steps
    submit_job 1.0  1.1  true  true  false ""  "repg1.1_merged_fixed"
    # Separate FP, fixed encode noise
    submit_job 1.0  1.1  false true  false ""  "repg1.1_separate_fixed"

    # --- D. Autoguidance with hidden 20K model ---
    echo ""
    AG_H20K="configs/sfd/hidden_1p0_h200_from_ft/autoguidance_hidden_20k.yaml"
    echo "--- D. CFG sweep with AG-Hidden20K ---"
    for cfg in 1.05 1.15 1.3; do
        submit_job ${cfg}  1.0  false true false "${AG_H20K}"  "cfg${cfg}_ag_h20k"
    done

    echo ""
    echo "Stage 1 submitted. Review FID results, then edit stage2 bests and run:"
    echo "  bash batch_guidance_search_1p0.sh stage2"

elif [ "${STAGE}" = "stage2" ]; then
    echo "============================================="
    echo "  STAGE 2: Cross-validate best CFG × repg × FP mode"
    echo "============================================="
    echo ""

    # ========== EDIT THESE after reviewing stage 1 results ==========
    BEST_CFG=1.1                    # best CFG scale from stage 1
    BEST_CFG_NOISE_HIDDEN=false     # whether cfg_noise_hidden helped
    BEST_REPG=1.1                   # best repg from stage 1
    # ================================================================

    echo "--- CFG=${BEST_CFG} (noise_h=${BEST_CFG_NOISE_HIDDEN}) × repg=${BEST_REPG} × FP modes ---"

    # Mode 1: Merged FP, fresh noise each step
    submit_job "${BEST_CFG}" "${BEST_REPG}" true  false "${BEST_CFG_NOISE_HIDDEN}" ""  "best_merged_fresh"

    # Mode 2: Merged FP, encode noise fixed across steps
    submit_job "${BEST_CFG}" "${BEST_REPG}" true  true  "${BEST_CFG_NOISE_HIDDEN}" ""  "best_merged_fixed"

    # Mode 3: Separate FP, encode noise fixed across steps
    submit_job "${BEST_CFG}" "${BEST_REPG}" false true  "${BEST_CFG_NOISE_HIDDEN}" ""  "best_separate_fixed"

    echo ""
    echo "Stage 2 submitted."

else
    echo "Unknown stage: ${STAGE}. Use 'stage1' or 'stage2'."
    exit 1
fi
