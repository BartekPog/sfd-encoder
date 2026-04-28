#!/bin/bash
# =============================================================================
# batch_run_inference_1p0_viper_ft_cfgrepg.sh — 6-job inference matrix on
# Viper-GPU (MI300A) for the cfg_repg-dropout FT of the 1p0B model.
#
# Reproduces the parent's best (encodereground + CFG + autoguidance) on the
# new FT checkpoint and runs a small ablation around it:
#
#   R  encodereground  cfg=1.4   AG       t_fix in {0.10, 0.15}    (replicate)
#   1  encodereground  cfg=1.05  no AG    t_fix=0.6                 (no AG)
#   2  encodereground  cfg=1.0   no AG    t_fix=0.5                 (no CFG, no AG)
#   3  linear          cfg=1.4   AG       —                         (linear baseline)
#   4  linear          cfg=1.0   no AG    —                         (linear, no guidance)
#   5  linear          cfg=1.4   no AG    —                         (linear, CFG only)
#
# Usage:
#   bash batch_run_inference_1p0_viper_ft_cfgrepg.sh [ckpt_step]
#
# Optional env overrides:
#   PER_PROC_BATCH_SIZE  (default 128 — conservative for 1p0B; raise after
#                         a successful run if memory headroom allows)
#   NUM_APUS             (default 1)
#   TIME                 (default 00-06:00:00)
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-100000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

TIME=${TIME:-"00-14:00:00"}
NUM_APUS=${NUM_APUS:-2}
CPUS_PER_APU=24
MEM_PER_APU=110000
CPUS_PER_TASK=$(( CPUS_PER_APU * NUM_APUS ))
MEM=$(( MEM_PER_APU * NUM_APUS ))
PRECISION=${PRECISION:-bf16}
VENV_PATH=${VENV_PATH:-.venv-sfd-rocm}

PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-128}
NUM_STEPS=${NUM_STEPS:-100}

CONFIG_PATH="configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5_ft_cfgrepg.yaml"
TRAIN_EXP_NAME="1p0_v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5_ft_cfgrepg"
CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"
INFER_EXP_NAME="${TRAIN_EXP_NAME}_${CKPT_NAME}"
INFERENCE_OUTPUT_DIR="outputs/inference"
AUTOGUIDANCE_CONFIG="configs/sfd/autoguidance_b/inference_ft.yaml"

if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: checkpoint ${CKPT_PATH} not found" >&2
    exit 1
fi
if [ ! -f "${AUTOGUIDANCE_CONFIG%/*}/$(basename ${AUTOGUIDANCE_CONFIG})" ]; then
    echo "ERROR: autoguidance config ${AUTOGUIDANCE_CONFIG} not found" >&2
    exit 1
fi

echo "============================================="
echo "  Inference matrix for ${INFER_EXP_NAME}"
echo "  per_proc_batch_size: ${PER_PROC_BATCH_SIZE}"
echo "  APUs/job:            ${NUM_APUS}"
echo "  Sampler:             Euler ${NUM_STEPS} steps"
echo "============================================="

SUBMITTED=0

# ------------------------------------------------------------------
# submit_job <tag> <kind> <cfg_scale> <ag:true|false> <t_fix_or_empty> [ag_ckpt]
#   - tag:        short label baked into job name + output dir
#   - kind:       reground | linear
#   - cfg_scale:  numeric; 1.0 disables --cfg_scale flag
#   - ag:         true → enables autoguidance (sample.autoguidance=true ...)
#   - t_fix:      encode_reground_t_fix; required for reground, ignored for linear
#   - ag_ckpt:    autoguidance checkpoint iteration in k (e.g. 80, 60, 40, 20)
# ------------------------------------------------------------------
submit_job() {
    local TAG=$1
    local KIND=$2
    local CFG=$3
    local AG=$4
    local T_FIX=$5
    local AG_CKPT=${6:-80}

    local INFER_FLAGS=""
    local SAVE_FLAGS=""
    local CFG_OVERRIDES=""
    local INFERENCE_TYPE=""
    local ENV_OVERRIDE=""

    # Schedule-specific flags
    if [ "${KIND}" = "reground" ]; then
        INFERENCE_TYPE="encodereground"
        INFER_FLAGS+=" --hidden_schedule encodereground --hidden_sphere_clamp --reground_fixed_enc_noise --encode_reground_t_fix ${T_FIX}"
        SAVE_FLAGS+=" --hidden_sphere_clamp --reground_fixed_enc_noise --encode_reground_t_fix ${T_FIX}"
    elif [ "${KIND}" = "linear" ]; then
        INFERENCE_TYPE="linear"
        INFER_FLAGS+=" --hidden_schedule linear --hidden_sphere_clamp"
        SAVE_FLAGS+=" --hidden_sphere_clamp"
    else
        echo "  bad kind: ${KIND}" >&2; exit 1
    fi

    # CFG (skip the flag when off — match existing launchers' convention)
    if (( $(echo "${CFG} > 1.0" | bc -l) )); then
        INFER_FLAGS+=" --cfg_scale ${CFG}"
        SAVE_FLAGS+=" --cfg_scale ${CFG}"
    fi

    # Autoguidance
    if [ "${AG}" = "true" ]; then
        CFG_OVERRIDES+=" sample.autoguidance=true sample.autoguidance_config=${AUTOGUIDANCE_CONFIG}"
        SAVE_FLAGS+=" --autoguidance_config ${AUTOGUIDANCE_CONFIG} --autoguidance_ckpt_iter ${AG_CKPT}"
        ENV_OVERRIDE="AUTOGUIDANCE_CKPT_ITER=${AG_CKPT}"
    fi

    local JOBSCRIPT="jobs/infer_1p0ftcfgrepg_${TAG}_${CKPT_NAME}.sh"
    local OUTPUT="job_outputs/infer_1p0ftcfgrepg_${TAG}_${CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")" "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash -l
#SBATCH --job-name infer1p0_${TAG}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:${NUM_APUS}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (${KIND}, ${TAG}): ${TRAIN_EXP_NAME} @ step ${CKPT_STEP}"

module purge
module load gcc/14 rocm/6.3 python-waterboa/2025.06
source ${VENV_PATH}/bin/activate

export TORCH_HOME=/viper/ptmp2/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/viper/ptmp2/bpogodzi/hidden-diffusion/cache/hf
export PYTORCH_ROCM_ARCH=gfx942
export HSA_XNACK=1
export PYTORCH_HIP_ALLOC_CONF=\${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}
export XFORMERS_DISABLED=1
export WANDB_MODE=\${WANDB_MODE:-offline}

# MIOpen find-db / kernel cache: per-job, node-local. Single-APU inference
# rarely hits the NFS rename race, but isolating costs nothing.
export MIOPEN_USER_DB_PATH=\${SLURM_TMPDIR:-/tmp}/miopen-\${SLURM_JOB_ID}/user-db
export MIOPEN_CUSTOM_CACHE_DIR=\${SLURM_TMPDIR:-/tmp}/miopen-\${SLURM_JOB_ID}/kernel-cache
mkdir -p \$MIOPEN_USER_DB_PATH \$MIOPEN_CUSTOM_CACHE_DIR

${ENV_OVERRIDE} GPUS_PER_NODE=${NUM_APUS} PRECISION=${PRECISION} \\
    bash run_inference.sh ${CONFIG_PATH} \\
    ckpt_path=${CKPT_PATH} \\
    sample.sampling_method=euler \\
    sample.num_sampling_steps=${NUM_STEPS} \\
    sample.per_proc_batch_size=${PER_PROC_BATCH_SIZE} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME}${CFG_OVERRIDES} \\
    ${INFER_FLAGS}

python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${CKPT_STEP} \\
    --inference_type ${INFERENCE_TYPE} \\
    --sampler euler \\
    --num_steps ${NUM_STEPS} \\
    ${SAVE_FLAGS}

# Best-effort cleanup of node-local MIOpen cache.
rm -rf \$(dirname \$MIOPEN_USER_DB_PATH) 2>/dev/null || true

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    local JOB_ID
    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${TAG} (${KIND}, cfg=${CFG}, ag=${AG}, t_fix=${T_FIX}, ag_ckpt=${AG_CKPT}k): submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
}

# ------------------------------------------------------------------
# Submit matrix: Evaluate autoguidance ckpts and t_fix for both schedules
# CFG scale is fixed at 1.4 (baseline)
# ------------------------------------------------------------------

# 1: Reground Schedule variations: t_fix in 0.15, 0.20, 0.25 | AG ckpt in 80, 60, 40, 20
for TFIX in 0.15 0.20 0.25; do
    for AG_ITER in 100 80 60 40 20; do
        # TAG name formatting (remove dot from t_fix, e.g. 0.15 -> 015)
        TFIX_TAG=$(echo ${TFIX} | tr -d '.')
        submit_job rg_t${TFIX_TAG}_ag${AG_ITER}k  reground 1.4  true  ${TFIX}  ${AG_ITER}
    done
done

# 2: Linear Schedule variations: AG ckpt in 80, 60, 40, 20
for AG_ITER in 100 80 60 40 20; do
    submit_job lin_ag${AG_ITER}k  linear   1.4  true  ""  ${AG_ITER}
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
