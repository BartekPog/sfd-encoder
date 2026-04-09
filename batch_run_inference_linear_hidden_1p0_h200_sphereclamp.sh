#!/bin/bash
# =============================================================================
# batch_run_inference_linear_hidden_1p0_h200_sphereclamp.sh
#
# FID50K inference for 1p0B models on H200.
# Includes non-hidden baselines (original pretrained + finetuned) and
# hidden-token model with linear schedule + sphere clamping.
#
# Usage:
#   bash batch_run_inference_linear_hidden_1p0_h200_sphereclamp.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step for hidden model (default: 40000)
#
# Optional env overrides (guidance benchmarks):
#   CFG_SCALE            — classifier-free guidance scale (default 1.0 = off).
#                          Set to 1.5 to match the best SFD setup from the
#                          README tables (XL 800ep / XXL 80ep / XXL 800ep).
#                          Applied to BOTH baselines and the hidden model.
#   HIDDEN_REP_GUIDANCE  — hidden representation guidance scale (default 1.0 = off).
#                          Uses the ramped weight w(t_h) = 1 + (s - 1) * t_h.
#                          Applied only to the hidden-token model (the rep-guidance
#                          code path has no effect on no-hidden baselines).
#
# Examples:
#   CFG_SCALE=1.5 bash batch_run_inference_linear_hidden_1p0_h200_sphereclamp.sh
#   HIDDEN_REP_GUIDANCE=2.0 bash batch_run_inference_linear_hidden_1p0_h200_sphereclamp.sh
#   CFG_SCALE=1.5 HIDDEN_REP_GUIDANCE=2.0 bash batch_run_inference_linear_hidden_1p0_h200_sphereclamp.sh
#
# All experiments use:  Euler sampler, 100 steps, FID50K
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-60000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings (H200 cluster / DAIS) ----
TIME=${TIME:-"00-6:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-512}

# ---- Guidance overrides (off by default; matches SFD best-FID setup when enabled) ----
CFG_SCALE=${CFG_SCALE:-1.0}                       # 1.0 = off; 1.5 = SFD best
HIDDEN_REP_GUIDANCE=${HIDDEN_REP_GUIDANCE:-1.0}   # 1.0 = off; e.g. 2.0 or 4.0 enables rep guidance

# CFG applies to both baselines and hidden runs; rep guidance only to hidden runs.
BASE_INFER_FLAGS=""
BASE_SAVE_FLAGS=""
BASE_TAG=""
HIDDEN_INFER_FLAGS=""
HIDDEN_SAVE_FLAGS=""
HIDDEN_TAG=""
if (( $(echo "${CFG_SCALE} > 1.0" | bc -l) )); then
    CFG_FLAG=" --cfg_scale ${CFG_SCALE}"
    CFG_TAG="_cfg$(printf '%.2f' ${CFG_SCALE} | tr -d '.')"
    BASE_INFER_FLAGS+="${CFG_FLAG}"
    BASE_SAVE_FLAGS+="${CFG_FLAG}"
    BASE_TAG+="${CFG_TAG}"
    HIDDEN_INFER_FLAGS+="${CFG_FLAG}"
    HIDDEN_SAVE_FLAGS+="${CFG_FLAG}"
    HIDDEN_TAG+="${CFG_TAG}"
fi
if (( $(echo "${HIDDEN_REP_GUIDANCE} > 1.0" | bc -l) )); then
    HRG_FLAG=" --hidden_rep_guidance ${HIDDEN_REP_GUIDANCE}"
    HRG_TAG="_hrg$(printf '%.1f' ${HIDDEN_REP_GUIDANCE} | tr -d '.')"
    HIDDEN_INFER_FLAGS+="${HRG_FLAG}"
    HIDDEN_SAVE_FLAGS+="${HRG_FLAG}"
    HIDDEN_TAG+="${HRG_TAG}"
fi

# ---- Inference output directory ----
INFERENCE_OUTPUT_DIR="outputs/inference"

# ---- Baseline experiments (no hidden tokens) ----
# Format: "config_yaml|train_exp_name|ckpt_step"
BASELINE_EXPERIMENTS=(
    # Original pretrained SFD 1p0B (4M steps)
    # "configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden.yaml|sfd_1p0|4000000"
    # Finetuned SFD 1p0B (no hidden tokens, 40K steps from 4M)
    "configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden.yaml|1p0_finetune_no_hidden|60000"
)

# ---- Hidden-token experiments (linear schedule + sphere clamp) ----
# Format: "config_yaml|train_exp_name[|ckpt_step_override]"
HIDDEN_EXPERIMENTS=(
    "configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml|1p0_v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5"
)

echo "============================================="
echo "  1p0B H200 — FID50K Inference (baselines + linear hidden + sphere clamp)"
echo "  Hidden ckpt step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler: Euler, 100 steps"
echo "  Batch size: ${PER_PROC_BATCH_SIZE}"
echo "  CFG scale: ${CFG_SCALE} (applied to baselines + hidden)"
echo "  Hidden rep guidance: ${HIDDEN_REP_GUIDANCE} (applied to hidden only)"
echo "  GPUs: ${NUM_GPUS} x H200"
echo "============================================="
echo ""

SUBMITTED=0

# ---- Run baseline experiments (no hidden tokens, standard inference) ----
echo "--- Baselines (no hidden tokens) ---"
for ENTRY in "${BASELINE_EXPERIMENTS[@]}"; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME EXP_CKPT_STEP <<< "${ENTRY}"
    EXP_CKPT_NAME=$(printf "%07d" "${EXP_CKPT_STEP}")
    CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${EXP_CKPT_NAME}.pt"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "  SKIP: ${TRAIN_EXP_NAME} — checkpoint ${CKPT_PATH} not found"
        continue
    fi

    INFER_EXP_NAME="${TRAIN_EXP_NAME}_${EXP_CKPT_NAME}"
    EXP_LABEL="${TRAIN_EXP_NAME}_${EXP_CKPT_NAME}"
    JOBSCRIPT="jobs/infer_1p0_base${BASE_TAG}_${EXP_LABEL}.sh"
    OUTPUT="job_outputs/infer_1p0_base${BASE_TAG}_${EXP_LABEL}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name 1p0_b_${TRAIN_EXP_NAME}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (baseline, no hidden): ${TRAIN_EXP_NAME} @ step ${EXP_CKPT_STEP}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh ${CONFIG_PATH} \\
    ckpt_path=${CKPT_PATH} \\
    sample.sampling_method=euler \\
    sample.num_sampling_steps=100 \\
    sample.per_proc_batch_size=${PER_PROC_BATCH_SIZE} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME}${BASE_INFER_FLAGS}
python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${EXP_CKPT_STEP} \\
    --inference_type standard \\
    --sampler euler \\
    --num_steps 100${BASE_SAVE_FLAGS}
echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${TRAIN_EXP_NAME} @ ${EXP_CKPT_STEP}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done

# ---- Run hidden-token experiments (linear + sphere clamp) ----
echo ""
echo "--- Hidden token models (linear + sphere clamp) ---"
for ENTRY in "${HIDDEN_EXPERIMENTS[@]}"; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME EXP_CKPT_STEP_OVERRIDE <<< "${ENTRY}"
    EXP_CKPT_STEP=${EXP_CKPT_STEP_OVERRIDE:-${CKPT_STEP}}
    EXP_CKPT_NAME=$(printf "%07d" "${EXP_CKPT_STEP}")
    CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${EXP_CKPT_NAME}.pt"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "  SKIP: ${TRAIN_EXP_NAME} — checkpoint ${CKPT_PATH} not found"
        continue
    fi

    INFER_EXP_NAME="${TRAIN_EXP_NAME}_${EXP_CKPT_NAME}"
    EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
    JOBSCRIPT="jobs/infer_1p0_linsc${HIDDEN_TAG}_${EXP_LABEL}_${EXP_CKPT_NAME}.sh"
    OUTPUT="job_outputs/infer_1p0_linsc${HIDDEN_TAG}_${EXP_LABEL}_${EXP_CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name 1p0_lin_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (linear hidden + sphere clamp): ${EXP_LABEL} @ step ${EXP_CKPT_STEP}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh ${CONFIG_PATH} \\
    ckpt_path=${CKPT_PATH} \\
    sample.sampling_method=euler \\
    sample.num_sampling_steps=100 \\
    sample.per_proc_batch_size=${PER_PROC_BATCH_SIZE} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME} \\
    --hidden_schedule linear \\
    --hidden_sphere_clamp${HIDDEN_INFER_FLAGS}
python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${EXP_CKPT_STEP} \\
    --inference_type linear \\
    --sampler euler \\
    --num_steps 100 \\
    --hidden_sphere_clamp${HIDDEN_SAVE_FLAGS}
echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${TRAIN_EXP_NAME}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} inference jobs (1p0B baselines + linear hidden)."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
