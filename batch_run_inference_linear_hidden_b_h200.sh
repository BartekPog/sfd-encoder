#!/bin/bash
# =============================================================================
# batch_run_inference_linear_hidden_b_h200.sh — FID50K inference with LINEAR
# hidden schedule for all B-size H200 hidden-token experiments
#
# Reuses the training configs directly, overriding inference settings
# via OmegaConf CLI overrides.  Only includes experiments that use hidden
# tokens (exp2 standard finetune is excluded — no hidden tokens).
#
# Usage:
#   bash batch_run_inference_linear_hidden_b_h200.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 300000)
#
# All experiments use:  Euler sampler, 100 steps, cfg_scale=1.0, FID50K,
#                       hidden_schedule=linear
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-400000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings (H200 cluster / DAIS) ----
TIME=${TIME:-"00-14:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"

# ---- Inference output directory ----
INFERENCE_OUTPUT_DIR="outputs/inference"

# ---- Hidden-token experiment definitions ----
# Format: "config_yaml|train_exp_name[|ckpt_step_override]"
# The optional third field overrides the global CKPT_STEP for that experiment.
EXPERIMENTS=(
    # Base 120K checkpoint (pretrained, no fine-tuning) — fixed step, always 120K
    # "configs/sfd/hidden_b_h200/exp0_base_120k.yaml|sfd_autoguidance_b|120000"
    # "configs/sfd/hidden_b_h200/exp1_hidden_scratch.yaml|hidden_b_h200_scratch"
    # "configs/sfd/hidden_b_h200/exp2_standard_finetune.yaml|sfd_b_h200_finetune"
    # "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained.yaml|hidden_b_h200_from_pretrained"
    # "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_h1.yaml|hidden_b_h200_from_pretrained_h1"
    # H16 skipped — not enough training steps yet
    # "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_separate_embedder.yaml|hidden_b_h200_from_pretrained_separate_embedder"
    # "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_hidden_pos_encoding.yaml|hidden_b_h200_from_pretrained_hidden_pos_encoding"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_weak_h_loss.yaml|hidden_b_h200_from_pretrained_weak_h_loss"
)

echo "============================================="
echo "  B-size H200 — FID50K Inference (LINEAR hidden schedule)"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler: Euler, 100 steps"
echo "  Hidden schedule: linear"
echo "  GPUs: ${NUM_GPUS} x H200"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "============================================="
echo ""

SUBMITTED=0

for ENTRY in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME EXP_CKPT_STEP_OVERRIDE <<< "${ENTRY}"
    # Use per-experiment override if provided, otherwise fall back to the global CKPT_STEP
    EXP_CKPT_STEP=${EXP_CKPT_STEP_OVERRIDE:-${CKPT_STEP}}
    EXP_CKPT_NAME=$(printf "%07d" "${EXP_CKPT_STEP}")
    CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${EXP_CKPT_NAME}.pt"

    # Verify checkpoint exists
    if [ ! -f "${CKPT_PATH}" ]; then
        echo "  SKIP: ${TRAIN_EXP_NAME} — checkpoint ${CKPT_PATH} not found"
        continue
    fi

    INFER_EXP_NAME="${TRAIN_EXP_NAME}_${EXP_CKPT_NAME}"
    EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
    JOBSCRIPT="jobs/infer_linear_${EXP_LABEL}_${EXP_CKPT_NAME}.sh"
    OUTPUT="job_outputs/infer_linear_${EXP_LABEL}_${EXP_CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name infer_lin_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (linear hidden): ${EXP_LABEL} @ step ${EXP_CKPT_STEP}"

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
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME} \\
    --hidden_schedule linear

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${TRAIN_EXP_NAME}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} inference jobs (linear hidden schedule)."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
