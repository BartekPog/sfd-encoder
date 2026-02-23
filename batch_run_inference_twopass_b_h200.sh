#!/bin/bash
# =============================================================================
# batch_run_inference_twopass_b_h200.sh — Two-pass FID50K inference for
# all B-size H200 hidden-token experiments.
#
# Pass 1: jointly generates image + hidden tokens (linear hidden schedule)
# Pass 2: regenerates the image from the same noise with clean hidden tokens
#
# Submits jobs for multiple step-per-pass configurations (e.g. 100+100, 50+50).
#
# Usage:
#   bash batch_run_inference_twopass_b_h200.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 300000)
#
# All experiments use:  Euler sampler, cfg_scale=1.0, FID50K,
#                       --two_pass --hidden_schedule linear
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-300000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- Steps-per-pass configurations ----
# Each value N means N steps for pass 1 + N steps for pass 2 (2N total)
STEPS_PER_PASS_LIST=(100 50)

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
# Only include experiments with hidden tokens (skip exp2 standard finetune and H16)
EXPERIMENTS=(
    "configs/sfd/hidden_b_h200/exp1_hidden_scratch.yaml|hidden_b_h200_scratch"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained.yaml|hidden_b_h200_from_pretrained"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_h1.yaml|hidden_b_h200_from_pretrained_h1"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_separate_embedder.yaml|hidden_b_h200_from_pretrained_separate_embedder"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_hidden_pos_encoding.yaml|hidden_b_h200_from_pretrained_hidden_pos_encoding"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_weak_h_loss.yaml|hidden_b_h200_from_pretrained_weak_h_loss"
)

echo "============================================="
echo "  B-size H200 — Two-Pass FID50K Inference"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler: Euler"
echo "  Steps per pass: ${STEPS_PER_PASS_LIST[*]}"
echo "  Mode: two-pass (linear → fixed)"
echo "  GPUs: ${NUM_GPUS} x H200"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "============================================="
echo ""

SUBMITTED=0

for STEPS_PER_PASS in "${STEPS_PER_PASS_LIST[@]}"; do
TOTAL_STEPS=$((STEPS_PER_PASS * 2))
echo "--- Steps per pass: ${STEPS_PER_PASS} (${TOTAL_STEPS} total) ---"

for ENTRY in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME <<< "${ENTRY}"
    CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "  SKIP: ${TRAIN_EXP_NAME} — checkpoint ${CKPT_PATH} not found"
        continue
    fi

    INFER_EXP_NAME="${TRAIN_EXP_NAME}_${CKPT_NAME}"
    EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
    JOBSCRIPT="jobs/infer_2p${STEPS_PER_PASS}_${EXP_LABEL}_${CKPT_NAME}.sh"
    OUTPUT="job_outputs/infer_2p${STEPS_PER_PASS}_${EXP_LABEL}_${CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name infer_2p${STEPS_PER_PASS}_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Two-pass inference (${STEPS_PER_PASS}+${STEPS_PER_PASS}): ${EXP_LABEL} @ step ${CKPT_STEP}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh ${CONFIG_PATH} \\
    ckpt_path=${CKPT_PATH} \\
    sample.sampling_method=euler \\
    sample.num_sampling_steps=${STEPS_PER_PASS} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME} \\
    --hidden_schedule linear \\
    --two_pass

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${TRAIN_EXP_NAME} (${STEPS_PER_PASS}+${STEPS_PER_PASS}): submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done
echo ""
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} two-pass inference jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
