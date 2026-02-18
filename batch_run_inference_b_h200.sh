#!/bin/bash
# =============================================================================
# batch_run_inference_b_h200.sh — FID50K inference for all B-size H200 experiments
#
# Reuses the training configs directly, overriding only inference-specific
# settings (ckpt_path, sampling method, output dir) via OmegaConf CLI overrides.
#
# Usage:
#   bash batch_run_inference_b_h200.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 200000)
#
# All experiments use:  Euler sampler, 100 steps, cfg_scale=1.0, FID50K
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-200000}
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

# ---- Experiment definitions ----
# Format: "config_yaml|exp_name_in_output_dir|ckpt_dir_name"
EXPERIMENTS=(
    "configs/sfd/hidden_b_h200/exp1_hidden_scratch.yaml|hidden_b_h200_scratch"
    "configs/sfd/hidden_b_h200/exp2_standard_finetune.yaml|sfd_b_h200_finetune"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained.yaml|hidden_b_h200_from_pretrained"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_h1.yaml|hidden_b_h200_from_pretrained_h1"
    # H16 skipped — only has up to 40K steps due to earlier crash
    # "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_h16.yaml|hidden_b_h200_from_pretrained_h16"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_separate_embedder.yaml|hidden_b_h200_from_pretrained_separate_embedder"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_hidden_pos_encoding.yaml|hidden_b_h200_from_pretrained_hidden_pos_encoding"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_weak_h_loss.yaml|hidden_b_h200_from_pretrained_weak_h_loss"
)

echo "============================================="
echo "  B-size H200 — FID50K Inference"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler: Euler, 100 steps"
echo "  GPUs: ${NUM_GPUS} x H200"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "============================================="
echo ""

SUBMITTED=0

for ENTRY in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME <<< "${ENTRY}"
    CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"

    # Verify checkpoint exists
    if [ ! -f "${CKPT_PATH}" ]; then
        echo "  SKIP: ${TRAIN_EXP_NAME} — checkpoint ${CKPT_PATH} not found"
        continue
    fi

    # Inference exp_name includes the checkpoint step for clarity
    INFER_EXP_NAME="${TRAIN_EXP_NAME}_${CKPT_NAME}"
    EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
    JOBSCRIPT="jobs/infer_${EXP_LABEL}_${CKPT_NAME}.sh"
    OUTPUT="job_outputs/infer_${EXP_LABEL}_${CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name infer_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference: ${EXP_LABEL} @ step ${CKPT_STEP}"

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
    train.exp_name=${INFER_EXP_NAME}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${TRAIN_EXP_NAME}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} inference jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
