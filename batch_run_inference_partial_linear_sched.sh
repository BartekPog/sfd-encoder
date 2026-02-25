#!/bin/bash
# =============================================================================
# batch_run_hidden_partial_linear_sched.sh
#
# Sweeps over hidden_schedule_max_t values for the "linear" hidden schedule,
# evaluating the effect of stopping hidden token denoising before t=1.
#
# Model:      v2_base_mse02 (H8, shared t-emb, MSE 0.2)
# Checkpoint: 100K steps
# Schedule:   linear hidden, max_t ∈ {0.95, 0.90, 0.80, 0.70}
# Sampler:    Euler, 100 steps, FID50K, balanced sampling
#
# Baseline (max_t=1.0) is not re-run here; see batch_run_fid_comparison_v2_base.sh.
#
# Usage:
#   bash batch_run_hidden_partial_linear_sched.sh
# =============================================================================

set -euo pipefail

# ---- Fixed settings ----
CONFIG_PATH="configs/sfd/hidden_b_h200/v2_base_mse02.yaml"
TRAIN_EXP_NAME="v2_base_mse02"
CKPT_STEP=80000
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")
CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"

# ---- Sweep values ----
MAX_T_VALUES=(0.95 0.90 0.80 0.70 0.60)

# ---- SLURM settings ----
TIME=${TIME:-"00-07:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
INFERENCE_OUTPUT_DIR="outputs/inference"

echo "============================================="
echo "  Partial linear hidden schedule sweep"
echo "  Model:      ${TRAIN_EXP_NAME}"
echo "  Checkpoint: ${CKPT_NAME}.pt"
echo "  Sampler:    Euler, 100 steps"
echo "  max_t values: ${MAX_T_VALUES[*]}"
echo "  GPUs: ${NUM_GPUS} x H200"
echo "============================================="
echo ""

if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

SUBMITTED=0

for MAX_T in "${MAX_T_VALUES[@]}"; do
    # Format max_t as two decimal places for naming (e.g. 0.95 → 095)
    MAX_T_TAG=$(printf "%.2f" "${MAX_T}" | tr -d '.')

    INFER_EXP_NAME="${TRAIN_EXP_NAME}_${CKPT_NAME}"
    EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
    JOBSCRIPT="jobs/infer_plin${MAX_T_TAG}_${EXP_LABEL}_${CKPT_NAME}.sh"
    OUTPUT="job_outputs/infer_plin${MAX_T_TAG}_${EXP_LABEL}_${CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name plin${MAX_T_TAG}_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Partial linear hidden schedule: max_t=${MAX_T}, ${EXP_LABEL} @ step ${CKPT_STEP}"

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
    --hidden_schedule linear \\
    --hidden_schedule_max_t ${MAX_T}
python save_fid_result.py \
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \
    --config     ${CONFIG_PATH} \
    --ckpt_step  ${CKPT_STEP} \
    --inference_type linear \
    --sampler euler \
    --num_steps 100 \
    --hidden_schedule_max_t ${MAX_T}
echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  max_t=${MAX_T}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} partial-linear schedule jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
