#!/bin/bash
# =============================================================================
# batch_run_inference_linear_1p0_hgd5.sh
#
# FID50K eval for 1p0B hgd_5 model on linear schedule + sphere clamp.
#
# Model: 1p0_v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5_warm8k
# Config: configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5.yaml
# Ckpt: 30K
#
# CFG sweep: 1.0 (no guidance), 1.2 (slight CFG)
# No hidden rep guidance.
#
# Total: 2 jobs
# =============================================================================

set -euo pipefail

CKPT_STEP=${CKPT_STEP:-30000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

TIME=${TIME:-"00-10:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-2048}

TRAIN_EXP_NAME="1p0_v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5_warm8k"
CONFIG_PATH="configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5.yaml"
CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"
INFER_EXP_NAME="${TRAIN_EXP_NAME}_${CKPT_NAME}"
INFERENCE_OUTPUT_DIR="outputs/inference"

if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: checkpoint ${CKPT_PATH} not found"
    exit 1
fi

CFG_VALUES=(1.0 1.2)

echo "============================================="
echo "  Linear inference — ${TRAIN_EXP_NAME}"
echo "  Ckpt: ${CKPT_STEP}"
echo "  CFGs: ${CFG_VALUES[*]}"
echo "  Schedule: linear + sphere clamp, no repg"
echo "============================================="
echo ""

SUBMITTED=0

for cfg in "${CFG_VALUES[@]}"; do
    GUIDE_INFER_FLAGS=""
    GUIDE_SAVE_FLAGS=""
    GUIDE_TAG=""
    if (( $(echo "${cfg} > 1.0" | bc -l) )); then
        GUIDE_INFER_FLAGS=" --cfg_scale ${cfg}"
        GUIDE_SAVE_FLAGS=" --cfg_scale ${cfg}"
        GUIDE_TAG="_cfg$(printf '%.2f' ${cfg} | tr -d '.')"
    fi

    EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
    JOBSCRIPT="jobs/infer_1p0_linsc${GUIDE_TAG}_${EXP_LABEL}_${CKPT_NAME}.sh"
    OUTPUT="job_outputs/infer_1p0_linsc${GUIDE_TAG}_${EXP_LABEL}_${CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")" "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name inlsc_1p0_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (linear hidden + sphere clamp): ${EXP_LABEL} @ step ${CKPT_STEP} cfg=${cfg}"

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
    --hidden_sphere_clamp${GUIDE_INFER_FLAGS}
python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${CKPT_STEP} \\
    --inference_type linear \\
    --sampler euler \\
    --num_steps 100 \\
    --hidden_sphere_clamp${GUIDE_SAVE_FLAGS}
echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  cfg=${cfg}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
