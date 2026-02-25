#!/bin/bash
# =============================================================================
# batch_run_fid_comparison_v2_base.sh
#
# FID50K comparison across checkpoints:
#   - v2_base_mse02        (H8, shared t-emb, MSE 0.2, linear hidden schedule)
#   - v2_finetune_no_hidden (baseline, no hidden tokens)
#
# Evaluated at steps: 10K, 30K, 50K, 100K
# Sampler: Euler, 100 steps, cfg_scale=1.0, balanced sampling
#
# Usage:
#   bash batch_run_fid_comparison_v2_base.sh
# =============================================================================

set -euo pipefail

# ---- SLURM settings ----
TIME=${TIME:-"00-14:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
INFERENCE_OUTPUT_DIR="outputs/inference"

# ---- Checkpoints to evaluate ----
CKPT_STEPS=(20000 40000 60000) #80000 100000

# ---- Experiment definitions ----
# Format: "config_yaml|train_exp_name|use_hidden_schedule"
# use_hidden_schedule: "yes" passes --hidden_schedule linear, "no" skips it
EXPERIMENTS=(
    "configs/sfd/hidden_b_h200/v2_finetune_no_hidden.yaml|v2_finetune_no_hidden|no"
    "configs/sfd/hidden_b_h200/v2_base_mse02.yaml|v2_base_mse02|yes"
)

echo "============================================="
echo "  V2 FID comparison: base_mse02 vs finetune_no_hidden"
echo "  Checkpoints: ${CKPT_STEPS[*]}"
echo "  Sampler: Euler, 100 steps"
echo "  GPUs: ${NUM_GPUS} x H200"
echo "============================================="
echo ""

SUBMITTED=0

for CKPT_STEP in "${CKPT_STEPS[@]}"; do
    CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")
    echo "--- Checkpoint ${CKPT_NAME} ---"

    for ENTRY in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME USE_HIDDEN_SCHED <<< "${ENTRY}"
        CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"

        if [ ! -f "${CKPT_PATH}" ]; then
            echo "  SKIP: ${TRAIN_EXP_NAME} @ ${CKPT_NAME} — checkpoint not found"
            continue
        fi

        INFER_EXP_NAME="${TRAIN_EXP_NAME}_${CKPT_NAME}"
        EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
        JOBSCRIPT="jobs/infer_fid_cmp_${EXP_LABEL}_${CKPT_NAME}.sh"
        OUTPUT="job_outputs/infer_fid_cmp_${EXP_LABEL}_${CKPT_NAME}.o%J"
        mkdir -p "$(dirname "${JOBSCRIPT}")"
        mkdir -p "$(dirname "${OUTPUT}")"

        # Build hidden schedule arg (only for hidden-token models)
        if [ "${USE_HIDDEN_SCHED}" = "yes" ]; then
            HIDDEN_SCHED_ARG="--hidden_schedule linear"
        else
            HIDDEN_SCHED_ARG=""
        fi

        cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name fid_cmp_${EXP_LABEL}_${CKPT_NAME}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "FID comparison: ${EXP_LABEL} @ step ${CKPT_STEP}"

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
    ${HIDDEN_SCHED_ARG}
python save_fid_result.py \
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \
    --config     ${CONFIG_PATH} \
    --ckpt_step  ${CKPT_STEP} \
    --inference_type linear \
    --sampler euler \
    --num_steps 100
echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

        JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
        echo "  ${TRAIN_EXP_NAME} @ ${CKPT_NAME}: submitted job ${JOB_ID}"
        rm -f "${JOBSCRIPT}"
        SUBMITTED=$((SUBMITTED + 1))
    done
    echo ""
done

echo "============================================="
echo "  Submitted ${SUBMITTED} inference jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
