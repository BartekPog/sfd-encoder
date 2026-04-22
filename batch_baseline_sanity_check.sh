#!/bin/bash
# =============================================================================
# batch_baseline_sanity_check.sh
#
# FID sanity check on non-hidden baseline models.
# Runs both fast (Euler 100 steps) and best-quality (dopri5 250 steps)
# with CFG 1.5, with and without autoguidance.
#
# Models:
#   1. sfd_1p0                @ 4M   — original SFD 1.0B model
#   2. 1p0_finetune_no_hidden @ 100K — fine-tuned from sfd_1p0, no hidden tokens
#
# Sampler configs:
#   A. Euler  100 steps  — fast, matches our hidden-model eval setup
#   B. dopri5 250 steps  — true upper bound
#
# Guidance configs:
#   a. CFG 1.5 only (standard CFG with null class label)
#   b. CFG 1.5 + autoguidance (B-size AG model @ 70K)
#
# Total: 2 models × 2 samplers × 2 guidance = 8 jobs
# =============================================================================

set -euo pipefail

# ---- SLURM settings ----
TIME="00-06:00:00"
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"

# ---- Inference settings ----
INFERENCE_OUTPUT_DIR="outputs/inference"
CFG_SCALE=1.5
AG_CONFIG="configs/sfd/autoguidance_b/inference.yaml"

# ---- Model definitions ----
# Format: CONFIG_PATH|EXP_NAME|CKPT_PATH
declare -A MODELS=(
    [sfd_1p0]="configs/sfd/lightningdit_1p0/inference_4m.yaml|sfd_1p0|outputs/train/sfd_1p0/checkpoints/4000000.pt"
    [ft_no_hidden]="configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden.yaml|1p0_finetune_no_hidden|outputs/train/1p0_finetune_no_hidden/checkpoints/0100000.pt"
)

# ---- Sampler configs ----
# Format: SAMPLER|NUM_STEPS|BATCH_SIZE
SAMPLER_CONFIGS=(
    "euler|100|512"
    "dopri5|250|64"
)

echo "============================================="
echo "  Baseline sanity check — FID upper bounds"
echo "  CFG ${CFG_SCALE}, FID50K, balanced sampling"
echo "  Euler 100 steps + dopri5 250 steps"
echo "============================================="
echo ""

SUBMITTED=0

for MODEL_KEY in sfd_1p0 ft_no_hidden; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME CKPT_PATH <<< "${MODELS[${MODEL_KEY}]}"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "  SKIP: ${TRAIN_EXP_NAME} — checkpoint ${CKPT_PATH} not found"
        continue
    fi

    CKPT_STEP=$(basename "${CKPT_PATH}" .pt | sed 's/^0*//' | sed 's/^$/0/')

    for SAMPLER_ENTRY in "${SAMPLER_CONFIGS[@]}"; do
        IFS='|' read -r SAMPLER NUM_STEPS BATCH_SIZE <<< "${SAMPLER_ENTRY}"

        for USE_AG in false true; do
            if [ "${USE_AG}" = "true" ]; then
                AG_OVERRIDE=" sample.autoguidance=true sample.autoguidance_config=${AG_CONFIG}"
                AG_SAVE_FLAGS="--autoguidance_config ${AG_CONFIG}"
                AG_TAG="_ag"
            else
                AG_OVERRIDE=""
                AG_SAVE_FLAGS=""
                AG_TAG=""
            fi

            INFER_EXP_NAME="${TRAIN_EXP_NAME}_baseline_sanity"
            LABEL="${TRAIN_EXP_NAME} ${SAMPLER}/${NUM_STEPS} cfg${CFG_SCALE}${AG_TAG}"

            CFG_TAG=$(printf '%.1f' ${CFG_SCALE} | tr '.' 'p')
            JOBNAME="bl_${MODEL_KEY}_${SAMPLER}_s${NUM_STEPS}_cfg${CFG_TAG}${AG_TAG}"
            JOBSCRIPT="jobs/${JOBNAME}.sh"
            OUTPUT="job_outputs/${JOBNAME}.o%J"
            mkdir -p "$(dirname "${JOBSCRIPT}")"
            mkdir -p "$(dirname "${OUTPUT}")"

            cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name ${JOBNAME}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Baseline sanity: ${LABEL}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh ${CONFIG_PATH} \\
    ckpt_path=${CKPT_PATH} \\
    sample.cfg_scale=${CFG_SCALE} \\
    sample.sampling_method=${SAMPLER} \\
    sample.num_sampling_steps=${NUM_STEPS} \\
    sample.per_proc_batch_size=${BATCH_SIZE} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME}${AG_OVERRIDE}

python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${CKPT_STEP} \\
    --inference_type standard \\
    --sampler ${SAMPLER} \\
    --num_steps ${NUM_STEPS} \\
    --cfg_scale ${CFG_SCALE} ${AG_SAVE_FLAGS}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

            echo "  Submitting: ${LABEL}"
            JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
            echo "    -> job ${JOB_ID}"
            rm -f "${JOBSCRIPT}"
            SUBMITTED=$((SUBMITTED + 1))
        done
    done
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} baseline sanity check jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
