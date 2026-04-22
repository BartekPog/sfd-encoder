#!/bin/bash
# =============================================================================
# batch_cfg_test_1p0_finetune.sh
#
# Test whether CFG effectiveness differs between the no-warmup and warmup
# finetunes of the 1p0 model (both without hidden tokens).
#
# Models (both at 60K FT steps):
#   1. finetune_no_hidden       — from EMA-only checkpoint, no warmup
#   2. finetune_no_hidden_warm8k — from full checkpoint, 8K LR warmup
#
# Guidance configs:
#   a. No CFG (cfg=1.0)
#   b. CFG 1.5
#   c. CFG 1.5 + autoguidance (B-size AG model)
#
# Sampler: Euler 100 steps (matches hidden-model eval setup)
#
# Total: 2 models × 3 guidance = 6 jobs
# =============================================================================

set -euo pipefail

# ---- SLURM settings ----
TIME="00-06:00:00"
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
BATCH_SIZE=512

# ---- Inference settings ----
INFERENCE_OUTPUT_DIR="outputs/inference"
AG_CONFIG="configs/sfd/autoguidance_b/inference.yaml"
SAMPLER="euler"
NUM_STEPS=100

# ---- Model definitions ----
# Format: CONFIG_PATH|EXP_NAME|CKPT_PATH
declare -A MODELS=(
    [ft_no_warmup]="configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden.yaml|1p0_finetune_no_hidden|outputs/train/1p0_finetune_no_hidden/checkpoints/0060000.pt"
    [ft_warm8k]="configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden_warm8k.yaml|1p0_finetune_no_hidden_warm8k|outputs/train/1p0_finetune_no_hidden_warm8k/checkpoints/0060000.pt"
)

# ---- Guidance configs ----
# Format: CFG_SCALE|USE_AG|LABEL
GUIDANCE_CONFIGS=(
    "1.0|false|no_cfg"
    "1.5|false|cfg1p5"
    "1.5|true|cfg1p5_ag"
)

echo "============================================="
echo "  CFG test on 1p0 finetune models (no hidden)"
echo "  Sampler: ${SAMPLER} ${NUM_STEPS} steps"
echo "  FID50K, balanced sampling"
echo "============================================="
echo ""

SUBMITTED=0

for MODEL_KEY in ft_no_warmup ft_warm8k; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME CKPT_PATH <<< "${MODELS[${MODEL_KEY}]}"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "  SKIP: ${TRAIN_EXP_NAME} — checkpoint ${CKPT_PATH} not found"
        continue
    fi

    CKPT_STEP=$(basename "${CKPT_PATH}" .pt | sed 's/^0*//' | sed 's/^$/0/')

    for GUIDE_ENTRY in "${GUIDANCE_CONFIGS[@]}"; do
        IFS='|' read -r CFG_SCALE USE_AG GUIDE_LABEL <<< "${GUIDE_ENTRY}"

        # Build guidance flags
        AG_OVERRIDE=""
        AG_SAVE_FLAGS=""
        if [ "${USE_AG}" = "true" ]; then
            AG_OVERRIDE=" sample.autoguidance=true sample.autoguidance_config=${AG_CONFIG}"
            AG_SAVE_FLAGS="--autoguidance_config ${AG_CONFIG}"
        fi

        INFER_EXP_NAME="${TRAIN_EXP_NAME}_cfg_test"
        LABEL="${TRAIN_EXP_NAME} ${GUIDE_LABEL}"

        JOBNAME="cfgtest_${MODEL_KEY}_${GUIDE_LABEL}"
        JOBSCRIPT="jobs/${JOBNAME}.sh"
        OUTPUT="job_outputs/${JOBNAME}.o%J"
        mkdir -p jobs job_outputs

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
echo "CFG test: ${LABEL}"

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

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} CFG test jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
