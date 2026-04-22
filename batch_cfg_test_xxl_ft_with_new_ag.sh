#!/bin/bash
# =============================================================================
# batch_cfg_test_xxl_ft_with_new_ag.sh
#
# Three-way comparison on the FT'd XXL (1p0_finetune_no_hidden_warm8k):
#   a. No guidance (cfg=1.0)
#   b. Regular CFG 1.5
#   c. CFG 1.5 + autoguidance using the NEW FT'd B model
#      (b_autoguidance_finetune_no_hidden_warm8k @ 100K)
#
# Sampler: Euler 100 steps, FID50K balanced.
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
AG_CONFIG="configs/sfd/autoguidance_b/inference_ft.yaml"
SAMPLER="euler"
NUM_STEPS=100

# ---- Model: FT'd XXL ----
XXL_CONFIG="configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden_warm8k.yaml"
XXL_TRAIN_EXP="1p0_finetune_no_hidden_warm8k"
XXL_CKPT=${XXL_CKPT:-"outputs/train/${XXL_TRAIN_EXP}/checkpoints/0100000.pt"}

# ---- Guidance configs ----
# Format: CFG_SCALE|USE_AG|LABEL
GUIDANCE_CONFIGS=(
    "1.0|false|no_cfg"
    "1.5|false|cfg1p5"
    "1.5|true|cfg1p5_ag_ft"
)

if [ ! -f "${XXL_CKPT}" ]; then
    echo "ERROR: XXL checkpoint ${XXL_CKPT} not found"
    exit 1
fi

CKPT_STEP=$(basename "${XXL_CKPT}" .pt | sed 's/^0*//' | sed 's/^$/0/')

echo "============================================="
echo "  XXL FT (${XXL_TRAIN_EXP} @ step ${CKPT_STEP})"
echo "  AG (when used): ${AG_CONFIG}"
echo "  Sampler: ${SAMPLER} ${NUM_STEPS} steps, FID50K balanced"
echo "============================================="

mkdir -p jobs job_outputs
SUBMITTED=0

for GUIDE_ENTRY in "${GUIDANCE_CONFIGS[@]}"; do
    IFS='|' read -r CFG_SCALE USE_AG GUIDE_LABEL <<< "${GUIDE_ENTRY}"

    AG_OVERRIDE=""
    AG_SAVE_FLAGS=""
    if [ "${USE_AG}" = "true" ]; then
        AG_OVERRIDE=" sample.autoguidance=true sample.autoguidance_config=${AG_CONFIG}"
        AG_SAVE_FLAGS="--autoguidance_config ${AG_CONFIG}"
    fi

    INFER_EXP_NAME="${XXL_TRAIN_EXP}_cfg_test_newag"
    LABEL="${XXL_TRAIN_EXP} ${GUIDE_LABEL}"

    JOBNAME="cfgtest_xxl_newag_${GUIDE_LABEL}"
    JOBSCRIPT="jobs/${JOBNAME}.sh"
    OUTPUT="job_outputs/${JOBNAME}.o%J"

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
echo "CFG test (XXL FT, new AG): ${LABEL}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh ${XXL_CONFIG} \\
    ckpt_path=${XXL_CKPT} \\
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
    --config     ${XXL_CONFIG} \\
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

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} XXL-FT inference jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
