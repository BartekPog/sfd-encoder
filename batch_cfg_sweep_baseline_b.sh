#!/bin/bash
# =============================================================================
# batch_cfg_sweep_baseline_b.sh
#
# CFG sweep on the B-size no-hidden baseline (v2_finetune_no_hidden) to
# establish what CFG alone can do without hidden tokens.
#
# Model: v2_finetune_no_hidden @ 1600K
#   (1540K base + 60K more training = same total steps as hidden models @ 60K FT)
#
# Existing results (no CFG, euler 100):
#   1600K: FID ~8.79
#
# Sweep: CFG {1.0, 1.3, 1.5, 2.0, 2.5, 3.0}
# Sampler: Euler 100 steps (matches hidden-model eval)
#
# Total: 6 jobs
# =============================================================================

set -euo pipefail

CKPT_STEP=1600000
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings ----
TIME="00-06:00:00"
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
BATCH_SIZE=${PER_PROC_BATCH_SIZE:-512}

# ---- Inference settings ----
INFERENCE_OUTPUT_DIR="outputs/inference"
CONFIG_PATH="configs/sfd/hidden_b_h200/v2_finetune_no_hidden.yaml"
TRAIN_EXP_NAME="v2_finetune_no_hidden"
CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"
SAMPLER="euler"
NUM_STEPS=100

if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: checkpoint ${CKPT_PATH} not found"
    exit 1
fi

echo "============================================="
echo "  CFG sweep on B-size no-hidden baseline"
echo "  Model: ${TRAIN_EXP_NAME} @ ${CKPT_STEP}"
echo "  Sampler: ${SAMPLER} ${NUM_STEPS} steps"
echo "  FID50K, balanced sampling"
echo "============================================="
echo ""

SUBMITTED=0
INFER_EXP_NAME="${TRAIN_EXP_NAME}_cfg_sweep"

# for CFG_SCALE in 1.0 1.3 1.5 2.0 2.5 3.0; do
for CFG_SCALE in 1.1 1.2 1.3 1.4 1.6; do
    CFG_TAG=$(printf '%.1f' ${CFG_SCALE} | tr '.' 'p')
    LABEL="${TRAIN_EXP_NAME} cfg=${CFG_SCALE}"

    JOBNAME="cfgsweep_baseline_b_cfg${CFG_TAG}"
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
echo "CFG sweep baseline: ${LABEL}"

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
    train.exp_name=${INFER_EXP_NAME}

python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${CKPT_STEP} \\
    --inference_type standard \\
    --sampler ${SAMPLER} \\
    --num_steps ${NUM_STEPS} \\
    --cfg_scale ${CFG_SCALE}

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
echo "  Submitted ${SUBMITTED} baseline CFG sweep jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="



