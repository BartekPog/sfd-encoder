#!/bin/bash
# =============================================================================
# batch_reground_1p0_hidden_newag_array.sh
#
# SLURM job-array sweep: encode-reground hidden schedule on 2 1p0B hidden
# models with the new B-FT autoguidance. No HRG. CFG sweep over {1.1, 1.3},
# t_fix sweep over {0.5, 0.2}.
#
# Models:
#   1. 1p0_v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5            @ 80000
#   2. 1p0_v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5_warm8k @ 40000
#
# Autoguidance: configs/sfd/autoguidance_b/inference_ft.yaml
#   (b_autoguidance_finetune_no_hidden_warm8k @ 80000)
#
# Schedule: encode-reground + sphere clamp, no --hidden_rep_guidance.
# Reground noise: --reground_fixed_enc_noise (matches existing reground sweeps).
# Sampler:  Euler 100 steps, FID50K balanced.
#
# Note: --cfg_noise_hidden is omitted — it lives only in the pure-CFG branch
# of the reground transport, and is bypassed when an autoguidance model is
# loaded (transport.py:1625-1642).
#
# Total: 2 models × 2 t_fix × 2 cfg = 8 array tasks (max parallel 4)
# =============================================================================

set -euo pipefail

# ---- SLURM settings ----
TIME="00-08:00:00"
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
PER_PROC_BATCH_SIZE=2048
MAX_PARALLEL=16

# ---- Inference settings ----
INFERENCE_OUTPUT_DIR="outputs/inference"
AG_CONFIG="configs/sfd/autoguidance_b/inference_ft.yaml"
SAMPLER="euler"
NUM_STEPS=100

# ---- Models: CONFIG_PATH|TRAIN_EXP_NAME|CKPT_STEP ----
MODELS=(
    "configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml|1p0_v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5|80000"
    # "configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5.yaml|1p0_v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5_warm8k|40000"
)
# T_FIX_VALUES=(0.5 0.2)
T_FIX_VALUES=( 0.1 0.2 0.3 0.4)
CFG_SCALES=(1.1 1.2 1.3 1.4 1.5)

# ---- Build matrix: MODEL_ENTRY|T_FIX|CFG_SCALE ----
MATRIX=()
for MODEL_ENTRY in "${MODELS[@]}"; do
    for T_FIX in "${T_FIX_VALUES[@]}"; do
        for CFG_SCALE in "${CFG_SCALES[@]}"; do
            MATRIX+=("${MODEL_ENTRY}|${T_FIX}|${CFG_SCALE}")
        done
    done
done
NUM_TASKS=${#MATRIX[@]}
LAST_IDX=$((NUM_TASKS - 1))

# ---- Verify checkpoints + AG config ----
echo "============================================="
echo "  Encode-reground hidden sweep — new B-FT autoguidance"
echo "  ${NUM_TASKS} tasks (max parallel ${MAX_PARALLEL})"
echo "  AG config: ${AG_CONFIG}"
echo "  Sampler: ${SAMPLER} ${NUM_STEPS}, FID50K balanced"
echo "============================================="

if [ ! -f "${AG_CONFIG}" ]; then
    echo "ERROR: AG config ${AG_CONFIG} missing"
    exit 1
fi
for ENTRY in "${MATRIX[@]}"; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME CKPT_STEP T_FIX CFG_SCALE <<< "${ENTRY}"
    CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")
    CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"
    if [ ! -f "${CKPT_PATH}" ]; then
        echo "ERROR: missing ${CKPT_PATH}"
        exit 1
    fi
    echo "  task: ${TRAIN_EXP_NAME} @ ${CKPT_STEP} t_fix=${T_FIX} cfg=${CFG_SCALE}"
done

mkdir -p jobs job_outputs

JOBNAME="rg_1p0_newag"
JOBSCRIPT="jobs/${JOBNAME}.sh"
TASKS_FILE="jobs/${JOBNAME}_tasks.tsv"

# Materialize matrix as a TSV the array reads at runtime.
: > "${TASKS_FILE}"
for i in "${!MATRIX[@]}"; do
    printf "%d\t%s\n" "${i}" "${MATRIX[${i}]}" >> "${TASKS_FILE}"
done

cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name ${JOBNAME}
#SBATCH --output job_outputs/${JOBNAME}_%A_%a.out
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}
#SBATCH --array=0-${LAST_IDX}%${MAX_PARALLEL}

set -euo pipefail

LINE=\$(awk -v i="\${SLURM_ARRAY_TASK_ID}" '\$1==i {print; exit}' "${TASKS_FILE}")
ENTRY=\$(echo "\${LINE}" | cut -f2-)
IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME CKPT_STEP T_FIX CFG_SCALE <<< "\${ENTRY}"

CKPT_NAME=\$(printf "%07d" "\${CKPT_STEP}")
CKPT_PATH="outputs/train/\${TRAIN_EXP_NAME}/checkpoints/\${CKPT_NAME}.pt"
INFER_EXP_NAME="\${TRAIN_EXP_NAME}_\${CKPT_NAME}"

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Reground (new AG) task \${SLURM_ARRAY_TASK_ID}: \${TRAIN_EXP_NAME} @ \${CKPT_STEP} t_fix=\${T_FIX} cfg=\${CFG_SCALE}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh \${CONFIG_PATH} \\
    ckpt_path=\${CKPT_PATH} \\
    sample.sampling_method=${SAMPLER} \\
    sample.num_sampling_steps=${NUM_STEPS} \\
    sample.per_proc_batch_size=${PER_PROC_BATCH_SIZE} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    sample.autoguidance=true \\
    sample.autoguidance_config=${AG_CONFIG} \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=\${INFER_EXP_NAME} \\
    --encode_reground_t_fix \${T_FIX} \\
    --reground_fixed_enc_noise \\
    --hidden_sphere_clamp \\
    --cfg_scale \${CFG_SCALE}

python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/\${INFER_EXP_NAME} \\
    --config     \${CONFIG_PATH} \\
    --ckpt_step  \${CKPT_STEP} \\
    --inference_type encodereground \\
    --sampler ${SAMPLER} \\
    --num_steps ${NUM_STEPS} \\
    --hidden_sphere_clamp \\
    --encode_reground_t_fix \${T_FIX} \\
    --reground_fixed_enc_noise \\
    --cfg_scale \${CFG_SCALE} \\
    --autoguidance_config ${AG_CONFIG}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

ARRAY_JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
echo ""
echo "Submitted array job ${ARRAY_JOB_ID} (tasks 0-${LAST_IDX}, %${MAX_PARALLEL})"
echo "Tasks file kept at: ${TASKS_FILE}"
echo "Monitor:  squeue -u \$USER -r"
