#!/bin/bash
# =============================================================================
# batch_reground_1p0_hidden_ag_array.sh
#
# Hidden-aware autoguidance experiment: use the B-size hidden model
# (v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5, same recipe) as the negative
# pass, instead of the no-hidden B-FT AG. Tests whether matching the hidden
# conditioning in both passes lets us push t_fix higher (so hidden tokens
# actually contribute) without saturation.
#
# Model: 1p0_v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5 @ 80000
# AG candidates (B repg_1p5): @ 40K (mid-training) and @ 80K (latest)
#   configs/sfd/hidden_b_h200_from_ft/inference_ag_repg_1p5_{40k,80k}.yaml
#
# Sweep: 2 AG ckpts × 3 t_fix {0.1, 0.3, 0.5} × 2 cfg {1.3, 1.4} = 12 tasks
# Schedule: encode-reground + sphere clamp + --reground_fixed_enc_noise
# Sampler:  Euler 100 steps, FID50K balanced
#
# Key prediction: if AG asymmetry was the bottleneck, the optimal t_fix should
# move upward from 0.1 with hidden-aware AG. If t_fix=0.1 still wins, either
# the encoder isn't producing useful h or training-repg saturation leaks
# through.
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
SAMPLER="euler"
NUM_STEPS=100

# ---- Model under test ----
CONFIG_PATH="configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml"
TRAIN_EXP_NAME="1p0_v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5"
CKPT_STEP=80000

# ---- AG candidates: AG_CONFIG|AG_TAG ----
AG_CONFIGS=(
    "configs/sfd/hidden_b_h200_from_ft/inference_ag_repg_1p5_40k.yaml|bhrepg40k"
    "configs/sfd/hidden_b_h200_from_ft/inference_ag_repg_1p5_80k.yaml|bhrepg80k"
)
T_FIX_VALUES=(0.1 0.3 0.5)
CFG_SCALES=(1.3 1.4)

# ---- Build matrix: AG_CONFIG|AG_TAG|T_FIX|CFG_SCALE ----
MATRIX=()
for AG_ENTRY in "${AG_CONFIGS[@]}"; do
    for T_FIX in "${T_FIX_VALUES[@]}"; do
        for CFG_SCALE in "${CFG_SCALES[@]}"; do
            MATRIX+=("${AG_ENTRY}|${T_FIX}|${CFG_SCALE}")
        done
    done
done
NUM_TASKS=${#MATRIX[@]}
LAST_IDX=$((NUM_TASKS - 1))

CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")
CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"

echo "============================================="
echo "  Hidden-aware AG sweep on ${TRAIN_EXP_NAME} @ ${CKPT_STEP}"
echo "  ${NUM_TASKS} tasks (max parallel ${MAX_PARALLEL})"
echo "============================================="

[ -f "${CKPT_PATH}" ] || { echo "ERROR: missing ${CKPT_PATH}"; exit 1; }
for ENTRY in "${MATRIX[@]}"; do
    IFS='|' read -r AG_CONFIG AG_TAG T_FIX CFG_SCALE <<< "${ENTRY}"
    [ -f "${AG_CONFIG}" ] || { echo "ERROR: missing AG config ${AG_CONFIG}"; exit 1; }
    echo "  task: ag=${AG_TAG} t_fix=${T_FIX} cfg=${CFG_SCALE}"
done

mkdir -p jobs job_outputs

JOBNAME="rg_1p0_hag"
JOBSCRIPT="jobs/${JOBNAME}.sh"
TASKS_FILE="jobs/${JOBNAME}_tasks.tsv"

: > "${TASKS_FILE}"
for i in "${!MATRIX[@]}"; do
    printf "%d\t%s\n" "${i}" "${MATRIX[${i}]}" >> "${TASKS_FILE}"
done

INFER_EXP_NAME="${TRAIN_EXP_NAME}_${CKPT_NAME}"

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
IFS='|' read -r AG_CONFIG AG_TAG T_FIX CFG_SCALE <<< "\${ENTRY}"

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Hidden-AG task \${SLURM_ARRAY_TASK_ID}: ag=\${AG_TAG} t_fix=\${T_FIX} cfg=\${CFG_SCALE}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh ${CONFIG_PATH} \\
    ckpt_path=${CKPT_PATH} \\
    sample.sampling_method=${SAMPLER} \\
    sample.num_sampling_steps=${NUM_STEPS} \\
    sample.per_proc_batch_size=${PER_PROC_BATCH_SIZE} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    sample.autoguidance=true \\
    sample.autoguidance_config=\${AG_CONFIG} \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME} \\
    --encode_reground_t_fix \${T_FIX} \\
    --reground_fixed_enc_noise \\
    --hidden_sphere_clamp \\
    --cfg_scale \${CFG_SCALE}

python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${CKPT_STEP} \\
    --inference_type encodereground \\
    --sampler ${SAMPLER} \\
    --num_steps ${NUM_STEPS} \\
    --hidden_sphere_clamp \\
    --encode_reground_t_fix \${T_FIX} \\
    --reground_fixed_enc_noise \\
    --cfg_scale \${CFG_SCALE} \\
    --autoguidance_config \${AG_CONFIG}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

ARRAY_JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
echo ""
echo "Submitted array job ${ARRAY_JOB_ID} (tasks 0-${LAST_IDX}, %${MAX_PARALLEL})"
echo "Tasks file: ${TASKS_FILE}"
echo "Monitor:  squeue -u \$USER -r"
