#!/bin/bash
# =============================================================================
# batch_linear_repg_sweep.sh
#
# Follow-up on the hgd_5 linear-schedule finding (CFG=1.3 + repg=1.3
# looked strong). Submits two dependent job arrays:
#
#   Array 1 — hgd_5 CFG refinement around repg=1.3 (runs first)
#       Model: v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5
#       CFG {1.2, 1.4, 1.5} × repg 1.3
#       = 3 jobs
#
#   Array 2 — transfer to other models (runs after Array 1 finishes)
#       Models:
#         v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5
#         v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2
#         v4_mse0001_noisy_enc_nocurr_shift1p5_repg_2
#         v4_mse0001_noisy_enc_nocurr_shift2_repg_1p5
#       CFG=1.3 × repg {1.0, 1.1, 1.3}
#       = 4 × 3 = 12 jobs
#
# Total: 15 tasks across two arrays.
# Array 2 uses --dependency=afterany:<array1_id>.
# Concurrency cap per array: MAX_CONCURRENT (default 16).
#
# All runs: Euler 100 steps, FID50K, balanced, ckpt=60K,
# --hidden_schedule linear --hidden_sphere_clamp.
# =============================================================================

set -euo pipefail

CKPT_STEP=60000
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings ----
TIME=${TIME:-"00-04:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-1024}
MAX_CONCURRENT=${MAX_CONCURRENT:-16}

INFERENCE_OUTPUT_DIR="outputs/inference"
CONFIG_DIR="configs/sfd/hidden_b_h200_from_ft"
NUM_STEPS=100

# ---- Array 1: hgd_5 CFG refinement around repg=1.3 ----
REFINE_MODEL="v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5"
REFINE_CFGS=(1.2 1.4 1.5)
REFINE_REPG=1.3

# ---- Array 2: transfer check ----
TRANSFER_MODELS=(
    "v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5"
    "v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2"
    "v4_mse0001_noisy_enc_nocurr_shift1p5_repg_2"
    "v4_mse0001_noisy_enc_nocurr_shift2_repg_1p5"
)
TRANSFER_CFG=1.3
TRANSFER_REPG=(1.0 1.1 1.3)

ARRAY_DIR="jobs/linear_repg_sweep"
mkdir -p "${ARRAY_DIR}" job_outputs

# Shared array job body — reads MODEL|CFG|REPG from tasks.tsv
build_array_script() {
    local jobscript=$1 output=$2 num_tasks=$3 table=$4 jobname=$5 dep=$6
    local dep_line=""
    if [ -n "${dep}" ]; then
        dep_line="#SBATCH --dependency=afterany:${dep}"
    fi
    cat > "${jobscript}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name ${jobname}
#SBATCH --output ${output}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}
#SBATCH --array=0-$((num_tasks - 1))%${MAX_CONCURRENT}
${dep_line}

set -euo pipefail

TABLE="${table}"
LINE=\$(awk -v i="\${SLURM_ARRAY_TASK_ID}" '\$1==i {print; exit}' "\${TABLE}")
MODEL=\$(echo "\${LINE}" | cut -f2 | cut -d'|' -f1)
CFG_SCALE=\$(echo "\${LINE}" | cut -f2 | cut -d'|' -f2)
HRG=\$(echo "\${LINE}" | cut -f2 | cut -d'|' -f3)

CONFIG_PATH="${CONFIG_DIR}/\${MODEL}.yaml"
CKPT_PATH="outputs/train/\${MODEL}/checkpoints/${CKPT_NAME}.pt"
INFER_EXP_NAME="\${MODEL}_${CKPT_NAME}"

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Task \${SLURM_ARRAY_TASK_ID}: model=\${MODEL} cfg=\${CFG_SCALE} repg=\${HRG}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GUIDE_INFER_FLAGS=" --cfg_scale \${CFG_SCALE}"
GUIDE_SAVE_FLAGS=" --cfg_scale \${CFG_SCALE}"
if (( \$(echo "\${HRG} > 1.0" | bc -l) )); then
    GUIDE_INFER_FLAGS+=" --hidden_rep_guidance \${HRG}"
    GUIDE_SAVE_FLAGS+=" --hidden_rep_guidance \${HRG}"
fi

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh \${CONFIG_PATH} \\
    ckpt_path=\${CKPT_PATH} \\
    sample.sampling_method=euler \\
    sample.num_sampling_steps=${NUM_STEPS} \\
    sample.per_proc_batch_size=${PER_PROC_BATCH_SIZE} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=\${INFER_EXP_NAME} \\
    --hidden_schedule linear \\
    --hidden_sphere_clamp\${GUIDE_INFER_FLAGS}

python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/\${INFER_EXP_NAME} \\
    --config     \${CONFIG_PATH} \\
    --ckpt_step  ${CKPT_STEP} \\
    --inference_type linear \\
    --sampler euler \\
    --num_steps ${NUM_STEPS} \\
    --hidden_sphere_clamp\${GUIDE_SAVE_FLAGS}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF
}

# ---- Build refine task table ----
REFINE_TASKS=()
ckpt="outputs/train/${REFINE_MODEL}/checkpoints/${CKPT_NAME}.pt"
if [ ! -f "${ckpt}" ]; then
    echo "ERROR: refine ckpt ${ckpt} missing"
    exit 1
fi
for cfg in "${REFINE_CFGS[@]}"; do
    REFINE_TASKS+=("${REFINE_MODEL}|${cfg}|${REFINE_REPG}")
done
REFINE_TABLE="${ARRAY_DIR}/refine_tasks.tsv"
: > "${REFINE_TABLE}"
for i in "${!REFINE_TASKS[@]}"; do
    echo -e "${i}\t${REFINE_TASKS[$i]}" >> "${REFINE_TABLE}"
done

# ---- Build transfer task table ----
TRANSFER_TASKS=()
for model in "${TRANSFER_MODELS[@]}"; do
    ckpt="outputs/train/${model}/checkpoints/${CKPT_NAME}.pt"
    if [ ! -f "${ckpt}" ]; then
        echo "SKIP: ${model} — ckpt ${ckpt} not found"
        continue
    fi
    for repg in "${TRANSFER_REPG[@]}"; do
        TRANSFER_TASKS+=("${model}|${TRANSFER_CFG}|${repg}")
    done
done
TRANSFER_TABLE="${ARRAY_DIR}/transfer_tasks.tsv"
: > "${TRANSFER_TABLE}"
for i in "${!TRANSFER_TASKS[@]}"; do
    echo -e "${i}\t${TRANSFER_TASKS[$i]}" >> "${TRANSFER_TABLE}"
done

NUM_REFINE=${#REFINE_TASKS[@]}
NUM_TRANSFER=${#TRANSFER_TASKS[@]}

echo "============================================="
echo "  Linear × CFG × repg sweep (2 dependent job arrays)"
echo "  Refine tasks:   ${NUM_REFINE}"
echo "  Transfer tasks: ${NUM_TRANSFER}"
echo "  Concurrency cap: ${MAX_CONCURRENT}"
echo "============================================="

# ---- Submit Array 1 (refine) ----
REFINE_SCRIPT="${ARRAY_DIR}/refine_array.sh"
build_array_script "${REFINE_SCRIPT}" \
    "job_outputs/linear_repg_refine_%A_%a.out" \
    "${NUM_REFINE}" "${REFINE_TABLE}" "linrepg_refine" ""
REFINE_ID=$(sbatch --parsable "${REFINE_SCRIPT}")
echo "Array 1 (refine) submitted: ${REFINE_ID}"

# ---- Submit Array 2 (transfer), dependent on Array 1 ----
if [ "${NUM_TRANSFER}" -gt 0 ]; then
    TRANSFER_SCRIPT="${ARRAY_DIR}/transfer_array.sh"
    build_array_script "${TRANSFER_SCRIPT}" \
        "job_outputs/linear_repg_transfer_%A_%a.out" \
        "${NUM_TRANSFER}" "${TRANSFER_TABLE}" "linrepg_transfer" "${REFINE_ID}"
    TRANSFER_ID=$(sbatch --parsable "${TRANSFER_SCRIPT}")
    echo "Array 2 (transfer) submitted: ${TRANSFER_ID} (afterany:${REFINE_ID})"
fi

echo ""
echo "Monitor:  squeue -u \$USER -r"
echo "Cancel all: scancel ${REFINE_ID}${TRANSFER_ID:+ ${TRANSFER_ID}}"
