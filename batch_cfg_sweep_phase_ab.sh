#!/bin/bash
# =============================================================================
# batch_cfg_sweep_phase_ab.sh
#
# Phase A+B: fine CFG sweep across B-size hidden-token models as a SLURM
# job array with a configurable concurrency cap.
#
# Phase A — 5 repg-trained models × CFG {1.2-1.6} × cfg_noise_hidden {false,true}
#   Models:
#     v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2
#     v4_mse0001_noisy_enc_nocurr_shift1p5_repg_2
#     v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p2
#     v4_mse0001_noisy_enc_nocurr_shift2_repg_1p5
#     v4_mse0001_noisy_enc_nocurr_shift3_repg_1p5
#   = 5 × 5 × 2 = 50 jobs
#
# Phase B — no-repg hgd_5 × same CFG/noise grid = 10 jobs
#
# Total: 60 array tasks. Concurrency cap: MAX_CONCURRENT (default 8).
#
# Usage:
#   bash batch_cfg_sweep_phase_ab.sh
#   MAX_CONCURRENT=4 bash batch_cfg_sweep_phase_ab.sh
# =============================================================================

set -euo pipefail

CKPT_STEP=60000
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")
T_FIX=0.88
T_FIX_TAG=$(printf "%.2f" "${T_FIX}" | tr -d '.')

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

PHASE_A_MODELS=(
    "v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2"
    "v4_mse0001_noisy_enc_nocurr_shift1p5_repg_2"
    "v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p2"
    "v4_mse0001_noisy_enc_nocurr_shift2_repg_1p5"
    "v4_mse0001_noisy_enc_nocurr_shift3_repg_1p5"
)
PHASE_B_MODEL="v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5"

CFG_VALUES=(1.2 1.3 1.4 1.5 1.6)
NOISE_H_VALUES=(false true)

# ---- Build task table ----
# Each line: MODEL|CFG|NOISE_H
TASKS=()
for model in "${PHASE_A_MODELS[@]}" "${PHASE_B_MODEL}"; do
    ckpt="outputs/train/${model}/checkpoints/${CKPT_NAME}.pt"
    if [ ! -f "${ckpt}" ]; then
        echo "SKIP: ${model} — checkpoint ${ckpt} not found"
        continue
    fi
    for cfg in "${CFG_VALUES[@]}"; do
        for noiseh in "${NOISE_H_VALUES[@]}"; do
            TASKS+=("${model}|${cfg}|${noiseh}")
        done
    done
done

NUM_TASKS=${#TASKS[@]}
if [ "${NUM_TASKS}" -eq 0 ]; then
    echo "No tasks — all checkpoints missing?"
    exit 1
fi

echo "============================================="
echo "  Phase A+B CFG sweep (job array)"
echo "  Tasks: ${NUM_TASKS}   Concurrency cap: ${MAX_CONCURRENT}"
echo "  t_fix=${T_FIX}, ckpt=${CKPT_STEP}, Euler ${NUM_STEPS} steps"
echo "============================================="

# ---- Write task table to disk so the array job can read it ----
ARRAY_DIR="jobs/cfg_sweep_phase_ab"
mkdir -p "${ARRAY_DIR}" job_outputs
TABLE="${ARRAY_DIR}/tasks.tsv"
: > "${TABLE}"
for i in "${!TASKS[@]}"; do
    echo -e "${i}\t${TASKS[$i]}" >> "${TABLE}"
done
echo "Task table: ${TABLE}"

JOBSCRIPT="${ARRAY_DIR}/array.sh"
OUTPUT="job_outputs/cfg_sweep_phase_ab_%A_%a.out"

cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name cfgsweep_ab
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}
#SBATCH --array=0-$((NUM_TASKS - 1))%${MAX_CONCURRENT}

set -euo pipefail

TABLE="${TABLE}"
LINE=\$(awk -v i="\${SLURM_ARRAY_TASK_ID}" '\$1==i {print; exit}' "\${TABLE}")
MODEL=\$(echo "\${LINE}" | cut -f2 | cut -d'|' -f1)
CFG_SCALE=\$(echo "\${LINE}" | cut -f2 | cut -d'|' -f2)
CFG_NOISE_HIDDEN=\$(echo "\${LINE}" | cut -f2 | cut -d'|' -f3)

CONFIG_PATH="${CONFIG_DIR}/\${MODEL}.yaml"
CKPT_PATH="outputs/train/\${MODEL}/checkpoints/${CKPT_NAME}.pt"
INFER_EXP_NAME="\${MODEL}_${CKPT_NAME}"

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Task \${SLURM_ARRAY_TASK_ID}: model=\${MODEL} cfg=\${CFG_SCALE} noise_h=\${CFG_NOISE_HIDDEN}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

EXTRA_INFER_FLAGS=" --reground_fixed_enc_noise"
EXTRA_SAVE_FLAGS=" --reground_fixed_enc_noise"
if [ "\${CFG_NOISE_HIDDEN}" = "true" ]; then
    EXTRA_INFER_FLAGS+=" --cfg_noise_hidden"
    EXTRA_SAVE_FLAGS+=" --cfg_noise_hidden"
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
    --encode_reground_t_fix ${T_FIX} \\
    --hidden_sphere_clamp \\
    --cfg_scale \${CFG_SCALE}\${EXTRA_INFER_FLAGS}

python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/\${INFER_EXP_NAME} \\
    --config     \${CONFIG_PATH} \\
    --ckpt_step  ${CKPT_STEP} \\
    --inference_type encodereground \\
    --sampler euler \\
    --num_steps ${NUM_STEPS} \\
    --hidden_sphere_clamp \\
    --encode_reground_t_fix ${T_FIX} \\
    --cfg_scale \${CFG_SCALE}\${EXTRA_SAVE_FLAGS}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

ARRAY_JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
echo ""
echo "Submitted array job ${ARRAY_JOB_ID} (${NUM_TASKS} tasks, max ${MAX_CONCURRENT} concurrent)."
echo "Monitor:  squeue -u \$USER -r"
echo "Cancel all: scancel ${ARRAY_JOB_ID}"
