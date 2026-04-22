#!/bin/bash
# =============================================================================
# batch_reground_1p0_norepg_scout.sh
#
# Scouting sweep on 1p0_v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5_warm8k
# @ 40K — the XXL hidden model trained WITHOUT training-repg. Since it wasn't
# trained with repg, inference repg is a genuinely novel axis here (no
# double-counting concern).
#
# Three branches:
#   A. cfg=1.0 (no guidance, no AG)               — 3 runs (t_fix sweep)
#   B. cfg=1.4 + no-hidden B-FT AG                — 6 runs (t_fix × repg)
#   C. cfg=1.4 + hidden B repg_1p5 @80K AG        — 3 runs (t_fix sweep, repg=1.0)
# Total: 12 array tasks (max parallel 6)
#
# Model:   1p0_v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5_warm8k @ 40000
# Config:  configs/sfd/hidden_1p0_h200_from_ft/
#            v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5.yaml
# Schedule: encode-reground + sphere clamp + --reground_fixed_enc_noise
# Sampler:  Euler 100 steps, FID50K balanced
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
MAX_PARALLEL=6

# ---- Inference settings ----
INFERENCE_OUTPUT_DIR="outputs/inference"
SAMPLER="euler"
NUM_STEPS=100

# ---- Model under test ----
CONFIG_PATH="configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5.yaml"
TRAIN_EXP_NAME="1p0_v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5_warm8k"
CKPT_STEP=40000

# ---- AG configs ----
AG_NOHIDDEN="configs/sfd/autoguidance_b/inference_ft.yaml"
AG_HIDDEN_80K="configs/sfd/hidden_b_h200_from_ft/inference_ag_repg_1p5_80k.yaml"

# ---- Matrix: AG_CONFIG|CFG|T_FIX|REPG  (AG_CONFIG empty means no AG) ----
MATRIX=(
    # Branch A: cfg=1.0, no AG
    "|1.0|0.1|1.0"
    "|1.0|0.4|1.0"
    "|1.0|0.8|1.0"
    # Branch B: cfg=1.4, no-hidden B FT AG, repg sweep
    "${AG_NOHIDDEN}|1.4|0.1|1.0"
    "${AG_NOHIDDEN}|1.4|0.4|1.0"
    "${AG_NOHIDDEN}|1.4|0.8|1.0"
    "${AG_NOHIDDEN}|1.4|0.1|1.2"
    "${AG_NOHIDDEN}|1.4|0.4|1.2"
    "${AG_NOHIDDEN}|1.4|0.8|1.2"
    # Branch C: cfg=1.4, hidden B @80K AG, repg=1.0
    "${AG_HIDDEN_80K}|1.4|0.1|1.0"
    "${AG_HIDDEN_80K}|1.4|0.4|1.0"
    "${AG_HIDDEN_80K}|1.4|0.8|1.0"
)
NUM_TASKS=${#MATRIX[@]}
LAST_IDX=$((NUM_TASKS - 1))

CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")
CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"

echo "============================================="
echo "  No-repg XXL scout: ${TRAIN_EXP_NAME} @ ${CKPT_STEP}"
echo "  ${NUM_TASKS} tasks (max parallel ${MAX_PARALLEL})"
echo "============================================="

[ -f "${CKPT_PATH}" ] || { echo "ERROR: missing ${CKPT_PATH}"; exit 1; }
for ENTRY in "${MATRIX[@]}"; do
    IFS='|' read -r AG_CONFIG CFG_SCALE T_FIX REPG <<< "${ENTRY}"
    if [ -n "${AG_CONFIG}" ] && [ ! -f "${AG_CONFIG}" ]; then
        echo "ERROR: missing AG config ${AG_CONFIG}"
        exit 1
    fi
    echo "  task: ag=${AG_CONFIG:-none} cfg=${CFG_SCALE} t_fix=${T_FIX} repg=${REPG}"
done

mkdir -p jobs job_outputs

JOBNAME="rg_1p0_norepg_scout"
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
IFS='|' read -r AG_CONFIG CFG_SCALE T_FIX REPG <<< "\${ENTRY}"

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "No-repg scout task \${SLURM_ARRAY_TASK_ID}: ag=\${AG_CONFIG:-none} cfg=\${CFG_SCALE} t_fix=\${T_FIX} repg=\${REPG}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

# Build conditional flags
AG_INFER_OVERRIDE=""
AG_SAVE_FLAGS=""
if [ -n "\${AG_CONFIG}" ]; then
    AG_INFER_OVERRIDE=" sample.autoguidance=true sample.autoguidance_config=\${AG_CONFIG}"
    AG_SAVE_FLAGS=" --autoguidance_config \${AG_CONFIG}"
fi

REPG_INFER_FLAGS=""
REPG_SAVE_FLAGS=""
if (( \$(echo "\${REPG} > 1.0" | bc -l) )); then
    REPG_INFER_FLAGS=" --hidden_rep_guidance \${REPG}"
    REPG_SAVE_FLAGS=" --hidden_rep_guidance \${REPG}"
fi

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh ${CONFIG_PATH} \\
    ckpt_path=${CKPT_PATH} \\
    sample.sampling_method=${SAMPLER} \\
    sample.num_sampling_steps=${NUM_STEPS} \\
    sample.per_proc_batch_size=${PER_PROC_BATCH_SIZE} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME}\${AG_INFER_OVERRIDE} \\
    --encode_reground_t_fix \${T_FIX} \\
    --reground_fixed_enc_noise \\
    --hidden_sphere_clamp \\
    --cfg_scale \${CFG_SCALE}\${REPG_INFER_FLAGS}

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
    --cfg_scale \${CFG_SCALE}\${REPG_SAVE_FLAGS}\${AG_SAVE_FLAGS}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

ARRAY_JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
echo ""
echo "Submitted array job ${ARRAY_JOB_ID} (tasks 0-${LAST_IDX}, %${MAX_PARALLEL})"
echo "Tasks file: ${TASKS_FILE}"
echo "Monitor:  squeue -u \$USER -r"
