#!/bin/bash
# =============================================================================
# batch_run_trajectory_analysis.sh
#
# Collect hidden-token trajectories for reground, recycle, and linear schedules.
# Runs a small number of images (100) per method+t_fix and saves h_clean at
# every ODE step as .npz files for later analysis (cosine similarity heatmaps).
#
# All methods use the same seed so initial noise and class labels are identical.
#
# Usage:
#   bash batch_run_trajectory_analysis.sh [ckpt_step]
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-60000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings ----
TIME=${TIME:-"00-3:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4

# ---- Analysis parameters ----
NUM_IMAGES=100
NUM_STEPS=100
BATCH_SIZE=25
SEED=42
OUTPUT_DIR="outputs/trajectory_analysis"

# ---- Methods and t_fix values ----
METHODS="reground recycle linear"
T_FIX_VALUES="0.7 0.8 0.9 0.95 1.0"

# ---- Reground noise options ----
REGROUND_SHARED_NOISE="--reground_shared_noise"
REGROUND_FIXED_ENC_NOISE="--reground_fixed_enc_noise"

# ---- Experiments ----
# Format: "config_yaml|train_exp_name"
EXPERIMENTS=(
    "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml|v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5"
)

echo "============================================="
echo "  Hidden Token Trajectory Analysis"
echo "  Checkpoint step: ${CKPT_STEP}"
echo "  Methods: ${METHODS}"
echo "  t_fix values: ${T_FIX_VALUES}"
echo "  Images per method: ${NUM_IMAGES}"
echo "  ODE steps: ${NUM_STEPS}"
echo "  Seed: ${SEED}"
echo "============================================="
echo ""

SUBMITTED=0

for ENTRY in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME <<< "${ENTRY}"
    CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "  SKIP: ${TRAIN_EXP_NAME} — checkpoint ${CKPT_PATH} not found"
        continue
    fi

    EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
    JOBSCRIPT="jobs/traj_${EXP_LABEL}_${CKPT_NAME}.sh"
    OUTPUT="job_outputs/traj_${EXP_LABEL}_${CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name traj_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Trajectory analysis: ${EXP_LABEL} @ step ${CKPT_STEP}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

python collect_hidden_trajectories.py \\
    --config ${CONFIG_PATH} \\
    --ckpt_path ${CKPT_PATH} \\
    --methods ${METHODS} \\
    --t_fix_values ${T_FIX_VALUES} \\
    --num_images ${NUM_IMAGES} \\
    --num_steps ${NUM_STEPS} \\
    --batch_size ${BATCH_SIZE} \\
    --output_dir ${OUTPUT_DIR} \\
    --seed ${SEED} \\
    ${REGROUND_SHARED_NOISE} \\
    ${REGROUND_FIXED_ENC_NOISE}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${TRAIN_EXP_NAME}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} trajectory analysis jobs."
echo "  Output: ${OUTPUT_DIR}/"
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="





