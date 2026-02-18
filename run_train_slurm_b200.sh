#!/bin/bash
# =============================================================================
# run_train_slurm_b200.sh — Submit a training job (or chain of jobs) to SLURM
#                            on the H200 cluster (DAIS)
#
# Usage:
#   bash run_train_slurm_b200.sh <config_path> [num_chains] [num_gpus]
#
# Arguments:
#   config_path  — path to YAML config
#   num_chains   — number of chained jobs (default: 6)
#   num_gpus     — number of H200 GPUs (default: 4)
#
# Each job resumes from the latest checkpoint. The config must have resume: true.
# =============================================================================

set -euo pipefail

CONFIG_PATH=${1:?'Usage: bash run_train_slurm_b200.sh <config_path> [num_chains] [num_gpus]'}
NUM_CHAINS=${2:-6}
NUM_GPUS=${3:-4}

# ---- SLURM settings (B200 cluster / DAIS) ----
# No partition flag — uses default
TIME=${TIME:-"00-06:00:00"}
GPUS="b200:${NUM_GPUS}"
MEM="350G"
CPUS_PER_TASK=4
PRECISION="bf16"
DEPENDENCY_TYPE=${DEPENDENCY_TYPE:-afterany}

EXP_NAME="b200$(basename "${CONFIG_PATH}" .yaml)"

echo "============================================="
echo "  Training: ${EXP_NAME}"
echo "  Config:   ${CONFIG_PATH}"
echo "  Chains:   ${NUM_CHAINS} x ${TIME}"
echo "  GPUs:     ${NUM_GPUS} x B200"
echo "============================================="

PREV_JOB_ID=""

for i in $(seq 1 "${NUM_CHAINS}"); do
    JOBSCRIPT="jobs/train_${EXP_NAME}_chain${i}.sh"
    OUTPUT="job_outputs/train_${EXP_NAME}_chain${i}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name ${EXP_NAME}_c${i}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Chain ${i}/${NUM_CHAINS} for ${EXP_NAME}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

# ---- Weights & Biases ----
export ENABLE_WANDB=\${ENABLE_WANDB:-1}
export WANDB_START_METHOD=\${WANDB_START_METHOD:-thread}
export WANDB_DIR=\${WANDB_DIR:-\${SLURM_TMPDIR:-\$PWD}/wandb}

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} bash run_train.sh ${CONFIG_PATH}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    if [ -z "${PREV_JOB_ID}" ]; then
        JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    else
        JOB_ID=$(sbatch --parsable --dependency=${DEPENDENCY_TYPE}:"${PREV_JOB_ID}" "${JOBSCRIPT}")
    fi

    echo "  Chain ${i}/${NUM_CHAINS}: submitted job ${JOB_ID}"
    PREV_JOB_ID="${JOB_ID}"
    rm -f "${JOBSCRIPT}"
done

echo ""
echo "All ${NUM_CHAINS} jobs submitted. Last job ID: ${PREV_JOB_ID}"
echo "Monitor with:  squeue -u \$USER"
echo "Cancel chain:  scancel ${PREV_JOB_ID}  (cancels pending dependents too)"
