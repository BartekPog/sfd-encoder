#!/bin/bash
# =============================================================================
# run_train_slurm.sh — Submit a training job (or chain of jobs) to SLURM
#
# Usage:
#   bash run_train_slurm.sh <config_path> [num_chains]
#
# Arguments:
#   config_path  — path to YAML config (e.g. configs/sfd/hidden_xl/exp1_hidden_scratch.yaml)
#   num_chains   — number of chained 4h jobs (default: 6, totalling ~24h)
#
# Each job resumes from the latest checkpoint. The config must have resume: true.
# Jobs are chained via --dependency=afterok so each starts after the previous completes.
# =============================================================================

set -euo pipefail

CONFIG_PATH=${1:?'Usage: bash run_train_slurm.sh <config_path> [num_chains]'}
NUM_CHAINS=${2:-6}

# ---- SLURM settings (3x L40 on gpu17) ----
PARTITION="gpu17"
TIME=${TIME:-"00-06:00:00"}        # 6 hours per chain link
NUM_GPUS=3
GPUS="l40:${NUM_GPUS}"
MEM="350G"
CPUS_PER_TASK=4
PRECISION="bf16"
DEPENDENCY_TYPE=${DEPENDENCY_TYPE:-afterany}

# ---- Derive a short experiment name for job naming ----
# e.g. configs/sfd/hidden_xl/exp1_hidden_scratch.yaml → exp1_hidden_scratch
EXP_NAME=$(basename "${CONFIG_PATH}" .yaml)

echo "============================================="
echo "  Training: ${EXP_NAME}"
echo "  Config:   ${CONFIG_PATH}"
echo "  Chains:   ${NUM_CHAINS} x ${TIME}"
echo "  GPUs:     ${NUM_GPUS} x L40"
echo "============================================="

PREV_JOB_ID=""

for i in $(seq 1 "${NUM_CHAINS}"); do
    JOBSCRIPT="jobs/train_${EXP_NAME}_chain${i}.sh"
    OUTPUT="job_outputs/train_${EXP_NAME}_chain${i}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH -p ${PARTITION}
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
source ./.venv-sfd/bin/activate

export TORCH_HOME=/scratch/inf0/user/bpogodzi/torch-cache
export HF_HOME=/BS/var-training/work/mdlm-decoding/tmp

# ---- Weights & Biases (optional) ----
# Enable with `ENABLE_WANDB=1` (default below turns it on for Slurm jobs).
# If compute nodes cannot reach the internet, set `WANDB_MODE=offline` and later run `wandb sync`.
export ENABLE_WANDB=${ENABLE_WANDB:-1}
export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}
export WANDB_DIR=${WANDB_DIR:-${SLURM_TMPDIR:-$PWD}/wandb}

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} bash run_train.sh ${CONFIG_PATH}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    # Submit with dependency on previous job (if any)
    if [ -z "${PREV_JOB_ID}" ]; then
        JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    else
        JOB_ID=$(sbatch --parsable --dependency=${DEPENDENCY_TYPE}:"${PREV_JOB_ID}" "${JOBSCRIPT}")
    fi

    echo "  Chain ${i}/${NUM_CHAINS}: submitted job ${JOB_ID}"
    PREV_JOB_ID="${JOB_ID}"

    # Clean up the generated jobscript (sbatch has already read it)
    rm -f "${JOBSCRIPT}"
done

echo ""
echo "All ${NUM_CHAINS} jobs submitted. Last job ID: ${PREV_JOB_ID}"
echo "Monitor with:  squeue -u \$USER"
echo "Cancel chain:  scancel ${PREV_JOB_ID}  (cancels pending dependents too)"
