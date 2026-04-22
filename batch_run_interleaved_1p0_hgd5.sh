#!/bin/bash
# =============================================================================
# batch_run_interleaved_1p0_hgd5.sh — Race 6-GPU vs 3-GPU training jobs
#
# For each "slot" in the chain, submits TWO competing jobs (6-GPU and 3-GPU).
# Both share the same job name, so SLURM's singleton dependency ensures at
# most one runs at a time. When one starts, it cancels its sibling via a
# sibling-ID file written at submission time.
#
# Effective batch size is identical for both variants:
#   6 GPUs x 42 per-gpu x 1 accum = 252
#   3 GPUs x 42 per-gpu x 2 accum = 252
#
# The 3-GPU job is submitted with --begin=now+20m, giving the 6-GPU job
# a 20-minute head start to get scheduled. If it doesn't, the 3-GPU job
# becomes eligible and will likely start first.
#
# Usage:
#   bash batch_run_interleaved_1p0_hgd5.sh [num_slots]
#
# Arguments:
#   num_slots — number of racing pairs to submit (default: 6)
# =============================================================================

set -euo pipefail

NUM_SLOTS=${1:-6}

CONFIG_6GPU="configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5.yaml"
CONFIG_3GPU="configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_no_repg_hgd_5_3gpu.yaml"

# Shared job name — singleton prevents concurrent runs
JOB_NAME="1p0_hgd5_warm8k"

TIME=${TIME:-"00-06:00:00"}
FALLBACK_DELAY=${FALLBACK_DELAY:-10}  # minutes before 3-GPU becomes eligible
PRECISION="bf16"
CPUS_PER_TASK=4

# Directory for sibling ID files
SIBLING_DIR="jobs/.siblings"
mkdir -p jobs job_outputs "${SIBLING_DIR}"

echo "============================================="
echo "  Racing Training: 1p0 HGD-5 (warm8k)"
echo "  Slots:   ${NUM_SLOTS}"
echo "  Pattern: 6-GPU races 3-GPU per slot"
echo "  Fallback delay: ${FALLBACK_DELAY} min"
echo "  Time per job: ${TIME}"
echo "============================================="
echo ""

PREV_6GPU_ID=""
PREV_3GPU_ID=""

for slot in $(seq 1 "${NUM_SLOTS}"); do
    # ---- 6-GPU job ----
    NUM_GPUS=6
    MEM=$((250000 * NUM_GPUS))
    SIBLING_FILE_6="${SIBLING_DIR}/slot${slot}_6gpu.sibling"
    SIBLING_FILE_3="${SIBLING_DIR}/slot${slot}_3gpu.sibling"
    JOBSCRIPT_6="jobs/train_hgd5_6gpu_s${slot}.sh"
    OUTPUT_6="job_outputs/train_hgd5_6gpu_s${slot}.o%J"

    cat > "${JOBSCRIPT_6}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name ${JOB_NAME}
#SBATCH --output ${OUTPUT_6}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:h200:${NUM_GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Slot ${slot}/${NUM_SLOTS} — 6-GPU variant"

# Cancel the sibling 3-GPU job for this slot if it's still pending
SIBLING_FILE="${SIBLING_FILE_6}"
if [ -f "\${SIBLING_FILE}" ]; then
    SIBLING_ID=\$(cat "\${SIBLING_FILE}")
    scancel "\${SIBLING_ID}" 2>/dev/null && echo "Cancelled sibling 3-GPU job \${SIBLING_ID}" || true
    rm -f "\${SIBLING_FILE}"
fi

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf
export ENABLE_WANDB=\${ENABLE_WANDB:-1}
export WANDB_START_METHOD=\${WANDB_START_METHOD:-thread}
export WANDB_DIR=\${WANDB_DIR:-\${SLURM_TMPDIR:-\$PWD}/wandb}

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} bash run_train.sh ${CONFIG_6GPU}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    # ---- 3-GPU job ----
    NUM_GPUS=3
    MEM=$((250000 * NUM_GPUS))
    JOBSCRIPT_3="jobs/train_hgd5_3gpu_s${slot}.sh"
    OUTPUT_3="job_outputs/train_hgd5_3gpu_s${slot}.o%J"

    cat > "${JOBSCRIPT_3}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name ${JOB_NAME}
#SBATCH --output ${OUTPUT_3}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:h200:${NUM_GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Slot ${slot}/${NUM_SLOTS} — 3-GPU fallback variant"

# Cancel the sibling 6-GPU job for this slot if it's still pending
SIBLING_FILE="${SIBLING_FILE_3}"
if [ -f "\${SIBLING_FILE}" ]; then
    SIBLING_ID=\$(cat "\${SIBLING_FILE}")
    scancel "\${SIBLING_ID}" 2>/dev/null && echo "Cancelled sibling 6-GPU job \${SIBLING_ID}" || true
    rm -f "\${SIBLING_FILE}"
fi

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf
export ENABLE_WANDB=\${ENABLE_WANDB:-1}
export WANDB_START_METHOD=\${WANDB_START_METHOD:-thread}
export WANDB_DIR=\${WANDB_DIR:-\${SLURM_TMPDIR:-\$PWD}/wandb}

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} bash run_train.sh ${CONFIG_3GPU}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    # ---- Submit both jobs for this slot ----

    # Dependency: must wait for BOTH jobs from previous slot to finish/cancel
    if [ -n "${PREV_6GPU_ID}" ]; then
        DEP="--dependency=afterany:${PREV_6GPU_ID}:${PREV_3GPU_ID},singleton"
    else
        DEP=""
    fi

    # Submit 6-GPU (starts immediately, or after prev slot)
    JOB_6=$(sbatch --parsable ${DEP} "${JOBSCRIPT_6}")

    # Submit 3-GPU with delay (starts after FALLBACK_DELAY min, or after prev slot)
    FALLBACK_SECONDS=$((FALLBACK_DELAY * 60))
    JOB_3=$(sbatch --parsable ${DEP} --begin=now+${FALLBACK_SECONDS} "${JOBSCRIPT_3}")

    # Write sibling IDs so each job can cancel the other
    echo "${JOB_3}" > "${SIBLING_FILE_6}"   # 6-GPU reads this to cancel 3-GPU
    echo "${JOB_6}" > "${SIBLING_FILE_3}"   # 3-GPU reads this to cancel 6-GPU

    echo "  Slot ${slot}/${NUM_SLOTS}: 6-GPU=${JOB_6}  3-GPU=${JOB_3} (delay ${FALLBACK_DELAY}m)"

    PREV_6GPU_ID="${JOB_6}"
    PREV_3GPU_ID="${JOB_3}"
    rm -f "${JOBSCRIPT_6}" "${JOBSCRIPT_3}"
done

echo ""
echo "All ${NUM_SLOTS} slots submitted ($((NUM_SLOTS * 2)) total jobs)."
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel -n ${JOB_NAME}"
