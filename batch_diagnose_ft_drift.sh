#!/bin/bash
# =============================================================================
# batch_diagnose_ft_drift.sh
#
# Submit diagnose_ft_drift.py on a single H200. Runs 3 probes to test whether
# the 1p0_finetune_no_hidden_warm8k drift is caused by a data/stats mismatch
# against the 4M pretrained weights.
#
# Usage:
#   bash batch_diagnose_ft_drift.sh
#   NUM_BATCHES=64 BATCH_SIZE=32 bash batch_diagnose_ft_drift.sh
# =============================================================================

set -euo pipefail

TIME=${TIME:-"00-01:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4

NUM_BATCHES=${NUM_BATCHES:-32}
BATCH_SIZE=${BATCH_SIZE:-16}
CONFIG=${CONFIG:-"configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden_warm8k.yaml"}
SRC_CKPT=${SRC_CKPT:-"outputs/train/sfd_1p0/checkpoints/4000000_full.pt"}
FT_CKPT=${FT_CKPT:-"outputs/train/1p0_finetune_no_hidden_warm8k/checkpoints/0060000.pt"}
STATS_DIR_OVERRIDE=${STATS_DIR_OVERRIDE:-""}

mkdir -p jobs job_outputs

JOBSCRIPT="jobs/diagnose_ft_drift.sh"
OUTPUT="job_outputs/diagnose_ft_drift.o%J"

cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name diag_ft_drift
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

set -euo pipefail

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Diagnose FT drift: num_batches=${NUM_BATCHES} batch_size=${BATCH_SIZE}"
echo "  config:   ${CONFIG}"
echo "  src_ckpt: ${SRC_CKPT}"
echo "  ft_ckpt:  ${FT_CKPT}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

STATS_FLAG=""
if [ -n "${STATS_DIR_OVERRIDE}" ]; then
    STATS_FLAG="--stats_dir_override ${STATS_DIR_OVERRIDE}"
fi

python diagnose_ft_drift.py \\
    --config     ${CONFIG} \\
    --src_ckpt   ${SRC_CKPT} \\
    --ft_ckpt    ${FT_CKPT} \\
    --num_batches ${NUM_BATCHES} \\
    --batch_size  ${BATCH_SIZE} \${STATS_FLAG}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
echo "Submitted diagnose_ft_drift job ${JOB_ID}"
echo "Output: ${OUTPUT/\%J/${JOB_ID}}"
echo "Monitor: squeue -u \$USER -j ${JOB_ID}"
