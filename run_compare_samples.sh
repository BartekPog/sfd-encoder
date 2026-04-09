#!/bin/bash
# =============================================================================
# run_compare_samples.sh
#
# Submit a SLURM job to generate qualitative sample comparison grids.
#
# Usage:
#   bash run_compare_samples.sh <comparison_config.yaml> [num_samples]
#
# Examples:
#   bash run_compare_samples.sh configs/comparisons/example_reground_vs_recycle.yaml
#   bash run_compare_samples.sh configs/comparisons/example_reground_vs_recycle.yaml 20
# =============================================================================

set -euo pipefail

COMPARISON_CONFIG=${1:?Usage: bash run_compare_samples.sh <comparison_config.yaml> [num_samples]}
NUM_SAMPLES=${2:-}

# ---- SLURM settings ----
TIME=${TIME:-"00-1:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4

# ---- Derive job name from config filename ----
CONFIG_BASENAME=$(basename "${COMPARISON_CONFIG}" .yaml)
JOBSCRIPT="jobs/compare_${CONFIG_BASENAME}.sh"
OUTPUT="job_outputs/compare_${CONFIG_BASENAME}.o%J"
mkdir -p "$(dirname "${JOBSCRIPT}")"
mkdir -p "$(dirname "${OUTPUT}")"

# ---- Build python command ----
PYTHON_CMD="python compare_samples.py ${COMPARISON_CONFIG}"
if [ -n "${NUM_SAMPLES}" ]; then
    PYTHON_CMD="${PYTHON_CMD} --num_samples ${NUM_SAMPLES}"
fi

cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name cmp_${CONFIG_BASENAME}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Sample comparison: ${COMPARISON_CONFIG}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

${PYTHON_CMD}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
echo "Submitted comparison job ${JOB_ID}: ${COMPARISON_CONFIG}"
echo "  Output: ${OUTPUT/\%J/${JOB_ID}}"
rm -f "${JOBSCRIPT}"
