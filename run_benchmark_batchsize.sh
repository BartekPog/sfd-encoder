#!/bin/bash
# =============================================================================
# Quick max-batch-size benchmark on a single H200.
#
# Usage:
#   bash run_benchmark_batchsize.sh           # default: start=256, step=128
#   bash run_benchmark_batchsize.sh 512 256   # start=512, step=256
# =============================================================================
set -euo pipefail

START=${1:-256}
STEP=${2:-128}

JOBSCRIPT=$(mktemp jobs/bench_bs_XXXXXX.sh)
mkdir -p jobs job_outputs

cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name bench_bs
#SBATCH --output job_outputs/bench_batchsize.o%J
#SBATCH --time 00-1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --gres gpu:h200:1

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf
export PYTHONUNBUFFERED=1

python -u benchmark_max_batch_size.py --start ${START} --step ${STEP}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
echo "Submitted batch size benchmark: job ${JOB_ID}"
echo "Output: job_outputs/bench_batchsize.o${JOB_ID}"
rm -f "${JOBSCRIPT}"
