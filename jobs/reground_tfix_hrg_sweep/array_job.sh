#!/bin/bash
#SBATCH --job-name rg_cfg_tfix_hrg
#SBATCH --output job_outputs/rg_cfg_tfix_hrg_%A_%a.o%J
#SBATCH --time 00-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --gres gpu:h200:1
#SBATCH --array=0-7%3

set -euo pipefail

TABLE="jobs/reground_tfix_hrg_sweep/tasks.tsv"
LINE=$(awk -v i="${SLURM_ARRAY_TASK_ID}" '$1==i {print; exit}' "${TABLE}")
CFG=$(echo "${LINE}" | cut -f2 | cut -d'|' -f1)
T_FIX=$(echo "${LINE}" | cut -f2 | cut -d'|' -f2)
HRG=$(echo "${LINE}" | cut -f2 | cut -d'|' -f3)
NOISEH=$(echo "${LINE}" | cut -f2 | cut -d'|' -f4)

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Task ${SLURM_ARRAY_TASK_ID}: cfg=${CFG} t_fix=${T_FIX} hrg=${HRG} noiseh=${NOISEH}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

EXPERIMENTS_OVERRIDE="configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p2.yaml|v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p2" \
CFG_SCALE="${CFG}" \
HIDDEN_REP_GUIDANCE="${HRG}" \
CFG_NOISE_HIDDEN="${NOISEH}" \
REGROUND_FIXED_ENC_NOISE=true \
T_FIX_VALUES_OVERRIDE="${T_FIX}" \
    bash batch_run_inference_reground_b_h200.sh 60000

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
