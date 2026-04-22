#!/bin/bash
#SBATCH --job-name linrepg_transfer
#SBATCH --output job_outputs/linear_repg_transfer_%A_%a.out
#SBATCH --time 00-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --gres gpu:h200:1
#SBATCH --array=0-11%16
#SBATCH --dependency=afterany:214496

set -euo pipefail

TABLE="jobs/linear_repg_sweep/transfer_tasks.tsv"
LINE=$(awk -v i="${SLURM_ARRAY_TASK_ID}" '$1==i {print; exit}' "${TABLE}")
MODEL=$(echo "${LINE}" | cut -f2 | cut -d'|' -f1)
CFG_SCALE=$(echo "${LINE}" | cut -f2 | cut -d'|' -f2)
HRG=$(echo "${LINE}" | cut -f2 | cut -d'|' -f3)

CONFIG_PATH="configs/sfd/hidden_b_h200_from_ft/${MODEL}.yaml"
CKPT_PATH="outputs/train/${MODEL}/checkpoints/0060000.pt"
INFER_EXP_NAME="${MODEL}_0060000"

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} cfg=${CFG_SCALE} repg=${HRG}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GUIDE_INFER_FLAGS=" --cfg_scale ${CFG_SCALE}"
GUIDE_SAVE_FLAGS=" --cfg_scale ${CFG_SCALE}"
if (( $(echo "${HRG} > 1.0" | bc -l) )); then
    GUIDE_INFER_FLAGS+=" --hidden_rep_guidance ${HRG}"
    GUIDE_SAVE_FLAGS+=" --hidden_rep_guidance ${HRG}"
fi

GPUS_PER_NODE=1 PRECISION=bf16 \
    bash run_inference.sh ${CONFIG_PATH} \
    ckpt_path=${CKPT_PATH} \
    sample.sampling_method=euler \
    sample.num_sampling_steps=100 \
    sample.per_proc_batch_size=1024 \
    sample.fid_num=50000 \
    sample.balanced_sampling=true \
    train.output_dir=outputs/inference \
    train.exp_name=${INFER_EXP_NAME} \
    --hidden_schedule linear \
    --hidden_sphere_clamp${GUIDE_INFER_FLAGS}

python save_fid_result.py \
    --output_dir outputs/inference/${INFER_EXP_NAME} \
    --config     ${CONFIG_PATH} \
    --ckpt_step  60000 \
    --inference_type linear \
    --sampler euler \
    --num_steps 100 \
    --hidden_sphere_clamp${GUIDE_SAVE_FLAGS}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
