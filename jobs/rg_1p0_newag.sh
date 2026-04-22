#!/bin/bash
#SBATCH --job-name rg_1p0_newag
#SBATCH --output job_outputs/rg_1p0_newag_%A_%a.out
#SBATCH --time 00-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --gres gpu:h200:1
#SBATCH --array=0-19%16

set -euo pipefail

LINE=$(awk -v i="${SLURM_ARRAY_TASK_ID}" '$1==i {print; exit}' "jobs/rg_1p0_newag_tasks.tsv")
ENTRY=$(echo "${LINE}" | cut -f2-)
IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME CKPT_STEP T_FIX CFG_SCALE <<< "${ENTRY}"

CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")
CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"
INFER_EXP_NAME="${TRAIN_EXP_NAME}_${CKPT_NAME}"

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Reground (new AG) task ${SLURM_ARRAY_TASK_ID}: ${TRAIN_EXP_NAME} @ ${CKPT_STEP} t_fix=${T_FIX} cfg=${CFG_SCALE}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=1 PRECISION=bf16 \
    bash run_inference.sh ${CONFIG_PATH} \
    ckpt_path=${CKPT_PATH} \
    sample.sampling_method=euler \
    sample.num_sampling_steps=100 \
    sample.per_proc_batch_size=2048 \
    sample.fid_num=50000 \
    sample.balanced_sampling=true \
    sample.autoguidance=true \
    sample.autoguidance_config=configs/sfd/autoguidance_b/inference_ft.yaml \
    train.output_dir=outputs/inference \
    train.exp_name=${INFER_EXP_NAME} \
    --encode_reground_t_fix ${T_FIX} \
    --reground_fixed_enc_noise \
    --hidden_sphere_clamp \
    --cfg_scale ${CFG_SCALE}

python save_fid_result.py \
    --output_dir outputs/inference/${INFER_EXP_NAME} \
    --config     ${CONFIG_PATH} \
    --ckpt_step  ${CKPT_STEP} \
    --inference_type encodereground \
    --sampler euler \
    --num_steps 100 \
    --hidden_sphere_clamp \
    --encode_reground_t_fix ${T_FIX} \
    --reground_fixed_enc_noise \
    --cfg_scale ${CFG_SCALE} \
    --autoguidance_config configs/sfd/autoguidance_b/inference_ft.yaml

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
