#!/bin/bash
#SBATCH --job-name cfgsweep_ab
#SBATCH --output job_outputs/cfg_sweep_phase_ab_%A_%a.out
#SBATCH --time 00-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --gres gpu:h200:1
#SBATCH --array=0-59%16

set -euo pipefail

TABLE="jobs/cfg_sweep_phase_ab/tasks.tsv"
LINE=$(awk -v i="${SLURM_ARRAY_TASK_ID}" '$1==i {print; exit}' "${TABLE}")
MODEL=$(echo "${LINE}" | cut -f2 | cut -d'|' -f1)
CFG_SCALE=$(echo "${LINE}" | cut -f2 | cut -d'|' -f2)
CFG_NOISE_HIDDEN=$(echo "${LINE}" | cut -f2 | cut -d'|' -f3)

CONFIG_PATH="configs/sfd/hidden_b_h200_from_ft/${MODEL}.yaml"
CKPT_PATH="outputs/train/${MODEL}/checkpoints/0060000.pt"
INFER_EXP_NAME="${MODEL}_0060000"

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} cfg=${CFG_SCALE} noise_h=${CFG_NOISE_HIDDEN}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

EXTRA_INFER_FLAGS=" --reground_fixed_enc_noise"
EXTRA_SAVE_FLAGS=" --reground_fixed_enc_noise"
if [ "${CFG_NOISE_HIDDEN}" = "true" ]; then
    EXTRA_INFER_FLAGS+=" --cfg_noise_hidden"
    EXTRA_SAVE_FLAGS+=" --cfg_noise_hidden"
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
    --encode_reground_t_fix 0.88 \
    --hidden_sphere_clamp \
    --cfg_scale ${CFG_SCALE}${EXTRA_INFER_FLAGS}

python save_fid_result.py \
    --output_dir outputs/inference/${INFER_EXP_NAME} \
    --config     ${CONFIG_PATH} \
    --ckpt_step  60000 \
    --inference_type encodereground \
    --sampler euler \
    --num_steps 100 \
    --hidden_sphere_clamp \
    --encode_reground_t_fix 0.88 \
    --cfg_scale ${CFG_SCALE}${EXTRA_SAVE_FLAGS}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
