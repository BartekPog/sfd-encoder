#!/bin/bash
#SBATCH --job-name rg_1p0_hag
#SBATCH --output job_outputs/rg_1p0_hag_%A_%a.out
#SBATCH --time 00-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --gres gpu:h200:1
#SBATCH --array=0-11%16

set -euo pipefail

LINE=$(awk -v i="${SLURM_ARRAY_TASK_ID}" '$1==i {print; exit}' "jobs/rg_1p0_hag_tasks.tsv")
ENTRY=$(echo "${LINE}" | cut -f2-)
IFS='|' read -r AG_CONFIG AG_TAG T_FIX CFG_SCALE <<< "${ENTRY}"

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Hidden-AG task ${SLURM_ARRAY_TASK_ID}: ag=${AG_TAG} t_fix=${T_FIX} cfg=${CFG_SCALE}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=1 PRECISION=bf16 \
    bash run_inference.sh configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml \
    ckpt_path=outputs/train/1p0_v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5/checkpoints/0080000.pt \
    sample.sampling_method=euler \
    sample.num_sampling_steps=100 \
    sample.per_proc_batch_size=2048 \
    sample.fid_num=50000 \
    sample.balanced_sampling=true \
    sample.autoguidance=true \
    sample.autoguidance_config=${AG_CONFIG} \
    train.output_dir=outputs/inference \
    train.exp_name=1p0_v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5_0080000 \
    --encode_reground_t_fix ${T_FIX} \
    --reground_fixed_enc_noise \
    --hidden_sphere_clamp \
    --cfg_scale ${CFG_SCALE}

python save_fid_result.py \
    --output_dir outputs/inference/1p0_v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5_0080000 \
    --config     configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml \
    --ckpt_step  80000 \
    --inference_type encodereground \
    --sampler euler \
    --num_steps 100 \
    --hidden_sphere_clamp \
    --encode_reground_t_fix ${T_FIX} \
    --reground_fixed_enc_noise \
    --cfg_scale ${CFG_SCALE} \
    --autoguidance_config ${AG_CONFIG}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
