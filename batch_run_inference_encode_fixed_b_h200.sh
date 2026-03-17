#!/bin/bash
# =============================================================================
# batch_run_inference_encode_fixed_b_h200.sh
#
# Encode-fixed FID50K inference for B-size H200 hidden-token experiments.
#
# Pass 1 (encode): A real dataset image is encoded in a single forward pass
#   (mirroring training Pass 1) to recover clean hidden tokens h_clean.
# Pass 2 (generate): Noise is added to h_clean at level start_t, and the
#   noisy hidden tokens are held FIXED (never updated) during image generation.
#
# This measures how much information the model can extract from partially
# noisy encoded hidden tokens without any hidden-token denoising.
#
# Usage:
#   bash batch_run_inference_encode_fixed_b_h200.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 120000)
#
# All experiments use:  Euler sampler, 100 steps, cfg_scale=1.0, FID50K,
#                       --encode_fixed_start_t <sweep> --hidden_sphere_clamp
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-100000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings (H200 cluster / DAIS) ----
TIME=${TIME:-"00-8:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"

# ---- Inference output directory ----
INFERENCE_OUTPUT_DIR="outputs/inference"

# ---- Start-t values to sweep ----
# 1.0 = clean (same as encode_first_pass), 0.0 = pure noise (hidden tokens useless)
START_T_VALUES=(0.0 0.1 0.3 0.5 0.6 0.8 1.0)

# ---- Hidden-token experiment definitions ----
# Format: "config_yaml|train_exp_name[|ckpt_step_override]"
EXPERIMENTS=(
    # ---- V4 experiments (from v2_finetune_no_hidden/1540000.pt) ----
    # "configs/sfd/hidden_b_h200_from_ft/v4_base_mse02.yaml|v4_base_mse02"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001.yaml|v4_mse01_cos001"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum.yaml|v4_mse01_cos001_noisy_enc_curriculum"
    "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum_hgd_scale_4.yaml|v4_mse01_cos001_noisy_enc_curriculum_hgd_scale_4"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_merged_noisy_enc.yaml|v4_mse002_cos0005_merged_noisy_enc"
)

echo "============================================="
echo "  B-size H200 — FID50K Inference (ENCODE-FIXED + SPHERE CLAMP)"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler: Euler, 100 steps"
echo "  Mode: encode real image → noisy h_init (start_t) → fixed hidden → generate"
echo "  Start-t sweep: ${START_T_VALUES[*]}"
echo "  GPUs: ${NUM_GPUS} x H200"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "============================================="
echo ""

SUBMITTED=0

for START_T in "${START_T_VALUES[@]}"; do
    # Format start_t for filenames (e.g. 0.30 → 030)
    START_T_TAG=$(echo "${START_T}" | tr -d '.')

    for ENTRY in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME EXP_CKPT_STEP_OVERRIDE <<< "${ENTRY}"
        EXP_CKPT_STEP=${EXP_CKPT_STEP_OVERRIDE:-${CKPT_STEP}}
        EXP_CKPT_NAME=$(printf "%07d" "${EXP_CKPT_STEP}")
        CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${EXP_CKPT_NAME}.pt"

        if [ ! -f "${CKPT_PATH}" ]; then
            echo "  SKIP: ${TRAIN_EXP_NAME} (start_t=${START_T}) — checkpoint ${CKPT_PATH} not found"
            continue
        fi

        INFER_EXP_NAME="${TRAIN_EXP_NAME}_${EXP_CKPT_NAME}"
        EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
        JOBSCRIPT="jobs/infer_encfix${START_T_TAG}_${EXP_LABEL}_${EXP_CKPT_NAME}.sh"
        OUTPUT="job_outputs/infer_encfix${START_T_TAG}_${EXP_LABEL}_${EXP_CKPT_NAME}.o%J"
        mkdir -p "$(dirname "${JOBSCRIPT}")"
        mkdir -p "$(dirname "${OUTPUT}")"

        cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name inef${START_T_TAG}_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (encode-fixed start_t=${START_T} + sphere clamp): ${EXP_LABEL} @ step ${EXP_CKPT_STEP}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh ${CONFIG_PATH} \\
    ckpt_path=${CKPT_PATH} \\
    sample.sampling_method=euler \\
    sample.num_sampling_steps=100 \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME} \\
    --encode_fixed_start_t ${START_T} \\
    --hidden_sphere_clamp
python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${EXP_CKPT_STEP} \\
    --inference_type encodefixed \\
    --sampler euler \\
    --num_steps 100 \\
    --hidden_sphere_clamp \\
    --encode_fixed_start_t ${START_T}
echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

        JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
        echo "  ${TRAIN_EXP_NAME} (start_t=${START_T}): submitted job ${JOB_ID}"
        rm -f "${JOBSCRIPT}"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} encode-fixed inference jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
