#!/bin/bash
# =============================================================================
# batch_run_inference_linear_hidden_b_h200_sphereclamp.sh
#
# Same as batch_run_inference_linear_hidden_b_h200.sh, but with
# --hidden_sphere_clamp enabled.  At each hidden-token ODE step the
# single-step clean prediction is projected onto the unit sphere
# (per token) before the velocity is computed.
#
# Usage:
#   bash batch_run_inference_linear_hidden_b_h200_sphereclamp.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 80000)
#
# All experiments use:  Euler sampler, 100 steps, cfg_scale=1.0, FID50K,
#                       hidden_schedule=linear, hidden_sphere_clamp=true
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-40000}
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

# ---- Hidden-token experiment definitions ----
# Format: "config_yaml|train_exp_name[|ckpt_step_override]"
EXPERIMENTS=(
    # ---- V2 experiments ----
    # "configs/sfd/hidden_b_h200/v2_base_mse02.yaml|v2_base_mse02"
    # "configs/sfd/hidden_b_h200/v2_mse01_cos01.yaml|v2_mse01_cos01"
    # "configs/sfd/hidden_b_h200/v2_mse01_cos01_same_t.yaml|v2_mse01_cos01_same_t"
    # "configs/sfd/hidden_b_h200/v2_mse02_cos02.yaml|v2_mse02_cos02"
    # "configs/sfd/hidden_b_h200/v2_cos02.yaml|v2_cos02"
    # "configs/sfd/hidden_b_h200/v2_nonshr_temb_mse01_cos01.yaml|v2_nonshr_temb_mse01_cos01"
    # "configs/sfd/hidden_b_h200/v2_sep_embedder_mse02.yaml|v2_sep_embedder_mse02"
    # H16 — longer to train, include when checkpoint is ready
    # "configs/sfd/hidden_b_h200/v2_base_h16_mse02.yaml|v2_base_h16_mse02"
    # ---- V4 experiments (from v2_finetune_no_hidden/1540000.pt) ----
    # "configs/sfd/hidden_b_h200_from_ft/v4_base_mse02.yaml|v4_base_mse02"
    # "configs/sfd/hidden_b_h200_from_ft/v4_base_h16_mse02.yaml|v4_base_h16_mse02"
    # "configs/sfd/hidden_b_h200_from_ft/v4_base_h16_mse02_merged.yaml|v4_base_h16_mse02_merged"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_same_t.yaml|v4_mse01_cos001_same_t"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001.yaml|v4_mse01_cos001"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_merged.yaml|v4_mse01_cos001_merged"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_merged_noisy_enc.yaml|v4_mse01_cos001_merged_noisy_enc"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse002_cos0005_merged_noisy_enc.yaml|v4_mse002_cos0005_merged_noisy_enc"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_merged_noisy_enc_curriculum.yaml|v4_mse01_cos001_merged_noisy_enc_curriculum"
    "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum.yaml|v4_mse01_cos001_noisy_enc_curriculum"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc.yaml|v4_mse01_cos001_noisy_enc"
)

echo "============================================="
echo "  B-size H200 — FID50K Inference (LINEAR hidden schedule + SPHERE CLAMP)"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler: Euler, 100 steps"
echo "  Hidden schedule: linear + sphere-clamping"
echo "  GPUs: ${NUM_GPUS} x H200"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "============================================="
echo ""

SUBMITTED=0

for ENTRY in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME EXP_CKPT_STEP_OVERRIDE <<< "${ENTRY}"
    EXP_CKPT_STEP=${EXP_CKPT_STEP_OVERRIDE:-${CKPT_STEP}}
    EXP_CKPT_NAME=$(printf "%07d" "${EXP_CKPT_STEP}")
    CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${EXP_CKPT_NAME}.pt"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "  SKIP: ${TRAIN_EXP_NAME} — checkpoint ${CKPT_PATH} not found"
        continue
    fi

    INFER_EXP_NAME="${TRAIN_EXP_NAME}_${EXP_CKPT_NAME}"
    EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
    JOBSCRIPT="jobs/infer_linsc_${EXP_LABEL}_${EXP_CKPT_NAME}.sh"
    OUTPUT="job_outputs/infer_linsc_${EXP_LABEL}_${EXP_CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name inlsc_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (linear hidden + sphere clamp): ${EXP_LABEL} @ step ${EXP_CKPT_STEP}"

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
    --hidden_schedule linear \\
    --hidden_sphere_clamp
python save_fid_result.py \
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \
    --config     ${CONFIG_PATH} \
    --ckpt_step  ${EXP_CKPT_STEP} \
    --inference_type linear \
    --sampler euler \
    --num_steps 100 \
    --hidden_sphere_clamp
echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${TRAIN_EXP_NAME}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} inference jobs (linear hidden + sphere clamp)."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
