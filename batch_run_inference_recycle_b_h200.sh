#!/bin/bash
# =============================================================================
# batch_run_inference_recycle_b_h200.sh
#
# Recycle FID50K inference for B-size H200 hidden-token experiments.
#
# Single forward pass per ODE step.  The model sees hidden tokens at a fixed
# noise level t_fix and predicts velocity for both image and hidden.  The
# hidden velocity is used to extract h_clean, which is then re-noised back to
# t_fix (using the frozen ODE hidden state as fixed noise).  This lets h_clean
# drift to track the evolving image without an extra forward pass.
#
# Usage:
#   bash batch_run_inference_recycle_b_h200.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 60000)
#
# All experiments use:  Euler sampler, cfg_scale=1.0, FID50K,
#                       --recycle_t_fix <sweep> --hidden_sphere_clamp
#                       num_steps swept over NUM_STEPS_VALUES
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-60000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings (H200 cluster / DAIS) ----
TIME=${TIME:-"00-3:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-1024}

# ---- Inference output directory ----
INFERENCE_OUTPUT_DIR="outputs/inference"

# ---- Number of ODE steps to sweep ----
NUM_STEPS_VALUES=(100)

# ---- t_fix values to sweep ----
# 1.0 = nearly clean hidden conditioning, 0.0 = pure noise (hidden useless)
# T_FIX_VALUES=(0.7 0.8 0.85 0.9 0.95)
T_FIX_VALUES=(0.5)

# ---- Hidden-token experiment definitions ----
# Format: "config_yaml|train_exp_name[|ckpt_step_override]"
EXPERIMENTS=(
    "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml|v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5"
)

echo "============================================="
echo "  B-size H200 — FID50K Inference (RECYCLE + SPHERE CLAMP)"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler: Euler"
echo "  Mode: predict h_clean from velocity, re-noise to t_fix (1 fwd pass/step)"
echo "  Steps sweep: ${NUM_STEPS_VALUES[*]}"
echo "  t_fix sweep: ${T_FIX_VALUES[*]}"
echo "  GPUs: ${NUM_GPUS} x H200"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "============================================="
echo ""

SUBMITTED=0

for NUM_STEPS in "${NUM_STEPS_VALUES[@]}"; do
for T_FIX in "${T_FIX_VALUES[@]}"; do
    # Format values for filenames (e.g. t_fix=1.00 → 100, steps=100 → s100)
    T_FIX_TAG=$(printf "%.2f" "${T_FIX}" | tr -d '.')

    for ENTRY in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME EXP_CKPT_STEP_OVERRIDE <<< "${ENTRY}"
        EXP_CKPT_STEP=${EXP_CKPT_STEP_OVERRIDE:-${CKPT_STEP}}
        EXP_CKPT_NAME=$(printf "%07d" "${EXP_CKPT_STEP}")
        CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${EXP_CKPT_NAME}.pt"

        if [ ! -f "${CKPT_PATH}" ]; then
            echo "  SKIP: ${TRAIN_EXP_NAME} (steps=${NUM_STEPS}, t_fix=${T_FIX}) — checkpoint ${CKPT_PATH} not found"
            continue
        fi

        INFER_EXP_NAME="${TRAIN_EXP_NAME}_${EXP_CKPT_NAME}"
        EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
        JOBSCRIPT="jobs/infer_rc_s${NUM_STEPS}_t${T_FIX_TAG}_${EXP_LABEL}_${EXP_CKPT_NAME}.sh"
        OUTPUT="job_outputs/infer_rc_s${NUM_STEPS}_t${T_FIX_TAG}_${EXP_LABEL}_${EXP_CKPT_NAME}.o%J"
        mkdir -p "$(dirname "${JOBSCRIPT}")"
        mkdir -p "$(dirname "${OUTPUT}")"

        cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name rc_s${NUM_STEPS}_t${T_FIX_TAG}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (recycle steps=${NUM_STEPS} t_fix=${T_FIX} + sphere clamp): ${EXP_LABEL} @ step ${EXP_CKPT_STEP}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

GPUS_PER_NODE=${NUM_GPUS} PRECISION=${PRECISION} \\
    bash run_inference.sh ${CONFIG_PATH} \\
    ckpt_path=${CKPT_PATH} \\
    sample.sampling_method=euler \\
    sample.num_sampling_steps=${NUM_STEPS} \\
    sample.per_proc_batch_size=${PER_PROC_BATCH_SIZE} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME} \\
    --recycle_t_fix ${T_FIX} \\
    --hidden_sphere_clamp
python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${EXP_CKPT_STEP} \\
    --inference_type recycle \\
    --sampler euler \\
    --num_steps ${NUM_STEPS} \\
    --hidden_sphere_clamp \\
    --recycle_t_fix ${T_FIX}
echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

        JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
        echo "  ${TRAIN_EXP_NAME} (steps=${NUM_STEPS}, t_fix=${T_FIX}): submitted job ${JOB_ID}"
        rm -f "${JOBSCRIPT}"
        SUBMITTED=$((SUBMITTED + 1))
    done
done
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} recycle inference jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
