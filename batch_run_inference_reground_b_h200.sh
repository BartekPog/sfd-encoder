#!/bin/bash
# =============================================================================
# batch_run_inference_reground_b_h200.sh
#
# Encode-reground FID50K inference for B-size H200 hidden-token experiments.
#
# At every ODE step the model re-encodes hidden tokens from the current noisy
# image x_t (Pass-1 style: pure-noise hidden input, t_hid=0), recovers h_clean,
# noises it to t_hid_fix, and uses the result to condition the image denoising
# step.  This lets hidden tokens track the evolving image state rather than
# committing to a fixed encoding at the start.
#
# Usage:
#   bash batch_run_inference_reground_b_h200.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 40000)
#
# All experiments use:  Euler sampler, cfg_scale=1.0, FID50K,
#                       --encode_reground_t_fix <sweep> --hidden_sphere_clamp
#                       num_steps swept over NUM_STEPS_VALUES
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-60000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings (H200 cluster / DAIS) ----
TIME=${TIME:-"00-4:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-1024}  # default 512; override: PER_PROC_BATCH_SIZE=256 bash ...

# ---- Inference output directory ----
INFERENCE_OUTPUT_DIR="outputs/inference"

# ---- Number of ODE steps to sweep ----
NUM_STEPS_VALUES=(100) #*2 inference steps because of the regrounding (each step has a Pass-1 re-encode + Pass-2 denoise)

# ---- t_fix values to sweep ----
# 1.0 = fully clean conditioning (default), 0.0 = no conditioning (ablation)
# T_FIX_VALUES=(0.7 0.9 0.95 1.0)
T_FIX_VALUES=(0.8 0.85 0.88 0.92)
# T_FIX_VALUES=(0.9 0.95 1.0)

# ---- Hidden-token experiment definitions ----
# Format: "config_yaml|train_exp_name[|ckpt_step_override]"
EXPERIMENTS=(
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1_repg_1p5"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml|v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5"

    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5"
    # "configs/sfd/hidden_b_h200_from_ft/e1_clean_enc_drop03_no_p3.yaml|e1_clean_enc_drop03_no_p3"
    "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml|v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_cos001_noisy_enc_curriculum_repg_1p5.yaml|v4_mse001_cos001_noisy_enc_curriculum_repg_1p5"
    # "configs/sfd/hidden_b_h200_from_ft/v4_noisy_enc_curriculum_repg_1p5_no_hloss.yaml|v4_noisy_enc_curriculum_repg_1p5_no_hloss"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5"

    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1_repg_1p5"
)

echo "============================================="
echo "  B-size H200 — FID50K Inference (ENCODE-REGROUND + SPHERE CLAMP)"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler: Euler"
echo "  Mode: re-encode h_clean from x_t at every ODE step → condition image step"
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
        JOBSCRIPT="jobs/infer_rg_s${NUM_STEPS}_t${T_FIX_TAG}_${EXP_LABEL}_${EXP_CKPT_NAME}.sh"
        OUTPUT="job_outputs/infer_rg_s${NUM_STEPS}_t${T_FIX_TAG}_${EXP_LABEL}_${EXP_CKPT_NAME}.o%J"
        mkdir -p "$(dirname "${JOBSCRIPT}")"
        mkdir -p "$(dirname "${OUTPUT}")"

        cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name rg_s${NUM_STEPS}_t${T_FIX_TAG}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (encode-reground steps=${NUM_STEPS} t_fix=${T_FIX} + sphere clamp): ${EXP_LABEL} @ step ${EXP_CKPT_STEP}"

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
    --encode_reground_t_fix ${T_FIX} \\
    --hidden_sphere_clamp
python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${EXP_CKPT_STEP} \\
    --inference_type encodereground \\
    --sampler euler \\
    --num_steps ${NUM_STEPS} \\
    --hidden_sphere_clamp \\
    --encode_reground_t_fix ${T_FIX}
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
echo "  Submitted ${SUBMITTED} encode-reground inference jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
