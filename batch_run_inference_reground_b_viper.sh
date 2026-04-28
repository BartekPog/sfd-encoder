#!/bin/bash
# =============================================================================
# batch_run_inference_reground_b_viper.sh — Encode-reground FID50K inference
# for the two hdrop0p1_sync_from20k experiments on Viper-GPU (MI300A).
#
# At every ODE step the model re-encodes hidden tokens from the current noisy
# image x_t (Pass-1 style: pure-noise hidden input, t_hid=0), recovers
# h_clean, noises it to t_hid_fix, and uses the result to condition the image
# denoising step.
#
# Usage:
#   bash batch_run_inference_reground_b_viper.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 40000)
#
# Optional env overrides:
#   CFG_SCALE                     — CFG scale (default 1.0 = off; 1.5 matches SFD best)
#   HIDDEN_REP_GUIDANCE           — hidden rep guidance scale (default 1.0 = off)
#   T_FIX_VALUES_OVERRIDE         — space-separated t_fix sweep (default: 0.88)
#   NUM_STEPS_VALUES_OVERRIDE     — space-separated step count sweep (default: 100)
#   PER_PROC_BATCH_SIZE           — per-APU sampling batch size (default 256)
#   NUM_APUS                      — APUs per job (default 1)
#   TIME                          — SLURM walltime (default 00-06:00:00)
#
# Notes:
#   - Reground costs 2 forwards / step (encode + cond).  With CFG or repg
#     it becomes 3, with both it becomes 4.  Set PER_PROC_BATCH_SIZE based
#     on the gen benchmark for the expected num_forwards.
#   - REGROUND_FIXED_ENC_NOISE=true by default (matches DAIS reground
#     experiments that showed improved stability).
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-40000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings (Viper-GPU MI300A) ----
TIME=${TIME:-"00-06:00:00"}
NUM_APUS=${NUM_APUS:-1}
CPUS_PER_APU=24
MEM_PER_APU=110000
CPUS_PER_TASK=$(( CPUS_PER_APU * NUM_APUS ))
MEM=$(( MEM_PER_APU * NUM_APUS ))
PRECISION=${PRECISION:-bf16}
VENV_PATH=${VENV_PATH:-.venv-sfd-rocm}

PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-256}

# ---- Guidance overrides ----
CFG_SCALE=${CFG_SCALE:-1.0}
HIDDEN_REP_GUIDANCE=${HIDDEN_REP_GUIDANCE:-1.0}

GUIDE_INFER_FLAGS=""
GUIDE_SAVE_FLAGS=""
GUIDE_TAG=""
if (( $(echo "${CFG_SCALE} > 1.0" | bc -l) )); then
    GUIDE_INFER_FLAGS+=" --cfg_scale ${CFG_SCALE}"
    GUIDE_SAVE_FLAGS+=" --cfg_scale ${CFG_SCALE}"
    GUIDE_TAG+="_cfg$(printf '%.2f' ${CFG_SCALE} | tr -d '.')"
fi
if (( $(echo "${HIDDEN_REP_GUIDANCE} > 1.0" | bc -l) )); then
    GUIDE_INFER_FLAGS+=" --hidden_rep_guidance ${HIDDEN_REP_GUIDANCE}"
    GUIDE_SAVE_FLAGS+=" --hidden_rep_guidance ${HIDDEN_REP_GUIDANCE}"
    GUIDE_TAG+="_hrg$(printf '%.1f' ${HIDDEN_REP_GUIDANCE} | tr -d '.')"
fi

INFERENCE_OUTPUT_DIR="outputs/inference"

# ---- Sweep axes ----
if [ -n "${T_FIX_VALUES_OVERRIDE:-}" ]; then
    read -ra T_FIX_VALUES <<< "${T_FIX_VALUES_OVERRIDE}"
else
    T_FIX_VALUES=(0.88)
fi

if [ -n "${NUM_STEPS_VALUES_OVERRIDE:-}" ]; then
    read -ra NUM_STEPS_VALUES <<< "${NUM_STEPS_VALUES_OVERRIDE}"
else
    NUM_STEPS_VALUES=(100)
fi

# ---- Reground noise / optimization flags ----
REGROUND_FIXED_ENC_NOISE=${REGROUND_FIXED_ENC_NOISE:-true}
REGROUND_FIXED_COND_NOISE=${REGROUND_FIXED_COND_NOISE:-false}
REGROUND_SHARED_NOISE=${REGROUND_SHARED_NOISE:-false}
REGROUND_REUSE_ENCODE_FOR_REPG=${REGROUND_REUSE_ENCODE_FOR_REPG:-false}
CFG_NOISE_HIDDEN=${CFG_NOISE_HIDDEN:-false}

# Format: "config_yaml|train_exp_name"
EXPERIMENTS=(
    "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5_hdrop0p1_sync_from20k.yaml|v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5_hdrop0p1_sync_from20k"
    "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2_hdrop0p1_sync_from20k.yaml|v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2_hdrop0p1_sync_from20k"
)

echo "============================================="
echo "  Encode-reground inference on Viper-GPU (MI300A)"
echo "  Checkpoint step:     ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler:             Euler, ${NUM_STEPS_VALUES[*]} steps"
echo "  t_fix values:        ${T_FIX_VALUES[*]}"
echo "  cfg_scale:           ${CFG_SCALE}"
echo "  hidden_rep_guidance: ${HIDDEN_REP_GUIDANCE}"
echo "  per_proc_batch_size: ${PER_PROC_BATCH_SIZE}"
echo "  APUs/job:            ${NUM_APUS}"
echo "  Experiments:         ${#EXPERIMENTS[@]}"
echo "============================================="
echo ""

SUBMITTED=0

for NUM_STEPS in "${NUM_STEPS_VALUES[@]}"; do
for T_FIX in "${T_FIX_VALUES[@]}"; do
    T_FIX_TAG=$(printf "%.2f" "${T_FIX}" | tr -d '.')

    for ENTRY in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME <<< "${ENTRY}"
        CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"

        if [ ! -f "${CKPT_PATH}" ]; then
            echo "  SKIP: ${TRAIN_EXP_NAME} (steps=${NUM_STEPS}, t_fix=${T_FIX}) — checkpoint ${CKPT_PATH} not found"
            continue
        fi

        INFER_EXP_NAME="${TRAIN_EXP_NAME}_${CKPT_NAME}"
        EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)

        EXTRA_INFER_FLAGS=""
        EXTRA_SAVE_FLAGS=""
        FIXNOISE_TAG=""
        if [ "${REGROUND_FIXED_ENC_NOISE}" = "true" ]; then
            EXTRA_INFER_FLAGS+=" --reground_fixed_enc_noise"
            EXTRA_SAVE_FLAGS+=" --reground_fixed_enc_noise"
            FIXNOISE_TAG+="_fxenc"
        fi
        if [ "${REGROUND_FIXED_COND_NOISE}" = "true" ]; then
            EXTRA_INFER_FLAGS+=" --reground_fixed_cond_noise"
            EXTRA_SAVE_FLAGS+=" --reground_fixed_cond_noise"
            FIXNOISE_TAG+="_fxcond"
        fi
        if [ "${REGROUND_SHARED_NOISE}" = "true" ]; then
            EXTRA_INFER_FLAGS+=" --reground_shared_noise"
            EXTRA_SAVE_FLAGS+=" --reground_shared_noise"
            FIXNOISE_TAG+="_shared"
        fi
        if [ "${REGROUND_REUSE_ENCODE_FOR_REPG}" = "true" ]; then
            EXTRA_INFER_FLAGS+=" --reground_reuse_encode_for_repg"
            EXTRA_SAVE_FLAGS+=" --reground_reuse_encode_for_repg"
            FIXNOISE_TAG+="_reuserepg"
        fi
        if [ "${CFG_NOISE_HIDDEN}" = "true" ]; then
            EXTRA_INFER_FLAGS+=" --cfg_noise_hidden"
            EXTRA_SAVE_FLAGS+=" --cfg_noise_hidden"
            FIXNOISE_TAG+="_cfgnoiseh"
        fi

        JOBSCRIPT="jobs/infer_rg_viper_s${NUM_STEPS}_t${T_FIX_TAG}${FIXNOISE_TAG}${GUIDE_TAG}_${EXP_LABEL}_${CKPT_NAME}.sh"
        OUTPUT="job_outputs/infer_rg_viper_s${NUM_STEPS}_t${T_FIX_TAG}${FIXNOISE_TAG}${GUIDE_TAG}_${EXP_LABEL}_${CKPT_NAME}.o%J"
        mkdir -p "$(dirname "${JOBSCRIPT}")" "$(dirname "${OUTPUT}")"

        cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash -l
#SBATCH --job-name rg_s${NUM_STEPS}_t${T_FIX_TAG}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:${NUM_APUS}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (encode-reground, Viper): ${EXP_LABEL} steps=${NUM_STEPS} t_fix=${T_FIX} @ step ${CKPT_STEP}"

module purge
module load gcc/14 rocm/6.3 python-waterboa/2025.06
source ${VENV_PATH}/bin/activate

export TORCH_HOME=/viper/ptmp2/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/viper/ptmp2/bpogodzi/hidden-diffusion/cache/hf
export PYTORCH_ROCM_ARCH=gfx942
export HSA_XNACK=1
export PYTORCH_HIP_ALLOC_CONF=\${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}
export XFORMERS_DISABLED=1
export WANDB_MODE=\${WANDB_MODE:-offline}

GPUS_PER_NODE=${NUM_APUS} PRECISION=${PRECISION} \\
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
    --hidden_sphere_clamp${EXTRA_INFER_FLAGS}${GUIDE_INFER_FLAGS}

python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${CKPT_STEP} \\
    --inference_type encodereground \\
    --sampler euler \\
    --num_steps ${NUM_STEPS} \\
    --hidden_sphere_clamp \\
    --encode_reground_t_fix ${T_FIX}${EXTRA_SAVE_FLAGS}${GUIDE_SAVE_FLAGS}

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
echo "  Submitted ${SUBMITTED} reground inference jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
