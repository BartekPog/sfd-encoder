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
# Optional env overrides (guidance benchmarks):
#   CFG_SCALE            — classifier-free guidance scale (default 1.0 = off).
#                          Set to 1.5 to match the best SFD setup from the
#                          README tables (XL 800ep / XXL 80ep / XXL 800ep).
#   HIDDEN_REP_GUIDANCE  — hidden representation guidance scale (default 1.0 = off).
#                          Uses the ramped weight w(t_h) = 1 + (s - 1) * t_h.
#
# Examples:
#   CFG_SCALE=1.5 bash batch_run_inference_reground_b_h200.sh
#   HIDDEN_REP_GUIDANCE=2.0 bash batch_run_inference_reground_b_h200.sh
#   CFG_SCALE=1.5 HIDDEN_REP_GUIDANCE=2.0 bash batch_run_inference_reground_b_h200.sh
#
# All experiments use:  Euler sampler, FID50K,
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

# ---- Guidance overrides (off by default; matches SFD best-FID setup when enabled) ----
CFG_SCALE=${CFG_SCALE:-1.0}                       # 1.0 = off; 1.5 = SFD best
HIDDEN_REP_GUIDANCE=${HIDDEN_REP_GUIDANCE:-1.0}   # 1.0 = off; e.g. 2.0 or 4.0 enables rep guidance

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

# ---- Inference output directory ----
INFERENCE_OUTPUT_DIR="outputs/inference"

# ---- Number of ODE steps to sweep ----
NUM_STEPS_VALUES=(100) #*2 inference steps because of the regrounding (each step has a Pass-1 re-encode + Pass-2 denoise)

# ---- t_fix values to sweep (override via env: T_FIX_VALUES_OVERRIDE="0.88" bash ...) ----
if [ -n "${T_FIX_VALUES_OVERRIDE:-}" ]; then
    read -ra T_FIX_VALUES <<< "${T_FIX_VALUES_OVERRIDE}"
else
    # T_FIX_VALUES=(0.7 0.9 0.95 1.0)
    T_FIX_VALUES=(0.8 0.85 0.88 0.9 0.92 1.0)
    # T_FIX_VALUES=(0.9 0.95 1.0)
fi

# ---- Fixed-noise flags for reground ----
# When true, the encode / conditioning noise is sampled once and reused across
# all ODE steps instead of being resampled each step.
# REGROUND_FIXED_ENC_NOISE=${REGROUND_FIXED_ENC_NOISE:-false}
REGROUND_FIXED_ENC_NOISE=${REGROUND_FIXED_ENC_NOISE:-true}
REGROUND_FIXED_COND_NOISE=${REGROUND_FIXED_COND_NOISE:-false}
# When true, use the same noise for encoding and conditioning (xt_h = x0 + t_fix * v_enc).
REGROUND_SHARED_NOISE=${REGROUND_SHARED_NOISE:-false}
# When true, reuse the encode-pass image velocity as the rep-guidance unconditional
# pass instead of issuing a separate forward call. Saves one forward per ODE step
# when --hidden_rep_guidance > 1.0; couples the repg-uncond noise to the encode-pass
# noise (which is in turn fixed when REGROUND_FIXED_ENC_NOISE=true).
REGROUND_REUSE_ENCODE_FOR_REPG=${REGROUND_REUSE_ENCODE_FOR_REPG:-false}

# When true, the pure-CFG negative pass uses fully noised hidden tokens (random
# noise, t_h=0) instead of re-grounded hidden, making it truly unconditional.
CFG_NOISE_HIDDEN=${CFG_NOISE_HIDDEN:-false}

# ---- Autoguidance config (for CFG negative pass) ----
# When set to a YAML config path, the inference loads that checkpoint as the
# autoguidance (negative) model. Required when CFG_SCALE > 1.0 — pure CFG
# without an autoguidance model is a no-op in the new reground guidance
# pipeline. Leave empty/unset to disable autoguidance entirely.
AUTOGUIDANCE_CONFIG=${AUTOGUIDANCE_CONFIG:-}

# ---- Hidden-token experiment definitions ----
# Format: "config_yaml|train_exp_name[|ckpt_step_override]"
# Override via env: EXPERIMENTS_OVERRIDE="config.yaml|exp_name" bash ...
if [ -n "${EXPERIMENTS_OVERRIDE:-}" ]; then
    EXPERIMENTS=("${EXPERIMENTS_OVERRIDE}")
else
EXPERIMENTS=(
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1_repg_1p5"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5"

    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5"
    # "configs/sfd/hidden_b_h200_from_ft/e1_clean_enc_drop03_no_p3.yaml|e1_clean_enc_drop03_no_p3"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5_merged.yaml|v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5_merged"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_cos001_noisy_enc_curriculum_repg_1p5.yaml|v4_mse001_cos001_noisy_enc_curriculum_repg_1p5"
    # "configs/sfd/hidden_b_h200_from_ft/v4_noisy_enc_curriculum_repg_1p5_no_hloss.yaml|v4_noisy_enc_curriculum_repg_1p5_no_hloss"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5"

    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1p5_repg_1p5"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse001_noisy_enc_nocurr_shift1_repg_1p5.yaml|v4_mse001_noisy_enc_nocurr_shift1_repg_1p5"

    "configs/sfd/hidden_b_h200_from_ft/v4_noisy_enc_nocurr_shift1_repg_1p5_no_hloss.yaml|v4_noisy_enc_nocurr_shift1_repg_1p5_no_hloss"
    "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift0p5_repg_1p5.yaml|v4_mse0001_noisy_enc_nocurr_shift0p5_repg_1p5"
    "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5.yaml|v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5"
    
    "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml|v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5"
)
fi

echo "============================================="
echo "  B-size H200 — FID50K Inference (ENCODE-REGROUND + SPHERE CLAMP)"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler: Euler"
echo "  Mode: re-encode h_clean from x_t at every ODE step → condition image step"
echo "  Steps sweep: ${NUM_STEPS_VALUES[*]}"
echo "  t_fix sweep: ${T_FIX_VALUES[*]}"
echo "  Fixed enc noise: ${REGROUND_FIXED_ENC_NOISE}"
echo "  Fixed cond noise: ${REGROUND_FIXED_COND_NOISE}"
echo "  Shared noise: ${REGROUND_SHARED_NOISE}"
echo "  Reuse encode for repg: ${REGROUND_REUSE_ENCODE_FOR_REPG}"
echo "  CFG scale: ${CFG_SCALE}"
echo "  CFG noise hidden: ${CFG_NOISE_HIDDEN}"
echo "  Hidden rep guidance: ${HIDDEN_REP_GUIDANCE}"
echo "  Autoguidance config: ${AUTOGUIDANCE_CONFIG:-<none>}"
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

        # Build extra CLI flags for fixed-noise options
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

        # Autoguidance config: enables autoguidance via OmegaConf overrides on
        # the run_inference.sh CLI, and is passed to save_fid_result.py for
        # folder filtering.
        AUTOG_INFER_OVERRIDE=""
        AUTOG_SAVE_FLAGS=""
        AUTOG_TAG=""
        if [ -n "${AUTOGUIDANCE_CONFIG}" ]; then
            AUTOG_INFER_OVERRIDE=" sample.autoguidance=true sample.autoguidance_config=${AUTOGUIDANCE_CONFIG}"
            AUTOG_SAVE_FLAGS+=" --autoguidance_config ${AUTOGUIDANCE_CONFIG}"
            AUTOG_BASENAME=$(basename "${AUTOGUIDANCE_CONFIG}" .yaml | tr '.' 'p')
            AUTOG_TAG="_ag_${AUTOG_BASENAME}"
        fi

        JOBSCRIPT="jobs/infer_rg_s${NUM_STEPS}_t${T_FIX_TAG}${FIXNOISE_TAG}${GUIDE_TAG}${AUTOG_TAG}_${EXP_LABEL}_${EXP_CKPT_NAME}.sh"
        OUTPUT="job_outputs/infer_rg_s${NUM_STEPS}_t${T_FIX_TAG}${FIXNOISE_TAG}${GUIDE_TAG}${AUTOG_TAG}_${EXP_LABEL}_${EXP_CKPT_NAME}.o%J"
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
    train.exp_name=${INFER_EXP_NAME}${AUTOG_INFER_OVERRIDE} \\
    --encode_reground_t_fix ${T_FIX} \\
    --hidden_sphere_clamp${EXTRA_INFER_FLAGS}${GUIDE_INFER_FLAGS}
python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${EXP_CKPT_STEP} \\
    --inference_type encodereground \\
    --sampler euler \\
    --num_steps ${NUM_STEPS} \\
    --hidden_sphere_clamp \\
    --encode_reground_t_fix ${T_FIX}${EXTRA_SAVE_FLAGS}${GUIDE_SAVE_FLAGS}${AUTOG_SAVE_FLAGS}
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
