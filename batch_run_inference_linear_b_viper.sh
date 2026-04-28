#!/bin/bash
# =============================================================================
# batch_run_inference_linear_b_viper.sh — FID50K inference with LINEAR hidden
# schedule for the two hdrop0p1_sync_from20k experiments on Viper-GPU
# (MI300A).
#
# Usage:
#   bash batch_run_inference_linear_b_viper.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 40000)
#
# Optional env overrides:
#   CFG_SCALE           — classifier-free guidance scale (default 1.0 = off)
#   PER_PROC_BATCH_SIZE — per-APU sampling batch size (default 256; raise once
#                         the gen benchmark reports how much fits)
#   NUM_APUS            — APUs per inference job (default 1; raise to 2 for
#                         2× throughput, single node)
#   TIME                — SLURM walltime (default 00-04:00:00)
#
# All experiments use: Euler sampler, 100 steps, FID50K, hidden_schedule=linear.
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-40000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings (Viper-GPU MI300A) ----
TIME=${TIME:-"00-04:00:00"}
NUM_APUS=${NUM_APUS:-1}
CPUS_PER_APU=24
MEM_PER_APU=110000
CPUS_PER_TASK=$(( CPUS_PER_APU * NUM_APUS ))
MEM=$(( MEM_PER_APU * NUM_APUS ))
PRECISION=${PRECISION:-bf16}
VENV_PATH=${VENV_PATH:-.venv-sfd-rocm}

# MI300A has less per-APU memory than an H200, so default lower; the gen
# benchmark will tell you how much further you can push this.
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-256}
CFG_SCALE=${CFG_SCALE:-1.0}

INFERENCE_OUTPUT_DIR="outputs/inference"

# Format: "config_yaml|train_exp_name"
EXPERIMENTS=(
    "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5_hdrop0p1_sync_from20k.yaml|v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5_hdrop0p1_sync_from20k"
    "configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2_hdrop0p1_sync_from20k.yaml|v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2_hdrop0p1_sync_from20k"
)

# Optional CFG tag to differentiate output dirs when sweeping cfg.
GUIDE_INFER_FLAGS=""
GUIDE_SAVE_FLAGS=""
GUIDE_TAG=""
if (( $(echo "${CFG_SCALE} > 1.0" | bc -l) )); then
    GUIDE_INFER_FLAGS+=" --cfg_scale ${CFG_SCALE}"
    GUIDE_SAVE_FLAGS+=" --cfg_scale ${CFG_SCALE}"
    GUIDE_TAG+="_cfg$(printf '%.2f' ${CFG_SCALE} | tr -d '.')"
fi

echo "============================================="
echo "  Linear inference on Viper-GPU (MI300A)"
echo "  Checkpoint step:     ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  Sampler:             Euler, 100 steps"
echo "  cfg_scale:           ${CFG_SCALE}"
echo "  per_proc_batch_size: ${PER_PROC_BATCH_SIZE}"
echo "  APUs/job:            ${NUM_APUS}"
echo "  Experiments:         ${#EXPERIMENTS[@]}"
echo "============================================="
echo ""

SUBMITTED=0

for ENTRY in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r CONFIG_PATH TRAIN_EXP_NAME <<< "${ENTRY}"
    CKPT_PATH="outputs/train/${TRAIN_EXP_NAME}/checkpoints/${CKPT_NAME}.pt"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "  SKIP: ${TRAIN_EXP_NAME} — checkpoint ${CKPT_PATH} not found"
        continue
    fi

    INFER_EXP_NAME="${TRAIN_EXP_NAME}_${CKPT_NAME}${GUIDE_TAG}"
    EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
    JOBSCRIPT="jobs/infer_lin_viper_${EXP_LABEL}_${CKPT_NAME}${GUIDE_TAG}.sh"
    OUTPUT="job_outputs/infer_lin_viper_${EXP_LABEL}_${CKPT_NAME}${GUIDE_TAG}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")" "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash -l
#SBATCH --job-name inlin_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:${NUM_APUS}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference (linear hidden, Viper): ${EXP_LABEL} @ step ${CKPT_STEP}"

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
    sample.num_sampling_steps=100 \\
    sample.per_proc_batch_size=${PER_PROC_BATCH_SIZE} \\
    sample.fid_num=50000 \\
    sample.balanced_sampling=true \\
    train.output_dir=${INFERENCE_OUTPUT_DIR} \\
    train.exp_name=${INFER_EXP_NAME} \\
    --hidden_schedule linear${GUIDE_INFER_FLAGS}

python save_fid_result.py \\
    --output_dir ${INFERENCE_OUTPUT_DIR}/${INFER_EXP_NAME} \\
    --config     ${CONFIG_PATH} \\
    --ckpt_step  ${CKPT_STEP} \\
    --inference_type linear \\
    --sampler euler \\
    --num_steps 100${GUIDE_SAVE_FLAGS}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${TRAIN_EXP_NAME}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} linear inference jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
