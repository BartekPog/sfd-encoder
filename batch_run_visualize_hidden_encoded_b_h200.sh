#!/bin/bash
# =============================================================================
# batch_run_visualize_hidden_encoded_b_h200.sh
#
# Visualize hidden token encodings from *real images* with sphere clamp.
#
# For each visualisation a real dataset image is encoded to obtain h_clean,
# then images are generated from varying hidden-noise levels (rows) and
# image-noise seeds (columns).  The original real image is shown in an
# extra column on the left.
#
# Usage:
#   bash batch_run_visualize_hidden_encoded_b_h200.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 80000)
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-40000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings ----
TIME=${TIME:-"00-04:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4

# ---- Hidden-token experiment definitions ----
EXPERIMENTS=(
    # ---- V2 experiments ----
    # "configs/sfd/hidden_b_h200/v2_base_mse02.yaml|v2_base_mse02"
    # "configs/sfd/hidden_b_h200/v2_mse01_cos01.yaml|v2_mse01_cos01"
    # "configs/sfd/hidden_b_h200/v2_mse01_cos01_same_t.yaml|v2_mse01_cos01_same_t"
    # "configs/sfd/hidden_b_h200/v2_mse02_cos02.yaml|v2_mse02_cos02"
    # "configs/sfd/hidden_b_h200/v2_cos02.yaml|v2_cos02"
    # "configs/sfd/hidden_b_h200/v2_nonshr_temb_mse01_cos01.yaml|v2_nonshr_temb_mse01_cos01"
    # "configs/sfd/hidden_b_h200/v2_sep_embedder_mse02.yaml|v2_sep_embedder_mse02"
    # # H16
    # "configs/sfd/hidden_b_h200/v2_base_h16_mse02.yaml|v2_base_h16_mse02"
    # "configs/sfd/hidden_b_h200_from_ft/v4_base_mse02.yaml|v4_base_mse02"
    # "configs/sfd/hidden_b_h200_from_ft/v4_base_h16_mse02.yaml|v4_base_h16_mse02"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_same_t.yaml|v4_mse01_cos001_same_t"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001.yaml|v4_mse01_cos001"
    # New batch of V4
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_merged_noisy_enc.yaml|v4_mse01_cos001_merged_noisy_enc"
    "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc.yaml|v4_mse01_cos001_noisy_enc"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_merged_noisy_enc_curriculum.yaml|v4_mse01_cos001_merged_noisy_enc_curriculum"
    # "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum.yaml|v4_mse01_cos001_noisy_enc_curriculum"
    # "configs/sfd/hidden_b_h200_from_ft/v4_base_h16_mse02_merged.yaml|v4_base_h16_mse02_merged"
)

echo "============================================="
echo "  B-size H200 — Hidden Token Encoded Visualisation + SPHERE CLAMP"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  5 visualisations per model, 5 noise seeds, 6 t_inith levels"
echo "  Mode: encode real image → noisy h → generate (with original shown)"
echo "  Outputs: outputs/visualizations/<exp>_<step>/vis_enc_XX_classYYY_sphereclamp.png"
echo "  GPUs: ${NUM_GPUS} x H200"
echo "  Experiments: ${#EXPERIMENTS[@]}"
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

    EXP_LABEL=$(basename "${CONFIG_PATH}" .yaml)
    JOBSCRIPT="jobs/viz_enc_${EXP_LABEL}_${CKPT_NAME}.sh"
    OUTPUT="job_outputs/viz_enc_${EXP_LABEL}_${CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name venc_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Hidden token encoded visualisation (sphere clamp): ${EXP_LABEL} @ step ${CKPT_STEP}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

python visualize_hidden_encoded.py \\
    --config ${CONFIG_PATH} \\
    --ckpt_path ${CKPT_PATH} \\
    --num_vis 5 \\
    --num_noise_seeds 5 \\
    --t_inith_values "0.0,0.2,0.4,0.6,0.8,1.0" \\
    --num_sampling_steps 100 \\
    --cfg_scale 1.0 \\
    --seed 42 \\
    --output_dir outputs/visualizations/${TRAIN_EXP_NAME}_${CKPT_NAME} \\
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
echo "  Submitted ${SUBMITTED} encoded visualisation jobs (sphere clamp)."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
