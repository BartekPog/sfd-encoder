#!/bin/bash
# =============================================================================
# batch_run_visualize_hidden_b_h200_sphereclamp.sh
#
# Same as batch_run_visualize_hidden_b_h200.sh, but with --hidden_sphere_clamp
# enabled.  Output filenames get a "_sphereclamp" suffix so they never
# overwrite the plain-generation visualisations from the base script.
#
# Usage:
#   bash batch_run_visualize_hidden_b_h200_sphereclamp.sh [ckpt_step]
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
    # # H16 — longer to train, include when checkpoint is ready
    # "configs/sfd/hidden_b_h200/v2_base_h16_mse02.yaml|v2_base_h16_mse02"
    "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc.yaml|v4_mse01_cos001_noisy_enc"
    "configs/sfd/hidden_b_h200_from_ft/v4_mse01_cos001_noisy_enc_curriculum.yaml|v4_mse01_cos001_noisy_enc_curriculum"
)

echo "============================================="
echo "  B-size H200 — Hidden Token Visualisation + SPHERE CLAMP"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  5 visualisations per model, 5 noise seeds, 6 t_inith levels"
echo "  Outputs: outputs/visualizations/<exp>_<step>/vis_XX_classYYY_sphereclamp.png"
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
    JOBSCRIPT="jobs/viz_hsc_${EXP_LABEL}_${CKPT_NAME}.sh"
    OUTPUT="job_outputs/viz_hsc_${EXP_LABEL}_${CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name vhsc_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Hidden token visualisation (sphere clamp): ${EXP_LABEL} @ step ${CKPT_STEP}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

python visualize_hidden.py \\
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
echo "  Submitted ${SUBMITTED} visualisation jobs (sphere clamp)."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
