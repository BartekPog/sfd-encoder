#!/bin/bash
# =============================================================================
# batch_run_visualize_hidden_b_h200.sh — Hidden token visualization for
# all B-size H200 hidden-token experiments.
#
# Generates grids showing what hidden tokens encode:
#   - Rows: different t_inith levels (0.0 to 1.0)
#   - Columns: reference image + images from different noise seeds
#
# Usage:
#   bash batch_run_visualize_hidden_b_h200.sh [ckpt_step]
#
# Arguments:
#   ckpt_step  — checkpoint step to evaluate (default: 300000)
# =============================================================================

set -euo pipefail

CKPT_STEP=${1:-300000}
CKPT_NAME=$(printf "%07d" "${CKPT_STEP}")

# ---- SLURM settings ----
TIME=${TIME:-"00-04:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4

# ---- Hidden-token experiment definitions ----
# Same set as two-pass / linear-hidden experiments (skip exp2 and H16)
EXPERIMENTS=(
    "configs/sfd/hidden_b_h200/exp1_hidden_scratch.yaml|hidden_b_h200_scratch"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained.yaml|hidden_b_h200_from_pretrained"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_h1.yaml|hidden_b_h200_from_pretrained_h1"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_separate_embedder.yaml|hidden_b_h200_from_pretrained_separate_embedder"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_hidden_pos_encoding.yaml|hidden_b_h200_from_pretrained_hidden_pos_encoding"
    "configs/sfd/hidden_b_h200/exp3_hidden_from_pretrained_weak_h_loss.yaml|hidden_b_h200_from_pretrained_weak_h_loss"
)

echo "============================================="
echo "  B-size H200 — Hidden Token Visualization"
echo "  Checkpoint step: ${CKPT_STEP} (${CKPT_NAME}.pt)"
echo "  5 visualisations per model, 5 noise seeds, 6 t_inith levels"
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
    JOBSCRIPT="jobs/viz_hidden_${EXP_LABEL}_${CKPT_NAME}.sh"
    OUTPUT="job_outputs/viz_hidden_${EXP_LABEL}_${CKPT_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH --job-name viz_h_${EXP_LABEL}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Hidden token visualization: ${EXP_LABEL} @ step ${CKPT_STEP}"

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
    --output_dir outputs/visualizations/${TRAIN_EXP_NAME}_${CKPT_NAME}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${TRAIN_EXP_NAME}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "============================================="
echo "  Submitted ${SUBMITTED} visualization jobs."
echo "  Monitor with:  squeue -u \$USER"
echo "============================================="
