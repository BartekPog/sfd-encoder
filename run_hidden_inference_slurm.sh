#!/bin/bash
# =============================================================================
# run_hidden_inference_slurm.sh — Submit FID50K inference jobs for exp2 and exp3
#
# Usage:
#   bash run_hidden_inference_slurm.sh
#
# Submits two independent SLURM jobs (no chaining):
#   1. Exp2: Standard fine-tune (LightningDiT-XL/1)  — no hidden tokens
#   2. Exp3: Hidden from pretrained (HiddenLightningDiT_XL_1_H8) — with hidden tokens
#
# Both use cfg_scale=1.0 (no guidance), FID_NUM=50000, balanced sampling, dopri5.
# =============================================================================

set -euo pipefail

# ---- SLURM settings ----
PARTITION="gpu17"
TIME="00-12:00:00"
NUM_GPUS=3
GPUS="l40:${NUM_GPUS}"
MEM="350G"
CPUS_PER_TASK=4

CONFIGS=(
    "configs/sfd/hidden_xl/inference_exp2_4k.yaml"
    "configs/sfd/hidden_xl/inference_exp3_4k.yaml"
)

echo "============================================="
echo "  Hidden experiments — FID50K inference"
echo "  GPUs: ${NUM_GPUS} x L40"
echo "============================================="

for CONFIG in "${CONFIGS[@]}"; do
    EXP_NAME=$(basename "${CONFIG}" .yaml)
    JOBSCRIPT="jobs/infer_${EXP_NAME}.sh"
    OUTPUT="job_outputs/infer_${EXP_NAME}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash
#SBATCH -p ${PARTITION}
#SBATCH --job-name infer_${EXP_NAME}
#SBATCH --output ${OUTPUT}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --gres gpu:${GPUS}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference: ${EXP_NAME}"

source ~/.bashrc
source ./.venv-sfd/bin/activate

export TORCH_HOME=/scratch/inf0/user/bpogodzi/torch-cache
export HF_HOME=/BS/var-training/work/mdlm-decoding/tmp

GPUS_PER_NODE=${NUM_GPUS} PRECISION=bf16 FID_NUM=50000 \
    bash run_inference.sh ${CONFIG}

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    echo "  ${EXP_NAME}: submitted job ${JOB_ID}"
    rm -f "${JOBSCRIPT}"
done

echo ""
echo "Both inference jobs submitted."
echo "Monitor with:  squeue -u \$USER"
