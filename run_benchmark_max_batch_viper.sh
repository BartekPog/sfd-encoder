#!/bin/bash
# =============================================================================
# run_benchmark_max_batch_viper.sh — Sweep training batch sizes on one MI300A.
#
# Submits a short 1-APU SLURM job that loops over candidate per-GPU batch
# sizes, calling benchmark_train_step.py as a subprocess per trial so an OOM
# in one trial does not poison the next. Prints a summary table at the end
# and writes it to benchmark_max_batch_viper_<exp>.txt.
#
# Usage:
#   bash run_benchmark_max_batch_viper.sh <config_path> [batch_sizes]
#
# Examples:
#   bash run_benchmark_max_batch_viper.sh \
#     configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5_hdrop0p1_sync_from20k.yaml
#   bash run_benchmark_max_batch_viper.sh \
#     configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2_hdrop0p1_sync_from20k.yaml \
#     "32,64,96,128,192,256"
# =============================================================================

set -euo pipefail

CONFIG_PATH=${1:?'Usage: bash run_benchmark_max_batch_viper.sh <config_path> [batch_sizes]'}
BATCH_SIZES=${2:-"256,512,768,1024,1280,1536,1792,2048"}

TIME=${TIME:-"00-00:45:00"}
VENV_PATH=${VENV_PATH:-.venv-sfd-rocm}
EXP_NAME=$(basename "${CONFIG_PATH}" .yaml)

JOBSCRIPT="jobs/benchmark_train_${EXP_NAME}_viper.sh"
OUTPUT_LOG="job_outputs/benchmark_train_${EXP_NAME}_viper.o%J"
SUMMARY="benchmark_max_batch_viper_${EXP_NAME}.txt"
mkdir -p "$(dirname "${JOBSCRIPT}")" "$(dirname "${OUTPUT_LOG}")"

echo "============================================="
echo "  Config:      ${CONFIG_PATH}"
echo "  Batch sizes: ${BATCH_SIZES}"
echo "  Time limit:  ${TIME}"
echo "  Summary →    ${SUMMARY}"
echo "============================================="

cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash -l
#SBATCH --job-name bench_train_${EXP_NAME}
#SBATCH --output ${OUTPUT_LOG}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Training batch-size benchmark for ${EXP_NAME}"

module purge
module load gcc/14 rocm/6.3 python-waterboa/2025.06
source ${VENV_PATH}/bin/activate

export TORCH_HOME=/viper/ptmp2/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/viper/ptmp2/bpogodzi/hidden-diffusion/cache/hf
export PYTORCH_ROCM_ARCH=gfx942
export HSA_XNACK=1
export PYTORCH_HIP_ALLOC_CONF=\${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}
export XFORMERS_DISABLED=1

SUMMARY_PATH="${SUMMARY}"
: > "\${SUMMARY_PATH}"
printf '%-8s  %-10s  %-10s  %-10s\n' 'bs' 'status' 'peak_MB' 'step_ms' | tee -a "\${SUMMARY_PATH}"
printf '%-8s  %-10s  %-10s  %-10s\n' '--' '------' '-------' '-------' | tee -a "\${SUMMARY_PATH}"

IFS=',' read -ra BSZS <<< "${BATCH_SIZES}"
for BS in "\${BSZS[@]}"; do
    LOG=\$(python benchmark_train_step.py --config ${CONFIG_PATH} --batch_size "\${BS}" 2>&1 || true)
    echo "---- bs=\${BS} ----"
    echo "\${LOG}"

    LINE=\$(echo "\${LOG}" | grep -E '^RESULT ' || true)
    if [ -n "\${LINE}" ]; then
        PEAK=\$(echo "\${LINE}" | sed -E 's/.*peak_mb=([0-9.]+).*/\1/')
        STEP=\$(echo "\${LINE}" | sed -E 's/.*step_ms=([0-9.]+).*/\1/')
        printf '%-8s  %-10s  %-10s  %-10s\n' "\${BS}" OK "\${PEAK}" "\${STEP}" | tee -a "\${SUMMARY_PATH}"
    elif echo "\${LOG}" | grep -qiE 'OutOfMemoryError|HIP out of memory|CUDA out of memory'; then
        printf '%-8s  %-10s  %-10s  %-10s\n' "\${BS}" OOM '-' '-' | tee -a "\${SUMMARY_PATH}"
    else
        printf '%-8s  %-10s  %-10s  %-10s\n' "\${BS}" FAIL '-' '-' | tee -a "\${SUMMARY_PATH}"
    fi
done

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
echo ""
echo "Summary written to \${SUMMARY_PATH}:"
cat "\${SUMMARY_PATH}"
SLURM_EOF

JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
echo "Submitted benchmark job ${JOB_ID}"
echo "Watch: tail -F ${OUTPUT_LOG/\%J/${JOB_ID}}"
rm -f "${JOBSCRIPT}"
