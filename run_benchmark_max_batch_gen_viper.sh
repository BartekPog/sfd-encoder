#!/bin/bash
# =============================================================================
# run_benchmark_max_batch_gen_viper.sh — Sweep inference batch sizes on one
# MI300A APU.
#
# Submits a short 1-APU SLURM job that loops over candidate per-GPU gen
# batch sizes, calling benchmark_gen_step.py as a subprocess per trial so an
# OOM in one trial does not poison the next. Writes a summary table to
# benchmark_max_batch_gen_viper_<exp>.txt.
#
# Usage:
#   bash run_benchmark_max_batch_gen_viper.sh <config_path> [batch_sizes] [num_forwards]
#
# num_forwards selects the inference scenario:
#   1 — linear hidden schedule (1 fwd/step)
#   2 — reground w/o CFG or repg (default)
#   3 — reground + pure CFG OR reground + repg
#   4 — reground + CFG + repg
#
# Examples:
#   bash run_benchmark_max_batch_gen_viper.sh \
#     configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5_hdrop0p1_sync_from20k.yaml
#   bash run_benchmark_max_batch_gen_viper.sh \
#     configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p5_hgd_2_hdrop0p1_sync_from20k.yaml \
#     "256,512,768,1024,1536" 3
# =============================================================================

set -euo pipefail

CONFIG_PATH=${1:?'Usage: bash run_benchmark_max_batch_gen_viper.sh <config_path> [batch_sizes] [num_forwards]'}
BATCH_SIZES=${2:-"128,256,384,512,768,1024"}
NUM_FORWARDS=${3:-2}

TIME=${TIME:-"00-00:45:00"}
VENV_PATH=${VENV_PATH:-.venv-sfd-rocm}
EXP_NAME=$(basename "${CONFIG_PATH}" .yaml)

JOBSCRIPT="jobs/benchmark_gen_${EXP_NAME}_viper.sh"
OUTPUT_LOG="job_outputs/benchmark_gen_${EXP_NAME}_viper.o%J"
SUMMARY="benchmark_max_batch_gen_viper_${EXP_NAME}_nfwd${NUM_FORWARDS}.txt"
mkdir -p "$(dirname "${JOBSCRIPT}")" "$(dirname "${OUTPUT_LOG}")"

echo "============================================="
echo "  Config:       ${CONFIG_PATH}"
echo "  gen_bsz:      ${BATCH_SIZES}"
echo "  num_forwards: ${NUM_FORWARDS}"
echo "  Time limit:   ${TIME}"
echo "  Summary →     ${SUMMARY}"
echo "============================================="

cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash -l
#SBATCH --job-name bench_gen_${EXP_NAME}
#SBATCH --output ${OUTPUT_LOG}
#SBATCH --time ${TIME}
#SBATCH --nodes=1
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Inference batch-size benchmark for ${EXP_NAME} (num_forwards=${NUM_FORWARDS})"

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
printf '%-8s  %-10s  %-10s  %-10s\n' 'bs' 'status' 'peak_MB' 'gen_ms' | tee -a "\${SUMMARY_PATH}"
printf '%-8s  %-10s  %-10s  %-10s\n' '--' '------' '-------' '------' | tee -a "\${SUMMARY_PATH}"

IFS=',' read -ra BSZS <<< "${BATCH_SIZES}"
for BS in "\${BSZS[@]}"; do
    LOG=\$(python benchmark_gen_step.py --config ${CONFIG_PATH} --gen_bsz "\${BS}" --num_forwards ${NUM_FORWARDS} 2>&1 || true)
    echo "---- bs=\${BS} ----"
    echo "\${LOG}"

    LINE=\$(echo "\${LOG}" | grep -E '^RESULT ' || true)
    if [ -n "\${LINE}" ]; then
        PEAK=\$(echo "\${LINE}" | sed -E 's/.*peak_mb=([0-9.]+).*/\1/')
        GEN=\$(echo "\${LINE}" | sed -E 's/.*gen_ms=([0-9.]+).*/\1/')
        printf '%-8s  %-10s  %-10s  %-10s\n' "\${BS}" OK "\${PEAK}" "\${GEN}" | tee -a "\${SUMMARY_PATH}"
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
echo "Submitted gen-benchmark job ${JOB_ID}"
echo "Watch: tail -F ${OUTPUT_LOG/\%J/${JOB_ID}}"
rm -f "${JOBSCRIPT}"
