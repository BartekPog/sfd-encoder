#!/bin/bash
# =============================================================================
# batch_reground_tfix_hrg_sweep_repg1p2.sh
#
# Reground sweep for repg_1p2 model with configurable CFG, t_fix, and hidden_rep_guidance.
# Tests with and without --cfg_noise_hidden (noising hidden tokens for the
# CFG negative pass).
#
# Submits a SLURM job array with max 3 concurrent tasks.
#
# Model: v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p2
# cfg_noise_hidden: true, false
# Ckpt: 60K
#
# Usage:
#   bash batch_reground_tfix_hrg_sweep_repg1p2.sh
#
# Environment overrides:
#   CFG_VALUES_ARR="1.2 1.3 1.5" bash ...
#   T_FIX_VALUES_ARR="0.4 0.5 0.6 0.7 0.8 0.9" bash ...
#   HRG_VALUES_ARR="1.0 1.5 2.0" bash ...
#   MAX_CONCURRENT=5 bash ...
#
# Example: Sweep all three dimensions
#   CFG_VALUES_ARR="1.3 1.5" HRG_VALUES_ARR="1.0 1.5 2.0" T_FIX_VALUES_ARR="0.6 0.8" bash batch_reground_tfix_hrg_sweep_repg1p2.sh
#   = 2 CFG × 3 HRG × 2 t_fix × 2 noiseh = 24 jobs
#
# Default: 1 CFG × 1 HRG × 4 t_fix × 2 noiseh = 8 jobs
# =============================================================================

set -euo pipefail

CKPT_STEP=60000
TIME=${TIME:-"00-04:00:00"}
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"
MEM="180G"
CPUS_PER_TASK=4
PRECISION="bf16"
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-1024}
MAX_CONCURRENT=${MAX_CONCURRENT:-3}

MODEL="v4_mse0001_noisy_enc_nocurr_shift1p5_repg_1p2"
CONFIG_DIR="configs/sfd/hidden_b_h200_from_ft"
CONFIG_PATH="${CONFIG_DIR}/${MODEL}.yaml"
INFERENCE_OUTPUT_DIR="outputs/inference"

# Configurable CFG values (override: CFG_VALUES_ARR="1.2 1.3 1.5" bash ...)
# CFG_VALUES_ARR=${CFG_VALUES_ARR:-"1.3"}
CFG_VALUES_ARR=${CFG_VALUES_ARR:-"1.4"}
read -ra CFG_VALUES <<< "${CFG_VALUES_ARR}"

# Configurable t_fix values (override: T_FIX_VALUES_ARR="0.4 0.8 0.9" bash ...)
T_FIX_VALUES_ARR=${T_FIX_VALUES_ARR:-"0.2 0.4 0.6 0.8"}
read -ra T_FIX_VALUES <<< "${T_FIX_VALUES_ARR}"

# Configurable hidden rep guidance values (override: HRG_VALUES_ARR="1.0 1.5 2.0" bash ...)
HRG_VALUES_ARR=${HRG_VALUES_ARR:-"1.0"}
# HRG_VALUES_ARR=${HRG_VALUES_ARR:-"1.0 1.1 1.3"}
read -ra HRG_VALUES <<< "${HRG_VALUES_ARR}"

ARRAY_DIR="jobs/reground_tfix_hrg_sweep"
mkdir -p "${ARRAY_DIR}" job_outputs

# Build task table: task_id | cfg | t_fix | hrg | noiseh
TASKS=()
for cfg in "${CFG_VALUES[@]}"; do
    for hrg in "${HRG_VALUES[@]}"; do
        for t_fix in "${T_FIX_VALUES[@]}"; do
            for noiseh in false true; do
                TASKS+=("${cfg}|${t_fix}|${hrg}|${noiseh}")
            done
        done
    done
done

TASK_TABLE="${ARRAY_DIR}/tasks.tsv"
: > "${TASK_TABLE}"
for i in "${!TASKS[@]}"; do
    echo -e "${i}\t${TASKS[$i]}" >> "${TASK_TABLE}"
done

NUM_TASKS=${#TASKS[@]}

echo "============================================="
echo "  Reground CFG × t_fix × HRG sweep (SLURM array)"
echo "  Model: ${MODEL}"
echo "  CFG: ${CFG_VALUES[*]}"
echo "  t_fix: ${T_FIX_VALUES[*]}"
echo "  hidden_rep_guidance: ${HRG_VALUES[*]}"
echo "  cfg_noise_hidden: false, true"
echo "  Total tasks: ${NUM_TASKS}"
echo "  Max concurrent: ${MAX_CONCURRENT}"
echo "  Ckpt: ${CKPT_STEP}"
echo "============================================="
echo ""

# Build array job script
JOBSCRIPT="${ARRAY_DIR}/array_job.sh"
cat > "${JOBSCRIPT}" <<'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name rg_cfg_tfix_hrg
#SBATCH --output job_outputs/rg_cfg_tfix_hrg_%A_%a.o%J
#SBATCH --time TIME_PLACEHOLDER
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --gres gpu:h200:1
#SBATCH --array=0-ARRAY_SIZE%MAX_CONCURRENT

set -euo pipefail

TABLE="TASK_TABLE"
LINE=$(awk -v i="${SLURM_ARRAY_TASK_ID}" '$1==i {print; exit}' "${TABLE}")
CFG=$(echo "${LINE}" | cut -f2 | cut -d'|' -f1)
T_FIX=$(echo "${LINE}" | cut -f2 | cut -d'|' -f2)
HRG=$(echo "${LINE}" | cut -f2 | cut -d'|' -f3)
NOISEH=$(echo "${LINE}" | cut -f2 | cut -d'|' -f4)

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Task ${SLURM_ARRAY_TASK_ID}: cfg=${CFG} t_fix=${T_FIX} hrg=${HRG} noiseh=${NOISEH}"

source ~/.bashrc
module load python-waterboa ffmpeg cuda/13.0
source ./.venv-sfd/bin/activate

export TORCH_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/dais/fs/scratch/bpogodzi/hidden-diffusion/cache/hf

EXPERIMENTS_OVERRIDE="CONFIG_PATH|MODEL" \
CFG_SCALE="${CFG}" \
HIDDEN_REP_GUIDANCE="${HRG}" \
CFG_NOISE_HIDDEN="${NOISEH}" \
REGROUND_FIXED_ENC_NOISE=true \
T_FIX_VALUES_OVERRIDE="${T_FIX}" \
    bash batch_run_inference_reground_b_h200.sh CKPT_STEP

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

# Replace placeholders
sed -i "s|ARRAY_SIZE|$((NUM_TASKS - 1))|g" "${JOBSCRIPT}"
sed -i "s|MAX_CONCURRENT|${MAX_CONCURRENT}|g" "${JOBSCRIPT}"
sed -i "s|TASK_TABLE|${TASK_TABLE}|g" "${JOBSCRIPT}"
sed -i "s|CONFIG_PATH|${CONFIG_PATH}|g" "${JOBSCRIPT}"
sed -i "s|MODEL|${MODEL}|g" "${JOBSCRIPT}"
sed -i "s|CKPT_STEP|${CKPT_STEP}|g" "${JOBSCRIPT}"
sed -i "s|TIME_PLACEHOLDER|${TIME}|g" "${JOBSCRIPT}"

ARRAY_ID=$(sbatch --parsable "${JOBSCRIPT}")
echo "Submitted job array: ${ARRAY_ID}"
echo ""
echo "Monitor with:  squeue -u \$USER -l"
echo "Cancel with:   scancel ${ARRAY_ID}"
