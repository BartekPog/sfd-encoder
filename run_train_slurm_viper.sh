#!/bin/bash
# =============================================================================
# run_train_slurm_viper.sh — Submit a training job (or chain of jobs) to SLURM
#                             on the Viper-GPU cluster (AMD MI300A APUs).
#
# Usage:
#   bash run_train_slurm_viper.sh <config_path> [num_chains] [num_apus]
#
# Arguments:
#   config_path  — path to YAML config (defines all training parameters)
#   num_chains   — number of chained jobs (default: 6)
#   num_apus     — total number of MI300A APUs (default: 4).
#                  Each node has 2 APUs, so num_apus must be 1, 2, or a
#                  multiple of 2 for multi-node.
#
# Each job resumes from the latest checkpoint. The config must have
# `train.resume: true`.
#
# NOTE: Viper compute nodes have NO internet access. Install dependencies
# on the login node first:
#   module purge
#   module load gcc/14 rocm/6.3 python-waterboa/2025.06
#   python -m venv .venv-sfd-rocm
#   source .venv-sfd-rocm/bin/activate
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
#   pip install -r requirements-rocm.txt
#
# Examples:
#   bash run_train_slurm_viper.sh \
#     configs/sfd/hidden_b_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1p5_no_repg_hgd_5_hdrop0p1_sync_from20k.yaml 6 4
# =============================================================================

set -euo pipefail

CONFIG_PATH=${1:?'Usage: bash run_train_slurm_viper.sh <config_path> [num_chains] [num_apus]'}
NUM_CHAINS=${2:-6}
NUM_APUS=${3:-4}

# Extra passthrough args to run_train.sh (unused today; reserved).
EXTRA_ARGS=${EXTRA_ARGS:-}

# Viper has 2 APUs per node.
APUS_PER_NODE=2
if [ "${NUM_APUS}" -le "${APUS_PER_NODE}" ]; then
    NUM_NODES=1
    APUS_THIS_NODE=${NUM_APUS}
else
    NUM_NODES=$(( (NUM_APUS + APUS_PER_NODE - 1) / APUS_PER_NODE ))
    APUS_THIS_NODE=${APUS_PER_NODE}
fi

EXP_NAME=$(basename "${CONFIG_PATH}" .yaml)

# ---- SLURM settings (Viper-GPU MI300A) ----
TIME=${TIME:-"00-12:00:00"}
CPUS_PER_APU=24
# MI300A has 128 GiB unified CPU+GPU memory per APU. Viper's documented --mem
# ceiling is 110000 MB/APU (220000 MB/node) — anything higher is rejected.
MEM_PER_APU=110000
# One Slurm task per node drives accelerate launch, which then spawns
# --num_processes workers locally. Asking for ntasks-per-node>1 and letting
# accelerate respawn collides on the TCPStore port.
NTASKS_PER_NODE=1
CPUS_PER_TASK=$(( CPUS_PER_APU * APUS_THIS_NODE ))
MEM=$(( MEM_PER_APU * APUS_THIS_NODE ))
DEPENDENCY_TYPE=${DEPENDENCY_TYPE:-afterany}
PRECISION=${PRECISION:-bf16}

VENV_PATH=${VENV_PATH:-.venv-sfd-rocm}

echo "============================================="
echo "  Experiment: ${EXP_NAME}"
echo "  Config:     ${CONFIG_PATH}"
echo "  Chains:     ${NUM_CHAINS} x ${TIME}"
echo "  APUs:       ${NUM_APUS} x MI300A (${NUM_NODES} node(s))"
echo "  Venv:       ${VENV_PATH}"
echo "============================================="

PREV_JOB_ID=""

for i in $(seq 1 "${NUM_CHAINS}"); do
    JOBSCRIPT="jobs/train_${EXP_NAME}_viper_chain${i}.sh"
    OUTPUT_LOG="job_outputs/train_${EXP_NAME}_viper_chain${i}.o%J"
    mkdir -p "$(dirname "${JOBSCRIPT}")"
    mkdir -p "$(dirname "${OUTPUT_LOG}")"

    cat > "${JOBSCRIPT}" <<SLURM_EOF
#!/bin/bash -l
#SBATCH --job-name ${EXP_NAME}_v${i}
#SBATCH --output ${OUTPUT_LOG}
#SBATCH --time ${TIME}
#SBATCH --nodes=${NUM_NODES}
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:${APUS_THIS_NODE}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}

echo -n 'date: '; date '+%Y-%m-%d %H:%M:%S'
echo "Chain ${i}/${NUM_CHAINS} for ${EXP_NAME} (Viper-GPU)"

module purge
module load gcc/14 rocm/6.3 python-waterboa/2025.06
source ${VENV_PATH}/bin/activate

export TORCH_HOME=/viper/ptmp2/bpogodzi/hidden-diffusion/cache/torch
export HF_HOME=/viper/ptmp2/bpogodzi/hidden-diffusion/cache/hf

# ---- W&B ----
export ENABLE_WANDB=\${ENABLE_WANDB:-1}
export WANDB_START_METHOD=\${WANDB_START_METHOD:-thread}
export WANDB_DIR=\${WANDB_DIR:-\${SLURM_TMPDIR:-\$PWD}/wandb}
# Viper compute nodes have no internet — wandb must run offline and be synced
# back to the cloud from the login node later (\`wandb sync <run-dir>\`).
export WANDB_MODE=\${WANDB_MODE:-offline}

# ---- ROCm / AMD GPU tunables ----
export PYTORCH_ROCM_ARCH=gfx942        # MI300A arch for torch.compile / Triton
export HSA_XNACK=1                      # unified CPU+GPU memory on-demand paging
export PYTORCH_HIP_ALLOC_CONF=\${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}

# sfd-encoder optionally imports xformers.SwiGLU; it has a torch-only fallback
# gated behind XFORMERS_DISABLED. xformers has no ROCm wheels, so hard-disable.
export XFORMERS_DISABLED=1

# MIOpen find-db / kernel cache: per-job, node-local.
# Default \$HOME/.config/miopen is on NFS — when multiple nodes share the home
# dir, MIOpen's rename-into-place of its find-db races across ranks and one
# worker crashes with std::filesystem_error. SLURM_TMPDIR is node-local and
# auto-cleaned; fall back to /tmp when unset. SLURM_JOB_ID in the path keeps
# concurrent jobs on the same node from colliding.
export MIOPEN_USER_DB_PATH=\${SLURM_TMPDIR:-/tmp}/miopen-\${SLURM_JOB_ID}/user-db
export MIOPEN_CUSTOM_CACHE_DIR=\${SLURM_TMPDIR:-/tmp}/miopen-\${SLURM_JOB_ID}/kernel-cache

export MASTER_PORT=\$(shuf -i 29500-65000 -n 1)

# Multi-node: pick the first node as master.
export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)

# IMPORTANT: wrap \`accelerate launch\` in \`bash -c '...'\` so \$SLURM_NODEID
# is expanded *on each remote task* (per-node) rather than once on node 0
# before srun dispatches. Without this wrapper every node launches with
# --machine_rank=0 and the TCPStore rendezvous times out after 15 min.
# The leading mkdir runs per-node so each compute node creates its own
# MIOpen cache dirs in its local SLURM_TMPDIR / /tmp.
srun --ntasks-per-node=${NTASKS_PER_NODE} bash -c "mkdir -p \\\$MIOPEN_USER_DB_PATH \\\$MIOPEN_CUSTOM_CACHE_DIR && accelerate launch \\
    --main_process_ip \\\$MASTER_ADDR \\
    --main_process_port \\\$MASTER_PORT \\
    --machine_rank \\\$SLURM_NODEID \\
    --num_processes $(( APUS_THIS_NODE * NUM_NODES )) \\
    --num_machines ${NUM_NODES} \\
    --mixed_precision ${PRECISION} \\
    train.py \\
    --config ${CONFIG_PATH} ${EXTRA_ARGS}"

# Clean up per-job MIOpen cache (no-op when SLURM_TMPDIR auto-cleans anyway;
# relevant only for the /tmp fallback). Per-node, best-effort.
srun --ntasks-per-node=${NTASKS_PER_NODE} bash -c "rm -rf \\\$(dirname \\\$MIOPEN_USER_DB_PATH) 2>/dev/null || true"

echo -n 'finished: '; date '+%Y-%m-%d %H:%M:%S'
SLURM_EOF

    if [ -z "${PREV_JOB_ID}" ]; then
        JOB_ID=$(sbatch --parsable "${JOBSCRIPT}")
    else
        JOB_ID=$(sbatch --parsable --dependency=${DEPENDENCY_TYPE}:"${PREV_JOB_ID}" "${JOBSCRIPT}")
    fi

    echo "  Chain ${i}/${NUM_CHAINS}: submitted job ${JOB_ID}"
    PREV_JOB_ID="${JOB_ID}"
    rm -f "${JOBSCRIPT}"
done

echo ""
echo "All ${NUM_CHAINS} jobs submitted. Last job ID: ${PREV_JOB_ID}"
echo "Monitor with:  squeue -u \$USER"
echo "Cancel chain:  scancel ${PREV_JOB_ID}  (cancels pending dependents too)"
