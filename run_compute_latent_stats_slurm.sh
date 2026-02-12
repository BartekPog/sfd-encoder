#!/bin/bash

# Job constants
# PARTITION="gpu17"   # gpu16 / gpu20 / gpu22
TIME="00-02:00:00"
NUM_GPUS=1
GPUS="h200:${NUM_GPUS}"

DATA_DIR=$1
JOB_NAME=${2:-'stats-sfd'}
RECOMPUTE=${3:-'false'}   # "true" or "false"

if [ -z "$DATA_DIR" ]; then
    echo "Usage: bash run_compute_latent_stats_slurm.sh <data_dir> [job_name] [recompute]"
    echo "Example:"
    echo "  bash run_compute_latent_stats_slurm.sh /path/to/imagenet_train_256 stats-sfd true"
    exit 1
fi

jobscript="jobs/${JOB_NAME}.sh"
output="job_outputs/${JOB_NAME}.o%J"
mkdir -p "$(dirname "${jobscript}")"
mkdir -p "$(dirname "${output}")"

echo "#!/bin/bash" > "$jobscript"
echo "#SBATCH --job-name ${JOB_NAME}" >> "$jobscript"
echo "#SBATCH --output ${output}" >> "$jobscript"
echo "#SBATCH --time ${TIME}" >> "$jobscript"
echo "#SBATCH --nodes=1" >> "$jobscript"
echo "#SBATCH --ntasks-per-node=${NUM_GPUS}" >> "$jobscript"
echo "#SBATCH --cpus-per-task=4" >> "$jobscript"
echo "#SBATCH --mem=64G" >> "$jobscript"
echo "#SBATCH --gres gpu:${GPUS}" >> "$jobscript"

echo "echo -n 'date: ';(date '+%Y-%m-%d %H:%M:%S')" >> "$jobscript"
echo "source ~/.bashrc" >> "$jobscript"
echo "module load python-waterboa ffmpeg cuda/13.0" >> "$jobscript"
echo "source ./.venv-sfd/bin/activate" >> "$jobscript"

if [ "$RECOMPUTE" = "true" ]; then
    echo "python compute_latent_stats.py --data_dir \"$DATA_DIR\" --recompute" >> "$jobscript"
else
    echo "python compute_latent_stats.py --data_dir \"$DATA_DIR\"" >> "$jobscript"
fi

echo "$jobscript"
sbatch "$jobscript"
rm "$jobscript"

