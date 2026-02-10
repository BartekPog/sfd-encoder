

# Job constants
PARTITION="gpu17"   # gpu16 / gpu20 / gpu22
TIME="00-24:00:00"  # NOTE make sure that one checkpointing interval takes less than this time!
# MEMORY="125G"
NUM_GPUS=3
# NUM_CORES=$((5* NUM_GPUS)) #12* NUM_GPUS
GPUS="l40:${NUM_GPUS}"            # a100:2




config=$1
id=${2:-'null'}
params=${*:3}

if [ "$id" == 'null' ]; then
    id=$(date '+%Y-%m-%d_%H-%M-%S')
fi

jobscript="jobs/${config}.sh"
output="job_outputs/${config}.o%J"
mkdir -p "$(dirname "${jobscript}")"
mkdir -p "$(dirname "${output}")"
echo "#!/bin/bash" > $jobscript
### Partition name
echo "#SBATCH -p ${PARTITION}" >> $jobscript
### Job name
echo "#SBATCH --job-name ${config}" >> $jobscript
### File for the output
echo "#SBATCH --output ${output}" >> $jobscript
### Time your job needs to execute, e.g. 15 min 30 sec
echo "#SBATCH --time ${TIME}" >> $jobscript
### Start time for delayed execution
# echo "#SBATCH --begin=now+120minutes" >> $jobscript

echo "#SBATCH --nodes=1" >> $jobscript


# echo "#SBATCH --partition=gpu1" >> $jobscript
# echo "#SBATCH --gres=gpu:h200:3" >> $jobscript
echo "#SBATCH --ntasks-per-node=${NUM_GPUS}" >> $jobscript       # 1 task per GPU"
echo "#SBATCH --cpus-per-task=4" >> $jobscript       # 12 CPU cores per task, MPCDF default ratio"
# echo "#SBATCH --threads-per-core=1" >> $jobscript

### Memory your job needs per node, e.g. 1 GB
echo "#SBATCH --mem=350G" >> $jobscript
### Number of GPUs per node, I want to use, e.g. 1
echo "#SBATCH --gres gpu:${GPUS}" >> $jobscript

# echo "#SBATCH -J pl_test" >> $jobscript
# echo "#SBATCH -D ./" >> $jobscript
# echo "#SBATCH -o slurm-%j.out" >> $jobscript
# echo "#SBATCH -e slurm-%j.err" >> $jobscript

# echo "export NCCL_DEBUG=INFO" >> $jobscript
# echo "export PYTHONFAULTHANDLER=1" >> $jobscript

echo "echo -n 'date: ';(date '+%Y-%m-%d %H:%M:%S')" >> $jobscript

echo "source ~/.bashrc" >> $jobscript
# echo "module load ffmpeg cuda/13.0" >> $jobscript
echo "source ./.venv-sfd/bin/activate" >> $jobscript
# echo "source ./.venv/bin/activate" >> $jobscript
echo "DEBUG=False TORCH_HOME=/scratch/inf0/user/bpogodzi/torch-cache HF_HOME=/BS/var-training/work/mdlm-decoding/tmp GPUS_PER_NODE=$NUM_GPUS bash run_extraction.sh tokenizer/configs/sdvae_f16d32_semvaebasech16.yaml semvae dinov2_vitb14_reg" >> $jobscript


echo $jobscript
sbatch $jobscript
rm $jobscript




## change GPU_NUM to the number of GPUs you have
# GPUS_PER_NODE=$GPU_NUM bash run_extraction.sh tokenizer/configs/sdvae_f16d32_semvaebasech16.yaml semvae dinov2_vitb14_reg