source .venv-sfd/bin/activate

python tokenizer/semvae/extract_dinov2_feature.py \
    --data_root /dais/fs/scratch/bpogodzi/hidden-diffusion/imagenet/ILSVRC2012/train \
    --output_root /dais/fs/scratch/bpogodzi/hidden-diffusion/datasets/imagenet-sfd/train \
    --model_name dinov2_vitb14_reg \
    --max_samples 1281167 \
    --batch_size 128 \
    --shuffle
