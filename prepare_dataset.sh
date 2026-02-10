source .venv-sfd/bin/activate

python tokenizer/semvae/extract_dinov2_feature.py \
    --data_root /scratch/inf0/user/mparcham/ILSVRC2012/train \
    --output_root /scratch/inf0/user/bpogodzi/datasets/imagenet-dino/train \
    --model_name dinov2_vitb14_reg \
    --max_samples 1281167 \
    --batch_size 128 \
    --shuffle
