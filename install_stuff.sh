source .venv-sfd/bin/activate
pip install torch
pip install -r requirements.txt
pip install numpy==1.24.3 protobuf==3.20.0
pip install piqa
## guided-diffusion evaluation environment
git clone https://github.com/openai/guided-diffusion.git
pip install tensorflow==2.8.0
sed -i 's/dtype=np\.bool)/dtype=np.bool_)/g' guided-diffusion/evaluations/evaluator.py  # or w

# Prepare the decoder of SD-VAE
mkdir -p outputs/model_weights/va-vae-imagenet256-experimental-variants
wget https://huggingface.co/hustvl/va-vae-imagenet256-experimental-variants/resolve/main/ldm-imagenet256-f16d32-50ep.ckpt \
    --no-check-certificate -O outputs/model_weights/va-vae-imagenet256-experimental-variants/ldm-imagenet256-f16d32-50ep.ckpt

# Prepare evaluation batches of ImageNet 256x256 from guided-diffusion
mkdir -p outputs/ADM_npz
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz -O outputs/ADM_npz/VIRTUAL_imagenet256_labeled.npz

# Download files from huggingface using Python script
python download_hf_files.py
# or you can directly download the checkpoints from huggingface: https://huggingface.co/SFD-Project/SFD. Put the files in model_weights/ of SFD-Project/SFD to outputs/train