# Hidden Diffusion / SFD Encoder

## What this project is

This codebase extends **SFD (Semantic-First Diffusion)** — a flow-matching image generation model — with **hidden tokens**: a small set of learnable tokens appended to the image-token sequence that the model uses as a self-encoding bottleneck. The goal is to make the diffusion model simultaneously denoise images and compress/reconstruct hidden representations, enabling richer internal conditioning during generation.

The base SFD model is a DiT (Diffusion Transformer) with velocity prediction on a linear interpolation path (`xt = t*x1 + (1-t)*x0`). It operates in a dual-latent space: 32 texture channels from SD-VAE and 16 semantic channels from a SemVAE (DINOv2 features compressed through a small transformer VAE). The "Semantic First" schedule (`semfirst_delta_t=0.3`) denotes that semantic channels begin denoising before texture channels.

## Architecture

### Models (`models/`)

- **LightningDiT** (`lightningdit.py`): Base DiT with RoPE, RMSNorm, QK-norm, SwiGLU FFN. Sizes: S, B, L, XL, 1p0B, 1p6B. Patch size /1 or /2.
- **HiddenLightningDiT** (`hidden_lightningdit.py`): Extends LightningDiT with `num_hidden_tokens` extra tokens (typically 8). Hidden tokens get their own timestep embedding and learnable position embeddings. RoPE applies only to image tokens. Supports separate or shared timestep embedder, optional encode-mode embedding.
  - Key model variants: `HiddenLightningDiT_B_1_H8` (B-size, 8 hidden), `HiddenLightningDiT_1p0B_H8` (1p0B-size, 8 hidden).

### Transport (`transport/transport.py`)

Three training loss functions:

1. **`training_losses`**: Standard velocity MSE + optional cosine loss + optional REPA (representation alignment with DINOv2 features). Handles SemFirst channel splitting.

2. **`training_losses_hidden`** (3-pass variant):
   - **Pass 1 (encode)**: Clean image (or noisy if `noisy_img_encode`) + pure-noise hidden tokens → predict `h_clean` (the encoding).
   - **Pass 3 (hidden denoise)**: Noisy image + noisy *detached* `h_clean` → hidden denoising loss. Run before Pass 2; if `backward_fn` provided, gradients freed immediately.
   - **Pass 2 (image denoise)**: Noisy image + noisy `h_clean` (with gradient) → image velocity MSE. Hidden loss backpropagates into encoder here.

3. **`training_losses_hidden_merged`** (2-pass variant): Merges Passes 2 and 3 into a single forward pass where both image and hidden losses are computed together. Hidden loss backpropagates into the encoder.

Key training hyperparameters for hidden tokens:
- `hidden_weight`: MSE loss weight for hidden denoising.
- `hidden_guidance_scale`: scale applied to hidden token velocity during encoding.
- `hidden_t_shift`: logit-normal bias for hidden timestep sampling (curriculum).
- `normalize_hidden`: project `h_clean` onto unit sphere.
- `hidden_reg_weight`: sphere-regularisation penalty.
- `noisy_img_encode`: encoder sees noisy image (logit-normal) instead of clean.

### Inference modes (`inference.py`)

For standard (non-hidden) models:
- **No guidance** (`cfg_scale=1.0`): Conditional generation only.
- **CFG** (`cfg_scale>1.0`): Standard classifier-free guidance with null class label. Nearly useless for flow matching models — barely improves FID.
- **Autoguidance** (`sample.autoguidance=true`): Uses a smaller/weaker model (typically B-size) as the "unconditional" prediction instead of null-class. This is what makes guidance work for SFD — the gap between a strong 1p0B and a weak B model provides a real guidance signal.

For hidden-token models, several inference schedules:
- **Semantic** (default): Hidden tokens follow the semantic-channel schedule.
- **Linear**: Hidden tokens denoised linearly from t=0 to `max_t` over ODE steps.
- **Fixed**: Hidden tokens held at a fixed noise level throughout.
- **Encode-first**: Encode real images from the dataset to get `h_clean`, then generate conditioned on those fixed hidden tokens.
- **Encode-reground**: At each ODE step, re-encode `h_clean` from the current `x_t`, noise it to `t_fix`, and condition the next step on it. Purely generative (no dataset lookup needed). This is our current best inference approach.
- **Recycle**: Single forward pass per step — extract `h_clean` from the hidden velocity output and re-noise it.
- **Two-pass**: Pass 1 with linear hidden schedule, Pass 2 with fixed hidden from Pass 1's output.

## Data pipeline

### Encoding (`extract_features.py`, `tokenizer/`)

Raw ImageNet images are pre-encoded into latents stored as safetensors:
- **Texture**: `img_transform` (center crop 256, normalize to [-1,1]) → SD-VAE encoder → 32-channel latent at 16×16.
- **Semantic**: `(x+1)/2` → ImageNet normalize → bicubic interpolate to 224×224 → DINOv2 ViT-B/14 (with registers) → patch tokens → SemVAE encoder → 16-channel latent at 16×16.
- **REPA features**: Same DINOv2 preprocessing → features at depth 2 for representation alignment loss.
- Both original and horizontally-flipped versions are stored.

Dataset path: `datasets/imagenet-sfd-latents/train/sdvae_f16d32_semvaebasech16_repadinov2_vitb14_reg/imagenet_train_256`

### Latent normalization

`ImgLatentDataset` computes per-channel mean/std from 10K random samples, cached as `latents_stats.pt` and `latents_sv_stats.pt`. Applied when `latent_norm: true` / `latent_sv_norm: true` in config.

## Training infrastructure

- **SLURM cluster** with H200 GPUs.
- Launcher: `run_train_slurm_h200.sh <config> <num_chains> <num_gpus>` — chains SLURM jobs for long training.
- Batch scripts: `batch_*.sh` files submit groups of experiments.
- Configs: `configs/sfd/` organized by model size and init source:
  - `hidden_b_h200_from_ft/` — B-size hidden models initialized from pretrained+finetuned checkpoints.
  - `hidden_1p0_h200_from_ft/` — 1p0B-size hidden models from pretrained checkpoints.
  - `autoguidance_b/` — B-size autoguidance (weak) model configs.
- Accelerate + DDP for multi-GPU. Mixed precision (bf16).
- W&B logging via `ENABLE_WANDB=1` env var.

## Evaluation

- **FID50K**: Primary metric. `inference.py` generates 50K balanced samples → computes FID against `outputs/ADM_npz/VIRTUAL_imagenet256_labeled.npz`.
- Results saved via `save_fid_result.py` to `results/fid_summary.csv`.
- Standard eval: Euler sampler, 100 steps, balanced class sampling.
- Inference batch scripts: `batch_run_inference_*.sh`, `batch_cfg_test_*.sh`.

## Current model zoo

### Pretrained (from SFD authors)
- `outputs/train/sfd_1p0/checkpoints/4000000.pt` — 1p0B EMA weights (4M steps). FID 2.52 (no guidance), 1.05 (autoguidance).
- `outputs/train/sfd_1p0/checkpoints/4000000_full.pt` — Same with model + EMA + optimizer (obtained directly from authors).
- `outputs/train/sfd_autoguidance_b/checkpoints/0070000.pt` — B-size autoguidance model, EMA only.

### Finetuned on our distribution
- `outputs/train/1p0_finetune_no_hidden_warm8k/` — 1p0B finetuned 100K steps on our encoded dataset, 8K LR warmup, from full 4M checkpoint.
- `outputs/train/b_autoguidance_finetune_no_hidden_warm8k/` — B-size autoguidance finetuned on our distribution.

### Hidden-token experiments
- `outputs/train/v4_*` — V4 generation hidden-token experiments (B-size and 1p0B).
- Config naming convention: `v4_mse{weight}_noisy_enc_{curriculum/nocurr}_shift{val}_repg_{val}[_hgd_{val}][_merged].yaml`

## Key files

| File | Purpose |
|------|---------|
| `train.py` | Main training loop. Handles weight init, optimizer restore, LR warmup, hidden-token routing. |
| `inference.py` | Sampling with all inference modes (standard, CFG, autoguidance, hidden schedules, encode-reground). |
| `transport/transport.py` | Flow matching: path sampling, training losses (standard, 3-pass hidden, merged hidden), ODE sampling. |
| `models/lightningdit.py` | Base LightningDiT architecture. |
| `models/hidden_lightningdit.py` | HiddenLightningDiT with hidden token support, CFG, and autoguidance forward passes. |
| `extract_features.py` | Encode raw images to dual-latent (SD-VAE + SemVAE) + REPA features. |
| `tokenizer/vavae.py` | VA-VAE wrapper (SD-VAE + SemVAE decode for image reconstruction). |
| `tokenizer/semvae/` | SemVAE: DINOv2 feature extraction + transformer VAE compression. |
| `dataset/img_latent_dataset.py` | Dataset class for pre-encoded latents in safetensors format. |
| `save_fid_result.py` | Append FID results + full config metadata to `results/fid_summary.csv`. |
| `run_train_slurm_h200.sh` | SLURM launcher with job chaining. |
| `run_inference.sh` | SLURM inference launcher. |
