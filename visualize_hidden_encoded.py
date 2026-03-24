"""
Visualize hidden token encodings from *real images* by generating grids where
the hidden state starts from a true encoding (optionally noised).

Each grid (one per real image):
  - Column 0: the original real image (ground truth)
  - Columns 1..N: images generated from different image-noise seeds,
    conditioned on the encoded (and possibly noised) hidden tokens
  - Rows: different t_inith values controlling hidden-noise level
    t_inith=0.0 → pure noise hidden tokens (encode_linear from 0.0)
    t_inith=1.0 → fully clean encoded hidden tokens (fixed)
"""

import os
import random
import argparse
import logging
import torch
import numpy as np
from glob import glob
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from safetensors import safe_open as _safe_open

from tokenizer.vavae import VA_VAE
from models import gen_models
from transport import create_transport, Sampler
from dataset.img_latent_dataset import ImgLatentDataset

logger = logging.getLogger(__name__)


def make_grid_image(images_2d, row_labels=None, col_labels=None,
                    cell_size=256, pad=4, label_height=70, font_size=20):
    """
    Compose a 2-D list of PIL images into a single labelled grid.

    Args:
        images_2d: list[list[PIL.Image]] — images_2d[row][col]
        row_labels: optional list of strings for each row
        col_labels: optional list of strings for each column
        label_height: pixel width reserved for row / column label margins
        font_size: font size for labels
    """
    nrows = len(images_2d)
    ncols = max(len(row) for row in images_2d)

    left_margin = label_height if row_labels else 0
    top_margin = label_height if col_labels else 0

    width = left_margin + ncols * (cell_size + pad) + pad
    height = top_margin + nrows * (cell_size + pad) + pad

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    if col_labels:
        for c, label in enumerate(col_labels):
            x = left_margin + pad + c * (cell_size + pad) + cell_size // 2
            draw.text((x, 4), label, fill=(0, 0, 0), font=font, anchor="mt")

    if row_labels:
        for r, label in enumerate(row_labels):
            y = top_margin + pad + r * (cell_size + pad) + cell_size // 2
            draw.text((4, y), label, fill=(0, 0, 0), font=font, anchor="lm")

    for r, row in enumerate(images_2d):
        for c, img in enumerate(row):
            if img is None:
                continue
            img_resized = img.resize((cell_size, cell_size), Image.LANCZOS)
            x = left_margin + pad + c * (cell_size + pad)
            y = top_margin + pad + r * (cell_size + pad)
            canvas.paste(img_resized, (x, y))

    return canvas


def encode_hidden_from_image(model, x1_batch, y_batch, semfirst_delta_t, normalize_hidden, device):
    """
    Single-step hidden-token encoding (mirrors Pass 1 in training_losses_hidden).

    Given clean image latents *x1_batch* (already normalised & on device), run one
    forward pass with t_img=1 (clean) and t_hid=0 (pure noise) to recover the
    clean hidden encoding  h_clean = x0_h + v_h.

    Returns:
        h_clean  (B, num_hidden_tokens, hidden_token_dim)  on *device*
    """
    B = x1_batch.shape[0]
    num_hidden_tokens = model.num_hidden_tokens
    hidden_token_dim = model.hidden_token_dim

    x0_h = torch.randn(B, num_hidden_tokens, hidden_token_dim, device=device)
    t_encode_img = torch.ones(B, device=device)
    t_encode_hid = torch.zeros(B, device=device)

    if semfirst_delta_t > 0:
        t_encode_img_for_model = (t_encode_img, t_encode_img)
    else:
        t_encode_img_for_model = t_encode_img

    with torch.no_grad():
        out = model(x1_batch, t=t_encode_img_for_model, y=y_batch,
                    x_hidden=x0_h, t_hidden=t_encode_hid)
    h_velocity = out[1]
    h_clean = x0_h + h_velocity

    if normalize_hidden:
        h_clean = h_clean / h_clean.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    return h_clean


def decode_latent(latent, vae, latent_mean, latent_std, latent_multiplier, semantic_chans):
    """Decode a single latent (or batch) to PIL images."""
    if semantic_chans > 0:
        latent = latent[:, :-semantic_chans]
    latent = (latent * latent_std) / latent_multiplier + latent_mean
    imgs_np = vae.decode_to_images(latent)
    return [Image.fromarray(img) for img in imgs_np]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hidden token encodings from real images")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--num_vis", type=int, default=5,
                        help="Number of visualisations (real images) per model")
    parser.add_argument("--num_noise_seeds", type=int, default=5,
                        help="Columns of different image noise seeds")
    parser.add_argument("--t_inith_values", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated t_inith values for rows")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_sphere_clamp", action="store_true", default=False,
                        help="Project each hidden token's clean prediction onto the unit sphere "
                             "at every sampling step.")
    parser.add_argument("--hidden_rep_guidance", type=float, default=1.0,
                        help="Hidden representation guidance scale. >1.0 amplifies the model's "
                             "response to hidden conditioning via a linearly-ramped CFG-like "
                             "combination w(t_h) = 1 + (scale-1)*t_h. Default 1.0 (off).")

    args, unknown = parser.parse_known_args()
    train_config = OmegaConf.load(args.config)
    if unknown:
        train_config = OmegaConf.merge(train_config, OmegaConf.from_dotlist(unknown))

    t_inith_values = [float(v) for v in args.t_inith_values.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # Output directory
    exp_name = train_config["train"]["exp_name"]
    if args.output_dir is None:
        output_dir = f"outputs/visualizations/{exp_name}"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset for real images + latent denormalization stats
    dataset = ImgLatentDataset(
        data_dir=train_config["data"]["data_path"],
        latent_norm=train_config["data"].get("latent_norm", False),
        latent_sv_norm=train_config["data"].get("latent_sv_norm", False),
        latent_multiplier=train_config["data"].get("latent_multiplier", 0.18215),
    )
    latent_mean, latent_std = dataset.get_latent_stats()
    latent_multiplier = train_config["data"].get("latent_multiplier", 0.18215)
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    # Build class → dataset-index mapping for class-matched sampling
    print("Building class index from dataset shards...")
    cls2idx = defaultdict(list)
    global_offset = 0
    for shard_path in sorted(glob(os.path.join(
            train_config["data"]["data_path"], "*.safetensors"))):
        with _safe_open(shard_path, framework="pt", device="cpu") as f:
            labels_tensor = f.get_slice("labels")[:]
        for local_i in range(labels_tensor.shape[0]):
            lbl = int(labels_tensor[local_i].item())
            cls2idx[lbl].append(global_offset + local_i)
        global_offset += labels_tensor.shape[0]
    for k in cls2idx:
        random.Random(args.seed).shuffle(cls2idx[k])
    print(f"  {len(cls2idx)} classes, {global_offset} total samples")

    # Model
    if "downsample_ratio" in train_config["vae"]:
        downsample_ratio = train_config["vae"]["downsample_ratio"]
    else:
        downsample_ratio = 16
    latent_size = train_config["data"]["image_size"] // downsample_ratio

    _hidden_kwargs = {}
    if "share_timestep_embedder" in train_config["model"]:
        _hidden_kwargs["share_timestep_embedder"] = train_config["model"]["share_timestep_embedder"]
    model = gen_models[train_config["model"]["model_type"]](
        input_size=latent_size,
        class_dropout_prob=train_config["model"].get("class_dropout_prob", 0.1),
        num_classes=train_config["data"]["num_classes"],
        use_qknorm=train_config["model"]["use_qknorm"],
        use_swiglu=train_config["model"].get("use_swiglu", False),
        use_rope=train_config["model"].get("use_rope", False),
        use_rmsnorm=train_config["model"].get("use_rmsnorm", False),
        wo_shift=train_config["model"].get("wo_shift", False),
        in_channels=train_config["model"].get("in_chans", 4),
        learn_sigma=train_config["model"].get("learn_sigma", False),
        use_repa=train_config["model"].get("use_repa", False),
        repa_dino_version=train_config["model"].get("repa_dino_version", None),
        repa_depth=train_config["model"].get("repa_feat_depth", None),
        semantic_chans=train_config["model"].get("semantic_chans", 0),
        semfirst_delta_t=train_config["model"].get("semfirst_delta_t", 0.0),
        semfirst_infer_interval_mode=train_config["model"].get("semfirst_infer_interval_mode", "both"),
        **_hidden_kwargs,
    )

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval().to(device)
    print(f"Loaded model from {args.ckpt_path}")

    assert hasattr(model, "num_hidden_tokens"), "Model must have hidden tokens for visualisation"

    semantic_chans = train_config["model"].get("semantic_chans", 0)
    semfirst_delta_t = train_config["model"].get("semfirst_delta_t", 0.0)
    normalize_hidden = train_config["model"].get("normalize_hidden", True)

    # Transport / sampler
    transport = create_transport(
        train_config["transport"]["path_type"],
        train_config["transport"]["prediction"],
        train_config["transport"]["loss_weight"],
        train_config["transport"]["train_eps"],
        train_config["transport"]["sample_eps"],
        use_cosine_loss=train_config["transport"].get("use_cosine_loss", False),
        use_lognorm=train_config["transport"].get("use_lognorm", False),
        semantic_weight=train_config["model"].get("semantic_weight", 1.0),
        semantic_chans=semantic_chans,
        semfirst_delta_t=semfirst_delta_t,
        repa_weight=train_config["model"].get("repa_weight", 1.0),
        repa_mode=train_config["model"].get("repa_mode", "cos"),
    )
    sampler = Sampler(transport)

    common_ode_kwargs = dict(
        sampling_method="euler",
        num_steps=args.num_sampling_steps,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=0.0,
        semfirst_delta_t=semfirst_delta_t,
        semantic_chans=semantic_chans,
        num_hidden_tokens=model.num_hidden_tokens,
        hidden_token_dim=model.hidden_token_dim,
        hidden_sphere_clamp=args.hidden_sphere_clamp,
        hidden_rep_guidance=args.hidden_rep_guidance,
    )

    if args.hidden_sphere_clamp:
        print("Hidden sphere clamping: enabled")
    if args.hidden_rep_guidance > 1.0:
        print(f"Hidden representation guidance: scale={args.hidden_rep_guidance}")

    # VAE
    vae = VA_VAE(f'tokenizer/configs/{train_config["vae"]["model_name"]}.yaml')

    num_classes = train_config["data"]["num_classes"]
    using_cfg = args.cfg_scale > 1.0

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    cls_ptr = defaultdict(int)  # round-robin pointer per class

    for vis_idx in range(args.num_vis):
        # Pick a random class, fetch a real image of that class
        class_label = torch.randint(0, num_classes, (1,), generator=rng, device=device).item()
        indices = cls2idx[class_label]
        ptr = cls_ptr[class_label]
        dataset_idx = indices[ptr % len(indices)]
        cls_ptr[class_label] = ptr + 1

        item = dataset[dataset_idx]
        x1_real = item[0].unsqueeze(0).to(device)  # (1, C, H, W)
        y_real = torch.tensor([class_label], device=device)

        print(f"Visualisation {vis_idx+1}/{args.num_vis}: class={class_label}, "
              f"dataset_idx={dataset_idx}")

        # Decode the original real image for the left column
        orig_pil = decode_latent(x1_real, vae, latent_mean, latent_std,
                                 latent_multiplier, semantic_chans)[0]

        # Encode hidden tokens from this real image
        h_clean = encode_hidden_from_image(
            model, x1_real, y_real, semfirst_delta_t, normalize_hidden, device)

        # Fixed noise for hidden tokens (used to re-noise h_clean)
        eps_h = torch.randn(1, model.num_hidden_tokens, model.hidden_token_dim,
                            device=device, generator=rng)

        # Image noise seeds for grid columns (pre-generate for reproducibility)
        z_imgs = []
        for _ in range(args.num_noise_seeds):
            z_imgs.append(torch.randn(1, model.in_channels, latent_size, latent_size,
                                      device=device, generator=rng))

        # --- Grid generation ---
        grid_rows = []
        for t_inith in t_inith_values:
            # Noise hidden tokens to level t_inith
            h_noised = t_inith * h_clean + (1.0 - t_inith) * eps_h

            # Create sample_fn with encode_linear schedule at this start_t
            sample_fn_grid = sampler.sample_ode_semantic_first_hidden(
                **common_ode_kwargs,
                hidden_schedule="encode_linear",
                hidden_schedule_start_t=t_inith,
            )

            row_images = [orig_pil]  # first column: original real image

            for seed_j in range(args.num_noise_seeds):
                z_img_j = z_imgs[seed_j]
                y_j = torch.tensor([class_label], device=device)

                if using_cfg:
                    z_j_cfg = torch.cat([z_img_j, z_img_j], 0)
                    y_j_cfg = torch.cat([y_j, torch.tensor([1000], device=device)], 0)
                    h_noised_cfg = torch.cat([h_noised, h_noised], 0)
                    mk = dict(y=y_j_cfg, cfg_scale=args.cfg_scale,
                              cfg_interval=True, cfg_interval_start=0)
                    mfn = model.forward_with_cfg
                else:
                    z_j_cfg = z_img_j
                    h_noised_cfg = h_noised
                    mk = dict(y=y_j)
                    mfn = model.forward

                gen_result = sample_fn_grid(z_j_cfg, mfn, _z_hidden=h_noised_cfg, **mk)
                gen_latent = gen_result[-1]
                if using_cfg:
                    gen_latent = gen_latent[:1]

                gen_pil = decode_latent(gen_latent, vae, latent_mean, latent_std,
                                        latent_multiplier, semantic_chans)[0]
                row_images.append(gen_pil)

            grid_rows.append(row_images)

        # --- Compose and save grid ---
        row_labels = [f"t={v:.1f}" for v in t_inith_values]
        col_labels = ["original"] + [f"seed {j}" for j in range(args.num_noise_seeds)]
        grid_img = make_grid_image(grid_rows, row_labels=row_labels, col_labels=col_labels)

        suffix = "_sphereclamp" if args.hidden_sphere_clamp else ""
        if args.hidden_rep_guidance > 1.0:
            suffix += f"_hrg{args.hidden_rep_guidance:.1f}"
        save_path = os.path.join(
            output_dir, f"vis_enc_{vis_idx:02d}_class{class_label}{suffix}.png")
        grid_img.save(save_path)
        print(f"  Saved: {save_path}")

    print(f"Done. {args.num_vis} visualisations saved to {output_dir}")


if __name__ == "__main__":
    main()
