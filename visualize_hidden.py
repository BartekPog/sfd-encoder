"""
Visualize what hidden tokens encode by generating grids of images
from varying hidden-noise levels and image-noise seeds.

Each grid (one per model/class):
  - Columns: reference + N different image noise seeds
  - Rows: different t_inith values (how clean the hidden tokens are)
    t_inith=0.0 → pure noise hidden tokens
    t_inith=1.0 → fully clean hidden tokens
"""

import os
import math
import argparse
import logging
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf

from tokenizer.vavae import VA_VAE
from models import gen_models
from transport import create_transport, Sampler
from dataset.img_latent_dataset import ImgLatentDataset

logger = logging.getLogger(__name__)


def make_grid_image(images_2d, row_labels=None, col_labels=None, cell_size=256, pad=4, label_height=30):
    """
    Compose a 2-D list of PIL images into a single labelled grid.

    Args:
        images_2d: list[list[PIL.Image]] — images_2d[row][col]
        row_labels: optional list of strings for each row
        col_labels: optional list of strings for each column
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
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    if col_labels:
        for c, label in enumerate(col_labels):
            x = left_margin + pad + c * (cell_size + pad) + cell_size // 2
            draw.text((x, 2), label, fill=(0, 0, 0), font=font, anchor="mt")

    if row_labels:
        for r, label in enumerate(row_labels):
            y = top_margin + pad + r * (cell_size + pad) + cell_size // 2
            draw.text((2, y), label, fill=(0, 0, 0), font=font, anchor="lm")

    for r, row in enumerate(images_2d):
        for c, img in enumerate(row):
            if img is None:
                continue
            img_resized = img.resize((cell_size, cell_size), Image.LANCZOS)
            x = left_margin + pad + c * (cell_size + pad)
            y = top_margin + pad + r * (cell_size + pad)
            canvas.paste(img_resized, (x, y))

    return canvas


def decode_latent(latent, vae, latent_mean, latent_std, latent_multiplier, semantic_chans):
    """Decode a single latent (or batch) to PIL images."""
    if semantic_chans > 0:
        latent = latent[:, :-semantic_chans]
    latent = (latent * latent_std) / latent_multiplier + latent_mean
    imgs_np = vae.decode_to_images(latent)
    return [Image.fromarray(img) for img in imgs_np]


def main():
    parser = argparse.ArgumentParser(description="Visualize hidden token encodings")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--num_vis", type=int, default=5, help="Number of visualisations per model")
    parser.add_argument("--num_noise_seeds", type=int, default=5, help="Columns of different image noise seeds")
    parser.add_argument("--t_inith_values", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated t_inith values for rows")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

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

    # Load dataset stats for latent denormalization
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
    )

    # Reference generation uses linear hidden schedule
    sample_fn_ref = sampler.sample_ode_semantic_first_hidden(
        **common_ode_kwargs, hidden_schedule="linear",
    )

    # VAE
    vae = VA_VAE(f'tokenizer/configs/{train_config["vae"]["model_name"]}.yaml')

    num_classes = train_config["data"]["num_classes"]
    using_cfg = args.cfg_scale > 1.0

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    for vis_idx in range(args.num_vis):
        class_label = torch.randint(0, num_classes, (1,), generator=rng, device=device).item()
        print(f"Visualisation {vis_idx+1}/{args.num_vis}: class={class_label}")

        # Fixed noise for hidden tokens (used to re-noise h_clean)
        eps_h = torch.randn(1, model.num_hidden_tokens, model.hidden_token_dim,
                            device=device, generator=rng)

        # Reference image noise
        z_img_ref = torch.randn(1, model.in_channels, latent_size, latent_size,
                                device=device, generator=rng)

        # Image noise seeds for grid columns (pre-generate for reproducibility)
        z_imgs = []
        for _ in range(args.num_noise_seeds):
            z_imgs.append(torch.randn(1, model.in_channels, latent_size, latent_size,
                                      device=device, generator=rng))

        # --- Reference generation (linear hidden schedule) ---
        y_ref = torch.tensor([class_label], device=device)
        if using_cfg:
            z_ref_cfg = torch.cat([z_img_ref, z_img_ref], 0)
            y_ref_cfg = torch.cat([y_ref, torch.tensor([1000], device=device)], 0)
            model_kwargs_ref = dict(y=y_ref_cfg, cfg_scale=args.cfg_scale,
                                    cfg_interval=True, cfg_interval_start=0)
            model_fn = model.forward_with_cfg
        else:
            z_ref_cfg = z_img_ref
            model_kwargs_ref = dict(y=y_ref)
            model_fn = model.forward

        result_ref = sample_fn_ref(z_ref_cfg, model_fn, _return_hidden=True, **model_kwargs_ref)
        img_ref_latent = result_ref[0]  # (B, C, H, W) — B=2 if CFG
        h_clean = result_ref[1]         # (B, num_hidden, dim)

        if using_cfg:
            img_ref_latent = img_ref_latent[:1]
            h_clean = h_clean[:1]

        ref_pil = decode_latent(img_ref_latent, vae, latent_mean, latent_std,
                                latent_multiplier, semantic_chans)[0]

        # --- Grid generation ---
        grid_rows = []
        for t_inith in t_inith_values:
            # Noise hidden tokens to level t_inith
            # h_noised = t_inith * h_clean + (1 - t_inith) * eps_h
            h_noised = t_inith * h_clean + (1.0 - t_inith) * eps_h

            # Create sample_fn for this t_inith
            sample_fn_grid = sampler.sample_ode_semantic_first_hidden(
                **common_ode_kwargs,
                hidden_schedule="linear_from",
                hidden_schedule_start_t=t_inith,
            )

            row_images = [ref_pil]  # first column is always the reference

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
        col_labels = ["ref"] + [f"seed {j}" for j in range(args.num_noise_seeds)]
        grid_img = make_grid_image(grid_rows, row_labels=row_labels, col_labels=col_labels)

        save_path = os.path.join(output_dir, f"vis_{vis_idx:02d}_class{class_label}.png")
        grid_img.save(save_path)
        print(f"  Saved: {save_path}")

    print(f"Done. {args.num_vis} visualisations saved to {output_dir}")


if __name__ == "__main__":
    main()
