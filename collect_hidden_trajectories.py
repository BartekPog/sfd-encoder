"""
Collect hidden-token trajectories (h_clean predictions at each ODE step) during
inference with reground, recycle, and linear schedules. For each method + t_fix
combination, runs a small number of samples and saves the predicted h_clean at
every ODE step.

Output: one .npz file per method+t_fix with:
  - trajectories: (num_steps, num_images, num_tokens, dim) float32
  - labels: (num_images,) int — class labels used
  - t_fix: float
  - method: str ("reground", "recycle", or "linear")
  - num_steps: int

Usage:
    python collect_hidden_trajectories.py --config <config.yaml> \\
        --methods reground recycle linear \\
        --t_fix_values 0.7 0.8 0.9 0.95 1.0 \\
        --num_images 100 --num_steps 100
"""

import argparse
import os
import numpy as np
import torch
from omegaconf import OmegaConf

from models import gen_models
from transport import create_transport, Sampler
from tokenizer.vavae import VA_VAE


def load_config(config_path):
    with open(config_path) as f:
        import yaml
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def main():
    parser = argparse.ArgumentParser(description="Collect hidden token trajectories")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--ckpt_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--methods", nargs="+", default=["reground", "recycle", "linear"],
                        choices=["reground", "recycle", "linear"],
                        help="Sampling methods to test")
    parser.add_argument("--t_fix_values", nargs="+", type=float,
                        default=[0.7, 0.8, 0.9, 0.95, 1.0],
                        help="t_fix values to sweep")
    parser.add_argument("--num_images", type=int, default=100,
                        help="Number of images to generate per method")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of Euler ODE steps")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for generation")
    parser.add_argument("--output_dir", default="outputs/trajectory_analysis",
                        help="Output directory for .npz files")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (shared across all methods for comparability)")
    parser.add_argument("--reground_shared_noise", action="store_true", default=False,
                        help="Use shared noise for reground (enc noise = cond noise)")
    parser.add_argument("--reground_fixed_enc_noise", action="store_true", default=False,
                        help="Fix encode noise across ODE steps for reground")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = load_config(args.config)

    # Determine experiment name from config
    exp_name = train_config["train"]["exp_name"]
    ckpt_path = args.ckpt_path

    # Model setup
    semfirst_delta_t = train_config["model"].get("semfirst_delta_t", 0.0)
    semantic_chans = train_config["model"].get("semantic_chans", 0)
    use_hidden = train_config["model"].get("use_hidden_tokens", False)
    assert use_hidden, "Model must have hidden tokens enabled"

    _hidden_kwargs = {}
    if "share_timestep_embedder" in train_config["model"]:
        _hidden_kwargs["share_timestep_embedder"] = train_config["model"]["share_timestep_embedder"]
    if train_config["model"].get("use_encode_mode_emb", False):
        _hidden_kwargs["use_encode_mode_emb"] = True

    model = gen_models[train_config["model"]["model_type"]](
        input_size=train_config["data"]["image_size"] // train_config["vae"]["downsample_ratio"],
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
        semantic_chans=semantic_chans,
        semfirst_delta_t=semfirst_delta_t,
        **_hidden_kwargs,
    ).to(device)

    # Load checkpoint
    from train import load_checkpoint_trusted, load_weights_with_shape_check
    checkpoint = load_checkpoint_trusted(ckpt_path, map_location="cpu")
    if "model" not in checkpoint and "ema" in checkpoint:
        checkpoint["model"] = checkpoint["ema"]
    checkpoint["model"] = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
    # Use EMA weights if available
    if "ema" in checkpoint:
        ema_state = {k.replace("module.", ""): v for k, v in checkpoint["ema"].items()}
        model = load_weights_with_shape_check(model, {"model": ema_state}, rank=0)
    else:
        model = load_weights_with_shape_check(model, checkpoint, rank=0)
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # Transport / sampler
    transport = create_transport(
        train_config["transport"]["path_type"],
        train_config["transport"]["prediction"],
        train_config["transport"].get("loss_weight", None),
        train_config["transport"].get("sample_eps", None),
        train_config["transport"].get("train_eps", None),
    )
    sampler = Sampler(transport)

    latent_size = train_config["data"]["image_size"] // train_config["vae"]["downsample_ratio"]
    in_channels = train_config["model"].get("in_chans", 4)
    num_classes = train_config["data"]["num_classes"]
    normalize_hidden = train_config["model"].get("normalize_hidden", True)
    hidden_rep_guidance = train_config["model"].get("hidden_guidance_scale", 1.0)

    # Output directory
    ckpt_step = os.path.basename(ckpt_path).replace(".pt", "")
    out_dir = os.path.join(args.output_dir, f"{exp_name}_{ckpt_step}")
    os.makedirs(out_dir, exist_ok=True)

    # Run each method + t_fix combination
    # Linear schedule has no t_fix — run it once with a dummy value
    for method in args.methods:
        t_fix_list = [None] if method == "linear" else args.t_fix_values
        for t_fix in t_fix_list:
            print(f"\n{'='*60}")
            if t_fix is not None:
                print(f"Method: {method}, t_fix: {t_fix}")
            else:
                print(f"Method: {method}")
            print(f"{'='*60}")

            # Create sample function with trajectory collection enabled
            sample_fn_kwargs = dict(
                sampling_method="euler",
                num_steps=args.num_steps,
                atol=float(train_config["sample"]["atol"]),
                rtol=float(train_config["sample"]["rtol"]),
                reverse=False,
                timestep_shift=0.0,
                semfirst_delta_t=semfirst_delta_t,
                semantic_chans=semantic_chans,
                num_hidden_tokens=model.num_hidden_tokens,
                hidden_token_dim=model.hidden_token_dim,
                hidden_sphere_clamp=True,
                hidden_rep_guidance=hidden_rep_guidance,
                normalize_hidden=normalize_hidden,
                collect_hidden_trajectory=True,
            )

            if method == "reground":
                sample_fn_kwargs["hidden_schedule"] = "reground"
                sample_fn_kwargs["hidden_reground_t_fix"] = t_fix
                sample_fn_kwargs["reground_shared_noise"] = args.reground_shared_noise
                sample_fn_kwargs["reground_fixed_enc_noise"] = args.reground_fixed_enc_noise
            elif method == "recycle":
                sample_fn_kwargs["hidden_schedule"] = "recycle"
                sample_fn_kwargs["recycle_t_fix"] = t_fix
            elif method == "linear":
                sample_fn_kwargs["hidden_schedule"] = "linear"

            sample_fn = sampler.sample_ode_semantic_first_hidden(**sample_fn_kwargs)

            # Generate images in batches, collecting trajectories
            all_trajectories = []
            all_labels = []
            num_generated = 0

            # Set seed for reproducibility — same seed for every method+t_fix
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            # Reground uses 2 forward passes per step — use smaller batch size
            effective_bs = args.batch_size // 2 if method == "reground" else args.batch_size
            effective_bs = max(effective_bs, 1)

            with torch.no_grad():
                while num_generated < args.num_images:
                    n = min(effective_bs, args.num_images - num_generated)
                    z = torch.randn(n, in_channels, latent_size, latent_size, device=device)
                    y = torch.arange(num_generated, num_generated + n, device=device) % num_classes

                    model_kwargs = dict(y=y)
                    result = sample_fn(z, model.forward, _return_trajectory=True, **model_kwargs)

                    # result: [img_final, trajectory(num_steps, B, tokens, dim)]
                    trajectory = result[-1]  # (num_steps, B, tokens, dim)
                    all_trajectories.append(trajectory.numpy())
                    all_labels.append(y.cpu().numpy())
                    num_generated += n
                    print(f"  Generated {num_generated}/{args.num_images}")

            # Concatenate across batches: (num_steps, total_images, tokens, dim)
            trajectories = np.concatenate(all_trajectories, axis=1)
            labels = np.concatenate(all_labels, axis=0)

            # Save
            extra_tag = ""
            if method == "reground":
                if args.reground_shared_noise:
                    extra_tag += "_shared"
                if args.reground_fixed_enc_noise:
                    extra_tag += "_fixenc"
            if t_fix is not None:
                t_fix_tag = f"_tfix{t_fix:.2f}".replace(".", "")
            else:
                t_fix_tag = ""
            fname = f"trajectory_{method}{t_fix_tag}{extra_tag}_s{args.num_steps}.npz"
            out_path = os.path.join(out_dir, fname)
            np.savez_compressed(
                out_path,
                trajectories=trajectories,
                labels=labels,
                t_fix=t_fix if t_fix is not None else -1.0,
                method=method,
                num_steps=args.num_steps,
                num_images=args.num_images,
                seed=args.seed,
                config=str(args.config),
                exp_name=exp_name,
            )
            print(f"  Saved: {out_path}")
            print(f"  Shape: {trajectories.shape}")

    print(f"\nAll trajectories saved to {out_dir}")


if __name__ == "__main__":
    main()
