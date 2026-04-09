"""
Generate and compare qualitative samples across different model configurations
and inference methods. Produces a side-by-side comparison grid.

Usage:
    python compare_samples.py comparison_config.yaml
    python compare_samples.py comparison_config.yaml --num_samples 20
    python compare_samples.py comparison_config.yaml --skip_generation

Config format (YAML):
    num_samples: 10
    num_steps: 100
    seed: 0
    comparison_name: my_comparison
    experiments:
      - label: "Method A"
        config: configs/path/to/config.yaml
        ckpt_step: 60000
        extra_args: "--hidden_sphere_clamp"
      - label: "Method B"
        config: configs/path/to/other.yaml
        ckpt_step: 40000
        ckpt_path: outputs/train/custom/checkpoints/0040000.pt  # optional override
        extra_args: "--encode_reground_t_fix 0.9 --hidden_sphere_clamp"
"""

import argparse
import os
import subprocess
import sys
import yaml
from glob import glob
from PIL import Image, ImageDraw, ImageFont


def derive_ckpt_path(config_path, ckpt_step):
    """Derive checkpoint path from training config and step number."""
    with open(config_path) as f:
        train_config = yaml.safe_load(f)
    exp_name = train_config["train"]["exp_name"]
    return f"outputs/train/{exp_name}/checkpoints/{ckpt_step:07d}.pt"


def run_experiment(exp, exp_name, comparison_dir, num_samples, num_steps, seed):
    """Run inference.py for a single experiment. Returns True on success."""
    config_path = exp["config"]

    if "ckpt_path" in exp:
        ckpt_path = exp["ckpt_path"]
    else:
        ckpt_step = exp.get("ckpt_step", 60000)
        ckpt_path = derive_ckpt_path(config_path, ckpt_step)

    if not os.path.exists(ckpt_path):
        print(f"  WARNING: checkpoint not found: {ckpt_path}")
        return False

    cmd = [
        sys.executable, "inference.py",
        "--config", config_path,
        f"ckpt_path={ckpt_path}",
        f"sample.num_sampling_steps={num_steps}",
        f"sample.per_proc_batch_size={num_samples}",
        f"sample.balanced_sampling=true",
        f"train.output_dir={comparison_dir}",
        f"train.exp_name={exp_name}",
        f"train.global_seed={seed}",
        "--save_samples", str(num_samples),
    ]

    extra_args = exp.get("extra_args", "")
    if extra_args:
        cmd.extend(extra_args.split())

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def find_sample_dir(comparison_dir, exp_name, num_samples=None):
    """Find the directory containing generated PNGs for an experiment."""
    base = os.path.join(comparison_dir, exp_name)
    if not os.path.isdir(base):
        return None
    save_tag = f"-save{num_samples}" if num_samples is not None else None
    candidates = []
    for d in os.listdir(base):
        full = os.path.join(base, d)
        if not os.path.isdir(full) or not glob(os.path.join(full, "*.png")):
            continue
        if save_tag and save_tag not in d:
            continue
        candidates.append(full)
    if not candidates:
        return None
    # Most recently modified if multiple matches
    return max(candidates, key=os.path.getmtime)


def make_comparison_grid(sample_dirs, labels, num_samples, output_path):
    """Assemble a comparison grid from saved sample PNGs."""
    valid = [(d, l) for d, l in zip(sample_dirs, labels) if d is not None]
    if not valid:
        print("No samples found for any experiment!")
        return

    dirs, labs = zip(*valid)

    # Load first image to get dimensions
    first_img = Image.open(os.path.join(dirs[0], "000000.png"))
    img_w, img_h = first_img.size

    n_cols = len(dirs)
    n_rows = min(num_samples, len(glob(os.path.join(dirs[0], "*.png"))))

    pad = 2
    header_h = 32
    grid_w = n_cols * img_w + (n_cols - 1) * pad
    grid_h = n_rows * img_h + (n_rows - 1) * pad + header_h

    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    # Column headers
    for col, label in enumerate(labs):
        x = col * (img_w + pad) + img_w // 2
        draw.text((x, 8), label, fill='black', anchor='mt', font=font)

    # Paste images
    for col, d in enumerate(dirs):
        for row in range(n_rows):
            img_path = os.path.join(d, f"{row:06d}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                x = col * (img_w + pad)
                y = row * (img_h + pad) + header_h
                grid.paste(img, (x, y))

    grid.save(output_path, quality=95)
    print(f"\nComparison grid saved to {output_path}")
    print(f"  {n_cols} experiments x {n_rows} samples, {grid_w}x{grid_h}px")


def main():
    parser = argparse.ArgumentParser(
        description="Compare qualitative samples across experiments")
    parser.add_argument("config", help="YAML config file defining experiments")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Override num_samples from config")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, only assemble grid from existing PNGs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    num_samples = args.num_samples or config.get("num_samples", 10)
    num_steps = config.get("num_steps", 100)
    seed = config.get("seed", 0)
    comparison_name = config.get("comparison_name", "comparison")
    output_base = config.get("output_dir", "outputs/comparisons")
    comparison_dir = os.path.join(output_base, comparison_name)
    experiments = config["experiments"]

    print(f"{'='*60}")
    print(f"  Sample Comparison: {comparison_name}")
    print(f"  Experiments: {len(experiments)}")
    print(f"  Samples per experiment: {num_samples}")
    print(f"  ODE steps: {num_steps}")
    print(f"  Seed: {seed}")
    print(f"  Output: {comparison_dir}/")
    print(f"{'='*60}")

    sample_dirs = []
    labels = []

    for i, exp in enumerate(experiments):
        label = exp.get("label", f"Experiment {i}")
        exp_name = f"exp_{i:02d}"

        print(f"\n--- [{i+1}/{len(experiments)}] {label} ---")

        if not args.skip_generation:
            success = run_experiment(exp, exp_name, comparison_dir,
                                     num_samples, num_steps, seed)
            if not success:
                print(f"  FAILED — skipping")

        sample_dir = find_sample_dir(comparison_dir, exp_name, num_samples)
        if sample_dir:
            print(f"  Samples: {sample_dir}")
        else:
            print(f"  No samples found")

        sample_dirs.append(sample_dir)
        labels.append(label)

    # Assemble grid
    grid_path = os.path.join(comparison_dir, f"{comparison_name}.png")
    os.makedirs(comparison_dir, exist_ok=True)
    make_comparison_grid(sample_dirs, labels, num_samples, grid_path)


if __name__ == "__main__":
    main()
