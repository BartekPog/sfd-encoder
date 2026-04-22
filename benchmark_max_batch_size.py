"""
Benchmark maximum inference batch size on a single GPU.

Tests increasingly large batch sizes until OOM, then reports the largest
successful size and per-step throughput for each scenario.

Scenarios tested:
  1. No-hidden baseline  (1 forward/step)
  2. Hidden reground      (2 forwards/step: encode + conditioning)
  3. Hidden reground + pure CFG  (3 forwards/step)
  4. Hidden reground + repg      (3 forwards/step)
  5. Hidden reground + CFG + repg (4 forwards/step)

Usage:
  python benchmark_max_batch_size.py [--config CONFIG] [--start 256] [--step 128]

  # Quick SLURM submission:
  sbatch --gres gpu:h200:1 --mem 180G --cpus-per-task 4 --time 00-0:30:00 \
         --wrap "source ~/.bashrc; module load python-waterboa cuda/13.0; \
                 source .venv-sfd/bin/activate; python benchmark_max_batch_size.py"
"""

import argparse
import gc
import os
import time
import warnings

os.environ["PYTHONUNBUFFERED"] = "1"

# Disable torch.compile — the @torch.compile decorators on model methods
# would trigger full recompilation for every new batch size.
# Must monkeypatch BEFORE importing any model code.
import torch
torch.compile = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda f: f)
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()

import sys
import yaml

warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")


def log(msg=""):
    """Print with immediate flush so SLURM job output is visible in real time."""
    print(msg, flush=True)


def load_config(path):
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def build_model(config, device):
    """Build model from config and load to device (no checkpoint)."""
    from models import gen_models
    _hidden_kwargs = {}
    if 'share_timestep_embedder' in config['model']:
        _hidden_kwargs['share_timestep_embedder'] = config['model']['share_timestep_embedder']
    model = gen_models[config['model']['model_type']](
        input_size=config['data']['image_size'] // config['vae']['downsample_ratio'],
        class_dropout_prob=config['model'].get('class_dropout_prob', 0.1),
        num_classes=config['data']['num_classes'],
        use_qknorm=config['model']['use_qknorm'],
        use_swiglu=config['model'].get('use_swiglu', False),
        use_rope=config['model'].get('use_rope', False),
        use_rmsnorm=config['model'].get('use_rmsnorm', False),
        wo_shift=config['model'].get('wo_shift', False),
        in_channels=config['model'].get('in_chans', 4),
        learn_sigma=config['model'].get('learn_sigma', False),
        use_repa=config['model'].get('use_repa', False),
        repa_dino_version=config['model'].get('repa_dino_version', None),
        repa_depth=config['model'].get('repa_feat_depth', None),
        semantic_chans=config['model'].get('semantic_chans', 0),
        semfirst_delta_t=config['model'].get('semfirst_delta_t', 0.0),
        semfirst_infer_interval_mode=config['model'].get('semfirst_infer_interval_mode', 'both'),
        **_hidden_kwargs,
    )
    model.eval().to(device)
    return model


def gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / 1e6


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def try_batch(model, batch_size, num_forwards, device, has_hidden):
    """Run `num_forwards` forward passes at `batch_size`.  Returns (ok, time_s, peak_mb)."""
    latent_size = 16  # 256 / 16
    in_chans = 48     # 32 texture + 16 semantic
    semfirst_delta_t = 0.3

    gc.collect()
    torch.cuda.empty_cache()
    reset_peak()

    try:
        B = batch_size
        x_img = torch.randn(B, in_chans, latent_size, latent_size, device=device, dtype=torch.bfloat16)
        y = torch.randint(0, 1000, (B,), device=device)
        t_sem = torch.rand(B, device=device)
        t_tex = (t_sem - semfirst_delta_t).clamp(min=0.0)

        hidden_kwargs = {}
        if has_hidden:
            hidden_kwargs['x_hidden'] = torch.randn(B, 8, model.hidden_token_dim, device=device, dtype=torch.bfloat16)
            hidden_kwargs['t_hidden'] = torch.full((B,), 0.9, device=device)

        # Warmup
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = model(x_img, t=(t_tex, t_sem), y=y, **hidden_kwargs)
        torch.cuda.synchronize()

        reset_peak()
        start = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(num_forwards):
                _ = model(x_img, t=(t_tex, t_sem), y=y, **hidden_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        peak = gpu_mem_mb()
        # Cleanup
        del x_img, y, hidden_kwargs
        gc.collect()
        torch.cuda.empty_cache()
        return True, elapsed, peak

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return False, 0.0, 0.0


def find_max_batch(model, num_forwards, device, has_hidden, start, step):
    """Exponential growth until OOM, then binary search to refine."""
    best_bs = 0
    best_time = 0.0
    best_peak = 0.0

    # Phase 1: double batch size until OOM
    bs = start
    while True:
        ok, t, peak = try_batch(model, bs, num_forwards, device, has_hidden)
        if ok:
            best_bs, best_time, best_peak = bs, t, peak
            log(f"    BS={bs:5d}  OK   {t:.2f}s  peak={peak:.0f}MB")
            bs *= 2
        else:
            log(f"    BS={bs:5d}  OOM")
            break

    # Phase 2: binary search between best_bs and bs
    lo, hi = best_bs, bs
    while hi - lo > 64:
        mid = (lo + hi) // 2
        mid = (mid // 8) * 8  # align to 8
        if mid <= lo:
            break
        ok, t, peak = try_batch(model, mid, num_forwards, device, has_hidden)
        if ok:
            lo = mid
            best_bs, best_time, best_peak = mid, t, peak
            log(f"    BS={mid:5d}  OK   {t:.2f}s  peak={peak:.0f}MB")
        else:
            hi = mid
            log(f"    BS={mid:5d}  OOM")

    return best_bs, best_time, best_peak


def main():
    parser = argparse.ArgumentParser(description="Find max inference batch size per scenario")
    parser.add_argument("--config-hidden",
                        default="configs/sfd/hidden_1p0_h200_from_ft/v4_mse0001_noisy_enc_nocurr_shift1_repg_1p5.yaml",
                        help="Config for hidden (1p0B) model")
    parser.add_argument("--config-nohidden",
                        default="configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden.yaml",
                        help="Config for no-hidden (1p0B) model")
    parser.add_argument("--start", type=int, default=256,
                        help="Starting batch size")
    parser.add_argument("--step", type=int, default=128,
                        help="Batch size increment for linear search phase")
    args = parser.parse_args()

    log("Starting benchmark...")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
    log(f"GPU: {gpu_name} ({gpu_total_mb:.0f} MB)")
    log(f"Search: start={args.start}, step={args.step}")
    log()

    # ---------- Scenario definitions ----------
    # (name, config_path, has_hidden, num_forwards_per_step)
    scenarios = [
        ("No-hidden baseline (1 fwd/step)",           args.config_nohidden, False, 1),
        ("Hidden reground (2 fwd/step)",               args.config_hidden,   True,  2),
        ("Hidden reground + pure CFG (3 fwd/step)",    args.config_hidden,   True,  3),
        ("Hidden reground + repg (3 fwd/step)",        args.config_hidden,   True,  3),
        ("Hidden reground + CFG + repg (4 fwd/step)",  args.config_hidden,   True,  4),
    ]

    results = []
    for name, config_path, has_hidden, num_fwd in scenarios:
        log(f"=== {name} ===")
        log(f"  Loading config: {config_path}")
        config = load_config(config_path)
        log(f"  Building model...")
        model = build_model(config, device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        log(f"  Model: {config['model']['model_type']} ({param_count:.0f}M params)")

        best_bs, best_time, best_peak = find_max_batch(
            model, num_fwd, device, has_hidden, args.start, args.step)

        if best_bs > 0:
            imgs_per_sec = best_bs / best_time * num_fwd  # total forwards / time
            step_time_ms = best_time / num_fwd * 1000
            results.append((name, num_fwd, best_bs, best_peak, step_time_ms, imgs_per_sec))
        else:
            results.append((name, num_fwd, 0, 0, 0, 0))

        # Free model before loading next
        del model
        gc.collect()
        torch.cuda.empty_cache()
        log()

    # ---------- Summary ----------
    log("=" * 90)
    log(f"{'Scenario':<45} {'Fwd/step':>8} {'MaxBS':>7} {'Peak MB':>8} {'ms/fwd':>8}")
    log("-" * 90)
    for name, num_fwd, bs, peak, step_ms, _ in results:
        if bs > 0:
            log(f"{name:<45} {num_fwd:>8} {bs:>7} {peak:>8.0f} {step_ms:>8.1f}")
        else:
            log(f"{name:<45} {num_fwd:>8} {'OOM':>7} {'—':>8} {'—':>8}")
    log("=" * 90)

    # Practical recommendation
    log()
    log("Recommendation for FID50K inference (50176 images):")
    for name, num_fwd, bs, peak, step_ms, _ in results:
        if bs > 0:
            # Leave ~10% headroom for VAE decode etc.
            safe_bs = (int(bs * 0.9) // 8) * 8
            num_batches = (50176 + safe_bs - 1) // safe_bs
            est_total_min = num_batches * num_fwd * step_ms / 1000 / 60 * 100  # 100 steps
            log(f"  {name}: safe BS={safe_bs}, "
                  f"~{num_batches} batches, "
                  f"~{est_total_min:.0f} min (100 steps)")


if __name__ == "__main__":
    main()
