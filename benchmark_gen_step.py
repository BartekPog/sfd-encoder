"""Single-trial max-batch-size probe for inference.

Builds the model, runs N forward passes at the requested per-GPU batch size
under bf16 autocast (no grads, no optimizer), and reports peak GPU memory +
mean forward time. The `--num_forwards` flag controls how many forwards are
issued per "step" to simulate the different inference scenarios:

  1  — linear hidden schedule (1 fwd/step)
  2  — reground w/o CFG or repg (encode + cond)
  3  — reground + pure CFG OR reground + repg (encode + cond + {uncond | repg})
  4  — reground + CFG + repg

Intended to be called repeatedly (once per gen_bsz) from
run_benchmark_max_batch_gen_viper.sh so that an OOM in one trial does not
poison the next.

Exit codes:
  0   — success; prints a "RESULT bs=<B> peak_mb=<M> gen_ms=<T>" line.
  non-zero — failure; the caller records it.
"""

import argparse
import time

import torch
from omegaconf import OmegaConf


def build_model(cfg, device):
    from models import gen_models

    extra = {}
    if 'share_timestep_embedder' in cfg['model']:
        extra['share_timestep_embedder'] = cfg['model']['share_timestep_embedder']

    model = gen_models[cfg['model']['model_type']](
        input_size=cfg['data']['image_size'] // cfg['vae']['downsample_ratio'],
        class_dropout_prob=cfg['model'].get('class_dropout_prob', 0.1),
        num_classes=cfg['data']['num_classes'],
        use_qknorm=cfg['model']['use_qknorm'],
        use_swiglu=cfg['model'].get('use_swiglu', False),
        use_rope=cfg['model'].get('use_rope', False),
        use_rmsnorm=cfg['model'].get('use_rmsnorm', False),
        wo_shift=cfg['model'].get('wo_shift', False),
        in_channels=cfg['model'].get('in_chans', 4),
        learn_sigma=cfg['model'].get('learn_sigma', False),
        use_repa=cfg['model'].get('use_repa', False),
        repa_dino_version=cfg['model'].get('repa_dino_version', None),
        repa_depth=cfg['model'].get('repa_feat_depth', None),
        semantic_chans=cfg['model'].get('semantic_chans', 0),
        semfirst_delta_t=cfg['model'].get('semfirst_delta_t', 0.0),
        semfirst_infer_interval_mode=cfg['model'].get('semfirst_infer_interval_mode', 'both'),
        **extra,
    )
    model.eval().to(device)
    return model


def run(cfg, gen_bsz, num_forwards, warmup_trials, measure_trials):
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)

    model = build_model(cfg, device)
    num_hidden = getattr(model, 'num_hidden_tokens', 8)
    hidden_token_dim = getattr(model, 'hidden_token_dim', None)

    latent_size = cfg['data']['image_size'] // cfg['vae']['downsample_ratio']
    in_chans = cfg['model']['in_chans']
    num_classes = cfg['data']['num_classes']
    has_hidden = cfg['model'].get('use_hidden_tokens', False)
    semfirst_delta_t = cfg['model'].get('semfirst_delta_t', 0.0)

    B = gen_bsz
    x = torch.randn(B, in_chans, latent_size, latent_size, device=device, dtype=torch.bfloat16)
    y = torch.randint(0, num_classes, (B,), device=device)
    t_sem = torch.rand(B, device=device)
    t_tex = (t_sem - semfirst_delta_t).clamp(min=0.0)

    hidden_kwargs = {}
    if has_hidden:
        assert hidden_token_dim is not None
        hidden_kwargs['x_hidden'] = torch.randn(B, num_hidden, hidden_token_dim,
                                                device=device, dtype=torch.bfloat16)
        hidden_kwargs['t_hidden'] = torch.full((B,), 0.9, device=device)

    # Warmup.
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(warmup_trials):
            for _ in range(num_forwards):
                _ = model(x, t=(t_tex, t_sem), y=y, **hidden_kwargs)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    trial_times = []
    for _ in range(measure_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(num_forwards):
                _ = model(x, t=(t_tex, t_sem), y=y, **hidden_kwargs)
        torch.cuda.synchronize()
        trial_times.append(time.perf_counter() - t0)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    mean_gen_ms = 1000.0 * sum(trial_times) / max(1, len(trial_times))
    return peak_bytes, mean_gen_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--gen_bsz', type=int, required=True)
    p.add_argument('--num_forwards', type=int, default=2,
                   help='Forwards per step: 1=linear, 2=reground, 3=reground+cfg or +repg, 4=+cfg+repg')
    p.add_argument('--warmup_trials', type=int, default=1)
    p.add_argument('--measure_trials', type=int, default=2)
    args = p.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    peak_bytes, gen_ms = run(cfg, args.gen_bsz, args.num_forwards,
                             args.warmup_trials, args.measure_trials)
    peak_mb = peak_bytes / (1024 * 1024)
    print(f"RESULT bs={args.gen_bsz} peak_mb={peak_mb:.0f} gen_ms={gen_ms:.0f}", flush=True)


if __name__ == '__main__':
    main()
