"""Single-trial max-batch-size probe for training.

Loads a training config, builds the real model + transport, runs a few warmup
+ measured training steps at the requested per-GPU batch size, and reports
peak GPU memory + mean step time. Intended to be called repeatedly (once per
batch size) from run_benchmark_max_batch_viper.sh so that an OOM in one trial
does not poison the next.

The probe mirrors the actual training loop in train.py: it dispatches to
`transport.training_losses_hidden` when `use_hidden_tokens` is set, and to
`transport.training_losses` otherwise. Inputs (image latents, labels, DINO
features) are random tensors of the shapes the real dataset produces.

Exit codes:
  0   — success; prints a "RESULT bs=<B> peak_mb=<M> step_ms=<T>" line.
  non-zero — failure (OOM, import error, etc.); the caller records it.
"""

import argparse
import contextlib
import os
import sys
import time

import torch
from omegaconf import OmegaConf


def build_inputs(cfg, batch_size, device):
    img_size = cfg['data']['image_size']
    down = cfg['vae']['downsample_ratio']
    latent_size = img_size // down
    in_chans = cfg['model']['in_chans']
    num_classes = cfg['data']['num_classes']

    x = torch.randn(batch_size, in_chans, latent_size, latent_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)

    feature_dino = None
    if cfg['model'].get('use_repa', False):
        dino_dim = {
            'dinov2_vits14': 384, 'dinov2_vits14_reg': 384,
            'dinov2_vitb14': 768, 'dinov2_vitb14_reg': 768,
            'dinov2_vitl14': 1024, 'dinov2_vitl14_reg': 1024,
        }[cfg['model']['repa_dino_version']]
        feature_dino = torch.randn(batch_size, dino_dim, latent_size, latent_size, device=device)

    return x, y, feature_dino


def build_model_and_transport(cfg, device):
    from models import gen_models
    from transport import create_transport

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
        **({'share_timestep_embedder': cfg['model']['share_timestep_embedder']}
           if 'share_timestep_embedder' in cfg['model'] else {}),
    )
    model.to(device).train()

    transport = create_transport(
        cfg['transport']['path_type'],
        cfg['transport']['prediction'],
        cfg['transport']['loss_weight'],
        cfg['transport']['train_eps'],
        cfg['transport']['sample_eps'],
        use_cosine_loss=cfg['transport'].get('use_cosine_loss', False),
        use_lognorm=cfg['transport'].get('use_lognorm', False),
        semantic_weight=cfg['model'].get('semantic_weight', 1.0),
        semantic_chans=cfg['model'].get('semantic_chans', 0),
        semfirst_delta_t=cfg['model'].get('semfirst_delta_t', 0.0),
        repa_weight=cfg['model'].get('repa_weight', 1.0),
        repa_mode=cfg['model'].get('repa_mode', 'cos'),
    )
    return model, transport


def training_step(model, transport, cfg, x, y, feature_dino):
    use_repa = cfg['model'].get('use_repa', False)
    use_hidden = cfg['model'].get('use_hidden_tokens', False)
    model_kwargs = dict(y=y)

    if use_hidden:
        def _backward_fn(loss):
            loss.backward()

        loss_dict = transport.training_losses_hidden(
            model, x, model_kwargs,
            use_repa=use_repa, feature_dino=feature_dino,
            hidden_weight=cfg['model'].get('hidden_weight', 1.0),
            normalize_hidden=cfg['model'].get('normalize_hidden', True),
            hidden_reg_weight=cfg['model'].get('hidden_reg_weight', 0.01),
            hidden_cos_weight=cfg['model'].get('hidden_cos_weight', 0.0),
            backward_fn=_backward_fn,
            hidden_same_t_as_img=cfg['model'].get('hidden_same_t_as_img', False),
            noisy_img_encode=cfg['model'].get('noisy_img_encode', False),
            hidden_t_shift=cfg['model'].get('hidden_t_shift_final', 0.0),
            hidden_loss_scale=1.0,
            hidden_grad_dyn_scale=cfg['model'].get('hidden_grad_dyn_scale', 0.0),
            hidden_grad_static_scale=cfg['model'].get('hidden_grad_static_scale', 1.0),
            use_encode_mode_emb=cfg['model'].get('use_encode_mode_emb', False),
            hidden_guidance_scale=cfg['model'].get('hidden_guidance_scale', 1.0),
            hidden_reuse_noise_pass2=cfg['model'].get('hidden_reuse_noise_pass2', False),
            hidden_reuse_noise_pass3=cfg['model'].get('hidden_reuse_noise_pass3', False),
            hidden_clean_only_pass2=cfg['model'].get('hidden_clean_only_pass2', False),
            hidden_dropout_prob=cfg['model'].get('hidden_dropout_prob', 0.0),
            sync_class_dropout=cfg['model'].get('sync_class_dropout', False),
            encoder_model=None,
        )
    else:
        loss_dict = transport.training_losses(
            model, x, model_kwargs,
            use_repa=use_repa, feature_dino=feature_dino,
        )

    if 'cos_loss' in loss_dict and 'repa_loss' in loss_dict:
        loss = loss_dict['loss'].mean() + loss_dict['cos_loss'].mean() + loss_dict['repa_loss'].mean()
    elif 'cos_loss' in loss_dict:
        loss = loss_dict['loss'].mean() + loss_dict['cos_loss'].mean()
    else:
        loss = loss_dict['loss'].mean()

    if 'hidden_loss' in loss_dict and loss_dict['hidden_loss'].requires_grad:
        loss = loss + cfg['model'].get('hidden_weight', 1.0) * loss_dict['hidden_loss'].mean()
    if 'hidden_reg_loss' in loss_dict and loss_dict['hidden_reg_loss'].requires_grad:
        loss = loss + cfg['model'].get('hidden_reg_weight', 0.0) * loss_dict['hidden_reg_loss'].mean()

    return loss


def run(cfg, batch_size, warmup_steps, measure_steps, precision):
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)

    model, transport = build_model_and_transport(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x, y, feature_dino = build_inputs(cfg, batch_size, device)

    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'no': torch.float32}[precision]
    autocast_ctx = (torch.amp.autocast('cuda', dtype=dtype)
                    if precision != 'no' else contextlib.nullcontext())

    step_times = []
    for i in range(warmup_steps + measure_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        with autocast_ctx:
            loss = training_step(model, transport, cfg, x, y, feature_dino)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup_steps:
            step_times.append(dt)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    mean_step_ms = 1000.0 * sum(step_times) / max(1, len(step_times))
    return peak_bytes, mean_step_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--batch_size', type=int, required=True)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--measure_steps', type=int, default=3)
    p.add_argument('--precision', default='bf16', choices=['bf16', 'fp16', 'no'])
    args = p.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    peak_bytes, step_ms = run(cfg, args.batch_size, args.warmup_steps,
                              args.measure_steps, args.precision)
    peak_mb = peak_bytes / (1024 * 1024)
    print(f"RESULT bs={args.batch_size} peak_mb={peak_mb:.0f} step_ms={step_ms:.0f}", flush=True)


if __name__ == '__main__':
    main()
