"""Diagnose whether the 1p0_finetune_no_hidden_warm8k drift is caused by
a data/stats mismatch between our dataset and the 4M pretrained weights.

Runs two probes on our dataset, loading weights through the same code path
as train.py's weight_init:

  Probe A — step-0 loss distribution.
    Load 4M ckpt into fresh model, run N batches forward, report per-batch
    velocity MSE + cos_loss + repa_loss. A well-matched pretrained model
    should show low, stable loss; a mismatched one has elevated loss.

  Probe B — gradient bias.
    For each probe batch accumulate the parameter gradient. Then compute
    per-parameter   ||mean(grad)|| / mean(||grad||).
    If near 0: gradients average out (converged, well-matched).
    If near 1: gradients all point the same way (persistent bias → drift).

  Probe C — drift alignment.
    Compare mean(-grad) against the actual 60K-vs-4M weight-delta direction.
    Cosine close to 1 means the drift direction is exactly what our data's
    persistent-gradient bias predicts (confirms hypothesis).

Usage:  python diagnose_ft_drift.py [--num_batches N] [--batch_size B]
"""

import argparse
import torch
import torch.nn.functional as F
from copy import deepcopy
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from models import gen_models
from dataset.img_latent_dataset import ImgLatentDataset
from transport import create_transport
from train import load_weights_with_shape_check, load_checkpoint_trusted


def build_model_from_config(cfg):
    m = cfg['model']
    return gen_models[m['model_type']](
        input_size=cfg['data']['image_size'] // cfg['vae'].get('downsample_ratio', 16),
        class_dropout_prob=m.get('class_dropout_prob', 0.1),
        num_classes=cfg['data']['num_classes'],
        use_qknorm=m['use_qknorm'],
        use_swiglu=m.get('use_swiglu', False),
        use_rope=m.get('use_rope', False),
        use_rmsnorm=m.get('use_rmsnorm', False),
        wo_shift=m.get('wo_shift', False),
        in_channels=m.get('in_chans', 4),
        use_checkpoint=m.get('use_checkpoint', False),
        use_repa=m.get('use_repa', False),
        repa_dino_version=m.get('repa_dino_version'),
        repa_depth=m.get('repa_feat_depth'),
        semantic_chans=m.get('semantic_chans', 0),
        semfirst_delta_t=m.get('semfirst_delta_t', 0.0),
        semfirst_infer_interval_mode=m.get('semfirst_infer_interval_mode', 'both'),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/sfd/hidden_1p0_h200_from_ft/finetune_no_hidden_warm8k.yaml')
    ap.add_argument('--src_ckpt', default='outputs/train/sfd_1p0/checkpoints/4000000_full.pt')
    ap.add_argument('--ft_ckpt', default='outputs/train/1p0_finetune_no_hidden_warm8k/checkpoints/0060000.pt')
    ap.add_argument('--num_batches', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--stats_dir_override', default=None,
                    help='Directory to load latents_stats.pt and latents_sv_stats.pt from '
                         '(e.g. the original pretrain dataset path). If set, overrides '
                         'the stats auto-loaded from the dataset dir.')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    m = cfg['model']
    d = cfg['data']

    # ---- Build model + load 4M ckpt through the same path as train.py ----
    print(f'Building {m["model_type"]} ...')
    model = build_model_from_config(cfg).to(device)

    print(f'Loading {args.src_ckpt} ...')
    ckpt = load_checkpoint_trusted(args.src_ckpt, map_location=lambda s, l: s)
    if 'model' not in ckpt and 'ema' in ckpt:
        ckpt['model'] = ckpt['ema']
    ckpt['model'] = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    model = load_weights_with_shape_check(model, ckpt, rank=0)

    # Use the EMA (that's what sampling uses)
    if 'ema' in ckpt and ckpt['ema'] is not ckpt['model']:
        ema_ckpt = {'model': {k.replace('module.', ''): v for k, v in ckpt['ema'].items()}}
        ema = deepcopy(model)
        ema = load_weights_with_shape_check(ema, ema_ckpt, rank=0).to(device)
    else:
        ema = deepcopy(model).to(device)

    # ---- Transport ----
    t_cfg = cfg['transport']
    transport = create_transport(
        t_cfg['path_type'], t_cfg['prediction'], t_cfg['loss_weight'],
        t_cfg['train_eps'], t_cfg['sample_eps'],
        use_cosine_loss=t_cfg.get('use_cosine_loss', False),
        use_lognorm=t_cfg.get('use_lognorm', False),
        semantic_weight=m.get('semantic_weight', 1.0),
        semantic_chans=m.get('semantic_chans', 0),
        semfirst_delta_t=m.get('semfirst_delta_t', 0.0),
        repa_weight=m.get('repa_weight', 1.0),
        repa_mode=m.get('repa_mode', 'cos'),
    )

    # ---- Data ----
    print(f'Dataset: {d["data_path"]}')
    ds = ImgLatentDataset(
        data_dir=d['data_path'],
        latent_norm=d.get('latent_norm', False),
        latent_sv_norm=d.get('latent_sv_norm', False),
        latent_multiplier=d.get('latent_multiplier', 0.18215),
    )
    if args.stats_dir_override is not None:
        import os as _os
        sd = args.stats_dir_override
        print(f'Overriding latent stats from: {sd}')
        if d.get('latent_norm', False):
            s = torch.load(_os.path.join(sd, 'latents_stats.pt'), weights_only=False)
            ds._latent_mean, ds._latent_std = s['mean'], s['std']
            print(f'  latents_stats.pt loaded: mean range [{s["mean"].min():.4f},{s["mean"].max():.4f}] std range [{s["std"].min():.4f},{s["std"].max():.4f}]')
        if d.get('latent_sv_norm', False):
            s = torch.load(_os.path.join(sd, 'latents_sv_stats.pt'), weights_only=False)
            ds._latent_sv_mean, ds._latent_sv_std = s['mean'], s['std']
            print(f'  latents_sv_stats.pt loaded: mean range [{s["mean"].min():.4f},{s["mean"].max():.4f}] std range [{s["std"].min():.4f},{s["std"].max():.4f}]')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # ---- Probes A + B ----
    print(f'\nRunning {args.num_batches} probe batches (bs={args.batch_size}) ...')
    model.train()  # so class-drop behaves like training
    for p in model.parameters():
        p.requires_grad_(p.requires_grad)  # keep existing
        if p.grad is not None:
            p.grad = None

    # Accumulators
    grad_sum = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    grad_abs_sum = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    grad_norm_sum = {n: 0.0 for n in grad_sum}
    losses = []
    cos_losses = []
    repa_losses = []

    use_repa = m.get('use_repa', False)

    for bi, batch_data in enumerate(loader):
        if bi >= args.num_batches:
            break
        if len(batch_data) == 2:
            x, y = batch_data
            feat = None
        else:
            x, y, feat = batch_data[0], batch_data[1], batch_data[2]
        x = x.to(device)
        y = y.to(device)
        if feat is not None:
            feat = feat.to(device)

        model.zero_grad(set_to_none=True)
        terms = transport.training_losses(
            model, x,
            model_kwargs={'y': y},
            use_repa=use_repa,
            feature_dino=feat,
        )
        total = terms['loss'].mean()
        if 'cos_loss' in terms:
            total = total + terms['cos_loss'].mean()
            cos_losses.append(terms['cos_loss'].mean().item())
        if 'repa_loss' in terms:
            total = total + terms['repa_loss'].mean()
            repa_losses.append(terms['repa_loss'].mean().item())
        losses.append(terms['loss'].mean().item())
        total.backward()

        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                grad_sum[n].add_(p.grad)
                grad_abs_sum[n].add_(p.grad.abs())
                grad_norm_sum[n] += p.grad.norm().item()

        if bi % 8 == 0:
            msg = f'  batch {bi:3d}: loss={losses[-1]:.4f}'
            if cos_losses:
                msg += f' cos={cos_losses[-1]:.4f}'
            if repa_losses:
                msg += f' repa={repa_losses[-1]:.4f}'
            print(msg)

    N = len(losses)
    print(f'\n=== Probe A: step-0 loss on our dataset ({N} batches) ===')
    t = torch.tensor(losses)
    print(f'  velocity MSE:  mean={t.mean():.5f}  std={t.std():.5f}  min={t.min():.5f}  max={t.max():.5f}')
    if cos_losses:
        tc = torch.tensor(cos_losses)
        print(f'  cos_loss:      mean={tc.mean():.5f}  std={tc.std():.5f}')
    if repa_losses:
        tr = torch.tensor(repa_losses)
        print(f'  repa_loss:     mean={tr.mean():.5f}  std={tr.std():.5f}')

    print(f'\n=== Probe B: per-param gradient bias (||mean grad|| / mean ||grad||) ===')
    # Bias ratio: ||sum_i g_i|| / sum_i ||g_i||.  Near 0 = averages out.  1 = perfectly aligned.
    bias = []
    for n in grad_sum:
        num = grad_sum[n].norm().item()
        den = grad_norm_sum[n]
        ratio = num / max(den, 1e-12)
        bias.append((n, ratio, num, den))
    bias.sort(key=lambda x: -x[1])
    print(f'  Top 15 most-biased params:')
    for n, r, num, den in bias[:15]:
        print(f'    {n:55s}  bias_ratio={r:.4f}   ||mean_g||={num/N:.4e}')
    print(f'  Bottom 5 (least biased):')
    for n, r, num, den in bias[-5:]:
        print(f'    {n:55s}  bias_ratio={r:.4f}')
    all_r = torch.tensor([b[1] for b in bias])
    print(f'  Overall bias_ratio: mean={all_r.mean():.4f}  median={all_r.median():.4f}  max={all_r.max():.4f}')

    # ---- Probe C: drift alignment ----
    print(f'\n=== Probe C: drift direction alignment ===')
    print(f'Loading 60K finetune ckpt for drift comparison: {args.ft_ckpt}')
    ft_ckpt = load_checkpoint_trusted(args.ft_ckpt, map_location='cpu')
    ft_state = {k.replace('module.', ''): v for k, v in ft_ckpt['model'].items()}
    src_state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}  # ckpt was modified above
    # Actually ckpt['model'] still holds the stripped source weights
    print('  param_name                                            cos(-mean_g, drift)   |drift|/|mean_g|')
    alignments = []
    for n in grad_sum:
        if n not in ft_state or n not in src_state:
            continue
        drift = (ft_state[n].float() - src_state[n].float()).to(device).flatten()
        mean_g = (grad_sum[n] / N).flatten()
        if drift.norm() < 1e-10 or mean_g.norm() < 1e-10:
            continue
        cos_neg = F.cosine_similarity(-mean_g.unsqueeze(0), drift.unsqueeze(0)).item()
        mag_ratio = drift.norm().item() / mean_g.norm().item()
        alignments.append((n, cos_neg, mag_ratio))
    alignments.sort(key=lambda x: -abs(x[1]))
    print(f'  Top 15 params by |cos(-mean_g, drift)|:')
    for n, c, r in alignments[:15]:
        print(f'    {n:55s}  cos={c:+.4f}   |drift|/|mean_g|={r:.3e}')
    cos_vals = torch.tensor([a[1] for a in alignments])
    print(f'  Overall cos(-mean_g, drift): mean={cos_vals.mean():+.4f}  median={cos_vals.median():+.4f}')
    print(f'  Fraction with cos > 0.5: {(cos_vals > 0.5).float().mean().item():.2%}')
    print(f'  Fraction with cos > 0.8: {(cos_vals > 0.8).float().mean().item():.2%}')

    # ---- Probe D: per-channel-group loss (texture vs semantic) ----
    print(f'\n=== Probe D: per-channel-group loss (texture vs semantic) ===')
    model.eval()
    from transport.transport import Sampler
    sem_chans = m.get('semantic_chans', 0)
    tex_chans = m.get('in_chans', 4) - sem_chans
    print(f'  texture chans = {tex_chans}, semantic chans = {sem_chans}')

    loader2 = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    tex_mses = []
    sem_mses = []
    semfirst_delta_t = m.get('semfirst_delta_t', 0.0)
    with torch.no_grad():
        for bi, batch_data in enumerate(loader2):
            if bi >= args.num_batches:
                break
            if len(batch_data) == 2:
                x, y = batch_data
                feat = None
            else:
                x, y, feat = batch_data[0], batch_data[1], batch_data[2]
            x1 = x.to(device)
            y = y.to(device)
            B = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device)

            if semfirst_delta_t > 0:
                t_sem = (t * (1 + semfirst_delta_t)).clamp(max=1.0)
                t_tex = (t_sem - semfirst_delta_t).clamp(min=0.0)
                x0_tex = x0[:, :tex_chans]; x0_sem = x0[:, tex_chans:]
                x1_tex = x1[:, :tex_chans]; x1_sem = x1[:, tex_chans:]
                xt_tex = t_tex.view(-1,1,1,1)*x1_tex + (1-t_tex.view(-1,1,1,1))*x0_tex
                xt_sem = t_sem.view(-1,1,1,1)*x1_sem + (1-t_sem.view(-1,1,1,1))*x0_sem
                ut_tex = x1_tex - x0_tex
                ut_sem = x1_sem - x0_sem
                xt = torch.cat([xt_tex, xt_sem], dim=1)
                ut = torch.cat([ut_tex, ut_sem], dim=1)
                t_arg = (t_tex, t_sem)
            else:
                xt = t.view(-1,1,1,1)*x1 + (1-t.view(-1,1,1,1))*x0
                ut = x1 - x0
                t_arg = t

            out = model(xt, t_arg, y=y)
            pred = out[0] if isinstance(out, tuple) else out

            err = (pred - ut) ** 2
            tex_mses.append(err[:, :tex_chans].mean().item())
            sem_mses.append(err[:, tex_chans:].mean().item())

    import statistics
    print(f'  texture  MSE: mean={statistics.mean(tex_mses):.4f}  min={min(tex_mses):.4f}  max={max(tex_mses):.4f}')
    print(f'  semantic MSE: mean={statistics.mean(sem_mses):.4f}  min={min(sem_mses):.4f}  max={max(sem_mses):.4f}')
    print(f'  ratio sem/tex = {statistics.mean(sem_mses)/max(statistics.mean(tex_mses),1e-9):.2f}')

    print('\nInterpretation:')
    print('  A: if velocity MSE is << typical early-training values (~0.5), pretrain is well-matched.')
    print('  B: if bias_ratio median << 0.1, gradients average out (healthy); >>0.1 → persistent bias.')
    print('  C: if cos(-mean_g, drift) is close to 1, drift IS the bias-induced update direction.')
    print('  D: if tex_mse << sem_mse, the semantic channels (SemVAE) are the mismatch source.')


if __name__ == '__main__':
    main()
