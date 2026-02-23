"""
Train Script of Semantic-First Diffusion (SFD).
"""

import torch
import os
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


def _dist_initialized():
    """Return True when a distributed process group is available."""
    return dist.is_available() and dist.is_initialized()
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_IMPORT_ERROR = None
except Exception as exc:
    # Keep training alive when tensorboard/protobuf versions are incompatible.
    SummaryWriter = None
    _TENSORBOARD_IMPORT_ERROR = exc

import math
import yaml
import json
import numpy as np
import logging
import os
import contextlib
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from diffusers.models import AutoencoderKL
from models import gen_models
from transport import create_transport, Sampler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from dataset.img_latent_dataset import ImgLatentDataset

# Enable W&B via env var for Slurm/non-interactive runs.
# Example: `export ENABLE_WANDB=1` (and optionally set `WANDB_PROJECT`, `WANDB_RUN_NAME`).
enable_wandb = os.getenv("ENABLE_WANDB", "0") == "1"
if enable_wandb:
    # pip install wandb
    import wandb


def load_checkpoint_trusted(path, map_location):
    """
    Load trusted training checkpoints across PyTorch versions.
    Raises a clear error when a Git LFS pointer is provided instead of real weights.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"Checkpoint file at '{path}' is a Git LFS pointer, not model weights. "
                "Download real checkpoints first (e.g. run download_hf_files.py or git lfs pull)."
            )
    except OSError:
        # Let torch.load raise the canonical file-not-found/permission error.
        pass

    return torch.load(path, map_location=map_location, weights_only=False)

def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
    """
    # Setup accelerator:
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tensorboard_dir_log = f"tensorboard_logs/{exp_name}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = None
        if SummaryWriter is not None:
            writer = SummaryWriter(log_dir=tensorboard_dir_log)

            # add configs to tensorboard
            # config_str=json.dumps(train_config, indent=4)
            config_str=json.dumps(OmegaConf.to_container(train_config), indent=4)
            writer.add_text('training configs', config_str, global_step=0)
        else:
            logger.warning(f"TensorBoard disabled due to import error: {_TENSORBOARD_IMPORT_ERROR}")
    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"

    # get rank
    rank = accelerator.local_process_index

    # Create model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    assert train_config['data']['image_size'] % downsample_ratio == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config['data']['image_size'] // downsample_ratio
    _hidden_kwargs = {}
    if 'share_timestep_embedder' in train_config['model']:
        _hidden_kwargs['share_timestep_embedder'] = train_config['model']['share_timestep_embedder']
    model = gen_models[train_config['model']['model_type']](
        input_size=latent_size,
        class_dropout_prob=train_config['model']['class_dropout_prob'] if 'class_dropout_prob' in train_config['model'] else 0.1,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        use_checkpoint=train_config['model']['use_checkpoint'] if 'use_checkpoint' in train_config['model'] else False,
        use_repa=train_config['model']['use_repa'] if 'use_repa' in train_config['model'] else False,
        repa_dino_version=train_config['model']['repa_dino_version'] if 'repa_dino_version' in train_config['model'] else None,
        repa_depth=train_config['model']['repa_feat_depth'] if 'repa_feat_depth' in train_config['model'] else None,
        semantic_chans=train_config['model']['semantic_chans'] if 'semantic_chans' in train_config['model'] else 0,
        semfirst_delta_t=train_config['model']['semfirst_delta_t'] if 'semfirst_delta_t' in train_config['model'] else 0.0,
        semfirst_infer_interval_mode=train_config['model']['semfirst_infer_interval_mode'] if 'semfirst_infer_interval_mode' in train_config['model'] else 'both',
        **_hidden_kwargs,
    )

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # load pretrained model
    if 'weight_init' in train_config['train']:
        checkpoint = load_checkpoint_trusted(
            train_config['train']['weight_init'],
            map_location=lambda storage, loc: storage,
        )
        # Handle checkpoints that only have 'ema' key (no 'model' key)
        if 'model' not in checkpoint and 'ema' in checkpoint:
            checkpoint['model'] = checkpoint['ema']
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = load_weights_with_shape_check(model, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")
    requires_grad(ema, False)
    
    model = model.to(device)
    if _dist_initialized():
        model = DDP(model, device_ids=[rank])
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
        semantic_weight = train_config['model']['semantic_weight'] if 'semantic_weight' in train_config['model'] else 1.0,
        semantic_chans = train_config['model']['semantic_chans'] if 'semantic_chans' in train_config['model'] else 0,
        semfirst_delta_t = train_config['model']['semfirst_delta_t'] if 'semfirst_delta_t' in train_config['model'] else 0.0,
        repa_weight = train_config['model']['repa_weight'] if 'repa_weight' in train_config['model'] else 1.0,
        repa_mode = train_config['model']['repa_mode'] if 'repa_mode' in train_config['model'] else 'cos',
    )  # default: velocity; 
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
        semantic_weight = train_config['model']['semantic_weight'] if 'semantic_weight' in train_config['model'] else 1.0
        semantic_chans = train_config['model']['semantic_chans'] if 'semantic_chans' in train_config['model'] else 0
        semfirst_delta_t = train_config['model']['semfirst_delta_t'] if 'semfirst_delta_t' in train_config['model'] else 0.0
        logger.info(f'Semantic weight: {semantic_weight}, Semantic channels: {semantic_chans}')
        repa_weight = train_config['model']['repa_weight'] if 'repa_weight' in train_config['model'] else 1.0
        logger.info(f'Repa weight: {repa_weight}')
        if semfirst_delta_t > 0 and semantic_chans > 0:
            logger.info(f'Semantic First enabled: delta_t={semfirst_delta_t}')
    opt = torch.optim.AdamW(model.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    
    # Setup data
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_sv_norm=train_config['data']['latent_sv_norm'] if 'latent_sv_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    grad_accum_steps = train_config['train'].get('grad_accum_steps', 1)
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / (accelerator.num_processes * grad_accum_steps)))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes * grad_accum_steps
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size (grad_accum={grad_accum_steps})")
    
    if 'valid_path' in train_config['data']:
        valid_dataset = ImgLatentDataset(
            data_dir=train_config['data']['valid_path'],
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_sv_norm=train_config['data']['latent_sv_norm'] if 'latent_sv_norm' in train_config['data'] else False,
            latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")

    # Prepare models for training:
    update_ema(ema, model.module if hasattr(model, 'module') else model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    train_config['train']['resume'] = train_config['train']['resume'] if 'resume' in train_config['train'] else False

    if train_config['train']['resume']:
        # Build ordered list of candidate checkpoints (prefer last.pt, then newest numeric)
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        numeric_checkpoints = sorted(
            [p for p in checkpoint_files
             if os.path.basename(p).split('.')[0].isdigit()],
            key=lambda x: int(os.path.basename(x).split('.')[0]),
            reverse=True,
        )
        candidate_checkpoints = []
        last_checkpoint = os.path.join(checkpoint_dir, "last.pt")
        if os.path.exists(last_checkpoint):
            candidate_checkpoints.append(last_checkpoint)
        candidate_checkpoints.extend(numeric_checkpoints)

        checkpoint = None
        latest_checkpoint = None
        for candidate in candidate_checkpoints:
            try:
                checkpoint = load_checkpoint_trusted(
                    candidate,
                    map_location=lambda storage, loc: storage,
                )
                latest_checkpoint = candidate
                break
            except Exception as exc:
                if accelerator.is_main_process:
                    logger.warning(
                        f"Checkpoint {candidate} is corrupted ({exc}), trying next..."
                    )
                # Remove the corrupted file so future runs don't hit it again
                try:
                    os.remove(candidate)
                    if accelerator.is_main_process:
                        logger.warning(f"Deleted corrupted checkpoint: {candidate}")
                except OSError:
                    pass

        if checkpoint is not None:
            state_dict = checkpoint['model']
            # Strip 'module.' prefix when not using DDP (single-GPU)
            if not _dist_initialized() and any(k.startswith('module.') for k in state_dict):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            ema_state = checkpoint['ema']
            if any(k.startswith('module.') for k in ema_state):
                ema_state = {k.replace('module.', '', 1): v for k, v in ema_state.items()}
            ema.load_state_dict(ema_state)
            train_steps = checkpoint.get('train_steps')
            if train_steps is None:
                base = os.path.basename(latest_checkpoint).split('.')[0]
                train_steps = int(base) if base.isdigit() else 0
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            train_steps = 0
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")

    train_config['train']['load_ckpt'] = train_config['train']['load_ckpt'] if 'load_ckpt' in train_config['train'] else False
    if train_config['train']['load_ckpt']:
        checkpoint = load_checkpoint_trusted(
            train_config['train']['load_ckpt'],
            map_location=lambda storage, loc: storage,
        )
        state_dict = checkpoint['model']
        if not _dist_initialized() and any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        ema_state = checkpoint['ema']
        if any(k.startswith('module.') for k in ema_state):
            ema_state = {k.replace('module.', '', 1): v for k, v in ema_state.items()}
        ema.load_state_dict(ema_state)
        if accelerator.is_main_process:
            logger.info(f"Loading checkpoint: {train_config['train']['load_ckpt']}")

    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    ckpt_every = train_config['train'].get('ckpt_every', 0)
    ckpt_last_every = train_config['train'].get('ckpt_last_every', ckpt_every)
    ckpt_keep_every = train_config['train'].get('ckpt_keep_every', ckpt_every)
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")

    while True:
        # for x, y in loader:
        for batch_data in loader:
            if len(batch_data) == 2:
                x, y = batch_data
                feature_dino = None
            elif len(batch_data) == 3:
                x, y, feature_dino = batch_data
            else:
                raise NotImplementedError()

            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32)
                y = y
            else:
                x = x.to(device)
                y = y.to(device)
            model_kwargs = dict(y=y)

            with accelerator.accumulate(model):
                # loss_dict = transport.training_losses(model, x, model_kwargs)
                use_repa=train_config['model']['use_repa'] if 'use_repa' in train_config['model'] else False
                use_hidden = train_config['model'].get('use_hidden_tokens', False)
                if use_hidden:
                    hidden_weight = train_config['model'].get('hidden_weight', 1.0)
                    normalize_hidden = train_config['model'].get('normalize_hidden', True)
                    hidden_reg_weight = train_config['model'].get('hidden_reg_weight', 0.01)
                    hidden_cos_weight = train_config['model'].get('hidden_cos_weight', 0.0)
                    hidden_same_t_as_img = train_config['model'].get('hidden_same_t_as_img', False)

                    # backward_fn: called inside training_losses_hidden to backward
                    # Pass 3's loss immediately, freeing its activations before Pass 2
                    # runs. Uses no_sync to defer gradient sync to the main backward.
                    def _hidden_backward_fn(loss):
                        no_sync = getattr(model, 'no_sync', None)
                        ctx = no_sync() if no_sync is not None else contextlib.nullcontext()
                        with ctx:
                            accelerator.backward(loss)

                    loss_dict = transport.training_losses_hidden(
                        model, x, model_kwargs,
                        use_repa=use_repa, feature_dino=feature_dino,
                        hidden_weight=hidden_weight,
                        normalize_hidden=normalize_hidden,
                        hidden_reg_weight=hidden_reg_weight,
                        hidden_cos_weight=hidden_cos_weight,
                        backward_fn=_hidden_backward_fn,
                        hidden_same_t_as_img=hidden_same_t_as_img,
                    )
                else:
                    loss_dict = transport.training_losses(model, x, model_kwargs, use_repa=use_repa, feature_dino=feature_dino)
                # if 'cos_loss' in loss_dict:
                #     mse_loss = loss_dict["loss"].mean()
                #     loss = loss_dict["cos_loss"].mean() + mse_loss
                # else:
                #     loss = loss_dict["loss"].mean()
                if 'cos_loss' in loss_dict and 'repa_loss' in loss_dict:
                    mse_loss = loss_dict["loss"].mean()
                    cos_loss = loss_dict["cos_loss"].mean()
                    repa_loss = loss_dict["repa_loss"].mean()
                    loss = cos_loss + mse_loss + repa_loss
                elif 'cos_loss' in loss_dict:
                    mse_loss = loss_dict["loss"].mean()
                    loss = loss_dict["cos_loss"].mean() + mse_loss
                else:
                    loss = loss_dict["loss"].mean()
                # Add hidden denoising loss if present and not yet backward-ed
                # (When backward_fn is used, hidden_loss is already detached/backward-ed)
                if 'hidden_loss' in loss_dict and loss_dict['hidden_loss'].requires_grad:
                    hidden_loss = loss_dict["hidden_loss"].mean()
                    loss = loss + hidden_loss
                if 'hidden_cos_loss' in loss_dict and loss_dict['hidden_cos_loss'].requires_grad:
                    loss = loss + loss_dict['hidden_cos_loss'].mean()
                if 'hidden_reg_loss' in loss_dict:
                    loss = loss + loss_dict["hidden_reg_loss"]
                accelerator.backward(loss)
                if 'max_grad_norm' in train_config['optimizer']:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
                opt.step()
                opt.zero_grad()

            # Only update EMA and count steps when gradients are actually synced
            if accelerator.sync_gradients:
                update_ema(ema, model.module if hasattr(model, 'module') else model)

                # Log loss values:
                if 'cos_loss' in loss_dict and 'repa_loss' in loss_dict:
                    running_loss += mse_loss.item()
                elif 'cos_loss' in loss_dict:
                    running_loss += mse_loss.item()
                else:
                    running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                if train_steps % train_config['train']['log_every'] == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    if _dist_initialized():
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                        avg_loss = avg_loss.item() / dist.get_world_size()
                    else:
                        avg_loss = avg_loss.item()
                    if accelerator.is_main_process:
                        logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                        if writer is not None:
                            writer.add_scalar('Loss/train', avg_loss, train_steps)
                        # Log loss to wandb
                        if enable_wandb:
                            wandb_losses = {'avg_loss': avg_loss}
                            wandb_losses['mse_loss'] = loss_dict["loss"].mean().item()
                            wandb_losses['total_loss'] = wandb_losses['mse_loss']
                            if 'cos_loss' in loss_dict:
                                wandb_losses['cos_loss'] = loss_dict["cos_loss"].mean().item()
                                wandb_losses['total_loss'] += wandb_losses['cos_loss']
                            if 'repa_loss' in loss_dict:
                                wandb_losses['repa_loss'] = loss_dict["repa_loss"].mean().item()
                                wandb_losses['total_loss'] += wandb_losses['repa_loss']
                            if 'hidden_loss' in loss_dict:
                                wandb_losses['hidden_loss'] = loss_dict["hidden_loss"].mean().item()
                                wandb_losses['total_loss'] += wandb_losses['hidden_loss']
                            if 'hidden_cos_loss' in loss_dict:
                                wandb_losses['hidden_cos_loss'] = loss_dict["hidden_cos_loss"].mean().item()
                                wandb_losses['total_loss'] += wandb_losses['hidden_cos_loss']
                            if 'hidden_reg_loss' in loss_dict:
                                wandb_losses['hidden_reg_loss'] = loss_dict["hidden_reg_loss"].item()
                                wandb_losses['total_loss'] += wandb_losses['hidden_reg_loss']
                            # print(wandb_losses)
                            wandb.log(wandb_losses, step=train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save checkpoint:
                save_last = ckpt_last_every and train_steps % ckpt_last_every == 0 and train_steps > 0
                save_keep = ckpt_keep_every and train_steps % ckpt_keep_every == 0 and train_steps > 0
                if save_last or save_keep:
                    if accelerator.is_main_process:
                        checkpoint = {
                            "model": (model.module if hasattr(model, 'module') else model).state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "config": train_config,
                            "train_steps": train_steps,
                            "wandb_run_id": os.getenv("WANDB_RUN_ID", None),
                        }
                        if save_last:
                            last_path = f"{checkpoint_dir}/last.pt"
                            tmp_path = last_path + ".tmp"
                            torch.save(checkpoint, tmp_path)
                            os.replace(tmp_path, last_path)
                            logger.info(f"Saved checkpoint to {last_path}")
                        if save_keep:
                            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                            tmp_path = checkpoint_path + ".tmp"
                            torch.save(checkpoint, tmp_path)
                            os.replace(tmp_path, checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if _dist_initialized():
                        dist.barrier()

                # Evaluate on validation set
                if save_keep and 'valid_path' in train_config['data']:
                    if accelerator.is_main_process:
                        logger.info(f"Start evaluating at step {train_steps}")
                    val_loss = evaluate(model, valid_loader, device, transport, (0.0, 1.0))
                    if _dist_initialized():
                        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                        val_loss = val_loss.item() / dist.get_world_size()
                    else:
                        val_loss = val_loss.item()
                    if accelerator.is_main_process:
                        logger.info(f"Validation Loss: {val_loss:.4f}")
                        if writer is not None:
                            writer.add_scalar('Loss/validation', val_loss, train_steps)
                    model.train()
                if train_steps >= train_config['train']['max_steps']:
                    break
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator

def load_weights_with_shape_check(model, checkpoint, rank=0):
    
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # special case for x_embedder.proj.weight
                # the pretrained model is trained with 256x256 images
                # we can load the weights by resizing the weights
                # and keep the first 3 channels the same
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
                model_state_dict[name] = weight
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

# def load_config(config_path):
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)
#     return config
def load_config(config_path):
    config = OmegaConf.load(config_path)
    return config

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    args = parser.parse_args()

    train_config = load_config(args.config)
    grad_accum_steps = train_config['train'].get('grad_accum_steps', 1) if 'grad_accum_steps' in train_config.get('train', {}) else 1
    use_hidden = train_config.get('model', {}).get('use_hidden_tokens', False)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=use_hidden,  # hidden-specific params only used in unwrapped passes
        broadcast_buffers=not use_hidden,   # RoPE buffers are deterministic; broadcasting
                                            # modifies them inplace, breaking multi-pass autograd
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_steps,
        kwargs_handlers=[ddp_kwargs] if use_hidden else [],
    )
    if accelerator.is_main_process and enable_wandb:
        # --- Determine wandb run ID for resume (merge chained jobs into one run) ---
        wandb_run_id = os.getenv("WANDB_RUN_ID", None)
        if wandb_run_id is None and train_config['train'].get('resume', False):
            # Try to load run ID from the latest checkpoint (with fallback)
            _exp_dir = os.path.join(train_config['train']['output_dir'], train_config['train']['exp_name'])
            _ckpt_dir = os.path.join(_exp_dir, 'checkpoints')
            _last_ckpt = os.path.join(_ckpt_dir, 'last.pt')
            _candidates = []
            if os.path.exists(_last_ckpt):
                _candidates.append(_last_ckpt)
            _candidates.extend(sorted(glob(f"{_ckpt_dir}/[0-9]*.pt"), reverse=True))
            for _ckpt_to_probe in _candidates:
                try:
                    _meta = load_checkpoint_trusted(_ckpt_to_probe, map_location='cpu')
                    wandb_run_id = _meta.get('wandb_run_id', None)
                    if wandb_run_id:
                        print(f"Resuming wandb run {wandb_run_id} from {_ckpt_to_probe}")
                    del _meta
                    break
                except Exception:
                    print(f"Warning: could not read {_ckpt_to_probe} for wandb resume, trying next...")

        # --- Collect tags from config and/or env ---
        wandb_tags = list(train_config.get('wandb', {}).get('tags', []))
        env_tags = os.getenv("WANDB_TAGS", "")
        if env_tags:
            wandb_tags.extend(t.strip() for t in env_tags.split(",") if t.strip())

        wandb.init(
            project="LightningDiT",
            name=train_config['train']['exp_name'],
            config=OmegaConf.to_container(train_config, resolve=True),
            tags=wandb_tags or None,
            id=wandb_run_id,             # None → new run, string → resume
            resume="allow",              # resume if id exists, else create new
        )
        # Export the (possibly new) run ID so it can be saved in checkpoints
        os.environ["WANDB_RUN_ID"] = wandb.run.id

    do_train(train_config, accelerator)