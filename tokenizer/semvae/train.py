"""
Semantic VAE Training Script - Pre-extracted Features Version
For training AutoEncoder to compress DINOv2 features
"""
import argparse
import datetime
import logging
import os
import sys
import glob
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from omegaconf import OmegaConf
from PIL import Image
from piqa import PSNR, SSIM
from torchvision import transforms
from torchvision.utils import save_image
import safetensors.torch as st
from diffusers import AutoencoderKL
# from tokenizer.vae_semantic.models.vae import SemanticAutoEncoder, SemanticVariationalAutoEncoder
from tokenizer.semvae.models.vae import create_semantic_autoencoder
from tokenizer.semvae.data.preextracted import build_preextracted_dataloader, build_eval_dataloader

enable_swandb = False
if enable_swandb:
    # pip install swanlab
    import swanlab
    swanlab.login(api_key="YOUR_SWANLAB_KEY", save=True)


def evaluate_on_preextracted(semantic_ae, eval_dataloader, variational, device, classification_head=None):
    """
    Evaluate semantic AutoEncoder on pre-extracted features
    """
    semantic_ae.eval()
    if classification_head is not None:
        classification_head.eval()

    mse_values = []
    cosine_sim_values = []
    l1_values = []
    kl_values = []  # Add KL loss value list
    classification_values = []  # Add classification loss value list

    with torch.no_grad():
        for batch_data in eval_dataloader:
            dino_features = batch_data['patch_tokens'].to(device)  # [B, num_patches, feat_dim]
            class_labels = batch_data['class_idx'].to(device) if classification_head is not None else None  # [B]

            # Pass through semantic AutoEncoder
            output = semantic_ae(dino_features)
            if variational:
                features_recon, bottleneck, kl_loss = output
                # Calculate KL loss same as training part, divided by pixel count
                # num_pixels = dino_features.shape[1] * 14 * 14  # dino_features.shape[1]=hxw; 14 is the downsample factor
                # kl_loss = kl_loss.mean() / num_pixels
                kl_loss = kl_loss.mean()
            else:
                features_recon, bottleneck = output
                kl_loss = torch.tensor(0).to(device)

            # Calculate reconstruction error
            mse_loss = F.mse_loss(features_recon, dino_features)
            l1_loss = F.l1_loss(features_recon, dino_features)

            # Calculate cosine similarity
            features_flat = dino_features.view(-1, dino_features.shape[-1])
            recon_flat = features_recon.view(-1, features_recon.shape[-1])
            cosine_sim = F.cosine_similarity(features_flat, recon_flat, dim=1).mean()

            # Calculate classification loss
            classification_loss = torch.tensor(0.0, device=device)
            if classification_head is not None and class_labels is not None:
                # Use bottleneck (VAE compressed representation) for classification, preserving more semantic information
                if bottleneck.dim() == 3:  # [B, num_patches, bottleneck_dim]
                    global_features = bottleneck.mean(dim=1)  # [B, bottleneck_dim]
                else:  # [B, bottleneck_dim]
                    global_features = bottleneck
                # Classify through classification head
                logits = classification_head(global_features)  # [B, num_classes]
                # Calculate cross-entropy loss
                classification_loss = F.cross_entropy(logits, class_labels)

            mse_values.append(mse_loss.item())
            l1_values.append(l1_loss.item())
            cosine_sim_values.append(cosine_sim.item())
            kl_values.append(kl_loss.item())  # Add KL loss value to list
            classification_values.append(classification_loss.item())  # Add classification loss value to list

    avg_mse = sum(mse_values) / len(mse_values) if mse_values else 0
    avg_l1 = sum(l1_values) / len(l1_values) if l1_values else 0
    avg_cosine_sim = sum(cosine_sim_values) / len(cosine_sim_values) if cosine_sim_values else 0
    avg_kl_loss = sum(kl_values) / len(kl_values) if kl_values else 0  # Calculate average KL loss
    avg_classification_loss = sum(classification_values) / len(classification_values) if classification_values else 0  # Calculate average classification loss

    return avg_mse, avg_l1, avg_cosine_sim, avg_kl_loss, avg_classification_loss  # Return average classification loss


# Modify eval_model function to print and log classification loss
def eval_model(eval_dataloader, semantic_ae, variational, iteration, log_dir, classification_head=None):
    """Evaluation function - uses pre-loaded evaluation data"""
    global accelerator

    if eval_dataloader is None:
        return

    semantic_ae.eval()
    unwrapped_ae = accelerator.unwrap_model(semantic_ae)
    unwrapped_classification_head = None
    if classification_head is not None:
        classification_head.eval()
        unwrapped_classification_head = accelerator.unwrap_model(classification_head)

    with torch.no_grad():
        avg_mse, avg_l1, avg_cosine_sim, avg_kl_loss, avg_classification_loss = evaluate_on_preextracted(  # Receive classification loss
            unwrapped_ae,
            eval_dataloader,
            variational,
            accelerator.device,
            unwrapped_classification_head
        )

        # Include all losses in print
        accelerator.print(f"[Evaluation] Iteration {iteration} | "
                        f"MSE: {avg_mse:.6f} | "
                        f"L1: {avg_l1:.6f} | "
                        f"Cosine Similarity: {avg_cosine_sim:.4f} | "
                        f"KL Loss: {avg_kl_loss:.6f} | "
                        f"Classification Loss: {avg_classification_loss:.6f}")

        # Save evaluation log
        eval_log_path = os.path.join(log_dir, "eval_log.txt")
        if accelerator.is_main_process:
            with open(eval_log_path, 'a') as f:
                f.write(f"Iteration {iteration}, MSE: {avg_mse:.6f}, "
                       f"L1: {avg_l1:.6f}, "
                       f"Cosine Similarity: {avg_cosine_sim:.4f}, "
                       f"KL Loss: {avg_kl_loss:.6f}, "
                       f"Classification Loss: {avg_classification_loss:.6f}\n")

        # SwanDB log evaluation metrics
        if enable_swandb:
            swanlab.log({
                "eval_mse": avg_mse,
                "eval_l1": avg_l1,
                "eval_cosine_sim": avg_cosine_sim,
                "eval_kl_loss": avg_kl_loss,
                "eval_classification_loss": avg_classification_loss,
            }, step=iteration)

    accelerator.wait_for_everyone()
    semantic_ae.train()
    if classification_head is not None:
        classification_head.train()


class LossTracker:
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.averages = {}
        
    def update(self, loss_dict):
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if key not in self.averages:
                self.averages[key] = value
            else:
                self.averages[key] = self.momentum * self.averages[key] + (1 - self.momentum) * value
                
    def get_average(self, key):
        return self.averages.get(key, 0.0)
        
    def get_all_averages(self):
        return self.averages.copy()
        
    def print_averages(self, accelerator, prefix=""):
        message = prefix
        for key, value in sorted(self.averages.items()):
            message += f" | {key}: {value:.6f}"
        accelerator.print(message)


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic VAE Training - Pre-extracted Features')
    parser.add_argument('--config', type=str, default='tokenizer/configs/semantic_vae/preextracted_demo.yaml', 
                       help='Path to config file')
    return parser.parse_args()


def setup_logger(config, accelerator):
    """Setup logging system"""
    log_dir = config['training']['log']['log_dir']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"semantic_vae_training_log_{timestamp}.txt")

    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logger = logging.getLogger("semantic_vae_training")
        logger.setLevel(logging.INFO)
        logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        original_print = accelerator.print

        def new_print(*args, **kwargs):
            message = " ".join(map(str, args))
            logger.info(message)

        accelerator.print = new_print
        accelerator.print(f"=== Semantic VAE training started at {timestamp} ===")
        accelerator.print(f"Log file: {log_file}")
    
    accelerator.wait_for_everyone()
    return accelerator


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, accelerator=None):
    """Load checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    accelerator.print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        msg = unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loading model state: {msg}')
    else:
        msg = model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loading model state: {msg}')

    start_iteration = checkpoint.get('iteration', 0)

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        accelerator.print("Optimizer state loaded")

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        accelerator.print("Scheduler state loaded")

    # start_iteration = 0

    # Print checkpoint info
    if 'loss' in checkpoint:
        accelerator.print(f"Checkpoint loss: {checkpoint['loss']:.6f}")
    if 'loss_averages' in checkpoint:
        accelerator.print("Loss statistics:", checkpoint['loss_averages'])

    accelerator.print(f"Resuming training from iteration {start_iteration}")
    return start_iteration+1


def setup_scheduler(optimizer, base_lr, warmup_steps, constant_steps, total_iterations, min_lr_factor=0.1):
    """Setup learning rate scheduler"""
    global accelerator

    if warmup_steps <= 0:
        accelerator.print(f"Set warmup steps to 1")
        warmup_steps = 1

    cosine_steps = total_iterations - warmup_steps - constant_steps
    if cosine_steps <= 0:
        cosine_steps = 1

    # Three-phase scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=constant_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=base_lr * min_lr_factor
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler, cosine_scheduler],
        milestones=[warmup_steps, warmup_steps + constant_steps]
    )

    accelerator.print(f"Learning rate scheduler setup: warmup={warmup_steps}, constant={constant_steps}, cosine={cosine_steps}")
    return scheduler


def main():
    # Parse arguments
    args = parse_args()
    config = OmegaConf.load(args.config)

    # Training parameters
    mixed_precision = config['training']['mixed_precision']
    total_iterations = config['training']['total_iterations']
    max_epochs = config['training']['max_epochs']

    # Model parameters - get from data config
    input_dim_dict = {
        'dinov2_vits14_reg': 384,
        'dinov2_vitb14_reg': 768,
        'dinov2_vitl14_reg': 1024,
        'dinov2_vitg14_reg': 1536,
        'mae_base': 768,
        'mae_large': 1024,
        'clip_vit_base_patch16': 768,
        'clip_vit_large_patch14': 1024,
        'siglip_base_patch16_224': 768,
    }
    model_name = config['data']['model_name']
    input_dim = input_dim_dict[model_name]
    bottleneck_dim = config['model']['bottleneck_dim']
    variational = config['model']['variational']
    
    # Checkpointing
    checkpoint_every = config['training']['checkpointing']['save_every']
    checkpoint_dir = config['training']['checkpointing']['checkpoint_dir']
    
    # Evaluation parameters
    eval_every = config['training']['evaluation']['eval_every']

    # Logging parameters
    log_every = config['training']['log']['log_every']

    # Setup Accelerator
    global accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision, step_scheduler_with_optimizer=False)
    accelerator = setup_logger(config, accelerator)

    accelerator.print(f"Using device: {accelerator.device}")
    accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Config file: {args.config}")
    accelerator.print(f"Config content:\n{OmegaConf.to_yaml(config)}")

    # Create output directories
    if accelerator.is_main_process:
        os.makedirs(config['training']['log']['log_dir'], exist_ok=True)
        os.makedirs(config['training']['checkpointing']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['training']['evaluation']['output_dir'], exist_ok=True)

    # Setup loss tracker
    loss_tracker = LossTracker(momentum=0.99)

    # Build model
    accelerator.print("Initializing semantic AutoEncoder...")
    accelerator.print(f"Using variational autoencoder: {variational}")
    model_arch_config = config['model']['params']
    semantic_ae = create_semantic_autoencoder(
        arch=config['model']['arch'],
        is_vae=variational,
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim,
        **model_arch_config
    )
    accelerator.print('Model Architecture Config:', model_arch_config)
    accelerator.print('Model Architecture:', semantic_ae)

    # Add classification head (for classification loss)
    classification_weight = config['training']['loss'].get('classification_weight', 0.0)
    classification_head = None
    if classification_weight > 0:
        # ImageNet has 1000 classes, use bottleneck_dim as input dimension
        num_classes = 1000
        classification_head = nn.Linear(bottleneck_dim, num_classes)
        accelerator.print(f"Classification loss enabled, weight: {classification_weight}")
        accelerator.print(f"Classification head: {classification_head}")
    else:
        accelerator.print("Classification loss weight is 0, classifier not enabled")

    semantic_ae.train()

    # Move classification head to device
    if classification_head is not None:
        classification_head.train()

    # Count parameters
    total_params = sum(p.numel() for p in semantic_ae.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in semantic_ae.parameters() if p.requires_grad) / 1e6
    if classification_head is not None:
        cls_total_params = sum(p.numel() for p in classification_head.parameters()) / 1e6
        cls_trainable_params = sum(p.numel() for p in classification_head.parameters() if p.requires_grad) / 1e6
        total_params += cls_total_params
        trainable_params += cls_trainable_params
        accelerator.print(f"Classifier parameters: {cls_total_params:.2f}M")

    accelerator.print(f"Semantic AE total parameters: {total_params:.2f}M")
    accelerator.print(f"Trainable parameters: {trainable_params:.2f}M")
    accelerator.print(f"Input dimension: {input_dim}")
    accelerator.print(f"Compression dimension: {bottleneck_dim}")
    accelerator.print(f"Compression ratio: {input_dim / bottleneck_dim:.2f}x")

    # Prepare all parameters to optimize
    all_parameters = list(semantic_ae.parameters())
    if classification_head is not None:
        all_parameters.extend(list(classification_head.parameters()))

    # Initialize optimizer
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(all_parameters, lr=optimizer_config['lr'])
    elif optimizer_config['type'] == 'Adam':
        optimizer = torch.optim.Adam(all_parameters, lr=optimizer_config['lr'])
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")

    # Initialize scheduler
    scheduler_config = config['training']['scheduler']
    base_lr = optimizer_config['lr']
    warmup_steps = scheduler_config['warmup_steps']
    constant_steps = scheduler_config['constant_steps']
    scheduler = setup_scheduler(optimizer, base_lr, warmup_steps, constant_steps, total_iterations)

    # Initialize SwanLab
    if enable_swandb:
        swanlab.init(
            # Set project name
            project=f"SemanticVAE-{model_name}",
            experiment_name=config['training']['log']['swandb_exp_name'],

            # Set hyperparameters
            config={
                "arch": config['model']['arch'],
                "is_vae": variational,
                "input_dim": input_dim,
                "bottleneck_dim": bottleneck_dim,
                "learning_rate": optimizer_config['lr'],
                "warmup_steps": scheduler_config['warmup_steps'],
                "constant_steps": scheduler_config['constant_steps'],
                "total_iterations": total_iterations,
                "loss_mse_weight": config['training']['loss']['mse_weight'],
                "loss_cos_weight": config['training']['loss']['cos_weight'],
                "loss_l1_weight": config['training']['loss']['l1_weight'],
                "loss_kl_weight": config['training']['loss']['kl_weight'],
                "loss_classification_weight": config['training']['loss'].get('classification_weight', 0.0),
            }
        )

    # Load pre-extracted feature data
    accelerator.print("Setting up training data loading...")
    dataloader = build_preextracted_dataloader(config)
    accelerator.print("Pre-extracted feature training data loading setup complete!")

    # Load evaluation data (one-time loading)
    accelerator.print("Setting up evaluation data loading...")
    eval_dataloader = None
    try:
        eval_dataloader = build_eval_dataloader(config)
        accelerator.print(f"Evaluation dataset sample count: {len(eval_dataloader.dataset)}")
        accelerator.print("Pre-extracted feature evaluation data loading setup complete!")
    except Exception as e:
        accelerator.print(f"Cannot load evaluation data: {e}")
        accelerator.print("Will skip evaluation steps")

    # Prepare components with accelerator
    if classification_head is not None:
        semantic_ae, classification_head, optimizer, dataloader, scheduler = accelerator.prepare(
            semantic_ae, classification_head, optimizer, dataloader, scheduler
        )
    else:
        semantic_ae, optimizer, dataloader, scheduler = accelerator.prepare(
            semantic_ae, optimizer, dataloader, scheduler
        )

    # Load checkpoint (if specified in config)
    start_iteration = 0
    if 'checkpoint_path' in config['training'] and config['training']['checkpoint_path']:
        checkpoint_path = config['training']['checkpoint_path']
        if os.path.exists(checkpoint_path):
            start_iteration = load_checkpoint(
                checkpoint_path,
                semantic_ae,
                optimizer,
                scheduler,
                accelerator
            )
        else:
            accelerator.print(f"Warning: Specified checkpoint file does not exist: {checkpoint_path}")
            accelerator.print("Will start training from scratch")

    # Pre-training evaluation
    if eval_dataloader is not None:
        accelerator.print(f"Evaluating on validation set before training...")
        eval_model(eval_dataloader, semantic_ae, variational, start_iteration, config['training']['log']['log_dir'], classification_head)
        accelerator.wait_for_everyone()

    # Start training
    accelerator.print(f"Starting training, target iterations: {total_iterations}")
    if start_iteration > 0:
        accelerator.print(f"Resuming from checkpoint, starting iteration: {start_iteration}")
    start_time = datetime.datetime.now()
    last_iter_time = start_time
    iteration = start_iteration
    
    for epoch in range(max_epochs):
        for batch_data in dataloader:

            # Get pre-extracted DINOv2 features and labels
            dino_features = batch_data['patch_tokens'].to(accelerator.device)  # [B, num_patches, feat_dim]
            class_labels = batch_data['class_idx'].to(accelerator.device) if classification_head is not None else None  # [B]

            # Forward pass
            # features_recon, bottleneck = semantic_ae(dino_features)
            output = semantic_ae(dino_features)
            if variational:
                features_recon, bottleneck, kl_loss = output
                # num_pixels = dino_features.shape[1] * 14 * 14 # dino_features.shape[1]=hxw; 14 is the downsample factor
                # kl_loss = kl_loss.mean() / num_pixels
                kl_loss = kl_loss.mean()
            else:
                features_recon, bottleneck = output
                kl_loss = 0

            # Calculate reconstruction loss
            mse_loss = F.mse_loss(features_recon, dino_features)
            l1_loss = F.l1_loss(features_recon, dino_features)
            cos_loss = 1 - F.cosine_similarity(features_recon, dino_features).mean()

            # Calculate classification loss
            classification_loss = torch.tensor(0.0, device=accelerator.device)
            if classification_head is not None and class_labels is not None:
                # Use bottleneck (VAE compressed representation) for classification, preserving more semantic information
                if bottleneck.dim() == 3:  # [B, num_patches, bottleneck_dim]
                    global_features = bottleneck.mean(dim=1)  # [B, bottleneck_dim]
                else:  # [B, bottleneck_dim]
                    global_features = bottleneck
                # Classify through classification head
                logits = classification_head(global_features)  # [B, num_classes]
                # Calculate cross-entropy loss
                classification_loss = F.cross_entropy(logits, class_labels)

            total_loss = mse_loss + l1_loss + kl_loss

            # Get loss weights
            mse_weight = config['training']['loss']['mse_weight']
            l1_weight = config['training']['loss']['l1_weight']
            cos_weight = config['training']['loss']['cos_weight']
            kl_weight = config['training']['loss']['kl_weight']
            classification_weight = config['training']['loss'].get('classification_weight', 0.0)

            # Total loss
            total_loss = (mse_weight * mse_loss +
                         l1_weight * l1_loss +
                         cos_weight * cos_loss +
                         kl_weight * kl_loss +
                         classification_weight * classification_loss) 
            
            current_losses = {
                "mse": mse_loss,
                "l1": l1_loss,
                "cos_loss": cos_loss,
                "kl_loss": kl_loss,
                "classification_loss": classification_loss,
                "total": total_loss,
                "bottleneck_std": bottleneck.std(),  # Monitor bottleneck standard deviation
                "bottleneck_mean": bottleneck.mean(),  # Monitor bottleneck mean
                "bottleneck_min": bottleneck.min(),  # Monitor bottleneck minimum
                "bottleneck_max": bottleneck.max(),  # Monitor bottleneck maximum
            }

            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()
            scheduler.step()

            # Update loss tracker
            loss_tracker.update(current_losses)

            # Print log
            if iteration % log_every == 0:
                current_time = datetime.datetime.now()
                iter_duration_log_every = (current_time - last_iter_time).total_seconds()
                elapsed = (current_time - start_time).total_seconds()
                iter_per_sec = log_every / iter_duration_log_every if iter_duration_log_every > 0 else 0
                iter_duration = iter_duration_log_every / log_every
                eta_seconds = (total_iterations - iteration) / iter_per_sec if iter_per_sec > 0 else 0
                
                current_lr = optimizer.param_groups[0]['lr']
                
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_secs = int(eta_seconds % 60)
                eta_str = f"{eta_hours}h {eta_minutes}m {eta_secs}s"
                
                status_msg = f"Iteration {iteration} | LR: {current_lr:.6f} | Time: {iter_duration:.2f}s/it ({iter_per_sec:.2f} it/s) | ETA: {eta_str}"
                loss_tracker.print_averages(accelerator, status_msg)
                
                last_iter_time = current_time

                # SwanLab log losses
                if enable_swandb:
                    swanlab.log(current_losses, step=iteration)

            # Evaluation
            if iteration % eval_every == 0 and iteration > 0:
                eval_model(eval_dataloader, semantic_ae, variational, iteration, config['training']['log']['log_dir'], classification_head)

                # Clean up memory
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                accelerator.wait_for_everyone()

            # Save checkpoint
            if iteration % checkpoint_every == 0 and accelerator.is_main_process and iteration > 0:
                try:
                    unwrapped_ae = accelerator.unwrap_model(semantic_ae)

                    save_path = os.path.join(checkpoint_dir, f'semantic_ae_checkpoint_iter_{iteration}.pt')
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': unwrapped_ae.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss_tracker.get_average('total'),
                        'loss_averages': loss_tracker.get_all_averages(),
                        'config': OmegaConf.to_yaml(config),
                        'input_dim': input_dim,
                        'bottleneck_dim': bottleneck_dim,
                    }, save_path)
                    accelerator.print(f"Model saved to {save_path}")
                except Exception as e:
                    accelerator.print(f"Error saving model: {e}")

            # Check if training is complete
            iteration += 1
            if iteration >= total_iterations:
                accelerator.print(f"Reached {total_iterations} iterations, stopping training")

                # Final evaluation
                if eval_dataloader is not None:
                    accelerator.print("Performing final evaluation...")
                    eval_model(eval_dataloader, semantic_ae, variational, iteration, config['training']['log']['log_dir'], classification_head)

                # Save final model
                if accelerator.is_main_process:
                    unwrapped_ae = accelerator.unwrap_model(semantic_ae)

                    final_save_path = os.path.join(checkpoint_dir, 'semantic_ae_final.pt')
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': unwrapped_ae.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss_tracker.get_average('total'),
                        'loss_averages': loss_tracker.get_all_averages(),
                        'config': OmegaConf.to_yaml(config),
                        'input_dim': input_dim,
                        'bottleneck_dim': bottleneck_dim,
                    }, final_save_path)
                    accelerator.print(f"Final model saved to {final_save_path}")

                    # Print training summary
                    total_time = datetime.datetime.now() - start_time
                    accelerator.print(f"Training complete! Total time: {total_time}")
                    accelerator.print(f"Final loss:")
                    loss_tracker.print_averages(accelerator, "  ")
                
                accelerator.wait_for_everyone()
                return


if __name__ == "__main__":
    main()