import torch
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
from dataset.img_latent_dataset import ImgLatentDataset
from tokenizer.vavae import VA_VAE

def main(args):
    """
    Run a tokenizer on full dataset and save the features.
    """
    assert torch.cuda.is_available(), "Extract features currently requires at least one GPU."

    # Setup DDP:
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup feature folders:
    # output_dir = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.config))[0], f'{args.data_split}_{args.image_size}')
    output_path = args.output_path
    basename =  os.path.splitext(os.path.basename(args.config))[0]
    if args.repa_dino_model_name != '':
        basename = basename+f"_repa{args.repa_dino_model_name}"
    output_dir = os.path.join(output_path, basename, f'{args.data_split}_{args.image_size}')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Create model:
    tokenizer = VA_VAE(
        args.config,
        img_size=args.image_size
    )
    if args.semantic_feat_type == 'dino_pca_ch8':
        @torch.no_grad()
        def load_dino_encoders(device, resolution=256):
            assert (resolution == 256) or (resolution == 512)
            encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14_reg')
            del encoder.head
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
            
            return encoder

        device_dino_pca = 'cuda'
        dino_encoder = load_dino_encoders(device_dino_pca)
        
        pca_rank = 8
        pca_stats = torch.load('redi_pcs/dino_pca_model.pth')
        pca_model = pca_stats["pca_model"]
        pca_components = torch.tensor(pca_model.components_[:pca_rank, :]).to(device_dino_pca)

        pca_mean = torch.tensor(pca_model.mean_).to(device_dino_pca)
        mean_dino = torch.tensor(pca_stats["mean"]).to(device_dino_pca)
        std_dino = torch.tensor(pca_stats["std"]).to(device_dino_pca)

        # prepare for dino pca
        latents_dino = []
        latents_dino_flip = []

        from torchvision.transforms import Normalize
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        @torch.no_grad()
        def preprocess_raw_image(x):
            resolution = x.shape[-1]
            x = x / 255.
            x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
            x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

            return x
    elif args.semantic_feat_type == 'semvae':
        device_semvae= 'cuda'

        # parse config
        from omegaconf import OmegaConf
        semvae_config = OmegaConf.load(args.config)['semvae']

        # Backward compatibility: support both old 'dino_model_name' and new 'model_name'
        if 'dino_model_name' in semvae_config:
            model_name = semvae_config['dino_model_name']
        elif 'model_name' in semvae_config:
            model_name = semvae_config['model_name']
        else:
            raise ValueError("Config must contain either 'dino_model_name' or 'model_name'")

        # Choose appropriate extractor based on model type
        if 'dinov2' in model_name:
            # DINOv2 models use DINOFeatureExtractor
            from tokenizer.semvae.extract_dinov2_feature import DINOFeatureExtractor
            if rank == 0:
                print(f'Loading DINOFeatureExtractor for model: {model_name}')
            sv_extractor = DINOFeatureExtractor(
                model_name,
                'cuda',
                semvae_config['semvae_config'],
                semvae_config['semvae_ckpt_path'])
            # DINOv2 uses ImageNet standard parameters
            from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            normalize_mean = IMAGENET_DEFAULT_MEAN
            normalize_std = IMAGENET_DEFAULT_STD
        else:
            # MAE, CLIP, SigLIP use UnifiedFeatureExtractor
            from tokenizer.semvae.extract_other_feature import UnifiedFeatureExtractor
            if rank == 0:
                print(f'Loading UnifiedFeatureExtractor for model: {model_name}')
            sv_extractor = UnifiedFeatureExtractor(
                model_name,
                'cuda',
                semvae_config['semvae_config'],
                semvae_config['semvae_ckpt_path'])
            # Use model-specific normalization parameters
            normalize_mean = sv_extractor.normalize_params['mean']
            normalize_std = sv_extractor.normalize_params['std']

        if rank == 0:
            print(f'Using normalize params: mean={normalize_mean}, std={normalize_std}')

        from torchvision import transforms
        import torch.nn as nn
        import einops
        sv_transform = transforms.Compose([transforms.Normalize(mean=normalize_mean, std=normalize_std)])

        latents_sv = []
        latents_sv_flip = []
    elif args.semantic_feat_type == '':
        pass
    else:
        raise ValueError(f"Invalid semantic_feat_type: {args.semantic_feat_type}")
    
    if args.repa_dino_model_name:
        device_repa = 'cuda'
        repa_model_name = args.repa_dino_model_name

        # Choose loading method based on model type
        if 'dinov2' in repa_model_name:
            # DINOv2 models loaded via torch.hub
            if rank == 0:
                print(f'Loading REPA DINOv2 model: {repa_model_name}')
            repa_model = torch.hub.load('facebookresearch/dinov2', repa_model_name)
            if hasattr(repa_model, 'head'):
                del repa_model.head
                repa_model.head = torch.nn.Identity()
            repa_model.to(device_repa)
            repa_model.eval()
            # DINOv2 uses ImageNet standard parameters
            from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            repa_normalize_mean = IMAGENET_DEFAULT_MEAN
            repa_normalize_std = IMAGENET_DEFAULT_STD
            repa_model_type = 'dinov2'
        else:
            # MAE, CLIP, SigLIP use UnifiedFeatureExtractor (without VAE)
            from tokenizer.semvae.extract_other_feature import UnifiedFeatureExtractor
            if rank == 0:
                print(f'Loading REPA model via UnifiedFeatureExtractor: {repa_model_name}')
            # Don't load VAE, only use encoder
            repa_extractor = UnifiedFeatureExtractor(
                repa_model_name,
                'cuda',
                vae_config_path=None,
                vae_checkpoint_path=None)
            repa_model = repa_extractor.encoder
            repa_model_type = repa_extractor.model_type
            repa_model_source = repa_extractor.model_source
            repa_patch_grid_size = repa_extractor.patch_grid_size
            # Use model-specific normalization parameters
            repa_normalize_mean = repa_extractor.normalize_params['mean']
            repa_normalize_std = repa_extractor.normalize_params['std']

        if rank == 0:
            print(f'REPA normalize params: mean={repa_normalize_mean}, std={repa_normalize_std}')

        from torchvision import transforms
        import torch.nn as nn
        import einops
        repa_transform = transforms.Compose([transforms.Normalize(mean=repa_normalize_mean, std=repa_normalize_std)])

        feats_repa = []
        feats_repa_flip = []

    # Setup data:
    datasets = [
        ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=0.0)),
        ImageFolder(args.data_path, transform=tokenizer.img_transform(p_hflip=1.0))
    ]
    samplers = [
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        ) for dataset in datasets
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        ) for dataset, sampler in zip(datasets, samplers)
    ]
    total_data_in_loop = len(loaders[0].dataset)
    if rank == 0:
        print(f"Total data in one loop: {total_data_in_loop}")

    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images} of {total_data_in_loop} images')
        
        for loader_idx, data in enumerate(batch_data):
            x = data[0]
            y = data[1]  # (N,)
            
            z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)

            if batch_idx == 0 and rank == 0:
                print('latent shape', z.shape, 'dtype', z.dtype)
            
            if loader_idx == 0:
                latents.append(z)
                labels.append(y)
            else:
                latents_flip.append(z)

            if args.semantic_feat_type == 'dino_pca_ch8':
                with torch.no_grad():
                    x_preprocessed = preprocess_raw_image((x+1)/2 * 255.).to(device_dino_pca)
                    z_dino = dino_encoder.forward_features(x_preprocessed)['x_norm_patchtokens'].detach()
                    z_dino = (z_dino - mean_dino) / std_dino
                    z_dino = z_dino - pca_mean

                    z_pca = z_dino @ pca_components.T
                    z_pca = z_pca.reshape(z_pca.shape[0], 16, 16, pca_rank).permute(0,3,1,2)
                    z_pca = z_pca.contiguous().to('cpu')

                if batch_idx == 0 and rank == 0:
                    print('dino latent shape', z_pca.shape, 'dtype', z_pca.dtype)
                
                if loader_idx == 0:
                    latents_dino.append(z_pca)
                else:
                    latents_dino_flip.append(z_pca)
            elif args.semantic_feat_type == 'semvae':
                with torch.no_grad():
                    if args.image_size == 512:
                        # 512×512: Split into 4x 256×256 patches, batch extract, rearrange to 32×32 using einops
                        B, C, H, W = x.shape
                        assert H == 512 and W == 512, f"Expected 512×512, got {H}×{W}"

                        # Use einops to rearrange spatial dimensions to batch dimension: [B, C, 512, 512] -> [4B, C, 256, 256]
                        x_patches = einops.rearrange(x, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', p1=2, p2=2)

                        # Batch preprocessing and extraction (process all patches at once)
                        x_patches_preprocessed = sv_transform((x_patches+1)/2)
                        x_patches_preprocessed = nn.functional.interpolate(x_patches_preprocessed, size=(224, 224), mode='bicubic').to(device_semvae)
                        z_sem_patches = sv_extractor.encode_images_to_vae_latents(x_patches_preprocessed)  # [4B, 256, D]
                        z_sem_patches = einops.rearrange(z_sem_patches, 'b (h w) d -> b d h w', h=16, w=16)  # [4B, D, 16, 16]

                        # Use einops to rearrange batch dimension back to spatial dimensions: [4B, D, 16, 16] -> [B, D, 32, 32]
                        z_sem = einops.rearrange(z_sem_patches, '(b p1 p2) d h w -> b d (p1 h) (p2 w)', b=B, p1=2, p2=2)
                        z_sem = z_sem.contiguous().to('cpu')
                    else:
                        # 256×256: Normal processing
                        x_preprocessed = sv_transform((x+1)/2)
                        x_preprocessed = nn.functional.interpolate(x_preprocessed, size=(224, 224), mode='bicubic').to(device_semvae)
                        z_sem = sv_extractor.encode_images_to_vae_latents(x_preprocessed)
                        z_sem = einops.rearrange(z_sem, 'b (h w) d -> b d h w', h=16, w=16)  # [B, D, 16, 16]
                        z_sem = z_sem.contiguous().to('cpu')

                if batch_idx == 0 and rank == 0:
                    print('semvae latent shape', z_sem.shape, 'dtype', z_sem.dtype)

                if loader_idx == 0:
                    latents_sv.append(z_sem)
                else:
                    latents_sv_flip.append(z_sem)

            if args.repa_dino_model_name:
                with torch.no_grad():
                    if args.image_size == 512:
                        # 512×512: Split into 4x 256×256 patches, batch extract, rearrange to 32×32 using einops
                        B, C, H, W = x.shape
                        assert H == 512 and W == 512, f"Expected 512×512, got {H}×{W}"

                        # Use einops to rearrange spatial dimensions to batch dimension: [B, C, 512, 512] -> [4B, C, 256, 256]
                        x_patches = einops.rearrange(x, 'b c (p1 h) (p2 w) -> (b p1 p2) c h w', p1=2, p2=2)

                        # Batch preprocessing and extraction (process all patches at once)
                        x_patches_preprocessed = repa_transform((x_patches+1)/2)
                        x_patches_preprocessed = nn.functional.interpolate(x_patches_preprocessed, size=(224, 224), mode='bicubic').to(device_repa)

                        # Extract features based on model type (batch process all patches)
                        if repa_model_type == 'dinov2':
                            # DINOv2 model
                            feat_repa_patches = repa_model.forward_features(x_patches_preprocessed)['x_norm_patchtokens'].detach()
                            # DINOv2 outputs 16×16, directly reshape
                            feat_repa_patches = einops.rearrange(feat_repa_patches, 'b (h w) d -> b d h w', h=16, w=16)
                        elif repa_model_source == 'timm':
                            # MAE model (timm)
                            features = repa_model.forward_features(x_patches_preprocessed)
                            feat_repa_patches = features[:, 1:, :]  # [4B, num_patches, feat_dim]
                            # Interpolate to 16×16
                            B_patches, num_patches, C_feat = feat_repa_patches.shape
                            H_patch = W_patch = repa_patch_grid_size
                            feat_repa_patches = feat_repa_patches.reshape(B_patches, H_patch, W_patch, C_feat).permute(0, 3, 1, 2)  # [4B, C, H, W]
                            if H_patch != 16:
                                feat_repa_patches = nn.functional.interpolate(feat_repa_patches, size=(16, 16), mode='bilinear', align_corners=False)
                        elif repa_model_source == 'hf':
                            # CLIP / SigLIP model
                            outputs = repa_model(x_patches_preprocessed)
                            if repa_model_type == 'clip':
                                last_hidden_state = outputs.last_hidden_state
                                feat_repa_patches = last_hidden_state[:, 1:, :]  # [4B, num_patches, feat_dim]
                            elif repa_model_type == 'siglip':
                                feat_repa_patches = outputs.last_hidden_state  # [4B, num_patches, feat_dim]

                            # Interpolate to 16×16
                            B_patches, num_patches, C_feat = feat_repa_patches.shape
                            H_patch = W_patch = repa_patch_grid_size
                            feat_repa_patches = feat_repa_patches.reshape(B_patches, H_patch, W_patch, C_feat).permute(0, 3, 1, 2)  # [4B, C, H, W]
                            if H_patch != 16:
                                feat_repa_patches = nn.functional.interpolate(feat_repa_patches, size=(16, 16), mode='bilinear', align_corners=False)

                        # Use einops to rearrange batch dimension back to spatial dimensions: [4B, D, 16, 16] -> [B, D, 32, 32]
                        feat_repa = einops.rearrange(feat_repa_patches, '(b p1 p2) d h w -> b d (p1 h) (p2 w)', b=B, p1=2, p2=2)
                        feat_repa = feat_repa.contiguous().to('cpu').half()
                    else:
                        # 256×256: Normal processing
                        x_for_repa = x

                        # Normalize -> Resize
                        x_preprocessed = repa_transform((x_for_repa+1)/2)
                        x_preprocessed = nn.functional.interpolate(x_preprocessed, size=(224, 224), mode='bicubic').to(device_repa)

                        # Extract features based on model type
                        if repa_model_type == 'dinov2':
                            # DINOv2 model
                            feat_repa = repa_model.forward_features(x_preprocessed)['x_norm_patchtokens'].detach()
                            # DINOv2 outputs 16×16, directly reshape
                            feat_repa = einops.rearrange(feat_repa, 'b (h w) d -> b d h w', h=16, w=16)
                        elif repa_model_source == 'timm':
                            # MAE model (timm)
                            features = repa_model.forward_features(x_preprocessed)
                            feat_repa = features[:, 1:, :]  # [B, num_patches, feat_dim]
                            # Interpolate to 16×16
                            B, num_patches, C = feat_repa.shape
                            H = W = repa_patch_grid_size
                            feat_repa = feat_repa.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
                            if H != 16:
                                feat_repa = nn.functional.interpolate(feat_repa, size=(16, 16), mode='bilinear', align_corners=False)
                        elif repa_model_source == 'hf':
                            # CLIP / SigLIP model
                            outputs = repa_model(x_preprocessed)
                            if repa_model_type == 'clip':
                                last_hidden_state = outputs.last_hidden_state
                                feat_repa = last_hidden_state[:, 1:, :]  # [B, num_patches, feat_dim]
                            elif repa_model_type == 'siglip':
                                feat_repa = outputs.last_hidden_state  # [B, num_patches, feat_dim]

                            # Interpolate to 16×16
                            B, num_patches, C = feat_repa.shape
                            H = W = repa_patch_grid_size
                            feat_repa = feat_repa.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
                            if H != 16:
                                feat_repa = nn.functional.interpolate(feat_repa, size=(16, 16), mode='bilinear', align_corners=False)

                        feat_repa = feat_repa.contiguous().to('cpu').half()

                if batch_idx == 0 and rank == 0:
                    print('repa feature shape', feat_repa.shape, 'dtype', feat_repa.dtype)

                if loader_idx == 0:
                    feats_repa.append(feat_repa)
                else:
                    feats_repa_flip.append(feat_repa)

        if len(latents) == 1000 // args.batch_size:
            latents = torch.cat(latents, dim=0)
            latents_flip = torch.cat(latents_flip, dim=0)
            labels = torch.cat(labels, dim=0)
            save_dict = {
                'latents': latents,
                'latents_flip': latents_flip,
                'labels': labels
            }
            if args.semantic_feat_type == 'dino_pca_ch8':
                latents_dino = torch.cat(latents_dino, dim=0)
                latents_dino_flip = torch.cat(latents_dino_flip, dim=0)
                save_dict['latents_dino'] = latents_dino
                save_dict['latents_dino_flip'] = latents_dino_flip
            elif args.semantic_feat_type == 'semvae':
                latents_sv = torch.cat(latents_sv, dim=0)
                latents_sv_flip = torch.cat(latents_sv_flip, dim=0)
                save_dict['latents_sv'] = latents_sv
                save_dict['latents_sv_flip'] = latents_sv_flip

            if args.repa_dino_model_name:
                feats_repa = torch.cat(feats_repa, dim=0)
                feats_repa_flip = torch.cat(feats_repa_flip, dim=0)
                save_dict['feats_dino'] = feats_repa
                save_dict['feats_dino_flip'] = feats_repa_flip

            for key in save_dict:
                if rank == 0:
                    print(key, save_dict[key].shape)
            save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
            )
            if rank == 0:
                print(f'Saved {save_filename}')
            
            latents = []
            latents_flip = []
            labels = []
            saved_files += 1

            if args.semantic_feat_type == 'dino_pca_ch8':
                latents_dino = []
                latents_dino_flip = []
            elif args.semantic_feat_type == 'semvae':
                latents_sv = []
                latents_sv_flip = []

            if args.repa_dino_model_name:
                feats_repa = []
                feats_repa_flip = []

    # Save remainder latents
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        latents_flip = torch.cat(latents_flip, dim=0)
        labels = torch.cat(labels, dim=0)
        save_dict = {
            'latents': latents,
            'latents_flip': latents_flip,
            'labels': labels
        }
        if args.semantic_feat_type == 'dino_pca_ch8':
            latents_dino = torch.cat(latents_dino, dim=0)
            latents_dino_flip = torch.cat(latents_dino_flip, dim=0)
            save_dict['latents_dino'] = latents_dino
            save_dict['latents_dino_flip'] = latents_dino_flip
        elif args.semantic_feat_type == 'semvae':
            latents_sv = torch.cat(latents_sv, dim=0)
            latents_sv_flip = torch.cat(latents_sv_flip, dim=0)
            save_dict['latents_sv'] = latents_sv
            save_dict['latents_sv_flip'] = latents_sv_flip

        if args.repa_dino_model_name:
            feats_repa = torch.cat(feats_repa, dim=0)
            feats_repa_flip = torch.cat(feats_repa_flip, dim=0)
            save_dict['feats_dino'] = feats_repa
            save_dict['feats_dino_flip'] = feats_repa_flip

        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
        )
        if rank == 0:
            print(f'Saved {save_filename}')

    # Calculate latents stats
    dist.barrier()
    if rank == 0:
        print('Calculating latents stats...')
        latent_sv_norm = args.semantic_feat_type == 'semvae'
        dataset = ImgLatentDataset(output_dir, latent_norm=True, latent_sv_norm=latent_sv_norm)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/scratch/inf0/user/mparcham/ILSVRC2012/train')
    parser.add_argument("--data_split", type=str, default='imagenet_train')
    parser.add_argument("--output_path", type=str, default="/scratch/inf0/user/bpogodzi/datasets/imagenet-sfd-latents/train")
    parser.add_argument("--config", type=str, default="config_details.yaml")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    
    parser.add_argument("--semantic_feat_type", type=str, default='')
    parser.add_argument('--repa_dino_model_name', type=str, default='')
    args = parser.parse_args()
    main(args)