import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import safetensors.torch as st
from tqdm import tqdm
import json
from omegaconf import OmegaConf
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from tokenizer.semvae.models.vae import create_semantic_autoencoder

class DINOFeatureExtractor:
    """DINOv2 feature extractor (supports VAE compression)"""
    def __init__(self, model_name='dinov2_vitb14_reg', device='cuda',
                 vae_config_path=None, vae_checkpoint_path=None):
        self.model_name = model_name
        self.device = device
        self.use_vae = vae_config_path is not None and vae_checkpoint_path is not None

        print(f"Loading {model_name} model...")
        # Load DINOv2 model
        self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)
        if hasattr(self.encoder, 'head'):
            del self.encoder.head
            self.encoder.head = torch.nn.Identity()
        self.encoder.to(device)
        self.encoder.eval()

        # Get feature dimensions
        self.feature_dims = {
            'dinov2_vits14_reg': 384,
            'dinov2_vitb14_reg': 768,
            'dinov2_vitl14_reg': 1024,
            'dinov2_vitg14_reg': 1536,
        }
        self.feature_dim = self.feature_dims[model_name]
        print(f"Feature dimension: {self.feature_dim}")

        # Load VAE model (if specified)
        self.vae = None
        self.vae_arch = None
        if self.use_vae:
            print(f"Loading VAE config: {vae_config_path}")
            print(f"Loading VAE checkpoint: {vae_checkpoint_path}")
            self.vae_config = OmegaConf.load(vae_config_path)
            self.vae_arch = self.vae_config['model']['arch']

            # Create VAE model
            self.vae = create_semantic_autoencoder(
                arch=self.vae_config['model']['arch'],
                is_vae=self.vae_config['model']['variational'],
                input_dim=self.feature_dim,
                bottleneck_dim=self.vae_config['model']['bottleneck_dim'],
                **self.vae_config['model']['params']
            )

            # Load checkpoint.
            # PyTorch 2.6+ changed torch.load default `weights_only=True`, which
            # breaks loading this training checkpoint format. Use
            # `weights_only=False` when available, and fall back for older torch.
            try:
                checkpoint = torch.load(
                    vae_checkpoint_path,
                    map_location='cpu',
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
            msg = self.vae.load_state_dict(checkpoint['model_state_dict'])
            print('Missing keys:', msg.missing_keys)
            print('Unexpected keys:', msg.unexpected_keys)
            self.vae.to(device)
            self.vae.eval()
            print(f"VAE model loaded, architecture: {self.vae_arch}")
        
    @staticmethod
    def get_preprocess_transform():
        """Get preprocessing transform for external data loaders"""
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            # TODO: should I directly use resizing shorter edge to 224 and then conducting center crop?
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
        
    def extract_features(self, images):
        """
        Extract DINOv2 features (optional VAE compression)
        Args:
            images: [B, 3, 224, 224] preprocessed tensor
        Returns:
            features: dict with 'patch_tokens' [B, num_patches, feat_dim] and 'cls_token' [B, feat_dim]
        """
        with torch.no_grad():
            # Extract DINOv2 features
            features = self.encoder.forward_features(images)

            # Get patch tokens and cls token
            patch_tokens = features['x_norm_patchtokens']  # [B, num_patches, feat_dim]
            cls_token = features['x_norm_clstoken']        # [B, feat_dim]

            # If using VAE, perform compression and reconstruction
            if self.use_vae and self.vae is not None:
                # VAE inputs patch tokens, outputs reconstructed patch tokens
                vae_output = self.vae(patch_tokens)
                if isinstance(vae_output, tuple) and len(vae_output) >= 2:
                    # Variational autoencoder returns (recon, bottleneck, kl_loss)
                    patch_tokens_recon = vae_output[0]
                else:
                    # Regular autoencoder returns (recon, bottleneck)
                    patch_tokens_recon = vae_output[0] if isinstance(vae_output, tuple) else vae_output

                # Use reconstructed patch tokens
                patch_tokens = patch_tokens_recon

                # Don't save cls token when using VAE to save storage space
                cls_token = None

        return {
            'patch_tokens': patch_tokens,
            'cls_token': cls_token,
            'num_patches': patch_tokens.shape[1]
        }

    def encode_images_to_vae_latents(self, images):
        """Encode images to latent representations
        Args:
            images: Input image tensor
        Returns:
            torch.Tensor: Encoded latent representation
        """
        with torch.no_grad():
            # Extract DINOv2 features
            features = self.encoder.forward_features(images)

            # Get patch tokens and cls token
            patch_tokens = features['x_norm_patchtokens']  # [B, num_patches, feat_dim]

            assert self.use_vae and self.vae is not None, "VAE is not used"
            output = self.vae.encode(patch_tokens)
            if isinstance(output, tuple) and len(output) >= 2:
                # Variational autoencoder returns (z, kl)
                z = output[0]
            else:
                # Regular autoencoder returns z
                z = output
                
            return z


class LimitedImageFolder(ImageFolder):
    """ImageFolder with limited sample count"""
    def __init__(self, root, transform=None, max_samples=None):
        super().__init__(root, transform=transform)

        if max_samples is not None and max_samples < len(self.samples):
            # Take first max_samples samples
            self.samples = self.samples[:max_samples]
            self.targets = self.targets[:max_samples]
            print(f"Limited sample count to: {max_samples}")

        print(f"Dataset sample count: {len(self.samples)}")
        print(f"Number of classes: {len(self.classes)}")


def extract_and_save_features(args):
    """Extract and save features"""

    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    print(f"Using device: {device}")

    # VAE compression parameter validation
    vae_config_path = None
    vae_checkpoint_path = None

    if args.use_vae_compression:
        if args.vae_checkpoint is None:
            raise ValueError("Must specify --vae_checkpoint parameter when using VAE compression")
        
        vae_config_path = args.vae_config
        vae_checkpoint_path = args.vae_checkpoint

        if not os.path.exists(vae_config_path):
            raise FileNotFoundError(f"VAE config file does not exist: {vae_config_path}")
        if not os.path.exists(vae_checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint file does not exist: {vae_checkpoint_path}")

        print(f"VAE compression enabled")

    # Create output directory (create model_name subdirectory under output_root)
    output_dir = os.path.join(args.output_root, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize feature extractor
    extractor = DINOFeatureExtractor(args.model_name, device, vae_config_path, vae_checkpoint_path)

    # Create dataset and data loader
    # Note: using preprocessing transform provided by DINOFeatureExtractor
    dataset = LimitedImageFolder(
        root=args.data_root,
        transform=DINOFeatureExtractor.get_preprocess_transform(),
        max_samples=args.max_samples
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=args.shuffle, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    # List to store all features
    all_patch_tokens = []
    all_cls_tokens = []
    all_patch_avg = []  # New: store average of patch tokens
    all_class_indices = []
    all_image_paths = []
    num_patches = None  # Record number of patches

    print(f"Starting feature extraction, total batches: {len(dataloader)}")

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
        images = images.to(device)

        # Extract features - images are already preprocessed [B, 3, 224, 224] tensors
        features = extractor.extract_features(images)

        # Record number of patches (first time only)
        if num_patches is None:
            num_patches = features['num_patches']

        # Calculate and save average of patch tokens
        patch_avg = features['patch_tokens'].mean(dim=1)  # [B, feat_dim]
        all_patch_avg.append(patch_avg.cpu().half())

        # Only save complete patch_tokens when not using VAE compression
        if not (args.use_vae_compression and extractor.use_vae):
            all_patch_tokens.append(features['patch_tokens'].cpu().half())
        if features['cls_token'] is not None:
            all_cls_tokens.append(features['cls_token'].cpu().half())
        all_class_indices.extend(labels.tolist())

        # Get image paths for current batch
        batch_start_idx = batch_idx * args.batch_size
        batch_end_idx = min(batch_start_idx + len(images), len(dataset.samples))
        batch_paths = [dataset.samples[i][0] for i in range(batch_start_idx, batch_end_idx)]
        all_image_paths.extend(batch_paths)

        # Periodically clean up GPU memory
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    # Merge all features
    print("Merging features...")
    all_patch_avg = torch.cat(all_patch_avg, dim=0)        # [N, feat_dim]

    if all_patch_tokens:
        all_patch_tokens = torch.cat(all_patch_tokens, dim=0)
        print(f"  Patch tokens shape: {all_patch_tokens.shape}")
    else:
        all_patch_tokens = None
        print(f"  Patch tokens: None")

    if all_cls_tokens:
        all_cls_tokens = torch.cat(all_cls_tokens, dim=0)
        print(f"  CLS tokens shape: {all_cls_tokens.shape}")
    else:
        all_cls_tokens = None
        print(f"  CLS tokens: None")

    print(f"Extraction complete:")
    print(f"  Sample count: {len(all_class_indices)}")
    print(f"  Patch average shape: {all_patch_avg.shape}")

    # Save as safetensors format (use fp16 to save storage space)
    tensors_to_save = {
        'patch_avg': all_patch_avg,
        'class_indices': torch.tensor(all_class_indices, dtype=torch.long)
    }

    if all_patch_tokens is not None:
        tensors_to_save['patch_tokens'] = all_patch_tokens
    if all_cls_tokens is not None:
        tensors_to_save['cls_tokens'] = all_cls_tokens

    # Ensure filename includes sample count
    actual_samples = len(all_class_indices)
    safetensors_path = os.path.join(output_dir, f'features_{actual_samples}samples.safetensors')
    st.save_file(tensors_to_save, safetensors_path)
    print(f"Feature tensors saved to: {safetensors_path}")

    # Get class name mapping
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[idx] for idx in all_class_indices]

    # Save metadata as JSON
    metadata = {
        'class_names': class_names,
        'image_paths': all_image_paths,
        'model_name': args.model_name,
        'feature_dim': extractor.feature_dim,
        'num_samples': len(all_class_indices),
        'num_patches': num_patches,
        'patch_tokens_shape': list(all_patch_tokens.shape) if all_patch_tokens is not None else None,
        'cls_tokens_shape': list(all_cls_tokens.shape) if all_cls_tokens is not None else None,
        'patch_avg_shape': list(all_patch_avg.shape),
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(dataset.classes),
        'extraction_config': {
            'data_root': args.data_root,
            'model_name': args.model_name,
            'max_samples': args.max_samples,
            'batch_size': args.batch_size,
            'shuffle': args.shuffle,
            'use_vae_compression': args.use_vae_compression,
            'vae_config': args.vae_config if args.use_vae_compression else None,
            'vae_checkpoint': args.vae_checkpoint if args.use_vae_compression else None
        }
    }
    
    metadata_path = os.path.join(output_dir, f'metadata_{actual_samples}samples.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print("Feature extraction complete!")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"First 10 classes: {dataset.classes[:10]}")
    print(f"Actual samples extracted: {actual_samples}")


def main():
    parser = argparse.ArgumentParser(description='Extract DINOv2 features (supports VAE compression)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='ImageNet dataset root directory (directory containing class folders)')
    parser.add_argument('--output_root', type=str, default='outputs/dataset/imagenet-dinov2',
                       help='Output root directory')
    parser.add_argument('--model_name', type=str, default='dinov2_vitb14_reg',
                       choices=['dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'],
                       help='DINOv2 model name')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading threads')
    parser.add_argument('--shuffle', action='store_true',
                       help='Whether to shuffle data')

    # VAE compression related parameters
    parser.add_argument('--use_vae_compression', action='store_true',
                       help='Whether to use VAE compression for features')
    parser.add_argument('--vae_config', type=str, default='tokenizer/configs/vae_semantic/vae/dinov2_vits14_reg/transformer_ch8.yaml',
                       help='VAE config file path')
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                       help='VAE checkpoint file path')
    
    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Data root directory: {args.data_root}")
    print(f"  Output root directory: {args.output_root}")
    print(f"  Model name: {args.model_name}")
    print(f"  Maximum samples: {args.max_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of data loading threads: {args.num_workers}")
    print(f"  Shuffle data: {args.shuffle}")

    if args.use_vae_compression:
        print(f"  VAE compression settings:")
        print(f"    VAE compression enabled: {args.use_vae_compression}")
        print(f"    VAE config file: {args.vae_config}")
        print(f"    VAE checkpoint: {args.vae_checkpoint}")
    else:
        print(f"  VAE compression: Not enabled")
    
    extract_and_save_features(args)


if __name__ == "__main__":
    main()