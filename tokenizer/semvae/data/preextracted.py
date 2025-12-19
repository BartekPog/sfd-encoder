import os
import torch
import json
import safetensors.torch as st


class PreExtractedDINODataset(torch.utils.data.Dataset):
    """Pre-extracted DINO features dataset"""
    def __init__(self, feature_dir, model_name, max_samples):
        self.feature_dir = feature_dir
        self.model_name = model_name

        # Construct file paths
        feature_path = os.path.join(feature_dir, model_name, f'features_{max_samples}samples.safetensors')
        metadata_path = os.path.join(feature_dir, model_name, f'metadata_{max_samples}samples.json')

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file does not exist: {feature_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

        print(f"Loading pre-extracted features: {feature_path}")

        # Load tensors
        tensors = st.load_file(feature_path)
        self.patch_tokens = tensors['patch_tokens']  # [N, num_patches, feat_dim]
        self.cls_tokens = tensors['cls_tokens']      # [N, feat_dim]
        self.class_indices = tensors['class_indices'] # [N]

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.class_names = metadata['class_names']
        self.image_paths = metadata['image_paths']
        self.feature_dim = metadata['feature_dim']
        self.num_samples = metadata['num_samples']

        print(f"Loading complete:")
        print(f"  Sample count: {self.num_samples}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Patch tokens shape: {self.patch_tokens.shape}")
        print(f"  CLS tokens shape: {self.cls_tokens.shape}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'patch_tokens': self.patch_tokens[idx].float(),  # [num_patches, feat_dim]
            'cls_tokens': self.cls_tokens[idx].float(),      # [feat_dim]
            'class_idx': self.class_indices[idx].item(),    # not verified, might have bug
            'class_name': self.class_names[idx],
            'image_path': self.image_paths[idx]
        }


def build_preextracted_dataloader(config):
    """Build data loader for pre-extracted features"""
    data_config = config['data']
    
    dataset = PreExtractedDINODataset(
        feature_dir=data_config['feature_dir'],
        model_name=data_config['model_name'],
        max_samples=data_config['max_samples']
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_config['batch_size'],
        shuffle=data_config['shuffle'],
        num_workers=data_config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader


def build_eval_dataloader(config):
    """Build evaluation data loader"""
    eval_config = config['training']['evaluation']

    dataset = PreExtractedDINODataset(
        feature_dir=eval_config['feature_dir'],
        model_name=eval_config['model_name'],
        max_samples=eval_config['max_samples']
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_config.get('batch_size', 32),
        shuffle=False,  # Don't shuffle during evaluation
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader