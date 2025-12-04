"""
ImageNet Latent Dataset with safetensors.
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from safetensors import safe_open


class ImgLatentDataset(Dataset):
    def __init__(self, data_dir, latent_norm=True, latent_sv_norm=False, latent_multiplier=1.0):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_sv_norm = latent_sv_norm
        self.latent_multiplier = latent_multiplier

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()
        if latent_sv_norm:
            self._latent_sv_mean, self._latent_sv_std = self.get_latent_stats(latent_key='latents_sv')

    def get_img_to_safefile_map(self):
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice('labels')
                labels_shape = labels.get_shape()
                num_imgs = labels_shape[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len+i] = {
                        'safe_file': safe_file,
                        'idx_in_file': i
                    }
        return img_to_file

    def get_latent_stats(self, latent_key='latents'):
        latent_stats_cache_file = os.path.join(self.data_dir, f"{latent_key}_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            latent_stats = self.compute_latent_stats(latent_key=latent_key)
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']
    
    def compute_latent_stats(self, latent_key='latents'):
        num_samples = min(10000, len(self.img_to_file_map))
        random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        latents = []
        for idx in tqdm(random_indices):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                features = f.get_slice(latent_key)
                feature = features[img_idx:img_idx+1]
                latents.append(feature)
        latents = torch.cat(latents, dim=0)
        mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        std = latents.std(dim=[0, 2, 3], keepdim=True)
        latent_stats = {'mean': mean, 'std': std}
        print(latent_stats)
        return latent_stats

    def __len__(self):
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            features = f.get_slice(tensor_key)
            labels = f.get_slice('labels')
            feature = features[img_idx:img_idx+1]
            label = labels[img_idx:img_idx+1]

            if self.latent_norm:
                feature = (feature - self._latent_mean) / self._latent_std

            # cat dino pca feature if needed; remember to flip following the feature order
            if 'latents_dino' in f.keys():
                tensor_key_dino = "latents_dino" if tensor_key=="latents" else "latents_dino_flip"
                features_dino = f.get_slice(tensor_key_dino)
                feature_dino = features_dino[img_idx:img_idx+1]
                feature = torch.cat((feature, feature_dino), dim=1)
            if 'latents_sv' in f.keys():
                tensor_key_sv = "latents_sv" if tensor_key=="latents" else "latents_sv_flip"
                features_sv = f.get_slice(tensor_key_sv)
                feature_sv = features_sv[img_idx:img_idx+1]
                if self.latent_sv_norm:
                    feature_sv = (feature_sv - self._latent_sv_mean) / self._latent_sv_std
                feature = torch.cat((feature, feature_sv), dim=1)
            if 'feats_dino' in f.keys():
                tensor_key_feature_dino = "feats_dino" if tensor_key=="latents" else "feats_dino_flip"
                features_dino = f.get_slice(tensor_key_feature_dino)
                feature_dino = features_dino[img_idx:img_idx+1].float()
                feature = feature * self.latent_multiplier
                feature = feature.squeeze(0)
                label = label.squeeze(0)

                feature_dino = feature_dino.squeeze(0)
                
                return feature, label, feature_dino
            
        feature = feature * self.latent_multiplier
        
        # remove the first batch dimension (=1) kept by get_slice()
        feature = feature.squeeze(0)
        label = label.squeeze(0)
        return feature, label