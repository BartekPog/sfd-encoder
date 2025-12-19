import math
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gate = nn.Linear(dim_in, dim_out)
        self.up = nn.Linear(dim_in, dim_out)
    
    def forward(self, x):
        gate = self.gate(x)
        up = self.up(x)
        return F.silu(gate) * up


class ResBlock(nn.Module):
    """Residual block"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        return x + self.mlp(self.norm(x))


class TransformerBlock(nn.Module):
    """Transformer block"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class SemanticAutoEncoder(nn.Module):
    """
    Semantic AutoEncoder, supports multiple architectures
    """
    def __init__(self, input_dim=1024, bottleneck_dim=32, hidden_dim=512,
                 arch='linear', num_layers=2, num_res_blocks=2,
                 transformer_heads=6, transformer_blocks=12, *args, **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.arch = arch

        # Build encoder and decoder
        if arch == 'linear':
            self.encoder = self._build_linear_encoder()
            self.decoder = self._build_linear_decoder()
        elif arch == 'mlp':
            self.encoder = self._build_mlp_encoder(hidden_dim, num_layers)
            self.decoder = self._build_mlp_decoder(hidden_dim, num_layers)
        elif arch == 'resnet':
            self.encoder = self._build_resnet_encoder(hidden_dim, num_layers, num_res_blocks)
            self.decoder = self._build_resnet_decoder(hidden_dim, num_layers, num_res_blocks)
        elif arch == 'transformer':
            self.encoder = self._build_transformer_encoder(hidden_dim, transformer_heads, transformer_blocks)
            self.decoder = self._build_transformer_decoder(hidden_dim, transformer_heads, transformer_blocks)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
    
    def _build_linear_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.bottleneck_dim)
        )
    
    def _build_linear_decoder(self):
        return nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.input_dim)
        )
    
    def _build_mlp_encoder(self, hidden_dim, num_layers):
        layers = []
        dims = [self.input_dim] + [hidden_dim] * (num_layers - 1) + [self.bottleneck_dim]

        for i in range(len(dims) - 1):
            if i < len(dims) - 2:  # Not the last layer
                layers.append(SwiGLU(dims[i], dims[i+1]))
            else:  # Last layer
                layers.append(nn.Linear(dims[i], dims[i+1]))
        
        return nn.Sequential(*layers)
    
    def _build_mlp_decoder(self, hidden_dim, num_layers):
        layers = []
        dims = [self.bottleneck_dim] + [hidden_dim] * (num_layers - 1) + [self.input_dim]

        for i in range(len(dims) - 1):
            if i < len(dims) - 2:  # Not the last layer
                layers.append(SwiGLU(dims[i], dims[i+1]))
            else:  # Last layer
                layers.append(nn.Linear(dims[i], dims[i+1]))
        
        return nn.Sequential(*layers)
    
    def _build_resnet_encoder(self, hidden_dim, num_layers, num_res_blocks):
        layers = []

        # Initial projection
        layers.append(nn.Linear(self.input_dim, hidden_dim))

        # ResBlocks
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dim))
        
        # MLP layers
        dims = [hidden_dim] * num_layers + [self.bottleneck_dim]
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layers.append(SwiGLU(dims[i], dims[i+1]))
            else:
                layers.append(nn.Linear(dims[i], dims[i+1]))
        
        return nn.Sequential(*layers)
    
    def _build_resnet_decoder(self, hidden_dim, num_layers, num_res_blocks):
        layers = []
        
        # MLP layers
        dims = [self.bottleneck_dim] + [hidden_dim] * num_layers
        for i in range(len(dims) - 1):
            layers.append(SwiGLU(dims[i], dims[i+1]))
        
        # ResBlocks
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dim))

        # Final projection
        layers.append(nn.Linear(hidden_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def _build_transformer_encoder(self, embed_dim, num_heads, num_blocks):
        layers = []

        # Initial projection
        layers.append(nn.Linear(self.input_dim, embed_dim))

        # Transformer blocks
        for _ in range(num_blocks):
            layers.append(TransformerBlock(embed_dim, num_heads))

        # Final projection to bottleneck
        layers.append(nn.LayerNorm(embed_dim))
        layers.append(nn.Linear(embed_dim, self.bottleneck_dim))
        
        return nn.Sequential(*layers)
    
    def _build_transformer_decoder(self, embed_dim, num_heads, num_blocks):
        layers = []

        # Projection from bottleneck
        layers.append(nn.Linear(self.bottleneck_dim, embed_dim))

        # Transformer blocks
        for _ in range(num_blocks):
            layers.append(TransformerBlock(embed_dim, num_heads))

        # Final projection
        layers.append(nn.LayerNorm(embed_dim))
        layers.append(nn.Linear(embed_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def encode(self, x):
        """Encode to bottleneck"""
        z = self.encoder(x)
        return z

    def decode(self, z):
        """Decode from bottleneck"""
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        """Forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class SemanticVariationalAutoEncoder(nn.Module):
    """
    Semantic Variational AutoEncoder, supports multiple architectures
    """
    def __init__(self, input_dim=1024, bottleneck_dim=32, hidden_dim=512,
                 arch='linear', num_layers=2, num_res_blocks=2,
                 transformer_heads=6, transformer_blocks=12, *args, **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.arch = arch

        # For VAE, we need to output 2*bottleneck_dim for mean and variance
        vae_bottleneck_dim = bottleneck_dim * 2

        # Build encoder and decoder
        if arch == 'linear':
            self.encoder = self._build_linear_encoder(vae_bottleneck_dim)
            self.decoder = self._build_linear_decoder()
        elif arch == 'mlp':
            self.encoder = self._build_mlp_encoder(hidden_dim, num_layers, vae_bottleneck_dim)
            self.decoder = self._build_mlp_decoder(hidden_dim, num_layers)
        elif arch == 'resnet':
            self.encoder = self._build_resnet_encoder(hidden_dim, num_layers, num_res_blocks, vae_bottleneck_dim)
            self.decoder = self._build_resnet_decoder(hidden_dim, num_layers, num_res_blocks)
        elif arch == 'transformer':
            self.encoder = self._build_transformer_encoder(hidden_dim, transformer_heads, transformer_blocks, vae_bottleneck_dim)
            self.decoder = self._build_transformer_decoder(hidden_dim, transformer_heads, transformer_blocks)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
    
    def _build_linear_encoder(self, output_dim):
        return nn.Sequential(
            nn.Linear(self.input_dim, output_dim)
        )
    
    def _build_linear_decoder(self):
        return nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.input_dim)
        )
    
    def _build_mlp_encoder(self, hidden_dim, num_layers, output_dim):
        layers = []
        dims = [self.input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layers.append(SwiGLU(dims[i], dims[i+1]))
            else:
                layers.append(nn.Linear(dims[i], dims[i+1]))
        
        return nn.Sequential(*layers)
    
    def _build_mlp_decoder(self, hidden_dim, num_layers):
        layers = []
        dims = [self.bottleneck_dim] + [hidden_dim] * (num_layers - 1) + [self.input_dim]
        
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layers.append(SwiGLU(dims[i], dims[i+1]))
            else:
                layers.append(nn.Linear(dims[i], dims[i+1]))
        
        return nn.Sequential(*layers)
    
    def _build_resnet_encoder(self, hidden_dim, num_layers, num_res_blocks, output_dim):
        layers = []
        
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dim))
        
        dims = [hidden_dim] * num_layers + [output_dim]
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layers.append(SwiGLU(dims[i], dims[i+1]))
            else:
                layers.append(nn.Linear(dims[i], dims[i+1]))
        
        return nn.Sequential(*layers)
    
    def _build_resnet_decoder(self, hidden_dim, num_layers, num_res_blocks):
        layers = []
        
        dims = [self.bottleneck_dim] + [hidden_dim] * num_layers
        for i in range(len(dims) - 1):
            layers.append(SwiGLU(dims[i], dims[i+1]))
        
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def _build_transformer_encoder(self, embed_dim, num_heads, num_blocks, output_dim):
        layers = []
        
        layers.append(nn.Linear(self.input_dim, embed_dim))
        
        for _ in range(num_blocks):
            layers.append(TransformerBlock(embed_dim, num_heads))
        
        layers.append(nn.LayerNorm(embed_dim))
        layers.append(nn.Linear(embed_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _build_transformer_decoder(self, embed_dim, num_heads, num_blocks):
        layers = []
        
        layers.append(nn.Linear(self.bottleneck_dim, embed_dim))
        
        for _ in range(num_blocks):
            layers.append(TransformerBlock(embed_dim, num_heads))
        
        layers.append(nn.LayerNorm(embed_dim))
        layers.append(nn.Linear(embed_dim, self.input_dim))
        
        return nn.Sequential(*layers)
    
    def encode(self, x):
        """Encode to bottleneck"""
        # x shape: [B, L, D] where L=256 (16x16)
        z_flat = self.encoder(x)  # [B, L, bottleneck_dim*2]

        # Assume L=256 can be reshaped to 16x16, adjust if not
        batch_size, seq_len, feat_dim = z_flat.shape

        # Check if seq_len is a perfect square
        hw = int(torch.sqrt(torch.tensor(seq_len)).item())
        if hw * hw != seq_len:
            # If not a perfect square, use the closest square for padding or truncation
            target_hw = 16  # Default to 16x16
            target_seq_len = target_hw * target_hw

            if seq_len > target_seq_len:
                z_flat = z_flat[:, :target_seq_len, :]  # truncate
            elif seq_len < target_seq_len:
                # padding
                padding = torch.zeros(batch_size, target_seq_len - seq_len, feat_dim,
                                    device=z_flat.device, dtype=z_flat.dtype)
                z_flat = torch.cat([z_flat, padding], dim=1)
            hw = target_hw

        # Reshape to 4D tensor for DiagonalGaussianDistribution
        z = einops.rearrange(z_flat, 'b (h w) c -> b c h w', h=hw, w=hw)
        
        posterior = DiagonalGaussianDistribution(z)
        sample = posterior.sample()

        # Reshape back to original format
        z_out = einops.rearrange(sample, 'b c h w -> b (h w) c')

        # If original seq_len differs from target, adjust back
        if seq_len != hw * hw:
            z_out = z_out[:, :seq_len, :]  # Restore original length

        kl = posterior.kl()
        return z_out, kl

    def decode(self, z):
        """Decode from bottleneck"""
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        """Forward pass"""
        z, kl = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, kl


# Factory function for quickly creating models with different architectures
def create_semantic_autoencoder(arch='linear', is_vae=False, **kwargs):
    """
    Create semantic AutoEncoder

    Args:
        arch: 'linear', 'mlp', 'resnet', 'transformer'
        is_vae: Whether to use VAE
        **kwargs: Other parameters
    """
    if is_vae:
        return SemanticVariationalAutoEncoder(arch=arch, **kwargs)
    else:
        return SemanticAutoEncoder(arch=arch, **kwargs)


# Usage example
if __name__ == "__main__":
    # Test different architectures
    batch_size, seq_len, input_dim = 2, 256, 384
    x = torch.randn(batch_size, seq_len, input_dim)
    bottleneck_dim = 32
    # is_vae = True
    is_vae = False
    
    # Linear AE
    model = create_semantic_autoencoder('linear', is_vae=is_vae, bottleneck_dim=bottleneck_dim, input_dim=input_dim)
    if is_vae:
        x_recon, z, kl = model(x)
        print(f"Linear VAE - Input: {x.shape}, Bottleneck: {z.shape}, Output: {x_recon.shape}, KL: {kl.shape}")
    else:
        x_recon, z = model(x)
        print(f"Linear AE - Input: {x.shape}, Bottleneck: {z.shape}, Output: {x_recon.shape}")
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    print('Model Architecture:')
    print(model)
    print('')
    
    # mlp
    model = create_semantic_autoencoder('mlp', is_vae=is_vae, bottleneck_dim=bottleneck_dim, num_layers=3, input_dim=input_dim)
    if is_vae:
        x_recon, z, kl = model(x)
        print(f"MLP VAE - Input: {x.shape}, Bottleneck: {z.shape}, Output: {x_recon.shape}, KL: {kl.shape}")
    else:
        x_recon, z = model(x)
        print(f"MLP AE - Input: {x.shape}, Bottleneck: {z.shape}, Output: {x_recon.shape}")
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    print('Model Architecture:')
    print(model)
    print('')

    # resnet VAE
    model = create_semantic_autoencoder('resnet', is_vae=is_vae, bottleneck_dim=bottleneck_dim, num_layers=3, num_res_blocks=2, input_dim=input_dim)
    if is_vae:
        x_recon, z, kl = model(x)
        print(f" ResNet VAE - Input: {x.shape}, Bottleneck: {z.shape}, Output: {x_recon.shape}, KL: {kl.shape}")
    else:
        x_recon, z = model(x)
        print(f"ResNet AE - Input: {x.shape}, Bottleneck: {z.shape}, Output: {x_recon.shape}")
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    print('Model Architecture:')
    print(model)
    print('')
    
    # Transformer AE
    model = create_semantic_autoencoder('transformer', is_vae=is_vae, bottleneck_dim=bottleneck_dim, input_dim=input_dim,
                                      hidden_dim=384, transformer_heads=6, transformer_blocks=4)
    if is_vae:
        x_recon, z, kl = model(x)
        print(f"Transformer VAE - Input: {x.shape}, Bottleneck: {z.shape}, Output: {x_recon.shape}, KL: {kl.shape}")

    else:
        x_recon, z = model(x)
        print(f"Transformer AE - Input: {x.shape}, Bottleneck: {z.shape}, Output: {x_recon.shape}")
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    print('Model Architecture:')
    print(model)
    print('')