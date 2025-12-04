"""
SFD is built upon LightningDiT: https://github.com/hustvl/LightningDiT.
Original file: https://github.com/hustvl/LightningDiT/blob/main/models/lightningdit.py
"""

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import PatchEmbed, Mlp
from models.swiglu_ffn import SwiGLUFFN 
from models.pos_embed import VisionRotaryEmbeddingFast
# from models.rmsnorm import RMSNorm

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

@torch.compile
def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding
    
    @torch.compile
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Same as DiT.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    @torch.compile
    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            
        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    @torch.compile
    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        use_repa=False,
        repa_dino_version=None,
        repa_depth=None,
        repa_projector_dim=2048,
        semantic_chans=0,
        semfirst_delta_t=0.0,
        semfirst_infer_interval_mode='both'
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.semantic_chans = semantic_chans
        self.semfirst_delta_t = semfirst_delta_t
        self.semfirst_infer_interval_mode = semfirst_infer_interval_mode
        assert semfirst_infer_interval_mode in ['tex', 'sem', 'both'], "semfirst_infer_interval_mode must be 'tex', 'sem', or 'both'"
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        # Initialize timestep embedder (always use single embedder, handle channelwise in forward)
        self.use_semantic_first = self.semfirst_delta_t > 0 and self.semantic_chans > 0
        if self.use_semantic_first:
            self.t_embedder_sem = TimestepEmbedder(hidden_size // 2)
            self.t_embedder_tex = TimestepEmbedder(hidden_size // 2)
        else:
            self.t_embedder = TimestepEmbedder(hidden_size)
            
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)

        self.use_repa = use_repa
        if use_repa:
            # reference code: https://github.com/sihyun-yu/REPA/blob/main/dataset.py
            assert repa_dino_version is not None, "REPA requires repa_dino_version"
            assert repa_depth is not None, "REPA requires repa_depth"
            repa_z_dim_dict = {
                "dinov2_vits14": 384,
                "dinov2_vits14_reg": 384,
                "dinov2_vitb14": 768,
                "dinov2_vitb14_reg": 768,
                "dinov2_vitl14": 1024,
                "dinov2_vitl14_reg": 1024,
                'mae_base': 768,
                'mae_large': 1024,
                'clip_vit_base_patch16': 768,
                'clip_vit_large_patch14': 1024,
                'siglip_base_patch16_224': 768,
            }
            repa_z_dim = repa_z_dim_dict[repa_dino_version]
            def build_mlp(hidden_size, projector_dim, z_dim):
                return nn.Sequential(
                            nn.Linear(hidden_size, projector_dim),
                            nn.SiLU(),
                            nn.Linear(projector_dim, projector_dim),
                            nn.SiLU(),
                            nn.Linear(projector_dim, z_dim),
                        )
            # self.projectors = nn.ModuleList([
            #     build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
            #     ])
            self.repa_projector = build_mlp(hidden_size, repa_projector_dim, repa_z_dim)
            self.repa_depth = repa_depth

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        if self.use_semantic_first:
            nn.init.normal_(self.t_embedder_sem.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder_sem.mlp[2].weight, std=0.02)
            nn.init.normal_(self.t_embedder_tex.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder_tex.mlp[2].weight, std=0.02)
        else:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t=None, y=None):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps or (N, C) tensor for channelwise semantic first mode
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """

        use_checkpoint = self.use_checkpoint

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape                        # debug, to be delete
        # Handle timestep embedding based on semantic first mode
        # if self.use_semantic_first and t.dim() == 2:
        #     # Channelwise timestep mode: t is (N, C)
        #     assert t.shape[1] == self.in_channels, f"Channelwise t must have {self.in_channels} channels, got {t.shape[1]}"

        #     # For channelwise, we can average the timesteps or use texture timestep for overall conditioning
        #     # Here we use the texture timestep (first C-semantic_chans channels)
        #     texture_chans = self.in_channels - self.semantic_chans
        #     t_for_embedding = t[:, :texture_chans].mean(dim=1)  # Average texture timesteps
        #     t_emb = self.t_embedder(t_for_embedding)  # (N, D)
        if self.use_semantic_first:
            t_tex, t_sem = t
            t_emb_sem = self.t_embedder_sem(t_sem)  # (N, D // 2)
            t_emb_tex = self.t_embedder_tex(t_tex)  # (N, D // 2)
            t = torch.cat([t_emb_sem, t_emb_tex], dim=1)  # (N, D)
        else:
            t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        # for block in self.blocks:
        for idx, block in enumerate(self.blocks):
            if use_checkpoint:
                x = checkpoint(block, x, c, self.feat_rope, use_reentrant=True)
            else:
                x = block(x, c, self.feat_rope)
            if self.use_repa and (idx+1) == self.repa_depth:
                repa_feat_proj = self.repa_projector(x)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        if self.use_repa:
            # if eval sample mode (no grad), return x
            if not self.training:
                return x
            return x, repa_feat_proj
        else:
            return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=None, cfg_interval_start=None):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        if self.use_semantic_first:
            # first, we fix the legacy bug in va-vae's version, removed rest of the model_out, and conducted cfg on the whole noisy latent
            eps = model_out
            texture_chans = self.in_channels - self.semantic_chans
            semantic_chans = self.semantic_chans
            eps_tex, eps_sem = eps[:, :texture_chans], eps[:, -semantic_chans:]

            cond_eps_tex, uncond_eps_tex = torch.split(eps_tex, len(eps_tex) // 2, dim=0)
            cond_eps_sem, uncond_eps_sem = torch.split(eps_sem, len(eps_sem) // 2, dim=0)
            half_eps_tex = uncond_eps_tex + cfg_scale * (cond_eps_tex - uncond_eps_tex)
            half_eps_sem = uncond_eps_sem + cfg_scale * (cond_eps_sem - uncond_eps_sem)

            if cfg_interval is True:
                t_tex, t_sem = t
                if self.semfirst_infer_interval_mode in ['both', 'tex'] and t_tex[0] < cfg_interval_start:
                    half_eps_tex = cond_eps_tex
                if self.semfirst_infer_interval_mode in ['both', 'sem'] and t_sem[0] < cfg_interval_start:
                    half_eps_sem = cond_eps_sem
            
            half_eps = torch.cat([half_eps_tex, half_eps_sem], dim=1)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return eps
            
        else:
            # For exact reproducibility reasons, we apply classifier-free guidance on only
            # three channels by default. The standard approach to cfg applies it to all channels.
            # This can be done by uncommenting the following line and commenting-out the line following that.
            eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
            # eps, rest = model_out[:, :3], model_out[:, 3:]              # <--------- legacy implementation in va-vae's version
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            
            if cfg_interval is True:
                timestep = t[0]
                if timestep < cfg_interval_start:
                    half_eps = cond_eps

            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

    def forward_with_autoguidance(self, x, t, y, cfg_scale, autoguidance_model, cfg_interval=False, cfg_interval_start=None,
                                 cfg_scale_sem=None, cfg_scale_tex=None):
        """
        Forward pass with AutoGuidance: main model for conditional, small model for unconditional.
        Supports separate CFG scales for semantic and texture channels.
        """
        # assert cfg_interval is False and cfg_interval_start == 0, "AutoGuidance does not support CFG interval."
        # Main model forward with conditional y
        cond_out = self.forward(x, t, y)

        # Small model forward with conditional y (serves as unconditional)
        uncond_out = autoguidance_model.forward(x, t, y)

        if self.use_semantic_first:
            # Handle semantic first mode
            texture_chans = self.in_channels - self.semantic_chans
            semantic_chans = self.semantic_chans

            cond_eps = cond_out
            uncond_eps = uncond_out

            cond_eps_tex, cond_eps_sem = cond_eps[:, :texture_chans], cond_eps[:, -semantic_chans:]
            uncond_eps_tex, uncond_eps_sem = uncond_eps[:, :texture_chans], uncond_eps[:, -semantic_chans:]

            # Use separate cfg scales if provided, otherwise use cfg_scale for both
            effective_cfg_tex = cfg_scale_tex if cfg_scale_tex is not None else cfg_scale
            effective_cfg_sem = cfg_scale_sem if cfg_scale_sem is not None else cfg_scale

            # Apply CFG formula with potentially different scales
            half_eps_tex = uncond_eps_tex + effective_cfg_tex * (cond_eps_tex - uncond_eps_tex)
            half_eps_sem = uncond_eps_sem + effective_cfg_sem * (cond_eps_sem - uncond_eps_sem)

            if cfg_interval is True:
                t_tex, t_sem = t
                if self.semfirst_infer_interval_mode in ['both', 'tex'] and t_tex[0] < cfg_interval_start:
                    half_eps_tex = cond_eps_tex
                if self.semfirst_infer_interval_mode in ['both', 'sem'] and t_sem[0] < cfg_interval_start:
                    half_eps_sem = cond_eps_sem

            half_eps = torch.cat([half_eps_tex, half_eps_sem], dim=1)
            return half_eps
        else:
            assert cfg_scale_sem is None and cfg_scale_tex is None, "Standard mode does not support separate CFG scales now."
            # Standard mode (without semantic first) - ignore separate scales
            cond_eps, rest = cond_out[:, :self.in_channels], cond_out[:, self.in_channels:]
            uncond_eps = uncond_out[:, :self.in_channels]

            # Apply CFG formula
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            if cfg_interval is True:
                timestep = t[0]
                if timestep < cfg_interval_start:
                    half_eps = cond_eps

            return torch.cat([half_eps, rest], dim=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                             LightningDiT Configs                              #
#################################################################################

def LightningDiT_XL_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def LightningDiT_XL_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def LightningDiT_L_1(**kwargs):
    return LightningDiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def LightningDiT_L_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def LightningDiT_B_1(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def LightningDiT_B_2(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def LightningDiT_S_1(**kwargs):
    return LightningDiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def LightningDiT_S_2(**kwargs):
    return LightningDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def LightningDiT_1p0B_1(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=1, num_heads=24, **kwargs)

def LightningDiT_1p0B_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=2, num_heads=24, **kwargs)

def LightningDiT_1p6B_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=1, num_heads=28, **kwargs)

def LightningDiT_1p6B_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=2, num_heads=28, **kwargs)

LightningDiT_models = {
    'LightningDiT-S/1': LightningDiT_S_1, 'LightningDiT-S/2': LightningDiT_S_2,
    'LightningDiT-B/1': LightningDiT_B_1, 'LightningDiT-B/2': LightningDiT_B_2,
    'LightningDiT-L/1': LightningDiT_L_1, 'LightningDiT-L/2': LightningDiT_L_2,
    'LightningDiT-XL/1': LightningDiT_XL_1, 'LightningDiT-XL/2': LightningDiT_XL_2,
    'LightningDiT-1p0B/1': LightningDiT_1p0B_1, 'LightningDiT-1p0B/2': LightningDiT_1p0B_2,
    'LightningDiT-1p6B/1': LightningDiT_1p6B_1, 'LightningDiT-1p6B/2': LightningDiT_1p6B_2,
}
