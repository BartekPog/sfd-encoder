import torch
from torch import nn, Tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from jaxtyping import Float

from .lightningdit import LightningDiT, TimestepEmbedder




from .pos_embed import rotate_half



class HiddenLightningDiT(LightningDiT):
    def __init__(
        self, 
        *args, 
        num_hidden_tokens: int=8, 
        use_per_token_encoding: bool=True, 
        share_patch_embedder: bool=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        assert num_hidden_tokens > 0, "num_hidden_tokens must be greater than 0"
        self.num_hidden_tokens = num_hidden_tokens
        self.use_per_token_encoding = use_per_token_encoding
        
        # Hidden token output dimension (matches what final_layer produces per token)
        self.hidden_token_dim = self.out_channels * (self.patch_size ** 2)
        
        # Create learnable token embeddings for the hidden tokens
        if use_per_token_encoding:
            self.h_embedding = nn.Parameter(torch.randn(num_hidden_tokens, self.hidden_size))
        else:
            self.h_embedding = nn.Parameter(torch.randn(1, self.hidden_size))
        self.h_embedding.requires_grad = True
        
        self.t_embedder_hid = TimestepEmbedder(self.hidden_size)

        
        # Create positional  embedding for the hidden tokens if we don't use per token encoding
        if not use_per_token_encoding:
            self.pos_emb_hid = nn.Parameter(torch.randn(1, self.num_hidden_tokens, self.hidden_size))
            self.pos_emb_hid.requires_grad = True
        else:            
            self.pos_emb_hid = None
            
            
        if not share_patch_embedder:
            self.x_embedder_hid = nn.Linear(self.in_channels, self.hidden_size)
        
        self.initialize_hidden_weights()
        
        
    def get_h_add_embeddings(self) -> Float[Tensor, "1 num_hidden_tokens model_dim"]:
        if self.use_per_token_encoding:
            return self.h_embedding.unsqueeze(0)  # (1, num_hidden_tokens, model_dim)
        else:
            return self.h_embedding + self.pos_emb_hid  # (1, model_dim) + (1, num_hidden_tokens, model_dim) -> (1, num_hidden_tokens, model_dim)
        
    def initialize_hidden_weights(self):
        if hasattr(self, 'x_embedder_hid'):
            nn.init.xavier_uniform_(self.x_embedder_hid.weight)
            if self.x_embedder_hid.bias is not None:
                nn.init.zeros_(self.x_embedder_hid.bias)
        nn.init.normal_(self.t_embedder_hid.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_hid.mlp[2].weight, std=0.02)
        if self.pos_emb_hid is not None:
            nn.init.normal_(self.pos_emb_hid, std=0.02)
        if self.h_embedding is not None:
            nn.init.normal_(self.h_embedding, std=0.02) 
    
    def embed_hidden(self, hidden_tokens: Float[Tensor, "B N input_h_dim"]) -> Float[Tensor, "B N hidden_dim"]:
        if hasattr(self, 'x_embedder_hid'):
            return self.x_embedder_hid(hidden_tokens)  # (B, N, model_dim)
        
        input_h_dim = hidden_tokens.shape[2]
        
        assert input_h_dim == self.in_channels * (self.patch_size ** 2), f"Expected hidden token input dimension to be {self.in_channels * (self.patch_size ** 2)}, got {input_h_dim}"
        
        patches  = hidden_tokens.view(hidden_tokens.shape[0], hidden_tokens.shape[1], self.in_channels, self.patch_size, self.patch_size)
        B, N, C, ph, pw = patches.shape
        x = patches.reshape(B * N, C, ph, pw)
        x = self.x_embedder.proj(x)              # (B*N, embed_dim, 1, 1) if ph=pw=patch_size
        x = x.flatten(2).transpose(1, 2)     # (B*N, 1, embed_dim)
        x = self.x_embedder.norm(x)              # (B*N, 1, embed_dim)
        x = x[:, 0].reshape(B, N, -1)        # (B, N, embed_dim)
        return x
        
    def _make_hidden_safe_rope(self, num_image_tokens):
        """
        Wrap self.feat_rope so it only applies RoPE to the first num_image_tokens
        tokens and leaves hidden tokens unrotated. Returns None if RoPE is disabled.
        
        Clones freqs_cos/freqs_sin buffers so each pass gets independent version
        counters. Without cloning, DDP's inplace buffer broadcast (or torch.compile's
        inplace optimisations) bump the version counter on the shared buffer, causing
        "modified by an inplace operation" errors when backward traverses an earlier
        pass's autograd graph.
        """
        if self.feat_rope is None:
            return None
        rope = self.feat_rope
        n_img = num_image_tokens
        # Clone once per forward call; all 28 blocks share these cloned tensors
        freqs_cos = rope.freqs_cos.clone()
        freqs_sin = rope.freqs_sin.clone()
        
        class _HiddenSafeRoPE(torch.nn.Module):
            """Applies RoPE only to image tokens, passes hidden tokens through."""
            def __init__(self, n_image, cos, sin):
                super().__init__()
                self.n_image = n_image
                self.cos = cos
                self.sin = sin
            def forward(self, t):
                t_img = t[:, :, :self.n_image, :]   # (B, H, n_img, D)
                t_hid = t[:, :, self.n_image:, :]   # (B, H, n_hid, D)
                # Inline RoPE with cloned buffers
                t_img = t_img * self.cos + rotate_half(t_img) * self.sin
                return torch.cat([t_img, t_hid], dim=2)
        
        return _HiddenSafeRoPE(n_img, freqs_cos, freqs_sin)

    def forward(self, x, t=None, y=None, x_hidden=None, t_hidden=None):
        """
        Forward pass of HiddenLightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps or tuple for semantic first mode
        y: (N,) tensor of class labels
        x_hidden: (N, num_hidden_tokens, hidden_token_dim) tensor of hidden token inputs
        t_hidden: (N,) tensor of timesteps for hidden tokens
        """
        # Backward compatibility: no hidden tokens -> use base class forward
        if x_hidden is None:
            return super().forward(x, t, y)

        use_checkpoint = self.use_checkpoint

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        num_image_tokens = x.shape[1]

        x_hidden_emb = self.embed_hidden(x_hidden)  # (N, num_hidden_tokens, D)
        x_hidden_emb = x_hidden_emb + self.get_h_add_embeddings()

        x = torch.cat([x, x_hidden_emb], dim=1)  # (N, T + num_hidden_tokens, D)

        # Wrap RoPE so it only applies to image tokens (first T), not hidden tokens
        hidden_safe_rope = self._make_hidden_safe_rope(num_image_tokens)

        # Build per-sample conditioning: combine both timestep embeddings + class label
        # Both timestep signals are summed into a single D-dim vector (like t + y in base class).
        # The model distinguishes image vs hidden tokens via the learnable h_embedding, not via conditioning.
        if self.use_semantic_first:
            t_tex, t_sem = t
            t_emb_sem = self.t_embedder_sem(t_sem)  # (N, D // 2)
            t_emb_tex = self.t_embedder_tex(t_tex)  # (N, D // 2)
            t_main = torch.cat([t_emb_sem, t_emb_tex], dim=1)  # (N, D)
        else:
            t_main = self.t_embedder(t)  # (N, D)

        t_hid_emb = self.t_embedder_hid(t_hidden)  # (N, D)
        y = self.y_embedder(y, self.training)        # (N, D)
        c = t_main + t_hid_emb + y                   # (N, D)
        c = c.unsqueeze(1)                            # (N, 1, D) broadcasted in blocks

        for idx, block in enumerate(self.blocks):
            if use_checkpoint:
                x = checkpoint(block, x, c, hidden_safe_rope, use_reentrant=True)
            else:
                x = block(x, c, hidden_safe_rope)
            if self.use_repa and (idx+1) == self.repa_depth:
                # Only project image tokens (exclude hidden tokens) to match feature_dino shape
                repa_feat_proj = self.repa_projector(x[:, :-self.num_hidden_tokens, :])

        x = self.final_layer(x, c)                # (N, T+num_hidden_tokens, patch_size ** 2 * out_channels)

        x_hidden_out = x[:, -self.num_hidden_tokens:, :]  # (N, num_hidden_tokens, patch_size ** 2 * out_channels)
        x = x[:, :-self.num_hidden_tokens, :]              # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
            x_hidden_out, _ = x_hidden_out.chunk(2, dim=2)
        if self.use_repa:
            if not self.training:
                return x, x_hidden_out
            return x, x_hidden_out, repa_feat_proj
        else:
            return x, x_hidden_out

    def forward_with_cfg(self, x, t, y, cfg_scale, x_hidden=None, t_hidden=None,
                         cfg_interval=None, cfg_interval_start=None):
        """
        Forward pass with classifier-free guidance for both image and hidden tokens.
        x is already doubled: [cond_x, uncond_x]
        y is already doubled: [cond_y, uncond_y]
        x_hidden and t_hidden are already doubled to match x's batch size.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        model_out, h_out = self.forward(combined, t, y,
                                         x_hidden=x_hidden,
                                         t_hidden=t_hidden)

        if self.use_semantic_first:
            texture_chans = self.in_channels - self.semantic_chans
            semantic_chans = self.semantic_chans
            eps_tex, eps_sem = model_out[:, :texture_chans], model_out[:, -semantic_chans:]

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
            img_eps = torch.cat([half_eps, half_eps], dim=0)
        else:
            eps = model_out[:, :self.in_channels]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            if cfg_interval is True:
                timestep = t[0]
                if timestep < cfg_interval_start:
                    half_eps = cond_eps

            img_eps = torch.cat([half_eps, half_eps], dim=0)

        # Apply CFG to hidden output
        cond_h, uncond_h = torch.split(h_out, len(h_out) // 2, dim=0)
        half_h = uncond_h + cfg_scale * (cond_h - uncond_h)
        h_eps = torch.cat([half_h, half_h], dim=0)

        return img_eps, h_eps

    def forward_with_autoguidance(self, x, t, y, cfg_scale, autoguidance_model,
                                  x_hidden=None, t_hidden=None,
                                  cfg_interval=False, cfg_interval_start=None,
                                  cfg_scale_sem=None, cfg_scale_tex=None):
        """
        Forward pass with AutoGuidance for both image and hidden tokens.
        """
        # Main model forward (conditional)
        cond_out, cond_h = self.forward(x, t, y, x_hidden=x_hidden, t_hidden=t_hidden)

        # Autoguidance model forward (serves as unconditional)
        uncond_out, uncond_h = autoguidance_model.forward(x, t, y, x_hidden=x_hidden, t_hidden=t_hidden)

        if self.use_semantic_first:
            texture_chans = self.in_channels - self.semantic_chans
            semantic_chans = self.semantic_chans

            cond_eps_tex, cond_eps_sem = cond_out[:, :texture_chans], cond_out[:, -semantic_chans:]
            uncond_eps_tex, uncond_eps_sem = uncond_out[:, :texture_chans], uncond_out[:, -semantic_chans:]

            effective_cfg_tex = cfg_scale_tex if cfg_scale_tex is not None else cfg_scale
            effective_cfg_sem = cfg_scale_sem if cfg_scale_sem is not None else cfg_scale

            half_eps_tex = uncond_eps_tex + effective_cfg_tex * (cond_eps_tex - uncond_eps_tex)
            half_eps_sem = uncond_eps_sem + effective_cfg_sem * (cond_eps_sem - uncond_eps_sem)

            if cfg_interval is True:
                t_tex, t_sem = t
                if self.semfirst_infer_interval_mode in ['both', 'tex'] and t_tex[0] < cfg_interval_start:
                    half_eps_tex = cond_eps_tex
                if self.semfirst_infer_interval_mode in ['both', 'sem'] and t_sem[0] < cfg_interval_start:
                    half_eps_sem = cond_eps_sem

            img_eps = torch.cat([half_eps_tex, half_eps_sem], dim=1)
        else:
            cond_eps = cond_out[:, :self.in_channels]
            uncond_eps = uncond_out[:, :self.in_channels]
            img_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            if cfg_interval is True:
                timestep = t[0]
                if timestep < cfg_interval_start:
                    img_eps = cond_eps

        # Apply CFG to hidden output
        h_eps = uncond_h + cfg_scale * (cond_h - uncond_h)

        return img_eps, h_eps


def HiddenLightningDiT_XL_1_H8(**kwargs):
    return HiddenLightningDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, num_hidden_tokens=8, **kwargs)

def HiddenLightningDiT_XL_1_H16(**kwargs):
    return HiddenLightningDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, num_hidden_tokens=16, **kwargs)

def HiddenLightningDiT_1p0B_H8(**kwargs):
    return HiddenLightningDiT(depth=24, hidden_size=1536, patch_size=1, num_heads=24, num_hidden_tokens=8, **kwargs)


HiddenLightningDiT_models = {
    "HiddenLightningDiT_XL_1_H8": HiddenLightningDiT_XL_1_H8,
    "HiddenLightningDiT_XL_1_H16": HiddenLightningDiT_XL_1_H16,
    "HiddenLightningDiT_1p0B_H8": HiddenLightningDiT_1p0B_H8,
}