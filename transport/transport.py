import torch as th
import numpy as np
import logging

import enum
import einops

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde
from scipy.stats import norm

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        use_cosine_loss=False,
        use_lognorm=False,
        partitial_train=None,
        partial_ratio=1.0,
        shift_lg=False,
        semantic_weight=1.0,
        semantic_chans=0,
        semfirst_delta_t=0.0,
        repa_weight=1.0,
        repa_mode='cos'
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.use_cosine_loss = use_cosine_loss
        self.use_lognorm = use_lognorm
        self.partitial_train = partitial_train
        self.partial_ratio = partial_ratio
        self.shift_lg = shift_lg
        self.semantic_weight = semantic_weight
        self.semantic_chans = semantic_chans
        self.semfirst_delta_t = semfirst_delta_t
        if semfirst_delta_t != 0:
            assert semfirst_delta_t > 0 and semfirst_delta_t < 1, 'semfirst_delta_t must be in (0, 1)'
            assert semantic_chans > 0, "semantic_chans must be greater than 0 if semfirst_delta_t is enabled"
        self.repa_weight = repa_weight
        assert repa_mode in ['cos', 'mse', 'cos_mse']
        self.repa_mode = repa_mode

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample_logit_normal(self, mu, sigma, size=1):
        # Generate samples from the normal distribution
        samples = norm.rvs(loc=mu, scale=sigma, size=size)
        
        # Transform samples to be in the range (0, 1) using the logistic function
        samples = 1 / (1 + np.exp(-samples))

        # Numpy to Tensor
        samples = th.tensor(samples, dtype=th.float32)

        return samples

    def sample_in_range(self, mu, sigma, target_size, range_min=0, range_max=0.5):
        samples = []
        while len(samples) < target_size:
            generated_samples = self.sample_logit_normal(mu, sigma, size=target_size)
            filtered_samples = generated_samples[(generated_samples >= range_min) & (generated_samples <= range_max)]
            samples.extend(filtered_samples)
        
        # If we have more than the target size, truncate the list
        samples = samples[:target_size]
        return th.tensor(samples)

    def sample(self, x1, sp_timesteps=None, shifted_mu=0):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        
        # # Use semantic first channelwise sampling if enabled
        # if self.semfirst_delta_t > 0 and self.semantic_chans > 0:
        #     return self.sample_channelwise(
        #         x1=x1,
        #         delta_t=self.semfirst_delta_t,
        #         t_max=1.0 + self.semfirst_delta_t,
        #         sp_timesteps=sp_timesteps,
        #         shifted_mu=shifted_mu,
        #         n_semantic=self.semantic_chans,
        #     )

        # Standard single timestep sampling
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if not self.use_lognorm:
            if self.partitial_train is not None and th.rand(1) < self.partial_ratio:
                t = th.rand((x1.shape[0],)) * (self.partitial_train[1] - self.partitial_train[0]) + self.partitial_train[0]
            else:
                t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        else:
            # random < partial_ratio, then sample from the partial range
            if not self.shift_lg:
                if self.partitial_train is not None and th.rand(1) < self.partial_ratio:
                    t = self.sample_in_range(0, 1, x1.shape[0], range_min=self.partitial_train[0], range_max=self.partitial_train[1])
                else:
                    t = self.sample_logit_normal(0, 1, size=x1.shape[0]) * (t1 - t0) + t0
            else:
                assert self.partitial_train is None, "Shifted lognormal distribution is not compatible with partial training"
                t = self.sample_logit_normal(shifted_mu, 1, size=x1.shape[0]) * (t1 - t0) + t0

        # overwrite t if sp_timesteps is provided (for validation)
        if sp_timesteps is not None:
            # uniform sampling between self.sp_timesteps[0] and self.sp_timesteps[1]
            t = th.rand((x1.shape[0],)) * (sp_timesteps[1] - sp_timesteps[0]) + sp_timesteps[0]

        t = t.to(x1)
        return t, x0, x1
    

    def training_losses(
        self, 
        model,  
        x1, 
        model_kwargs=None,
        sp_timesteps=None,
        shifted_mu=0,
        use_repa=False,
        feature_dino=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}
        
        t, x0, x1 = self.sample(x1, sp_timesteps, shifted_mu)
        if self.semfirst_delta_t > 0:
            t_sem = t * (1 + self.semfirst_delta_t)
            t_tex = t_sem - self.semfirst_delta_t
            t_sem = t_sem.clamp(max=1.0)
            t_tex = t_tex.clamp(min=0.0)
            x0_sem = x0[:, -self.semantic_chans:]
            x0_tex = x0[:, :-self.semantic_chans]
            x1_sem = x1[:, -self.semantic_chans:]
            x1_tex = x1[:, :-self.semantic_chans]
            t_sem, xt_sem, ut_sem = self.path_sampler.plan(t_sem, x0_sem, x1_sem)
            t_tex, xt_tex, ut_tex = self.path_sampler.plan(t_tex, x0_tex, x1_tex)
            t = (t_tex, t_sem)
            xt = th.cat([xt_tex, xt_sem], dim=1)
            ut = th.cat([ut_tex, ut_sem], dim=1)
        else:
            t, xt, ut = self.path_sampler.plan(t, x0, x1)

        # model_output = model(xt, model_t, **model_kwargs)
        if use_repa:
            assert feature_dino is not None
            model_output, repa_feat_proj = model(xt, t, **model_kwargs)
        else:
            model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}
        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            # terms['loss'] = mean_flat(((model_output - ut) ** 2))
            # Calculate loss with semantic weighting
            ch_num = ut.shape[1]
            loss_per_channel = (model_output - ut) ** 2  # Shape: [B, C, H, W]

            if self.semantic_chans > 0 and self.semantic_weight != 1.0:
                # Split into regular and semantic channels
                regular_chans = ch_num - self.semantic_chans
                regular_loss = loss_per_channel[:, :regular_chans]  # First N-semantic_chans channels
                semantic_loss = loss_per_channel[:, regular_chans:]  # Last semantic_chans channels

                # Apply semantic weight
                weighted_semantic_loss = semantic_loss * self.semantic_weight

                # Combine losses
                total_loss = th.cat([regular_loss, weighted_semantic_loss], dim=1)
                terms['loss'] = mean_flat(total_loss)
            else:
                # Default behavior when no semantic weighting
                terms['loss'] = mean_flat(loss_per_channel)
            if self.use_cosine_loss:
                terms['cos_loss'] = mean_flat(1 - th.nn.functional.cosine_similarity(model_output, ut, dim=1))
            if use_repa:
                # import pdb; pdb.set_trace()
                # check the size of feature_dino and repa_feat_proj, make sure the dimiension is correct
                feature_dino = einops.rearrange(feature_dino, 'b c h w -> b (h w) c')
                # terms['repa_loss'] = mean_flat((1 - th.nn.functional.cosine_similarity(feature_dino, repa_feat_proj, dim=-1)))
                # repa_loss = mean_flat((1 - th.nn.functional.cosine_similarity(feature_dino, repa_feat_proj, dim=-1)))
                # TODO: debug here
                if self.repa_mode == 'cos':
                    repa_loss = mean_flat((1 - th.nn.functional.cosine_similarity(feature_dino, repa_feat_proj, dim=-1)))
                elif self.repa_mode == 'mse':
                    repa_loss = mean_flat((feature_dino - repa_feat_proj) ** 2)
                elif self.repa_mode == 'cos_mse':
                    cos_loss = mean_flat(1 - th.nn.functional.cosine_similarity(feature_dino, repa_feat_proj, dim=-1))
                    mse_loss = mean_flat((feature_dino - repa_feat_proj) ** 2)
                    repa_loss = (cos_loss + mse_loss) / 2

                terms['repa_loss'] = repa_loss * self.repa_weight
                # visualize to see if correct
                # import torchvision, torch
                # feature_dino_vis = einops.rearrange(feature_dino, 'b (h w) c -> b c h w', h=16, w=16)[:16]
                # repa_feat_proj_vis = einops.rearrange(repa_feat_proj, 'b (h w) c -> b c h w', h=16, w=16)[:16]
                # feature_dino_vis = torch.nn.functional.interpolate(feature_dino_vis, size=(256, 256), mode='bilinear', align_corners=False)
                # repa_feat_proj_vis = torch.nn.functional.interpolate(repa_feat_proj_vis.detach().float(), size=(256, 256), mode='bilinear', align_corners=False)
                # torchvision.utils.save_image(feature_dino_vis[:,:3], 'delete_feature_dino.png', normalize=True)
                # torchvision.utils.save_image(repa_feat_proj_vis[:,:3], 'delete_repa_feat_proj.png', normalize=True)
        else: 
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
                
        return terms

    def training_losses_hidden(
        self,
        model,
        x1,
        model_kwargs=None,
        sp_timesteps=None,
        shifted_mu=0,
        use_repa=False,
        feature_dino=None,
        hidden_weight=1.0,
        normalize_hidden=True,
        hidden_reg_weight=0.01,
    ):
        """
        Self-encoder training loss with hidden tokens.

        Three forward passes:
          Pass 1: Clean image + pure noise hidden -> predict encoding (single-step)
          Pass 2: Noisy image + noisy encoding (grad flows through encoding) -> image denoising loss
          Pass 3: Noisy image + noisy DETACHED encoding -> hidden denoising loss

        Args:
            model: HiddenLightningDiT model (possibly wrapped in DDP)
            x1: clean data (B, C, H, W)
            model_kwargs: dict with 'y' (class labels)
            hidden_weight: weight for hidden denoising loss (pass 3)
            normalize_hidden: if True, project h_clean tokens onto unit sphere to
                prevent variance explosion (default True)
            hidden_reg_weight: weight for regularization loss that penalizes hidden
                token norms deviating from 1. Set to 0 to disable. (default 0.01)
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Get model's hidden token config (works with DDP .module)
        m = model.module if hasattr(model, 'module') else model
        num_hidden_tokens = m.num_hidden_tokens
        hidden_token_dim = m.hidden_token_dim
        B = x1.shape[0]
        device = x1.device

        # ============ PASS 1: Encode ============
        # Convention: x_t = t * x1 + (1-t) * x0, where x0=noise, x1=clean data
        #   t=0 → pure noise, t=1 → clean data
        # We pass the clean image (t_img=1) with pure noise hidden tokens (t_hid=0).
        # The model predicts velocity v = x1 - x0 for hidden, so:
        #   h_clean = x0_h + v_h  (recover clean encoding from noise + predicted velocity)
        x0_h = th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)
        t_encode_img = th.ones(B, device=device)    # t=1: image is clean data
        t_encode_hid = th.zeros(B, device=device)   # t=0: hidden is pure noise

        # Forward pass 1: model sees clean image (t=1) + pure noise hidden (t=0)
        # Model predicts velocity for hidden: v_h = x1_h - x0_h
        # Recover clean encoding: h_clean = x0_h + v_h
        if self.semfirst_delta_t > 0:
            t_encode_img_for_model = (t_encode_img, t_encode_img)  # both tex and sem at t=1
        else:
            t_encode_img_for_model = t_encode_img

        _, h_velocity = model(x1, t=t_encode_img_for_model, x_hidden=x0_h,
                              t_hidden=t_encode_hid, **model_kwargs)
        # Reconstruct clean encoding from velocity prediction
        h_clean = x0_h + h_velocity  # x1 = x0 + v

        # Regularization: penalize hidden token norms deviating from unit sphere
        if hidden_reg_weight > 0:
            h_radius = h_clean.norm(dim=-1)  # (B, num_hidden_tokens)
            hidden_reg_loss = (h_radius - 1).pow(2).mean()
        
        # Project each hidden token onto the unit sphere to prevent variance explosion
        if normalize_hidden:
            h_clean = h_clean / h_clean.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # ============ PASS 2: Image denoising conditioned on encoding ============
        # Sample timesteps and noise for image
        t_img, x0_img, x1_img = self.sample(x1, sp_timesteps, shifted_mu)

        # Handle semantic first for image timesteps
        if self.semfirst_delta_t > 0:
            t_sem = t_img * (1 + self.semfirst_delta_t)
            t_tex = t_sem - self.semfirst_delta_t
            t_sem = t_sem.clamp(max=1.0)
            t_tex = t_tex.clamp(min=0.0)
            x0_sem = x0_img[:, -self.semantic_chans:]
            x0_tex = x0_img[:, :-self.semantic_chans]
            x1_sem = x1_img[:, -self.semantic_chans:]
            x1_tex = x1_img[:, :-self.semantic_chans]
            t_sem, xt_sem, ut_sem = self.path_sampler.plan(t_sem, x0_sem, x1_sem)
            t_tex, xt_tex, ut_tex = self.path_sampler.plan(t_tex, x0_tex, x1_tex)
            t_img_for_model = (t_tex, t_sem)
            xt_img = th.cat([xt_tex, xt_sem], dim=1)
            ut_img = th.cat([ut_tex, ut_sem], dim=1)
        else:
            t_img, xt_img, ut_img = self.path_sampler.plan(t_img, x0_img, x1_img)
            t_img_for_model = t_img

        # Sample timestep and noise for hidden tokens
        t0_h, t1_h = self.check_interval(self.train_eps, self.sample_eps)
        t_h = th.rand(B, device=device) * (t1_h - t0_h) + t0_h
        x0_h2 = th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)

        # Noise the encoding (gradient flows through h_clean from Pass 1!)
        t_h_expanded = t_h.view(B, 1, 1)
        xt_h = t_h_expanded * h_clean + (1 - t_h_expanded) * x0_h2  # linear interpolation
        ut_h = h_clean - x0_h2  # velocity target for linear path

        # Forward pass 2: predict image and hidden velocities
        if use_repa:
            assert feature_dino is not None
            img_pred, h_pred, repa_feat_proj = model(xt_img, t=t_img_for_model, x_hidden=xt_h,
                                                     t_hidden=t_h, **model_kwargs)
        else:
            img_pred, h_pred = model(xt_img, t=t_img_for_model, x_hidden=xt_h,
                                     t_hidden=t_h, **model_kwargs)

        # Image denoising loss (velocity matching)
        loss_per_channel = (img_pred - ut_img) ** 2
        if self.semantic_chans > 0 and self.semantic_weight != 1.0:
            ch_num = ut_img.shape[1]
            regular_chans = ch_num - self.semantic_chans
            regular_loss = loss_per_channel[:, :regular_chans]
            semantic_loss = loss_per_channel[:, regular_chans:] * self.semantic_weight
            total_img_loss = th.cat([regular_loss, semantic_loss], dim=1)
            img_loss = mean_flat(total_img_loss)
        else:
            img_loss = mean_flat(loss_per_channel)

        terms = {}
        terms['loss'] = img_loss  # image denoising loss (gradient flows through h_clean -> Pass 1)
        terms['pred'] = img_pred

        if self.use_cosine_loss:
            terms['cos_loss'] = mean_flat(1 - th.nn.functional.cosine_similarity(img_pred, ut_img, dim=1))

        if use_repa:
            feature_dino = einops.rearrange(feature_dino, 'b c h w -> b (h w) c')
            if self.repa_mode == 'cos':
                repa_loss = mean_flat((1 - th.nn.functional.cosine_similarity(feature_dino, repa_feat_proj, dim=-1)))
            elif self.repa_mode == 'mse':
                repa_loss = mean_flat((feature_dino - repa_feat_proj) ** 2)
            elif self.repa_mode == 'cos_mse':
                cos_loss = mean_flat(1 - th.nn.functional.cosine_similarity(feature_dino, repa_feat_proj, dim=-1))
                mse_loss = mean_flat((feature_dino - repa_feat_proj) ** 2)
                repa_loss = (cos_loss + mse_loss) / 2
            terms['repa_loss'] = repa_loss * self.repa_weight

        # Hidden token regularization loss
        if hidden_reg_weight > 0:
            terms['hidden_reg_loss'] = hidden_reg_loss * hidden_reg_weight

        # ============ PASS 3: Hidden denoising (detached encoding) ============
        h_clean_detached = h_clean.detach()

        # Sample new timestep for hidden
        t_h3 = th.rand(B, device=device) * (t1_h - t0_h) + t0_h
        x0_h3 = th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)

        t_h3_expanded = t_h3.view(B, 1, 1)
        xt_h3 = t_h3_expanded * h_clean_detached + (1 - t_h3_expanded) * x0_h3
        ut_h3 = h_clean_detached - x0_h3

        # Forward pass 3: reuse the same noisy image from pass 2 (detached from img path)
        xt_img_detached = xt_img.detach()
        # Detach t_img_for_model if it's a tuple
        if isinstance(t_img_for_model, tuple):
            t_img_detached = tuple(t.detach() for t in t_img_for_model)
        else:
            t_img_detached = t_img_for_model.detach()

        _, h_pred3 = model(xt_img_detached, t=t_img_detached, x_hidden=xt_h3,
                           t_hidden=t_h3, **model_kwargs)

        # Hidden denoising loss
        hidden_loss = mean_flat((h_pred3 - ut_h3) ** 2)
        terms['hidden_loss'] = hidden_loss * hidden_weight

        return terms
    

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_drift_semantic_first(
        self,
        semfirst_delta_t=0.3,
        semantic_chans=8,
    ):
        """
        Member function for obtaining the drift of the probability flow ODE with semantic first handling

        Args:
            semfirst_delta_t: time difference (semantic leads texture)
            semantic_chans: number of semantic channels (at the end)
        """
        def velocity_ode_semantic_first(x, t, model, **model_kwargs):
            """
            Semantic first drift function using tuple timestep interface
            """
            B, C = x.shape[:2]
            device = x.device

            # Calculate t_sem and t_tex based on global time t
            t_sem = t.clamp(max=1.0)  # Semantic time clamped to [0, 1]
            t_tex = (t - semfirst_delta_t).clamp(min=0.0)  # Texture time with delay

            # Create tuple for model (matches training interface)
            t_tuple = (t_tex, t_sem)

            # Get model output with tuple timesteps
            model_output = model(x, t_tuple, **model_kwargs)
            return model_output

        if self.model_type == ModelType.VELOCITY:
            drift_fn = velocity_ode_semantic_first
        else:
            # For other model types, would need to implement similar adaptations
            raise NotImplementedError("Semantic first currently only supports VELOCITY model type")

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn
    

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=0.0,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            timestep_shift=timestep_shift,
        )
        
        return _ode.sample

    def sample_ode_hidden(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=0.0,
        num_hidden_tokens=8,
        hidden_token_dim=32,
    ):
        """
        Returns a sampling function that jointly generates image and hidden tokens via ODE.
        Uses tuple state (x_img, x_hidden) compatible with torchdiffeq.

        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: number of integration steps
        - atol: absolute error tolerance
        - rtol: relative error tolerance
        - reverse: whether to reverse time
        - timestep_shift: timestep shift parameter
        - num_hidden_tokens: number of hidden tokens per sample
        - hidden_token_dim: dimension of each hidden token
        """

        def joint_drift(state, t, model, **model_kwargs):
            """
            Joint drift for image + hidden tokens.
            state: tuple (x_img, x_hidden)
              x_img: (B, C, H, W)
              x_hidden: (B, num_hidden_tokens, hidden_token_dim)
            t: (B,) scalar timestep (same for image and hidden)
            Returns: tuple (img_velocity, hidden_velocity)
            """
            x_img, x_hidden = state

            # Model returns (img_velocity, hidden_velocity) — works with
            # forward, forward_with_cfg, and forward_with_autoguidance
            # because they all accept x_hidden/t_hidden as kwargs
            result = model(x_img, t=t, x_hidden=x_hidden, t_hidden=t,
                           **model_kwargs)
            if isinstance(result, tuple):
                img_vel, hidden_vel = result[0], result[1]
            else:
                raise ValueError("Model must return (img_output, hidden_output) tuple for hidden token sampling")

            return (img_vel, hidden_vel)

        if reverse:
            drift = lambda state, t, model, **kwargs: joint_drift(
                state, th.ones_like(t) * (1 - t), model, **kwargs
            )
        else:
            drift = joint_drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            timestep_shift=timestep_shift,
        )

        def _sample_fn(z_img, z_hidden, model, **model_kwargs):
            """
            Sample image + hidden tokens jointly via ODE.
            Args:
                z_img: (B, C, H, W) initial noise for image
                z_hidden: (B, num_hidden_tokens, hidden_token_dim) initial noise for hidden tokens
                model: model forward function
            Returns:
                (img_samples, hidden_samples) tuple at t=1
            """
            init_state = (z_img, z_hidden)
            samples = _ode.sample(init_state, model, **model_kwargs)
            # odeint returns samples at each timepoint; last one is at t=1
            if isinstance(samples, tuple):
                return samples[0][-1], samples[1][-1]
            else:
                return samples[-1]

        return _sample_fn

    def sample_ode_semantic_first(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=0.0,
        semfirst_delta_t=0.3,
        semantic_chans=8,
    ):
        """
        Returns a sampling function with semantic first ODE settings.

        Inference strategy:
        - Phase 1: t ∈ [0, delta_t] → only semantic channels update (texture frozen)
        - Phase 2: t ∈ [delta_t, 1.0] → both semantic and texture channels update
        - Phase 3: t ∈ [1.0, 1.0+delta_t] → only texture channels update (semantic frozen)
        - Total denoising time: 1.0 + delta_t

        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: number of integration steps
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        - timestep_shift: timestep shift parameter
        - semfirst_delta_t: time difference between semantic and texture channels
        - semantic_chans: number of semantic channels (located at the end)
        """
        t_max = 1.0 + semfirst_delta_t
        texture_chans = None  # Will be set in drift function

        # Get the semantic first drift function
        drift_semantic_first = self.transport.get_drift_semantic_first(
            semfirst_delta_t=semfirst_delta_t,
            semantic_chans=semantic_chans
        )

        def semantic_first_drift_with_masking(x, t, model, **model_kwargs):
            """
            Semantic first drift function that handles selective updating of channels
            """
            nonlocal texture_chans
            B, C = x.shape[:2]
            device = x.device

            if texture_chans is None:
                texture_chans = C - semantic_chans

            # Get the full drift from the model using semantic first method
            full_drift = drift_semantic_first(x, t, model, **model_kwargs)

            # Apply channel masking based on current time
            # Phase 1: t ∈ [0, delta_t] → only semantic updates (last semantic_chans channels)
            # Phase 2: t ∈ [delta_t, 1.0] → both update
            # Phase 3: t ∈ [1.0, 1.0+delta_t] → only texture updates (first texture_chans channels)

            mask = th.zeros_like(x)
            # Texture mask: active when t >= delta_t (first texture_chans channels)
            mask[:, :texture_chans, ...] = (t >= semfirst_delta_t).view(B, 1, *[1] * (x.dim() - 2))
            # Semantic mask: active when t <= 1.0 (last semantic_chans channels)
            mask[:, -semantic_chans:, ...] = (t <= 1.0).view(B, 1, *[1] * (x.dim() - 2))

            return full_drift * mask

        # Set up time interval for the entire process
        t0 = 0.0
        t1 = t_max  # Total time is 1.0 + delta_t

        if reverse:
            # For reverse sampling, would need to adapt the logic
            raise NotImplementedError("Reverse sampling not yet implemented for semantic first")

        _ode = ode(
            drift=semantic_first_drift_with_masking,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            timestep_shift=timestep_shift,
            semfirst_delta_t=semfirst_delta_t,
        )

        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn