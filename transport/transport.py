import torch as th
import numpy as np
import logging
from jaxtyping import Float 
from torch import Tensor 

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

    def grad_scale(self, x: Float[Tensor, "b c h w"], scale: Float[Tensor, "b c h w"] | float) -> Float[Tensor, "b c h w"]:
        return x * scale - (scale - 1) * x.detach()
    
    def dyn_grad_scale(self, h_clean, t_h, hidden_grad_dyn_scale, eps=1e-4):
        inverted_scale = (1/(hidden_grad_dyn_scale + eps))

        weight_t = (1+inverted_scale) / (t_h + inverted_scale)
        return self.grad_scale(h_clean, weight_t.view(-1, 1, 1))
    

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
        hidden_cos_weight=0.0,
        backward_fn=None,
        hidden_same_t_as_img=False,
        noisy_img_encode=False,
        hidden_t_shift=0.0,
        hidden_loss_scale=1.0,
        hidden_grad_dyn_scale=0.0,
        hidden_grad_static_scale=1.0,
        use_encode_mode_emb=False,
        hidden_guidance_scale=1.0,
        hidden_reuse_noise_pass2=False,
        hidden_reuse_noise_pass3=False,
        hidden_clean_only_pass2=False,
        hidden_dropout_prob=0.0,
        sync_class_dropout=False,
        encoder_model=None,
    ):
        """
        Self-encoder training loss with hidden tokens (3-pass variant).

        Three forward passes, structured for memory efficiency:
          Pass 1  (encode):  Image + pure-noise hidden x0_h → h_clean
          Pass 3  (hidden):  Noisy image + noisy DETACHED h_clean on same x0_h trajectory → hidden loss
                             (Run before Pass 2; if backward_fn is provided,
                              activations freed immediately via backward)
          Pass 2  (image):   Noisy image + noisy h_clean on same x0_h trajectory (grad flows) → image loss

        The encoder image in Pass 1 is clean (t=1) by default.  When
        ``noisy_img_encode=True``, two timesteps are sampled from the training
        distribution; the higher one is used for the encoder (Pass 1, cleaner
        image) and the lower one for the denoiser (Passes 2/3, noisier image).
        Both share the same Gaussian noise, staying on the same interpolation
        trajectory.  This keeps the model's internal features in-distribution
        (logit-normal sampling puts near-zero density at t=1).

        When backward_fn is provided, Pass 3's loss is backward-ed right after
        computation and before Pass 2 runs.  This reduces peak activation
        memory from ~3× to ~2× a single forward pass.

        Args:
            model: HiddenLightningDiT model (possibly wrapped in DDP).
            x1: clean data (B, C, H, W).
            model_kwargs: dict with 'y' (class labels).
            sp_timesteps: optional (t_lo, t_hi) for uniform timestep override.
            shifted_mu: mean shift for logit-normal sampling.
            use_repa: if True, compute REPA representation-alignment loss.
            feature_dino: DINOv2 features (required when use_repa=True).
            hidden_weight: weight for hidden denoising MSE loss (default 1.0).
            normalize_hidden: if True, project h_clean onto unit sphere to
                prevent variance explosion (default True).
            hidden_reg_weight: weight for sphere-regularisation loss
                (penalises ||h|| deviating from 1). 0 to disable (default 0.01).
            hidden_cos_weight: weight for cosine similarity loss between the
                single-step predicted clean hidden tokens and the target
                encodings. 0 to disable (default 0.0).
            backward_fn: callable(loss) that runs backward on a loss tensor.
                Should wrap with model.no_sync() under DDP to defer gradient
                sync.  If None, all losses returned for a single backward.
            hidden_same_t_as_img: if True, hidden noise level equals t_img
                rather than independent sampling.  Matches the "linear"
                inference schedule (default False).
            noisy_img_encode: if True, the encoder sees a noisy image at
                t = max(t_a, t_b) sampled from the logit-normal training
                distribution instead of a clean image at t=1.  The denoiser
                sees t = min(t_a, t_b). Both use the same noise (default False).
            hidden_t_shift: logit-normal mu for hidden timestep sampling.
                Positive values bias t_h toward 1 (cleaner hidden tokens).
                Use with a curriculum that decays from a large value to a
                small permanent bias (default 0.0 = uniform).
            hidden_loss_scale: multiplier for Pass 3 inclusion.  In the
                3-pass variant this is used as a Bernoulli probability:
                Pass 3 is run with probability hidden_loss_scale and skipped
                otherwise.  1.0 = always run (default).
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Get model's hidden token config (unwrap DDP / Accelerate wrappers)
        m = model
        while hasattr(m, 'module'):
            m = m.module
        num_hidden_tokens = m.num_hidden_tokens
        hidden_token_dim = m.hidden_token_dim
        B = x1.shape[0]
        device = x1.device
        
        if hidden_dropout_prob > 0.0:
            drop_mask = (th.rand(B, device=device) < hidden_dropout_prob).view(B)
        else:
            drop_mask = None

        force_drop_ids = None
        if sync_class_dropout and ("y" in model_kwargs):
            y_embedder = getattr(m, "y_embedder", None)
            p = getattr(y_embedder, "dropout_prob", 0.0) if y_embedder is not None else 0.0
            if p > 0:
                force_drop_ids = (th.rand(B, device=device) < p).to(th.int64)

        # ============ PASS 1: Encode ============
        # Convention: x_t = t * x1 + (1-t) * x0, where x0=noise, x1=clean data
        #   t=0 → pure noise, t=1 → clean data
        # We pass the clean image (t_img=1) with pure noise hidden tokens (t_hid=0).
        # The model predicts velocity v = x1 - x0 for hidden, so:
        #   h_clean = x0_h + v_h  (recover clean encoding from noise + predicted velocity)
        x0_h = th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)
        t_encode_hid = th.zeros(B, device=device)   # t=0: hidden is pure noise

        if noisy_img_encode:
            # Sample two timesteps from training distribution; assign
            # max -> encoder (cleaner image), min -> denoiser (noisier image).
            # Shared noise keeps both on the same interpolation trajectory.
            t_a, _, _ = self.sample(x1, sp_timesteps, shifted_mu)
            t_b, _, _ = self.sample(x1, sp_timesteps, shifted_mu)
            t_encode_raw = th.max(t_a, t_b)
            t_denoise_raw = th.min(t_a, t_b)
            x0_img = th.randn_like(x1)

            if self.semfirst_delta_t > 0:
                x0_sem = x0_img[:, -self.semantic_chans:]
                x0_tex = x0_img[:, :-self.semantic_chans]
                x1_sem = x1[:, -self.semantic_chans:]
                x1_tex = x1[:, :-self.semantic_chans]
                # Encoder image at t_encode_raw
                t_sem_enc = (t_encode_raw * (1 + self.semfirst_delta_t)).clamp(max=1.0)
                t_tex_enc = (t_sem_enc - self.semfirst_delta_t).clamp(min=0.0)
                _, xt_sem_enc, _ = self.path_sampler.plan(t_sem_enc, x0_sem, x1_sem)
                _, xt_tex_enc, _ = self.path_sampler.plan(t_tex_enc, x0_tex, x1_tex)
                xt_encode = th.cat([xt_tex_enc, xt_sem_enc], dim=1)
                t_encode_img_for_model = (t_tex_enc, t_sem_enc)
                # Denoiser image at t_denoise_raw
                t_sem_den = (t_denoise_raw * (1 + self.semfirst_delta_t)).clamp(max=1.0)
                t_tex_den = (t_sem_den - self.semfirst_delta_t).clamp(min=0.0)
                t_sem_den, xt_sem, ut_sem = self.path_sampler.plan(t_sem_den, x0_sem, x1_sem)
                t_tex_den, xt_tex, ut_tex = self.path_sampler.plan(t_tex_den, x0_tex, x1_tex)
                t_img_for_model = (t_tex_den, t_sem_den)
                xt_img = th.cat([xt_tex, xt_sem], dim=1)
                ut_img = th.cat([ut_tex, ut_sem], dim=1)
            else:
                _, xt_encode, _ = self.path_sampler.plan(t_encode_raw, x0_img, x1)
                t_encode_img_for_model = t_encode_raw
                t_denoise_raw, xt_img, ut_img = self.path_sampler.plan(t_denoise_raw, x0_img, x1)
                t_img_for_model = t_denoise_raw
            t_img = t_denoise_raw  # for hidden_same_t_as_img
        else:
            t_encode_img = th.ones(B, device=device)    # t=1: image is clean data
            if self.semfirst_delta_t > 0:
                t_encode_img_for_model = (t_encode_img, t_encode_img)
            else:
                t_encode_img_for_model = t_encode_img
            xt_encode = x1  # clean image

        model_for_encode = encoder_model if encoder_model is not None else m
        pass1_out = model_for_encode(
            xt_encode,
            t=t_encode_img_for_model,
            x_hidden=x0_h,
            t_hidden=t_encode_hid,
            force_drop_ids=force_drop_ids,
            encode_mode=use_encode_mode_emb,
            **model_kwargs,
        )
        h_velocity = pass1_out[1]
        h_clean = x0_h + h_velocity

        # Regularization: penalize hidden token norms deviating from unit sphere
        if hidden_reg_weight > 0:
            h_radius = h_clean.norm(dim=-1)  # (B, num_hidden_tokens)
            hidden_reg_loss = (h_radius - 1).pow(2).mean()
        
        # Project each hidden token onto the unit sphere to prevent variance explosion
        if normalize_hidden:
            h_clean = h_clean / h_clean.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # ============ SAMPLE IMAGE NOISE ============
        if not noisy_img_encode:
            t_img, x0_img, x1_img = self.sample(x1, sp_timesteps, shifted_mu)

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

        # ============ PASS 3: Hidden denoising (detached, run BEFORE Pass 2) ============
        # By running Pass 3 before Pass 2, we can backward it immediately (via
        # backward_fn) and free its activation graph. This way, at most 2 forward
        # pass activation graphs coexist at any time (Pass 1 + Pass 3, then
        # Pass 1 + Pass 2), giving ~2x peak memory instead of ~3x.
        h_clean_detached = h_clean.detach()
        t0_h, t1_h = self.check_interval(self.train_eps, self.sample_eps)

        # Stochastic Pass 3 gating: run with probability hidden_loss_scale.
        # When skipped, hidden_loss is zero (no backward, saving compute).
        run_pass3 = (hidden_loss_scale >= 1.0) or (th.rand(1).item() < hidden_loss_scale)

        if run_pass3:
            if hidden_clean_only_pass2:
                t_h3 = th.ones(B, device=device) * t1_h
            elif hidden_same_t_as_img:
                t_h3 = t_img.clone().clamp(t0_h, t1_h)
            elif hidden_t_shift != 0:
                t_h3 = self.sample_logit_normal(hidden_t_shift, 1, size=B).to(device)
                t_h3 = t_h3 * (t1_h - t0_h) + t0_h
            else:
                t_h3 = th.rand(B, device=device) * (t1_h - t0_h) + t0_h
            
            if drop_mask is not None:
                t_h3 = th.where(drop_mask, th.zeros_like(t_h3) + t0_h, t_h3)
            
            x0_h3 = x0_h if hidden_reuse_noise_pass3 else th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)
            t_h3_expanded = t_h3.view(B, 1, 1)
            xt_h3 = t_h3_expanded * h_clean_detached + (1 - t_h3_expanded) * x0_h3
            ut_h3 = h_clean_detached - x0_h3

            # Detach image inputs for Pass 3 (no grad flow to image path)
            xt_img_detached = xt_img.detach()
            if isinstance(t_img_for_model, tuple):
                t_img_detached = tuple(t.detach() for t in t_img_for_model)
            else:
                t_img_detached = t_img_for_model.detach()

            # Use unwrapped model for Pass 3 (avoids multiple DDP forwards;
            # backward_fn uses no_sync to defer gradient allreduce to Pass 2)
            pass3_out = m(
                xt_img_detached,
                t=t_img_detached,
                x_hidden=xt_h3,
                t_hidden=t_h3,
                force_drop_ids=force_drop_ids,
                **model_kwargs,
            )
            h_pred3 = pass3_out[1]

            hidden_loss_unreduced = (h_pred3 - ut_h3) ** 2
            if drop_mask is not None:
                hidden_loss_unreduced = hidden_loss_unreduced * (~drop_mask).view(B, 1, 1).float()
            hidden_loss = mean_flat(hidden_loss_unreduced)

            if hidden_cos_weight > 0:
                h1_pred3 = xt_h3 + (1 - t_h3_expanded) * h_pred3
                hidden_cos_loss_unreduced = 1 - th.nn.functional.cosine_similarity(h1_pred3, h_clean_detached, dim=-1)
                if drop_mask is not None:
                    hidden_cos_loss_unreduced = hidden_cos_loss_unreduced * (~drop_mask).view(B, 1).float()
                hidden_cos_loss = mean_flat(hidden_cos_loss_unreduced)
            else:
                hidden_cos_loss = None

            # If backward_fn provided, backward Pass 3 immediately to free activations
            if backward_fn is not None:
                pass3_loss = hidden_loss.mean() * hidden_weight
                if hidden_cos_loss is not None:
                    pass3_loss = pass3_loss + hidden_cos_loss.mean() * hidden_cos_weight
                backward_fn(pass3_loss)
                hidden_loss_for_terms = hidden_loss.detach()
                hidden_cos_loss_for_terms = hidden_cos_loss.detach() if hidden_cos_loss is not None else None
            else:
                hidden_loss_for_terms = hidden_loss * hidden_weight
                hidden_cos_loss_for_terms = hidden_cos_loss * hidden_cos_weight if hidden_cos_loss is not None else None
        else:
            # Pass 3 skipped — report zero hidden loss (detached, no backward needed)
            hidden_loss_for_terms = th.zeros(B, device=device)
            hidden_cos_loss_for_terms = None

        # ============ PASS 2: Image denoising conditioned on encoding ============
        if hidden_clean_only_pass2:
            t_h = th.ones(B, device=device) * t1_h
        elif hidden_same_t_as_img:
            t_h = t_img.clone().clamp(t0_h, t1_h)
        elif hidden_t_shift != 0:
            t_h = self.sample_logit_normal(hidden_t_shift, 1, size=B).to(device)
            t_h = t_h * (t1_h - t0_h) + t0_h
        else:
            t_h = th.rand(B, device=device) * (t1_h - t0_h) + t0_h
            
        if drop_mask is not None:
            t_h = th.where(drop_mask, th.zeros_like(t_h) + t0_h, t_h)
            
        if hidden_grad_dyn_scale > 0.0:
            h_clean = self.dyn_grad_scale(h_clean, t_h, hidden_grad_dyn_scale)

        if hidden_grad_static_scale != 1.0:
            h_clean = self.grad_scale(h_clean, hidden_grad_static_scale)

        x0_h2 = x0_h if hidden_reuse_noise_pass2 else th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)

        # Noise the encoding (gradient flows through h_clean from Pass 1!)
        t_h_expanded = t_h.view(B, 1, 1)
        xt_h = t_h_expanded * h_clean + (1 - t_h_expanded) * x0_h2
        ut_h = h_clean - x0_h2

        # Forward pass 2: predict image and hidden velocities
        if use_repa:
            assert feature_dino is not None
            img_pred, h_pred, repa_feat_proj = model(
                xt_img,
                t=t_img_for_model,
                x_hidden=xt_h,
                t_hidden=t_h,
                force_drop_ids=force_drop_ids,
                **model_kwargs,
            )
        else:
            img_pred, h_pred = model(
                xt_img,
                t=t_img_for_model,
                x_hidden=xt_h,
                t_hidden=t_h,
                force_drop_ids=force_drop_ids,
                **model_kwargs,
            )

        # ============ Hidden-conditioning guidance (amplified target) ============
        # When hidden_guidance_scale > 1, construct a target that overshoots v_gt
        # away from the unconditional prediction, so the model internalises guidance.
        #   v_uncond = sg(model(xt_img, h_noise, t_h=0))
        #   v_target = v_uncond + w(t_h) * (v_gt - v_uncond)
        # w(t_h) ramps linearly from 1.0 (at t_h=0, no amplification) to
        # hidden_guidance_scale (at t_h=1, full amplification).
        if hidden_guidance_scale > 1.0:
            h_uncond = th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)
            t_h_uncond = th.zeros(B, device=device)
            with th.no_grad():
                v_uncond_img = m(
                    xt_img.detach(),
                    t=t_img_for_model if not isinstance(t_img_for_model, tuple)
                    else tuple(t.detach() for t in t_img_for_model),
                    x_hidden=h_uncond,
                    t_hidden=t_h_uncond,
                    force_drop_ids=force_drop_ids,
                    **model_kwargs,
                )[0]
            w_guidance = 1.0 + (hidden_guidance_scale - 1.0) * t_h.view(B, 1, 1, 1)
            ut_img_guided = v_uncond_img + w_guidance * (ut_img - v_uncond_img)
        else:
            ut_img_guided = ut_img

        # Image denoising loss (velocity matching)
        loss_per_channel = (img_pred - ut_img_guided) ** 2
        if self.semantic_chans > 0 and self.semantic_weight != 1.0:
            ch_num = ut_img_guided.shape[1]
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
        terms['hidden_loss'] = hidden_loss_for_terms
        if hidden_cos_loss_for_terms is not None:
            terms['hidden_cos_loss'] = hidden_cos_loss_for_terms

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

        return terms

    def training_losses_hidden_merged(
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
        hidden_cos_weight=0.0,
        hidden_same_t_as_img=False,
        noisy_img_encode=False,
        hidden_t_shift=0.0,
        hidden_loss_scale=1.0,
        hidden_grad_dyn_scale=0.0,
        hidden_grad_static_scale=1.0,
        use_encode_mode_emb=False,
        hidden_guidance_scale=1.0,
        hidden_reuse_noise_pass2=False,
        hidden_reuse_noise_pass3=False,
        sync_class_dropout=False,
    ):
        """
        Two-pass self-encoder training with merged image + hidden denoising.

        Unlike ``training_losses_hidden`` (3 passes), this variant lets the
        hidden denoising loss backpropagate through the noised hidden *input*
        all the way back to Pass 1 (the encoder).  Because gradients now flow
        from both the image loss and the hidden loss into the encoder, Passes 2
        and 3 can be merged into a single forward pass:

          Pass 1  (encode):  Image + pure-noise hidden x0_h → h_clean
          Pass 2  (denoise): Noisy image + noisy h_clean on same x0_h trajectory → (img_pred, h_pred)
                             image loss   = MSE(img_pred, ut_img)
                             hidden loss  = MSE(h_pred, ut_h_detached)

        The hidden velocity *target* ``ut_h`` is detached so that MSE does not
        push directly on the encoding; supervision reaches the encoder only
        through the *predicted* velocity ``h_pred`` and the noised input ``xt_h``.

        When ``noisy_img_encode=True``, the encoder in Pass 1 sees a noisy
        image at t = max(t_a, t_b) (sampled from the logit-normal training
        distribution) instead of a clean image at t=1, and the denoiser in
        Pass 2 uses t = min(t_a, t_b).  Both share the same noise.  This keeps
        the model's internal features in-distribution.

        This is simpler (2 forward passes, single backward) and ~33 % less
        peak activation memory than the 3-pass variant with ``backward_fn``.

        Args:
            model: HiddenLightningDiT model (possibly wrapped in DDP).
            x1: clean data (B, C, H, W).
            model_kwargs: dict with 'y' (class labels).
            sp_timesteps: optional (t_lo, t_hi) for uniform timestep override.
            shifted_mu: mean shift for logit-normal sampling.
            use_repa: if True, compute REPA representation-alignment loss.
            feature_dino: DINOv2 features (required when use_repa=True).
            hidden_weight: weight for hidden denoising MSE loss (default 1.0).
            normalize_hidden: if True, project h_clean onto unit sphere
                (default True).
            hidden_reg_weight: weight for sphere-regularisation loss
                (default 0.01; 0 to disable).
            hidden_cos_weight: weight for cosine similarity loss on
                single-step clean hidden prediction (default 0.0; 0 to disable).
            hidden_same_t_as_img: if True, hidden noise level equals t_img
                rather than independent sampling (default False).
            noisy_img_encode: if True, the encoder sees a noisy image at
                t = max(t_a, t_b) from the logit-normal distribution instead
                of a clean image at t=1 (default False).
            hidden_t_shift: logit-normal mu for hidden timestep sampling.
                Positive values bias t_h toward 1 (cleaner hidden tokens).
                Use with a curriculum that decays from a large value to a
                small permanent bias (default 0.0 = uniform).
            hidden_loss_scale: multiplier for the hidden denoising loss.
                In the merged variant this directly scales hidden_loss.
                Ramp from 0 to 1 over training to delay hidden denoising
                pressure (default 1.0).
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Unwrap DDP / Accelerate wrappers
        m = model
        while hasattr(m, 'module'):
            m = m.module
        num_hidden_tokens = m.num_hidden_tokens
        hidden_token_dim = m.hidden_token_dim
        B = x1.shape[0]
        device = x1.device

        force_drop_ids = None
        if sync_class_dropout and ("y" in model_kwargs):
            y_embedder = getattr(m, "y_embedder", None)
            p = getattr(y_embedder, "dropout_prob", 0.0) if y_embedder is not None else 0.0
            if p > 0:
                force_drop_ids = (th.rand(B, device=device) < p).to(th.int64)

        # ============ PASS 1: Encode ============
        x0_h = th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)
        t_encode_hid = th.zeros(B, device=device)

        if noisy_img_encode:
            # Sample two timesteps; max -> encoder (cleaner), min -> denoiser
            t_a, _, _ = self.sample(x1, sp_timesteps, shifted_mu)
            t_b, _, _ = self.sample(x1, sp_timesteps, shifted_mu)
            t_encode_raw = th.max(t_a, t_b)
            t_denoise_raw = th.min(t_a, t_b)
            x0_img = th.randn_like(x1)

            if self.semfirst_delta_t > 0:
                x0_sem = x0_img[:, -self.semantic_chans:]
                x0_tex = x0_img[:, :-self.semantic_chans]
                x1_sem = x1[:, -self.semantic_chans:]
                x1_tex = x1[:, :-self.semantic_chans]
                t_sem_enc = (t_encode_raw * (1 + self.semfirst_delta_t)).clamp(max=1.0)
                t_tex_enc = (t_sem_enc - self.semfirst_delta_t).clamp(min=0.0)
                _, xt_sem_enc, _ = self.path_sampler.plan(t_sem_enc, x0_sem, x1_sem)
                _, xt_tex_enc, _ = self.path_sampler.plan(t_tex_enc, x0_tex, x1_tex)
                xt_encode = th.cat([xt_tex_enc, xt_sem_enc], dim=1)
                t_encode_img_for_model = (t_tex_enc, t_sem_enc)
                t_sem_den = (t_denoise_raw * (1 + self.semfirst_delta_t)).clamp(max=1.0)
                t_tex_den = (t_sem_den - self.semfirst_delta_t).clamp(min=0.0)
                t_sem_den, xt_sem, ut_sem = self.path_sampler.plan(t_sem_den, x0_sem, x1_sem)
                t_tex_den, xt_tex, ut_tex = self.path_sampler.plan(t_tex_den, x0_tex, x1_tex)
                t_img_for_model = (t_tex_den, t_sem_den)
                xt_img = th.cat([xt_tex, xt_sem], dim=1)
                ut_img = th.cat([ut_tex, ut_sem], dim=1)
            else:
                _, xt_encode, _ = self.path_sampler.plan(t_encode_raw, x0_img, x1)
                t_encode_img_for_model = t_encode_raw
                t_denoise_raw, xt_img, ut_img = self.path_sampler.plan(t_denoise_raw, x0_img, x1)
                t_img_for_model = t_denoise_raw
            t_img = t_denoise_raw
        else:
            t_encode_img = th.ones(B, device=device)
            if self.semfirst_delta_t > 0:
                t_encode_img_for_model = (t_encode_img, t_encode_img)
            else:
                t_encode_img_for_model = t_encode_img
            xt_encode = x1

        # Use unwrapped model for Pass 1
        pass1_out = m(
            xt_encode,
            t=t_encode_img_for_model,
            x_hidden=x0_h,
            t_hidden=t_encode_hid,
            force_drop_ids=force_drop_ids,
            encode_mode=use_encode_mode_emb,
            **model_kwargs,
        )
        h_velocity = pass1_out[1]
        h_clean = x0_h + h_velocity  # reconstruct clean encoding

        # Regularization: penalize hidden token norms deviating from unit sphere
        if hidden_reg_weight > 0:
            h_radius = h_clean.norm(dim=-1)  # (B, num_hidden_tokens)
            hidden_reg_loss = (h_radius - 1).pow(2).mean()

        # Project onto unit sphere
        if normalize_hidden:
            h_clean = h_clean / h_clean.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # ============ SAMPLE IMAGE NOISE ============
        if not noisy_img_encode:
            t_img, x0_img, x1_img = self.sample(x1, sp_timesteps, shifted_mu)

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

        # ============ PASS 2: Merged image + hidden denoising ============
        t0_h, t1_h = self.check_interval(self.train_eps, self.sample_eps)

        if hidden_same_t_as_img:
            t_h = t_img.clone().clamp(t0_h, t1_h)
        elif hidden_t_shift != 0:
            t_h = self.sample_logit_normal(hidden_t_shift, 1, size=B).to(device)
            t_h = t_h * (t1_h - t0_h) + t0_h
        else:
            t_h = th.rand(B, device=device) * (t1_h - t0_h) + t0_h
        if hidden_grad_dyn_scale > 0.0:
            h_clean = self.dyn_grad_scale(h_clean, t_h, hidden_grad_dyn_scale)

        if hidden_grad_static_scale != 1.0:
            h_clean = self.grad_scale(h_clean, hidden_grad_static_scale)

        x0_h2 = x0_h if hidden_reuse_noise_pass2 else th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)

        # Noise the encoding — grad flows through h_clean back to Pass 1
        t_h_expanded = t_h.view(B, 1, 1)
        xt_h = t_h_expanded * h_clean + (1 - t_h_expanded) * x0_h2

        # Detached target: gradient from hidden MSE does NOT push on h_clean
        # directly; it only reaches the encoder through h_pred → xt_h → h_clean.
        h_clean_detached = h_clean.detach()
        x0_h3 = x0_h if hidden_reuse_noise_pass3 else th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)
        ut_h = h_clean_detached - x0_h3

        # Single forward pass: predict image velocity + hidden velocity
        if use_repa:
            assert feature_dino is not None
            img_pred, h_pred, repa_feat_proj = model(
                xt_img,
                t=t_img_for_model,
                x_hidden=xt_h,
                t_hidden=t_h,
                force_drop_ids=force_drop_ids,
                **model_kwargs,
            )
        else:
            img_pred, h_pred = model(
                xt_img,
                t=t_img_for_model,
                x_hidden=xt_h,
                t_hidden=t_h,
                force_drop_ids=force_drop_ids,
                **model_kwargs,
            )

        # --- Hidden-conditioning guidance (amplified target) ---
        if hidden_guidance_scale > 1.0:
            h_uncond = th.randn(B, num_hidden_tokens, hidden_token_dim, device=device)
            t_h_uncond = th.zeros(B, device=device)
            with th.no_grad():
                v_uncond_img = m(
                    xt_img.detach(),
                    t=t_img_for_model if not isinstance(t_img_for_model, tuple)
                    else tuple(t.detach() for t in t_img_for_model),
                    x_hidden=h_uncond,
                    t_hidden=t_h_uncond,
                    force_drop_ids=force_drop_ids,
                    **model_kwargs,
                )[0]
            w_guidance = 1.0 + (hidden_guidance_scale - 1.0) * t_h.view(B, 1, 1, 1)
            ut_img_guided = v_uncond_img + w_guidance * (ut_img - v_uncond_img)
        else:
            ut_img_guided = ut_img

        # --- Image denoising loss ---
        loss_per_channel = (img_pred - ut_img_guided) ** 2
        if self.semantic_chans > 0 and self.semantic_weight != 1.0:
            ch_num = ut_img_guided.shape[1]
            regular_chans = ch_num - self.semantic_chans
            regular_loss = loss_per_channel[:, :regular_chans]
            semantic_loss = loss_per_channel[:, regular_chans:] * self.semantic_weight
            total_img_loss = th.cat([regular_loss, semantic_loss], dim=1)
            img_loss = mean_flat(total_img_loss)
        else:
            img_loss = mean_flat(loss_per_channel)

        # --- Hidden denoising loss (target detached, prediction has grad) ---
        hidden_loss = mean_flat((h_pred - ut_h) ** 2) * hidden_weight * hidden_loss_scale

        # --- Optional cosine loss on single-step clean prediction ---
        if hidden_cos_weight > 0:
            h1_pred = xt_h + (1 - t_h_expanded) * h_pred
            hidden_cos_loss = mean_flat(
                1 - th.nn.functional.cosine_similarity(h1_pred, h_clean_detached, dim=-1)
            ) * hidden_cos_weight * hidden_loss_scale
        else:
            hidden_cos_loss = None

        # --- Assemble terms ---
        terms = {}
        terms['loss'] = img_loss
        terms['pred'] = img_pred
        terms['hidden_loss'] = hidden_loss
        if hidden_cos_loss is not None:
            terms['hidden_cos_loss'] = hidden_cos_loss

        if self.use_cosine_loss:
            terms['cos_loss'] = mean_flat(
                1 - th.nn.functional.cosine_similarity(img_pred, ut_img, dim=1))

        if use_repa:
            feature_dino = einops.rearrange(feature_dino, 'b c h w -> b (h w) c')
            if self.repa_mode == 'cos':
                repa_loss = mean_flat(1 - th.nn.functional.cosine_similarity(
                    feature_dino, repa_feat_proj, dim=-1))
            elif self.repa_mode == 'mse':
                repa_loss = mean_flat((feature_dino - repa_feat_proj) ** 2)
            elif self.repa_mode == 'cos_mse':
                cos_loss = mean_flat(1 - th.nn.functional.cosine_similarity(
                    feature_dino, repa_feat_proj, dim=-1))
                mse_loss = mean_flat((feature_dino - repa_feat_proj) ** 2)
                repa_loss = (cos_loss + mse_loss) / 2
            terms['repa_loss'] = repa_loss * self.repa_weight

        if hidden_reg_weight > 0:
            terms['hidden_reg_loss'] = hidden_reg_loss * hidden_reg_weight

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

    def sample_ode_semantic_first_hidden(
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
        num_hidden_tokens=8,
        hidden_token_dim=32,
        hidden_schedule="semantic",
        hidden_schedule_start_t=0.0,
        hidden_schedule_max_t=1.0,
        hidden_sphere_clamp=False,
        hidden_rep_guidance=1.0,
        hidden_reground_t_fix=1.0,
        normalize_hidden=True,
        reground_fixed_enc_noise=False,
        reground_fixed_cond_noise=False,
        reground_shared_noise=False,
        reground_reuse_encode_for_repg=False,
        recycle_t_fix=None,
        collect_hidden_trajectory=False,
        cfg_scale=1.0,
        autoguidance_model=None,
        null_class_label=1000,
        cfg_noise_hidden=False,
    ):
        """
        Returns a sampling function for joint image + hidden token ODE with
        semantic-first scheduling.

        Args:
            hidden_schedule:
                - "semantic": Hidden tokens sync with semantic tokens (frozen during texture-only phase).
                - "linear": Hidden tokens evolve linearly from 0 to hidden_schedule_max_t over the full
                  [0, 1+dt] trajectory. With hidden_schedule_max_t=1.0 (default) this is equivalent to
                  full denoising; values < 1.0 leave hidden tokens partially noisy at generation end,
                  which can be beneficial when the hidden token generator is imperfect.
                - "fixed": Hidden tokens are kept fixed (t_hid=1.0, zero velocity). Use with _z_hidden.
                - "linear_from": Like "linear", but hidden tokens stay frozen at hidden_schedule_start_t
                  until the linear schedule catches up, then evolve normally.
                - "encode_linear": Hidden tokens start at hidden_schedule_start_t and linearly reach 1.0
                  by the end. Intended for GT-initialised hidden tokens:
                  h_init = start_t * h_clean + (1-start_t) * noise, passed via _z_hidden.
                - "reground": At every ODE step, re-encode hidden tokens from the current noisy image
                  x_t by running Pass-1 (pure-noise hidden input, t_hid=0), then condition the image
                  denoising step on the recovered h_clean (optionally noised to hidden_reground_t_fix).
                  This allows hidden tokens to track the image as it is denoised.
                  Two model calls per step; hidden ODE state is never updated.
                - "recycle": Single forward pass per step. The model sees hidden tokens at a fixed
                  noise level recycle_t_fix and predicts velocity for both image and hidden. The
                  hidden velocity is used to extract h_clean, which is then re-noised back to
                  recycle_t_fix using the frozen ODE hidden state as fixed noise. This lets
                  h_clean drift to track the evolving image without an extra forward pass.
            hidden_schedule_start_t: Used with "linear_from" and "encode_linear". For "linear_from",
                hidden tokens are frozen until t_hid would exceed this value. For "encode_linear",
                this is the starting hidden timestep (noise level of the GT-initialised hidden state).
            hidden_schedule_max_t: Only used with "linear". Maximum hidden timestep reached at
                the end of generation (default 1.0 = fully clean). Values in (0, 1) leave hidden
                tokens partially noisy; the per-step Δt_hid is scaled proportionally.
            hidden_reground_t_fix: Only used with "reground". The noise level at which the
                re-encoded h_clean is presented to the model for the conditioning step.
                1.0 = fully clean (default), 0.0 = pure noise (no conditioning).
            normalize_hidden: Only used with "reground". Whether to L2-normalise h_clean onto
                the unit sphere before noising/conditioning (mirrors training normalize_hidden,
                default True).
            reground_fixed_enc_noise: Only used with "reground". If True, sample the encode-pass
                noise once on the first ODE step and reuse it for all subsequent steps.
            reground_fixed_cond_noise: Only used with "reground". If True, sample the conditioning
                noise (used to noise h_clean to t_fix) once and reuse it for all subsequent steps.
            reground_shared_noise: Only used with "reground". If True, use the same noise for both
                the encode pass and the conditioning re-noising. This makes the conditioning input
                xt_h = x0 + t_fix * v_enc, i.e. stopping partway along the encoding flow line.
                Overrides reground_fixed_cond_noise (cond noise is just enc noise).
            recycle_t_fix: Only used with "recycle". The fixed noise level at which hidden tokens
                are presented to the model. After each step, h_clean is extracted from the velocity
                prediction and re-noised back to this level. 1.0 = nearly clean, 0.0 = pure noise.
        """
        t_max = 1.0 + semfirst_delta_t
        texture_chans = None
        _reground_cache = {}
        _recycle_cache = {}
        _hidden_trajectory = []  # collects h_clean at each step when enabled

        def joint_drift_semantic_first(state, t, model, **model_kwargs):
            """
            Joint drift for (x_img, x_hidden) with semantic-first scheduling.
            """
            nonlocal texture_chans
            x_img, x_hidden = state
            B, C = x_img.shape[:2]
            device = x_img.device

            if texture_chans is None:
                texture_chans = C - semantic_chans

            # Compute per-component times
            t_sem = t.clamp(max=1.0)                       # semantic time
            t_tex = (t - semfirst_delta_t).clamp(min=0.0)  # texture time (delayed)
            t_tuple = (t_tex, t_sem)

            # ----------------------------------------------------------------
            # "reground" schedule: 2-3 model calls per step, single batch.
            #
            # Guidance (CFG/autoguidance/repg) is assembled inside this branch
            # rather than via model.forward_with_cfg, so the encode pass never
            # sees a doubled CFG batch (which would otherwise cause the cond
            # and uncond halves to compute different h_clean from different
            # class conditioning — diluting the eventual guidance signal).
            #
            # Pass layout:
            #   1. Encode pass (single batch, plain forward, conditional y):
            #      → h_clean for the conditioning pass
            #      → also serves as v_repg_uncond when reground_reuse_encode_for_repg=True
            #   2. Conditioning pass: image denoising with re-grounded hidden → v_main
            #   3. Optional repg uncond pass: skipped if reused from encode pass
            #   4. Optional autoguidance pass: separate (degraded/no-hidden) model
            #
            # Combined velocity:
            #   v = v_main
            #     + (cfg_scale - 1)        * (v_main - v_autog)         if cfg_scale > 1
            #     + (hidden_rep_guidance-1) * (v_main - v_repg_uncond)  if repg > 1
            #
            # NOTE: this changes the historical reground repg weight from
            #   w = 1 + (s-1) * t_fix   →   w = 1 + (s-1)
            # so existing reground+repg numbers will shift slightly.
            # ----------------------------------------------------------------
            if hidden_schedule == "reground":
                # ----- Pass 1: encode -----
                if reground_fixed_enc_noise and "enc_noise" in _reground_cache:
                    x0_h_enc = _reground_cache["enc_noise"]
                else:
                    x0_h_enc = th.randn_like(x_hidden)
                    if reground_fixed_enc_noise:
                        _reground_cache["enc_noise"] = x0_h_enc
                t_h_enc = th.zeros(B, device=device)
                enc_result = model(x_img, t=t_tuple, x_hidden=x0_h_enc,
                                   t_hidden=t_h_enc, **model_kwargs)
                # enc_result[0] is the image velocity at (x_img, noise hidden, t_h=0)
                # — exactly the inputs the repg uncond pass would use. Cache it
                # when reuse is requested to skip a redundant forward.
                img_vel_repg_uncond = enc_result[0] if reground_reuse_encode_for_repg else None
                h_clean = x0_h_enc + enc_result[1]
                if normalize_hidden:
                    h_clean = h_clean / h_clean.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                if collect_hidden_trajectory:
                    _hidden_trajectory.append(h_clean.detach().cpu())

                # Noise h_clean to t_fix level for the conditioning call
                if reground_shared_noise:
                    x0_h_cond = x0_h_enc
                elif reground_fixed_cond_noise and "cond_noise" in _reground_cache:
                    x0_h_cond = _reground_cache["cond_noise"]
                else:
                    x0_h_cond = th.randn_like(h_clean)
                    if reground_fixed_cond_noise:
                        _reground_cache["cond_noise"] = x0_h_cond
                xt_h = hidden_reground_t_fix * h_clean + (1.0 - hidden_reground_t_fix) * x0_h_cond
                t_hid_cond = th.full((B,), hidden_reground_t_fix, device=device)

                # ----- Pass 2: conditioning (v_main) -----
                cond_result = model(x_img, t=t_tuple, x_hidden=xt_h,
                                    t_hidden=t_hid_cond, **model_kwargs)
                img_vel = cond_result[0]

                # ----- Pass 3 (optional): hidden representation guidance -----
                # ----- Pass 4 (optional): classifier-free / autoguidance -----
                # Both guidance terms are computed as deltas from the original
                # v_main and summed additively to avoid cross-terms:
                #   v = v_main
                #     + (repg - 1) * (v_main - v_repg_uncond)
                #     + (cfg  - 1) * (v_main - v_neg)
                img_vel_main = img_vel  # save before modification

                if hidden_rep_guidance > 1.0:
                    if img_vel_repg_uncond is None:
                        h_uncond = th.randn(B, num_hidden_tokens, hidden_token_dim,
                                            device=device)
                        t_h_uncond = th.zeros(B, device=device)
                        repg_result = model(x_img, t=t_tuple, x_hidden=h_uncond,
                                            t_hidden=t_h_uncond, **model_kwargs)
                        img_vel_repg_uncond = repg_result[0]
                    img_vel = img_vel + (hidden_rep_guidance - 1.0) * (img_vel_main - img_vel_repg_uncond)

                if cfg_scale > 1.0:
                    if autoguidance_model is not None:
                        if hasattr(autoguidance_model, 'num_hidden_tokens'):
                            # (a) Hidden autoguidance: mirror the conditioning pass.
                            autog_result = autoguidance_model(
                                x_img, t=t_tuple, x_hidden=xt_h,
                                t_hidden=t_hid_cond, **model_kwargs)
                        else:
                            # (b) No-hidden autoguidance (e.g. base SFD checkpoint):
                            autog_result = autoguidance_model(
                                x_img, t=t_tuple, **model_kwargs)
                    else:
                        # (c) Pure CFG: same model with the null class label.
                        y_cond = model_kwargs['y']
                        y_null = th.full_like(y_cond, null_class_label)
                        null_kwargs = {**model_kwargs, 'y': y_null}
                        if cfg_noise_hidden:
                            # Fully noise hidden tokens for the negative pass so
                            # CFG captures the full conditioning effect (class + hidden).
                            h_cfg_neg = th.randn(B, num_hidden_tokens, hidden_token_dim,
                                                 device=device)
                            t_h_cfg_neg = th.zeros(B, device=device)
                            autog_result = model(
                                x_img, t=t_tuple, x_hidden=h_cfg_neg,
                                t_hidden=t_h_cfg_neg, **null_kwargs)
                        else:
                            autog_result = model(
                                x_img, t=t_tuple, x_hidden=xt_h,
                                t_hidden=t_hid_cond, **null_kwargs)
                    img_vel_autog = autog_result[0] if isinstance(autog_result, tuple) else autog_result
                    img_vel = img_vel + (cfg_scale - 1.0) * (img_vel_main - img_vel_autog)

                # Semantic-first image velocity masking (applied once to final v)
                img_mask = th.zeros_like(x_img)
                img_mask[:, :texture_chans, ...] = (t >= semfirst_delta_t).view(
                    B, 1, *[1] * (x_img.dim() - 2))
                img_mask[:, -semantic_chans:, ...] = (t <= 1.0).view(
                    B, 1, *[1] * (x_img.dim() - 2))
                img_vel = img_vel * img_mask

                # Hidden state not evolved by the ODE (re-encoded fresh each step)
                return (img_vel, th.zeros_like(x_hidden))

            # ----------------------------------------------------------------
            # "recycle" schedule: single model call per step.
            # The model sees hidden tokens at fixed noise level recycle_t_fix.
            # Hidden velocity is used to extract h_clean, which is re-noised
            # back to recycle_t_fix using the frozen ODE x_hidden as noise.
            # ----------------------------------------------------------------
            if hidden_schedule == "recycle":
                # Build hidden input: on first call use ODE state (random noise),
                # on subsequent calls mix cached h_clean with ODE noise at t_fix.
                if "h_clean" in _recycle_cache:
                    x_h = recycle_t_fix * _recycle_cache["h_clean"] + (1.0 - recycle_t_fix) * x_hidden
                else:
                    x_h = x_hidden  # first step: pure noise
                t_hid = th.full((B,), recycle_t_fix, device=device)

                # Single model call
                result = model(x_img, t=t_tuple, x_hidden=x_h,
                               t_hidden=t_hid, **model_kwargs)
                img_vel = result[0]
                hidden_vel = result[1]

                # Extract h_clean from velocity: x_1 = x_t + (1 - t) * v
                h_clean = x_h + (1.0 - recycle_t_fix) * hidden_vel
                if normalize_hidden:
                    h_clean = h_clean / h_clean.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                _recycle_cache["h_clean"] = h_clean
                if collect_hidden_trajectory:
                    _hidden_trajectory.append(h_clean.detach().cpu())

                # Optional hidden-representation guidance
                if hidden_rep_guidance > 1.0:
                    h_uncond = th.randn(B, num_hidden_tokens, hidden_token_dim,
                                        device=device)
                    t_h_uncond = th.zeros(B, device=device)
                    result_uncond = model(x_img, t=t_tuple, x_hidden=h_uncond,
                                          t_hidden=t_h_uncond, **model_kwargs)
                    img_vel_uncond = result_uncond[0]
                    w = 1.0 + (hidden_rep_guidance - 1.0) * recycle_t_fix
                    img_vel = img_vel_uncond + w * (img_vel - img_vel_uncond)

                # Semantic-first image velocity masking
                img_mask = th.zeros_like(x_img)
                img_mask[:, :texture_chans, ...] = (t >= semfirst_delta_t).view(
                    B, 1, *[1] * (x_img.dim() - 2))
                img_mask[:, -semantic_chans:, ...] = (t <= 1.0).view(
                    B, 1, *[1] * (x_img.dim() - 2))
                img_vel = img_vel * img_mask

                # Hidden ODE state is a dummy — return zero velocity
                return (img_vel, th.zeros_like(x_hidden))

            if hidden_schedule == "fixed":
                t_hid = th.ones_like(t)
            elif hidden_schedule == "linear":
                # Scale linearly from 0 to hidden_schedule_max_t over the full trajectory.
                # With max_t=1.0 (default) hidden tokens are fully denoised at the end;
                # max_t < 1.0 leaves them partially noisy, reducing each step's Δt_hid.
                t_hid = (t / t_max) * hidden_schedule_max_t
            elif hidden_schedule == "linear_from":
                t_hid_raw = t / t_max
                t_hid = th.where(t_hid_raw >= hidden_schedule_start_t,
                                 t_hid_raw,
                                 th.full_like(t_hid_raw, hidden_schedule_start_t))
            elif hidden_schedule == "encode_linear":
                # Compressed linear: t_hid goes from start_t → 1.0 over full trajectory
                t_hid = hidden_schedule_start_t + (t / t_max) * (1.0 - hidden_schedule_start_t)
            else:
                t_hid = t_sem

            # Model expects tuple timestep for semantic first (already set above)

            # Forward: model returns (img_velocity, hidden_velocity)
            result = model(x_img, t=t_tuple, x_hidden=x_hidden,
                           t_hidden=t_hid, **model_kwargs)
            if isinstance(result, tuple):
                img_vel, hidden_vel = result[0], result[1]
            else:
                raise ValueError("Model must return (img_output, hidden_output) tuple")

            # Hidden representation guidance: amplify the image velocity in the
            # direction away from the unconditional (no-hidden-info) prediction.
            #   w(t_hid) = 1 + (hidden_rep_guidance - 1) * t_hid
            # At t_hid≈0 both passes are nearly equivalent so w≈1 (no effect);
            # at t_hid=1 (clean hidden) the full guidance weight is applied.
            if hidden_rep_guidance > 1.0:
                h_uncond = th.randn(B, num_hidden_tokens, hidden_token_dim,
                                    device=device)
                t_h_uncond = th.zeros(B, device=device)
                with th.no_grad():
                    result_uncond = model(x_img, t=t_tuple,
                                          x_hidden=h_uncond,
                                          t_hidden=t_h_uncond, **model_kwargs)
                img_vel_uncond = result_uncond[0]
                w = 1.0 + (hidden_rep_guidance - 1.0) * t_hid.view(B, 1, 1, 1)
                img_vel = img_vel_uncond + w * (img_vel - img_vel_uncond)

            # Apply channel masking to image velocity
            img_mask = th.zeros_like(x_img)
            img_mask[:, :texture_chans, ...] = (t >= semfirst_delta_t).view(
                B, 1, *[1] * (x_img.dim() - 2))
            img_mask[:, -semantic_chans:, ...] = (t <= 1.0).view(
                B, 1, *[1] * (x_img.dim() - 2))
            img_vel = img_vel * img_mask

            # Sphere-clamping for hidden tokens (before masking).
            # At each step we recover the single-step clean prediction h_1_pred,
            # project it onto the unit sphere token-wise, then reconstruct the
            # velocity so that it points toward the sphere-normalised target while
            # keeping the noise estimate h_0_pred unchanged.
            #
            #   h_1_pred  = h_t + (1-t_hid) * v          (rectified-flow clean pred)
            #   h_0_pred  = h_t - t_hid     * v           (noise pred)
            #   h_1_clamp = h_1_pred / ||h_1_pred||_token  (per-token L2 norm)
            #   v_clamp   = h_1_clamp - h_0_pred
            #
            # Applied only when the schedule actively moves hidden tokens; the
            # subsequent masking will zero-out v_clamp whenever the schedule is
            # frozen (e.g. linear_from before start_t), so order is safe.
            if hidden_sphere_clamp and hidden_schedule != "fixed":
                t_hid_exp = t_hid.view(B, 1, 1)  # (B, 1, 1) broadcast over (tokens, dim)
                h_clean_pred = x_hidden + (1.0 - t_hid_exp) * hidden_vel
                h_noise_pred = x_hidden - t_hid_exp * hidden_vel
                # Token-wise L2 norm; clamp to avoid division by zero
                h_norms = h_clean_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                h_clean_clamped = h_clean_pred / h_norms
                hidden_vel = h_clean_clamped - h_noise_pred

            # Collect h_clean prediction for trajectory analysis
            if collect_hidden_trajectory and hidden_schedule != "fixed":
                t_hid_exp_traj = t_hid.view(B, 1, 1)
                if hidden_sphere_clamp:
                    # h_clean_clamped already computed above
                    _hidden_trajectory.append(h_clean_clamped.detach().cpu())
                else:
                    h_clean_pred_traj = x_hidden + (1.0 - t_hid_exp_traj) * hidden_vel
                    if normalize_hidden:
                        h_clean_pred_traj = h_clean_pred_traj / h_clean_pred_traj.norm(
                            dim=-1, keepdim=True).clamp(min=1e-6)
                    _hidden_trajectory.append(h_clean_pred_traj.detach().cpu())

            # Hidden velocity masking
            if hidden_schedule == "fixed":
                hidden_vel = th.zeros_like(hidden_vel)
            elif hidden_schedule in ("linear", "encode_linear"):
                pass  # active throughout
            elif hidden_schedule == "linear_from":
                t_hid_raw = t / t_max
                hid_active = (t_hid_raw >= hidden_schedule_start_t).view(
                    B, *[1] * (x_hidden.dim() - 1)).float()
                hidden_vel = hidden_vel * hid_active
            else:
                hid_mask = (t <= 1.0).view(B, *[1] * (x_hidden.dim() - 1)).float()
                hidden_vel = hidden_vel * hid_mask

            return (img_vel, hidden_vel)

        if reverse:
            raise NotImplementedError("Reverse sampling not yet implemented for semantic first hidden")

        t0 = 0.0
        t1 = t_max

        _ode = ode(
            drift=joint_drift_semantic_first,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            timestep_shift=timestep_shift,
            semfirst_delta_t=semfirst_delta_t,
        )

        def _sample_fn(z_img, model_fn, **model_kwargs):
            """
            Sample image + hidden tokens jointly via semantic-first ODE.

            Special kwargs (popped before forwarding to model):
                _z_hidden: (B, num_hidden_tokens, hidden_token_dim) override initial hidden state
                _return_hidden: if True, return [img_final, h_final] instead of [img_final]
                _return_trajectory: if True, append the h_clean trajectory list as the last element
            """
            _z_hidden = model_kwargs.pop('_z_hidden', None)
            _return_hidden = model_kwargs.pop('_return_hidden', False)
            _return_trajectory = model_kwargs.pop('_return_trajectory', False)

            # Clear trajectory from any previous call
            _hidden_trajectory.clear()

            B = z_img.shape[0]
            device = z_img.device
            if _z_hidden is not None:
                z_hidden = _z_hidden
            else:
                z_hidden = th.randn(B, num_hidden_tokens, hidden_token_dim,
                                    device=device)
            init_state = (z_img, z_hidden)
            samples = _ode.sample(init_state, model_fn, **model_kwargs)
            if isinstance(samples, tuple):
                img_final = samples[0][-1]
                h_final = samples[1][-1]
            else:
                img_final = samples[-1]
                h_final = None

            result = [img_final]
            if _return_hidden:
                result.append(h_final)
            if _return_trajectory and _hidden_trajectory:
                # Stack into (num_steps, B, num_tokens, dim)
                result.append(th.stack(_hidden_trajectory, dim=0))
            return result

        return _sample_fn

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