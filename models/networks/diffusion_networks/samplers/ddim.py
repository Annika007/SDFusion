"""SAMPLING ONLY."""
""" Reference: https://github.com/CompVis/latent-diffusion/tree/main/ldm/models/diffusion """

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from functools import partial

from models.networks.diffusion_networks.ldm_diffusion_util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like
)

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def _interpolate_spatial(self, x, size, mode=None):
        if x.dim() == 4:
            mode = mode or "bilinear"
            if mode == "nearest":
                return F.interpolate(x, size=size, mode=mode)
            return F.interpolate(x, size=size, mode=mode, align_corners=False)
        if x.dim() == 5:
            mode = mode or "trilinear"
            if mode == "nearest":
                return F.interpolate(x, size=size, mode=mode)
            return F.interpolate(x, size=size, mode=mode, align_corners=False)
        return x

    def _resize_conditioning(self, cond, size, mode=None):
        if cond is None:
            return None
        if torch.is_tensor(cond):
            if cond.dim() in (4, 5) and tuple(cond.shape[2:]) != tuple(size):
                return self._interpolate_spatial(cond, size=size, mode=mode)
            return cond
        if isinstance(cond, list):
            return [self._resize_conditioning(item, size, mode=mode) for item in cond]
        if isinstance(cond, tuple):
            return tuple(self._resize_conditioning(item, size, mode=mode) for item in cond)
        if isinstance(cond, dict):
            resized = {}
            for key, value in cond.items():
                if key in ("img_w", "txt_w"):
                    resized[key] = value
                else:
                    resized[key] = self._resize_conditioning(value, size, mode=mode)
            return resized
        return cond

    def _resize_mask(self, mask, size):
        if mask is None:
            return None
        if torch.is_tensor(mask) and mask.dim() in (4, 5) and tuple(mask.shape[2:]) != tuple(size):
            return self._interpolate_spatial(mask, size=size, mode="nearest")
        return mask

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               mm_cls_free=False,
               pyramid_list=None,
               pyramid_interp_mode=None,
               pyramid_use_up_v2=False,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                # cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        if len(shape) == 4:
            C, D, H, W = shape
            size = (batch_size, C, D, H, W)
        else:
            C, H, W = shape
            size = (batch_size, C, H, W)
        
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        if pyramid_list is not None:
            return self.ddim_sampling_pyramid(
                conditioning, size, pyramid_list,
                x_T=x_T,
                ddim_use_original_steps=False,
                callback=callback,
                timesteps=None,
                quantize_denoised=quantize_x0,
                mask=mask,
                x0=x0,
                img_callback=img_callback,
                log_every_t=log_every_t,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mm_cls_free=mm_cls_free,
                verbose=verbose,
                eta=eta,
                pyramid_interp_mode=pyramid_interp_mode,
                pyramid_use_up_v2=pyramid_use_up_v2,
            )


        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    mm_cls_free=mm_cls_free,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling_pyramid(self, cond, shape, pyramid_list,
                              x_T=None, ddim_use_original_steps=False,
                              callback=None, timesteps=None, quantize_denoised=False,
                              mask=None, x0=None, img_callback=None, log_every_t=100,
                              temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                              unconditional_guidance_scale=1., unconditional_conditioning=None,
                              mm_cls_free=False, verbose=True, eta=0.,
                              pyramid_interp_mode=None, pyramid_use_up_v2=False):
        if len(shape) == 5:
            spatial_dims = 3
        elif len(shape) == 4:
            spatial_dims = 2
        else:
            raise ValueError(f"Unsupported latent shape: {shape}")

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if len(pyramid_list) != total_steps:
            raise ValueError(f"len(pyramid_list) must equal ddim steps, got {len(pyramid_list)} and {total_steps}")

        device = self.model.betas.device
        b = shape[0]
        full_spatial = tuple(shape[2:])
        coarse_spatial = tuple(max(1, full_spatial[i] // int(pyramid_list[-1])) for i in range(spatial_dims))

        if x_T is None:
            img = torch.randn((b, shape[1], *coarse_spatial), device=device)
        else:
            img = x_T
            if tuple(img.shape[2:]) != coarse_spatial:
                img = self._interpolate_spatial(img, coarse_spatial, mode=pyramid_interp_mode)

        intermediates = {
            'x_inter': [self._interpolate_spatial(img, full_spatial, mode=pyramid_interp_mode)],
            'pred_x0': [self._interpolate_spatial(img, full_spatial, mode=pyramid_interp_mode)]
        }

        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        if verbose:
            print(f"Running pyramid DDIM Sampling with {total_steps} timesteps and pyramid {pyramid_list}")

        iterator = tqdm(time_range, desc='Pyramid DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            step_scale = int(pyramid_list[index])
            step_spatial = tuple(max(1, full_spatial[d] // step_scale) for d in range(spatial_dims))

            if tuple(img.shape[2:]) != step_spatial:
                img = self._interpolate_spatial(img, step_spatial, mode=pyramid_interp_mode)

            cond_step = self._resize_conditioning(cond, step_spatial, mode=pyramid_interp_mode)
            mask_step = self._resize_mask(mask, step_spatial)
            x0_step = self._resize_conditioning(x0, step_spatial, mode=pyramid_interp_mode)

            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if mask_step is not None:
                assert x0_step is not None
                img_orig = self.model.q_sample(x0_step, ts)
                img = img_orig * mask_step + (1. - mask_step) * img

            img, pred_x0 = self.p_sample_ddim(
                img, cond_step, ts, index=index, use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised, temperature=temperature,
                noise_dropout=noise_dropout, score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mm_cls_free=mm_cls_free,
            )

            if pyramid_use_up_v2 and index > 0 and int(pyramid_list[index - 1]) != step_scale:
                next_scale = int(pyramid_list[index - 1])
                next_spatial = tuple(max(1, full_spatial[d] // next_scale) for d in range(spatial_dims))
                img = self._interpolate_spatial(img, next_spatial, mode=pyramid_interp_mode)
                pred_x0 = self._interpolate_spatial(pred_x0, next_spatial, mode=pyramid_interp_mode)

            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(self._interpolate_spatial(img, full_spatial, mode=pyramid_interp_mode))
                intermediates['pred_x0'].append(self._interpolate_spatial(pred_x0, full_spatial, mode=pyramid_interp_mode))

        return img, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      mm_cls_free=False,
                      ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      mm_cls_free=mm_cls_free,
                                      )
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, mm_cls_free=False):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        elif mm_cls_free:

            uc_img, uc_txt = unconditional_conditioning['uc_img'], unconditional_conditioning['uc_txt']
            mm_uc_feat = torch.cat([uc_img, uc_txt], dim=1)

            c_img, c_txt, img_w, txt_w = c['c_img'], c['c_txt'], c['img_w'], c['txt_w']
            c_img_with_txt = torch.cat([c_img, torch.zeros_like(c_txt)], dim=1)
            c_txt_with_img  = torch.cat([torch.zeros_like(c_img), c_txt], dim=1)
            
            x_in = torch.cat([x] * 3)
            t_in = torch.cat([t] * 3)
            c_in = torch.cat([mm_uc_feat, c_img_with_txt, c_txt_with_img])
            e_t_uncond, e_t_img, e_t_txt = self.model.apply_model(x_in, t_in, c_in).chunk(3)
            e_t = e_t_uncond + \
                  unconditional_guidance_scale * img_w * (e_t_img - e_t_uncond) + \
                  unconditional_guidance_scale * txt_w * (e_t_txt - e_t_uncond)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        
        if x.dim() == 5:
            param_shape = (b, 1, 1, 1, 1)
        else:
            param_shape = (b, 1, 1, 1)

        a_t = torch.full(param_shape, alphas[index], device=device)
        a_prev = torch.full(param_shape, alphas_prev[index], device=device)
        sigma_t = torch.full(param_shape, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(param_shape, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            # pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            pred_x0, _, *_ = self.model.vqvae.quantize(pred_x0, is_voxel=True)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
