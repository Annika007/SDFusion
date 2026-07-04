import os
import sys
from pathlib import Path
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if 'einops' not in sys.modules:
    einops_stub = types.ModuleType('einops')

    def _repeat(tensor, pattern, **kwargs):
        if pattern == 'b -> b d':
            return tensor[:, None].repeat(1, kwargs['d'])
        raise NotImplementedError(f'Unsupported repeat pattern: {pattern}')

    einops_stub.repeat = _repeat
    sys.modules['einops'] = einops_stub

from models.networks.diffusion_networks.samplers.ddim import DDIMSampler


class DummyVQVAE:
    def quantize(self, x, is_voxel=True):
        return x, None, None, None


class DummyModel:
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')
        self.num_timesteps = 8
        self.betas = torch.linspace(1e-4, 2e-2, self.num_timesteps, device=self.device)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.ddim_sigmas_for_original_num_steps = torch.zeros_like(self.betas)
        self.ddim_sigmas = torch.zeros_like(self.betas)
        self.ddim_alphas = self.alphas_cumprod
        self.ddim_alphas_prev = self.alphas_cumprod_prev
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - self.ddim_alphas)
        self.parameterization = 'eps'
        self.vqvae = DummyVQVAE()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.zeros_like(x_start)
        return x_start + noise * 0.0

    def apply_model(self, x, t, cond):
        return torch.zeros_like(x)


def main():
    model = DummyModel()
    sampler = DDIMSampler(model)
    samples, intermediates = sampler.sample(
        S=4,
        batch_size=1,
        shape=(4, 8, 8, 8),
        conditioning=None,
        verbose=False,
        unconditional_guidance_scale=1.0,
        eta=0.0,
        pyramid_list=[1, 1, 2, 2],
        pyramid_interp_mode='trilinear',
        pyramid_use_up_v2=True,
    )
    assert samples.shape == (1, 4, 8, 8, 8), samples.shape
    assert 'x_inter' in intermediates and 'pred_x0' in intermediates
    print('pyramid_ddim_smoke_ok', tuple(samples.shape), len(intermediates['x_inter']))


if __name__ == '__main__':
    main()
