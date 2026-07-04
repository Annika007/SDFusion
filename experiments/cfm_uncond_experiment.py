import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent


def setup_repo(repo_root: Path):
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_opt(args, is_train=False):
    return SimpleNamespace(
        isTrain=is_train,
        gpu_ids=[0],
        gpu_ids_str="0",
        device="cuda",
        distributed=False,
        local_rank=0,
        rank=0,
        df_cfg="configs/sdfusion_snet.yaml",
        vq_cfg="configs/vqvae_snet.yaml",
        vq_ckpt=args.vq_ckpt,
        ckpt=args.base_ckpt,
        dataset_mode="snet",
        debug="0",
        model="sdfusion",
        cat=args.cat,
        dataroot=args.dataroot,
        max_dataset_size=args.max_dataset_size,
        trunc_thres=args.trunc_thres,
        res=64,
        batch_size=args.batch_size,
        logs_dir=str(args.out_dir),
        name=args.name,
    )


def install_cfm_time_embedding(model):
    """Add an r-time embedder and a CFM forward method without editing repo files."""
    from models.networks.diffusion_networks.ldm_diffusion_util import timestep_embedding
    import torch as th

    unet = model.df.diffusion_net
    if not hasattr(unet, "time_embed_r"):
        unet.time_embed_r = copy.deepcopy(unet.time_embed)

    def forward_cfm(self, x, timesteps=None, r_timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (self.num_classes is not None), (
            "must specify y if and only if the model is class-conditional"
        )
        if r_timesteps is None:
            r_timesteps = timesteps
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        r_emb = timestep_embedding(r_timesteps, self.model_channels, repeat_only=False)
        r_cond_mode = getattr(self, "cfm_r_cond_mode", "avg")
        if r_cond_mode == "avg":
            emb = 0.5 * (self.time_embed(t_emb) + self.time_embed_r(r_emb))
        elif r_cond_mode == "delta_add":
            delta_timesteps = (timesteps - r_timesteps).clamp_min(0)
            delta_emb = timestep_embedding(delta_timesteps, self.model_channels, repeat_only=False)
            zero_emb = timestep_embedding(th.zeros_like(delta_timesteps), self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb) + self.time_embed_r(delta_emb) - self.time_embed_r(zero_emb)
        else:
            raise ValueError(f"unknown cfm_r_cond_mode: {r_cond_mode}")
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        return self.out(h)

    unet.forward_cfm = MethodType(forward_cfm, unet)


def load_uncond_model(args):
    import models.sdfusion_model as sdf_mod

    # Rendering is not needed for training/eval metrics and costs extra GPU memory.
    sdf_mod.init_mesh_renderer = lambda *a, **k: None
    model = sdf_mod.SDFusionModel()
    model.initialize(make_opt(args, is_train=False))
    model.switch_eval()
    install_cfm_time_embedding(model)
    model.df.diffusion_net.cfm_r_cond_mode = args.r_cond_mode
    return model


def load_cfm_ckpt(model, ckpt_path: Path):
    state = torch.load(str(ckpt_path), map_location="cuda")
    install_cfm_time_embedding(model)
    model.df.load_state_dict(state["df"], strict=True)
    if "global_step" in state:
        print(f"[*] loaded CFM checkpoint {ckpt_path} at step {state['global_step']}")


def disable_diffusion_checkpointing(model):
    def no_checkpoint(func, inputs, params, flag):
        return func(*inputs)

    try:
        import models.networks.diffusion_networks.ldm_diffusion_util as ldm_util
        import models.networks.diffusion_networks.openai_model_3d as openai_model_3d
        import models.networks.diffusion_networks.attention as attention

        ldm_util.checkpoint = no_checkpoint
        openai_model_3d.checkpoint = no_checkpoint
        attention.checkpoint = no_checkpoint
    except Exception as exc:
        print(f"[!] failed to monkey-patch checkpoint helpers: {exc}", flush=True)

    for module in model.df.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False
        if isinstance(getattr(module, "checkpoint", None), bool):
            module.checkpoint = False


def save_cfm_ckpt(model, out_dir: Path, step: int):
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "df": model.df.state_dict(),
            "global_step": step,
            "note": "Cumulative Flow Maps DDIM-CFM fine-tune for SDFusion uncond diffusion model.",
        },
        ckpt_dir / "cfm_steps-latest.pth",
    )


def pred_x0_from_eps(model, x_t, t, eps):
    from models.networks.diffusion_networks.ldm_diffusion_util import extract_into_tensor

    return (
        x_t - extract_into_tensor(model.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps
    ) / extract_into_tensor(model.sqrt_alphas_cumprod, t, x_t.shape)


def interp_schedule(values, t, x_shape=None):
    """Linear interpolation for SDFusion's discrete scheduler at float timesteps."""
    values = values.to(device=t.device, dtype=t.dtype)
    t = t.clamp(0, values.shape[0] - 1)
    lo = t.floor().long()
    hi = (lo + 1).clamp(max=values.shape[0] - 1)
    w = (t - lo.to(t.dtype))
    out = values[lo] * (1.0 - w) + values[hi] * w
    if x_shape is not None:
        out = out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))
    return out


def pred_x0_from_eps_float(model, x_t, t, eps):
    sqrt_alpha = interp_schedule(model.sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_one_minus = interp_schedule(model.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    return (x_t - sqrt_one_minus * eps) / sqrt_alpha.clamp_min(1e-8)


def q_sample_float(model, x_start, t, noise):
    sqrt_alpha = interp_schedule(model.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus = interp_schedule(model.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alpha * x_start + sqrt_one_minus * noise


def q_sample(model, x_start, t, noise):
    from models.networks.diffusion_networks.ldm_diffusion_util import extract_into_tensor

    return (
        extract_into_tensor(model.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + extract_into_tensor(model.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )


def apply_cfm(model, x_t, t, r):
    return model.df.diffusion_net.forward_cfm(x_t, t, r_timesteps=r)


def apply_cfm_x0(model, x_t, t, r):
    eps = apply_cfm(model, x_t, t, r)
    return pred_x0_from_eps_float(model, x_t, t.float(), eps)


def ddim_transport(model, x_t, t, r, pred_x0):
    from models.networks.diffusion_networks.ldm_diffusion_util import extract_into_tensor

    a_t = extract_into_tensor(model.alphas_cumprod, t, x_t.shape)
    a_r = extract_into_tensor(model.alphas_cumprod, r, x_t.shape)
    eps_implied = (x_t - a_t.sqrt() * pred_x0) / (1.0 - a_t).sqrt()
    return a_r.sqrt() * pred_x0 + (1.0 - a_r).sqrt() * eps_implied


def ddim_transport_float(model, x_t, t, r, pred_x0):
    a_t = interp_schedule(model.alphas_cumprod, t.float(), x_t.shape)
    a_r = interp_schedule(model.alphas_cumprod, r.float(), x_t.shape)
    eps_implied = (x_t - a_t.sqrt() * pred_x0) / (1.0 - a_t).sqrt().clamp_min(1e-8)
    return a_r.sqrt() * pred_x0 + (1.0 - a_r).sqrt() * eps_implied


def normalized_to_index(model, tau):
    return tau * float(model.num_timesteps - 1)


def index_to_normalized(model, t):
    return t / float(model.num_timesteps - 1)


def sample_ddim_cfm_times(
    model,
    batch_size,
    device,
    instant_mix,
    max_gap=None,
    min_gap=1.0,
    time_sampler="curriculum",
    train_ddim_steps=None,
    ddim_endpoint_indices=None,
):
    if time_sampler == "ddim_endpoint_pair":
        from models.networks.diffusion_networks.ldm_diffusion_util import make_ddim_timesteps

        step_choices = train_ddim_steps or [4]
        t_chunks = []
        r_chunks = []
        for _ in range(batch_size):
            n_steps = int(step_choices[torch.randint(0, len(step_choices), (1,), device=device).item()])
            if n_steps < 1:
                raise ValueError(f"invalid train_ddim_steps value: {n_steps}")
            ddim_steps = make_ddim_timesteps(
                ddim_discr_method="uniform",
                num_ddim_timesteps=n_steps,
                num_ddpm_timesteps=model.num_timesteps,
                verbose=False,
            )
            endpoints = torch.as_tensor(
                np.concatenate([np.flip(ddim_steps).copy(), np.asarray([0])]),
                device=device,
                dtype=torch.float32,
            )
            if ddim_endpoint_indices:
                valid_indices = [int(i) for i in ddim_endpoint_indices if 0 <= int(i) < n_steps]
                if not valid_indices:
                    raise ValueError(f"no valid endpoint indices for n_steps={n_steps}: {ddim_endpoint_indices}")
                idx = valid_indices[torch.randint(0, len(valid_indices), (1,), device=device).item()]
            else:
                idx = torch.randint(0, n_steps, (1,), device=device).item()
            t_chunks.append(endpoints[idx])
            r_chunks.append(endpoints[idx + 1])
        t = torch.stack(t_chunks)
        r = torch.stack(r_chunks)
    else:
        t = 1.0 + torch.rand(batch_size, device=device) * (model.num_timesteps - 2)
    if time_sampler == "uniform_pair":
        r = 1.0 + torch.rand(batch_size, device=device) * (model.num_timesteps - 2)
        # DDIM runs from high-noise T toward clean 0, so enforce r <= t.
        t, r = torch.maximum(t, r), torch.minimum(t, r)
    elif time_sampler == "curriculum":
        if max_gap is None or max_gap <= 0:
            r = 1.0 + torch.rand(batch_size, device=device) * (model.num_timesteps - 2)
            t, r = torch.maximum(t, r), torch.minimum(t, r)
        else:
            gap_hi = torch.minimum(torch.full_like(t, float(max_gap)), t - 1.0)
            gap_lo = torch.minimum(torch.full_like(t, float(min_gap)), gap_hi)
            gap = gap_lo + torch.rand(batch_size, device=device) * (gap_hi - gap_lo).clamp_min(0.0)
            r = (t - gap).clamp_min(0.0)
    elif time_sampler == "ddim_endpoint_pair":
        pass
    else:
        raise ValueError(f"unknown time_sampler: {time_sampler}")
    mix = torch.rand(batch_size, device=device) < instant_mix
    r = torch.where(mix, t, r)
    return t, r


def sample_ddim_cfm_rollout_times(model, batch_size, device, train_ddim_steps=None, ddim_endpoint_indices=None):
    from models.networks.diffusion_networks.ldm_diffusion_util import make_ddim_timesteps

    step_choices = train_ddim_steps or [10]
    t_chunks = []
    m_chunks = []
    r_chunks = []
    for _ in range(batch_size):
        n_steps = int(step_choices[torch.randint(0, len(step_choices), (1,), device=device).item()])
        if n_steps < 2:
            raise ValueError(f"rollout FM needs at least 2 DDIM steps, got {n_steps}")
        ddim_steps = make_ddim_timesteps(
            ddim_discr_method="uniform",
            num_ddim_timesteps=n_steps,
            num_ddpm_timesteps=model.num_timesteps,
            verbose=False,
        )
        endpoints = torch.as_tensor(
            np.concatenate([np.flip(ddim_steps).copy(), np.asarray([0])]),
            device=device,
            dtype=torch.float32,
        )
        if ddim_endpoint_indices:
            valid_indices = [int(i) for i in ddim_endpoint_indices if 0 <= int(i) < n_steps - 1]
            if not valid_indices:
                raise ValueError(f"no valid rollout endpoint indices for n_steps={n_steps}: {ddim_endpoint_indices}")
            idx = valid_indices[torch.randint(0, len(valid_indices), (1,), device=device).item()]
        else:
            idx = torch.randint(0, n_steps - 1, (1,), device=device).item()
        t_chunks.append(endpoints[idx])
        m_chunks.append(endpoints[idx + 1])
        r_chunks.append(endpoints[idx + 2])
    return torch.stack(t_chunks), torch.stack(m_chunks), torch.stack(r_chunks)


def sample_ddim_cfm_multistep_path(model, device, train_ddim_steps=None, ddim_endpoint_indices=None, rollout_hops=3):
    from models.networks.diffusion_networks.ldm_diffusion_util import make_ddim_timesteps

    step_choices = train_ddim_steps or [10]
    n_steps = int(step_choices[torch.randint(0, len(step_choices), (1,), device=device).item()])
    if n_steps < 2:
        raise ValueError(f"multi-step rollout FM needs at least 2 DDIM steps, got {n_steps}")
    max_hops = max(2, int(rollout_hops))
    hops = min(max_hops, n_steps)
    ddim_steps = make_ddim_timesteps(
        ddim_discr_method="uniform",
        num_ddim_timesteps=n_steps,
        num_ddpm_timesteps=model.num_timesteps,
        verbose=False,
    )
    endpoints = torch.as_tensor(
        np.concatenate([np.flip(ddim_steps).copy(), np.asarray([0])]),
        device=device,
        dtype=torch.float32,
    )
    if ddim_endpoint_indices:
        valid_indices = [int(i) for i in ddim_endpoint_indices if 0 <= int(i) <= n_steps - hops]
        if not valid_indices:
            raise ValueError(
                f"no valid multi-step start indices for n_steps={n_steps}, rollout_hops={hops}: {ddim_endpoint_indices}"
            )
        start = valid_indices[torch.randint(0, len(valid_indices), (1,), device=device).item()]
    else:
        start = torch.randint(0, n_steps - hops + 1, (1,), device=device).item()
    return endpoints[start : start + hops + 1], torch.tensor(float(n_steps), device=device)


def ddim_cfm_jvp_loss(model, z, args):
    noise = torch.randn_like(z)
    bs = z.shape[0]
    t, r = sample_ddim_cfm_times(
        model,
        bs,
        z.device,
        args.instant_mix,
        args.max_gap,
        args.min_gap,
        args.time_sampler,
        args.train_ddim_steps,
        args.ddim_endpoint_indices,
    )
    x_t = q_sample_float(model, z, t, noise)

    pred = apply_cfm_x0(model, x_t, t, r)

    # Eq. 7 DDIM instantiation: target is stop-gradient and contains
    # J_x m · v and d_t m for m = x0_{t->r}(x).
    with torch.enable_grad():
        x_jvp = x_t.detach().requires_grad_(True)
        r_const = r.detach()
        if args.time_param == "normalized":
            t_jvp = index_to_normalized(model, t.detach()).requires_grad_(True)

            def jvp_time_to_index(t_in):
                return normalized_to_index(model, t_in)
        elif args.time_param == "index":
            t_jvp = t.detach().requires_grad_(True)

            def jvp_time_to_index(t_in):
                return t_in
        else:
            raise ValueError(f"unknown time_param: {args.time_param}")

        t_jvp_idx = jvp_time_to_index(t_jvp)
        a_t = interp_schedule(model.alphas_cumprod, t_jvp_idx, z.shape)
        a_r = interp_schedule(model.alphas_cumprod, r_const, z.shape)
        beta_t = interp_schedule(model.betas, t_jvp_idx, z.shape).clamp_min(1e-8)

        # Paper notation: (sqrt(alpha_bar_t) * x1 - alpha_bar_t * x) partial_x m.
        v_x = a_t.sqrt() * z.detach() - a_t * x_jvp.detach()

        def x0_field(x_in, t_in):
            return apply_cfm_x0(model, x_in, jvp_time_to_index(t_in), r_const)

        _, jvp_x = torch.autograd.functional.jvp(
            x0_field,
            (x_jvp, t_jvp),
            (v_x, torch.zeros_like(t_jvp)),
            create_graph=False,
            strict=False,
        )
        _, jvp_t = torch.autograd.functional.jvp(
            x0_field,
            (x_jvp, t_jvp),
            (torch.zeros_like(x_jvp), torch.ones_like(t_jvp)),
            create_graph=False,
            strict=False,
        )
        if args.time_param == "normalized":
            # jvp_t above is d m / d tau for tau=t/(T-1), while the DDPM
            # coefficient below is defined against the original timestep index.
            jvp_t = jvp_t / float(model.num_timesteps - 1)

        coeff = ((1.0 - a_t).sqrt() * a_r.sqrt() / ((1.0 - a_r).sqrt() * a_t.sqrt()).clamp_min(1e-8)) - 1.0
        if args.coeff_clip > 0:
            coeff = coeff.clamp(-args.coeff_clip, args.coeff_clip)
        dt_coeff = 2.0 * (1.0 - a_t) * (1.0 - beta_t) / beta_t
        target = coeff * (jvp_x - dt_coeff * jvp_t) + z.detach()
        if args.target_clip > 0:
            delta = target - z.detach()
            flat_norm = delta.flatten(1).norm(dim=1).clamp_min(1e-8)
            max_norm = torch.full_like(flat_norm, args.target_clip)
            scale = torch.minimum(torch.ones_like(flat_norm), max_norm / flat_norm)
            target = z.detach() + delta * scale.view(bs, *((1,) * (delta.dim() - 1)))
        target = target.detach()

    if args.loss_kind == "huber":
        loss_cfm = F.smooth_l1_loss(pred, target, beta=args.huber_beta)
    elif args.loss_kind == "mse":
        loss_cfm = F.mse_loss(pred, target)
    else:
        raise ValueError(f"unknown loss_kind: {args.loss_kind}")
    loss_instant = F.mse_loss(pred, z)
    return loss_cfm, {
        "loss_cfm": loss_cfm,
        "loss_instant_x0": loss_instant,
        "t_mean": t.mean(),
        "r_mean": r.mean(),
        "gap_mean": (t - r).mean(),
        "time_param_id": torch.tensor(1.0 if args.time_param == "normalized" else 0.0, device=z.device),
    }


def teacher_endpoint_distill_loss(model, z, args):
    """Distill original SDFusion DDIM endpoint transitions into forward_cfm."""
    noise = torch.randn_like(z)
    bs = z.shape[0]
    t, r = sample_ddim_cfm_times(
        model,
        bs,
        z.device,
        args.instant_mix,
        args.max_gap,
        args.min_gap,
        args.time_sampler,
        args.train_ddim_steps,
        args.ddim_endpoint_indices,
    )
    x_t = q_sample_float(model, z, t, noise)

    with torch.no_grad():
        teacher_eps = model.df.diffusion_net(x_t, t)
        teacher_x0 = pred_x0_from_eps_float(model, x_t, t, teacher_eps)
        teacher_xr = ddim_transport_float(model, x_t, t, r, teacher_x0)

    student_x0 = apply_cfm_x0(model, x_t, t, r)
    student_xr = ddim_transport_float(model, x_t, t, r, student_x0)

    loss_xr = F.mse_loss(student_xr, teacher_xr)
    loss_x0 = F.mse_loss(student_x0, teacher_x0)
    loss = args.w_xr * loss_xr + args.w_x0 * loss_x0
    return loss, {
        "loss_xr": loss_xr,
        "loss_x0": loss_x0,
        "t_mean": t.mean(),
        "r_mean": r.mean(),
        "gap_mean": (t - r).mean(),
    }


def ddim_endpoint_fm_loss(model, z, args):
    """Pure CMF endpoint flow matching using the analytic forward noising path."""
    noise = torch.randn_like(z)
    bs = z.shape[0]
    t, r = sample_ddim_cfm_times(
        model,
        bs,
        z.device,
        args.instant_mix,
        args.max_gap,
        args.min_gap,
        args.time_sampler,
        args.train_ddim_steps,
        args.ddim_endpoint_indices,
    )
    x_t = q_sample_float(model, z, t, noise)
    x_r_true = q_sample_float(model, z, r, noise)

    student_x0 = apply_cfm_x0(model, x_t, t, r)
    student_xr = ddim_transport_float(model, x_t, t, r, student_x0)

    if args.loss_kind == "huber":
        loss_xr = F.smooth_l1_loss(student_xr, x_r_true.detach(), beta=args.huber_beta)
        loss_x0 = F.smooth_l1_loss(student_x0, z.detach(), beta=args.huber_beta)
    elif args.loss_kind == "mse":
        loss_xr = F.mse_loss(student_xr, x_r_true.detach())
        loss_x0 = F.mse_loss(student_x0, z.detach())
    else:
        raise ValueError(f"unknown loss_kind: {args.loss_kind}")
    loss = args.w_xr * loss_xr + args.w_x0 * loss_x0
    return loss, {
        "loss_xr": loss_xr,
        "loss_x0": loss_x0,
        "t_mean": t.mean(),
        "r_mean": r.mean(),
        "gap_mean": (t - r).mean(),
    }


def ddim_rollout_fm_loss(model, z, args):
    """Pure CMF rollout correction: train on states produced by the current CMF."""
    noise = torch.randn_like(z)
    bs = z.shape[0]
    t, m, r = sample_ddim_cfm_rollout_times(
        model,
        bs,
        z.device,
        args.train_ddim_steps,
        args.ddim_endpoint_indices,
    )
    x_t = q_sample_float(model, z, t, noise)
    x_r_true = q_sample_float(model, z, r, noise)

    with torch.no_grad():
        first_x0 = apply_cfm_x0(model, x_t, t, m)
        x_m_hat = ddim_transport_float(model, x_t, t, m, first_x0)

    student_x0 = apply_cfm_x0(model, x_m_hat.detach(), m, r)
    x_r_hat = ddim_transport_float(model, x_m_hat.detach(), m, r, student_x0)

    if args.loss_kind == "huber":
        loss_xr = F.smooth_l1_loss(x_r_hat, x_r_true.detach(), beta=args.huber_beta)
        loss_x0 = F.smooth_l1_loss(student_x0, z.detach(), beta=args.huber_beta)
    elif args.loss_kind == "mse":
        loss_xr = F.mse_loss(x_r_hat, x_r_true.detach())
        loss_x0 = F.mse_loss(student_x0, z.detach())
    else:
        raise ValueError(f"unknown loss_kind: {args.loss_kind}")
    loss = args.w_xr * loss_xr + args.w_x0 * loss_x0
    return loss, {
        "loss_xr": loss_xr,
        "loss_x0": loss_x0,
        "t_mean": t.mean(),
        "mid_mean": m.mean(),
        "r_mean": r.mean(),
        "gap_mean": (t - r).mean(),
    }


def ddim_multistep_rollout_fm_loss(model, z, args):
    """Pure CMF correction from states after several current-CMF DDIM hops."""
    noise = torch.randn_like(z)
    bs = z.shape[0]
    path, n_steps = sample_ddim_cfm_multistep_path(
        model,
        z.device,
        args.train_ddim_steps,
        args.ddim_endpoint_indices,
        args.rollout_hops,
    )
    x = q_sample_float(model, z, path[0].repeat(bs), noise)

    with torch.no_grad():
        for i in range(path.numel() - 2):
            t_i = path[i].repeat(bs)
            r_i = path[i + 1].repeat(bs)
            pred_x0_i = apply_cfm_x0(model, x, t_i, r_i)
            x = ddim_transport_float(model, x, t_i, r_i, pred_x0_i)

    m = path[-2].repeat(bs)
    r = path[-1].repeat(bs)
    x_r_true = q_sample_float(model, z, r, noise)
    student_x0 = apply_cfm_x0(model, x.detach(), m, r)
    x_r_hat = ddim_transport_float(model, x.detach(), m, r, student_x0)

    if args.loss_kind == "huber":
        loss_xr = F.smooth_l1_loss(x_r_hat, x_r_true.detach(), beta=args.huber_beta)
        loss_x0 = F.smooth_l1_loss(student_x0, z.detach(), beta=args.huber_beta)
    elif args.loss_kind == "mse":
        loss_xr = F.mse_loss(x_r_hat, x_r_true.detach())
        loss_x0 = F.mse_loss(student_x0, z.detach())
    else:
        raise ValueError(f"unknown loss_kind: {args.loss_kind}")
    loss = args.w_xr * loss_xr + args.w_x0 * loss_x0
    return loss, {
        "loss_xr": loss_xr,
        "loss_x0": loss_x0,
        "start_t": path[0],
        "train_t": path[-2],
        "r_mean": path[-1],
        "gap_mean": path[0] - path[-1],
        "rollout_hops": torch.tensor(float(path.numel() - 1), device=z.device),
        "train_ddim_steps": n_steps,
    }


def ddim_randomhop_rollout_fm_loss(model, z, args):
    """Pure CMF rollout correction sampled from any hop in a multi-step path.

    `ddim_multistep_rollout_fm_loss` only trains the final hop after a fixed
    number of detached CMF hops. This variant samples the supervised hop inside
    the rollout window, so the model sees states after 0, 1, ..., K-1 accumulated
    CMF errors without keeping multiple gradient graphs in memory.
    """
    noise = torch.randn_like(z)
    bs = z.shape[0]
    path, n_steps = sample_ddim_cfm_multistep_path(
        model,
        z.device,
        args.train_ddim_steps,
        args.ddim_endpoint_indices,
        args.rollout_hops,
    )
    train_offset = torch.randint(0, path.numel() - 1, (1,), device=z.device).item()
    x = q_sample_float(model, z, path[0].repeat(bs), noise)

    with torch.no_grad():
        for i in range(train_offset):
            t_i = path[i].repeat(bs)
            r_i = path[i + 1].repeat(bs)
            pred_x0_i = apply_cfm_x0(model, x, t_i, r_i)
            x = ddim_transport_float(model, x, t_i, r_i, pred_x0_i)

    t = path[train_offset].repeat(bs)
    r = path[train_offset + 1].repeat(bs)
    x_r_true = q_sample_float(model, z, r, noise)
    student_x0 = apply_cfm_x0(model, x.detach(), t, r)
    x_r_hat = ddim_transport_float(model, x.detach(), t, r, student_x0)

    if args.loss_kind == "huber":
        loss_xr = F.smooth_l1_loss(x_r_hat, x_r_true.detach(), beta=args.huber_beta)
        loss_x0 = F.smooth_l1_loss(student_x0, z.detach(), beta=args.huber_beta)
    elif args.loss_kind == "mse":
        loss_xr = F.mse_loss(x_r_hat, x_r_true.detach())
        loss_x0 = F.mse_loss(student_x0, z.detach())
    else:
        raise ValueError(f"unknown loss_kind: {args.loss_kind}")
    loss = args.w_xr * loss_xr + args.w_x0 * loss_x0
    return loss, {
        "loss_xr": loss_xr,
        "loss_x0": loss_x0,
        "start_t": path[0],
        "train_t": path[train_offset],
        "r_mean": path[train_offset + 1],
        "gap_mean": path[0] - path[-1],
        "train_offset": torch.tensor(float(train_offset), device=z.device),
        "rollout_hops": torch.tensor(float(path.numel() - 1), device=z.device),
        "train_ddim_steps": n_steps,
    }


def ddim_allhop_rollout_fm_loss(model, z, args):
    """Pure CMF rollout correction supervised at every hop in one rollout window.

    Each supervised transition has its own gradient graph, but the state passed
    to the next hop is detached. This trains accumulated-error states without
    backpropagating through the full sampler chain.
    """
    noise = torch.randn_like(z)
    bs = z.shape[0]
    path, n_steps = sample_ddim_cfm_multistep_path(
        model,
        z.device,
        args.train_ddim_steps,
        args.ddim_endpoint_indices,
        args.rollout_hops,
    )
    x = q_sample_float(model, z, path[0].repeat(bs), noise)

    losses_xr = []
    losses_x0 = []
    for i in range(path.numel() - 1):
        t = path[i].repeat(bs)
        r = path[i + 1].repeat(bs)
        x_r_true = q_sample_float(model, z, r, noise)
        student_x0 = apply_cfm_x0(model, x.detach(), t, r)
        x_r_hat = ddim_transport_float(model, x.detach(), t, r, student_x0)

        if args.loss_kind == "huber":
            loss_xr_i = F.smooth_l1_loss(x_r_hat, x_r_true.detach(), beta=args.huber_beta)
            loss_x0_i = F.smooth_l1_loss(student_x0, z.detach(), beta=args.huber_beta)
        elif args.loss_kind == "mse":
            loss_xr_i = F.mse_loss(x_r_hat, x_r_true.detach())
            loss_x0_i = F.mse_loss(student_x0, z.detach())
        else:
            raise ValueError(f"unknown loss_kind: {args.loss_kind}")
        losses_xr.append(loss_xr_i)
        losses_x0.append(loss_x0_i)
        x = x_r_hat.detach()

    loss_xr = torch.stack(losses_xr).mean()
    loss_x0 = torch.stack(losses_x0).mean()
    loss = args.w_xr * loss_xr + args.w_x0 * loss_x0
    return loss, {
        "loss_xr": loss_xr,
        "loss_x0": loss_x0,
        "start_t": path[0],
        "end_r": path[-1],
        "gap_mean": path[0] - path[-1],
        "supervised_hops": torch.tensor(float(path.numel() - 1), device=z.device),
        "train_ddim_steps": n_steps,
    }


def ddim_endpoint_rollout_fm_loss(model, z, args):
    """Pure CMF mix of on-manifold endpoint FM and off-manifold rollout FM."""
    endpoint_loss, endpoint_aux = ddim_endpoint_fm_loss(model, z, args)
    rollout_loss, rollout_aux = ddim_rollout_fm_loss(model, z, args)
    loss = args.w_endpoint * endpoint_loss + args.w_rollout * rollout_loss
    aux = {
        "loss_endpoint": endpoint_loss,
        "loss_rollout": rollout_loss,
    }
    for key, value in endpoint_aux.items():
        aux[f"endpoint_{key}"] = value
    for key, value in rollout_aux.items():
        aux[f"rollout_{key}"] = value
    return loss, aux


def ddim_endpoint_randomhop_rollout_fm_loss(model, z, args):
    """Pure CMF mix of endpoint FM and random-hop off-manifold rollout FM."""
    endpoint_loss, endpoint_aux = ddim_endpoint_fm_loss(model, z, args)
    rollout_loss, rollout_aux = ddim_randomhop_rollout_fm_loss(model, z, args)
    loss = args.w_endpoint * endpoint_loss + args.w_rollout * rollout_loss
    aux = {
        "loss_endpoint": endpoint_loss,
        "loss_randomhop_rollout": rollout_loss,
    }
    for key, value in endpoint_aux.items():
        aux[f"endpoint_{key}"] = value
    for key, value in rollout_aux.items():
        aux[f"randomhop_{key}"] = value
    return loss, aux


def ddim_endpoint_multistep_rollout_fm_loss(model, z, args):
    """Pure CMF mix of endpoint FM and multi-step off-manifold rollout FM."""
    endpoint_loss, endpoint_aux = ddim_endpoint_fm_loss(model, z, args)
    rollout_loss, rollout_aux = ddim_multistep_rollout_fm_loss(model, z, args)
    loss = args.w_endpoint * endpoint_loss + args.w_rollout * rollout_loss
    aux = {
        "loss_endpoint": endpoint_loss,
        "loss_multistep_rollout": rollout_loss,
    }
    for key, value in endpoint_aux.items():
        aux[f"endpoint_{key}"] = value
    for key, value in rollout_aux.items():
        aux[f"multistep_{key}"] = value
    return loss, aux


def ddim_endpoint_allhop_rollout_fm_loss(model, z, args):
    """Pure CMF mix of endpoint FM and all-hop off-manifold rollout FM."""
    endpoint_loss, endpoint_aux = ddim_endpoint_fm_loss(model, z, args)
    rollout_loss, rollout_aux = ddim_allhop_rollout_fm_loss(model, z, args)
    loss = args.w_endpoint * endpoint_loss + args.w_rollout * rollout_loss
    aux = {
        "loss_endpoint": endpoint_loss,
        "loss_allhop_rollout": rollout_loss,
    }
    for key, value in endpoint_aux.items():
        aux[f"endpoint_{key}"] = value
    for key, value in rollout_aux.items():
        aux[f"allhop_{key}"] = value
    return loss, aux


def make_loader(args):
    from datasets.snet_dataset import ShapeNetDataset

    opt = make_opt(args, is_train=False)
    ds = ShapeNetDataset()
    ds.initialize(opt, phase="train", cat=args.cat, res=64)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def next_batch(loader):
    while True:
        for batch in loader:
            yield batch


def train(args):
    out_dir = args.out_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    model = load_uncond_model(args)
    if args.resume_cfm:
        load_cfm_ckpt(model, Path(args.resume_cfm))
    model.df.train()
    if args.loss_type == "ddim_cfm_jvp":
        disable_diffusion_checkpointing(model)
        print("[*] disabled internal gradient checkpointing for DDIM-CFM JVP loss", flush=True)
    model.vqvae.eval()
    for p in model.vqvae.parameters():
        p.requires_grad_(False)
    if args.train_scope == "r_embed":
        for p in model.df.parameters():
            p.requires_grad_(False)
        for p in model.df.diffusion_net.time_embed_r.parameters():
            p.requires_grad_(True)
        disable_diffusion_checkpointing(model)
        print("[*] train_scope=r_embed: frozen diffusion backbone; training only time_embed_r", flush=True)
        print("[*] disabled internal gradient checkpointing for frozen-backbone training", flush=True)
    elif args.train_scope == "r_embed_out":
        for p in model.df.parameters():
            p.requires_grad_(False)
        for p in model.df.diffusion_net.time_embed_r.parameters():
            p.requires_grad_(True)
        for p in model.df.diffusion_net.out.parameters():
            p.requires_grad_(True)
        disable_diffusion_checkpointing(model)
        print("[*] train_scope=r_embed_out: training time_embed_r and final output layer", flush=True)
        print("[*] disabled internal gradient checkpointing for partial-backbone training", flush=True)
    elif args.train_scope == "r_embed_tail":
        for p in model.df.parameters():
            p.requires_grad_(False)
        for p in model.df.diffusion_net.time_embed_r.parameters():
            p.requires_grad_(True)
        tail_blocks = max(1, int(args.train_tail_blocks))
        for block in list(model.df.diffusion_net.output_blocks)[-tail_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for p in model.df.diffusion_net.out.parameters():
            p.requires_grad_(True)
        disable_diffusion_checkpointing(model)
        print(
            f"[*] train_scope=r_embed_tail: training time_embed_r, last {tail_blocks} output blocks, and final output layer",
            flush=True,
        )
        print("[*] disabled internal gradient checkpointing for tail-block training", flush=True)
    elif args.train_scope == "all":
        for p in model.df.parameters():
            p.requires_grad_(True)
        print("[*] train_scope=all: training full diffusion model", flush=True)
    else:
        raise ValueError(f"unknown train_scope: {args.train_scope}")

    loader = make_loader(args)
    batches = next_batch(loader)
    trainable = [p for p in model.df.parameters() if p.requires_grad]
    print(f"[*] trainable parameters: {sum(p.numel() for p in trainable)}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, foreach=False)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    log_path = out_dir / "train_metrics.jsonl"
    t_start = time.time()
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    for step in range(1, args.iters + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_for_log = 0.0
        aux_for_log = {}
        for micro_step in range(grad_accum_steps):
            batch = next(batches)
            x = batch["sdf"].cuda(non_blocking=True)
            with torch.no_grad():
                z = model.vqvae(x, forward_no_quant=True, encode_only=True)

            bs = z.shape[0]
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.loss_type == "teacher_endpoint_distill":
                    loss, aux = teacher_endpoint_distill_loss(model, z, args)
                elif args.loss_type == "ddim_endpoint_fm":
                    loss, aux = ddim_endpoint_fm_loss(model, z, args)
                elif args.loss_type == "ddim_rollout_fm":
                    loss, aux = ddim_rollout_fm_loss(model, z, args)
                elif args.loss_type == "ddim_multistep_rollout_fm":
                    loss, aux = ddim_multistep_rollout_fm_loss(model, z, args)
                elif args.loss_type == "ddim_randomhop_rollout_fm":
                    loss, aux = ddim_randomhop_rollout_fm_loss(model, z, args)
                elif args.loss_type == "ddim_endpoint_rollout_fm":
                    loss, aux = ddim_endpoint_rollout_fm_loss(model, z, args)
                elif args.loss_type == "ddim_endpoint_randomhop_rollout_fm":
                    loss, aux = ddim_endpoint_randomhop_rollout_fm_loss(model, z, args)
                elif args.loss_type == "ddim_endpoint_multistep_rollout_fm":
                    loss, aux = ddim_endpoint_multistep_rollout_fm_loss(model, z, args)
                elif args.loss_type == "ddim_endpoint_allhop_rollout_fm":
                    loss, aux = ddim_endpoint_allhop_rollout_fm_loss(model, z, args)
                elif args.loss_type == "ddim_cfm_jvp":
                    loss, aux = ddim_cfm_jvp_loss(model, z, args)
                elif args.loss_type == "legacy":
                    t = torch.randint(1, model.num_timesteps, (bs,), device="cuda").long()
                    mix = torch.rand(bs, device="cuda") < args.instant_mix
                    r_random = torch.floor(torch.rand(bs, device="cuda") * t.float()).long()
                    r = torch.where(mix, t, r_random)
                    noise = torch.randn_like(z)
                    x_t = q_sample(model, z, t, noise)

                    eps_tr = apply_cfm(model, x_t, t, r)
                    loss_eps = F.mse_loss(eps_tr, noise)
                    pred_x0_tr = pred_x0_from_eps(model, x_t, t, eps_tr)
                    loss_x0 = F.mse_loss(pred_x0_tr, z)

                    # Long-range DDIM consistency: t -> r prediction should agree with the
                    # model's instantaneous prediction at the reached r state.
                    x_r = ddim_transport(model, x_t, t, r, pred_x0_tr.detach())
                    eps_rr = apply_cfm(model, x_r, r, r)
                    pred_x0_rr = pred_x0_from_eps(model, x_r, r, eps_rr)
                    loss_cons = F.mse_loss(pred_x0_rr, pred_x0_tr.detach())

                    loss = args.w_eps * loss_eps + args.w_x0 * loss_x0 + args.w_cons * loss_cons
                    aux = {
                        "loss_eps": loss_eps,
                        "loss_x0": loss_x0,
                        "loss_cons": loss_cons,
                    }
                else:
                    raise ValueError(f"unknown loss_type: {args.loss_type}")

            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()
            loss_for_log += float(loss.detach()) / grad_accum_steps
            for key, value in aux.items():
                aux_for_log[key] = aux_for_log.get(key, 0.0) + float(value.detach()) / grad_accum_steps

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.df.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.print_freq == 0:
            row = {
                "step": step,
                "elapsed_sec": round(time.time() - t_start, 3),
                "loss_type": args.loss_type,
                "loss": round(loss_for_log, 6),
                "grad_accum_steps": grad_accum_steps,
                "grad_norm": round(float(grad_norm), 6),
                "gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
            }
            for key, value in aux_for_log.items():
                row[key] = round(value, 6)
            print(json.dumps(row), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if step % args.save_freq == 0 or step == args.iters:
            save_cfm_ckpt(model, out_dir, step)

    print(f"[*] train complete: {out_dir}", flush=True)


@torch.no_grad()
def sample_cfm_latent(model, steps: int, batch_size: int):
    from models.networks.diffusion_networks.ldm_diffusion_util import make_ddim_timesteps

    shape = (batch_size, *model.z_shape)
    x = torch.randn(shape, device="cuda")
    ddim_steps = make_ddim_timesteps(
        ddim_discr_method="uniform",
        num_ddim_timesteps=steps,
        num_ddpm_timesteps=model.num_timesteps,
        verbose=False,
    )
    endpoints = torch.as_tensor(
        np.concatenate([np.flip(ddim_steps).copy(), np.asarray([0])]),
        device="cuda",
        dtype=torch.float32,
    )
    for i in range(steps):
        t = endpoints[i].repeat(batch_size)
        r = endpoints[i + 1].repeat(batch_size)
        pred_x0 = apply_cfm_x0(model, x, t, r)
        x = ddim_transport_float(model, x, t, r, pred_x0)
    return x


def eval_model(args):
    model = None
    if args.eval_only in ("all", "baseline"):
        model = load_uncond_model(args)
    cfm_model = load_uncond_model(args)
    load_cfm_ckpt(cfm_model, Path(args.cfm_ckpt))
    if model is not None:
        model.df.eval()
    cfm_model.df.eval()

    out_dir = args.out_dir / args.name / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seed_all(args.seed)
    if model is not None:
        _ = model.uncond(ngen=1, ddim_steps=1, ddim_eta=0.0)
    _ = cfm_model.vqvae_module.decode_no_quant(sample_cfm_latent(cfm_model, 1, 1))
    torch.cuda.synchronize()
    for steps in args.eval_steps:
        if model is not None:
            torch.cuda.reset_peak_memory_stats()
            seed_all(args.seed)
            t0 = time.time()
            gen = model.uncond(ngen=args.eval_batch, ddim_steps=steps, ddim_eta=0.0)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            row = {
                "sampler": "baseline_notebook_uncond",
                "steps": steps,
                "batch": args.eval_batch,
                "elapsed_sec": round(elapsed, 4),
                "sec_per_shape": round(elapsed / args.eval_batch, 4),
                "sdf_mean": round(float(gen.mean()), 6),
                "sdf_std": round(float(gen.std()), 6),
                "finite": bool(torch.isfinite(gen).all()),
                "gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
                "note": "Uses SDFusionModel.uncond(), the same public inference path used by demo_uncond_shape_comp.ipynb.",
            }
            rows.append(row)
            torch.save({"sdf": gen.detach().cpu(), "row": row}, out_dir / f"baseline_notebook_uncond_{steps}step.pt")
            print(json.dumps(row), flush=True)

        if args.eval_only in ("all", "cfm"):
            torch.cuda.reset_peak_memory_stats()
            seed_all(args.seed)
            t0 = time.time()
            z = sample_cfm_latent(cfm_model, steps, args.eval_batch)
            gen = cfm_model.vqvae_module.decode_no_quant(z)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            row = {
                "sampler": "cfm_custom_tr",
                "steps": steps,
                "batch": args.eval_batch,
                "elapsed_sec": round(elapsed, 4),
                "sec_per_shape": round(elapsed / args.eval_batch, 4),
                "latent_mean": round(float(z.mean()), 6),
                "latent_std": round(float(z.std()), 6),
                "sdf_mean": round(float(gen.mean()), 6),
                "sdf_std": round(float(gen.std()), 6),
                "finite": bool(torch.isfinite(gen).all()),
                "gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
            }
            rows.append(row)
            torch.save({"latent": z.detach().cpu(), "sdf": gen.detach().cpu(), "row": row}, out_dir / f"cfm_custom_tr_{steps}step.pt")
            print(json.dumps(row), flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[*] eval complete: {out_dir}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    p.add_argument("--out-dir", type=Path, default=Path("runs/cfm_sdfusion"))
    p.add_argument("--name", default="cfm-chair-stage1-smoke")
    p.add_argument("--dataroot", default="data")
    p.add_argument("--cat", default="chair")
    p.add_argument("--base-ckpt", default="saved_ckpt/sdfusion-snet-all.pth")
    p.add_argument("--vq-ckpt", default="saved_ckpt/vqvae-snet-all.pth")
    p.add_argument("--resume-cfm", default=None)
    p.add_argument("--cfm-ckpt", default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-dataset-size", type=int, default=256)
    p.add_argument("--trunc-thres", type=float, default=0.2)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument(
        "--loss-type",
        choices=[
            "ddim_cfm_jvp",
            "ddim_endpoint_fm",
            "ddim_rollout_fm",
            "ddim_multistep_rollout_fm",
            "ddim_randomhop_rollout_fm",
            "ddim_endpoint_rollout_fm",
            "ddim_endpoint_randomhop_rollout_fm",
            "ddim_endpoint_multistep_rollout_fm",
            "ddim_endpoint_allhop_rollout_fm",
            "teacher_endpoint_distill",
            "legacy",
        ],
        default="ddim_cfm_jvp",
    )
    p.add_argument("--train-scope", choices=["all", "r_embed", "r_embed_out", "r_embed_tail"], default="all")
    p.add_argument("--train-tail-blocks", type=int, default=1)
    p.add_argument("--r-cond-mode", choices=["avg", "delta_add"], default="avg")
    p.add_argument("--time-param", choices=["index", "normalized"], default="index")
    p.add_argument("--time-sampler", choices=["curriculum", "uniform_pair", "ddim_endpoint_pair"], default="curriculum")
    p.add_argument("--train-ddim-steps", type=int, nargs="+", default=[4])
    p.add_argument("--ddim-endpoint-indices", type=int, nargs="+", default=None)
    p.add_argument("--rollout-hops", type=int, default=3)
    p.add_argument("--instant-mix", type=float, default=0.5)
    p.add_argument("--max-gap", type=float, default=0.0)
    p.add_argument("--min-gap", type=float, default=1.0)
    p.add_argument("--coeff-clip", type=float, default=0.0)
    p.add_argument("--target-clip", type=float, default=0.0)
    p.add_argument("--loss-kind", choices=["mse", "huber"], default="mse")
    p.add_argument("--huber-beta", type=float, default=1.0)
    p.add_argument("--w-eps", type=float, default=1.0)
    p.add_argument("--w-xr", type=float, default=1.0)
    p.add_argument("--w-x0", type=float, default=0.25)
    p.add_argument("--w-cons", type=float, default=0.25)
    p.add_argument("--w-endpoint", type=float, default=1.0)
    p.add_argument("--w-rollout", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--print-freq", type=int, default=10)
    p.add_argument("--save-freq", type=int, default=100)
    p.add_argument("--eval-steps", type=int, nargs="+", default=[4, 10, 50])
    p.add_argument("--eval-batch", type=int, default=2)
    p.add_argument("--eval-only", choices=["all", "baseline", "cfm"], default="all")
    args = p.parse_args()
    args.repo_root = args.repo_root.resolve()
    if not args.out_dir.is_absolute():
        args.out_dir = args.repo_root / args.out_dir
    if args.mode == "eval" and not args.cfm_ckpt:
        p.error("--cfm-ckpt is required for eval")
    return args


def main():
    args = parse_args()
    setup_repo(args.repo_root)
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    if args.mode == "train":
        train(args)
    else:
        eval_model(args)


if __name__ == "__main__":
    main()
