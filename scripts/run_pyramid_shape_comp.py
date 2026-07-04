import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.base_model import create_model
from utils.demo_util import SDFusionOpt
from utils.util_3d import read_sdf, render_sdf, sdf_to_mesh, save_mesh_as_gif


def parse_args():
    parser = argparse.ArgumentParser(description='Run SDFusion shape completion with pyramid DDIM sampling')
    parser.add_argument('--sdf_path', required=True, help='Input SDF file (.h5, .pt, .pth, or .npy)')
    parser.add_argument('--out_dir', default='demo_results_wzy/pyramid_shape_comp', help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--ddim_steps', type=int, default=4, help='DDIM steps')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--scale', type=float, default=1.0, help='Unconditional guidance scale')
    parser.add_argument('--pyramid_list', type=str, default='1,1,2,2', help='Comma-separated pyramid schedule')
    parser.add_argument('--pyramid_interp_mode', type=str, default='trilinear',
                        choices=['nearest', 'bilinear', 'trilinear'], help='Interpolation mode')
    parser.add_argument('--pyramid_use_up_v2', action='store_true',
                        help='Use extra upsample re-noise step when scale changes')
    parser.add_argument('--ckpt', type=str, default='saved_ckpt/sdfusion-snet-all.pth',
                        help='Diffusion checkpoint')
    parser.add_argument('--vq_ckpt', type=str, default='saved_ckpt/vqvae-snet-all.pth',
                        help='VQVAE checkpoint')
    parser.add_argument('--dataroot', type=str, default='data', help='Dataset root used by the repo')
    parser.add_argument('--render_size', type=int, default=256, help='Rendered image size')
    parser.add_argument('--clip_sdf', type=float, default=0.2, help='Clamp loaded SDF values to [-clip_sdf, clip_sdf]')
    parser.add_argument('--x_min', type=float, default=-1.0, help='Missing region x min')
    parser.add_argument('--x_max', type=float, default=1.0, help='Missing region x max')
    parser.add_argument('--y_min', type=float, default=0.0, help='Missing region y min')
    parser.add_argument('--y_max', type=float, default=1.0, help='Missing region y max')
    parser.add_argument('--z_min', type=float, default=-1.0, help='Missing region z min')
    parser.add_argument('--z_max', type=float, default=1.0, help='Missing region z max')
    parser.add_argument('--output_level', type=float, default=0.02, help='Marching cubes level for visualization')
    return parser.parse_args()


def load_sdf_tensor(path: str, resolution: int = 64):
    suffix = Path(path).suffix.lower()
    if suffix in {'.h5', '.hdf5'}:
        return read_sdf(path, resolution=resolution)
    if suffix in {'.pt', '.pth'}:
        sdf = torch.load(path, map_location='cpu')
        if isinstance(sdf, dict):
            for key in ('sdf', 'pc_sdf_sample', 'data', 'tensor'):
                if key in sdf:
                    sdf = sdf[key]
                    break
        sdf = torch.as_tensor(sdf).float()
    elif suffix == '.npy':
        sdf = torch.from_numpy(np.load(path)).float()
    else:
        raise ValueError(f'Unsupported SDF file type: {path}')

    if sdf.dim() == 3:
        sdf = sdf.unsqueeze(0).unsqueeze(0)
    elif sdf.dim() == 4:
        sdf = sdf.unsqueeze(0)
    elif sdf.dim() != 5:
        raise ValueError(f'Unsupported SDF tensor shape: {tuple(sdf.shape)}')
    return sdf


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pyramid_list = [int(item.strip()) for item in args.pyramid_list.split(',') if item.strip()]
    if not pyramid_list:
        raise ValueError('pyramid_list is empty')

    sdf = load_sdf_tensor(args.sdf_path)
    sdf = sdf.clamp(-args.clip_sdf, args.clip_sdf)

    opt = SDFusionOpt(gpu_ids=args.gpu_id, seed=args.seed)
    opt.init_dset_args(dataroot=args.dataroot, dataset_mode='snet', cat='all', res=sdf.shape[-1], cached_dir=None)
    opt.init_model_args(ckpt_path=args.ckpt, vq_ckpt_path=args.vq_ckpt)

    model = create_model(opt)

    xyz_dict = {
        'x': (args.x_min, args.x_max),
        'y': (args.y_min, args.y_max),
        'z': (args.z_min, args.z_max),
    }

    output_shape_comp = model.shape_comp(
        sdf.to(model.device),
        xyz_dict,
        ngen=1,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        scale=args.scale,
        pyramid_list=pyramid_list,
        pyramid_interp_mode=args.pyramid_interp_mode,
        pyramid_use_up_v2=args.pyramid_use_up_v2,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    partial = getattr(model, 'x_part', None)
    missing = getattr(model, 'x_missing', None)
    if partial is not None:
        partial_render = render_sdf(model.renderer, partial, render_imsize=args.render_size)
        torch.save(partial.detach().cpu(), out_dir / 'pyramid_shape_comp_partial.pt')
        torch.save(missing.detach().cpu(), out_dir / 'pyramid_shape_comp_missing.pt')
        torch.save(partial_render, out_dir / 'pyramid_shape_comp_partial_render.pt')

    mesh_shape_comp = sdf_to_mesh(output_shape_comp, level=args.output_level)
    sc_output_name = out_dir / 'pyramid_shape_comp_output.gif'
    save_mesh_as_gif(model.renderer, mesh_shape_comp, nrow=1, out_name=str(sc_output_name))

    rendered = render_sdf(model.renderer, output_shape_comp, render_imsize=args.render_size)
    torch.save(output_shape_comp.detach().cpu(), out_dir / 'pyramid_shape_comp_tensor.pt')
    torch.save(rendered, out_dir / 'pyramid_shape_comp_render.pt')

    print(f'[*] saved render to {out_dir / "pyramid_shape_comp_render.pt"}')
    print(f'[*] saved tensor to {out_dir / "pyramid_shape_comp_tensor.pt"}')
    print(f'[*] saved gif to {sc_output_name}')


if __name__ == '__main__':
    main()
