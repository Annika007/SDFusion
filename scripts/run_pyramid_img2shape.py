import argparse
from pathlib import Path
import sys

import torch
import torchvision.utils as vutils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.demo_util import SDFusionImage2ShapeOpt
from utils.util_3d import render_sdf
from models.base_model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Run SDFusion img2shape with pyramid DDIM sampling')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--mask', required=True, help='Input mask path')
    parser.add_argument('--out_dir', default='demo_results_wzy/pyramid_img2shape', help='Output directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--ddim_steps', type=int, default=4, help='DDIM steps')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--uc_scale', type=float, default=1.0, help='Classifier-free guidance scale')
    parser.add_argument('--pyramid_list', type=str, default='1,1,2,2', help='Comma-separated pyramid schedule')
    parser.add_argument('--pyramid_interp_mode', type=str, default='trilinear', choices=['nearest', 'bilinear', 'trilinear'], help='Interpolation mode')
    parser.add_argument('--pyramid_use_up_v2', action='store_true', help='Use extra upsample re-noise step when scale changes')
    parser.add_argument('--ckpt', type=str, default='saved_ckpt/sdfusion-img2shape.pth', help='Diffusion checkpoint')
    parser.add_argument('--vq_ckpt', type=str, default='saved_ckpt/vqvae-snet-all.pth', help='VQVAE checkpoint')
    parser.add_argument('--dataroot', type=str, default='data', help='Dataset root used by the repo')
    parser.add_argument('--render_size', type=int, default=256, help='Rendered image size')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pyramid_list = [int(item.strip()) for item in args.pyramid_list.split(',') if item.strip()]
    if not pyramid_list:
        raise ValueError('pyramid_list is empty')

    opt = SDFusionImage2ShapeOpt(gpu_ids=args.gpu_id, seed=args.seed)
    opt.init_dset_args(dataroot=args.dataroot, dataset_mode='pix3d_img2shape', cat='all', res=64, cached_dir=None)
    opt.init_model_args(ckpt_path=args.ckpt, vq_ckpt_path=args.vq_ckpt)

    model = create_model(opt)
    model.gen_df = model.img2shape(
        args.image,
        args.mask,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        uc_scale=args.uc_scale,
        pyramid_list=pyramid_list,
        pyramid_interp_mode=args.pyramid_interp_mode,
        pyramid_use_up_v2=args.pyramid_use_up_v2,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = render_sdf(model.renderer, model.gen_df, render_imsize=args.render_size)
    vutils.save_image(rendered, out_dir / 'pyramid_img2shape_render.png', nrow=1)
    torch.save(model.gen_df.detach().cpu(), out_dir / 'pyramid_img2shape_tensor.pt')

    print(f'[*] saved render to {out_dir / "pyramid_img2shape_render.png"}')
    print(f'[*] saved tensor to {out_dir / "pyramid_img2shape_tensor.pt"}')


if __name__ == '__main__':
    main()
