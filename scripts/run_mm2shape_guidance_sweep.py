#!/usr/bin/env python3
"""Guidance sweep for partial-shape + text/image multimodal generation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.base_model import create_model
from utils.demo_util import SDFusionMM2ShapeOpt, preprocess_image
from utils.util_3d import read_sdf, sdf_to_mesh, save_mesh_as_gif, render_sdf


DEFAULT_PROMPTS = [
    "chair with one leg",
    "a wooden chair",
    "an office chair with armrests",
    "a rounded chair with a backrest",
]


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len] or "prompt"


def export_mesh(mesh, out_path: Path) -> None:
    verts = mesh.verts_list()[0].detach().cpu().numpy()
    faces = mesh.faces_list()[0].detach().cpu().numpy()
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tri.export(out_path)


def load_sdf_tensor(path: str) -> torch.Tensor:
    suffix = Path(path).suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return read_sdf(path, resolution=64)
    sdf = torch.load(path, map_location="cpu")
    if isinstance(sdf, dict):
        for key in ("sdf", "pc_sdf_sample", "data", "tensor"):
            if key in sdf:
                sdf = sdf[key]
                break
    sdf = torch.as_tensor(sdf).float()
    if sdf.dim() == 3:
        sdf = sdf.unsqueeze(0).unsqueeze(0)
    elif sdf.dim() == 4:
        sdf = sdf.unsqueeze(0)
    if sdf.dim() != 5:
        raise ValueError(f"Unsupported SDF tensor shape: {tuple(sdf.shape)}")
    return sdf


def load_pix3d_record(dataroot: str, category: str, image_id: str) -> dict:
    pix3d_json = Path(dataroot) / "pix3d" / "pix3d.json"
    with open(pix3d_json, "r") as f:
        records = json.load(f)
    wanted = f"img/{category}/{image_id}"
    for rec in records:
        if rec.get("category") != category:
            continue
        if rec.get("img", "").endswith(wanted) or rec.get("img", "").endswith(f"{wanted}.jpg") or rec.get("img", "").endswith(f"{wanted}.png") or rec.get("img", "").endswith(f"{wanted}.jpeg"):
            return rec
    raise FileNotFoundError(f"Could not find Pix3D record for {wanted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep mm2shape txt/img guidance weights.")
    parser.add_argument("--category", default="chair", help="Pix3D category.")
    parser.add_argument("--image_id", default="1858", help="Pix3D image id, for example 1858.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Optional text file with one prompt per line.")
    parser.add_argument("--prompts", nargs="*", default=None, help="Prompts to sweep. Defaults to a short built-in list.")
    parser.add_argument("--txt_scales", nargs="*", type=float, default=[0.0, 0.5, 1.0, 2.0], help="Text guidance weights.")
    parser.add_argument("--img_scales", nargs="*", type=float, default=[0.0, 0.5, 1.0, 2.0], help="Image guidance weights.")
    parser.add_argument("--mask_mode", default="top", help="Partial-shape mask mode.")
    parser.add_argument("--sdf_path", default="demo_data/chair-IKEA-FUSION.h5", help="Input partial SDF path.")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps.")
    parser.add_argument("--ddim_eta", type=float, default=0.1, help="DDIM eta.")
    parser.add_argument("--uc_scale", type=float, default=5.0, help="Overall classifier-free guidance scale.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/FinalProject/ybr/data", help="Dataset root.")
    parser.add_argument("--ckpt", type=str, default="/root/autodl-tmp/FinalProject/ybr/DiffusionShapeNet/SDFusion/saved_ckpt/sdfusion-mm2shape.pth", help="Diffusion checkpoint.")
    parser.add_argument("--vq_ckpt", type=str, default="saved_ckpt/vqvae-snet-all.pth", help="VQVAE checkpoint.")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if prompts:
            return prompts
    if args.prompts:
        prompts = [p.strip() for p in args.prompts if p.strip()]
        if prompts:
            return prompts
    return DEFAULT_PROMPTS


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompts = load_prompts(args)
    rec = load_pix3d_record(args.dataroot, args.category, args.image_id)
    image_path = Path(args.dataroot) / "pix3d" / rec["img"]
    mask_path = Path(args.dataroot) / "pix3d" / rec["mask"]
    model_id = Path(rec["model"]).parent.name

    sdf = load_sdf_tensor(args.sdf_path).clamp(-0.2, 0.2)
    rend_sdf = None

    opt = SDFusionMM2ShapeOpt(gpu_ids=args.gpu_id, seed=args.seed)
    opt.init_dset_args(dataroot=args.dataroot, dataset_mode="snet_mm2shape", cat="all", res=64, cached_dir=None)
    opt.init_model_args(ckpt_path=args.ckpt, vq_ckpt_path=args.vq_ckpt)
    model = create_model(opt)

    out_root = ROOT.parent / "results" / "mm2shape_guidance_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    img_vis, img_clean = preprocess_image(str(image_path), str(mask_path))
    img_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((256, 256)),
    ])
    img_clean = img_tf(img_clean).unsqueeze(0)

    summary = []
    for prompt in prompts:
        prompt_slug = slugify(prompt)
        test_data = {
            "sdf": sdf,
            "img": img_clean,
            "text": [prompt],
        }

        for txt_scale in args.txt_scales:
            for img_scale in args.img_scales:
                run_dir = out_root / args.category / prompt_slug / model_id / f"txt_{txt_scale:g}_img_{img_scale:g}"
                run_dir.mkdir(parents=True, exist_ok=True)

                model.mm_inference(
                    test_data,
                    mask_mode=args.mask_mode,
                    ddim_steps=args.ddim_steps,
                    ddim_eta=args.ddim_eta,
                    uc_scale=args.uc_scale,
                    txt_scale=txt_scale,
                    img_scale=img_scale,
                )

                sdf_gen = model.gen_df
                mesh = sdf_to_mesh(sdf_gen)
                if mesh is None:
                    raise RuntimeError(f"Failed to convert mm2shape output for prompt={prompt}")

                save_mesh_as_gif(model.renderer, mesh, nrow=1, out_name=str(run_dir / "pred.gif"))
                torch.save(sdf_gen.detach().cpu(), run_dir / "pred_tensor.pt")
                export_mesh(mesh, run_dir / "pred.obj")

                if rend_sdf is None:
                    rend_sdf = render_sdf(model.renderer, sdf.to(model.device), render_imsize=256)
                    torch.save(rend_sdf, run_dir / "input_shape_render.pt")

                meta = {
                    "category": args.category,
                    "model_id": model_id,
                    "image_id": args.image_id,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "prompt": prompt,
                    "txt_scale": txt_scale,
                    "img_scale": img_scale,
                    "uc_scale": args.uc_scale,
                    "ddim_steps": args.ddim_steps,
                    "ddim_eta": args.ddim_eta,
                    "mask_mode": args.mask_mode,
                }
                with open(run_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2)
                summary.append(meta)
                print(f"[*] saved {run_dir}")

    with open(out_root / f"{args.category}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
