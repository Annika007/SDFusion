#!/usr/bin/env python3
"""Batch sweep for txt2shape prompts and guidance scales."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.base_model import create_model
from utils.demo_util import SDFusionText2ShapeOpt
from utils.util_3d import sdf_to_mesh, save_mesh_as_gif


DEFAULT_PROMPTS = [
    "A rocking chair",
    "A wooden chair with armrests",
    "A simple office chair",
    "A round table with four legs",
    "A modern sofa",
    "A tall lamp",
]


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len] or "prompt"


def export_batch_meshes(mesh, out_dir: Path, basename: str) -> None:
    verts_list = mesh.verts_list()
    faces_list = mesh.faces_list()
    for i, (verts, faces) in enumerate(zip(verts_list, faces_list)):
        tri = trimesh.Trimesh(
            vertices=verts.detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
            process=False,
        )
        tri.export(out_dir / f"{basename}_{i:02d}.obj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep txt2shape prompts and guidance scales.")
    parser.add_argument("--out_root", default="results/txt2shape_sweep", help="Output root directory.")
    parser.add_argument("--prompts", nargs="*", default=None, help="Prompts to run. If omitted, built-in prompts are used.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Optional text file with one prompt per line.")
    parser.add_argument("--uc_scales", nargs="*", type=float, default=[2.0, 3.0, 5.0, 7.0], help="Guidance scales to sweep.")
    parser.add_argument("--ngen", type=int, default=4, help="Number of samples per prompt/scale.")
    parser.add_argument("--ddim_steps", type=int, default=100, help="DDIM steps.")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/FinalProject/ybr/data", help="Dataset root.")
    parser.add_argument("--ckpt", type=str, default="saved_ckpt/sdfusion-txt2shape.pth", help="Diffusion checkpoint.")
    parser.add_argument("--vq_ckpt", type=str, default="saved_ckpt/vqvae-snet-all.pth", help="VQVAE checkpoint.")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = []
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompts:
        prompts = [p.strip() for p in args.prompts if p.strip()]
    if not prompts:
        prompts = DEFAULT_PROMPTS
    return prompts


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompts = load_prompts(args)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    opt = SDFusionText2ShapeOpt(gpu_ids=args.gpu_id, seed=args.seed)
    opt.init_dset_args(dataroot=args.dataroot, dataset_mode="text2shape", cat="all", res=64, cached_dir=None)
    opt.init_model_args(ckpt_path=args.ckpt, vq_ckpt_path=args.vq_ckpt)
    model = create_model(opt)

    runs = []
    for prompt in prompts:
        prompt_slug = slugify(prompt)
        for uc_scale in args.uc_scales:
            run_dir = out_root / prompt_slug / f"uc_{uc_scale:g}"
            run_dir.mkdir(parents=True, exist_ok=True)

            sdf_gen = model.txt2shape(
                input_txt=prompt,
                ngen=args.ngen,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                uc_scale=uc_scale,
            )
            mesh = sdf_to_mesh(sdf_gen)
            if mesh is None:
                raise RuntimeError(f"Failed to convert txt2shape output for prompt: {prompt}")

            save_mesh_as_gif(model.renderer, mesh, nrow=min(3, args.ngen), out_name=str(run_dir / "pred.gif"))
            torch.save(sdf_gen.detach().cpu(), run_dir / "pred_tensor.pt")
            export_batch_meshes(mesh, run_dir, "pred")

            meta = {
                "prompt": prompt,
                "prompt_slug": prompt_slug,
                "uc_scale": uc_scale,
                "ngen": args.ngen,
                "ddim_steps": args.ddim_steps,
                "ddim_eta": args.ddim_eta,
            }
            with open(run_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            runs.append(meta)
            print(f"[*] saved {run_dir}")

    with open(out_root / "summary.json", "w") as f:
        json.dump(runs, f, indent=2)


if __name__ == "__main__":
    main()
