#!/usr/bin/env python3
"""Batch sweep for ShapeNet shape completion."""

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
from utils.demo_util import SDFusionOpt
from utils.util_3d import sdf_to_mesh, save_mesh_as_gif
from datasets.snet_dataset import ShapeNetDataset


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len] or "sample"


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
    parser = argparse.ArgumentParser(description="Sweep ShapeNet shape completion samples.")
    parser.add_argument("--category", default="all", help="ShapeNet category, or all.")
    parser.add_argument("--indices", nargs="*", type=int, default=None, help="Optional dataset indices to run.")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of samples to run if indices are omitted.")
    parser.add_argument("--ngen", type=int, default=6, help="Number of completions per sample.")
    parser.add_argument("--ddim_steps", type=int, default=100, help="DDIM steps.")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta.")
    parser.add_argument("--scale", type=float, default=1.0, help="Unconditional guidance scale.")
    parser.add_argument("--pyramid_list", type=str, default="1,1,2,2", help="Pyramid schedule.")
    parser.add_argument("--pyramid_interp_mode", type=str, default="trilinear", choices=["nearest", "bilinear", "trilinear"], help="Interpolation mode.")
    parser.add_argument("--pyramid_use_up_v2", action="store_true", help="Use extra upsample re-noise step.")
    parser.add_argument("--mask_mode", default="top", help="Partial-shape mask mode.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/FinalProject/ybr/data", help="Dataset root.")
    parser.add_argument("--ckpt", type=str, default="saved_ckpt/sdfusion-snet-all.pth", help="Diffusion checkpoint.")
    parser.add_argument("--vq_ckpt", type=str, default="saved_ckpt/vqvae-snet-all.pth", help="VQVAE checkpoint.")
    return parser.parse_args()


def choose_indices(dataset: ShapeNetDataset, indices: list[int] | None, num_samples: int) -> list[int]:
    if indices:
        return indices
    return list(range(min(num_samples, len(dataset))))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pyramid_list = [int(x.strip()) for x in args.pyramid_list.split(",") if x.strip()]
    if not pyramid_list:
        raise ValueError("pyramid_list is empty")

    opt = SDFusionOpt(gpu_ids=args.gpu_id, seed=args.seed)
    opt.init_dset_args(dataroot=args.dataroot, dataset_mode="snet", cat=args.category, res=64, cached_dir=None)
    opt.init_model_args(ckpt_path=args.ckpt, vq_ckpt_path=args.vq_ckpt)
    model = create_model(opt)

    dataset = ShapeNetDataset()
    dataset.initialize(opt, phase="test", cat=args.category, res=64)
    run_indices = choose_indices(dataset, args.indices, args.num_samples)

    out_root = ROOT.parent / "results" / "shape_comp_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    xyz_dict = {"x": (-1.0, 1.0), "y": (0.0, 1.0), "z": (-1.0, 1.0)}
    summary = []
    for ix in run_indices:
        item = dataset[ix]
        model_id = Path(item["path"]).parent.name
        synset = item["cat_id"]
        sample_slug = f"{synset}_{model_id}"
        shape = item["sdf"].unsqueeze(0).to(model.device)

        run_dir = out_root / synset / model_id
        run_dir.mkdir(parents=True, exist_ok=True)

        output = model.shape_comp(
            shape,
            xyz_dict,
            ngen=args.ngen,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            scale=args.scale,
            pyramid_list=pyramid_list,
            pyramid_interp_mode=args.pyramid_interp_mode,
            pyramid_use_up_v2=args.pyramid_use_up_v2,
        )

        mesh = sdf_to_mesh(output)
        if mesh is None:
            raise RuntimeError(f"Failed to convert shape completion output for {sample_slug}")

        save_mesh_as_gif(model.renderer, mesh, nrow=3, out_name=str(run_dir / "pred.gif"))
        torch.save(output.detach().cpu(), run_dir / "pred_tensor.pt")
        export_batch_meshes(mesh, run_dir, "pred")

        partial = getattr(model, "x_part", None)
        missing = getattr(model, "x_missing", None)
        if partial is not None:
            torch.save(partial.detach().cpu(), run_dir / "partial.pt")
        if missing is not None:
            torch.save(missing.detach().cpu(), run_dir / "missing.pt")

        meta = {
            "dataset_index": ix,
            "sample_slug": sample_slug,
            "synset": synset,
            "model_id": model_id,
            "path": item["path"],
            "scale": args.scale,
            "ddim_steps": args.ddim_steps,
            "ddim_eta": args.ddim_eta,
            "ngen": args.ngen,
            "pyramid_list": pyramid_list,
            "pyramid_interp_mode": args.pyramid_interp_mode,
            "pyramid_use_up_v2": args.pyramid_use_up_v2,
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
