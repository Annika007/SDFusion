#!/usr/bin/env python3
"""Batch sweep for Pix3D img2shape with automatic image/mask lookup."""

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
from utils.demo_util import SDFusionImage2ShapeOpt
from utils.util_3d import sdf_to_mesh, save_mesh_as_gif


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len] or "sample"


def export_mesh(mesh, out_path: Path) -> None:
    verts = mesh.verts_list()[0].detach().cpu().numpy()
    faces = mesh.faces_list()[0].detach().cpu().numpy()
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tri.export(out_path)


def load_pix3d_records(dataroot: str, category: str) -> list[dict]:
    pix3d_json = Path(dataroot) / "pix3d" / "pix3d.json"
    with open(pix3d_json, "r") as f:
        records = json.load(f)
    out = []
    for rec in records:
        if rec.get("category") != category:
            continue
        img_rel = rec.get("img")
        mask_rel = rec.get("mask")
        model_rel = rec.get("model")
        if not img_rel or not mask_rel or not model_rel:
            continue
        out.append(rec)
    return out


def choose_records(records: list[dict], sample_ids: list[str] | None, num_samples: int | None) -> list[dict]:
    if sample_ids:
        chosen = []
        wanted = set(sample_ids)
        for rec in records:
            if Path(rec["img"]).stem in wanted:
                chosen.append(rec)
        return chosen

    # Fallback: first unique models from the category.
    chosen = []
    seen_models = set()
    for rec in records:
        model_id = Path(rec["model"]).parent.name
        if model_id in seen_models:
            continue
        chosen.append(rec)
        seen_models.add(model_id)
        if num_samples is not None and len(chosen) >= num_samples:
            break
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Pix3D img2shape samples and guidance scales.")
    parser.add_argument("--category", default="chair", help="Pix3D category, for example chair or tool.")
    parser.add_argument("--sample_ids", nargs="*", default=None, help="Optional image ids like 1858 1859.")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of unique models to sample if sample_ids is omitted.")
    parser.add_argument("--uc_scales", nargs="*", type=float, default=[1.0, 2.0, 3.0, 5.0], help="Guidance scales to sweep.")
    parser.add_argument("--ddim_steps", type=int, default=100, help="DDIM steps.")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/FinalProject/ybr/data", help="Dataset root.")
    parser.add_argument("--ckpt", type=str, default="saved_ckpt/sdfusion-img2shape.pth", help="Diffusion checkpoint.")
    parser.add_argument("--vq_ckpt", type=str, default="saved_ckpt/vqvae-snet-all.pth", help="VQVAE checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = load_pix3d_records(args.dataroot, args.category)
    chosen = choose_records(records, args.sample_ids, args.num_samples)
    if not chosen:
        raise RuntimeError(f"No Pix3D records found for category={args.category}")

    out_root = ROOT.parent / "results" / "img2shape_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    opt = SDFusionImage2ShapeOpt(gpu_ids=args.gpu_id, seed=args.seed)
    opt.init_dset_args(dataroot=args.dataroot, dataset_mode="pix3d_img2shape", cat="all", res=64, cached_dir=None)
    opt.init_model_args(ckpt_path=args.ckpt, vq_ckpt_path=args.vq_ckpt)
    model = create_model(opt)

    summary = []
    for rec in chosen:
        img_rel = rec["img"]
        mask_rel = rec["mask"]
        model_rel = rec["model"]
        model_id = Path(model_rel).parent.name
        image_id = Path(img_rel).stem
        image_path = Path(args.dataroot) / "pix3d" / img_rel
        mask_path = Path(args.dataroot) / "pix3d" / mask_rel

        for uc_scale in args.uc_scales:
            run_dir = out_root / args.category / model_id / f"{image_id}_uc_{uc_scale:g}"
            run_dir.mkdir(parents=True, exist_ok=True)

            sdf_gen = model.img2shape(
                image=str(image_path),
                mask=str(mask_path),
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                uc_scale=uc_scale,
            )
            mesh = sdf_to_mesh(sdf_gen)
            if mesh is None:
                raise RuntimeError(f"Failed to convert img2shape output for {image_path}")

            save_mesh_as_gif(model.renderer, mesh, nrow=1, out_name=str(run_dir / "pred.gif"))
            torch.save(sdf_gen.detach().cpu(), run_dir / "pred_tensor.pt")
            export_mesh(mesh, run_dir / "pred.obj")

            meta = {
                "category": args.category,
                "model_id": model_id,
                "image_id": image_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "uc_scale": uc_scale,
                "ddim_steps": args.ddim_steps,
                "ddim_eta": args.ddim_eta,
            }
            with open(run_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            summary.append(meta)
            print(f"[*] saved {run_dir}")

    with open(out_root / f"{args.category}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
