#!/usr/bin/env python3
"""Convert pyramid demo outputs from SDF tensors to eval-ready meshes.

The demo scripts in this repo save predictions as ``*_tensor.pt`` files.
`utils/eval_matrics.py` can already read `.pt` / `.pth` / `.npy`, but it
matches predictions to GT by filename stem or parent directory name.

For the experiment folders under ``exp_img2shape`` and ``exp_shape_comp``,
the saved tensor names are generic (for example ``pyramid_shape_comp_tensor.pt``),
so they do not match dataset sample keys. This script converts each experiment
subdirectory into a mesh file named with an explicit sample key, which makes
the evaluation script pick it up correctly.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.util_3d import sdf_to_mesh


def load_sdf_tensor(path: Path) -> torch.Tensor:
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
        raise ValueError(f"Unsupported SDF tensor shape in {path}: {tuple(sdf.shape)}")
    return sdf


def export_mesh_from_sdf(sdf: torch.Tensor, out_path: Path, level: float) -> None:
    mesh = sdf_to_mesh(sdf, level=level)
    if mesh is None:
        raise RuntimeError("sdf_to_mesh returned None")

    verts = mesh.verts_list()[0].detach().cpu().numpy()
    faces = mesh.faces_list()[0].detach().cpu().numpy()
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tri.export(out_path)


def collect_tensor_files(run_dir: Path) -> list[Path]:
    files = sorted(run_dir.glob("*_tensor.pt"))
    if files:
        return files
    return sorted(p for p in run_dir.rglob("*_tensor.pt") if p.is_file())


def convert_run_dir(run_dir: Path, dst_run_dir: Path, sample_key: str, level: float, copy_aux: bool, overwrite: bool) -> None:
    tensor_files = collect_tensor_files(run_dir)
    if not tensor_files:
        print(f"[*] skip {run_dir}: no *_tensor.pt found")
        return

    dst_run_dir.mkdir(parents=True, exist_ok=True)
    out_mesh = dst_run_dir / f"{sample_key}.obj"
    if out_mesh.exists() and not overwrite:
        print(f"[*] skip {out_mesh}: already exists")
    else:
        sdf = load_sdf_tensor(tensor_files[0])
        export_mesh_from_sdf(sdf, out_mesh, level=level)
        print(f"[*] wrote {out_mesh}")

    if copy_aux:
        for src in run_dir.iterdir():
            if not src.is_file():
                continue
            if src.name.endswith("_tensor.pt"):
                continue
            dst = dst_run_dir / src.name
            if dst.exists() and not overwrite:
                continue
            shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pyramid experiment outputs to eval-ready meshes.")
    parser.add_argument(
        "--src_root",
        required=True,
        help="Experiment root, for example DiffusionShapeNet/exp_shape_comp or DiffusionShapeNet/exp_img2shape",
    )
    parser.add_argument(
        "--dst_root",
        required=True,
        help="Output root for converted meshes.",
    )
    parser.add_argument(
        "--sample_key",
        required=True,
        help="GT sample key used as output filename stem, for example 662f95... or IKEA_FUSION.",
    )
    parser.add_argument("--level", type=float, default=0.02, help="Marching-cubes level used for conversion.")
    parser.add_argument("--copy_aux", action="store_true", help="Copy non-tensor sidecar files into the output tree.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(src_root)

    # Convert immediate experiment subdirs; if the root already contains tensor files,
    # treat the root itself as a single run directory.
    run_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()])
    if not run_dirs:
        run_dirs = [src_root]

    for run_dir in run_dirs:
        tensor_files = collect_tensor_files(run_dir)
        if not tensor_files:
            continue
        dst_run_dir = dst_root / run_dir.name
        convert_run_dir(
            run_dir=run_dir,
            dst_run_dir=dst_run_dir,
            sample_key=args.sample_key,
            level=args.level,
            copy_aux=args.copy_aux,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
