#!/usr/bin/env python3
"""Aggregate results, render figures, and write LaTeX tables for the project report.

This script is intentionally self-contained:
- Collects quantitative metrics from existing eval JSONs.
- Extracts geometry statistics from generated OBJ files.
- Builds qualitative grids from saved GIFs.
- Writes CSV/JSON/PNG/TEX assets into `results/report_assets/`.

Usage:
    python scripts/report_assets.py collect
    python scripts/report_assets.py plot
    python scripts/report_assets.py tables
    python scripts/report_assets.py all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TOP_ROOT = ROOT.parent
RESULT_ROOTS = [TOP_ROOT / "results", ROOT / "results"]
OUT_ROOT = TOP_ROOT / "results" / "report_assets"
DATA_DIR = OUT_ROOT / "data"
FIG_DIR = OUT_ROOT / "figures"
TEX_DIR = OUT_ROOT / "tables"


def ensure_dirs() -> None:
    for d in (OUT_ROOT, DATA_DIR, FIG_DIR, TEX_DIR):
        d.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return np.nan, np.nan, 0
    arr = np.array(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), int(arr.size)


def load_shapenet_cat_map() -> dict[str, str]:
    info_path = ROOT / "dataset_info_files" / "info-shapenet.json"
    if not info_path.exists():
        return {}
    info = read_json(info_path)
    return {v: k for k, v in info.get("cats", {}).items()}


def load_pix3d_cat_map() -> dict[str, str]:
    info_path = ROOT / "dataset_info_files" / "info-pix3d.json"
    if not info_path.exists():
        return {}
    info = read_json(info_path)
    return {v: k for k, v in info.get("cats", {}).items()}


def find_existing(*parts: str) -> Path | None:
    for root in RESULT_ROOTS:
        p = root.joinpath(*parts)
        if p.exists():
            return p
    return None


def iter_obj_files(run_dir: Path) -> list[Path]:
    objs = [p for p in run_dir.glob("pred_*.obj") if p.is_file()]
    if not objs:
        p = run_dir / "pred.obj"
        if p.exists():
            objs = [p]
    return sorted(objs)


def mesh_stats_from_obj(obj_path: Path) -> dict:
    mesh = trimesh.load_mesh(obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    extents = mesh.extents if hasattr(mesh, "extents") else np.array([np.nan, np.nan, np.nan])
    bbox_volume = float(np.prod(extents)) if np.all(np.isfinite(extents)) else np.nan
    volume = float(abs(mesh.volume)) if getattr(mesh, "is_volume", False) else np.nan
    return {
        "obj": str(obj_path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "surface_area": float(mesh.area) if hasattr(mesh, "area") else np.nan,
        "volume": volume,
        "bbox_volume": bbox_volume,
        "is_watertight": bool(getattr(mesh, "is_watertight", False)),
        "is_volume": bool(getattr(mesh, "is_volume", False)),
    }


def geom_stats_for_run(run_dir: Path) -> dict:
    objs = iter_obj_files(run_dir)
    stats = [mesh_stats_from_obj(p) for p in objs]
    if not stats:
        return {"mesh_count": 0}
    out = {"mesh_count": len(stats)}
    for key in ("vertex_count", "face_count", "surface_area", "volume", "bbox_volume"):
        vals = [s[key] for s in stats if s.get(key) is not None and not (isinstance(s[key], float) and np.isnan(s[key]))]
        m, s, n = mean_std(vals)
        out[f"{key}_mean"] = m
        out[f"{key}_std"] = s
        out[f"{key}_count"] = n
    out["watertight_rate"] = float(np.mean([1.0 if s.get("is_watertight") else 0.0 for s in stats]))
    out["volume_rate"] = float(np.mean([1.0 if s.get("is_volume") else 0.0 for s in stats]))
    return out


def collect_eval_json(eval_path: Path, task: str) -> list[dict]:
    data = read_json(eval_path)
    rows = []
    if isinstance(data, dict) and "per_sample" in data:
        for sample, metrics in data["per_sample"].items():
            row = {"task": task, "sample": sample}
            row.update({k: safe_float(v) for k, v in metrics.items()})
            rows.append(row)
    return rows


def collect_shape_comp() -> tuple[list[dict], dict]:
    rows = collect_eval_json(TOP_ROOT / "results" / "shape_comp_sweep.json", "shape_comp")
    summary = read_json(TOP_ROOT / "results" / "shape_comp_sweep.json")["summary"]
    return rows, summary


def collect_img2shape() -> tuple[list[dict], dict]:
    rows = []
    summaries = {}
    for cat in ("chair", "tool"):
        p = TOP_ROOT / "results" / f"img2shape_sweep_{cat}.json"
        if not p.exists():
            continue
        rows.extend(collect_eval_json(p, f"img2shape_{cat}"))
        summaries[cat] = read_json(p)["summary"]
    return rows, summaries


def collect_summary_rows(summary_path: Path, task: str) -> list[dict]:
    data = read_json(summary_path)
    rows = []
    if isinstance(data, list):
        for r in data:
            row = {"task": task}
            row.update(r)
            rows.append(row)
    return rows


def extract_txt2shape_rows() -> list[dict]:
    summary_path = find_existing("txt2shape_sweep", "summary.json")
    if summary_path is None:
        return []
    summary = read_json(summary_path)
    rows = []
    for r in summary:
        prompt_slug = r["prompt_slug"]
        uc_scale = r["uc_scale"]
        run_dir = summary_path.parent / prompt_slug / f"uc_{uc_scale:g}"
        g = geom_stats_for_run(run_dir)
        row = dict(r)
        row.update(g)
        rows.append(row)
    return rows


def extract_mm2shape_rows() -> list[dict]:
    summary_path = find_existing("mm2shape_guidance_sweep", "chair_summary.json")
    if summary_path is None:
        return []
    summary = read_json(summary_path)
    rows = []
    for r in summary:
        prompt_slug = r["prompt"].lower().strip().replace(" ", "-")
        prompt_slug = "".join(ch for ch in prompt_slug if ch.isalnum() or ch == "-")
        prompt_slug = "-".join([s for s in prompt_slug.split("-") if s])
        run_dir = summary_path.parent / "chair" / prompt_slug / r["model_id"] / f"txt_{r['txt_scale']:g}_img_{r['img_scale']:g}"
        g = geom_stats_for_run(run_dir)
        row = dict(r)
        row.update(g)
        rows.append(row)
    return rows


def extract_sweep_rows(base_dir: Path) -> list[dict]:
    rows = []
    if not base_dir.exists():
        return rows
    for meta_path in base_dir.rglob("meta.json"):
        row = read_json(meta_path)
        row["run_dir"] = str(meta_path.parent)
        row.update(geom_stats_for_run(meta_path.parent))
        rows.append(row)
    return rows


def collect_all() -> dict:
    ensure_dirs()
    output = {}

    shape_rows, shape_summary = collect_shape_comp()
    output["shape_comp_eval_rows"] = shape_rows
    output["shape_comp_eval_summary"] = shape_summary
    write_csv(DATA_DIR / "shape_comp_eval_rows.csv", shape_rows, sorted({k for r in shape_rows for k in r.keys()}))
    write_json(DATA_DIR / "shape_comp_eval_summary.json", shape_summary)

    img_rows, img_summaries = collect_img2shape()
    output["img2shape_eval_rows"] = img_rows
    output["img2shape_eval_summary"] = img_summaries
    if img_rows:
        write_csv(DATA_DIR / "img2shape_eval_rows.csv", img_rows, sorted({k for r in img_rows for k in r.keys()}))
    write_json(DATA_DIR / "img2shape_eval_summary.json", img_summaries)

    txt_rows = extract_txt2shape_rows()
    output["txt2shape_rows"] = txt_rows
    if txt_rows:
        write_csv(DATA_DIR / "txt2shape_rows.csv", txt_rows, sorted({k for r in txt_rows for k in r.keys()}))

    mm_rows = extract_mm2shape_rows()
    output["mm2shape_rows"] = mm_rows
    if mm_rows:
        write_csv(DATA_DIR / "mm2shape_rows.csv", mm_rows, sorted({k for r in mm_rows for k in r.keys()}))

    # Also ingest per-run sweep rows if you want to inspect geometry stats directly.
    output["shape_comp_runs"] = extract_sweep_rows(TOP_ROOT / "results" / "shape_comp_sweep")
    output["img2shape_runs"] = extract_sweep_rows(TOP_ROOT / "results" / "img2shape_sweep")
    write_json(DATA_DIR / "shape_comp_runs.json", output["shape_comp_runs"])
    write_json(DATA_DIR / "img2shape_runs.json", output["img2shape_runs"])
    return output


def first_frame(path: Path) -> Image.Image:
    frames = imageio.mimread(path)
    if not frames:
        raise RuntimeError(f"Empty gif: {path}")
    arr = np.asarray(frames[0])
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr.astype(np.uint8))


def resize_to_fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(img, size)


def draw_label(img: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 28), "white")
    canvas.paste(img, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), text, fill="black")
    return canvas


def make_grid(images: list[Image.Image], rows: int, cols: int, cell_size: tuple[int, int], pad: int = 8, bg=(255, 255, 255)) -> Image.Image:
    w, h = cell_size
    out = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad) + pad), bg)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        if r >= rows:
            break
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        img = resize_to_fit(img, (w, h))
        tile = Image.new("RGB", (w, h), bg)
        tile.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
        out.paste(tile, (x, y))
    return out


def plot_shape_comp(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    metrics = [k for k in ("chamfer", "fscore_0.01", "fscore_0.02", "uhd", "tmd") if k in summary]
    means = [summary[m]["mean"] for m in metrics]
    stds = [summary[m]["std"] for m in metrics]
    ax.bar(metrics, means, yerr=stds, color="#4C78A8", alpha=0.9, capsize=4)
    ax.set_title("Shape Completion Summary")
    ax.set_ylabel("Metric value")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shape_comp_summary_bar.png", dpi=200)
    plt.close(fig)


def plot_img2shape(summaries: dict) -> None:
    rows = []
    for cat, summary in summaries.items():
        rows.append((cat, summary))
    if not rows:
        return
    metrics = ["chamfer", "fscore_0.01", "fscore_0.02", "iou"]
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 3.5 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, (cat, summary) in zip(axes, rows):
        means = [summary[m]["mean"] for m in metrics if m in summary]
        labels = [m for m in metrics if m in summary]
        ax.bar(labels, means, color="#72B7B2")
        ax.set_title(f"Pix3D img2shape: {cat}")
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "img2shape_summary_bar.png", dpi=200)
    plt.close(fig)


def plot_txt2shape_grid(txt_rows: list[dict]) -> None:
    if not txt_rows:
        return
    prompts = []
    scales = []
    by_key = {}
    for r in txt_rows:
        prompts.append(r["prompt_slug"])
        scales.append(r["uc_scale"])
        by_key[(r["prompt_slug"], r["uc_scale"])] = r
    prompts = sorted(set(prompts))
    scales = sorted(set(scales))

    images = []
    labels = []
    for p in prompts:
        for s in scales:
            r = by_key[(p, s)]
            gif_path = Path(find_existing("txt2shape_sweep", p, f"uc_{s:g}", "pred.gif"))
            img = first_frame(gif_path)
            images.append(draw_label(img, f"{p}\nuc={s:g}"))
            labels.append(f"{p}:{s}")
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    grid = make_grid(images, rows=len(prompts), cols=len(scales), cell_size=(cell_w, cell_h))
    grid.save(FIG_DIR / "txt2shape_uc_grid.png")


def plot_mm2shape_heatmaps(mm_rows: list[dict]) -> None:
    if not mm_rows:
        return
    prompts = sorted(set(r["prompt"] for r in mm_rows))
    txt_scales = sorted(set(float(r["txt_scale"]) for r in mm_rows))
    img_scales = sorted(set(float(r["img_scale"]) for r in mm_rows))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, prompt in zip(axes, prompts):
        pivot = np.full((len(txt_scales), len(img_scales)), np.nan, dtype=np.float32)
        lookup = {(float(r["txt_scale"]), float(r["img_scale"])): r for r in mm_rows if r["prompt"] == prompt}
        for i, t in enumerate(txt_scales):
            for j, im in enumerate(img_scales):
                r = lookup.get((t, im))
                if r:
                    pivot[i, j] = r.get("bbox_volume_mean", np.nan)
        im = ax.imshow(pivot, cmap="viridis")
        ax.set_title(prompt)
        ax.set_xticks(range(len(img_scales)))
        ax.set_xticklabels([str(s) for s in img_scales])
        ax.set_yticks(range(len(txt_scales)))
        ax.set_yticklabels([str(s) for s in txt_scales])
        ax.set_xlabel("img_scale")
        ax.set_ylabel("txt_scale")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                if np.isfinite(pivot[i, j]):
                    ax.text(j, i, f"{pivot[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.75, label="Mean bbox volume")
    fig.suptitle("mm2shape guidance sweep: geometry sensitivity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mm2shape_guidance_heatmaps.png", dpi=200)
    plt.close(fig)


def plot_guidance_geometry(txt_rows: list[dict], mm_rows: list[dict]) -> None:
    # txt2shape: face-count trend over uc_scale
    if txt_rows:
        prompts = sorted(set(r["prompt_slug"] for r in txt_rows))
        scales = sorted(set(float(r["uc_scale"]) for r in txt_rows))
        fig, axes = plt.subplots(len(prompts), 1, figsize=(9, 3.2 * len(prompts)))
        if len(prompts) == 1:
            axes = [axes]
        for ax, prompt in zip(axes, prompts):
            ys = []
            for s in scales:
                vals = [r.get("face_count_mean", np.nan) for r in txt_rows if r["prompt_slug"] == prompt and float(r["uc_scale"]) == s]
                ys.append(np.nanmean(vals))
            ax.plot(scales, ys, marker="o")
            ax.set_title(f"txt2shape geometry: {prompt}")
            ax.set_xlabel("uc_scale")
            ax.set_ylabel("mean face count")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "txt2shape_facecount_vs_uc.png", dpi=200)
        plt.close(fig)

    # mm2shape: volume trend per prompt averaged over grid
    if mm_rows:
        prompts = sorted(set(r["prompt"] for r in mm_rows))
        fig, axes = plt.subplots(len(prompts), 1, figsize=(9, 3.2 * len(prompts)))
        if len(prompts) == 1:
            axes = [axes]
        for ax, prompt in zip(axes, prompts):
            rows = [r for r in mm_rows if r["prompt"] == prompt]
            vals = [r.get("bbox_volume_mean", np.nan) for r in rows]
            ax.hist([v for v in vals if np.isfinite(v)], bins=8, color="#F58518", alpha=0.85)
            ax.set_title(f"mm2shape geometry: {prompt}")
            ax.set_xlabel("mean bbox volume")
            ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "mm2shape_bbox_volume_hist.png", dpi=200)
        plt.close(fig)


def write_latex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]], align: str = "l") -> None:
    cols = align + " ".join(["c"] * (len(headers) - 1))
    lines = [
        "\\begin{table}[t]",
        f"\\caption{{{caption}}}",
        "\\centering",
        "\\setlength{\\tabcolsep}{0.35em}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\label{{{label}}}", "\\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def fmt_mean_std(summary_item: dict, precision: int = 4) -> str:
    if summary_item is None:
        return "N/A"
    mean = summary_item.get("mean", np.nan)
    std = summary_item.get("std", np.nan)
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def make_tables(shape_summary: dict, img_summaries: dict, txt_rows: list[dict], mm_rows: list[dict]) -> None:
    # Main quantitative tables.
    shape_rows = [[
        "Ours (ShapeNet)",
        fmt_mean_std(shape_summary.get("chamfer")),
        fmt_mean_std(shape_summary.get("fscore_0.01")),
        fmt_mean_std(shape_summary.get("fscore_0.02")),
        fmt_mean_std(shape_summary.get("uhd")),
        fmt_mean_std(shape_summary.get("tmd")),
        str(shape_summary.get("chamfer", {}).get("count", 0)),
    ]]
    write_latex_table(
        TEX_DIR / "tab_shape_comp_auto.tex",
        "Automatic summary of the ShapeNet completion sweep.",
        "tab:auto_shape_comp",
        ["Method", "CD", "F@1\\%", "F@2\\%", "UHD", "TMD", "N"],
        shape_rows,
    )

    img_rows = []
    for cat, summary in img_summaries.items():
        img_rows.append([
            f"Ours ({cat})",
            fmt_mean_std(summary.get("chamfer")),
            fmt_mean_std(summary.get("fscore_0.01")),
            fmt_mean_std(summary.get("fscore_0.02")),
            fmt_mean_std(summary.get("iou")),
            str(summary.get("chamfer", {}).get("count", 0)),
        ])
    write_latex_table(
        TEX_DIR / "tab_img2shape_auto.tex",
        "Automatic summary of the Pix3D reconstruction sweep.",
        "tab:auto_img2shape",
        ["Method", "CD", "F@1\\%", "F@2\\%", "IoU", "N"],
        img_rows,
    )

    # Geometry-analysis tables for guidance sweeps.
    txt_table = []
    for prompt in sorted(set(r["prompt"] for r in txt_rows)):
        for s in sorted(set(float(r["uc_scale"]) for r in txt_rows)):
            subset = [r for r in txt_rows if r["prompt"] == prompt and float(r["uc_scale"]) == s]
            if not subset:
                continue
            row = subset[0]
            txt_table.append([
                prompt,
                f"{s:g}",
                f"{row.get('face_count_mean', np.nan):.1f}",
                f"{row.get('bbox_volume_mean', np.nan):.3f}",
                f"{row.get('watertight_rate', np.nan):.2f}",
            ])
    if txt_table:
        write_latex_table(
            TEX_DIR / "tab_txt2shape_geometry.tex",
            "Geometry statistics of txt2shape guidance sweeps.",
            "tab:txt2shape_geometry",
            ["Prompt", "uc", "Faces", "BBoxVol", "Watertight"],
            txt_table,
        )

    mm_table = []
    for prompt in sorted(set(r["prompt"] for r in mm_rows)):
        subset = [r for r in mm_rows if r["prompt"] == prompt]
        if not subset:
            continue
        vols = [r.get("bbox_volume_mean", np.nan) for r in subset if np.isfinite(r.get("bbox_volume_mean", np.nan))]
        faces = [r.get("face_count_mean", np.nan) for r in subset if np.isfinite(r.get("face_count_mean", np.nan))]
        mm_table.append([
            prompt,
            f"{np.nanmean(faces):.1f}" if faces else "N/A",
            f"{np.nanmean(vols):.3f}" if vols else "N/A",
            f"{np.nanmean([r.get('watertight_rate', np.nan) for r in subset]):.2f}",
        ])
    if mm_table:
        write_latex_table(
            TEX_DIR / "tab_mm2shape_geometry.tex",
            "Geometry statistics of mm2shape guidance sweeps.",
            "tab:mm2shape_geometry",
            ["Prompt", "Faces", "BBoxVol", "Watertight"],
            mm_table,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["collect", "plot", "tables", "all"], help="Which report assets to build.")
    args = parser.parse_args()

    ensure_dirs()
    cache = {}

    if args.mode in {"collect", "all", "plot", "tables"}:
        cache = collect_all()

    if args.mode in {"plot", "all"}:
        shape_summary = read_json(DATA_DIR / "shape_comp_eval_summary.json")
        img_summaries = read_json(DATA_DIR / "img2shape_eval_summary.json")
        txt_rows = read_json(DATA_DIR / "txt2shape_rows.csv") if False else extract_txt2shape_rows()
        mm_rows = extract_mm2shape_rows()
        plot_shape_comp(shape_summary)
        plot_img2shape(img_summaries)
        plot_txt2shape_grid(txt_rows)
        plot_mm2shape_heatmaps(mm_rows)
        plot_guidance_geometry(txt_rows, mm_rows)

    if args.mode in {"tables", "all"}:
        shape_summary = read_json(DATA_DIR / "shape_comp_eval_summary.json")
        img_summaries = read_json(DATA_DIR / "img2shape_eval_summary.json")
        txt_rows = extract_txt2shape_rows()
        mm_rows = extract_mm2shape_rows()
        make_tables(shape_summary, img_summaries, txt_rows, mm_rows)

    print(f"[*] report assets written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
