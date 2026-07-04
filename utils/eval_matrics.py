# Evaluation utilities for SDFusion / pyramid experiments.
# Supports mesh metrics (Chamfer / F-score / IoU) and shape completion metrics (UHD / TMD).
import argparse
import json
import os
import random
import sys
from glob import glob
from itertools import combinations

import numpy as np
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from pytorch3d.loss import chamfer_distance
    _USE_PYTORCH3D = True
except Exception:
    _USE_PYTORCH3D = False

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

_HAS_TRIMESH = True
try:
    import trimesh
except Exception:
    trimesh = None
    _HAS_TRIMESH = False

try:
    from utils.util_3d import sdf_to_mesh
except Exception:
    sdf_to_mesh = None

from utils.util import iou as voxel_iou


SUPPORTED_EXTS = {'.obj', '.ply', '.off', '.stl', '.glb', '.gltf', '.h5', '.hdf5', '.npz', '.pt', '.pth', '.npy'}


def sample_points_from_trimesh(mesh, n_pts):
    if trimesh is None:
        raise RuntimeError('trimesh is required')
    pts, _ = trimesh.sample.sample_surface(mesh, n_pts)
    return pts.astype(np.float32)


def ensure_trimesh(mesh):
    if trimesh is None:
        raise RuntimeError('trimesh is required')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if hasattr(mesh, 'verts_list') and hasattr(mesh, 'faces_list'):
        verts = mesh.verts_list()
        faces = mesh.faces_list()
        if len(verts) == 1 and len(faces) == 1:
            return trimesh.Trimesh(
                vertices=verts[0].detach().cpu().numpy(),
                faces=faces[0].detach().cpu().numpy(),
                process=False,
            )
    if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
        return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    raise RuntimeError(f'Unsupported mesh type: {type(mesh)}')


def candidate_keys(path):
    path = os.path.abspath(path)
    keys = []
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    for key in (stem, parent, grandparent):
        if key and key not in keys:
            keys.append(key)
    return keys


def load_mesh_file(mesh_path):
    if trimesh is None:
        raise RuntimeError('trimesh is required to load meshes')
    return ensure_trimesh(trimesh.load_mesh(mesh_path, process=False))


def load_sdf_tensor(path, resolution=64):
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {'.h5', '.hdf5'}:
        import h5py
        with h5py.File(path, 'r') as hf:
            datasets = []
            def collect(group, prefix=''):
                for k in group.keys():
                    obj = group[k]
                    if isinstance(obj, h5py.Dataset):
                        datasets.append((prefix + k, obj))
                    elif isinstance(obj, h5py.Group):
                        collect(obj, prefix + k + '/')
            collect(hf)
            if not datasets:
                raise RuntimeError(f'No datasets found in {path}')
            candidates = []
            target_size = resolution ** 3
            keywords = ['pc_sdf_sample', 'ori_sample_grid', 'sdf', 'sample', 'grid']
            for name, ds in datasets:
                try:
                    shape = tuple(int(x) for x in ds.shape)
                except Exception:
                    continue
                size = int(np.prod(shape))
                has_kw = any(k in name.lower() for k in keywords)
                candidates.append((name, ds, shape, size, has_kw))
            pick = None
            for _, ds, _, size, _ in candidates:
                if size == target_size:
                    pick = ds
                    break
            if pick is None:
                for _, ds, _, _, has_kw in candidates:
                    if has_kw:
                        pick = ds
                        break
            if pick is None and candidates:
                candidates.sort(key=lambda x: x[3], reverse=True)
                pick = candidates[0][1]
            if pick is None:
                raise RuntimeError(f'Could not find suitable SDF dataset in {path}')
            arr = pick[()]
    elif suffix == '.npz':
        data = np.load(path)
        arr = None
        for k in data.files:
            a = data[k]
            if hasattr(a, 'ndim') and a.ndim >= 3:
                arr = a
                break
        if arr is None:
            arr = data[data.files[0]]
    elif suffix in {'.pt', '.pth'}:
        sdf = torch.load(path, map_location='cpu')
        if isinstance(sdf, dict):
            for key in ('sdf', 'pc_sdf_sample', 'data', 'tensor'):
                if key in sdf:
                    sdf = sdf[key]
                    break
        arr = torch.as_tensor(sdf).cpu().numpy()
    elif suffix == '.npy':
        arr = np.load(path)
    else:
        raise RuntimeError('Unsupported SDF format: ' + suffix)

    arr = np.array(arr).astype(np.float32)
    if arr.ndim == 1:
        t = torch.from_numpy(arr.reshape(resolution, resolution, resolution))[None, None, ...]
    elif arr.ndim == 2:
        t = torch.from_numpy(arr.flatten().reshape(resolution, resolution, resolution))[None, None, ...]
    elif arr.ndim == 3:
        t = torch.from_numpy(arr)[None, None, ...]
    elif arr.ndim == 4:
        t = torch.from_numpy(arr)[None, ...]
    elif arr.ndim == 5:
        t = torch.from_numpy(arr)
    else:
        raise RuntimeError(f'Unsupported SDF array shape: {arr.shape}')
    return t.float()


def load_sdf_as_mesh(sdf_path, resolution=64, level=0.02):
    if sdf_to_mesh is None:
        raise RuntimeError('utils.util_3d.sdf_to_mesh is required')
    sdf = load_sdf_tensor(sdf_path, resolution=resolution)
    mesh = sdf_to_mesh(sdf, level=level)
    return ensure_trimesh(mesh)


def load_geometry_as_mesh(path, resolution=64, sdf_level=0.02):
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.obj', '.ply', '.off', '.stl', '.glb', '.gltf'}:
        return load_mesh_file(path)
    if ext in {'.h5', '.hdf5', '.npz', '.pt', '.pth', '.npy'}:
        return load_sdf_as_mesh(path, resolution=resolution, level=sdf_level)
    raise RuntimeError('Unsupported geometry format: ' + ext)


def compute_chamfer_and_fscore(pred_mesh, gt_mesh, num_points=10000, thresholds=(0.01,)):
    pts_pred = sample_points_from_trimesh(pred_mesh, num_points)
    pts_gt = sample_points_from_trimesh(gt_mesh, num_points)

    results = {}
    success_pytorch3d = False

    if _USE_PYTORCH3D:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            p_pred = torch.from_numpy(pts_pred).unsqueeze(0).to(device)
            p_gt = torch.from_numpy(pts_gt).unsqueeze(0).to(device)
            cd_p2g, cd_g2p = chamfer_distance(p_pred, p_gt)
            if cd_p2g is not None and cd_g2p is not None:
                cd = (cd_p2g.mean().item() + cd_g2p.mean().item()) / 2.0
                results['chamfer'] = float(cd)
                dists = torch.cdist(p_pred, p_gt)
                min_pred_gt, _ = dists.min(dim=2)
                min_gt_pred, _ = dists.min(dim=1)
                success_pytorch3d = True
        except Exception:
            success_pytorch3d = False

    if not success_pytorch3d:
        if KDTree is not None:
            tree_gt = KDTree(pts_gt)
            d_pred_gt, _ = tree_gt.query(pts_pred, k=1)
            tree_pred = KDTree(pts_pred)
            d_gt_pred, _ = tree_pred.query(pts_gt, k=1)
        else:
            dists_full = np.sqrt(((pts_pred[:, None, :] - pts_gt[None, :, :]) ** 2).sum(-1))
            d_pred_gt = dists_full.min(axis=1)
            d_gt_pred = dists_full.min(axis=0)
        cd = float(np.mean(d_pred_gt ** 2) + np.mean(d_gt_pred ** 2)) / 2.0
        results['chamfer'] = float(cd)
        min_pred_gt = torch.from_numpy(d_pred_gt[None, :])
        min_gt_pred = torch.from_numpy(d_gt_pred[None, :])

    for thr in thresholds:
        if isinstance(min_pred_gt, torch.Tensor):
            prec = (min_pred_gt < thr).float().mean().item()
            rec = (min_gt_pred < thr).float().mean().item()
        else:
            prec = float((min_pred_gt < thr).mean())
            rec = float((min_gt_pred < thr).mean())
        f = 2 * prec * rec / (prec + rec + 1e-9)
        results[f'fscore_{thr}'] = float(f)
        results[f'prec_{thr}'] = float(prec)
        results[f'rec_{thr}'] = float(rec)

    return results


def mesh_contains_voxel(mesh, resolution=64):
    if trimesh is None:
        raise RuntimeError('trimesh is required for IoU')
    rng = np.linspace(-1.0, 1.0, resolution)
    xs, ys, zs = np.meshgrid(rng, rng, rng, indexing='xy')
    pts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    contained = mesh.contains(pts)
    occ = contained.reshape(resolution, resolution, resolution)
    occ_t = torch.from_numpy(occ.astype(np.float32))[None, None, ...]
    return occ_t


def compute_iou_from_mesh_and_sdf(pred_mesh, gt_sdf_file, resolution=64, thres=0.0):
    gt_sdf = load_sdf_tensor(gt_sdf_file, resolution=resolution)
    try:
        pred_occ = mesh_contains_voxel(pred_mesh, resolution=resolution)
    except Exception as e:
        raise RuntimeError('Failed to rasterize pred mesh to voxel occupancy: ' + str(e))
    iou_val = voxel_iou(gt_sdf, pred_occ, thres)
    return float(iou_val.mean().item())


def compute_uhd(pred_mesh, partial_mesh, num_points=10000):
    pred_pts = sample_points_from_trimesh(pred_mesh, num_points)
    partial_pts = sample_points_from_trimesh(partial_mesh, num_points)
    if KDTree is not None:
        tree_pred = KDTree(pred_pts)
        d_partial_pred, _ = tree_pred.query(partial_pts, k=1)
    else:
        dists_full = np.sqrt(((partial_pts[:, None, :] - pred_pts[None, :, :]) ** 2).sum(-1))
        d_partial_pred = dists_full.min(axis=1)
    return float(np.max(d_partial_pred))


def compute_tmd(mesh_list, num_points=10000):
    if len(mesh_list) < 2:
        return None
    pair_vals = []
    for m1, m2 in combinations(mesh_list, 2):
        pts_1 = sample_points_from_trimesh(m1, num_points)
        pts_2 = sample_points_from_trimesh(m2, num_points)
        if KDTree is not None:
            tree_2 = KDTree(pts_2)
            d_1_2, _ = tree_2.query(pts_1, k=1)
            tree_1 = KDTree(pts_1)
            d_2_1, _ = tree_1.query(pts_2, k=1)
        else:
            dists_full = np.sqrt(((pts_1[:, None, :] - pts_2[None, :, :]) ** 2).sum(-1))
            d_1_2 = dists_full.min(axis=1)
            d_2_1 = dists_full.min(axis=0)
        cd = float(np.mean(d_1_2 ** 2) + np.mean(d_2_1 ** 2)) / 2.0
        pair_vals.append(cd)
    return float(np.mean(pair_vals)) if pair_vals else None


def build_lookup(root_dir):
    lookup = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            full = os.path.join(root, f)
            for key in candidate_keys(full):
                if key not in lookup:
                    lookup[key] = full
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory of predicted geometry')
    parser.add_argument('--gt_dir', type=str, required=False, help='Directory of ground-truth meshes')
    parser.add_argument('--gt_sdf_dir', type=str, required=False, help='Directory of ground-truth SDF files for IoU')
    parser.add_argument('--partial_dir', type=str, required=False, help='Directory of partial shapes for UHD/TMD')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--thresh', type=str, default='0.01,0.02', help='comma-separated thresholds for F-score')
    parser.add_argument('--sdf_resolution', type=int, default=64)
    parser.add_argument('--sdf_level', type=float, default=0.02)
    parser.add_argument('--shuffle', action='store_true', help='Shuffle matched samples before truncating to n_samples')
    parser.add_argument('--out', type=str, default='results/eval_results.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    preds = sorted(glob(os.path.join(args.pred_dir, '**', '*'), recursive=True))
    preds = [
        p for p in preds
        if os.path.isfile(p)
        and os.path.splitext(p)[1].lower() in SUPPORTED_EXTS
        and not os.path.basename(p).endswith('_tensor.pt')
    ]
    if not preds:
        print('No predicted geometry found in', args.pred_dir)
        return

    gt_map = build_lookup(args.gt_dir) if args.gt_dir else {}
    gt_sdf_map = build_lookup(args.gt_sdf_dir) if args.gt_sdf_dir else {}
    partial_map = build_lookup(args.partial_dir) if args.partial_dir else {}

    matched = []
    for p in preds:
        match_key = None
        for key in candidate_keys(p):
            if key in gt_map or key in gt_sdf_map or key in partial_map:
                match_key = key
                break
        if match_key is None:
            continue
        matched.append((match_key, p, gt_map.get(match_key), gt_sdf_map.get(match_key), partial_map.get(match_key)))

    if not matched:
        print('No matching GT found for predicted geometry.')
        return

    if args.shuffle:
        random.shuffle(matched)
    matched = matched[:args.n_samples]

    thresholds = [float(x) for x in args.thresh.split(',') if x.strip()]

    grouped = {}
    for key, pred_p, gt_mesh_p, gt_sdf_p, partial_p in matched:
        grouped.setdefault(key, []).append((pred_p, gt_mesh_p, gt_sdf_p, partial_p))

    per_sample = {}
    for key, items in grouped.items():
        pred_meshes = []
        gt_mesh_p = items[0][1]
        gt_sdf_p = items[0][2]
        partial_p = items[0][3]

        for pred_p, _, _, _ in items:
            try:
                pred_tm = load_geometry_as_mesh(pred_p, resolution=args.sdf_resolution, sdf_level=args.sdf_level)
                pred_meshes.append(pred_tm)
            except Exception as e:
                print('Failed to load predicted geometry', pred_p, e)

        if not pred_meshes:
            continue

        sample_metrics = {}
        if gt_mesh_p:
            try:
                gt_tm = load_geometry_as_mesh(gt_mesh_p, resolution=args.sdf_resolution, sdf_level=args.sdf_level)
                sample_metrics.update(compute_chamfer_and_fscore(pred_meshes[0], gt_tm, num_points=args.num_points, thresholds=thresholds))
            except Exception as e:
                print('Failed to process GT mesh', gt_mesh_p, e)

        if gt_sdf_p:
            try:
                sample_metrics['iou'] = compute_iou_from_mesh_and_sdf(pred_meshes[0], gt_sdf_p, resolution=args.sdf_resolution)
            except Exception as e:
                print('Failed IoU for', key, e)

        if partial_p:
            try:
                partial_tm = load_geometry_as_mesh(partial_p, resolution=args.sdf_resolution, sdf_level=args.sdf_level)
                sample_metrics['uhd'] = float(np.mean([compute_uhd(pm, partial_tm, num_points=args.num_points) for pm in pred_meshes]))
                tmd_val = compute_tmd(pred_meshes, num_points=args.num_points)
                if tmd_val is not None:
                    sample_metrics['tmd'] = float(tmd_val)
            except Exception as e:
                print('Failed completion metrics for', key, e)

        per_sample[key] = sample_metrics

    summary = {}
    metric_values = {}
    for sample_metrics in per_sample.values():
        for k, v in sample_metrics.items():
            metric_values.setdefault(k, []).append(float(v))

    for k, vals in metric_values.items():
        summary[k] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'count': int(len(vals)),
        }

    payload = {
        'summary': summary,
        'per_sample': per_sample,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)

    print('Saved results to', args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
