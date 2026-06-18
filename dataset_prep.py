"""Recreate the dataset's *derived* data from the raw TUM Traffic download, so the
whole pipeline is reproducible in-app (no external preprocessing scripts).

Tool 1 — Crop to ROI:
    The bundled `cropped` clouds are the raw south LiDAR clouds clipped to the
    **road polygons** in site_geometry.json (verified: clipping to the road polygon
    reproduces the existing cropped clouds exactly). This regenerates them so.
"""
import glob
import os

import numpy as np
import open3d as o3d

from bg_filter_core import sorted_by_frame_index
from geometry_config import get_road_polygon, points_in_polygon


def crop_points_to_region(points, polygon):
    """Keep points whose (x, y) fall inside `polygon`."""
    if points.shape[0] == 0:
        return points
    mask = points_in_polygon(polygon, points[:, :2])
    return points[mask]


def _polygon(margin=0.0):
    poly = get_road_polygon()
    return poly.buffer(float(margin)) if margin else poly


def crop_dataset(src_dir, out_dir, margin=0.0, max_frames=0, progress=None):
    """Clip every raw cloud in src_dir to the research polygon, writing to out_dir.
    Returns (n_written, total_kept, total_points)."""
    files = sorted_by_frame_index(glob.glob(os.path.join(src_dir, "*.pcd")))
    if max_frames and max_frames > 0:
        files = files[:max_frames]
    poly = _polygon(margin)
    os.makedirs(out_dir, exist_ok=True)
    total_kept, total_pts = 0, 0
    for i, f in enumerate(files):
        pts = np.asarray(o3d.io.read_point_cloud(f).points)
        kept = crop_points_to_region(pts, poly)
        total_pts += len(pts); total_kept += len(kept)
        pc = o3d.geometry.PointCloud()
        if kept.size:
            pc.points = o3d.utility.Vector3dVector(kept)
        o3d.io.write_point_cloud(os.path.join(out_dir, os.path.basename(f)), pc, write_ascii=True)
        if progress:
            progress(i + 1, len(files))
    return len(files), total_kept, total_pts


def validate_crop(src_dir, existing_dir, margin=0.0, n=5):
    """Crop n sample raw frames and compare point counts against the EXISTING
    cropped folder (matched by frame order). Returns a list of per-frame dicts."""
    src = sorted_by_frame_index(glob.glob(os.path.join(src_dir, "*.pcd")))
    exist = sorted_by_frame_index(glob.glob(os.path.join(existing_dir, "*.pcd")))
    poly = _polygon(margin)
    rows = []
    m = min(n, len(src), len(exist))
    idxs = np.unique(np.linspace(0, min(len(src), len(exist)) - 1, max(m, 1)).astype(int))
    for i in idxs:
        raw = np.asarray(o3d.io.read_point_cloud(src[int(i)]).points)
        kept = crop_points_to_region(raw, poly)
        existing = np.asarray(o3d.io.read_point_cloud(exist[int(i)]).points)
        denom = max(len(existing), 1)
        rows.append({
            "frame": int(i),
            "raw": len(raw),
            "ours": len(kept),
            "existing": len(existing),
            "match %": round(100.0 * (1 - abs(len(kept) - len(existing)) / denom), 1),
        })
    return rows
