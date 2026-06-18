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


def road_polygon(margin=0.0):
    """The road polygon used for cropping (optionally expanded by `margin` metres)."""
    poly = get_road_polygon()
    return poly.buffer(float(margin)) if margin else poly


_polygon = road_polygon  # backwards-compatible alias


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


def crop_preview_figure(points_cropped, margin=0.0, height=620, title=""):
    """Top-down BEV of the cropped points + the road boundary as a black dashed
    line (matches the bundled cropped/vis previews)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    if points_cropped is not None and len(points_cropped):
        fig.add_trace(go.Scattergl(
            x=points_cropped[:, 0], y=points_cropped[:, 1], mode="markers",
            marker=dict(size=2, color="#1f77b4"), hoverinfo="skip", name="cropped"))
    poly = road_polygon(margin)
    geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
    for g in geoms:
        x, y = g.exterior.xy
        fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines",
                                 line=dict(color="black", width=2, dash="dash"),
                                 hoverinfo="skip", showlegend=False))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=30, b=0), title=title, showlegend=False,
        xaxis=dict(title="x (m)"),
        yaxis=dict(title="y (m)", scaleanchor="x", scaleratio=1))
    return fig
