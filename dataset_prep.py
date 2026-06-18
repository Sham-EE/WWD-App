"""Recreate the dataset's *derived* data from the raw TUM Traffic download, so the
whole pipeline is reproducible in-app (no external preprocessing scripts).

Tool 1 — Crop to ROI:
    The bundled `cropped` clouds are the raw south LiDAR clouds clipped to the
    **road polygons** in site_geometry.json (verified: clipping to the road polygon
    reproduces the existing cropped clouds exactly). This regenerates them so.
"""
import copy
import glob
import json
import os

import numpy as np
import open3d as o3d

from bg_filter_core import sorted_by_frame_index
from geometry_config import get_road_polygon, get_research_polygon, points_in_polygon


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


def crop_preview_figure(points, margin=0.0, height=620, title="", draw_boundary=True):
    """Top-down BEV of `points` + (optionally) the road boundary as a black dashed
    line (matches the bundled cropped/vis previews). `points` may be the cropped or
    the full/uncropped cloud."""
    import plotly.graph_objects as go
    poly = road_polygon(margin)
    fig = go.Figure()
    if points is not None and len(points):
        fig.add_trace(go.Scattergl(
            x=points[:, 0], y=points[:, 1], mode="markers",
            marker=dict(size=2, color="#1f77b4"), hoverinfo="skip", name="points"))
    if draw_boundary:
        geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for g in geoms:
            x, y = g.exterior.xy
            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines",
                                     line=dict(color="limegreen", width=3, dash="dash"),
                                     hoverinfo="skip", showlegend=False))
    # Lock the view to the road region (square) so cropped vs full share the SAME
    # zoom — full no longer zooms out to the ~200 m raw extent.
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny) / 2 + 6.0
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=30, b=0), title=title, showlegend=False,
        dragmode="pan",
        xaxis=dict(title="x (m)", range=[cx - half, cx + half]),
        yaxis=dict(title="y (m)", range=[cy - half, cy + half], scaleanchor="x", scaleratio=1),
        # Lock the user's pan/zoom across frame steps AND the cropped/full toggle.
        # Set your view once; double-click in the plot to reset to this default.
        uirevision="dp_crop")
    return fig


# ============== Tool 2: Scorable ground truth (visible-only) ==============
# The bundled `labels_visible_south` was made by an opaque per-frame visibility
# check we can't reproduce from the labels. Instead this builds a transparent,
# reproducible "scorable GT": objects in the processed ROI (research polygon) that
# actually have LiDAR points — the right basis for fair evaluation.

def research_region(margin=0.0):
    poly = get_research_polygon()
    return poly.buffer(float(margin)) if margin else poly


def _obj_num_points(cuboid):
    for a in cuboid.get("attributes", {}).get("num", []):
        if a.get("name") == "num_points":
            return a.get("val", 0) or 0
    return 0


def _obj_occlusion(cuboid):
    for a in cuboid.get("attributes", {}).get("text", []):
        if a.get("name") == "occlusion_level":
            return a.get("val", "")
    return ""


OCCLUSION_LEVELS = ["NOT_OCCLUDED", "PARTIALLY_OCCLUDED", "MOSTLY_OCCLUDED", "FULLY_OCCLUDED"]
SCORABLE_CLASSES = ["CAR", "TRUCK", "VAN", "BUS", "TRAILER", "MOTORCYCLE",
                    "PEDESTRIAN", "BICYCLE", "EMERGENCY_VEHICLE", "OTHER"]
DEFAULT_CRITERIA = {"min_points": 1, "max_points": None, "max_range": None,
                    "drop_occlusion": (), "classes": None}


def _keep_object(cuboid, obj_type, region_poly, crit):
    """Decide if an object is scorable, given a criteria dict (see DEFAULT_CRITERIA)."""
    v = cuboid.get("val")
    if not v or len(v) < 3:
        return False
    x, y = v[0], v[1]
    if not points_in_polygon(region_poly, np.array([[x, y]]))[0]:
        return False
    npn = _obj_num_points(cuboid)
    if npn < crit.get("min_points", 1):
        return False
    mx = crit.get("max_points")
    if mx is not None and npn > mx:
        return False
    mr = crit.get("max_range")
    if mr is not None and (x * x + y * y) ** 0.5 > mr:
        return False
    if _obj_occlusion(cuboid) in (crit.get("drop_occlusion") or ()):
        return False
    classes = crit.get("classes")
    if classes is not None and str(obj_type).upper() not in classes:
        return False
    return True


def _box_footprint(val):
    """Closed top-down (x, y) rectangle for a cuboid [x,y,z, qx,qy,qz,qw, l,w,h]."""
    x, y = val[0], val[1]
    qx, qy, qz, qw = val[3], val[4], val[5], val[6]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    l, w = val[7] / 2.0, val[8] / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[l, w], [l, -w], [-l, -w], [-l, w], [l, w]])
    return local @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def scorable_classify(label_path, region_poly, crit=None):
    """Return (kept_boxes, dropped_boxes): lists of closed (x,y) box footprints."""
    crit = crit or DEFAULT_CRITERIA
    ol = json.load(open(label_path))["openlabel"]
    frames = ol.get("frames", {})
    kept, dropped = [], []
    if frames:
        fr = frames[next(iter(frames))]
        for o in fr.get("objects", {}).values():
            cub = o["object_data"].get("cuboid", {})
            v = cub.get("val")
            if not v or len(v) < 10:
                continue
            fp = _box_footprint(v)
            ok = _keep_object(cub, o["object_data"].get("type"), region_poly, crit)
            (kept if ok else dropped).append(fp)
    return kept, dropped


def generate_scorable_gt(src_label_dir, out_dir, region_poly, crit=None,
                         max_frames=0, progress=None):
    """Write filtered OpenLABEL files (same structure, subset of objects) to out_dir.
    Returns (n_files, kept, total)."""
    crit = crit or DEFAULT_CRITERIA
    files = sorted_by_frame_index(glob.glob(os.path.join(src_label_dir, "*.json")))
    if max_frames and max_frames > 0:
        files = files[:max_frames]
    os.makedirs(out_dir, exist_ok=True)
    kept_tot, total = 0, 0
    for i, f in enumerate(files):
        ol = json.load(open(f))["openlabel"]
        out = copy.deepcopy(ol)
        for fid, fr in out.get("frames", {}).items():
            objs = fr.get("objects", {})
            keep = {}
            for oid, o in objs.items():
                total += 1
                if _keep_object(o["object_data"].get("cuboid", {}), o["object_data"].get("type"),
                                region_poly, crit):
                    keep[oid] = o
                    kept_tot += 1
            fr["objects"] = keep
        with open(os.path.join(out_dir, os.path.basename(f)), "w", encoding="utf-8") as fp:
            json.dump({"openlabel": out}, fp)
        if progress:
            progress(i + 1, len(files))
    return len(files), kept_tot, total


def _boxes_xy(boxes):
    """Flatten a list of footprints into one polyline (None-separated) for Plotly."""
    xs, ys = [], []
    for fp in boxes:
        xs += list(fp[:, 0]) + [None]
        ys += list(fp[:, 1]) + [None]
    return xs, ys


def scorable_preview_figure(points, kept_boxes, dropped_boxes, region_poly,
                            height=620, title="", max_points=40000):
    """BEV like the bundled vis: point cloud (blue) + kept GT boxes (green) and
    dropped GT boxes (red), locked to the ROI region."""
    import plotly.graph_objects as go
    fig = go.Figure()
    if points is not None and len(points):
        if len(points) > max_points:
            points = points[np.random.default_rng(0).choice(len(points), max_points, replace=False)]
        fig.add_trace(go.Scattergl(x=points[:, 0], y=points[:, 1], mode="markers",
                                   marker=dict(size=2, color="#1f77b4"), hoverinfo="skip", showlegend=False))
    dx, dy = _boxes_xy(dropped_boxes)
    if dx:
        fig.add_trace(go.Scatter(x=dx, y=dy, mode="lines", line=dict(color="red", width=2),
                                 name="dropped", hoverinfo="skip"))
    kx, ky = _boxes_xy(kept_boxes)
    if kx:
        fig.add_trace(go.Scatter(x=kx, y=ky, mode="lines", line=dict(color="limegreen", width=2),
                                 name="kept (scorable)", hoverinfo="skip"))
    # Fit the view to the ROI AND every object (kept + dropped) so all boxes are on
    # screen — the visible green+red count then equals the title's kept/total.
    xs = [region_poly.bounds[0], region_poly.bounds[2]]
    ys = [region_poly.bounds[1], region_poly.bounds[3]]
    for fp in list(kept_boxes) + list(dropped_boxes):
        xs += [float(fp[:, 0].min()), float(fp[:, 0].max())]
        ys += [float(fp[:, 1].min()), float(fp[:, 1].max())]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 6.0
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), title=title,
                      legend=dict(orientation="h", y=1.02, x=0),
                      dragmode="pan",
                      xaxis=dict(title="x (m)", range=[cx - half, cx + half]),
                      yaxis=dict(title="y (m)", range=[cy - half, cy + half], scaleanchor="x", scaleratio=1),
                      uirevision="dp_gt")  # keep scroll-zoom/pan across frame steps
    return fig
