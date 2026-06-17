"""Render OpenLABEL 3D labels (and the LiDAR point cloud) onto camera images.

Self-contained replacement for the TUM Traffic dev-kit's image visualization:
the camera calibration is read straight out of each OpenLABEL label JSON
(``coordinate_systems[cam].pose_wrt_parent`` for extrinsics +
``streams[cam].stream_properties.intrinsics_pinhole.camera_matrix_3x4`` for
intrinsics), so no dev-kit, environment, or external calibration files are needed.

Projection:  pixel ~ K_3x4 · T_cam_from_lidar · [X, Y, Z, 1]ᵀ  (divide by depth).
"""
import json
import os

import numpy as np

# Bump when the rendering/palette changes so cached overlays auto-invalidate
# (the Road Viewer includes this in the cache folder name).
RENDER_VERSION = "v5"

# Per-category colours (RGB) — EXACTLY the TUM Traffic dev-kit values
# (id_to_class_name_mapping[...]["color_rgb"] in src/utils/utils.py).
CATEGORY_COLORS = {
    "CAR": (0, 204, 246), "TRUCK": (63, 233, 185), "TRAILER": (90, 255, 126),
    "VAN": (235, 207, 54), "MOTORCYCLE": (185, 164, 84), "BUS": (217, 138, 134),
    "PEDESTRIAN": (233, 118, 249), "BICYCLE": (177, 140, 255),
    "EMERGENCY_VEHICLE": (102, 107, 250), "OTHER": (199, 199, 199),
    "LICENSE_PLATE_LOCATION": (0, 0, 0),
}
_DEFAULT_COLOR = (199, 199, 199)


def _font(size=15):
    from PIL import ImageFont
    try:
        import matplotlib
        return ImageFont.truetype(
            os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def _label_text(obj):
    """Dev-kit style: TYPE_<first 3 chars of the id suffix>, e.g. CAR_048."""
    name = obj.get("name") or ""
    suffix = name.rsplit("_", 1)[-1] if "_" in name else str(obj.get("id", ""))
    return f"{obj['type']}_{suffix[:3]}" if suffix else str(obj["type"])

# Box edges (corner indices); corners 0-3 = top face (+z), 4-7 = bottom face.
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]
# +x ("front") face = corners 0,1,5,4 -> diagonals mark the heading direction.
_FRONT_DIAGONALS = [(0, 5), (1, 4)]


def camera_id_from_image(path):
    """`1646..._057..._s110_camera_basler_south1_8mm.jpg` -> camera id."""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else base


def load_calibration(label_json_path):
    """{camera_id: {K(3x4), T(cam_from_lidar 4x4), width, height}} from a label JSON."""
    with open(label_json_path, "r", encoding="utf-8") as f:
        ol = json.load(f)["openlabel"]
    cs = ol.get("coordinate_systems", {})
    streams = ol.get("streams", {})
    out = {}
    for cam, info in streams.items():
        if info.get("type") != "camera":
            continue
        pin = info.get("stream_properties", {}).get("intrinsics_pinhole", {})
        K = pin.get("camera_matrix_3x4")
        pose = cs.get(cam, {}).get("pose_wrt_parent", {}).get("matrix4x4")
        if K is None or pose is None:
            continue
        K = np.asarray(K, dtype=float)                      # 3x4
        # In TUMTraf, pose_wrt_parent is stored as the camera-from-LiDAR transform
        # (verified empirically: labels project into the image with it directly,
        # not its inverse).
        T_cam_from_lidar = np.asarray(pose, dtype=float).reshape(4, 4)
        out[cam] = {"K": K, "T": T_cam_from_lidar,
                    "width": pin.get("width_px"), "height": pin.get("height_px")}
    return out


def load_objects(label_json_path):
    """List of {id, type, val(10)} cuboids from a single-frame label JSON."""
    with open(label_json_path, "r", encoding="utf-8") as f:
        ol = json.load(f)["openlabel"]
    frames = ol.get("frames", {})
    if not frames:
        return []
    fr = frames[next(iter(frames))]
    objs = []
    for oid, o in fr.get("objects", {}).items():
        od = o.get("object_data", {})
        val = od.get("cuboid", {}).get("val")
        if val and len(val) >= 10:
            objs.append({"id": oid, "type": od.get("type", "OTHER"),
                         "name": od.get("name", ""), "val": val})
    return objs


def _project(points_xyz, K, T):
    """Project Nx3 LiDAR points -> (u, v, depth, valid_in_front)."""
    n = points_xyz.shape[0]
    hom = np.hstack([points_xyz, np.ones((n, 1))])     # N x 4
    cam = (T @ hom.T).T                                # N x 4 (camera frame)
    z = cam[:, 2]
    uvw = (K @ cam.T).T                                # N x 3
    w = uvw[:, 2]
    safe = np.abs(w) > 1e-6
    u = np.where(safe, uvw[:, 0] / np.where(safe, w, 1), -1)
    v = np.where(safe, uvw[:, 1] / np.where(safe, w, 1), -1)
    return u, v, z, (z > 0.1) & safe


def cuboid_corners(val):
    """8 corners (world/LiDAR frame) from [x,y,z, qx,qy,qz,qw, l,w,h]."""
    from scipy.spatial.transform import Rotation
    x, y, z = val[0], val[1], val[2]
    R = Rotation.from_quat([val[3], val[4], val[5], val[6]]).as_matrix()
    dx, dy, dz = val[7] / 2.0, val[8] / 2.0, val[9] / 2.0
    local = np.array([[dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
                      [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz]])
    return (R @ local.T).T + np.array([x, y, z])


def _color_for(obj, color_mode):
    if color_mode == "by_track_id":
        h = abs(hash(obj["id"]))
        return (60 + h % 180, 60 + (h // 180) % 180, 60 + (h // 32400) % 180)
    return CATEGORY_COLORS.get(str(obj["type"]).upper(), _DEFAULT_COLOR)


def _depth_colors(distances, dmax=None):
    """Distance -> RGB via 'jet' (near=blue, far=red). The full spectrum is
    stretched across the 1st–99th percentile of the visible ranges, so the
    gradient uses its whole range (wider colour spread, like the dev-kit)."""
    from matplotlib import cm
    d = np.asarray(distances, dtype=float)
    if d.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo = float(np.percentile(d, 1))
    hi = float(dmax) if dmax else float(np.percentile(d, 99))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((d - lo) / (hi - lo), 0, 1)          # near 0 -> blue, far 1 -> red
    rgba = cm.get_cmap("jet")(norm)
    return (rgba[:, :3] * 255).astype(np.uint8)


def load_pointcloud(pcd_path):
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(pcd_path).points, dtype=np.float32)


def render_frame(image_path, label_json_path, camera_id, mode="box3d",
                 color_mode="by_category", pcd_path=None, depth_max=None,
                 point_size=2, line_width=2, draw_labels=True, label_size=26):
    """Render labels (and optionally the point cloud) onto one camera image.
    mode: 'box3d' (3D wireframes) or 'point_cloud' (points + 3D wireframes).
    Returns a PIL.Image."""
    from PIL import Image, ImageDraw
    calib = load_calibration(label_json_path).get(camera_id)
    img = Image.open(image_path).convert("RGB")
    if calib is None:
        return img
    K, T = calib["K"], calib["T"]
    W, H = img.size

    # --- point-cloud overlay (vectorised pixel writes) ---
    if mode == "point_cloud" and pcd_path and os.path.exists(pcd_path):
        pts = load_pointcloud(pcd_path)
        if pts.size:
            u, v, z, valid = _project(pts, K, T)
            rng = np.linalg.norm(pts[:, :3], axis=1)         # range from the sensor
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            inb = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if np.any(inb):
                arr = np.asarray(img).copy()
                d, uu, vv = rng[inb], ui[inb], vi[inb]
                order = np.argsort(-d)                        # far first; near drawn on top
                d, uu, vv = d[order], uu[order], vv[order]
                cols = _depth_colors(d, depth_max)
                r = max(1, int(point_size))                  # filled square, radius r
                for dyp in range(-r, r + 1):
                    for dxp in range(-r, r + 1):
                        yy, xx = np.clip(vv + dyp, 0, H - 1), np.clip(uu + dxp, 0, W - 1)
                        arr[yy, xx] = cols
                img = Image.fromarray(arr)

    # --- 3D boxes ---
    draw = ImageDraw.Draw(img)
    font = _font(label_size) if draw_labels else None
    for obj in load_objects(label_json_path):
        corners = cuboid_corners(obj["val"])
        u, v, z, valid = _project(corners, K, T)
        if valid.sum() < 4:
            continue
        col = _color_for(obj, color_mode)
        pix = list(zip(u, v))
        for a, b in _EDGES + _FRONT_DIAGONALS:
            if valid[a] and valid[b]:
                draw.line([pix[a], pix[b]], fill=col, width=line_width)
        if draw_labels and valid.any():
            tx = float(np.min(u[valid])); ty = float(np.min(v[valid]))
            if 0 <= tx < W and 0 <= ty < H:
                text = _label_text(obj)
                tx = min(max(tx, 0), W - 1); ty = max(0, ty - label_size - 6)
                bb = draw.textbbox((tx, ty), text, font=font)
                draw.rectangle([bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2], fill=col)  # solid tag
                draw.text((tx, ty), text, fill=(0, 0, 0), font=font)                    # black text
    return img


def render_cached(image_path, label_path, camera_id, mode, out_dir,
                  color_mode="by_category", pcd_path=None, point_size=1, force=False):
    """Render (or reuse a cached) labelled image. Returns the output path."""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(image_path))
    if force or not os.path.exists(out):
        im = render_frame(image_path, label_path, camera_id, mode=mode,
                          color_mode=color_mode, pcd_path=pcd_path, point_size=point_size)
        im.save(out, quality=90)
    return out
