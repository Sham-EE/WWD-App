"""Multi-LiDAR point-cloud registration for the s110 site (calibration-first).

Both Ouster LiDARs (south + north) carry their *sensor → s110_base* extrinsic in
every OpenLABEL label, under
``coordinate_systems[<sensor>].pose_wrt_parent.matrix4x4`` with ``parent =
s110_base``. The sensors are static, so this 4x4 is **constant across frames**
(verified: max frame-to-frame difference = 0). Registration is therefore
*deterministic*: transform each cloud by its sensor→base matrix and the two
clouds overlay in the common ``s110_base`` frame — no guessing, no global
optimisation needed.

This module:
  * reads those calibration matrices straight from the labels (cached),
  * nearest-timestamp pairs south/north frames (the two sensors fire ~async),
  * fuses a pair into ``s110_base`` (tagging each point with its source sensor),
  * optionally ICP-refines north→south as a *check/correction* on the bundled
    calibration (reports fitness + RMSE so you can trust-but-verify),
  * batch-writes a fused dataset to ``data/derived/point_clouds/registered/`` plus a
    ``registration.json`` manifest (matrices + ICP result + provenance).

The dev kit bakes the calibration into the labels and trusts it; we follow that
(it is the objective ground truth) and add the ICP read-out on top so the
overlay quality is *measured*, not assumed.
"""
import glob
import json
import os

import numpy as np

# OpenLABEL coordinate-system names for the two LiDARs, keyed by our short side id.
SENSORS = {
    "south": "s110_lidar_ouster_south",
    "north": "s110_lidar_ouster_north",
}
BASE_FRAME = "s110_base"

# Distinct colours for the per-sensor ("do they line up?") QA view.
SENSOR_COLORS = {"south": "#36c5f0", "north": "#ff7a59"}


# --------------------------------------------------------------------------- #
# Calibration (sensor -> s110_base), read from the labels
# --------------------------------------------------------------------------- #
def read_sensor_to_base(label_path, side):
    """4x4 ``sensor → s110_base`` transform from one OpenLABEL label file."""
    with open(label_path, "r", encoding="utf-8") as f:
        ol = json.load(f)["openlabel"]
    cs = ol.get("coordinate_systems", {})
    node = cs.get(SENSORS[side], {})
    pose = node.get("pose_wrt_parent", {}).get("matrix4x4")
    if pose is None:
        raise ValueError(f"No pose_wrt_parent for {SENSORS[side]} in {label_path}")
    parent = node.get("parent")
    if parent and parent != BASE_FRAME:
        raise ValueError(f"{SENSORS[side]} parent is {parent!r}, expected {BASE_FRAME!r}")
    return np.asarray(pose, dtype=float).reshape(4, 4)


def calibration_for(label_dir, side):
    """Read the (constant) ``sensor → s110_base`` matrix from the first label in
    a directory. Returns ``None`` if no usable label is found."""
    files = sorted(glob.glob(os.path.join(label_dir, "*.json")))
    for f in files:
        try:
            return read_sensor_to_base(f, side)
        except Exception:
            continue
    return None


def calibration_is_constant(label_dir, side, sample=8):
    """Sanity check: confirm the matrix doesn't drift across frames. Returns the
    max element-wise deviation over a sample of frames (0.0 = perfectly static)."""
    files = sorted(glob.glob(os.path.join(label_dir, "*.json")))
    if not files:
        return None
    idx = np.unique(np.linspace(0, len(files) - 1, min(sample, len(files))).astype(int))
    mats = []
    for i in idx:
        try:
            mats.append(read_sensor_to_base(files[int(i)], side))
        except Exception:
            pass
    if not mats:
        return None
    m0 = mats[0]
    return float(max(np.abs(m - m0).max() for m in mats))


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def transform_points(pts, M):
    """Apply a 4x4 homogeneous transform to an Nx3 array of points."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] == 0:
        return pts.reshape(0, 3)
    hom = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1))])
    return (M @ hom.T).T[:, :3]


def load_xyz(pcd_path, max_points=0, seed=0):
    """Load a .pcd as an Nx3 float array, optionally random-downsampled."""
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(pcd_path).points, dtype=np.float64)
    if max_points and pts.shape[0] > max_points:
        idx = np.random.default_rng(seed).choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    return pts


def write_xyz(pcd_path, pts):
    """Write an Nx3 array to a .pcd (binary)."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64)[:, :3])
    os.makedirs(os.path.dirname(pcd_path) or ".", exist_ok=True)
    o3d.io.write_point_cloud(pcd_path, pc, write_ascii=False)


# --------------------------------------------------------------------------- #
# Frame pairing (sensors fire asynchronously -> match nearest timestamp)
# --------------------------------------------------------------------------- #
def _ts_ns(path):
    """Capture timestamp in nanoseconds from a TUMTraf filename `<sec>_<nsec>_...`."""
    base = os.path.basename(path)
    parts = base.split("_")
    try:
        return int(parts[0]) * 1_000_000_000 + int(parts[1])
    except (IndexError, ValueError):
        digits = "".join(c for c in os.path.splitext(base)[0] if c.isdigit())
        return int(digits) if digits else 0


def match_frame_pairs(south_files, north_files, max_dt_ms=None):
    """Pair each south file with its nearest-in-time north file.

    Returns a list of ``(south_path, north_path, dt_ms)`` sorted by time. If
    ``max_dt_ms`` is set, pairs further apart than that are dropped."""
    south = sorted(south_files, key=_ts_ns)
    north = sorted(north_files, key=_ts_ns)
    if not south or not north:
        return []
    n_ts = np.array([_ts_ns(p) for p in north])
    order = np.argsort(n_ts)
    n_ts_sorted = n_ts[order]
    pairs = []
    for sp in south:
        st = _ts_ns(sp)
        j = int(np.searchsorted(n_ts_sorted, st))
        cands = [k for k in (j - 1, j) if 0 <= k < len(n_ts_sorted)]
        best = min(cands, key=lambda k: abs(int(n_ts_sorted[k]) - st))
        dt_ms = abs(int(n_ts_sorted[best]) - st) / 1e6
        if max_dt_ms is None or dt_ms <= max_dt_ms:
            pairs.append((sp, north[order[best]], dt_ms))
    return pairs


# --------------------------------------------------------------------------- #
# Fusion + ICP
# --------------------------------------------------------------------------- #
def fuse_pair(south_pts, north_pts, M_south, M_north, refine=None):
    """Transform both clouds into ``s110_base`` and stack them.

    ``refine`` (optional 4x4) is an extra correction applied to the north cloud
    *after* its calibration transform (e.g. an ICP delta). Returns a dict with
    the per-sensor base-frame clouds, the stacked cloud, and an int source tag
    (0 = south, 1 = north)."""
    s_base = transform_points(south_pts, M_south)
    n_base = transform_points(north_pts, M_north)
    if refine is not None:
        n_base = transform_points(n_base, refine)
    pts = np.vstack([s_base, n_base]) if (len(s_base) or len(n_base)) else np.zeros((0, 3))
    src = np.concatenate([np.zeros(len(s_base), dtype=int), np.ones(len(n_base), dtype=int)])
    return {"south": s_base, "north": n_base, "points": pts, "source": src}


def icp_refine(north_base, south_base, max_dist=0.5, max_iter=60, voxel=0.25,
               point_to_plane=True, coarse_to_fine=True):
    """ICP aligning the north cloud onto the south cloud (both already in
    ``s110_base``). Returns ``(delta_4x4, info)`` where ``delta`` is the
    correction to apply to the north cloud and ``info`` has fitness/RMSE + the
    yaw/translation magnitude of the correction.

    The bundled extrinsics get the ground plane right but carry a notable
    **relative yaw** error (~8°) between the two sensors, so a single tight ICP
    pass gets stuck in a local minimum. This runs **coarse-to-fine**
    (5 → 2 → 1 → ``max_dist`` m correspondence distance) with **point-to-plane**
    by default, which reliably recovers the yaw. Because the rig is static the
    resulting correction is constant, so it can be computed once and applied to
    every frame."""
    import open3d as o3d

    def _pc(p):
        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(np.asarray(p, dtype=np.float64)[:, :3])
        return c

    src = _pc(north_base)
    tgt = _pc(south_base)
    if voxel and voxel > 0:
        src = src.voxel_down_sample(voxel)
        tgt = tgt.voxel_down_sample(voxel)
    if point_to_plane:
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 4, 1.0), max_nn=30))
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    if coarse_to_fine:
        schedule = sorted({d for d in (5.0, 2.0, 1.0, float(max_dist)) if d >= float(max_dist)},
                          reverse=True)
    else:
        schedule = [float(max_dist)]

    T = np.eye(4)
    res = None
    for md in schedule:
        res = o3d.pipelines.registration.registration_icp(
            src, tgt, md, T, estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter)))
        T = np.asarray(res.transformation, dtype=float)

    t = T[:3, 3]
    R = T[:3, :3]
    cos = (np.trace(R) - 1.0) / 2.0
    ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    info = {
        "fitness": float(res.fitness),          # fraction of points with a match
        "inlier_rmse": float(res.inlier_rmse),  # metres
        "translation_m": float(np.linalg.norm(t)),
        "rotation_deg": ang,
        "yaw_deg": yaw,                          # the dominant error component
        "max_dist": float(max_dist),
        "max_iter": int(max_iter),
        "point_to_plane": bool(point_to_plane),
    }
    return T, info


# --------------------------------------------------------------------------- #
# Batch registration -> data/derived/point_clouds/registered/
# --------------------------------------------------------------------------- #
def register_dataset(south_pcd_dir, north_pcd_dir, M_south, M_north, out_dir,
                     refine=None, max_dt_ms=None, max_frames=0, progress=None,
                     manifest_extra=None):
    """Fuse every matched south/north pair into ``s110_base`` and write the
    combined clouds to ``out_dir`` (named by the south timestamp). Writes a
    ``registration.json`` manifest alongside. Returns (n_written, n_pairs)."""
    south_files = glob.glob(os.path.join(south_pcd_dir, "*.pcd"))
    north_files = glob.glob(os.path.join(north_pcd_dir, "*.pcd"))
    pairs = match_frame_pairs(south_files, north_files, max_dt_ms=max_dt_ms)
    if max_frames and max_frames > 0:
        pairs = pairs[:max_frames]
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    dts = []
    for sp, npath, dt_ms in pairs:
        s_pts = load_xyz(sp)
        n_pts = load_xyz(npath)
        fused = fuse_pair(s_pts, n_pts, M_south, M_north, refine=refine)
        name = os.path.splitext(os.path.basename(sp))[0]
        name = name.replace("_s110_lidar_ouster_south", "") + "_registered.pcd"
        write_xyz(os.path.join(out_dir, name), fused["points"])
        dts.append(dt_ms)
        n += 1
        if progress:
            progress(n, len(pairs))
    manifest = {
        "method": "calibration-first (sensor->s110_base from OpenLABEL labels)",
        "base_frame": BASE_FRAME,
        "M_south_to_base": np.asarray(M_south).tolist(),
        "M_north_to_base": np.asarray(M_north).tolist(),
        "icp_refine_applied": refine is not None,
        "icp_refine_delta": (np.asarray(refine).tolist() if refine is not None else None),
        "n_pairs": len(pairs),
        "n_written": n,
        "max_pair_dt_ms": (max(dts) if dts else None),
        "mean_pair_dt_ms": (float(np.mean(dts)) if dts else None),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    with open(os.path.join(out_dir, "registration.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return n, len(pairs)


# --------------------------------------------------------------------------- #
# Preview figures (south vs north overlay / height colour + geometry overlays)
# --------------------------------------------------------------------------- #
def _road_window(margin=12.0):
    """(xlo, ylo, xhi, yhi) road bounding box (+margin) used to frame the view —
    same trick as the background-filtering inspector: clipping to the road window
    keeps the camera zoom calibrated regardless of stray distant returns."""
    try:
        from geometry_config import get_road_polygon
        b = get_road_polygon().bounds
        return b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin
    except Exception:
        return -60.0, -60.0, 60.0, 60.0


def _clip_to_window(p, win):
    if len(p) == 0:
        return p
    xlo, ylo, xhi, yhi = win
    m = (p[:, 0] >= xlo) & (p[:, 0] <= xhi) & (p[:, 1] >= ylo) & (p[:, 1] <= yhi)
    return p[m]


def _geometry_overlay_traces(go, show_road, show_roi, z_floor=0.0):
    traces = []
    try:
        if show_road:
            from geometry_config import get_road_polygon
            xs, ys = get_road_polygon().exterior.xy
            traces.append(go.Scatter3d(x=list(xs), y=list(ys), z=[z_floor] * len(xs),
                                       mode="lines", name="Road",
                                       line=dict(color="#39ff14", width=4), showlegend=True))
        if show_roi:
            from geometry_config import get_research_polygon
            xs, ys = get_research_polygon().exterior.xy
            traces.append(go.Scatter3d(x=list(xs), y=list(ys), z=[z_floor] * len(xs),
                                       mode="lines", name="ROI",
                                       line=dict(color="#00e5ff", width=3, dash="dot"), showlegend=True))
    except Exception:
        pass
    return traces


def _apply_scene(go, traces, height, title, zoom, azimuth, elevation, rev):
    el = np.radians(elevation)
    az = np.radians(azimuth)
    r = float(zoom) * np.sqrt(3.0)
    eye = dict(x=float(r * np.cos(el) * np.cos(az)),
               y=float(r * np.cos(el) * np.sin(az)),
               z=float(r * np.sin(el)))
    fig = go.Figure(data=traces)
    fig.update_layout(
        height=height, title=title, showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(14,17,23,0.6)"),
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        paper_bgcolor="#0e1117", font=dict(color="#c9d1d9"),
        uirevision=rev,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#0e1117",
            camera=dict(eye=eye, up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
            uirevision=rev,
        ),
    )
    return fig


# Marker colours for the LiDAR position overlays (coordinated with, but brighter
# than, the south(blue)/north(red) cloud scheme).
SENSOR_MARKER_COLORS = {"South": "#1e90ff", "North": "#ff1744"}


def lidar_markers(ds, sensor):
    """LiDAR-position markers for a viewer showing `sensor` data, as a list of
    ``{name, pos, color}``. For **south/north** the data is in that sensor's own
    frame, so the LiDAR sits at the origin. For **registered** (s110_base) both
    LiDARs sit at their calibrated positions (north shifted by the ICP delta from
    the registration manifest, if present). Returns [] if positions can't be read."""
    if sensor in ("south", "north"):
        name = sensor.capitalize()
        return [{"name": name, "pos": [0.0, 0.0, 0.0], "color": SENSOR_MARKER_COLORS[name]}]
    # registered -> both sensors in s110_base
    out = []
    Ms = calibration_for(ds.raw_labels_south_dir, "south")
    if Ms is not None:
        out.append({"name": "South", "pos": [float(v) for v in Ms[:3, 3]],
                    "color": SENSOR_MARKER_COLORS["South"]})
    Mn = calibration_for(ds.raw_labels_north_dir, "north")
    if Mn is not None:
        npos = Mn[:3, 3]
        man = os.path.join(ds.registered_dir, "registration.json")
        try:
            with open(man, "r", encoding="utf-8") as f:
                delta = json.load(f).get("icp_refine_delta")
            if delta:
                npos = (np.asarray(delta) @ np.append(npos, 1.0))[:3]
        except Exception:
            pass
        out.append({"name": "North", "pos": [float(v) for v in npos],
                    "color": SENSOR_MARKER_COLORS["North"]})
    return out


def sensor_marker_traces(go, sensors, z_floor=0.0):
    """Diamond at each LiDAR's position + dotted plumb line to its nadir (the
    ground point under it = the blank spot in that sensor's points)."""
    traces = []
    for s in sensors or []:
        x, y, z = s["pos"]
        col = s.get("color", "#ffffff")
        name = s["name"]
        traces.append(go.Scatter3d(
            x=[x], y=[y], z=[z], mode="markers+text", name=f"{name} LiDAR",
            text=[f"  {name} LiDAR ({x:.1f}, {y:.1f}, {z:.1f})"], textposition="top center",
            textfont=dict(color=col, size=12),
            marker=dict(size=6, color=col, symbol="diamond", line=dict(color="white", width=1)),
            showlegend=True))
        traces.append(go.Scatter3d(
            x=[x, x], y=[y, y], z=[z, z_floor], mode="lines", showlegend=False,
            line=dict(color=col, width=3, dash="dot")))
        traces.append(go.Scatter3d(
            x=[x], y=[y], z=[z_floor], mode="markers", showlegend=False,
            marker=dict(size=5, color=col, symbol="x")))
    return traces


def registration_figure(fused, color_mode="by_sensor", height_span=4.0,
                        show_south=True, show_north=True, show_road=False,
                        show_roi=False, height=640, title="",
                        zoom=0.9, azimuth=45.0, elevation=35.0, margin=12.0,
                        clip=True, sensors=None):
    """Plotly 3D figure of a south/north pair.

    Works for both the **registered** view (points already in ``s110_base``;
    ``clip=True`` frames the site by the road window) and the **raw** view
    (points still in each sensor's own frame; pass ``clip=False`` so they're not
    clipped against the base-frame road box).

    ``color_mode``: ``"by_sensor"`` (south/north distinct — the alignment QA
    view) or ``"by_height"`` (Turbo z-ramp like the dev kit)."""
    import plotly.graph_objects as go

    s_base = fused["south"]
    n_base = fused["north"]
    traces = []
    win = _road_window(margin)

    def _clip(p):
        return _clip_to_window(p, win) if clip else p

    if color_mode == "by_height":
        allp = np.vstack([p for p, show in ((s_base, show_south), (n_base, show_north)) if show and len(p)]) \
            if (show_south and len(s_base)) or (show_north and len(n_base)) else np.zeros((0, 3))
        z0 = float(np.percentile(allp[:, 2], 1)) if len(allp) else 0.0
        for p, show, name in ((s_base, show_south, "South"), (n_base, show_north, "North")):
            if not show:
                continue
            p = _clip(p)
            if not len(p):
                continue
            traces.append(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name=name,
                marker=dict(size=1.5, color=p[:, 2], colorscale="Turbo",
                            cmin=z0, cmax=z0 + float(height_span), opacity=0.6, showscale=False)))
    else:
        for p, show, name, col in ((s_base, show_south, "South", SENSOR_COLORS["south"]),
                                   (n_base, show_north, "North", SENSOR_COLORS["north"])):
            if not show:
                continue
            p = _clip(p)
            if not len(p):
                continue
            traces.append(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name=name,
                marker=dict(size=1.5, color=col, opacity=0.55)))

    traces += _geometry_overlay_traces(go, show_road, show_roi)
    traces += sensor_marker_traces(go, sensors)
    rev = f"reg_{zoom}_{azimuth}_{elevation}"
    return _apply_scene(go, traces, height, title, zoom, azimuth, elevation, rev)
