"""3D LiDAR label visualization (dev-kit feature 1.2) using Plotly.

Draws a LiDAR scan + its ground-truth 3D boxes in the sensor frame — no
projection/calibration needed (points and cuboids share the LiDAR frame), and no
Open3D GUI / OpenGL (the thing that breaks cross-platform): pure Plotly, so it
runs anywhere Streamlit does. Reuses the validated cuboid corners + dev-kit
category colours from label_projection.
"""
import numpy as np

import label_projection as lp

# Oblique, z-up perspective — the same "horizontal angle" as the Background
# Filtering viewer (azimuth 45°, elevation 35° → eye ≈ a (1,1,1) corner view).
_OBLIQUE = dict(eye=dict(x=1.05, y=1.05, z=1.0), up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0), projection=dict(type="perspective"))
_BG = "#0e1117"
_POINT_COLOR = "#8b929c"


def load_points(pcd_path, max_points=20000):
    """Load a .pcd as Nx3, randomly downsampled to max_points for snappy 3D."""
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(pcd_path).points, dtype=np.float32)
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    return pts


def _hex(rgb):
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _box_edge_groups(objs, color_mode):
    """Group all box edges by colour -> (xs, ys, zs) with None separators so each
    colour is a single Scatter3d line trace."""
    groups = {}
    for o in objs:
        c = lp.cuboid_corners(o["val"])
        col = _hex(lp._color_for(o, color_mode))
        xs, ys, zs = groups.setdefault(col, ([], [], []))
        for a, b in lp._EDGES + lp._FRONT_DIAGONALS:
            xs += [c[a, 0], c[b, 0], None]
            ys += [c[a, 1], c[b, 1], None]
            zs += [c[a, 2], c[b, 2], None]
    return groups


def _dashed_xyz(coords, z, dash=2.5, gap=1.8, step=0.4):
    """Walk a polygon boundary emitting dashed line points (None = gap) at height z."""
    pts = np.asarray(coords, dtype=float)
    xs, ys, zs = [], [], []
    period, dist = dash + gap, 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seglen = float(np.hypot(*(b - a)))
        if seglen == 0:
            continue
        nsteps = max(1, int(seglen / step))
        for s in range(nsteps + 1):
            d = dist + seglen * s / nsteps
            if (d % period) < dash:
                p = a + (b - a) * (s / nsteps)
                xs.append(p[0]); ys.append(p[1]); zs.append(z)
            else:
                xs.append(None); ys.append(None); zs.append(None)
        dist += seglen
    return xs, ys, zs


def orbit_camera(azimuth=45.0, elevation=35.0, zoom=1.0, pan_x=0.0, pan_y=0.0, roll=0.0,
                 base_radius=1.5):
    """A Plotly 3D `camera` dict from intuitive controls. Azimuth/elevation spin the
    eye around the scene; `zoom` moves it in (higher = closer, glitch-free true 3D
    zoom); pan_x/pan_y shift the look-at centre (normalized scene units); `roll`
    tilts the horizon by rotating the up-vector about the view axis."""
    import math
    az, el, rl = math.radians(azimuth), math.radians(elevation), math.radians(roll)
    r = base_radius / max(float(zoom), 0.05)
    dx, dy, dz = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
    center = dict(x=float(pan_x), y=float(pan_y), z=0.0)
    eye = dict(x=center["x"] + r * dx, y=center["y"] + r * dy, z=center["z"] + r * dz)
    # Roll: Rodrigues-rotate world-up (0,0,1) about the unit view direction.
    d = np.array([dx, dy, dz], dtype=float); d /= (np.linalg.norm(d) or 1.0)
    u0 = np.array([0.0, 0.0, 1.0])
    u = u0 * math.cos(rl) + np.cross(d, u0) * math.sin(rl) + d * float(d @ u0) * (1 - math.cos(rl))
    return dict(eye=eye, center=center, up=dict(x=float(u[0]), y=float(u[1]), z=float(u[2])),
                projection=dict(type="perspective"))


def build_figure(points, objs, color_mode="by_category", view="oblique",
                 height=560, line_width=4, show_labels=True, road_poly=None, bounds=None,
                 sensors=None, hdmap_lanes=None, camera=None, uirevision="lidar_oblique"):
    """Plotly 3D figure: grey points + category-coloured GT boxes, in an oblique
    z-up perspective (the same horizontal angle as the Background Filtering viewer).
    If `road_poly` is given, its boundary is drawn as a green dashed outline at
    ground level. `hdmap_lanes` (cloud-frame polylines [[x, y], ...]) draws the real
    HD-map road network at ground level — the dev-kit 'digital twin' look. `bounds`
    (xmin, xmax, ymin, ymax) locks the x/y view extent (keeps a stable zoom)."""
    import plotly.graph_objects as go
    fig = go.Figure()
    _z0 = float(np.percentile(points[:, 2], 2)) if (points is not None and len(points)) else -7.5
    if hdmap_lanes:
        hx, hy, hz = [], [], []
        for poly in hdmap_lanes:
            for p in poly:
                hx.append(p[0]); hy.append(p[1]); hz.append(_z0)
            hx.append(None); hy.append(None); hz.append(None)
        fig.add_trace(go.Scatter3d(x=hx, y=hy, z=hz, mode="lines",
                                   line=dict(color="#c2c8d2", width=1),
                                   opacity=0.55, hoverinfo="skip", showlegend=False))
    if points is not None and len(points):
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            marker=dict(size=1.2, color=_POINT_COLOR, opacity=0.5),
            hoverinfo="skip", showlegend=False))
    for col, (xs, ys, zs) in _box_edge_groups(objs, color_mode).items():
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                   line=dict(color=col, width=line_width),
                                   hoverinfo="skip", showlegend=False))
    if show_labels and objs:
        lx, ly, lz, lt, lcol = [], [], [], [], []
        for o in objs:
            top = lp.cuboid_corners(o["val"])[:4].mean(axis=0)   # top-face centre
            lx.append(top[0]); ly.append(top[1]); lz.append(top[2] + 0.4)
            lt.append(lp._label_text(o)); lcol.append(_hex(lp._color_for(o, color_mode)))
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="text", text=lt,
                                   textfont=dict(size=11, color=lcol),
                                   hoverinfo="skip", showlegend=False))
    if road_poly is not None:
        z0 = float(np.percentile(points[:, 2], 2)) if (points is not None and len(points)) else -7.5
        geoms = [road_poly] if road_poly.geom_type == "Polygon" else list(road_poly.geoms)
        for g in geoms:
            gx, gy = g.exterior.xy
            dx, dy, dz = _dashed_xyz(list(zip(gx, gy)), z0)
            fig.add_trace(go.Scatter3d(x=dx, y=dy, z=dz, mode="lines",
                                       line=dict(color="limegreen", width=5),
                                       hoverinfo="skip", showlegend=False))
    if sensors:
        import registration as reg
        zf = float(np.percentile(points[:, 2], 2)) if (points is not None and len(points)) else -7.5
        mfloor = 0.0 if any(s["pos"][2] > 2.0 for s in sensors) else zf
        for tr in reg.sensor_marker_traces(go, sensors, z_floor=mfloor):
            fig.add_trace(tr)
    xr = dict(visible=False, range=[bounds[0], bounds[1]]) if bounds else dict(visible=False)
    yr = dict(visible=False, range=[bounds[2], bounds[3]]) if bounds else dict(visible=False)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=_BG,
        scene=dict(aspectmode="data", bgcolor=_BG, camera=(camera or _OBLIQUE),
                   xaxis=xr, yaxis=yr, zaxis=dict(visible=False)),
        uirevision=uirevision)   # keep the user's rotation/zoom across frame changes
    return fig


# --------------------------------------------------------------------------- #
#  Plotly + kaleido video renderer — true 3D perspective & clipping, so it
#  matches the interactive preview EXACTLY and zooming never streaks off-screen
#  geometry across the frame (the matplotlib failure mode). Slower (kaleido spins
#  a headless browser per frame, ~seconds/frame) but accurate.
# --------------------------------------------------------------------------- #
def render_lidar_video_plotly(pcd_files, label_files, out_dir, basename, *, camera,
                              fps=10, width=960, height=560, max_points=12000,
                              color_mode="by_category", road_poly=None, hdmap_lanes=None,
                              sensors=None, bounds=None, show_labels=True, max_frames=0,
                              progress=None):
    """Render the LiDAR scan + GT 3D boxes to a video with Plotly/kaleido at the given
    `camera` (use `orbit_camera(...)`). Each frame is the SAME `build_figure` the live
    preview draws, so the clip is pixel-faithful to what you framed. MP4 (ffmpeg) or
    GIF fallback → `out_dir`. Returns (path, kind)."""
    import os
    import io
    import imageio.v2 as imageio

    os.makedirs(out_dir, exist_ok=True)
    n = len(pcd_files)
    if max_frames and max_frames > 0:
        n = min(n, max_frames)
    if n == 0:
        raise ValueError("no frames to render")

    mp4 = os.path.join(out_dir, basename + ".mp4")
    try:
        writer = imageio.get_writer(mp4, fps=fps, codec="libx264", macro_block_size=16, quality=7)
        out, kind = mp4, "mp4"
    except Exception:
        out = os.path.join(out_dir, basename + ".gif")
        writer = imageio.get_writer(out, mode="I", duration=1.0 / max(1, fps), loop=0)
        kind = "gif"

    try:
        for i in range(n):
            pts = load_points(pcd_files[i], max_points)
            objs = lp.load_objects(label_files[i]) if i < len(label_files) else []
            fig = build_figure(pts, objs, color_mode, height=height, road_poly=road_poly,
                               bounds=bounds, sensors=sensors, hdmap_lanes=hdmap_lanes,
                               camera=camera, show_labels=show_labels)
            png = fig.to_image(format="png", width=width, height=height, engine="kaleido")
            writer.append_data(np.asarray(imageio.imread(io.BytesIO(png)))[:, :, :3])
            if progress:
                progress(i + 1, n)
    finally:
        writer.close()
    return out, kind


# --------------------------------------------------------------------------- #
#  Matplotlib renderer (fast, ~ms/frame — unlike kaleido). A single frame and
#  the full video share ONE drawing path (`_draw_scene`), so the still preview a
#  user dials the angle on is *exactly* what the video saves (WYSIWYG).
# --------------------------------------------------------------------------- #
def scene_limits(pcd_files, sample_idx=None):
    """Stable (xlim, ylim, zlim) from one representative frame, so the view doesn't
    jitter — and so the still preview and the video use the SAME extent."""
    fallback = ((-60, 80), (-50, 50), (-10, 6))
    if not pcd_files:
        return fallback
    idx = (len(pcd_files) // 2) if sample_idx is None else max(0, min(sample_idx, len(pcd_files) - 1))
    p0 = load_points(pcd_files[idx], 40000)
    if not len(p0):
        return fallback
    return ((float(np.percentile(p0[:, 0], 1)) - 4, float(np.percentile(p0[:, 0], 99)) + 4),
            (float(np.percentile(p0[:, 1], 1)) - 4, float(np.percentile(p0[:, 1], 99)) + 4),
            (float(np.percentile(p0[:, 2], 1)) - 1, float(np.percentile(p0[:, 2], 99)) + 3))


def _hd_segments(hdmap_lanes, gz):
    """Flatten HD-map polylines into one list of ground-level segments (built once;
    thousands of separate ax.plot calls would dominate per-frame time)."""
    segs = []
    for poly in (hdmap_lanes or []):
        arr = np.asarray(poly, dtype=float)
        for k in range(len(arr) - 1):
            segs.append([(arr[k, 0], arr[k, 1], gz), (arr[k + 1, 0], arr[k + 1, 1], gz)])
    return segs


def _draw_scene(ax, pts, objs, hd_segs, sensors, *, color_mode, elev, azim, roll,
                focal, xlim, ylim, zlim):
    """Draw one LiDAR frame (HD map + points + GT boxes + LiDAR markers) into `ax` at
    the given perspective angle. The single source of truth for both the preview and
    the video, so they can never drift."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    ax.set_facecolor("#0e1117")
    try:
        ax.set_proj_type("persp", focal_length=float(focal))
    except TypeError:
        ax.set_proj_type("persp")

    if hd_segs:
        ax.add_collection3d(Line3DCollection(hd_segs, colors="#39d353", linewidths=0.6, alpha=0.5))
    if pts is not None and len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.25, c="#9aa6b2",
                   alpha=0.35, depthshade=False, linewidths=0)
    bsegs, bcols = [], []
    for o in (objs or []):
        c = lp.cuboid_corners(o["val"])
        col = _hex(lp._color_for(o, color_mode))
        for a, b in lp._EDGES:
            bsegs.append([(c[a, 0], c[a, 1], c[a, 2]), (c[b, 0], c[b, 1], c[b, 2])])
            bcols.append(col)
    if bsegs:
        ax.add_collection3d(Line3DCollection(bsegs, colors=bcols, linewidths=1.2))
    for s in (sensors or []):
        p = s.get("pos", [0, 0, 0])
        col = s.get("color", "#ffffff")
        if isinstance(col, (list, tuple)):
            col = _hex(col)
        ax.scatter([p[0]], [p[1]], [p[2]], c=col, marker="x", s=70, linewidths=2.2)

    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0],
                       max(zlim[1] - zlim[0], (xlim[1] - xlim[0]) / 12)))
    ax.set_axis_off()


def render_lidar_frame(pcd_file, label_file, *, elev=12.0, azim=-90.0, roll=0.0,
                       focal=0.3, max_points=12000, color_mode="by_category",
                       hdmap_lanes=None, sensors=None, xlim=None, ylim=None, zlim=None,
                       width=960, height=480):
    """Render ONE frame to an RGB image (H×W×3 uint8) via the same matplotlib path as
    `render_lidar_video` — so a still preview is exactly what the video will save.
    Pass xlim/ylim/zlim from `scene_limits(...)` so the preview matches the clip's extent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if xlim is None or ylim is None or zlim is None:
        xlim, ylim, zlim = scene_limits([pcd_file])
    hd_segs = _hd_segments(hdmap_lanes, zlim[0] + 0.3)
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#0e1117")
    try:
        ax = fig.add_subplot(111, projection="3d")
        pts = load_points(pcd_file, max_points)
        objs = lp.load_objects(label_file) if label_file else []
        _draw_scene(ax, pts, objs, hd_segs, sensors, color_mode=color_mode, elev=elev,
                    azim=azim, roll=roll, focal=focal, xlim=xlim, ylim=ylim, zlim=zlim)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    finally:
        plt.close(fig)
    return rgb


def render_lidar_video(pcd_files, label_files, out_dir, basename, *, fps=10,
                       width=960, height=480, elev=12.0, azim=-90.0, roll=0.0,
                       focal=0.3, max_points=12000, color_mode="by_category",
                       hdmap_lanes=None, sensors=None, xlim=None, ylim=None,
                       zlim=None, max_frames=0, progress=None):
    """Render the LiDAR scan + GT 3D boxes to a video with matplotlib (Agg) — fast
    enough for all frames — from a fixed perspective angle (elev/azim/roll/focal).
    Saves MP4 (ffmpeg) or GIF fallback to `out_dir`. Returns (path, kind).

    pcd_files / label_files : aligned per-frame paths (same order/length).
    hdmap_lanes : list of sensor-frame polylines (drawn green at ground).
    sensors     : list of {pos:[x,y,z], color} LiDAR markers (drawn as X's).
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    os.makedirs(out_dir, exist_ok=True)
    n = len(pcd_files)
    if max_frames and max_frames > 0:
        n = min(n, max_frames)
    if n == 0:
        raise ValueError("no frames to render")

    if xlim is None or ylim is None or zlim is None:
        _xl, _yl, _zl = scene_limits(pcd_files, sample_idx=n // 2)
        xlim = xlim or _xl; ylim = ylim or _yl; zlim = zlim or _zl
    hd_segs = _hd_segments(hdmap_lanes, zlim[0] + 0.3)

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#0e1117")
    mp4 = os.path.join(out_dir, basename + ".mp4")
    try:
        writer = imageio.get_writer(mp4, fps=fps, codec="libx264", macro_block_size=16, quality=7)
        out, kind = mp4, "mp4"
    except Exception:
        out = os.path.join(out_dir, basename + ".gif")
        writer = imageio.get_writer(out, mode="I", duration=1.0 / max(1, fps), loop=0)
        kind = "gif"

    try:
        for i in range(n):
            fig.clf()
            ax = fig.add_subplot(111, projection="3d")
            pts = load_points(pcd_files[i], max_points)
            objs = lp.load_objects(label_files[i]) if i < len(label_files) else []
            _draw_scene(ax, pts, objs, hd_segs, sensors, color_mode=color_mode, elev=elev,
                        azim=azim, roll=roll, focal=focal, xlim=xlim, ylim=ylim, zlim=zlim)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3].copy())
            if progress:
                progress(i + 1, n)
    finally:
        writer.close()
        plt.close(fig)
    return out, kind
