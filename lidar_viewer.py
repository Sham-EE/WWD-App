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


def build_figure(points, objs, color_mode="by_category", view="oblique",
                 height=560, line_width=4, show_labels=True, road_poly=None, bounds=None,
                 sensors=None, hdmap_lanes=None):
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
        scene=dict(aspectmode="data", bgcolor=_BG, camera=_OBLIQUE,
                   xaxis=xr, yaxis=yr, zaxis=dict(visible=False)),
        uirevision="lidar_oblique")   # keep the user's rotation/zoom across frame changes
    return fig


# --------------------------------------------------------------------------- #
#  Matplotlib video renderer (fast, ~ms/frame — unlike kaleido) at a fixed
#  forward-looking angle that mirrors the south cameras (the dev-kit view).
# --------------------------------------------------------------------------- #
def render_lidar_video(pcd_files, label_files, out_dir, basename, *, fps=10,
                       width=960, height=480, elev=12.0, azim=-90.0, roll=0.0,
                       focal=0.3, max_points=12000, color_mode="by_category",
                       hdmap_lanes=None, sensors=None, xlim=None, ylim=None,
                       zlim=None, max_frames=0, progress=None):
    """Render the LiDAR scan + GT 3D boxes to a video with matplotlib (Agg) — fast
    enough for all frames — from a fixed perspective angle (elev/azim/roll/focal)
    chosen to look the way the south cameras do. Saves MP4 (ffmpeg) or GIF
    fallback to `out_dir`. Returns (path, kind).

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

    # Stable limits from a representative frame (so the view doesn't jitter).
    if xlim is None or ylim is None or zlim is None:
        p0 = load_points(pcd_files[n // 2], 40000)
        if len(p0):
            xlim = xlim or (float(np.percentile(p0[:, 0], 1)) - 4, float(np.percentile(p0[:, 0], 99)) + 4)
            ylim = ylim or (float(np.percentile(p0[:, 1], 1)) - 4, float(np.percentile(p0[:, 1], 99)) + 4)
            zlim = zlim or (float(np.percentile(p0[:, 2], 1)) - 1, float(np.percentile(p0[:, 2], 99)) + 3)
        else:
            xlim, ylim, zlim = xlim or (-60, 80), ylim or (-50, 50), zlim or (-10, 6)
    gz = zlim[0] + 0.3

    # Batch every HD-map polyline into ONE collection — thousands of separate
    # ax.plot calls would dominate the per-frame time. Static, so build once.
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    hd_segs = []
    for poly in (hdmap_lanes or []):
        arr = np.asarray(poly, dtype=float)
        for k in range(len(arr) - 1):
            hd_segs.append([(arr[k, 0], arr[k, 1], gz), (arr[k + 1, 0], arr[k + 1, 1], gz)])

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
            ax.set_facecolor("#0e1117")
            try:
                ax.set_proj_type("persp", focal_length=float(focal))
            except TypeError:
                ax.set_proj_type("persp")

            # HD-map road network (green, at ground) — under everything, one collection.
            if hd_segs:
                ax.add_collection3d(Line3DCollection(hd_segs, colors="#39d353",
                                                     linewidths=0.6, alpha=0.5))

            # Point cloud.
            pts = load_points(pcd_files[i], max_points)
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.25, c="#9aa6b2",
                           alpha=0.35, depthshade=False, linewidths=0)

            # GT 3D boxes (coloured by category, dev-kit palette) — one collection.
            objs = lp.load_objects(label_files[i]) if i < len(label_files) else []
            bsegs, bcols = [], []
            for o in objs:
                c = lp.cuboid_corners(o["val"])
                col = _hex(lp._color_for(o, color_mode))
                for a, b in lp._EDGES:
                    bsegs.append([(c[a, 0], c[a, 1], c[a, 2]), (c[b, 0], c[b, 1], c[b, 2])])
                    bcols.append(col)
            if bsegs:
                ax.add_collection3d(Line3DCollection(bsegs, colors=bcols, linewidths=1.2))

            # LiDAR station markers (X's).
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

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3].copy())
            if progress:
                progress(i + 1, n)
    finally:
        writer.close()
        plt.close(fig)
    return out, kind
