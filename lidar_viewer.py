"""3D LiDAR label visualization (dev-kit feature 1.2) using Plotly.

Draws a LiDAR scan + its ground-truth 3D boxes in the sensor frame — no
projection/calibration needed (points and cuboids share the LiDAR frame), and no
Open3D GUI / OpenGL (the thing that breaks cross-platform): pure Plotly, so it
runs anywhere Streamlit does. Reuses the validated cuboid corners + dev-kit
category colours from label_projection.
"""
import numpy as np

import label_projection as lp

# Camera presets matching the dev-kit's two renders.
_CAMERAS = {
    "bev": dict(eye=dict(x=0, y=0, z=2.2), up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0), projection=dict(type="orthographic")),
    "side": dict(eye=dict(x=1.5, y=-1.5, z=0.9), up=dict(x=0, y=0, z=1),
                 center=dict(x=0, y=0, z=0), projection=dict(type="perspective")),
}
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


def build_figure(points, objs, color_mode="by_category", view="side",
                 height=560, line_width=4, show_labels=True, road_poly=None, bounds=None):
    """Plotly 3D figure: grey points + category-coloured GT boxes, with a preset
    camera ('bev' = top-down, 'side' = angled). If `road_poly` is given, its
    boundary is drawn as a green dashed outline at ground level. `bounds`
    (xmin, xmax, ymin, ymax) locks the x/y view extent (keeps a stable zoom)."""
    import plotly.graph_objects as go
    fig = go.Figure()
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
    xr = dict(visible=False, range=[bounds[0], bounds[1]]) if bounds else dict(visible=False)
    yr = dict(visible=False, range=[bounds[2], bounds[3]]) if bounds else dict(visible=False)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=_BG,
        scene=dict(aspectmode="data", bgcolor=_BG, camera=_CAMERAS.get(view, _CAMERAS["side"]),
                   xaxis=xr, yaxis=yr, zaxis=dict(visible=False)),
        uirevision=view)   # keep the user's rotation/zoom across frame changes
    return fig
