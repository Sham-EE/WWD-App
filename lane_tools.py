"""Helpers for the Lane Editor page.

Turns a tracks.csv (produced by detection/tracking) into an editable set of lane
boxes: auto-clusters the vehicle motion into N travel directions, builds an
axis-aligned box + mean heading per direction, and round-trips to the GeoJSON
format consumed by wwd_detection / geometry_config.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans


def load_tracks(source) -> pd.DataFrame:
    """Read a tracks.csv (path or uploaded buffer). Ensures a 'heading' column:
    if absent/NaN it is recomputed from vx, vy."""
    df = pd.read_csv(source)
    if 'heading' not in df.columns:
        df['heading'] = np.nan
    need = df['heading'].isna()
    if need.any() and {'vx', 'vy'}.issubset(df.columns):
        df.loc[need, 'heading'] = np.arctan2(
            pd.to_numeric(df.loc[need, 'vy'], errors='coerce'),
            pd.to_numeric(df.loc[need, 'vx'], errors='coerce'),
        )
    return df


def moving_points(df: pd.DataFrame, min_speed: float = 1.0) -> np.ndarray:
    """(N, 3) array of [cx, cy, heading_rad] for moving rows with valid heading."""
    d = df
    if 'speed' in d.columns:
        d = d[pd.to_numeric(d['speed'], errors='coerce') >= min_speed]
    d = d[d['heading'].notna()]
    if len(d) == 0:
        return np.zeros((0, 3))
    return d[['cx', 'cy', 'heading']].to_numpy(dtype=float)


def auto_lanes(points: np.ndarray, k: int = 4, buffer_m: float = 2.0):
    """Cluster motion into k travel directions; one axis-aligned lane box per
    cluster (bounding box of its points + buffer, circular-mean heading)."""
    lanes = []
    if len(points) == 0:
        return lanes
    k = max(1, min(k, len(points)))
    H = points[:, 2]
    feats = np.column_stack([np.cos(H), np.sin(H)])  # cluster on direction only
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit(feats).labels_
    for c in range(k):
        pts = points[labels == c]
        if len(pts) == 0:
            continue
        # Robust circular mean of the travel heading: start from the plain mean,
        # then iteratively drop points whose heading is far from it. TURNING
        # vehicles sweep through headings and would otherwise pull the lane's
        # direction diagonal — this keeps it on the straight-through flow.
        Hc = pts[:, 2]
        mh = np.arctan2(np.sin(Hc).mean(), np.cos(Hc).mean())
        for _ in range(4):
            dev = np.abs((np.degrees(Hc - mh) + 180.0) % 360.0 - 180.0)
            keep = dev <= 30.0
            if keep.sum() >= max(5, int(0.25 * len(Hc))):
                mh = np.arctan2(np.sin(Hc[keep]).mean(), np.cos(Hc[keep]).mean())
        lanes.append(dict(
            lane_id=f"lane_{c+1}",
            xmin=float(pts[:, 0].min() - buffer_m), xmax=float(pts[:, 0].max() + buffer_m),
            ymin=float(pts[:, 1].min() - buffer_m), ymax=float(pts[:, 1].max() + buffer_m),
            heading_deg=float(np.degrees(mh)), n=int(len(pts)),
        ))
    lanes.sort(key=lambda l: -l['n'])
    return lanes


def snap_to_cardinal(deg: float) -> float:
    """Nearest cardinal heading to `deg` (E=0°, N=90°, W=180°, S=−90°), in degrees —
    so the lane arrow points straight along an axis, ignoring vehicle scatter."""
    cards = [0.0, 90.0, 180.0, -90.0]
    return float(min(cards, key=lambda c: abs((float(deg) - c + 180.0) % 360.0 - 180.0)))


# A lane is just a travel direction: pick one and the heading + color follow.
LANE_DIRECTIONS = ["Eastbound", "Westbound", "Northbound", "Southbound"]
_DIR_LETTER = {"Eastbound": "E", "Westbound": "W", "Northbound": "N", "Southbound": "S"}
_LETTER_DIR = {"E": "Eastbound", "W": "Westbound", "N": "Northbound", "S": "Southbound"}


def cardinal_heading(letter: str, true_north_deg=None) -> float:
    """Sensor-frame math heading (deg) whose arrow points along the given TRUE
    compass cardinal (N/E/S/W). With a georeference (`true_north_deg` set) this is
    the real-world direction; without one it falls back to the sensor axes
    (E=0, N=90, W=180, S=−90). Kept consistent with true_cardinal_buckets (E/W
    swap already baked in via +90 for East, −90 for West)."""
    if true_north_deg is None:
        return {"E": 0.0, "N": 90.0, "W": 180.0, "S": -90.0}[letter]
    off = {"N": 0.0, "E": 90.0, "S": 180.0, "W": -90.0}[letter]
    return float(((float(true_north_deg) + off + 180.0) % 360.0) - 180.0)


def direction_to_heading(direction: str, true_north_deg=None) -> float:
    """Heading (deg) for a lane travel-direction name (Eastbound/Westbound/…)."""
    return cardinal_heading(_DIR_LETTER[direction], true_north_deg)


def heading_to_direction(heading_deg: float, true_north_deg=None) -> str:
    """Which travel-direction name a heading currently reads as (real compass when
    `true_north_deg` is given, else sensor-frame)."""
    bucket_fn, _ = _cardinal_scheme(true_north_deg)
    letter = str(bucket_fn([heading_deg])[0])[0]
    return _LETTER_DIR.get(letter, "Eastbound")


def lane_ring(lane):
    """Closed [[x,y], ...] ring for a lane — its drawn ``polygon`` if present, else
    the axis-aligned box from xmin/xmax/ymin/ymax."""
    if lane.get('polygon'):
        ring = [[float(x), float(y)] for x, y in lane['polygon']]
        if len(ring) >= 1 and ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        return ring
    xmin, xmax = sorted((float(lane['xmin']), float(lane['xmax'])))
    ymin, ymax = sorted((float(lane['ymin']), float(lane['ymax'])))
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]


def _is_axis_box(verts) -> bool:
    """True if `verts` (open ring) is a 4-corner axis-aligned rectangle."""
    if len(verts) != 4:
        return False
    xs = sorted(set(round(float(x), 3) for x, _ in verts))
    ys = sorted(set(round(float(y), 3) for _, y in verts))
    return len(xs) == 2 and len(ys) == 2


def lanes_to_geojson(lanes) -> dict:
    feats = []
    for l in lanes:
        feats.append({
            "type": "Feature",
            "properties": {"lane_id": str(l['lane_id']),
                           "heading_deg": round(float(l['heading_deg']), 1),
                           "calibrated": True},
            "geometry": {"type": "Polygon", "coordinates": [lane_ring(l)]},
        })
    return {"type": "FeatureCollection",
            "_comment": "Generated by the Lane Editor page.",
            "features": feats}


def geojson_to_lanes(gj: dict):
    """Inverse of lanes_to_geojson. A 4-corner axis-aligned rectangle loads as a box
    lane (xmin/xmax/ymin/ymax); any other shape keeps its vertices as a ``polygon``."""
    lanes = []
    for f in gj.get('features', []):
        p = f.get('properties', {})
        ring = f['geometry']['coordinates'][0]
        verts = ring[:-1] if (len(ring) > 1 and ring[0] == ring[-1]) else ring
        xs = [c[0] for c in verts]
        ys = [c[1] for c in verts]
        lane = dict(lane_id=p.get('lane_id', f"lane_{len(lanes)+1}"),
                    xmin=min(xs), xmax=max(xs), ymin=min(ys), ymax=max(ys),
                    heading_deg=float(p.get('heading_deg', 0.0)), n=0)
        if not _is_axis_box(verts):
            lane['polygon'] = [[float(x), float(y)] for x, y in verts]
        lanes.append(lane)
    return lanes


_PALETTE = ['#1f77b4', '#9467bd', '#17becf', '#e377c2', '#bcbd22',
            '#7f7f7f', '#ff7f0e', '#2ca02c']

# Cardinal buckets in the sensor frame (atan2(vy,vx) degrees). Named by axis AND
# the lane convention so the legend is unambiguous (note: +Y is the sensor's
# axis, only "north" by convention — not verified geographic north).
_CARDINAL = [
    ("E  (+X, ~0°)",   '#d62728'),
    ("N  (+Y, ~90°)",  '#2ca02c'),
    ("W  (-X, ~180°)", '#1f77b4'),
    ("S  (-Y, ~-90°)", '#ff7f0e'),
]


def _cardinal_bucket(deg: np.ndarray) -> np.ndarray:
    d = (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0  # -> [-180, 180)
    lab = np.empty(d.shape, dtype=object)
    lab[(d >= -45) & (d < 45)] = _CARDINAL[0][0]
    lab[(d >= 45) & (d < 135)] = _CARDINAL[1][0]
    lab[(d >= 135) | (d < -135)] = _CARDINAL[2][0]
    lab[(d >= -135) & (d < -45)] = _CARDINAL[3][0]
    return lab


def load_pcd_background(pcd_dir: str, n_frames: int = 15, grid: float = 0.5,
                        max_points: int = 60000) -> np.ndarray:
    """Accumulate an (X, Y, Z) road footprint from several PCD frames,
    grid-downsampled in XY, for a faint backdrop in the editor preview."""
    import glob
    import os
    import open3d as o3d
    files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    if not files:
        return np.zeros((0, 3))
    idxs = np.unique(np.linspace(0, len(files) - 1, min(n_frames, len(files))).astype(int))
    acc = []
    for i in idxs:
        pts = np.asarray(o3d.io.read_point_cloud(files[int(i)]).points)
        if pts.size:
            acc.append(pts[:, :3])
    if not acc:
        return np.zeros((0, 3))
    xyz = np.vstack(acc)
    _, idx = np.unique(np.floor(xyz[:, :2] / grid).astype(np.int64), axis=0, return_index=True)
    xyz = xyz[idx]
    if len(xyz) > max_points:
        xyz = xyz[np.random.choice(len(xyz), max_points, replace=False)]
    return xyz


def load_track_paths(df, min_frames: int = 4):
    """Per-object trajectories from a tracks DataFrame (needs tid, cx, cy; frame to
    order). Each item: {tid, xy:(N,2), bearing (deg, start->end), disp (m),
    straight (0..1)}. Restricts to vehicles when an is_vehicle column is present.

    The net start->end bearing is used (not the per-frame heading column, which is
    unreliable on this dataset); `straight` = displacement / path-length flags
    turners/loiterers (≈1 straight-through, low = curved/parked)."""
    out = []
    if df is None or not {'tid', 'cx', 'cy'}.issubset(df.columns):
        return out
    d0 = df
    if 'is_vehicle' in d0.columns:
        iv = d0['is_vehicle']
        d0 = d0[(iv == True) | iv.astype(str).str.lower().isin(['true', '1', '1.0'])]  # noqa: E712
    for tid, d in d0.groupby('tid'):
        if 'frame' in d.columns:
            d = d.sort_values('frame')
        xy = d[['cx', 'cy']].to_numpy(dtype=float)
        if len(xy) < min_frames:
            continue
        dvec = xy[-1] - xy[0]
        disp = float(np.hypot(dvec[0], dvec[1]))
        seg = np.diff(xy, axis=0)
        path_len = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        out.append(dict(
            tid=tid, xy=xy, disp=disp,
            bearing=float(np.degrees(np.arctan2(dvec[1], dvec[0]))),
            straight=(disp / path_len if path_len > 1e-6 else 0.0)))
    return out


def _circular_mean_deg(degs) -> float:
    a = np.radians(np.asarray(degs, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def heading_from_tracks_in_lane(lane, paths, min_disp: float = 3.0,
                                min_straight: float = 0.7, min_frac_inside: float = 0.5,
                                tol_deg: float = 40.0):
    """Robust travel heading (deg, sensor frame) for a lane from the through-traffic
    whose path lies inside its polygon. Takes each qualifying car's net start->end
    bearing and returns the dominant direction (trimmed circular mean, so the opposite
    / cross flow is rejected). Returns (heading, n_used); (None, 0) when nothing
    qualifies — e.g. a lane only turned into/out of has no straight-through sample."""
    from matplotlib.path import Path
    poly = Path(np.asarray(lane_ring(lane), dtype=float))
    bearings = [p['bearing'] for p in paths
                if p['disp'] >= min_disp and p['straight'] >= min_straight
                and poly.contains_points(p['xy']).mean() >= min_frac_inside]
    if not bearings:
        return None, 0
    b = np.asarray(bearings, dtype=float)
    # Seed from the DENSEST bearing (the car with the most like-directed neighbours), so
    # a lane carrying both a correct and an opposing/cross flow locks onto the majority
    # direction instead of averaging to something in between.
    within = lambda ref: np.abs((b - ref + 180.0) % 360.0 - 180.0) <= tol_deg
    seed = b[int(np.argmax([within(x).sum() for x in b]))]
    m = _circular_mean_deg(b[within(seed)])
    for _ in range(2):  # settle the mean on that cluster
        if within(m).any():
            m = _circular_mean_deg(b[within(m)])
    return float(m), int(within(m).sum())


def assign_points_to_lanes(points: np.ndarray, lanes) -> np.ndarray:
    """First-match lane index for each point (point-in-polygon, handles boxes AND
    drawn polygons); -1 if the point is outside every lane."""
    n = len(points)
    a = np.full(n, -1, dtype=int)
    if n == 0:
        return a
    from matplotlib.path import Path
    xy = np.asarray(points[:, :2])
    for i, l in enumerate(lanes):
        inside = Path(np.asarray(lane_ring(l), dtype=float)).contains_points(xy)
        m = (a < 0) & inside
        a[m] = i
    return a


# TRUE geographic cardinals (used when a georeference is available). `north_deg` is
# the sensor-frame math heading (CCW from +X) that points at true North.
_TRUE_CARDINAL = [("N", '#2ca02c'), ("E", '#d62728'), ("S", '#ff7f0e'), ("W", '#1f77b4')]


def true_cardinal_buckets(heading_deg, north_deg):
    """Label sensor-frame math headings (deg) by TRUE compass cardinal N/E/S/W.

    The georeference's east/west is mirrored relative to reality (verified: the
    first-frame truck heads sensor +X, which the georef calls West, but on the map
    it drives East toward the Jägerhof). So compass = heading − north (a reflection
    across the N–S axis: N/S unchanged, E↔W swapped)."""
    comp = (np.asarray(heading_deg, dtype=float) - float(north_deg)) % 360.0
    lab = np.empty(comp.shape, dtype=object)
    lab[(comp < 45) | (comp >= 315)] = 'N'
    lab[(comp >= 45) & (comp < 135)] = 'E'
    lab[(comp >= 135) & (comp < 225)] = 'S'
    lab[(comp >= 225) & (comp < 315)] = 'W'
    return lab


def _cardinal_scheme(true_north_deg):
    """(bucket_fn(heading_deg_array) -> labels, palette[(label,color)]) for the active
    cardinal convention: TRUE compass when `true_north_deg` is set, else sensor-frame."""
    if true_north_deg is None:
        return (lambda h: _cardinal_bucket(np.asarray(h))), _CARDINAL
    return (lambda h: true_cardinal_buckets(h, true_north_deg)), _TRUE_CARDINAL


def _compass_dirs(north_deg):
    """Math angle (CCW from +X, deg) of each true-cardinal arrow tip on screen.
    E/W swapped vs a naive rose because the georef's east/west is mirrored (see
    true_cardinal_buckets) — keeps N/S, puts East where the scene's east really is."""
    return [('N', north_deg, True), ('E', north_deg + 90.0, False),
            ('S', north_deg + 180.0, False), ('W', north_deg - 90.0, False)]


def lane_display_colors(lanes, color_mode: str, true_north_deg=None):
    """One display color per lane, chosen so boxes match the vehicle dots:
    'lane' -> per-lane palette; otherwise -> the lane heading's cardinal color
    (TRUE compass cardinal when `true_north_deg` is given, else sensor-frame)."""
    bucket_fn, palette = _cardinal_scheme(true_north_deg)
    card = dict(palette)
    cols = []
    for i, l in enumerate(lanes):
        if color_mode == 'lane':
            cols.append(_PALETTE[i % len(_PALETTE)])
        else:
            lab = bucket_fn([l['heading_deg']])[0]
            cols.append(card.get(lab, '#888888'))
    return cols


def build_preview(points: np.ndarray, lanes, color_mode: str = 'cardinal',
                  bg_xyz: np.ndarray = None, top_down: bool = True,
                  hdmap_lanes=None, true_north_deg=None) -> go.Figure:
    """3D bird's-eye preview (same mouse UX as the detection viewer): optional
    point-cloud backdrop + vehicle positions + lane boxes with direction arrows.
    Box colors always match the dot colors. ``top_down`` toggles the camera.

    ``hdmap_lanes`` (list of sensor-frame polylines [[x, y], ...]) draws the real
    intersection's HD-map road network underneath, so lane boxes can be aligned to
    the actual roads. ``true_north_deg`` (sensor math-heading of true North) switches
    the cardinal colouring to TRUE compass directions and draws a compass rose.

    color_mode: 'cardinal' (dots by direction bucket), 'lane' (dots by which box
    they fall in; gray if outside all), or 'heading' (continuous HSV)."""
    Z_DOT, Z_BOX = -6.0, -7.2
    fig = go.Figure()
    bucket_fn, palette = _cardinal_scheme(true_north_deg)

    # Real intersection roads (HD map) — one batched trace, under everything.
    if hdmap_lanes:
        hx, hy, hz = [], [], []
        for poly in hdmap_lanes:
            for px, py in poly:
                hx.append(float(px)); hy.append(float(py)); hz.append(Z_BOX)
            hx.append(None); hy.append(None); hz.append(None)
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz, mode='lines', line=dict(color='#5b6573', width=1),
            opacity=0.65, name='intersection (HD map)', hoverinfo='skip'))

    if bg_xyz is not None and len(bg_xyz):
        bg_xyz = np.asarray(bg_xyz)
        # Tolerate a 2-column (X,Y) background (e.g. an older cached result) by
        # placing it on the box plane instead of crashing on a missing Z column.
        bz = bg_xyz[:, 2] if bg_xyz.shape[1] >= 3 else np.full(len(bg_xyz), Z_BOX)
        fig.add_trace(go.Scatter3d(
            x=bg_xyz[:, 0], y=bg_xyz[:, 1], z=bz, mode='markers',
            marker=dict(size=1.5, color='#9a9a9a', opacity=0.35),
            name='point cloud', hoverinfo='skip'))

    lane_cols = lane_display_colors(lanes, color_mode, true_north_deg)

    if len(points):
        zc = np.full(len(points), Z_DOT)
        if color_mode == 'cardinal':
            buckets = bucket_fn(np.degrees(points[:, 2]))
            for label, col in palette:
                m = buckets == label
                if m.any():
                    fig.add_trace(go.Scatter3d(
                        x=points[m, 0], y=points[m, 1], z=zc[m], mode='markers',
                        marker=dict(size=3, color=col), name=label, hoverinfo='skip'))
        elif color_mode == 'lane':
            assign = assign_points_to_lanes(points, lanes)
            m = assign < 0
            if m.any():
                fig.add_trace(go.Scatter3d(
                    x=points[m, 0], y=points[m, 1], z=zc[m], mode='markers',
                    marker=dict(size=3, color='#cccccc'),
                    name='outside any lane', hoverinfo='skip'))
            for i, l in enumerate(lanes):
                m = assign == i
                if m.any():
                    fig.add_trace(go.Scatter3d(
                        x=points[m, 0], y=points[m, 1], z=zc[m], mode='markers',
                        marker=dict(size=3, color=lane_cols[i]),
                        name=str(l['lane_id']), hoverinfo='skip'))
        else:
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=zc, mode='markers',
                marker=dict(size=3, color=np.degrees(points[:, 2]), colorscale='HSV',
                            cmin=-180, cmax=180, colorbar=dict(title='heading°')),
                name='vehicle positions', hoverinfo='skip'))

    for i, l in enumerate(lanes):
        col = lane_cols[i]
        ring = lane_ring(l)
        rx = [p[0] for p in ring]; ry = [p[1] for p in ring]
        fig.add_trace(go.Scatter3d(
            x=rx, y=ry, z=[Z_BOX] * len(ring), mode='lines', line=dict(color=col, width=5),
            name=str(l['lane_id'])))
        vx = np.asarray(rx[:-1]); vy = np.asarray(ry[:-1])
        cx, cy = float(vx.mean()), float(vy.mean())
        hd = np.radians(float(l['heading_deg']))
        L = 0.35 * min(vx.max() - vx.min(), vy.max() - vy.min()) + 3.0
        tx, ty = cx + L * np.cos(hd), cy + L * np.sin(hd)
        fig.add_trace(go.Scatter3d(x=[cx, tx], y=[cy, ty], z=[Z_BOX, Z_BOX],
                                   mode='lines', line=dict(color=col, width=6),
                                   showlegend=False, hoverinfo='skip'))
        wl, wa = 2.4, np.radians(28)
        fig.add_trace(go.Scatter3d(
            x=[tx - wl * np.cos(hd - wa), tx, tx - wl * np.cos(hd + wa)],
            y=[ty - wl * np.sin(hd - wa), ty, ty - wl * np.sin(hd + wa)],
            z=[Z_BOX] * 3, mode='lines', line=dict(color=col, width=6),
            showlegend=False, hoverinfo='skip'))

    if top_down:
        cam = {'up': {'x': 0, 'y': 1, 'z': 0}, 'center': {'x': 0, 'y': 0, 'z': 0},
               'eye': {'x': 0, 'y': 0, 'z': 2.6}}
    else:
        cam = {'up': {'x': 0, 'y': 0, 'z': 1}, 'center': {'x': 0, 'y': 0, 'z': 0},
               'eye': {'x': 1.55, 'y': 1.55, 'z': 1.55}}
    # uirevision keeps the user's zoom/rotation across Streamlit reruns (e.g.
    # while editing box numbers); it only resets when the camera mode changes.
    uirev = f"td_{top_down}"
    # Default to a zoomed-OUT view (the whole intersection / research region) so it
    # doesn't sit tight on just the lanes when the point-cloud backdrop is off.
    try:
        from geometry_config import get_research_polygon
        bx0, by0, bx1, by1 = get_research_polygon().bounds
        m = 5.0
        xrange, yrange = [bx0 - m, bx1 + m], [by0 - m, by1 + m]
    except Exception:
        xrange = yrange = None

    # True-North compass rose in the bottom-right corner (data coords @ box plane).
    if true_north_deg is not None and xrange and yrange:
        cxc = xrange[1] - 0.10 * (xrange[1] - xrange[0])
        cyc = yrange[0] + 0.12 * (yrange[1] - yrange[0])
        r = 0.06 * min(xrange[1] - xrange[0], yrange[1] - yrange[0])
        for lab, ang, is_n in _compass_dirs(true_north_deg):
            a = np.radians(ang)
            ex, ey = cxc + r * np.cos(a), cyc + r * np.sin(a)
            c = '#ffffff' if is_n else '#9aa3b2'
            fig.add_trace(go.Scatter3d(x=[cxc, ex], y=[cyc, ey], z=[Z_BOX, Z_BOX], mode='lines',
                                       line=dict(color=c, width=5 if is_n else 2),
                                       showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter3d(x=[cxc + 1.25 * r * np.cos(a)], y=[cyc + 1.25 * r * np.sin(a)],
                                       z=[Z_BOX], mode='text', text=[lab],
                                       textfont=dict(color=c, size=13), showlegend=False, hoverinfo='skip'))

    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=30, b=0),
        uirevision=uirev,
        scene=dict(xaxis=dict(title='X (m)', range=xrange), yaxis=dict(title='Y (m)', range=yrange),
                   zaxis_title='Z (m)', aspectmode='data', camera=cam, uirevision=uirev),
        legend=dict(orientation='h', y=-0.02),
        title='Top-down (drag to rotate, scroll to zoom, right-drag to pan)' if top_down else '3D view')
    return fig


def simplify_path(xs, ys, max_pts: int = 24, min_pts: int = 3):
    """Downsample a lasso path to <= max_pts evenly-spaced vertices → list of [x,y].
    Returns [] if the path is too short to be a polygon."""
    xs = list(xs); ys = list(ys)
    n = min(len(xs), len(ys))
    if n < min_pts:
        return []
    idx = range(n) if n <= max_pts else np.unique(np.linspace(0, n - 1, max_pts).astype(int))
    return [[float(xs[i]), float(ys[i])] for i in idx]


def build_draw_figure(points: np.ndarray, lanes, hdmap_lanes=None,
                      color_mode: str = 'cardinal', xrange=None, yrange=None,
                      true_north_deg=None) -> go.Figure:
    """Flat top-down 2D figure for DRAWING lane polygons: vehicle dots + HD-map roads
    + existing lane rings, with lasso/box select enabled (the drawn path becomes a new
    lane polygon). Equal aspect so shapes aren't distorted. ``true_north_deg`` switches
    the cardinal colouring to TRUE compass directions and draws a compass rose."""
    fig = go.Figure()
    bucket_fn, palette = _cardinal_scheme(true_north_deg)
    if hdmap_lanes:
        hx, hy = [], []
        for poly in hdmap_lanes:
            for px, py in poly:
                hx.append(float(px)); hy.append(float(py))
            hx.append(None); hy.append(None)
        fig.add_trace(go.Scattergl(x=hx, y=hy, mode='lines', line=dict(color='#5b6573', width=1),
                                   opacity=0.6, name='intersection (HD map)', hoverinfo='skip'))
    if points is not None and len(points):
        if color_mode == 'cardinal':
            buckets = bucket_fn(np.degrees(points[:, 2]))
            for label, col in palette:
                m = buckets == label
                if m.any():
                    fig.add_trace(go.Scattergl(x=points[m, 0], y=points[m, 1], mode='markers',
                                               marker=dict(size=4, color=col), name=label, hoverinfo='skip'))
        else:
            fig.add_trace(go.Scattergl(x=points[:, 0], y=points[:, 1], mode='markers',
                                       marker=dict(size=4, color='#8a93a5'), name='vehicles', hoverinfo='skip'))
    lane_cols = lane_display_colors(lanes, color_mode, true_north_deg)
    for i, l in enumerate(lanes):
        ring = lane_ring(l)
        fig.add_trace(go.Scattergl(x=[p[0] for p in ring], y=[p[1] for p in ring], mode='lines',
                                   line=dict(color=lane_cols[i], width=3),
                                   name=str(l['lane_id']), hoverinfo='skip'))

    # True-North compass rose, bottom-right (data coords; equal aspect keeps angles true).
    if true_north_deg is not None and xrange and yrange:
        cxc = xrange[1] - 0.10 * (xrange[1] - xrange[0])
        cyc = yrange[0] + 0.12 * (yrange[1] - yrange[0])
        r = 0.06 * min(xrange[1] - xrange[0], yrange[1] - yrange[0])
        for lab, ang, is_n in _compass_dirs(true_north_deg):
            a = np.radians(ang)
            ex, ey = cxc + r * np.cos(a), cyc + r * np.sin(a)
            c = '#ffffff' if is_n else '#9aa3b2'
            fig.add_trace(go.Scattergl(x=[cxc, ex], y=[cyc, ey], mode='lines',
                                       line=dict(color=c, width=4 if is_n else 1.5),
                                       hoverinfo='skip', showlegend=False))
            fig.add_trace(go.Scattergl(x=[cxc + 1.28 * r * np.cos(a)], y=[cyc + 1.28 * r * np.sin(a)],
                                       mode='text', text=[lab], textfont=dict(color=c, size=13),
                                       hoverinfo='skip', showlegend=False))

    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=30, b=0), dragmode='lasso',
        xaxis=dict(title='X (m)', range=xrange),
        yaxis=dict(title='Y (m)', range=yrange, scaleanchor='x', scaleratio=1),
        legend=dict(orientation='h', y=-0.05), uirevision='lane_draw',
        title='✏️ Lasso (or Box-Select) a shape to make a lane polygon')
    return fig
