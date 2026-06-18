"""Load / save / preview the editable site geometry (research polygon, road
polygons, foreground-exclusion rects) for the active dataset.

Everything downstream reads this through geometry_config.get_research_polygon() /
get_road_polygon() / get_fg_exclusion_rects(), which re-load the active dataset's
site_geometry.json whenever it changes — so saving here updates the WHOLE pipeline
(background filtering, cropping, scorable GT, previews, road outline) automatically.
"""
import json
import os

import dataset_manager as dm

GEOM_KEYS = ("research_polygon", "road_polygons", "foreground_exclusion_rects")


def load_site_geometry(ds):
    """Return the dataset's site geometry as plain lists; derive a starter if missing."""
    g = {}
    if os.path.exists(ds.site_geometry_path):
        try:
            g = json.load(open(ds.site_geometry_path))
        except Exception:
            g = {}
    if not g.get("research_polygon"):
        g = dm.derive_site_geometry(ds.pcd_dir)
    g.setdefault("research_polygon", [])
    g.setdefault("road_polygons", [])
    g.setdefault("foreground_exclusion_rects", [])
    g.setdefault("coarse_grid", {"NX": 5, "NY": 5})
    # normalise to lists of [x, y] floats
    g["research_polygon"] = [[float(p[0]), float(p[1])] for p in g["research_polygon"]]
    g["road_polygons"] = [[[float(p[0]), float(p[1])] for p in poly] for poly in g["road_polygons"]]
    g["foreground_exclusion_rects"] = [[[float(p[0]), float(p[1])] for p in r]
                                       for r in g["foreground_exclusion_rects"]]
    return g


def save_site_geometry(ds, geom):
    """Write geometry to the dataset's site_geometry.json (updates everything)."""
    os.makedirs(ds.config_dir, exist_ok=True)
    out = {
        "_comment": "Edited via the Geometry Editor. Coordinates in the sensor frame (metres). "
                    "Used by Background Filtering / cropping / scorable GT across the app.",
        "research_polygon": [[float(x), float(y)] for x, y in geom.get("research_polygon", [])],
        "road_polygons": [[[float(x), float(y)] for x, y in poly] for poly in geom.get("road_polygons", [])],
        "foreground_exclusion_rects": [[[float(x), float(y)] for x, y in r]
                                       for r in geom.get("foreground_exclusion_rects", [])],
        "coarse_grid": geom.get("coarse_grid", {"NX": 5, "NY": 5}),
    }
    with open(ds.site_geometry_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _rect_corners(cx0, cy0, cx1, cy1):
    return [[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1]]


def default_rect(geom):
    """A small starter exclusion rect near the centre of the research polygon."""
    rp = geom.get("research_polygon") or [[-10, -10], [10, -10], [10, 10], [-10, 10]]
    xs = [p[0] for p in rp]; ys = [p[1] for p in rp]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return _rect_corners(cx - 3, cy - 3, cx + 3, cy + 3)


def preview_figure(points, geom, height=640, title=""):
    """BEV: point cloud + research (cyan dotted), road (green), exclusion (red)."""
    import numpy as np
    import plotly.graph_objects as go
    fig = go.Figure()
    if points is not None and len(points):
        if len(points) > 40000:
            points = points[np.random.default_rng(0).choice(len(points), 40000, replace=False)]
        fig.add_trace(go.Scattergl(x=points[:, 0], y=points[:, 1], mode="markers",
                                   marker=dict(size=2, color="#1f77b4"), hoverinfo="skip", showlegend=False))

    def _closed(poly):
        return (list(poly) + [poly[0]]) if poly else []

    rp = _closed(geom.get("research_polygon", []))
    if rp:
        fig.add_trace(go.Scatter(x=[p[0] for p in rp], y=[p[1] for p in rp], mode="lines",
                                 line=dict(color="#17becf", width=2, dash="dot"), name="research (ROI)"))
    for i, poly in enumerate(geom.get("road_polygons", [])):
        c = _closed(poly)
        if c:
            fig.add_trace(go.Scatter(x=[p[0] for p in c], y=[p[1] for p in c], mode="lines",
                                     line=dict(color="limegreen", width=3),
                                     name="road", legendgroup="road", showlegend=(i == 0)))
    for i, r in enumerate(geom.get("foreground_exclusion_rects", [])):
        c = _closed(r)
        if c:
            fig.add_trace(go.Scatter(x=[p[0] for p in c], y=[p[1] for p in c], mode="lines",
                                     line=dict(color="red", width=2),
                                     name="exclusion", legendgroup="excl", showlegend=(i == 0)))

    allx, ally = [], []
    for poly in [geom.get("research_polygon", [])] + list(geom.get("road_polygons", [])):
        allx += [p[0] for p in poly]; ally += [p[1] for p in poly]
    if allx:
        cx, cy = (min(allx) + max(allx)) / 2, (min(ally) + max(ally)) / 2
        half = max(max(allx) - min(allx), max(ally) - min(ally)) / 2 + 8.0
    else:
        cx, cy, half = 0.0, 0.0, 60.0
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), title=title, dragmode="pan",
                      legend=dict(orientation="h", y=1.02, x=0),
                      xaxis=dict(title="x (m)", range=[cx - half, cx + half]),
                      yaxis=dict(title="y (m)", range=[cy - half, cy + half], scaleanchor="x", scaleratio=1),
                      uirevision="geom_edit")
    return fig
