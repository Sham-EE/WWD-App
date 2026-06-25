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


def load_default_geometry(ds):
    """The factory-default geometry snapshot (config/defaults/), or the current
    saved geometry if no snapshot exists."""
    path = ds.default_site_geometry_path
    if os.path.exists(path):
        try:
            g = json.load(open(path))
            g.setdefault("research_polygon", [])
            g.setdefault("road_polygons", [])
            g.setdefault("foreground_exclusion_rects", [])
            g["research_polygon"] = [[float(p[0]), float(p[1])] for p in g["research_polygon"]]
            g["road_polygons"] = [[[float(p[0]), float(p[1])] for p in poly] for poly in g["road_polygons"]]
            g["foreground_exclusion_rects"] = [[[float(p[0]), float(p[1])] for p in r]
                                               for r in g["foreground_exclusion_rects"]]
            return g
        except Exception:
            pass
    return load_site_geometry(ds)


def has_defaults(ds):
    return os.path.exists(ds.default_site_geometry_path)


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


def apply_geometry_crop(fg, geom):
    """Split foreground Nx3 into (kept, excluded) by the CURRENT in-editor geometry:
    a point is excluded if it falls inside any exclusion rect OR outside every road
    polygon (the crop). Lets the editor show — live, before saving — what the geometry
    would remove (grey) vs keep (red)."""
    import numpy as np
    from matplotlib.path import Path
    if fg is None or not len(fg):
        empty = fg if fg is not None else None
        return empty, empty
    xy = fg[:, :2]
    excluded = np.zeros(len(xy), dtype=bool)
    roads = geom.get("road_polygons", [])
    if roads:
        in_road = np.zeros(len(xy), dtype=bool)
        for poly in roads:
            if len(poly) >= 3:
                in_road |= Path(np.asarray(poly, dtype=float)).contains_points(xy)
        excluded |= ~in_road
    for r in geom.get("foreground_exclusion_rects", []):
        if len(r) >= 3:
            excluded |= Path(np.asarray(r, dtype=float)).contains_points(xy)
    return fg[~excluded], fg[excluded]


def preview_figure(points, geom, height=640, title="", fg_points=None, gt_objs=None,
                   dragmode="pan", show_vertex_labels=False, fg_excluded_points=None,
                   color_by_height=False, height_span=4.0, off_object_points=None,
                   det_objs=None, det_scatter=None):
    """BEV: point cloud + research (cyan dotted), road (green), exclusion (magenta).
    `fg_points` (kept foreground) is drawn red; `fg_excluded_points` (foreground that
    the current geometry crops out) is drawn grey. If `gt_objs` is given, overlay GT
    box footprints + TYPE_id labels, category-coloured like the Visualizer.
    `off_object_points` (kept foreground outside every GT box) is drawn yellow — the
    same clutter/false-foreground cue as the Background-Filtering viewer, so you can
    place crop/exclusion zones over the points the filter wrongly keeps.
    `det_objs` = this-frame detections [(cx,cy,yaw,l,w,is_static), …] drawn as oriented
    boxes; `det_scatter` = (centres Nx2, is_static bool N) for the all-frames aggregate
    drawn as ✕ markers. Moving detections are green, never-moving (static, FP/pole risk)
    are purple — so you can see where the detector keeps placing phantom objects.
    `color_by_height` colours the backdrop cloud by z (Turbo) like the dev-kit."""
    import numpy as np
    import plotly.graph_objects as go
    fig = go.Figure()
    if points is not None and len(points):
        if len(points) > 40000:
            points = points[np.random.default_rng(0).choice(len(points), 40000, replace=False)]
        if color_by_height and points.shape[1] >= 3:
            z = points[:, 2]; z0 = float(np.percentile(z, 1))
            mk = dict(size=2, color=z, colorscale="Turbo", cmin=z0, cmax=z0 + float(height_span),
                      showscale=False)
        else:
            mk = dict(size=2, color="#1f77b4")
        fig.add_trace(go.Scattergl(x=points[:, 0], y=points[:, 1], mode="markers",
                                   marker=mk, hoverinfo="skip", showlegend=False))
    if fg_excluded_points is not None and len(fg_excluded_points):
        fig.add_trace(go.Scattergl(x=fg_excluded_points[:, 0], y=fg_excluded_points[:, 1], mode="markers",
                                   marker=dict(size=3, color="#888888"),
                                   name="cropped-out foreground", hoverinfo="skip"))
    if fg_points is not None and len(fg_points):
        fig.add_trace(go.Scattergl(x=fg_points[:, 0], y=fg_points[:, 1], mode="markers",
                                   marker=dict(size=3, color="red"), name="foreground (kept)",
                                   hoverinfo="skip"))
    # Off-object foreground (outside every GT box) — yellow, on top of the red, so
    # clutter the filter keeps stands out where you'd draw an exclusion rect.
    if off_object_points is not None and len(off_object_points):
        fig.add_trace(go.Scattergl(x=off_object_points[:, 0], y=off_object_points[:, 1], mode="markers",
                                   marker=dict(size=3.5, color="#ffd400"), name="off-object FG",
                                   hoverinfo="skip"))
    # Detections — what the detector reported as objects. Green = moving, purple = static
    # (never-moving over its track; FP/pole risk). Per-frame: oriented boxes; all-frames:
    # ✕ markers at each centre.
    _DET_MOV, _DET_STAT = "#16c60c", "#b14cff"
    if det_objs:
        _shown = {False: False, True: False}
        for (cx, cy, yaw, l, w, stat) in det_objs:
            c, s = np.cos(yaw), np.sin(yaw)
            loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2], [l / 2, w / 2]])
            corners = loc @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])
            fig.add_trace(go.Scatter(x=corners[:, 0], y=corners[:, 1], mode="lines",
                                     line=dict(color=_DET_STAT if stat else _DET_MOV, width=2),
                                     name=("static det" if stat else "detection"),
                                     legendgroup=("sdet" if stat else "det"),
                                     showlegend=not _shown[stat], hoverinfo="skip"))
            _shown[stat] = True
    if det_scatter is not None:
        cen, stat = det_scatter
        if len(cen):
            stat = np.asarray(stat, dtype=bool)
            for mask, col, nm in ((~stat, _DET_MOV, "detections (all frames)"),
                                  (stat, _DET_STAT, "static detections (all frames)")):
                if mask.any():
                    fig.add_trace(go.Scattergl(x=cen[mask, 0], y=cen[mask, 1], mode="markers",
                                               marker=dict(size=4, color=col, symbol="x"),
                                               name=nm, hoverinfo="skip"))
    if gt_objs:
        import label_projection as lp
        import lidar_viewer as lv
        import dataset_prep as dp
        for k, o in enumerate(gt_objs):
            fp = dp._box_footprint(o["val"])  # closed (x, y) rectangle
            col = lv._hex(lp._color_for(o, "by_category"))
            fig.add_trace(go.Scatter(x=fp[:, 0], y=fp[:, 1], mode="lines",
                                     line=dict(color=col, width=2), name="GT boxes",
                                     legendgroup="gt", showlegend=(k == 0), hoverinfo="skip"))
        lx = [float(o["val"][0]) for o in gt_objs]; ly = [float(o["val"][1]) for o in gt_objs]
        lt = [lp._label_text(o) for o in gt_objs]
        lc = [lv._hex(lp._color_for(o, "by_category")) for o in gt_objs]
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="text", text=lt,
                                 textfont=dict(size=10, color=lc), hoverinfo="skip", showlegend=False))

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
            # magenta (not red) so it stays distinct from the red foreground overlay
            fig.add_trace(go.Scatter(x=[p[0] for p in c], y=[p[1] for p in c], mode="lines",
                                     line=dict(color="#ff5fec", width=2),
                                     name="exclusion", legendgroup="excl", showlegend=(i == 0)))

    if show_vertex_labels:
        # tag each vertex so you know its polygon + index while editing:
        #   ROI1.. / R<road>.<v> / X<rect>.<v>
        for tag, color, items in (("ROI", "#17becf", [geom.get("research_polygon", [])]),
                                   ("R", "limegreen", geom.get("road_polygons", [])),
                                   ("X", "#ff5fec", geom.get("foreground_exclusion_rects", []))):
            tx, ty, tt = [], [], []
            multi = tag != "ROI"
            for pi, poly in enumerate(items):
                for vi, (x, y) in enumerate(poly):
                    tx.append(x); ty.append(y)
                    tt.append(f"{tag}{pi+1}.{vi+1}" if multi else f"{tag}{vi+1}")
            if tx:
                fig.add_trace(go.Scatter(x=tx, y=ty, mode="markers+text", text=tt,
                                         textposition="top center", marker=dict(size=5, color=color),
                                         textfont=dict(size=9, color=color),
                                         hoverinfo="skip", showlegend=False))

    allx, ally = [], []
    for poly in [geom.get("research_polygon", [])] + list(geom.get("road_polygons", [])):
        allx += [p[0] for p in poly]; ally += [p[1] for p in poly]
    if allx:
        cx, cy = (min(allx) + max(allx)) / 2, (min(ally) + max(ally)) / 2
        half = max(max(allx) - min(allx), max(ally) - min(ally)) / 2 + 8.0
    else:
        cx, cy, half = 0.0, 0.0, 60.0
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), title=title, dragmode=dragmode,
                      legend=dict(orientation="h", y=1.02, x=0),
                      xaxis=dict(title="x (m)", range=[cx - half, cx + half]),
                      yaxis=dict(title="y (m)", range=[cy - half, cy + half], scaleanchor="x", scaleratio=1),
                      uirevision="geom_edit")
    return fig
