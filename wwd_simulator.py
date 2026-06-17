"""Wrong-way driver simulator.

Real wrong-way events are rare, and the available data is all legal-direction
traffic. This module SYNTHESISES a wrong-way driver as a moving detection that
flows through the *real* WWD detector (wwd_detection.detect_wrong_way) — so the
flag it raises is produced by the actual algorithm, not hard-coded.

The driver is injected at the track level: a smooth, correctly-formed detection
(position, velocity, heading) travelling along a chosen lane in the direction
OPPOSITE to that lane's legal heading. Only genuinely-wrong-way scenarios are
offered (e.g. "North-bound in the southbound lane").
"""
import json
import os

import numpy as np

SIM_TID = 90001  # high id so it never collides with real track ids

V2X_HTML_PATH = os.path.join(os.path.dirname(__file__), "assets", "wwd_v2x_dashboard.html")

# JS injected into the V2X React app: when window.__WWD_EVENT__ is present, fire
# the app's alert pipeline once on mount with the detected speed/heading. Inserted
# right after the app's fireAlert useCallback (anchored on its dependency list).
_V2X_AUTO_FIRE = r"""
  // ===== External LiDAR/WWD trigger (injected by the Streamlit simulator) =====
  const __wwdFireRef = useRef(null);
  useEffect(function () { __wwdFireRef.current = fireAlert; });
  useEffect(function () {
    var ev = window.__WWD_EVENT__;
    if (!ev || window.__WWD_AUTO_FIRED__) return;
    window.__WWD_AUTO_FIRED__ = true;
    if (ev.speed != null) setSpeed(Number(ev.speed));
    setTimeout(function () {
      var nodes = current.laneNodes;
      var mid = (nodes && nodes.length) ? nodes[Math.floor(nodes.length / 2)] : current.center;
      if (__wwdFireRef.current) __wwdFireRef.current(mid, ev.heading != null ? Number(ev.heading) : 270);
    }, 700);
  }, []);
"""

_FIREALERT_ANCHOR = "}, [current, speed, threshold]);"


def math_heading_to_compass(heading_rad):
    """Convert a math-convention heading (0=+X/east, CCW) to a compass bearing
    (0=north, clockwise) for the V2X message."""
    return (90.0 - np.degrees(heading_rad)) % 360.0


def v2x_dashboard_html(event):
    """Load the user's V2X dashboard, inject the detection event + auto-fire hook.
    Returns the HTML string, or None if the dashboard file isn't present."""
    if not os.path.exists(V2X_HTML_PATH):
        return None
    with open(V2X_HTML_PATH, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    ev_script = "<script>window.__WWD_EVENT__ = %s;</script>" % json.dumps(event)
    if '<div id="root"></div>' in html:
        html = html.replace('<div id="root"></div>', '<div id="root"></div>\n' + ev_script, 1)
    else:
        html = ev_script + html
    if _FIREALERT_ANCHOR in html:
        html = html.replace(_FIREALERT_ANCHOR, _FIREALERT_ANCHOR + _V2X_AUTO_FIRE, 1)
    return html


def cardinal_color(heading_rad) -> str:
    """Cardinal travel-direction colour — sourced from the Detection view
    (visualization.CARDINAL_BINS) so the two pages always use the same palette."""
    if heading_rad is None:
        return "#9a9a9a"
    from visualization import cardinal_color as _vc  # lazy import (open3d is heavy)
    return _vc(heading_rad)


def cardinal_name(deg: float) -> str:
    d = (float(deg) + 180.0) % 360.0 - 180.0
    if -45 <= d < 45:
        return "East"
    if 45 <= d < 135:
        return "North"
    if d >= 135 or d < -135:
        return "West"
    return "South"


def wrong_way_options(lanes):
    """One wrong-way scenario per lane: drive opposite its legal heading."""
    opts = []
    for ln in lanes:
        legal = ln["heading_deg"]
        wrong = legal + 180.0
        opts.append({
            "lane_id": ln["lane_id"],
            "legal_name": cardinal_name(legal),
            "wrong_name": cardinal_name(wrong),
            "label": f"{cardinal_name(wrong)}-bound in the “{ln['lane_id']}” lane "
                     f"(legal direction: {cardinal_name(legal)})",
            "lane": ln,
            "wrong_deg": wrong,
        })
    return opts


def make_wrong_way_track(lane, fps=10.0, speed=8.0, start_frac=0.0, lateral_frac=0.5,
                         max_frames=400):
    """Synthesise the per-step detections of a wrong-way driver crossing `lane`.

    The driver travels along the lane's long axis in the OPPOSITE direction to
    the lane's legal heading, at `speed` m/s. start_frac positions the entry
    point along the lane (0 = far upstream end); lateral_frac (0..1) places it
    across the lane width. Returns a list of detection dicts (one per step)."""
    poly = lane["polygon"]
    wrong = np.radians(lane["heading_deg"] + 180.0)
    u = np.array([np.cos(wrong), np.sin(wrong)])          # travel direction
    vperp = np.array([-u[1], u[0]])                        # across the lane

    xs, ys = poly.exterior.xy
    coords = np.column_stack([np.asarray(xs), np.asarray(ys)])
    C = np.array([poly.centroid.x, poly.centroid.y])
    proj = coords @ u
    pmin, pmax = float(proj.min()), float(proj.max())
    pperp = coords @ vperp
    perp_min, perp_max = float(pperp.min()), float(pperp.max())

    start_along = pmin + float(start_frac) * (pmax - pmin)
    lat = perp_min + float(lateral_frac) * (perp_max - perp_min)
    start = C + (start_along - C @ u) * u + (lat - C @ vperp) * vperp

    dt = 1.0 / float(fps) if fps else 0.1
    step = float(speed) * dt
    dets = []
    pos = start.astype(float).copy()
    f = 0
    while f < max_frames:
        if (pos @ u) > pmax + 1.0:          # has crossed the whole lane
            break
        dets.append(dict(
            tid=SIM_TID, cls="Car", cx=float(pos[0]), cy=float(pos[1]),
            yaw=float(wrong), heading=float(wrong),
            vx=float(u[0] * speed), vy=float(u[1] * speed), speed=float(speed),
            l=4.5, w=1.9, length=4.5, width=1.9, moving=True, hit=True,
            score=300.0, is_vehicle=True, simulated=True))
        pos = pos + u * step
        f += 1
    return dets


def _box_corners(cx, cy, yaw, l, w):
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = l / 2.0, w / 2.0
    loc = [(dx, dy), (dx, -dy), (-dx, -dy), (-dx, dy), (dx, dy)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in loc]


def simulator_figure(lanes, sim_track, frame_idx, flagged, base_dets=None,
                     x_range=None, y_range=None, height=620):
    """Top-down view: lane boxes + legal-direction arrows, the wrong-way driver's
    path and current position (red when flagged), and optional real traffic."""
    import plotly.graph_objects as go
    fig = go.Figure()

    # lanes: outline coloured by the lane's LEGAL travel direction (so the legend
    # is just the lanes, each in its direction colour) + a matching legal arrow.
    for ln in lanes:
        lcol = cardinal_color(np.radians(ln["heading_deg"]))
        xs, ys = ln["polygon"].exterior.xy
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines",
                                 line=dict(color=lcol, width=2),
                                 name=f"{ln['lane_id']} (legal {cardinal_name(ln['heading_deg'])})",
                                 hoverinfo="skip"))
        cx, cy = ln["polygon"].centroid.x, ln["polygon"].centroid.y
        hd = np.radians(ln["heading_deg"])
        L = 8.0
        fig.add_annotation(x=cx + L * np.cos(hd), y=cy + L * np.sin(hd), ax=cx, ay=cy,
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowsize=1.4, arrowwidth=2, arrowcolor=lcol)

    # real traffic backdrop (optional), spheres coloured by cardinal travel direction
    if base_dets:
        cols = [cardinal_color(d.get("heading")) for d in base_dets]
        fig.add_trace(go.Scatter(x=[d["cx"] for d in base_dets], y=[d["cy"] for d in base_dets],
                                 mode="markers", marker=dict(size=7, color=cols,
                                 line=dict(color="#000", width=0.5)),
                                 name="real traffic", showlegend=False, hoverinfo="skip"))

    # driver full path (faint) + travelled path + box, coloured by cardinal direction
    if sim_track:
        px = [d["cx"] for d in sim_track]
        py = [d["cy"] for d in sim_track]
        k = min(frame_idx, len(sim_track) - 1)
        d = sim_track[k]
        ccol = cardinal_color(d["heading"])   # the driver's (wrong-way) direction colour
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color="#ddd", width=1, dash="dot"),
                                 name="planned path", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=px[:k + 1], y=py[:k + 1], mode="lines",
                                 line=dict(color=ccol, width=2),
                                 name="driver path", showlegend=False, hoverinfo="skip"))
        corners = _box_corners(d["cx"], d["cy"], d["yaw"], d["l"], d["w"])
        # cardinal fill; red outline (thicker) once flagged wrong-way
        fig.add_trace(go.Scatter(x=[c[0] for c in corners], y=[c[1] for c in corners],
                                 mode="lines", fill="toself", fillcolor=ccol, opacity=0.85,
                                 line=dict(color="#ff2b2b" if flagged else "#000",
                                           width=4 if flagged else 2),
                                 name="⚠ wrong-way driver" if flagged else "driver",
                                 showlegend=False, hoverinfo="skip"))
        hd = d["heading"]; L = 6.0
        fig.add_annotation(x=d["cx"] + L * np.cos(hd), y=d["cy"] + L * np.sin(hd),
                           ax=d["cx"], ay=d["cy"], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=3, arrowcolor=ccol)

    if x_range:
        fig.update_xaxes(range=list(x_range))
    if y_range:
        fig.update_yaxes(range=list(y_range))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="Y (m)")
    fig.update_xaxes(title="X (m)")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0),
                      plot_bgcolor="#111", dragmode="pan", uirevision="sim",
                      title="Wrong-way driver simulation (top-down)",
                      legend=dict(orientation="h", y=-0.06))
    return fig


def build_sim_det_frames(sim_dets, start_frame=0, base_det_frames=None, total_frames=None):
    """Place the synthetic driver into a per-frame detection list, optionally on
    top of real traffic (base_det_frames). Returns a fresh list (originals are
    not mutated)."""
    if base_det_frames:
        frames = [list(f) for f in base_det_frames]
    else:
        n = total_frames if total_frames else (start_frame + len(sim_dets) + 5)
        frames = [[] for _ in range(n)]
    for k, d in enumerate(sim_dets):
        fi = start_frame + k
        if 0 <= fi < len(frames):
            frames[fi].append(dict(d))
    return frames
