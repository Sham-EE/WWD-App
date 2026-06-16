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
import numpy as np

SIM_TID = 90001  # high id so it never collides with real track ids


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

    # lanes: outline (grey) + legal-direction arrow (green)
    for ln in lanes:
        xs, ys = ln["polygon"].exterior.xy
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines",
                                 line=dict(color="#888", width=1.5),
                                 name=f"lane {ln['lane_id']}", hoverinfo="skip"))
        cx, cy = ln["polygon"].centroid.x, ln["polygon"].centroid.y
        hd = np.radians(ln["heading_deg"])
        L = 8.0
        fig.add_annotation(x=cx + L * np.cos(hd), y=cy + L * np.sin(hd), ax=cx, ay=cy,
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowsize=1.4, arrowwidth=2, arrowcolor="#2ca02c")

    # real traffic backdrop (optional)
    if base_dets:
        fig.add_trace(go.Scatter(x=[d["cx"] for d in base_dets], y=[d["cy"] for d in base_dets],
                                 mode="markers", marker=dict(size=5, color="#1f77b4"),
                                 name="real traffic", hoverinfo="skip"))

    # driver full path (faint) + travelled path (red)
    if sim_track:
        px = [d["cx"] for d in sim_track]
        py = [d["cy"] for d in sim_track]
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color="#ddd", width=1, dash="dot"),
                                 name="planned path", hoverinfo="skip"))
        k = min(frame_idx, len(sim_track) - 1)
        fig.add_trace(go.Scatter(x=px[:k + 1], y=py[:k + 1], mode="lines",
                                 line=dict(color="#ff2b2b", width=2),
                                 name="driver path", hoverinfo="skip"))
        d = sim_track[k]
        col = "#ff2b2b" if flagged else "#ff7f0e"
        corners = _box_corners(d["cx"], d["cy"], d["yaw"], d["l"], d["w"])
        fig.add_trace(go.Scatter(x=[c[0] for c in corners], y=[c[1] for c in corners],
                                 mode="lines", fill="toself", fillcolor=col, opacity=0.85,
                                 line=dict(color="black", width=2),
                                 name="⚠ wrong-way driver" if flagged else "driver", hoverinfo="skip"))
        # heading arrow
        hd = d["heading"]; L = 6.0
        fig.add_annotation(x=d["cx"] + L * np.cos(hd), y=d["cy"] + L * np.sin(hd),
                           ax=d["cx"], ay=d["cy"], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=3, arrowcolor=col)

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
