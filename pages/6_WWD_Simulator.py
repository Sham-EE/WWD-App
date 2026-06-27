import streamlit as st
import numpy as np

import streamlit.components.v1 as components

from wwd_detection import load_lane_config, lanes_calibrated, detect_wrong_way
from wwd_simulator import (wrong_way_options, make_wrong_way_track,
                           build_sim_det_frames, simulator_figure, SIM_TID,
                           v2x_dashboard_html, math_heading_to_compass)
import viewer_ui as vu
import geo_reference as geo
import nav

st.set_page_config(layout="wide", page_title="WWD Simulator")
nav.render_sidebar()
st.title("🚨 Wrong-Way Driver Simulator")
st.markdown(
    "Real wrong-way events are rare and this dataset is all legal traffic, so this page "
    "**spawns a synthetic wrong-way driver** and runs it through the *real* WWD detector. "
    "Two ways to set up the scenario:"
)
st.markdown(
    "- **🛣️ Hand-drawn lanes** — use the lane geometry from the Lane Editor (the abstract "
    "sensor-frame view).\n"
    "- **🗺️ Draw on the real map** — draw the lanes *and* the driver's path straight onto the "
    "satellite map of the real intersection; it's georeferenced into the sensor frame and run "
    "through the same detector."
)


# ============================ Tab 1: hand-drawn lanes ============================
def render_hand_drawn_tab():
    lanes = load_lane_config()
    if not lanes:
        st.error("No lane configuration found (config/lanes.geojson). Calibrate lanes in the Lane Editor first.")
        return
    if not lanes_calibrated(lanes):
        st.warning("Lane geometry is not fully calibrated — the simulated directions may be off. "
                   "Calibrate in the Lane Editor for trustworthy results.")

    opts = wrong_way_options(lanes)
    # lane extent for a stable view
    allx, ally = [], []
    for ln in lanes:
        x0, y0, x1, y1 = ln["polygon"].bounds
        allx += [x0, x1]; ally += [y0, y1]
    m = 6.0
    x_range = (min(allx) - m, max(allx) + m)
    y_range = (min(ally) - m, max(ally) + m)

    # ---------------- Setup panel: one collapsible card, tabbed inside ----------------
    vu.ensure_toggle_defaults({
        "sim_show_lanes": True, "sim_show_legal_arrows": True, "sim_show_path": True,
        "sim_show_heading": True, "sim_show_real": True, "sim_show_grid": False,
        "sim_show_legend": True, "sim_show_hdmap_bev": True,
    })
    have_real = bool(st.session_state.get("detection_results"))

    with st.expander("⚙️ Simulation setup", expanded=True):
        t_scn, t_det, t_disp = st.tabs(["🚗 Scenario", "🛡️ Detector", "🎛️ Display"])

        with t_scn:
            labels = [o["label"] for o in opts]
            c1, c2 = st.columns([3, 2])
            choice = c1.selectbox("Wrong-way scenario (only illegal directions are offered)", labels,
                                  help="Each option drives a vehicle OPPOSITE to a lane's legal heading.")
            opt = opts[labels.index(choice)]
            speed = c2.slider("Driver speed (m/s)", 1.0, 25.0, 9.0, 0.5)
            c3, c4, c5 = st.columns(3)
            start_frac = c3.slider("Entry point along lane", 0.0, 0.8, 0.0, 0.05,
                                   help="Where along the lane the driver enters (0 = far end).")
            lateral_frac = c4.slider("Lane position (across)", 0.0, 1.0, 0.5, 0.05)
            fps = c5.number_input("FPS", 1.0, 30.0, 10.0, 1.0, help="Simulation frame rate.")

        with t_det:
            d1, d2, d3 = st.columns(3)
            conf_frames = d1.slider("Confirmation frames", 1, 30, 5, 1,
                                    help="Consecutive wrong-way frames required before the detector confirms it "
                                         "(WWD min_frames). Higher = more cautious, later detection.")
            min_speed_wwd = d2.slider("Min speed (m/s)", 0.5, 10.0, 1.0, 0.5,
                                      help="Below this the heading is treated as unreliable.")
            angle_thresh = d3.slider("Angle vs. flow (deg)", 90, 180, 120, 5,
                                     help="How far against the lane's legal direction counts as wrong-way.")

        with t_disp:
            st.caption("Show / hide overlays on the simulation view.")
            _disp_keys = ["sim_show_lanes", "sim_show_legal_arrows", "sim_show_path",
                          "sim_show_heading", "sim_show_grid", "sim_show_legend"]
            if have_real:
                _disp_keys.insert(4, "sim_show_real")
            vu.bulk_toggle_buttons(_disp_keys, "sim_disp", rerun_scope="app")
            tc1, tc2 = st.columns(2)
            tc1.toggle("🛣️ Lane boxes", key="sim_show_lanes")
            tc1.toggle("🗺️ HD-map roads (BEV)", key="sim_show_hdmap_bev",
                       help="Overlay the dataset's real HD-map road network on the BEV view "
                            "(the dev-kit 'digital twin' look).")
            tc1.toggle("➡️ Legal-direction arrows", key="sim_show_legal_arrows",
                       help="Per-lane arrow showing the legal flow direction.")
            tc1.toggle("〰️ Driver path", key="sim_show_path",
                       help="Planned + travelled path of the simulated driver.")
            tc2.toggle("🧭 Driver heading arrow", key="sim_show_heading")
            tc2.toggle("▦ Grid", key="sim_show_grid")
            tc2.toggle("🏷️ Legend", key="sim_show_legend")
            if have_real:
                st.toggle("🚗 Overlay real traffic", key="sim_show_real",
                          help="Animate the actual detected vehicles (from the last detection run) "
                               "alongside the simulated driver, colored by direction.")
            else:
                st.caption("Run **Object Detection and Tracking** first to overlay real moving traffic.")

    mix_real = have_real and st.session_state.sim_show_real

    # ---------------- Build + detect ----------------
    sim_track = make_wrong_way_track(opt["lane"], fps=fps, speed=speed,
                                     start_frac=start_frac, lateral_frac=lateral_frac)
    if not sim_track:
        st.error("Could not generate a path for this lane — try a different entry point.")
        return

    base = None
    start_frame = 0
    if mix_real:
        base = st.session_state.detection_results.get("det_frames")
        start_frame = st.slider("Start frame (in the real sequence)", 0, max(len(base) - len(sim_track), 0), 0)

    det_frames = build_sim_det_frames(sim_track, start_frame=start_frame, base_det_frames=base,
                                      total_frames=len(sim_track) + 5)
    ww = detect_wrong_way(det_frames, lanes, {"min_speed": min_speed_wwd, "min_frames": int(conf_frames),
                                              "angle_thresh_deg": float(angle_thresh)})
    sim_res = ww["tracks"].get(SIM_TID, {})
    is_flagged = SIM_TID in ww["wrong_way_tids"]
    first_flag = sim_res.get("first_flag_frame")  # det-frame index

    # ---------------- Playback + view ----------------
    st.subheader("▶️ Run the simulation")
    n_steps = len(sim_track)
    step, playing, play_delay = vu.nav_row("sim_step", n_steps, "sim", label="🎞️ Step")

    cur_frame_idx = start_frame + step
    confirm_frame = (first_flag + int(conf_frames) - 1) if (is_flagged and first_flag is not None) else None
    flagged_now = confirm_frame is not None and cur_frame_idx >= confirm_frame

    left, right = st.columns([3, 2], gap="medium")
    with left:
        base_dets = det_frames[cur_frame_idx] if (base and cur_frame_idx < len(det_frames)) else None
        if base_dets:
            base_dets = [d for d in base_dets if d.get("tid") != SIM_TID]
        _hdmap_bev = geo.hdmap_lanes_sensor_frame("south", 130.0) \
            if st.session_state.sim_show_hdmap_bev else None
        fig = simulator_figure(lanes, sim_track, step, flagged_now, base_dets=base_dets,
                               x_range=x_range, y_range=y_range,
                               show_lanes=st.session_state.sim_show_lanes,
                               show_legal_arrows=st.session_state.sim_show_legal_arrows,
                               show_path=st.session_state.sim_show_path,
                               show_heading=st.session_state.sim_show_heading,
                               show_real=st.session_state.sim_show_real,
                               show_grid=st.session_state.sim_show_grid,
                               show_legend=st.session_state.sim_show_legend,
                               hdmap_lanes=_hdmap_bev)
        st.plotly_chart(fig, use_container_width=True, key="sim_fig",
                        config={"scrollZoom": True})

    with right:
        st.markdown("#### Detector verdict")
        if is_flagged:
            start_step = first_flag - start_frame
            confirm_step = confirm_frame - start_frame
            t_s = confirm_step / float(fps)
            st.error("🚨 **WRONG-WAY DRIVING DETECTED**")
            st.write(f"- **Direction:** {opt['wrong_name']}-bound in the **{opt['lane_id']}** lane "
                     f"(legal: {opt['legal_name']})")
            st.write(f"- **Wrong-way motion starts:** step {start_step}")
            st.write(f"- **Confirmed (alert fires):** step {confirm_step} (~{t_s:.1f}s) — "
                     f"after {int(conf_frames)} confirmation frames")
            st.write(f"- **Max angle vs flow:** {sim_res.get('max_angle_deg',0):.0f}°  ·  "
                     f"**speed:** {speed:.1f} m/s")
            if flagged_now:
                st.success("Alert is ACTIVE at the current step.")
            else:
                st.info(f"Scrub to step {confirm_step} to reach the confirmation/alert.")
        else:
            st.success("No wrong-way flag — the driver did not sustain wrong-way motion for "
                       f"{int(conf_frames)} frames. Lower the confirmation frames, increase speed/length, "
                       "or check lane calibration.")

    # ---------------- Live geo map (native, real-time — the real intersection) ----------------
    st.divider()
    st.subheader("🗺️ Live geo map — real intersection")
    st.caption(f"**{geo.site_name()}** — the driver moves here in real time as the sim plays "
               "(no broadcast needed).")
    vu.ensure_toggle_defaults({"map_satellite": True, "map_hdmap": True,
                               "map_sensors": True, "map_compass": True})
    _mt = st.columns(4)
    _mt[0].toggle("🛰️ Satellite", key="map_satellite", help="Esri World Imagery vs street map.")
    _mt[1].toggle("🛣️ HD-map roads", key="map_hdmap",
                  help="The dataset's real HD-map lane network, laid on the imagery.")
    _mt[2].toggle("📡 LiDAR stations", key="map_sensors", help="The two gantry LiDARs + 120 m range rings.")
    _mt[3].toggle("🧭 Compass + cardinals", key="map_compass", help="True-north rose + per-lane cardinal labels.")
    try:
        import math as _math
        import pydeck as pdk

        def _card(b):                       # true bearing (deg) → cardinal letter
            return ["N", "E", "S", "W"][int(((b + 45.0) % 360.0) // 90.0)]

        _rings, _verts = [], []
        for ln in lanes:
            xs, ys = ln["polygon"].exterior.xy
            ring = list(zip([float(x) for x in xs], [float(y) for y in ys]))
            _rings.append((ln, ring))
            _verts.extend(ring)
        _proj = geo.make_projector("south", ref_points_xy=_verts)

        def _ll(x, y):                      # → [lon, lat] for pydeck
            lat, lon = _proj(x, y)
            return [lon, lat]

        _lls = [_proj(x, y) for x, y in _verts]
        _clat = sum(p[0] for p in _lls) / len(_lls)
        _clon = sum(p[1] for p in _lls) / len(_lls)
        _center = (_clat, _clon)
        _lane_data = [{"polygon": [_ll(x, y) for x, y in ring], "name": ln["lane_id"]}
                      for ln, ring in _rings]

        _k = min(step, len(sim_track) - 1)
        _path = [_ll(d["cx"], d["cy"]) for d in sim_track[:_k + 1]]
        _dpos = _ll(sim_track[_k]["cx"], sim_track[_k]["cy"])
        _dcol = [255, 43, 43] if flagged_now else [255, 165, 0]

        _layers = []
        if st.session_state.map_hdmap and _proj.exact:
            _roads = geo.hdmap_paths_near(_center, 130.0)
            if _roads:
                _layers.append(pdk.Layer("PathLayer", [{"path": p} for p in _roads],
                                         get_path="path", get_color=[255, 255, 255, 110],
                                         width_min_pixels=1))
        _layers += [
            pdk.Layer("PolygonLayer", _lane_data, get_polygon="polygon",
                      get_fill_color=[56, 132, 255, 35], get_line_color=[56, 132, 255, 220],
                      line_width_min_pixels=2, stroked=True, filled=True, pickable=True),
            pdk.Layer("PathLayer", [{"path": _path}] if len(_path) > 1 else [],
                      get_path="path", get_color=_dcol, width_min_pixels=3),
            pdk.Layer("ScatterplotLayer", [{"position": _dpos}], get_position="position",
                      get_fill_color=_dcol, get_line_color=[255, 255, 255], get_radius=4,
                      radius_min_pixels=7, radius_max_pixels=16, stroked=True, line_width_min_pixels=1),
        ]
        if mix_real and base_dets:
            _rt = [{"position": _ll(d["cx"], d["cy"])} for d in base_dets]
            _layers.append(pdk.Layer("ScatterplotLayer", _rt, get_position="position",
                                     get_fill_color=[150, 150, 150, 170], get_radius=3,
                                     radius_min_pixels=4))
        if st.session_state.map_sensors and _proj.exact:
            _sens, _ringp, _slbl = [], [], []
            for _sn, _col in (("south", [0, 200, 255]), ("north", [255, 122, 89])):
                _p = geo.sensor_position_latlon(_sn)
                if _p is None:
                    continue
                _sens.append({"position": [_p[1], _p[0]], "color": _col})
                _slbl.append({"position": [_p[1], _p[0]], "text": f"LiDAR {_sn}"})
                _ringp.append({"path": geo.circle_latlon(_p, 120.0), "color": _col})
            if _ringp:
                _layers.append(pdk.Layer("PathLayer", _ringp, get_path="path",
                                         get_color="color", width_min_pixels=1, opacity=0.5))
            if _sens:
                _layers.append(pdk.Layer("ScatterplotLayer", _sens, get_position="position",
                                         get_fill_color="color", get_line_color=[0, 0, 0],
                                         get_radius=5, radius_min_pixels=8, stroked=True,
                                         line_width_min_pixels=2))
                _layers.append(pdk.Layer("TextLayer", _slbl, get_position="position",
                                         get_text="text", get_size=13, get_color=[255, 255, 255],
                                         get_alignment_baseline="'top'", get_pixel_offset=[0, 10]))
        if st.session_state.map_compass:
            _rose = []
            for _name, _ang in (("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", 270.0)):
                _la, _lo = geo._enu_offset_latlon(90.0 * _math.sin(_math.radians(_ang)),
                                                  90.0 * _math.cos(_math.radians(_ang)), _center)
                _rose.append({"position": [_lo, _la], "text": _name})
            _layers.append(pdk.Layer("TextLayer", _rose, get_position="position", get_text="text",
                                     get_size=20, get_color=[255, 235, 120],
                                     get_alignment_baseline="'center'"))
            _clbl = []
            for ln, ring in _rings:
                cx = sum(p[0] for p in ring) / len(ring)
                cy = sum(p[1] for p in ring) / len(ring)
                _b = geo.bearing_at(cx, cy, _math.radians(float(ln["heading_deg"])), "south")
                _clbl.append({"position": _ll(cx, cy), "text": f"{_card(_b)} {_b:.0f}°"})
            _layers.append(pdk.Layer("TextLayer", _clbl, get_position="position", get_text="text",
                                     get_size=12, get_color=[120, 230, 255],
                                     get_alignment_baseline="'center'"))

        if st.session_state.map_satellite:
            _layers.insert(0, pdk.Layer(
                "TileLayer",
                data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                min_zoom=0, max_zoom=19, tile_size=256))
            _deck = pdk.Deck(layers=_layers, map_provider=None,
                             initial_view_state=pdk.ViewState(latitude=_clat, longitude=_clon,
                                                              zoom=17, pitch=0),
                             tooltip={"text": "{name}"})
        else:
            _deck = pdk.Deck(layers=_layers, map_provider="carto", map_style="road",
                             initial_view_state=pdk.ViewState(latitude=_clat, longitude=_clon,
                                                              zoom=17, pitch=0),
                             tooltip={"text": "{name}"})
        st.pydeck_chart(_deck, use_container_width=True)
        st.caption(("🛰️ **Exact georeferenced position** — HD-map roads, LiDAR stations and "
                    "lanes are placed from the dataset's surveyed HD-map anchor."
                    if _proj.exact else
                    "📍 **Approximate placement** (no exact georef — install pyproj + place the HD "
                    "map). Correct shape & orientation, centred on the site.")
                   + f"  Driver: {'🔴 wrong-way (alerting)' if flagged_now else '🟠 tracking'}.")
    except Exception as e:
        st.info(f"Live map unavailable ({type(e).__name__}: {e}). The view above and the V2X "
                "dashboard below still work.")

    # ---------------- V2X broadcast (external dashboard) ----------------
    st.divider()
    st.subheader("📡 V2X broadcast — WWD V2X Dashboard")
    _geo_ok = geo.has_georef("south")
    _d = sim_track[step]
    _true_bearing = geo.heading_to_true_bearing(_d["heading"], "south") if _geo_ok \
        else math_heading_to_compass(_d["heading"])
    _latlon = geo.sensor_xy_to_latlon(_d["cx"], _d["cy"], "south")
    _loc = _latlon if _latlon is not None else geo.site_latlon()
    st.caption(f"📍 **{geo.site_name()}**  ·  "
               + (f"driver @ {_latlon[0]:.6f}, {_latlon[1]:.6f} (exact)"
                  if _latlon is not None else
                  f"~{geo.site_latlon()[0]:.4f}, {geo.site_latlon()[1]:.4f} "
                  "(approx site — add the HD-map UTM anchor in geo_reference.py for exact per-driver lat/lon)")
               + (f"  ·  bearing **{_true_bearing:.0f}°** (true)" if _geo_ok
                  else "  ·  bearing assumes +y=north (no georef found)"))
    st.session_state.setdefault("v2x_armed", False)
    st.session_state.setdefault("v2x_event", None)

    if not is_flagged:
        st.caption("No wrong-way flag yet — once detected, broadcast it to your V2X dashboard here.")
    else:
        bc1, bc2 = st.columns([2, 1])
        if bc1.button("📡 Broadcast detection to the V2X Dashboard", type="primary",
                      disabled=not flagged_now,
                      help="Fires your dashboard's full J2735 TIM / C-V2X / nav-push pipeline with the "
                           "detected speed, true compass heading & geo-location. Play to the detection "
                           "step to enable."):
            st.session_state.v2x_event = {
                "speed": round(float(speed), 1), "heading": round(float(_true_bearing)),
                "lane": opt["lane_id"], "direction": opt["wrong_name"],
                "lat": round(float(_loc[0]), 6), "lon": round(float(_loc[1]), 6),
                "lat_exact": _latlon is not None, "site": geo.site_name(),
            }
            st.session_state.v2x_armed = True
            st.rerun()
        if st.session_state.v2x_armed and bc2.button("✖ Close dashboard", use_container_width=True):
            st.session_state.v2x_armed = False
            st.session_state.v2x_event = None
            st.rerun()
        if not flagged_now and not st.session_state.v2x_armed and confirm_frame is not None:
            st.caption(f"Scrub to step {confirm_frame - start_frame} (the confirmation moment) to enable the broadcast.")

    if st.session_state.v2x_armed and st.session_state.v2x_event:
        html = v2x_dashboard_html(st.session_state.v2x_event)
        if html is None:
            st.warning("V2X dashboard not found. Save your single-file app to "
                       "`assets/wwd_v2x_dashboard.html` (see assets/README.md), then broadcast again.")
        else:
            ev = st.session_state.v2x_event
            _loc_txt = (f"{ev['lat']:.6f}, {ev['lon']:.6f}"
                        + ("" if ev.get("lat_exact") else " (approx)"))
            st.success(f"🚨 Broadcasting: {ev['direction']}-bound in the {ev['lane']} lane · "
                       f"{ev['speed']} m/s · heading {ev['heading']}° (true) · 📍 {_loc_txt} — "
                       "the dashboard fired its alert pipeline below.")
            components.html(html, height=1500, scrolling=True)

    # auto-advance (paused while the dashboard is embedded to avoid re-render churn)
    if playing and step < n_steps - 1 and not st.session_state.v2x_armed:
        import time
        time.sleep(float(play_delay))
        st.session_state.sim_step = step + 1
        st.rerun()


# ============================ Tab 2: draw on the real map ============================
def _track_from_path(pts, fps=10.0, speed=9.0, max_frames=800):
    """Resample a hand-drawn path (sensor-frame (x,y) vertices) into per-frame
    detection dicts at constant speed — the shape detect_wrong_way expects."""
    P = [np.asarray(p, float) for p in pts]
    segs = []
    for i in range(len(P) - 1):
        d = P[i + 1] - P[i]
        L = float(np.hypot(d[0], d[1]))
        if L > 1e-9:
            segs.append((P[i], d / L, L))
    if not segs:
        return []
    total = sum(L for _, _, L in segs)
    dt = 1.0 / float(fps) if fps else 0.1
    step = max(float(speed) * dt, 1e-3)
    dets, s = [], 0.0
    while s <= total + 1e-9 and len(dets) < max_frames:
        acc, pos, u = 0.0, None, segs[-1][1]
        for a, du, L in segs:
            if s <= acc + L:
                pos, u = a + du * (s - acc), du
                break
            acc += L
        if pos is None:
            a, u, L = segs[-1][0], segs[-1][1], segs[-1][2]
            pos = a + u * L
        head = float(np.arctan2(u[1], u[0]))
        dets.append(dict(tid=SIM_TID, cls="Car", cx=float(pos[0]), cy=float(pos[1]),
                         yaw=head, heading=head, vx=float(u[0] * speed), vy=float(u[1] * speed),
                         speed=float(speed), l=4.5, w=1.9, length=4.5, width=1.9,
                         moving=True, hit=True, score=300.0, is_vehicle=True, simulated=True))
        s += step
    return dets


def _lane_from_latlon(latlon, lane_id, width_m=3.5):
    """Drawn lat/lon polyline → a lane dict (sensor-frame polygon + legal heading)."""
    from shapely.geometry import LineString
    sensor = []
    for lat, lon in latlon:
        xy = geo.latlon_to_sensor(lat, lon, "south")
        if xy is None:
            return None
        sensor.append(xy)
    if len(sensor) < 2:
        return None
    a, b = np.array(sensor[0]), np.array(sensor[-1])
    head = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
    return {"lane_id": lane_id, "name": lane_id, "heading_deg": head, "calibrated": True,
            "polygon": LineString(sensor).buffer(width_m / 2.0, cap_style=2), "latlon": list(latlon)}


def _ls_coords(drawing):
    """(lat, lon) vertices of a drawn LineString GeoJSON feature, else None."""
    if not drawing:
        return None
    g = drawing.get("geometry", {})
    if g.get("type") != "LineString" or len(g.get("coordinates", [])) < 2:
        return None
    return [(c[1], c[0]) for c in g["coordinates"]]   # GeoJSON is [lon, lat]


def render_real_map_tab():
    st.markdown("Draw your scenario **right on the real intersection** — lanes first (with their "
                "legal direction = the way you draw them), then the wrong-way driver's path. "
                "It's georeferenced into the sensor frame and run through the same WWD detector.")

    if not geo.has_exact_georef("south"):
        st.warning("Exact georeferencing isn't available (needs the HD map + `pyproj`), so map "
                   "drawings can't be converted into the sensor frame. Use the **Hand-drawn lanes** "
                   "tab, or add the HD map + pyproj to enable this.")
        return

    try:
        import folium
        from folium.plugins import Draw
        from streamlit_folium import st_folium
    except Exception as e:
        st.error(f"Map drawing needs `folium` + `streamlit-folium` ({type(e).__name__}: {e}). "
                 "Install them: `pip install folium streamlit-folium`.")
        return

    ss = st.session_state
    ss.setdefault("rm_lanes", [])      # list of lane dicts (with .latlon)
    ss.setdefault("rm_path", None)     # list of (lat, lon)
    ss.setdefault("rm_result", None)   # last detection result
    ss.setdefault("rm_v2x", None)      # armed broadcast event

    center = geo.site_latlon()

    # ---- scenario knobs ----
    cset = st.columns(4)
    speed = cset[0].slider("Driver speed (m/s)", 1.0, 25.0, 9.0, 0.5, key="rm_speed")
    fps = cset[1].number_input("FPS", 1.0, 30.0, 10.0, 1.0, key="rm_fps")
    conf_frames = cset[2].slider("Confirmation frames", 1, 30, 5, 1, key="rm_conf")
    angle_thresh = cset[3].slider("Angle vs flow (deg)", 90, 180, 120, 5, key="rm_angle")

    # ---- the drawing map ----
    fmap = folium.Map(location=[center[0], center[1]], zoom_start=19, max_zoom=22,
                      tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
        max_zoom=22, max_native_zoom=19).add_to(fmap)   # over-zoom (upscale) past native 19
    flagged = bool(ss.rm_result and ss.rm_result.get("flagged"))
    for ln in ss.rm_lanes:
        folium.PolyLine([[la, lo] for la, lo in ln["latlon"]], color="#3884ff", weight=5, opacity=.9,
                        tooltip=f"{ln['lane_id']} · legal "
                                f"{geo.heading_to_true_bearing(np.radians(ln['heading_deg']),'south'):.0f}°"
                        ).add_to(fmap)
        folium.CircleMarker([ln["latlon"][-1][0], ln["latlon"][-1][1]], radius=4,
                            color="#3884ff", fill=True, tooltip="lane exit →").add_to(fmap)
    if ss.rm_path:
        folium.PolyLine([[la, lo] for la, lo in ss.rm_path],
                        color="#ff2b2b" if flagged else "#ffa500", weight=4, opacity=.95,
                        dash_array="6", tooltip="driver path").add_to(fmap)
    Draw(export=False, draw_options={"polyline": True, "polygon": False, "rectangle": False,
                                     "circle": False, "marker": False, "circlemarker": False},
         edit_options={"edit": False}).add_to(fmap)

    out = st_folium(fmap, key="rm_map", height=520, use_container_width=True,
                    returned_objects=["last_active_drawing"])
    drawn = _ls_coords((out or {}).get("last_active_drawing"))

    st.caption("Use the **line tool** (top-left of the map) to draw, then add it below. Draw the lane "
               "in its **legal** direction; the driver path is what the car actually does.")

    # ---- capture drawings ----
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("➕ Add as lane", disabled=drawn is None, use_container_width=True):
        ln = _lane_from_latlon(drawn, f"lane_{len(ss.rm_lanes)+1}")
        if ln:
            ss.rm_lanes.append(ln)
            ss.rm_result = None
            st.rerun()
        else:
            st.warning("Couldn't convert that line — draw a polyline with 2+ points.")
    if b2.button("🧍 Set as driver path", disabled=drawn is None, use_container_width=True):
        ss.rm_path = drawn
        ss.rm_result = None
        st.rerun()
    if b3.button("↩️ Remove last lane", disabled=not ss.rm_lanes, use_container_width=True):
        ss.rm_lanes.pop()
        ss.rm_result = None
        st.rerun()
    if b4.button("🗑️ Clear all", use_container_width=True):
        ss.rm_lanes, ss.rm_path, ss.rm_result, ss.rm_v2x = [], None, None, None
        st.rerun()

    st.write(f"**Lanes drawn:** {len(ss.rm_lanes)}  ·  **Driver path:** "
             f"{'✅ set' if ss.rm_path else '— not set'}")

    # ---- run the check ----
    ready = bool(ss.rm_lanes and ss.rm_path)
    if st.button("🚦 Check for wrong-way", type="primary", disabled=not ready):
        sensor_path = [geo.latlon_to_sensor(la, lo, "south") for la, lo in ss.rm_path]
        track = _track_from_path(sensor_path, fps=fps, speed=speed)
        if not track:
            st.warning("The driver path is too short — draw a longer line.")
        else:
            det_frames = build_sim_det_frames(track, total_frames=len(track) + 5)
            ww = detect_wrong_way(det_frames, ss.rm_lanes,
                                  {"min_speed": 1.0, "min_frames": int(conf_frames),
                                   "angle_thresh_deg": float(angle_thresh)})
            res = ww["tracks"].get(SIM_TID, {})
            ff = res.get("first_flag_frame")
            cidx = (ff + int(conf_frames) - 1) if ff is not None else None
            d_at = track[min(cidx, len(track) - 1)] if cidx is not None else track[-1]
            ss.rm_result = {
                "flagged": SIM_TID in ww["wrong_way_tids"],
                "lane_id": res.get("lane_id"), "max_angle": res.get("max_angle_deg", 0.0),
                "confirm_step": cidx, "t_s": (cidx / float(fps)) if cidx is not None else None,
                "speed": float(speed),
                "latlon": geo.sensor_xy_to_latlon(d_at["cx"], d_at["cy"], "south"),
                "bearing": geo.heading_to_true_bearing(d_at["heading"], "south"),
            }
            st.rerun()

    if not ready:
        st.info("Draw **at least one lane** and **a driver path**, then run the check.")

    # ---- verdict + broadcast ----
    res = ss.rm_result
    if res:
        st.divider()
        if res["flagged"]:
            st.error("🚨 **WRONG-WAY DRIVING DETECTED**")
            st.write(f"- **Lane:** {res['lane_id']}  ·  **max angle vs flow:** {res['max_angle']:.0f}°")
            st.write(f"- **Confirmed at step:** {res['confirm_step']} (~{res['t_s']:.1f}s) "
                     f"after {int(conf_frames)} frames")
            if res["latlon"]:
                st.write(f"- **Location:** {res['latlon'][0]:.6f}, {res['latlon'][1]:.6f}  ·  "
                         f"**bearing:** {res['bearing']:.0f}° (true)")
            st.caption("The driver path on the map above is drawn red.")

            bc1, bc2 = st.columns([2, 1])
            if bc1.button("📡 Broadcast to the V2X Dashboard", type="primary", use_container_width=True):
                loc = res["latlon"] or geo.site_latlon()
                _b = float(res["bearing"])
                _dir = ["N", "E", "S", "W"][int(((_b + 45.0) % 360.0) // 90.0)]
                ss.rm_v2x = {"speed": round(res["speed"], 1), "heading": round(_b),
                             "lane": res["lane_id"] or "lane", "direction": _dir,
                             "lat": round(float(loc[0]), 6), "lon": round(float(loc[1]), 6),
                             "lat_exact": res["latlon"] is not None, "site": geo.site_name()}
                st.rerun()
            if ss.rm_v2x and bc2.button("✖ Close dashboard", use_container_width=True):
                ss.rm_v2x = None
                st.rerun()
        else:
            st.success("No wrong-way flag — the drawn path stayed within the lanes' legal direction "
                       "(or didn't sustain wrong-way motion long enough). Try drawing the path against "
                       "a lane's direction, or lower the confirmation frames.")

    if ss.rm_v2x:
        html = v2x_dashboard_html(ss.rm_v2x)
        if html is None:
            st.warning("V2X dashboard not found. Save your single-file app to "
                       "`assets/wwd_v2x_dashboard.html` (see assets/README.md), then broadcast again.")
        else:
            ev = ss.rm_v2x
            st.success(f"🚨 Broadcasting: {ev['lane']} · {ev['speed']} m/s · heading {ev['heading']}° "
                       f"(true) · 📍 {ev['lat']:.6f}, {ev['lon']:.6f} — dashboard fired below.")
            components.html(html, height=1500, scrolling=True)


# ============================ tabs ============================
tab_lanes, tab_map = st.tabs(["🛣️ Hand-drawn lanes", "🗺️ Draw on the real map"])
with tab_lanes:
    render_hand_drawn_tab()
with tab_map:
    render_real_map_tab()
