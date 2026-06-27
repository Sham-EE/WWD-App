import streamlit as st
import numpy as np

import streamlit.components.v1 as components
import plotly.graph_objects as go

from wwd_detection import load_lane_config, lanes_calibrated, detect_wrong_way
from wwd_simulator import (wrong_way_options, make_wrong_way_track,
                           build_sim_det_frames, simulator_figure, SIM_TID,
                           v2x_dashboard_html, math_heading_to_compass)
import viewer_ui as vu
import geo_reference as geo
import dataset_manager as dm
import road_viewer as rv
import label_projection as lp
import lidar_viewer as lv
import dataset_prep as dp
import registration as reg
import nav

st.set_page_config(layout="wide", page_title="WWD Simulator")
nav.render_sidebar()
st.title("🚨 Wrong-Way Driver Simulator")
st.markdown(
    "Real wrong-way events are rare and this dataset is all legal traffic, so this page "
    "**spawns a synthetic wrong-way driver** and runs it through the *real* WWD detector — then "
    "shows it **driving through the actual LiDAR scan** (the same 3D view as the Visualizer)."
)

ds = dm.get_active()
lanes = load_lane_config()
if not lanes:
    st.error("No lane configuration found (config/lanes.geojson). Calibrate lanes in the Lane Editor first.")
    st.stop()
if not lanes_calibrated(lanes):
    st.warning("Lane geometry is not fully calibrated — the simulated directions may be off. "
               "Calibrate in the Lane Editor for trustworthy results.")

opts = wrong_way_options(lanes)
# lane extent for a stable BEV view
allx, ally = [], []
for ln in lanes:
    x0, y0, x1, y1 = ln["polygon"].bounds
    allx += [x0, x1]; ally += [y0, y1]
m = 6.0
x_range = (min(allx) - m, max(allx) + m)
y_range = (min(ally) - m, max(ally) + m)

# Real LiDAR frames (south/cropped — the frame the lanes + sim driver live in).
_pcds = rv.list_by_frame(ds.input_pcd_for_sensor("south", "cropped"), [".pcd"])
_labels = rv.list_by_frame(ds.labels_dir_for("south", "scorable"), [".json"])
_n_frames = len(_pcds)


@st.cache_data(show_spinner=False, max_entries=128)
def _sim_load_pts(path, max_pts):
    return lv.load_points(path, max_points=max_pts)


def _ground_z(pts, gt_objs):
    """Best ground-height estimate so the synthetic box sits on the road: median GT
    box centre-z if boxes exist, else a low percentile of the cloud."""
    if gt_objs:
        return float(np.median([o["val"][2] for o in gt_objs]))
    if pts is not None and len(pts):
        return float(np.percentile(pts[:, 2], 50)) + 0.8
    return 0.0


def _driver_val(d, z, length=4.5, width=1.9, height=1.6):
    """OpenLABEL cuboid [x,y,z, qx,qy,qz,qw, l,w,h] for the synthetic driver."""
    yaw = float(d["heading"])
    return [float(d["cx"]), float(d["cy"]), float(z), 0.0, 0.0,
            float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0)), length, width, height]


def _add_box(fig, val, color, name, width=6):
    c = lp.cuboid_corners(val)
    xs, ys, zs = [], [], []
    for a, b in lp._EDGES + lp._FRONT_DIAGONALS:
        xs += [c[a, 0], c[b, 0], None]
        ys += [c[a, 1], c[b, 1], None]
        zs += [c[a, 2], c[b, 2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                               line=dict(color=color, width=width), name=name,
                               hoverinfo="name", showlegend=True))


def _add_ground_polyline(fig, xy, z, color, name, width=3, closed=False):
    xs = [float(p[0]) for p in xy]
    ys = [float(p[1]) for p in xy]
    if closed and xy:
        xs.append(xs[0]); ys.append(ys[0])
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[z] * len(xs), mode="lines",
                               line=dict(color=color, width=width), name=name,
                               hoverinfo="name", showlegend=False))


# ---------------- Setup panel: one collapsible card, tabbed inside ----------------
vu.ensure_toggle_defaults({
    "sim_show_lanes": True, "sim_show_legal_arrows": True, "sim_show_path": True,
    "sim_show_heading": True, "sim_show_real": True, "sim_show_grid": False,
    "sim_show_legend": True, "sim_show_hdmap_bev": True,
    # 3D-scan view
    "sim3d_points": True, "sim3d_boxes": True, "sim3d_road": True,
    "sim3d_hdmap": True, "sim3d_lidar": True, "sim3d_lanes": True,
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
        view_mode = st.radio("Main view", ["🧊 LiDAR scan (3D)", "▦ Abstract BEV"], horizontal=True,
                             help="3D scan = the synthetic driver overlaid on the real LiDAR frame "
                                  "(same view as the Visualizer). BEV = the abstract top-down lanes.")
        st.divider()
        if view_mode.startswith("🧊"):
            st.caption("3D scan overlays.")
            _pts_shown = st.select_slider("Points shown", [10000, 20000, 30000, 50000], value=20000,
                                          key="sim3d_maxpts")
            st.caption("(The 🧊 Point cloud toggle is above the view — turn it off for smooth playback.)")
            q1, q2, q3 = st.columns(3)
            q1.toggle("📦 GT boxes", key="sim3d_boxes")
            q1.toggle("🛣️ WWD lanes", key="sim3d_lanes")
            q2.toggle("🛣️ Road outline", key="sim3d_road")
            q2.toggle("🗺️ HD-map roads", key="sim3d_hdmap")
            q3.toggle("📍 LiDAR stations", key="sim3d_lidar")
        else:
            st.caption("Show / hide overlays on the abstract BEV view.")
            _disp_keys = ["sim_show_lanes", "sim_show_legal_arrows", "sim_show_path",
                          "sim_show_heading", "sim_show_grid", "sim_show_legend"]
            if have_real:
                _disp_keys.insert(4, "sim_show_real")
            vu.bulk_toggle_buttons(_disp_keys, "sim_disp", rerun_scope="app")
            tc1, tc2 = st.columns(2)
            tc1.toggle("🛣️ Lane boxes", key="sim_show_lanes")
            tc1.toggle("🗺️ HD-map roads (BEV)", key="sim_show_hdmap_bev")
            tc1.toggle("➡️ Legal-direction arrows", key="sim_show_legal_arrows")
            tc1.toggle("〰️ Driver path", key="sim_show_path")
            tc2.toggle("🧭 Driver heading arrow", key="sim_show_heading")
            tc2.toggle("▦ Grid", key="sim_show_grid")
            tc2.toggle("🏷️ Legend", key="sim_show_legend")
            if have_real:
                st.toggle("🚗 Overlay real traffic", key="sim_show_real")
            else:
                st.caption("Run **Object Detection and Tracking** first to overlay real moving traffic.")

is_3d = view_mode.startswith("🧊")
mix_real = (not is_3d) and have_real and st.session_state.sim_show_real

# ---------------- Build + detect ----------------
sim_track = make_wrong_way_track(opt["lane"], fps=fps, speed=speed,
                                 start_frac=start_frac, lateral_frac=lateral_frac)
if not sim_track:
    st.error("Could not generate a path for this lane — try a different entry point.")
    st.stop()

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
_dcol = "#ff2b2b" if flagged_now else "#ffa500"

left, right = st.columns([3, 2], gap="medium")
with left:
    if is_3d and _n_frames:
        st.toggle("🧊 Point cloud", key="sim3d_points",
                  help="Turn the LiDAR points OFF for much smoother playback — only the "
                       "lightweight boxes / lanes / driver animate (no 20k-point reload each frame).")
        # The driver steps through the real LiDAR frames as it plays.
        scene_i = min(start_frame + step, _n_frames - 1)
        pts = _sim_load_pts(_pcds[scene_i], int(st.session_state.get("sim3d_maxpts", 20000)))
        gt_objs = lp.load_objects(_labels[scene_i]) if scene_i < len(_labels) else []
        road = dp.road_polygon(0.0) if st.session_state.sim3d_road else None
        sensors = reg.lidar_markers(ds, "south") if st.session_state.sim3d_lidar else None
        hdmap = geo.hdmap_lanes_sensor_frame("south", 130.0) if st.session_state.sim3d_hdmap else None
        fig = lv.build_figure(pts if st.session_state.sim3d_points else np.zeros((0, 3)),
                              gt_objs if st.session_state.sim3d_boxes else [],
                              "by_category", height=620, road_poly=road,
                              sensors=sensors, hdmap_lanes=hdmap)
        gz = _ground_z(pts, gt_objs)
        # WWD lanes as ground rectangles
        if st.session_state.sim3d_lanes:
            for ln in lanes:
                xs, ys = ln["polygon"].exterior.xy
                _add_ground_polyline(fig, list(zip(xs, ys)), gz, "#3884ff", f"lane {ln['lane_id']}", width=3)
        # driver path travelled so far + current box
        d = sim_track[step]
        _add_ground_polyline(fig, [(dd["cx"], dd["cy"]) for dd in sim_track[:step + 1]],
                             gz, _dcol, "driver path", width=4)
        _add_box(fig, _driver_val(d, gz + 0.8), _dcol, "🚨 WWD driver", width=7)
        st.plotly_chart(fig, use_container_width=True, key="sim_fig3d")
        st.caption(f"Real frame {scene_i + 1}/{_n_frames} · {len(gt_objs)} GT boxes · "
                   f"{len(pts):,} points · driver step {step + 1}/{n_steps} "
                   f"({'🔴 wrong-way' if flagged_now else '🟠 tracking'})")
    else:
        if is_3d and not _n_frames:
            st.info("No LiDAR frames found for south/cropped — showing the abstract BEV instead. "
                    "Run **Dataset Prep → Crop** to generate the cloud.")
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
    _dcol_rgb = [255, 43, 43] if flagged_now else [255, 165, 0]

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
                  get_path="path", get_color=_dcol_rgb, width_min_pixels=3),
        pdk.Layer("ScatterplotLayer", [{"position": _dpos}], get_position="position",
                  get_fill_color=_dcol_rgb, get_line_color=[255, 255, 255], get_radius=4,
                  radius_min_pixels=7, radius_max_pixels=16, stroked=True, line_width_min_pixels=1),
    ]
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
