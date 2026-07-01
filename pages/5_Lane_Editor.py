import os
import json

import streamlit as st
import numpy as np

from lane_tools import (
    load_tracks, moving_points, auto_lanes,
    LANE_DIRECTIONS, direction_to_heading, heading_to_direction,
    load_track_paths, heading_from_tracks_in_lane,
    lanes_to_geojson, geojson_to_lanes, build_preview, build_draw_figure,
    simplify_path, load_pcd_background,
)

import nav
st.set_page_config(layout="wide", page_title="Lane Editor")
nav.render_sidebar()


@st.cache_data(show_spinner="Loading point-cloud background…")
def _cached_background(pcd_dir, n_frames):
    return load_pcd_background(pcd_dir, n_frames=n_frames)


import dataset_manager as dm
_ds = dm.get_active()
LANES_PATH = _ds.lanes_path
PREVIEW_H = 640

st.session_state.setdefault('le_lanes', [])
st.session_state.setdefault('le_v', 0)  # widget key version (bump to reseed)


# The direction dropdown and the numeric Heading° field are two views of the SAME
# lane heading — edit either and the other follows (via these on_change callbacks).
def _sync_dir_to_heading(idx, dirkey, hdkey, north):
    """Dropdown picked → snap heading to that compass cardinal; mirror into the number."""
    h = direction_to_heading(st.session_state[dirkey], north)
    st.session_state.le_lanes[idx]['heading_deg'] = h
    st.session_state[hdkey] = h


def _sync_heading_to_dir(idx, hdkey, dirkey, north):
    """Heading number edited → store it; mirror the matching direction into the dropdown."""
    h = float(st.session_state[hdkey])
    st.session_state.le_lanes[idx]['heading_deg'] = h
    st.session_state[dirkey] = heading_to_direction(h, north)


def _reseed_lane_widgets():
    """Bump the widget-key version so the dropdown/heading re-derive in the newly
    selected cardinal frame (used when the True-cardinals toggle flips)."""
    st.session_state.le_v += 1


st.title("🛣️ Lane Editor")

# Follow the same sensor/source as the rest of the app (shared pipeline_* state), so the
# point-cloud backdrop AND the auto-detect tracks come from the active pipeline
# (registered/cropped by default) — not a hardcoded source.
_lsc, _lic = st.columns(2)
_le_sensor = _lsc.radio("Sensor", ["Registered", "South", "North"], key="pipeline_sensor",
                        horizontal=True, help="Which LiDAR's detection tracks + cloud backdrop to use. "
                        "Shared with Background Filtering / Detection / Geometry Editor.").lower()
_le_src = "cropped" if _lic.radio("Input cloud", ["Cropped (road)", "Full (uncropped)"],
                                  key="pipeline_source", horizontal=True).startswith("Cropped") else "full"
DEFAULT_TRACKS = os.path.join(_ds.detection_dir_for_sensor(_le_sensor, _le_src), "tracks.csv")
DEFAULT_PCD_BG = _ds.input_pcd_for_sensor(_le_sensor, _le_src)
st.caption(f"🛰️ {_le_sensor.capitalize()} · {'Cropped' if _le_src=='cropped' else 'Full'}  ·  "
           f"tracks: `{'found' if os.path.exists(DEFAULT_TRACKS) else 'none — run Detection'}`")

# Georeference for THIS sensor — used to map lane directions (Eastbound/…) to the
# real compass heading, so the dropdown/colors reflect true-life cardinals. Falls
# back to sensor-frame axes when no georeference exists.
import geo_reference as geo
_gref = "north" if _le_sensor == "north" else "south"
try:
    _has_geo = geo.has_georef(_gref)
except Exception:
    _has_geo = False
_dir_north = None
if _has_geo:
    try:
        _dir_north = geo.heading_to_true_bearing(0.0, _gref)  # sensor math-heading of true North
    except Exception:
        _dir_north = None
# The "🧭 True cardinals" toggle (rendered in the right panel) also governs the
# frame the direction dropdown/name map in — read its remembered state up here so
# names, colors and the compass all stay in the same frame. Off / no georeference
# → sensor-frame axes (E=+X, N=+Y, …).
_use_true = bool(st.session_state.get("le_true_card", _has_geo)) and _has_geo
_active_north = _dir_north if _use_true else None

# ---------------- Top bar: data + template actions ----------------
top = st.columns([2.3, 1.1, 1.1, 1.3, 1.3, 1.3])
uploaded = top[0].file_uploader("tracks.csv", type="csv", label_visibility="collapsed")
src = uploaded if uploaded is not None else (DEFAULT_TRACKS if os.path.exists(DEFAULT_TRACKS) else None)
min_speed = top[1].slider("Min speed", 0.0, 10.0, 1.0, 0.5, help="m/s; exclude slow/parked points.")
k = top[2].number_input("# lanes", 1, 8, 4, help="Lanes to auto-detect.")

points, paths = None, []
if src is not None:
    try:
        _df_tracks = load_tracks(src)
        points = moving_points(_df_tracks, min_speed=min_speed)
        # Full per-car trajectories (start->end bearings) for "set heading from vehicles".
        paths = load_track_paths(_df_tracks)
    except Exception as e:
        st.error(f"Could not read tracks: {e}")

if top[3].button("✨ Auto-generate", use_container_width=True,
                 disabled=points is None or len(points) == 0,
                 help="Cluster the moving vehicles into the chosen number of directions and "
                      "create one starting lane box (with a measured heading) per direction. "
                      "A rough template you then refine by hand."):
    st.session_state.le_lanes = auto_lanes(points, k=int(k), buffer_m=2.0)
    st.session_state.le_v += 1
    st.rerun()
if top[4].button("📂 Load saved", use_container_width=True, disabled=not os.path.exists(LANES_PATH),
                 help="Load the lanes from the active config/lanes.geojson into the editor so "
                      "you can view or adjust them."):
    with open(LANES_PATH) as f:
        st.session_state.le_lanes = geojson_to_lanes(json.load(f))
    st.session_state.le_v += 1
    st.rerun()
if top[5].button("🔄 Default lanes", use_container_width=True,
                 disabled=not os.path.exists(_ds.default_lanes_path),
                 help="Reset to the dataset's default calibrated lanes "
                      "(config/defaults/lanes.geojson) — an honest representation of the road, "
                      "in case you messed up your edits."):
    with open(_ds.default_lanes_path) as f:
        st.session_state.le_lanes = geojson_to_lanes(json.load(f))
    st.session_state.le_v += 1
    st.rerun()

st.divider()

# ---------------- Symmetric split: editor (left) | live preview (right) ----------------
left, right = st.columns(2, gap="large")
lanes = st.session_state.le_lanes
v = st.session_state.le_v

# Default lanes keyed by lane_id, so each lane can be individually reset to its
# calibrated baseline (config/defaults/lanes.geojson).
_def_lanes_by_id = {}
if os.path.exists(_ds.default_lanes_path):
    try:
        with open(_ds.default_lanes_path) as _f:
            for _dl in geojson_to_lanes(json.load(_f)):
                _def_lanes_by_id[str(_dl['lane_id'])] = _dl
    except Exception:
        pass

with left:
    st.markdown("##### Lanes")
    _m = st.session_state.pop('le_veh_msg', None)   # from a "set heading from vehicles" action
    if _m:
        (st.success if _m[0] == 'success' else st.warning)(_m[1])
    box = st.container(height=PREVIEW_H, border=False)
    with box:
        if not lanes:
            st.info("No lanes yet — use ✨ Auto-generate above, or ➕ Add lane below.")
        delete_idx = None
        reset_action = None  # (idx, default_lane) — applied after the loop
        veh_idx = None       # idx — set this lane's heading from the vehicles inside it
        for i, l in enumerate(lanes):
            dft = _def_lanes_by_id.get(str(l['lane_id']))
            # A lane IS a travel direction. The dropdown snaps to a compass cardinal;
            # the Heading° field fine-tunes it — both are kept in sync (they seed from
            # the lane heading; the version key `v` re-seeds them on load/reset/toggle).
            _wkey, _hdkey = f"dir_{i}_{v}", f"hd_{i}_{v}"
            st.session_state.setdefault(_hdkey, float(l['heading_deg']))
            st.session_state.setdefault(_wkey, heading_to_direction(l['heading_deg'], _active_north))
            _name = st.session_state[_wkey]
            _name = _name if _name in LANE_DIRECTIONS else heading_to_direction(l['heading_deg'], _active_north)
            with st.expander(f"🛣️ {_name}", expanded=False):
                a = st.columns([1.9, 1.3, 0.8])
                a[0].selectbox("Direction", LANE_DIRECTIONS, key=_wkey,
                               on_change=_sync_dir_to_heading, args=(i, _wkey, _hdkey, _active_north),
                               help="Snap the lane to a compass cardinal — sets its heading, name and color.")
                a[1].number_input("Heading°", step=1.0, format="%.1f", key=_hdkey,
                                  on_change=_sync_heading_to_dir, args=(i, _hdkey, _wkey, _active_north),
                                  help="Fine-tune the exact heading (0=+X, 90=+Y, 180=−X, −90=−Y). "
                                       "The direction dropdown + name follow it.")
                a[2].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                if a[2].button("🗑", key=f"del_{i}_{v}", help="Delete lane"):
                    delete_idx = i
                # Reconcile the lane object with the (possibly just-edited) widgets so the
                # preview + save use the live values.
                l['heading_deg'] = float(st.session_state[_hdkey])
                l['lane_id'] = st.session_state[_wkey]  # the lane's name is its direction
                if l.get('polygon'):
                    st.caption(f"✏️ Drawn polygon · {len(l['polygon'])} vertices — re-shape it in "
                               "**✏️ Draw mode** (right) via *Replace selected lane*.")
                else:
                    xc = st.columns(2)
                    l['xmin'] = xc[0].number_input("X min", value=float(l['xmin']), step=1.0,
                                                   format="%.1f", key=f"xmin_{i}_{v}")
                    l['xmax'] = xc[1].number_input("X max", value=float(l['xmax']), step=1.0,
                                                   format="%.1f", key=f"xmax_{i}_{v}")
                    yc = st.columns(2)
                    l['ymin'] = yc[0].number_input("Y min", value=float(l['ymin']), step=1.0,
                                                   format="%.1f", key=f"ymin_{i}_{v}")
                    l['ymax'] = yc[1].number_input("Y max", value=float(l['ymax']), step=1.0,
                                                   format="%.1f", key=f"ymax_{i}_{v}")
                rc = st.columns(2)
                if rc[0].button("🚗 Heading from vehicles", key=f"veh_{i}_{v}",
                                use_container_width=True, disabled=not paths,
                                help="Set this lane's heading to the dominant travel direction of the "
                                     "cars whose path runs through it (their real start→end bearing, "
                                     "ignoring turners). Turn-only lanes have no through-traffic to use — "
                                     "you'll be told to set those by hand." if paths
                                     else "No track trajectories loaded."):
                    veh_idx = i
                if rc[1].button("↺ Reset to default", key=f"rst_{i}_{v}",
                                use_container_width=True, disabled=dft is None,
                                help=("Restore just this lane's box + heading from "
                                      "config/defaults/lanes.geojson." if dft is not None
                                      else "No default with this lane name to restore from.")):
                    reset_action = (i, dft)
        if delete_idx is not None:
            lanes.pop(delete_idx)
            st.session_state.le_v += 1
            st.rerun()
        if veh_idx is not None:
            _h, _n = heading_from_tracks_in_lane(lanes[veh_idx], paths)
            _lid = lanes[veh_idx]['lane_id']
            if _h is None:
                st.session_state['le_veh_msg'] = ('warning',
                    f"No straight-through vehicles inside “{_lid}” — it's likely only turned "
                    f"into/out of. Set its Heading° by hand (or copy a neighbouring lane).")
            else:
                lanes[veh_idx]['heading_deg'] = _h
                st.session_state['le_veh_msg'] = ('success',
                    f"Set “{_lid}” heading to {_h:.1f}° from {_n} vehicle(s) through the lane.")
            st.session_state.le_v += 1
            st.rerun()
        if reset_action is not None:
            idx, dft = reset_action
            lanes[idx].pop('polygon', None)   # defaults are boxes → back to a box lane
            lanes[idx].update(xmin=dft['xmin'], xmax=dft['xmax'], ymin=dft['ymin'],
                              ymax=dft['ymax'], heading_deg=dft['heading_deg'])
            st.session_state.le_v += 1
            st.rerun()

    if st.button("🚗 Set ALL lane headings from their vehicles", use_container_width=True,
                 disabled=not lanes or not paths,
                 help="Recompute every lane's heading from the through-traffic inside it (real "
                      "start→end bearings). Lanes with no straight-through cars (turn-only) are "
                      "left as-is and listed for you to set by hand."):
        _changed, _skipped = 0, []
        for _l in lanes:
            _h, _n = heading_from_tracks_in_lane(_l, paths)
            if _h is None:
                _skipped.append(str(_l['lane_id']))
            else:
                _l['heading_deg'] = _h
                _changed += 1
        _msg = f"Set {_changed} lane heading(s) from vehicles."
        if _skipped:
            _msg += f" No through-traffic for: {', '.join(_skipped)} — set those by hand."
        st.session_state['le_veh_msg'] = ('warning' if _skipped else 'success', _msg)
        st.session_state.le_v += 1
        st.rerun()

    add, dl, sv, sd = st.columns([1.4, 1, 1, 1.2])
    if add.button("➕ Add lane", use_container_width=True,
                  help="Add a new Eastbound lane box you can then re-point (direction dropdown) and resize."):
        lanes.append(dict(lane_id="Eastbound", xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0,
                          heading_deg=direction_to_heading("Eastbound", _active_north)))
        st.session_state.le_v += 1
        st.rerun()
    txt = json.dumps(lanes_to_geojson(lanes), indent=2) if lanes else ""
    dl.download_button("⬇️ Download", data=txt, file_name="lanes.geojson",
                       mime="application/json", use_container_width=True, disabled=not lanes,
                       help="Download the current lanes as a lanes.geojson file to your computer. "
                            "Does NOT change the active config.")
    if sv.button("💾 Save", use_container_width=True, disabled=not lanes,
                 help="Overwrite config/lanes.geojson with the current lanes. The Wrong-Way "
                      "Detection page picks this up on its next run."):
        os.makedirs(os.path.dirname(LANES_PATH), exist_ok=True)
        with open(LANES_PATH, "w") as f:
            f.write(txt)
        st.success(f"Saved {len(lanes)} lanes → {LANES_PATH}")
    if sd.button("📌 Set as new default", use_container_width=True, disabled=not lanes,
                 help="Snapshot the CURRENT lanes as this dataset's default — what every "
                      "'🔄 Default lanes' reset restores from here on. Mirrors the Geometry "
                      "Editor's set-as-default."):
        os.makedirs(os.path.dirname(_ds.default_lanes_path), exist_ok=True)
        with open(_ds.default_lanes_path, "w") as f:
            f.write(txt)
        st.success(f"Default updated → `{_ds.default_lanes_path}`. "
                   "'🔄 Default lanes' now restores these.")

with right:
    head = st.columns([2.6, 1, 1])
    color_choice = head[0].segmented_control(
        "Color points by", ["Cardinal", "Lane", "Heading"], default="Cardinal",
        help="How the vehicle dots are colored:\n\n"
             "• Cardinal — by travel direction bucket (E/N/W/S); lane boxes are colored to match.\n\n"
             "• Lane — each dot takes the color of the lane box it falls inside (gray if outside every box). "
             "Good for checking which boxes capture which vehicles.\n\n"
             "• Heading — continuous color by exact heading angle.") or "Cardinal"
    color_mode = {"Cardinal": "cardinal", "Lane": "lane", "Heading": "heading"}[color_choice]
    draw_mode = head[1].toggle("✏️ Draw", value=False,
                               help="Flat top-down view where you **lasso/box-draw** a lane polygon. "
                                    "Off = the normal 3D preview.")
    top_down = head[2].toggle("⬇️ Top-down", value=True, disabled=draw_mode,
                              help="On = bird's-eye. Off = oblique 3D. (Draw mode is always top-down.)")

    bgc, hmc, tnc = st.columns(3)
    show_bg = bgc.checkbox("🛰️ Point cloud", value=True, disabled=draw_mode)
    show_hdmap = hmc.checkbox("🗺️ Intersection (HD map)", value=True,
                              help="Overlay the real intersection's road network (the dev-kit HD map) "
                                   "so you can line lanes up with the actual roads.")
    true_cardinals = tnc.checkbox("🧭 True cardinals", value=_has_geo, disabled=not _has_geo,
                                  key="le_true_card", on_change=_reseed_lane_widgets,
                                  help="Colour vehicles/lanes by REAL compass direction (N/E/S/W from the "
                                       "georeference) and show a compass rose, instead of the sensor-frame "
                                       "axes. Also drives the lane direction dropdown/names. Needs a "
                                       "georeference for this sensor."
                                       if _has_geo else "No georeference for this sensor.")
    # Same frame the direction dropdown/names used above, so display stays consistent.
    true_north_deg = _active_north
    bg_xyz = None
    if show_bg and not draw_mode and os.path.isdir(DEFAULT_PCD_BG):
        try:
            bg_xyz = _cached_background(DEFAULT_PCD_BG, 15)
        except Exception as e:
            st.warning(f"Background load failed: {e}")

    hdmap_lanes = None
    if show_hdmap:
        try:
            import geo_reference as geo
            hdmap_lanes = geo.hdmap_lanes_sensor_frame("north" if _le_sensor == "north" else "south", 130.0)
            if not hdmap_lanes:
                st.caption("ℹ️ HD-map overlay needs `map/lane_samples.json` (from the dev-kit's map.zip).")
        except Exception:
            hdmap_lanes = None

    _pts = points if points is not None else np.zeros((0, 3))
    if draw_mode:
        try:
            from geometry_config import get_research_polygon
            _bx0, _by0, _bx1, _by1 = get_research_polygon().bounds
            _m = 5.0
            _xr, _yr = [_bx0 - _m, _bx1 + _m], [_by0 - _m, _by1 + _m]
        except Exception:
            _xr = _yr = None
        dfig = build_draw_figure(_pts, lanes, hdmap_lanes=hdmap_lanes, color_mode=color_mode,
                                 xrange=_xr, yrange=_yr, true_north_deg=true_north_deg)
        ev = st.plotly_chart(dfig, use_container_width=True, key="le_draw", on_select="rerun",
                             config={"scrollZoom": True, "displaylogo": False,
                                     "modeBarButtonsToRemove": ["autoScale2d"]})
        poly = None
        try:
            sel = ev.get("selection") or {}
            lasso, boxsel = sel.get("lasso") or [], sel.get("box") or []
            if lasso:
                poly = simplify_path(lasso[0]["x"], lasso[0]["y"])
            elif boxsel:
                xs, ys = boxsel[0]["x"], boxsel[0]["y"]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        except Exception:
            poly = None
        if poly and len(poly) >= 3:
            st.caption(f"✏️ Drawn shape · {len(poly)} vertices — add it as a lane, or replace an existing one.")
            d1, d2, d3 = st.columns([1.3, 1.6, 1])
            if d1.button("➕ Add as new lane", use_container_width=True):
                lanes.append(dict(lane_id="Eastbound", polygon=poly, n=0,
                                  heading_deg=direction_to_heading("Eastbound", _active_north)))
                st.session_state.le_v += 1
                st.rerun()
            if lanes:
                _ri = d2.selectbox("replace which lane", list(range(len(lanes))),
                                   format_func=lambda j: str(lanes[j]['lane_id']),
                                   key=f"le_repl_{v}", label_visibility="collapsed")
                if d3.button("✏️ Replace", use_container_width=True):
                    lanes[_ri]['polygon'] = poly
                    st.session_state.le_v += 1
                    st.rerun()
        else:
            st.caption("Pick the **Lasso** (or **Box Select**) tool at the top-right of the chart, then "
                       "drag a shape around the lane — the outline becomes the lane polygon.")
    else:
        fig = build_preview(_pts, lanes, color_mode=color_mode, bg_xyz=bg_xyz, top_down=top_down,
                            hdmap_lanes=hdmap_lanes, true_north_deg=true_north_deg)
        fig.update_layout(height=PREVIEW_H)
        st.plotly_chart(fig, use_container_width=True, key="le_preview")
