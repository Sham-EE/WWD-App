import os
import json

import streamlit as st
import numpy as np

from lane_tools import (
    load_tracks, moving_points, auto_lanes, snap_to_cardinal,
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

# ---------------- Top bar: data + template actions ----------------
top = st.columns([2.3, 1.1, 1.1, 1.3, 1.3, 1.3])
uploaded = top[0].file_uploader("tracks.csv", type="csv", label_visibility="collapsed")
src = uploaded if uploaded is not None else (DEFAULT_TRACKS if os.path.exists(DEFAULT_TRACKS) else None)
min_speed = top[1].slider("Min speed", 0.0, 10.0, 1.0, 0.5, help="m/s; exclude slow/parked points.")
k = top[2].number_input("# lanes", 1, 8, 4, help="Lanes to auto-detect.")

points = None
if src is not None:
    try:
        points = moving_points(load_tracks(src), min_speed=min_speed)
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
    box = st.container(height=PREVIEW_H, border=False)
    with box:
        if not lanes:
            st.info("No lanes yet — use ✨ Auto-generate above, or ➕ Add lane below.")
        delete_idx = None
        reset_action = None  # (idx, default_lane) — applied after the loop
        snap_idx = None      # idx — snap this lane's heading to a cardinal, after the loop
        for i, l in enumerate(lanes):
            dft = _def_lanes_by_id.get(str(l['lane_id']))
            # NOTE: the heading is deliberately NOT in the expander title — putting a
            # value that changes on every edit there changes the expander's identity,
            # so Streamlit collapses it on each +/click. Title is the (stable) lane id.
            with st.expander(f"🛣️ {l['lane_id']}", expanded=False):
                a = st.columns([3, 2, 0.8])
                l['lane_id'] = a[0].text_input("Lane", value=str(l['lane_id']), key=f"id_{i}_{v}")
                l['heading_deg'] = a[1].number_input("Heading°", value=float(l['heading_deg']),
                                                     step=1.0, format="%.1f", key=f"hd_{i}_{v}",
                                                     help="0=+X, 90=+Y, 180=-X, -90=-Y")
                a[2].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                if a[2].button("🗑", key=f"del_{i}_{v}", help="Delete lane"):
                    delete_idx = i
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
                bc = st.columns(2)
                if bc[0].button("🧭 Snap to cardinal", key=f"snap_{i}_{v}", use_container_width=True,
                                help="Point this lane's arrow straight along the nearest cardinal "
                                     "direction — E 0°, N 90°, W 180°, S −90° (ignores vehicle scatter)."):
                    snap_idx = i
                if bc[1].button("↺ Reset to default", key=f"rst_{i}_{v}",
                                use_container_width=True, disabled=dft is None,
                                help=("Restore just this lane's box + heading from "
                                      "config/defaults/lanes.geojson." if dft is not None
                                      else "No default with this lane name to restore from.")):
                    reset_action = (i, dft)
        if delete_idx is not None:
            lanes.pop(delete_idx)
            st.session_state.le_v += 1
            st.rerun()
        if snap_idx is not None:
            lanes[snap_idx]['heading_deg'] = snap_to_cardinal(lanes[snap_idx]['heading_deg'])
            st.session_state.le_v += 1
            st.rerun()
        if reset_action is not None:
            idx, dft = reset_action
            lanes[idx].pop('polygon', None)   # defaults are boxes → back to a box lane
            lanes[idx].update(xmin=dft['xmin'], xmax=dft['xmax'], ymin=dft['ymin'],
                              ymax=dft['ymax'], heading_deg=dft['heading_deg'])
            st.session_state.le_v += 1
            st.rerun()

    if st.button("🧭 Snap ALL headings to cardinal", use_container_width=True, disabled=not lanes,
                 help="Point every lane's arrow straight along its nearest cardinal direction (E/N/W/S)."):
        for _l in lanes:
            _l['heading_deg'] = snap_to_cardinal(_l['heading_deg'])
        st.session_state.le_v += 1
        st.rerun()

    add, dl, sv, sd = st.columns([1.4, 1, 1, 1.2])
    if add.button("➕ Add lane", use_container_width=True):
        lanes.append(dict(lane_id=f"lane_{len(lanes)+1}", xmin=-5.0, xmax=5.0,
                          ymin=-5.0, ymax=5.0, heading_deg=0.0))
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

    bgc, hmc = st.columns(2)
    show_bg = bgc.checkbox("🛰️ Point cloud background", value=True, disabled=draw_mode)
    show_hdmap = hmc.checkbox("🗺️ Intersection (HD map)", value=True,
                              help="Overlay the real intersection's road network (the dev-kit HD map) "
                                   "so you can line lanes up with the actual roads.")
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
                                 xrange=_xr, yrange=_yr)
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
                lanes.append(dict(lane_id=f"lane_{len(lanes)+1}", polygon=poly, heading_deg=0.0, n=0))
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
                            hdmap_lanes=hdmap_lanes)
        fig.update_layout(height=PREVIEW_H)
        st.plotly_chart(fig, use_container_width=True, key="le_preview")
