import os
import json

import streamlit as st
import numpy as np

from lane_tools import (
    load_tracks, moving_points, auto_lanes,
    lanes_to_geojson, geojson_to_lanes, build_preview, load_pcd_background,
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
        for i, l in enumerate(lanes):
            dft = _def_lanes_by_id.get(str(l['lane_id']))
            with st.expander(f"🛣️ {l['lane_id']}  ·  {float(l['heading_deg']):.0f}°", expanded=False):
                a = st.columns([3, 2, 0.8])
                l['lane_id'] = a[0].text_input("Lane", value=str(l['lane_id']), key=f"id_{i}_{v}")
                l['heading_deg'] = a[1].number_input("Heading°", value=float(l['heading_deg']),
                                                     step=1.0, format="%.1f", key=f"hd_{i}_{v}",
                                                     help="0=+X, 90=+Y, 180=-X, -90=-Y")
                a[2].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                if a[2].button("🗑", key=f"del_{i}_{v}", help="Delete lane"):
                    delete_idx = i
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
                if st.button("↺ Reset this lane to default", key=f"rst_{i}_{v}",
                             use_container_width=True, disabled=dft is None,
                             help=("Restore just this lane's box + heading from "
                                   "config/defaults/lanes.geojson." if dft is not None
                                   else "No default with this lane name to restore from.")):
                    reset_action = (i, dft)
        if delete_idx is not None:
            lanes.pop(delete_idx)
            st.session_state.le_v += 1
            st.rerun()
        if reset_action is not None:
            idx, dft = reset_action
            lanes[idx].update(xmin=dft['xmin'], xmax=dft['xmax'], ymin=dft['ymin'],
                              ymax=dft['ymax'], heading_deg=dft['heading_deg'])
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
    head = st.columns([3, 1.4])
    color_choice = head[0].segmented_control(
        "Color points by", ["Cardinal", "Lane", "Heading"], default="Cardinal",
        help="How the vehicle dots are colored:\n\n"
             "• Cardinal — by travel direction bucket (E/N/W/S); lane boxes are colored to match.\n\n"
             "• Lane — each dot takes the color of the lane box it falls inside (gray if outside every box). "
             "Good for checking which boxes capture which vehicles.\n\n"
             "• Heading — continuous color by exact heading angle.") or "Cardinal"
    color_mode = {"Cardinal": "cardinal", "Lane": "lane", "Heading": "heading"}[color_choice]
    top_down = head[1].toggle("⬇️ Top-down", value=True,
                              help="On = bird's-eye. Off = oblique 3D.")

    show_bg = st.checkbox("🛰️ Point cloud background", value=True)
    bg_xyz = None
    if show_bg and os.path.isdir(DEFAULT_PCD_BG):
        try:
            bg_xyz = _cached_background(DEFAULT_PCD_BG, 15)
        except Exception as e:
            st.warning(f"Background load failed: {e}")

    fig = build_preview(points if points is not None else np.zeros((0, 3)),
                        lanes, color_mode=color_mode, bg_xyz=bg_xyz, top_down=top_down)
    fig.update_layout(height=PREVIEW_H)
    st.plotly_chart(fig, use_container_width=True, key="le_preview")
