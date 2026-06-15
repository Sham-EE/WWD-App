import os
import json

import streamlit as st

from lane_tools import (
    load_tracks, moving_points, auto_lanes,
    lanes_to_geojson, geojson_to_lanes, build_preview, load_pcd_background,
)


@st.cache_data(show_spinner="Loading point-cloud background…")
def _cached_background(pcd_dir, n_frames):
    return load_pcd_background(pcd_dir, n_frames=n_frames)

st.set_page_config(layout="wide", page_title="Lane Editor")
st.title("🛣️ Lane Editor")
st.markdown(
    "Build the wrong-way lane geometry from data. Load a `tracks.csv`, auto-generate "
    "a lane template from the observed traffic, **adjust each box and heading** with the "
    "controls (watching the live top-down preview), then **export** to `config/lanes.geojson`."
)

LANES_PATH = "config/lanes.geojson"
DEFAULT_TRACKS = "outputs/object_detection/tracks.csv"

# ---- session state ----
st.session_state.setdefault('le_lanes', [])
st.session_state.setdefault('le_v', 0)          # key version: bump to reset widgets
st.session_state.setdefault('le_points', None)

# =================== 1. Load tracks ===================
st.subheader("1. Load trajectory data")
c1, c2 = st.columns([2, 1])
uploaded = c1.file_uploader("Upload tracks.csv", type="csv")
src = None
if uploaded is not None:
    src = uploaded
elif os.path.exists(DEFAULT_TRACKS):
    if c2.checkbox(f"Use {DEFAULT_TRACKS}", value=True):
        src = DEFAULT_TRACKS

min_speed = st.slider("Min speed to include a point (m/s)", 0.0, 10.0, 1.0, 0.5,
                      help="Slow/parked points have unreliable heading; exclude them.")

points = None
if src is not None:
    try:
        df = load_tracks(src)
        points = moving_points(df, min_speed=min_speed)
        st.session_state.le_points = points
        st.success(f"Loaded {len(df)} rows → {len(points)} moving points with valid heading.")
    except Exception as e:
        st.error(f"Could not read tracks: {e}")
points = st.session_state.le_points

# =================== 2. Auto-template ===================
st.subheader("2. Auto-generate a lane template")
a1, a2, a3 = st.columns([1, 1, 2])
k = a1.number_input("Number of lane directions", 1, 8, 4)
buf = a2.number_input("Box buffer (m)", 0.0, 10.0, 2.0, 0.5)
if a3.button("✨ Auto-generate template from data", use_container_width=True,
             disabled=points is None or len(points) == 0):
    st.session_state.le_lanes = auto_lanes(points, k=int(k), buffer_m=float(buf))
    st.session_state.le_v += 1
    st.rerun()

le1, le2 = st.columns(2)
if le1.button("➕ Add empty lane", use_container_width=True):
    st.session_state.le_lanes.append(dict(lane_id=f"lane_{len(st.session_state.le_lanes)+1}",
                                           xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0,
                                           heading_deg=0.0, n=0))
    st.session_state.le_v += 1
    st.rerun()
if le2.button("📂 Load current config/lanes.geojson to edit", use_container_width=True,
              disabled=not os.path.exists(LANES_PATH)):
    with open(LANES_PATH) as f:
        st.session_state.le_lanes = geojson_to_lanes(json.load(f))
    st.session_state.le_v += 1
    st.rerun()

# =================== 3. Edit lanes ===================
lanes = st.session_state.le_lanes
v = st.session_state.le_v

if not lanes:
    st.info("No lanes yet — auto-generate a template or add an empty lane above.")
else:
    st.subheader("3. Adjust lanes")
    delete_idx = None
    for i, l in enumerate(lanes):
        with st.expander(f"🛣️ {l['lane_id']}  —  heading {float(l['heading_deg']):.0f}°"
                         + (f"  ({l['n']} pts)" if l.get('n') else ""), expanded=True):
            r1 = st.columns([2, 2, 1])
            l['lane_id'] = r1[0].text_input("Lane ID", value=str(l['lane_id']), key=f"id_{i}_{v}")
            l['heading_deg'] = r1[1].number_input("Heading (deg)", value=float(l['heading_deg']),
                                                  step=1.0, format="%.1f", key=f"hd_{i}_{v}",
                                                  help="Expected travel dir: 0=+X, 90=+Y, 180=-X, -90=-Y.")
            if r1[2].button("🗑 Delete", key=f"del_{i}_{v}", use_container_width=True):
                delete_idx = i
            r2 = st.columns(4)
            l['xmin'] = r2[0].number_input("X min", value=float(l['xmin']), step=0.5, key=f"xmin_{i}_{v}")
            l['xmax'] = r2[1].number_input("X max", value=float(l['xmax']), step=0.5, key=f"xmax_{i}_{v}")
            l['ymin'] = r2[2].number_input("Y min", value=float(l['ymin']), step=0.5, key=f"ymin_{i}_{v}")
            l['ymax'] = r2[3].number_input("Y max", value=float(l['ymax']), step=0.5, key=f"ymax_{i}_{v}")
    if delete_idx is not None:
        lanes.pop(delete_idx)
        st.session_state.le_v += 1
        st.rerun()

# =================== 4. Preview ===================
st.subheader("4. Preview")
p1, p2 = st.columns(2)
color_choice = p1.radio(
    "Color vehicle points by",
    ["Cardinal direction", "Lane membership", "Heading (continuous)"], horizontal=True,
    help="Cardinal = direction buckets (boxes colored to match). "
         "Lane membership = each dot takes its box's color, gray if outside every box. "
         "Heading = continuous HSV.")
color_mode = {"Cardinal direction": "cardinal", "Lane membership": "lane",
              "Heading (continuous)": "heading"}[color_choice]
top_down = p2.toggle("⬇️ Top-down view", value=True,
                     help="On = bird's-eye for box fitting. Off = oblique 3D. "
                          "Drag to rotate, scroll to zoom, right-drag to pan.")

b1, b2, b3 = st.columns([1, 2, 2])
show_bg = b1.checkbox("🛰️ Point cloud background", value=False,
                      help="Overlay an accumulated road footprint to align boxes to the road.")
bg_xyz = None
if show_bg:
    bg_dir = b2.text_input("PCD directory", value="data/point_clouds/cropped/cropped_pcd")
    n_bg = b3.slider("Frames to accumulate", 1, 60, 15)
    if os.path.isdir(bg_dir):
        try:
            bg_xyz = _cached_background(bg_dir, n_bg)
        except Exception as e:
            st.warning(f"Could not load point cloud background: {e}")
    else:
        st.warning(f"PCD directory not found: {bg_dir}")

st.caption("Arrows show each lane's expected direction; box and dot colors match. A lane is "
           "correct when its arrow agrees with the dots inside its box.")
import numpy as _np
fig = build_preview(points if points is not None else _np.zeros((0, 3)),
                    lanes, color_mode=color_mode, bg_xyz=bg_xyz, top_down=top_down)
st.plotly_chart(fig, use_container_width=True)

# =================== 5. Export ===================
st.subheader("5. Export")
if lanes:
    gj = lanes_to_geojson(lanes)
    txt = json.dumps(gj, indent=2)
    e1, e2 = st.columns(2)
    e1.download_button("⬇️ Download lanes.geojson", data=txt,
                       file_name="lanes.geojson", mime="application/json",
                       use_container_width=True)
    if e2.button(f"💾 Save to {LANES_PATH} (overwrites active config)", use_container_width=True):
        os.makedirs(os.path.dirname(LANES_PATH), exist_ok=True)
        with open(LANES_PATH, "w") as f:
            f.write(txt)
        st.success(f"Saved {len(lanes)} lanes to {LANES_PATH}. The WWD page will use it on next run.")
    with st.expander("Preview GeoJSON"):
        st.code(txt, language="json")
