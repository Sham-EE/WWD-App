import os
import json

import streamlit as st
import numpy as np

from lane_tools import (
    load_tracks, moving_points, auto_lanes,
    lanes_to_geojson, geojson_to_lanes, build_preview, load_pcd_background,
)

st.set_page_config(layout="wide", page_title="Lane Editor")


@st.cache_data(show_spinner="Loading point-cloud background…")
def _cached_background(pcd_dir, n_frames):
    return load_pcd_background(pcd_dir, n_frames=n_frames)


LANES_PATH = "config/lanes.geojson"
DEFAULT_TRACKS = "outputs/object_detection/tracks.csv"
PREVIEW_H = 640

st.session_state.setdefault('le_lanes', [])
st.session_state.setdefault('le_v', 0)  # widget key version (bump to reseed)

st.title("🛣️ Lane Editor")

# ---------------- Top bar: data + template actions ----------------
top = st.columns([3, 1.3, 1.3, 1.4, 1.4])
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

st.divider()

# ---------------- Symmetric split: editor (left) | live preview (right) ----------------
left, right = st.columns(2, gap="large")
lanes = st.session_state.le_lanes
v = st.session_state.le_v

with left:
    st.markdown("##### Lanes")
    box = st.container(height=PREVIEW_H, border=False)
    with box:
        if not lanes:
            st.info("No lanes yet — use ✨ Auto-generate above, or ➕ Add lane below.")
        delete_idx = None
        for i, l in enumerate(lanes):
            with st.container(border=True):
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
        if delete_idx is not None:
            lanes.pop(delete_idx)
            st.session_state.le_v += 1
            st.rerun()

    add, dl, sv = st.columns([1.4, 1, 1])
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

with right:
    head = st.columns([3, 1.4])
    color_choice = head[0].segmented_control(
        "Color", ["Cardinal", "Lane", "Heading"], default="Cardinal",
        label_visibility="collapsed",
        help="How the vehicle dots are colored:\n\n"
             "• Cardinal — by travel direction bucket (E/N/W/S); lane boxes are colored to match.\n\n"
             "• Lane — each dot takes the color of the lane box it falls inside (gray if outside every box). "
             "Good for checking which boxes capture which vehicles.\n\n"
             "• Heading — continuous color by exact heading angle.") or "Cardinal"
    color_mode = {"Cardinal": "cardinal", "Lane": "lane", "Heading": "heading"}[color_choice]
    top_down = head[1].toggle("⬇️ Top-down", value=True,
                              help="On = bird's-eye. Off = oblique 3D.")

    show_bg = st.checkbox("🛰️ Point cloud background", value=False)
    bg_xyz = None
    if show_bg and os.path.isdir("data/point_clouds/cropped/cropped_pcd"):
        try:
            bg_xyz = _cached_background("data/point_clouds/cropped/cropped_pcd", 15)
        except Exception as e:
            st.warning(f"Background load failed: {e}")

    fig = build_preview(points if points is not None else np.zeros((0, 3)),
                        lanes, color_mode=color_mode, bg_xyz=bg_xyz, top_down=top_down)
    fig.update_layout(height=PREVIEW_H)
    st.plotly_chart(fig, use_container_width=True, key="le_preview")
