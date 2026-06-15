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

st.session_state.setdefault('le_lanes', [])
st.session_state.setdefault('le_v', 0)  # data-editor key version (bump to reseed)

st.title("🛣️ Lane Editor")

# ---------------- Top bar: data + template actions (compact) ----------------
top = st.columns([3, 1.4, 1.6, 1.4, 1.4])
uploaded = top[0].file_uploader("tracks.csv", type="csv", label_visibility="collapsed")
src = uploaded if uploaded is not None else (DEFAULT_TRACKS if os.path.exists(DEFAULT_TRACKS) else None)
min_speed = top[1].slider("Min speed", 0.0, 10.0, 1.0, 0.5, help="m/s; exclude slow/parked points.")
k = top[2].number_input("# directions", 1, 8, 4, help="Lanes to auto-detect.")

points = None
if src is not None:
    try:
        df = load_tracks(src)
        points = moving_points(df, min_speed=min_speed)
    except Exception as e:
        st.error(f"Could not read tracks: {e}")

if top[3].button("✨ Auto-generate", use_container_width=True,
                 disabled=points is None or len(points) == 0):
    st.session_state.le_lanes = auto_lanes(points, k=int(k), buffer_m=2.0)
    st.session_state.le_v += 1
    st.rerun()
if top[4].button("📂 Load saved", use_container_width=True, disabled=not os.path.exists(LANES_PATH)):
    with open(LANES_PATH) as f:
        st.session_state.le_lanes = geojson_to_lanes(json.load(f))
    st.session_state.le_v += 1
    st.rerun()

cap = f"{len(points)} moving points" if points is not None else "No tracks loaded"
st.caption(f"{cap} · edit the table on the left and watch the preview on the right — no scrolling.")

# ---------------- Side-by-side: editor table (left) | live preview (right) ----------------
left, right = st.columns([5, 7], gap="medium")

with left:
    lanes = st.session_state.le_lanes
    v = st.session_state.le_v
    W = [1.5, 1, 1, 1, 1, 1, 0.5]  # column widths: id, xmin, xmax, ymin, ymax, hdg, del

    if not lanes:
        st.info("No lanes yet — use ✨ Auto-generate above or ➕ Add lane below.")
    else:
        head = st.columns(W)
        for col, lab in zip(head, ["Lane", "X min", "X max", "Y min", "Y max", "Hdg°", ""]):
            col.markdown(f"<div style='font-size:0.75rem;color:#888'>{lab}</div>",
                         unsafe_allow_html=True)
        delete_idx = None
        for i, l in enumerate(lanes):
            r = st.columns(W)
            l['lane_id'] = r[0].text_input("id", value=str(l['lane_id']),
                                           key=f"id_{i}_{v}", label_visibility="collapsed")
            # step=1.0 -> the +/- steppers increment/decrement each value by one.
            l['xmin'] = r[1].number_input("xmin", value=float(l['xmin']), step=1.0, format="%.1f",
                                          key=f"xmin_{i}_{v}", label_visibility="collapsed")
            l['xmax'] = r[2].number_input("xmax", value=float(l['xmax']), step=1.0, format="%.1f",
                                          key=f"xmax_{i}_{v}", label_visibility="collapsed")
            l['ymin'] = r[3].number_input("ymin", value=float(l['ymin']), step=1.0, format="%.1f",
                                          key=f"ymin_{i}_{v}", label_visibility="collapsed")
            l['ymax'] = r[4].number_input("ymax", value=float(l['ymax']), step=1.0, format="%.1f",
                                          key=f"ymax_{i}_{v}", label_visibility="collapsed")
            l['heading_deg'] = r[5].number_input("hdg", value=float(l['heading_deg']), step=1.0,
                                                 format="%.1f", key=f"hd_{i}_{v}",
                                                 label_visibility="collapsed")
            if r[6].button("🗑", key=f"del_{i}_{v}", help="Delete lane"):
                delete_idx = i
        if delete_idx is not None:
            lanes.pop(delete_idx)
            st.session_state.le_v += 1
            st.rerun()

    if st.button("➕ Add lane", use_container_width=True):
        lanes.append(dict(lane_id=f"lane_{len(lanes)+1}", xmin=-5.0, xmax=5.0,
                          ymin=-5.0, ymax=5.0, heading_deg=0.0))
        st.session_state.le_v += 1
        st.rerun()

    if lanes:
        txt = json.dumps(lanes_to_geojson(lanes), indent=2)
        ex1, ex2 = st.columns(2)
        ex1.download_button("⬇️ Download", data=txt, file_name="lanes.geojson",
                            mime="application/json", use_container_width=True)
        if ex2.button("💾 Save to config", use_container_width=True):
            os.makedirs(os.path.dirname(LANES_PATH), exist_ok=True)
            with open(LANES_PATH, "w") as f:
                f.write(txt)
            st.success(f"Saved {len(lanes)} lanes → {LANES_PATH}")

with right:
    rc = st.columns([3, 2])
    color_choice = rc[0].radio(
        "Color points by", ["Cardinal direction", "Lane membership", "Heading (continuous)"],
        horizontal=True, label_visibility="collapsed")
    color_mode = {"Cardinal direction": "cardinal", "Lane membership": "lane",
                  "Heading (continuous)": "heading"}[color_choice]
    top_down = rc[1].toggle("⬇️ Top-down", value=True,
                            help="On = bird's-eye. Off = oblique 3D (drag-rotate, scroll-zoom, right-drag pan).")

    show_bg = st.checkbox("🛰️ Point cloud background", value=False)
    bg_xyz = None
    if show_bg:
        bg_dir = "data/point_clouds/cropped/cropped_pcd"
        if os.path.isdir(bg_dir):
            try:
                bg_xyz = _cached_background(bg_dir, 15)
            except Exception as e:
                st.warning(f"Background load failed: {e}")

    fig = build_preview(points if points is not None else np.zeros((0, 3)),
                        lanes, color_mode=color_mode, bg_xyz=bg_xyz, top_down=top_down)
    # Stable key + uirevision so the camera/zoom survives table edits.
    st.plotly_chart(fig, use_container_width=True, key="le_preview")
