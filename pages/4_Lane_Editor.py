import os
import json

import streamlit as st
import numpy as np
import pandas as pd

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
COLS = ["lane_id", "xmin", "xmax", "ymin", "ymax", "heading_deg"]

st.session_state.setdefault('le_lanes', [])
st.session_state.setdefault('le_v', 0)  # data-editor key version (bump to reseed)
st.session_state.setdefault('le_zoom', 3.4)  # camera distance; bigger = wider view

ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 1.6, 7.0, 0.4

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
    seed = pd.DataFrame(
        [{c: l.get(c) for c in COLS} for l in st.session_state.le_lanes],
        columns=COLS,
    )
    edited = st.data_editor(
        seed, num_rows="dynamic", use_container_width=True, height=430,
        key=f"lane_table_{st.session_state.le_v}",
        column_config={
            "lane_id": st.column_config.TextColumn("Lane", width="small"),
            "xmin": st.column_config.NumberColumn("X min", step=0.5, format="%.1f"),
            "xmax": st.column_config.NumberColumn("X max", step=0.5, format="%.1f"),
            "ymin": st.column_config.NumberColumn("Y min", step=0.5, format="%.1f"),
            "ymax": st.column_config.NumberColumn("Y max", step=0.5, format="%.1f"),
            "heading_deg": st.column_config.NumberColumn("Heading°", step=1.0, format="%.1f",
                                                         help="0=+X, 90=+Y, 180=-X, -90=-Y"),
        },
    )
    # Parse the edited table into lane dicts (skip incomplete rows).
    lanes = []
    for i, r in edited.iterrows():
        if any(pd.isna(r[c]) for c in ("xmin", "xmax", "ymin", "ymax")):
            continue
        lanes.append(dict(
            lane_id=(str(r["lane_id"]) if not pd.isna(r["lane_id"]) else f"lane_{len(lanes)+1}"),
            xmin=float(r["xmin"]), xmax=float(r["xmax"]),
            ymin=float(r["ymin"]), ymax=float(r["ymax"]),
            heading_deg=float(r["heading_deg"]) if not pd.isna(r["heading_deg"]) else 0.0,
        ))

    st.caption("➕ add a row at the bottom · select a row's checkbox + ⌫ to delete.")
    ex1, ex2 = st.columns(2)
    if lanes:
        txt = json.dumps(lanes_to_geojson(lanes), indent=2)
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

    zc = st.columns([1, 1, 1, 3])
    if zc[0].button("🔍➖ Wider", use_container_width=True, help="Zoom out (see more)"):
        st.session_state.le_zoom = round(min(ZOOM_MAX, st.session_state.le_zoom + ZOOM_STEP), 2)
        st.rerun()
    if zc[1].button("🔍➕ Closer", use_container_width=True, help="Zoom in"):
        st.session_state.le_zoom = round(max(ZOOM_MIN, st.session_state.le_zoom - ZOOM_STEP), 2)
        st.rerun()
    if zc[2].button("↺ Fit", use_container_width=True, help="Reset to the default wide view"):
        st.session_state.le_zoom = 3.4
        st.rerun()
    show_bg = zc[3].checkbox("🛰️ Point cloud background", value=False)

    bg_xyz = None
    if show_bg:
        bg_dir = "data/point_clouds/cropped/cropped_pcd"
        if os.path.isdir(bg_dir):
            try:
                bg_xyz = _cached_background(bg_dir, 15)
            except Exception as e:
                st.warning(f"Background load failed: {e}")

    fig = build_preview(points if points is not None else np.zeros((0, 3)),
                        lanes, color_mode=color_mode, bg_xyz=bg_xyz, top_down=top_down,
                        cam_dist=st.session_state.le_zoom)
    # Stable key + uirevision so the camera/zoom survives table edits.
    st.plotly_chart(fig, use_container_width=True, key="le_preview")
