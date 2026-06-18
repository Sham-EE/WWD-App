import streamlit as st
import numpy as np
import os
import plotly.graph_objects as go
import logging
import glob
import open3d as o3d
import pickle

from bg_filter_core import (
    build_background_model,
    filter_points_with_model,
    sorted_by_frame_index,
)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Background Filtering")
st.title("🔬 Background Filtering")
logging.info("--- Background Filter Page Loaded ---")

# ---------------- Active-dataset paths -----------------
import dataset_manager as dm
_ds = dm.get_active()
st.sidebar.caption(f"📂 Dataset: **{_ds.name}**")
_src_label = st.sidebar.radio("Input cloud", ["Cropped (road)", "Full (uncropped)"],
                              key="pipeline_source", horizontal=True,
                              help="Cropped = road-clipped clouds; Full = raw clouds (research region). "
                                   "Each writes to its own model/filtered/detection folders so you can "
                                   "compare eval metrics. The choice is shared across Filtering / Detection / Evaluation.")
_src = "cropped" if _src_label.startswith("Cropped") else "full"
DEFAULT_MODEL_PATH = _ds.model_path_for(_src)
DEFAULT_PCD = _ds.input_pcd_for(_src)
DEFAULT_GT = _ds.gt_dir
DEFAULT_OUT = _ds.filtered_dir_for(_src)

@st.cache_data(show_spinner="Discovering PCD files...")
def discover_pcd_files(dir_path: str):
    if not os.path.isdir(dir_path): return []
    files = sorted_by_frame_index(glob.glob(os.path.join(dir_path, "*.pcd")))
    logging.info(f"Discovered {len(files)} PCD files in {dir_path}")
    return files

def create_filtered_figure(foreground_pts, original_pts):
    fig = go.Figure()
    if original_pts.size > 0:
        fig.add_trace(go.Scatter3d(x=original_pts[:, 0], y=original_pts[:, 1], z=original_pts[:, 2],
            mode="markers", name="Original", marker=dict(size=1.5, color="#8fa3bd", opacity=0.2)))
    if foreground_pts.size > 0:
        fig.add_trace(go.Scatter3d(x=foreground_pts[:, 0], y=foreground_pts[:, 1], z=foreground_pts[:, 2],
            mode="markers", name="Foreground", marker=dict(size=2.5, color="red", opacity=0.9)))
    # Lock the x/y extent to the road region so toggling Cropped<->Full keeps the
    # SAME zoom (Full no longer auto-fits out to the wider research extent).
    try:
        from geometry_config import get_road_polygon
        minx, miny, maxx, maxy = get_road_polygon().bounds
        xr = dict(range=[minx - 3, maxx + 3]); yr = dict(range=[miny - 3, maxy + 3])
    except Exception:
        xr, yr = {}, {}
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(aspectmode="data", xaxis=xr, yaxis=yr), uirevision="bf_view")
    return fig

# ---------------- Sidebar parameters ----------------
side = st.sidebar
side.header("Input and Model")
model_save_path = side.text_input("Background Model Path", DEFAULT_MODEL_PATH)
use_saved_model = side.checkbox("Load saved model if available", value=True)

config = {
    "pcd_dir": side.text_input("PCD Directory", DEFAULT_PCD),
    # "gt_dir": side.text_input("Ground Truth Directory (for Z-ranges)", DEFAULT_GT), # Hidden per user request
    "build_frames": side.number_input('Build Frames (0=all)', min_value=0, value=0),
}
config['gt_dir'] = DEFAULT_GT # Assign default value without showing UI element

side.header("Ground Removal")
config.update({
    "ground_grid": side.number_input("Ground Grid Size (m)", 0.1, 2.0, 0.5, 0.1),
    "dz_thresh": side.number_input("Ground Z Threshold (m)", 0.1, 1.0, 0.3, 0.05),
})

side.header("Background Model: Voxel Occupancy")
config.update({
    "bg_voxel": side.slider("BG Voxel Size (m)", 0.5, 2.0, 1.0, 0.1),
    "bg_ratio": side.slider("BG Presence Ratio", 0.5, 1.0, 0.98, 0.01),
})

side.header("Background Model: Cluster Persistence")
config.update({
    "cell_size": side.slider('Grid Cell Size (m)', 0.5, 5.0, 1.0, 0.1),
    "cell_ratio": side.slider('Presence Ratio', 0.5, 1.0, 0.9, 0.01),
})

side.header("Clustering (DBSCAN)")
cluster_params = {
    "ds_voxel": side.slider('Downsample Voxel (Build)', 0.05, 0.5, 0.15, 0.01),
    "eps0": side.number_input('eps0', value=0.35),
    "eps_k": side.number_input('eps_k', value=0.008, format="%.4f"),
    "eps_min": side.number_input('eps_min', value=0.35),
    "eps_max": side.number_input('eps_max', value=2.0),
    "min_samples": side.number_input('min_samples', value=16),
}
config['cluster'] = cluster_params

side.header("Pole-like Geometry Filter")
config.update({
    'enable_pole_filter': side.checkbox('Enable Pole Filter', value=True),
    'pole_min_height': side.number_input('Min Height', value=1.5),
    'pole_min_aspect_xy': side.number_input('Min Aspect XY', value=6.0),
    'pole_max_xy_area': side.number_input('Max XY Area', value=1.0),
    'pole_min_linearity': side.number_input('Min Linearity', value=0.75),
    'pole_min_points': side.number_input('Min Points', value=8),
    'pole_max_points': side.number_input('Max Points (pole)', value=80,
        help="A tall, small-footprint cluster is only treated as a pole if it has at most this many "
             "points. Dense objects (trucks/vans) exceed it and are kept. Raise to delete more; "
             "lower to protect more vehicles."),
})

side.header("Misc Filters")
config.update({
    'inward_buffer_m': side.number_input('Road Edge Inward Buffer (m)', value=2.0),
    'coarse_5x5': {'NX': 5, 'NY': 5}, # Hard-coded
})

side.header("Output")
save_filtered = side.checkbox("Save filtered foreground points (PCD)", value=True)
output_dir = side.text_input("Output Directory", DEFAULT_OUT, disabled=not save_filtered)

# ---------------- Load or Build background model ----------------
if "bg_model" not in st.session_state: st.session_state.bg_model = None

if use_saved_model and st.session_state.bg_model is None and os.path.exists(model_save_path):
    try:
        with open(model_save_path, "rb") as fp:
            st.session_state.bg_model = pickle.load(fp)
        st.success(f"Loaded saved background model from: {model_save_path}")
    except Exception as e:
        st.warning(f"Could not load saved model: {e}")
        st.session_state.bg_model = None

if side.button("Build Background Model", use_container_width=True, type="primary"):
    pcd_files = discover_pcd_files(config["pcd_dir"])
    if not pcd_files:
        st.error("No PCD files found!")
    else:
        progress_bar = st.progress(0.0, text="Starting model build...")
        def progress_callback(p, txt):
            progress_bar.progress(p, text=txt)
        with st.spinner("Building background model..."):
            model = build_background_model(config, pcd_files, config['gt_dir'], progress_callback)
            st.session_state.bg_model = model
        progress_bar.empty()
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, "wb") as fp:
            pickle.dump(st.session_state.bg_model, fp)
        st.success(f"Background model built and saved to: {model_save_path}")

        if save_filtered:
            os.makedirs(output_dir, exist_ok=True)
            save_bar = st.progress(0.0, text="Saving filtered clouds...")
            buildN = len(pcd_files) if config['build_frames'] == 0 else min(config['build_frames'], len(pcd_files))
            for idx, pcd_path in enumerate(pcd_files[:buildN]):
                pts = np.asarray(o3d.io.read_point_cloud(pcd_path).points)
                fg, _ = filter_points_with_model(pts, st.session_state.bg_model, config)
                pc_out = o3d.geometry.PointCloud()
                if fg.size > 0: pc_out.points = o3d.utility.Vector3dVector(fg)
                o3d.io.write_point_cloud(os.path.join(output_dir, os.path.basename(pcd_path)), pc_out, write_ascii=True)
                save_bar.progress((idx + 1) / buildN, text=f"Saving {idx+1}/{buildN}")
            save_bar.empty()
            st.success(f"Saved all filtered point clouds to: {output_dir}")

# ---------------- Visualization ----------------
@st.cache_data(show_spinner=False, max_entries=128)
def _load_raw(path):
    return np.asarray(o3d.io.read_point_cloud(path).points)


if st.session_state.bg_model:
    st.divider()
    st.subheader("Filtered Point Cloud Viewer")

    pcd_files = discover_pcd_files(config["pcd_dir"])
    if pcd_files:
        n_bf = len(pcd_files)
        st.session_state.setdefault("bf_frame", 0)

        @st.fragment
        def _bf_viewer():
            st.session_state.bf_frame = max(0, min(st.session_state.bf_frame, n_bf - 1))
            nav = st.columns([1, 1, 1, 1, 1.3, 3])
            if nav[0].button("⏮ First", use_container_width=True):
                st.session_state.bf_frame = 0
            if nav[1].button("◀ Prev", use_container_width=True):
                st.session_state.bf_frame = max(0, st.session_state.bf_frame - 1)
            if nav[2].button("Next ▶", use_container_width=True):
                st.session_state.bf_frame = min(n_bf - 1, st.session_state.bf_frame + 1)
            if nav[3].button("Last ⏭", use_container_width=True):
                st.session_state.bf_frame = n_bf - 1
            playing = nav[4].toggle("▶ Play", value=False)
            play_delay = nav[5].slider("Play delay (s)", 0.0, 1.0, 0.15, 0.05)
            i = st.slider("Frame", 0, max(n_bf - 1, 1), st.session_state.bf_frame)
            st.session_state.bf_frame = i

            pts = _load_raw(pcd_files[i])
            fg, _ = filter_points_with_model(pts, st.session_state.bg_model, config)
            with st.container(height=560):
                st.plotly_chart(create_filtered_figure(fg, pts), use_container_width=True, key="bf_fig")
            st.caption(f"{os.path.basename(pcd_files[i])} · frame {i+1}/{n_bf} · "
                       f"{len(fg)} foreground / {len(pts)} points")

            if playing and i < n_bf - 1:
                import time
                time.sleep(float(play_delay))
                st.session_state.bf_frame = i + 1
                st.rerun(scope="fragment")

        _bf_viewer()
else:
    st.info("Adjust parameters and click 'Build Background Model' to begin.")
