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

def _frame_key(path):
    """Leading <timestamp1>_<timestamp2> token shared by a PCD and its GT label."""
    return "_".join(os.path.basename(path).split("_")[:2])

@st.cache_data(show_spinner=False)
def discover_gt_index(gt_dir: str):
    """Map frame key -> GT label .json so boxes can be matched to each cloud frame."""
    idx = {}
    if gt_dir and os.path.isdir(gt_dir):
        for f in glob.glob(os.path.join(gt_dir, "*.json")):
            idx[_frame_key(f)] = f
    return idx

def create_filtered_figure(foreground_pts, original_pts, margin=12.0, zoom=1.25,
                           show_road=False, road_dashed=False, show_roi=False,
                           show_excl=False, gt_objs=None):
    fig = go.Figure()
    if original_pts.size > 0:
        fig.add_trace(go.Scatter3d(x=original_pts[:, 0], y=original_pts[:, 1], z=original_pts[:, 2],
            mode="markers", name="Original", marker=dict(size=1.5, color="#8fa3bd", opacity=0.2)))
    if foreground_pts.size > 0:
        fig.add_trace(go.Scatter3d(x=foreground_pts[:, 0], y=foreground_pts[:, 1], z=foreground_pts[:, 2],
            mode="markers", name="Foreground", marker=dict(size=2.5, color="red", opacity=0.9)))
    # 3D zoom is the CAMERA distance (eye). No uirevision so the camera below applies
    # on every render; smaller eye = more zoomed in. The slider persists across frames.
    poly = None
    try:
        from geometry_config import get_road_polygon
        poly = get_road_polygon()
        minx, miny, maxx, maxy = poly.bounds
        m = float(margin)
        xr = dict(range=[minx - m, maxx + m]); yr = dict(range=[miny - m, maxy + m])
    except Exception:
        xr, yr = {}, {}

    # Optional geometry overlays (off by default). Each region affects filtering:
    #   road = edge-band removal, ROI = bounds processing, exclusion = always dropped.
    z_floor = float(original_pts[:, 2].min()) if original_pts.size else -8.0

    def _outline(geom, color, dash, name, show_legend=True):
        if geom is None:
            return
        geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for j, g in enumerate(geoms):
            gx, gy = g.exterior.xy
            fig.add_trace(go.Scatter3d(x=list(gx), y=list(gy), z=[z_floor] * len(gx),
                mode="lines", line=dict(color=color, width=4, dash=dash), name=name,
                legendgroup=name, showlegend=(show_legend and j == 0), hoverinfo="skip"))

    if show_road:
        # Dashed on the full cloud = reference only (the full cloud isn't cropped);
        # solid on the cropped cloud = the actual crop boundary.
        _outline(poly, "limegreen", "dash" if road_dashed else "solid",
                 "Road outline (uncropped, ref)" if road_dashed else "Road outline")
    if show_roi:
        try:
            from geometry_config import get_research_polygon
            _outline(get_research_polygon(), "#17becf", "dot", "ROI (research region)")
        except Exception:
            pass
    if show_excl:
        try:
            from geometry_config import get_fg_exclusion_rects
            for k, r in enumerate(get_fg_exclusion_rects() or []):
                _outline(r, "#ff5fec", "dash", "Exclusion zones", show_legend=(k == 0))
        except Exception:
            pass

    if gt_objs:
        import label_projection as lp
        import lidar_viewer as lv
        # Category-coloured boxes + TYPE_id text labels, matching the Visualizer's
        # 3D LiDAR view (same _box_edge_groups / _color_for / _label_text helpers).
        for col, (xs, ys, zs) in lv._box_edge_groups(gt_objs, "by_category").items():
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=col, width=3), hoverinfo="skip", showlegend=False))
        lx, ly, lz, lt, lcol = [], [], [], [], []
        for o in gt_objs:
            top = lp.cuboid_corners(o["val"])[:4].mean(axis=0)   # top-face centre
            lx.append(top[0]); ly.append(top[1]); lz.append(top[2] + 0.4)
            lt.append(lp._label_text(o)); lcol.append(lv._hex(lp._color_for(o, "by_category")))
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="text", text=lt,
            textfont=dict(size=11, color=lcol), hoverinfo="skip", showlegend=False))

    eye = float(zoom)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True,
                      scene=dict(aspectmode="data", xaxis=xr, yaxis=yr,
                                 zaxis=dict(range=[-12.0, 1.0]),
                                 camera=dict(eye=dict(x=eye, y=eye, z=eye))))
    return fig


def foreground_quality(fg_pts, original_pts, gt_objs, min_pts=10):
    """How well the filter preserved real objects, vs the GT boxes for this frame.
    A *proxy* for downstream detectability, not the detection F1.

    Counts foreground (and original) points inside each GT box footprint:
      - covered / scanned : objects with >= min_pts foreground pts, out of objects the
        LiDAR actually hit (>=1 original pt) — i.e. of what's visible, what survived.
      - recall            : foreground pts on objects / original pts on objects.
      - off_object        : foreground pts outside every box (clutter / false-fg proxy).
    """
    import numpy as np
    from matplotlib.path import Path
    import dataset_prep as dp
    fg_xy = fg_pts[:, :2] if (fg_pts is not None and len(fg_pts)) else np.zeros((0, 2))
    or_xy = original_pts[:, :2] if (original_pts is not None and len(original_pts)) else np.zeros((0, 2))
    fg_on = np.zeros(len(fg_xy), dtype=bool)
    or_on = np.zeros(len(or_xy), dtype=bool)
    scanned = covered = 0
    for o in gt_objs or []:
        path = Path(dp._box_footprint(o["val"]))
        fin = path.contains_points(fg_xy) if len(fg_xy) else np.zeros(0, dtype=bool)
        oin = path.contains_points(or_xy) if len(or_xy) else np.zeros(0, dtype=bool)
        fc = int(fin.sum()); oc = int(oin.sum())
        if oc >= 1:                       # object actually hit by the LiDAR
            scanned += 1
            if fc >= min_pts:
                covered += 1
        if len(fg_xy): fg_on |= fin
        if len(or_xy): or_on |= oin
    on_fg, on_or = int(fg_on.sum()), int(or_on.sum())
    return {
        "scanned": scanned, "covered": covered, "min_pts": int(min_pts),
        "recall": (on_fg / on_or) if on_or else None,
        "off_object": int(len(fg_xy) - on_fg), "total_fg": int(len(fg_xy)),
    }

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
    gt_index = discover_gt_index(config["gt_dir"])
    has_gt = bool(gt_index)
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
            # Per-source default zoom (lower = more zoomed in). Edit these two numbers
            # to set the zoom each source loads at. Each source has its own slider/key,
            # so cropped and full remember their own zoom independently.
            ZOOM_DEFAULTS = {"cropped": 0.7, "full": 1.25}
            zoom = st.slider("3D zoom (lower = closer)", 0.6, 2.0,
                             ZOOM_DEFAULTS.get(_src, 1.25), 0.05, key=f"bf_zoom_{_src}")

            # Geometry overlays — each region actually affects filtering:
            geo = st.columns(3)
            road_on = geo[0].toggle("🛣️ Road outline", value=(_src == "cropped"),
                                    key=f"bf_road_{_src}",
                                    help="Road polygon boundary. Solid on cropped (the actual crop); "
                                         "dashed on full (reference only — shows what's outside it).")
            roi_on = geo[1].toggle("🔵 ROI (research region)", value=False, key="bf_roi",
                                   help="Bounds what the filter processes — points outside are never kept.")
            excl_on = geo[2].toggle("🔴 Exclusion zones", value=False, key="bf_excl",
                                    help="Foreground-exclusion rects — points inside are always dropped "
                                         "as background.")
            # GT overlays / metric:
            opt = st.columns([1.1, 1.2, 1.0, 2.7])
            gt_on = opt[0].toggle("🏷️ GT boxes", value=False, disabled=not has_gt,
                                  key="bf_gt",
                                  help="Overlay this frame's ground-truth 3D boxes."
                                       if has_gt else "No ground truth for this dataset.")
            metric_on = opt[1].toggle("📊 FG quality", value=False, disabled=not has_gt,
                                      key="bf_metric",
                                      help="Live foreground-vs-GT quality for this frame "
                                           "(a tuning proxy, not the detection F1)."
                                           if has_gt else "No ground truth for this dataset.")
            min_pts = opt[2].number_input("≥ pts", 1, 200, 10, 1, key="bf_minpts",
                                          help="A GT object counts as 'covered' with at "
                                               "least this many surviving foreground points.")

            pts = _load_raw(pcd_files[i])
            fg, _ = filter_points_with_model(pts, st.session_state.bg_model, config)
            gt_objs = None
            if (gt_on or metric_on) and has_gt:
                import label_projection as lp
                gp = gt_index.get(_frame_key(pcd_files[i]))
                if gp:
                    gt_objs = lp.load_objects(gp)
            with st.container(height=560):
                st.plotly_chart(
                    create_filtered_figure(fg, pts, margin=12.0, zoom=zoom,
                                           show_road=road_on, road_dashed=(_src != "cropped"),
                                           show_roi=roi_on, show_excl=excl_on,
                                           gt_objs=gt_objs if gt_on else None),
                    use_container_width=True, key="bf_fig")
            st.caption(f"{os.path.basename(pcd_files[i])} · frame {i+1}/{n_bf} · "
                       f"{len(fg)} foreground / {len(pts)} points")

            if metric_on and gt_objs is not None:
                q = foreground_quality(fg, pts, gt_objs, min_pts=int(min_pts))
                m = st.columns(3)
                m[0].metric(f"Objects covered (≥{q['min_pts']} pts)",
                            f"{q['covered']} / {q['scanned']}",
                            help="GT objects with enough surviving foreground points, out of "
                                 "objects the LiDAR actually hit this frame.")
                m[1].metric("On-object point recall",
                            f"{q['recall']*100:.0f}%" if q['recall'] is not None else "—",
                            help="Foreground points inside GT boxes ÷ original points inside GT "
                                 "boxes — are the real movers being kept?")
                m[2].metric("Off-object foreground", f"{q['off_object']:,}",
                            help="Foreground points outside every GT box (clutter / false-"
                                 "foreground proxy).")
            elif metric_on and has_gt:
                st.caption("No GT for this frame.")

            if playing and i < n_bf - 1:
                import time
                time.sleep(float(play_delay))
                st.session_state.bf_frame = i + 1
                st.rerun(scope="fragment")

        _bf_viewer()
else:
    st.info("Adjust parameters and click 'Build Background Model' to begin.")
