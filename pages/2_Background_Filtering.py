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
import registration as reg
import viewer_ui as vu

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Background Filtering")
st.title("🔬 Background Filtering")
logging.info("--- Background Filter Page Loaded ---")

# ---------------- Active-dataset paths -----------------
import dataset_manager as dm
_ds = dm.get_active()
st.sidebar.caption(f"📂 Dataset: **{_ds.name}**")
_sensor_label = st.sidebar.radio("Sensor", ["Registered", "South", "North"],
                                 key="pipeline_sensor", horizontal=True,
                                 help="Which LiDAR to filter. Registered = the fused south+north cloud "
                                      "(default). Each sensor writes to its own model/filtered/detection "
                                      "folders so their metrics can be compared.")
_sensor = _sensor_label.lower()
_src_label = st.sidebar.radio("Input cloud", ["Cropped (road)", "Full (uncropped)"],
                              key="pipeline_source", horizontal=True,
                              help="Cropped = road-clipped clouds; Full = raw/fused clouds (research region). "
                                   "Each writes to its own model/filtered/detection folders so you can "
                                   "compare eval metrics. The choice is shared across Filtering / Detection / Evaluation.")
_src = "cropped" if _src_label.startswith("Cropped") else "full"
DEFAULT_MODEL_PATH = _ds.model_path_for_sensor(_sensor, _src)
DEFAULT_PCD = _ds.input_pcd_for_sensor(_sensor, _src)
# auto-pick the GT matching the input sensor so the overlay + FG-quality metric
# line up across south / north / registered.
DEFAULT_GT = _ds.gt_dir_for_input(DEFAULT_PCD)
DEFAULT_OUT = _ds.filtered_dir_for_sensor(_sensor, _src)
st.sidebar.caption(f"🛰️ Input: `{os.path.basename(DEFAULT_PCD.rstrip('/'))}`"
                   + (f"  ·  🏷️ GT: `{os.path.basename(DEFAULT_GT.rstrip('/'))}`"
                      if os.path.isdir(DEFAULT_GT) else "  ·  🏷️ GT: none found"))

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
                           show_excl=False, gt_objs=None, color_by_height=False, height_span=4.0,
                           show_foreground=True, show_original=True, height=560,
                           azimuth=45.0, elevation=35.0, sensors=None):
    fig = go.Figure()
    # Resolve the road window first so we can lock the view AND clip the plotted points
    # to it. Clipping is the key: aspectmode="data" sizes the 3D box from the POINT
    # extent, so the full cloud's ~200 m span would crush z (flatten the view). Points
    # outside the locked window are off-screen anyway, so clipping is visually lossless
    # and keeps the SAME camera/zoom feel as the cropped view (back to aspectmode="data").
    poly = None
    zmin, zmax = -12.0, 1.0
    xlo = xhi = ylo = yhi = None
    try:
        from geometry_config import get_road_polygon
        poly = get_road_polygon()
        minx, miny, maxx, maxy = poly.bounds
        m = float(margin)
        xlo, xhi, ylo, yhi = minx - m, maxx + m, miny - m, maxy + m
        xr = dict(range=[xlo, xhi]); yr = dict(range=[ylo, yhi])
    except Exception:
        xr, yr = {}, {}

    def _clip(p):
        if p is None or not len(p) or xlo is None:
            return p
        mask = (p[:, 0] >= xlo) & (p[:, 0] <= xhi) & (p[:, 1] >= ylo) & (p[:, 1] <= yhi)
        return p[mask]
    original_pts = _clip(original_pts)
    foreground_pts = _clip(foreground_pts)

    if show_original and original_pts.size > 0:
        if color_by_height:
            z = original_pts[:, 2]; z0 = float(np.percentile(z, 1))
            omk = dict(size=1.5, color=z, colorscale="Turbo", cmin=z0, cmax=z0 + float(height_span),
                       opacity=0.5, showscale=False)
        else:
            omk = dict(size=1.5, color="#8fa3bd", opacity=0.2)
        fig.add_trace(go.Scatter3d(x=original_pts[:, 0], y=original_pts[:, 1], z=original_pts[:, 2],
            mode="markers", name="Original", marker=omk))
    if show_foreground and foreground_pts.size > 0:
        fig.add_trace(go.Scatter3d(x=foreground_pts[:, 0], y=foreground_pts[:, 1], z=foreground_pts[:, 2],
            mode="markers", name="Foreground", marker=dict(size=2.5, color="red", opacity=0.9)))

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

    if sensors:
        import registration as reg
        for tr in reg.sensor_marker_traces(go, sensors, z_floor=z_floor):
            fig.add_trace(tr)

    # Camera is driven entirely by the (persistent) sliders: zoom = distance,
    # azimuth = orbit angle, elevation = tilt. Building eye from these angles is what
    # makes the view persist across frames/toggles (the slider values live in
    # session_state). Default az=45, el=35 reproduces the classic (z,z,z) view.
    r = float(zoom) * np.sqrt(3.0)
    az, el = np.radians(float(azimuth)), np.radians(float(elevation))
    eye = dict(x=float(r * np.cos(el) * np.cos(az)),
               y=float(r * np.cos(el) * np.sin(az)),
               z=float(r * np.sin(el)))
    urev = f"bf_{zoom}_{azimuth}_{elevation}"
    fig.update_layout(height=height, margin=dict(l=0, r=0, b=0, t=0), showlegend=True,
                      uirevision=urev,
                      scene=dict(aspectmode="data", xaxis=xr, yaxis=yr,
                                 zaxis=dict(range=[zmin, zmax]),
                                 camera=dict(eye=eye, up=dict(x=0, y=0, z=1)),
                                 uirevision=urev))
    return fig


from dataset_prep import foreground_quality  # shared with the Geometry Editor

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
            # clear stale filtered clouds (e.g. from a previous naming generation)
            # so the folder always matches THIS run's frames — outputs are named by
            # input basename, so a renamed input would otherwise leave orphans.
            for _old in glob.glob(os.path.join(output_dir, "*.pcd")):
                try:
                    os.remove(_old)
                except OSError:
                    pass
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
            # Compact one-line frame navigation.
            i, playing, play_delay = vu.nav_row("bf_frame", n_bf, "bf")

            # Persistent camera (one line): zoom = distance, rotate/tilt = orbit.
            ZOOM_DEFAULTS = {"cropped": 0.7, "full": 0.9}
            cz, ca, ce = st.columns(3)
            zoom = cz.slider("🔍 Zoom", 0.35, 2.0, ZOOM_DEFAULTS.get(_src, 0.9), 0.05,
                             key=f"bf_zoom_{_src}", help="Lower = closer.")
            azimuth = ca.slider("🔄 Rotate", 0, 360, 45, 5, key="bf_az",
                                help="Orbit angle around the scene (degrees).")
            elevation = ce.slider("📐 Tilt", 5, 88, 35, 1, key="bf_el",
                                  help="Camera height angle; 88 ≈ straight-down bird's-eye.")

            road_key = f"bf_road_{_src}"
            # Seed every toggle so bulk on/off can flip them without value= conflicts.
            toggle_defaults = {
                "bf_show_fg": True, "bf_show_orig": True, road_key: (_src == "cropped"),
                "bf_roi": False, "bf_excl": False, "bf_gt": False, "bf_sensors": True,
                "bf_height": False, "bf_metric": False,
            }
            vu.ensure_toggle_defaults(toggle_defaults)
            overlay_keys = ["bf_show_fg", "bf_show_orig", road_key, "bf_roi",
                            "bf_excl", "bf_gt", "bf_sensors"]

            with st.expander("🎛️ Layers & overlays", expanded=True):
                vu.bulk_toggle_buttons(overlay_keys, "bf_bulk")
                r1 = st.columns(4)
                show_fg_pts = r1[0].toggle("🔴 Foreground", key="bf_show_fg",
                                           help="The red moving-foreground points.")
                show_orig = r1[1].toggle("⚪ Original", key="bf_show_orig",
                                         help="The faint background/original cloud.")
                color_h = r1[2].toggle("🌈 Height", key="bf_height",
                                       help="Colour the original cloud by z (Turbo).")
                sensors_on = r1[3].toggle("📍 LiDAR", key="bf_sensors",
                                          help="Mark the LiDAR position(s) + nadir.")
                r2 = st.columns(4)
                road_on = r2[0].toggle("🛣️ Road", key=road_key,
                                       help="Road polygon. Solid on cropped (the crop); dashed on full.")
                roi_on = r2[1].toggle("🔵 ROI", key="bf_roi",
                                      help="Bounds what the filter processes.")
                excl_on = r2[2].toggle("🟣 Exclusion", key="bf_excl",
                                       help="Foreground-exclusion rects — always dropped.")
                gt_on = r2[3].toggle("🏷️ GT", key="bf_gt", disabled=not has_gt,
                                     help="Overlay this frame's GT boxes."
                                          if has_gt else "No ground truth for this dataset.")
                r3 = st.columns([1.3, 1, 2.7])
                metric_on = r3[0].toggle("📊 FG quality", key="bf_metric", disabled=not has_gt,
                                         help="Live foreground-vs-GT quality (a tuning proxy)."
                                              if has_gt else "No ground truth for this dataset.")
                min_pts = r3[1].number_input("≥ pts", 1, 200, 10, 1, key="bf_minpts",
                                             help="Pts for a GT object to count as 'covered'.")
                h_span = r3[2].slider("Height span (m)", 1.5, 12.0, 4.0, 0.5, key="bf_hspan",
                                      help="Colour spreads over this many metres above ground.") \
                    if color_h else 4.0

            sensors = reg.lidar_markers(_ds, _sensor) if sensors_on else None
            pts = _load_raw(pcd_files[i])
            fg, _ = filter_points_with_model(pts, st.session_state.bg_model, config)
            gt_objs = None
            if (gt_on or metric_on) and has_gt:
                import label_projection as lp
                gp = gt_index.get(_frame_key(pcd_files[i]))
                if gp:
                    gt_objs = lp.load_objects(gp)
            # NOTE: render the chart directly (no fixed-height st.container) — wrapping
            # it in a container remounts the plot each rerun and wipes the camera, which
            # defeats uirevision. Height is set on the figure instead.
            st.plotly_chart(
                create_filtered_figure(fg, pts, margin=12.0, zoom=zoom,
                                       show_road=road_on, road_dashed=(_src != "cropped"),
                                       show_roi=roi_on, show_excl=excl_on,
                                       gt_objs=gt_objs if gt_on else None,
                                       color_by_height=color_h, height_span=h_span,
                                       show_foreground=show_fg_pts, show_original=show_orig,
                                       height=560, azimuth=azimuth, elevation=elevation,
                                       sensors=sensors),
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
