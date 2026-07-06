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
import run_history as rh

# --- Page Configuration ---
import nav
st.set_page_config(layout="wide", page_title="Background Filtering", page_icon="assets/favicon.png")
nav.render_sidebar()
st.title("🔬 Background Filtering")
logging.info("--- Background Filter Page Loaded ---")

# --- Input and Output Paths ---
st.subheader("📁 Input and Output")

# ---------------- Active-dataset paths -----------------
import dataset_manager as dm
_ds = dm.get_active()
st.caption(f"📂 Dataset: **{_ds.name}**")
_sc, _ic = st.columns(2)
_sensor_label = _sc.radio("Sensor", ["Registered", "South", "North"],
                          key="pipeline_sensor", horizontal=True,
                          help="Which LiDAR to filter. Registered = the fused south+north cloud "
                               "(default). Each sensor writes to its own model/filtered/detection "
                               "folders so their metrics can be compared.")
_sensor = _sensor_label.lower()
_src_label = _ic.radio("Input cloud", ["Cropped (road)", "Full (uncropped)"],
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

# Folder paths default from the Sensor/Input toggles; tuck them into a collapsible
# section (keyed by sensor/source so they re-follow the toggles when you switch) —
# same position/pattern as Object Detection and Tracking.
with st.expander("📁 Folder paths & model (advanced override)", expanded=False):
    fp1, fp2 = st.columns(2)
    model_save_path = fp1.text_input("Background Model Path", DEFAULT_MODEL_PATH,
                                     key=f"bf_model_{_sensor}_{_src}")
    pcd_dir_in = fp2.text_input("PCD Directory", DEFAULT_PCD, key=f"bf_pcd_{_sensor}_{_src}")
    fp3, fp4 = st.columns(2)
    output_dir = fp3.text_input("Output Directory (filtered PCDs)", DEFAULT_OUT,
                                key=f"bf_out_{_sensor}_{_src}")
    build_frames = fp4.number_input("Build Frames (0 = all)", min_value=0, value=0, key="bf_buildN")
    fp5, fp6 = st.columns(2)
    use_saved_model = fp5.checkbox("Load saved model if available", value=True, key="bf_loadsaved")
    save_filtered = fp6.checkbox("Save filtered foreground points (PCD)", value=True, key="bf_savefilt")

# Path relative to data/ (not just the basename) so it's clear which KIND of folder
# this is — raw/ vs derived/, point_clouds/ vs labels/, cropped vs not — not just
# which sensor (that's already shown by the radio above).
_pcd_rel = os.path.relpath(DEFAULT_PCD.rstrip('/'), _ds.data_dir)
_gt_ok = os.path.isdir(DEFAULT_GT)
_io_parts = [f"🛰️ Input: `{_pcd_rel}`"]
if not _gt_ok:
    _io_parts.append("🏷️ GT: none found")
else:
    _io_parts.append(f"🏷️ GT: `{os.path.relpath(DEFAULT_GT.rstrip('/'), _ds.data_dir)}`")
st.caption("  ·  ".join(_io_parts))

config = {"pcd_dir": pcd_dir_in, "build_frames": build_frames, "gt_dir": DEFAULT_GT}

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
                           azimuth=45.0, elevation=35.0, sensors=None,
                           uncovered_objs=None, off_object_pts=None, split=None):
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
    off_object_pts = _clip(off_object_pts)

    # Sensor split (registered only): colour the original cloud by source LiDAR —
    # south vs north — like the registration tab, so you can see fusion coverage /
    # which sensor a region's points come from. Overrides the plain/height original.
    # Split shows whenever it's supplied (independent of the ⚪ Original toggle — it IS
    # the original cloud, just coloured by source LiDAR).
    if split is not None:
        s_pts, n_pts = _clip(split[0]), _clip(split[1])
        # Match the Registration tab's by-sensor palette exactly (reg.SENSOR_COLORS).
        if s_pts is not None and len(s_pts):
            fig.add_trace(go.Scatter3d(x=s_pts[:, 0], y=s_pts[:, 1], z=s_pts[:, 2], mode="markers",
                name="South", marker=dict(size=1.5, color=reg.SENSOR_COLORS["south"], opacity=0.55)))
        if n_pts is not None and len(n_pts):
            fig.add_trace(go.Scatter3d(x=n_pts[:, 0], y=n_pts[:, 1], z=n_pts[:, 2], mode="markers",
                name="North", marker=dict(size=1.5, color=reg.SENSOR_COLORS["north"], opacity=0.55)))
    elif show_original and original_pts.size > 0:
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
    # Off-object foreground (clutter / false-foreground): the foreground points that
    # fall outside every GT box — yellow, on top of the red foreground.
    if off_object_pts is not None and off_object_pts.size > 0:
        fig.add_trace(go.Scatter3d(x=off_object_pts[:, 0], y=off_object_pts[:, 1], z=off_object_pts[:, 2],
            mode="markers", name="Off-object FG", marker=dict(size=2.8, color="#ffd400", opacity=0.95)))

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

    # Uncovered objects (hit by the LiDAR but too little foreground kept) — bright
    # red wireframes, so the objects the filter under-preserved stand out.
    if uncovered_objs:
        import label_projection as lp
        ux, uy, uz = [], [], []
        for o in uncovered_objs:
            c = lp.cuboid_corners(o["val"])
            for a, b in lp._EDGES:
                ux += [c[a, 0], c[b, 0], None]
                uy += [c[a, 1], c[b, 1], None]
                uz += [c[a, 2], c[b, 2], None]
        fig.add_trace(go.Scatter3d(x=ux, y=uy, z=uz, mode="lines",
            line=dict(color="#ff2b2b", width=6), name="Uncovered obj", hoverinfo="skip"))

    if sensors:
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

# ---------------- Model & filter parameters (collapsible, on the page) ----------------
st.subheader("⚙️ Model & Filter Parameters")

with st.expander("🧹 Ground removal", expanded=False):
    g1, g2 = st.columns(2)
    config["ground_grid"] = g1.number_input("Ground Grid Size (m)", 0.1, 2.0, 0.5, 0.1)
    config["dz_thresh"] = g2.number_input("Ground Z Threshold (m)", 0.1, 1.0, 0.3, 0.05)

with st.expander("🧱 Background model (occupancy + persistence)", expanded=False):

    b1, b2 = st.columns(2)
    config["bg_voxel"] = b1.slider("BG Voxel Size (m)", 0.5, 2.0, 1.0, 0.1)
    config["bg_ratio"] = b2.slider("BG Presence Ratio", 0.5, 1.0, 0.85, 0.01)
    b3, b4 = st.columns(2)
    config["cell_size"] = b3.slider("Grid Cell Size (m)", 0.5, 5.0, 1.0, 0.1)
    config["cell_ratio"] = b4.slider("Cluster Presence Ratio", 0.5, 1.0, 0.75, 0.01)

with st.expander("🔗 Clustering (DBSCAN)", expanded=False):
    cluster_mode = st.radio(
        "Clusterer", ["Density-adaptive (recommended)", "Global (legacy)"],
        key="bf_cluster_mode", horizontal=True,
        help="Density-adaptive measures the local point spacing per range tier and sets "
             "eps/min_samples from it — fixes the fused cloud where 'far' is actually dense. "
             "Global is the original single-median-eps clusterer (eps0/eps_k below).")
    _mode = "density" if cluster_mode.startswith("Density") else "global"
    dc1, dc2, dc3 = st.columns(3)
    ds_voxel = dc1.slider("Downsample Voxel (Build)", 0.05, 0.5, 0.15, 0.01)
    eps_min = dc2.number_input("eps_min", value=0.35)
    eps_max = dc3.number_input("eps_max", value=2.0)
    if _mode == "density":
        dd1, dd2, dd3 = st.columns(3)
        eps_scale = dd1.number_input("eps_scale (× spacing)", value=2.5, step=0.1,
            help="eps = eps_scale × measured k-NN spacing of the tier, clamped to [eps_min, eps_max].")
        n_tiers = int(dd2.number_input("Range tiers", min_value=1, max_value=6, value=3, step=1,
            help="Split the cloud into this many range bands; eps/min_samples adapt per band."))
        dd4, dd5 = st.columns(2)
        min_samples = dd4.number_input("min_samples (dense)", value=16)
        min_samples_far = dd5.number_input("min_samples (sparse tier)", value=8,
            help="Eased point count for the sparsest tier so distant/thin objects still cluster.")
        eps0 = eps_k = 0.0  # unused in density mode
    else:
        dc4, dc5, dc6 = st.columns(3)
        eps0 = dc4.number_input("eps0", value=0.35)
        eps_k = dc5.number_input("eps_k", value=0.008, format="%.4f")
        min_samples = dc6.number_input("min_samples", value=16)
        eps_scale, n_tiers, min_samples_far = 2.5, 3, 8
    config["cluster"] = {"mode": _mode, "ds_voxel": ds_voxel, "eps0": eps0, "eps_k": eps_k,
                         "eps_min": eps_min, "eps_max": eps_max, "min_samples": min_samples,
                         "eps_scale": eps_scale, "n_tiers": n_tiers,
                         "min_samples_far": min_samples_far}

with st.expander("📍 Pole-like geometry filter", expanded=False):
    config["enable_pole_filter"] = st.checkbox("Enable Pole Filter", value=True)
    p1, p2, p3 = st.columns(3)
    config["pole_min_height"] = p1.number_input("Min Height", value=1.5)
    config["pole_min_aspect_xy"] = p2.number_input("Min Aspect XY", value=6.0)
    config["pole_max_xy_area"] = p3.number_input("Max XY Area", value=1.0)
    p4, p5, p6 = st.columns(3)
    config["pole_min_linearity"] = p4.number_input("Min Linearity", value=0.75)
    config["pole_min_points"] = p5.number_input("Min Points", value=8)
    config["pole_max_points"] = p6.number_input("Max Points (pole)", value=80,
        help="A tall, small-footprint cluster is only treated as a pole if it has at most this many "
             "points. Dense objects (trucks/vans) exceed it and are kept. Raise to delete more; "
             "lower to protect more vehicles.")

with st.expander("⚙️ Misc filters", expanded=False):
    config["inward_buffer_m"] = st.number_input("Road Edge Inward Buffer (m)", value=2.0)

with st.expander("🧽 Denoise (statistical outlier removal)", expanded=False):
    config["enable_sor"] = st.checkbox("Enable SOR on foreground output", value=False,
        help="Drop scattered isolated points after subtraction. NOTE (measured on the "
             "detection ablation): SOR is a pure precision↔recall DIAL, not a net win — it "
             "raises precision but removes real on-object points, costing mid-field recall "
             "(20-40 m). Default OFF because wrong-way detection is recall-critical. Turn on "
             "(with a gentle std ≥ 3) only if you specifically need fewer false positives.")
    so1, so2 = st.columns(2)
    config["sor_k"] = int(so1.number_input("SOR neighbours (k)", min_value=4, max_value=64,
        value=12, step=1, help="Mean distance is computed over k nearest neighbours."))
    config["sor_std"] = so2.number_input("SOR std ratio", min_value=0.5, max_value=5.0,
        value=2.0, step=0.1, help="Lower = more aggressive (drops more points).")

# ---------------- Load or Build background model ----------------
if "bg_model" not in st.session_state:
    st.session_state.bg_model = None
    st.session_state.bg_model_path = None

# The model in session_state is a SINGLE slot, but each sensor/source has its OWN
# background model (the path encodes <sensor>/<crop>). If the selected path changed
# — switching the Sensor/Input radios or editing the path — drop the loaded model so
# we never apply one sensor's model to another's cloud (which would paint bogus
# foreground everywhere). It then auto-loads the matching saved model below, or, if
# none exists, the viewer prompts you to build it.
if st.session_state.get("bg_model_path") != model_save_path:
    st.session_state.bg_model = None
    st.session_state.bg_model_path = None

if use_saved_model and st.session_state.bg_model is None and os.path.exists(model_save_path):
    try:
        with open(model_save_path, "rb") as fp:
            st.session_state.bg_model = pickle.load(fp)
        st.session_state.bg_model_path = model_save_path
        st.success(f"Loaded saved background model from: {model_save_path}")
    except Exception as e:
        st.warning(f"Could not load saved model: {e}")
        st.session_state.bg_model = None
        st.session_state.bg_model_path = None

if st.button("🧠 Build Background Model", use_container_width=True, type="primary", key="bf_build"):
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
            st.session_state.bg_model_path = model_save_path
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

    # ---------------- Run tracker: "is it getting better?" ----------------
    # Scores the CURRENT model+config over a sample of frames with the foreground-
    # quality proxy and logs it, so each tuning change shows an explicit delta vs the
    # last run + a trend line. (A fast stand-in for the full detection eval.)
    if has_gt and pcd_files:
        # run_history/bgfilter/ — separate from run_history/eval/ (Evaluation page's
        # detection-metric + A/B trackers) so the folder makes it obvious at a glance
        # which tracker each logged file belongs to.
        _RH_CAT = "bgfilter"
        _tag = f"{_sensor}_{_src}"

        def _aggregate_quality(_pcd_files, _gt_index, _bg_model, _cfg, min_pts, stride):
            import label_projection as lp
            cov = scan = nframes = 0
            recalls, offs = [], []
            for f in _pcd_files[::max(1, stride)]:
                gp = _gt_index.get(_frame_key(f))
                if not gp:
                    continue
                pts = _load_raw(f)
                fg, _ = filter_points_with_model(pts, _bg_model, _cfg)
                q = foreground_quality(fg, pts, lp.load_objects(gp), min_pts=min_pts)
                cov += q["covered"]; scan += q["scanned"]; offs.append(q["off_object"])
                if q["recall"] is not None:
                    recalls.append(q["recall"])
                nframes += 1
            return dict(
                covered=int(cov), scanned=int(scan),
                covered_pct=(100.0 * cov / scan if scan else 0.0),
                recall=(100.0 * float(np.mean(recalls)) if recalls else 0.0),
                off_object=(float(np.mean(offs)) if offs else 0.0),
                frames=int(nframes))

        with st.expander("📈 Run tracker — is it getting better?", expanded=False):
            tc = st.columns(3)
            track_stride = int(tc[0].number_input("Sample every Nth frame", 1, 50, 10,
                key="bf_track_stride", help="Higher = faster, coarser estimate."))
            track_minpts = int(tc[1].number_input("≥ pts for 'covered'", 1, 200, 5,
                key="bf_track_minpts"))
            note = tc[2].text_input("Note (optional)", key="bf_track_note",
                placeholder="e.g. density eps + SOR")
            bcol = st.columns([3, 1])
            if bcol[0].button("📊 Evaluate current settings & log", use_container_width=True,
                              type="primary", key="bf_track_eval"):
                with st.spinner("Scoring foreground quality over sampled frames..."):
                    metrics = _aggregate_quality(pcd_files, gt_index, st.session_state.bg_model,
                                                 config, track_minpts, track_stride)
                    rh.log_run(_ds, _RH_CAT, _tag, metrics, rh.summarize_params(config), note=note)
                st.success(f"Logged: {metrics['covered']}/{metrics['scanned']} objects covered "
                           f"over {metrics['frames']} sampled frames.")
            if bcol[1].button("🗑️ Clear", use_container_width=True, key="bf_track_clear"):
                rh.clear_history(_ds, _RH_CAT, _tag)
                st.rerun()

            hist = rh.load_history(_ds, _RH_CAT, _tag)
            if hist:
                cur = hist[-1]["metrics"]
                prev = hist[-2]["metrics"] if len(hist) > 1 else None
                mc = st.columns(3)
                mc[0].metric("Objects covered", f"{cur['covered_pct']:.1f}%",
                             f"{cur['covered_pct']-prev['covered_pct']:+.1f}%" if prev else None,
                             help="GT objects with ≥ min pts foreground, over a frame sample.")
                mc[1].metric("On-object point recall", f"{cur['recall']:.1f}%",
                             f"{cur['recall']-prev['recall']:+.1f}%" if prev else None)
                mc[2].metric("Off-object FG (avg/frame)", f"{cur['off_object']:.0f}",
                             f"{cur['off_object']-prev['off_object']:+.0f}" if prev else None,
                             delta_color="inverse",
                             help="Lower is better — scattered false-foreground (green = improved).")
                import pandas as pd
                df = pd.DataFrame([{"run": i + 1,
                                    "covered %": h["metrics"]["covered_pct"],
                                    "recall %": h["metrics"]["recall"]}
                                   for i, h in enumerate(hist)]).set_index("run")
                st.line_chart(df)
                if prev:
                    diff = rh.param_diff(hist[-2]["params"], hist[-1]["params"])
                    st.caption("🔧 Changed since previous run: "
                               + (", ".join(f"`{k}` {a}→{b}" for k, (a, b) in diff.items())
                                  if diff else "nothing tracked changed."))
                with st.expander("Full run log", expanded=False):
                    st.dataframe(pd.DataFrame([{"time": h["time"], "note": h.get("note", ""),
                                                **h["metrics"]} for h in hist]),
                                 use_container_width=True)
            else:
                st.caption("No runs logged yet — build/adjust, then click "
                           "**Evaluate current settings & log** to start the trend.")

    if pcd_files:
        n_bf = len(pcd_files)
        st.session_state.setdefault("bf_frame", 0)

        @st.fragment
        def _bf_viewer():
            # Compact one-line frame navigation.
            i, playing, play_delay = vu.nav_row("bf_frame", n_bf, "bf")

            # Persistent camera (one line): zoom = distance, rotate/tilt = orbit,
            # view margin = how far past the road the window extends (clips points
            # AND bounds the axes — raise it to see objects/boxes on the approaches).
            ZOOM_DEFAULTS = {"cropped": 0.7, "full": 0.9}
            cz, ca, ce, cv = st.columns(4)
            zoom = cz.slider("🔍 Zoom", 0.35, 2.0, ZOOM_DEFAULTS.get(_src, 0.9), 0.05,
                             key=f"bf_zoom_{_src}", help="Lower = closer.")
            azimuth = ca.slider("🔄 Rotate", 0, 360, 45, 5, key="bf_az",
                                help="Orbit angle around the scene (degrees).")
            elevation = ce.slider("📐 Tilt", 5, 88, 35, 1, key="bf_el",
                                  help="Camera height angle; 88 ≈ straight-down bird's-eye.")
            view_margin = cv.slider("🔭 View margin (m)", 12, 100, 12, 4, key="bf_margin",
                                    help="How far beyond the road the view extends. The default keeps the "
                                         "height axis readable; raise it to see vehicles / GT boxes out on "
                                         "the approaches (use a high Tilt for a top-down look — wide "
                                         "margins flatten the height axis).")

            road_key = f"bf_road_{_src}"
            # Seed every toggle so bulk on/off can flip them without value= conflicts.
            toggle_defaults = {
                "bf_show_fg": True, "bf_show_orig": True, road_key: (_src == "cropped"),
                "bf_roi": False, "bf_excl": False, "bf_gt": False, "bf_sensors": True,
                "bf_height": False, "bf_metric": False, "bf_uncov": False, "bf_offfg": False,
                "bf_split": False,
            }
            vu.ensure_toggle_defaults(toggle_defaults)
            # "All" also turns Height on (per request).
            overlay_keys = ["bf_show_fg", "bf_show_orig", road_key, "bf_roi", "bf_excl",
                            "bf_gt", "bf_uncov", "bf_offfg", "bf_sensors", "bf_height"]

            with st.expander("🎛️ Layers & overlays", expanded=False):
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
                r3 = st.columns(4)
                uncov_on = r3[0].toggle("❌ Uncovered obj", key="bf_uncov", disabled=not has_gt,
                                        help="Red-outline GT objects the LiDAR hit but the filter kept "
                                             "< '≥ pts' foreground for — the objects you're losing."
                                             if has_gt else "No ground truth for this dataset.")
                offfg_on = r3[1].toggle("🟡 Off-object FG", key="bf_offfg", disabled=not has_gt,
                                        help="Recolour (yellow) the foreground points that fall OUTSIDE "
                                             "every GT box — clutter / false-foreground."
                                             if has_gt else "No ground truth for this dataset.")
                metric_on = r3[2].toggle("📊 FG quality", key="bf_metric", disabled=not has_gt,
                                         help="Live foreground-vs-GT quality numbers (a tuning proxy)."
                                              if has_gt else "No ground truth for this dataset.")
                min_pts = r3[3].number_input("≥ pts", 1, 200, 5, 1, key="bf_minpts",
                                             help="Foreground pts for a GT object to count as 'covered'.")
                r4 = st.columns(4)
                split_on = r4[0].toggle("🔵🟠 S/N split", key="bf_split",
                                        disabled=(_sensor != "registered"),
                                        help="Colour the original cloud by source LiDAR — south (blue) vs "
                                             "north (orange) — like the Registration tab, to see fusion "
                                             "coverage. Registered only."
                                             if _sensor == "registered"
                                             else "Only for the Registered (fused) cloud.")
                off_buf = r4[1].number_input("🟡 box buffer (m)", 0.0, 2.0, 0.3, 0.1,
                                             key="bf_offbuf", disabled=not has_gt,
                                             help="Grow each GT box by this margin before deciding which "
                                                  "foreground is 'off-object' (yellow). A return spilling "
                                                  "just past a tight / mis-placed box then counts as on-object, "
                                                  "not clutter. Moves the yellow overlay AND the off-object / "
                                                  "covered numbers together (set 0 for the strict count).")
                h_span = st.slider("Height span (m)", 1.5, 12.0, 4.0, 0.5, key="bf_hspan",
                                   help="Colour spreads over this many metres above ground.") \
                    if color_h else 4.0

            sensors = reg.lidar_markers(_ds, _sensor) if sensors_on else None
            pts = _load_raw(pcd_files[i])
            fg, _ = filter_points_with_model(pts, st.session_state.bg_model, config)
            gt_objs = None
            if (gt_on or metric_on or uncov_on or offfg_on) and has_gt:
                import label_projection as lp
                gp = gt_index.get(_frame_key(pcd_files[i]))
                if gp:
                    gt_objs = lp.load_objects(gp)
            # Foreground-quality analysis, computed ONCE with the box buffer and shared by
            # the metric readout, the caption, and the overlays — so the 🟡 box-buffer
            # slider moves the yellow points AND the off-object count together (a wider
            # buffer counts truck-edge spillover as on-object, not clutter).
            q = None
            if gt_objs is not None and (metric_on or uncov_on or offfg_on):
                q = foreground_quality(fg, pts, gt_objs, min_pts=int(min_pts), box_buffer=float(off_buf))
            uncovered_objs = q["uncovered"] if (q and uncov_on) else None
            off_object_pts = fg[~q["fg_on_mask"]] if (q and offfg_on and len(fg)) else None
            # Per-frame south/north split for the registered cloud (on-the-fly re-fuse).
            split = reg.registered_split_for_frame(_ds, _frame_key(pcd_files[i])) \
                if (split_on and _sensor == "registered") else None
            # NOTE: render the chart directly (no fixed-height st.container) — wrapping
            # it in a container remounts the plot each rerun and wipes the camera, which
            # defeats uirevision. Height is set on the figure instead.
            st.plotly_chart(
                create_filtered_figure(fg, pts, margin=float(view_margin), zoom=zoom,
                                       show_road=road_on, road_dashed=(_src != "cropped"),
                                       show_roi=roi_on, show_excl=excl_on,
                                       gt_objs=gt_objs if gt_on else None,
                                       color_by_height=color_h, height_span=h_span,
                                       show_foreground=show_fg_pts, show_original=show_orig,
                                       height=560, azimuth=azimuth, elevation=elevation,
                                       sensors=sensors, uncovered_objs=uncovered_objs,
                                       off_object_pts=off_object_pts, split=split),
                use_container_width=True, key="bf_fig")
            cap = (f"{os.path.basename(pcd_files[i])} · frame {i+1}/{n_bf} · "
                   f"{len(fg)} foreground / {len(pts)} points")
            if q is not None:
                cap += (f"  ·  ✅ {q['covered']}/{q['scanned']} objects covered  ·  "
                        f"🟡 {q['off_object']:,} off-object FG")
            st.caption(cap)

            if metric_on and q is not None:
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
