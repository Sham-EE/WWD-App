import streamlit as st
import os
import glob

from detection_logic import run_detection_and_tracking, sorted_by_frame_index, DEFAULT_DETECTION_PARAMS
from visualization import create_3d_figure, generate_tracking_animation
from wwd_detection import load_lane_config, lanes_calibrated, detect_wrong_way, summarize_wrong_way
import registration as reg
import viewer_ui as vu
import nav

st.set_page_config(layout="wide", page_title="Object Detection and Tracking", page_icon="assets/favicon.png")
nav.render_sidebar()

st.title("📦 Object Detection and Tracking")

# Initialize session state
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

# --- Input and Output Paths ---
st.subheader("📁 Input and Output")

import dataset_manager as dm
_ds = dm.get_active()
st.caption(f"📂 Dataset: **{_ds.name}**")
_sc, _ic = st.columns(2)
_sensor_label = _sc.radio("Sensor", ["Registered", "South", "North"],
                          key="pipeline_sensor", horizontal=True,
                          help="Which LiDAR to run on. Must match what you built in Background Filtering "
                               "(each sensor has its own filtered/detection folders). Shared across pages.")
_sensor = _sensor_label.lower()
_src_label = _ic.radio("Input cloud", ["Cropped (road)", "Full (uncropped)"],
                       key="pipeline_source", horizontal=True,
                       help="Must match what you ran in Background Filtering. Each source has its own "
                            "filtered/detection folders so you can compare eval metrics. **Cropped is "
                            "recommended for detection** — the full research-region cloud's off-road "
                            "clutter roughly doubles false positives (measured F1 ≈0.52 cropped vs ≈0.36 full).")
_src = "cropped" if _src_label.startswith("Cropped") else "full"
if _src == "full":
    st.warning("⚠️ Running detection on the **full** cloud — off-road clutter inflates false positives. "
               "**Cropped (road)** scores markedly higher (F1 ≈0.52 vs ≈0.36). Switch unless you "
               "specifically need the full research region.", icon="⚠️")

output_dir = _ds.detection_dir_for_sensor(_sensor, _src)
# Folder paths default from the Sensor/Input toggles; tuck them into a collapsible
# section (keyed by sensor/source so they re-follow the toggles when you switch).
with st.expander("📁 Folder paths (advanced override)", expanded=False):
    filtered_pcd_dir = st.text_input(
        "FILTERED PCD files (for detection):",
        value=_ds.filtered_dir_for_sensor(_sensor, _src), key=f"odt_filt_{_sensor}_{_src}")
    original_pcd_dir = st.text_input(
        "ORIGINAL PCD files (for visualization):",
        value=_ds.input_pcd_for_sensor(_sensor, _src), key=f"odt_orig_{_sensor}_{_src}")
    st.caption(f"Detection output → `{output_dir}`")

# --- Parameters (each group collapses into its own section) ---
st.subheader("⚙️ Algorithm Parameters")

with st.expander("🔧 Clustering & Detection", expanded=False):
    adaptive_eps = st.checkbox("Range-adaptive eps", value=True,
        help="Grow the clustering radius with distance (eps = eps0 + eps_k·range, clipped). Near, dense "
             "objects use a small eps (kept separate → precision); far, sparse objects use a larger eps "
             "(scattered points still join → recall). Measured better than a fixed eps on both precision "
             "and recall. Uncheck to use the fixed DBSCAN eps below.")
    if adaptive_eps:
        ae1, ae2 = st.columns(2)
        aeps0 = ae1.number_input("eps0 (base)", 0.1, 3.0, 0.8, 0.1)
        aeps_k = ae2.number_input("eps_k (per m)", 0.0, 0.2, 0.04, 0.01, format="%.3f")
        ae3, ae4 = st.columns(2)
        aeps_min = ae3.number_input("eps min", 0.1, 3.0, 1.0, 0.1)
        aeps_max = ae4.number_input("eps max", 0.5, 6.0, 3.0, 0.1)
        dbscan_eps = 2.0  # unused when adaptive is on
    else:
        dbscan_eps = st.slider("DBSCAN Epsilon (eps)", 0.1, 5.0, 2.0, 0.1, help="Fixed cluster radius (m).")
        aeps0, aeps_k, aeps_min, aeps_max = 0.8, 0.04, 1.0, 3.0
    cc1, cc2 = st.columns(2)
    min_cluster_pts = cc1.slider("Min Cluster Points", 1, 50, 1, 1, help="Minimum points to form a cluster.")
    min_hits = cc2.slider("Min Temporal Hits", 1, 10, 2, 1,
        help="Frames a candidate must exist to be confirmed. Higher = fewer spurious tracks and fewer "
             "ID switches, but lower recall (e.g. 3 cut ID switches ~half but dropped recall).")
    strong_pts = st.slider("Auto-accept point count (strong_pts)", 20, 400,
        int(DEFAULT_DETECTION_PARAMS["strong_pts"]), 10,
        help="A cluster with at least this many points is accepted immediately, skipping temporal "
             "confirmation — dense road clusters are unambiguously real vehicles. Lower to catch dense "
             "fast movers that fail temporal confirmation (lifts near-field recall); too low starts "
             "auto-accepting dense clutter. Lowered 200→100 by default.")

with st.expander("🚗 Vehicle class gate", expanded=False):
    vehicle_gate = st.checkbox("Drop non-vehicles (peds/bikes)", value=False,
        help="OFF (default) = detect everything, including pedestrians & bicycles (each detection is "
             "still tagged is_vehicle). ON = drop clusters below BOTH size thresholds. NOTE: on the "
             "background-FILTERED clouds, vehicles are very sparse (often fewer points than a "
             "pedestrian), so a size gate can drop vehicles — leave OFF unless using denser input. "
             "For fair vehicle-only metrics, use the Evaluation page's 'Vehicles only' instead (it "
             "ignores ped/bike detections rather than dropping them).")
    vg1, vg2 = st.columns(2)
    vehicle_min_length = vg1.number_input("Veh. min length (m)", 0.0, 10.0, 2.5, 0.1,
        help="A track counts as a vehicle if it is at least this long OR has enough points.")
    vehicle_min_points = vg2.number_input("Veh. min points", 1, 500, 40, 1,
        help="A track counts as a vehicle if it has at least this many points OR is long enough.")

with st.expander("🛰️ Tracking & Association", expanded=False):
    tc1, tc2 = st.columns(2)
    fps = tc1.slider("Frames Per Second (FPS)", 1.0, 30.0, 10.0, 0.5, help="Data frame rate for velocity calculation.")
    max_missed = tc2.slider("Max Missed Frames", 0, 20, 5, 1, help="Frames to keep a track alive without detection.")
    tc3, tc4 = st.columns(2)
    moving_speed_thresh = tc3.slider("Moving Speed Threshold (m/s)", 0.0, 10.0, 3.0, 0.1, help="Speed above which an object is 'moving'.")
    roi_abs_y = tc4.slider("ROI Absolute Y (m)", 5.0, 100.0, 40.0, 1.0, help="Y-coordinate processing range.")
    tc5, _tc6 = st.columns(2)
    truck_merge_dist = tc5.slider("Truck merge distance (m)", 0.0, 15.0,
                                  float(DEFAULT_DETECTION_PARAMS["truck_merge_dist"]), 0.5,
                                  help="Merge two clusters up to this far apart when one is truck-length. "
                                       "Lower avoids over-merging adjacent vehicles in dense (fused) clouds; "
                                       "higher re-joins a split-up truck. Default lowered 10→5 after the "
                                       "Registered A/B (10 m tanked far-field recall on the fused cloud).")

    st.markdown("**Static-phantom suppression**")
    suppress_static = st.checkbox("Drop persistent, never-moving tracks", value=True,
        help="Removes tracks that BOTH persist a long time AND never exceed the speed floor — the "
             "static-leak signature (barriers/poles the background model missed, detected in the same "
             "spot every frame). Measured +5 F1 / +8.5 precision for −0.9 recall on registered/cropped; "
             "real vehicles always break the speed floor, so recall is barely touched. Also task-aligned "
             "(a never-moving object is never a wrong-way driver).")
    sc1, sc2 = st.columns(2)
    static_min_frames = sc1.slider("Min lifetime (frames)", 5, 120,
        int(DEFAULT_DETECTION_PARAMS["static_min_frames"]), 5,
        help="Only suppress tracks seen for at least this many frames (so briefly-seen real objects are safe).")
    static_max_speed = sc2.slider("Static speed floor (m/s)", 0.1, 3.0,
        float(DEFAULT_DETECTION_PARAMS["static_max_speed"]), 0.1,
        help="A track is 'static' only if its lifetime MAX speed stays below this. Real vehicles exceed it.")

with st.expander("🎥 Visualization", expanded=False):
    col_v1, col_v2, col_v3 = st.columns(3)
    eye_x = col_v1.number_input("Camera Eye X", value=0.8, step=0.05)
    eye_y = col_v2.number_input("Camera Eye Y", value=0.8, step=0.05)
    eye_z = col_v3.number_input("Camera Eye Z", value=0.8, step=0.05)
    max_frames_to_animate = st.number_input("Max Frames to Animate (0 for all)", min_value=0, max_value=2000,
        value=0, help="Limit the number of frames to process for the animation to save time.")

st.divider()

if st.button("🚀 Start Detection and Tracking", use_container_width=True):
    st.session_state.detection_results = None # Reset results

    # --- Validate Paths and Scan Files ---
    if not all(os.path.isdir(p) for p in [filtered_pcd_dir, original_pcd_dir]):
        st.error("One of the input directories is invalid. Please check the paths.")
    else:
        filtered_files = sorted_by_frame_index(glob.glob(os.path.join(filtered_pcd_dir, "*.pcd")))
        original_files = sorted_by_frame_index(glob.glob(os.path.join(original_pcd_dir, "*.pcd")))

        # --- File Count Validation ---
        if not filtered_files:
            st.error("No PCD files found in the filtered directory.")
        elif len(filtered_files) != len(original_files):
            st.error(f"PCD file count mismatch: {len(filtered_files)} filtered vs {len(original_files)} original files.")
        else:
            # Overlay the UI-controlled values on the shared defaults (single source of
            # truth in detection_logic) — non-UI knobs (yaw_bias_deg, merge_dist,
            # yaw_merge_deg, truck_len_thresh) come straight from the defaults.
            params = {
                **DEFAULT_DETECTION_PARAMS,
                'dbscan_eps': dbscan_eps, 'min_cluster_pts': min_cluster_pts, 'min_hits': min_hits,
                'roi_abs_y': roi_abs_y, 'fps': fps, 'max_missed': max_missed,
                'moving_speed_thresh': moving_speed_thresh, 'truck_merge_dist': truck_merge_dist,
                'vehicle_gate': vehicle_gate, 'vehicle_min_length': vehicle_min_length,
                'vehicle_min_points': vehicle_min_points,
                'adaptive_eps': adaptive_eps, 'aeps0': aeps0, 'aeps_k': aeps_k,
                'aeps_min': aeps_min, 'aeps_max': aeps_max,
                'suppress_static': suppress_static, 'static_min_frames': static_min_frames,
                'static_max_speed': static_max_speed, 'strong_pts': strong_pts,
            }
            st.info(f"Processing {len(filtered_files)} files from: {filtered_pcd_dir}...")
            progress_bar = st.progress(0, text="Starting...")
            def update_progress(current, total, message):
                progress_bar.progress(current / total, text=f"{message}: {current}/{total} frames")
            try:
                results, error_message = run_detection_and_tracking(filtered_pcd_dir, output_dir, params, update_progress)
                if error_message:
                    st.error(error_message)
                else:
                    # Add the sorted lists of files to the results for the UI
                    results['original_pcd_files'] = original_files
                    results['params'] = params # Pass params to visualization
                    results['sensor'] = _sensor      # tag which sensor/source these
                    results['source'] = _src         # results are for (eval cross-checks)
                    st.session_state.detection_results = results
                    st.success(f"✅ Processing finished! Found {len(results['pcd_files'])} frames "
                               f"({_sensor_label} · {_src_label}). Use the slider below to visualize.")
                progress_bar.empty()
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                progress_bar.empty()

# --- Visualization Section ---
if st.session_state.detection_results:
    st.divider()
    st.subheader("🖼️ Interactive Visualization")
    results = st.session_state.detection_results
    # Warn if the toggles were changed since these results were computed — the
    # viewer/eval use the in-memory results, so a stale sensor/source is a footgun.
    _r_sensor, _r_source = results.get('sensor'), results.get('source')
    if (_r_sensor and _r_sensor != _sensor) or (_r_source and _r_source != _src):
        st.warning(f"⚠️ Showing results for **{(_r_sensor or '?').capitalize()} · "
                   f"{'Cropped' if _r_source == 'cropped' else 'Full'}**, but the toggles are now set to "
                   f"**{_sensor_label} · {_src_label}**. Click **Start Detection** to recompute for the "
                   "current selection (Evaluation scores whatever is loaded here).")

    # Update camera eye in results so visualization uses current UI settings
    results['camera_eye'] = {'x': eye_x, 'y': eye_y, 'z': eye_z}

    # --- Wrong-Way Driving (WWD) analysis ---
    st.divider()
    lanes = load_lane_config()
    if not lanes:
        st.warning("No lane configuration found (config/lanes.geojson). WWD is disabled.")
    else:
        if not lanes_calibrated(lanes):
            st.warning(
                "Lane geometry in config/lanes.geojson is NOT calibrated (placeholder "
                "headings). WWD will run, but results are not trustworthy until you set "
                "the real per-lane headings — see README → 'Calibrating lane geometry'."
            )
        with st.expander("🚨 Wrong-Way Driving Detection", expanded=False):
            wc1, wc2, wc3, wc4 = st.columns(4)
            ww_angle = wc1.slider("Angle vs. flow (deg)", 90.0, 180.0, 120.0, 5.0,
                                  help="How far against the expected lane direction counts as wrong-way.")
            ww_speed = wc2.slider("Min speed (m/s)", 0.5, 10.0, 2.0, 0.5,
                                  help="Below this, heading is unreliable.")
            ww_frames = wc3.slider("Sustained frames", 1, 30, 5, 1,
                                   help="Consecutive flagged frames required.")
            ww_disp = wc4.slider("Min displacement (m)", 0.0, 20.0, 3.0, 0.5,
                                 help="Net travel over the flagged span.")
            wc5, wc6 = st.columns(2)
            ww_exempt = wc5.checkbox("Exempt junction turns", value=True,
                                     help="Ignore wrong-way inside the intersection (where lane boxes "
                                          "overlap and turning is legal). Fixes turning vehicles being "
                                          "misflagged.")
            ww_consist = wc6.slider("Min heading steadiness", 0.0, 1.0, 0.85, 0.05,
                                    help="A real wrong-way vehicle holds a steady heading; a turn sweeps "
                                         "through headings. Higher = reject turns more aggressively (1.0 = "
                                         "perfectly steady).")
        ww_params = {"angle_thresh_deg": ww_angle, "min_speed": ww_speed,
                     "min_frames": ww_frames, "min_displacement_m": ww_disp,
                     "exempt_junction": ww_exempt, "min_heading_consistency": ww_consist}
        wwd_result = detect_wrong_way(results['det_frames'], lanes, ww_params)  # annotates det_frames in place
        rows = summarize_wrong_way(wwd_result, fps=results.get('params', {}).get('fps', 10.0))
        n_ww = len(wwd_result['wrong_way_tids'])
        if n_ww == 0:
            st.success("No wrong-way vehicles detected with the current settings.")
        else:
            st.error(f"⚠ {n_ww} wrong-way vehicle(s) detected.")
            st.dataframe(rows, use_container_width=True)

    # --- Compact one-line frame navigation ---
    n_frames = len(results['pcd_files'])
    frame_idx, playing, play_delay = vu.nav_row("odt_frame", n_frames, "odt")

    # GT index for the active sensor (scorable set; same one Evaluation scores).
    _gt_dir = _ds.gt_dir_for_input(original_pcd_dir)

    @st.cache_data(show_spinner=False)
    def _gt_index(gt_dir):
        idx = {}
        if gt_dir and os.path.isdir(gt_dir):
            for f in glob.glob(os.path.join(gt_dir, "*.json")):
                idx["_".join(os.path.basename(f).split("_")[:2])] = f
        return idx
    gt_index = _gt_index(_gt_dir)
    has_gt = bool(gt_index)

    # --- Collapsible layers & overlays (bulk on/off; matches Background Filtering) ---
    have_lanes = bool(lanes)
    toggle_defaults = {
        "odt_orig": True, "odt_fg": False, "odt_objects": True, "odt_gt": False,
        "odt_missed": False, "odt_lanes": have_lanes, "odt_road": True, "odt_roi": False,
        "odt_excl": False, "odt_sensors": True, "odt_height": False, "odt_topdown": False,
    }
    vu.ensure_toggle_defaults(toggle_defaults)
    # "All" also turns Height on (per request); top-down stays manual.
    overlay_keys = ["odt_orig", "odt_fg", "odt_objects", "odt_gt", "odt_missed", "odt_lanes",
                    "odt_road", "odt_roi", "odt_excl", "odt_sensors", "odt_height"]
    with st.expander("🎛️ Layers & overlays", expanded=False):
        vu.bulk_toggle_buttons(overlay_keys, "odt_bulk", rerun_scope="app")
        r1 = st.columns(4)
        show_orig = r1[0].toggle("⚪ Point cloud", key="odt_orig", help="The original cloud (grey/Turbo).")
        show_fg = r1[1].toggle("🟠 Foreground", key="odt_fg",
                               help="The filtered foreground points the detector actually ran on "
                                    "(orange — distinct from the red object markers and the cyan CAR "
                                    "GT boxes).")
        show_objects = r1[2].toggle("🔴 Objects/tracks", key="odt_objects",
                                    help="Detection markers, wrong-way diamonds, heading arrows, trails.")
        show_gt = r1[3].toggle("🏷️ GT boxes", key="odt_gt", disabled=not has_gt,
                               help="Overlay this frame's ground-truth boxes (category-coloured) to "
                                    "eyeball detections against truth." if has_gt
                                    else "No ground truth found for this sensor.")
        r2 = st.columns(4)
        show_missed = r2[0].toggle("❌ Missed GT", key="odt_missed", disabled=not has_gt,
                                   help="Outline the GT boxes with NO detection within 2 m (false "
                                        "negatives) in bright red, so you can see which objects were "
                                        "missed." if has_gt else "No ground truth found for this sensor.")
        show_lanes = r2[1].toggle("🟦 Lanes", key="odt_lanes", disabled=not have_lanes,
                                  help="Lane polygons + expected-direction arrows."
                                       if have_lanes else "No calibrated lanes for this dataset.")
        show_road = r2[2].toggle("🛣️ Road", key="odt_road", help="Road polygon outline.")
        show_roi = r2[3].toggle("🔵 ROI", key="odt_roi", help="Research region boundary.")
        r3 = st.columns(4)
        show_excl = r3[0].toggle("🟣 Exclusion", key="odt_excl", help="Foreground-exclusion rects.")
        sensors_on = r3[1].toggle("📍 LiDAR", key="odt_sensors", help="Mark the LiDAR position(s).")
        color_h = r3[2].toggle("🌈 Height", key="odt_height", help="Colour the cloud by z (Turbo).")
        top_down = r3[3].toggle("⬇️ Top-down", key="odt_topdown",
                                help="Snap the camera straight down (bird's-eye).")
        h_span = st.slider("Height span (m)", 1.5, 12.0, 4.0, 0.5, key="odt_hspan",
                           help="Colour spreads over this many metres above ground.") if color_h else 4.0

    results['lanes'] = lanes
    results['show_lanes'] = bool(show_lanes and have_lanes)
    results['top_down'] = top_down
    sensors = reg.lidar_markers(_ds, _sensor) if sensors_on else None

    st.subheader("3D Point Cloud View")
    original_pcd_path = results['original_pcd_files'][frame_idx]
    if not os.path.exists(original_pcd_path):
        st.error(f"Original PCD file not found for this frame: {original_pcd_path}")
    else:
        # Load GT once if it's needed for display, the missed overlay, or the metric.
        gt_objs = None
        if (show_gt or show_missed) and has_gt:
            import label_projection as lp
            gp = gt_index.get("_".join(os.path.basename(original_pcd_path).split("_")[:2]))
            if gp:
                gt_objs = lp.load_objects(gp)

        # Match detections -> GT once (BEV centre, 2 m) for both the missed overlay
        # and the per-frame coverage metric.
        covered = ng = None
        missed_objs = None
        if gt_objs is not None:
            from evaluation import _match_frame
            det_xy = [{"cx": d["cx"], "cy": d["cy"]} for d in results["det_frames"][frame_idx]]
            gt_xy = [{"cx": o["val"][0], "cy": o["val"][1]} for o in gt_objs]
            matches, _, _, _ = _match_frame(det_xy, gt_xy, 2.0)
            matched_gt = {j for _, j, _ in matches}
            ng = len(gt_objs)
            covered = len(matched_gt)
            missed_objs = [gt_objs[j] for j in range(ng) if j not in matched_gt]

        # filtered foreground for this frame (the cloud detection ran on)
        _fg_files = results.get('pcd_files', [])
        fg_path = _fg_files[frame_idx] if (show_fg and frame_idx < len(_fg_files)) else None
        fig = create_3d_figure(results, frame_idx, original_pcd_path,
                               color_by_height=color_h, height_span=h_span,
                               show_original=show_orig, show_road=show_road,
                               show_roi=show_roi, show_excl=show_excl,
                               show_objects=show_objects, sensors=sensors,
                               gt_objs=(gt_objs if show_gt else None),
                               missed_objs=(missed_objs if show_missed else None),
                               foreground_path=fg_path)
        st.plotly_chart(fig, use_container_width=True, height=800)
        if show_gt or show_missed:
            if not has_gt:
                st.caption("🏷️ No GT found for this sensor.")
            elif gt_objs is None:
                st.caption("🏷️ No GT for this frame.")
            else:
                pct = (100.0 * covered / ng) if ng else 0.0
                st.caption(f"🏷️ GT: {ng} boxes · **{covered}/{ng} detected** ({pct:.0f}%) · "
                           f"❌ {ng - covered} missed — a GT box counts as detected if a track centre is "
                           f"within 2 m  ·  `{os.path.basename(_gt_dir.rstrip('/'))}`")

        # Static-detection stat: a track that barely moves over its whole life (lifetime
        # max speed under a noise-tolerant floor) is static — pole / clutter / parked, not a
        # mover. Uses 1.5 m/s (≈ walking pace) so Kalman velocity noise on a truly-static
        # object doesn't flip it to "moving". With GT, how many are unmatched (= static FP).
        _cur = results["det_frames"][frame_idx]
        if _cur:
            _STATIC_V = 1.5
            _lm = {}
            for _dets in results["det_frames"]:
                for _d in _dets:
                    _t = _d.get("tid")
                    _lm[_t] = max(_lm.get(_t, 0.0), float(_d.get("speed", 0.0)))
            _is_static = lambda d: _lm.get(d.get("tid"), 9.9) < _STATIC_V
            _n_static = sum(1 for d in _cur if _is_static(d))
            _cap = f"🟣 {_n_static}/{len(_cur)} static (poles/parked)"
            if gt_objs is not None:
                _matched_det = {i for i, _, _ in matches}
                _n_sfp = sum(1 for i, d in enumerate(_cur) if i not in _matched_det and _is_static(d))
                _cap += f" · {_n_sfp} unmatched FP"
            st.caption(_cap)

    # Auto-play: advance one frame and rerun until the end or until paused.
    if playing and frame_idx < n_frames - 1:
        import time
        time.sleep(float(play_delay))
        st.session_state.odt_frame = frame_idx + 1
        st.rerun()

    # --- Animation Generation Section ---
    st.divider()
    st.subheader("🎥 Tracking Animation")

    animation_path = os.path.join(output_dir, "tracking_animation.gif")

    # Always try to display the animation if it exists in the output directory
    if os.path.exists(animation_path):
        st.image(animation_path, caption="Tracking animation (last generated)")
    else:
        st.info("No animation generated yet. Click the button below to create one.")

    if st.button("🎬 Generate / Update Tracking Animation", use_container_width=True):
        st.info("Starting animation generation... This may take a few minutes.")
        progress_bar = st.progress(0, text="Initializing...")

        def animation_progress_callback(current, total):
            progress_bar.progress(current / total, text=f"Processing frame {current}/{total}")

        try:
            generate_tracking_animation(results, animation_path, animation_progress_callback, max_frames=max_frames_to_animate)
            st.success(f"✅ Animation successfully saved to: {animation_path}")
            st.balloons()
            # Force a rerun to display the new animation immediately
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during animation generation: {e}")
        finally:
            progress_bar.empty()
