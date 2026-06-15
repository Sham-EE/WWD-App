import streamlit as st
import os
import glob

from detection_logic import run_detection_and_tracking, sorted_by_frame_index
from visualization import create_3d_figure, generate_tracking_animation
from wwd_detection import load_lane_config, lanes_calibrated, detect_wrong_way, summarize_wrong_way

st.set_page_config(layout="wide", page_title="Object Detection and Tracking")

st.title("📦 Object Detection and Tracking")

# Initialize session state
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

# --- Input and Output Paths ---
st.subheader("📁 Input and Output")

filtered_pcd_dir = st.text_input(
    "Enter the path to the FILTERED PCD files (for detection):",
    value=r"outputs/background_filtering"
)

original_pcd_dir = st.text_input(
    "Enter the path to the ORIGINAL PCD files (for visualization):",
    value=r"data/point_clouds/cropped/cropped_pcd"
)

output_dir = "outputs/object_detection"

# --- Parameters ---
st.subheader("⚙️ Algorithm Parameters")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Clustering and Detection")
    dbscan_eps = st.slider("DBSCAN Epsilon (eps)", 0.1, 5.0, 2.0, 0.1, help="Controls cluster density.")
    min_cluster_pts = st.slider("Min Cluster Points", 1, 50, 1, 1, help="Minimum points to form a cluster.")
    min_hits = st.slider("Min Temporal Hits", 1, 10, 2, 1, help="Frames a candidate must exist to be confirmed.")

with col2:
    st.markdown("#### Tracking and Association")
    fps = st.slider("Frames Per Second (FPS)", 1.0, 30.0, 10.0, 0.5, help="Data frame rate for velocity calculation.")
    max_missed = st.slider("Max Missed Frames", 0, 20, 5, 1, help="Frames to keep a track alive without detection.")
    moving_speed_thresh = st.slider("Moving Speed Threshold (m/s)", 0.0, 10.0, 3.0, 0.1, help="Speed above which an object is 'moving'.")
    roi_abs_y = st.slider("ROI Absolute Y (m)", 5.0, 100.0, 40.0, 1.0, help="Y-coordinate processing range.")

st.markdown("#### Visualization")
col_v1, col_v2, col_v3 = st.columns(3)
with col_v1:
    eye_x = st.number_input("Camera Eye X", value=0.8, step=0.05)
with col_v2:
    eye_y = st.number_input("Camera Eye Y", value=0.8, step=0.05)
with col_v3:
    eye_z = st.number_input("Camera Eye Z", value=0.8, step=0.05)

max_frames_to_animate = st.number_input("Max Frames to Animate (0 for all)", min_value=0, max_value=2000, value=0, help="Limit the number of frames to process for the animation to save time.")

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
            params = {
                'dbscan_eps': dbscan_eps, 'min_cluster_pts': min_cluster_pts, 'min_hits': min_hits,
                'roi_abs_y': roi_abs_y, 'yaw_bias_deg': -90.0,
                'fps': fps, 'max_missed': max_missed, 'moving_speed_thresh': moving_speed_thresh,
                'merge_dist': 2.5, 'yaw_merge_deg': 15.0, 'truck_len_thresh': 7.0, 'truck_merge_dist': 10.0,
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
                    st.session_state.detection_results = results
                    st.success(f"✅ Processing finished! Found {len(results['pcd_files'])} frames. Use the slider below to visualize.")
                progress_bar.empty()
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                progress_bar.empty()

# --- Visualization Section --- 
if st.session_state.detection_results:
    st.divider()
    st.subheader("🖼️ Interactive Visualization")
    results = st.session_state.detection_results

    # Update camera eye in results so visualization uses current UI settings
    results['camera_eye'] = {'x': eye_x, 'y': eye_y, 'z': eye_z}

    # --- Wrong-Way Driving (WWD) analysis ---
    st.divider()
    st.subheader("🚨 Wrong-Way Driving Detection")
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
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            ww_angle = st.slider("Angle vs. flow (deg)", 90.0, 180.0, 120.0, 5.0,
                                 help="How far against the expected lane direction counts as wrong-way.")
        with wc2:
            ww_speed = st.slider("Min speed (m/s)", 0.5, 10.0, 2.0, 0.5,
                                 help="Below this, heading is unreliable.")
        with wc3:
            ww_frames = st.slider("Sustained frames", 1, 30, 5, 1,
                                  help="Consecutive flagged frames required.")
        with wc4:
            ww_disp = st.slider("Min displacement (m)", 0.0, 20.0, 3.0, 0.5,
                                help="Net travel over the flagged span.")
        wc5, wc6 = st.columns(2)
        with wc5:
            ww_exempt = st.checkbox("Exempt junction turns", value=True,
                                    help="Ignore wrong-way inside the intersection (where lane boxes "
                                         "overlap and turning is legal). Fixes turning vehicles being "
                                         "misflagged.")
        with wc6:
            ww_consist = st.slider("Min heading steadiness", 0.0, 1.0, 0.85, 0.05,
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

    # --- View toggles (lane overlay + bird's-eye) ---
    vc1, vc2 = st.columns(2)
    if lanes:
        results['lanes'] = lanes
        results['show_lanes'] = vc1.checkbox(
            "🛣️ Show lane directions", value=True,
            help="Overlay the calibrated lane polygons and their expected travel-direction arrows.")
    else:
        results['show_lanes'] = False
    results['top_down'] = vc2.checkbox(
        "⬇️ Top-down (bird's-eye) view", value=False,
        help="Snap the camera straight down to verify lane alignment against the road.")

    # --- Frame playback controls (steps the live viewer; no animation render) ---
    n_frames = len(results['pcd_files'])
    if 'odt_frame' not in st.session_state:
        st.session_state.odt_frame = 0
    st.session_state.odt_frame = max(0, min(st.session_state.odt_frame, n_frames - 1))

    pc = st.columns([1, 1, 1, 1, 1.4, 3])
    if pc[0].button("⏮ First", use_container_width=True):
        st.session_state.odt_frame = 0; st.rerun()
    if pc[1].button("◀ Prev", use_container_width=True):
        st.session_state.odt_frame = max(0, st.session_state.odt_frame - 1); st.rerun()
    if pc[2].button("Next ▶", use_container_width=True):
        st.session_state.odt_frame = min(n_frames - 1, st.session_state.odt_frame + 1); st.rerun()
    if pc[3].button("Last ⏭", use_container_width=True):
        st.session_state.odt_frame = n_frames - 1; st.rerun()
    playing = pc[4].toggle("▶ Play", value=False, help="Auto-advance frames in the live viewer.")
    play_delay = pc[5].slider("Play delay (s/frame)", 0.0, 1.0, 0.2, 0.05)

    frame_idx = st.slider("Select Frame", 0, max(n_frames - 1, 1), st.session_state.odt_frame)
    st.session_state.odt_frame = frame_idx

    st.subheader("3D Point Cloud View")
    original_pcd_path = results['original_pcd_files'][frame_idx]
    if not os.path.exists(original_pcd_path):
        st.error(f"Original PCD file not found for this frame: {original_pcd_path}")
    else:
        fig = create_3d_figure(results, frame_idx, original_pcd_path)
        st.plotly_chart(fig, use_container_width=True, height=800)

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
