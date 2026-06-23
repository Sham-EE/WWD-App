import os
import time

import streamlit as st

import dataset_manager as dm
import road_viewer as rv
import label_projection as lp
import lidar_viewer as lv
import dataset_prep as dp
import registration as reg
import viewer_ui as vu

st.set_page_config(layout="wide", page_title="Visualizer")
st.title("🎬 Visualizer")

ds = dm.get_active()
st.caption(f"📂 Dataset: **{ds.name}**")

with st.expander("📁 Input folders", expanded=False):
    images_root = st.text_input("Images folder (one subfolder per camera)", value=ds.images_dir)
    label_dir = st.text_input("OpenLABEL labels folder", value=ds.gt_dir)
    pcd_dir = st.text_input("Point-cloud folder", value=ds.pcd_dir)

labels = rv.list_by_frame(label_dir, [".json"])
pcds = rv.list_by_frame(pcd_dir, [".pcd"])


@st.cache_data(show_spinner=False)
def _all_centers(label_paths):
    """Per-frame {object_id: (x,y,z)} for the whole sequence (cached)."""
    return [lp.frame_centers(p) for p in label_paths]


@st.cache_data(show_spinner=False, max_entries=128)
def _load_pts(pcd_path, max_pts):
    """Cached point load so stepping frames is instant (no reload flicker)."""
    return lv.load_points(pcd_path, max_points=max_pts)


# ======================= Camera (Road Viewer) tab =======================
def render_camera_tab():
    cameras = rv.list_cameras(images_root)
    if len(cameras) < 1:
        st.warning(f"No camera subfolders found in `{images_root}`.")
        return
    have_labels = len(labels) > 0

    c1, c2, c3 = st.columns([1, 1, 1.4])

    def _cam_idx(token, fallback):
        for k, cam in enumerate(cameras):
            if token in cam:
                return k
        return fallback

    left_default = _cam_idx("south2", 0)
    right_default = _cam_idx("south1", 1 if len(cameras) > 1 else 0)
    left_cam = c1.selectbox("Left camera", cameras, index=left_default)
    right_cam = c2.selectbox("Right camera", cameras, index=right_default)

    modes = {"Raw camera": None, "Bounding boxes (3D)": "box3d", "Boxes + point cloud": "point_cloud"}
    variant_labels = list(modes) if have_labels else ["Raw camera"]
    variant_label = c3.radio("Visualization", variant_labels, index=(1 if have_labels else 0),
                             horizontal=True,
                             help="Box/point-cloud overlays are generated from the OpenLABEL labels + "
                                  "calibration and cached under the dataset's outputs/rendered/.")
    mode = modes[variant_label]
    if not have_labels:
        st.info(f"No OpenLABEL label files in `{label_dir}` — only raw images can be shown.")

    color_mode, point_size = "by_category", 2
    track_hist, hist_window, trail_width = False, 30, 8
    pc_full = False
    if mode:
        oc1, oc2, oc3 = st.columns([1, 1.4, 1])
        color_mode = oc1.radio("Box colour", ["by_category", "by_track_id"], horizontal=True,
                               help="by_category: colour encodes the class. by_track_id: each object "
                                    "gets its own colour to follow it across frames.")
        if mode == "point_cloud":
            point_size = oc2.select_slider("Point size", [1, 2, 3], value=2)
        if oc3.button("♻️ Regenerate (clear cache)", help="Delete cached overlays and re-render."):
            import shutil
            shutil.rmtree(os.path.join(ds.outputs_dir, "rendered"), ignore_errors=True)
            st.session_state.road_video = None
            st.rerun()
        th1, th2, th3 = st.columns([1.3, 1, 1])
        track_hist = th1.checkbox("🛤️ Track history (trails)", value=False,
                                  help="Draw each object's recent path as a tapering trail (coloured to "
                                       "match its box), computed from preceding frames.")
        if track_hist:
            hist_window = th2.slider("Trail length (frames)", 5, 80, 30, 5)
            trail_width = th3.slider("Trail thickness (px)", 2, 24, 8, 1)
        if mode == "point_cloud":
            pc_full = st.radio("Projected point cloud", ["Cropped (road)", "Full (uncropped)"],
                               horizontal=True, key="cam_pc",
                               help="Project the road-cropped cloud or the full raw south cloud "
                                    "(full reaches distant structures).") == "Full (uncropped)"

    raw_left = rv.frames_for(images_root, left_cam, "raw")
    raw_right = rv.frames_for(images_root, right_cam, "raw")
    # which clouds to project for 'Boxes + point cloud'
    pc_list = rv.list_by_frame(ds.raw_lidar_south_dir, [".pcd"]) if (mode == "point_cloud" and pc_full) else pcds
    counts = [len(raw_left), len(raw_right)]
    if mode:
        counts.append(len(labels))
    if mode == "point_cloud":
        counts.append(len(pc_list))
    n = min(counts) if counts else 0
    if n == 0:
        st.warning("Not enough synchronized frames.")
        return
    st.caption(f"{n} frames · **{left_cam}** (left) ↔ **{right_cam}** (right) · “{variant_label}”")

    left_id = lp.camera_id_from_image(raw_left[0])
    right_id = lp.camera_id_from_image(raw_right[0])
    all_centers = _all_centers(tuple(labels)) if (mode and track_hist) else None

    def _histories(i):
        if not (mode and track_hist and all_centers):
            return None
        cur = all_centers[i]
        lo = max(0, i - hist_window)
        out = {}
        for oid in cur:
            seq = [all_centers[f][oid] for f in range(lo, i + 1) if oid in all_centers[f]]
            if len(seq) >= 2:
                out[oid] = seq
        return out

    def _cache_dir(cam):
        ps = f"_ps{point_size}" if mode == "point_cloud" else ""
        pc = (f"_pc{'full' if pc_full else 'crop'}") if mode == "point_cloud" else ""
        th = f"_th{hist_window}w{trail_width}" if track_hist else ""
        return os.path.join(ds.outputs_dir, "rendered", cam, f"{mode}_{color_mode}{ps}{pc}{th}_{lp.RENDER_VERSION}")

    def _render(i, cam, cam_id, raw):
        if not mode:
            return raw[i]
        return lp.render_cached(raw[i], labels[i], cam_id, mode, _cache_dir(cam),
                                color_mode=color_mode,
                                pcd_path=(pc_list[i] if mode == "point_cloud" else None),
                                point_size=point_size, histories=_histories(i), trail_width=trail_width)

    st.session_state.setdefault("road_frame", 0)

    @st.fragment
    def _viewer():
        i, playing, play_delay = vu.nav_row("road_frame", n, "road")

        left_img = _render(i, left_cam, left_id, raw_left)
        right_img = _render(i, right_cam, right_id, raw_right)
        lc, rc = st.columns(2)
        lc.image(left_img, use_container_width=True, caption=f"{left_cam} (left) · frame {i+1}/{n}")
        rc.image(right_img, use_container_width=True, caption=f"{right_cam} (right) · frame {i+1}/{n}")

        if playing and i < n - 1:
            time.sleep(float(play_delay))
            st.session_state.road_frame = i + 1
            st.rerun(scope="fragment")

    _viewer()

    # ---- video ----
    st.divider()
    st.subheader("🎬 Generate road video")
    if not rv.mp4_available():
        st.caption("ℹ️ MP4 needs `imageio-ffmpeg`; otherwise an animated **GIF** is produced.")
    g1, g2, g3 = st.columns(3)
    v_fps = g1.slider("Video FPS", 1, 30, 10, 1)
    v_height = g2.select_slider("Frame height (px)", [360, 480, 540, 720], value=480)
    v_max = g3.number_input("Max frames (0 = all)", 0, n, 0)
    video_dir = os.path.join(ds.outputs_dir, "road_videos")
    tag = mode or "raw"
    basename = f"road_{left_cam}_{right_cam}_{tag}_{color_mode if mode else 'plain'}"
    st.session_state.setdefault("road_video", None)
    if st.button("🎬 Generate side-by-side video", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="Preparing frames…")
        m = n if v_max in (0, None) else min(int(v_max), n)
        try:
            if mode:
                lefts, rights = [], []
                for k in range(m):
                    lefts.append(_render(k, left_cam, left_id, raw_left))
                    rights.append(_render(k, right_cam, right_id, raw_right))
                    bar.progress(0.5 * (k + 1) / m, text=f"Rendering frame {k+1}/{m}")
            else:
                lefts, rights = raw_left[:m], raw_right[:m]
            path, kind = rv.generate_side_by_side_video(
                lefts, rights, video_dir, basename, fps=int(v_fps), height=int(v_height),
                max_frames=m, progress=lambda c, t: bar.progress(0.5 + 0.5 * c / t, text=f"Encoding {c}/{t}"))
            bar.empty()
            st.session_state.road_video = (path, kind)
            st.success(f"Saved {kind.upper()} → `{path}`"
                       + ("  (GIF fallback — install imageio-ffmpeg for MP4)" if kind == "gif" else ""))
        except Exception as e:
            bar.empty()
            st.error(f"Video generation failed: {e}")
    rvid = st.session_state.get("road_video")
    if rvid and os.path.exists(rvid[0]):
        path, kind = rvid
        st.video(path) if kind == "mp4" else st.image(path, caption="Road animation (GIF)")
        st.caption(f"Last generated: `{path}`")


# ======================= LiDAR labels (3D) tab =======================
def render_lidar_tab():
    st.markdown("LiDAR scan + ground-truth 3D boxes — **Bird's Eye** and **Side** view (rotate/zoom each).")
    n = min(len(labels), len(pcds))
    if n == 0:
        st.warning("Need both OpenLABEL labels and point clouds (set them in **Input folders** above).")
        return
    o1, o2, o3, o4 = st.columns(4)
    color_mode = o1.radio("Box colour", ["by_category", "by_track_id"], horizontal=True, key="lv_color")
    max_pts = o2.select_slider("Points shown", [10000, 20000, 30000, 50000], value=20000, key="lv_pts")
    show_road = o3.checkbox("🛣️ Road outline", value=True, key="lv_road",
                            help="Green road boundary from site_geometry.json.")
    show_sensors = o4.checkbox("📍 LiDAR", value=True, key="lv_sensors",
                               help="Mark the LiDAR position + nadir (inferred from the point-cloud folder).")
    road = dp.road_polygon(0.0) if show_road else None
    # infer which sensor the point-cloud folder belongs to so markers land right
    _pl = (pcd_dir or "").lower()
    _sensor = "registered" if "registered" in _pl else ("north" if "north" in _pl else "south")
    sensors = reg.lidar_markers(ds, _sensor) if show_sensors else None
    st.session_state.setdefault("lidar_frame", 0)

    @st.fragment
    def _viewer3d():
        i, playing, delay = vu.nav_row("lidar_frame", n, "lv")

        pts = _load_pts(pcds[i], int(max_pts))
        objs = lp.load_objects(labels[i])
        with st.container(height=600):
            cbev, cside = st.columns(2)
            with cbev:
                st.markdown("**Bird's Eye View**")
                st.plotly_chart(lv.build_figure(pts, objs, color_mode, "bev", height=520,
                                                road_poly=road, sensors=sensors),
                                use_container_width=True, key="lv_bev")
            with cside:
                st.markdown("**Side View**")
                st.plotly_chart(lv.build_figure(pts, objs, color_mode, "side", height=520,
                                                road_poly=road, sensors=sensors),
                                use_container_width=True, key="lv_side")
        st.caption(f"Frame {i+1}/{n} · {len(objs)} labelled objects · {len(pts):,} points shown")

        if playing and i < n - 1:
            time.sleep(float(delay))
            st.session_state.lidar_frame = i + 1
            st.rerun(scope="fragment")

    _viewer3d()


tab_cam, tab_lidar = st.tabs(["🎥 Road Viewer (cameras)", "🧊 LiDAR labels (3D)"])
with tab_cam:
    render_camera_tab()
with tab_lidar:
    render_lidar_tab()
