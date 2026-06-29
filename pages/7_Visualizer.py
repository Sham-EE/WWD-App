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
import geo_reference as geo
import nav

st.set_page_config(layout="wide", page_title="Visualizer")
nav.render_sidebar()

import re as _re
def _short_cam(cam):
    """s110_camera_basler_south1_8mm -> 'south1' (for compact folder/file/UI names)."""
    m = _re.search(r"(south|north|east|west)\d*", cam or "")
    return m.group(0) if m else (cam or "cam")

_MODE_SHORT = {"box3d": "box", "point_cloud": "pcd"}      # render mode -> short token
_COLOR_SHORT = {"by_category": "cat", "by_track_id": "track"}
st.title("🎬 Visualizer")

ds = dm.get_active()
st.caption(f"📂 Dataset: **{ds.name}**")

# Sensor + input-cloud toggle — drives BOTH the Road Viewer (camera projection)
# and the LiDAR labels (3D) tabs. Registered is written in the south frame, so it
# reuses the south GT boxes + south camera calibration; south/north use their own.
_sc, _ic, _gc = st.columns(3)
_sensor_label = _sc.radio("Sensor", ["Registered", "South", "North"], horizontal=True,
                          key="pipeline_sensor",
                          help="Which LiDAR to visualize. Registered = fused south+north (written in the "
                               "south frame, so the south GT boxes and south camera calibration apply to "
                               "it directly). South / North use each sensor's own labels + calibration. "
                               "Shared with the Filtering / Detection / Evaluation pages.")
_sensor = _sensor_label.lower()
_src_label = _ic.radio("Input cloud", ["Cropped (road)", "Full (uncropped)"], horizontal=True,
                       key="pipeline_source",
                       help="Cropped = road-clipped cloud; Full = the raw/fused cloud. Drives both the "
                            "3D view and the cloud projected onto the camera.")
_src = "cropped" if _src_label.startswith("Cropped") else "full"
_gt_label = _gc.radio("GT boxes", ["Scorable", "All (raw)"], horizontal=True, key="viz_gt_kind",
                      help="Scorable = only the boxes Dataset Prep kept (inside the ROI + enough LiDAR "
                           "points) — what Evaluation scores against. All (raw) = every annotated box, "
                           "including ones outside the ROI or with too few points to score.")
_gt_kind = "raw" if _gt_label.startswith("All") else "scorable"

# Resolve the cloud + labels for the selected sensor/source/GT-kind (same helpers
# the pipeline pages use). Registered resolves to the fused south∪north union.
_pcd_default = ds.input_pcd_for_sensor(_sensor, _src)
_label_default = ds.labels_dir_for(_sensor, _gt_kind)

with st.expander("📁 Input folders (advanced override)", expanded=False):
    images_root = st.text_input("Images folder (one subfolder per camera)", value=ds.images_dir,
                                key="viz_images")
    # keyed by sensor/source/kind so the fields re-follow the radios when you switch
    label_dir = st.text_input("OpenLABEL labels folder", value=_label_default,
                              key=f"viz_label_{_sensor}_{_src}_{_gt_kind}")
    pcd_dir = st.text_input("Point-cloud folder", value=_pcd_default,
                            key=f"viz_pcd_{_sensor}_{_src}")
st.caption(f"🛰️ **{_sensor_label} · {_src_label} · {_gt_label} GT** — cloud "
           f"`{os.path.basename(pcd_dir.rstrip('/'))}` · GT `{os.path.basename(label_dir.rstrip('/'))}`"
           + ("" if os.path.isdir(label_dir) else "  ·  ⚠️ GT folder not found"))

labels = rv.list_by_frame(label_dir, [".json"])
pcds = rv.list_by_frame(pcd_dir, [".pcd"])


def _fkey(path):
    """Leading <ts1>_<ts2> token shared by a frame's cloud + its label(s)."""
    return "_".join(os.path.basename(path).split("_")[:2])


@st.cache_data(show_spinner=False)
def _box_counts(label_dir):
    """{frame-key: #GT boxes} for a label folder (cached so stepping is instant)."""
    import glob
    out = {}
    if label_dir and os.path.isdir(label_dir):
        for f in glob.glob(os.path.join(label_dir, "*.json")):
            try:
                out[_fkey(f)] = len(lp.load_objects(f))
            except Exception:
                out[_fkey(f)] = 0
    return out


# Per-frame raw vs scorable box counts for the active sensor (both shown so you can
# see, per frame, how many of the annotated boxes survive the scorable filter).
_raw_counts = _box_counts(ds.labels_dir_for(_sensor, "raw"))
_scorable_counts = _box_counts(ds.labels_dir_for(_sensor, "scorable"))


def _box_count_str(ref_path):
    k = _fkey(ref_path)
    return f"🏷️ {_scorable_counts.get(k, 0)} scorable / {_raw_counts.get(k, 0)} raw boxes"


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
    left_cam = c1.selectbox("Left camera", cameras, index=left_default, format_func=_short_cam)
    right_cam = c2.selectbox("Right camera", cameras, index=right_default, format_func=_short_cam)

    modes = {"Raw camera": None, "Bounding boxes (3D)": "box3d", "Boxes + point cloud": "point_cloud"}
    variant_labels = list(modes) if have_labels else ["Raw camera"]
    variant_label = c3.radio("Visualization", variant_labels, index=(1 if have_labels else 0),
                             horizontal=True,
                             help="Box/point-cloud overlays are generated from the OpenLABEL labels + "
                                  "calibration and cached under the dataset's outputs/visualizer/rendered/.")
    mode = modes[variant_label]
    if not have_labels:
        st.info(f"No OpenLABEL label files in `{label_dir}` — only raw images can be shown.")

    color_mode, point_size = "by_category", 2
    track_hist, hist_window, trail_width = False, 30, 8
    if mode:
        oc1, oc2, oc3 = st.columns([1, 1.4, 1])
        color_mode = oc1.radio("Box colour", ["by_category", "by_track_id"], horizontal=True,
                               help="by_category: colour encodes the class. by_track_id: each object "
                                    "gets its own colour to follow it across frames.")
        if mode == "point_cloud":
            point_size = oc2.select_slider("Point size", [1, 2, 3], value=2)
        if oc3.button("♻️ Regenerate (clear cache)", help="Delete cached overlays and re-render."):
            import shutil
            shutil.rmtree(ds.rendered_dir, ignore_errors=True)
            st.session_state.road_video = None
            st.rerun()
        th1, th2, th3 = st.columns([1.3, 1, 1])
        track_hist = th1.checkbox("🛤️ Track history (trails)", value=False,
                                  help="Draw each object's recent path as a tapering trail (coloured to "
                                       "match its box), computed from preceding frames.")
        if track_hist:
            hist_window = th2.slider("Trail length (frames)", 5, 80, 30, 5)
            trail_width = th3.slider("Trail thickness (px)", 2, 24, 8, 1)
    raw_left = rv.frames_for(images_root, left_cam, "raw")
    raw_right = rv.frames_for(images_root, right_cam, "raw")
    # Cloud projected for 'Boxes + point cloud' = the sensor/source cloud picked
    # above (registered is in the south frame, so it projects with the south camera
    # calibration); for north it uses the north calibration carried in north labels.
    pc_list = pcds
    counts = [len(raw_left), len(raw_right)]
    if mode:
        counts.append(len(labels))
    if mode == "point_cloud":
        counts.append(len(pc_list))
    n = min(counts) if counts else 0
    if n == 0:
        st.warning("Not enough synchronized frames.")
        return
    st.caption(f"{n} frames · **{_short_cam(left_cam)}** (left) ↔ **{_short_cam(right_cam)}** (right) · “{variant_label}”")

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
        pc = (f"_{'full' if _src == 'full' else 'crop'}") if mode == "point_cloud" else ""
        th = f"_trail{hist_window}-{trail_width}" if track_hist else ""
        m = _MODE_SHORT.get(mode, mode); col = _COLOR_SHORT.get(color_mode, color_mode)
        # include sensor + GT kind so south/north/registered and scorable/raw renders
        # don't collide (same camera + mode, but different boxes + projected cloud).
        gt = "" if not mode else f"_{_gt_kind}"
        return os.path.join(ds.rendered_dir, _short_cam(cam), _sensor, f"{m}_{col}{ps}{pc}{th}{gt}")

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
        lc.image(left_img, use_container_width=True, caption=f"{_short_cam(left_cam)} (left) · frame {i+1}/{n}")
        rc.image(right_img, use_container_width=True, caption=f"{_short_cam(right_cam)} (right) · frame {i+1}/{n}")
        if mode and labels:
            st.caption(f"Frame {i+1}/{n} · {_box_count_str(labels[i])}")

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
    video_dir = ds.road_videos_dir
    tag = mode or "raw"
    basename = (f"road_{_short_cam(left_cam)}_{_short_cam(right_cam)}_{_sensor}_"
                f"{_MODE_SHORT.get(tag, tag)}_{_COLOR_SHORT.get(color_mode, color_mode) if mode else 'plain'}"
                + (f"_{_gt_kind}" if mode else ""))
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
        if kind == "mp4":
            st.video(path)
        else:
            st.image(path, caption="Road animation (GIF)")
        st.caption(f"Last generated: `{path}`")


# ======================= LiDAR labels (3D) tab =======================
def render_lidar_tab():
    st.markdown("LiDAR scan + ground-truth 3D boxes — **Bird's Eye** and **Side** view (rotate/zoom each).")
    n = min(len(labels), len(pcds))
    if n == 0:
        st.warning("Need both OpenLABEL labels and point clouds (set them in **Input folders** above).")
        return
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    color_mode = o1.radio("Box colour", ["by_category", "by_track_id"], horizontal=True, key="lv_color")
    max_pts = o2.select_slider("Points shown", [10000, 20000, 30000, 50000], value=20000, key="lv_pts")
    show_boxes = o3.checkbox("📦 Boxes", value=True, key="lv_boxes",
                             help="Show the 3D boxes + LiDAR markers.")
    show_road = o4.checkbox("🛣️ Road outline", value=True, key="lv_road",
                            help="Green road boundary from site_geometry.json.")
    show_sensors = o5.checkbox("📍 LiDAR", value=True, key="lv_sensors",
                               help="Mark the LiDAR position + nadir for the selected sensor.")
    show_hdmap = o6.checkbox("🗺️ HD map", value=True, key="lv_hdmap",
                             help="Overlay the dataset's real HD-map road network (lane_samples.json) "
                                  "at ground level — the dev-kit digital-twin look.")
    road = dp.road_polygon(0.0) if show_road else None
    # markers follow the Sensor toggle (registered = both LiDARs in the south frame)
    sensors = reg.lidar_markers(ds, _sensor) if show_sensors else None
    # registered cloud is written in the SOUTH frame, so HD-map lanes use south unless north.
    hdmap_lanes = (geo.hdmap_lanes_sensor_frame("north" if _sensor == "north" else "south", 130.0)
                   if show_hdmap else None)
    if show_hdmap and not hdmap_lanes:
        st.caption("ℹ️ HD-map overlay needs `map/lane_samples.json` (from the dev-kit's src/map/map.zip).")

    st.session_state.setdefault("lidar_frame", 0)

    @st.fragment
    def _viewer3d():
        i, playing, delay = vu.nav_row("lidar_frame", n, "lv", label="🎞️ Scene frame")

        pts = _load_pts(pcds[i], int(max_pts))
        objs = lp.load_objects(labels[i])
        st.plotly_chart(lv.build_figure(pts, objs if show_boxes else [], color_mode,
                                        height=820, road_poly=road,
                                        sensors=sensors if show_boxes else None,
                                        hdmap_lanes=hdmap_lanes),
                        use_container_width=True, key="lv_main")
        st.caption(f"Frame {i+1}/{n} · {len(objs)} shown ({_box_count_str(labels[i])}) · "
                   f"{len(pts):,} points")

        if playing and i < n - 1:
            time.sleep(float(delay))
            st.session_state.lidar_frame = i + 1
            st.rerun(scope="fragment")

    _viewer3d()

    # ---- 3D LiDAR video — Plotly/kaleido, WYSIWYG with the interactive preview ----
    st.divider()
    st.subheader("🎬 3D LiDAR video")
    st.caption("This preview uses the **same Plotly engine** as the render, so what you frame is exactly "
               "what you get — and zooming in won't streak the HD-map lines across the screen. **Orbit it "
               "with the mouse** to explore, then lock the camera with the sliders (the render uses the "
               "sliders). Found the look? Note the values and I'll hardcode them.")
    a1, a2, a3 = st.columns(3)
    c_az = a1.slider("Azimuth°", -180, 180, 45, key="vid3d_az",
                     help="Spin the camera around the scene (compass heading).")
    c_el = a2.slider("Elevation°", 0, 89, 30, key="vid3d_el",
                     help="Camera height above the ground plane. Low ≈ eye-level / camera-like.")
    c_roll = a3.slider("Roll°", -45, 45, 0, key="vid3d_roll", help="Tilt the horizon.")
    z1, z2, z3 = st.columns(3)
    c_zoom = z1.slider("Zoom", 0.4, 4.0, 1.0, 0.05, key="vid3d_zoom",
                       help="Higher = closer (moves the camera in). True 3D zoom — no distortion or glitch.")
    c_px = z2.slider("Pan X", -1.0, 1.0, 0.0, 0.05, key="vid3d_px",
                     help="Shift the look-at point left/right (normalized scene units).")
    c_py = z3.slider("Pan Y", -1.0, 1.0, 0.0, 0.05, key="vid3d_py",
                     help="Shift the look-at point forward/back (normalized scene units).")
    s1, s2, s3 = st.columns(3)
    v_fps = s1.slider("FPS", 1, 30, 10, key="vid3d_fps")
    v_pts = s2.select_slider("Points", [3000, 6000, 12000, 20000], value=6000, key="vid3d_pts")
    v_max = s3.number_input("Max frames (0 = all)", 0, n, 0, key="vid3d_max")

    camera = lv.orbit_camera(azimuth=c_az, elevation=c_el, zoom=c_zoom,
                             pan_x=c_px, pan_y=c_py, roll=c_roll)
    prev_i = min(int(st.session_state.get("lidar_frame", 0)), n - 1)
    _pp = _load_pts(pcds[prev_i], int(v_pts))
    _po = lp.load_objects(labels[prev_i])
    # uirevision keyed to the camera: moving a slider applies the new camera, but
    # stepping the scene frame (same camera) preserves any manual mouse-orbit.
    st.plotly_chart(
        lv.build_figure(_pp, _po, color_mode, height=560, road_poly=road,
                        sensors=sensors, hdmap_lanes=hdmap_lanes, camera=camera,
                        uirevision=f"vid_{c_az}_{c_el}_{c_roll}_{c_zoom}_{c_px}_{c_py}"),
        use_container_width=True, key="vid3d_preview")
    st.caption(f"Preview — frame {prev_i+1}/{n} · az {c_az}° · el {c_el}° · roll {c_roll}° · "
               f"zoom {c_zoom}× · pan ({c_px:g}, {c_py:g})  ·  the render uses this exact camera.")
    st.info("⏱️ kaleido renders ~1–3 s per frame (a headless browser per frame), so all "
            f"{n} frames can take several minutes. Set **Max frames** low to test the angle first.")

    if st.button("🎬 Generate 3D LiDAR video (kaleido)", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="Rendering 3D frames (kaleido)…")
        try:
            path, kind = lv.render_lidar_video_plotly(
                pcds[:n], labels[:n], ds.road_videos_dir, f"lidar3d_{_sensor}_{_src}",
                camera=camera, fps=int(v_fps), max_points=int(v_pts), color_mode=color_mode,
                road_poly=road, hdmap_lanes=hdmap_lanes, sensors=sensors, max_frames=int(v_max),
                progress=lambda c, t: bar.progress(c / t, text=f"Rendering {c}/{t} frames…"))
            st.session_state.lidar_video = (path, kind)
            bar.empty()
            st.success(f"Saved {kind.upper()} → `{path}`")
        except Exception as e:
            bar.empty()
            st.error(f"3D video render failed: {e}")

    vid = st.session_state.get("lidar_video")
    if vid and os.path.exists(vid[0]):
        if vid[1] == "mp4":
            st.video(vid[0])
        else:
            st.image(vid[0], caption="3D LiDAR (GIF)")
        with open(vid[0], "rb") as f:
            st.download_button("⬇️ Download 3D LiDAR video", f, file_name=os.path.basename(vid[0]),
                               key="dl_3d_vid")
        st.caption(f"Last generated: `{vid[0]}` · saved to the dataset's road_videos folder.")


# ======================= Real intersection (Google Maps) tab =======================
def render_real_tab():
    import streamlit.components.v1 as components
    st.markdown(f"The dataset's **real location** — a live, interactive Google Map of "
                f"**{geo.site_name()}**. Pan, zoom, switch Map/Satellite (bottom-left), or click a "
                "place like the **Jägerhof**.")
    lat, lon = geo.sensor_position_latlon("south") or geo.site_latlon()
    src = f"https://maps.google.com/maps?q={lat},{lon}&t=h&z=18&hl=en&output=embed"   # t=h → satellite
    components.html(
        f'<iframe width="100%" height="660" style="border:0;border-radius:10px" loading="lazy" '
        f'referrerpolicy="no-referrer-when-downgrade" src="{src}"></iframe>', height=680)
    pano = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
    st.caption(f"📍 LiDAR gantry @ **{lat:.6f}, {lon:.6f}** — placed by the exact georeference.  "
               f"[Open in Google Maps ↗](https://www.google.com/maps/search/?api=1&query={lat},{lon})  ·  "
               f"[👤 Street View — walk it in first-person ↗]({pano})")


tab_cam, tab_lidar, tab_real = st.tabs(
    ["🎥 Road Viewer (cameras)", "🧊 LiDAR labels (3D)", "🛰️ Real intersection"])
with tab_cam:
    render_camera_tab()
with tab_lidar:
    render_lidar_tab()
with tab_real:
    render_real_tab()
