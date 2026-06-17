import os

import streamlit as st

import dataset_manager as dm
import road_viewer as rv
import label_projection as lp

st.set_page_config(layout="wide", page_title="Road Viewer")
st.title("🛣️ Road Viewer")
st.markdown("Browse the cameras side by side and **generate** label/point-cloud overlays on the fly "
            "(no dev-kit needed) — then render a continuous road video.")

ds = dm.get_active()
st.caption(f"📂 Dataset: **{ds.name}**")

with st.expander("📁 Input folders", expanded=False):
    images_root = st.text_input("Images folder (one subfolder per camera)", value=ds.images_dir)
    label_dir = st.text_input("OpenLABEL labels folder (for box/point-cloud overlays)", value=ds.gt_dir)
    pcd_dir = st.text_input("Point-cloud folder (for the point-cloud overlay)", value=ds.pcd_dir)

cameras = rv.list_cameras(images_root)
if len(cameras) < 1:
    st.warning(f"No camera subfolders found in `{images_root}`.")
    st.stop()

labels = rv.list_by_frame(label_dir, [".json"])
pcds = rv.list_by_frame(pcd_dir, [".pcd"])
have_labels = len(labels) > 0

# ---------------- Camera + variant selection ----------------
c1, c2, c3 = st.columns([1, 1, 1.4])
left_default = cameras.index("south2") if "south2" in cameras else 0
right_default = cameras.index("south1") if "south1" in cameras else (1 if len(cameras) > 1 else 0)
left_cam = c1.selectbox("Left camera", cameras, index=left_default)
right_cam = c2.selectbox("Right camera", cameras, index=right_default)

MODES = {"Raw camera": None, "Bounding boxes (3D)": "box3d", "Boxes + point cloud": "point_cloud"}
variant_labels = list(MODES) if have_labels else ["Raw camera"]
variant_label = c3.radio("Visualization", variant_labels,
                         index=(1 if have_labels else 0), horizontal=True,
                         help="Box/point-cloud overlays are generated from the OpenLABEL labels + "
                              "calibration and cached under the dataset's outputs/rendered/.")
mode = MODES[variant_label]
if not have_labels:
    st.info(f"No OpenLABEL label files in `{label_dir}` — only raw images can be shown. "
            "Add labels to enable generated box / point-cloud overlays.")

color_mode, point_size = "by_category", 2
track_hist, hist_window = False, 30
if mode:
    oc1, oc2, oc3 = st.columns([1, 1.4, 1])
    color_mode = oc1.radio("Box colour", ["by_category", "by_track_id"], horizontal=True,
                           help="by_category: colour encodes the class (all cars cyan, trucks teal…). "
                                "by_track_id: each object gets its own colour to follow it across frames.")
    if mode == "point_cloud":
        point_size = oc2.select_slider("Point size", [1, 2, 3], value=2)
    if oc3.button("♻️ Regenerate (clear cache)", help="Delete cached overlays and re-render."):
        import shutil
        shutil.rmtree(os.path.join(ds.outputs_dir, "rendered"), ignore_errors=True)
        st.session_state.road_video = None
        st.rerun()
    th1, th2 = st.columns([1, 2])
    track_hist = th1.checkbox("🛤️ Track history (cyan trails)", value=False,
                              help="Draw each object's recent path as a tapering cyan trail, "
                                   "computed from its position in the preceding frames.")
    if track_hist:
        hist_window = th2.slider("Trail length (frames)", 5, 80, 30, 5)

# ---------------- Frame resolution + pairing ----------------
raw_left = rv.frames_for(images_root, left_cam, "raw")
raw_right = rv.frames_for(images_root, right_cam, "raw")
counts = [len(raw_left), len(raw_right)]
if mode:
    counts.append(len(labels))
if mode == "point_cloud":
    counts.append(len(pcds))
n = min(counts) if counts else 0
if n == 0:
    st.warning("Not enough synchronized frames (need raw images" +
               (" + labels" if mode else "") + (" + point clouds" if mode == "point_cloud" else "") + ").")
    st.stop()
st.caption(f"{n} synchronized frames · **{left_cam}** (left) ↔ **{right_cam}** (right) · “{variant_label}”")

left_id = lp.camera_id_from_image(raw_left[0])
right_id = lp.camera_id_from_image(raw_right[0])


@st.cache_data(show_spinner=False)
def _all_centers(label_paths):
    """Per-frame {object_id: (x,y,z)} for the whole sequence (cached)."""
    return [lp.frame_centers(p) for p in label_paths]


all_centers = _all_centers(tuple(labels)) if (mode and track_hist) else None


def _histories(i):
    """{object_id: [centres from frames i-window .. i]} for objects in frame i."""
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
    th = f"_th{hist_window}" if track_hist else ""
    return os.path.join(ds.outputs_dir, "rendered", cam, f"{mode}_{color_mode}{ps}{th}_{lp.RENDER_VERSION}")


def _render(i, cam, cam_id, raw):
    """Path to the frame to show for camera `cam` at index i (raw or generated)."""
    if not mode:
        return raw[i]
    return lp.render_cached(raw[i], labels[i], cam_id, mode, _cache_dir(cam),
                            color_mode=color_mode,
                            pcd_path=(pcds[i] if mode == "point_cloud" else None),
                            point_size=point_size, histories=_histories(i))


# ---------------- Playback ----------------
# Isolated in a fragment so stepping frames reruns ONLY this block, not the whole
# page — no full-page refresh/flicker between frames.
st.session_state.setdefault("road_frame", 0)


@st.fragment
def _viewer():
    st.session_state.road_frame = max(0, min(st.session_state.road_frame, n - 1))
    nav = st.columns([1, 1, 1, 1, 1.3, 3])
    if nav[0].button("⏮ First", use_container_width=True):
        st.session_state.road_frame = 0
    if nav[1].button("◀ Prev", use_container_width=True):
        st.session_state.road_frame = max(0, st.session_state.road_frame - 1)
    if nav[2].button("Next ▶", use_container_width=True):
        st.session_state.road_frame = min(n - 1, st.session_state.road_frame + 1)
    if nav[3].button("Last ⏭", use_container_width=True):
        st.session_state.road_frame = n - 1
    playing = nav[4].toggle("▶ Play", value=False)
    play_delay = nav[5].slider("Play delay (s)", 0.0, 1.0, 0.15, 0.05)
    i = st.slider("Frame", 0, max(n - 1, 1), st.session_state.road_frame)
    st.session_state.road_frame = i

    left_img = _render(i, left_cam, left_id, raw_left)
    right_img = _render(i, right_cam, right_id, raw_right)
    lc, rc = st.columns(2)
    lc.image(left_img, use_container_width=True, caption=f"{left_cam} (left) · frame {i+1}/{n}")
    rc.image(right_img, use_container_width=True, caption=f"{right_cam} (right) · frame {i+1}/{n}")

    if playing and i < n - 1:
        import time
        time.sleep(float(play_delay))
        st.session_state.road_frame = i + 1
        st.rerun(scope="fragment")


_viewer()

# ---------------- Generate video ----------------
st.divider()
st.subheader("🎬 Generate road video")
if not rv.mp4_available():
    st.caption("ℹ️ MP4 needs `imageio-ffmpeg` in the running environment; otherwise an animated **GIF** is produced.")
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
        if mode:   # render (cache) the needed frames first
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
