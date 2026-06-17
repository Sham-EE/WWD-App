import os

import streamlit as st

import dataset_manager as dm
import road_viewer as rv

st.set_page_config(layout="wide", page_title="Road Viewer")
st.title("🛣️ Road Viewer")
st.markdown("Browse the camera images of the road — both cameras side by side — and render a "
            "continuous side-by-side video to showcase the scene.")

ds = dm.get_active()
st.caption(f"📂 Dataset: **{ds.name}**")

images_root = st.text_input("Path to the images folder (one subfolder per camera)",
                            value=ds.images_dir)
cameras = rv.list_cameras(images_root)
if len(cameras) < 1:
    st.warning(f"No camera subfolders found in `{images_root}`.")
    st.stop()

# ---------------- Camera + variant selection ----------------
c1, c2, c3 = st.columns(3)
# Default layout requested: south2 on the LEFT, south1 on the RIGHT.
left_default = cameras.index("south2") if "south2" in cameras else 0
right_default = cameras.index("south1") if "south1" in cameras else (1 if len(cameras) > 1 else 0)
left_cam = c1.selectbox("Left camera", cameras, index=left_default)
right_cam = c2.selectbox("Right camera", cameras, index=right_default)

variants = [v for v in rv.available_variants(images_root, left_cam)
            if v in rv.available_variants(images_root, right_cam)] or list(rv.VARIANTS.keys())
vdefault = variants.index("Bounding boxes") if "Bounding boxes" in variants else 0
variant_label = c3.radio("Visualization", variants, index=vdefault, horizontal=True,
                         help="Raw camera · Bounding boxes · Boxes + point cloud overlay.")
vkey = rv.VARIANTS[variant_label]

left_frames = rv.frames_for(images_root, left_cam, vkey)
right_frames = rv.frames_for(images_root, right_cam, vkey)
n = min(len(left_frames), len(right_frames))
if n == 0:
    st.warning("No images found for that camera/variant combination.")
    st.stop()
st.caption(f"{n} synchronized frame pairs · **{left_cam}** (left) ↔ **{right_cam}** (right) · “{variant_label}”")

# ---------------- Frame playback ----------------
st.session_state.setdefault("road_frame", 0)
st.session_state.road_frame = max(0, min(st.session_state.road_frame, n - 1))
nav = st.columns([1, 1, 1, 1, 1.4, 3])
if nav[0].button("⏮ First", use_container_width=True):
    st.session_state.road_frame = 0; st.rerun()
if nav[1].button("◀ Prev", use_container_width=True):
    st.session_state.road_frame = max(0, st.session_state.road_frame - 1); st.rerun()
if nav[2].button("Next ▶", use_container_width=True):
    st.session_state.road_frame = min(n - 1, st.session_state.road_frame + 1); st.rerun()
if nav[3].button("Last ⏭", use_container_width=True):
    st.session_state.road_frame = n - 1; st.rerun()
playing = nav[4].toggle("▶ Play", value=False)
play_delay = nav[5].slider("Play delay (s)", 0.0, 1.0, 0.15, 0.05)
i = st.slider("Frame", 0, max(n - 1, 1), st.session_state.road_frame)
st.session_state.road_frame = i

lc, rc = st.columns(2)
lc.image(left_frames[i], use_container_width=True,
         caption=f"{left_cam} (left) · frame {i+1}/{n} · {os.path.basename(left_frames[i])}")
rc.image(right_frames[i], use_container_width=True,
         caption=f"{right_cam} (right) · frame {i+1}/{n} · {os.path.basename(right_frames[i])}")

if playing and i < n - 1:
    import time
    time.sleep(float(play_delay))
    st.session_state.road_frame = i + 1
    st.rerun()

# ---------------- Generate video ----------------
st.divider()
st.subheader("🎬 Generate road video")
if not rv.mp4_available():
    st.caption("ℹ️ MP4 export needs `imageio-ffmpeg` in the environment running this app "
               "(`pip install imageio-ffmpeg`). Until then, an animated **GIF** is produced instead.")
g1, g2, g3 = st.columns(3)
v_fps = g1.slider("Video FPS", 1, 30, 10, 1)
v_height = g2.select_slider("Frame height (px)", [360, 480, 540, 720], value=480)
v_max = g3.number_input("Max frames (0 = all)", 0, n, 0)

video_dir = os.path.join(ds.outputs_dir, "road_videos")
basename = f"road_{left_cam}_{right_cam}_{vkey}"
st.session_state.setdefault("road_video", None)

if st.button("🎬 Generate side-by-side video", type="primary", use_container_width=True):
    bar = st.progress(0.0, text="Rendering…")
    try:
        path, kind = rv.generate_side_by_side_video(
            left_frames, right_frames, video_dir, basename, fps=int(v_fps), height=int(v_height),
            max_frames=int(v_max), progress=lambda c, t: bar.progress(c / t, text=f"Frame {c}/{t}"))
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
