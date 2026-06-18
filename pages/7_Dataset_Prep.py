import os
import time

import numpy as np
import open3d as o3d
import streamlit as st

import dataset_manager as dm
import dataset_prep as dp
import road_viewer as rv

st.set_page_config(layout="wide", page_title="Dataset Prep")
st.title("🧰 Dataset Prep")
st.markdown("Recreate the dataset's **derived** data from the raw TUM Traffic download, in-app — so "
            "everything the pipeline needs is reproducible, no external scripts.")

ds = dm.get_active()
st.caption(f"📂 Dataset: **{ds.name}**")


@st.cache_data(show_spinner=False, max_entries=64)
def _load_raw(path):
    return np.asarray(o3d.io.read_point_cloud(path).points)


tab_crop, tab_gt = st.tabs(["✂️ Crop to road (ROI)", "🏷️ Scorable GT (visible-only)"])

# ===================== Tab 1: Crop to road =====================
with tab_crop:
    st.caption("Clip a LiDAR's point clouds to the **road polygons** in `site_geometry.json`. "
               "Scales across sensors — crop the south, the north, or (later) the registered cloud. "
               "Verified to reproduce the bundled cropped clouds exactly.")

    # Source -> (raw input dir, cropped output dir)
    sources = {
        "South LiDAR": (ds.raw_lidar_south_dir, ds.pcd_dir),
        "North LiDAR": (ds.raw_lidar_north_dir, os.path.join(ds.derived_dir, "cropped_north", "cropped_pcd")),
        "Registered (south + north)": (os.path.join(ds.derived_dir, "registered"),
                                       os.path.join(ds.derived_dir, "cropped_registered", "cropped_pcd")),
    }
    sc1, sc2 = st.columns([1, 2])
    source = sc1.selectbox("Source LiDAR", list(sources), index=0)
    src_dir, out_dir = sources[source]
    margin = sc2.slider("Road margin (m)", 0.0, 5.0, 0.0, 0.5,
                        help="Expand the road polygon outward before clipping (0 = exact).")

    frames = rv.list_by_frame(src_dir, [".pcd"])
    if not frames:
        st.warning(f"No `.pcd` files in `{src_dir}`."
                   + ("  — register the clouds first (Registration page)." if "Registered" in source else ""))
    else:
        st.text_input("Output (cropped) folder", value=out_dir, key="dp_out", disabled=True)
        if st.button("✂️ Generate cropped clouds", type="primary", use_container_width=True):
            bar = st.progress(0.0, text="Cropping…")
            n, kept, tot = dp.crop_dataset(src_dir, out_dir, margin=margin,
                                           progress=lambda c, t: bar.progress(c / t, text=f"Cropping {c}/{t}"))
            bar.empty()
            pct = 100.0 * kept / max(tot, 1)
            st.success(f"Wrote **{n}** cropped clouds → `{out_dir}`  (kept {kept:,} / {tot:,} points, {pct:.0f}%).")

        # ---- frame-by-frame preview (BEV + dashed road boundary) ----
        st.divider()
        st.subheader("👁 Preview")
        n = len(frames)
        st.session_state.setdefault("dp_frame", 0)

        @st.fragment
        def _crop_preview():
            st.session_state.dp_frame = max(0, min(st.session_state.dp_frame, n - 1))
            nav = st.columns([1, 1, 1, 1, 1.3, 3])
            if nav[0].button("⏮ First", use_container_width=True):
                st.session_state.dp_frame = 0
            if nav[1].button("◀ Prev", use_container_width=True):
                st.session_state.dp_frame = max(0, st.session_state.dp_frame - 1)
            if nav[2].button("Next ▶", use_container_width=True):
                st.session_state.dp_frame = min(n - 1, st.session_state.dp_frame + 1)
            if nav[3].button("Last ⏭", use_container_width=True):
                st.session_state.dp_frame = n - 1
            playing = nav[4].toggle("▶ Play", value=False)
            delay = nav[5].slider("Play delay (s)", 0.0, 1.0, 0.15, 0.05)
            i = st.slider("Frame", 0, max(n - 1, 1), st.session_state.dp_frame)
            st.session_state.dp_frame = i

            pts = _load_raw(frames[i])
            kept = dp.crop_points_to_region(pts, dp.road_polygon(margin))
            with st.container(height=660):
                st.plotly_chart(dp.crop_preview_figure(kept, margin=margin,
                                title=f"{source} · frame {i+1}/{n} · {len(kept):,} pts in road"),
                                use_container_width=True, key="dp_fig")

            if playing and i < n - 1:
                time.sleep(float(delay))
                st.session_state.dp_frame = i + 1
                st.rerun(scope="fragment")

        _crop_preview()

# ===================== Tab 2: Scorable GT =====================
with tab_gt:
    st.caption("Filter the raw labels down to objects that are in-region and actually perceived by "
               "the LiDAR — reproduces `labels_visible_south` for fair evaluation.")
    st.info("Coming next — this is the step we'll build after the crop preview.")
