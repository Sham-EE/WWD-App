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
    st.text_input("Output (cropped) folder", value=out_dir, key="dp_out", disabled=True)
    if not frames:
        st.warning(f"No `.pcd` files in `{src_dir}`."
                   + ("  — register the clouds first (Registration page)." if "Registered" in source else ""))
    elif st.button("✂️ Generate cropped clouds", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="Cropping…")
        n, kept, tot = dp.crop_dataset(src_dir, out_dir, margin=margin,
                                       progress=lambda c, t: bar.progress(c / t, text=f"Cropping {c}/{t}"))
        bar.empty()
        pct = 100.0 * kept / max(tot, 1)
        st.success(f"Wrote **{n}** cropped clouds → `{out_dir}`  (kept {kept:,} / {tot:,} points, {pct:.0f}%).")

    # ---- side-by-side preview: south | north, cropped/uncropped + road outline ----
    st.divider()
    st.subheader("👁 Preview")
    pv_sensors = {"South": ds.raw_lidar_south_dir, "North": ds.raw_lidar_north_dir}
    pc1, pc2, pc3, pc4 = st.columns([1, 1, 1.4, 1])
    pv_left = pc1.selectbox("Left LiDAR", list(pv_sensors), index=0, key="dp_left")
    pv_right = pc2.selectbox("Right LiDAR", list(pv_sensors), index=1, key="dp_right")
    crop_mode = pc3.radio("Points", ["Cropped (road)", "Full (uncropped)"], horizontal=True, key="dp_crop")
    show_road = pc4.checkbox("🛣️ Road outline", value=True, key="dp_road")
    cropped = crop_mode.startswith("Cropped")

    Lf = rv.list_by_frame(pv_sensors[pv_left], [".pcd"])
    Rf = rv.list_by_frame(pv_sensors[pv_right], [".pcd"])
    npv = min(len(Lf), len(Rf))
    if npv == 0:
        st.warning("Need point clouds for both selected sensors.")
    else:
        st.session_state.setdefault("dp_frame", 0)

        def _pv_panel(sensor, files, i, key):
            pts = _load_raw(files[i])
            shown = dp.crop_points_to_region(pts, dp.road_polygon(margin)) if cropped else pts
            st.markdown(f"**{sensor} LiDAR** · {len(shown):,} pts")
            st.plotly_chart(dp.crop_preview_figure(shown, margin=margin, height=520, draw_boundary=show_road),
                            use_container_width=True, key=key)

        @st.fragment
        def _crop_preview():
            st.session_state.dp_frame = max(0, min(st.session_state.dp_frame, npv - 1))
            nav = st.columns([1, 1, 1, 1, 1.3, 3])
            if nav[0].button("⏮ First", use_container_width=True):
                st.session_state.dp_frame = 0
            if nav[1].button("◀ Prev", use_container_width=True):
                st.session_state.dp_frame = max(0, st.session_state.dp_frame - 1)
            if nav[2].button("Next ▶", use_container_width=True):
                st.session_state.dp_frame = min(npv - 1, st.session_state.dp_frame + 1)
            if nav[3].button("Last ⏭", use_container_width=True):
                st.session_state.dp_frame = npv - 1
            playing = nav[4].toggle("▶ Play", value=False)
            delay = nav[5].slider("Play delay (s)", 0.0, 1.0, 0.15, 0.05)
            i = st.slider("Frame", 0, max(npv - 1, 1), st.session_state.dp_frame)
            st.session_state.dp_frame = i

            with st.container(height=600):
                cl, cr = st.columns(2)
                with cl:
                    _pv_panel(pv_left, Lf, i, "dp_fig_l")
                with cr:
                    _pv_panel(pv_right, Rf, i, "dp_fig_r")
            st.caption(f"Frame {i+1}/{npv} · {crop_mode} · {pv_left} ↔ {pv_right}")

            if playing and i < npv - 1:
                time.sleep(float(delay))
                st.session_state.dp_frame = i + 1
                st.rerun(scope="fragment")

        _crop_preview()

# ===================== Tab 2: Scorable GT =====================
with tab_gt:
    st.caption("Build a **scorable** ground-truth set: keep only objects inside the processed region "
               "(the eval ROI) that actually have LiDAR points. Transparent + reproducible — the basis "
               "for fair evaluation. (The bundled `labels_visible_south` used an opaque per-frame "
               "visibility check that can't be reproduced from the labels; this is the principled "
               "equivalent.)")

    gt_src = ds.raw_labels_south_dir
    gt_out = ds.gt_dir
    gc1, gc2, gc3 = st.columns([1.2, 1, 1])
    gt_margin = gc1.slider("Region margin (m)", 0.0, 15.0, 0.0, 1.0,
                           help="Expand the research/ROI polygon outward before filtering.")
    min_points = gc2.slider("Min LiDAR points", 0, 50, 1, 1,
                            help="Drop objects with fewer than this many points (0 = keep all in-region).")
    excl_occ = gc3.checkbox("Drop mostly/fully occluded", value=False)
    region = dp.research_region(gt_margin)

    st.text_input("Source labels", value=gt_src, key="gt_src", disabled=True)
    st.text_input("Output (scorable GT) folder", value=gt_out, key="gt_out", disabled=True)
    gt_labels = rv.list_by_frame(gt_src, [".json"])
    if not gt_labels:
        st.warning(f"No label files in `{gt_src}`.")
    elif st.button("🏷️ Generate scorable GT", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="Filtering labels…")
        nfiles, kept, total = dp.generate_scorable_gt(
            gt_src, gt_out, region, min_points=int(min_points), exclude_occluded=excl_occ,
            progress=lambda c, t: bar.progress(c / t, text=f"Filtering {c}/{t}"))
        bar.empty()
        st.success(f"Wrote **{nfiles}** label files → `{gt_out}`  (kept {kept:,} / {total:,} objects, "
                   f"{100*kept/max(total,1):.0f}%).")

    # ---- preview: kept (green) vs dropped (grey) + ROI outline ----
    st.divider()
    st.subheader("👁 Preview")
    if gt_labels:
        n_gt = len(gt_labels)
        st.session_state.setdefault("gt_frame", 0)

        @st.fragment
        def _gt_preview():
            st.session_state.gt_frame = max(0, min(st.session_state.gt_frame, n_gt - 1))
            nav = st.columns([1, 1, 1, 1, 1.3, 3])
            if nav[0].button("⏮ First", use_container_width=True, key="gt_first"):
                st.session_state.gt_frame = 0
            if nav[1].button("◀ Prev", use_container_width=True, key="gt_prev"):
                st.session_state.gt_frame = max(0, st.session_state.gt_frame - 1)
            if nav[2].button("Next ▶", use_container_width=True, key="gt_next"):
                st.session_state.gt_frame = min(n_gt - 1, st.session_state.gt_frame + 1)
            if nav[3].button("Last ⏭", use_container_width=True, key="gt_last"):
                st.session_state.gt_frame = n_gt - 1
            playing = nav[4].toggle("▶ Play", value=False, key="gt_play")
            delay = nav[5].slider("Play delay (s)", 0.0, 1.0, 0.15, 0.05, key="gt_delay")
            i = st.slider("Frame", 0, max(n_gt - 1, 1), st.session_state.gt_frame, key="gt_slider")
            st.session_state.gt_frame = i

            kept_xy, dropped_xy = dp.scorable_classify(gt_labels[i], region,
                                                       min_points=int(min_points), exclude_occluded=excl_occ)
            with st.container(height=620):
                st.plotly_chart(dp.scorable_preview_figure(
                    kept_xy, dropped_xy, region,
                    title=f"frame {i+1}/{n_gt} · kept {len(kept_xy)} · dropped {len(dropped_xy)}"),
                    use_container_width=True, key="gt_fig")

            if playing and i < n_gt - 1:
                time.sleep(float(delay))
                st.session_state.gt_frame = i + 1
                st.rerun(scope="fragment")

        _gt_preview()
