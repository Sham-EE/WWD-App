import os
import time

import numpy as np
import open3d as o3d
import streamlit as st

import dataset_manager as dm
import dataset_prep as dp
import geometry_editor as ge
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


def _bbox_editor(poly, label, step=1.0):
    """Lane-Editor-style rectangle editor: X/Y min-max with +/- steppers. Returns 4 corners.
    (No widget keys -> the value re-seeds from `poly` each run, so reset-to-default works.)"""
    xs = [p[0] for p in poly] or [-10.0, 10.0]
    ys = [p[1] for p in poly] or [-10.0, 10.0]
    # 2 columns (not 4) so each number_input is wide enough to show its +/- steppers
    r1c1, r1c2 = st.columns(2)
    xmin = r1c1.number_input(f"{label} X min", value=float(min(xs)), step=step, format="%.1f")
    xmax = r1c2.number_input(f"{label} X max", value=float(max(xs)), step=step, format="%.1f")
    r2c1, r2c2 = st.columns(2)
    ymin = r2c1.number_input(f"{label} Y min", value=float(min(ys)), step=step, format="%.1f")
    ymax = r2c2.number_input(f"{label} Y max", value=float(max(ys)), step=step, format="%.1f")
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def _vertex_editor(poly, label, step=1.0):
    """Per-vertex x/y +/- steppers (for arbitrary polygons). Returns list of [x,y]."""
    poly = [list(p) for p in (poly or [[0.0, 0.0]])]
    out = []
    for i, (x, y) in enumerate(poly):
        c1, c2 = st.columns(2)
        nx = c1.number_input(f"{label} v{i+1} x", value=float(x), step=step, format="%.1f")
        ny = c2.number_input(f"{label} v{i+1} y", value=float(y), step=step, format="%.1f")
        out.append([nx, ny])
    return out


tab_crop, tab_gt, tab_geom = st.tabs(
    ["✂️ Crop to road (ROI)", "🏷️ Scorable GT (visible-only)", "🗺️ Geometry Editor"])

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
                            use_container_width=True, key=key, config={"scrollZoom": True})

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

    # Source -> (raw labels dir, scorable-GT output dir, raw cloud dir for preview)
    gt_sources = {
        "South": (ds.raw_labels_south_dir, ds.gt_dir, ds.raw_lidar_south_dir),
        "North": (ds.raw_labels_north_dir, os.path.join(ds.derived_dir, "labels_visible_north"),
                  ds.raw_lidar_north_dir),
    }
    if os.path.isdir(os.path.join(ds.derived_dir, "registered_labels")):
        gt_sources["Registered (south + north)"] = (
            os.path.join(ds.derived_dir, "registered_labels"),
            os.path.join(ds.derived_dir, "labels_visible_registered"),
            os.path.join(ds.derived_dir, "registered"))

    gs1, gs2 = st.columns([1.3, 1])
    gt_source = gs1.selectbox("Source LiDAR", list(gt_sources), index=0, key="gt_source")
    gt_src, gt_out, gt_cloud_dir = gt_sources[gt_source]
    gt_margin = gs2.slider("Region margin (m)", 0.0, 15.0, 0.0, 1.0,
                           help="Expand the research/ROI region. Edit its SHAPE in the Geometry Editor; "
                                "this just buffers it outward.")
    region = dp.research_region(gt_margin)

    with st.expander("⚙️ Keep / drop criteria", expanded=True):
        st.caption("Define exactly what counts as a scorable object. The region (above) is the ROI; "
                   "these conditions filter within it. Tune live against the preview below.")
        cc1, cc2, cc3 = st.columns(3)
        min_points = cc1.slider("Min LiDAR points", 0, 50, 1, 1,
                                help="Drop objects with fewer points than this (sparse / blind-spot).")
        max_points = cc2.number_input("Max LiDAR points (0 = none)", 0, 100000, 0)
        max_range = cc3.slider("Max range from sensor (m, 0 = none)", 0, 150, 0, 5)
        oc1, oc2 = st.columns([1, 1.4])
        drop_occ = oc1.multiselect("Drop occlusion levels", dp.OCCLUSION_LEVELS, default=[])
        classes = oc2.multiselect("Classes to keep (empty = all)", dp.SCORABLE_CLASSES, default=[])
        crit = {
            "min_points": int(min_points),
            "max_points": (int(max_points) or None),
            "max_range": (float(max_range) or None),
            "drop_occlusion": tuple(drop_occ),
            "classes": (set(classes) if classes else None),
        }

    st.text_input("Source labels", value=gt_src, key="gt_src", disabled=True)
    st.text_input("Output (scorable GT) folder", value=gt_out, key="gt_out", disabled=True)
    gt_labels = rv.list_by_frame(gt_src, [".json"])
    if not gt_labels:
        st.warning(f"No label files in `{gt_src}`.")
    elif st.button("🏷️ Generate scorable GT", type="primary", use_container_width=True):
        bar = st.progress(0.0, text="Filtering labels…")
        nfiles, kept, total = dp.generate_scorable_gt(
            gt_src, gt_out, region, crit=crit,
            progress=lambda c, t: bar.progress(c / t, text=f"Filtering {c}/{t}"))
        bar.empty()
        st.success(f"Wrote **{nfiles}** label files → `{gt_out}`  (kept {kept:,} / {total:,} objects, "
                   f"{100*kept/max(total,1):.0f}%).")

    # ---- preview: point cloud + kept (green) vs dropped (red) boxes ----
    st.divider()
    st.subheader("👁 Preview")
    st.caption("Point cloud (blue) with **kept** ground-truth boxes in **green** and **dropped** boxes "
               "in **red**. Far drops sit outside the ROI window; in-region red boxes are objects with "
               "too few LiDAR points (e.g. blind spots).")
    gt_clouds = rv.list_by_frame(gt_cloud_dir, [".pcd"])
    if gt_labels:
        n_gt = min(len(gt_labels), len(gt_clouds)) if gt_clouds else len(gt_labels)
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
            i = st.slider("GT frame", 0, max(n_gt - 1, 1), st.session_state.gt_frame)
            st.session_state.gt_frame = i

            kept_boxes, dropped_boxes = dp.scorable_classify(gt_labels[i], region, crit)
            pts = _load_raw(gt_clouds[i]) if (gt_clouds and i < len(gt_clouds)) else None
            tot = len(kept_boxes) + len(dropped_boxes)
            with st.container(height=640):
                st.plotly_chart(dp.scorable_preview_figure(
                    pts, kept_boxes, dropped_boxes, region,
                    title=f"frame {i+1}/{n_gt} · kept {len(kept_boxes)}/{tot}"),
                    use_container_width=True, key="gt_fig", config={"scrollZoom": True})

            if playing and i < n_gt - 1:
                time.sleep(float(delay))
                st.session_state.gt_frame = i + 1
                st.rerun(scope="fragment")

        _gt_preview()


# ===================== Tab 3: Geometry Editor =====================
with tab_geom:
    st.caption("Edit the **site geometry** — research/ROI polygon, road polygons (used for cropping), "
               "and exclusion rectangles. **Saving updates the whole pipeline** (Background Filtering, "
               "cropping, scorable GT, the road outline) — they all read this file.")

    if "geom_edit" not in st.session_state or st.session_state.get("geom_ds") != ds.id:
        st.session_state.geom_edit = ge.load_site_geometry(ds)
        st.session_state.geom_ds = ds.id
    geom = st.session_state.geom_edit

    default_geom = ge.load_default_geometry(ds)
    gt1, gt2, gt3 = st.columns([1, 1, 1])
    if gt1.button("💾 Save (updates everything)", type="primary", use_container_width=True, key="geom_save"):
        ge.save_site_geometry(ds, geom)
        st.success(f"Saved → `{ds.site_geometry_path}`. The whole pipeline now uses this geometry.")
    if gt2.button("↩️ Reload saved", use_container_width=True, key="geom_reload"):
        st.session_state.geom_edit = ge.load_site_geometry(ds)
        st.rerun()
    if gt3.button("🔄 Reset ALL to default", use_container_width=True, key="geom_reset_all",
                  disabled=not ge.has_defaults(ds)):
        import copy
        st.session_state.geom_edit = copy.deepcopy(default_geom)
        st.rerun()

    geom_clouds = rv.list_by_frame(ds.raw_lidar_south_dir, [".pcd"]) or rv.list_by_frame(ds.pcd_dir, [".pcd"])
    geom_bg = None
    if geom_clouds:
        gfi = st.slider("Backdrop frame", 0, len(geom_clouds) - 1, 0, key="geom_bg_frame")
        geom_bg = _load_raw(geom_clouds[gfi])
    step = st.select_slider("Stepper increment (m)", [0.5, 1.0, 2.0, 5.0], value=1.0, key="geom_step")

    g_left, g_right = st.columns([1, 1.3], gap="medium")
    with g_left:
        with st.expander("🔵 Research polygon (ROI)", expanded=True):
            st.caption("Overall analysed region (a rectangle); the default scorable-GT region.")
            geom["research_polygon"] = _bbox_editor(geom.get("research_polygon", []), "ROI", step)
            rr1, rr2 = st.columns(2)
            if rr1.button("🔄 Reset to default", key="ge_reset_research", disabled=not ge.has_defaults(ds)):
                geom["research_polygon"] = [list(p) for p in default_geom["research_polygon"]]; st.rerun()
            if rr2.button("📐 From data extent", key="ge_derive_research"):
                geom["research_polygon"] = dm.derive_site_geometry(ds.pcd_dir)["research_polygon"]; st.rerun()

        with st.expander("🟢 Road polygons (cropping)", expanded=True):
            st.caption("Drivable area (any shape). Crop-to-road keeps only points inside these.")
            roads = geom.get("road_polygons", [])
            if st.button("🔄 Reset road polygons to default", key="ge_reset_road",
                         disabled=not ge.has_defaults(ds)):
                geom["road_polygons"] = [[list(p) for p in poly] for poly in default_geom["road_polygons"]]
                st.rerun()
            rc1, rc2 = st.columns(2)
            if rc1.button("➕ Add polygon", key="ge_add_road"):
                roads.append(ge.default_rect(geom)); geom["road_polygons"] = roads; st.rerun()
            if roads:
                st.session_state.setdefault("ge_road_idx", 0)
                ridx = min(st.session_state.ge_road_idx, len(roads) - 1)
                ridx = st.selectbox("Road polygon", range(len(roads)), index=ridx,
                                    format_func=lambda i: f"#{i+1}")
                st.session_state.ge_road_idx = ridx
                rcv1, rcv2 = st.columns(2)
                if rcv1.button("➕ Add vertex", key="ge_road_addv"):
                    roads[ridx].append(list(roads[ridx][-1])); geom["road_polygons"] = roads; st.rerun()
                if rcv2.button("🗑️ Delete polygon", key="ge_del_road"):
                    roads.pop(ridx); geom["road_polygons"] = roads
                    st.session_state.ge_road_idx = 0; st.rerun()
                roads[ridx] = _vertex_editor(roads[ridx], f"Road{ridx+1}", step)
                geom["road_polygons"] = roads
            else:
                st.info("No road polygons — add one (cropping keeps everything until you do).")

        with st.expander("🔴 Exclusion rectangles", expanded=False):
            st.caption("Regions always treated as background (static clutter).")
            rects = geom.get("foreground_exclusion_rects", [])
            if st.button("🔄 Reset exclusions to default", key="ge_reset_excl",
                         disabled=not ge.has_defaults(ds)):
                geom["foreground_exclusion_rects"] = [[list(p) for p in r]
                                                      for r in default_geom["foreground_exclusion_rects"]]
                st.rerun()
            if st.button("➕ Add rectangle", key="ge_add_excl"):
                rects.append(ge.default_rect(geom)); geom["foreground_exclusion_rects"] = rects; st.rerun()
            if rects:
                st.session_state.setdefault("ge_excl_idx", 0)
                eidx = min(st.session_state.ge_excl_idx, len(rects) - 1)
                eidx = st.selectbox("Rectangle", range(len(rects)), index=eidx,
                                    format_func=lambda i: f"#{i+1}")
                st.session_state.ge_excl_idx = eidx
                if st.button("🗑️ Delete this rect", key="ge_del_excl"):
                    rects.pop(eidx); geom["foreground_exclusion_rects"] = rects
                    st.session_state.ge_excl_idx = 0; st.rerun()
                rects[eidx] = _bbox_editor(rects[eidx], f"Rect{eidx+1}", step)
                geom["foreground_exclusion_rects"] = rects
            else:
                st.info("No exclusion rectangles.")
    with g_right:
        st.markdown("**👁 Live preview** — scroll to zoom, drag to pan.")
        st.plotly_chart(ge.preview_figure(geom_bg, geom, height=640),
                        use_container_width=True, config={"scrollZoom": True})

    st.session_state.geom_edit = geom
