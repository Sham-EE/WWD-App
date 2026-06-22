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


# Filter-time config (matches the Background Filtering page defaults) — lets the
# geometry editor reuse a saved background model to show foreground classification.
_GEOM_FILTER_CFG = {
    "ground_grid": 0.5, "dz_thresh": 0.3, "bg_voxel": 1.0, "bg_ratio": 0.98,
    "cell_size": 1.0, "cell_ratio": 0.9,
    "cluster": {"ds_voxel": 0.15, "eps0": 0.35, "eps_k": 0.008, "eps_min": 0.35,
                "eps_max": 2.0, "min_samples": 16},
    "enable_pole_filter": True, "pole_min_height": 1.5, "pole_min_aspect_xy": 6.0,
    "pole_max_xy_area": 1.0, "pole_min_linearity": 0.75, "pole_min_points": 8,
    "pole_max_points": 80,
}


@st.cache_resource(show_spinner=False)
def _load_bg_model(path, _mtime):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Filtering foreground…", max_entries=16)
def _geom_foreground(cloud_path, model_path, model_mtime, geom_mtime):
    """Run the saved background model over one cloud, return foreground Nx3.
    Cached by model + geometry mtime, so it only recomputes when you rebuild the
    model or Save the geometry (geom_mtime busts it via remove_fg_rects)."""
    import bg_filter_core as bf
    try:
        pts = _load_raw(cloud_path)
        model = _load_bg_model(model_path, model_mtime)
        fg, _ = bf.filter_points_with_model(pts, model, _GEOM_FILTER_CFG)
        return fg
    except Exception:
        return None


def _resolve_bg_model(src):
    """Saved model path matching `src`, else the other source's, else None."""
    for p in (ds.model_path_for(src), ds.model_path_for("cropped" if src == "full" else "full")):
        if p and os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def _gt_map(label_dir):
    """{frame key -> label .json} for matching GT to a backdrop cloud frame."""
    files = rv.list_by_frame(label_dir, [".json"])
    return {"_".join(os.path.basename(f).split("_")[:2]): f for f in files}


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
    pc1, pc2, pc3, pc4, pc5 = st.columns([1, 1, 1.3, 1, 1])
    pv_left = pc1.selectbox("Left LiDAR", list(pv_sensors), index=0, key="dp_left")
    pv_right = pc2.selectbox("Right LiDAR", list(pv_sensors), index=1, key="dp_right")
    crop_mode = pc3.radio("Points", ["Cropped (road)", "Full (uncropped)"], horizontal=True, key="dp_crop")
    show_road = pc4.checkbox("🛣️ Road outline", value=True, key="dp_road")
    color_h = pc5.checkbox("🌈 Color by height", value=False, key="dp_height",
                           help="Colour points by z (Turbo) like the dev-kit.")
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
            st.plotly_chart(dp.crop_preview_figure(shown, margin=margin, height=520, draw_boundary=show_road,
                                                   color_by_height=color_h),
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

            # Toggle each Geometry-Editor boundary onto the preview.
            bt = st.columns(4)
            show_roi = bt[0].toggle("🔵 ROI", value=True, key="gt_show_roi",
                                    help="Research region — objects outside it are dropped (red).")
            show_road = bt[1].toggle("🟢 Road outline", value=False, key="gt_show_road",
                                     help="Drivable area used for cropping.")
            show_excl = bt[2].toggle("🟣 Exclusion zones", value=False, key="gt_show_excl",
                                     help="Foreground-exclusion rectangles (static clutter).")
            color_h = bt[3].toggle("🌈 Color by height", value=False, key="gt_height",
                                   help="Colour points by z (Turbo) like the dev-kit.")

            kept_boxes, dropped_boxes = dp.scorable_classify(gt_labels[i], region, crit)
            pts = _load_raw(gt_clouds[i]) if (gt_clouds and i < len(gt_clouds)) else None
            tot = len(kept_boxes) + len(dropped_boxes)
            with st.container(height=640):
                st.plotly_chart(dp.scorable_preview_figure(
                    pts, kept_boxes, dropped_boxes, region,
                    title=f"frame {i+1}/{n_gt} · kept {len(kept_boxes)}/{tot}",
                    show_roi=show_roi, show_road=show_road, show_exclusion=show_excl,
                    color_by_height=color_h),
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

    geom_clouds = rv.list_by_frame(ds.raw_lidar_south_dir, [".pcd"])
    geom_src = "full"
    if not geom_clouds:
        geom_clouds = rv.list_by_frame(ds.pcd_dir, [".pcd"]); geom_src = "cropped"
    geom_bg = None
    if geom_clouds:
        gfi = st.slider("Backdrop frame", 0, len(geom_clouds) - 1, 0, key="geom_bg_frame")
        geom_bg = _load_raw(geom_clouds[gfi])
    gt_map = _gt_map(ds.raw_labels_south_dir) or _gt_map(ds.gt_dir)
    model_path = _resolve_bg_model(geom_src)
    bc1, bc2, bc3, bc4 = st.columns([1.2, 1, 1, 1.1])
    step = bc1.select_slider("Stepper increment (m)", [0.5, 1.0, 2.0, 5.0], value=1.0, key="geom_step")
    show_fg = bc2.toggle("🔴 Foreground", value=False, key="geom_show_fg", disabled=not model_path,
                         help="Overlay what the background model classifies as foreground (red), so "
                              "you can drop an exclusion rect over poles/clutter that leak through. "
                              "Save geometry to see the effect." if model_path
                              else "No saved background model — build one on the Background Filtering page.")
    show_gt = bc3.toggle("🏷️ GT boxes", value=False, key="geom_show_gt", disabled=not gt_map,
                         help="Overlay this frame's ground-truth boxes (category-coloured + labels)."
                              if gt_map else "No ground truth for this dataset.")
    show_metric = bc4.toggle("📊 FG quality", value=False, key="geom_metric",
                             disabled=not (model_path and gt_map),
                             help="Live foreground-vs-GT quality for this frame (needs a model + GT). "
                                  "Edit/Save geometry and watch the numbers move."
                                  if (model_path and gt_map) else "Needs a saved model AND ground truth.")

    # Compute foreground / GT if any of their consumers (overlay or metric) is on.
    geom_fg = None
    if (show_fg or show_metric) and model_path and geom_clouds:
        _gmt = os.path.getmtime(ds.site_geometry_path) if os.path.exists(ds.site_geometry_path) else 0.0
        geom_fg = _geom_foreground(geom_clouds[gfi], model_path, os.path.getmtime(model_path), _gmt)
    geom_gt = None
    if (show_gt or show_metric) and gt_map and geom_clouds:
        import label_projection as lp
        _gp = gt_map.get("_".join(os.path.basename(geom_clouds[gfi]).split("_")[:2]))
        if _gp:
            geom_gt = lp.load_objects(_gp)

    # Split the model foreground by the CURRENT (unsaved) geometry: kept (red) vs
    # cropped-out (grey). Metrics use only the kept set, so editing updates them live.
    geom_fg_kept, geom_fg_excl = (None, None)
    if geom_fg is not None:
        geom_fg_kept, geom_fg_excl = ge.apply_geometry_crop(geom_fg, geom)

    if show_metric and geom_fg_kept is not None and geom_gt is not None:
        mpts = st.number_input("Covered if ≥ pts", 1, 200, 10, 1, key="geom_minpts",
                               help="A GT object counts as 'covered' with at least this many "
                                    "surviving foreground points.")
        q = dp.foreground_quality(geom_fg_kept, geom_bg, geom_gt, min_pts=int(mpts))
        n_excl = int(len(geom_fg_excl)) if geom_fg_excl is not None else 0
        mm = st.columns(4)
        mm[0].metric(f"Objects covered (≥{q['min_pts']} pts)", f"{q['covered']} / {q['scanned']}",
                     help="GT objects with enough surviving foreground points, out of objects "
                          "the LiDAR actually hit this frame.")
        mm[1].metric("On-object recall",
                     f"{q['recall']*100:.0f}%" if q['recall'] is not None else "—",
                     help="Foreground points inside GT boxes ÷ original points inside GT boxes.")
        mm[2].metric("Off-object foreground", f"{q['off_object']:,}",
                     help="Kept foreground outside every GT box (clutter / false-foreground proxy).")
        mm[3].metric("Cropped out (grey)", f"{n_excl:,}",
                     help="Foreground removed by the CURRENT geometry (exclusion rects / outside "
                          "road) — live, before you Save.")
        st.caption("Live: reflects unsaved edits. Grey dots = removed by current geometry.")

    g_left, g_right = st.columns([1, 1.3], gap="medium")
    with g_left:
        with st.expander("🔵 Research polygon (ROI)", expanded=False):
            st.caption("Overall analysed region (a rectangle); the default scorable-GT region.")
            geom["research_polygon"] = _bbox_editor(geom.get("research_polygon", []), "ROI", step)
            rr1, rr2, rr3 = st.columns(3)
            if rr1.button("🔄 Default", key="ge_reset_research", disabled=not ge.has_defaults(ds),
                          help="Reset the ROI to the dataset default."):
                geom["research_polygon"] = [list(p) for p in default_geom["research_polygon"]]; st.rerun()
            if rr2.button("📐 Data extent", key="ge_derive_research",
                          help="Fit the ROI to the full point-cloud extent."):
                geom["research_polygon"] = dm.derive_site_geometry(ds.pcd_dir)["research_polygon"]; st.rerun()
            _roads = geom.get("road_polygons", [])
            if rr3.button("🛣️ From road", key="ge_roi_from_road", disabled=not _roads,
                          help="Set the ROI to the road bounds + a 5 m margin (keeps off-road edge)."):
                _ax = [p[0] for poly in _roads for p in poly]; _ay = [p[1] for poly in _roads for p in poly]
                _m = 5.0
                _x0, _x1, _y0, _y1 = min(_ax) - _m, max(_ax) + _m, min(_ay) - _m, max(_ay) + _m
                geom["research_polygon"] = [[_x0, _y0], [_x1, _y0], [_x1, _y1], [_x0, _y1]]; st.rerun()

        with st.expander("🟢 Road polygons (cropping)", expanded=False):
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
                # Move the whole polygon by `step` metres (the working drag-move).
                st.caption(f"Move polygon #{ridx+1} (±{step:g} m)")
                mv = st.columns(4)
                _moves = {"◀": (-step, 0), "▶": (step, 0), "▲": (0, step), "▼": (0, -step)}
                for _c, (_lbl, (_dx, _dy)) in zip(mv, _moves.items()):
                    if _c.button(_lbl, key=f"ge_road_mv_{_lbl}", use_container_width=True):
                        roads[ridx] = [[p[0] + _dx, p[1] + _dy] for p in roads[ridx]]
                        geom["road_polygons"] = roads; st.rerun()
                roads[ridx] = _vertex_editor(roads[ridx], f"Road{ridx+1}", step)
                geom["road_polygons"] = roads
            else:
                st.info("No road polygons — add one (cropping keeps everything until you do).")

        with st.expander("🟣 Exclusion rectangles", expanded=False):
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
        pv1, pv2, pv3, pv4 = st.columns([1, 1, 1, 1])
        pv1.markdown("**👁 Live preview**")
        mode = pv2.radio("Mouse", ["🖐 Pan", "⬛ Draw box"], horizontal=True, key="geom_drawmode",
                         label_visibility="collapsed",
                         help="Draw box: drag a rectangle on the plot, then add it as an exclusion "
                              "zone or set it as the ROI.")
        show_verts = pv3.toggle("🔖 Vertex labels", value=False, key="geom_verts",
                                help="Tag every vertex (ROIn / R<road>.<v> / X<rect>.<v>) so you know "
                                     "which polygon and vertex you're editing.")
        color_h = pv4.toggle("🌈 Color by height", value=False, key="geom_height",
                             help="Colour the backdrop cloud by z (Turbo) like the dev-kit — ground vs "
                                  "poles/vehicles separate by hue (great for spotting clutter in 2D).")
        h_span = 4.0
        if color_h:
            h_span = st.slider("Height span (m)", 1.5, 12.0, 4.0, 0.5, key="geom_hspan",
                               help="Colour spreads over this many metres above the ground. Smaller = "
                                    "more colour detail on short objects (cars show a gradient too); "
                                    "taller things saturate at the top colour.")
        dm_mode = "select" if mode.startswith("⬛") else "pan"
        ev = st.plotly_chart(ge.preview_figure(geom_bg, geom, height=620,
                                               fg_points=geom_fg_kept if show_fg else None,
                                               fg_excluded_points=geom_fg_excl if show_fg else None,
                                               gt_objs=geom_gt if show_gt else None,
                                               dragmode=dm_mode, show_vertex_labels=show_verts,
                                               color_by_height=color_h, height_span=h_span),
                             use_container_width=True, config={"scrollZoom": True},
                             on_select="rerun", key="geom_preview")

        # Read a drawn box from the selection and let the user apply it.
        drawn = None
        try:
            bx = (ev.get("selection") or {}).get("box") or []
            if bx:
                xs, ys = bx[0]["x"], bx[0]["y"]
                drawn = (min(xs), min(ys), max(xs), max(ys))
        except Exception:
            drawn = None
        if drawn:
            x0, y0, x1, y1 = drawn
            rect = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            st.caption(f"⬛ Drawn box: x[{x0:.1f}, {x1:.1f}]  y[{y0:.1f}, {y1:.1f}]")
            db1, db2, db3 = st.columns(3)
            if db1.button("🟢 Add as road", use_container_width=True, key="geom_box_road",
                          help="Add a rough road polygon, then refine its vertices in the Road panel."):
                geom.setdefault("road_polygons", []).append(rect)
                st.session_state.geom_edit = geom; st.rerun()
            if db2.button("🟣 Add exclusion", use_container_width=True, key="geom_box_excl"):
                geom.setdefault("foreground_exclusion_rects", []).append(rect)
                st.session_state.geom_edit = geom; st.rerun()
            if db3.button("🔵 Set as ROI", use_container_width=True, key="geom_box_roi"):
                geom["research_polygon"] = rect
                st.session_state.geom_edit = geom; st.rerun()
        elif dm_mode == "select":
            st.caption("Drag a rectangle on the plot to draw a box.")

    st.session_state.geom_edit = geom
