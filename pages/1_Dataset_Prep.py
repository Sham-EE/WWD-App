import os
import glob
import time

import numpy as np
import open3d as o3d
import streamlit as st

import dataset_manager as dm
import dataset_prep as dp
import geometry_editor as ge
import registration as reg
import road_viewer as rv
import viewer_ui as vu
import nav

st.set_page_config(layout="wide", page_title="Dataset Prep")
nav.render_sidebar()
st.title("🧰 Dataset Prep")
st.markdown("Recreate the dataset's **derived** data from the raw TUM Traffic download, in-app — so "
            "everything the pipeline needs is reproducible, no external scripts.")

ds = dm.get_active()
st.caption(f"📂 Dataset: **{ds.name}**")


@st.cache_data(show_spinner=False, max_entries=64)
def _load_raw(path):
    return np.asarray(o3d.io.read_point_cloud(path).points)


# Filter-time config — must mirror the Background Filtering page DEFAULTS so the geometry
# editor's foreground matches what that page would produce (density clusterer, 5x5 on, SOR
# off). Keep in sync if those defaults change.
_GEOM_FILTER_CFG = {
    "ground_grid": 0.5, "dz_thresh": 0.3, "bg_voxel": 1.0, "bg_ratio": 0.98,
    "cell_size": 1.0, "cell_ratio": 0.9, "inward_buffer_m": 2.0,
    "cluster": {"mode": "density", "ds_voxel": 0.15, "eps0": 0.35, "eps_k": 0.008,
                "eps_min": 0.35, "eps_max": 2.0, "min_samples": 16,
                "eps_scale": 2.5, "n_tiers": 3, "min_samples_far": 8},
    "enable_pole_filter": True, "pole_min_height": 1.5, "pole_min_aspect_xy": 6.0,
    "pole_max_xy_area": 1.0, "pole_min_linearity": 0.75, "pole_min_points": 8,
    "pole_max_points": 80, "enable_5x5": True, "enable_sor": False,
    "coarse_5x5": {"NX": 5, "NY": 5},
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


@st.cache_data(show_spinner="Aggregating foreground over ALL frames…", max_entries=3)
def _geom_foreground_all(pcd_dir, gt_dir, model_path, model_mtime, geom_mtime, box_buffer, voxel=0.25):
    """Foreground (and off-object foreground) accumulated over EVERY frame of pcd_dir,
    voxel-downsampled for display. Lets the editor show all clutter across the whole
    sequence at once, so you can place exclusion zones over persistent off-object FG.
    Cached by model/geometry/buffer so it only recomputes when those change."""
    import bg_filter_core as bf
    import label_projection as lp
    from dataset_prep import foreground_quality
    try:
        model = _load_bg_model(model_path, model_mtime)
    except Exception:
        return None, None, 0
    gtmap = {}
    if gt_dir and os.path.isdir(gt_dir):
        for f in glob.glob(os.path.join(gt_dir, "*.json")):
            gtmap["_".join(os.path.basename(f).split("_")[:2])] = f
    files = rv.list_by_frame(pcd_dir, [".pcd"])
    fg_list, off_list = [], []
    for cp in files:
        pts = _load_raw(cp)
        fg, _ = bf.filter_points_with_model(pts, model, _GEOM_FILTER_CFG)
        if not len(fg):
            continue
        fg_list.append(fg[:, :3])
        gp = gtmap.get("_".join(os.path.basename(cp).split("_")[:2]))
        if gp:
            q = foreground_quality(fg, pts, lp.load_objects(gp), box_buffer=box_buffer)
            off_list.append(fg[~q["fg_on_mask"]][:, :3])
        else:
            off_list.append(fg[:, :3])  # no GT this frame -> all foreground is "off-object"
    def _vox(lst):
        return bf.voxel_downsample_numpy(np.vstack(lst), voxel) if lst else None
    return _vox(fg_list), _vox(off_list), len(files)


@st.cache_data(show_spinner=False)
def _geom_detections(csv_path, _mtime):
    """Read a detection tracks.csv -> (by_frame dict, all-centres Nx2, all-static bool N).
    'static' = the track's lifetime MAX speed never crossed 1.5 m/s (FP / pole risk).
    Boxes fall back to nominal car size when no extent was measured."""
    import csv
    rows, maxsp = [], {}
    try:
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                tid = r.get("tid", "")
                fr = int(float(r["frame"]))
                cx, cy = float(r["cx"]), float(r["cy"])
                yaw = float(r.get("yaw", 0) or 0)
                l = float(r.get("length", 0) or 0) or 4.5
                w = float(r.get("width", 0) or 0) or 1.9
                rows.append((fr, tid, cx, cy, yaw, l, w))
                maxsp[tid] = max(maxsp.get(tid, 0.0), float(r.get("speed", 0) or 0))
    except Exception:
        return {}, np.zeros((0, 2)), np.zeros((0,), dtype=bool)
    by_frame, cen, stat = {}, [], []
    for fr, tid, cx, cy, yaw, l, w in rows:
        is_static = maxsp.get(tid, 9.9) < 1.5
        by_frame.setdefault(fr, []).append((cx, cy, yaw, l, w, is_static))
        cen.append((cx, cy)); stat.append(is_static)
    return by_frame, (np.array(cen) if cen else np.zeros((0, 2))), np.array(stat, dtype=bool)


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


# ---------------- Prep pipeline (mirrors the Home stepper) ----------------
# The four tabs below build on each other; this stepper shows the recommended
# order and which steps are already done for the active dataset.
def _has_pcd(d):
    return bool(glob.glob(os.path.join(d, "*.pcd")))

_status = ds.status()
# (icon, name, done?, optional?)
_prep_steps = [
    ("🧭", "Registration", _has_pcd(ds.registered_dir), True),
    ("🗺️", "Geometry Editor", os.path.exists(ds.site_geometry_path), False),
    ("✂️", "Crop to road", _status["pcd"], False),
    ("🏷️", "Scorable GT", _status["gt"], False),
]
_prep_next = next((i for i, (_ic, _nm, dn, op) in enumerate(_prep_steps) if not dn and not op), None)

st.markdown("##### 🧭 Prep order")
_pc = st.columns(len(_prep_steps))
for i, (ic, nm, dn, op) in enumerate(_prep_steps):
    if dn:
        badge, color, bg, border = "✅ Done", "#4ade80", "#101a13", "#234a2c"
    elif i == _prep_next:
        badge, color, bg, border = "🔵 Next", "#60a5fa", "#0f1722", "#2b4a78"
    elif op:
        badge, color, bg, border = "⚪ Optional", "#c8a96a", "#1a1710", "#473a24"
    else:
        badge, color, bg, border = "⬜ To do", "#6b7480", "#14181f", "#2a3340"
    with _pc[i]:
        st.markdown(
            f"""<div style="background:{bg};border:1px solid {border};border-radius:12px;
                        padding:10px 6px;text-align:center;min-height:104px">
                  <div style="font-size:.7rem;color:#6b7480">STEP {i+1}</div>
                  <div style="font-size:1.35rem;line-height:1.55rem">{ic}</div>
                  <div style="font-weight:600;font-size:.84rem;margin-top:2px">{nm}</div>
                  <div style="color:{color};font-size:.74rem;margin-top:3px">{badge}</div>
                </div>""",
            unsafe_allow_html=True,
        )
st.caption("Registration is optional — only when fusing south + north. "
           "Geometry Editor draws the road polygons, Crop clips the clouds to them, "
           "then Scorable GT builds the visible-only ground truth.")

# Tabs follow the recommended order; the `with` blocks keep their original
# variable names, so only the labels + unpack order changed.
tab_reg, tab_geom, tab_crop, tab_gt = st.tabs(
    ["🧭 Registration", "🗺️ Geometry Editor", "✂️ Crop to road (ROI)", "🏷️ Scorable GT (visible-only)"])

# ===================== Step 3: Crop to road =====================
with tab_crop:
    st.caption("Clip a LiDAR's point clouds to the **road polygons** in `site_geometry.json`. "
               "Scales across sensors — crop the south, the north, or (later) the registered cloud. "
               "Verified to reproduce the bundled cropped clouds exactly.")

    # Source -> (raw input dir, cropped output dir)
    sources = {
        "South LiDAR": (ds.raw_lidar_south_dir, ds.cropped_dir_for("south")),
        "North LiDAR": (ds.raw_lidar_north_dir, ds.cropped_dir_for("north")),
        "Registered (south + north)": (ds.registered_dir, ds.cropped_dir_for("registered")),
    }
    sc1, sc2 = st.columns([1, 2])
    source = sc1.selectbox("Source LiDAR", list(sources), index=0)
    src_dir, out_dir = sources[source]
    margin = sc2.slider("Road margin (m)", 0.0, 5.0, 0.0, 0.5,
                        help="Expand the road polygon outward before clipping (0 = exact).")

    frames = rv.list_by_frame(src_dir, [".pcd"])
    st.caption("Output (cropped) folder")
    st.code(out_dir, language="text")
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
    # Registered is in the south frame, so the road polygon + road-window framing
    # apply to it exactly as they do to the south cloud.
    pv_sensors = {"South": ds.raw_lidar_south_dir, "North": ds.raw_lidar_north_dir,
                  "Registered": ds.registered_dir}
    pc1, pc2, pc3 = st.columns([1, 1, 1.3])
    pv_left = pc1.selectbox("Left LiDAR", list(pv_sensors), index=0, key="dp_left")
    pv_right = pc2.selectbox("Right LiDAR", list(pv_sensors), index=1, key="dp_right")
    crop_mode = pc3.radio("Points", ["Cropped (road)", "Full (uncropped)"], horizontal=True, key="dp_crop")
    cropped = crop_mode.startswith("Cropped")

    Lf = rv.list_by_frame(pv_sensors[pv_left], [".pcd"])
    Rf = rv.list_by_frame(pv_sensors[pv_right], [".pcd"])
    npv = min(len(Lf), len(Rf))
    if npv == 0:
        st.warning("Need point clouds for both selected sensors.")
    else:
        st.session_state.setdefault("dp_frame", 0)

        @st.fragment
        def _crop_preview():
            i, playing, delay = vu.nav_row("dp_frame", npv, "dp", label="🎞️ Crop frame")
            vu.ensure_toggle_defaults({"dp_road": True, "dp_height": False})
            with st.expander("🎛️ Layers & overlays", expanded=True):
                vu.bulk_toggle_buttons(["dp_road", "dp_height"], "dp_bulk")
                tg = st.columns(2)
                show_road = tg[0].toggle("🛣️ Road outline", key="dp_road",
                                         help="Draw the road crop boundary.")
                color_h = tg[1].toggle("🌈 Height", key="dp_height",
                                       help="Colour points by z (Turbo) like the dev-kit.")

            def _panel(sensor, files, key):
                pts = _load_raw(files[i])
                shown = dp.crop_points_to_region(pts, dp.road_polygon(margin)) if cropped else pts
                label = "Registered (fused)" if sensor == "Registered" else f"{sensor} LiDAR"
                st.markdown(f"**{label}** · {len(shown):,} pts")
                st.plotly_chart(dp.crop_preview_figure(shown, margin=margin, height=520,
                                                       draw_boundary=show_road, color_by_height=color_h),
                                use_container_width=True, key=key, config={"scrollZoom": True})

            with st.container(height=600):
                cl, cr = st.columns(2)
                with cl:
                    _panel(pv_left, Lf, "dp_fig_l")
                with cr:
                    _panel(pv_right, Rf, "dp_fig_r")
            st.caption(f"Frame {i+1}/{npv} · {crop_mode} · {pv_left} ↔ {pv_right}")

            if playing and i < npv - 1:
                time.sleep(float(delay))
                st.session_state.dp_frame = i + 1
                st.rerun(scope="fragment")

        _crop_preview()

# ===================== Step 4: Scorable GT =====================
with tab_gt:
    st.caption("Build a **scorable** ground-truth set: keep only objects inside the processed region "
               "(the eval ROI) that actually have LiDAR points. Transparent + reproducible — the basis "
               "for fair evaluation. (The bundled `labels_visible_south` used an opaque per-frame "
               "visibility check that can't be reproduced from the labels; this is the principled "
               "equivalent.)")

    # Source -> (raw labels dir, scorable-GT output dir, raw cloud dir for preview)
    gt_sources = {
        "South": (ds.raw_labels_south_dir, ds.gt_dir, ds.raw_lidar_south_dir),
        "North": (ds.raw_labels_north_dir, ds.scorable_gt_dir_for("north"),
                  ds.raw_lidar_north_dir),
    }
    _reg_raw_labels = os.path.join(ds.derived_dir, "labels", "registered")  # future fused GT
    if os.path.isdir(_reg_raw_labels):
        gt_sources["Registered (south + north)"] = (
            _reg_raw_labels,
            ds.scorable_gt_dir_for("registered"),
            ds.registered_dir)

    gs1, gs2 = st.columns([1.3, 1])
    gt_source = gs1.selectbox("Source LiDAR", list(gt_sources), index=0, key="gt_source")
    gt_src, gt_out, gt_cloud_dir = gt_sources[gt_source]
    gt_margin = gs2.slider("Region margin (m)", 0.0, 15.0, 0.0, 1.0,
                           help="Expand the research/ROI region. Edit its SHAPE in the Geometry Editor; "
                                "this just buffers it outward.")
    # The ROI is defined in the SOUTH frame; express it in the selected source's frame
    # so it lines up with that sensor's cloud (identity for south/registered; rotated
    # into the north frame for north — both are physically the same intersection ROI).
    _gt_sensor = ("north" if gt_source.startswith("North")
                  else "registered" if "Registered" in gt_source else "south")
    region = dp.research_region(gt_margin)
    if _gt_sensor == "north":
        region = reg.transform_polygon(region, reg.south_to_sensor_4x4(ds, "north"))

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

    st.caption("Source labels")
    st.code(gt_src, language="text")
    st.caption("Output (scorable GT) folder")
    st.code(gt_out, language="text")
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
            i, playing, delay = vu.nav_row("gt_frame", n_gt, "gt", label="🏷️ GT frame")
            # Toggle each Geometry-Editor boundary onto the preview.
            vu.ensure_toggle_defaults({"gt_show_roi": True, "gt_show_road": False,
                                       "gt_show_excl": False, "gt_height": False})
            with st.expander("🎛️ Layers & overlays", expanded=True):
                vu.bulk_toggle_buttons(["gt_show_roi", "gt_show_road", "gt_show_excl", "gt_height"],
                                       "gt_bulk")
                bt = st.columns(4)
                show_roi = bt[0].toggle("🔵 ROI", key="gt_show_roi",
                                        help="Research region — objects outside it are dropped (red).")
                show_road = bt[1].toggle("🟢 Road outline", key="gt_show_road",
                                         help="Drivable area used for cropping.")
                show_excl = bt[2].toggle("🟣 Exclusion zones", key="gt_show_excl",
                                         help="Foreground-exclusion rectangles (static clutter).")
                color_h = bt[3].toggle("🌈 Height", key="gt_height",
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


# ===================== Step 2: Geometry Editor =====================
with tab_geom:
    st.caption("Edit the **site geometry** — research/ROI polygon, road polygons (used for cropping), "
               "and exclusion rectangles. **Saving updates the whole pipeline** (Background Filtering, "
               "cropping, scorable GT, the road outline) — they all read this file.")

    if "geom_edit" not in st.session_state or st.session_state.get("geom_ds") != ds.id:
        st.session_state.geom_edit = ge.load_site_geometry(ds)
        st.session_state.geom_ds = ds.id
    geom = st.session_state.geom_edit

    default_geom = ge.load_default_geometry(ds)
    gt1, gt2, gt3, gt4 = st.columns([1.2, 1, 1, 1.2])
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
    if gt4.button("📌 Set as new default", use_container_width=True, key="geom_set_default",
                  help="Snapshot the CURRENT geometry as this dataset's default — what every "
                       "'Reset to default' button restores from here on."):
        ge.save_default_geometry(ds, geom)
        st.success(f"Default updated → `{ds.default_site_geometry_path}`. "
                   "'Reset to default' now restores this geometry.")

    # Follow the SAME sensor/source the rest of the app uses (shared pipeline_* state),
    # so the foreground / FG-quality here matches Background Filtering & Detection — not a
    # hardcoded south/full reference. Resolve the cloud, model and GT for that selection.
    gsc, gic = st.columns(2)
    _g_sensor = gsc.radio("Sensor", ["Registered", "South", "North"], key="pipeline_sensor",
                          horizontal=True, help="Which LiDAR's cloud + background model to show the "
                          "foreground for. Shared with Background Filtering / Detection.").lower()
    _g_src = "cropped" if gic.radio("Input cloud", ["Cropped (road)", "Full (uncropped)"],
                                    key="pipeline_source", horizontal=True).startswith("Cropped") else "full"
    geom_src = _g_src
    _g_pcd_dir = ds.input_pcd_for_sensor(_g_sensor, _g_src)
    geom_clouds = rv.list_by_frame(_g_pcd_dir, [".pcd"])
    if not geom_clouds:  # fallback so the editor still works if that selection isn't built
        geom_clouds = rv.list_by_frame(ds.raw_lidar_south_dir, [".pcd"]) or rv.list_by_frame(ds.pcd_dir, [".pcd"])
    geom_bg = None
    if geom_clouds:
        gfi = st.slider("Backdrop frame", 0, len(geom_clouds) - 1, 0, key="geom_bg_frame")
        geom_bg = _load_raw(geom_clouds[gfi])
    gt_map = _gt_map(ds.gt_dir_for_input(_g_pcd_dir)) or _gt_map(ds.raw_labels_south_dir) or _gt_map(ds.gt_dir)
    _g_model = ds.model_path_for_sensor(_g_sensor, _g_src)
    model_path = _g_model if (_g_model and os.path.exists(_g_model)) else _resolve_bg_model(geom_src)
    st.caption(f"🛰️ {_g_sensor.capitalize()} · {'Cropped' if _g_src=='cropped' else 'Full'}  ·  "
               f"model: `{os.path.basename(model_path) if model_path else 'none — build on Background Filtering'}`")
    step = st.select_slider("Stepper increment (m)", [0.5, 1.0, 2.0, 5.0], value=1.0, key="geom_step",
                            help="Step size for the +/- vertex / box editors below.")
    _nframes = len(geom_clouds) if geom_clouds else 0
    _det_csv = os.path.join(ds.detection_dir_for_sensor(_g_sensor, _g_src), "tracks.csv")
    _has_det = os.path.exists(_det_csv)
    _ov_keys = ["geom_show_fg", "geom_show_gt", "geom_show_off", "geom_metric",
                "geom_show_det", "geom_height", "geom_verts"]
    vu.ensure_toggle_defaults({k: False for k in _ov_keys})
    with st.expander("🎛️ Layers & overlays", expanded=True):
        vu.bulk_toggle_buttons(_ov_keys, "geom_bulk", rerun_scope="app")
        gr1 = st.columns(3)
        show_fg = gr1[0].toggle("🔴 Foreground", key="geom_show_fg", disabled=not model_path,
                                help="What the background model classifies as foreground (red) — drop an "
                                     "exclusion rect over poles/clutter that leak through."
                                     if model_path else "No saved background model.")
        show_gt = gr1[1].toggle("🏷️ GT boxes", key="geom_show_gt", disabled=not gt_map,
                                help="This frame's ground-truth boxes (category-coloured + labels)."
                                     if gt_map else "No ground truth for this dataset.")
        show_off = gr1[2].toggle("🟡 Off-object FG", key="geom_show_off", disabled=not (model_path and gt_map),
                                 help="Foreground OUTSIDE every GT box (yellow) — clutter to place exclusion "
                                      "zones over." if (model_path and gt_map) else "Needs a model AND ground truth.")
        gr2 = st.columns(3)
        show_metric = gr2[0].toggle("📊 FG quality", key="geom_metric", disabled=not (model_path and gt_map),
                                    help="Live foreground-vs-GT quality for this frame (model + GT)."
                                         if (model_path and gt_map) else "Needs a model AND ground truth.")
        color_h = gr2[1].toggle("🌈 Height", key="geom_height",
                                help="Colour the backdrop cloud by z (Turbo) — spot clutter in 2D.")
        show_verts = gr2[2].toggle("🔖 Vertex labels", key="geom_verts",
                                   help="Tag every vertex (ROIn / R<road>.<v> / X<rect>.<v>) while editing.")
        gr3 = st.columns(3)
        show_det = gr3[0].toggle("🎯 Detections", key="geom_show_det", disabled=not _has_det,
                                 help="What the detector reported as objects — green = moving, purple = "
                                      "static (never-moving = FP / pole risk). This frame's boxes, or (with "
                                      "Show ALL frames) every detection centre across the sequence." if _has_det
                                      else "No detection tracks.csv for this sensor/source — run Detection first.")
        off_buf = st.number_input("🟡 Off-object box buffer (m)", 0.0, 2.0, 0.3, 0.1, key="geom_offbuf",
                                  help="Grow each GT box by this margin before flagging foreground as off-object "
                                       "(yellow). Moves the yellow overlay AND the FG-quality numbers together "
                                       "(0 = strict).") if (show_off or show_metric) else 0.3
        mpts = st.number_input("📊 Covered if ≥ pts", 1, 200, 10, 1, key="geom_minpts",
                               help="A GT object counts as 'covered' with at least this many surviving "
                                    "foreground points.") if show_metric else 10
        h_span = st.slider("🌈 Height span (m)", 1.5, 12.0, 4.0, 0.5, key="geom_hspan",
                           help="Colour spreads over this many metres above the ground.") if color_h else 4.0
        show_all = st.checkbox(f"🕒 Show ALL {_nframes} frames at once (whole-sequence FG + off-object)",
                               value=False, key="geom_allframes", disabled=not (model_path and geom_clouds),
                               help="Accumulate foreground (red) + off-object FG (yellow) across every frame so "
                                    "you can place exclusion zones over persistent clutter. First run filters all "
                                    "frames (slow); then cached." if (model_path and geom_clouds)
                                    else "Needs a saved background model.")

    _gmt = os.path.getmtime(ds.site_geometry_path) if os.path.exists(ds.site_geometry_path) else 0.0
    fg_all = off_all = None
    if show_all and model_path and geom_clouds:
        fg_all, off_all, _ = _geom_foreground_all(
            _g_pcd_dir, ds.gt_dir_for_input(_g_pcd_dir), model_path,
            os.path.getmtime(model_path), _gmt, float(off_buf))

    # Detections (the detector's reported objects) for this frame, or all frames.
    det_objs = det_scatter = None
    if show_det and _has_det:
        _by_frame, _det_cen, _det_stat = _geom_detections(_det_csv, os.path.getmtime(_det_csv))
        if show_all:
            det_scatter = (_det_cen, _det_stat)
        else:
            det_objs = _by_frame.get(gfi)

    # Compute foreground / GT if any of their consumers (overlay or metric) is on.
    geom_fg = None
    if (show_fg or show_metric or show_off) and model_path and geom_clouds:
        geom_fg = _geom_foreground(geom_clouds[gfi], model_path, os.path.getmtime(model_path), _gmt)
    geom_gt = None
    if (show_gt or show_metric or show_off) and gt_map and geom_clouds:
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
        q = dp.foreground_quality(geom_fg_kept, geom_bg, geom_gt, min_pts=int(mpts),
                                  box_buffer=float(off_buf))
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
        st.markdown("**👁 Live preview** — **drag to pan**, scroll to zoom. To draw a road / exclusion / "
                    "ROI box, click the ⬚ **Box Select** tool in the chart's top-right toolbar, then drag "
                    "a rectangle. (Overlay toggles are in 🎛️ Layers & overlays.)")
        dm_mode = "pan"  # left-drag pans; drawing uses the modebar Box-Select tool
        # Off-object foreground = kept foreground outside every GT box (yellow).
        geom_off = None
        if show_off and geom_fg_kept is not None and len(geom_fg_kept) and geom_gt:
            # 0.3 m box buffer so real returns spilling just past a tight box aren't
            # flagged as clutter (overlay-only; the FG-quality metric stays unbuffered).
            _q = dp.foreground_quality(geom_fg_kept, geom_bg, geom_gt, min_pts=1, box_buffer=float(off_buf))
            geom_off = geom_fg_kept[~_q["fg_on_mask"]]
        # All-frames mode shows the whole-sequence aggregates instead of the current frame's
        # foreground, so every persistent clutter point is visible for exclusion-zone placement.
        if show_all:
            _fg_disp, _fgx_disp, _off_disp = fg_all, None, off_all
            if fg_all is not None:
                st.caption(f"🕒 All-frames view: {len(fg_all):,} foreground voxels"
                           + (f" · {len(off_all):,} off-object" if off_all is not None else "")
                           + " (downsampled). Yellow = persistent clutter to exclude.")
        else:
            _fg_disp = geom_fg_kept if show_fg else None
            _fgx_disp = geom_fg_excl if show_fg else None
            _off_disp = geom_off
        if show_det and (det_objs or det_scatter is not None):
            if show_all and det_scatter is not None:
                _ds_cen = det_scatter[0]
                st.caption(f"🎯 All-frames detections: {len(_ds_cen):,} object centres "
                           f"({int(det_scatter[1].sum()):,} static — purple = FP/pole risk).")
            elif det_objs:
                _ns = sum(1 for d in det_objs if d[5])
                st.caption(f"🎯 Frame {gfi}: {len(det_objs)} detections ({_ns} static).")
        ev = st.plotly_chart(ge.preview_figure(geom_bg, geom, height=620,
                                               fg_points=_fg_disp,
                                               fg_excluded_points=_fgx_disp,
                                               gt_objs=geom_gt if show_gt else None,
                                               off_object_points=_off_disp,
                                               det_objs=det_objs, det_scatter=det_scatter,
                                               dragmode=dm_mode, show_vertex_labels=show_verts,
                                               color_by_height=color_h, height_span=h_span),
                             use_container_width=True,
                             config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False,
                                     "modeBarButtonsToRemove": ["lasso2d", "autoScale2d"]},
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
        else:
            st.caption("Tip: click the ⬚ Box Select tool (top-right of the chart), then drag a "
                       "rectangle to draw a road / exclusion / ROI box.")

    st.session_state.geom_edit = geom


# ===================== Step 1: Registration (south + north) =====================
with tab_reg:
    st.caption("Fuse the **south** and **north** Ouster LiDARs into one cloud. The sensor→base "
               "extrinsics are read straight from the OpenLABEL labels (static rig → constant 4×4), so "
               "registration is **deterministic** — no guessing. The fused cloud is written in the "
               "**south LiDAR frame**, so it's a drop-in superset of the south cloud (the GT, camera "
               "calibration, and road/ROI polygons all match it). Optional **ICP** then *measures* how "
               "well the bundled calibration aligns the clouds.")

    label_dirs = {"south": ds.raw_labels_south_dir, "north": ds.raw_labels_north_dir}
    pcd_dirs = {"south": ds.raw_lidar_south_dir, "north": ds.raw_lidar_north_dir}
    reg_out = ds.registered_dir

    @st.cache_data(show_spinner=False)
    def _calib(side, label_dir):
        M = reg.calibration_for(label_dir, side)
        dev = reg.calibration_is_constant(label_dir, side)
        return (M.tolist() if M is not None else None), dev

    Ms_l, dev_s = _calib("south", label_dirs["south"])
    Mn_l, dev_n = _calib("north", label_dirs["north"])
    Ms = np.array(Ms_l) if Ms_l else None
    Mn = np.array(Mn_l) if Mn_l else None

    if Ms is None or Mn is None:
        st.error("Couldn't read the sensor→base calibration from the labels. "
                 f"Need OpenLABEL labels with `s110_lidar_ouster_*` poses in "
                 f"`{label_dirs['south']}` and `{label_dirs['north']}`.")
        st.stop()

    # --- calibration read-out ---
    with st.expander("📐 Calibration (sensor → s110_base, read from labels)", expanded=False):
        for side, M, dev in (("South", Ms, dev_s), ("North", Mn, dev_n)):
            t = M[:3, 3]
            st.markdown(f"**{side}** → `s110_base`  ·  translation "
                        f"(x={t[0]:.2f}, y={t[1]:.2f}, z={t[2]:.2f}) m  ·  "
                        f"frame-to-frame drift {dev:.1e} "
                        + ("✅ static" if (dev is not None and dev < 1e-6) else "⚠️ varies"))
            st.code(np.array2string(M, precision=4, suppress_small=True), language="text")

    # --- frame pairing ---
    sp_files = rv.list_by_frame(pcd_dirs["south"], [".pcd"])
    np_files = rv.list_by_frame(pcd_dirs["north"], [".pcd"])
    if not sp_files or not np_files:
        st.warning("Need raw point clouds for **both** south and north LiDARs under "
                   f"`{pcd_dirs['south']}` and `{pcd_dirs['north']}`.")
        st.stop()
    pairs = reg.match_frame_pairs(sp_files, np_files)
    dts = [p[2] for p in pairs]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Matched pairs", f"{len(pairs)}")
    mc2.metric("Mean Δt", f"{np.mean(dts):.0f} ms")
    mc3.metric("Max Δt", f"{max(dts):.0f} ms")
    st.caption("Sensors fire asynchronously, so each south frame is paired with its nearest-in-time "
               "north frame. Small Δt ⇒ negligible motion between the two captures.")

    # --- ICP refine ---
    st.divider()
    with st.expander("🔧 ICP refinement", expanded=False):
        st.caption("The bundled extrinsics align the ground plane but carry a relative **yaw + "
                   "translation** error between the two sensors. Coarse-to-fine point-to-plane ICP "
                   "measures and corrects it; the rig is static, so one correction applies to every frame.")
        ic1, ic2, ic3, ic4 = st.columns([1.4, 1, 1, 1])
        use_refine = ic1.toggle("Apply ICP correction", value=True, key="reg_use_icp",
                                help="On = north corrected by ICP on top of the calibration (recommended — "
                                     "the raw calibration is visibly off). Off = pure calibration only.")
        icp_dist = ic2.slider("ICP fine dist (m)", 0.2, 2.0, 0.5, 0.1, key="reg_icp_dist",
                              help="Finest correspondence distance (coarse-to-fine starts at 5 m).")
        icp_iter = ic3.slider("ICP iters/level", 10, 100, 60, 10, key="reg_icp_iter")
        icp_voxel = ic4.slider("ICP voxel (m)", 0.0, 1.0, 0.25, 0.05, key="reg_icp_voxel",
                               help="Downsample before ICP for speed (0 = full resolution).")

    st.session_state.setdefault("reg_frame", 0)
    st.session_state.setdefault("reg_delta", None)

    st.divider()
    st.subheader("👁 Preview — Raw vs Registered")
    st.caption("One viewer, two modes. **Raw** overlays the two clouds in their own sensor frames "
               "(the 'before' — they won't line up). **Registered** fuses them (the 'after'); use the "
               "**Frame** toggle to view in the south LiDAR frame (what's written) or the neutral "
               "`s110_base`. Toggle **By sensor** to colour south (blue) / north (orange) — if a "
               "vehicle shows up as two offset copies in Registered view, that's a registration error.")

    @st.cache_data(show_spinner=False)
    def _fused(sp, npath, Ms_l, Mn_l, refine_l):
        s_pts = reg.load_xyz(sp)
        n_pts = reg.load_xyz(npath)
        refine = np.array(refine_l) if refine_l else None
        f = reg.fuse_pair(s_pts, n_pts, np.array(Ms_l), np.array(Mn_l), refine=refine)
        return {"south": f["south"], "north": f["north"]}

    @st.cache_data(show_spinner="Running ICP…")
    def _icp_for(sp, npath, Ms_l, Mn_l, dist, iters, voxel):
        """Coarse-to-fine ICP correction (north→south) for one pair, cached by
        pair + params. Returns (delta_list, info)."""
        f = reg.fuse_pair(reg.load_xyz(sp), reg.load_xyz(npath), np.array(Ms_l), np.array(Mn_l))
        delta, info = reg.icp_refine(f["north"], f["south"], max_dist=dist, max_iter=iters, voxel=voxel)
        return delta.tolist(), info

    @st.fragment
    def _reg_preview():
        i, playing, delay = vu.nav_row("reg_frame", len(pairs), "reg", label="🧭 Pair")
        sp, npath, dt_ms = pairs[i]

        # view controls — one cohesive viewer with a Raw <-> Registered toggle
        tc1, tc2, tc3 = st.columns([1.2, 1, 1.2])
        view = tc1.radio("View", ["Raw (unregistered)", "Registered (fused)"],
                         horizontal=True, index=1, key="reg_view",
                         help="Raw = each cloud in its OWN sensor frame, overlaid (the 'before' — they "
                              "won't line up). Registered = both fused together (the 'after').")
        color_mode = tc2.radio("Color", ["By sensor", "By height"], horizontal=True,
                               key="reg_color", help="By sensor = south (blue) / north (orange) distinct "
                                                     "— alignment QA. By height = Turbo z-ramp.")
        frame = tc3.radio("Frame", ["South LiDAR", "s110_base"], horizontal=True, key="reg_frame_sel",
                          help="A 'frame' is just where (0,0,0) sits and which way the axes point — like "
                               "measuring a room from the ceiling camera vs. the floor corner. The cloud "
                               "doesn't move, only the numbers do. **South LiDAR**: origin is AT the south "
                               "sensor (so its marker sits at 0,0,0; ground ≈ −8.6). **s110_base**: origin "
                               "is on the GROUND (the sensor is up its pole; ground ≈ 0). Same picture, "
                               "shifted + rotated together. We write in the south frame because the GT, "
                               "camera calibration, and road/ROI polygons are all measured from the south "
                               "sensor too — so registered lines up with them for free (south marker at "
                               "the origin = the cloud now speaks the same numbers as the GT).")
        vc1, _s1, vc2, _s2, vc3 = st.columns([3, 0.3, 3, 0.3, 3])
        zoom = vc1.slider("🔍 Zoom", 0.35, 2.0, 0.9, 0.05, key="reg_zoom")
        az = vc2.slider("🔄 Rotate", 0, 360, 45, key="reg_az")
        el = vc3.slider("📐 Tilt", 5, 88, 35, key="reg_el")
        _reg_ov = ["reg_show_s", "reg_show_n", "reg_road", "reg_roi", "reg_sensors"]
        vu.ensure_toggle_defaults({"reg_show_s": True, "reg_show_n": True, "reg_road": False,
                                   "reg_roi": False, "reg_sensors": True, "reg_crop": True})
        with st.expander("🎛️ Layers & overlays", expanded=True):
            vu.bulk_toggle_buttons(_reg_ov, "reg_bulk")  # fragment-scoped rerun
            rr1 = st.columns(3)
            show_s = rr1[0].toggle("🔵 South", key="reg_show_s", help="South cloud (blue).")
            show_n = rr1[1].toggle("🟠 North", key="reg_show_n", help="North cloud (orange).")
            show_sensors = rr1[2].toggle("📍 Sensors", key="reg_sensors",
                                         help="Mark the LiDAR positions + a plumb line to each nadir "
                                              "(blank spot) — Registered view only.")
            rr2 = st.columns(3)
            show_road = rr2[0].toggle("🟢 Road", key="reg_road",
                                      help="Road outline (south frame — Registered + South-frame view only).")
            show_roi = rr2[1].toggle("🔵 ROI", key="reg_roi",
                                     help="Research region (south frame — Registered + South-frame view only).")
            crop_road = rr2[2].toggle("✂️ Crop", key="reg_crop",
                                      help="Clip the fused cloud to the road region (same window used to make "
                                           "the cropped clouds). Off = full scene incl. background structures. "
                                           "Not part of All/None (it's a clip, not an overlay).")
            hspan = st.slider("🌈 Height span (m)", 1.0, 12.0, 4.0, 0.5, key="reg_hspan",
                              help="For 'By height' colour: metres above ground the Turbo ramp spans.")

        # ICP correction for this pair (auto-computed + cached). Always measured
        # so the read-out quantifies the calibration error; applied only if the
        # toggle is on. Stored as reg_delta so the batch writer reuses it.
        delta_l, inf = _icp_for(sp, npath, Ms.tolist(), Mn.tolist(), icp_dist, icp_iter, icp_voxel)
        st.session_state.reg_delta = delta_l
        refine_l = delta_l if use_refine else None
        f = _fused(sp, npath, Ms.tolist(), Mn.tolist(), refine_l)

        big_yaw = abs(inf["yaw_deg"]) > 1.0 or inf["translation_m"] > 0.5
        # the yaw/shift describe how far OFF the raw calibration is; the verdict
        # then says whether that error is currently being corrected.
        if not big_yaw:
            verdict = "✅ raw calibration already aligns well — correction negligible"
        elif use_refine:
            verdict = "✅ corrected (this is how far off the raw calibration was)"
        else:
            verdict = "⚠️ raw calibration is off by this much — toggle on to correct"
        st.info(f"ICP correction (north→south): yaw **{inf['yaw_deg']:+.2f}°** · "
                f"shift **{inf['translation_m']:.2f} m** · fitness **{inf['fitness']:.2f}** · "
                f"RMSE **{inf['inlier_rmse']:.3f} m** — {verdict}")

        is_raw = view.startswith("Raw")
        if is_raw:
            # raw clouds in their OWN sensor frames — don't clip to the base-frame
            # road window and don't draw base-frame overlays (they wouldn't line up)
            data = {"south": _load_raw(sp)[:, :3], "north": _load_raw(npath)[:, :3]}
            tag, clip, road_on, roi_on, sensors = "RAW · unregistered", False, False, False, None
        else:
            # `f` is the fused cloud in s110_base. The road clip window + road/ROI
            # polygons are defined in the SOUTH frame, so clip there, then re-express
            # the (clipped) result in whichever frame is being viewed. Crop is now an
            # independent toggle that works in both frames.
            in_south = frame.startswith("South")
            Tf_s = reg.to_south_frame(Ms)                         # s110_base -> south
            s_pts = reg.transform_points(f["south"], Tf_s)
            n_pts = reg.transform_points(f["north"], Tf_s)
            if crop_road:
                win = reg._road_window()
                s_pts, n_pts = reg._clip_to_window(s_pts, win), reg._clip_to_window(n_pts, win)
            Tf = Tf_s if in_south else np.eye(4)                  # display frame (from base)
            if in_south:
                data = {"south": s_pts, "north": n_pts}
            else:                                                 # back to s110_base for display
                data = {"south": reg.transform_points(s_pts, Ms),
                        "north": reg.transform_points(n_pts, Ms)}
            clip = False                                          # already clipped above
            road_on, roi_on = (show_road and in_south), (show_roi and in_south)
            tag = ("REGISTERED · south frame" if in_south else "REGISTERED · s110_base") + \
                  (" · road-cropped" if crop_road else " · full")
            sensors = None
            if show_sensors:
                # sensor origin in base = translation column of its calibration; the
                # north sensor moves with the ICP correction when it's applied. Re-
                # express both in the chosen view frame (in south frame, South → origin).
                south_pos = Ms[:3, 3]
                n_pos = Mn[:3, 3]
                if use_refine:
                    n_pos = (np.array(delta_l) @ np.append(n_pos, 1.0))[:3]
                south_pos = reg.transform_points(np.asarray(south_pos).reshape(1, 3), Tf)[0]
                n_pos = reg.transform_points(np.asarray(n_pos).reshape(1, 3), Tf)[0]
                # vivid blue / red — coordinated with the south(blue)/north(orange-red)
                # clouds, but brighter (+ white outline) so the markers stand out.
                sensors = [{"name": "South", "pos": [float(v) for v in south_pos], "color": "#1e90ff"},
                           {"name": "North", "pos": [float(v) for v in n_pos], "color": "#ff1744"}]
        title = (f"{tag} · pair {i+1}/{len(pairs)} · Δt {dt_ms:.0f} ms · "
                 f"{len(data['south']):,}+{len(data['north']):,} pts")
        with st.container(height=680):
            st.plotly_chart(
                reg.registration_figure(
                    data, color_mode=("by_height" if color_mode == "By height" else "by_sensor"),
                    height_span=hspan, show_south=show_s, show_north=show_n,
                    show_road=road_on, show_roi=roi_on, clip=clip, sensors=sensors,
                    height=660, title=title, zoom=zoom, azimuth=float(az), elevation=float(el)),
                use_container_width=True, key="reg_fig", config={"scrollZoom": True})

        if playing and i < len(pairs) - 1:
            time.sleep(float(delay))
            st.session_state.reg_frame = i + 1
            st.rerun(scope="fragment")

    _reg_preview()

    # --- batch register ---
    st.divider()
    st.subheader("💾 Register all frames")
    st.text_input("Output (registered) folder", value=reg_out, key="reg_out", disabled=True)
    batch_refine = st.session_state.reg_delta if (use_refine and st.session_state.reg_delta) else None
    if batch_refine is not None:
        st.caption("ICP refinement **will** be baked into the written clouds (north corrected by the "
                   "last-run ICP delta).")
    else:
        st.caption("Pure calibration will be written (recommended). Run + apply ICP above to bake a "
                   "refinement instead.")
    if st.button("🧭 Register & write fused clouds", type="primary", use_container_width=True,
                 key="reg_go"):
        bar = st.progress(0.0, text="Registering…")
        n, total = reg.register_dataset(
            pcd_dirs["south"], pcd_dirs["north"], Ms, Mn, reg_out,
            refine=(np.array(batch_refine) if batch_refine is not None else None),
            progress=lambda c, t: bar.progress(c / t, text=f"Registering {c}/{t}"))
        bar.empty()
        st.success(f"Wrote **{n}** fused clouds (south LiDAR frame) → `{reg_out}`  "
                   "(manifest: `registration.json`). Now crop or score the **Registered (south + north)** "
                   "source in the other tabs — it reuses the south GT, calibration, and polygons.")

    # --- fused GT labels (union of south + north boxes) ---
    st.divider()
    st.subheader("🏷️ Build fused GT labels (south ∪ north)")
    reg_label_out = os.path.join(ds.derived_dir, "labels", "registered")
    st.caption("Each sensor only annotates the objects **it** can see, so the registered cloud — which "
               "borrows the **south** GT — is missing the objects only **north** saw (they have points in "
               "the fused cloud but no box). This builds a **union** GT: north boxes are transformed into "
               "the south frame and the ones south didn't annotate are added (shared objects keep their "
               "south box). Each box's **`num_points` is recomputed against the fused cloud** so the "
               "scorable gate is honest for registered (the stored counts are south-only). Afterwards, "
               "build the **Registered** scorable set in the 🏷️ Scorable GT tab.")
    fc1, fc2 = st.columns([1.4, 2])
    fuse_dist = fc1.slider("Duplicate-merge distance (m)", 0.5, 5.0, 2.5, 0.5,
                           help="A north box within this distance of a south box is treated as the SAME "
                                "object and kept once. Larger = more aggressive de-duplication.")
    fc2.text_input("Output (fused labels) folder", value=reg_label_out, key="reg_lbl_out", disabled=True)
    if st.button("🏷️ Build fused GT labels", use_container_width=True, key="reg_fuse_lbl"):
        bar = st.progress(0.0, text="Fusing labels…")
        nfr, added, shared = reg.fuse_labels(
            label_dirs["south"], label_dirs["north"], Ms, Mn, reg_label_out,
            refine=(np.array(batch_refine) if batch_refine is not None else None),
            match_dist=float(fuse_dist), registered_pcd_dir=ds.registered_dir,
            progress=lambda c, t: bar.progress(c / t, text=f"Fusing {c}/{t}"))
        bar.empty()
        st.success(f"Wrote **{nfr}** fused label files → `{reg_label_out}`  "
                   f"(**+{added}** north-only boxes added · {shared} shared de-duplicated). "
                   "Now build the **Registered (south + north)** scorable GT in the 🏷️ Scorable GT tab.")
