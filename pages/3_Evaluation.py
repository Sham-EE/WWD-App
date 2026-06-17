import streamlit as st
import os
import numpy as np

from evaluation import (evaluate, save_report, load_gt_by_key, bev_figure,
                        _frame_key, _match_frame, BEV_CONFIG)


@st.cache_data(show_spinner="Loading ground-truth frames…")
def _gt_cached(gt_dir):
    return load_gt_by_key(gt_dir)


@st.cache_data(show_spinner=False)
def _pcd_xy(path, grid=0.3):
    """Downsampled top-down (X, Y) of a point cloud frame, for the BEV backdrop."""
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(path).points)
    if pts.size == 0:
        return np.zeros((0, 2))
    xy = pts[:, :2]
    _, idx = np.unique(np.floor(xy / grid).astype(np.int64), axis=0, return_index=True)
    return xy[idx]

st.set_page_config(layout="wide", page_title="Evaluation")
st.title("📊 Quantitative Evaluation")

st.markdown(
    """
    Evaluate the most recent detection/tracking run against ground-truth cuboids.
    Reports **precision / recall / F1** (detection) and **MOTA / MOTP / ID-switches**
    (tracking), matching predictions to GT by bird's-eye-view centre distance.

    Run **Object Detection and Tracking** first — this page reuses its results.
    """
)

if not st.session_state.get('detection_results'):
    st.info("No detection results in memory. Open the 'Object Detection and Tracking' "
            "page and run a detection first.")
    st.stop()

results = st.session_state.detection_results

import dataset_manager as dm
_ds = dm.get_active()
st.caption(f"📂 Dataset: **{_ds.name}**")

gt_dir = st.text_input(
    "Ground-truth directory (OpenLABEL .json files)",
    value=_ds.gt_dir,
)
ec1, ec2, ec3 = st.columns(3)
match_dist = ec1.slider("BEV match distance gate (m)", 0.5, 5.0, 2.0, 0.5,
                        help="Max centre distance for a prediction to count as a true positive.")
veh_only = ec2.checkbox("Vehicles only", value=False,
                        help="Score only CAR/TRUCK/VAN/BUS/TRAILER/MOTORCYCLE (exclude pedestrians & bicycles).")
roi_only = ec3.checkbox("Restrict to processed region (ROI)", value=True,
                        help="Only score GT inside the area the detector actually processes "
                             "(research polygon ∩ |y|≤ROI). Objects outside the sensor's operational "
                             "region aren't counted as misses — this is the fair number.")
output_dir = _ds.detection_dir

if st.button("📐 Run Evaluation", use_container_width=True, type="primary"):
    if not os.path.isdir(gt_dir):
        st.error(f"GT directory not found: {gt_dir}")
    else:
        classes = {'CAR', 'TRUCK', 'VAN', 'BUS', 'TRAILER', 'MOTORCYCLE'} if veh_only else None
        rb = results.get('research_poly_bounds', (-25, -50, 45, 50))
        roi_y = float(results.get('params', {}).get('roi_abs_y', 40.0))
        roi_bounds = (rb[0], rb[2], -roi_y, roi_y) if roi_only else None
        report = evaluate(results['det_frames'], results['pcd_files'], gt_dir,
                          match_dist=match_dist, classes=classes, roi_bounds=roi_bounds)
        s = report['summary']
        if s['evaluated_frames'] == 0:
            st.warning("No detection frames aligned to GT files. Check that .pcd and .json "
                       "filenames share the same leading <timestamp1>_<timestamp2> token.")
        else:
            st.success(f"Evaluated {s['evaluated_frames']} frame(s) "
                       f"({report['gt_frames_available']} GT files available).")
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{s['precision']:.3f}",
                      help="Of everything the detector reported, the fraction that was a real object. "
                           "TP / (TP + FP). **Higher is better** (1.0 = no false positives). Low precision "
                           "= too many spurious detections.")
            c2.metric("Recall", f"{s['recall']:.3f}",
                      help="Of all the ground-truth objects, the fraction the detector found. "
                           "TP / (TP + FN). **Higher is better** (1.0 = nothing missed). Low recall = "
                           "missing real objects.")
            c3.metric("F1", f"{s['f1']:.3f}",
                      help="Harmonic mean of precision and recall — a single balanced score. "
                           "2·P·R / (P + R). **Higher is better.** Good when you want both few misses and "
                           "few false alarms.")
            c4, c5, c6 = st.columns(3)
            c4.metric("MOTA", f"{s['MOTA']:.3f}",
                      help="Multi-Object Tracking Accuracy: overall error rate combining misses, false "
                           "positives and identity switches. 1 − (FN + FP + IDSW) / GT. **Higher is "
                           "better** (1.0 = perfect; can go negative if errors exceed the number of GT "
                           "objects).")
            c5.metric("MOTP (m)", f"{s['MOTP_m']:.3f}",
                      help="Multi-Object Tracking Precision: average distance (metres) between a matched "
                           "detection and its ground-truth centre — how *accurately* matched boxes are "
                           "placed. **Lower is better** (0 = perfect localization).")
            c6.metric("ID switches", s['ID_switches'],
                      help="How many times a tracked object's ID changed to a different ground-truth "
                           "object across frames (identity got swapped/lost and reassigned). "
                           "**Lower is better** (0 = every object kept one consistent ID).")
            st.caption(f"TP={s['TP']}  FP={s['FP']}  FN={s['FN']}  GT objects={s['gt_objects_total']}  "
                       f"(match gate = {s['match_dist_m']} m)")

            # Record the exact settings that produced this result, plus the
            # detection parameters from the last run, so a number is never ambiguous.
            import datetime
            report['config']['evaluated_at'] = datetime.datetime.now().isoformat(timespec='seconds')
            report['config']['detection_params'] = results.get('params', {})

            st.markdown("**⚙️ Settings used** (saved with the report):")
            cfg = report['config']
            roi_txt = (f"x[{cfg['roi_bounds'][0]:.0f}, {cfg['roi_bounds'][1]:.0f}] · "
                       f"|y|≤{abs(cfg['roi_bounds'][2]):.0f}" if cfg['roi_bounds'] else "off (full area)")
            st.caption(
                f"Match gate: **{cfg['match_dist_m']} m**  ·  Scored classes: **{cfg['scored_classes']}**  ·  "
                f"ROI: **{roi_txt}**  ·  Ignored (don't-care) GT present: **{cfg['ignore_classes_present']}**  ·  "
                f"Run at {cfg['evaluated_at']}")
            with st.expander("🔧 Full settings (readable JSON)"):
                st.json(report['config'])

            with st.expander("Per-frame breakdown"):
                st.dataframe(report['per_frame'], use_container_width=True)

            json_path, csv_path = save_report(report, output_dir)
            st.info(f"Saved report to `{json_path}` and `{csv_path}`.")
            with st.expander("📄 Full report summary (readable JSON)"):
                st.json(report['summary'])
                st.caption("Full report (incl. per-frame rows + settings) saved at the path above.")

# ---------------- Visual evaluation: GT vs Detection, side-by-side top-down ----------------
st.divider()
st.subheader("🔍 Visual Evaluation — Ground Truth vs Detection (top-down)")
st.caption("Step through frames to compare human-annotated boxes (left) against the algorithm's "
           "detections (right). Both use the same sensor X/Y axes, so the LiDAR blind spot and "
           "every object line up.")

if not os.path.isdir(gt_dir):
    st.warning(f"GT directory not found: {gt_dir}")
else:
    gt_by_key = _gt_cached(gt_dir)
    pcd_files = results['pcd_files']
    orig_files = results.get('original_pcd_files', pcd_files)
    n = len(pcd_files)

    # Default to data-driven bounds (an "auto-fit") so the whole scene is visible
    # without pressing autoscale; shared by both panels so they stay aligned.
    gxs, gys = [], []
    for dets in results['det_frames']:
        for d in dets:
            gxs.append(d['cx']); gys.append(d['cy'])
    for boxes in gt_by_key.values():
        for b in boxes:
            gxs.append(b['cx']); gys.append(b['cy'])
    if gxs:
        m = 6.0  # margin so edge boxes (and their extent) aren't clipped
        x_range = (min(gxs) - m, max(gxs) + m)
        y_range = (min(gys) - m, max(gys) + m)
    else:
        rb = results.get('research_poly_bounds', (-25, -50, 45, 50))
        x_range, y_range = (rb[0], rb[2]), (rb[1], rb[3])

    # --- view options ---
    opt = st.columns(4)
    separate = opt[0].toggle("Separate views", value=True,
                             help="On = GT and Detection side by side. Off = both overlaid in one "
                                  "plot (GT drawn on top of Detection).")
    show_pc = opt[1].checkbox("Show point cloud", value=False,
                              help="Overlay the LiDAR point cloud (top-down) behind the boxes.")
    show_missed = opt[2].checkbox("Show missed (red)", value=False, disabled=not separate,
                                  help="In separate view, draw GT objects that have NO matching "
                                       "detection (false negatives) in red on the Detection panel.")

    st.session_state.setdefault('ev_frame', 0)
    st.session_state.ev_frame = max(0, min(st.session_state.ev_frame, n - 1))
    nav = st.columns([1, 1, 1, 1, 4])
    if nav[0].button("⏮ First", use_container_width=True):
        st.session_state.ev_frame = 0; st.rerun()
    if nav[1].button("◀ Prev", use_container_width=True):
        st.session_state.ev_frame = max(0, st.session_state.ev_frame - 1); st.rerun()
    if nav[2].button("Next ▶", use_container_width=True):
        st.session_state.ev_frame = min(n - 1, st.session_state.ev_frame + 1); st.rerun()
    if nav[3].button("Last ⏭", use_container_width=True):
        st.session_state.ev_frame = n - 1; st.rerun()
    i = st.slider("Frame", 0, max(n - 1, 1), st.session_state.ev_frame)
    st.session_state.ev_frame = i

    key = _frame_key(pcd_files[i])
    gt_raw = gt_by_key.get(key, [])
    gt_boxes = [dict(cx=b['cx'], cy=b['cy'], yaw=b.get('yaw', 0.0), l=b['l'], w=b['w'],
                     label=str(b['cls'])[:3]) for b in gt_raw]
    det_boxes = [dict(cx=d['cx'], cy=d['cy'], yaw=d.get('yaw', 0.0),
                      l=d.get('l', 4.5), w=d.get('w', 1.9),
                      label=str(d.get('tid', ''))) for d in results['det_frames'][i]]

    bg_xy = _pcd_xy(orig_files[i]) if (show_pc and i < len(orig_files) and os.path.exists(orig_files[i])) else None

    if key not in gt_by_key:
        st.warning(f"No GT frame matches this detection frame (key {key}).")

    if separate:
        # Compute missed GT (false negatives) for the detection panel.
        missed_boxes = []
        if show_missed and (gt_boxes or det_boxes):
            matches, _, _, _ = _match_frame(det_boxes, gt_boxes, float(match_dist))
            matched_gt = {gj for _, gj, _ in matches}
            missed_boxes = [gt_boxes[j] for j in range(len(gt_boxes)) if j not in matched_gt]
        g, d = st.columns(2)
        g.plotly_chart(
            bev_figure([(gt_boxes, '#2ca02c')], f"Ground Truth — {len(gt_boxes)} objects (frame {i})",
                       x_range, y_range, bg_xy=bg_xy, uirev='ev_left'),
            use_container_width=True, key="ev_gt", config=BEV_CONFIG)
        det_groups = [(det_boxes, '#1f77b4')]
        if missed_boxes:
            det_groups.append((missed_boxes, '#ff2b2b'))
        title = f"Detection — {len(det_boxes)} objects (frame {i})"
        if show_missed:
            title += f" · {len(missed_boxes)} missed"
        d.plotly_chart(bev_figure(det_groups, title, x_range, y_range, bg_xy=bg_xy, uirev='ev_right'),
                       use_container_width=True, key="ev_det", config=BEV_CONFIG)
    else:
        # Overlap: detection first (bottom), GT on top, distinguishable by color.
        fig = bev_figure([(det_boxes, '#1f77b4'), (gt_boxes, '#2ca02c')],
                         f"Overlay — GT (green) over Detection (blue), frame {i}",
                         x_range, y_range, height=680, bg_xy=bg_xy, uirev='ev_overlay')
        st.plotly_chart(fig, use_container_width=True, key="ev_overlay", config=BEV_CONFIG)

    st.caption(f"Frame {i}/{n-1} · GT objects: {len(gt_boxes)} · Detections: {len(det_boxes)}"
               + ("  ·  🟩 GT  🟦 Detection" + ("  🟥 missed" if (separate and show_missed) else "")))
