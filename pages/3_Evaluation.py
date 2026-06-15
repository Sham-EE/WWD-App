import streamlit as st
import os

from evaluation import evaluate, save_report, load_gt_by_key, bev_figure, _frame_key


@st.cache_data(show_spinner="Loading ground-truth frames…")
def _gt_cached(gt_dir):
    return load_gt_by_key(gt_dir)

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

gt_dir = st.text_input(
    "Ground-truth directory (OpenLABEL .json files)",
    value="data/labels_point_clouds/a9_gt_visible_only_south",
)
match_dist = st.slider("BEV match distance gate (m)", 0.5, 5.0, 2.0, 0.5,
                       help="Max centre distance for a prediction to count as a true positive.")
output_dir = "outputs/object_detection"

if st.button("📐 Run Evaluation", use_container_width=True, type="primary"):
    if not os.path.isdir(gt_dir):
        st.error(f"GT directory not found: {gt_dir}")
    else:
        report = evaluate(results['det_frames'], results['pcd_files'], gt_dir, match_dist=match_dist)
        s = report['summary']
        if s['evaluated_frames'] == 0:
            st.warning("No detection frames aligned to GT files. Check that .pcd and .json "
                       "filenames share the same leading <timestamp1>_<timestamp2> token.")
        else:
            st.success(f"Evaluated {s['evaluated_frames']} frame(s) "
                       f"({report['gt_frames_available']} GT files available).")
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{s['precision']:.3f}")
            c2.metric("Recall", f"{s['recall']:.3f}")
            c3.metric("F1", f"{s['f1']:.3f}")
            c4, c5, c6 = st.columns(3)
            c4.metric("MOTA", f"{s['MOTA']:.3f}")
            c5.metric("MOTP (m)", f"{s['MOTP_m']:.3f}")
            c6.metric("ID switches", s['ID_switches'])
            st.caption(f"TP={s['TP']}  FP={s['FP']}  FN={s['FN']}  GT objects={s['gt_objects_total']}  "
                       f"(match gate = {s['match_dist_m']} m)")

            with st.expander("Per-frame breakdown"):
                st.dataframe(report['per_frame'], use_container_width=True)

            json_path, csv_path = save_report(report, output_dir)
            st.info(f"Saved report to `{json_path}` and `{csv_path}`.")

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
    n = len(pcd_files)
    rb = results.get('research_poly_bounds', (-25, -50, 45, 50))
    x_range, y_range = (rb[0], rb[2]), (rb[1], rb[3])

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
                     color='#2ca02c', label=str(b['cls'])[:3]) for b in gt_raw]
    det_boxes = [dict(cx=d['cx'], cy=d['cy'], yaw=d.get('yaw', 0.0),
                      l=d.get('l', 4.5), w=d.get('w', 1.9),
                      color='orange' if d.get('wrong_way') else '#1f77b4',
                      label=str(d.get('tid', ''))) for d in results['det_frames'][i]]

    if key not in gt_by_key:
        st.warning(f"No GT frame matches this detection frame (key {key}).")
    g, d = st.columns(2)
    g.plotly_chart(bev_figure(gt_boxes, f"Ground Truth — {len(gt_boxes)} objects (frame {i})",
                              x_range, y_range, default_color='#2ca02c'),
                   use_container_width=True, key="ev_gt")
    d.plotly_chart(bev_figure(det_boxes, f"Detection — {len(det_boxes)} objects (frame {i})",
                              x_range, y_range, default_color='#1f77b4'),
                   use_container_width=True, key="ev_det")
    st.caption(f"Frame {i}/{n-1} · GT objects: {len(gt_boxes)} · Detections: {len(det_boxes)}")
