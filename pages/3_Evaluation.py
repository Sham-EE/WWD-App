import streamlit as st
import os

from evaluation import evaluate, save_report

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
