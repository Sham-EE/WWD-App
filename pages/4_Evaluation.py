import glob
import os

import numpy as np
import pandas as pd
import streamlit as st

import dataset_manager as dm
from detection_logic import run_detection_and_tracking, DEFAULT_DETECTION_PARAMS
from evaluation import (evaluate, recall_by_distance, save_report, load_gt_by_key,
                        bev_figure, _frame_key, _match_frame, BEV_CONFIG)
import viewer_ui as vu
import run_history as rh


def _eval_metrics(s):
    """Flatten an evaluate() summary into the metrics dict we log per run."""
    return {"precision": round(100 * s["precision"], 2), "recall": round(100 * s["recall"], 2),
            "f1": round(100 * s["f1"], 2), "MOTA": round(100 * s["MOTA"], 2),
            "MOTP_m": round(s["MOTP_m"], 3), "TP": s["TP"], "FP": s["FP"], "FN": s["FN"],
            "ID_switches": s["ID_switches"], "gt_total": s["gt_objects_total"],
            "frames": s["evaluated_frames"]}


def _render_eval_history(_ds, tag):
    """Persistent eval-run history: current-vs-previous deltas + a trend + a param diff +
    the full log, mirroring the Background-Filtering run tracker but for the real
    detection metrics. Reads outputs/run_history/<tag>.jsonl."""
    hist = rh.load_history(_ds, tag)
    if not hist:
        st.caption("No logged eval runs yet for this selection — each **Run Evaluation** appends one.")
        return
    cur = hist[-1]["metrics"]
    prev = hist[-2]["metrics"] if len(hist) > 1 else None
    m = st.columns(4)
    def _d(k, fmt="{:+.1f}"):
        return fmt.format(cur[k] - prev[k]) if prev else None
    m[0].metric("Precision", f"{cur['precision']:.1f}%", _d("precision"))
    m[1].metric("Recall", f"{cur['recall']:.1f}%", _d("recall"))
    m[2].metric("F1", f"{cur['f1']:.1f}%", _d("f1"))
    m[3].metric("FP", f"{cur['FP']:,}", _d("FP", "{:+d}"), delta_color="inverse")
    df = pd.DataFrame([{"run": i + 1, "P": h["metrics"]["precision"], "R": h["metrics"]["recall"],
                        "F1": h["metrics"]["f1"]} for i, h in enumerate(hist)]).set_index("run")
    st.line_chart(df)
    if prev:
        diff = rh.param_diff(hist[-2]["params"], hist[-1]["params"])
        st.caption("🔧 Changed since previous run: "
                   + (", ".join(f"`{k}` {a}→{b}" for k, (a, b) in diff.items())
                      if diff else "nothing tracked changed."))
    with st.expander("Full eval log"):
        st.dataframe(pd.DataFrame([{"time": h["time"], "note": h.get("note", ""), **h["metrics"]}
                                   for h in hist]), use_container_width=True)


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

_ds = dm.get_active()
st.caption(f"📂 Dataset: **{_ds.name}**")


# ===================== Tab 1: Single run =====================
def _render_single_run():
    st.markdown(
        "Evaluate the most recent detection/tracking run against ground-truth cuboids. "
        "Reports **precision / recall / F1** (detection) and **MOTA / MOTP / ID-switches** "
        "(tracking), matching predictions to GT by bird's-eye-view centre distance. "
        "Run **Object Detection and Tracking** first — this tab reuses its results."
    )
    results = st.session_state.get('detection_results')
    if not results:
        st.info("No detection results in memory. Open the 'Object Detection and Tracking' "
                "page and run a detection first.")
        return

    _sc, _ic = st.columns(2)
    _sensor_label = _sc.radio("Sensor", ["Registered", "South", "North"],
                              key="pipeline_sensor", horizontal=True,
                              help="Which LiDAR's run to score. Must match what you detected; GT auto-resolves "
                                   "to this sensor. Shared across pages.")
    _sensor = _sensor_label.lower()
    _src_label = _ic.radio("Input cloud (which run to score)", ["Cropped (road)", "Full (uncropped)"],
                           key="pipeline_source", horizontal=True,
                           help="Scores the detection results in memory; pick the source you just ran so the "
                                "report saves to the matching folder. Run the pipeline once per source, then "
                                "compare these metrics.")
    _src = "cropped" if _src_label.startswith("Cropped") else "full"

    gt_dir = st.text_input(
        "Ground-truth directory (OpenLABEL .json files)",
        value=_ds.gt_dir_for_input(_ds.input_pcd_for_sensor(_sensor, _src)),
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
    output_dir = _ds.detection_dir_for_sensor(_sensor, _src)
    eval_note = st.text_input("Run note (optional — logged with this eval)", key="eval_note_single",
                              placeholder="e.g. strong_pts=100, suppress_static on")

    # Cross-check: the in-memory detection must be for the sensor/source being scored,
    # otherwise we'd match one sensor's frames against another's GT (the cryptic
    # "no frames aligned" error). Surface it clearly instead.
    _r_sensor, _r_source = results.get('sensor'), results.get('source')
    _mismatch = (_r_sensor and _r_sensor != _sensor) or (_r_source and _r_source != _src)
    _r_src_lbl = 'Cropped' if _r_source == 'cropped' else 'Full'
    if _mismatch:
        st.warning(f"⚠️ Loaded detection is for **{(_r_sensor or '?').capitalize()} · {_r_src_lbl}**, "
                   f"but you're scoring **{_sensor_label} · {_src_label}**. Re-run **Start Detection** for "
                   f"this selection first (or switch the toggles above to match the loaded run).")

    if st.button("📐 Run Evaluation", use_container_width=True, type="primary"):
        if _mismatch:
            st.error(f"Can't evaluate: in-memory detection is for {(_r_sensor or '?').capitalize()} · "
                     f"{_r_src_lbl}, not {_sensor_label} · {_src_label}. Re-run Detection for this "
                     "selection, then evaluate.")
        elif not os.path.isdir(gt_dir):
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

                # Append to the PERSISTENT eval history (settings + metrics) so runs are
                # comparable over time — save_report only keeps the latest report.
                _dp = results.get('params', {})
                _eparams = {"match_dist": match_dist, "vehicles_only": bool(veh_only),
                            "roi": bool(roi_only), "gt": os.path.basename(gt_dir.rstrip('/'))}
                for _k in ("strong_pts", "suppress_static", "static_max_speed", "truck_merge_dist",
                           "min_cluster_pts", "min_hits", "adaptive_eps", "dbscan_eps"):
                    if _k in _dp:
                        _eparams[_k] = _dp[_k]
                rh.log_run(_ds, f"eval_{_sensor}_{_src}", _eval_metrics(s), _eparams, note=eval_note)

    st.divider()
    st.markdown("#### 📈 Eval history — this sensor · source")
    st.caption("Persistent log of every evaluation you've run for this selection (settings + metrics), "
               "with current-vs-previous deltas. Stored in `outputs/run_history/`.")
    _render_eval_history(_ds, f"eval_{_sensor}_{_src}")

    # ---------------- Visual evaluation: GT vs Detection, side-by-side top-down ----------------
    st.divider()
    st.subheader("🔍 Visual Evaluation — Ground Truth vs Detection (top-down)")
    st.caption("Step through frames to compare human-annotated boxes (left) against the algorithm's "
               "detections (right). Both use the same sensor X/Y axes, so the LiDAR blind spot and "
               "every object line up.")

    if not os.path.isdir(gt_dir):
        st.warning(f"GT directory not found: {gt_dir}")
        return
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

    i, playing, play_delay = vu.nav_row("ev_frame", n, "ev")

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

    if playing and i < n - 1:
        import time
        time.sleep(float(play_delay))
        st.session_state.ev_frame = i + 1
        st.rerun()


# ===================== Tab 2: Registered vs South (A/B) =====================
# Identical detection params for BOTH runs — the whole point of the controlled
# comparison. Uses the shared DEFAULT_DETECTION_PARAMS (single source of truth) so the
# A/B and the live detector can't drift.
_AB_PARAMS = DEFAULT_DETECTION_PARAMS
_VEHICLE_CLASSES = {"CAR", "TRUCK", "VAN", "BUS", "TRAILER", "MOTORCYCLE", "EMERGENCY_VEHICLE"}
_DIST_BINS = (0, 20, 40, 60, 1e9)
_AB_SENSORS = ["south", "registered"]


def _render_ab():
    st.markdown(
        "Run the **same** detection on the **South** and **Registered** filtered clouds and score each "
        "against its own scorable GT — a controlled test of whether fusion (south + north) actually "
        "improves detection, especially **recall on far / occluded objects**. Both clouds are in the "
        "south frame, so ranges-from-sensor and the ROI are directly comparable (apples-to-apples)."
    )
    c1, c2, c3 = st.columns(3)
    src_label = c1.radio("Input cloud", ["Cropped (road)", "Full (uncropped)"], horizontal=True, key="ab_src",
                         help="Filtered clouds to score (must be built in Background Filtering for BOTH sensors).")
    src = "cropped" if src_label.startswith("Cropped") else "full"
    match_dist = c2.slider("Match distance (m)", 0.5, 5.0, 2.0, 0.5, key="ab_md",
                           help="BEV centre gate for matching a detection to a GT box.")
    vehicles_only = c3.checkbox("Vehicles only", value=False, key="ab_veh",
                                help="Score only vehicle classes (peds/bikes become ignore, not penalised).")
    roi_on = st.checkbox("Restrict to processed ROI", value=True, key="ab_roi",
                         help="Only score GT inside the research region — the fair number.")
    classes = _VEHICLE_CLASSES if vehicles_only else None

    g1, g2 = st.columns(2)
    score_mode = g1.radio("Score both against", ["Registered union (shared)", "Each sensor's own GT"],
                          horizontal=True, key="ab_gtmode",
                          help="**Registered union (shared)** = grade BOTH pipelines against the SAME "
                               "objective GT — the registered union, which includes the north-only vehicles "
                               "south's GT lacks. So south is fairly penalised for objects it physically "
                               "can't see (the limitation fusion fixes), and recall shares one denominator. "
                               "**Each sensor's own GT** = grade each on its own annotations (south is "
                               "flattered by its incomplete GT).")
    gt_kind = g2.radio("GT boxes", ["Scorable", "All (raw)"], horizontal=True, key="ab_gtkind",
                       help="Scorable = only objects with enough LiDAR points to be detectable (num_points "
                            "recomputed on the fused cloud). All (raw) = every annotated box, including ones "
                            "too sparse to fairly detect.")
    shared_gt = score_mode.startswith("Registered union")
    gt_kind_key = "raw" if gt_kind.startswith("All") else "scorable"

    roi_bounds = None
    if roi_on:
        try:
            from geometry_config import get_research_polygon
            b = get_research_polygon().bounds  # minx, miny, maxx, maxy
            roi_bounds = (b[0], b[2], b[1], b[3])  # x0, x1, y0, y1
        except Exception:
            roi_bounds = None

    with st.expander("⚙️ Detection parameters (identical for both runs)", expanded=False):
        st.json(_AB_PARAMS, expanded=False)
        st.caption("Fixed so the only thing differing between the two runs is the input cloud "
                   "(south-only vs fused). Tune the live detector on the Object Detection page.")

    ready = True
    rc = st.columns(2)
    for col, s in zip(rc, _AB_SENSORS):
        fdir = _ds.filtered_dir_for_sensor(s, src)
        nf = len(glob.glob(os.path.join(fdir, "*.pcd"))) if os.path.isdir(fdir) else 0
        if nf == 0:
            col.error(f"**{s.capitalize()}** — no filtered clouds. Build the BG model for "
                      f"**{s.capitalize()} · {src_label}** first.")
            ready = False
        else:
            col.success(f"**{s.capitalize()}** — {nf} filtered clouds ready.")

    if st.button("▶ Run A/B comparison", type="primary", use_container_width=True, disabled=not ready):
        out = {}
        bar = st.progress(0.0, text="Starting…")
        for si, s in enumerate(_AB_SENSORS):
            fdir = _ds.filtered_dir_for_sensor(s, src)
            outdir = _ds.detection_dir_for_sensor(s, src)
            # Shared objective GT = the registered union for both; else each sensor's own.
            gtdir = _ds.labels_dir_for("registered" if shared_gt else s, gt_kind_key)

            def _pcb(c, t, m, _s=s, _si=si):
                bar.progress((_si + c / max(t, 1)) / len(_AB_SENSORS), text=f"{_s.capitalize()}: {m} {c}/{t}")

            res, err = run_detection_and_tracking(fdir, outdir, _AB_PARAMS, _pcb)
            if err:
                st.error(f"{s.capitalize()}: {err}")
                continue
            det_frames, paths = res["det_frames"], res["pcd_files"]
            rep = evaluate(det_frames, paths, gtdir, match_dist=match_dist, classes=classes, roi_bounds=roi_bounds)
            rbd = recall_by_distance(det_frames, paths, gtdir, bins=_DIST_BINS, match_dist=match_dist,
                                     classes=classes, roi_bounds=roi_bounds)
            out[s] = {"summary": rep["summary"], "rbd": rbd, "gt_dir": os.path.basename(gtdir.rstrip("/"))}
            # Log each arm to the persistent eval history (shared GT mode in the tag so
            # shared-vs-own-GT runs don't mix), so A/B runs are comparable over time too.
            _abtag = f"eval_ab_{s}_{src}_{'shared' if shared_gt else 'own'}"
            _abparams = {"match_dist": match_dist, "vehicles_only": bool(vehicles_only),
                         "roi": bool(roi_on), "gt_kind": gt_kind_key, "gt": out[s]["gt_dir"],
                         "strong_pts": _AB_PARAMS.get("strong_pts"),
                         "suppress_static": _AB_PARAMS.get("suppress_static"),
                         "truck_merge_dist": _AB_PARAMS.get("truck_merge_dist")}
            rh.log_run(_ds, _abtag, _eval_metrics(rep["summary"]), _abparams,
                       note=f"A/B {s}·{src}·{'shared' if shared_gt else 'own'} GT")
        bar.empty()
        st.session_state.ab_results = {"results": out, "src": src_label, "match_dist": match_dist,
                                       "vehicles_only": vehicles_only, "roi_on": roi_on,
                                       "gt_mode": score_mode, "gt_kind": gt_kind, "shared": shared_gt}

    state = st.session_state.get("ab_results")
    if not (state and all(s in state["results"] for s in _AB_SENSORS)):
        st.info("Pick the input cloud + scoring options, then **Run A/B comparison**. "
                "Both South and Registered need their filtered clouds built first (Background Filtering).")
        return

    res = state["results"]
    ss, rs = res["south"]["summary"], res["registered"]["summary"]
    cfg = (f"{state['src']} · match {state['match_dist']} m · "
           f"{'vehicles only' if state['vehicles_only'] else 'all classes'} · "
           f"{'ROI' if state['roi_on'] else 'no ROI'} · GT: {state.get('gt_kind', '')}")
    st.divider()
    st.subheader("Overall metrics")
    if state.get("shared"):
        st.caption(f"⚖️ Both scored against the **shared registered-union GT** "
                   f"(`{res['south']['gt_dir']}`, {ss['gt_objects_total']} objects — same denominator)  ·  {cfg}")
    else:
        st.caption(f"South GT: `{res['south']['gt_dir']}` · Registered GT: `{res['registered']['gt_dir']}`  "
                   f"·  {cfg}")

    def _delta(rv, sv):
        d = rv - sv
        return f"{d:+.4f}" if isinstance(d, float) else f"{d:+d}"

    metrics = [("recall", "Recall ↑"), ("precision", "Precision ↑"), ("f1", "F1 ↑"),
               ("MOTA", "MOTA ↑"), ("MOTP_m", "MOTP (m) ↓"), ("TP", "TP ↑"), ("FP", "FP ↓"),
               ("FN", "FN ↓"), ("ID_switches", "ID switches ↓"), ("gt_objects_total", "GT objects")]
    rows = [{"Metric": label, "South": ss[k], "Registered": rs[k], "Δ (Reg−South)": _delta(rs[k], ss[k])}
            for k, label in metrics]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    dr = rs["recall"] - ss["recall"]
    (st.success if dr > 0 else st.warning if dr == 0 else st.error)(
        f"Recall {'+' if dr >= 0 else ''}{dr*100:.1f} pts on Registered vs South "
        f"({ss['recall']*100:.1f}% → {rs['recall']*100:.1f}%).")

    st.subheader("Recall by distance — the occlusion-shadow test")
    st.caption("Range = BEV distance from the south sensor. The fusion hypothesis predicts the biggest "
               "recall gains in the **far** bins, where south alone has occlusion shadows.")
    sb = {d["bin"]: d for d in res["south"]["rbd"]}
    rb = {d["bin"]: d for d in res["registered"]["rbd"]}
    drows, chart = [], {"South": [], "Registered": []}
    for b in sb:
        s_r, r_r = sb[b]["recall"], rb[b]["recall"]
        drows.append({"Distance": b,
                      "South recall": (round(s_r, 3) if s_r is not None else None),
                      "South m/n": f"{sb[b]['matched']}/{sb[b]['total']}",
                      "Registered recall": (round(r_r, 3) if r_r is not None else None),
                      "Registered m/n": f"{rb[b]['matched']}/{rb[b]['total']}",
                      "Δ": (round(r_r - s_r, 3) if (s_r is not None and r_r is not None) else None)})
        chart["South"].append(s_r if s_r is not None else np.nan)
        chart["Registered"].append(r_r if r_r is not None else np.nan)
    st.dataframe(pd.DataFrame(drows), use_container_width=True, hide_index=True)
    st.bar_chart(pd.DataFrame(chart, index=list(sb)))

    st.divider()
    with st.expander("📈 A/B eval history (per arm — current source + GT mode)", expanded=False):
        st.caption("Every A/B run appends to a persistent log. Trend below is for the current Input "
                   "cloud + scoring mode; switch those to see other histories.")
        for _s in _AB_SENSORS:
            st.markdown(f"**{_s.capitalize()}**")
            _render_eval_history(_ds, f"eval_ab_{_s}_{src}_{'shared' if shared_gt else 'own'}")


tab_single, tab_ab = st.tabs(["📐 Single run", "📊 Registered vs South (A/B)"])
with tab_single:
    _render_single_run()
with tab_ab:
    _render_ab()
