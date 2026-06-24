import glob
import os

import numpy as np
import pandas as pd
import streamlit as st

import dataset_manager as dm
from detection_logic import run_detection_and_tracking
from evaluation import evaluate, recall_by_distance

st.set_page_config(layout="wide", page_title="Registered vs South (A/B)")
st.title("📊 Registered vs South — A/B")

ds = dm.get_active()
st.caption(f"📂 Dataset: **{ds.name}**")
st.markdown(
    "Run the **same** detection on the **South** and **Registered** filtered clouds and score each "
    "against its own scorable GT — a controlled test of whether fusion (south + north) actually "
    "improves detection, especially **recall on far / occluded objects**. Both clouds are in the "
    "south frame, so ranges-from-sensor and the ROI are directly comparable (apples-to-apples)."
)

# Identical detection params for BOTH runs (the detection-page defaults) — this is the
# whole point of the controlled comparison, so they're fixed rather than per-run.
PARAMS = {
    "dbscan_eps": 2.0, "min_cluster_pts": 1, "min_hits": 2, "roi_abs_y": 40.0, "yaw_bias_deg": -90.0,
    "fps": 10.0, "max_missed": 5, "moving_speed_thresh": 3.0,
    "merge_dist": 2.5, "yaw_merge_deg": 15.0, "truck_len_thresh": 7.0, "truck_merge_dist": 10.0,
    "vehicle_gate": False, "vehicle_min_length": 2.5, "vehicle_min_points": 40,
    "adaptive_eps": True, "aeps0": 0.8, "aeps_k": 0.04, "aeps_min": 1.0, "aeps_max": 3.0,
}
VEHICLE_CLASSES = {"CAR", "TRUCK", "VAN", "BUS", "TRAILER", "MOTORCYCLE", "EMERGENCY_VEHICLE"}
DIST_BINS = (0, 20, 40, 60, 1e9)
SENSORS = ["south", "registered"]

# --- shared scoring knobs ---
c1, c2, c3 = st.columns(3)
_src_label = c1.radio("Input cloud", ["Cropped (road)", "Full (uncropped)"], horizontal=True, key="ab_src",
                      help="Filtered clouds to score (must be built in Background Filtering for BOTH sensors).")
_src = "cropped" if _src_label.startswith("Cropped") else "full"
match_dist = c2.slider("Match distance (m)", 0.5, 5.0, 2.0, 0.5, key="ab_md",
                       help="BEV centre gate for matching a detection to a GT box.")
vehicles_only = c3.checkbox("Vehicles only", value=False, key="ab_veh",
                            help="Score only vehicle classes (peds/bikes become ignore, not penalised).")
roi_on = st.checkbox("Restrict to processed ROI", value=True, key="ab_roi",
                     help="Only score GT inside the research region — the fair number.")
classes = VEHICLE_CLASSES if vehicles_only else None

roi_bounds = None
if roi_on:
    try:
        from geometry_config import get_research_polygon
        b = get_research_polygon().bounds  # minx, miny, maxx, maxy
        roi_bounds = (b[0], b[2], b[1], b[3])  # x0, x1, y0, y1
    except Exception:
        roi_bounds = None

with st.expander("⚙️ Detection parameters (identical for both runs)", expanded=False):
    st.json(PARAMS, expanded=False)
    st.caption("Fixed so the only thing that differs between the two runs is the input cloud "
               "(south-only vs fused). Tune the live detector on the Object Detection page.")

# --- readiness check: both sensors need filtered clouds ---
ready = True
rc = st.columns(2)
for col, s in zip(rc, SENSORS):
    fdir = ds.filtered_dir_for_sensor(s, _src)
    n = len(glob.glob(os.path.join(fdir, "*.pcd"))) if os.path.isdir(fdir) else 0
    if n == 0:
        col.error(f"**{s.capitalize()}** — no filtered clouds. Build the BG model for "
                  f"**{s.capitalize()} · {_src_label}** first.")
        ready = False
    else:
        col.success(f"**{s.capitalize()}** — {n} filtered clouds ready.")

run = st.button("▶ Run A/B comparison", type="primary", use_container_width=True, disabled=not ready)

if run:
    results = {}
    bar = st.progress(0.0, text="Starting…")
    for si, s in enumerate(SENSORS):
        fdir = ds.filtered_dir_for_sensor(s, _src)
        outdir = ds.detection_dir_for_sensor(s, _src)
        gtdir = ds.gt_dir_for_input(ds.input_pcd_for_sensor(s, _src))

        def _pcb(c, t, m, _s=s, _si=si):
            bar.progress((_si + c / max(t, 1)) / len(SENSORS), text=f"{_s.capitalize()}: {m} {c}/{t}")

        res, err = run_detection_and_tracking(fdir, outdir, PARAMS, _pcb)
        if err:
            st.error(f"{s.capitalize()}: {err}")
            continue
        det_frames, paths = res["det_frames"], res["pcd_files"]
        rep = evaluate(det_frames, paths, gtdir, match_dist=match_dist, classes=classes, roi_bounds=roi_bounds)
        rbd = recall_by_distance(det_frames, paths, gtdir, bins=DIST_BINS, match_dist=match_dist,
                                 classes=classes, roi_bounds=roi_bounds)
        results[s] = {"summary": rep["summary"], "rbd": rbd,
                      "gt_dir": os.path.basename(gtdir.rstrip("/"))}
    bar.empty()
    st.session_state.ab_results = {"results": results, "src": _src_label, "match_dist": match_dist,
                                   "vehicles_only": vehicles_only, "roi_on": roi_on}

# --- display ---
state = st.session_state.get("ab_results")
if state and all(s in state["results"] for s in SENSORS):
    res = state["results"]
    ss, rs = res["south"]["summary"], res["registered"]["summary"]
    cfg = (f"{state['src']} · match {state['match_dist']} m · "
           f"{'vehicles only' if state['vehicles_only'] else 'all classes'} · "
           f"{'ROI' if state['roi_on'] else 'no ROI'}")
    st.divider()
    st.subheader("Overall metrics")
    st.caption(f"South GT: `{res['south']['gt_dir']}` · Registered GT: `{res['registered']['gt_dir']}`  ·  {cfg}")

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
else:
    st.info("Pick the input cloud + scoring options, then **Run A/B comparison**. "
            "Both South and Registered need their filtered clouds built first (Background Filtering).")
