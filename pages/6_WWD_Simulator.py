import streamlit as st
import numpy as np

import streamlit.components.v1 as components

from wwd_detection import load_lane_config, lanes_calibrated, detect_wrong_way
from wwd_simulator import (wrong_way_options, make_wrong_way_track,
                           build_sim_det_frames, simulator_figure, SIM_TID,
                           v2x_dashboard_html, math_heading_to_compass)
import viewer_ui as vu

st.set_page_config(layout="wide", page_title="WWD Simulator")
st.title("🚨 Wrong-Way Driver Simulator")
st.markdown(
    "Real wrong-way events are rare and this dataset is all legal traffic, so this page "
    "**spawns a synthetic wrong-way driver** and runs it through the *real* WWD detector. "
    "Pick a wrong-way scenario, watch the driver cross the lane, and the alert fires the "
    "moment the algorithm flags it."
)

lanes = load_lane_config()
if not lanes:
    st.error("No lane configuration found (config/lanes.geojson). Calibrate lanes in the Lane Editor first.")
    st.stop()
if not lanes_calibrated(lanes):
    st.warning("Lane geometry is not fully calibrated — the simulated directions may be off. "
               "Calibrate in the Lane Editor for trustworthy results.")

opts = wrong_way_options(lanes)
# lane extent for a stable view
import numpy as _np
allx, ally = [], []
for ln in lanes:
    x0, y0, x1, y1 = ln["polygon"].bounds
    allx += [x0, x1]; ally += [y0, y1]
m = 6.0
x_range = (min(allx) - m, max(allx) + m)
y_range = (min(ally) - m, max(ally) + m)

# ---------------- Setup panel: one collapsible card, tabbed inside ----------------
# Display-toggle keys are seeded first so the figure + the bulk All/None buttons share them.
vu.ensure_toggle_defaults({
    "sim_show_lanes": True, "sim_show_legal_arrows": True, "sim_show_path": True,
    "sim_show_heading": True, "sim_show_real": True, "sim_show_grid": False,
    "sim_show_legend": True,
})
have_real = bool(st.session_state.get("detection_results"))

with st.expander("⚙️ Simulation setup", expanded=True):
    t_scn, t_det, t_disp = st.tabs(["🚗 Scenario", "🛡️ Detector", "🎛️ Display"])

    with t_scn:
        labels = [o["label"] for o in opts]
        c1, c2 = st.columns([3, 2])
        choice = c1.selectbox("Wrong-way scenario (only illegal directions are offered)", labels,
                              help="Each option drives a vehicle OPPOSITE to a lane's legal heading.")
        opt = opts[labels.index(choice)]
        speed = c2.slider("Driver speed (m/s)", 1.0, 25.0, 9.0, 0.5)
        c3, c4, c5 = st.columns(3)
        start_frac = c3.slider("Entry point along lane", 0.0, 0.8, 0.0, 0.05,
                               help="Where along the lane the driver enters (0 = far end).")
        lateral_frac = c4.slider("Lane position (across)", 0.0, 1.0, 0.5, 0.05)
        fps = c5.number_input("FPS", 1.0, 30.0, 10.0, 1.0, help="Simulation frame rate.")

    with t_det:
        d1, d2, d3 = st.columns(3)
        conf_frames = d1.slider("Confirmation frames", 1, 30, 5, 1,
                                help="Consecutive wrong-way frames required before the detector confirms it "
                                     "(WWD min_frames). Higher = more cautious, later detection.")
        min_speed_wwd = d2.slider("Min speed (m/s)", 0.5, 10.0, 1.0, 0.5,
                                  help="Below this the heading is treated as unreliable.")
        angle_thresh = d3.slider("Angle vs. flow (deg)", 90, 180, 120, 5,
                                 help="How far against the lane's legal direction counts as wrong-way.")

    with t_disp:
        st.caption("Show / hide overlays on the simulation view.")
        _disp_keys = ["sim_show_lanes", "sim_show_legal_arrows", "sim_show_path",
                      "sim_show_heading", "sim_show_grid", "sim_show_legend"]
        if have_real:
            _disp_keys.insert(4, "sim_show_real")
        vu.bulk_toggle_buttons(_disp_keys, "sim_disp", rerun_scope="app")
        tc1, tc2 = st.columns(2)
        tc1.toggle("🛣️ Lane boxes", key="sim_show_lanes")
        tc1.toggle("➡️ Legal-direction arrows", key="sim_show_legal_arrows",
                   help="Per-lane arrow showing the legal flow direction.")
        tc1.toggle("〰️ Driver path", key="sim_show_path",
                   help="Planned + travelled path of the simulated driver.")
        tc2.toggle("🧭 Driver heading arrow", key="sim_show_heading")
        tc2.toggle("▦ Grid", key="sim_show_grid")
        tc2.toggle("🏷️ Legend", key="sim_show_legend")
        if have_real:
            st.toggle("🚗 Overlay real traffic", key="sim_show_real",
                      help="Animate the actual detected vehicles (from the last detection run) "
                           "alongside the simulated driver, colored by direction.")
        else:
            st.caption("Run **Object Detection and Tracking** first to overlay real moving traffic.")

mix_real = have_real and st.session_state.sim_show_real

# ---------------- Build + detect ----------------
sim_track = make_wrong_way_track(opt["lane"], fps=fps, speed=speed,
                                 start_frac=start_frac, lateral_frac=lateral_frac)
if not sim_track:
    st.error("Could not generate a path for this lane — try a different entry point.")
    st.stop()

base = None
start_frame = 0
if mix_real:
    base = st.session_state.detection_results.get("det_frames")
    start_frame = st.slider("Start frame (in the real sequence)", 0, max(len(base) - len(sim_track), 0), 0)

det_frames = build_sim_det_frames(sim_track, start_frame=start_frame, base_det_frames=base,
                                  total_frames=len(sim_track) + 5)
ww = detect_wrong_way(det_frames, lanes, {"min_speed": min_speed_wwd, "min_frames": int(conf_frames),
                                          "angle_thresh_deg": float(angle_thresh)})
sim_res = ww["tracks"].get(SIM_TID, {})
is_flagged = SIM_TID in ww["wrong_way_tids"]
first_flag = sim_res.get("first_flag_frame")  # det-frame index

# ---------------- Playback + view ----------------
st.subheader("▶️ Run the simulation")
n_steps = len(sim_track)
step, playing, play_delay = vu.nav_row("sim_step", n_steps, "sim", label="🎞️ Step")

cur_frame_idx = start_frame + step
# The detector confirms only AFTER `conf_frames` consecutive wrong-way frames, so
# the alert/"flagged at" moment is the run start + (confirmation frames - 1).
confirm_frame = (first_flag + int(conf_frames) - 1) if (is_flagged and first_flag is not None) else None
flagged_now = confirm_frame is not None and cur_frame_idx >= confirm_frame

left, right = st.columns([3, 2], gap="medium")
with left:
    base_dets = det_frames[cur_frame_idx] if (base and cur_frame_idx < len(det_frames)) else None
    if base_dets:
        base_dets = [d for d in base_dets if d.get("tid") != SIM_TID]
    fig = simulator_figure(lanes, sim_track, step, flagged_now, base_dets=base_dets,
                           x_range=x_range, y_range=y_range,
                           show_lanes=st.session_state.sim_show_lanes,
                           show_legal_arrows=st.session_state.sim_show_legal_arrows,
                           show_path=st.session_state.sim_show_path,
                           show_heading=st.session_state.sim_show_heading,
                           show_real=st.session_state.sim_show_real,
                           show_grid=st.session_state.sim_show_grid,
                           show_legend=st.session_state.sim_show_legend)
    st.plotly_chart(fig, use_container_width=True, key="sim_fig",
                    config={"scrollZoom": True})

with right:
    st.markdown("#### Detector verdict")
    if is_flagged:
        start_step = first_flag - start_frame
        confirm_step = confirm_frame - start_frame
        t_s = confirm_step / float(fps)
        st.error("🚨 **WRONG-WAY DRIVING DETECTED**")
        st.write(f"- **Direction:** {opt['wrong_name']}-bound in the **{opt['lane_id']}** lane "
                 f"(legal: {opt['legal_name']})")
        st.write(f"- **Wrong-way motion starts:** step {start_step}")
        st.write(f"- **Confirmed (alert fires):** step {confirm_step} (~{t_s:.1f}s) — "
                 f"after {int(conf_frames)} confirmation frames")
        st.write(f"- **Max angle vs flow:** {sim_res.get('max_angle_deg',0):.0f}°  ·  "
                 f"**speed:** {speed:.1f} m/s")
        if flagged_now:
            st.success("Alert is ACTIVE at the current step.")
        else:
            st.info(f"Scrub to step {confirm_step} to reach the confirmation/alert.")
    else:
        st.success("No wrong-way flag — the driver did not sustain wrong-way motion for "
                   f"{int(conf_frames)} frames. Lower the confirmation frames, increase speed/length, "
                   "or check lane calibration.")

# ---------------- V2X broadcast (external dashboard) ----------------
st.divider()
st.subheader("📡 V2X broadcast — WWD V2X Dashboard")
st.session_state.setdefault("v2x_armed", False)
st.session_state.setdefault("v2x_event", None)

if not is_flagged:
    st.caption("No wrong-way flag yet — once detected, broadcast it to your V2X dashboard here.")
else:
    bc1, bc2 = st.columns([2, 1])
    if bc1.button("📡 Broadcast detection to the V2X Dashboard", type="primary",
                  disabled=not flagged_now,
                  help="Fires your dashboard's full J2735 TIM / C-V2X / nav-push pipeline with the "
                       "detected speed & heading. Play to the detection step to enable."):
        compass = math_heading_to_compass(sim_track[step]["heading"])
        st.session_state.v2x_event = {
            "speed": round(float(speed), 1), "heading": round(float(compass)),
            "lane": opt["lane_id"], "direction": opt["wrong_name"],
        }
        st.session_state.v2x_armed = True
        st.rerun()
    if st.session_state.v2x_armed and bc2.button("✖ Close dashboard", use_container_width=True):
        st.session_state.v2x_armed = False
        st.session_state.v2x_event = None
        st.rerun()
    if not flagged_now and not st.session_state.v2x_armed and confirm_frame is not None:
        st.caption(f"Scrub to step {confirm_frame - start_frame} (the confirmation moment) to enable the broadcast.")

if st.session_state.v2x_armed and st.session_state.v2x_event:
    html = v2x_dashboard_html(st.session_state.v2x_event)
    if html is None:
        st.warning("V2X dashboard not found. Save your single-file app to "
                   "`assets/wwd_v2x_dashboard.html` (see assets/README.md), then broadcast again.")
    else:
        ev = st.session_state.v2x_event
        st.success(f"🚨 Broadcasting: {ev['direction']}-bound in the {ev['lane']} lane · "
                   f"{ev['speed']} m/s · heading {ev['heading']}° — the dashboard fired its alert pipeline below.")
        components.html(html, height=1500, scrolling=True)

# auto-advance (paused while the dashboard is embedded to avoid re-render churn)
if playing and step < n_steps - 1 and not st.session_state.v2x_armed:
    import time
    time.sleep(float(play_delay))
    st.session_state.sim_step = step + 1
    st.rerun()
