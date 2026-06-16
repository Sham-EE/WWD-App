import streamlit as st
import numpy as np

from wwd_detection import load_lane_config, lanes_calibrated, detect_wrong_way
from wwd_simulator import (wrong_way_options, make_wrong_way_track,
                           build_sim_det_frames, simulator_figure, SIM_TID)

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

# ---------------- Scenario controls ----------------
st.subheader("1. Define the wrong-way driver")
c1, c2 = st.columns([3, 2])
labels = [o["label"] for o in opts]
choice = c1.selectbox("Wrong-way scenario (only illegal directions are offered)", labels,
                      help="Each option drives a vehicle OPPOSITE to a lane's legal heading.")
opt = opts[labels.index(choice)]
speed = c2.slider("Driver speed (m/s)", 1.0, 25.0, 9.0, 0.5)
c3, c4, c5 = st.columns(3)
start_frac = c3.slider("Entry point along lane", 0.0, 0.8, 0.0, 0.05,
                       help="Where along the lane the driver enters (0 = far end).")
lateral_frac = c4.slider("Lane position (across)", 0.0, 1.0, 0.5, 0.05)
fps = c5.number_input("FPS", 1.0, 30.0, 10.0, 1.0, help="Simulation frame rate.")

mix_real = False
if st.session_state.get("detection_results"):
    mix_real = st.checkbox("Mix with real traffic from the last detection run", value=False)

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
ww = detect_wrong_way(det_frames, lanes, {"min_speed": 1.0})
sim_res = ww["tracks"].get(SIM_TID, {})
is_flagged = SIM_TID in ww["wrong_way_tids"]
first_flag = sim_res.get("first_flag_frame")  # det-frame index

# ---------------- Playback + view ----------------
st.subheader("2. Run the simulation")
n_steps = len(sim_track)
st.session_state.setdefault("sim_step", 0)
st.session_state.sim_step = min(st.session_state.sim_step, n_steps - 1)
nav = st.columns([1, 1, 1, 1, 4])
if nav[0].button("⏮ Start", use_container_width=True):
    st.session_state.sim_step = 0; st.rerun()
if nav[1].button("◀ Prev", use_container_width=True):
    st.session_state.sim_step = max(0, st.session_state.sim_step - 1); st.rerun()
if nav[2].button("Next ▶", use_container_width=True):
    st.session_state.sim_step = min(n_steps - 1, st.session_state.sim_step + 1); st.rerun()
if nav[3].button("End ⏭", use_container_width=True):
    st.session_state.sim_step = n_steps - 1; st.rerun()
playing = nav[4].toggle("▶ Play", value=False)
step = st.slider("Step", 0, max(n_steps - 1, 1), st.session_state.sim_step)
st.session_state.sim_step = step

cur_frame_idx = start_frame + step
flagged_now = is_flagged and first_flag is not None and cur_frame_idx >= first_flag

left, right = st.columns([3, 2], gap="medium")
with left:
    base_dets = det_frames[cur_frame_idx] if (base and cur_frame_idx < len(det_frames)) else None
    if base_dets:
        base_dets = [d for d in base_dets if d.get("tid") != SIM_TID]
    fig = simulator_figure(lanes, sim_track, step, flagged_now, base_dets=base_dets,
                           x_range=x_range, y_range=y_range)
    st.plotly_chart(fig, use_container_width=True, key="sim_fig",
                    config={"scrollZoom": True})

with right:
    st.markdown("#### Detector verdict")
    if is_flagged:
        t_s = (first_flag - start_frame) / float(fps) if first_flag is not None else 0.0
        st.error(f"🚨 **WRONG-WAY DRIVING DETECTED**")
        st.write(f"- **Direction:** {opt['wrong_name']}-bound in the **{opt['lane_id']}** lane "
                 f"(legal: {opt['legal_name']})")
        st.write(f"- **Flagged at:** step {first_flag - start_frame} (~{t_s:.1f}s), "
                 f"sustained {sim_res.get('run_len','?')} frames")
        st.write(f"- **Max angle vs flow:** {sim_res.get('max_angle_deg',0):.0f}°  ·  "
                 f"**speed:** {speed:.1f} m/s")
        if flagged_now:
            st.success("Alert is ACTIVE at the current step.")
        else:
            st.info(f"Scrub to step {first_flag - start_frame} to reach the alert.")
    else:
        st.success("No wrong-way flag — the driver did not trigger the detector. "
                   "Increase speed/length or check lane calibration.")

# ---------------- Alert / external-app messaging hook ----------------
st.divider()
st.subheader("3. Alert & messaging")
if flagged_now:
    details = {
        "lane": opt["lane_id"], "direction": opt["wrong_name"],
        "speed_mps": round(speed, 1), "frame": cur_frame_idx,
        "max_angle_deg": round(sim_res.get("max_angle_deg", 0), 0),
    }
    # ===================================================================
    # INTEGRATION POINT for the external alerting app (single HTML file).
    # Paste that app's HTML here and it will render/trigger on detection,
    # e.g. via st.components.v1.html(your_html, height=...). Until then,
    # a placeholder alert banner is shown.
    # ===================================================================
    st.markdown(
        f"""
        <div style="background:#b00020;color:white;padding:18px;border-radius:10px;
                    font-size:1.1rem;border:2px solid #ff5252">
          🚨 <b>WRONG-WAY ALERT</b> — {details['direction']}-bound vehicle in the
          <b>{details['lane']}</b> lane at {details['speed_mps']} m/s.
          <br><span style="opacity:.85;font-size:.95rem">[Your alert app's messaging renders here]</span>
        </div>
        """, unsafe_allow_html=True)
    st.caption("⬆ This banner is the hook for your external messaging app — paste its HTML and "
               "I'll wire it to fire here on detection.")
else:
    st.caption("No active alert at this step. Play through to the moment of detection.")

# auto-advance
if playing and step < n_steps - 1:
    import time
    time.sleep(max(0.03, 1.0 / float(fps)))
    st.session_state.sim_step = step + 1
    st.rerun()
