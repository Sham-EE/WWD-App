import streamlit as st
import streamlit.components.v1 as components

import nav
import v2x_dashboard as v2x

st.set_page_config(layout="wide", page_title="V2X Dashboard")
nav.render_sidebar()
st.title("📡 V2X Dashboard")

event = st.session_state.get("v2x_event")
armed = st.session_state.get("v2x_armed")

if not (event and armed):
    st.info(
        "No active broadcast. Open the **🚨 WWD Simulator**, run a wrong-way scenario until the "
        "detector **confirms** it, then click **📡 Broadcast** — this dashboard will light up with the "
        "live alert: an accurate map of the wrong-way driver on the real intersection, the J2735 TIM "
        "message, the detection→broadcast pipeline, and the receivers notified."
    )
    st.page_link("pages/6_WWD_Simulator.py", label="→ Go to the WWD Simulator", icon="🚨")
    st.stop()

st.caption(f"Live broadcast for **{event.get('site','the intersection')}** — the map **plays on its own**: "
           "the wrong-way driver drives its real path while nearby C-V2X vehicles light up with the "
           "WRONG-WAY warning as it enters their range. Generated in-app (no external dashboard asset).")
components.html(v2x.build_dashboard_html(event), height=1000, scrolling=True)

c1, c2 = st.columns([1, 4])
if c1.button("✖ Clear broadcast", use_container_width=True):
    st.session_state.v2x_armed = False
    st.session_state.v2x_event = None
    st.rerun()
c2.page_link("pages/6_WWD_Simulator.py", label="Back to the WWD Simulator", icon="🚨")
