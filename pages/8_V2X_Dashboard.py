import streamlit as st
import streamlit.components.v1 as components

import nav
import geo_reference as geo
import dataset_manager as dm
from wwd_detection import load_lane_config
from wwd_simulator import (wrong_way_options, make_wrong_way_track,
                           build_v2x_intersection, v2x_dashboard_html)

st.set_page_config(layout="wide", page_title="V2X Dashboard")
nav.render_sidebar()
st.title("📡 V2X Dashboard")

lanes = load_lane_config()
if not lanes:
    st.info("No lanes configured yet — build them in the 🛣️ Lane Editor, then come back.")
    st.page_link("pages/5_Lane_Editor.py", label="→ Lane Editor", icon="🛣️")
    st.stop()

opts = wrong_way_options(lanes)
labels = [o["label"] for o in opts]

st.caption("Your V2X dashboard (`assets/index.html`), with the map re-centred on **our** intersection and "
           "lane geometry injected in place of the built-in Houston one. Pick which lane the wrong-way "
           "driver runs against; the dashboard's own controls drive the rest.")

# Pre-select the simulator's last broadcast lane, if one was sent over.
ext = st.session_state.get("v2x_event") if st.session_state.get("v2x_armed") else None
default_idx = next((i for i, o in enumerate(opts) if ext and o["lane_id"] == ext.get("lane")), 0)

c1, c2 = st.columns([3, 1])
choice = c1.selectbox("Wrong-way scenario", labels, index=default_idx)
opt = opts[labels.index(choice)]
speed = c2.slider("Speed (m/s)", 1.0, 25.0, float(ext["speed"]) if ext else 9.0, 0.5)

to_ll = lambda x, y: geo.sensor_xy_to_latlon(x, y, "south")
track = make_wrong_way_track(opt["lane"], fps=10, speed=speed, start_frac=0.0, lateral_frac=0.5)
inter = build_v2x_intersection(track, to_ll, geo.site_name())
mid = track[len(track) // 2]
mll = to_ll(mid["cx"], mid["cy"]) or geo.site_latlon()
event = {
    "speed": round(speed, 1), "heading": round(geo.heading_to_true_bearing(mid["heading"], "south")),
    "lane": opt["lane_id"], "direction": opt["wrong_name"],
    "lat": round(float(mll[0]), 6), "lon": round(float(mll[1]), 6),
    "lat_exact": True, "site": geo.site_name(), "intersection": inter,
}

html = v2x_dashboard_html(event)
if html is None:
    st.error("Dashboard not found — save your single-file app to `assets/index.html`.")
else:
    components.html(html, height=1500, scrolling=True)
