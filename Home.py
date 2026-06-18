import streamlit as st
import dataset_manager as dm

st.set_page_config(page_title="LiDAR WWD Toolkit", page_icon="🚗", layout="wide")

# ---------------- Header ----------------
st.markdown(
    """
    <div style="padding:6px 0 2px 0">
      <h1 style="margin-bottom:0">🚗 LiDAR Wrong-Way Driving Toolkit</h1>
      <p style="color:#8b97a7;margin-top:4px;font-size:1.02rem">
        Roadside-LiDAR pipeline — background filtering · detection &amp; tracking ·
        wrong-way detection · evaluation · V2X alerting.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.success("Select a tool above.")

# ---------------- Active dataset banner ----------------
_active = dm.get_active()
_status = _active.status()
_chips = " ".join(c for c in [
    "🟢 PCDs" if _status["pcd"] else "⚪ no PCDs",
    "🏷️ GT" if _status["gt"] else "",
    "🛣️ lanes" if _status["lanes"] else "",
    "📦 model" if _status["model"] else "",
    "✨ filtered" if _status["filtered"] else "",
] if c)
ab1, ab2 = st.columns([4, 1])
with ab1:
    st.markdown(
        f"""<div style="background:#14181f;border:1px solid #2a3340;border-radius:12px;padding:12px 16px">
              <span style="color:#8b97a7;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em">Active dataset</span><br>
              <span style="font-size:1.15rem;font-weight:600">🗂️ {_active.name}</span>
              <span style="color:#8b97a7"> · <code>{_active.id}</code></span><br>
              <span style="color:#8b97a7;font-size:.9rem">{_chips}</span>
            </div>""",
        unsafe_allow_html=True,
    )
with ab2:
    st.write("")
    if st.button("Manage datasets →", use_container_width=True):
        st.switch_page("pages/0_Datasets.py")

st.write("")

# ---------------- Tool groups ----------------
GROUPS = [
    ("📂 Data & setup", [
        ("🗂️", "Datasets", "Choose the active dataset or add your own.", "pages/0_Datasets.py"),
        ("🧰", "Dataset Prep", "Recreate derived data (crop, scorable GT) from the raw download.", "pages/7_Dataset_Prep.py"),
        ("🎬", "Visualizer", "Cameras side by side + 3D LiDAR labels (BEV/side); overlays, trails, video.", "pages/6_Visualizer.py"),
    ]),
    ("⚙️ Detection pipeline", [
        ("🔬", "Background Filtering", "Build a background model; keep moving foreground points.", "pages/1_Background_Filtering.py"),
        ("📦", "Detection & Tracking", "Cluster, Kalman-track, and flag wrong-way vehicles.", "pages/2_Object_Detection_and_Tracking.py"),
        ("📊", "Evaluation", "Score vs ground truth (P/R/F1, MOTA) + visual compare.", "pages/3_Evaluation.py"),
    ]),
    ("🚨 Wrong-way driving", [
        ("🛣️", "Lane Editor", "Build/adjust the lane directions used for WWD.", "pages/4_Lane_Editor.py"),
        ("🚨", "WWD Simulator", "Spawn a synthetic wrong-way driver; fire the V2X alert.", "pages/5_WWD_Simulator.py"),
    ]),
]

for title, tools in GROUPS:
    st.markdown(f"#### {title}")
    cols = st.columns(3)
    for i, (icon, name, desc, page) in enumerate(tools):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{icon}&nbsp; {name}**", unsafe_allow_html=True)
                st.caption(desc)
                if st.button("Open", key=f"go_{page}", use_container_width=True):
                    st.switch_page(page)
    st.write("")
