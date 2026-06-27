"""Shared navigation: a single source of truth for the app's tools, grouped into
collapsible sidebar sections. Used by both the custom sidebar (every page calls
`render_sidebar()`) and the Home landing cards.

The default Streamlit page nav is hidden via `.streamlit/config.toml`
(`showSidebarNavigation = false`) so this collapsible sidebar replaces it.
"""
import streamlit as st

# (section title, [ (page_path, icon, name, one-line description), ... ])
SECTIONS = [
    ("📂 Data & setup", [
        ("pages/0_Datasets.py",     "🗂️", "Datasets",      "Choose the active dataset or add your own."),
        ("pages/1_Dataset_Prep.py", "🧰", "Dataset Prep",  "Crop, scorable GT, and the geometry editor — recreate derived data."),
        ("pages/7_Visualizer.py",   "🎬", "Visualizer",    "Cameras + 3D LiDAR labels, overlays, and the real intersection."),
    ]),
    ("⚙️ Detection pipeline", [
        ("pages/2_Background_Filtering.py",          "🔬", "Background Filtering", "Build a background model; keep moving foreground points."),
        ("pages/3_Object_Detection_and_Tracking.py", "📦", "Detection & Tracking", "Cluster, Kalman-track, and flag wrong-way vehicles."),
        ("pages/4_Evaluation.py",                    "📊", "Evaluation",           "Score vs ground truth (P/R/F1, MOTA) + visual compare."),
    ]),
    ("🚨 Wrong-way driving", [
        ("pages/5_Lane_Editor.py",   "🛣️", "Lane Editor",   "Build/adjust the lane directions used for WWD."),
        ("pages/6_WWD_Simulator.py", "🚨", "WWD Simulator", "Spawn a synthetic wrong-way driver; fire the V2X alert."),
    ]),
]


def render_sidebar():
    """Draw the custom collapsible sidebar. Call once near the top of every page
    (after `st.set_page_config`)."""
    with st.sidebar:
        st.page_link("Home.py", label="🏠  Home")
        for title, tools in SECTIONS:
            with st.expander(title, expanded=True):
                for path, icon, name, _desc in tools:
                    st.page_link(path, label=f"{icon}  {name}")
