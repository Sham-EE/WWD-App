"""Shared navigation: a single source of truth for the app's tools, grouped into
collapsible sidebar sections. Used by both the custom sidebar (every page calls
`render_sidebar()`) and the Home landing cards.

The default Streamlit page nav is hidden via `.streamlit/config.toml`
(`showSidebarNavigation = false`) so this collapsible sidebar replaces it.
"""
import os
import streamlit as st

# (section title, [ (page_path, icon, name, one-line description, [sub-tabs]), ... ])
# Sub-tabs mirror the st.tabs INSIDE each page (Streamlit can't deep-link to a
# tab, so they're surfaced as labels in the Home pipeline + cards for orientation).
SECTIONS = [
    ("📂 Data & setup", [
        ("pages/0_Datasets.py",     "🗂️", "Datasets",      "Choose the active dataset or add your own.", []),
        ("pages/1_Dataset_Prep.py", "🧰", "Dataset Prep",  "Crop, scorable GT, geometry & registration — recreate derived data.",
            ["✂️ Crop to road", "🏷️ Scorable GT", "🗺️ Geometry Editor", "🧭 Registration"]),
        ("pages/7_Visualizer.py",   "🎬", "Visualizer",    "Cameras + 3D LiDAR labels, overlays, and the real intersection.",
            ["🎥 Road Viewer", "🧊 LiDAR 3D", "🛰️ Real intersection"]),
    ]),
    ("⚙️ Detection pipeline", [
        ("pages/2_Background_Filtering.py",          "🔬", "Background Filtering", "Build a background model; keep moving foreground points.", []),
        ("pages/3_Object_Detection_and_Tracking.py", "📦", "Detection & Tracking", "Cluster, Kalman-track, and flag wrong-way vehicles.", []),
        ("pages/4_Evaluation.py",                    "📊", "Evaluation",           "Score vs ground truth (P/R/F1, MOTA) + visual compare.",
            ["📐 Single run", "📊 A/B compare"]),
    ]),
    ("🚨 Wrong-way driving", [
        ("pages/5_Lane_Editor.py",   "🛣️", "Lane Editor",   "Build/adjust the lane directions used for WWD.", []),
        ("pages/6_WWD_Simulator.py", "🚨", "WWD Simulator", "Spawn a synthetic wrong-way driver; fire the V2X alert.", []),
    ]),
]

# page_path -> tool tuple, and the linear workflow order for the Home stepper
# (Visualizer is a view-only tool, used anytime → kept in the cards, off the stepper).
TOOLS = {t[0]: t for _title, tools in SECTIONS for t in tools}
PIPELINE = [
    "pages/0_Datasets.py",
    "pages/1_Dataset_Prep.py",
    "pages/2_Background_Filtering.py",
    "pages/3_Object_Detection_and_Tracking.py",
    "pages/4_Evaluation.py",
    "pages/5_Lane_Editor.py",
    "pages/6_WWD_Simulator.py",
]


def render_sidebar():
    """Draw the custom collapsible sidebar. Call once near the top of every page
    (after `st.set_page_config`)."""
    with st.sidebar:
        st.page_link("Home.py", label="🏠  Home")
        for title, tools in SECTIONS:
            with st.expander(title, expanded=True):
                for path, icon, name, _desc, _subs in tools:
                    st.page_link(path, label=f"{icon}  {name}")


def _has_files(d, ext=None):
    """True if `d` (recursively) holds at least one file (optionally ending in `ext`)."""
    for _root, _dirs, files in os.walk(d):
        for f in files:
            if ext is None or f.endswith(ext):
                return True
    return False


def tool_states(ds):
    """Per-tool completion for the active dataset, keyed by page path:
    'done'    — its output/artifact exists,
    'todo'    — actionable but not yet done,
    'anytime' — view-only (no completion; never counted in a section fraction).

    Used by Home to show each section step's progress (e.g. 2/3) and to pick the
    single global "next" step (first actionable 'todo' in pipeline order)."""
    s = ds.status()
    det = _has_files(os.path.join(ds.outputs_dir, "detection"))
    ev = _has_files(os.path.join(ds.outputs_dir, "run_history"), ".jsonl")
    return {
        "pages/0_Datasets.py":                       "done" if s["pcd"] else "todo",
        "pages/1_Dataset_Prep.py":                   "done" if s["gt"] else "todo",
        "pages/7_Visualizer.py":                     "anytime",
        "pages/2_Background_Filtering.py":           "done" if s["filtered"] else "todo",
        "pages/3_Object_Detection_and_Tracking.py":  "done" if det else "todo",
        "pages/4_Evaluation.py":                     "done" if ev else "todo",
        "pages/5_Lane_Editor.py":                    "done" if s["lanes"] else "todo",
        "pages/6_WWD_Simulator.py":                  "done" if (s["lanes"] and s["filtered"]) else "todo",
    }
