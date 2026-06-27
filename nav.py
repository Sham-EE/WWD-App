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
        ("pages/1_Dataset_Prep.py", "🧰", "Dataset Prep",  "Register, draw road geometry, crop & make scorable GT — recreate derived data.",
            ["🧭 Registration", "🗺️ Geometry Editor", "✂️ Crop to road", "🏷️ Scorable GT"]),
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


# Compact, flush styling for the sidebar nav — section headers (tertiary
# buttons) and page links share the same left edge (no indent) with tight gaps.
_SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{gap:.12rem}
section[data-testid="stSidebar"] [data-testid="stButton"] button{
  justify-content:flex-start;text-align:left;font-weight:600;
  padding:.28rem .5rem;border:none}
section[data-testid="stSidebar"] [data-testid="stButton"] button:hover{
  background:rgba(255,255,255,.05)}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a{padding:.22rem .5rem}
section[data-testid="stSidebar"] [data-testid="stPageLink"]{margin:0}
</style>
"""


def render_sidebar():
    """Draw the custom collapsible sidebar. Call once near the top of every page
    (after `st.set_page_config`).

    Sections are collapsible and their open/closed state persists across page
    navigation (kept in `st.session_state["_nav_open"]`). `st.expander` can't
    report its own state, so each section header is a tertiary toggle button that
    drives the persisted set; page links render only while a section is open.
    Whatever you leave open stays open after clicking a tool. New sessions start
    fully collapsed."""
    ss = st.session_state
    if "_nav_open" not in ss:
        ss["_nav_open"] = set()
    open_sections = ss["_nav_open"]
    with st.sidebar:
        st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
        st.page_link("Home.py", label="🏠  Home")
        for i, (title, tools) in enumerate(SECTIONS):
            is_open = title in open_sections
            if st.button(f"{'▾' if is_open else '▸'}  {title}", key=f"nav_sec_{i}",
                         use_container_width=True, type="tertiary"):
                open_sections ^= {title}
                st.rerun()
            if is_open:
                for path, icon, name, _desc, _subs in tools:
                    st.page_link(path, label=f"{icon}  {name}")


# state -> (bg, border, text, trailing glyph) for the compact stepper pills
_STEP_STYLE = {
    "done":     ("#101a13", "#2c5036", "#86d6a0", "✓"),
    "next":     ("#13233a", "#3a6abf", "#bcd8ff", "●"),
    "todo":     ("#14181f", "#2a3340", "#8b97a7", ""),
    "optional": ("#1a1710", "#4a3c22", "#d2bd86", "○"),
}


def render_stepper(steps):
    """Render a compact, single-line stepper: small pills joined by arrows.

    `steps` is a list of (icon, name, state) where state is one of
    done / next / todo / optional. Wraps gracefully on narrow screens."""
    parts = []
    for j, (icon, name, state) in enumerate(steps):
        bg, border, txt, glyph = _STEP_STYLE.get(state, _STEP_STYLE["todo"])
        weight = "600" if state in ("next", "done") else "500"
        tail = f"&nbsp;<span style='opacity:.85'>{glyph}</span>" if glyph else ""
        parts.append(
            f"<span style='display:inline-flex;align-items:center;background:{bg};"
            f"border:1px solid {border};border-radius:999px;padding:3px 12px;"
            f"font-size:.8rem;font-weight:{weight};color:{txt};white-space:nowrap'>"
            f"{icon}&nbsp;{name}{tail}</span>"
        )
        if j < len(steps) - 1:
            parts.append("<span style='color:#3a4452;font-size:1rem;margin:0 1px'>→</span>")
    st.markdown(
        "<div style='display:flex;align-items:center;gap:7px;flex-wrap:wrap;padding:2px 0 6px'>"
        + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


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
