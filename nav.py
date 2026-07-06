"""Shared navigation: a single source of truth for the app's tools, grouped into
collapsible sidebar sections. Used by both the custom sidebar (every page calls
`render_sidebar()`) and the Home landing cards.

The default Streamlit page nav is hidden via `.streamlit/config.toml`
(`showSidebarNavigation = false`) so this collapsible sidebar replaces it.
"""
import inspect
import os
import streamlit as st

# (section title, [ (page_path, icon, name, one-line description, [sub-tabs]), ... ])
# Sub-tabs mirror the st.tabs INSIDE each page (Streamlit can't deep-link to a
# tab, so they're surfaced as labels in the Home pipeline + cards for orientation).
SECTIONS = [
    ("📂 Data & setup", [
        ("pages/0_Datasets.py",     "🗂️", "Datasets",      "Choose the active dataset or add your own.", []),
        ("pages/1_Dataset_Prep.py", "🧰", "Dataset Prep",  "Register, draw road geometry, crop & make scorable GT.",
            ["🧭 Registration", "🗺️ Geometry Editor", "✂️ Crop to road", "🏷️ Scorable GT"]),
        ("pages/7_Visualizer.py",   "🎬", "Visualizer",    "Cameras + 3D LiDAR labels, synced video, and the real intersection.",
            ["🎥 Road Viewer", "🧊 LiDAR 3D", "🎬 Videos", "🛰️ Real intersection"]),
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
        ("pages/8_V2X_Dashboard.py", "📡", "V2X Dashboard", "Live wrong-way alert: map, J2735 TIM, pipeline & receivers.", []),
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


# page_path -> section title, for "which section is the current page in?"
_PAGE_SECTION = {t[0]: title for title, tools in SECTIONS for t in tools}


def _current_page():
    """Best-effort path of the page that called render_sidebar ("pages/2_*.py" or
    "Home.py"), read from the call stack — Streamlit execs each page with its real
    filename. Used to auto-open that page's sidebar section."""
    try:
        for fr in inspect.stack():
            base = os.path.basename(fr.filename)
            if base == "Home.py":
                return "Home.py"
            if base[:1].isdigit() and base.endswith(".py"):
                return "pages/" + base
    except Exception:
        pass
    return None


def render_sidebar():
    """Draw the custom collapsible sidebar. Call once near the top of every page
    (after `st.set_page_config`).

    Keeps the native `st.expander` look. `st.expander` can't report its own
    open/closed state, so to survive navigation we auto-expand the section that
    owns the current page — i.e. whatever section you're working in stays open
    after you click a tool (instead of everything collapsing). Other sections
    start collapsed; on Home everything is collapsed."""
    cur_section = _PAGE_SECTION.get(_current_page())
    with st.sidebar:
        st.page_link("Home.py", label="🏠  Home")
        for title, tools in SECTIONS:
            with st.expander(title, expanded=(title == cur_section)):
                for path, icon, name, _desc, _subs in tools:
                    st.page_link(path, label=f"{icon}  {name}")


# state -> (bg, border, text, trailing glyph) for the compact stepper pills
_STEP_STYLE = {
    "done":     ("#101a13", "#2c5036", "#86d6a0", "✓"),
    "next":     ("#13233a", "#3a6abf", "#bcd8ff", "●"),
    "todo":     ("#14181f", "#2a3340", "#8b97a7", ""),
    "optional": ("#1a1710", "#4a3c22", "#d2bd86", "○"),
}


# Dataset artifact chips (present/missing), styled like the stepper's done/todo pills.
_STATUS_ITEMS = [("Cropped PCDs", "pcd"), ("Scorable GT", "gt"), ("Lane geometry", "lanes"),
                 ("Background model", "model"), ("Filtered PCDs", "filtered")]


def status_pills_html(status):
    """Row of rounded pills for a dataset's status dict (from Dataset.status()) —
    green + check when the artifact is present, red + cross when it's missing. Shared
    by Home's active-dataset banner and the Datasets page dataset list so both match."""
    parts = []
    for label, key in _STATUS_ITEMS:
        ok = bool(status.get(key))
        bg, bd, tx, glyph = ("#101a13", "#2c5036", "#86d6a0", "✓") if ok \
            else ("#1a1113", "#512c31", "#d98a90", "✕")
        parts.append(
            f"<span style='display:inline-flex;align-items:center;gap:5px;background:{bg};"
            f"border:1px solid {bd};border-radius:999px;padding:3px 12px;font-size:.78rem;"
            f"font-weight:600;color:{tx};white-space:nowrap'>{label}"
            f"<span style='opacity:.85'>{glyph}</span></span>"
        )
    return "".join(parts)


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
        "pages/8_V2X_Dashboard.py":                  "anytime",
    }
