import streamlit as st
import dataset_manager as dm
import nav

st.set_page_config(page_title="LiDAR WWD Toolkit", page_icon="assets/favicon.png", layout="wide")
nav.render_sidebar()

# ---------------- Header ----------------
st.markdown(
    """
    <div style="padding:6px 0 2px 0">
      <h1 style="margin-bottom:0">🚗 LiDAR Wrong-Way Driving Toolkit</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Active dataset banner ----------------
_active = dm.get_active()
_chips = nav.status_pills_html(_active.status())
ab1, ab2 = st.columns([4, 1])
with ab1:
    st.markdown(
        f"""<div style="background:#14181f;border:1px solid #2a3340;border-radius:12px;padding:14px 18px">
              <div style="color:#8b97a7;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px">Active dataset</div>
              <div style="font-size:1.15rem;font-weight:600;line-height:1.3">🗂️ {_active.name}</div>
              <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px">{_chips}</div>
            </div>""",
        unsafe_allow_html=True,
    )
with ab2:
    st.write("")
    if st.button("Manage datasets →", use_container_width=True):
        st.switch_page("pages/0_Datasets.py")

st.write("")

# ---------------- Dynamic pipeline (compact stepper) ----------------
# A single-line stepper (pills joined by arrows); the first not-done tool is
# "Next". In-page tab order lives on each page itself.
_states = nav.tool_states(_active)
_next_page = next((p for p in nav.PIPELINE if _states[p] == "todo"), None)

st.markdown("#### 🧭 Pipeline")

# The reset button is styled to match the stepper's rounded pills (same radius
# language), but smaller and in a neutral dashed-border tone.
st.markdown(
    """<style>
    .st-key-reset_pill_btn button {
        border-radius: 999px; padding: 1px 10px; font-size: .72rem; font-weight: 500;
        background: #14181f; border: 1px dashed #3a4452; color: #9aa6b2; min-height: 0;
    }
    </style>""",
    unsafe_allow_html=True,
)
pcol, rcol = st.columns([5.5, 1.3])
with pcol:
    nav.render_stepper([
        (nav.TOOLS[p][1], nav.TOOLS[p][2], "next" if p == _next_page else _states[p])
        for p in nav.PIPELINE
    ])
with rcol:
    if st.button("🧨 Reset pipeline", key="reset_pill_btn", use_container_width=True,
                 help="Delete everything the pipeline has generated for this dataset "
                      "(data/derived/ + outputs/), so you can re-run it end-to-end from "
                      "just the raw download. Raw data, the HD map, and your calibration "
                      "(config/) are never touched."):
        st.session_state["confirm_reset_pipeline"] = True

if st.session_state.get("confirm_reset_pipeline"):
    with st.container(border=True):
        st.warning(
            f"This permanently deletes **all generated data** for **{_active.name}**: cropped "
            "clouds, the registered cloud, scorable GT, background models, filtered clouds, "
            "detection tracks, evaluation reports, and tuning run history "
            "(`data/derived/` + `outputs/`).\n\n"
            "**Not touched:** the raw download (`data/raw/`), the HD map (`map/`), and your "
            "calibration (`config/` site geometry, lanes, georeference).",
            icon="🧨")
        keep_videos = st.checkbox(
            "Keep generated videos (road/3D videos, slow to regenerate)",
            value=True, key="reset_keep_videos")
        rc1, rc2 = st.columns(2)
        if rc1.button("Yes, delete it all", type="primary", use_container_width=True):
            info = dm.reset_pipeline(_active, keep_videos=keep_videos)
            # Drop in-memory results that now point at deleted files.
            for _k in ("bg_model", "bg_model_path", "detection_results", "ab_results",
                       "road_video", "lidar_video"):
                st.session_state.pop(_k, None)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state["confirm_reset_pipeline"] = False
            _msg = "Reset done — data/derived/ and outputs/ cleared."
            if info["videos_kept"]:
                _msg += f" Kept {info['videos_kept']} video file(s)."
            st.success(_msg)
            st.rerun()
        if rc2.button("Cancel", use_container_width=True):
            st.session_state["confirm_reset_pipeline"] = False
            st.rerun()

st.write("")
st.divider()

# ---------------- Tool groups (one card per section, sub-tabs listed) ----------------
# Scoped CSS so every card's "Open" button sits flush at the bottom of its column,
# regardless of how many lines the description/sub-tab chips wrap to — the columns
# already stretch to equal height (Streamlit's row is a flex container), this just
# turns each column into a flex column and pins the last element (the button) down.
st.markdown(
    """<style>
    div.st-key-tool_cards div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {
        height: 100%;
    }
    div.st-key-tool_cards div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] >
    div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    div.st-key-tool_cards div[data-testid="stColumn"] div[data-testid="stElementContainer"]:has(button) {
        margin-top: auto;
        padding-top: 10px;
    }
    </style>""",
    unsafe_allow_html=True,
)
with st.container(key="tool_cards"):
    for title, tools in nav.SECTIONS:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            cols = st.columns(len(tools))
            for i, (page, icon, name, desc, subs) in enumerate(tools):
                with cols[i]:
                    st.markdown(f"**{icon}&nbsp; {name}**", unsafe_allow_html=True)
                    st.caption(desc)
                    if subs:
                        st.markdown(
                            "".join(
                                f"<span style='display:inline-block;background:#1b212b;border:1px solid #2a3340;"
                                f"border-radius:6px;padding:1px 7px;margin:0 4px 4px 0;font-size:.72rem;"
                                f"color:#9aa6b2'>{s}</span>" for s in subs
                            ),
                            unsafe_allow_html=True,
                        )
                    if st.button("Open", key=f"go_{page}", use_container_width=True):
                        st.switch_page(page)
        st.write("")
