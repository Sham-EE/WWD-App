import streamlit as st
import dataset_manager as dm
import nav

st.set_page_config(page_title="LiDAR WWD Toolkit", page_icon="🚗", layout="wide")
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
_status = _active.status()
def _status_pill(label, ok):
    """A rounded status pill matching the pipeline stepper — green + ✓ when the
    artifact is present, red + ✕ when it's missing."""
    bg, bd, tx, glyph = (("#101a13", "#2c5036", "#86d6a0", "✓") if ok
                         else ("#1a1113", "#512c31", "#d98a90", "✕"))
    return (f"<span style='display:inline-flex;align-items:center;gap:5px;background:{bg};"
            f"border:1px solid {bd};border-radius:999px;padding:3px 12px;font-size:.78rem;"
            f"font-weight:600;color:{tx};white-space:nowrap'>{label}"
            f"<span style='opacity:.85'>{glyph}</span></span>")


_chips = "".join(_status_pill(lbl, ok) for lbl, ok in [
    ("PCDs", _status["pcd"]),
    ("GT", _status["gt"]),
    ("lanes", _status["lanes"]),
    ("model", _status["model"]),
    ("filtered", _status["filtered"]),
])
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
nav.render_stepper([
    (nav.TOOLS[p][1], nav.TOOLS[p][2], "next" if p == _next_page else _states[p])
    for p in nav.PIPELINE
])

st.write("")
st.divider()

# ---------------- Tool groups (one card per section, sub-tabs listed) ----------------
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
