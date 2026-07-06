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
nav.render_stepper([
    (nav.TOOLS[p][1], nav.TOOLS[p][2], "next" if p == _next_page else _states[p])
    for p in nav.PIPELINE
])

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
