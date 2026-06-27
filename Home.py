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
      <p style="color:#8b97a7;margin-top:4px;font-size:1.02rem">
        Roadside-LiDAR pipeline — background filtering · detection &amp; tracking ·
        wrong-way detection · evaluation · V2X alerting.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

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

# ---------------- Dynamic pipeline (horizontal stepper) ----------------
# One even-height card per workflow tool (in order) with its status badge. The
# first not-done tool is "Next". (In-page tab order lives on each page itself.)
_states = nav.tool_states(_active)
_next_page = next((p for p in nav.PIPELINE if _states[p] == "todo"), None)

st.markdown("#### 🧭 Pipeline")
st.caption("The recommended order — status updates as the active dataset progresses.")
_cols = st.columns(len(nav.PIPELINE))
for i, page in enumerate(nav.PIPELINE):
    _p, icon, name, _desc, _subs = nav.TOOLS[page]
    state = "next" if page == _next_page else _states[page]
    if state == "done":
        badge, color, bg, border = "✅ Done", "#4ade80", "#101a13", "#234a2c"
    elif state == "next":
        badge, color, bg, border = "🔵 Next", "#60a5fa", "#0f1722", "#2b4a78"
    else:
        badge, color, bg, border = "⬜ To do", "#6b7480", "#14181f", "#2a3340"
    with _cols[i]:
        st.markdown(
            f"""<div style="background:{bg};border:1px solid {border};border-radius:12px;
                        padding:10px 6px;text-align:center;min-height:118px">
                  <div style="font-size:.7rem;color:#6b7480">STEP {i+1}</div>
                  <div style="font-size:1.35rem;line-height:1.55rem">{icon}</div>
                  <div style="font-weight:600;font-size:.84rem;margin-top:2px">{name}</div>
                  <div style="color:{color};font-size:.74rem;margin-top:3px">{badge}</div>
                </div>""",
            unsafe_allow_html=True,
        )

if _next_page is None:
    st.success("🎉 All set — every pipeline stage is complete. Jump into the **WWD Simulator** or **Evaluation**.")
else:
    _ni, _nicon, _nname, _ndesc, _ns = nav.TOOLS[_next_page]
    nc1, nc2 = st.columns([4, 1])
    nc1.info(f"👉 **Next step — {_nicon} {_nname}:** {_ndesc}")
    with nc2:
        st.write("")
        if st.button(f"Go to {_nname} →", use_container_width=True, type="primary"):
            st.switch_page(_next_page)

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
