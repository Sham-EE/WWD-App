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

# ---------------- Dynamic pipeline ----------------
# Each step maps to a status flag for the active dataset; the terminal step is
# "ready" once everything before it is done. The first incomplete step is "Next".
PIPELINE = [
    ("Load dataset",      "🗂️", "pages/0_Datasets.py",              _status["pcd"],      "Point clouds loaded."),
    ("Background model",  "🔬", "pages/2_Background_Filtering.py",  _status["model"],    "Background model built."),
    ("Filter clouds",     "✨", "pages/2_Background_Filtering.py",  _status["filtered"], "Foreground clouds saved."),
    ("Define lanes",      "🛣️", "pages/5_Lane_Editor.py",          _status["lanes"],    "Lane directions set."),
    ("Run WWD",           "🚨", "pages/6_WWD_Simulator.py",         None,                "Simulate & broadcast V2X."),
]

# Resolve the terminal step (done once every earlier step is done) and find "Next".
_steps, _prev_all_done = [], True
for name, icon, page, done, hint in PIPELINE:
    d = _prev_all_done if done is None else bool(done)
    _steps.append({"name": name, "icon": icon, "page": page, "done": d, "hint": hint})
    _prev_all_done = _prev_all_done and d
_next_idx = next((i for i, s in enumerate(_steps) if not s["done"]), None)

st.markdown("#### 🧭 Pipeline")
st.caption("The recommended order — status updates as the active dataset progresses.")
_cols = st.columns(len(_steps))
for i, s in enumerate(_steps):
    is_next = (i == _next_idx)
    if s["done"]:
        badge, color, bg, border = "✅ Done", "#4ade80", "#101a13", "#234a2c"
    elif is_next:
        badge, color, bg, border = "🔵 Next", "#60a5fa", "#0f1722", "#2b4a78"
    else:
        badge, color, bg, border = "⬜ To do", "#6b7480", "#14181f", "#2a3340"
    with _cols[i]:
        st.markdown(
            f"""<div style="background:{bg};border:1px solid {border};border-radius:12px;
                        padding:10px 8px;text-align:center;min-height:104px">
                  <div style="font-size:.72rem;color:#6b7480">STEP {i+1}</div>
                  <div style="font-size:1.5rem;line-height:1.7rem">{s['icon']}</div>
                  <div style="font-weight:600;font-size:.9rem;margin-top:2px">{s['name']}</div>
                  <div style="color:{color};font-size:.78rem;margin-top:4px">{badge}</div>
                </div>""",
            unsafe_allow_html=True,
        )

if _next_idx is None:
    st.success("🎉 All set — every pipeline stage is complete. Jump into the **WWD Simulator** or **Evaluation**.")
else:
    _ns = _steps[_next_idx]
    nc1, nc2 = st.columns([4, 1])
    nc1.info(f"👉 **Next step — {_ns['name']}:** {_ns['hint']}")
    with nc2:
        st.write("")
        if st.button(f"Go to {_ns['name']} →", use_container_width=True, type="primary"):
            st.switch_page(_ns["page"])

st.write("")
st.divider()

# ---------------- Tool groups (one card per section) ----------------
for title, tools in nav.SECTIONS:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        cols = st.columns(len(tools))
        for i, (page, icon, name, desc) in enumerate(tools):
            with cols[i]:
                st.markdown(f"**{icon}&nbsp; {name}**", unsafe_allow_html=True)
                st.caption(desc)
                if st.button("Open", key=f"go_{page}", use_container_width=True):
                    st.switch_page(page)
    st.write("")
