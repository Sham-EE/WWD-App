import os

import streamlit as st

import dataset_manager as dm

st.set_page_config(layout="wide", page_title="Datasets")
st.title("🗂️ Datasets")
st.markdown(
    "Pick which dataset the whole app works on, or add your own. Each dataset keeps its **own** "
    "inputs, lane geometry, background model, outputs and settings in a separate workspace, so "
    "switching never overwrites another dataset's work."
)

active = dm.get_active()
st.success(f"**Active dataset:** {active.name}  ·  `{active.id}`")

st.info("ℹ️ Increment 1: this page manages and selects datasets. Wiring the Background Filtering / "
        "Detection / Evaluation / Lane Editor pages to read & write the *active* dataset's folders "
        "is the next step — for now they still use the default (TUMTraf) locations.", icon="ℹ️")

# ---------------- Select ----------------
st.subheader("Select a dataset")
datasets = dm.all_datasets()
ids = [d.id for d in datasets]


def _label(d):
    s = d.status()
    chips = []
    chips.append("🟢 PCDs" if s["pcd"] else "⚪ PCDs")
    chips.append("🏷️ GT" if s["gt"] else "")
    chips.append("🛣️ lanes" if s["lanes"] else "")
    chips.append("📦 model" if s["model"] else "")
    chips.append("✨ filtered" if s["filtered"] else "")
    tag = " · ".join(c for c in chips if c)
    return f"{d.name}{'  (template)' if d.is_template else ''}  —  {tag}"


choice = st.radio("Datasets", ids, index=ids.index(active.id) if active.id in ids else 0,
                  format_func=lambda i: _label(dm.get_dataset(i)), label_visibility="collapsed")
if choice != active.id:
    if st.button(f"✅ Switch to “{dm.get_dataset(choice).name}”", type="primary"):
        dm.set_active(choice)
        st.rerun()

# ---------------- Active dataset paths ----------------
ds = dm.get_dataset(choice)
with st.expander("📁 Workspace paths for this dataset", expanded=True):
    st.json({
        "input PCD dir": ds.pcd_dir,
        "ground-truth dir": ds.gt_dir,
        "config (lanes / site geometry)": ds.config_dir,
        "background model": ds.model_path,
        "filtered clouds": ds.filtered_dir,
        "detection outputs": ds.detection_dir,
        "settings": ds.settings_path,
    })
    s = ds.status()
    if not s["pcd"]:
        st.warning(f"No .pcd files found in `{ds.pcd_dir}`.")

if not ds.is_template:
    if st.button(f"🗑️ Remove “{ds.name}” from the list (keeps its files)"):
        dm.delete_dataset(ds.id)
        st.rerun()

# ---------------- Add a dataset ----------------
st.divider()
st.subheader("➕ Add your own dataset")
st.caption("Point the app at a folder of `.pcd` frames on disk (not copied). A workspace for its "
           "config/model/outputs is created under `datasets/<id>/`.")
with st.form("new_dataset"):
    name = st.text_input("Dataset name", placeholder="e.g. My Intersection — North Camera")
    pcd_dir = st.text_input("Path to the folder of PCD frames", placeholder="/path/to/pcd_frames")
    gt_dir = st.text_input("Path to ground-truth labels (optional, OpenLABEL .json)", placeholder="")
    submitted = st.form_submit_button("Create dataset", type="primary")
    if submitted:
        if not name.strip():
            st.error("Please enter a name.")
        elif not pcd_dir.strip() or not os.path.isdir(pcd_dir.strip()):
            st.error(f"PCD folder not found: {pcd_dir}")
        else:
            n_pcd = len([f for f in os.listdir(pcd_dir.strip()) if f.endswith(".pcd")])
            if n_pcd == 0:
                st.warning(f"No .pcd files in {pcd_dir} — creating anyway, but check the path.")
            new = dm.create_dataset(name, pcd_dir, gt_dir)
            st.success(f"Created **{new.name}** (`{new.id}`) and set it active. "
                       f"Workspace: `{new.workspace}`")
            st.rerun()
