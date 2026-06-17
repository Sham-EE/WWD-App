import os
import json
import glob

import streamlit as st

import dataset_manager as dm

st.set_page_config(layout="wide", page_title="Datasets")
st.title("🗂️ Datasets")
st.markdown(
    "Pick which dataset the whole app works on, or add your own. Each dataset keeps its **own** "
    "inputs, lane geometry, site geometry, background model, outputs and settings in a separate "
    "workspace — switching never overwrites another dataset's work."
)

active = dm.get_active()
st.success(f"**Active dataset:** {active.name}   ·   `{active.id}`")


def _pcd_count(p):
    try:
        return len(glob.glob(os.path.join(p, "*.pcd")))
    except Exception:
        return 0


def _gt_count(p):
    try:
        return len(glob.glob(os.path.join(p or "", "*.json")))
    except Exception:
        return 0


def _clear_session():
    for k in dm.SESSION_KEYS_TO_CLEAR:
        st.session_state.pop(k, None)


# ---------------- Select ----------------
st.subheader("Select a dataset")
datasets = dm.all_datasets()
ids = [d.id for d in datasets]


def _label(d):
    s = d.status()
    chips = ["🟢 PCDs" if s["pcd"] else "⚪ no PCDs"]
    if s["gt"]:
        chips.append("🏷️ GT")
    if s["lanes"]:
        chips.append("🛣️ lanes")
    if s["model"]:
        chips.append("📦 model")
    if s["filtered"]:
        chips.append("✨ filtered")
    return f"{d.name}{'  ·  template' if d.is_template else ''}  —  {' · '.join(chips)}"


choice = st.radio("Datasets", ids, index=ids.index(active.id) if active.id in ids else 0,
                  format_func=lambda i: _label(dm.get_dataset(i)), label_visibility="collapsed")
ds = dm.get_dataset(choice)

if choice != active.id:
    if st.button(f"✅ Switch to “{ds.name}” (clears in-memory results)", type="primary"):
        dm.set_active(choice)
        _clear_session()
        st.rerun()

# ---------------- Active/selected dataset detail ----------------
st.divider()
st.subheader(f"📋 {ds.name}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("PCD frames", _pcd_count(ds.pcd_dir))
m2.metric("GT label files", _gt_count(ds.gt_dir))
m3.metric("Type", "template" if ds.is_template else "user")
m4.metric("Created", (ds.d.get("created", "—") or "—").split("T")[0])
if ds.d.get("description"):
    st.caption(ds.d["description"])

with st.expander("📁 Workspace paths"):
    st.json({
        "input PCD dir": ds.pcd_dir,
        "ground-truth dir": ds.gt_dir,
        "lane geometry": ds.lanes_path,
        "site geometry": ds.site_geometry_path,
        "background model": ds.model_path,
        "filtered clouds": ds.filtered_dir,
        "detection outputs": ds.detection_dir,
    })
    if not ds.status()["pcd"]:
        st.warning(f"No .pcd files found in `{ds.pcd_dir}`.")
        if ds.is_template:
            st.info("Template **config** ships with the repo, but its large LiDAR/image **data** "
                    "does not. Download the dataset and place the frames under "
                    f"`{os.path.join(ds.workspace, 'data')}` (see that folder's README). "
                    "You don't edit any paths — the template already points here.")

# --- site geometry view (and edit for user datasets) ---
with st.expander("🗺️ Site geometry (background-filtering region)"):
    st.caption("**Site geometry** (`site_geometry.json`) is the *processed region* used by "
               "**Background Filtering** — the research polygon, road polygons, and exclusion "
               "rectangles. It is **separate from the WWD lanes** you draw in the Lane Editor "
               "(`lanes.geojson`): editing lanes does NOT change this. For the A9 template these "
               "are curated values; for a new dataset they're auto-derived from the data extent.")
    geo = None
    if os.path.exists(ds.site_geometry_path):
        try:
            geo = json.load(open(ds.site_geometry_path))
        except Exception:
            pass
    if geo and geo.get("research_polygon"):
        rp = geo["research_polygon"]
        xs = [p[0] for p in rp]; ys = [p[1] for p in rp]
        st.write(f"Research polygon bounds: X [{min(xs):.1f}, {max(xs):.1f}] · "
                 f"Y [{min(ys):.1f}, {max(ys):.1f}] · {len(geo.get('road_polygons', []))} road polygon(s) · "
                 f"{len(geo.get('foreground_exclusion_rects', []))} exclusion rect(s)")
    else:
        st.info("No site_geometry.json — the pipeline falls back to the built-in default geometry.")

    if not ds.is_template:
        c1, c2 = st.columns(2)
        if c1.button("🔄 Re-derive geometry from this dataset's PCDs"):
            geo2 = dm.derive_site_geometry(ds.pcd_dir)
            os.makedirs(ds.config_dir, exist_ok=True)
            json.dump(geo2, open(ds.site_geometry_path, "w"), indent=2)
            st.success("Re-derived research region from the data extent.")
            st.rerun()
        st.caption("Tip: build lanes for this dataset on the **Lane Editor** page (it saves to this "
                   "dataset's config), then run Background Filtering → Detection.")
    else:
        st.caption("Template geometry is curated and read-only here.")

# --- rename / remove (user datasets) ---
if not ds.is_template:
    with st.expander("✏️ Rename / remove"):
        new_name = st.text_input("Rename", value=ds.name, key="rename_field")
        rc1, rc2 = st.columns(2)
        if rc1.button("Save name"):
            dm.rename_dataset(ds.id, new_name)
            st.rerun()
        if rc2.button(f"🗑️ Remove “{ds.name}” (keeps its files on disk)"):
            dm.delete_dataset(ds.id)
            _clear_session()
            st.rerun()

# ---------------- Add a dataset ----------------
st.divider()
st.subheader("➕ Add your own dataset")
st.caption("Point the app at a folder of `.pcd` frames on disk (not copied). A workspace for its "
           "config/model/outputs is created under `datasets/<id>/`, and a starter site geometry is "
           "derived from the data so it runs right away.")
with st.form("new_dataset"):
    name = st.text_input("Dataset name", placeholder="e.g. My Intersection — North Camera")
    pcd_dir = st.text_input("Path to the folder of PCD frames", placeholder="/path/to/pcd_frames")
    gt_dir = st.text_input("Path to ground-truth labels (optional, OpenLABEL .json)", placeholder="")
    description = st.text_input("Description (optional)", placeholder="sensor, location, notes…")
    submitted = st.form_submit_button("Create dataset", type="primary")
    if submitted:
        if not name.strip():
            st.error("Please enter a name.")
        elif not pcd_dir.strip() or not os.path.isdir(pcd_dir.strip()):
            st.error(f"PCD folder not found: {pcd_dir}")
        else:
            n_pcd = _pcd_count(pcd_dir.strip())
            new = dm.create_dataset(name, pcd_dir, gt_dir, description)
            _clear_session()
            st.success(f"Created **{new.name}** (`{new.id}`) with {n_pcd} PCD frames and set it active. "
                       f"Starter geometry written to `{new.site_geometry_path}`.")
            if n_pcd == 0:
                st.warning("No .pcd files were found in that folder — double-check the path.")
            st.rerun()
