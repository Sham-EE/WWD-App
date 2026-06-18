import os

import streamlit as st

import dataset_manager as dm
import dataset_prep as dp

st.set_page_config(layout="wide", page_title="Dataset Prep")
st.title("🧰 Dataset Prep")
st.markdown("Recreate the dataset's **derived** data from the raw TUM Traffic download, in-app — so "
            "everything the pipeline needs is reproducible, no external preprocessing scripts required.")

ds = dm.get_active()
st.caption(f"📂 Dataset: **{ds.name}**")

# ---------------- 1) Crop to road (ROI) ----------------
st.subheader("1 · Crop point clouds to the road (ROI)")
st.caption("Clips the raw south LiDAR clouds to the **road polygons** in `site_geometry.json`. "
           "This reproduces the `cropped` clouds that Background Filtering / Detection run on "
           "(verified to match the bundled cropped clouds exactly).")

c1, c2 = st.columns(2)
src = c1.text_input("Raw south LiDAR folder", value=ds.raw_lidar_south_dir)
out = c2.text_input("Output (cropped) folder", value=ds.pcd_dir)
margin = st.slider("Road margin (m)", 0.0, 5.0, 0.0, 0.5,
                   help="Expand the road polygon outward before clipping (0 = exact match to the bundled crop).")

vcol, gcol = st.columns(2)
if vcol.button("🔎 Validate against existing cropped", use_container_width=True):
    if not os.path.isdir(src):
        st.error(f"Raw folder not found: `{src}`")
    elif not os.path.isdir(out):
        st.warning("No existing cropped folder to compare against — use Generate instead.")
    else:
        with st.spinner("Cropping sample frames…"):
            rows = dp.validate_crop(src, out, margin=margin, n=8)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            avg = sum(r["match %"] for r in rows) / len(rows)
            (st.success if avg >= 99.5 else st.warning)(
                f"Average point-count match vs the existing cropped clouds: **{avg:.1f}%**")
        else:
            st.warning("No frames found to validate.")

if gcol.button("✂️ Generate cropped clouds", type="primary", use_container_width=True):
    if not os.path.isdir(src):
        st.error(f"Raw folder not found: `{src}`")
    else:
        bar = st.progress(0.0, text="Cropping…")
        n, kept, tot = dp.crop_dataset(src, out, margin=margin,
                                       progress=lambda c, t: bar.progress(c / t, text=f"Cropping {c}/{t}"))
        bar.empty()
        pct = 100.0 * kept / max(tot, 1)
        st.success(f"Wrote **{n}** cropped clouds → `{out}`  (kept {kept:,} / {tot:,} points, {pct:.0f}%).")

# ---------------- 2) Scorable GT (next) ----------------
st.divider()
st.subheader("2 · Generate scorable ground truth (visible-only)")
st.info("Coming next — filters the raw labels down to objects that are in-region and actually "
        "perceived by the LiDAR (reproduces `a9_gt_visible_only_south` for fair evaluation).")
