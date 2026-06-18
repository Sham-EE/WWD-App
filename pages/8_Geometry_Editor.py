import os

import numpy as np
import open3d as o3d
import pandas as pd
import streamlit as st

import dataset_manager as dm
import geometry_editor as ge
import road_viewer as rv

st.set_page_config(layout="wide", page_title="Geometry Editor")
st.title("🗺️ Geometry Editor")
st.markdown("Edit the **site geometry** — the research/ROI polygon, the road polygons (used for "
            "cropping), and exclusion rectangles. **Saving updates the whole pipeline** (Background "
            "Filtering, cropping, scorable GT, the road outline) — they all read this file.")

ds = dm.get_active()
st.caption(f"📂 Dataset: **{ds.name}**")


@st.cache_data(show_spinner=False, max_entries=64)
def _load_raw(path):
    return np.asarray(o3d.io.read_point_cloud(path).points)


# ---- editable geometry in session state ----
if "geom_edit" not in st.session_state or st.session_state.get("geom_ds") != ds.id:
    st.session_state.geom_edit = ge.load_site_geometry(ds)
    st.session_state.geom_ds = ds.id
geom = st.session_state.geom_edit

tc1, tc2, tc3 = st.columns([1, 1, 2])
if tc1.button("💾 Save (updates everything)", type="primary", use_container_width=True):
    ge.save_site_geometry(ds, geom)
    st.success(f"Saved → `{ds.site_geometry_path}`. The whole pipeline now uses this geometry.")
if tc2.button("↩️ Reload saved", use_container_width=True):
    st.session_state.geom_edit = ge.load_site_geometry(ds)
    st.rerun()

clouds = rv.list_by_frame(ds.raw_lidar_south_dir, [".pcd"]) or rv.list_by_frame(ds.pcd_dir, [".pcd"])
bg_pts = None
if clouds:
    fi = st.slider("Backdrop frame", 0, len(clouds) - 1, 0)
    bg_pts = _load_raw(clouds[fi])


def _poly_editor(poly, key):
    """Vertex (x,y) table editor -> list of [x,y]."""
    df = pd.DataFrame(poly or [[0.0, 0.0]], columns=["x", "y"])
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=key,
                            column_config={"x": st.column_config.NumberColumn(format="%.2f"),
                                           "y": st.column_config.NumberColumn(format="%.2f")})
    out = edited.dropna(how="any")[["x", "y"]].values.tolist()
    return [[float(x), float(y)] for x, y in out]


left, right = st.columns([1, 1.3], gap="medium")

with left:
    st.subheader("✏️ Edit")
    with st.expander("🔵 Research polygon (ROI)", expanded=True):
        st.caption("The overall analysed region. Used by Background Filtering and as the default "
                   "scorable-GT region.")
        geom["research_polygon"] = _poly_editor(geom.get("research_polygon", []), "ge_research")
        if st.button("Reset research → data extent", key="ge_reset_research"):
            geom["research_polygon"] = dm.derive_site_geometry(ds.pcd_dir)["research_polygon"]
            st.rerun()

    with st.expander("🟢 Road polygons (cropping)", expanded=True):
        st.caption("Drivable area. Points outside these are removed by **Crop to road**.")
        roads = geom.get("road_polygons", [])
        rc1, rc2 = st.columns(2)
        if rc1.button("➕ Add road polygon", key="ge_add_road"):
            roads.append(ge.default_rect(geom)); geom["road_polygons"] = roads; st.rerun()
        if roads:
            ridx = st.selectbox("Road polygon", range(len(roads)),
                                format_func=lambda i: f"#{i+1}", key="ge_road_sel")
            if rc2.button("🗑️ Delete this polygon", key="ge_del_road"):
                roads.pop(ridx); geom["road_polygons"] = roads; st.rerun()
            roads[ridx] = _poly_editor(roads[ridx], f"ge_road_{ridx}")
            geom["road_polygons"] = roads
        else:
            st.info("No road polygons — add one (cropping keeps everything until you do).")

    with st.expander("🔴 Exclusion rectangles", expanded=False):
        st.caption("Regions always treated as background (e.g. static clutter).")
        rects = geom.get("foreground_exclusion_rects", [])
        ec1, ec2 = st.columns(2)
        if ec1.button("➕ Add rectangle", key="ge_add_excl"):
            rects.append(ge.default_rect(geom)); geom["foreground_exclusion_rects"] = rects; st.rerun()
        if rects:
            eidx = st.selectbox("Rectangle", range(len(rects)),
                                format_func=lambda i: f"#{i+1}", key="ge_excl_sel")
            if ec2.button("🗑️ Delete this rect", key="ge_del_excl"):
                rects.pop(eidx); geom["foreground_exclusion_rects"] = rects; st.rerun()
            rects[eidx] = _poly_editor(rects[eidx], f"ge_excl_{eidx}")
            geom["foreground_exclusion_rects"] = rects
        else:
            st.info("No exclusion rectangles.")

with right:
    st.subheader("👁 Live preview")
    st.caption("Scroll to zoom, drag to pan. Edits show immediately; click **Save** to apply everywhere.")
    st.plotly_chart(ge.preview_figure(bg_pts, geom, height=640),
                    use_container_width=True, config={"scrollZoom": True})

st.session_state.geom_edit = geom
