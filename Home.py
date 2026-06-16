import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Home")

st.sidebar.success("Select a tool above.")

st.markdown(
    """
    Welcome to the Lidar Processing Toolkit!

    This application provides tools for visualizing and processing 3D point cloud data.

    **👈 Select a tool from the sidebar** to get started.

    ### Available Tools:
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Background Filtering")
    st.write("Build a background model and filter dynamic objects.")
    if st.button("Go to Background Filtering", use_container_width=True):
        st.switch_page("pages/1_Background_Filtering.py")

with col2:
    st.subheader("Detection, Tracking & WWD")
    st.write("Detect, track, and flag wrong-way vehicles in filtered clouds.")
    if st.button("Go to Object Detection and Tracking", use_container_width=True):
        st.switch_page("pages/2_Object_Detection_and_Tracking.py")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Evaluation")
    st.write("Score detection/tracking against ground truth (P/R/F1, MOTA).")
    if st.button("Go to Evaluation", use_container_width=True):
        st.switch_page("pages/3_Evaluation.py")

with col4:
    st.subheader("Lane Editor")
    st.write("Build/adjust wrong-way lane geometry from tracks.csv and export it.")
    if st.button("Go to Lane Editor", use_container_width=True):
        st.switch_page("pages/4_Lane_Editor.py")

col5, _ = st.columns(2)

with col5:
    st.subheader("🚨 WWD Simulator")
    st.write("Spawn a synthetic wrong-way driver, watch the detector flag it, fire the alert.")
    if st.button("Go to WWD Simulator", use_container_width=True):
        st.switch_page("pages/5_WWD_Simulator.py")



