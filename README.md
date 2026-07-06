# LiDAR Wrong-Way Driving (WWD) Detection Toolkit

A Streamlit pipeline for detecting wrong-way driving from roadside LiDAR point
clouds: background filtering → clustering/tracking → wrong-way flagging →
quantitative evaluation, plus tools to calibrate a new site (lane geometry,
scene geometry, sensor registration) and to demo a wrong-way alert over a
simulated V2X broadcast.

**Contents:** [Quickstart](#quickstart) · [What's inside](#whats-inside) ·
[How it works](#how-it-works) · [Datasets](#datasets) ·
[Current benchmark](#current-benchmark) · [Project layout](#project-layout) ·
[Known limitations / next steps](#known-limitations--next-steps)

---

## Quickstart

```bash
# 1. Install Git, if you don't have it: https://git-scm.com/downloads

# 2. Clone this repository
git clone <this-repository-url>
cd <repository-folder>

# 3. Install Conda, if you don't have it: https://docs.conda.io/en/latest/miniconda.html

# 4. Create the environment (Python 3.11 — open3d doesn't yet support 3.13)
conda create -n lidar_env python=3.11 -y

# 5. Activate it
conda activate lidar_env

# 6. Install dependencies
pip install -r requirements.txt

# 7. Run the app
streamlit run Home.py
# (if `streamlit` isn't on PATH: python -m streamlit run Home.py)
```

The bundled dataset (`datasets/A9_r02_s02/`) ships with sample derived data, so
the app is usable immediately — see **Datasets** below for adding your own, and
**End-to-end workflow** if you need to regenerate derived data from scratch.

> **Optional — HD-map overlay:** the Visualizer's HD-map/digital-twin view and
> exact lat/lon features read `datasets/<id>/map/lane_samples.json`, extracted
> from the TUM Traffic dev-kit's `src/map/map.zip`. Everything degrades
> gracefully without it.

---

## What's inside

The sidebar groups pages into three sections (Home shows a dataset-aware
**pipeline stepper** marking each stage Done / Next / To-do):

**Data & setup**
- **Datasets** — pick the active dataset, or add your own (a folder of `.pcd`
  frames + optional OpenLABEL GT).
- **Dataset Prep** — registration (fuse south+north LiDARs), the scene-geometry
  editor (research/road/exclusion polygons the whole pipeline reads), crop-to-road,
  and scorable-GT generation.
- **Visualizer** — camera + 3D LiDAR label viewers and a real-intersection map;
  a native replacement for the TUM Traffic dev-kit's visualization tools.

**Detection pipeline**
- **Background Filtering** — build a static-background model and keep only the
  moving foreground, with a full 3D inspector for tuning.
- **Detection & Tracking** — cluster the foreground, track objects with a
  Kalman filter, associate detections frame-to-frame.
- **Evaluation** — score against ground truth (precision/recall/F1, MOTA/MOTP/
  ID-switches) with a visual GT-vs-detection comparison, plus a Registered-vs-South
  A/B benchmark.

**Wrong-way driving**
- **Lane Editor** — draw lane geometry (boxes or arbitrary polygons) and set each
  lane's legal direction of travel — the reference WWD compares vehicle heading against.
- **WWD Simulator** — inject a synthetic wrong-way driver through the real
  detector/flagging logic (no scripted result) and fire a simulated V2X alert.
- **V2X Dashboard** — a standalone alert dashboard seeded with the real,
  georeferenced intersection and a chosen wrong-way scenario.

---

## How it works

- **Background filtering** (`bg_filter_core.py`): a per-cell/voxel occupancy +
  cluster-recurrence model of what's static, refreshed by a geometric pole-shape
  filter and optional denoising. Cropped input dramatically outperforms full/raw
  input since off-road clutter is the dominant false-positive source (see
  *Current benchmark* below).
- **Detection & tracking** (`detection_logic.py`): range-adaptive clustering,
  PCA-oriented boxes, temporal confirmation, a Kalman tracker with gated Hungarian
  association, and a static-phantom suppression pass. Defaults live in one place,
  `DEFAULT_DETECTION_PARAMS` — the UI and the A/B benchmark both read it, so they
  can't silently drift apart.
- **Wrong-way flagging** (`wwd_detection.py`): a box's PCA yaw is 180°-ambiguous, so
  WWD instead compares the **Kalman velocity direction** to the expected lane
  heading (`config/lanes.geojson`). A track is flagged only for a sustained run of
  frames that's fast enough, wrong enough, travels far enough, and holds a steady
  heading — with the intersection interior exempted so legal turns aren't mis-flagged.
- **Evaluation** (`evaluation.py`): OpenLABEL GT aligned to detection frames by
  filename timestamp, Hungarian BEV-centre matching gated *before* assignment, CLEAR-MOT
  metrics, class/ROI filters, and a Registered-vs-South A/B harness with per-distance-bin
  recall (the direct test of "fusion fills occlusion shadows").
- **Registration** (`registration.py`): fuses south+north via calibration-init +
  coarse-to-fine point-to-plane ICP (the bundled extrinsics alone miss a
  systematic ~8° yaw), written in the south LiDAR frame so it's a drop-in
  superset — south GT/calibration/geometry all apply to it directly. Also builds a
  de-duplicated **fused-union GT** with per-object point counts recomputed on the
  fused cloud.
- **Georeferencing** (`geo_reference.py`): composes the OpenLABEL + HD-map chain
  into exact WGS84 lat/lon + true compass bearings, so the Lane Editor, WWD
  Simulator, and V2X Dashboard all agree on real-world directions and position.

---

## Datasets

The app is **multi-dataset**; each has its own self-contained workspace:

```
datasets/<id>/
  config/            lanes.geojson, site_geometry.json, georef.json  ← tracked (small)
    defaults/        factory snapshot for "reset to default"
  map/               lane_samples.json (HD map, optional, gitignored)
  data/              ← gitignored (large, local)
    raw/             point_clouds/{..._south,_north}/, labels/, images/   # untouched download
    derived/         point_clouds/{registered, cropped/<sensor>}/, labels/scorable/<sensor>/
  outputs/           ← gitignored, nested by stage
    background/{background_model,background_filtering}/<sensor>/<crop>/
    detection/object_detection/<sensor>/<crop>/     # tracks.csv, animation, eval report
    run_history/{bgfilter,eval}/<tag>.jsonl          # logged tuning runs, tracked over time
    visualizer/{rendered,road_videos}/
```

Two toggles — **Sensor** (Registered / South / North) and **Input cloud**
(Cropped / Full) — are shared across Filtering → Detection → Evaluation via
session state; each combination writes to its own folder so results never
collide and metrics stay comparable. GT auto-resolves to match the selected
sensor.

### End-to-end workflow (regenerating a dataset from scratch)
1. **Dataset Prep** → Registration (optional) → Geometry Editor → Crop to road → Scorable GT.
2. **Background Filtering** → pick Sensor + Input cloud → *Build Background Model*
   (with "Save filtered foreground points" checked).
3. **Detection & Tracking** → *Start Detection* (same Sensor/Input) → writes `tracks.csv`.
4. **Evaluation** → *Run Evaluation* (Restrict to ROI on for the fair number).
5. **Lane Editor** → calibrate lanes → Save, so WWD has a reference direction per lane.

---

## Current benchmark

*(Registered/cropped, gated matcher, exclusion zones, ROI on, 2.0 m BEV match gate —
tuned 2026-07-04.)*

| classes | Precision | Recall | F1 | MOTP |
|---|---|---|---|---|
| vehicles only | 0.724 | 0.752 | 0.738 | ~1.25 m |
| all classes | 0.746 | 0.740 | 0.743 | ~1.22 m |

Fusion (Registered vs South) is a clear recall win, concentrated near/mid-field.
Cropping to the road polygon is the single biggest lever in the pipeline (off-road
clutter dominates false positives on the full/raw cloud).

> The full sweep tables, the A/B fusion benchmark, and the static-phantom FP
> analysis behind these numbers live in a local `RESULTS.md` working document
> (git-ignored — a lab notebook, not shipped with the repo).

---

## Project layout

| File | Purpose |
|------|---------|
| `Home.py` | Landing page — pipeline stepper + tool cards |
| `nav.py` | Shared sidebar/nav, tool definitions, per-tool completion |
| `pages/0_Datasets.py` | Select/add a dataset |
| `pages/1_Dataset_Prep.py` | Registration, Geometry Editor, Crop to road, Scorable GT (4 tabs) |
| `pages/2_Background_Filtering.py` | Background model build + foreground 3D inspector |
| `pages/3_Object_Detection_and_Tracking.py` | Detection, tracking, WWD, viewer |
| `pages/4_Evaluation.py` | Metrics + visual GT-vs-detection + A/B benchmark |
| `pages/5_Lane_Editor.py` | Build/adjust lane geometry |
| `pages/6_WWD_Simulator.py` | Synthetic wrong-way driver + embedded V2X broadcast |
| `pages/7_Visualizer.py` | Camera + 3D LiDAR viewers, real-intersection map |
| `pages/8_V2X_Dashboard.py` | Standalone V2X alert dashboard |
| `dataset_manager.py` | Dataset registry + per-dataset/sensor/source path resolution |
| `geometry_config.py` / `geometry_editor.py` | Scene geometry: load/save + point-in-polygon helpers |
| `bg_filter_core.py` | Background modelling + filtering |
| `detection_logic.py` | Clustering, Kalman tracker, association, `DEFAULT_DETECTION_PARAMS` |
| `wwd_detection.py` | Wrong-way logic (velocity vs. lane direction) |
| `evaluation.py` | CLEAR-MOT metrics, BEV figures, class/ROI filters |
| `registration.py` | South+north LiDAR fusion (calibration + ICP + fused GT) |
| `lane_tools.py` | Lane Editor helpers (auto-cluster, drawable polygons, heading fit) |
| `geo_reference.py` | Georeferencing: sensor↔WGS84, true bearings, HD-map road network |
| `wwd_simulator.py` | Synthetic wrong-way track + V2X dashboard integration |
| `run_history.py` | Persistent tuning-run history (`outputs/run_history/`) |
| `visualization.py` / `lidar_viewer.py` / `label_projection.py` | 3D/camera rendering helpers |
| `viewer_ui.py` / `road_viewer.py` | Shared viewer controls + video helpers |
| `dataset_prep.py` | Crop + scorable-GT generation, BEV previews |

---

## Known limitations / next steps

Full detail and measured evidence for each of these lives in the local
`RESULTS.md` working notebook mentioned above. Headline items:
- **A learned detector** — the clustering pipeline is at its measured precision
  ceiling; candidates include PointPillars, SECOND, CenterPoint, PV-RCNN, or
  InfraDet3D (purpose-built for roadside infrastructure LiDAR).
- **Validate beyond this one site** — every number here comes from a single
  282-frame clip at one intersection. The pipeline is dataset-aware and
  georeference-agnostic by design; the next real test is a second intersection.
- **Tracker association tuning** to reduce ID-switches further (gate/noise terms
  were never swept).
- **A real hardware pipeline** — move the V2X broadcast from simulation to an
  actual RSU/OBU exchanging real C-V2X/J2735 TIM messages.

> Streamlit resets all sliders to their defaults on a browser refresh — for
> reproducible numbers, note the settings used (especially the eval match-distance
> gate), or check `outputs/run_history/` for the logged trend.
