# LiDAR Wrong-Way Driving (WWD) Detection Toolkit

A Streamlit pipeline for detecting wrong-way driving from roadside LiDAR point
clouds. The pages:

0. **Datasets** — choose which dataset the app works on, or add your own.
1. **Background Filtering** — build a static-background model and keep only the
   moving foreground points.
2. **Object Detection, Tracking & WWD** — cluster the foreground, track objects
   with a Kalman filter, and **flag wrong-way vehicles** by comparing each
   vehicle's velocity direction against the expected lane direction.
3. **Evaluation** — score detection/tracking against ground-truth cuboids
   (precision / recall / F1, MOTA / MOTP / ID-switches) with a side-by-side
   GT-vs-detection visual comparison.
4. **Lane Editor** — build/adjust the wrong-way lane geometry from data and
   export it.
5. **WWD Simulator** — spawn a synthetic wrong-way driver through the real
   detector and fire the V2X dashboard's messaging on detection.
6. **Visualizer** — two tabs: a **camera** viewer (both cameras side by side with
   generated bounding-box / point-cloud overlays + track-history trails + video),
   and a **3D LiDAR** viewer (the scan + ground-truth boxes in Bird's-Eye and Side
   views). A native, dependency-light replacement for the TUM Traffic dev-kit.

## Datasets

The app is **multi-dataset**. Each dataset has its own self-contained workspace
under `datasets/<id>/`:

```
datasets/A9_r02_s02/         # the bundled TUMTraf A9 template
  config/   lanes.geojson, site_geometry.json     ← tracked in git (small)
  data/     point_clouds/…, labels_point_clouds/… ← gitignored (local)
  outputs/  background_model/, background_filtering/, object_detection/ ← gitignored
```

On the **Datasets** page you can switch the active dataset (every page reads/writes
the active one) or **add your own**: point it at a folder of `.pcd` frames on disk
(not copied) and optional OpenLABEL GT. A starter `site_geometry.json` is derived
from the data extent so it runs immediately; datasets **without GT** still filter
(the background model derives its height band from the point cloud). Build lanes
for a new dataset on the **Lane Editor** page, then run Background Filtering →
Detection → Evaluation.

## Visualizer (page 6) — native dev-kit replacement

The TUM Traffic dev-kit's label visualizations need a heavy native/OpenGL setup
that's painful to install on Mac/Windows. This page reproduces them with pure
Python (PIL + Plotly), reading the camera calibration **straight out of each
OpenLABEL label JSON** (`coordinate_systems` extrinsics + `streams` intrinsics) —
so no dev-kit, no calibration files, and no extra dependencies.

- **Camera tab** — both cameras side by side (south2 left, south1 right by
  default). Switch between **Raw**, **Bounding boxes (3D)**, and **Boxes + point
  cloud** (LiDAR projected onto the image, depth-coloured blue→red). Overlays are
  generated on the fly from labels + calibration and cached under
  `outputs/rendered/`. Optional **track-history trails** (coloured per object,
  adjustable length/thickness). Exact dev-kit class colours + `TYPE_id` labels.
  **Generate** a side-by-side MP4 (GIF fallback) of any variant.
- **3D LiDAR tab** — the point cloud + ground-truth 3D boxes in the sensor frame,
  shown as **Bird's-Eye** and **Side** views (rotate/zoom), coloured by category.

Frame stepping is isolated with `st.fragment` + cached loads so playback is smooth
(no full-page refresh). The same playback control is on **Background Filtering**.

## Running

```bash
# From this folder, with your Python environment active (see requirements.txt:
# shapely 2.x, scipy, scikit-learn, open3d, streamlit, plotly, pandas, matplotlib,
# imageio + imageio-ffmpeg for MP4 export):
streamlit run Home.py
```

### End-to-end workflow
1. **Background Filtering** → *Build Background Model* with **"Save filtered
   foreground points (PCD)" checked**. This writes `outputs/background_filtering/`.
   ⚠️ The live viewer filters on the fly; only the **Build** button writes the
   files that detection reads. Re-run this whenever you change a filter setting.
2. **Object Detection and Tracking** → *Start Detection*. Reads the filtered
   clouds, produces tracks + `outputs/object_detection/tracks.csv`, runs WWD.
3. **Evaluation** → *Run Evaluation* (keep **Restrict to ROI** on for the fair
   number). Use the **Visual Evaluation** panel to step through frames.
4. **Lane Editor** → calibrate/adjust lanes, **Save to config**.

> Note: refreshing the browser resets all Streamlit sliders to their defaults,
> so a run after a refresh uses default parameters. For reproducible numbers,
> record the settings you used (especially the eval **match-distance gate**).

## Project layout

| File | Purpose |
|------|---------|
| `Home.py` | Landing page / navigation (grouped tool cards + active dataset) |
| `pages/0_Datasets.py` | Select active dataset / add your own; per-dataset workspace |
| `pages/1_Background_Filtering.py` | Background model build + foreground viewer |
| `pages/2_Object_Detection_and_Tracking.py` | Detection, tracking, **WWD**, viewer + GIF |
| `pages/3_Evaluation.py` | Metrics + **visual GT-vs-detection** comparison |
| `pages/4_Lane_Editor.py` | Build/adjust lane geometry, export `lanes.geojson` |
| `pages/5_WWD_Simulator.py` | Synthetic wrong-way driver + **V2X dashboard** broadcast |
| `pages/6_Visualizer.py` | Camera overlays + **3D LiDAR labels** (dev-kit replacement) |
| `dataset_manager.py` | Dataset registry, active-dataset, per-dataset path resolution |
| `bg_filter_core.py` | Background modelling + filtering (GT-less z-band fallback) |
| `detection_logic.py` | Candidate extraction (adaptive eps), Kalman tracker, association |
| `wwd_detection.py` | **Wrong-way logic** (velocity vs. lane direction) |
| `evaluation.py` | CLEAR-MOT metrics + BEV figures + class/ROI filters |
| `geometry_config.py` | Loads the active dataset's scene geometry + point-in-polygon |
| `lane_tools.py` | Lane Editor helpers (auto-cluster, geojson, 3D preview) |
| `visualization.py` | 3D interactive view + matplotlib GIF (cardinal arrows) |
| `label_projection.py` | OpenLABEL boxes/point-cloud → camera image (calibration from JSON) |
| `lidar_viewer.py` | 3D LiDAR scan + GT boxes (BEV / side) via Plotly |
| `road_viewer.py` | Camera browsing + side-by-side video helpers |
| `wwd_simulator.py` | Synthetic wrong-way track + V2X dashboard integration |
| `datasets/<id>/config/site_geometry.json` | Per-dataset scene geometry (research/road/exclusion) |
| `datasets/<id>/config/lanes.geojson` | Per-dataset **lane directions for WWD** |

---

## How wrong-way detection works

The bounding-box yaw from PCA is 180°-ambiguous, so it can't distinguish
wrong-way from right-way. Instead WWD uses the **Kalman velocity** direction
(`atan2(vy, vx)`) and compares it to the **expected heading** of the lane the
vehicle is in (the active dataset's `config/lanes.geojson`). A track is flagged only when, for a
sustained run of frames, it is fast enough, points far enough against the lane,
travels far enough, **and** holds a steady heading — with the **intersection
interior exempted** (where lane boxes overlap and turning is legal). This is what
prevents turning vehicles at the junction from being mis-flagged.

WWD parameters live on the Detection page: *Angle vs. flow*, *Min speed*,
*Sustained frames*, *Min displacement*, *Exempt junction turns*, *Min heading
steadiness*.

### Visualization
- Object markers and their heading **arrows are color-coded by cardinal
  direction** (E=red, N=green, W=blue, S=orange); stationary/undefined = gray.
- Wrong-way vehicles show an **orange diamond + "WRONG WAY"** label.
- Lane boxes + expected-direction arrows can be overlaid (toggle on the page).

---

## Calibrating lane geometry (Lane Editor — page 4)

Each dataset's `config/lanes.geojson` defines, per road region, the **expected
legal direction of travel** (degrees, math convention: `0=+X`, `90=+Y`, `180/-180=-X`, `-90=-Y`,
i.e. `atan2(vy,vx)` in the sensor frame). The file is **currently calibrated**
(eastbound / westbound / northbound / southbound). To recalibrate or retarget:

1. Run a detection first (produces `outputs/object_detection/tracks.csv`).
2. Open the **Lane Editor**. Click **Auto-generate** to cluster the observed
   traffic into N directions and create starting boxes with measured headings.
3. **Adjust** each lane's box (X/Y min/max) and heading in the table; watch the
   live top-down preview (color points by *Cardinal*, *Lane membership*, or
   *Heading*; toggle the point-cloud backdrop; pan/scroll-zoom).
4. **Save to config** (overwrites the active dataset's `config/lanes.geojson`).
5. **Validate:** re-run on normal traffic → expect zero wrong-way flags; run a
   known wrong-way clip → expect it flagged. Tune the WWD sliders.

> The names (eastbound, etc.) are a convention — `+Y` is the sensor's axis, not
> verified geographic north. It doesn't affect WWD correctness (only relative
> direction matters), just the labels.

---

## Evaluation notes

- GT is read from OpenLABEL `.json` cuboids and aligned to detection frames by
  the leading `<timestamp1>_<timestamp2>` filename token.
- **Restrict to processed region (ROI)** (on by default): only scores GT inside
  the area the detector actually processes (research polygon ∩ `|y| ≤ roi_abs_y`).
  Objects outside the sensor's operational region aren't counted as misses — this
  is the fair number. (Many GT objects sit beyond x=45 / |y|>40, where the system
  never looks.)
- **Vehicles only**: scores only CAR/TRUCK/VAN/BUS/TRAILER/MOTORCYCLE. Note this
  currently *lowers* precision, because the detector emits boxes that match
  pedestrians/bicycles (it does not yet classify), so excluding those classes
  turns matches into false positives.
- **Match-distance gate** strongly affects the numbers — always report it.
- **Visual Evaluation** panel: step frame-by-frame, GT (left) vs detections
  (right) in identical axes; toggle separate/overlay views, point-cloud backdrop,
  and "show missed (red)" to see false negatives.

### Current baseline (defaults, ROI on, all classes)
| match gate | Precision | Recall | F1 | MOTA | MOTP |
|---|---|---|---|---|---|
| 2.0 m (strict) | 0.736 | 0.635 | 0.682 | 0.400 | ~1.0 m |
| 2.5 m | 0.813 | 0.701 | 0.753 | 0.531 | ~1.0 m |

(Detection is deterministic: identical settings → identical results.)

---

## Changelog (highlights since the pipeline came together)

**Core detection / WWD**
- **WWD** implemented (Kalman velocity vs. lane direction) with junction
  exemption + heading-steadiness so turning vehicles aren't mis-flagged.
- **Quantitative evaluation** (P/R/F1, MOTA/MOTP/ID-switches) + **visual
  GT-vs-detection** comparison; **class** and **ROI** filters; the **settings
  used** are written into `evaluation_report.json` and shown in-app.
- **Lane Editor** (auto-cluster from `tracks.csv`, editable table, live 3D
  preview, export). **Cardinal-direction colour encoding** for markers/arrows.
- **Range-adaptive DBSCAN eps** (`eps = eps0 + eps_k·range`, clipped) — better on
  both precision and recall for near/dense vs far/sparse objects. Optional
  **vehicle class gate**.
- Recall bug-fixes: pole filter no longer deletes dense **trucks**
  (`pole_max_points`); fast dense clusters bypass temporal confirmation
  (`strong_pts`).

**Datasets**
- **Multi-dataset workspaces** (`datasets/<id>/`): switch the active dataset or
  add your own (point at a PCD folder); per-dataset config/model/outputs; starter
  site-geometry auto-derived from data; **GT-less datasets still filter** (z-band
  derived from the point cloud). The whole pipeline is dataset-aware.

**Visualization (V2X + dev-kit replacement)**
- **WWD Simulator** + embedded **V2X dashboard**: a synthetic wrong-way driver
  runs through the real detector and auto-fires the dashboard's TIM/C-V2X messaging
  on detection.
- **Visualizer** page reproduces the TUM Traffic dev-kit's label visualizations
  natively (calibration read from the OpenLABEL JSON): camera box / point-cloud
  overlays with track-history trails + video, and a 3D LiDAR BEV/Side view.
- Smooth playback everywhere (`st.fragment` + cached loads).

## Open follow-ups
- Reduce **ID switches** further (association / `max_missed` tuning).
- Remaining **GT leakage** for the bundled A9 dataset: the background filter still
  uses GT cuboids to set per-region z-bands when GT is present (new datasets
  already use the label-free fallback).
- **Raw (uncropped) point clouds**: the bundled data is cropped (~63 m), so the
  point-cloud overlays / 3D view don't reach distant structures — adding the full
  clouds would extend them automatically.
