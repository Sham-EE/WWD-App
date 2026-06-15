# LiDAR Wrong-Way Driving (WWD) Detection Toolkit

A Streamlit pipeline for detecting wrong-way driving from roadside LiDAR point
clouds. Three stages:

1. **Background Filtering** — build a static-background model and keep only the
   moving foreground points.
2. **Object Detection, Tracking & WWD** — cluster the foreground, track objects
   with a Kalman filter, and **flag wrong-way vehicles** by comparing each
   vehicle's velocity direction against the expected lane direction.
3. **Evaluation** — score detection/tracking against ground-truth cuboids
   (precision / recall / F1, MOTA / MOTP / ID-switches).

## Running

```bash
# Run from THIS directory (it contains data/ and outputs/), with your Python
# environment active (needs shapely 2.x, scipy, open3d, streamlit, etc.):
streamlit run Home.py
```
Run from *this* folder — the default paths (`data/...`, `outputs/...`) are
relative to the working directory, which is why this copy works out of the box.

## Project layout

| File | Purpose |
|------|---------|
| `Home.py` | Landing page / navigation |
| `pages/1_Background_Filtering.py` | Background model build + foreground viewer |
| `pages/2_Object_Detection_and_Tracking.py` | Detection, tracking, **WWD** |
| `pages/3_Evaluation.py` | Quantitative metrics vs. ground truth |
| `bg_filter_core.py` | Background modelling + filtering algorithms |
| `detection_logic.py` | Candidate extraction, Kalman tracker, association |
| `wwd_detection.py` | **Wrong-way logic** (velocity vs. lane direction) |
| `evaluation.py` | CLEAR-MOT style metrics |
| `geometry_config.py` | Loads scene geometry + fast point-in-polygon |
| `config/site_geometry.json` | Editable scene geometry (research/road/exclusion) |
| `config/lanes.geojson` | **Lane directions for WWD (must be calibrated)** |

## Retargeting to a new site

All scene geometry lives in `config/site_geometry.json` (research polygon, road
polygons, foreground exclusion rectangles). Edit that file — no Python changes
needed. If the file is missing/invalid, the original TUMTraf `s110_ouster_south`
geometry is used as a fallback.

---

## Calibrating lane geometry (required for trustworthy WWD)

`config/lanes.geojson` defines, for each region of road, the **expected
direction of legal travel**. WWD compares each vehicle's measured velocity
heading to this expected heading; if it points sufficiently *against* the lane
(default ≥ 120°) for enough consecutive frames, it is flagged.

The shipped file contains **placeholder** headings with `"calibrated": false`,
and the app shows a warning until you fix them. Calibrate like this:

1. **Pick the heading convention.** Heading is in degrees, math convention:
   `0 = +X`, `90 = +Y`, `180/-180 = -X`, `-90 = -Y`, computed as
   `atan2(vy, vx)` in the sensor frame.

2. **Find the real direction of each lane from your data.**
   - Run **Object Detection and Tracking** on a clip you *know* contains only
     normal (correct-direction) traffic.
   - Open `outputs/object_detection/tracks.csv`. It now contains `vx, vy,
     heading` columns. For vehicles in a given lane, compute the typical heading:
     ```bash
     # rough average heading (radians) of moving vehicles, convert to degrees
     ../venv/bin/python - <<'PY'
     import pandas as pd, numpy as np
     df = pd.read_csv('outputs/object_detection/tracks.csv')
     df = df[(df.moving == 1) & df.heading.notna()]
     # mean of a circular quantity: average the unit vectors
     ang = np.arctan2(np.sin(df.heading).mean(), np.cos(df.heading).mean())
     print('mean heading deg =', np.degrees(ang))
     PY
     ```
   - If a lane carries both directions, split it into two polygons (one per
     direction) so each has a single expected heading.

3. **Draw lane polygons.** Each feature's `geometry.coordinates` is a polygon in
   sensor-frame metres. Start from the road polygons in
   `config/site_geometry.json` and subdivide them per travel direction. (Tip:
   the 3-D viewer shows the road outline in green; read off approximate X/Y
   corners there, or overlay on a labelled GT frame.)

4. **Set `heading_deg` and `"calibrated": true`** for every feature once you are
   confident. The warning banner disappears when all lanes are calibrated.

5. **Validate.** Re-run on the normal-traffic clip: it should report *zero*
   wrong-way vehicles. Then run on a clip with a known wrong-way event and
   confirm it is flagged. Tune the angle / min-speed / sustained-frames /
   displacement sliders on the WWD panel to trade off sensitivity vs.
   false alarms.

> If you have map data (HD map / OpenDRIVE) or the dataset's lane annotations,
> you can instead export lane polygons + headings directly into this GeoJSON,
> which is more accurate than reading them off trajectories.

## Ground truth for evaluation

The Evaluation page reads OpenLABEL `.json` cuboid files from a directory you
specify and aligns them to detection frames by the leading
`<timestamp1>_<timestamp2>` token in the filenames (already true for the TUMTraf
data here). No extra labelling is needed beyond the cuboid GT you already have.
