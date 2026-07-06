# A9_r02_s02 — TUM Traffic Intersection r02_s02 (s110)

Bundled template dataset. **Config travels with the repo; `data/` and `outputs/` do not**
(LiDAR/image data is far too large for GitHub, so both are gitignored).

## Layout
```
config/                              ← tracked in git
  lanes.geojson, site_geometry.json, georef.json
  defaults/                          factory snapshot for "reset to default"
map/                                 ← optional, gitignored
  lane_samples.json                  HD-map road network (dev-kit's src/map/map.zip)
data/                                ← gitignored (local)
  raw/                               ← the untouched TUM Traffic download
    point_clouds/s110_lidar_ouster_{south,north}/   *.pcd
    labels/s110_lidar_ouster_{south,north}/         *.json   (OpenLABEL)
    images/s110_camera_basler_{south1,south2}_8mm/  camera frames
  derived/                           ← regenerated in-app via the Dataset Prep page
    point_clouds/registered/         fused south+north cloud (south LiDAR frame)
    point_clouds/cropped/<sensor>/   clouds clipped to the road (south/north/registered)
    labels/registered/               fused union GT (south∪north, raw)
    labels/scorable/<sensor>/        scorable GT (in-region + LiDAR-visible objects)
outputs/                             ← gitignored, nested by stage:
  background/{background_model,background_filtering}/<sensor>/<crop>/
  detection/object_detection/<sensor>/<crop>/   (tracks.csv, animation, eval report)
  run_history/{bgfilter,eval}/<tag>.jsonl       logged tuning runs, tracked over time
  visualizer/{rendered,road_videos}/
```

## Getting the data
Download the **TUM Traffic Intersection Dataset** (release r02_s02). Its raw folders
map onto `data/raw/` as above:
`_point_clouds → raw/point_clouds`, `_labels → raw/labels`, `_images → raw/images`.

Then open the **Dataset Prep** page to regenerate `data/derived/` (registration, crop
to road, scorable GT) — no external scripts needed. The **Datasets** page shows 🟢 PCDs
once the raw clouds are present.

> Optional: for the HD-map/digital-twin overlay and exact lat/lon, extract
> `lane_samples.json` from the dev-kit's `src/map/map.zip` into `map/`.
