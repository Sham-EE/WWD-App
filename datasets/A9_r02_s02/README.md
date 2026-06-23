# A9_r02_s02 — TUM Traffic Intersection r02_s02 (s110)

Bundled template dataset. **Config travels with the repo; raw/derived data do not**
(LiDAR/image data is far too large for GitHub, so `data/` is gitignored).

## Layout
```
config/                              ← tracked in git
  lanes.geojson, site_geometry.json
data/                                ← gitignored (local)
  raw/                               ← the untouched TUM Traffic download
    point_clouds/s110_lidar_ouster_south/   *.pcd
    point_clouds/s110_lidar_ouster_north/   *.pcd
    labels/s110_lidar_ouster_south/         *.json   (OpenLABEL)
    labels/s110_lidar_ouster_north/         *.json
    images/s110_camera_basler_south1_8mm/   camera frames
    images/s110_camera_basler_south2_8mm/
  derived/                           ← regenerated in-app via the Dataset Prep page
    point_clouds/registered/         fused south+north clouds (s110_base frame)
    point_clouds/cropped/<sensor>/   clouds clipped to the road (south/north/registered)
    labels/scorable/<sensor>/        scorable GT (in-region + LiDAR-visible objects)
outputs/                             ← gitignored, nested by stage:
  background/{background_model,background_filtering}/<sensor>/<crop>/
  detection/object_detection/<sensor>/<crop>/   (tracks.csv, animation, eval report)
  visualizer/{rendered,road_videos}/
```

## Getting the data
Download the **TUM Traffic Intersection Dataset** (release r02_s02). Its raw folders
map onto `data/raw/` as above:
`_points_clouds → raw/point_clouds`, `_labels → raw/labels`, `_images → raw/images`.

Then open the **Dataset Prep** page to regenerate `data/derived/` (crop to ROI +
scorable GT) — no external scripts needed. The **Datasets** page shows 🟢 PCDs once
the raw clouds are present.
