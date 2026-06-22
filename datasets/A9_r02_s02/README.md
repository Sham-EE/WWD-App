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
    images/south1/ , images/south2/         camera frames
  derived/                           ← regenerated in-app via the Dataset Prep page
    cropped_south/                   south clouds clipped to the road (pipeline input)
    cropped_north/                   north clouds clipped to the road
    registered/                      fused south+north clouds (s110_base frame)
    labels_visible_south/            scorable GT (in-region + LiDAR-visible objects)
    labels_visible_north/            scorable GT for the north sensor
outputs/                             ← gitignored (background model, filtered clouds, detection, videos)
```

## Getting the data
Download the **TUM Traffic Intersection Dataset** (release r02_s02). Its raw folders
map onto `data/raw/` as above:
`_points_clouds → raw/point_clouds`, `_labels → raw/labels`, `_images → raw/images`.

Then open the **Dataset Prep** page to regenerate `data/derived/` (crop to ROI +
scorable GT) — no external scripts needed. The **Datasets** page shows 🟢 PCDs once
the raw clouds are present.
