# A9_r02_s02 — TUMTraf A9 (s110 ouster south) template

This is the bundled template dataset. **Its config travels with the repo; its raw
data does not** (LiDAR/image data is far too large for GitHub, so `data/` and
`outputs/` are gitignored).

## What's in git
```
config/   lanes.geojson, site_geometry.json     ← tracked (small)
README.md                                        ← this file
```

## What you must obtain separately (not in git)
```
data/point_clouds/cropped/cropped_pcd/   *.pcd   (LiDAR frames)
data/images/south1/ , data/images/south2/        (camera images: raw, bb, bb_pcd)
data/labels_point_clouds/a9_gt_visible_only_south/  *.json  (OpenLABEL GT)
```
Source: the **TUMTraf Intersection Dataset** (A9, sensor `s110_lidar_ouster_south`).
Download it and place the frames at the paths above (under this folder). The app's
**Datasets** page shows 🟢 PCDs once they're present.

> You do **not** edit any paths by hand — the template already points here. Just
> drop the data into `data/`. To use a *different* site, use the Datasets page's
> "Add your own dataset" (point it at a folder of `.pcd` frames on disk).
