"""Georeferencing for the TUMTraf s110 intersection.

The dataset is recorded at the Providentia++ / TUMTraf Intersection test field at the
intersection of Schleißheimer Straße (B471) × Zeppelinstraße, Garching-Hochbrück,
just north of Munich, DE (Zimmer et al., ITSC 2023, §III.A).

EXACT georeference (survey-grade) — the dataset's HD map carries the geodetic anchor.
`map/lane_samples.json` (from the dev-kit's src/map/map.zip) has:
  - geoReference: a PROJ Transverse-Mercator string == UTM zone 32N
    (+proj=tmerc +lon_0=9 +k=0.9996 +x_0=500000 +datum=WGS84)
  - origin: the projected (easting, northing, elev) of the HD-map frame's (0,0,0).
Chain to WGS84:

    sensor (s110_lidar_ouster_south/north)
      → s110_base            (OpenLABEL pose_wrt_parent, from the label files)
      → HD-map frame         (inverse of the dev-kit hd_map.py map→s110_base transform)
      → projected E/N         (+ HD-map origin offset)
      → lat/lon              (pyproj, via geoReference)

Verified: south LiDAR origin → 48.24946, 11.63086 (the gantry, on the junction).

Fallback: if the HD map or pyproj is unavailable, we fall back to an APPROXIMATE
placement (correct shape + orientation from the sensor→map rotation, centred on the
site centroid) so the live map still renders.
"""
import functools
import glob
import json
import math
import os
import re

import numpy as np

# Human-readable site, for dashboard/UI display.
SITE_NAME = "Schleißheimer Str. (B471) × Zeppelinstr., Garching-Hochbrück (Munich), DE"
# Real crossroads centre (OSM Overpass nodes 860592919 / 1941137894). Used only as the
# fallback map centre when the exact HD-map chain is unavailable.
SITE_LATLON_APPROX = (48.2494, 11.6308)

# HD-map → s110_base rigid transform, lifted verbatim from the TUMTraf dev-kit
# (src/map/hd_map.py, _get_transform_map2local("s110_base")). map_point → s110_base.
_MAP2BASE_R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_MAP2BASE_T = np.array([-854.96568588, -631.98486299, 0.0])

# s110_base → point-cloud (native LiDAR) frame, lifted verbatim from the dev-kit
# (src/visualization/visualize_point_cloud_with_3d_boxes.py): it subtracts this
# translation then rotates by 77.8° about z (applied as points @ Rz). This is the
# transform that overlays the HD-map on the raw point cloud — copying it exactly
# fixes the rotation/offset the OpenLABEL-extrinsic chain got wrong.
_S110_TO_CLOUD_T = np.array([-15.87257873, 2.30019086, 7.48077521])
_S110_TO_CLOUD_RZ_DEG = 77.8


def _rz(deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

# Candidate locations for the HD-map lane_samples.json (extracted from map.zip).
_HDMAP_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "map", "lane_samples.json"),
    os.path.join(os.path.dirname(__file__), "datasets", "A9_r02_s02", "map", "lane_samples.json"),
]


# --------------------------------------------------------------------------- #
#  OpenLABEL chain (sensor → s110_base, and sensor → hd_map_origin fallback)
# --------------------------------------------------------------------------- #
def _read_coord_systems(sensor="south"):
    """OpenLABEL coordinate_systems from the first raw label of the active dataset
    (calibration is static across frames). None if unavailable."""
    try:
        import dataset_manager as dm
        ds = dm.get_active()
        d = ds.raw_labels_south_dir if sensor == "south" else ds.raw_labels_north_dir
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        if not files:
            return None
        with open(files[0], "r") as f:
            return json.load(f)["openlabel"]["coordinate_systems"]
    except Exception:
        return None


def _compose(cs, sensor):
    """4x4 transform: sensor frame → chain root (hd_map_origin), walking pose_wrt_parent
    (child→parent) up the tree. None if missing."""
    key = f"s110_lidar_ouster_{sensor}"
    if cs is None or key not in cs:
        return None
    M = np.eye(4)
    node = key
    while node and node in cs and cs[node].get("parent"):
        m = cs[node].get("pose_wrt_parent", {}).get("matrix4x4")
        if m:
            M = np.array(m, dtype=float).reshape(4, 4) @ M
        node = cs[node]["parent"]
    return M


@functools.lru_cache(maxsize=4)
def sensor_to_s110_base(sensor="south"):
    """Cached 4x4 sensor→s110_base (None if no labels)."""
    cs = _read_coord_systems(sensor)
    if cs is None:
        return None
    key = f"s110_lidar_ouster_{sensor}"
    # s110_base is the direct parent of the lidar, so its single pose_wrt_parent is it.
    m = cs.get(key, {}).get("pose_wrt_parent", {}).get("matrix4x4")
    return np.array(m, dtype=float).reshape(4, 4) if m else None


@functools.lru_cache(maxsize=4)
def sensor_to_map_transform(sensor="south"):
    """Cached 4x4 sensor→hd_map_origin transform (OpenLABEL root). Used only by the
    approximate fallback path."""
    return _compose(_read_coord_systems(sensor), sensor)


# --------------------------------------------------------------------------- #
#  HD-map geodetic anchor (the exact path)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _hdmap_georef():
    """(proj_string, origin_xyz) from the HD-map lane_samples.json, or None. Reads only
    the small header (geoReference + origin precede the huge `roads` array), so it does
    NOT parse the whole 48 MB file."""
    for path in _HDMAP_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                head = f.read(4096)
            gr = re.search(r'"geoReference"\s*:\s*"([^"]*)"', head)
            og = re.search(r'"origin"\s*:\s*\[([^\]]*)\]', head)
            if gr and og:
                origin = [float(v) for v in og.group(1).split(",")]
                return gr.group(1), origin
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=2)
def _transformer(proj_string):
    """Cached pyproj Transformer proj→WGS84. None if pyproj is missing."""
    try:
        from pyproj import Transformer
        return Transformer.from_crs(proj_string, "EPSG:4326", always_xy=True)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _geod():
    try:
        from pyproj import Geod
        return Geod(ellps="WGS84")
    except Exception:
        return None


def has_exact_georef(sensor="south"):
    """True iff lat/lon projection is available (HD map + pyproj)."""
    gr = _hdmap_georef()
    return gr is not None and _transformer(gr[0]) is not None


def sensor_to_projected(x, y, z=0.0, sensor="south"):
    """Point-cloud-frame (x,y,z) → projected (easting, northing) in the HD-map CRS
    (metres). Inverse of the dev-kit recipe: cloud → s110_base → HD-map. None if the
    HD map is unavailable."""
    gr = _hdmap_georef()
    if gr is None:
        return None
    _, origin = gr
    cloud = np.array([float(x), float(y), float(z)])
    s110 = _rz(_S110_TO_CLOUD_RZ_DEG) @ cloud + _S110_TO_CLOUD_T   # cloud → s110_base
    mp = _MAP2BASE_R.T @ (s110 - _MAP2BASE_T)                      # s110_base → HD-map
    return origin[0] + mp[0], origin[1] + mp[1]


def sensor_xy_to_latlon(x, y, sensor="south"):
    """Exact WGS84 (lat, lon) for a point-cloud-frame (x, y) via the HD-map anchor.
    None if the HD map / pyproj are unavailable."""
    gr = _hdmap_georef()
    if gr is None:
        return None
    tr = _transformer(gr[0])
    en = sensor_to_projected(x, y, 0.0, sensor)
    if tr is None or en is None:
        return None
    lon, lat = tr.transform(en[0], en[1])
    return float(lat), float(lon)


# --------------------------------------------------------------------------- #
#  Public: bearing + projector (exact when possible, graceful fallback)
# --------------------------------------------------------------------------- #
def _naive_compass(heading_rad):
    """Last-resort: math heading (0=+x, CCW) → compass (0=N, CW), assuming +y=north."""
    return (90.0 - np.degrees(heading_rad)) % 360.0


def heading_to_true_bearing(heading_rad, sensor="south"):
    """Sensor-frame math heading (0=+x, CCW) → TRUE compass bearing (0=N, CW).

    Exact path: project the point and a point 1 m ahead to lat/lon and take the
    geodesic azimuth (accounts for meridian convergence). Falls back to the sensor→
    hd_map_origin rotation, then to the naive +y=north assumption.
    """
    if has_exact_georef(sensor):
        # need an anchor point; use the scene-ish origin direction at (0,0)
        return bearing_at(0.0, 0.0, heading_rad, sensor)
    M = sensor_to_map_transform(sensor)
    if M is None:
        return _naive_compass(heading_rad)
    v = M[:3, :3] @ np.array([np.cos(heading_rad), np.sin(heading_rad), 0.0])
    return float((90.0 - np.degrees(np.arctan2(v[1], v[0]))) % 360.0)


def bearing_at(x, y, heading_rad, sensor="south"):
    """TRUE compass bearing of `heading_rad` evaluated at sensor-frame point (x, y).
    Exact (geodesic) when the HD-map chain is available, else the rotation/naive
    fallback (position-independent)."""
    g = _geod()
    if has_exact_georef(sensor) and g is not None:
        p0 = sensor_xy_to_latlon(x, y, sensor)
        p1 = sensor_xy_to_latlon(x + math.cos(heading_rad), y + math.sin(heading_rad), sensor)
        if p0 and p1:
            az, _, _ = g.inv(p0[1], p0[0], p1[1], p1[0])   # forward azimuth, 0=N, CW
            return float(az % 360.0)
    return heading_to_true_bearing(heading_rad, sensor)


def _enu_offset_latlon(east_m, north_m, center):
    """Flat-earth: ENU metre offset from a centre (lat, lon) → (lat, lon)."""
    lat0, lon0 = center
    lat = lat0 + north_m / 111320.0
    lon = lon0 + east_m / (111320.0 * math.cos(math.radians(lat0)))
    return lat, lon


def make_projector(sensor="south", ref_points_xy=None, center=None):
    """Build `proj(x, y) -> (lat, lon)` for map display, plus a bool `.exact`.

    - EXACT: HD-map geodetic chain → survey-grade absolute WGS84.
    - APPROX: scene map-frame metres centred on `center` (default site centroid) using
      `ref_points_xy`'s centroid as local origin — correct shape + orientation, only
      the absolute position approximate.
    """
    if has_exact_georef(sensor):
        def proj(x, y):
            ll = sensor_xy_to_latlon(x, y, sensor)
            return ll if ll is not None else SITE_LATLON_APPROX
        proj.exact = True
        return proj

    # ---- approximate fallback ----
    M = sensor_to_map_transform(sensor)
    c = tuple(center) if center else SITE_LATLON_APPROX
    ref = np.zeros(2)
    if M is not None and ref_points_xy is not None and len(ref_points_xy):
        rp = np.asarray(ref_points_xy, dtype=float)
        hom = np.column_stack([rp[:, 0], rp[:, 1], np.zeros(len(rp)), np.ones(len(rp))])
        ref = (M @ hom.T).T[:, :2].mean(axis=0)

    def proj(x, y):
        if M is None:
            return _enu_offset_latlon(float(x), float(y), c)
        p = M @ np.array([float(x), float(y), 0.0, 1.0])
        return _enu_offset_latlon(p[0] - ref[0], p[1] - ref[1], c)

    proj.exact = False
    return proj


def has_georef(sensor="south"):
    """Any georef at all (exact or approximate)."""
    return has_exact_georef(sensor) or sensor_to_map_transform(sensor) is not None


def sensor_position_latlon(sensor="south"):
    """(lat, lon) of a LiDAR station (its frame origin) on the gantry. None if no
    exact georef."""
    return sensor_xy_to_latlon(0.0, 0.0, sensor)


def sensor_points_to_latlon(xy, sensor="south"):
    """Batch: point-cloud-frame (N,2) array → (N,2) [lat, lon] via the HD-map anchor
    (inverse dev-kit recipe). None if the HD map / pyproj are unavailable."""
    gr = _hdmap_georef()
    if gr is None:
        return None
    tr = _transformer(gr[0])
    if tr is None:
        return None
    xy = np.asarray(xy, dtype=float)
    cloud = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])
    s110 = cloud @ _rz(_S110_TO_CLOUD_RZ_DEG).T + _S110_TO_CLOUD_T   # cloud → s110_base
    mp = (s110 - _MAP2BASE_T) @ _MAP2BASE_R                          # s110_base → HD-map
    lon, lat = tr.transform(gr[1][0] + mp[:, 0], gr[1][1] + mp[:, 1])
    return np.column_stack([lat, lon])


def camera_fovs_latlon(sensor="south", fov_deg=68.8, length_m=50.0):
    """Per south camera: {name, pos:(lat,lon), bearing, cone:[[lon,lat],...]}. Reads the
    OpenLABEL camera poses (relative to the LiDAR), runs them through the exact georef,
    and builds an FOV wedge along the optical axis. [] if unavailable."""
    cs = _read_coord_systems(sensor)
    g = _geod()
    if cs is None or g is None or not has_exact_georef(sensor):
        return []
    out = []
    for cam in ("s110_camera_basler_south1_8mm", "s110_camera_basler_south2_8mm"):
        node = cs.get(cam)
        if not node or "pose_wrt_parent" not in node:
            continue
        M = np.array(node["pose_wrt_parent"]["matrix4x4"], dtype=float).reshape(4, 4)
        pos = M[:3, 3]                                   # camera origin in cloud frame
        fxy = (M[:3, :3] @ np.array([0.0, 0.0, 1.0]))[:2]   # optical axis (+z) on the ground
        nrm = np.linalg.norm(fxy)
        if nrm < 1e-6:
            continue
        fxy /= nrm
        p0 = sensor_xy_to_latlon(pos[0], pos[1], sensor)
        p1 = sensor_xy_to_latlon(pos[0] + fxy[0], pos[1] + fxy[1], sensor)
        if p0 is None or p1 is None:
            continue
        bearing = g.inv(p0[1], p0[0], p1[1], p1[0])[0]
        cone = [[p0[1], p0[0]]]
        for a in (bearing - fov_deg / 2.0, bearing + fov_deg / 2.0):
            lon2, lat2, _ = g.fwd(p0[1], p0[0], a, length_m)
            cone.append([lon2, lat2])
        cone.append([p0[1], p0[0]])
        out.append({"name": cam.replace("s110_camera_basler_", "").replace("_8mm", ""),
                    "pos": p0, "bearing": float(bearing % 360.0), "cone": cone})
    return out


def circle_latlon(center_latlon, radius_m, n=72):
    """A closed ring of [lon, lat] points at radius_m around center (for FOV/range
    rings on the map)."""
    out = []
    for i in range(n + 1):
        th = 2.0 * math.pi * i / n
        lat, lon = _enu_offset_latlon(radius_m * math.cos(th), radius_m * math.sin(th),
                                      center_latlon)
        out.append([lon, lat])
    return out


@functools.lru_cache(maxsize=1)
def _hdmap_lanes_raw():
    """Every HD-map lane centerline as a map-frame np.ndarray([[x, y, z], ...]) (lanes
    with ≥2 samples). Parses the 48 MB file once (cached). [] if unavailable. Needs
    only the file — no pyproj."""
    path = next((p for p in _HDMAP_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return []
    try:
        with open(path, "r") as f:
            d = json.load(f)
    except Exception:
        return []
    out = []
    for road in d.get("roads", []):
        for ls in road.get("laneSections", []):
            for ln in ls.get("lanes", []):
                s = ln.get("samples") or []
                if len(s) >= 2:
                    out.append(np.asarray(s, dtype=float)[:, :3])
    return out


@functools.lru_cache(maxsize=1)
def _hdmap_lanes_latlon():
    """Every HD-map lane centerline as a lat/lon polyline. Returns a list of
    np.ndarray([[lat, lon], ...]), or []. Batch-projects all samples to WGS84 once."""
    gr = _hdmap_georef()
    lanes = _hdmap_lanes_raw()
    if gr is None or not lanes:
        return []
    tr = _transformer(gr[0])
    if tr is None:
        return []
    origin = gr[1]
    bounds, n = [], 0
    for arr in lanes:
        bounds.append((n, n + len(arr)))
        n += len(arr)
    allp = np.vstack([a[:, :2] for a in lanes])
    lon, lat = tr.transform(origin[0] + allp[:, 0], origin[1] + allp[:, 1])
    latlon = np.column_stack([lat, lon])
    return [latlon[a:b] for a, b in bounds]


@functools.lru_cache(maxsize=4)
def hdmap_lanes_sensor_frame(sensor="south", radius_m=130.0):
    """HD-map lane centerlines in the point-cloud frame (metres), as polylines
    [[x, y], ...] clipped to radius_m around the origin. For BEV digital-twin
    overlays. Needs only the HD-map file (no pyproj / no labels). [] if unavailable.

    Uses the dev-kit's exact recipe: map sample → s110_base (map→base) → cloud
    (subtract _S110_TO_CLOUD_T, then `@ Rz(77.8°)`). Valid for the south / registered
    (south-anchored) clouds the app renders.
    """
    lanes = _hdmap_lanes_raw()
    if not lanes:
        return []
    Rz = _rz(_S110_TO_CLOUD_RZ_DEG)
    r2 = radius_m * radius_m
    out = []
    for arr in lanes:
        s110 = (_MAP2BASE_R @ arr.T).T + _MAP2BASE_T          # map → s110_base
        cloud = (s110 - _S110_TO_CLOUD_T) @ Rz               # s110_base → point cloud
        xy = cloud[:, :2]
        mask = (xy[:, 0] ** 2 + xy[:, 1] ** 2) < r2
        if int(mask.sum()) >= 2:
            out.append(xy[mask].tolist())
    return out


def hdmap_paths_near(center_latlon, radius_m=130.0):
    """HD-map lane centerlines near `center_latlon`, as polylines [[lon, lat], ...] for
    pydeck. Each lane is clipped to the bbox of the given radius (≥2 pts kept). []
    if the HD map is unavailable."""
    lanes = _hdmap_lanes_latlon()
    if not lanes:
        return []
    clat, clon = center_latlon
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * math.cos(math.radians(clat)))
    out = []
    for arr in lanes:
        mask = (np.abs(arr[:, 0] - clat) < dlat) & (np.abs(arr[:, 1] - clon) < dlon)
        if int(mask.sum()) < 2:
            continue
        out.append([[float(lo), float(la)] for la, lo in arr[mask]])
    return out
