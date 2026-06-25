"""Georeferencing for the TUMTraf s110 intersection.

The dataset is recorded at the Providentia++ / TUMTraf Intersection test field on
the B471 (Schleißheimer Straße) in Garching-Hochbrück, just north of Munich, DE.
Every OpenLABEL label file carries a coordinate_systems tree:

    hd_map_origin            (georeferenced map frame; +x≈UTM east, +y≈UTM north)
      └─ s110_base           (pose_wrt_parent: rigid 4x4)
           └─ s110_lidar_ouster_south / _north   (the sensor frame the app works in)

Tier 1 (exact, no external data): compose the sensor→base→map rotations to turn a
sensor-frame heading into a bearing in the georeferenced map frame — replacing the
old "+y = north (unverified)" assumption with the real ~82° sensor rotation.

Tier 2 (exact lat/lon): additionally needs the published UTM origin of
hd_map_origin (HD_MAP_ORIGIN_UTM). Until that one constant is filled in, the
lat/lon helpers return None and callers fall back to the approximate site centroid.
"""
import functools
import glob
import json
import math
import os

import numpy as np

# Human-readable site, for dashboard/UI display.
SITE_NAME = "TUMTraf s110 intersection — B471, Garching-Hochbrück (Munich), DE"
# Rough centroid of the intersection, ONLY a display fallback for the map pin until
# the exact UTM anchor below is provided (do not treat as survey-grade).
SITE_LATLON_APPROX = (48.2486, 11.6250)

# --- Tier 2 anchor -----------------------------------------------------------
# UTM zone 32N (easting, northing) of the hd_map_origin frame, as published in the
# TUMTraf release / HD map. Fill this in to enable exact per-driver lat/lon.
HD_MAP_ORIGIN_UTM = None          # e.g. (692000.0, 5_339_000.0)
HD_MAP_ORIGIN_UTM_EPSG = "EPSG:32632"  # WGS84 / UTM zone 32N


def _read_coord_systems(sensor="south"):
    """The OpenLABEL coordinate_systems block from the first raw label file of the
    active dataset (calibration is static across frames). None if unavailable."""
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


def _compose_to_map(cs, sensor):
    """4x4 transform mapping sensor-frame homogeneous coords → hd_map_origin frame,
    by walking pose_wrt_parent up the chain. None if the chain is missing."""
    key = f"s110_lidar_ouster_{sensor}"
    if cs is None or key not in cs:
        return None
    M = np.eye(4)
    node = key
    # pose_wrt_parent maps child→parent, so composing M = M_parent @ M_child up the
    # chain (sensor → base → map) yields sensor→map.
    while node and node in cs and cs[node].get("parent"):
        m = cs[node].get("pose_wrt_parent", {}).get("matrix4x4")
        if m:
            M = np.array(m, dtype=float).reshape(4, 4) @ M
        node = cs[node]["parent"]
    return M


@functools.lru_cache(maxsize=4)
def sensor_to_map_transform(sensor="south"):
    """Cached 4x4 sensor→hd_map_origin transform (None if no labels)."""
    return _compose_to_map(_read_coord_systems(sensor), sensor)


def has_georef(sensor="south"):
    return sensor_to_map_transform(sensor) is not None


def _naive_compass(heading_rad):
    """Old fallback: math heading (0=+x/east, CCW) → compass (0=N, CW), assuming
    the sensor +y axis is north. Used only when the transform chain is absent."""
    return (90.0 - np.degrees(heading_rad)) % 360.0


def heading_to_true_bearing(heading_rad, sensor="south"):
    """Sensor-frame math heading (0=+x, CCW) → true compass bearing (0=N, CW) in the
    georeferenced map frame. Exact from the OpenLABEL chain; falls back to the naive
    +y=north assumption only if the dataset has no labels."""
    M = sensor_to_map_transform(sensor)
    if M is None:
        return _naive_compass(heading_rad)
    R = M[:3, :3]
    v = R @ np.array([np.cos(heading_rad), np.sin(heading_rad), 0.0])
    # map frame: +x≈east, +y≈north → bearing measured from north, clockwise.
    return float((90.0 - np.degrees(np.arctan2(v[1], v[0]))) % 360.0)


def sensor_xy_to_map(x, y, sensor="south"):
    """(x,y) in the sensor frame → (east, north) metres in hd_map_origin. None if no
    chain. (Map-frame metres; add HD_MAP_ORIGIN_UTM for absolute UTM/lat-lon.)"""
    M = sensor_to_map_transform(sensor)
    if M is None:
        return None
    p = M @ np.array([float(x), float(y), 0.0, 1.0])
    return float(p[0]), float(p[1])


def sensor_xy_to_latlon(x, y, sensor="south"):
    """Exact WGS84 (lat, lon) for a sensor-frame (x,y). Requires HD_MAP_ORIGIN_UTM
    (Tier 2) and pyproj; returns None until that anchor is provided."""
    if HD_MAP_ORIGIN_UTM is None:
        return None
    mp = sensor_xy_to_map(x, y, sensor)
    if mp is None:
        return None
    try:
        from pyproj import Transformer
    except ImportError:
        return None
    e0, n0 = HD_MAP_ORIGIN_UTM
    tr = Transformer.from_crs(HD_MAP_ORIGIN_UTM_EPSG, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(e0 + mp[0], n0 + mp[1])
    return float(lat), float(lon)


def _enu_offset_latlon(east_m, north_m, center):
    """Flat-earth: ENU metre offset from a centre (lat, lon) → (lat, lon)."""
    lat0, lon0 = center
    lat = lat0 + north_m / 111320.0
    lon = lon0 + east_m / (111320.0 * math.cos(math.radians(lat0)))
    return lat, lon


def make_projector(sensor="south", ref_points_xy=None, center=None):
    """Build a function mapping a sensor-frame (x, y) → (lat, lon) for map display.

    - If the Tier-2 anchor (HD_MAP_ORIGIN_UTM) is set → exact WGS84 (survey-grade,
      absolutely positioned).
    - Otherwise → APPROXIMATE placement: the scene's map-frame metres are centred on
      `center` (default the site centroid) using `ref_points_xy`'s centroid as local
      origin. Shape + orientation are correct (from the real sensor→map rotation);
      only the absolute position is approximate until the anchor is provided.

    Returns a `proj(x, y) -> (lat, lon)` closure, plus a bool `exact`.
    """
    M = sensor_to_map_transform(sensor)
    c = tuple(center) if center else SITE_LATLON_APPROX
    exact = HD_MAP_ORIGIN_UTM is not None and M is not None
    ref = np.zeros(2)
    if M is not None and ref_points_xy is not None and len(ref_points_xy):
        rp = np.asarray(ref_points_xy, dtype=float)
        hom = np.column_stack([rp[:, 0], rp[:, 1], np.zeros(len(rp)), np.ones(len(rp))])
        ref = (M @ hom.T).T[:, :2].mean(axis=0)

    def proj(x, y):
        if exact:
            ll = sensor_xy_to_latlon(x, y, sensor)
            if ll is not None:
                return ll
        if M is None:                       # no georef at all (last-resort scatter)
            return _enu_offset_latlon(float(x), float(y), c)
        p = M @ np.array([float(x), float(y), 0.0, 1.0])
        return _enu_offset_latlon(p[0] - ref[0], p[1] - ref[1], c)

    proj.exact = exact
    return proj
