"""Centralised site geometry + fast point-in-polygon helpers.

This module is the single source of truth for the scene geometry that used to be
hard-coded (twice) inside ``bg_filter_core.py`` and ``detection_logic.py``:

  * the research polygon (overall area of interest),
  * the road polygon(s),
  * the foreground exclusion rectangles,
  * the coarse 5x5 grid size.

Geometry is loaded from ``config/site_geometry.json`` so the pipeline can be
retargeted to a new intersection by editing JSON only. If the file is missing or
unreadable, the original TUMTraf ``s110_ouster_south`` values are used as a
fallback, so existing behaviour is preserved.

It also exposes ``points_in_polygon`` / ``points_in_prepared`` which replace the
per-point Python ``shapely.Point`` loops that were the main runtime bottleneck.
"""
import json
import os

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "site_geometry.json")

# --- Hard-coded fallback (original TUMTraf s110_ouster_south values) ----------
_FALLBACK = {
    "research_polygon": [[-25.0, -50.0], [45.0, -50.0], [45.0, 50.0], [-25.0, 50.0]],
    "road_polygons": [
        [[-25.0, -20.0], [-25.0, 10.0], [45.0, 20.0], [45.0, -15.0]],
        [[0.0, 0.0], [10.0, 50.0], [35.0, 50.0], [25.0, 0.0]],
        [[0.0, 0.0], [30.0, 0.0], [40.0, -50.0], [10.0, -50.0]],
    ],
    "foreground_exclusion_rects": [
        [[16.0, -22.0], [19.0, -22.0], [19.0, -19.0], [16.0, -19.0]],
        [[28.0, 0.0], [42.0, 0.0], [42.0, 5.0], [28.0, 5.0]],
        [[40.0, 15.0], [44.0, 15.0], [44.0, 20.0], [40.0, 20.0]],
        [[4.0, 18.0], [8.0, 18.0], [8.0, 23.0], [4.0, 23.0]],
        [[14.0, -15.0], [17.0, -15.0], [17.0, -12.0], [14.0, -12.0]],
    ],
    "coarse_grid": {"NX": 5, "NY": 5},
}


def _load_config():
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Merge so a partial config still works.
        merged = dict(_FALLBACK)
        merged.update({k: v for k, v in cfg.items() if not k.startswith("_")})
        return merged
    except Exception:
        return dict(_FALLBACK)


_CFG = _load_config()


def get_research_polygon() -> Polygon:
    return Polygon(_CFG["research_polygon"])


def get_road_polygon():
    polys = [Polygon(ring).buffer(0) for ring in _CFG["road_polygons"]]
    return unary_union(polys)


def get_fg_exclusion_rects():
    return [Polygon(r) for r in _CFG["foreground_exclusion_rects"]]


def get_coarse_grid():
    g = _CFG.get("coarse_grid", {"NX": 5, "NY": 5})
    return int(g["NX"]), int(g["NY"])


# --- Vectorised point-in-polygon ---------------------------------------------

def points_in_polygon(poly, xy: np.ndarray) -> np.ndarray:
    """Boolean mask of which (N, 2) points fall inside ``poly``.

    Uses shapely 2.x's vectorised ``contains_xy`` (C-level, GEOS) instead of a
    Python loop over ``shapely.Point`` objects — typically 10-100x faster on the
    tens of thousands of points in a LiDAR frame.
    """
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    return np.asarray(contains_xy(poly, xy[:, 0], xy[:, 1]), dtype=bool)


# ``contains_xy`` already accepts a prepared geometry implicitly via GEOS, so a
# single entry point is enough. Kept as an alias for call-site readability.
points_in_prepared = points_in_polygon
