"""Wrong-way driving (WWD) detection.

This is the research deliverable that was missing from the pipeline: the tracker
produced positions, velocities and trajectories, but nothing decided whether a
vehicle was driving against the legal direction of travel.

Key idea
--------
A bounding-box yaw derived from PCA is 180-degree ambiguous (a north-bound and a
south-bound car give the same axis), so it cannot, on its own, tell wrong-way
from right-way. Instead we use the *velocity* direction from the Kalman tracker
(``atan2(vy, vx)``), which is unambiguous once a vehicle is moving, and compare
it against an *expected* direction of travel defined per lane region in
``config/lanes.geojson``.

A track is flagged as wrong-way only when ALL of the following hold, which keeps
the false-alarm rate low (U-turns, jitter and parked vehicles are rejected):

  * its speed exceeds ``min_speed`` (m/s),
  * its velocity heading differs from the lane's expected heading by at least
    ``angle_thresh_deg`` degrees,
  * this holds for a run of at least ``min_frames`` consecutive frames,
  * the net displacement across the flagged span is at least
    ``min_displacement_m`` metres.
"""
import json
import os
from collections import defaultdict

import numpy as np
from shapely.geometry import shape

from geometry_config import points_in_polygon

DEFAULT_LANES_PATH = os.path.join(os.path.dirname(__file__), "config", "lanes.geojson")

DEFAULT_PARAMS = {
    "angle_thresh_deg": 120.0,   # how far against the flow counts as wrong-way
    "min_speed": 2.0,            # m/s; below this, heading is unreliable
    "min_frames": 5,             # consecutive flagged frames required
    "min_displacement_m": 3.0,   # net travel over the flagged span
}


def load_lane_config(path: str = DEFAULT_LANES_PATH):
    """Load lane regions + expected headings from a GeoJSON FeatureCollection.

    Returns a list of dicts: {lane_id, name, heading_deg, calibrated, polygon}.
    Returns an empty list (WWD effectively disabled) if the file is missing.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    lanes = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        try:
            poly = shape(feat["geometry"])
        except Exception:
            continue
        lanes.append({
            "lane_id": props.get("lane_id", f"lane_{len(lanes)}"),
            "name": props.get("name", ""),
            "heading_deg": float(props.get("heading_deg", 0.0)),
            "calibrated": bool(props.get("calibrated", False)),
            "polygon": poly,
        })
    return lanes


def lanes_calibrated(lanes) -> bool:
    """True only if every lane has been marked calibrated. Used to warn the user
    that WWD numbers from placeholder geometry are not trustworthy."""
    return bool(lanes) and all(l.get("calibrated", False) for l in lanes)


def angular_diff_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute angle between two headings, in [0, 180]."""
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def _lane_for_point(lanes, x: float, y: float):
    pt = np.array([[x, y]], dtype=np.float64)
    for lane in lanes:
        if points_in_polygon(lane["polygon"], pt)[0]:
            return lane
    return None


def _longest_consecutive_run(frames):
    """Length of the longest run of consecutive integers in a sorted unique list,
    plus the (start, end) frame numbers of that run."""
    if not frames:
        return 0, None, None
    best_len, best_start, best_end = 1, frames[0], frames[0]
    cur_len, cur_start = 1, frames[0]
    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1] + 1:
            cur_len += 1
        else:
            cur_len, cur_start = 1, frames[i]
        if cur_len > best_len:
            best_len, best_start, best_end = cur_len, cur_start, frames[i]
    return best_len, best_start, best_end


def detect_wrong_way(det_frames, lanes, params=None):
    """Analyse tracked detections and flag wrong-way tracks.

    Parameters
    ----------
    det_frames : list[list[dict]]
        Per-frame detections as produced by ``run_detection_and_tracking``;
        each dict must contain tid, cx, cy, speed and heading.
    lanes : list[dict]
        Output of ``load_lane_config``.
    params : dict, optional
        Overrides for ``DEFAULT_PARAMS``.

    Returns
    -------
    dict with keys:
        'tracks'        : {tid: {is_wrong_way, lane_id, max_angle_deg,
                                 run_len, flagged_frames(set), first_flag_frame}}
        'wrong_way_tids': set of tids flagged as wrong-way
        'frame_flags'   : list[set] per frame of wrong-way tids active that frame
    Also annotates each detection dict in-place with d['wrong_way'] (bool).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})

    angle_thresh = float(p["angle_thresh_deg"])
    min_speed = float(p["min_speed"])
    min_frames = int(p["min_frames"])
    min_disp = float(p["min_displacement_m"])

    # Group detections by track id, preserving frame order.
    series = defaultdict(list)
    for fi, dets in enumerate(det_frames):
        for d in dets:
            d['wrong_way'] = False  # reset / default
            if 'tid' in d:
                series[d['tid']].append((fi, d))

    track_results = {}
    if not lanes:
        return {"tracks": {}, "wrong_way_tids": set(),
                "frame_flags": [set() for _ in det_frames]}

    for tid, items in series.items():
        flagged = []          # frame indices where the instantaneous test passes
        flagged_meta = {}     # frame -> (angle, lane_id, det)
        pos_by_frame = {}
        for fi, d in items:
            pos_by_frame[fi] = (d['cx'], d['cy'])
            hdg = d.get('heading', None)
            speed = float(d.get('speed', 0.0))
            if hdg is None or speed < min_speed:
                continue
            lane = _lane_for_point(lanes, d['cx'], d['cy'])
            if lane is None:
                continue
            diff = angular_diff_deg(np.degrees(hdg), lane['heading_deg'])
            if diff >= angle_thresh:
                flagged.append(fi)
                flagged_meta[fi] = (diff, lane['lane_id'], d)

        flagged_sorted = sorted(flagged)
        run_len, run_start, run_end = _longest_consecutive_run(flagged_sorted)

        is_ww = False
        disp = 0.0
        if run_len >= min_frames and run_start is not None:
            p0 = pos_by_frame.get(run_start)
            p1 = pos_by_frame.get(run_end)
            if p0 is not None and p1 is not None:
                disp = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            is_ww = disp >= min_disp

        run_frames = set(f for f in flagged_sorted if run_start is not None and run_start <= f <= run_end)
        max_angle = max((flagged_meta[f][0] for f in run_frames), default=0.0)
        lane_id = None
        if run_frames:
            # most common lane over the flagged run
            lane_counts = defaultdict(int)
            for f in run_frames:
                lane_counts[flagged_meta[f][1]] += 1
            lane_id = max(lane_counts, key=lane_counts.get)

        if is_ww:
            for f in run_frames:
                flagged_meta[f][2]['wrong_way'] = True

        track_results[tid] = {
            "is_wrong_way": is_ww,
            "lane_id": lane_id,
            "max_angle_deg": max_angle,
            "run_len": run_len,
            "displacement_m": disp,
            "flagged_frames": run_frames,
            "first_flag_frame": run_start if is_ww else None,
        }

    wrong_way_tids = {tid for tid, r in track_results.items() if r["is_wrong_way"]}
    frame_flags = []
    for fi, dets in enumerate(det_frames):
        frame_flags.append({d['tid'] for d in dets if d.get('wrong_way')})

    return {"tracks": track_results, "wrong_way_tids": wrong_way_tids,
            "frame_flags": frame_flags}


def summarize_wrong_way(wwd_result, fps: float = 10.0):
    """Human-readable summary rows for the UI / report."""
    rows = []
    for tid, r in sorted(wwd_result["tracks"].items()):
        if not r["is_wrong_way"]:
            continue
        rows.append({
            "Track ID": tid,
            "Lane": r["lane_id"],
            "First frame": r["first_flag_frame"],
            "Approx. time (s)": round(r["first_flag_frame"] / fps, 2) if r["first_flag_frame"] is not None and fps else None,
            "Sustained frames": r["run_len"],
            "Max angle vs flow (deg)": round(r["max_angle_deg"], 1),
            "Displacement (m)": round(r["displacement_m"], 1),
        })
    return rows
