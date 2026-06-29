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


def _active_lanes_path():
    """Active dataset's lanes.geojson (falls back to the top-level config/)."""
    try:
        import dataset_manager as dm
        return dm.get_active().lanes_path
    except Exception:
        return DEFAULT_LANES_PATH

DEFAULT_PARAMS = {
    "angle_thresh_deg": 120.0,   # how far against the flow counts as wrong-way
    "min_speed": 2.0,            # m/s; below this, heading is unreliable
    "min_frames": 5,             # consecutive flagged frames required
    "min_displacement_m": 3.0,   # net travel over the flagged span
    "exempt_junction": True,     # skip frames inside 2+ overlapping lanes (turns)
    "min_heading_consistency": 0.85,  # resultant length of flagged headings (turn reject)
}


def load_lane_config(path: str = None):
    """Load lane regions + expected headings from a GeoJSON FeatureCollection.
    Defaults to the active dataset's lanes.geojson.

    Returns a list of dicts: {lane_id, name, heading_deg, calibrated, polygon}.
    Returns an empty list (WWD effectively disabled) if the file is missing.
    """
    if path is None:
        path = _active_lanes_path()
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


def _lanes_containing(lanes, x: float, y: float):
    """All lanes whose box contains the point (2+ => ambiguous junction/overlap)."""
    pt = np.array([[x, y]], dtype=np.float64)
    return [lane for lane in lanes if points_in_polygon(lane["polygon"], pt)[0]]


def _resultant_length(headings_rad):
    """Circular concentration R in [0,1]: ~1 = steady heading, low = rotating/turning."""
    if not headings_rad:
        return 0.0
    c = np.cos(headings_rad).mean()
    s = np.sin(headings_rad).mean()
    return float(np.hypot(c, s))


def _lane_for_point(lanes, x: float, y: float):
    pt = np.array([[x, y]], dtype=np.float64)
    for lane in lanes:
        if points_in_polygon(lane["polygon"], pt)[0]:
            return lane
    return None


def _runs(frames):
    """All maximal runs of consecutive integers in a sorted unique list, as a list
    of (start, end, length)."""
    if not frames:
        return []
    out = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            out.append((start, prev, prev - start + 1))
            start = prev = f
    out.append((start, prev, prev - start + 1))
    return out


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
    exempt_junction = bool(p.get("exempt_junction", True))
    min_consistency = float(p.get("min_heading_consistency", 0.0))

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
            here = _lanes_containing(lanes, d['cx'], d['cy'])
            # In the junction (2+ overlapping lanes) the legal direction is
            # ambiguous and turns are normal, so don't judge wrong-way there.
            if not here or (exempt_junction and len(here) >= 2):
                continue
            lane = here[0]
            diff = angular_diff_deg(np.degrees(hdg), lane['heading_deg'])
            if diff >= angle_thresh:
                flagged.append(fi)
                flagged_meta[fi] = (diff, lane['lane_id'], d, float(hdg))

        flagged_sorted = sorted(flagged)

        # Evaluate EVERY consecutive run of flagged frames (the junction exemption
        # can split one wrong-way pass into several runs). A run is a genuine
        # wrong-way pass only when it is long enough, travels far enough, holds a
        # STEADY heading, AND its NET travel direction itself opposes the lane.
        # That last test is what rejects turning cars: a turn's net displacement
        # curves diagonally, so it never opposes the lane by `angle_thresh`, even
        # though a few instantaneous headings during the turn might.
        qualifying = []   # (start, end, length, disp, consistency, max_angle, lane_id, frames)
        for rs, re_, rl in _runs(flagged_sorted):
            if rl < min_frames:
                continue
            rframes = [f for f in flagged_sorted if rs <= f <= re_]
            p0, p1 = pos_by_frame.get(rs), pos_by_frame.get(re_)
            if p0 is None or p1 is None:
                continue
            dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
            disp = float(np.hypot(dx, dy))
            consistency = _resultant_length([flagged_meta[f][3] for f in rframes])
            lane_counts = defaultdict(int)
            for f in rframes:
                lane_counts[flagged_meta[f][1]] += 1
            r_lane_id = max(lane_counts, key=lane_counts.get)
            r_lane = next((l for l in lanes if l["lane_id"] == r_lane_id), None)
            disp_opposes = (disp > 1e-6 and r_lane is not None and
                            angular_diff_deg(np.degrees(np.arctan2(dy, dx)),
                                             r_lane["heading_deg"]) >= angle_thresh)
            if disp >= min_disp and consistency >= min_consistency and disp_opposes:
                r_max = max(flagged_meta[f][0] for f in rframes)
                qualifying.append((rs, re_, rl, disp, consistency, r_max, r_lane_id, set(rframes)))

        is_ww = bool(qualifying)
        first_flag = qualifying[0][0] if is_ww else None          # EARLIEST qualifying run
        run_frames = set().union(*[q[7] for q in qualifying]) if qualifying else set()
        max_angle = max((q[5] for q in qualifying), default=0.0)
        run_len = max((q[2] for q in qualifying), default=0)
        disp = max((q[3] for q in qualifying), default=0.0)
        consistency = max((q[4] for q in qualifying), default=0.0)
        lane_id = None
        if qualifying:
            lane_counts = defaultdict(int)
            for q in qualifying:
                lane_counts[q[6]] += len(q[7])
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
            "heading_consistency": consistency,
            "flagged_frames": run_frames,
            "first_flag_frame": first_flag,
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
