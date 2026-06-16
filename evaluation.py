"""Quantitative evaluation of detection/tracking against ground-truth cuboids.

Computes the standard metrics a reviewer will expect:

  * Detection quality: precision, recall, F1 (per-frame BEV centre matching).
  * Tracking quality: MOTA, MOTP and ID-switch count (CLEAR-MOT style).

Matching is done in bird's-eye-view: each predicted box is matched to a GT box
by 2-D centre distance using the Hungarian algorithm, gated at ``match_dist_m``.
This keeps the implementation dependency-light while remaining meaningful for
WWD work, where BEV position and identity are what matter.

Ground truth is read from the OpenLABEL JSON files in the GT directory and
aligned to detection frames by the leading ``<timestamp1>_<timestamp2>`` token
shared by the .json and .pcd filenames.
"""
import json
import glob
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def _frame_key(path: str) -> str:
    """Leading timestamp key shared by GT .json and .pcd filenames."""
    base = os.path.basename(path)
    toks = base.split('_')
    return '_'.join(toks[:2]) if len(toks) >= 2 else os.path.splitext(base)[0]


def _quat_yaw(qx, qy, qz, qw):
    """Yaw (rotation about Z) from a quaternion."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def parse_gt_frame(json_path: str):
    """Return GT boxes for a single OpenLABEL frame.

    Each box: dict(gid, cls, cx, cy, l, w, yaw). gid is the persistent GT object
    id (used for ID-switch counting). val = [cx,cy,cz, qx,qy,qz,qw, l,w,h]."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    boxes = []
    try:
        frames = data['openlabel']['frames']
        frame_obj = frames[next(iter(frames))]
        for gid, od in frame_obj.get('objects', {}).items():
            obj_data = od.get('object_data', {})
            val = obj_data.get('cuboid', {}).get('val')
            cls = obj_data.get('type', obj_data.get('name', 'Unknown'))
            if isinstance(val, list) and len(val) >= 10:
                boxes.append(dict(gid=gid, cls=cls,
                                  cx=float(val[0]), cy=float(val[1]),
                                  yaw=_quat_yaw(val[3], val[4], val[5], val[6]),
                                  l=float(val[7]), w=float(val[8])))
    except Exception:
        pass
    return boxes


def load_gt_by_key(gt_dir: str):
    """Map each GT frame's timestamp key -> list of GT boxes (for visual eval)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(gt_dir, '*.json'))):
        out[_frame_key(f)] = parse_gt_frame(f)
    return out


def _match_frame(pred, gt, match_dist):
    """Hungarian BEV-centre matching of one frame.

    Returns (matches, n_fp, n_fn, dist_sum) where matches is a list of
    (pred_index, gt_index, distance)."""
    if not pred or not gt:
        return [], len(pred), len(gt), 0.0
    cost = np.zeros((len(pred), len(gt)), dtype=np.float64)
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            cost[i, j] = np.hypot(p['cx'] - g['cx'], p['cy'] - g['cy'])
    row, col = linear_sum_assignment(cost)
    matches, dist_sum = [], 0.0
    matched_pred, matched_gt = set(), set()
    for i, j in zip(row, col):
        if cost[i, j] <= match_dist:
            matches.append((i, j, float(cost[i, j])))
            dist_sum += float(cost[i, j])
            matched_pred.add(i); matched_gt.add(j)
    n_fp = len(pred) - len(matched_pred)
    n_fn = len(gt) - len(matched_gt)
    return matches, n_fp, n_fn, dist_sum


def evaluate(det_frames, pred_frame_paths, gt_dir, match_dist=2.0, classes=None, roi_bounds=None):
    """Evaluate detection + tracking against GT.

    Parameters
    ----------
    det_frames : list[list[dict]]   tracked detections (tid, cx, cy, ...)
    pred_frame_paths : list[str]    pcd path per detection frame (for alignment)
    gt_dir : str                    directory of OpenLABEL .json files
    match_dist : float              BEV centre gate (m)
    classes : set[str] | None       if given, only score GT of these classes
                                     (upper-case, e.g. {'CAR','TRUCK','VAN','BUS'}).
                                     GT of OTHER classes (in ROI) become "ignore":
                                     a detection matching one is neither TP nor FP,
                                     so the detector isn't penalised for correctly
                                     finding a pedestrian/bicycle that we simply
                                     aren't scoring.

    Returns a report dict.
    """
    cls_filter = {c.upper() for c in classes} if classes else None

    def _in_roi(b):
        if roi_bounds is None:
            return True
        x0, x1, y0, y1 = roi_bounds
        return (x0 <= b['cx'] <= x1) and (y0 <= b['cy'] <= y1)

    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.json')))
    gt_by_key = {}       # scored GT (the classes we evaluate)
    ignore_by_key = {}   # in-ROI GT of non-scored classes -> don't-care
    for f in gt_files:
        k = _frame_key(f)
        scored, ignore = [], []
        for b in parse_gt_frame(f):
            if not _in_roi(b):
                continue
            if cls_filter is None or str(b['cls']).upper() in cls_filter:
                scored.append(b)
            else:
                ignore.append(b)
        gt_by_key[k] = scored
        if ignore:
            ignore_by_key[k] = ignore
    # Filter predictions to the same ROI so the comparison is apples-to-apples.
    if roi_bounds is not None:
        x0, x1, y0, y1 = roi_bounds
        det_frames = [[d for d in dets if x0 <= d['cx'] <= x1 and y0 <= d['cy'] <= y1]
                      for dets in det_frames]

    total_tp = total_fp = total_fn = total_idsw = 0
    total_gt = 0
    dist_sum = 0.0
    n_matched = 0
    evaluated_frames = 0
    last_gid_for_tid = {}   # tid -> last GT gid it matched (for ID switches)

    per_frame = []
    for fi, dets in enumerate(det_frames):
        if fi >= len(pred_frame_paths):
            break
        key = _frame_key(pred_frame_paths[fi])
        if key not in gt_by_key:
            continue  # no GT for this frame; skip rather than penalise
        gt = gt_by_key[key]
        evaluated_frames += 1
        total_gt += len(gt)

        matches, n_fp, n_fn, ds = _match_frame(dets, gt, match_dist)
        tp = len(matches)
        # Detections that didn't match a scored GT but DO match an ignore-class
        # GT (e.g. a real pedestrian when scoring vehicles only) are not FPs.
        ign = ignore_by_key.get(key)
        if ign and n_fp > 0:
            matched_di = {di for di, _, _ in matches}
            unmatched = [dets[di] for di in range(len(dets)) if di not in matched_di]
            if unmatched:
                im, _, _, _ = _match_frame(unmatched, ign, match_dist)
                n_fp = max(0, n_fp - len(im))
        total_tp += tp; total_fp += n_fp; total_fn += n_fn
        dist_sum += ds; n_matched += tp

        # ID switches: a tracked id now matches a different GT object than before
        idsw = 0
        for pi, gj, _ in matches:
            tid = dets[pi].get('tid')
            gid = gt[gj]['gid']
            if tid in last_gid_for_tid and last_gid_for_tid[tid] != gid:
                idsw += 1
            last_gid_for_tid[tid] = gid
        total_idsw += idsw

        prec = tp / (tp + n_fp) if (tp + n_fp) else 0.0
        rec = tp / (tp + n_fn) if (tp + n_fn) else 0.0
        per_frame.append(dict(frame=fi, tp=tp, fp=n_fp, fn=n_fn, idsw=idsw,
                              precision=round(prec, 4), recall=round(rec, 4)))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mota = 1.0 - (total_fn + total_fp + total_idsw) / total_gt if total_gt else 0.0
    motp = (dist_sum / n_matched) if n_matched else 0.0

    return {
        "summary": {
            "evaluated_frames": evaluated_frames,
            "gt_objects_total": total_gt,
            "TP": total_tp, "FP": total_fp, "FN": total_fn, "ID_switches": total_idsw,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "MOTA": round(mota, 4),
            "MOTP_m": round(motp, 4),
            "match_dist_m": match_dist,
        },
        "per_frame": per_frame,
        "gt_frames_available": len(gt_files),
    }


def _box_corners(cx, cy, yaw, l, w):
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = l / 2.0, w / 2.0
    loc = [(dx, dy), (dx, -dy), (-dx, -dy), (-dx, dy), (dx, dy)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in loc]


def bev_figure(box_groups, title, x_range, y_range, height=620, bg_xy=None, uirev='eval'):
    """Top-down (bird's-eye) plot of oriented boxes in the sensor X/Y frame.
    All figures share identical axes/ranges so the LiDAR blind spot and objects
    line up. Pan/scroll-zoom enabled; uirevision keeps the view across frames.

    box_groups: list of (boxes, color) — later groups render on top. Each box:
    dict(cx, cy, yaw, l, w, [label]). bg_xy: optional (N,2) point cloud."""
    import plotly.graph_objects as go
    fig = go.Figure()
    if bg_xy is not None and len(bg_xy):
        fig.add_trace(go.Scatter(x=bg_xy[:, 0], y=bg_xy[:, 1], mode='markers',
                                 marker=dict(size=1.5, color='#666'),
                                 showlegend=False, hoverinfo='skip'))
    mx, my, mtext, mcol = [], [], [], []
    for boxes, color in box_groups:
        xs, ys = [], []
        for b in boxes:
            l = float(b.get('l') or 4.5)
            w = float(b.get('w') or 1.9)
            corners = _box_corners(float(b['cx']), float(b['cy']), float(b.get('yaw', 0.0)), l, w)
            xs.extend([p[0] for p in corners] + [None])
            ys.extend([p[1] for p in corners] + [None])
            mx.append(float(b['cx'])); my.append(float(b['cy']))
            mtext.append(str(b.get('label', ''))); mcol.append(color)
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                                     line=dict(color=color, width=2),
                                     showlegend=False, hoverinfo='skip'))
    if mx:
        fig.add_trace(go.Scatter(x=mx, y=my, mode='markers+text', text=mtext,
                                 textposition='top center', textfont=dict(size=9),
                                 marker=dict(size=4, color=mcol), showlegend=False,
                                 hoverinfo='skip'))
    fig.update_xaxes(range=list(x_range), title='X (m)', constrain='domain')
    fig.update_yaxes(range=list(y_range), title='Y (m)', scaleanchor='x', scaleratio=1)
    fig.update_layout(title=title, height=height, margin=dict(l=0, r=0, t=40, b=0),
                      plot_bgcolor='#111', dragmode='pan', uirevision=uirev)
    return fig


BEV_CONFIG = {'scrollZoom': True, 'displayModeBar': True,
              'modeBarButtonsToRemove': ['select2d', 'lasso2d']}


def save_report(report, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'evaluation_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    csv_path = os.path.join(out_dir, 'evaluation_per_frame.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('frame,tp,fp,fn,idsw,precision,recall\n')
        for r in report['per_frame']:
            f.write(f"{r['frame']},{r['tp']},{r['fp']},{r['fn']},{r['idsw']},{r['precision']},{r['recall']}\n")
    return json_path, csv_path
