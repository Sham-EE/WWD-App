"""In-app port of the TUM Traffic dev-kit's 3D object-detection evaluation
(`src/eval/evaluation.py`), so the official per-class Precision / Recall / AP@0.1
metric runs straight from the Evaluation page — no separate repo, no numba-GPU /
pytorch3d / camera-calibration setup.

What's faithful to the dev-kit (copied verbatim): the AP machinery —
`get_thresholds`, `accumulate_scores`, `compute_statistics`, `filter_data`,
`overall_filter`, and `get_evaluation_results` (the p-r-curve → AP, superclass
collapsing, occurrence counts).

What's replaced:
- The 3D-IoU primitive `rotate_iou_cpu_eval` (dev-kit uses pytorch3d's
  `box3d_overlap`) → an equivalent shapely computation. For upright boxes (yaw
  only, no roll/pitch) 3D IoU = rotated-BEV-intersection × z-overlap / union,
  which is exactly what pytorch3d returns — so the numbers match.
- The box parser: the dev-kit's `load_lidar_boxes_into_s110` re-projects boxes
  through a camera into the s110_base ground plane (needs calibration). Here GT
  and predictions are already in the same LiDAR/registered frame, so we read the
  boxes directly — a consistent, camera-free comparison.

Box convention (matching the dev-kit): `[x, y, z, l, w, h, yaw]`.
"""
import glob
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon

# Per the dev-kit: AP@0.1 for every (super)class.
_SUPERCLASS_IOU = {"VEHICLE": 0.1, "PEDESTRIAN": 0.1, "BICYCLE": 0.1}
_CLASS_IOU = {c: 0.1 for c in ["CAR", "TRUCK", "TRAILER", "VAN", "MOTORCYCLE", "BUS",
                               "PEDESTRIAN", "BICYCLE", "EMERGENCY_VEHICLE", "OTHER"]}
_ALL_CLASSES = list(_CLASS_IOU)


# --------------------------------------------------------------------------- #
#  IoU — shapely replacement for the dev-kit's pytorch3d primitive
# --------------------------------------------------------------------------- #
def _rect(x, y, l, w, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = l / 2.0, w / 2.0
    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    return Polygon([(x + cx * c - cy * s, y + cx * s + cy * c) for cx, cy in corners])


def rotate_iou_cpu_eval(gt_boxes, pred_boxes):
    """[N,7] gt, [M,7] pred (x,y,z,l,w,h,yaw) → [N,M] 3D IoU (upright boxes)."""
    n, m = len(gt_boxes), len(pred_boxes)
    out = np.zeros((n, m), dtype=np.float64)
    if n == 0 or m == 0:
        return out
    g_rect = [_rect(b[0], b[1], b[3], b[4], b[6]) for b in gt_boxes]
    p_rect = [_rect(b[0], b[1], b[3], b[4], b[6]) for b in pred_boxes]
    g_zlo = gt_boxes[:, 2] - gt_boxes[:, 5] / 2.0; g_zhi = gt_boxes[:, 2] + gt_boxes[:, 5] / 2.0
    p_zlo = pred_boxes[:, 2] - pred_boxes[:, 5] / 2.0; p_zhi = pred_boxes[:, 2] + pred_boxes[:, 5] / 2.0
    g_vol = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]
    p_vol = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    for i in range(n):
        gi = g_rect[i]
        if gi.area <= 0:
            continue
        for j in range(m):
            inter2d = gi.intersection(p_rect[j]).area
            if inter2d <= 0:
                continue
            ih = min(g_zhi[i], p_zhi[j]) - max(g_zlo[i], p_zlo[j])
            if ih <= 0:
                continue
            inter3d = inter2d * ih
            union = g_vol[i] + p_vol[j] - inter3d
            if union > 0:
                out[i, j] = inter3d / union
    return out


def compute_iou3d_cpu(gt_annos, pred_annos):
    return [rotate_iou_cpu_eval(gt_annos[i]["boxes_3d"], pred_annos[i]["boxes_3d"])
            for i in range(len(gt_annos))]


# --------------------------------------------------------------------------- #
#  AP machinery — copied verbatim from the dev-kit (numba removed; pure numpy)
# --------------------------------------------------------------------------- #
def get_thresholds(scores, num_gt, num_pr_points):
    eps = 1e-6
    scores = np.sort(scores)[::-1]
    recall_level = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        r_recall = (i + 2) / num_gt if i < (len(scores) - 1) else l_recall
        if (r_recall + l_recall < 2 * recall_level) and i < (len(scores) - 1):
            continue
        thresholds.append(score)
        recall_level += 1 / num_pr_points
        while r_recall + l_recall + eps > 2 * recall_level:
            thresholds.append(score)
            recall_level += 1 / num_pr_points
    return thresholds


def accumulate_scores(gt_shapes, pred_shapes, iou, pred_scores, gt_flag, pred_flag, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    accum_scores = np.zeros(num_gt)
    accum_ious = np.zeros(num_gt)
    accum_pos_rmse = np.zeros(num_gt)
    accum_rot_rmse = np.zeros(num_gt)
    accum_idx = 0
    for gt_id, gt_shape in enumerate(gt_shapes):
        if gt_flag[gt_id] == -1:
            continue
        det_idx = -1
        detected_score = -1
        det_iou = -1
        det_pos_rmse = -1
        det_rot_rmse = -1
        for pred_id, pred_shape in enumerate(pred_shapes):
            if pred_flag[pred_id] == -1 or assigned[pred_id]:
                continue
            iou_ij = iou[gt_id, pred_id]
            pred_score = pred_scores[pred_id]
            if (iou_ij > iou_threshold) and (pred_score > detected_score) and (iou_ij > det_iou):
                det_idx = pred_id
                detected_score = pred_score
                det_iou = iou_ij
                det_pos_rmse = np.linalg.norm(gt_shape[:3] - pred_shape[:3])
                det_rot_rmse = abs(gt_shape[6] - pred_shape[6]) % math.pi
                if det_rot_rmse > math.pi * 0.5:
                    det_rot_rmse = det_rot_rmse - math.pi * 0.5
        if (detected_score == -1) and (gt_flag[gt_id] == 0):
            pass
        elif (detected_score != -1) and (gt_flag[gt_id] == 1 or pred_flag[det_idx] == 1):
            assigned[det_idx] = True
        elif detected_score != -1:
            accum_scores[accum_idx] = pred_scores[det_idx]
            accum_ious[accum_idx] = det_iou
            accum_pos_rmse[accum_idx] = det_pos_rmse
            accum_rot_rmse[accum_idx] = det_rot_rmse
            accum_idx += 1
            assigned[det_idx] = True
    return accum_scores[:accum_idx], accum_ious[:accum_idx], accum_pos_rmse[:accum_idx], accum_rot_rmse[:accum_idx]


def compute_statistics(iou, pred_scores, gt_flag, pred_flag, score_threshold, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    under_threshold = pred_scores < score_threshold
    tp, fp, fn = 0, 0, 0
    for i in range(num_gt):
        if gt_flag[i] == -1:
            continue
        det_idx = -1
        detected = False
        best_matched_iou = 0
        gt_assigned_to_ignore = False
        for j in range(num_pred):
            if pred_flag[j] == -1 or assigned[j] or under_threshold[j]:
                continue
            iou_ij = iou[i, j]
            if (iou_ij > iou_threshold) and (iou_ij > best_matched_iou or gt_assigned_to_ignore) and pred_flag[j] == 0:
                best_matched_iou = iou_ij
                det_idx = j
                detected = True
                gt_assigned_to_ignore = False
            elif (iou_ij > iou_threshold) and (not detected) and pred_flag[j] == 1:
                det_idx = j
                detected = True
                gt_assigned_to_ignore = True
        if (not detected) and gt_flag[i] == 0:
            fn += 1
        elif detected and (gt_flag[i] == 1 or pred_flag[det_idx] == 1):
            assigned[det_idx] = True
        elif detected:
            tp += 1
            assigned[det_idx] = True
    for j in range(num_pred):
        if not (assigned[j] or pred_flag[j] == -1 or pred_flag[j] == 1 or under_threshold[j]):
            fp += 1
    return tp, fp, fn


def overall_filter(boxes, level):
    ignore = np.ones(boxes.shape[0], dtype=bool)
    if len(boxes) == 0:
        return ignore
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))
    if level == 0:
        flag = dist < 64
    elif level == 1:
        flag = dist < 40
    elif level == 2:
        flag = (dist >= 40) & (dist < 50)
    else:
        flag = (dist >= 50) & (dist < 64)
    ignore[flag] = False
    return ignore


def filter_data(gt_anno, pred_anno, difficulty_level, class_name, use_superclass):
    num_gt = len(gt_anno["name"])
    gt_flag = np.zeros(num_gt, dtype=np.int64)
    if num_gt > 0:
        if use_superclass and class_name == "VEHICLE":
            reject = np.logical_or(gt_anno["name"] == "PEDESTRIAN",
                                   np.logical_or(gt_anno["name"] == "BICYCLE", gt_anno["name"] == "MOTORCYCLE"))
        elif use_superclass and class_name == "BICYCLE":
            reject = ~np.logical_or(gt_anno["name"] == "BICYCLE", gt_anno["name"] == "MOTORCYCLE")
        else:
            reject = gt_anno["name"] != class_name
        gt_flag[reject] = -1
    num_pred = len(pred_anno["name"])
    pred_flag = np.zeros(num_pred, dtype=np.int64)
    if num_pred > 0:
        if use_superclass and class_name == "VEHICLE":
            reject = np.logical_or(pred_anno["name"] == "PEDESTRIAN",
                                   np.logical_or(pred_anno["name"] == "BICYCLE", pred_anno["name"] == "MOTORCYCLE"))
        elif use_superclass and class_name == "BICYCLE":
            reject = ~np.logical_or(pred_anno["name"] == "BICYCLE", pred_anno["name"] == "MOTORCYCLE")
        else:
            reject = pred_anno["name"] != class_name
        pred_flag[reject] = -1
    gt_flag[overall_filter(gt_anno["boxes_3d"], difficulty_level)] = 1
    pred_flag[overall_filter(pred_anno["boxes_3d"], difficulty_level)] = 1
    return gt_flag, pred_flag


def get_evaluation_results(gt_annotation_frames, pred_annotation_frames, classes,
                           use_superclass=True, iou_thresholds=None, num_pr_points=50):
    if iou_thresholds is None:
        iou_thresholds = _SUPERCLASS_IOU if use_superclass else _CLASS_IOU
    assert len(gt_annotation_frames) == len(pred_annotation_frames)

    if use_superclass:
        nv = npd = nb = 0
        for g in gt_annotation_frames:
            for oc in g["name"]:
                u = str(oc).upper()
                nv += u in ["CAR", "TRUCK", "BUS", "TRAILER", "VAN", "EMERGENCY_VEHICLE", "OTHER"]
                npd += u == "PEDESTRIAN"
                nb += u in ["BICYCLE", "MOTORCYCLE"]
        classes = ([c for c, k in (("VEHICLE", nv), ("PEDESTRIAN", npd), ("BICYCLE", nb)) if k > 0])

    ious = compute_iou3d_cpu(gt_annotation_frames, pred_annotation_frames)
    num_classes = len(classes)
    num_difficulties = 4
    precision = np.zeros([num_classes, num_difficulties, num_pr_points + 1])
    recall = np.zeros([num_classes, num_difficulties, num_pr_points + 1])
    iou_3d = np.zeros([num_classes, num_difficulties])
    gt_occ = {c: 0 for c in classes}
    pred_occ = {c: 0 for c in classes}

    def _count(anno, occ):
        if anno["name"].size == 0:
            return
        npd = (anno["name"] == "PEDESTRIAN").sum()
        nb = np.logical_or(anno["name"] == "BICYCLE", anno["name"] == "MOTORCYCLE").sum()
        nv = len(anno["name"]) - npd - nb
        for c in classes:
            occ[c] += {"VEHICLE": nv, "BICYCLE": nb, "PEDESTRIAN": npd}.get(c.upper(),
                      (anno["name"] == c.upper()).sum())

    for g, p in zip(gt_annotation_frames, pred_annotation_frames):
        if use_superclass:
            _count(g, gt_occ); _count(p, pred_occ)
        else:
            for c in classes:
                gt_occ[c] += (g["name"] == c.upper()).sum() if g["name"].size else 0
                pred_occ[c] += (p["name"] == c.upper()).sum() if p["name"].size else 0

    num_samples = len(gt_annotation_frames)
    for cls_idx, cur_class in enumerate(classes):
        iou_threshold = iou_thresholds[cur_class.upper()]
        for diff_idx in range(num_difficulties):
            accum_scores, accum_ious, gt_flags, pred_flags = [], [], [], []
            num_valid_gt = 0
            for s in range(num_samples):
                g, p = gt_annotation_frames[s], pred_annotation_frames[s]
                iou = ious[s]
                gf, pf = filter_data(g, p, diff_idx, cur_class.upper(), use_superclass)
                gt_flags.append(gf); pred_flags.append(pf)
                num_valid_gt += int(np.sum(gf == 0))
                if iou.size > 0:
                    sc, io, _, _ = accumulate_scores(g["boxes_3d"], p["boxes_3d"], iou,
                                                     p["score"], gf, pf, iou_threshold)
                else:
                    sc, io = np.array([]), np.array([])
                accum_scores.append(sc); accum_ious.append(io)
            all_scores = np.concatenate(accum_scores, axis=0) if accum_scores else np.array([])
            all_ious = np.concatenate(accum_ious, axis=0) if accum_ious else np.array([])
            iou_3d[cls_idx, diff_idx] = np.average(all_ious) if len(all_ious) else 0.0
            if num_valid_gt == 0:
                continue
            thresholds = get_thresholds(all_scores, num_valid_gt, num_pr_points)
            confusion = np.zeros([len(thresholds), 3])
            for s in range(num_samples):
                iou = ious[s]
                if iou.size == 0:
                    continue
                ps = pred_annotation_frames[s]["score"]
                gf, pf = gt_flags[s], pred_flags[s]
                for th_idx, score_th in enumerate(thresholds):
                    tp, fp, fn = compute_statistics(iou, ps, gf, pf, score_th, iou_threshold)
                    confusion[th_idx] += (tp, fp, fn)
            for th_idx in range(len(thresholds)):
                tp, fp, fn = confusion[th_idx]
                recall[cls_idx, diff_idx, th_idx] = tp / (tp + fn) if (tp + fn) else 0.0
                precision[cls_idx, diff_idx, th_idx] = tp / (tp + fp) if (tp + fp) else 0.0
            for th_idx in range(len(thresholds)):
                precision[cls_idx, diff_idx, th_idx] = np.max(precision[cls_idx, diff_idx, th_idx:], axis=-1)
                recall[cls_idx, diff_idx, th_idx] = np.max(recall[cls_idx, diff_idx, th_idx:], axis=-1)

    AP = np.zeros([num_classes, num_difficulties])
    for i in range(1, precision.shape[-1]):
        AP += precision[:, :, i]
    AP = AP / num_pr_points * 100

    rows = []
    for idx, cur_class in enumerate(classes):
        rows.append({"class": cur_class, "occ_pred": int(pred_occ[cur_class]), "occ_gt": int(gt_occ[cur_class]),
                     "precision": float(np.mean(precision[idx], axis=-1)[0] * 100),
                     "recall": float(np.mean(recall[idx], axis=-1)[0] * 100),
                     "ap": float(AP[idx, 0])})
    total = {"class": f"Total ({num_classes} classes)",
             "occ_pred": int(sum(pred_occ.values())), "occ_gt": int(sum(gt_occ.values())),
             "precision": float(np.mean([r["precision"] for r in rows])) if rows else 0.0,
             "recall": float(np.mean([r["recall"] for r in rows])) if rows else 0.0,
             "ap": float(np.mean(AP[:, 0])) if num_classes else 0.0}
    return rows, total


# --------------------------------------------------------------------------- #
#  Parsing (LiDAR-frame, camera-free) + top-level runner
# --------------------------------------------------------------------------- #
def _attr(attrs, name):
    for a in attrs or []:
        if a.get("name") == name:
            return a.get("val")
    return None


def parse_folder(folder, object_min_points=0, is_prediction=False, flatten_z=True):
    """OpenLABEL folder → list of per-frame annotation dicts
    {name, boxes_3d[N,7]=(x,y,z,l,w,h,yaw), score, num_points_in_gt}, sorted by filename.

    `flatten_z` sets every box centre to z=0 to mirror the dev-kit's
    `project_to_ground` step (its official LiDAR eval projects all boxes onto the
    s110_base ground plane), so the 3D IoU is BEV-footprint × height-overlap and
    isn't penalised by absolute-height differences."""
    out = []
    for fp in sorted(glob.glob(os.path.join(folder, "*.json"))):
        name, boxes, npg, scores = [], [], [], []
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception:
            data = {}
        frames = data.get("openlabel", {}).get("frames", {})
        for _fid, fo in frames.items():
            for _oid, lab in fo.get("objects", {}).items():
                cub = lab.get("object_data", {}).get("cuboid", {})
                val = cub.get("val")
                if not val or len(val) < 10:
                    continue
                q = [float(val[3]), float(val[4]), float(val[5]), float(val[6])]
                if np.linalg.norm(q) == 0.0:
                    continue
                attrs = cub.get("attributes", {}).get("num", [])
                np_attr = _attr(attrs, "num_points")
                num_points = int(float(np_attr)) if np_attr is not None else (10 if is_prediction else 0)
                if num_points < object_min_points:
                    continue
                yaw = float(R.from_quat(q).as_euler("zyx", degrees=False)[0])
                name.append(str(lab["object_data"].get("type", "OTHER")).upper())
                boxes.append([float(val[0]), float(val[1]), 0.0 if flatten_z else float(val[2]),
                              float(val[7]), float(val[8]), float(val[9]), yaw])
                npg.append(num_points)
                sc = _attr(attrs, "score")
                scores.append(float(sc) if sc is not None else 1.0)
        out.append({"name": np.array(name),
                    "boxes_3d": np.array(boxes, dtype=np.float64) if boxes else np.zeros((0, 7)),
                    "num_points_in_gt": np.array(npg),
                    "score": np.array(scores, dtype=np.float64)})
    return out


def run_benchmark(gt_dir, pred_dir, object_min_points=5, use_superclass=True):
    """Score predictions against GT with the dev-kit's AP@0.1 protocol. Frames are
    paired by sorted filename (export names them to match the GT). Returns
    (rows, total) — per-(super)class Precision/Recall/AP plus a totals row."""
    gt = parse_folder(gt_dir, object_min_points=object_min_points, is_prediction=False)
    pred = parse_folder(pred_dir, object_min_points=0, is_prediction=True)
    n = min(len(gt), len(pred))
    if n == 0:
        raise ValueError("no GT or prediction frames found")
    return get_evaluation_results(gt[:n], pred[:n], _ALL_CLASSES, use_superclass=use_superclass)
