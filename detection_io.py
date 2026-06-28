"""Bridge between the app's in-memory detections and the TUM Traffic dev-kit's
benchmark formats.

- `export_openlabel` / `export_kitti`: write the current detector's `det_frames`
  as one prediction file per frame (named to match the GT), so the dev-kit's
  `src/eval/evaluation.py` can score them → per-class Precision/Recall/AP@0.1,
  comparable to the published InfraDet3D table.
- `load_openlabel_detections`: read a folder of OpenLABEL predictions back into
  the same `det_frames` shape — so a learned model's output (e.g. PointPillars
  run on Colab) drops straight into the app's tracking / WWD / evaluation.

The classical pipeline produces BEV boxes (cx, cy, l, w, yaw) with no height, so
exports place boxes on an estimated ground plane with a per-class default height.
AP@0.1 is dominated by the BEV footprint, so this is fine for benchmarking; a
learned detector carries real z/h and round-trips exactly through the loader.
"""
import glob
import json
import os

import numpy as np

# Per-class default box height (m) for the height-less BEV detector.
_CLASS_H = {"car": 1.5, "van": 2.0, "truck": 3.2, "bus": 3.2, "trailer": 3.5,
            "motorcycle": 1.5, "bicycle": 1.5, "pedestrian": 1.8, "other": 1.6}


def _h_for(cls):
    return _CLASS_H.get(str(cls).lower(), 1.6)


def _lw(d):
    return float(d.get("l", d.get("length", 0)) or 0), float(d.get("w", d.get("width", 0)) or 0)


def _yaw_to_quat(yaw):
    """z-axis yaw → quaternion [qx, qy, qz, qw] (OpenLABEL cuboid convention)."""
    return [0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))]


def _quat_to_yaw(qx, qy, qz, qw):
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def _box_z(d, ground_z):
    """(z_center, height) for a detection: real z/h if present, else ground + default."""
    if d.get("h") and d.get("z") is not None:
        return float(d["z"]), float(d["h"])
    h = _h_for(d.get("cls", "Car"))
    return float(ground_z) + h / 2.0, h


def export_kitti(det_frames, names, out_dir, ground_z=-7.5, progress=None):
    """One `<name>.txt` per frame: `class x y z l w h yaw score` per object."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(det_frames)
    for i, (dets, name) in enumerate(zip(det_frames, names)):
        with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            for d in dets:
                l, w = _lw(d)
                z, h = _box_z(d, ground_z)
                f.write(f"{d.get('cls','Car')} {float(d['cx']):.4f} {float(d['cy']):.4f} {z:.4f} "
                        f"{l:.4f} {w:.4f} {h:.4f} {float(d.get('yaw',0.0)):.4f} "
                        f"{float(d.get('score',0.0)):.4f}\n")
        if progress:
            progress(i + 1, n)
    return out_dir


def export_openlabel(det_frames, names, out_dir, ground_z=-7.5, progress=None):
    """One `<name>.json` per frame, mirroring the GT OpenLABEL cuboid schema, with
    the detector confidence stored as a `score` cuboid attribute."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(det_frames)
    for i, (dets, name) in enumerate(zip(det_frames, names)):
        objects = {}
        for j, d in enumerate(dets):
            l, w = _lw(d)
            z, h = _box_z(d, ground_z)
            q = _yaw_to_quat(float(d.get("yaw", 0.0)))
            oid = str(d.get("tid", j))
            objects[oid] = {
                "object_data": {
                    "type": str(d.get("cls", "CAR")).upper(),
                    "cuboid": {
                        "name": f"cuboid_{oid}",
                        "val": [float(d["cx"]), float(d["cy"]), z, q[0], q[1], q[2], q[3], l, w, h],
                        "attributes": {"num": [{"name": "score", "val": float(d.get("score", 0.0))}]},
                    },
                }
            }
        ol = {"openlabel": {"metadata": {"schema_version": "1.0.0"},
                            "frames": {"0": {"objects": objects}}}}
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(ol, f)
        if progress:
            progress(i + 1, n)
    return out_dir


def load_openlabel_detections(folder):
    """Read a folder of OpenLABEL prediction JSONs → (files, det_frames), sorted by
    filename. Each detection becomes dict(cls, cx, cy, z, l, w, h, yaw, score) — the
    shape the app's tracking/WWD/eval consume."""
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    frames = []
    for fp in files:
        dets = []
        try:
            ol = json.load(open(fp, encoding="utf-8"))["openlabel"]
            fr = ol.get("frames", {})
            objs = (next(iter(fr.values())) if fr else {}).get("objects", {})
        except Exception:
            objs = {}
        for oid, o in objs.items():
            cub = o.get("object_data", {}).get("cuboid", {})
            val = cub.get("val")
            if not val or len(val) < 10:
                continue
            x, y, z, qx, qy, qz, qw, l, w, h = (float(v) for v in val[:10])
            score = 0.0
            for a in cub.get("attributes", {}).get("num", []):
                if a.get("name") in ("score", "confidence"):
                    score = float(a.get("val", 0.0))
            dets.append(dict(cls=o.get("object_data", {}).get("type", "Car"),
                             cx=x, cy=y, z=z, l=l, w=w, h=h,
                             yaw=_quat_to_yaw(qx, qy, qz, qw), score=score))
        frames.append(dets)
    return files, frames
