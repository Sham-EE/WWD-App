import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import open3d as o3d
import imageio.v2 as imageio
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Scene geometry is shared with the background-filtering stage via
# geometry_config (config-driven, with a hard-coded fallback).
from geometry_config import get_research_polygon, get_road_polygon

# ---------------------- Helper functions ------------------------

def sorted_by_frame_index(files):
    def keyfn(f):
        base = os.path.splitext(os.path.basename(f))[0]
        digits = ''.join([c for c in base if c.isdigit()])
        return int(digits) if digits else 0
    return sorted(files, key=keyfn)

def load_points_from_pcd(pcd_path: str) -> np.ndarray:
    pc = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pc.points, dtype=np.float32)

def angle_wrap(a):
    while a > np.pi: a -= 2*np.pi
    while a <= -np.pi: a += 2*np.pi
    return a

def align_yaw_to_ref(yaw, ref):
    best = yaw
    best_diff = abs(angle_wrap(yaw-ref))
    for k in (-3,-2,-1,1,2,3):
        cand = yaw + k*(np.pi/2)
        d = abs(angle_wrap(cand-ref))
        if d < best_diff: best, best_diff = cand, d
    return best

# ---------------------- Candidate extraction -------------------------------

def extract_candidates(points_xyz: np.ndarray, bounds, eps: float):
    if points_xyz.shape[0] == 0:
        return []
    rminx, rminy, rmaxx, rmaxy = bounds
    res = 0.25
    ix = np.floor((points_xyz[:,0] - rminx) / res).astype(np.int32)
    iy = np.floor((points_xyz[:,1] - rminy) / res).astype(np.int32)
    Dx = int(np.ceil((rmaxx - rminx) / res))
    Dy = int(np.ceil((rmaxy - rminy) / res))
    mask = (ix >= 0) & (ix < Dx) & (iy >= 0) & (iy < Dy)
    pts = points_xyz[mask]
    if pts.shape[0] == 0:
        return []
    labels = DBSCAN(eps=eps, min_samples=1).fit(pts[:,:2]).labels_
    cands = []
    for lbl in set(labels) - {-1}:
        m = labels == lbl
        ptsk = pts[m]
        n = int(ptsk.shape[0])
        xy = ptsk[:,:2]
        mu = xy.mean(axis=0)
        cov = np.cov((xy - mu).T) if n > 1 else np.eye(2)
        vals, vecs = np.linalg.eig(cov)
        idx = int(np.argmax(vals.real))
        v = vecs[:,idx].real
        yaw = float(np.arctan2(v[1], v[0]))
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, s],[-s, c]], dtype=np.float32)
        local = (xy @ R.T)
        min_xy = local.min(axis=0)
        max_xy = local.max(axis=0)
        l_len = float(max_xy[0] - min_xy[0])
        w_len = float(max_xy[1] - min_xy[1])
        if w_len > l_len: l_len, w_len = w_len, l_len; yaw += np.pi/2.0
        cx, cy = float(mu[0]), float(mu[1])
        cands.append(dict(n=n, cx=cx, cy=cy, l=l_len, w=w_len, yaw=yaw))
    return cands

# ---------------------- Temporal logic -------------------------------------

def gate_radius(cx, cy):
    return 0.8 + 0.02 * np.hypot(cx, cy)

def _similar(c1, c2):
    return np.hypot(c1['cx']-c2['cx'], c1['cy']-c2['cy']) <= gate_radius(c1['cx'], c1['cy'])

def accept_with_temporal(cand, neighs, min_hits=2, min_pts=3, strong_pts=200):
    # Large dense clusters (e.g. a 900-point truck) are unambiguously real, so
    # accept them without temporal confirmation. This catches fast vehicles
    # (whose per-frame motion exceeds the tight gate) without widening the gate
    # for everything, which would let noise confirm itself and hurt precision.
    if cand['n'] < min_pts:
        return False
    if strong_pts and cand['n'] >= strong_pts:
        return True
    hits = 1
    for n in neighs:
        if _similar(cand, n):
            hits += 1
            if hits >= min_hits:
                return True
    return False

# ---------------------- Main Processing Function ---------------------------

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def run_detection_and_tracking(pcd_dir, out_dir, params, progress_callback=None):
    args = Args(**params)

    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, 'vis'); os.makedirs(vis_dir, exist_ok=True)

    pcd_files = sorted_by_frame_index(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    if not pcd_files:
        return None, "No PCD files found in the specified directory."

    research_poly = get_research_polygon(); rminx,rminy,rmaxx,rmaxy = research_poly.bounds
    road_poly = get_road_polygon()
    yaw_bias = np.deg2rad(args.yaw_bias_deg)

    cand_frames = []
    for idx, p in enumerate(pcd_files):
        pts = load_points_from_pcd(p)
        if pts.size > 0 and args.roi_abs_y is not None:
            pts = pts[np.abs(pts[:, 1]) <= float(args.roi_abs_y)]
        cand_frames.append(extract_candidates(pts, (rminx, rminy, rmaxx, rmaxy), eps=args.dbscan_eps))
        if progress_callback:
            progress_callback(idx + 1, len(pcd_files), "Extracting candidates")

    dt = 1.0 / float(args.fps) if args.fps and args.fps > 0 else 0.1

    class Track:
        __slots__ = ('tid', 'x_state', 'P', 'yaw', 'score', 'age', 'hits', 'missed', 'moving_history', 'cls', 'length', 'width', 'F', 'H', 'Q', 'R')
        def __init__(self, tid, x, y, yaw, score, cls='Car', length=0.0, width=0.0):
            self.tid = int(tid)
            self.x_state = np.array([float(x), float(y), 0.0, 0.0], dtype=np.float64)
            self.P = np.diag([1.0, 1.0, 9.0, 9.0]).astype(np.float64)
            self.yaw = float(yaw)
            self.score = float(score)
            self.cls = str(cls)
            self.length = float(length)
            self.width = float(width)
            self.age = 1
            self.hits = 1
            self.missed = 0
            self.moving_history = []
            self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
            self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
            sigma_a = 2.0; dt2=dt*dt; dt3=dt2*dt; dt4=dt2*dt2; q=sigma_a**2
            self.Q = q * np.array([[dt4/4,0,dt3/2,0],[0,dt4/4,0,dt3/2],[dt3/2,0,dt2,0],[0,dt3/2,0,dt2]], dtype=np.float64)
            sigma_z = 0.8
            self.R = np.diag([sigma_z**2, sigma_z**2]).astype(np.float64)
        def predict(self):
            self.x_state = self.F @ self.x_state
            self.P = self.F @ self.P @ self.F.T + self.Q
            self.age += 1
            self.missed += 1
        def innovation(self, meas_x, meas_y):
            z = np.array([float(meas_x), float(meas_y)], dtype=np.float64)
            y = z - (self.H @ self.x_state)
            S = self.H @ self.P @ self.H.T + self.R
            return y, S
        def gating_dist2(self, meas_x, meas_y):
            y, S = self.innovation(meas_x, meas_y)
            try: Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError: Sinv = np.linalg.pinv(S)
            return float(y.T @ Sinv @ y)
        def update(self, meas_x, meas_y, meas_yaw, meas_score, meas_cls='Car', meas_len=0.0, meas_w=0.0):
            z = np.array([float(meas_x), float(meas_y)], dtype=np.float64)
            y = z - (self.H @ self.x_state)
            S = self.H @ self.P @ self.H.T + self.R
            try: Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError: Sinv = np.linalg.pinv(S)
            K = self.P @ self.H.T @ Sinv
            self.x_state = self.x_state + K @ y
            I = np.eye(4, dtype=np.float64)
            self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
            self.yaw = align_yaw_to_ref(float(meas_yaw), self.yaw)
            self.score = float(meas_score)
            self.cls = str(meas_cls)
            if meas_len > 0: self.length = float(meas_len)
            if meas_w > 0: self.width = float(meas_w)
            self.hits += 1
            self.missed = 0
        def heading(self):
            """Direction of travel from the Kalman velocity (unambiguous, unlike
            the PCA-derived yaw which is 180-degree ambiguous). Returns radians,
            or None if the track is essentially stationary."""
            sp = self.speed()
            if sp < 1e-3:
                return None
            return float(np.arctan2(self.vy, self.vx))
        @property
        def x(self): return float(self.x_state[0])
        @property
        def y(self): return float(self.x_state[1])
        @property
        def vx(self): return float(self.x_state[2])
        @property
        def vy(self): return float(self.x_state[3])
        def speed(self): return float(np.hypot(self.vx, self.vy))
        def is_moving(self, speed_thresh, window=10):
            s = self.speed()
            self.moving_history.append(s)
            if len(self.moving_history) > window: self.moving_history = self.moving_history[-window:]
            return (np.median(self.moving_history) >= float(speed_thresh))

    raw_det_frames = []
    for i in range(len(pcd_files)):
        neighs = []
        for dti in (-2, -1, 1, 2):
            j = i + dti
            if 0 <= j < len(cand_frames): neighs.extend(cand_frames[j])
        dets = []
        for c in cand_frames[i]:
            if accept_with_temporal(c, neighs, min_hits=args.min_hits, min_pts=args.min_cluster_pts,
                                     strong_pts=getattr(args, 'strong_pts', 200)):
                dets.append(dict(cls='Car', cx=c['cx'], cy=c['cy'], l=float(c.get('l',0)), w=float(c.get('w',0)), yaw=c['yaw']+yaw_bias, score=c['n']))
        raw_det_frames.append(dets)

    tracks = []
    next_tid = 1
    track_det_frames = [[] for _ in range(len(pcd_files))]
    pending_by_tid = {}

    def _yaw_diff(a, b): return abs(angle_wrap(float(a) - float(b)))

    def merge_detections_with_yaw(dets_in, merge_dist, yaw_merge_rad, truck_merge_dist, truck_len_thresh):
        dets = [dict(d) for d in dets_in]
        if len(dets) < 2: return dets
        merged = True
        while merged:
            merged = False
            n = len(dets)
            for i in range(n):
                if merged: break
                for j in range(i + 1, n):
                    a, b = dets[i], dets[j]
                    dist = float(np.hypot(a['cx'] - b['cx'], a['cy'] - b['cy']))
                    yd = _yaw_diff(a.get('yaw', 0), b.get('yaw', 0))
                    if yd > yaw_merge_rad: continue
                    do_merge = False
                    if dist <= float(merge_dist): do_merge = True
                    elif dist <= float(truck_merge_dist):
                        yawm = align_yaw_to_ref(float(b.get('yaw',0)), float(a.get('yaw',0)))
                        yawm = 0.5 * (float(a.get('yaw',0)) + float(yawm))
                        c, s = np.cos(yawm), np.sin(yawm); ux, uy = c, s
                        la = max(0, float(a.get('l',0))); lb = max(0, float(b.get('l',0)))
                        a0 = (a['cx']*ux+a['cy']*uy)-la/2; a1 = (a['cx']*ux+a['cy']*uy)+la/2
                        b0 = (b['cx']*ux+b['cy']*uy)-lb/2; b1 = (b['cx']*ux+b['cy']*uy)+lb/2
                        merged_len = float(max(a1,b1) - min(a0,b0))
                        if merged_len >= float(truck_len_thresh): do_merge = True
                    if not do_merge: continue
                    wa = float(max(1,a.get('score',1))); wb = float(max(1,b.get('score',1))); wsum = wa+wb
                    cx = (wa*a['cx']+wb*b['cx'])/wsum; cy = (wa*a['cy']+wb*b['cy'])/wsum
                    yawb_aligned = align_yaw_to_ref(float(b.get('yaw',0)), float(a.get('yaw',0)))
                    yaw = 0.5 * (float(a.get('yaw',0)) + float(yawb_aligned))
                    c,s=np.cos(yaw),np.sin(yaw); ux,uy=c,s; vx,vy=-s,c
                    def extent(det):
                        l=max(0,float(det.get('l',0))); w=max(0,float(det.get('w',0)))
                        u=det['cx']*ux+det['cy']*uy; v=det['cx']*vx+det['cy']*vy
                        return (u-l/2, u+l/2, v-w/2, v+w/2)
                    a_u0,a_u1,a_v0,a_v1=extent(a); b_u0,b_u1,b_v0,b_v1=extent(b)
                    u0,u1=min(a_u0,b_u0),max(a_u1,b_u1); v0,v1=min(a_v0,b_v0),max(a_v1,b_v1)
                    l=float(max(0,u1-u0)); w=float(max(0,v1-v0))
                    score=float(a.get('score',0))+float(b.get('score',0))
                    m=dict(cls='Car',cx=cx,cy=cy,l=l,w=w,yaw=yaw,score=score)
                    dets[i]=m; dets.pop(j); merged=True; break
        return dets

    for fi in range(len(pcd_files)):
        dets = raw_det_frames[fi]
        dets = merge_detections_with_yaw(dets, float(args.merge_dist), np.deg2rad(args.yaw_merge_deg), float(args.truck_merge_dist), float(args.truck_len_thresh))
        for t in tracks: t.predict()
        used_tracks = set(); used_dets = set()
        chi2_gate_2d = 9.21
        # Globally optimal detection<->track association via the Hungarian
        # algorithm on the Mahalanobis gating distance, replacing the previous
        # greedy first-match loop (which could mis-pair in dense scenes).
        if dets and tracks:
            cost = np.full((len(dets), len(tracks)), 1e9, dtype=np.float64)
            for di, d in enumerate(dets):
                for ti, t in enumerate(tracks):
                    d2 = t.gating_dist2(d['cx'], d['cy'])
                    if d2 <= chi2_gate_2d:
                        cost[di, ti] = d2
            row_ind, col_ind = linear_sum_assignment(cost)
            for di, ti in zip(row_ind, col_ind):
                if cost[di, ti] <= chi2_gate_2d:
                    tracks[ti].update(dets[di]['cx'], dets[di]['cy'], dets[di]['yaw'], dets[di]['score'],
                                      meas_cls=dets[di].get('cls','Car'), meas_len=dets[di].get('l',0), meas_w=dets[di].get('w',0))
                    used_tracks.add(ti); used_dets.add(di)
        for di, d in enumerate(dets):
            if di in used_dets: continue
            t = Track(next_tid, d['cx'], d['cy'], d['yaw'], d.get('score',0), cls=d.get('cls','Car'), length=d.get('l',0), width=d.get('w',0)); next_tid+=1; tracks.append(t)
        tracks = [t for t in tracks if t.missed <= int(args.max_missed)]
        if len(tracks) >= 2 and args.merge_dist is not None and float(args.merge_dist) > 0:
            md = float(args.merge_dist)
            merged = True
            while merged:
                merged = False; n = len(tracks)
                for i in range(n):
                    if merged: break
                    for j in range(i + 1, n):
                        ti, tj = tracks[i], tracks[j]
                        if float(np.hypot(ti.x-tj.x, ti.y-tj.y)) <= md:
                            hi=(ti.missed==0); hj=(tj.missed==0)
                            if hi and not hj: keep,drop=ti,tj
                            elif hj and not hi: keep,drop=tj,ti
                            else: keep,drop=(ti,tj) if ti.hits>tj.hits else (tj,ti) if tj.hits>ti.hits else (ti,tj) if ti.age>=tj.age else (tj,ti)
                            keep.x_state=0.5*(keep.x_state+drop.x_state); keep.P=0.5*(keep.P+drop.P)
                            keep.yaw=align_yaw_to_ref(drop.yaw,keep.yaw); keep.score=max(float(keep.score),float(drop.score))
                            keep.hits=max(int(keep.hits),int(drop.hits)); keep.age=max(int(keep.age),int(drop.age))
                            keep.missed=min(int(keep.missed),int(drop.missed))
                            if drop.cls=='Truck': keep.cls='Truck'
                            tracks.remove(drop); merged=True; break
        for t in tracks:
            hit = (t.missed == 0)
            # Call is_moving exactly once per frame: it mutates moving_history,
            # so the previous two-call version double-counted every frame.
            moving = t.is_moving(args.moving_speed_thresh, window=10)
            # Classification by size only; a stopped truck is still a truck.
            final_cls = 'Truck' if t.length >= float(args.truck_len_thresh) else 'Car'
            # Report the measured box size; fall back to nominal sizes only when
            # no extent was ever observed.
            out_l = t.length if t.length > 0 else (9.0 if final_cls == 'Truck' else 4.5)
            out_w = t.width if t.width > 0 else 1.9
            hdg = t.heading()
            det_dict = dict(tid=t.tid, cls=final_cls, cx=t.x, cy=t.y, l=out_l, w=out_w, yaw=t.yaw,
                            score=t.score, hit=bool(hit), speed=t.speed(),
                            vx=t.vx, vy=t.vy, heading=(float(hdg) if hdg is not None else None),
                            moving=moving, length=t.length)
            if hit:
                pend = pending_by_tid.pop(t.tid, [])
                for pf, pd in pend: pd['tid']=t.tid; track_det_frames[pf].append(pd)
                track_det_frames[fi].append(det_dict)
            else:
                pending_by_tid.setdefault(t.tid, []).append((fi, det_dict))
        alive_tids = set([t.tid for t in tracks])
        for tid in list(pending_by_tid.keys()):
            if tid not in alive_tids: pending_by_tid.pop(tid, None)
        if progress_callback:
            progress_callback(fi + 1, len(pcd_files), "Tracking objects")

    # Instead of creating a GIF, return the raw data for 3D visualization
    results = {
        "pcd_files": pcd_files,
        "det_frames": track_det_frames,
        "road_poly": road_poly,
        "research_poly_bounds": research_poly.bounds
    }

    # Save track results as CSV for post-processing
    csv_path = os.path.join(out_dir, 'tracks.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('frame,tid,cls,cx,cy,yaw,vx,vy,heading,hit,speed,moving,score,length,width\n')
        for frame_idx, dets in enumerate(track_det_frames):
            for d in dets:
                hdg = d.get('heading', None)
                hdg_s = f"{hdg:.6f}" if hdg is not None else ''
                f.write(f"{frame_idx},{d.get('tid','')},{d.get('cls','')},{d.get('cx',0):.6f},{d.get('cy',0):.6f},"
                        f"{d.get('yaw',0):.6f},{d.get('vx',0):.6f},{d.get('vy',0):.6f},{hdg_s},"
                        f"{int(bool(d.get('hit',True)))},{d.get('speed',0):.6f},{int(bool(d.get('moving',False)))},"
                        f"{d.get('score',0):.6f},{d.get('length',0):.6f},{d.get('w',0):.6f}\n")

    if progress_callback:
        progress_callback(len(pcd_files), len(pcd_files), "Finished")

    return results, None
