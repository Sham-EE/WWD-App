import numpy as np
import open3d as o3d
import glob
import os
import json
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN

# Geometry now comes from config/site_geometry.json via geometry_config (with a
# hard-coded fallback to the original values). The vectorised points_in_polygon
# replaces the old per-point shapely.Point loops.
from geometry_config import (
    get_research_polygon,
    get_road_polygon,
    get_fg_exclusion_rects,
    get_coarse_grid,
    points_in_polygon,
)

# FG exclusion rects are resolved lazily (per active dataset), not cached at
# import — so switching datasets uses the right geometry. geometry_config caches
# the parse, so calling this per frame is cheap.
def _fg_rects_buffered():
    return [p.buffer(0) for p in get_fg_exclusion_rects()]

# --- Core Utility Functions (from original script) ---

def sorted_by_frame_index(files):
    def keyfn(f):
        base = os.path.splitext(os.path.basename(f))[0]
        digits = ''.join([c for c in base if c.isdigit()])
        return int(digits) if digits else 0
    return sorted(files, key=keyfn)

def parse_gt_bboxes(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    boxes = []
    try:
        frames = data['openlabel']['frames']
        frame_obj = frames[next(iter(frames))]
        for _, od in frame_obj.get('objects', {}).items():
            obj_data = od.get('object_data', {})
            val = obj_data.get('cuboid', {}).get('val')
            if isinstance(val, list) and len(val) >= 10:
                cx, cy, cz = float(val[0]), float(val[1]), float(val[2])
                l, w, h = float(val[7]), float(val[8]), float(val[9])
                boxes.append(dict(cx=cx, cy=cy, cz=cz, l=l, w=w, h=h))
    except Exception:
        pass
    return boxes

# --- Core Algorithm Functions (from original script) ---

def remove_fg_rects(pts: np.ndarray):
    if pts.size == 0: return pts
    keep = np.ones(pts.shape[0], dtype=bool)
    for poly in _fg_rects_buffered():
        keep &= ~points_in_polygon(poly, pts[:, :2])
    return pts[keep]

def remove_ground_grid_minz(points_xyz: np.ndarray, grid=0.5, dz_thresh=0.35):
    """Remove ground using per-grid-cell minimum z.

    This version is written defensively to avoid shape/dtype issues with
    np.minimum.at on different NumPy versions.
    """
    if points_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    # Ensure we are always working with a plain (N, 3) float array
    pts = np.asarray(points_xyz, dtype=np.float32)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Quantise to grid and build structured keys
    gx = np.floor(x / grid).astype(np.int64)
    gy = np.floor(y / grid).astype(np.int64)
    keys = np.stack([gx, gy], axis=1)
    k_view = np.ascontiguousarray(keys).view([("gx", np.int64), ("gy", np.int64)])

    # uniq: unique cells, inverse: for each point, index into uniq
    uniq, inverse = np.unique(k_view, return_inverse=True)
    inverse = inverse.ravel().astype(np.int64)
    z_flat = z.ravel().astype(np.float64)

    # Per-cell min z
    minz = np.full(uniq.shape[0], np.inf, dtype=np.float64)
    np.minimum.at(minz, inverse, z_flat)

    # Mark points that are sufficiently above the local minimum z as non-ground
    return z_flat > (minz[inverse] + float(dz_thresh))

def adaptive_dbscan(points_xy, distances, eps0=0.35, eps_k=0.008, eps_min=0.35, eps_max=2.0, min_samples=16):
    if points_xy.shape[0] == 0: return np.empty((0,), dtype=int)
    eps = eps0 + eps_k * distances
    eps = np.clip(eps, eps_min, eps_max)
    eps_global = float(np.median(eps))
    ms = int(max(8, min_samples))
    return DBSCAN(eps=eps_global, min_samples=ms).fit(points_xy).labels_

def voxel_downsample_numpy(points, voxel_size=0.15):
    if points.shape[0] == 0: return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[idx]

def build_grid_regions(research_poly: Polygon, cell_size: float = 20.0):
    minx, miny, maxx, maxy = research_poly.bounds
    regions = []
    for ix in range(int(np.ceil((maxx - minx) / cell_size))):
        for iy in range(int(np.ceil((maxy - miny) / cell_size))):
            x0, y0 = minx + ix * cell_size, miny + iy * cell_size
            poly = Polygon([(x0, y0), (x0 + cell_size, y0), (x0 + cell_size, y0 + cell_size), (x0, y0 + cell_size)])
            regions.append((f"cell_{ix}_{iy}", poly))
    return regions

def compute_region_z_ranges(gt_dir, regions):
    json_files = sorted_by_frame_index(glob.glob(os.path.join(gt_dir, '*.json')))
    z_ranges = {name: {'zmin': None, 'zmax': None} for name, _ in regions}
    if not json_files: return z_ranges, -5.0, 10.0
    all_zmins, all_zmaxs = [], []
    for jp in json_files:
        for b in parse_gt_bboxes(jp):
            gt_poly = Polygon([(b['cx']-b['l']/2, b['cy']-b['w']/2), (b['cx']+b['l']/2, b['cy']-b['w']/2), 
                               (b['cx']+b['l']/2, b['cy']+b['w']/2), (b['cx']-b['l']/2, b['cy']+b['w']/2)])
            if not gt_poly.is_valid: gt_poly = gt_poly.buffer(0)
            if gt_poly.is_empty: continue
            zmin, zmax = b['cz'] - b['h'] / 2.0, b['cz'] + b['h'] / 2.0
            all_zmins.append(zmin)
            all_zmaxs.append(zmax)
            for name, poly in regions:
                if gt_poly.intersects(poly):
                    zr = z_ranges[name]
                    zr['zmin'] = zmin if zr['zmin'] is None else min(zr['zmin'], zmin)
                    zr['zmax'] = zmax if zr['zmax'] is None else max(zr['zmax'], zmax)
    z_floor = float(min(all_zmins)) if all_zmins else -5.0
    z_ceil = float(max(all_zmaxs)) if all_zmaxs else 10.0
    return z_ranges, z_floor, z_ceil

def compute_region_z_ranges_from_data(pcd_files, regions, config, max_frames=20,
                                      object_height=5.0):
    """Label-free fallback for datasets WITHOUT ground-truth: estimate the ground
    level from the point cloud and keep a generous band (ground -> ground+height)
    for every region. Anchored to the global low percentile so it works on any
    site; without this a GT-less dataset would keep zero foreground."""
    files = sorted_by_frame_index(pcd_files)
    if not files:
        return {name: {'zmin': None, 'zmax': None} for name, _ in regions}, -5.0, 10.0
    idxs = np.unique(np.linspace(0, len(files) - 1, min(max_frames, len(files))).astype(int))
    grounds = []
    for i in idxs:
        pts = np.asarray(o3d.io.read_point_cloud(files[int(i)]).points, dtype=np.float32)
        if pts.size:
            grounds.append(float(np.percentile(pts[:, 2], 1.0)))
    ground = float(np.median(grounds)) if grounds else -5.0
    zmin, zmax = ground - 0.5, ground + float(object_height)
    z_ranges = {name: {'zmin': zmin, 'zmax': zmax} for name, _ in regions}
    return z_ranges, zmin, zmax

def cluster_shape_metrics(cluster_pts: np.ndarray):
    if cluster_pts.shape[0] < 3: return dict(H=0.0, area_xy=0.0, aspect_xy=0.0, L=0.0, P=0.0)
    H = float(cluster_pts[:,2].max() - cluster_pts[:,2].min())
    cov = np.cov(cluster_pts[:,:2].T)
    vals = np.sort(np.real(np.linalg.eigvalsh(cov)))[::-1]
    vals = np.maximum(vals, 1e-12)
    aspect = float(np.sqrt(vals[0]) / np.sqrt(vals[1])) if vals[1] > 0 else float('inf')
    area_xy = float(np.pi * np.sqrt(vals[0] * vals[1]))
    cov3 = np.cov(cluster_pts.T)
    e = np.sort(np.real(np.linalg.eigvalsh(cov3)))[::-1] + 1e-12
    L = float((e[0] - e[1]) / e[0])
    return dict(H=H, area_xy=area_xy, aspect_xy=aspect, L=L)

def geometry_filter_pole(points_xyz, params, z_ceiling=None):
    if not params['enable_pole_filter'] or points_xyz.shape[0] == 0:
        return points_xyz
    dists = np.linalg.norm(points_xyz[:,:2], axis=1)
    cluster_args = {k: v for k, v in params['cluster'].items() if k != 'ds_voxel'}
    labels = adaptive_dbscan(points_xyz[:,:2], dists, **cluster_args)
    keep_mask = np.ones(points_xyz.shape[0], dtype=bool)
    for lbl in set(labels) - {-1}:
        inds = np.where(labels == lbl)[0]
        if inds.size < max(3, params['pole_min_points']):
            continue
        m = cluster_shape_metrics(points_xyz[inds])
        # Poles are sparse; vehicles are dense. A small XY footprint only counts
        # as pole-like when the cluster is also sparse, so a tall but compact
        # truck/van (many points) is no longer mistaken for a pole and deleted.
        pole_max_points = params.get('pole_max_points', 80)
        pole_like = (m['H'] >= params['pole_min_height']) and \
                    ((m['aspect_xy'] >= params['pole_min_aspect_xy']) or \
                     (m['L'] >= params['pole_min_linearity']) or \
                     (m['area_xy'] <= params['pole_max_xy_area'] and inds.size <= pole_max_points))
        if not pole_like and z_ceiling is not None:
            pole_like |= (m['H'] >= max(0.0, z_ceiling - points_xyz[inds, 2].min()) - 0.3) and \
                         (m['aspect_xy'] >= params['pole_min_aspect_xy'])
        if pole_like:
            keep_mask[inds] = False
    return points_xyz[keep_mask]

# --- Main Logic Functions ---

def build_background_model(config: dict, pcd_files: list, gt_dir: str, progress_callback=None):
    research_poly = get_research_polygon()
    rminx, rminy, rmaxx, rmaxy = research_poly.bounds
    road_poly = get_road_polygon()
    
    edge_band = road_poly.difference(road_poly.buffer(-config['inward_buffer_m'])) if config['inward_buffer_m'] > 0 else Polygon()

    regions = build_grid_regions(research_poly, cell_size=20.0)
    # Use GT cuboids for per-region height bands when GT exists; otherwise derive
    # a label-free band from the point cloud so GT-less datasets still filter.
    has_gt = bool(gt_dir and glob.glob(os.path.join(gt_dir, '*.json')))
    if has_gt:
        z_ranges, z_floor, z_ceil = compute_region_z_ranges(gt_dir, regions)
    else:
        z_ranges, z_floor, z_ceil = compute_region_z_ranges_from_data(pcd_files, regions, config)

    # Init models
    Dx, Dy, Dz = (int(np.ceil((rmaxx - rminx) / config['bg_voxel'])),
                  int(np.ceil((rmaxy - rminy) / config['bg_voxel'])),
                  int(np.ceil((z_ceil - z_floor) / config['bg_voxel'])))
    presence_counts = np.zeros((Dx, Dy, Dz), dtype=np.uint16)
    Cx, Cy = (int(np.ceil((rmaxx - rminx) / config['cell_size'])),
              int(np.ceil((rmaxy - rminy) / config['cell_size'])))
    cell_counts = np.zeros((Cx, Cy), dtype=np.uint16)
    NX5, NY5 = config['coarse_5x5']['NX'], config['coarse_5x5']['NY']
    runlen5 = np.zeros((NX5, NY5), dtype=np.int16)
    has_run3 = np.zeros((NX5, NY5), dtype=bool)

    buildN = len(pcd_files) if config['build_frames'] == 0 else min(config['build_frames'], len(pcd_files))
    for i, pcd_path in enumerate(pcd_files[:buildN]):
        if progress_callback: progress_callback(i / buildN, f"Building frame {i+1}/{buildN}")
        pts = np.asarray(o3d.io.read_point_cloud(pcd_path).points, dtype=np.float32)

        # Pre-filtering pipeline
        pts = remove_fg_rects(pts)
        if pts.size == 0: continue
        pts_ng = pts[remove_ground_grid_minz(pts, grid=config['ground_grid'], dz_thresh=config['dz_thresh'])]
        if pts_ng.size == 0: continue
        
        kept_list = []
        for name, poly in regions:
            zr = z_ranges[name]
            if zr['zmin'] is None or zr['zmax'] is None: continue
            minx, miny, maxx, maxy = poly.bounds
            cand = pts_ng[(pts_ng[:,0] >= minx) & (pts_ng[:,0] <= maxx) & (pts_ng[:,1] >= miny) & (pts_ng[:,1] <= maxy)]
            if cand.size == 0: continue
            inside = cand[points_in_polygon(poly, cand[:, :2])]
            if inside.size > 0:
                kept_list.append(inside[(inside[:,2] >= zr['zmin']) & (inside[:,2] <= zr['zmax'])])
        pre_pts = np.vstack(kept_list) if kept_list else np.empty((0,3), dtype=np.float32)
        if pre_pts.size == 0: continue

        if edge_band is not None and not edge_band.is_empty and pre_pts.size > 0:
            inside_mask = points_in_polygon(edge_band, pre_pts[:, :2])
            pre_pts = pre_pts[~inside_mask]
        if pre_pts.size == 0: continue

        pre_pts = geometry_filter_pole(pre_pts, config, z_ceil)
        if pre_pts.size == 0: continue

        # Accumulate models
        vx, vy, vz = (np.floor((pre_pts[:,i] - offset) / config['bg_voxel']).astype(np.int32) for i, offset in enumerate([rminx, rminy, z_floor]))
        valid = (vx >= 0) & (vx < Dx) & (vy >= 0) & (vy < Dy) & (vz >= 0) & (vz < Dz)
        if np.any(valid): presence_counts[vx[valid], vy[valid], vz[valid]] += 1

        ds = voxel_downsample_numpy(pre_pts, voxel_size=config['cluster']['ds_voxel'])
        if ds.shape[0] >= config['cluster']['min_samples']:
            cluster_args = {k: v for k, v in config['cluster'].items() if k != 'ds_voxel'}
            labels = adaptive_dbscan(ds[:,:2], np.linalg.norm(ds[:,:2], axis=1), **cluster_args)
            for lbl in set(labels) - {-1}:
                cen = ds[labels == lbl,:2].mean(axis=0)
                cx, cy = int(np.floor((cen[0] - rminx) / config['cell_size'])), int(np.floor((cen[1] - rminy) / config['cell_size']))
                if 0 <= cx < Cx and 0 <= cy < Cy: cell_counts[cx, cy] += 1

        presence5 = np.zeros((NX5, NY5), dtype=bool)
        ix5, iy5 = (np.floor((pre_pts[:,i] - offset) / size).astype(np.int32) for i,(offset,size) in enumerate([(rminx, (rmaxx-rminx)/NX5), (rminy, (rmaxy-rminy)/NY5)]))
        valid5 = (ix5 >= 0) & (ix5 < NX5) & (iy5 >= 0) & (iy5 < NY5)
        if np.any(valid5): presence5[ix5[valid5], iy5[valid5]] = True
        runlen5 = (runlen5 + 1) * presence5
        has_run3 |= (runlen5 >= 3)

    # Return a serializable model (without prepared geometries)
    return {
        'voxel_mask': presence_counts / float(buildN) >= config['bg_ratio'],
        'cell_mask': cell_counts / float(buildN) >= config['cell_ratio'],
        '5x5_mask': ~has_run3,
        'z_ranges': z_ranges, 'z_floor': z_floor, 'z_ceil': z_ceil,
        'rminx': rminx, 'rminy': rminy, 'rmaxx': rmaxx, 'rmaxy': rmaxy,
        'road_poly': road_poly, 'edge_band': edge_band,
        'regions': regions
    }

def filter_points_with_model(points: np.ndarray, bg_model: dict, config: dict):
    if points.size == 0: return points, np.array([])

    # --- Prepare geometries for this run ---
    edge_band = bg_model['edge_band']

    # Pre-filtering
    pts = remove_fg_rects(points)
    pts_ng = pts[remove_ground_grid_minz(pts, grid=config['ground_grid'], dz_thresh=config['dz_thresh'])]

    kept_list = []
    for name, poly in bg_model['regions']:
        zr = bg_model['z_ranges'][name]
        if zr['zmin'] is None or zr['zmax'] is None: continue
        minx, miny, maxx, maxy = poly.bounds
        cand = pts_ng[(pts_ng[:,0] >= minx) & (pts_ng[:,0] <= maxx) & (pts_ng[:,1] >= miny) & (pts_ng[:,1] <= maxy)]
        if cand.size == 0: continue
        inside = cand[points_in_polygon(poly, cand[:, :2])]
        if inside.size > 0:
            kept_list.append(inside[(inside[:,2] >= zr['zmin']) & (inside[:,2] <= zr['zmax'])])
    pre_pts = np.vstack(kept_list) if kept_list else np.empty((0,3), dtype=np.float32)

    if edge_band is not None and not edge_band.is_empty and pre_pts.size > 0:
        inside_mask = points_in_polygon(edge_band, pre_pts[:, :2])
        pre_pts = pre_pts[~inside_mask]

    # Filtering stage 1: Cluster persistence
    cluster_args = {k: v for k, v in config['cluster'].items() if k != 'ds_voxel'}
    labels = adaptive_dbscan(pre_pts[:,:2], np.linalg.norm(pre_pts[:,:2], axis=1), **cluster_args)
    keep_mask = np.ones(pre_pts.shape[0], dtype=bool)
    Cx, Cy = bg_model['cell_mask'].shape
    for lbl in set(labels) - {-1}:
        inds = np.where(labels == lbl)[0]
        cen = pre_pts[inds,:2].mean(axis=0)
        cx, cy = int(np.floor((cen[0] - bg_model['rminx']) / config['cell_size'])), int(np.floor((cen[1] - bg_model['rminy']) / config['cell_size']))
        if 0 <= cx < Cx and 0 <= cy < Cy and bg_model['cell_mask'][cx, cy]:
            keep_mask[inds] = False
    remained = pre_pts[keep_mask]

    # Filtering stage 2: Pole-like geometry
    remained = geometry_filter_pole(remained, config, bg_model['z_ceil'])

    # Filtering stage 3: Voxel background
    vx, vy, vz = (np.floor((remained[:,i] - offset) / config['bg_voxel']).astype(np.int32) for i, offset in enumerate([bg_model['rminx'], bg_model['rminy'], bg_model['z_floor']]))
    Dx, Dy, Dz = bg_model['voxel_mask'].shape
    valid = (vx >= 0) & (vx < Dx) & (vy >= 0) & (vy < Dy) & (vz >= 0) & (vz < Dz)
    keep_v = np.ones(remained.shape[0], dtype=bool)
    keep_v[valid] = ~bg_model['voxel_mask'][vx[valid], vy[valid], vz[valid]]
    fpts = remained[keep_v]

    # Filtering stage 4: 5x5 coarse background
    if fpts.shape[0] > 0:
        NX5, NY5 = bg_model['5x5_mask'].shape
        ix5, iy5 = (np.floor((fpts[:,i] - offset) / size).astype(np.int32) for i,(offset,size) in enumerate([(bg_model['rminx'], (bg_model['rmaxx']-bg_model['rminx'])/NX5), (bg_model['rminy'], (bg_model['rmaxy']-bg_model['rminy'])/NY5)]))
        valid5 = (ix5 >= 0) & (ix5 < NX5) & (iy5 >= 0) & (iy5 < NY5)
        if np.any(valid5):
            rm_mask = bg_model['5x5_mask'][ix5[valid5], iy5[valid5]]
            f_keep = np.ones(fpts.shape[0], dtype=bool)
            f_keep[np.where(valid5)[0]] = ~rm_mask
            fpts = fpts[f_keep]

    return fpts, np.array([]) # Second return is background points, not implemented here
