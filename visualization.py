import os
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import imageio
import open3d as o3d
from detection_logic import load_points_from_pcd

# Cardinal direction colour encoding (sensor frame, atan2(vy,vx)).
# Matches the Lane Editor scheme: E=+X red, N=+Y green, W=-X blue, S=-Y orange.
CARDINAL_BINS = [
    ("→ E (+X)", '#d62728'),
    ("→ N (+Y)", '#2ca02c'),
    ("→ W (-X)", '#1f77b4'),
    ("→ S (-Y)", '#ff7f0e'),
]


def cardinal_index(heading_rad):
    """0=E, 1=N, 2=W, 3=S for a heading in radians."""
    deg = (np.degrees(heading_rad) + 180.0) % 360.0 - 180.0
    if -45 <= deg < 45:
        return 0
    if 45 <= deg < 135:
        return 1
    if deg >= 135 or deg < -135:
        return 2
    return 3


def cardinal_color(heading_rad):
    return CARDINAL_BINS[cardinal_index(heading_rad)][1]


def object_marker_color(d, speed_threshold, default='red'):
    """Cardinal color when the object is moving with a defined heading; otherwise
    `default` (red) for stationary / undefined-direction objects."""
    hdg = d.get('heading', None)
    if hdg is not None and d.get('speed', 0.0) >= speed_threshold:
        return cardinal_color(hdg)
    return default


def _arrow_segments(cx, cy, hdg, z, length=4.0, head=1.6, head_ang=np.radians(30)):
    """Return (xs, ys, zs) for a shaft + arrowhead 'V', with None separators so a
    whole set of arrows can live in one Scatter3d trace."""
    tx, ty = cx + length * np.cos(hdg), cy + length * np.sin(hdg)
    lwx, lwy = tx - head * np.cos(hdg - head_ang), ty - head * np.sin(hdg - head_ang)
    rwx, rwy = tx - head * np.cos(hdg + head_ang), ty - head * np.sin(hdg + head_ang)
    xs = [cx, tx, None, lwx, tx, rwx, None]
    ys = [cy, ty, None, lwy, ty, rwy, None]
    zs = [z, z, None, z, z, z, None]
    return xs, ys, zs


def create_3d_figure(results, frame_index_to_render, original_pcd_path, camera_dict=None):
    """
    Creates an interactive 3D Plotly figure for a given frame.
    (Keep this for interactive UI display as it doesn't need Kaleido)
    """
    fig = go.Figure()

    # 1. Add Original Point Cloud
    points = load_points_from_pcd(original_pcd_path)
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers', name='Original Point Cloud',
        marker=dict(size=1, color='grey', opacity=0.5)
    ))

    # 2. Add Road Polygon
    road_poly = results['road_poly']
    if road_poly.geom_type == 'Polygon': polys = [road_poly]
    else: polys = road_poly.geoms
    for poly in polys:
        x, y = poly.exterior.xy
        x_np = np.array(x)
        y_np = np.array(y)
        fig.add_trace(go.Scatter3d(
            x=x_np, y=y_np, z=np.full_like(x_np, -7.5),
            mode='lines', name='Road',
            line=dict(color='green', width=4)
        ))

    # 2b. Optional lane-direction overlay (for WWD calibration / sanity check)
    if results.get('show_lanes') and results.get('lanes'):
        lane_colors = ['#1f77b4', '#9467bd', '#17becf', '#e377c2', '#bcbd22', '#7f7f7f']
        z_lane = -7.3
        for li, lane in enumerate(results['lanes']):
            poly = lane['polygon']
            col = lane_colors[li % len(lane_colors)]
            lx, ly = poly.exterior.xy
            lx, ly = np.array(lx), np.array(ly)
            # lane boundary
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=np.full_like(lx, z_lane),
                mode='lines', name=f"Lane: {lane['lane_id']}",
                line=dict(color=col, width=5)
            ))
            # expected-direction arrow from the lane centroid -> tip
            cxl, cyl = float(poly.centroid.x), float(poly.centroid.y)
            hd = np.radians(lane['heading_deg'])
            arrow_len = 9.0
            tx, ty = cxl + arrow_len * np.cos(hd), cyl + arrow_len * np.sin(hd)
            # shaft (with the lane label at its base)
            fig.add_trace(go.Scatter3d(
                x=[cxl, tx], y=[cyl, ty], z=[z_lane, z_lane],
                mode='lines+text',
                text=[f"{lane['lane_id']} ({lane['heading_deg']:.0f}°)", ''],
                textposition='top center', textfont=dict(size=12, color=col),
                line=dict(color=col, width=8),
                showlegend=False, hoverinfo='none'
            ))
            # arrowhead: two wings forming a ">" at the tip, pointing along hd
            wing_len, wing_ang = 2.4, np.radians(28)
            lwx, lwy = tx - wing_len * np.cos(hd - wing_ang), ty - wing_len * np.sin(hd - wing_ang)
            rwx, rwy = tx - wing_len * np.cos(hd + wing_ang), ty - wing_len * np.sin(hd + wing_ang)
            fig.add_trace(go.Scatter3d(
                x=[lwx, tx, rwx], y=[lwy, ty, rwy], z=[z_lane, z_lane, z_lane],
                mode='lines', line=dict(color=col, width=8),
                showlegend=False, hoverinfo='none'
            ))

    # 3. Add Detections
    det_frames = results['det_frames']
    current_dets = det_frames[frame_index_to_render]
    params = results.get('params', {})
    speed_threshold = params.get('moving_speed_thresh', 3.0)

    # Split detections into normal vs. wrong-way so the latter stand out. Normal
    # markers are colored by cardinal direction when moving, else red.
    norm_x, norm_y, norm_z, norm_text, norm_color = [], [], [], [], []
    ww_x, ww_y, ww_z, ww_text = [], [], [], []
    for d in current_dets:
        speed = d.get('speed', 0.0)
        label = f"ID: {d['tid']}" + (f"<br>{speed:.1f} m/s" if speed >= speed_threshold else "")
        if d.get('wrong_way'):
            ww_x.append(d['cx']); ww_y.append(d['cy']); ww_z.append(-6.5)
            ww_text.append(f"⚠ WRONG WAY<br>{label}")
        else:
            norm_x.append(d['cx']); norm_y.append(d['cy']); norm_z.append(-6.5)
            norm_text.append(label)
            norm_color.append(object_marker_color(d, speed_threshold))

    fig.add_trace(go.Scatter3d(
        x=norm_x, y=norm_y, z=norm_z,
        mode='markers+text',
        name='Objects',
        marker=dict(size=8, color=norm_color or 'red', symbol='circle'),
        text=norm_text,
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hoverinfo='none'
    ))

    if ww_x:
        fig.add_trace(go.Scatter3d(
            x=ww_x, y=ww_y, z=ww_z,
            mode='markers+text',
            name='⚠ Wrong-way',
            marker=dict(size=14, color='orange', symbol='diamond',
                        line=dict(color='black', width=2)),
            text=ww_text,
            textposition='top center',
            textfont=dict(size=12, color='darkorange'),
            hoverinfo='none'
        ))

    # Heading arrows for moving objects, color-encoded by cardinal direction and
    # drawn with an arrowhead. Grouped into one trace per direction so the legend
    # shows the colour->direction key.
    bins_xyz = {i: ([], [], []) for i in range(len(CARDINAL_BINS))}
    for d in current_dets:
        if d.get('speed', 0.0) < speed_threshold:
            continue
        hdg = d.get('heading', None)
        if hdg is None:
            continue
        ci = cardinal_index(hdg)
        xs, ys, zs = _arrow_segments(d['cx'], d['cy'], hdg, -6.5)
        bins_xyz[ci][0].extend(xs)
        bins_xyz[ci][1].extend(ys)
        bins_xyz[ci][2].extend(zs)
    for ci, (name, color) in enumerate(CARDINAL_BINS):
        xs, ys, zs = bins_xyz[ci]
        if not xs:
            continue
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines', name=name,
            line=dict(color=color, width=6), connectgaps=False, hoverinfo='none'))

    # 4. Add Trajectories
    moving_tids = {d['tid'] for d in current_dets if d.get('moving', False)}
    for tid in sorted(list(moving_tids)):
        traj_x, traj_y, traj_z = [], [], []
        for i in range(frame_index_to_render + 1):
            for d in det_frames[i]:
                if d['tid'] == tid:
                    traj_x.append(d['cx']); traj_y.append(d['cy']); traj_z.append(-6.5)
                    break
        if len(traj_x) >= 2:
            fig.add_trace(go.Scatter3d(
                x=traj_x, y=traj_y, z=traj_z,
                mode='lines', name=f'Track {tid}',
                line=dict(color='magenta', width=3),
                showlegend=False
            ))

    road_poly = results['road_poly']
    minx, miny, maxx, maxy = road_poly.bounds
    buffer_x, buffer_y = 5.0, 10.0
    
    layout_dict = dict(
        margin=dict(l=0, r=0, b=0, t=40),
        title=f"Frame {frame_index_to_render}",
        scene=dict(
            xaxis=dict(title='X (m)', range=[minx - buffer_x, maxx + buffer_x]),
            yaxis=dict(title='Y (m)', range=[miny - buffer_y, maxy + buffer_y]),
            zaxis=dict(title='Z (m)', range=[-15, 10]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.15)
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    if results.get('top_down'):
        # Straight-down bird's-eye: look along -Z with +Y as screen-up so the
        # view matches the X/Y sensor grid (best for verifying lane alignment).
        layout_dict['scene']['camera'] = {
            'up': {'x': 0, 'y': 1, 'z': 0},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'eye': {'x': 0, 'y': 0, 'z': 2.5}
        }
    else:
        camera_eye = results.get('camera_eye', {'x': 1.25, 'y': 1.25, 'z': 1.25})
        layout_dict['scene']['camera'] = {
            'up': {'x': 0, 'y': 0, 'z': 1},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'eye': camera_eye
        }
    fig.update_layout(**layout_dict)
    return fig

def generate_tracking_animation(results, output_gif_path, progress_callback=None, max_frames=0):
    """
    Generate tracking animation using Matplotlib 3D engine, 100% aligned with Plotly interactive view visual details and perspective.
    Perspective converted from camera_eye parameters to ensure 100% alignment and support loop playback.
    """
    pcd_files = results['original_pcd_files']
    det_frames = results['det_frames']
    road_poly = results['road_poly']
    
    if road_poly.geom_type == 'Polygon': polys = [road_poly]
    else: polys = road_poly.geoms

    all_frames = len(det_frames)
    num_frames = min(max_frames, all_frames) if max_frames > 0 else all_frames
    
    params = results.get('params', {})
    speed_threshold = params.get('moving_speed_thresh', 3.0)
    minx, miny, maxx, maxy = road_poly.bounds
    buffer_x, buffer_y = 5.0, 10.0

    camera_eye = results.get('camera_eye', {'x': 1.25, 'y': 1.25, 'z': 1.25})
    # Correct camera mapping logic
    ex, ey, ez = camera_eye['x'], camera_eye['y'], camera_eye['z']
    # Precise conversion: Plotly camera_eye [1.25, 1.25, 1.25] mapped to Matplotlib elevation/azimuth
    azim = np.degrees(np.arctan2(ey, ex)) # 45 degrees
    dist_xy = np.sqrt(ex**2 + ey**2)
    elev = np.degrees(np.arctan2(ez, dist_xy)) # approx 35 degrees

    # Use white background
    images = []
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    
    # Use imageio writer mode to ensure multi-frame stream writing
    # Completely solve the issue of "only displaying the first or last frame"
    with imageio.get_writer(output_gif_path, mode='I', fps=10, loop=0) as writer:
        for i in range(num_frames):
            plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('white')
            
            # 1. Style alignment: transparent panes, remove borders
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('white')
            ax.yaxis.pane.set_edgecolor('white')
            ax.zaxis.pane.set_edgecolor('white')
            
            # 2. Road lines (Green, aligned with Plotly thickness)
            for poly in polys:
                x, y = poly.exterior.xy
                z = np.full_like(x, -7.5)
                ax.plot(x, y, z, color='#2ca02c', linewidth=1.5, alpha=0.8)
                
            # 3. Point cloud (Grey points, aligned with Plotly texture)
            pcd = o3d.io.read_point_cloud(pcd_files[i])
            pts = np.asarray(pcd.points)
            if pts.size > 0:
                sample_size = min(10000, pts.shape[0])
                idx = np.random.choice(pts.shape[0], sample_size, replace=False)
                ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], s=0.2, c='#a9a9a9', alpha=0.3, depthshade=True)
                
            # 4. Detected objects and trajectories (Red dots, magenta lines;
            #    wrong-way vehicles in orange with a warning label + heading arrow)
            current_dets = det_frames[i]
            for d in current_dets:
                is_ww = d.get('wrong_way', False)
                if is_ww:
                    ax.scatter([d['cx']], [d['cy']], [-6.5], s=90, c='orange',
                               marker='D', edgecolors='black', linewidths=1.0, alpha=0.95)
                else:
                    mcol = object_marker_color(d, speed_threshold)
                    ax.scatter([d['cx']], [d['cy']], [-6.5], s=30, c=mcol, edgecolors='none', alpha=0.9)
                speed = d.get('speed', 0.0)
                base = f"ID:{d['tid']}\n{speed:.1f}m/s" if speed >= speed_threshold else f"ID:{d['tid']}"
                label = ("WRONG WAY\n" + base) if is_ww else base
                ax.text(d['cx'], d['cy'], -5.5, label, fontsize=7, fontweight='bold' if is_ww else 'normal',
                        color='darkorange' if is_ww else 'black', ha='center')

                # Heading arrow (true travel direction), color-encoded by cardinal
                # direction with an arrowhead.
                hdg = d.get('heading', None)
                if speed >= speed_threshold and hdg is not None:
                    col = cardinal_color(hdg)
                    xs, ys, zs = _arrow_segments(d['cx'], d['cy'], hdg, -6.5)
                    seg_x, seg_y, seg_z = [], [], []
                    for px, py, pz in zip(xs, ys, zs):
                        if px is None:
                            if seg_x:
                                ax.plot(seg_x, seg_y, seg_z, color=col, linewidth=2.0, alpha=0.95)
                            seg_x, seg_y, seg_z = [], [], []
                        else:
                            seg_x.append(px); seg_y.append(py); seg_z.append(pz)
                    if seg_x:
                        ax.plot(seg_x, seg_y, seg_z, color=col, linewidth=2.0, alpha=0.95)

                if d.get('moving', False):
                    tid = d['tid']
                    traj_x, traj_y, traj_z = [], [], []
                    for prev_i in range(i + 1):
                        for prev_d in det_frames[prev_i]:
                            if prev_d['tid'] == tid:
                                traj_x.append(prev_d['cx']); traj_y.append(prev_d['cy']); traj_z.append(-6.5)
                                break
                    if len(traj_x) >= 2:
                        ax.plot(traj_x, traj_y, traj_z, color='magenta', linewidth=1.2, alpha=0.7)
            
            # 5. Perspective and scale alignment
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(minx - buffer_x, maxx + buffer_x)
            ax.set_ylim(miny - buffer_y, maxy + buffer_y)
            ax.set_zlim(-15, 10)
            ax.set_box_aspect([1, 1, 0.15])
            
            # Hide black axis lines, keep light grey grid
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.grid(True, linestyle='-', color='#dddddd', linewidth=0.5)
            
            ax.set_xlabel('X (m)', fontsize=8, labelpad=-10)
            ax.set_ylabel('Y (m)', fontsize=8, labelpad=-10)
            ax.set_zlabel('Z (m)', fontsize=8, labelpad=-10)
            ax.set_title(f"Frame {i}", color='black', fontsize=10, y=0.95)

            # Cardinal-direction colour key for the heading arrows.
            handles = [Line2D([0], [0], color=c, lw=2, label=n) for n, c in CARDINAL_BINS]
            ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.6,
                      title='Heading', title_fontsize=7)
            
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3].copy()) # Important: must use .copy()
            
            if progress_callback:
                progress_callback(i + 1, num_frames)
        
        plt.close(fig)
            
    return output_gif_path
