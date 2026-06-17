"""Helpers for the Road Viewer page.

Browses a dataset's camera images (multiple cameras side by side, multiple
visualization variants) and renders a continuous side-by-side video of the road.
"""
import glob
import os

# Visualization variants -> the subfolder convention used in this data.
# 'raw' is whatever subfolder isn't bb/bb_pcd (the long camera-named one), or the
# camera folder itself if images sit directly in it.
VARIANTS = {
    "Raw camera": "raw",
    "Bounding boxes": "bb",
    "Boxes + point cloud": "bb_pcd",
}


def _frame_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(c for c in base if c.isdigit())
    return int(digits) if digits else 0


def list_cameras(images_root):
    """Camera subfolders under the images root (e.g. south1, south2)."""
    if not images_root or not os.path.isdir(images_root):
        return []
    return sorted(d for d in os.listdir(images_root)
                  if os.path.isdir(os.path.join(images_root, d)))


def _images_in(d):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files += glob.glob(os.path.join(d, ext))
    return sorted(files, key=_frame_key)


def variant_dir(camera_dir, variant_key):
    """Resolve a camera folder + variant to the folder holding those images."""
    if not os.path.isdir(camera_dir):
        return None
    subs = [d for d in os.listdir(camera_dir) if os.path.isdir(os.path.join(camera_dir, d))]
    if variant_key in ("bb", "bb_pcd") and variant_key in subs:
        return os.path.join(camera_dir, variant_key)
    if variant_key == "raw":
        raw = [d for d in subs if d not in ("bb", "bb_pcd")]
        if raw:
            return os.path.join(camera_dir, raw[0])
        if _images_in(camera_dir):       # images directly in the camera folder
            return camera_dir
    # fallbacks: requested variant missing -> raw -> camera folder
    raw = [d for d in subs if d not in ("bb", "bb_pcd")]
    if raw:
        return os.path.join(camera_dir, raw[0])
    return camera_dir if _images_in(camera_dir) else None


def frames_for(images_root, camera, variant_key):
    d = variant_dir(os.path.join(images_root, camera), variant_key)
    return _images_in(d) if d else []


def available_variants(images_root, camera):
    """Which of the known variants actually exist for a camera (label list)."""
    out = []
    for label, key in VARIANTS.items():
        if frames_for(images_root, camera, key):
            out.append(label)
    return out or list(VARIANTS.keys())


def combine_side_by_side(left_path, right_path, height=480, gap=8):
    """One image: left | right, scaled to a common height (black background)."""
    from PIL import Image
    ims = []
    for p in (left_path, right_path):
        try:
            ims.append(Image.open(p).convert("RGB"))
        except Exception:
            ims.append(Image.new("RGB", (height, height), "black"))
    scaled = [im.resize((max(1, int(im.width * height / im.height)), height)) for im in ims]
    W = sum(im.width for im in scaled) + gap
    canvas = Image.new("RGB", (W, height), "black")
    x = 0
    for im in scaled:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def generate_side_by_side_video(left_frames, right_frames, out_path, fps=10,
                                height=480, max_frames=0, progress=None):
    """Write an MP4 of left|right frames. Returns the path."""
    import imageio.v2 as imageio
    import numpy as np
    n = min(len(left_frames), len(right_frames))
    if max_frames and max_frames > 0:
        n = min(n, max_frames)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                macro_block_size=16, quality=7)
    try:
        for i in range(n):
            frame = combine_side_by_side(left_frames[i], right_frames[i], height=height)
            writer.append_data(np.asarray(frame))
            if progress:
                progress(i + 1, n)
    finally:
        writer.close()
    return out_path
