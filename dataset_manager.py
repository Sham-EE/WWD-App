"""Per-dataset workspace management.

Lets the app work with multiple datasets (templates + user-supplied), each with
its OWN inputs, config (lane geometry / site geometry), background model,
outputs and settings — so switching datasets preserves every dataset's state.

Layout
------
A dataset has *input* dirs (raw PCDs + optional GT, which may live anywhere on
disk) and a *workspace* root that holds everything the app generates/edits:

    <workspace>/config/   lanes.geojson, site_geometry.json
    <workspace>/outputs/  background/{background_model,background_filtering}/<sensor>/<crop>/,
                          detection/object_detection/<sensor>/<crop>/, visualizer/{rendered,road_videos}/
    <workspace>/settings.json

The built-in **TUMTraf** template maps to the project's existing top-level
`config/` and `outputs/` (workspace = ".") so nothing has to move and current
behaviour is unchanged. User datasets get their own `datasets/<id>/` workspace.

This module only RESOLVES paths and manages the registry; wiring the pipeline
pages to use the active dataset is done separately.
"""
import datetime
import glob
import json
import os

# Session-state keys that belong to ONE dataset's run; cleared when switching so
# stale results from another dataset never linger in Detection/Eval/Simulator.
SESSION_KEYS_TO_CLEAR = [
    "detection_results", "bg_model",
    "le_lanes", "le_v", "le_points",
    "sim_step", "v2x_armed", "v2x_event", "ev_frame", "odt_frame",
]

_ROOT = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(_ROOT, "datasets")
REGISTRY_PATH = os.path.join(DATASETS_DIR, "registry.json")

# Built-in template(s). Paths are relative to the repo root (the app's CWD).
TEMPLATES = [
    {
        "id": "A9_r02_s02",
        "name": "TUMTraf s110 · Garching intersection (A9 r02_s02) — template",
        "template": True,
        "pcd_dir": "datasets/A9_r02_s02/data/derived/point_clouds/cropped/south",
        "gt_dir": "datasets/A9_r02_s02/data/derived/labels/scorable/south",
        "workspace": "datasets/A9_r02_s02",
        "description": (
            "Real TUM Traffic intersection — sensor station **s110** at "
            "Schleißheimer Str. (B471) × Zeppelinstr., Garching-Hochbrück, Munich "
            "(48.2494 °N, 11.6308 °E). Ships curated derived data: road-cropped **south** "
            "LiDAR + visible-only **scorable GT**, **WWD lane** directions and **site "
            "geometry**, a south+north **registered** (fused) cloud, **georeferencing** "
            "config (exact WGS84 + true compass bearings) and the **HD-map** road network "
            "for the digital twin. The large raw LiDAR/image data is downloaded separately "
            "into `data/` — the template config already points at it."
        ),
    },
]


class Dataset:
    """Resolves all paths for one dataset from its input dirs + workspace root."""

    def __init__(self, d):
        self.d = dict(d)

    # --- identity ---
    @property
    def id(self):
        return self.d["id"]

    @property
    def name(self):
        return self.d.get("name", self.d["id"])

    @property
    def is_template(self):
        return bool(self.d.get("template", False))

    @property
    def workspace(self):
        return self.d.get("workspace") or os.path.join("datasets", self.id)

    # --- inputs ---
    @property
    def pcd_dir(self):
        return self.d.get("pcd_dir", os.path.join(self.workspace, "data", "point_clouds"))

    @property
    def gt_dir(self):
        return self.d.get("gt_dir", os.path.join(self.workspace, "data", "labels"))

    def _has_json(self, d):
        try:
            return os.path.isdir(d) and any(f.endswith(".json") for f in os.listdir(d))
        except OSError:
            return False

    def _north_gt_dir(self):
        """North GT: prefer the generated scorable set, else the raw north labels."""
        scorable = self.scorable_gt_dir_for("north")
        return scorable if self._has_json(scorable) else self.raw_labels_north_dir

    def _registered_gt_dir(self):
        """Registered GT, best available, in order:
        1. the scorable registered set (fused south∪north, visibility-filtered),
        2. the fused raw registered labels (south∪north union, before filtering),
        3. the south GT (registered is in the south frame, so south boxes apply —
           but this MISSES objects only north saw; build the fused GT to fix that).
        """
        scorable = self.scorable_gt_dir_for("registered")
        if self._has_json(scorable):
            return scorable
        fused = os.path.join(self.derived_dir, "labels", "registered")
        return fused if self._has_json(fused) else self.gt_dir

    def gt_dir_for_input(self, input_pcd_dir):
        """Pick the GT folder whose sensor matches the input clouds, so the GT
        overlay + FG-quality metric line up no matter which sensor is filtered.
        North/registered inputs -> their GT; everything else -> the configured
        `gt_dir` (south). Lets south/north/registered A/B work without hand-
        editing `gt_dir`."""
        p = (input_pcd_dir or "").lower()
        if "registered" in p:
            return self._registered_gt_dir()
        if "north" in p:
            return self._north_gt_dir()
        return self.gt_dir

    def labels_dir_for(self, sensor, kind="scorable"):
        """Label folder for a sensor and kind:
        - ``kind="raw"``     -> EVERY annotated box (pre visibility/region filter);
          registered = the fused south∪north union (falls back to south raw).
        - ``kind="scorable"``-> the visibility/region-filtered set used by Evaluation
          (same fallbacks as ``gt_dir_for_input``).
        """
        if kind == "raw":
            if sensor == "north":
                return self.raw_labels_north_dir
            if sensor == "registered":
                fused = os.path.join(self.derived_dir, "labels", "registered")
                return fused if self._has_json(fused) else self.raw_labels_south_dir
            return self.raw_labels_south_dir
        if sensor == "north":
            return self._north_gt_dir()
        if sensor == "registered":
            return self._registered_gt_dir()
        return self.gt_dir

    @property
    def images_dir(self):
        return self.d.get("images_dir", os.path.join(self.data_dir, "raw", "images"))

    # --- RAW inputs (the untouched TUM Traffic download, under data/raw/) ---
    @property
    def data_dir(self):
        return os.path.join(self.workspace, "data")

    @property
    def raw_dir(self):
        return os.path.join(self.data_dir, "raw")

    @property
    def map_dir(self):
        """Per-dataset HD-map dir. For TUMTraf this holds lane_samples.json (from the
        dev-kit's src/map/map.zip) — used by geo_reference for the georeferenced
        overlays. Optional: features degrade gracefully if it's absent."""
        return self.d.get("map_dir", os.path.join(self.workspace, "map"))

    @property
    def hdmap_path(self):
        return os.path.join(self.map_dir, "lane_samples.json")

    @property
    def derived_dir(self):
        return os.path.join(self.data_dir, "derived")

    # --- DERIVED data, nested to mirror raw/ (point_clouds/ + labels/) ---
    #   derived/point_clouds/registered/*.pcd               (fused, south LiDAR frame)
    #   derived/point_clouds/cropped/<sensor>/*.pcd         (road-clipped)
    #   derived/labels/scorable/<sensor>/*.json             (visible-only GT)
    @property
    def registered_dir(self):
        return os.path.join(self.derived_dir, "point_clouds", "registered")

    def cropped_dir_for(self, sensor):
        return os.path.join(self.derived_dir, "point_clouds", "cropped", sensor)

    def scorable_gt_dir_for(self, sensor):
        return os.path.join(self.derived_dir, "labels", "scorable", sensor)

    @property
    def raw_lidar_south_dir(self):
        return self.d.get("raw_lidar_south_dir",
                          os.path.join(self.raw_dir, "point_clouds", "s110_lidar_ouster_south"))

    @property
    def raw_lidar_north_dir(self):
        return self.d.get("raw_lidar_north_dir",
                          os.path.join(self.raw_dir, "point_clouds", "s110_lidar_ouster_north"))

    @property
    def raw_labels_south_dir(self):
        return self.d.get("raw_labels_south_dir",
                          os.path.join(self.raw_dir, "labels", "s110_lidar_ouster_south"))

    @property
    def raw_labels_north_dir(self):
        return self.d.get("raw_labels_north_dir",
                          os.path.join(self.raw_dir, "labels", "s110_lidar_ouster_north"))

    # --- workspace (generated / edited) ---
    @property
    def config_dir(self):
        return os.path.join(self.workspace, "config")

    @property
    def outputs_dir(self):
        return os.path.join(self.workspace, "outputs")

    @property
    def lanes_path(self):
        return os.path.join(self.config_dir, "lanes.geojson")

    @property
    def site_geometry_path(self):
        return os.path.join(self.config_dir, "site_geometry.json")

    # --- factory-default snapshots (config/defaults/), for "reset to default" ---
    @property
    def defaults_dir(self):
        return os.path.join(self.config_dir, "defaults")

    @property
    def default_site_geometry_path(self):
        return os.path.join(self.defaults_dir, "site_geometry.json")

    @property
    def default_lanes_path(self):
        return os.path.join(self.defaults_dir, "lanes.geojson")

    @property
    def settings_path(self):
        return os.path.join(self.workspace, "settings.json")

    # --- visualizer outputs (camera-overlay cache + road videos) ---
    @property
    def visualizer_dir(self):
        return os.path.join(self.outputs_dir, "visualizer")

    @property
    def rendered_dir(self):
        return os.path.join(self.visualizer_dir, "rendered")

    @property
    def road_videos_dir(self):
        return os.path.join(self.visualizer_dir, "road_videos")

    # --- pipeline INPUT source (cropped road vs full/uncropped), per sensor ---
    def _crop(self, source):
        return "cropped" if source == "cropped" else "full"

    def _dir_has_pcd(self, d):
        try:
            return os.path.isdir(d) and any(f.endswith(".pcd") for f in os.listdir(d))
        except OSError:
            return False

    def input_pcd_for(self, source):
        return self.pcd_dir if source == "cropped" else self.raw_lidar_south_dir

    def input_pcd_for_sensor(self, sensor, source):
        if sensor == "north":
            return (self.cropped_dir_for("north")
                    if source == "cropped" else self.raw_lidar_north_dir)
        if sensor == "registered":
            crop = self.cropped_dir_for("registered")
            if source == "cropped":
                # fall back to the full fused cloud if it hasn't been cropped yet
                return crop if self._dir_has_pcd(crop) else self.registered_dir
            return self.registered_dir
        return self.input_pcd_for(source)  # south

    # --- per-SENSOR OUTPUT paths, nested so outputs/ stays tidy ---
    #   outputs/background/background_model/<sensor>/<crop>/background_model.pkl
    #   outputs/background/background_filtering/<sensor>/<crop>/*.pcd
    #   outputs/detection/object_detection/<sensor>/<crop>/  (tracks.csv, gif, eval report)
    # <sensor> is south|north|registered, <crop> is cropped|full — so every
    # combination has its own folder and all results coexist for comparison.
    def model_path_for_sensor(self, sensor, source):
        return os.path.join(self.outputs_dir, "background", "background_model",
                            sensor, self._crop(source), "background_model.pkl")

    def filtered_dir_for_sensor(self, sensor, source):
        return os.path.join(self.outputs_dir, "background", "background_filtering",
                            sensor, self._crop(source))

    def detection_dir_for_sensor(self, sensor, source):
        return os.path.join(self.outputs_dir, "detection", "object_detection",
                            sensor, self._crop(source))

    # back-compat bare helpers/properties default to the SOUTH sensor.
    def model_path_for(self, source):
        return self.model_path_for_sensor("south", source)

    def filtered_dir_for(self, source):
        return self.filtered_dir_for_sensor("south", source)

    def detection_dir_for(self, source):
        return self.detection_dir_for_sensor("south", source)

    @property
    def model_path(self):
        return self.model_path_for_sensor("south", "cropped")

    @property
    def filtered_dir(self):
        return self.filtered_dir_for_sensor("south", "cropped")

    @property
    def detection_dir(self):
        return self.detection_dir_for_sensor("south", "cropped")

    def ensure_workspace(self):
        """Create the workspace folders (config/outputs) for a user dataset."""
        for p in (self.config_dir, os.path.dirname(self.model_path),
                  self.filtered_dir, self.detection_dir,
                  self.rendered_dir, self.road_videos_dir):
            os.makedirs(p, exist_ok=True)

    def status(self):
        """Quick presence flags for the UI."""
        def has_pcd(p):
            try:
                return os.path.isdir(p) and any(f.endswith(".pcd") for f in os.listdir(p))
            except Exception:
                return False
        return {
            "pcd": has_pcd(self.pcd_dir),
            "gt": os.path.isdir(self.gt_dir or ""),
            "lanes": os.path.exists(self.lanes_path),
            "model": os.path.exists(self.model_path),
            "filtered": has_pcd(self.filtered_dir),
        }

    def to_dict(self):
        return dict(self.d)


# ---------------- registry ----------------

def _default_registry():
    return {"active": TEMPLATES[0]["id"], "datasets": []}


def load_registry():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    reg = _default_registry()
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                stored = json.load(f)
            reg["active"] = stored.get("active", reg["active"])
            reg["datasets"] = [d for d in stored.get("datasets", []) if not d.get("template")]
        except Exception:
            pass
    return reg


def save_registry(reg):
    os.makedirs(DATASETS_DIR, exist_ok=True)
    # never persist templates (they're code-defined)
    out = {"active": reg.get("active"), "datasets": [d for d in reg.get("datasets", []) if not d.get("template")]}
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def all_datasets():
    """Templates first, then user datasets."""
    reg = load_registry()
    return [Dataset(t) for t in TEMPLATES] + [Dataset(d) for d in reg["datasets"]]


def get_dataset(ds_id):
    for ds in all_datasets():
        if ds.id == ds_id:
            return ds
    return None


def get_active():
    reg = load_registry()
    ds = get_dataset(reg.get("active"))
    return ds or Dataset(TEMPLATES[0])


def set_active(ds_id):
    reg = load_registry()
    if get_dataset(ds_id) is not None:
        reg["active"] = ds_id
        save_registry(reg)


def _slugify(name):
    s = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
    return s or "dataset"


def derive_site_geometry(pcd_dir, max_frames=10, pad=3.0):
    """Build a starter site_geometry from the PCD extent: a research polygon
    covering the data (padded), the whole area as one road polygon, no exclusion
    rects. Lets a brand-new dataset run immediately; the user refines later."""
    import numpy as np
    try:
        import open3d as o3d
        files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
        xs, ys = [], []
        if files:
            idxs = np.unique(np.linspace(0, len(files) - 1, min(max_frames, len(files))).astype(int))
            for i in idxs:
                p = np.asarray(o3d.io.read_point_cloud(files[int(i)]).points)
                if p.size:
                    xs += [float(p[:, 0].min()), float(p[:, 0].max())]
                    ys += [float(p[:, 1].min()), float(p[:, 1].max())]
        if xs:
            x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
        else:
            x0, x1, y0, y1 = -50.0, 50.0, -50.0, 50.0
    except Exception:
        x0, x1, y0, y1 = -50.0, 50.0, -50.0, 50.0
    rect = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {
        "_comment": "Auto-derived from PCD extent on dataset creation. Refine the "
                    "research polygon / road polygons / exclusion rects as needed.",
        "research_polygon": rect,
        "road_polygons": [rect],
        "foreground_exclusion_rects": [],
        "coarse_grid": {"NX": 5, "NY": 5},
    }


def create_dataset(name, pcd_dir, gt_dir="", description=""):
    """Register a user dataset that points at an on-disk PCD folder (not copied),
    with its own workspace under datasets/<id>/. Scaffolds a starter
    site_geometry.json (from the data extent) + a README. Returns the Dataset."""
    reg = load_registry()
    existing = {d["id"] for d in reg["datasets"]} | {t["id"] for t in TEMPLATES}
    base = _slugify(name)
    ds_id = base
    n = 2
    while ds_id in existing:
        ds_id = f"{base}_{n}"; n += 1
    d = {"id": ds_id, "name": name.strip() or ds_id, "template": False,
         "pcd_dir": pcd_dir.strip(), "gt_dir": gt_dir.strip(),
         "workspace": os.path.join("datasets", ds_id),
         "description": description.strip(),
         "created": datetime.datetime.now().isoformat(timespec="seconds")}
    ds = Dataset(d)
    ds.ensure_workspace()
    # scaffold starter geometry + readme so the dataset is immediately runnable
    try:
        starter = derive_site_geometry(pcd_dir)
        with open(ds.site_geometry_path, "w", encoding="utf-8") as f:
            json.dump(starter, f, indent=2)
        # snapshot it as the factory default so "reset to default" works
        os.makedirs(ds.defaults_dir, exist_ok=True)
        with open(ds.default_site_geometry_path, "w", encoding="utf-8") as f:
            json.dump(starter, f, indent=2)
    except Exception:
        pass
    try:
        with open(os.path.join(ds.workspace, "README.md"), "w", encoding="utf-8") as f:
            f.write(f"# {d['name']}\n\nCreated {d['created']}.\n\n"
                    f"- PCD frames: `{pcd_dir}`\n- GT labels: `{gt_dir or '(none)'}`\n\n"
                    "config/ (lane geometry, site geometry) and outputs/ (model, filtered clouds, "
                    "detection results) for this dataset live in this folder.\n")
    except Exception:
        pass
    reg["datasets"].append(d)
    reg["active"] = ds_id
    save_registry(reg)
    return ds


def rename_dataset(ds_id, new_name):
    reg = load_registry()
    for d in reg["datasets"]:
        if d["id"] == ds_id:
            d["name"] = new_name.strip() or d["name"]
            save_registry(reg)
            return


def delete_dataset(ds_id):
    """Remove a user dataset from the registry (does NOT delete its files)."""
    reg = load_registry()
    reg["datasets"] = [d for d in reg["datasets"] if d["id"] != ds_id]
    if reg.get("active") == ds_id:
        reg["active"] = TEMPLATES[0]["id"]
    save_registry(reg)
