"""Per-dataset workspace management.

Lets the app work with multiple datasets (templates + user-supplied), each with
its OWN inputs, config (lane geometry / site geometry), background model,
outputs and settings — so switching datasets preserves every dataset's state.

Layout
------
A dataset has *input* dirs (raw PCDs + optional GT, which may live anywhere on
disk) and a *workspace* root that holds everything the app generates/edits:

    <workspace>/config/   lanes.geojson, site_geometry.json
    <workspace>/outputs/  background_model/, background_filtering/, object_detection/
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
        "name": "TUMTraf A9 r02_s02 (s110 ouster south) — template",
        "template": True,
        "pcd_dir": "datasets/A9_r02_s02/data/derived/cropped_south/cropped_pcd",
        "gt_dir": "datasets/A9_r02_s02/data/derived/labels_visible_south",
        "workspace": "datasets/A9_r02_s02",
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

    def _north_gt_dir(self):
        """North GT: prefer the generated scorable set, else the raw north labels."""
        scorable = os.path.join(self.derived_dir, "labels_visible_north")
        try:
            if os.path.isdir(scorable) and any(f.endswith(".json") for f in os.listdir(scorable)):
                return scorable
        except OSError:
            pass
        return self.raw_labels_north_dir

    def gt_dir_for_input(self, input_pcd_dir):
        """Pick the GT folder whose sensor matches the input clouds, so the GT
        overlay + FG-quality metric line up no matter which sensor is filtered.
        North inputs -> north GT; everything else -> the configured `gt_dir`
        (south). Lets south/north A/B work without hand-editing `gt_dir`."""
        if "north" in (input_pcd_dir or "").lower():
            return self._north_gt_dir()
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
    def derived_dir(self):
        return os.path.join(self.data_dir, "derived")

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
    def model_path(self):
        return os.path.join(self.outputs_dir, "background_model", "background_model.pkl")

    @property
    def filtered_dir(self):
        return os.path.join(self.outputs_dir, "background_filtering")

    @property
    def detection_dir(self):
        return os.path.join(self.outputs_dir, "object_detection")

    @property
    def settings_path(self):
        return os.path.join(self.workspace, "settings.json")

    # --- pipeline input source (cropped road vs full/uncropped) for A/B eval ---
    # Each source writes to its own model/filtered/detection folders so the two
    # don't clash and their eval metrics can be compared.
    def _sfx(self, source):
        return "" if source == "cropped" else "_full"

    def input_pcd_for(self, source):
        return self.pcd_dir if source == "cropped" else self.raw_lidar_south_dir

    def model_path_for(self, source):
        return os.path.join(self.outputs_dir, "background_model" + self._sfx(source), "background_model.pkl")

    def filtered_dir_for(self, source):
        return self.filtered_dir + self._sfx(source)

    def detection_dir_for(self, source):
        return self.detection_dir + self._sfx(source)

    def ensure_workspace(self):
        """Create the workspace folders (config/outputs) for a user dataset."""
        for p in (self.config_dir, os.path.join(self.outputs_dir, "background_model"),
                  self.filtered_dir, self.detection_dir):
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
