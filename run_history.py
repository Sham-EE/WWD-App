"""Lightweight run history so you can SEE whether a tuning change actually helped.

Each filtering "evaluation" (current model + config scored over a sample of frames
via the foreground-quality proxy) appends one JSON line to
  <workspace>/outputs/run_history/<tag>.jsonl
where <tag> encodes the sensor + input-source so south / north / registered and
cropped / full each keep their own trend. The Background Filtering page reads this
back to show current-vs-previous deltas + a trend chart, so you don't have to
remember last run's numbers.
"""
import os
import json
import time


def _dir(ds):
    return os.path.join(ds.outputs_dir, "run_history")


def _path(ds, tag):
    return os.path.join(_dir(ds), f"{tag}.jsonl")


# Only the knobs worth tracking — paths/frame-counts are noise for a tuning trend.
_TRACKED_KEYS = (
    "ground_grid", "dz_thresh", "bg_voxel", "bg_ratio", "cell_size", "cell_ratio",
    "inward_buffer_m", "enable_pole_filter", "enable_5x5", "enable_sor",
    "sor_k", "sor_std",
)
_TRACKED_CLUSTER = (
    "mode", "ds_voxel", "eps0", "eps_k", "eps_min", "eps_max", "eps_scale",
    "min_samples", "min_samples_far", "n_tiers",
)


def summarize_params(config):
    """Flatten the tuning-relevant subset of a filter config for logging/diffing."""
    out = {k: config.get(k) for k in _TRACKED_KEYS if k in config}
    cl = config.get("cluster", {}) or {}
    for k in _TRACKED_CLUSTER:
        if k in cl:
            out[f"cluster.{k}"] = cl[k]
    return out


def log_run(ds, tag, metrics, params, note=""):
    """Append one run record. metrics + params are plain dicts of JSON-able values."""
    os.makedirs(_dir(ds), exist_ok=True)
    rec = {
        "ts": time.time(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
        "metrics": metrics,
        "params": params,
    }
    with open(_path(ds, tag), "a", encoding="utf-8") as fp:
        fp.write(json.dumps(rec) + "\n")
    return rec


def load_history(ds, tag):
    """Return the run records (oldest first); [] if none yet."""
    p = _path(ds, tag)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def clear_history(ds, tag):
    p = _path(ds, tag)
    if os.path.exists(p):
        os.remove(p)


def param_diff(prev_params, cur_params):
    """Keys whose value changed between two summarize_params() dicts -> (old, new)."""
    keys = set(prev_params or {}) | set(cur_params or {})
    diff = {}
    for k in sorted(keys):
        a = (prev_params or {}).get(k)
        b = (cur_params or {}).get(k)
        if a != b:
            diff[k] = (a, b)
    return diff
