# V2X dashboard integration

Save your **WWD V2X Dashboard** single-file app here as:

```
assets/index.html
```

The **V2X Dashboard** page (and the WWD Simulator's *Broadcast*) load this exact
file, inject a small in-memory trigger, and embed it.

What the injection does: it passes the detector's event into the dashboard and —
critically — **replaces the dashboard's intersection with the exact live scenario**.
`build_v2x_intersection()` turns the wrong-way lane into a true-lat/lon centerline
(from the LiDAR→GPS georeference), injects it via `setIntersections` +
`setCurrentId`, and fires the alert pipeline (J2735 TIM 8708 → C-V2X → law
enforcement) on it. The file's own built-in default is already the real TUMTraf
s110 site (not a generic placeholder), so the injection's job is swapping in the
*specific* scenario/lane/driver position, not switching sites.

Also here: `favicon.png` — the app's tab icon, referenced as `page_icon=` on
every page.

Notes
- The integration does **not** modify this file on disk — the trigger is injected
  in-memory at load time, so your original app is untouched.
- Pick which lane the wrong-way driver runs from the scenario selector on the
  V2X Dashboard page.
- The embedded app loads Leaflet/Tailwind/React/Babel from CDNs, so it needs internet.
