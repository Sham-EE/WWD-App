# V2X dashboard integration

Save your **WWD V2X Dashboard** single-file app here as:

```
assets/wwd_v2x_dashboard.html
```

The WWD Simulator page loads this exact file, injects a small trigger, and
embeds it. When the simulator's detector flags the wrong-way driver and you
click **“Broadcast to V2X Dashboard”**, the embedded app **auto-fires its alert
pipeline** (J2735 TIM 8708 → C-V2X broadcast → nav push → law enforcement) using
the detected speed/heading.

Notes
- Keep your original file (the integration does not modify it on disk — the
  trigger is injected in-memory at load time).
- The dashboard fires against its **currently-selected intersection** (default:
  the Houston OSM site). To make the alert use your real intersection, add it in
  the dashboard's *Add Intersection* tab (Snap to OSM one-way road) and select it
  — there is no georeference from the LiDAR sensor frame to GPS in this app.
- The embedded app loads Leaflet/Tailwind/React from CDNs, so it needs internet.
