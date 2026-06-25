# Results & Ablations — Multi-LiDAR Wrong-Way Detection

Paper-ready evidence base for the s110 (TUMTraf A9) pipeline. Every number below is from
a controlled run on the same dataset (`A9_r02_s02`, 282 frames). Unless noted: registered
(south+north fused) cloud, **cropped to road**, ROI on, vehicle classes only, BEV-centre
match gate 2.0 m, identical detector. Detection is deterministic (same settings → same
numbers).

> **All tables re-baselined together** under (a) the **corrected gated matcher** — the
> Hungarian assignment now applies the distance gate *before* solving, so dense scenes no
> longer strand a genuinely-close detection/GT pair as both a miss *and* a false positive;
> and (b) the **current site geometry**, including the **exclusion zones** placed over the
> fixed static infrastructure (poles/barriers). So every absolute number here is mutually
> consistent.

> Headline arc: **fusion fills occlusion shadows (recall) → cropped input + exclusion zones
> + static-suppression remove the false positives (precision) → the classical detector is
> then Pareto-tuned** (further gains need a learned detector, not parameter tuning).

**Current registered/cropped baseline:** **P 0.691 / R 0.654 / F1 0.672** (veh-only);
**P 0.723 / R 0.682 / F1 0.702** (all classes). MOTP ≈ 1.2 m.

---

## 1. Does fusion help? (Registered vs South A/B)

Same detection on each sensor's filtered cloud, graded against the **shared registered-union
scorable GT** (so south is fairly penalised for the north-only vehicles it physically can't
see; both share one denominator). Cropped, ROI, veh-only.

| pipeline | Precision | Recall | F1 |
|---|---|---|---|
| South | **0.745** | 0.505 | 0.602 |
| **Registered** | 0.691 | **0.654** | **0.672** |

**Registered now wins both recall and F1.** Fusion's core benefit is **recall +15 pts
(0.505 → 0.654)** — it sees vehicles a single sensor physically can't (occlusion shadows),
which is the metric that matters for not *missing* a wrong-way driver. South keeps a small
precision edge (a sparser cloud yields fewer candidates), but exclusion zones + static
suppression close registered's old precision gap, so registered leads on F1 too. Earlier,
the `truck_merge_dist = 10 m` default over-merged adjacent vehicles in the denser fused
far-field; **lowering it to 5 m** recovered far-field recall and was the fix that turned
"should help" into a measured win.

---

## 2. Background-filter ablation (is the filter the bottleneck?)

Registered, **full** research region, all classes, shared scorable GT — same detector.
Tests the BG-occupancy knobs we added (density-adaptive clustering, SOR denoise).

| BG-filter variant | Precision | Recall | F1 |
|---|---|---|---|
| Global clusterer | 36.5 | 67.8 | 47.4 |
| Density-adaptive clusterer | 35.8 | **68.3** | 47.0 |
| Global + SOR (std 2.0) | 44.3 | 51.3 | 47.5 |
| Density + SOR (std 2.0) | **45.6** | 53.9 | **49.4** |

**Findings.** (1) The clusterer choice is **metrically neutral** — cleaner clusters (≈62 vs
170 clusters, ≈2 % vs 12 % noise) don't move detection F1 (47.0 vs 47.4). (2) **SOR is a
pure precision↔recall dial**, not a net win (raises precision, drops recall, F1 ≈ flat) — so
it ships **off** by default. The BG-occupancy model is near its useful limit; the low
precision on the full cloud is a **clutter** problem, fixed by cropping + zones (§3), not by
the filter.

---

## 3. Input matters: cropped ≫ full

Same registered detector, only the input cloud differs (veh-only, ROI).

| input cloud | Precision | Recall | F1 |
|---|---|---|---|
| Full (research region) | 32.2 | 65.6 | 43.2 |
| **Cropped (road)** | **69.1** | 65.4 | **67.2** |

The full cloud's off-road clutter **more than doubles false positives** (FP 3928 → 835) at
identical recall. Cropped is the detection default (with an in-app warning if you switch to
full).

---

## 4. Removing the false positives: exclusion zones + static-phantom suppression

**FP analysis (cropped):** ~70 % of false positives came from tracks that persist ≥ 20
frames — the worst offenders appear in *every one* of the 282 frames. These are static-leak
phantoms (poles/barriers/vegetation the occupancy model can't remove because they're
geometrically identical to parked cars — but they never move). Two complementary fixes:

- **Exclusion zones** (site calibration) — the poles are fixed infrastructure at known
  locations, so a zone removes them at *zero recall cost*. Placing them lifted
  registered/cropped precision from ≈0.56 to **0.69** *and* nudged recall up (removing
  clutter that had been contaminating adjacent vehicle clusters and shifting their centroids
  past the match gate). The all-frames detection overlay in the Geometry Editor shows exactly
  where the persistent static detections cluster.
- **Static-phantom suppression** (`suppress_static`, automatic) — drop a track if it **both**
  persists ≥ `static_min_frames` (30) **and** never exceeds `static_max_speed` (0.5 m/s
  lifetime max). Real vehicles always break the floor, so recall is untouched.

| variant (registered/cropped, veh-only) | Precision | Recall | F1 |
|---|---|---|---|
| suppress static **off** | 67.8 | 65.4 | 66.6 |
| suppress static **on** (default) | **69.1** | 65.4 | **67.2** |

The suppression delta is now modest (+0.6 F1) because the **exclusion zones already remove
most static phantoms** — the two overlap. (Before zones, suppression alone was worth ≈ +5
F1.) Both are kept: zones do the heavy lifting on the known poles, suppression catches the
rest generically.

---

## 5. Precision is then Pareto-capped (negative result)

With zones + suppression in, **registered/cropped baseline = P 69.1 / R 65.4 / F1 67.2**.
Every remaining precision lever is a pure precision↔recall trade — **none beats the baseline
F1**:

| variant | Precision | Recall | F1 |
|---|---|---|---|
| **baseline (current defaults)** | 69.1 | 65.4 | **67.2** |
| min_cluster_pts = 3 | **81.1** | 54.8 | 65.4 |
| min_hits = 3 | 75.6 | 57.5 | 65.3 |
| merge_dist = 3.5 | 70.6 | 61.7 | 65.9 |
| static_max_speed = 1.0 | 63.0 | 46.3 | 53.3 |

The remaining FPs are tiny clusters in the mid-field, **irreducibly ambiguous** against real
sparse vehicles; near-duplicate detections within ≤ 2.5 m are already merged by the tracker.
**Conclusion:** once the structural wins are in (fusion, cropping, zones, suppression), the
classical pipeline is Pareto-tuned — further precision needs a learned detector, not tuning.
The `min_cluster_pts` slider still lets a user dial toward precision (e.g. **P 0.81** at the
cost of recall) when fewer false alarms matter more than coverage.

---

## 6. Failure-mode trace → a recall fix the precision sweep missed

Tracing every scorable GT box through the pipeline (cluster → accept → track → match)
localized the misses: **dense FNs** split into *no cluster formed*, *cluster rejected by the
acceptance gate*, and *accepted but lost in tracking* (association confusion between adjacent
vehicles), plus **split detections** (one car → two boxes from a fragmenting sparse cluster).

The acceptance-gate misses pointed at `strong_pts` (auto-accept a cluster without temporal
confirmation once it has this many points). The old cutoff **200** was arbitrary; a dense
fast mover (e.g. a 184-pt car at frame 0 with no prior frame) fell just under it and failed
temporal confirmation. **`strong_pts` is a recall lever** the precision sweep never touched,
and lowering it is a clean **Pareto improvement** (recall up, precision flat):

| strong_pts | Precision | Recall | F1 |
|---|---|---|---|
| 200 (old) | 69.2 | 65.0 | 67.0 |
| **100 (new default)** | 69.1 | 65.4 | 67.2 |
| 60 | 69.2 | 65.7 | 67.5 |

Shipped `strong_pts = 100`. The **split** and **tracking-association** misses remain — they
need a better *model* (L-shape box fitting, smarter adjacent-vehicle association), not a
knob, consistent with §5.

---

## 7. Correctness fixes

- **Evaluator gated matching.** The Hungarian matcher applied the distance gate *after* the
  global assignment; in dense scenes the global optimum paired a detection with a far GT,
  stranding a genuinely-close pair so both dropped (a real in-box detection scored as miss
  *and* FP). Gating *before* assigning fixed it — this alone raised absolute F1 ≈ 4 points
  (both precision and recall), hardest on the dense fused cloud. **All numbers above use the
  fixed matcher.**
- **Cross-sensor GT dedup (IoU-aware).** The ~8° yaw / ~2 m calibration residual pushed a
  shared vehicle's south/north boxes apart (up to ~5 m at range), past the 2.5 m centre gate,
  so they were emitted twice (phantom FN + a red "twin"). A BEV-IoU check (`dedup_iou = 0.10`)
  removed the residual twins. (The intuitive "give the twins the same ID" would *not* help —
  eval matches by geometry, not ID.)
- **`num_points` recomputed on the fused cloud** for the scorable gate (the stored counts are
  south-only, which would unfairly drop objects sparse for south but dense once fused).

---

## Reproducing

The harness scripts (`rebaseline1/2/3.py` in the session scratchpad) build each filtered
cloud **fresh from the current geometry** (so the exclusion zones apply), run
`detection_logic.run_detection_and_tracking` with `DEFAULT_DETECTION_PARAMS` overlaid, and
score with `evaluation.evaluate` / `recall_by_distance` against
`dataset_manager.labels_dir_for("registered","scorable")`. Defaults are centralized in
`detection_logic.DEFAULT_DETECTION_PARAMS`, so the A/B and the live Detection page can't drift.
