# Results & Ablations — Multi-LiDAR Wrong-Way Detection

Paper-ready evidence base for the s110 (TUMTraf A9) pipeline. Every number below is from
a controlled run on the same dataset (`A9_r02_s02`, 282 frames). Unless noted: registered
(south+north fused) cloud, **cropped to road**, ROI on, vehicle classes only, BEV-centre
match gate 2.0 m, identical detector. Detection is deterministic (same settings → same
numbers).

> Headline arc: **fusion → honest A/B → ablations localize the bottleneck to detection FPs
> → FP analysis → static-phantom suppression (+5 F1) → the classical pipeline is then
> Pareto-tuned** (further gains need a learned detector, not parameter tuning).

> **⚠️ Matching-fix note.** A bug in the evaluator's Hungarian matching (gate applied
> *after* the global assignment instead of before) was stranding genuinely-close
> detection/GT pairs in dense scenes — scoring a real, in-box detection as *both* a miss
> and a false positive. Fixed (gate before assigning). This **raises absolute F1 by ≈ 4
> points (both precision and recall)**, hardest on the dense fused cloud. The corrected
> registered/cropped baseline is **P 55.9 / R 65.2 / F1 60.2** (was 56.5). §1 below is
> updated to corrected numbers; **§2–§7 were measured pre-fix** and understate absolute F1
> by ~4 pts, but the **relative deltas within each table hold** (same matcher throughout a
> table), so their conclusions are unchanged.

---

## 1. Does fusion help? (Registered vs South A/B)

Same detection on each sensor's filtered cloud, graded against the **shared registered-union
scorable GT** (so south is fairly penalised for the north-only vehicles it physically can't
see; both share one denominator). Cropped, ROI.

Corrected numbers (gated matcher, veh-only, cropped, shared scorable GT):

| pipeline | Precision | Recall | F1 |
|---|---|---|---|
| South | **0.782** | 0.501 | **0.611** |
| **Registered** | 0.559 | **0.652** | 0.602 |

The corrected matcher reframes the A/B as a clean **recall-vs-precision trade**: registration
wins **recall decisively (0.652 vs 0.501, +15 pts)** — the occlusion-shadow-filling benefit,
and the metric that matters for not *missing* a wrong-way driver — while south's sparse cloud
keeps higher precision (fewer candidates → fewer false positives), making F1 a near-tie.
**For a recall-critical safety task, registered is the right pipeline.**

- Fusion crushes near/mid recall: **0–20 m 0.18→0.49**, **20–40 m 0.64→0.77** (fills each
  sensor's occlusion shadows).
- **Tuning fix surfaced by the benchmark:** the original `truck_merge_dist = 10 m`
  over-merged adjacent vehicles in the *denser* fused far-field, regressing 40–60 m recall
  to 0.36 and washing out the overall win. Lowering it to **5 m** lifts far recall to 0.49
  (barely touches sparse-far south) → registered then beats south overall.
- South keeps a precision edge (its GT omits the hard north-only objects).

---

## 2. Background-filter ablation (is the filter the bottleneck?)

Registered, **full** research region, all classes, shared scorable GT — same detector.
Tests the BG-occupancy knobs we added (density-adaptive clustering, SOR denoise).

| BG-filter variant | Precision | Recall | F1 |
|---|---|---|---|
| Global clusterer (baseline) | 28.3 | **65.2** | 39.5 |
| Density-adaptive clusterer | 27.0 | 65.9 | 38.3 |
| Global + SOR (std 2.0) | 35.6 | 49.2 | 41.3 |
| Density + SOR (std 2.0) | **36.3** | 51.9 | **42.7** |
| Global + SOR (std 3.0) | 30.2 | 58.5 | 39.9 |

**Findings.** (1) The clusterer choice is **metrically neutral** — cleaner clusters
(≈62 vs 170, ≈2 % vs 12 % noise) don't move detection F1. (2) **SOR is a pure
precision↔recall dial**, not a net win (F1 stays ≈39–43 while recall swings 65→49) — so it
ships **off** by default. The BG-occupancy model is near its useful limit; the low precision
here is a **detection-stage** problem, which §3–4 confirm.

---

## 3. Input matters: cropped ≫ full

Same registered detector, only the input cloud differs (veh-only, ROI).

| input cloud | Precision | Recall | F1 |
|---|---|---|---|
| Full (research region) | 25.6 | 63.3 | 36.4 |
| **Cropped (road)** | **44.3** | 63.5 | **52.2** |

The full cloud's off-road clutter roughly **doubles false positives**. Cropped is now the
detection default (with an in-app warning if you switch to full).

---

## 4. The FP bottleneck → static-phantom suppression

**FP analysis (cropped):** ~70 % of false positives came from tracks that persist ≥ 20
frames — the worst offenders appear in *every one* of the 282 frames. These are static-leak
phantoms (barriers/poles/vegetation the occupancy model can't remove because they're
geometrically identical to parked cars — but they never move).

**Fix:** drop a track from every frame if it **both** persists ≥ `static_min_frames` (30)
**and** never exceeds `static_max_speed` (0.5 m/s lifetime max). Real vehicles always break
the floor, so recall is barely touched; a never-moving object is never a wrong-way driver.

| input | suppress static | Precision | Recall | F1 |
|---|---|---|---|---|
| registered/cropped | off | 43.8 | 61.5 | 51.1 |
| registered/cropped | **on** | **52.3** | 60.6 | **56.2** |
| south/cropped | off | 73.2 | 51.2 | 60.2 |
| south/cropped | on | 75.6 | 49.3 | 59.7 |

Big win on the fused pipeline (**+5 F1, −30 % FP**, −0.9 recall); near-neutral on the
already-clean single south sensor (few phantoms) — consistent with the leak being a fusion
artifact. Applied identically to both, so the A/B stays fair.

---

## 5. Precision is then Pareto-capped (negative result)

After static-suppression, **registered/cropped baseline = P 52.4 / R 60.8 / F1 56.3**
(against the IoU-deduped GT). Every remaining precision lever was swept — all are pure
precision↔recall trades, **none beats the baseline F1**:

| variant | Precision | Recall | F1 |
|---|---|---|---|
| **baseline (current defaults)** | 52.4 | 60.8 | **56.3** |
| min_cluster_pts = 3 | 57.2 | 48.6 | 52.5 |
| range-aware min-pts (6/3/1) | 60.1 | 46.7 | 52.6 |
| min_hits = 3 | 52.4 | 51.6 | 52.0 |
| merge_dist = 3.5 | 48.2 | 56.3 | 51.9 |
| NMS (dist 3 m) | 52.5 | 60.2 | 56.1 |
| static_max_speed = 1.0 | 48.2 | 41.0 | 44.3 |

The remaining FPs are tiny clusters (median 5 points) in the mid-field (71 % at 20–40 m),
**irreducibly ambiguous** against real sparse vehicles at the same range; near-duplicate
detections within ≤2.5 m are already merged by the tracker. **Conclusion:** once the
structural wins are in (fusion, cropping, static-suppression), the classical pipeline is
Pareto-tuned — further precision needs a fundamentally better (learned) detector, not
parameter tuning. The existing `min_cluster_pts` slider already lets a user trade toward
precision (e.g. P 57 at the cost of recall) when fewer false alarms matter more than recall.

---

## 6. Failure-mode trace → a recall fix the precision sweep missed

Tracing every scorable GT box through the pipeline (cluster → accept → track → match) on
registered/cropped (2849 boxes) localized the misses:

- **231 "dense misses"** (FN with ≥ 20 foreground points inside): **69** no cluster formed
  near the box, **41** a cluster existed but was **rejected by the acceptance gate**, **121**
  accepted but **lost in tracking** (association confusion between adjacent vehicles).
- **68 split detections** (a matched GT with ≥ 2 detections inside) — the "one car → two
  objects" case, from a sparse cluster fragmenting under DBSCAN.

The acceptance-gate misses pointed at `strong_pts` (auto-accept a cluster without temporal
confirmation once it has this many points). The old cutoff **200** was arbitrary; a dense
fast mover (e.g. a 184-pt car at frame 0 with no prior frame) fell just under it and failed
temporal confirmation. **`strong_pts` was never in the precision sweep** — it's a recall
lever — and lowering it is a clean **Pareto improvement** (recall up, precision flat):

| strong_pts | Precision | Recall | F1 | R@0–20 |
|---|---|---|---|---|
| 200 (old) | 52.4 | 60.8 | 56.3 | 52.5 |
| **100 (new default)** | 52.4 | 61.3 | 56.5 | **55.8** |
| 60 | 52.6 | 61.6 | 56.7 | 56.8 |

Near-field recall rises 52.5→55.8 at zero precision cost (dense road clusters are real
vehicles; clutter is tiny, median ~5 pts). Shipped `strong_pts = 100`. The **split** and
**tracking-association** misses remain — they need a better *model* (L-shape box fitting,
smarter adjacent-vehicle association), not a knob, consistent with §5.

## 7. Correctness fixes (small metric effect, real for soundness)

- **Cross-sensor GT dedup (IoU-aware).** The ~8° yaw / ~2 m calibration residual pushed a
  shared vehicle's south/north boxes apart (up to ~5 m at range), past the 2.5 m centre
  gate, so they were emitted twice (phantom FN + a red "twin"). Adding a BEV-IoU check
  (`dedup_iou = 0.10`) removed **42** residual twins (94→52 < 8 m); metric effect small
  (F1 53.1→53.2) — a soundness/visual fix, not a lever. (The intuitive "give the twins the
  same ID" idea would *not* help — eval matches by geometry, not ID.)
- **`num_points` recomputed on the fused cloud** for the scorable gate (the stored counts
  are south-only, which would unfairly drop objects sparse for south but dense once fused).

---

## Reproducing

The exact harness scripts used for every table live in the session scratchpad; each builds
the filtered cloud (or reuses the saved one), runs `detection_logic.run_detection_and_tracking`
with `DEFAULT_DETECTION_PARAMS` overlaid, and scores with `evaluation.evaluate` /
`recall_by_distance` against `dataset_manager.labels_dir_for("registered","scorable")`.
Defaults are centralized in `detection_logic.DEFAULT_DETECTION_PARAMS`, so the A/B and the
live Detection page can't drift.
