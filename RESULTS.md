# Results & Ablations — Multi-LiDAR Wrong-Way Detection

> **⚠️ Partially stale — re-verified 2026-07-03.** Numbers keep drifting as detector tuning
> continues; re-checking a sample of the tables below against live runs found the baseline,
> §1 (fusion A/B), and §4's suppression sub-table measurably out of date (baseline F1 was
> understated by ~4 pts; recall by ~8-10 pts). Those three are corrected in place below with
> a re-verified note. **§2 (BG-filter ablation), §4's exclusion-zone-specific precision lift,
> §5 (Pareto sweep), and §6 were NOT re-run** — each needs multiple full
> background-model/detection/eval cycles to re-verify, out of scope for this pass. Given how
> much the baseline moved, §5's "no lever beats baseline" conclusion in particular is **not
> confirmed at today's baseline** — treat §2, §4's zone-specific number, §5, and §6 as
> historical until re-swept. The current, re-verified headline numbers also live in the main
> [`README.md`](README.md#current-baseline-registeredcropped-gated-matcher-exclusion-zones-roi-20-m-gate).

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
> + static-suppression remove the false positives (precision) → a hyperparameter sweep
> (§8) tunes the classical detector further** (learned-model territory is what's left after that).

**Current registered/cropped baseline** *(tuned 2026-07-04, §8 — supersedes the 2026-07-03
figure below)*: **P 0.724 / R 0.752 / F1 0.738** (veh-only); **P 0.746 / R 0.740 / F1 0.743**
(all classes). MOTP ≈ 1.22–1.25 m.

*(2026-07-03, pre-tuning, kept for the trail: P 0.687 / R 0.737 / F1 0.711 veh-only;
P 0.720 / R 0.757 / F1 0.738 all classes.)*

---

## 1. Does fusion help? (Registered vs South A/B)

*(re-verified 2026-07-03)*

Same detection on each sensor's filtered cloud, graded against the **shared registered-union
scorable GT** (so south is fairly penalised for the north-only vehicles it physically can't
see; both share one denominator). Cropped, ROI, veh-only.

| pipeline | Precision | Recall | F1 |
|---|---|---|---|
| South | **0.777** | 0.564 | 0.653 |
| **Registered** | 0.687 | **0.737** | **0.711** |

Per-distance-bin recall (shared GT, veh-only): 0–20 m 0.191→0.579, 20–40 m 0.628→0.828,
40–60 m 0.572→0.609 — registered now wins at all three bins (previously close to a wash
far-field).

**Registered wins both recall and F1.** Fusion's core benefit is **recall +17 pts
(0.564 → 0.737)** — it sees vehicles a single sensor physically can't (occlusion shadows),
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

**FP analysis (cropped):** an earlier breakdown reported ~70 % of false positives coming
from tracks that persist ≥ 20 frames — static-leak phantoms (poles/barriers the occupancy
model can't remove because they're geometrically identical to parked cars, but never
move). **Re-tested 2026-07-03: this no longer reproduces** — only ~5.6 % of current FPs
match that signature (see the main [`README.md`](README.md) for the re-test). Two
complementary fixes were built against the original finding:

- **Exclusion zones** (site calibration) — the poles are fixed infrastructure at known
  locations, so a zone removes them at *zero recall cost*. Placing them reportedly lifted
  registered/cropped precision from ≈0.56 to ≈0.69 *and* nudged recall up (removing
  clutter that had been contaminating adjacent vehicle clusters and shifting their centroids
  past the match gate). *(This specific lift number — zones on vs. off — was not
  independently re-tested in the 2026-07-03 pass; treat as historical pending a re-run with
  zones toggled off.)* The all-frames detection overlay in the Geometry Editor shows exactly
  where the persistent static detections cluster.
- **Static-phantom suppression** (`suppress_static`, automatic) — drop a track if it **both**
  persists ≥ `static_min_frames` (30) **and** never exceeds `static_max_speed` (0.5 m/s
  lifetime max). Real vehicles always break the floor, so recall is untouched.

*(Re-verified 2026-07-03, registered/cropped, veh-only, own scorable GT — supersedes the
table this replaced, which read 67.8/69.1 precision and 66.6/67.2 F1:)*

| variant (registered/cropped, veh-only) | Precision | Recall | F1 | FP |
|---|---|---|---|---|
| suppress static **off** | 67.4 | 73.7 | 70.4 | 896 |
| suppress static **on** (default) | **68.7** | 73.7 | **71.1** | **846** |

The suppression delta is now small (+0.7 F1, −5.6 % FP, no recall cost) because the
**exclusion zones already remove most static phantoms** — the two overlap, and zones catch
the bulk of it before tracking ever sees them. (Before zones existed, suppression alone was
reportedly worth ≈ +5 F1 — see the flagged caveat above.) Both are kept: zones do the heavy
lifting on the known poles, suppression catches whatever's left, for free.

---

## 5. Precision is then Pareto-capped (negative result)

With zones + suppression in, **registered/cropped baseline = P 69.1 / R 65.4 / F1 67.2**
*(the baseline at the time this sweep was run — today's re-verified baseline is P 68.7 /
R 73.7 / F1 71.1, see §0/README; this sweep itself was NOT re-run against it, see the
banner at the top of this file)*. Every remaining precision lever *was* a pure
precision↔recall trade at that baseline — **none beat the baseline F1 then**:

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

## 8. Hyperparameter sweep (2026-07-04) — new defaults

Every number below is real (live detection + eval runs, registered/cropped, veh-only, ROI on,
shared registered-union scorable GT, **match_dist = 2.0 m** unless the table says otherwise).
"Baseline" in this section = the defaults **before** this sweep (`strong_pts=100`,
`truck_merge_dist=5.0`, `yaw_merge_deg=15.0`, `truck_len_thresh=7.0`, `bg_ratio=0.98`,
`cell_ratio=0.90`) — **P 0.6866 / R 0.7365 / F1 0.7106 / MOTP 1.341 m / 18 ID-switches → 16**
(one 282-frame clip). This was a one-at-a-time (coordinate-descent) sweep, not an exhaustive
grid, on a single dataset — a strong lead, not a provably global optimum.

### 8a. Match-distance gate — investigated, NOT adopted

Raised because F1 climbs sharply with a looser gate (0.71 → 0.80 by 4.0 m) with only a small
ID-switch cost. Measured at the (pre-sweep) baseline detection settings:

| match_dist (m) | P | R | F1 | MOTP (m) | ID-switches | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.0124 | 0.0135 | 0.0129 | 0.354 | 1 | 34 | 2710 | 2482 |
| 1.0 | 0.2117 | 0.2277 | 0.2194 | 0.755 | 7 | 573 | 2134 | 1943 |
| 1.5 | 0.3575 | 0.3839 | 0.3703 | 0.958 | 13 | 966 | 1736 | 1550 |
| **2.0 (standard)** | **0.6866** | **0.7365** | **0.7106** | **1.341** | **16** | **1853** | **846** | **663** |
| 2.5 | 0.7410 | 0.7949 | 0.7670 | 1.402 | 22 | 2000 | 699 | 516 |
| 3.0 | 0.7566 | 0.8116 | 0.7831 | 1.430 | 22 | 2042 | 657 | 474 |
| 3.5 | 0.7640 | 0.8196 | 0.7908 | 1.447 | 23 | 2062 | 637 | 454 |
| 4.0 | 0.7684 | 0.8243 | 0.7954 | 1.460 | 23 | 2074 | 625 | 442 |
| 4.5 | 0.7701 | 0.8255 | 0.7969 | 1.466 | 25 | 2077 | 620 | 439 |
| 5.0 | 0.7709 | 0.8263 | 0.7976 | 1.469 | 25 | 2079 | 618 | 437 |

**Rejected.** Going 2.0→4.0 m shifts TP/FP/FN by the exact same amount (+221/−221/−221) — one
reclassification event (the same 221 borderline pairs newly accepted), not new detections. The
two numbers that would move if this were a real improvement both move the *wrong* way: **MOTP
gets worse** (looser matches are, on average, less accurately localized) and **ID-switches rise**
(a track more easily locks onto the wrong nearby vehicle). Kept **2.0 m** as the reported gate.

### 8b. Detection/tracking parameters (one-at-a-time, cheap — reuses the already-filtered cloud)

| Parameter | Values tried | F1 | Verdict |
|---|---|---|---|
| `min_cluster_pts` (default 1) | 1 / 2 / 3 / 5 | 0.7106 / 0.7056 / 0.6979 / 0.6533 | monotone P↔R trade — no win, matches the earlier Pareto finding |
| `min_hits` (default 2) | 1 / 2 / 3 / 4 | 0.7106 / 0.7106 / 0.6951 / 0.6524 | no win |
| `strong_pts` (default 100) | 50 / 75 / 100 / 150 / 200 | **0.7141** / 0.7120 / 0.7106 / 0.7098 / 0.7093 | **50 wins** — MOTP/ID-switches flat |
| `truck_merge_dist` (default 5.0) | 2.5 / 3.5 / 5.0 / 6.5 / 8.0 | **0.7207** / **0.7207** / 0.7106 / 0.7091 / 0.6344 | **3.5 wins** — MOTP 1.341→1.290, ID-switches 16→15 |
| `merge_dist` (default 2.5) | 1.5 / 2.0 / 2.5 / 3.0 / 3.5 | 0.7043 / 0.7074 / 0.7106 / 0.7016 / 0.6980 | current default is already the local peak |
| `static_min_frames` (default 30) | 15 / 20 / 30 / 45 / 60 | 0.7132 / 0.7093 / 0.7106 / 0.7106 / 0.7039 | flat — no clear win |
| `static_max_speed` (default 0.5) | 0.3 / 0.5 / 0.7 / 1.0 | 0.7106 / 0.7106 / 0.5672 / 0.5672 | **sharp cliff above 0.7** — real slow-moving vehicles start getting suppressed; do not raise |
| `roi_abs_y` (default 40.0) | 30 / 40 / 50 / 60 | 0.6448 / 0.7106 / 0.7045 / 0.7045 | current default already best (50/60 tie — no GT beyond it) |
| `yaw_merge_deg` (default 15.0) | 10 / 15 / 20 / 30 | **0.7167** / 0.7106 / 0.7084 / 0.7080 | **10 wins** — MOTP 1.341→1.316 |
| `truck_len_thresh` (default 7.0) | 5.5 / 6.0 / 7.0 / 8.0 | 0.6943 / 0.7025 / 0.7106 / **0.7203** | **8.0 wins** — MOTP 1.341→1.290, ID-switches 16→15 |

**Combined** (`strong_pts=50, truck_merge_dist=3.5, yaw_merge_deg=10, truck_len_thresh=8.0`):
**P 0.6969 / R 0.7532 / F1 0.7240 / MOTP 1.287 m / 16 ID-switches** — better than any single
change alone; the four stack rather than interfere.

### 8c. Background-filtering parameters (expensive — full rebuild + refilter + redetect + re-eval per row, ~53 s each)

Detection held at its (pre-sweep) baseline settings while these varied:

| Parameter | Values tried | F1 | Verdict |
|---|---|---|---|
| `bg_ratio` (default 0.98) | 0.80 / 0.85 / 0.90 | **0.7327** / **0.7327** / 0.7094 | **0.85 wins** (0.80 ties it, so 0.85 kept as the more conservative choice) — P/R/MOTP/ID-switches **all** improve together, not a trade |
| `cell_ratio` (default 0.90) | 0.75 | **0.7131** | small additional win |
| clusterer mode (default density-adaptive) | global (legacy) | 0.7124 | metrically neutral — confirms the earlier ablation |
| `enable_5x5` (default on) | off | 0.7106 (identical to 4 decimals) | **confirmed true no-op** on this data |
| `enable_sor` (default off) | on, std=3.0 | 0.5965 | confirms SOR is a precision↔recall dial, not a net win |
| `pole_max_points` (default 80) | 40 / 150 | 0.7097 / 0.7129 | flat — no clear win |
| `dz_thresh` (default 0.3) | 0.2 (stricter) | 0.4430 | **large regression** — strips real low vehicle points; do not tighten |
| `inward_buffer_m` (default 2.0) | 0.0 (disabled) | 0.6770 | **real regression** — the road-edge trim is doing genuine work |

### 8d. Final combined result — adopted as the new defaults

| Configuration | P | R | F1 | MOTP (m) | ID-switches |
|---|---|---|---|---|---|
| Old defaults (pre-sweep) | 0.6866 | 0.7365 | 0.7106 | 1.341 | 16 |
| bg_ratio=0.85 + detection combo | 0.7230 | 0.7500 | 0.7362 | 1.251 | 16 |
| **bg_ratio=0.85, cell_ratio=0.75 + detection combo (adopted)** | **0.7238** | **0.7520** | **0.7376** | **1.252** | **17** |

**+2.70 F1 points, tighter localization (−0.089 m MOTP), +1 ID-switch — at the same fair
match_dist=2.0 m gate.** Verified end-to-end against the literal new defaults (nothing manually
overridden) before being adopted in `detection_logic.DEFAULT_DETECTION_PARAMS` and the
`bg_ratio`/`cell_ratio` sliders in `pages/2_Background_Filtering.py`.

---

## Reproducing

The harness scripts (`rebaseline1/2/3.py` in the session scratchpad) build each filtered
cloud **fresh from the current geometry** (so the exclusion zones apply), run
`detection_logic.run_detection_and_tracking` with `DEFAULT_DETECTION_PARAMS` overlaid, and
score with `evaluation.evaluate` / `recall_by_distance` against
`dataset_manager.labels_dir_for("registered","scorable")`. Defaults are centralized in
`detection_logic.DEFAULT_DETECTION_PARAMS`, so the A/B and the live Detection page can't drift.
