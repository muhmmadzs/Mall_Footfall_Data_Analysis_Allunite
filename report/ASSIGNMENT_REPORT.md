# Technical Assignment: Mall Footfall Data Analysis

**Location:** Liverpool One (UK) · **Period:** 20–26 April 2026 (UTC) · **Mall ID:** `facility_master_num` 66328

## Executive summary

- Processed **2,248,477** sensor sessions across **10** facility records (9 active sensors + mall cluster row).
- Calibrated a **Plan C trend-preserving** footfall model on **4** one-hour manual counting windows (2 sensors × 2 days), using **clean unique devices** for hourly shape and manual counts for scale.
- The old trusted-device linear model has lower validation error, but its large intercept flattens the hourly output; the recommended model uses elasticity **β = 0.10** with MAPE **6.04%** and leave-one-out MAPE **12.16%**.
- **6.43%** of clean unique devices were detected at more than one sensor; strongest pairwise overlap is **66330 ↔ 66331** (58,355 shared devices).
- Most common journey path: **66330 → 66331** (42,826 devices); dominant transition edge matches this corridor.
- Built-in session flags are rare (**0.11%** any flag); **87** additional high-activity devices look suspicious but are not flagged.
- Hourly footfall z-scores highlight peak anomalies (e.g. facility 66340 on 25 Apr 15:00 UTC, z ≈ 4.3).
- Daily estimated footfall is now less flat (~91k–111k/day); **local MAC devices** dominate detections; week total **707k** estimated footfall vs **2.04M** clean unique devices (mall dedup).

## Data and cleaning

### Sources

| File | Role |
|------|------|
| `data/allunite_device_session.csv` | ~2.25M detection sessions |
| `data/facility_information - Sheet1.csv` | Sensor metadata, GPS, `box_macs` |
| `data/Manual Counting - Sheet1.csv` | Ground-truth counts (pedestrians + bicycles) |

### Preprocessing

1. Parsed `session_start` as **UTC**; derived `hour_start`, `date`, `hour_of_day`.
2. Dropped rows missing `device_id`, `facility_num`, or `session_start` only.
3. Kept `session_duration = 0` (valid momentary detections per assignment notes).
4. **Clean traffic mask** for modelling:

```text
~is_excluded & ~is_anomaly & ~is_fake & ~is_permanent_device
```

5. Manual target = **sum of Pedestrians + Bicycles** per 60-minute window.

### Manual counting coverage

| Facility | Name | Windows |
|----------|------|---------|
| 66330 | GB-LVO-DD-08003 | 2026-04-23 13:00, 2026-04-24 13:00 |
| 66333 | GB-LVO-DD-08009 | 2026-04-23 15:00, 2026-04-24 15:00 |

No manual counts exist for the other seven sensors.

---

## Task 1 — Traffic model (sensor → estimated footfall)

### Method

1. For each manual window, computed sensor features (raw/clean sessions, unique devices, trusted/local unique devices).
2. Compared linear, no-intercept, capture-rate, blended, and clean-device elasticity models.
3. Kept the old trusted-device linear model as a diagnostic because it validates well but creates flat hourly totals through a large intercept.
4. Selected **Plan C: trend-preserving clean-device elasticity** for the delivered estimate.
5. Applied calibrated/manual anchors to hourly clean-device volume for the full week; uncalibrated sensors borrow the nearest calibrated anchor.

### Model naming and selection rationale

| Label | Output column | Role |
|-------|---------------|------|
| Plan A | `footfall_plan_a` / `footfall_trusted_linear` | Trusted-device linear baseline; useful diagnostic but flatter over time. |
| Plan B | `footfall_plan_b` | Capture-rate + hour-of-day/density comparison model; also produces mall-deduped visitors. |
| Plan C | `footfall_plan_c` / `estimated_total_footfall` | Delivered Task 1 estimate; trend-preserving clean-device elasticity. |

Plan A has the lowest error on the four calibration windows, but that fit relies on a large intercept and compresses hourly/daily variation. Plan C is selected for delivery because it keeps the hourly shape tied to clean unique device volume while staying anchored to manual counts. With only four manual windows, this is a judgement call: the reported error metrics should be read together with the trend-shape plots and the limitations below.

### Selected model

| Metric | Value |
|--------|-------|
| Feature | `clean_unique_devices` |
| Intercept | 0.00 |
| Elasticity beta | 0.10 |
| MAE (4 windows) | 35.09 |
| MAPE | 6.04% |
| R² | 0.240 |
| Leave-one-out MAPE | 12.16% |

### Daily estimated mall totals (sum of facility-hours; not deduplicated visitors)

| Date | Estimated footfall |
|------|-------------------|
| 2026-04-20 | 97,453.78 |
| 2026-04-21 | 91,470.95 |
| 2026-04-22 | 99,979.64 |
| 2026-04-23 | 90,914.00 |
| 2026-04-24 | 106,925.19 |
| 2026-04-25 | 109,576.51 |
| 2026-04-26 | 110,890.14 |

**Outputs:** `outputs/task1_calibration_windows.csv`, `outputs/task1_hourly_estimated_footfall.csv`, `outputs/task1_model_comparison.csv`, `outputs/task1_leave_one_out_cv.csv`, `outputs/task1_bootstrap_selected_model.csv`, `outputs/task1_calibration_scatter.png`

---

## Task 2 — Sensor intersection

### Method

- Reduced clean data to unique `(device_id, facility_num)` pairs.
- Counted devices seen at **>1** facility.
- For each sensor pair (A < B): shared devices, Jaccard index, overlap % of smaller side.
- Added same-day shared-device counts to separate broad weekly overlap from more time-local movement evidence.

### Findings

| Metric | Value |
|--------|-------|
| Clean unique devices | 2,042,923 |
| Multi-sensor devices | 131,320 (**6.43%**) |

**Top overlap pair:** GB-LVO-DD-08003 (66330) ↔ GB-LVO-DD-08005 (66331) — **58,355** shared devices, Jaccard **0.103**.

**Second:** GB-LVO-DD-08009 (66333) ↔ GB-LVO-DD-08023 (66342) — **43,278** shared, Jaccard **0.105**.

High overlap reflects pedestrian movement and adjacent coverage, not necessarily double-counting error.

**Outputs:** `outputs/task2_sensor_overlap_pairs.csv`, `outputs/task2_top_overlap_pairs.png`

---

## Task 3 — Journey mapping

### Method

1. Devices with **>1** distinct facility on clean data.
2. Sorted sessions by time and split journeys when the same device had a gap of more than **60 minutes**.
3. Collapsed consecutive duplicate facilities within each journey session.
4. Built path strings and `facility → next_facility` transition counts.

### Findings

| Metric | Value |
|--------|-------|
| Devices with multi-facility paths | 131,320 |
| Most common path | `66330 -> 66331` (see `outputs/task3_top_journeys.csv` for sessionized journey and device counts) |
| Top transition | 66330 → 66331 (**51,395** transitions) |

Sensor GPS map saved for spatial context; floor-plan overlay can be added when a raster image is exported from the provided PPTX.

**Outputs:** `outputs/task3_transition_matrix.csv`, `outputs/task3_top_journeys.csv`, `outputs/task3_sample_device_journeys.csv`, `outputs/task3_sensor_map.png`

---

## Task 4 — Anomaly detection and patterns

### A. Existing flags

| Flag | Session rate |
|------|----------------|
| `is_excluded` | 0.105% |
| `is_fake` | 0.003% |
| `is_anomaly` | 0.000% |
| Any flag | 0.108% |

Flags catch a small fraction of sessions; most traffic is unflagged.

### B. Suspicious unflagged devices

Per-device stats: total sessions, facility spread, active days, night ratio (hours 0–5 UTC).

Flagged if **not** permanent/flagged and either:

- Top **0.01%** by session count (≥ **32** sessions), or
- Night ratio ≥ **0.6** with ≥ **120** sessions.

Threshold rationale:

| Rule | Threshold | Why used |
|------|-----------|----------|
| Long-tail device activity | Top 0.01% by session count | Catches extreme repeated detections unlikely to be ordinary visitors. |
| Night-heavy devices | Night ratio ≥ 0.60 and sessions ≥ 120 | Catches always-on/background devices active outside normal mall movement. |
| Hourly footfall outliers | Facility-level \|z\| ≥ 3 | Standard high-confidence outlier rule within each sensor's weekly pattern. |

**Result:** **87** suspicious devices identified.

### C. Hourly footfall outliers

Per-facility z-score on `estimated_total_footfall`; flag \|z\| ≥ 3.

Example peaks:

- Facility **66330**, 2026-04-23 16:00 UTC — footfall **672.04**, z **4.60**
- Facility **66331**, 2026-04-26 17:00 UTC — footfall **627.40**, z **4.51**
- Facility **66340**, 2026-04-25 15:00 UTC — footfall **733.13**, z **4.31**

**Outputs:** `outputs/task4_suspicious_unflagged_devices.csv`, `outputs/task4_hourly_anomalies.csv`

**Figures (patterns & anomalies):**

| Chart | File |
|-------|------|
| Built-in flag rates | `task4_flag_rates.png` |
| Sessions by hour (clean vs flagged) | `task4_sessions_by_hour_flagged.png` |
| Device activity long-tail | `task4_device_session_distribution.png` |
| Suspicious: sessions vs night ratio | `task4_suspicious_device_patterns.png` |
| Suspicious: multi-facility spread | `task4_facility_spread_patterns.png` |
| Hourly footfall with peaks | `task4_hourly_footfall_anomalies.png` |
| Z-score heatmap | `task4_anomaly_zscore_heatmap.png` |
| Top anomaly hours | `task4_top_anomaly_hours.png` |

Anomaly outputs: `outputs/task4_hourly_anomalies.csv`, `plan_b/outputs/task4_hourly_anomalies.csv`.

---

## Device and footfall trends (daily / weekly)

### Summary

| Metric | Week total (Apr 20–26) |
|--------|------------------------|
| Total sessions | 2,248,477 |
| Clean sessions | 2,237,846 |
| Unique devices (mall dedup, clean) | 2,042,923 |
| Sum of facility daily uniques | 2,209,972 (over-counts multi-sensor visits) |
| Estimated footfall (facility-hour sum) | 707,210.21 |
| Trusted devices (week) | 13,126 |
| Local devices (week) | 2,038,900 |
| Other devices (week) | 3,881 |

Daily estimated footfall now moves in a wider but controlled band (~91k–111k per day). **Local (randomised MAC) devices** dominate each day; **trusted** devices are a small fraction (~0.9% of sessions) and are retained as diagnostics, while the recommended footfall shape is driven by clean unique devices. **Other devices** (clean, neither trusted nor local) are a small residual.

### Charts

| File | Description |
|------|-------------|
| `outputs/footfall_daily_sessions_and_devices.png` | Sessions vs mall-deduplicated unique devices per day |
| `outputs/footfall_daily_device_types_stacked.png` | Trusted / local / other unique devices per day |
| `outputs/footfall_daily_estimated_vs_detected.png` | Estimated footfall vs clean unique devices |
| `outputs/footfall_weekly_totals.png` | Week totals and dedup vs summed-facility comparison |
| `outputs/footfall_daily_by_facility_heatmap.png` | Per-sensor daily unique devices |
| `outputs/footfall_manual_calibration_overlay.png` | Manual vs model for 4 calibration windows |
| `outputs/daily_device_footfall_summary.csv` | Daily metrics table |

See [notebooks/mall_footfall_analysis.ipynb](../notebooks/mall_footfall_analysis.ipynb) for Plan A vs Plan B and mall visitors.

---

## Assumptions and limitations

1. **Four calibration windows** on two sensors — model extrapolation to all facilities/hours is uncertain.
2. **Pedestrians + bicycles** used as ground truth; pedestrians-only would change coefficients.
3. **Daily mall totals** sum facility-hour estimates without cross-sensor deduplication.
4. **Randomised MACs** (`is_local`) weaken cross-session identity for journeys.
5. Temporal path order does not imply shortest physical route (dwell, backtracking, sensor range).

## Reproducibility

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
```

Notebook: [notebooks/mall_footfall_analysis.ipynb](../notebooks/mall_footfall_analysis.ipynb). Full summary: `outputs/analysis_summary.json`.

## Suggested next steps

1. Collect more manual counting windows (more sensors, hours, days).
2. Facility-specific or hierarchical calibration models with uncertainty bands.
3. Overlay top transitions on mall floor plan for spatial analytics.
4. Review the 87 suspicious devices against staff lists / permanent device rules.
