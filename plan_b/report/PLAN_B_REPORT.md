# Plan B — Enhanced Analysis Report

## Overview

Plan B lives in [`plan_b/`](../) and reuses [`data/`](../data/) from the main repo. It implements the alternative methods discussed for limited manual-count calibration (4 windows, 2 sensors).

**Primary footfall estimate:** `footfall_hod_adjusted` → column `estimated_total_footfall`.

---

## Task 1 — Calibration methods

### Models compared (`task1_models_comparison.csv`)

| Model | Description |
|-------|-------------|
| `global_linear_trusted_unique_devices` | Plan A–style trusted linear (global) |
| `global_linear_clean_unique_devices` | Linear on clean unique devices |
| `global_linear_local_unique_devices` | Linear on local MAC uniques |
| `blend_trusted_clean` | Two-feature regression (trusted + clean) |
| Pedestrians-only target | Separate calibration on pedestrian counts only |

### Per-facility capture rate

For each manual window: `capture_rate = manual_total / clean_unique_devices`.

Averaged per facility (66330, 66333). Other sensors inherit the **nearest calibrated sensor** by GPS (see `plan_b_meta.json` → `neighbor_map`).

### Hour-of-day (HOD) adjustment

1. Build median `clean_unique_devices` per `(facility, hour_of_day)` for the full week.
2. Scale hourly footfall: `devices × capture_rate × (profile_ref / profile_current_hour)`.
3. `profile_ref` = median profile at hours where manual counting occurred for that facility (or its neighbor).

### Mall-level daily totals (`task1_mall_daily_dedup.csv`)

| Column | Meaning |
|--------|---------|
| `unique_devices_mall` | Deduped clean devices per day |
| `estimated_total_footfall` | Sum of Plan B facility-hour estimates |
| `estimated_footfall_dedup_adjusted` | Facility sum × (1 − multi-sensor rate) |
| `estimated_from_mall_uniques` | Mall uniques × mean capture rate |

### Validation

- **Leave-one-out CV** (`task1_leave_one_out_cv.csv`) — train on 3 windows, predict the 4th.
- **Bootstrap** (`task1_bootstrap.json`) — 500 resamples of 4 points for intercept/slope uncertainty.

---

## Task 4 — Anomaly methods

| Method | Column | Rule |
|--------|--------|------|
| Plan A style | `zscore_global` | vs facility weekly mean |
| Robust | `zscore_robust` | MAD-based, \|z\| ≥ 3.5 |
| HOD baseline | `zscore_hod_baseline` | vs same facility × hour-of-day mean |
| Consensus | `is_anomaly_consensus` | ≥2 of global / robust / HOD |
| STL (if installed) | `is_anomaly_stl` | Robust seasonal residual |

---

## Comparison with Plan A

See `comparison_daily_footfall.csv` and `comparison_daily_footfall.png`.

Typical differences:

- Plan B tracks **clean device scale** and **time-of-day** better than trusted-only global linear.
- Plan A often floors near intercept (~517) when trusted devices = 0.
- Daily totals may differ by several % depending on hour profile and capture rates.

---

## How to run

```bash
python run_analysis.py          # Plan A
python plan_b/run_analysis.py   # Plan B
```

Notebook: [`notebooks/mall_footfall_analysis.ipynb`](../notebooks/mall_footfall_analysis.ipynb)
