# Plan B — Sophisticated footfall + mall visitors

Runs alongside **Plan A** (repo root `run_analysis.py`).

## Two methods

| Method | Column | Meaning |
|--------|--------|---------|
| **Plan A** | `footfall_plan_a` | Global linear on trusted devices: intercept + slope × trusted (calibration fit) |
| **Plan B** | `footfall_plan_b` | Capture rate × **HOD v2** × density on **mall-hour deduped** devices |
| **Plan B (no dedup)** | `footfall_plan_b_per_sensor` | Same HOD v2 formula but per-sensor device counts (can double-count) |
| **Plan B (legacy HOD)** | `footfall_plan_b_profile_hod` | Old profile-ratio HOD (0.5–2.0 clamp) for comparison |
| **Mall visitors** | `estimated_mall_visitors` | Mall-deduped unique devices × mall capture rate (daily or hourly) |

## HOD v2 (Plan B)

\(`footfall_plan_b = devices_dedup × capture_rate × hod_factor × density_factor`\)

1. **Manual anchors (B):** at each calibration window, `hod_anchor = manual / (devices × capture)`; latest window wins per facility × hour.
2. **Smooth 24h curve:** Gaussian blend of anchors across clock hours (`smooth_tau`, LOO-tuned).
3. **Shrink (D):** `hod = 1 + w × (hod_smooth − 1)` with `w` decaying by circular distance to nearest calibration hour (`shrink_tau`).
4. **Density (A):** `(devices / profile_median)^β`, clipped (default β=0.15).

Legacy profile-ratio HOD is kept in `hod_factor_profile` / `footfall_plan_b_profile_hod`.

## Run

```bash
python run_analysis.py          # Plan A
python plan_b/run_analysis.py   # Plan B + comparison chart
```

## Key outputs (`plan_b/outputs/`)

- `task1_hourly_estimated_footfall.csv` — `footfall_plan_a`, `footfall_plan_b`, `hod_factor`, `density_factor`, `capture_rate_facility`
- `hod_v2_calibration_validation.csv` — Plan B vs manual at calibration windows
- `task1_mall_visitors_daily.csv` — daily mall visitors
- `task1_mall_visitors_hourly.csv` — hourly mall visitors
- `comparison_daily_footfall.csv` / `.png` — Plan A vs Plan B vs mall visitors

## Notebook

See [notebooks/mall_footfall_analysis.ipynb](../notebooks/mall_footfall_analysis.ipynb).
