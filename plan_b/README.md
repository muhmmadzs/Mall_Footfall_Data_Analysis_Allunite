# Plan B — Sophisticated footfall + mall visitors

Runs alongside **Plan A** (repo root `run_analysis.py`).

## Two methods

| Method | Column | Meaning |
|--------|--------|---------|
| **Plan A** | `footfall_plan_a` | Global linear on trusted devices: intercept + slope × trusted (calibration fit) |
| **Plan B** | `footfall_plan_b` | Per-sensor capture rate + GPS neighbor transfer + hour-of-day adjustment |
| **Mall visitors** | `estimated_mall_visitors` | Mall-deduped unique devices × mall capture rate (daily or hourly) |

## Run

```bash
python run_analysis.py          # Plan A
python plan_b/run_analysis.py   # Plan B + comparison chart
```

## Key outputs (`plan_b/outputs/`)

- `task1_hourly_estimated_footfall.csv` — `footfall_plan_a`, `footfall_plan_b`, `hod_factor`, `capture_rate_facility`
- `task1_mall_visitors_daily.csv` — daily mall visitors
- `task1_mall_visitors_hourly.csv` — hourly mall visitors
- `comparison_daily_footfall.csv` / `.png` — Plan A vs Plan B vs mall visitors

## Notebook

See [notebooks/mall_footfall_analysis.ipynb](../notebooks/mall_footfall_analysis.ipynb).
