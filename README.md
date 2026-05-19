# Mall Footfall Data Analysis (AllUnite)

Technical assignment: sensor-based footfall modelling and visitor behaviour analysis for Liverpool One (Apr 20–26, 2026).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

By default, place the three input files under `data/`:

```text
data/
  allunite_device_session.csv
  facility_information - Sheet1.csv
  Manual Counting - Sheet1.csv
```

You can also point the code at another folder with environment variables:

```bash
export MALL_FOOTFALL_DATA_DIR="/path/to/my/data"
python run_analysis.py
```

Or override individual files:

```bash
export MALL_FOOTFALL_SESSION_CSV="/path/to/allunite_device_session.csv"
export MALL_FOOTFALL_FACILITY_CSV="/path/to/facility_information - Sheet1.csv"
export MALL_FOOTFALL_MANUAL_CSV="/path/to/Manual Counting - Sheet1.csv"
python run_analysis.py
```

Optional output overrides:

```bash
export MALL_FOOTFALL_OUTPUT_DIR="/path/to/outputs"
export MALL_FOOTFALL_PLAN_B_OUTPUT_DIR="/path/to/plan_b_outputs"
```

## Run analysis

```bash
python run_analysis.py
```

Outputs are written to `outputs/`. Footfall/device trend charts (`footfall_*.png`) and `daily_device_footfall_summary.csv` are generated automatically. See `report/ASSIGNMENT_REPORT.md` for methods and findings.

Interactive dashboard: [notebooks/mall_footfall_analysis.ipynb](notebooks/mall_footfall_analysis.ipynb)

## Project layout

- `data/` — raw CSV inputs
- `src/` — load, clean, and task modules (**Plan A**)
- `plan_b/` — enhanced pipeline for comparison (**Plan B**)
- `notebooks/` — single analysis notebook (`mall_footfall_analysis.ipynb`)
- `outputs/` — Plan A CSVs and summary JSON
- `plan_b/outputs/` — Plan B outputs + A vs B comparison charts
- `report/` — written assignment report

### Plan A vs Plan B

| | Plan A (root) | Plan B (`plan_b/`) |
|---|---------------|---------------------|
| Footfall | Global linear on `trusted_unique_devices` (`footfall_plan_a`) | Capture rate + HOD (`footfall_plan_b`) |
| People in mall | — | `estimated_mall_visitors` (dedup devices × capture) |
| Anomalies | Global z-score | Robust z, HOD baseline, consensus |

```bash
python run_analysis.py
python plan_b/run_analysis.py
```

See [plan_b/README.md](plan_b/README.md).

### Notebook

```python
from notebook_helpers import run_both_plans, plot_comparison_interactive, load_comparison_daily
run_both_plans(force_recompute=False)
plot_comparison_interactive(load_comparison_daily())
```

Open [notebooks/mall_footfall_analysis.ipynb](notebooks/mall_footfall_analysis.ipynb).
