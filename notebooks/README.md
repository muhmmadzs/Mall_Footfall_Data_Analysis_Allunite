# Notebooks

## Single entry point

**[mall_footfall_analysis.ipynb](mall_footfall_analysis.ipynb)** — runs the full pipeline (Tasks 1–4, Plan A + B, advanced anomalies).

**[00_trusted_devices_eda.ipynb](00_trusted_devices_eda.ipynb)** — first-pass EDA for manual counts, `trusted_unique_devices`, candidate calibration features, and weekly trusted-device trends.

```python
from notebook_helpers import tune_plan_b_parameters, run_full_analysis, plot_full_analysis

tuning = tune_plan_b_parameters(method="bayesian", n_calls=50)  # Gaussian-process search
results = run_full_analysis(force_recompute=True, hod_params=tuning["hod_params"])
plot_full_analysis(results, save_plots=True)
```

Install tuning dependency: `pip install scikit-optimize`

- **`write_outputs=False`** (default) — keep results in memory; no CSV files under `outputs/`
- **`save_plots=True`** — PNGs only under `outputs/plots/`

## Legacy per-task notebooks

`task1_*.ipynb` … `task4_*.ipynb` are optional; prefer the main notebook above.

## CLI

```bash
python run_analysis.py
```

Same orchestration as the notebook; writes plots to `outputs/plots/` and a small `analysis_summary.json`.
