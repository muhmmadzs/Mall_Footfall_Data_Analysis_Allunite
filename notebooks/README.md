# Notebooks

## Single entry point

**[mall_footfall_analysis.ipynb](mall_footfall_analysis.ipynb)** — runs the full pipeline (Tasks 1–4, Plan A + B, advanced anomalies).

```python
from notebook_helpers import run_full_analysis, plot_full_analysis

results = run_full_analysis(force_recompute=True, write_outputs=False)
plot_full_analysis(results, save_plots=True)
```

- **`write_outputs=False`** (default) — keep results in memory; no CSV files under `outputs/`
- **`save_plots=True`** — PNGs only under `outputs/plots/`

## Legacy per-task notebooks

`task1_*.ipynb` … `task4_*.ipynb` are optional; prefer the main notebook above.

## CLI

```bash
python run_analysis.py
```

Same orchestration as the notebook; writes plots to `outputs/plots/` and a small `analysis_summary.json`.
