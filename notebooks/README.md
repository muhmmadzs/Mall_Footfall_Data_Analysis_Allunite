# Notebooks

Single entry point: **[mall_footfall_analysis.ipynb](mall_footfall_analysis.ipynb)**

Run with kernel cwd = `notebooks/` (or repo root — `notebook_helpers.py` adds the repo to `sys.path`).

```python
from notebook_helpers import run_both_plans, plot_comparison_interactive, load_comparison_daily

run_both_plans(force_recompute=False)
plot_comparison_interactive(load_comparison_daily())
```

CLI alternative:

```bash
python run_analysis.py
python plan_b/run_analysis.py
```
