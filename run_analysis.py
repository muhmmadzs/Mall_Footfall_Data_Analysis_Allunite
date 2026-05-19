#!/usr/bin/env python3
"""Run full mall footfall analysis (delegates to notebook_helpers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notebook_helpers import plot_full_analysis, run_full_analysis
from src.paths import OUTPUT_DIR


def main() -> None:
    results = run_full_analysis(force_recompute=True, write_outputs=True)
    plot_full_analysis(results, save_plots=True, plan="both")
    task1_result = results["plan_a"]["task1_result"]
    summary = {
        "task1": {
            "recommended_model": task1_result.model_name,
            "selected_feature": task1_result.feature_name,
            "intercept": task1_result.intercept,
            "slope": task1_result.slope,
            "in_sample_mape": task1_result.mape,
            "in_sample_r2": task1_result.r2,
            "leave_one_out_mape": task1_result.loo_mape,
            "leave_one_out_max_ape": task1_result.loo_max_ape,
            "calibration_windows": task1_result.n_calibration_windows,
        },
        "task2": results["task2"]["summary"],
        "task3": {"journey_devices": results["task3"].get("journey_devices")},
        "task4_b": results["task4_b"]["anomaly_summary"],
        "task4_advanced_a": results["task4_advanced"]["plan_a"]["summary"],
        "task4_advanced_b": results["task4_advanced"]["plan_b"]["summary"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Plots written under {OUTPUT_DIR / 'plots'}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
