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
    results = run_full_analysis(force_recompute=True, write_outputs=False)
    plot_full_analysis(results, save_plots=True, plan="both")
    summary = {
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
