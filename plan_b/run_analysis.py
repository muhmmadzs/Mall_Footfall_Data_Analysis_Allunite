#!/usr/bin/env python3
"""Run Plan B analysis pipeline (sophisticated method + mall visitors)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PLAN_B_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN_B_ROOT))

from src.compare import generate_comparison_outputs
from src.load_data import load_all
from src.paths import OUTPUT_DIR
from src.plots_anomalies import generate_anomaly_plots
from src.task1 import run_task1_plan_b
from src.task2 import run_task2
from src.task3 import run_task3
from src.task4 import run_task4_plan_b


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions, facilities, manuals = load_all()

    _, intersection_summary = run_task2(sessions, facilities)
    overlap_rate = float(intersection_summary.get("multi_sensor_rate", 0.0))

    cal_df, models_df, loo, hourly, mall_daily, mall_hourly, capture_rates, meta = (
        run_task1_plan_b(sessions, facilities, manuals, overlap_rate=overlap_rate)
    )
    suspicious, anomaly_hours, anomaly_summary, _ = run_task4_plan_b(sessions, hourly)
    top_paths, transitions, sample_paths, journey_devices = run_task3(sessions, facilities)

    cal_df.to_csv(OUTPUT_DIR / "task1_calibration_windows.csv", index=False)
    models_df.to_csv(OUTPUT_DIR / "task1_models_comparison.csv", index=False)
    loo.to_csv(OUTPUT_DIR / "task1_leave_one_out_cv.csv", index=False)
    hourly.to_csv(OUTPUT_DIR / "task1_hourly_estimated_footfall.csv", index=False)
    mall_daily.to_csv(OUTPUT_DIR / "task1_mall_visitors_daily.csv", index=False)
    mall_hourly.to_csv(OUTPUT_DIR / "task1_mall_visitors_hourly.csv", index=False)
    if not capture_rates.empty:
        capture_rates.to_csv(OUTPUT_DIR / "task1_capture_rates_by_facility.csv", index=False)

    suspicious.head(200).to_csv(OUTPUT_DIR / "task4_suspicious_unflagged_devices.csv", index=False)
    anomaly_hours.to_csv(OUTPUT_DIR / "task4_hourly_anomalies.csv", index=False)
    transitions.to_csv(OUTPUT_DIR / "task3_transition_matrix.csv", index=False)
    top_paths.head(100).to_csv(OUTPUT_DIR / "task3_top_journeys.csv", index=False)
    sample_paths.to_csv(OUTPUT_DIR / "task3_sample_device_journeys.csv", index=False)

    with (OUTPUT_DIR / "plan_b_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)

    comp = generate_comparison_outputs(hourly, mall_daily)
    generate_anomaly_plots(anomaly_hours)

    summary = {
        "plan": "B",
        "methods": {
            "plan_a": "footfall_plan_a (global trusted linear, sensor-hour)",
            "plan_b": "footfall_plan_b (capture rate + HOD, sensor-hour)",
            "mall_visitors": "estimated_mall_visitors (mall dedup devices × capture)",
        },
        "task1_models": models_df.to_dict(orient="records"),
        "task1_loo_mae_mean": float(loo["abs_error"].mean()) if len(loo) else None,
        "task2": intersection_summary,
        "task3": {"journey_devices": journey_devices},
        "task4": anomaly_summary,
        "mall_visitors_daily": mall_daily.to_dict(orient="records"),
        "daily_comparison": comp.to_dict(orient="records") if len(comp) else [],
        "meta": meta,
    }
    with (OUTPUT_DIR / "analysis_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Plan B outputs written to {OUTPUT_DIR}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
