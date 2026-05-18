#!/usr/bin/env python3
"""Run all four assignment tasks and write outputs/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import load_all
from src.paths import OUTPUT_DIR
from src.task1 import run_task1
from src.task2 import run_task2
from src.task3 import run_task3
from src.task4 import run_task4
from src.plots_footfall import generate_all_footfall_plots
from src.plots_anomalies import generate_all_anomaly_plots


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sessions, facilities, manuals = load_all()

    calibration_df, cal_result, hourly = run_task1(sessions, manuals)
    pair_counts, intersection_summary = run_task2(sessions, facilities)
    top_paths, transitions, sample_paths, journey_devices = run_task3(
        sessions, facilities
    )
    suspicious, anomaly_hours, anomaly_summary, device_stats = run_task4(
        sessions, hourly
    )
    generate_all_anomaly_plots(
        sessions,
        facilities,
        device_stats,
        suspicious,
        anomaly_hours,
        anomaly_summary,
    )

    calibration_df.to_csv(OUTPUT_DIR / "task1_calibration_windows.csv", index=False)
    hourly.to_csv(OUTPUT_DIR / "task1_hourly_estimated_footfall.csv", index=False)
    pair_counts.to_csv(OUTPUT_DIR / "task2_sensor_overlap_pairs.csv", index=False)
    top_paths.head(100).to_csv(OUTPUT_DIR / "task3_top_journeys.csv", index=False)
    transitions.to_csv(OUTPUT_DIR / "task3_transition_matrix.csv", index=False)
    sample_paths.to_csv(OUTPUT_DIR / "task3_sample_device_journeys.csv", index=False)
    suspicious.head(200).to_csv(
        OUTPUT_DIR / "task4_suspicious_unflagged_devices.csv", index=False
    )
    anomaly_hours.to_csv(OUTPUT_DIR / "task4_hourly_anomalies.csv", index=False)

    daily_device_df, weekly_footfall_stats = generate_all_footfall_plots(
        sessions,
        facilities,
        manuals,
        hourly,
        multi_sensor_rate=intersection_summary.get("multi_sensor_rate"),
    )

    daily = (
        hourly.assign(date=lambda df: df["hour_start"].dt.date)
        .groupby("date", as_index=False)["estimated_total_footfall"]
        .sum()
        .sort_values("date")
    )

    summary = {
        "rows_sessions": int(len(sessions)),
        "rows_facilities": int(len(facilities)),
        "rows_manual_counts": int(len(manuals)),
        "task1": {
            "selected_feature": cal_result.feature_name,
            "intercept": cal_result.intercept,
            "slope": cal_result.slope,
            "mae": cal_result.mae,
            "mape": cal_result.mape,
            "r2": cal_result.r2,
            "calibration_windows": int(len(calibration_df)),
        },
        "task2": intersection_summary,
        "task3": {
            "devices_with_multifacility_paths": journey_devices,
            "top_path": top_paths.iloc[0]["path"] if len(top_paths) else None,
            "top_path_devices": int(top_paths.iloc[0]["device_count"])
            if len(top_paths)
            else 0,
        },
        "task4": anomaly_summary,
        "daily_estimated_footfall": daily.to_dict(orient="records"),
        "footfall_device_trends": {
            "daily_summary_rows": int(len(daily_device_df)),
            **weekly_footfall_stats,
        },
    }
    with (OUTPUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote outputs to {OUTPUT_DIR}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
