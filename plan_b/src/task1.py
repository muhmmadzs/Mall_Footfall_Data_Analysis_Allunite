from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.calibration import (
    apply_plan_b_estimates,
    assign_facility_neighbors,
    bootstrap_coefficients,
    build_calibration_windows,
    capture_rates_per_facility,
    fit_blend_trusted_clean,
    fit_global_linear,
    hour_of_day_profile,
    hourly_sensor_features,
    leave_one_out_cv,
    mall_daily_dedup,
    mall_hour_primary_facility_counts,
    mall_hourly_visitors,
)
from src.hod_v2 import (
    calibration_validation_hod_v2,
    tune_hod_v2_params,
)
from src.cleaning import build_quality_mask
from src.paths import CALIBRATED_FACILITIES


def run_task1_plan_b(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    manuals: pd.DataFrame,
    overlap_rate: float = 0.0,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    cal_total = build_calibration_windows(sessions, manuals, pedestrians_only=False)
    cal_ped = build_calibration_windows(sessions, manuals, pedestrians_only=True)

    models_comparison: List[Dict[str, Any]] = []
    for feat in ["trusted_unique_devices", "clean_unique_devices", "local_unique_devices"]:
        spec = fit_global_linear(cal_total, feat)
        models_comparison.append(
            {
                "target": "total_ped_bikes",
                "model": spec.name,
                "feature": feat,
                "intercept": spec.intercept,
                "slope": spec.slope,
                "mae": spec.mae,
                "mape": spec.mape,
                "r2": spec.r2,
            }
        )
    spec_ped = fit_global_linear(cal_ped, "clean_unique_devices")
    models_comparison.append(
        {
            "target": "pedestrians_only",
            "model": "global_linear_clean_unique_devices",
            "feature": "clean_unique_devices",
            "intercept": spec_ped.intercept,
            "slope": spec_ped.slope,
            "mae": spec_ped.mae,
            "mape": spec_ped.mape,
            "r2": spec_ped.r2,
        }
    )
    blend = fit_blend_trusted_clean(cal_total)
    models_comparison.append(
        {
            "target": "total_ped_bikes",
            "model": "blend_trusted_clean",
            "feature": "trusted+clean",
            "intercept": blend.intercept,
            "slope": blend.coef_trusted,
            "mae": blend.mae,
            "mape": blend.mape,
            "r2": blend.r2,
            "coef_clean": blend.coef_clean,
        }
    )
    models_df = pd.DataFrame(models_comparison)

    global_trusted = fit_global_linear(cal_total, "trusted_unique_devices")
    global_clean = fit_global_linear(cal_total, "clean_unique_devices")
    capture_rates = capture_rates_per_facility(cal_total)
    loo = leave_one_out_cv(cal_total, "clean_unique_devices")
    bootstrap = bootstrap_coefficients(cal_total, "clean_unique_devices")

    clean_sessions = sessions.loc[build_quality_mask(sessions)]
    profile = hour_of_day_profile(clean_sessions)
    neighbor_map = assign_facility_neighbors(facilities)
    hod_params, hod_loo_mae, hod_tune_grid = tune_hod_v2_params(
        cal_total, neighbor_map, profile, capture_rates_per_facility
    )
    cap_map = dict(zip(capture_rates["facility_num"], capture_rates["mean_capture_rate"]))
    hod_calibration_val = calibration_validation_hod_v2(
        cal_total, cap_map, neighbor_map, profile, hod_params
    )

    hourly = hourly_sensor_features(sessions)
    dedup_counts = mall_hour_primary_facility_counts(sessions)
    hourly = hourly.merge(dedup_counts, on=["facility_num", "hour_start"], how="left")
    hourly["clean_unique_devices_dedup"] = hourly["clean_unique_devices_dedup"].fillna(0)
    hourly["trusted_unique_devices_dedup"] = hourly["trusted_unique_devices_dedup"].fillna(0)
    hourly = apply_plan_b_estimates(
        hourly,
        cal_total,
        capture_rates,
        profile,
        global_trusted,
        neighbor_map,
        hod_params=hod_params,
    )

    mean_capture = float(
        cal_total["manual_total_count"].sum() / cal_total["clean_unique_devices"].sum()
    )
    mall_daily = mall_daily_dedup(sessions, hourly, overlap_rate, mean_capture)
    mall_hourly = mall_hourly_visitors(sessions, mean_capture)

    capture_export = capture_rates.rename(columns={"mean_capture_rate": "capture_rate"})
    capture_export["neighbor_facility"] = capture_export["facility_num"].map(neighbor_map)

    cal_total["pred_global_clean"] = (
        global_clean.intercept + global_clean.slope * cal_total["clean_unique_devices"]
    ).clip(lower=0)
    cal_total["pred_capture_rate"] = (
        cal_total["facility_num"].map(
            dict(zip(capture_rates["facility_num"], capture_rates["mean_capture_rate"]))
        )
        * cal_total["clean_unique_devices"]
    )

    meta = {
        "primary_method_plan_b": "footfall_plan_b",
        "plan_b_hod": "hod_v2 manual-anchored + shrinkage + density (LOO-tuned)",
        "hod_v2_params": hod_params.__dict__,
        "hod_v2_loo_mae": hod_loo_mae,
        "hod_v2_calibration_validation": hod_calibration_val.to_dict(orient="records"),
        "hod_v2_tune_grid_top": hod_tune_grid.head(10).to_dict(orient="records"),
        "plan_b_device_assignment": "one primary sensor per device per hour (max signal_level)",
        "plan_b_dedup_note": "footfall_plan_b uses clean_unique_devices_dedup; footfall_plan_b_per_sensor uses per-sensor counts",
        "plan_b_legacy_hod_column": "hod_factor_profile / footfall_plan_b_profile_hod",
        "primary_mall_metric": "estimated_mall_visitors",
        "global_clean": global_clean.__dict__,
        "global_trusted": global_trusted.__dict__,
        "bootstrap_clean": bootstrap,
        "neighbor_map": {str(k): v for k, v in neighbor_map.items()},
        "mean_mall_capture_rate": mean_capture,
        "calibrated_facilities": list(CALIBRATED_FACILITIES),
        "capture_rates": capture_export.to_dict(orient="records"),
    }
    return cal_total, models_df, loo, hourly, mall_daily, mall_hourly, capture_export, meta
