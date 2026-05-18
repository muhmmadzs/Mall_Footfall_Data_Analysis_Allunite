from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from src.cleaning import build_quality_mask


@dataclass
class CalibrationResult:
    feature_name: str
    intercept: float
    slope: float
    mae: float
    mape: float
    r2: float


CANDIDATE_FEATURES = [
    "raw_unique_devices",
    "clean_unique_devices",
    "raw_sessions",
    "clean_sessions",
    "trusted_unique_devices",
    "local_unique_devices",
]


def _window_features(raw: pd.DataFrame, clean: pd.DataFrame) -> Dict[str, float]:
    return {
        "raw_sessions": float(len(raw)),
        "raw_unique_devices": float(raw["device_id"].nunique()),
        "clean_sessions": float(len(clean)),
        "clean_unique_devices": float(clean["device_id"].nunique()),
        "trusted_unique_devices": float(
            clean.loc[clean["is_trusted"], "device_id"].nunique()
        ),
        "local_unique_devices": float(
            clean.loc[clean["is_local"], "device_id"].nunique()
        ),
        "mean_signal_level": float(clean["signal_level"].mean()) if len(clean) else 0.0,
        "median_session_duration": float(clean["session_duration"].median())
        if len(clean)
        else 0.0,
    }


def run_task1(
    sessions: pd.DataFrame, manuals: pd.DataFrame
) -> tuple[pd.DataFrame, CalibrationResult, pd.DataFrame]:
    manual_total = (
        manuals.groupby(["facility_num", "started", "duration", "window_end"], as_index=False)[
            "count"
        ]
        .sum()
        .rename(columns={"count": "manual_total_count"})
    )
    manual_ped = (
        manuals.loc[manuals["category"] == "Pedestrians"]
        .groupby(["facility_num", "started"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "manual_pedestrian_count"})
    )
    windows = manual_total.merge(manual_ped, on=["facility_num", "started"], how="left")

    calibration_rows: List[Dict] = []
    for row in windows.itertuples(index=False):
        mask = (
            (sessions["facility_num"] == row.facility_num)
            & (sessions["session_start"] >= row.started)
            & (sessions["session_start"] < row.window_end)
        )
        raw_window = sessions.loc[mask]
        clean_window = raw_window.loc[build_quality_mask(raw_window)]
        features = _window_features(raw_window, clean_window)
        features.update(
            {
                "facility_num": int(row.facility_num),
                "started": row.started,
                "window_end": row.window_end,
                "manual_total_count": float(row.manual_total_count),
                "manual_pedestrian_count": float(row.manual_pedestrian_count),
            }
        )
        calibration_rows.append(features)

    calibration_df = pd.DataFrame(calibration_rows)

    correlations: Dict[str, float] = {}
    for feature in CANDIDATE_FEATURES:
        if calibration_df[feature].std() == 0:
            correlations[feature] = 0.0
        else:
            correlations[feature] = float(
                calibration_df[feature].corr(calibration_df["manual_total_count"])
            )
    selected = max(correlations, key=lambda k: abs(correlations[k]))

    x = calibration_df[[selected]].to_numpy()
    y = calibration_df["manual_total_count"].to_numpy()
    model = LinearRegression()
    model.fit(x, y)
    calibration_df["estimated_total_count"] = model.predict(x).clip(min=0)

    result = CalibrationResult(
        feature_name=selected,
        intercept=float(model.intercept_),
        slope=float(model.coef_[0]),
        mae=float(mean_absolute_error(y, calibration_df["estimated_total_count"])),
        mape=float(
            mean_absolute_percentage_error(y, calibration_df["estimated_total_count"])
        ),
        r2=float(r2_score(y, calibration_df["estimated_total_count"])),
    )

    quality_mask = build_quality_mask(sessions)
    raw_hourly = (
        sessions.groupby(["facility_num", "hour_start"], as_index=False)
        .agg(
            raw_sessions=("device_id", "size"),
            raw_unique_devices=("device_id", "nunique"),
        )
    )
    clean_sessions = sessions.loc[quality_mask]
    clean_hourly = (
        clean_sessions.groupby(["facility_num", "hour_start"], as_index=False)
        .agg(
            clean_sessions=("device_id", "size"),
            clean_unique_devices=("device_id", "nunique"),
            mean_signal_level=("signal_level", "mean"),
            median_session_duration=("session_duration", "median"),
        )
    )
    trusted_hourly = (
        clean_sessions.loc[clean_sessions["is_trusted"]]
        .groupby(["facility_num", "hour_start"], as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "trusted_unique_devices"})
    )
    local_hourly = (
        clean_sessions.loc[clean_sessions["is_local"]]
        .groupby(["facility_num", "hour_start"], as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "local_unique_devices"})
    )

    hourly = raw_hourly.merge(clean_hourly, on=["facility_num", "hour_start"], how="left")
    hourly = hourly.merge(trusted_hourly, on=["facility_num", "hour_start"], how="left")
    hourly = hourly.merge(local_hourly, on=["facility_num", "hour_start"], how="left")
    hourly = hourly.fillna(
        {
            "clean_sessions": 0,
            "clean_unique_devices": 0,
            "trusted_unique_devices": 0,
            "local_unique_devices": 0,
            "mean_signal_level": 0,
            "median_session_duration": 0,
        }
    )
    hourly["footfall_plan_a"] = (
        result.intercept + result.slope * hourly[selected]
    ).clip(lower=0).round(2)
    hourly["estimated_total_footfall"] = hourly["footfall_plan_a"]

    return calibration_df, result, hourly
