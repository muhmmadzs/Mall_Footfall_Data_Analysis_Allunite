from __future__ import annotations

from typing import Dict

import pandas as pd

SUSPICIOUS_SESSION_QUANTILE = 0.9999
SUSPICIOUS_NIGHT_RATIO_MIN = 0.60
SUSPICIOUS_NIGHT_SESSION_MIN = 120
HOURLY_FOOTFALL_Z_THRESHOLD = 3.0


def build_device_stats(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per-device behaviour metrics for anomaly pattern analysis."""
    flag_cols = ["is_excluded", "is_anomaly", "is_fake"]
    sessions = sessions.copy()
    sessions["flagged_any"] = sessions[flag_cols].any(axis=1)
    device_stats = (
        sessions.groupby("device_id", as_index=False)
        .agg(
            total_sessions=("device_id", "size"),
            unique_facilities=("facility_num", "nunique"),
            active_days=("date", "nunique"),
            night_sessions=("hour_of_day", lambda x: int((x <= 5).sum())),
            flagged_sessions=("flagged_any", "sum"),
            permanent_device=("is_permanent_device", "max"),
        )
        .sort_values("total_sessions", ascending=False)
    )
    device_stats["night_session_ratio"] = (
        device_stats["night_sessions"] / device_stats["total_sessions"]
    )
    return device_stats


def run_task4(
    sessions: pd.DataFrame, hourly_footfall: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], pd.DataFrame]:
    flag_cols = ["is_excluded", "is_anomaly", "is_fake"]
    sessions = sessions.copy()
    sessions["flagged_any"] = sessions[flag_cols].any(axis=1)

    flag_summary = {col: float(sessions[col].mean()) for col in flag_cols}
    flag_summary["flagged_any"] = float(sessions["flagged_any"].mean())

    device_stats = build_device_stats(sessions)
    threshold = float(device_stats["total_sessions"].quantile(SUSPICIOUS_SESSION_QUANTILE))

    suspicious = device_stats.loc[
        (~device_stats["permanent_device"])
        & (device_stats["flagged_sessions"] == 0)
        & (
            (device_stats["total_sessions"] >= threshold)
            | (
                (device_stats["night_session_ratio"] >= SUSPICIOUS_NIGHT_RATIO_MIN)
                & (device_stats["total_sessions"] >= SUSPICIOUS_NIGHT_SESSION_MIN)
            )
        )
    ].sort_values(["total_sessions", "night_session_ratio"], ascending=[False, False])

    anomaly_hours = hourly_footfall.copy()
    anomaly_hours["zscore"] = anomaly_hours.groupby("facility_num")[
        "estimated_total_footfall"
    ].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1))
    anomaly_hours["is_anomaly_hour"] = anomaly_hours["zscore"].abs() >= HOURLY_FOOTFALL_Z_THRESHOLD
    anomaly_hours = anomaly_hours.sort_values("zscore", ascending=False)

    summary = {
        "high_session_threshold_quantile": SUSPICIOUS_SESSION_QUANTILE,
        "high_session_threshold": threshold,
        "night_ratio_threshold": SUSPICIOUS_NIGHT_RATIO_MIN,
        "night_session_minimum": SUSPICIOUS_NIGHT_SESSION_MIN,
        "hourly_footfall_z_threshold": HOURLY_FOOTFALL_Z_THRESHOLD,
        "threshold_rationale": (
            "Suspicious devices are unflagged/non-permanent outliers in total sessions "
            "or heavily night-active devices; hourly footfall anomalies use |z| >= 3 "
            "within each facility."
        ),
        "suspicious_unflagged_devices": int(len(suspicious)),
        **flag_summary,
    }
    return suspicious, anomaly_hours, summary, device_stats
