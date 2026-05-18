"""Plan B anomaly detection: robust z-score, hour-of-day baseline, device weighting."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.cleaning import build_quality_mask


def _robust_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else 1.0
    return (series - med) / scale


def build_device_stats(sessions: pd.DataFrame) -> pd.DataFrame:
    flag_cols = ["is_excluded", "is_anomaly", "is_fake"]
    sessions = sessions.copy()
    sessions["flagged_any"] = sessions[flag_cols].any(axis=1)
    stats = (
        sessions.groupby("device_id", as_index=False)
        .agg(
            total_sessions=("device_id", "size"),
            unique_facilities=("facility_num", "nunique"),
            active_days=("date", "nunique"),
            night_sessions=("hour_of_day", lambda x: int((x <= 5).sum())),
            flagged_sessions=("flagged_any", "sum"),
            permanent_device=("is_permanent_device", "max"),
            trusted_sessions=("is_trusted", "sum"),
        )
        .sort_values("total_sessions", ascending=False)
    )
    stats["night_session_ratio"] = stats["night_sessions"] / stats["total_sessions"]
    return stats


def run_task4_plan_b(
    sessions: pd.DataFrame, hourly_footfall: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], pd.DataFrame]:
    flag_cols = ["is_excluded", "is_anomaly", "is_fake"]
    sessions = sessions.copy()
    sessions["flagged_any"] = sessions[flag_cols].any(axis=1)
    flag_summary = {col: float(sessions[col].mean()) for col in flag_cols}
    flag_summary["flagged_any"] = float(sessions["flagged_any"].mean())

    device_stats = build_device_stats(sessions)
    threshold = float(device_stats["total_sessions"].quantile(0.9999))

    suspicious = device_stats.loc[
        (~device_stats["permanent_device"])
        & (device_stats["flagged_sessions"] == 0)
        & (
            (device_stats["total_sessions"] >= threshold)
            | (
                (device_stats["night_session_ratio"] >= 0.6)
                & (device_stats["total_sessions"] >= 120)
            )
        )
    ].sort_values(["total_sessions", "night_session_ratio"], ascending=[False, False])

    anomaly = hourly_footfall.copy()
    anomaly["hour_of_day"] = anomaly["hour_start"].dt.hour
    footfall_col = "footfall_plan_b" if "footfall_plan_b" in anomaly.columns else "estimated_total_footfall"

    anomaly["zscore_global"] = anomaly.groupby("facility_num")[footfall_col].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1)
    )
    anomaly["zscore_robust"] = anomaly.groupby("facility_num")[footfall_col].transform(_robust_z)

    hod_mean = anomaly.groupby(["facility_num", "hour_of_day"])[footfall_col].transform("mean")
    hod_std = anomaly.groupby(["facility_num", "hour_of_day"])[footfall_col].transform(
        lambda s: s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0
    )
    anomaly["zscore_hod_baseline"] = (anomaly[footfall_col] - hod_mean) / hod_std
    anomaly["residual_hod"] = anomaly[footfall_col] - hod_mean
    anomaly["is_anomaly_global"] = anomaly["zscore_global"].abs() >= 3
    anomaly["is_anomaly_robust"] = anomaly["zscore_robust"].abs() >= 3.5
    anomaly["is_anomaly_hod"] = anomaly["zscore_hod_baseline"].abs() >= 3
    anomaly["is_anomaly_consensus"] = (
        anomaly["is_anomaly_robust"].astype(int)
        + anomaly["is_anomaly_hod"].astype(int)
        + anomaly["is_anomaly_global"].astype(int)
    ) >= 2

    # Optional STL per facility
    try:
        from statsmodels.tsa.seasonal import STL

        stl_residuals = []
        for fac, grp in anomaly.groupby("facility_num", sort=False):
            grp = grp.sort_values("hour_start")
            vals = grp[footfall_col].to_numpy(dtype=float)
            if len(vals) >= 48:
                res = STL(vals, period=24, robust=True).fit().resid
            else:
                res = vals - np.mean(vals)
            stl_residuals.extend(res.tolist())
        anomaly["stl_residual"] = stl_residuals
        anomaly["stl_zscore"] = anomaly.groupby("facility_num")["stl_residual"].transform(
            _robust_z
        )
        anomaly["is_anomaly_stl"] = anomaly["stl_zscore"].abs() >= 3.5
    except ImportError:
        anomaly["stl_residual"] = np.nan
        anomaly["stl_zscore"] = np.nan
        anomaly["is_anomaly_stl"] = False

    anomaly = anomaly.sort_values("zscore_hod_baseline", key=abs, ascending=False)

    summary = {
        "high_session_threshold": threshold,
        "suspicious_unflagged_devices": int(len(suspicious)),
        "anomaly_hours_global": int(anomaly["is_anomaly_global"].sum()),
        "anomaly_hours_robust": int(anomaly["is_anomaly_robust"].sum()),
        "anomaly_hours_hod": int(anomaly["is_anomaly_hod"].sum()),
        "anomaly_hours_consensus": int(anomaly["is_anomaly_consensus"].sum()),
        **flag_summary,
    }
    return suspicious, anomaly, summary, device_stats
