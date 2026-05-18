"""EDA summary tables for the assignment."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.cleaning import build_quality_mask


def flag_rates(sessions: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "is_excluded",
        "is_anomaly",
        "is_fake",
        "is_permanent_device",
        "is_local",
        "is_trusted",
    ]
    rates = {c: sessions[c].mean() for c in cols}
    rates["is_clean"] = build_quality_mask(sessions).mean()
    return pd.DataFrame([rates]).T.rename(columns={0: "rate"})


def volume_by_facility(sessions: pd.DataFrame) -> pd.DataFrame:
    return (
        sessions.groupby("facility_num", as_index=False)
        .agg(
            sessions=("device_id", "size"),
            unique_devices=("device_id", "nunique"),
            clean_sessions=("is_clean", "sum"),
        )
        .sort_values("sessions", ascending=False)
    )


def volume_by_hour(sessions: pd.DataFrame) -> pd.DataFrame:
    return (
        sessions.groupby("hour_of_day", as_index=False)
        .agg(sessions=("device_id", "size"), unique_devices=("device_id", "nunique"))
        .sort_values("hour_of_day")
    )


def mall_sanity(sessions: pd.DataFrame) -> pd.DataFrame:
    return sessions.groupby(["company_num", "facility_master_num"], as_index=False).agg(
        sessions=("device_id", "size")
    )


def daily_device_footfall_summary(
    sessions: pd.DataFrame, hourly_footfall: pd.DataFrame
) -> pd.DataFrame:
    """One row per UTC day: sessions, device categories, and estimated footfall."""
    clean = sessions.loc[build_quality_mask(sessions)].copy()
    sessions = sessions.copy()
    sessions["flagged_any"] = sessions[["is_excluded", "is_anomaly", "is_fake"]].any(
        axis=1
    )

    daily_sessions = sessions.groupby("date", as_index=False).agg(
        total_sessions=("device_id", "size"),
        unique_devices_mall=("device_id", "nunique"),
        flagged_sessions=("flagged_any", "sum"),
    )
    daily_clean = clean.groupby("date", as_index=False).agg(
        clean_sessions=("device_id", "size"),
        unique_devices_clean=("device_id", "nunique"),
    )
    trusted_by_day = (
        clean.loc[clean["is_trusted"]]
        .groupby("date", as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "trusted_devices"})
    )
    local_by_day = (
        clean.loc[clean["is_local"]]
        .groupby("date", as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "local_devices"})
    )
    other_by_day = (
        clean.loc[~clean["is_trusted"] & ~clean["is_local"]]
        .groupby("date", as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "other_devices"})
    )
    permanent_by_day = (
        sessions.loc[sessions["is_permanent_device"]]
        .groupby("date", as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "permanent_devices"})
    )
    sum_facility_uniques = (
        clean.groupby(["date", "facility_num"], as_index=False)["device_id"]
        .nunique()
        .groupby("date", as_index=False)["device_id"]
        .sum()
        .rename(columns={"device_id": "sum_facility_unique_devices"})
    )
    footfall_by_day = (
        hourly_footfall.assign(date=lambda df: df["hour_start"].dt.date)
        .groupby("date", as_index=False)["estimated_total_footfall"]
        .sum()
        .rename(columns={"estimated_total_footfall": "estimated_footfall"})
    )

    daily = daily_sessions.merge(daily_clean, on="date", how="left")
    for part in (
        trusted_by_day,
        local_by_day,
        other_by_day,
        permanent_by_day,
        sum_facility_uniques,
        footfall_by_day,
    ):
        daily = daily.merge(part, on="date", how="left")

    fill_cols = [
        "clean_sessions",
        "unique_devices_clean",
        "trusted_devices",
        "local_devices",
        "other_devices",
        "permanent_devices",
        "flagged_sessions",
        "sum_facility_unique_devices",
        "estimated_footfall",
    ]
    daily[fill_cols] = daily[fill_cols].fillna(0)
    return daily.sort_values("date").reset_index(drop=True)


def weekly_device_footfall_summary(
    daily_df: pd.DataFrame, sessions: pd.DataFrame
) -> Dict[str, Any]:
    """Week-level totals; unique device counts use mall-wide nunique, not summed daily."""
    clean = sessions.loc[build_quality_mask(sessions)]
    return {
        "total_sessions": int(daily_df["total_sessions"].sum()),
        "clean_sessions": int(daily_df["clean_sessions"].sum()),
        "flagged_sessions": int(daily_df["flagged_sessions"].sum()),
        "estimated_footfall": float(daily_df["estimated_footfall"].sum()),
        "unique_devices_mall_week": int(sessions["device_id"].nunique()),
        "unique_devices_clean_week": int(clean["device_id"].nunique()),
        "trusted_devices_week": int(clean.loc[clean["is_trusted"], "device_id"].nunique()),
        "local_devices_week": int(clean.loc[clean["is_local"], "device_id"].nunique()),
        "other_devices_week": int(
            clean.loc[~clean["is_trusted"] & ~clean["is_local"], "device_id"].nunique()
        ),
        "sum_facility_unique_devices_week": int(
            daily_df["sum_facility_unique_devices"].sum()
        ),
    }
