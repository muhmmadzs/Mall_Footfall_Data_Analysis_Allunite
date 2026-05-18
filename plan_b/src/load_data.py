"""Load data from shared repo data/ folder."""

from __future__ import annotations

import json
from typing import Tuple

import pandas as pd

from src.cleaning import build_quality_mask
from src.paths import FACILITY_CSV, MANUAL_CSV, SESSION_CSV


def load_sessions() -> pd.DataFrame:
    sessions = pd.read_csv(SESSION_CSV, parse_dates=["session_start"], low_memory=False)
    numeric_cols = [
        "device_id",
        "company_num",
        "facility_num",
        "facility_master_num",
        "session_duration",
        "signal_level",
        "vendor_id",
    ]
    for col in numeric_cols:
        sessions[col] = pd.to_numeric(sessions[col], errors="coerce")
    bool_cols = [
        "is_permanent_device",
        "is_local",
        "is_trusted",
        "is_excluded",
        "is_anomaly",
        "is_fake",
    ]
    for col in bool_cols:
        sessions[col] = sessions[col].astype("boolean").fillna(False).astype(bool)
    sessions = sessions.dropna(subset=["device_id", "facility_num", "session_start"]).copy()
    sessions["device_id"] = sessions["device_id"].astype("int64")
    sessions["facility_num"] = sessions["facility_num"].astype("int32")
    sessions["session_duration"] = sessions["session_duration"].fillna(0).astype("int32")
    sessions["signal_level"] = sessions["signal_level"].fillna(0).astype("int16")
    if sessions["session_start"].dt.tz is None:
        sessions["session_start"] = sessions["session_start"].dt.tz_localize("UTC")
    sessions["hour_start"] = sessions["session_start"].dt.floor("h")
    sessions["date"] = sessions["session_start"].dt.date
    sessions["hour_of_day"] = sessions["session_start"].dt.hour
    sessions["is_clean"] = build_quality_mask(sessions)
    return sessions


def load_facilities() -> pd.DataFrame:
    facilities = pd.read_csv(FACILITY_CSV)
    facilities["box_mac_list"] = facilities["box_macs"].apply(_parse_box_macs)
    return facilities


def _parse_box_macs(value: object) -> list[str]:
    if pd.isna(value) or value == "[]":
        return []
    try:
        parsed = json.loads(str(value).replace('""', '"'))
        return [str(m).lower() for m in parsed]
    except json.JSONDecodeError:
        return []


def load_manual_counts() -> pd.DataFrame:
    manuals = pd.read_csv(MANUAL_CSV)
    manuals["started"] = pd.to_datetime(manuals["started"], utc=True)
    manuals["duration"] = manuals["duration"].astype(int)
    manuals["window_end"] = manuals["started"] + pd.to_timedelta(manuals["duration"], unit="m")
    return manuals


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_sessions(), load_facilities(), load_manual_counts()
