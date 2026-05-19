from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.seasonal import STL


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _path_from_env("MALL_FOOTFALL_DATA_DIR", PROJECT_ROOT / "data")
SESSION_CSV = _path_from_env(
    "MALL_FOOTFALL_SESSION_CSV",
    DATA_DIR / "allunite_device_session.csv",
)
FACILITY_CSV = _path_from_env(
    "MALL_FOOTFALL_FACILITY_CSV",
    DATA_DIR / "facility_information - Sheet1.csv",
)
MANUAL_CSV = _path_from_env(
    "MALL_FOOTFALL_MANUAL_CSV",
    DATA_DIR / "Manual Counting - Sheet1.csv",
)

OUTPUT_DIR = _path_from_env("MALL_FOOTFALL_OUTPUT_DIR", PROJECT_ROOT / "outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CalibrationResult:
    feature_name: str
    intercept: float
    slope: float
    mae: float
    mape: float
    r2: float


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = pd.read_csv(
        SESSION_CSV,
        parse_dates=["session_start"],
        low_memory=False,
    )

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
    sessions["company_num"] = sessions["company_num"].fillna(-1).astype("int32")
    sessions["facility_num"] = sessions["facility_num"].astype("int32")
    sessions["facility_master_num"] = sessions["facility_master_num"].fillna(-1).astype("int32")
    sessions["session_duration"] = sessions["session_duration"].fillna(0).astype("int32")
    sessions["signal_level"] = sessions["signal_level"].fillna(0).astype("int16")
    sessions["vendor_id"] = sessions["vendor_id"].fillna(0).astype("int32")
    sessions["box_mac"] = sessions["box_mac"].astype("string")
    if sessions["session_start"].dt.tz is None:
        sessions["session_start"] = sessions["session_start"].dt.tz_localize("UTC")
    sessions["hour_start"] = sessions["session_start"].dt.floor("h")
    sessions["date"] = sessions["session_start"].dt.date
    sessions["hour_of_day"] = sessions["session_start"].dt.hour

    facilities = pd.read_csv(FACILITY_CSV)
    manuals = pd.read_csv(MANUAL_CSV)
    manuals["started"] = pd.to_datetime(manuals["started"], utc=True)
    manuals["duration"] = manuals["duration"].astype(int)
    manuals["window_end"] = manuals["started"] + pd.to_timedelta(manuals["duration"], unit="m")
    return sessions, facilities, manuals


def build_quality_mask(df: pd.DataFrame) -> pd.Series:
    return (
        ~df["is_excluded"]
        & ~df["is_anomaly"]
        & ~df["is_fake"]
        & ~df["is_permanent_device"]
    )


def aggregate_window_features(raw_window: pd.DataFrame, clean_window: pd.DataFrame) -> Dict[str, float]:
    features = {
        "raw_sessions": float(len(raw_window)),
        "raw_unique_devices": float(raw_window["device_id"].nunique()),
        "clean_sessions": float(len(clean_window)),
        "clean_unique_devices": float(clean_window["device_id"].nunique()),
        "trusted_unique_devices": float(
            clean_window.loc[clean_window["is_trusted"], "device_id"].nunique()
        ),
        "local_unique_devices": float(
            clean_window.loc[clean_window["is_local"], "device_id"].nunique()
        ),
        "mean_signal_level": float(clean_window["signal_level"].mean()) if len(clean_window) else 0.0,
        "median_session_duration": float(clean_window["session_duration"].median()) if len(clean_window) else 0.0,
    }
    return features


def task1_calibration(
    sessions: pd.DataFrame, manuals: pd.DataFrame
) -> tuple[pd.DataFrame, CalibrationResult, pd.DataFrame]:
    manual_total = (
        manuals.groupby(["facility_num", "started", "duration", "window_end"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "manual_total_count"})
    )
    manual_ped = (
        manuals.loc[manuals["category"] == "Pedestrians"]
        .groupby(["facility_num", "started"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "manual_pedestrian_count"})
    )
    windows = manual_total.merge(
        manual_ped, on=["facility_num", "started"], how="left"
    ).sort_values(["facility_num", "started"])

    quality_mask = build_quality_mask(sessions)
    calibration_rows: List[Dict[str, float]] = []
    for row in windows.itertuples(index=False):
        raw_window = sessions.loc[
            (sessions["facility_num"] == row.facility_num)
            & (sessions["session_start"] >= row.started)
            & (sessions["session_start"] < row.window_end)
        ]
        clean_window = raw_window.loc[build_quality_mask(raw_window)]
        feature_row = aggregate_window_features(raw_window, clean_window)
        feature_row.update(
            {
                "facility_num": int(row.facility_num),
                "started": row.started,
                "window_end": row.window_end,
                "manual_total_count": float(row.manual_total_count),
                "manual_pedestrian_count": float(row.manual_pedestrian_count),
            }
        )
        calibration_rows.append(feature_row)

    calibration_df = pd.DataFrame(calibration_rows)

    candidate_features = [
        "raw_unique_devices",
        "clean_unique_devices",
        "raw_sessions",
        "clean_sessions",
        "trusted_unique_devices",
        "local_unique_devices",
    ]
    correlations: Dict[str, float] = {}
    for feature in candidate_features:
        if calibration_df[feature].std() == 0:
            correlations[feature] = 0.0
            continue
        correlations[feature] = float(
            calibration_df[feature].corr(calibration_df["manual_total_count"])
        )
    selected_feature = max(correlations, key=lambda key: abs(correlations[key]))

    x_train = calibration_df[[selected_feature]].to_numpy()
    y_train = calibration_df["manual_total_count"].to_numpy()
    model = LinearRegression()
    model.fit(x_train, y_train)
    calibration_df["estimated_total_count"] = model.predict(x_train).clip(min=0)

    cal_result = CalibrationResult(
        feature_name=selected_feature,
        intercept=float(model.intercept_),
        slope=float(model.coef_[0]),
        mae=float(mean_absolute_error(y_train, calibration_df["estimated_total_count"])),
        mape=float(mean_absolute_percentage_error(y_train, calibration_df["estimated_total_count"])),
        r2=float(r2_score(y_train, calibration_df["estimated_total_count"])),
    )

    # Build hourly feature frame for full-week estimates.
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

    hourly["estimated_total_footfall"] = (
        cal_result.intercept + cal_result.slope * hourly[selected_feature]
    ).clip(lower=0)
    hourly["estimated_total_footfall"] = hourly["estimated_total_footfall"].round(2)

    return calibration_df, cal_result, hourly


def task2_sensor_intersection(
    sessions: pd.DataFrame, facilities: pd.DataFrame
) -> tuple[pd.DataFrame, Dict[str, float]]:
    clean_sessions = sessions.loc[build_quality_mask(sessions), ["device_id", "facility_num"]]
    device_fac = clean_sessions.drop_duplicates()

    facility_counts = (
        device_fac.groupby("facility_num", as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "devices_at_facility"})
    )
    facility_counts_dict = dict(zip(facility_counts["facility_num"], facility_counts["devices_at_facility"]))

    device_facility_n = (
        device_fac.groupby("device_id")["facility_num"].nunique().rename("facility_count")
    )
    multi_sensor_devices = int((device_facility_n > 1).sum())
    total_devices = int(device_facility_n.shape[0])

    pairs = device_fac.merge(device_fac, on="device_id")
    pairs = pairs.loc[pairs["facility_num_x"] < pairs["facility_num_y"]]
    pair_counts = (
        pairs.groupby(["facility_num_x", "facility_num_y"], as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "shared_devices"})
    )
    pair_counts["devices_facility_x"] = pair_counts["facility_num_x"].map(facility_counts_dict)
    pair_counts["devices_facility_y"] = pair_counts["facility_num_y"].map(facility_counts_dict)
    pair_counts["union_devices"] = (
        pair_counts["devices_facility_x"] + pair_counts["devices_facility_y"] - pair_counts["shared_devices"]
    )
    pair_counts["jaccard_overlap"] = (
        pair_counts["shared_devices"] / pair_counts["union_devices"]
    ).round(6)
    pair_counts["overlap_pct_smaller_side"] = (
        pair_counts["shared_devices"]
        / pair_counts[["devices_facility_x", "devices_facility_y"]].min(axis=1)
    ).round(6)

    facility_lookup = facilities[["facility_num", "facility_name"]].drop_duplicates()
    pair_counts = pair_counts.merge(
        facility_lookup.rename(columns={"facility_num": "facility_num_x", "facility_name": "facility_name_x"}),
        on="facility_num_x",
        how="left",
    ).merge(
        facility_lookup.rename(columns={"facility_num": "facility_num_y", "facility_name": "facility_name_y"}),
        on="facility_num_y",
        how="left",
    )
    pair_counts = pair_counts.sort_values("shared_devices", ascending=False)

    summary = {
        "total_unique_devices_clean": total_devices,
        "multi_sensor_devices": multi_sensor_devices,
        "multi_sensor_rate": float(multi_sensor_devices / total_devices if total_devices else 0.0),
    }
    return pair_counts, summary


def task3_journeys(
    sessions: pd.DataFrame, facilities: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    clean_sessions = sessions.loc[build_quality_mask(sessions), ["device_id", "facility_num", "session_start"]]
    device_facility_n = clean_sessions.groupby("device_id")["facility_num"].nunique()
    multi_devices = device_facility_n.loc[device_facility_n > 1].index
    journey_events = clean_sessions.loc[clean_sessions["device_id"].isin(multi_devices)].copy()

    journey_events = journey_events.sort_values(["device_id", "session_start"])
    journey_events["prev_facility"] = journey_events.groupby("device_id")["facility_num"].shift()
    journey_events = journey_events.loc[
        journey_events["prev_facility"].isna()
        | (journey_events["facility_num"] != journey_events["prev_facility"])
    ].copy()

    journey_events["next_facility"] = journey_events.groupby("device_id")["facility_num"].shift(-1)
    transitions = journey_events.loc[
        journey_events["next_facility"].notna(),
        ["facility_num", "next_facility"],
    ].copy()
    transitions["next_facility"] = transitions["next_facility"].astype(int)
    transition_counts = (
        transitions.groupby(["facility_num", "next_facility"], as_index=False)
        .size()
        .rename(columns={"size": "transition_count"})
        .sort_values("transition_count", ascending=False)
    )

    journey_paths = (
        journey_events.groupby("device_id")
        .agg(
            first_seen=("session_start", "min"),
            last_seen=("session_start", "max"),
            unique_facilities=("facility_num", "nunique"),
            path_steps=("facility_num", "size"),
            path=("facility_num", lambda x: " -> ".join(map(str, x.tolist()))),
        )
        .reset_index()
    )
    top_paths = (
        journey_paths.groupby("path", as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "device_count"})
        .sort_values("device_count", ascending=False)
    )
    top_paths["path_length"] = top_paths["path"].str.count("->") + 1

    sample_paths = journey_paths.sample(n=min(200, len(journey_paths)), random_state=42)

    facility_lookup = facilities[["facility_num", "facility_name"]].drop_duplicates()
    transition_counts = transition_counts.merge(
        facility_lookup.rename(columns={"facility_num": "facility_num", "facility_name": "facility_name_from"}),
        on="facility_num",
        how="left",
    ).merge(
        facility_lookup.rename(columns={"facility_num": "next_facility", "facility_name": "facility_name_to"}),
        on="next_facility",
        how="left",
    )

    return top_paths, transition_counts, sample_paths, int(len(journey_paths))


def task4_anomalies(
    sessions: pd.DataFrame, hourly_footfall: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    flag_cols = ["is_excluded", "is_anomaly", "is_fake"]
    sessions = sessions.copy()
    sessions["flagged_any"] = sessions[flag_cols].any(axis=1)

    flag_summary = {col: float(sessions[col].mean()) for col in flag_cols}
    flag_summary["flagged_any"] = float(sessions["flagged_any"].mean())

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
    high_session_threshold = float(device_stats["total_sessions"].quantile(0.9999))

    suspicious = device_stats.loc[
        (~device_stats["permanent_device"])
        & (device_stats["flagged_sessions"] == 0)
        & (
            (device_stats["total_sessions"] >= high_session_threshold)
            | (
                (device_stats["night_session_ratio"] >= 0.6)
                & (device_stats["total_sessions"] >= 120)
            )
        )
    ].copy()
    suspicious = suspicious.sort_values(
        ["total_sessions", "night_session_ratio"], ascending=[False, False]
    )

    anomaly_hours = hourly_footfall.copy()
    anomaly_hours["zscore"] = anomaly_hours.groupby("facility_num")[
        "estimated_total_footfall"
    ].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1))
    anomaly_hours["is_anomaly_hour"] = anomaly_hours["zscore"].abs() >= 3
    anomaly_hours = anomaly_hours.sort_values("zscore", ascending=False)

    summary = {
        "high_session_threshold_top_0_1pct": high_session_threshold,
        "suspicious_unflagged_devices": int(len(suspicious)),
        **flag_summary,
    }
    return suspicious, anomaly_hours, summary


def _robust_abs_zscore(values: pd.Series) -> pd.Series:
    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = 1.4826 * mad if mad > 0 else 1.0
    return (values - median).abs() / scale


def _minmax(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)
    lo = float(valid.min())
    hi = float(valid.max())
    if hi <= lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def _matrix_profile_single_series(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    k = n - window + 1
    if k < 2:
        return np.array([], dtype=float)

    subseqs = np.lib.stride_tricks.sliding_window_view(values, window)
    means = subseqs.mean(axis=1, keepdims=True)
    stds = subseqs.std(axis=1, keepdims=True)
    stds_safe = np.where(stds > 1e-9, stds, 1.0)
    zsubseqs = (subseqs - means) / stds_safe
    valid = (stds[:, 0] > 1e-9)

    profile = np.full(k, np.nan, dtype=float)
    trivial_zone = int(np.ceil(window / 2))
    for idx in range(k):
        if not valid[idx]:
            continue
        distances = np.linalg.norm(zsubseqs - zsubseqs[idx], axis=1)
        left = max(0, idx - trivial_zone)
        right = min(k, idx + trivial_zone + 1)
        distances[left:right] = np.inf
        distances[~valid] = np.inf
        nearest = distances.min()
        profile[idx] = float(nearest) if np.isfinite(nearest) else np.nan
    return profile


def matrix_profile_hourly_anomalies(
    hourly_footfall: pd.DataFrame, windows: tuple[int, ...] = (6, 12, 24)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: List[pd.DataFrame] = []
    details: List[pd.DataFrame] = []
    for facility_num, grp in hourly_footfall.groupby("facility_num", sort=False):
        grp = grp.sort_values("hour_start").copy()
        values = grp["estimated_total_footfall"].to_numpy(dtype=float)
        base = grp[["facility_num", "hour_start"]].copy()
        base["matrix_profile_score"] = 0.0
        base["matrix_profile_window"] = np.nan

        for window in windows:
            profile = _matrix_profile_single_series(values, window)
            if profile.size == 0:
                continue
            centers = np.arange(window // 2, window // 2 + len(profile))
            mp_z = _robust_abs_zscore(pd.Series(profile, dtype=float)).to_numpy()
            window_scores = np.zeros(len(base), dtype=float)
            window_scores[centers] = mp_z

            better = window_scores > base["matrix_profile_score"].to_numpy()
            base.loc[better, "matrix_profile_score"] = window_scores[better]
            base.loc[better, "matrix_profile_window"] = window

            detail = pd.DataFrame(
                {
                    "facility_num": facility_num,
                    "hour_start": grp.iloc[centers]["hour_start"].to_numpy(),
                    "window_size": window,
                    "matrix_profile_distance": profile,
                    "matrix_profile_robust_z": mp_z,
                }
            )
            details.append(detail)

        threshold = float(base["matrix_profile_score"].quantile(0.97))
        base["is_matrix_profile_anomaly"] = base["matrix_profile_score"] >= threshold
        summaries.append(base)

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(
        columns=[
            "facility_num",
            "hour_start",
            "matrix_profile_score",
            "matrix_profile_window",
            "is_matrix_profile_anomaly",
        ]
    )
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame(
        columns=[
            "facility_num",
            "hour_start",
            "window_size",
            "matrix_profile_distance",
            "matrix_profile_robust_z",
        ]
    )
    return summary_df.sort_values(["facility_num", "hour_start"]), detail_df


def stl_hourly_anomalies(
    hourly_footfall: pd.DataFrame, period: int = 24
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for facility_num, grp in hourly_footfall.groupby("facility_num", sort=False):
        grp = grp.sort_values("hour_start").copy()
        series = grp["estimated_total_footfall"].astype(float).to_numpy()

        if len(series) < period * 2:
            grp["stl_trend"] = np.nan
            grp["stl_seasonal"] = np.nan
            grp["stl_resid"] = series - np.mean(series)
            grp["stl_score"] = _robust_abs_zscore(grp["stl_resid"])
        else:
            stl_res = STL(series, period=period, robust=True).fit()
            grp["stl_trend"] = stl_res.trend
            grp["stl_seasonal"] = stl_res.seasonal
            grp["stl_resid"] = stl_res.resid
            grp["stl_score"] = _robust_abs_zscore(grp["stl_resid"])

        threshold = float(grp["stl_score"].quantile(0.97))
        grp["is_stl_anomaly"] = grp["stl_score"] >= threshold
        rows.append(grp)

    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "facility_num",
            "hour_start",
            "stl_trend",
            "stl_seasonal",
            "stl_resid",
            "stl_score",
            "is_stl_anomaly",
        ]
    ].sort_values(["facility_num", "hour_start"])


def isolation_forest_hourly_anomalies(
    hourly_footfall: pd.DataFrame,
) -> pd.DataFrame:
    feature_cols = [
        "raw_sessions",
        "raw_unique_devices",
        "clean_sessions",
        "clean_unique_devices",
        "trusted_unique_devices",
        "local_unique_devices",
        "mean_signal_level",
        "median_session_duration",
        "estimated_total_footfall",
    ]
    rows: List[pd.DataFrame] = []
    for facility_num, grp in hourly_footfall.groupby("facility_num", sort=False):
        grp = grp.sort_values("hour_start").copy()
        x = grp[feature_cols].fillna(0.0).to_numpy(dtype=float)
        model = IsolationForest(
            n_estimators=300,
            contamination=0.03,
            random_state=42,
        )
        model.fit(x)
        # score_samples is higher for inliers; negate for anomaly-intuitive direction.
        scores = -model.score_samples(x)
        grp["isolation_forest_score"] = scores
        threshold = float(np.quantile(scores, 0.97))
        grp["is_isolation_forest_anomaly"] = grp["isolation_forest_score"] >= threshold
        rows.append(grp)

    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "facility_num",
            "hour_start",
            "isolation_forest_score",
            "is_isolation_forest_anomaly",
        ]
    ].sort_values(["facility_num", "hour_start"])


def _build_journey_events(sessions: pd.DataFrame) -> pd.DataFrame:
    clean_sessions = sessions.loc[build_quality_mask(sessions), ["device_id", "facility_num", "session_start"]]
    device_facility_n = clean_sessions.groupby("device_id")["facility_num"].nunique()
    multi_devices = device_facility_n.loc[device_facility_n > 1].index
    journey_events = clean_sessions.loc[clean_sessions["device_id"].isin(multi_devices)].copy()
    journey_events = journey_events.sort_values(["device_id", "session_start"])
    journey_events["prev_facility"] = journey_events.groupby("device_id")["facility_num"].shift()
    journey_events = journey_events.loc[
        journey_events["prev_facility"].isna()
        | (journey_events["facility_num"] != journey_events["prev_facility"])
    ].copy()
    journey_events["next_facility"] = journey_events.groupby("device_id")["facility_num"].shift(-1)
    return journey_events


def markov_journey_anomalies(sessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = _build_journey_events(sessions)
    trans = events.loc[events["next_facility"].notna(), ["facility_num", "next_facility"]].copy()
    trans["next_facility"] = trans["next_facility"].astype(int)
    states = np.sort(events["facility_num"].dropna().unique())
    state_count = len(states)
    alpha = 1.0

    trans_counts = (
        trans.groupby(["facility_num", "next_facility"], as_index=False)
        .size()
        .rename(columns={"size": "transition_count"})
    )
    row_totals = trans_counts.groupby("facility_num", as_index=False)["transition_count"].sum().rename(
        columns={"transition_count": "row_total"}
    )
    trans_probs = trans_counts.merge(row_totals, on="facility_num", how="left")
    trans_probs["transition_prob"] = (
        (trans_probs["transition_count"] + alpha)
        / (trans_probs["row_total"] + alpha * state_count)
    )
    prob_lookup = {
        (int(r.facility_num), int(r.next_facility)): float(r.transition_prob)
        for r in trans_probs.itertuples(index=False)
    }
    row_lookup = {int(r.facility_num): int(r.row_total) for r in row_totals.itertuples(index=False)}

    journey_rows: List[Dict[str, object]] = []
    for device_id, grp in events.groupby("device_id", sort=False):
        seq = grp["facility_num"].astype(int).tolist()
        times = grp["session_start"].tolist()
        if len(seq) < 2:
            continue
        nll_values: List[float] = []
        rare_count = 0
        for i in range(len(seq) - 1):
            src = seq[i]
            dst = seq[i + 1]
            prob = prob_lookup.get(
                (src, dst),
                alpha / (row_lookup.get(src, 0) + alpha * state_count),
            )
            prob = max(prob, 1e-12)
            nll = -np.log(prob)
            if prob < 0.02:
                rare_count += 1
            nll_values.append(float(nll))

        avg_nll = float(np.mean(nll_values))
        journey_rows.append(
            {
                "device_id": int(device_id),
                "start_time": times[0],
                "end_time": times[-1],
                "end_hour_start": pd.Timestamp(times[-1]).floor("h"),
                "start_facility": int(seq[0]),
                "end_facility": int(seq[-1]),
                "path_steps": len(seq),
                "path": " -> ".join(map(str, seq)),
                "markov_avg_neg_log_prob": avg_nll,
                "rare_transition_ratio": float(rare_count / max(1, len(nll_values))),
            }
        )

    journey_df = pd.DataFrame(journey_rows)
    if journey_df.empty:
        empty_j = pd.DataFrame(
            columns=[
                "device_id",
                "start_time",
                "end_time",
                "end_hour_start",
                "start_facility",
                "end_facility",
                "path_steps",
                "path",
                "markov_avg_neg_log_prob",
                "rare_transition_ratio",
                "markov_score",
                "is_markov_journey_anomaly",
            ]
        )
        empty_h = pd.DataFrame(
            columns=[
                "facility_num",
                "hour_start",
                "markov_score",
                "markov_journey_count",
                "is_markov_hour_anomaly",
            ]
        )
        return empty_j, empty_h

    journey_df["markov_score"] = (
        _robust_abs_zscore(journey_df["markov_avg_neg_log_prob"])
        + _robust_abs_zscore(journey_df["rare_transition_ratio"])
    ) / 2.0
    threshold = float(journey_df["markov_score"].quantile(0.97))
    journey_df["is_markov_journey_anomaly"] = journey_df["markov_score"] >= threshold
    journey_df = journey_df.sort_values("markov_score", ascending=False)

    markov_hour = (
        journey_df.groupby(["end_facility", "end_hour_start"], as_index=False)
        .agg(
            markov_score=("markov_score", "max"),
            markov_journey_count=("device_id", "size"),
        )
        .rename(columns={"end_facility": "facility_num", "end_hour_start": "hour_start"})
    )
    if len(markov_hour):
        hour_threshold = float(markov_hour["markov_score"].quantile(0.97))
    else:
        hour_threshold = 0.0
    markov_hour["is_markov_hour_anomaly"] = markov_hour["markov_score"] >= hour_threshold
    markov_hour = markov_hour.sort_values(["facility_num", "hour_start"])
    return journey_df, markov_hour


def task4_advanced_anomalies(
    sessions: pd.DataFrame, hourly_footfall: pd.DataFrame, anomaly_hours: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, float],
]:
    matrix_profile_df, matrix_profile_detail = matrix_profile_hourly_anomalies(hourly_footfall)
    stl_df = stl_hourly_anomalies(hourly_footfall)
    iso_df = isolation_forest_hourly_anomalies(hourly_footfall)
    markov_journey_df, markov_hour_df = markov_journey_anomalies(sessions)

    comparison = (
        hourly_footfall[["facility_num", "hour_start", "estimated_total_footfall"]]
        .merge(
            anomaly_hours[["facility_num", "hour_start", "zscore", "is_anomaly_hour"]],
            on=["facility_num", "hour_start"],
            how="left",
        )
        .merge(
            matrix_profile_df[
                ["facility_num", "hour_start", "matrix_profile_score", "is_matrix_profile_anomaly"]
            ],
            on=["facility_num", "hour_start"],
            how="left",
        )
        .merge(
            stl_df[["facility_num", "hour_start", "stl_score", "is_stl_anomaly"]],
            on=["facility_num", "hour_start"],
            how="left",
        )
        .merge(
            iso_df[
                [
                    "facility_num",
                    "hour_start",
                    "isolation_forest_score",
                    "is_isolation_forest_anomaly",
                ]
            ],
            on=["facility_num", "hour_start"],
            how="left",
        )
        .merge(
            markov_hour_df[
                [
                    "facility_num",
                    "hour_start",
                    "markov_score",
                    "markov_journey_count",
                    "is_markov_hour_anomaly",
                ]
            ],
            on=["facility_num", "hour_start"],
            how="left",
        )
    )
    comparison["zscore_score"] = comparison["zscore"].abs()
    comparison["zscore_norm"] = _minmax(comparison["zscore_score"].fillna(0))
    comparison["matrix_profile_norm"] = _minmax(comparison["matrix_profile_score"].fillna(0))
    comparison["stl_norm"] = _minmax(comparison["stl_score"].fillna(0))
    comparison["isolation_forest_norm"] = _minmax(comparison["isolation_forest_score"].fillna(0))
    comparison["markov_norm"] = _minmax(comparison["markov_score"].fillna(0))

    bool_cols = [
        "is_anomaly_hour",
        "is_matrix_profile_anomaly",
        "is_stl_anomaly",
        "is_isolation_forest_anomaly",
        "is_markov_hour_anomaly",
    ]
    for col in bool_cols:
        comparison[col] = comparison[col].fillna(False).astype(bool)

    comparison["ensemble_score"] = (
        comparison[
            [
                "zscore_norm",
                "matrix_profile_norm",
                "stl_norm",
                "isolation_forest_norm",
                "markov_norm",
            ]
        ]
        .mean(axis=1)
        .round(6)
    )
    comparison["method_agreement_count"] = comparison[bool_cols].sum(axis=1)
    comparison["is_consensus_anomaly"] = comparison["method_agreement_count"] >= 2
    comparison = comparison.sort_values("ensemble_score", ascending=False)

    summary = {
        "matrix_profile_anomaly_hours": int(matrix_profile_df["is_matrix_profile_anomaly"].sum()),
        "stl_anomaly_hours": int(stl_df["is_stl_anomaly"].sum()),
        "isolation_forest_anomaly_hours": int(iso_df["is_isolation_forest_anomaly"].sum()),
        "markov_journey_anomalies": int(markov_journey_df["is_markov_journey_anomaly"].sum()),
        "markov_hour_anomalies": int(markov_hour_df["is_markov_hour_anomaly"].sum())
        if len(markov_hour_df)
        else 0,
        "consensus_anomaly_hours": int(comparison["is_consensus_anomaly"].sum()),
        "max_method_agreement_count": int(comparison["method_agreement_count"].max()),
    }
    return (
        matrix_profile_df,
        matrix_profile_detail,
        stl_df,
        iso_df,
        markov_journey_df,
        markov_hour_df,
        comparison,
        summary,
    )


def main() -> None:
    sessions, facilities, manuals = load_data()

    calibration_df, cal_result, hourly_footfall = task1_calibration(sessions, manuals)
    pair_counts, intersection_summary = task2_sensor_intersection(sessions, facilities)
    top_paths, transitions, sample_paths, total_journey_devices = task3_journeys(sessions, facilities)
    suspicious_devices, anomaly_hours, anomaly_summary = task4_anomalies(sessions, hourly_footfall)
    (
        matrix_profile_df,
        matrix_profile_detail,
        stl_df,
        iso_df,
        markov_journey_df,
        markov_hour_df,
        anomaly_comparison_df,
        advanced_anomaly_summary,
    ) = task4_advanced_anomalies(sessions, hourly_footfall, anomaly_hours)

    calibration_df.to_csv(OUTPUT_DIR / "task1_calibration_windows.csv", index=False)
    hourly_footfall.to_csv(OUTPUT_DIR / "task1_hourly_estimated_footfall.csv", index=False)
    pair_counts.to_csv(OUTPUT_DIR / "task2_sensor_overlap_pairs.csv", index=False)
    top_paths.head(100).to_csv(OUTPUT_DIR / "task3_top_journeys.csv", index=False)
    transitions.to_csv(OUTPUT_DIR / "task3_transition_matrix.csv", index=False)
    sample_paths.to_csv(OUTPUT_DIR / "task3_sample_device_journeys.csv", index=False)
    suspicious_devices.head(200).to_csv(OUTPUT_DIR / "task4_suspicious_unflagged_devices.csv", index=False)
    anomaly_hours.to_csv(OUTPUT_DIR / "task4_hourly_anomalies.csv", index=False)
    matrix_profile_df.to_csv(OUTPUT_DIR / "task4_matrix_profile_anomalies.csv", index=False)
    matrix_profile_detail.to_csv(OUTPUT_DIR / "task4_matrix_profile_details.csv", index=False)
    stl_df.to_csv(OUTPUT_DIR / "task4_stl_anomalies.csv", index=False)
    iso_df.to_csv(OUTPUT_DIR / "task4_isolation_forest_anomalies.csv", index=False)
    markov_journey_df.to_csv(OUTPUT_DIR / "task4_markov_journey_anomalies.csv", index=False)
    markov_hour_df.to_csv(OUTPUT_DIR / "task4_markov_hourly_anomalies.csv", index=False)
    anomaly_comparison_df.to_csv(OUTPUT_DIR / "task4_anomaly_comparison.csv", index=False)

    daily_estimates = (
        hourly_footfall.assign(date=lambda df: df["hour_start"].dt.date)
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
            "mean_manual_total_count": float(calibration_df["manual_total_count"].mean()),
            "mean_estimated_total_count": float(calibration_df["estimated_total_count"].mean()),
        },
        "task2": intersection_summary,
        "task3": {
            "devices_with_multifacility_paths": total_journey_devices,
            "sampled_journeys_exported": int(len(sample_paths)),
            "top_path": top_paths.iloc[0]["path"] if len(top_paths) else None,
            "top_path_devices": int(top_paths.iloc[0]["device_count"]) if len(top_paths) else 0,
            "top_transition_from": int(transitions.iloc[0]["facility_num"]) if len(transitions) else None,
            "top_transition_to": int(transitions.iloc[0]["next_facility"]) if len(transitions) else None,
            "top_transition_count": int(transitions.iloc[0]["transition_count"]) if len(transitions) else 0,
        },
        "task4": anomaly_summary,
        "task4_advanced": {
            **advanced_anomaly_summary,
            "top_consensus_event": anomaly_comparison_df.iloc[0]["hour_start"]
            if len(anomaly_comparison_df)
            else None,
            "top_consensus_facility": int(anomaly_comparison_df.iloc[0]["facility_num"])
            if len(anomaly_comparison_df)
            else None,
            "top_consensus_ensemble_score": float(anomaly_comparison_df.iloc[0]["ensemble_score"])
            if len(anomaly_comparison_df)
            else 0.0,
        },
        "daily_estimated_footfall": daily_estimates.to_dict(orient="records"),
    }
    with (OUTPUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
