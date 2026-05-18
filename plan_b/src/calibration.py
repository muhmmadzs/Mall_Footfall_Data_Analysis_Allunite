"""Plan B calibration: multi-model, per-facility, hour-of-day, capture rate, validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from src.cleaning import build_quality_mask
from src.hod_v2 import (
    HodV2Params,
    build_hod_anchors,
    build_hod_curves,
    hod_v2_components,
)
from src.paths import CALIBRATED_FACILITIES

# Legacy profile-ratio HOD (kept for comparison)
HOD_FACTOR_MIN = 0.5
HOD_FACTOR_MAX = 2.0


FEATURES = [
    "trusted_unique_devices",
    "clean_unique_devices",
    "local_unique_devices",
    "clean_sessions",
]


@dataclass
class ModelSpec:
    name: str
    feature: str
    intercept: float
    slope: float
    mae: float
    mape: float
    r2: float
    per_facility: bool = False


def _window_features(raw: pd.DataFrame, clean: pd.DataFrame) -> Dict[str, float]:
    return {
        "raw_sessions": float(len(raw)),
        "clean_sessions": float(len(clean)),
        "clean_unique_devices": float(clean["device_id"].nunique()),
        "trusted_unique_devices": float(
            clean.loc[clean["is_trusted"], "device_id"].nunique()
        ),
        "local_unique_devices": float(
            clean.loc[clean["is_local"], "device_id"].nunique()
        ),
    }


def build_calibration_windows(
    sessions: pd.DataFrame, manuals: pd.DataFrame, pedestrians_only: bool = False
) -> pd.DataFrame:
    if pedestrians_only:
        target = (
            manuals.loc[manuals["category"] == "Pedestrians"]
            .groupby(["facility_num", "started", "duration", "window_end"], as_index=False)["count"]
            .sum()
            .rename(columns={"count": "manual_total_count"})
        )
    else:
        target = (
            manuals.groupby(["facility_num", "started", "duration", "window_end"], as_index=False)[
                "count"
            ]
            .sum()
            .rename(columns={"count": "manual_total_count"})
        )
    rows: List[Dict] = []
    for row in target.itertuples(index=False):
        mask = (
            (sessions["facility_num"] == row.facility_num)
            & (sessions["session_start"] >= row.started)
            & (sessions["session_start"] < row.window_end)
        )
        raw_w = sessions.loc[mask]
        clean_w = raw_w.loc[build_quality_mask(raw_w)]
        feat = _window_features(raw_w, clean_w)
        feat.update(
            {
                "facility_num": int(row.facility_num),
                "started": row.started,
                "window_end": row.window_end,
                "hour_of_day": int(row.started.hour),
                "manual_total_count": float(row.manual_total_count),
            }
        )
        rows.append(feat)
    return pd.DataFrame(rows)


def fit_global_linear(cal_df: pd.DataFrame, feature: str) -> ModelSpec:
    x = cal_df[[feature]].to_numpy()
    y = cal_df["manual_total_count"].to_numpy()
    m = LinearRegression().fit(x, y)
    pred = np.clip(m.predict(x), 0, None)
    return ModelSpec(
        name=f"global_linear_{feature}",
        feature=feature,
        intercept=float(m.intercept_),
        slope=float(m.coef_[0]),
        mae=float(mean_absolute_error(y, pred)),
        mape=float(mean_absolute_percentage_error(y, pred)),
        r2=float(r2_score(y, pred)),
    )


def fit_global_linear_no_intercept(cal_df: pd.DataFrame, feature: str) -> ModelSpec:
    """Regression through origin: footfall = slope × feature (zero when feature is zero)."""
    x = cal_df[[feature]].to_numpy()
    y = cal_df["manual_total_count"].to_numpy()
    m = LinearRegression(fit_intercept=False).fit(x, y)
    pred = np.clip(m.predict(x), 0, None)
    return ModelSpec(
        name=f"origin_{feature}",
        feature=feature,
        intercept=0.0,
        slope=float(m.coef_[0]),
        mae=float(mean_absolute_error(y, pred)),
        mape=float(mean_absolute_percentage_error(y, pred)),
        r2=float(r2_score(y, pred)),
    )


def fit_blend_two_feature(cal_df: pd.DataFrame, f1: str, f2: str) -> ModelSpec:
    x = cal_df[[f1, f2]].to_numpy()
    y = cal_df["manual_total_count"].to_numpy()
    m = LinearRegression().fit(x, y)
    pred = np.clip(m.predict(x), 0, None)
    return ModelSpec(
        name=f"blend_{f1}_{f2}",
        feature=f"{f1}+{f2}",
        intercept=float(m.intercept_),
        slope=float(m.coef_[0]),  # store coef[0] only; use predict_blend for apply
        mae=float(mean_absolute_error(y, pred)),
        mape=float(mean_absolute_percentage_error(y, pred)),
        r2=float(r2_score(y, pred)),
    )


@dataclass
class BlendModel:
    intercept: float
    coef_trusted: float
    coef_clean: float
    mae: float
    mape: float
    r2: float


def fit_blend_trusted_clean(cal_df: pd.DataFrame) -> BlendModel:
    x = cal_df[["trusted_unique_devices", "clean_unique_devices"]].to_numpy()
    y = cal_df["manual_total_count"].to_numpy()
    m = LinearRegression().fit(x, y)
    pred = np.clip(m.predict(x), 0, None)
    return BlendModel(
        intercept=float(m.intercept_),
        coef_trusted=float(m.coef_[0]),
        coef_clean=float(m.coef_[1]),
        mae=float(mean_absolute_error(y, pred)),
        mape=float(mean_absolute_percentage_error(y, pred)),
        r2=float(r2_score(y, pred)),
    )


def capture_rates_per_facility(
    cal_df: pd.DataFrame,
    method: str = "mean",
    global_scale: float = 1.0,
) -> pd.DataFrame:
    """Facility capture rates from manual calibration windows."""
    df = cal_df.copy()
    df["capture_rate"] = df["manual_total_count"] / df["clean_unique_devices"].clip(lower=1)
    if method == "latest":
        df = df.sort_values("started").drop_duplicates("facility_num", keep="last")
        out = df[["facility_num", "capture_rate"]].rename(columns={"capture_rate": "mean_capture_rate"})
    elif method == "median":
        out = (
            df.groupby("facility_num", as_index=False)["capture_rate"]
            .median()
            .rename(columns={"capture_rate": "mean_capture_rate"})
        )
    elif method == "geo_mean":
        out = (
            df.groupby("facility_num")["capture_rate"]
            .apply(lambda s: float(np.exp(np.log(s.clip(lower=1e-9)).mean())))
            .reset_index()
            .rename(columns={"capture_rate": "mean_capture_rate"})
        )
    else:
        out = (
            df.groupby("facility_num", as_index=False)["capture_rate"]
            .mean()
            .rename(columns={"capture_rate": "mean_capture_rate"})
        )
    if global_scale != 1.0:
        out["mean_capture_rate"] = out["mean_capture_rate"] * float(global_scale)
    return out


def leave_one_out_cv(cal_df: pd.DataFrame, feature: str = "clean_unique_devices") -> pd.DataFrame:
    rows = []
    for i in range(len(cal_df)):
        train = cal_df.drop(index=i)
        test = cal_df.iloc[i]
        spec = fit_global_linear(train, feature)
        pred = max(0, spec.intercept + spec.slope * test[feature])
        rows.append(
            {
                "held_out_facility": int(test["facility_num"]),
                "held_out_started": test["started"],
                "manual_total_count": float(test["manual_total_count"]),
                "predicted": pred,
                "abs_error": abs(pred - test["manual_total_count"]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_coefficients(
    cal_df: pd.DataFrame, feature: str = "clean_unique_devices", n: int = 500, seed: int = 42
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    intercepts, slopes = [], []
    y = cal_df["manual_total_count"].to_numpy()
    x = cal_df[[feature]].to_numpy()
    n_rows = len(cal_df)
    for _ in range(n):
        idx = rng.integers(0, n_rows, size=n_rows)
        m = LinearRegression().fit(x[idx], y[idx])
        intercepts.append(m.intercept_)
        slopes.append(m.coef_[0])
    return {
        "intercept_mean": float(np.mean(intercepts)),
        "intercept_p05": float(np.percentile(intercepts, 5)),
        "intercept_p95": float(np.percentile(intercepts, 95)),
        "slope_mean": float(np.mean(slopes)),
        "slope_p05": float(np.percentile(slopes, 5)),
        "slope_p95": float(np.percentile(slopes, 95)),
    }


def hourly_sensor_features(sessions: pd.DataFrame) -> pd.DataFrame:
    quality = build_quality_mask(sessions)
    clean = sessions.loc[quality]
    raw_h = (
        sessions.groupby(["facility_num", "hour_start"], as_index=False)
        .agg(raw_sessions=("device_id", "size"))
    )
    clean_h = (
        clean.groupby(["facility_num", "hour_start"], as_index=False)
        .agg(
            clean_sessions=("device_id", "size"),
            clean_unique_devices=("device_id", "nunique"),
        )
    )
    trusted_h = (
        clean.loc[clean["is_trusted"]]
        .groupby(["facility_num", "hour_start"], as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "trusted_unique_devices"})
    )
    local_h = (
        clean.loc[clean["is_local"]]
        .groupby(["facility_num", "hour_start"], as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "local_unique_devices"})
    )
    hourly = raw_h.merge(clean_h, on=["facility_num", "hour_start"], how="left")
    hourly = hourly.merge(trusted_h, on=["facility_num", "hour_start"], how="left")
    hourly = hourly.merge(local_h, on=["facility_num", "hour_start"], how="left")
    hourly = hourly.fillna(0)
    hourly["hour_of_day"] = hourly["hour_start"].dt.hour
    hourly["date"] = hourly["hour_start"].dt.date
    return hourly


def hour_of_day_profile(clean_sessions: pd.DataFrame) -> pd.DataFrame:
    """Median clean unique devices per facility × hour-of-day (week)."""
    return (
        clean_sessions.groupby(["facility_num", "hour_of_day"], as_index=False)["device_id"]
        .nunique()
        .groupby(["facility_num", "hour_of_day"])["device_id"]
        .median()
        .reset_index(name="profile_median_devices")
    )


def mall_hour_primary_facility_counts(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each device to one primary sensor per clock hour (strongest signal),
    then count unique devices per (facility, hour). Avoids double-counting the same
    phone at multiple sensors in the same hour.
    """
    clean = sessions.loc[build_quality_mask(sessions)].copy()
    clean["hour_start"] = pd.to_datetime(clean["session_start"]).dt.floor("h")
    idx = clean.groupby(["hour_start", "device_id"], sort=False)["signal_level"].idxmax()
    assigned = clean.loc[idx, ["hour_start", "device_id", "facility_num", "is_trusted"]]

    counts = assigned.groupby(["facility_num", "hour_start"], as_index=False).agg(
        clean_unique_devices_dedup=("device_id", "nunique"),
    )
    trusted = (
        assigned.loc[assigned["is_trusted"]]
        .groupby(["facility_num", "hour_start"], as_index=False)
        .agg(trusted_unique_devices_dedup=("device_id", "nunique"))
    )
    counts = counts.merge(trusted, on=["facility_num", "hour_start"], how="left")
    counts["trusted_unique_devices_dedup"] = counts["trusted_unique_devices_dedup"].fillna(0)
    return counts


def assign_facility_neighbors(
    facilities: pd.DataFrame, calibrated: Optional[set[int]] = None
) -> Dict[int, int]:
    calibrated = calibrated or CALIBRATED_FACILITIES
    sensors = facilities.dropna(subset=["latitude", "longitude"]).copy()
    sensors = sensors[sensors["facility_num"].isin(sensors["facility_num"])]

    cal_sensors = sensors.loc[sensors["facility_num"].isin(calibrated)]
    other = sensors.loc[~sensors["facility_num"].isin(calibrated)]

    mapping: Dict[int, int] = {int(f): int(f) for f in calibrated}
    for _, row in other.iterrows():
        dists = np.sqrt(
            (cal_sensors["latitude"] - row["latitude"]) ** 2
            + (cal_sensors["longitude"] - row["longitude"]) ** 2
        )
        nearest = int(cal_sensors.iloc[int(dists.values.argmin())]["facility_num"])
        mapping[int(row["facility_num"])] = nearest
    return mapping


def _profile_ratio_hod(
    fac: int,
    hod: int,
    prof_devices: float,
    cal_df: pd.DataFrame,
    prof: pd.Series,
    neighbor_map: Dict[int, int],
) -> float:
    """Legacy HOD: median calibration-hour profile / this hour's profile."""
    try:
        prof_val = float(prof.loc[(fac, hod)])
    except KeyError:
        prof_val = prof_devices if prof_devices > 0 else 1.0
    prof_val = max(prof_val, 1.0)
    ref_fac = fac if fac in CALIBRATED_FACILITIES else neighbor_map.get(fac, fac)
    ref_hours = cal_df.loc[cal_df["facility_num"] == ref_fac, "hour_of_day"].tolist()
    if ref_hours:
        ref_prof = np.median(
            [prof.loc[(ref_fac, h)] for h in ref_hours if (ref_fac, h) in prof.index]
        )
    else:
        ref_prof = prof_val
    ref_prof = max(float(ref_prof), 1.0)
    return float(np.clip(ref_prof / prof_val, HOD_FACTOR_MIN, HOD_FACTOR_MAX))


def apply_plan_b_estimates(
    hourly: pd.DataFrame,
    cal_df: pd.DataFrame,
    capture_rates: pd.DataFrame,
    profile: pd.DataFrame,
    global_trusted: ModelSpec,
    neighbor_map: Dict[int, int],
    hod_params: Optional[HodV2Params] = None,
) -> pd.DataFrame:
    """Plan B sensor-hour footfall (HOD v2) + legacy profile HOD for comparison."""
    out = hourly.copy()
    params = hod_params or HodV2Params()
    cap_rates = capture_rates.copy()
    if params.capture_global_scale != 1.0:
        cap_rates = cap_rates.copy()
        cap_rates["mean_capture_rate"] = (
            cap_rates["mean_capture_rate"] * params.capture_global_scale
        )
    cap_map = dict(zip(cap_rates["facility_num"], cap_rates["mean_capture_rate"]))
    prof = profile.set_index(["facility_num", "hour_of_day"])["profile_median_devices"]

    out["footfall_plan_a"] = (
        global_trusted.intercept + global_trusted.slope * out["trusted_unique_devices"]
    ).clip(lower=0).round(2)

    out["capture_rate_facility"] = out["facility_num"].map(cap_map)
    for fac, neighbor in neighbor_map.items():
        if fac not in cap_map and neighbor in cap_map:
            out.loc[out["facility_num"] == fac, "capture_rate_facility"] = cap_map[neighbor]

    has_dedup = "clean_unique_devices_dedup" in out.columns
    if not has_dedup:
        out["clean_unique_devices_dedup"] = out["clean_unique_devices"]
        out["trusted_unique_devices_dedup"] = out.get(
            "trusted_unique_devices", pd.Series(0, index=out.index)
        )

    anchors = build_hod_anchors(cal_df, cap_map)
    facilities = sorted(out["facility_num"].astype(int).unique().tolist())
    hod_curves = build_hod_curves(anchors, neighbor_map, facilities, params)

    hod_factors: List[float] = []
    hod_profile: List[float] = []
    density_factors: List[float] = []
    footfall_b_dedup: List[float] = []
    footfall_b_sensor: List[float] = []
    footfall_profile: List[float] = []

    for row in out.itertuples(index=False):
        fac = int(row.facility_num)
        hod = int(row.hour_of_day)
        devices_sensor = float(row.clean_unique_devices)
        devices_dedup = float(row.clean_unique_devices_dedup)
        rate = float(cap_map.get(fac) or cap_map.get(neighbor_map.get(fac, fac), 0.1))
        prof_devices = devices_dedup if devices_dedup > 0 else devices_sensor

        hod_prof = _profile_ratio_hod(fac, hod, prof_devices, cal_df, prof, neighbor_map)
        hod_v2, dens, _ = hod_v2_components(
            fac, hod, prof_devices, cal_df, cap_map, prof, neighbor_map, hod_curves, params
        )

        hod_factors.append(hod_v2)
        hod_profile.append(hod_prof)
        density_factors.append(dens)
        footfall_b_dedup.append(devices_dedup * rate * hod_v2 * dens)
        footfall_b_sensor.append(devices_sensor * rate * hod_v2 * dens)
        footfall_profile.append(devices_dedup * rate * hod_prof)

    out["hod_factor"] = hod_factors
    out["hod_factor_profile"] = hod_profile
    out["density_factor"] = density_factors
    out["footfall_plan_b_per_sensor"] = np.clip(footfall_b_sensor, 0, None).round(2)
    out["footfall_plan_b"] = np.clip(footfall_b_dedup, 0, None).round(2)
    out["footfall_plan_b_profile_hod"] = np.clip(footfall_profile, 0, None).round(2)
    return out


def mall_daily_dedup(
    sessions: pd.DataFrame,
    hourly: pd.DataFrame,
    overlap_rate: float,
    capture_rate_mall: float,
) -> pd.DataFrame:
    clean = sessions.loc[build_quality_mask(sessions)]
    daily = (
        clean.groupby("date", as_index=False)
        .agg(
            unique_devices_mall=("device_id", "nunique"),
            clean_sessions=("device_id", "size"),
        )
    )
    sum_fac = (
        clean.groupby(["date", "facility_num"], as_index=False)["device_id"]
        .nunique()
        .groupby("date", as_index=False)["device_id"]
        .sum()
        .rename(columns={"device_id": "sum_facility_uniques"})
    )
    footfall_b = hourly.groupby("date", as_index=False)["footfall_plan_b"].sum()
    footfall_a = hourly.groupby("date", as_index=False)["footfall_plan_a"].sum()
    daily = daily.merge(sum_fac, on="date", how="left")
    daily = daily.merge(footfall_b, on="date", how="left")
    daily = daily.merge(footfall_a, on="date", how="left")
    daily["estimated_mall_visitors"] = (
        daily["unique_devices_mall"] * capture_rate_mall
    ).round(2)
    return daily


def mall_hourly_visitors(sessions: pd.DataFrame, capture_rate_mall: float) -> pd.DataFrame:
    """Mall-wide unique devices per hour × capture rate (people-in-mall proxy)."""
    clean = sessions.loc[build_quality_mask(sessions)]
    hourly = (
        clean.groupby("hour_start", as_index=False)
        .agg(unique_devices_mall=("device_id", "nunique"))
    )
    hourly["estimated_mall_visitors"] = (
        hourly["unique_devices_mall"] * capture_rate_mall
    ).round(2)
    hourly["date"] = hourly["hour_start"].dt.date
    return hourly
