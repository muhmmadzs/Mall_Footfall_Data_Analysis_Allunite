from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
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
    model_name: str = "linear_trusted_unique_devices"
    target_name: str = "manual_total_count"
    prediction_col: str = "footfall_recommended"
    selected_by: str = "lowest leave-one-out MAPE among single-feature linear models"
    n_calibration_windows: int = 0
    loo_mae: float = 0.0
    loo_mape: float = 0.0
    loo_max_ape: float = 0.0
    bootstrap_intercept_p05: float = np.nan
    bootstrap_intercept_p50: float = np.nan
    bootstrap_intercept_p95: float = np.nan
    bootstrap_slope_p05: float = np.nan
    bootstrap_slope_p50: float = np.nan
    bootstrap_slope_p95: float = np.nan
    model_comparison: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    leave_one_out: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    bootstrap: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


CANDIDATE_FEATURES = [
    "raw_unique_devices",
    "clean_unique_devices",
    "raw_sessions",
    "clean_sessions",
    "trusted_unique_devices",
    "local_unique_devices",
]

DIAGNOSTIC_FEATURES = [
    *CANDIDATE_FEATURES,
    "mean_signal_level",
    "median_session_duration",
    "zero_duration_ratio",
    "trusted_share_of_clean_unique",
    "local_share_of_clean_unique",
]

HOURLY_PREDICTION_MODELS = [
    "linear_trusted_unique_devices",
    "linear_clean_unique_devices",
    "linear_local_unique_devices",
    "origin_clean_unique_devices",
    "capture_median_clean_unique_devices",
]

TREND_ELASTICITY_BETA = 0.10
TREND_ELASTICITY_BETAS = (0.05, 0.10, 0.25)
CAPTURE_RATE_SHRINK_WEIGHT = 0.65
TREND_RECOMMENDED_MODEL = "trend_clean_elasticity_beta_010"


def _window_features(raw: pd.DataFrame, clean: pd.DataFrame) -> Dict[str, float]:
    clean_unique = float(clean["device_id"].nunique())
    trusted_unique = float(clean.loc[clean["is_trusted"], "device_id"].nunique())
    local_unique = float(clean.loc[clean["is_local"], "device_id"].nunique())
    return {
        "raw_sessions": float(len(raw)),
        "raw_unique_devices": float(raw["device_id"].nunique()),
        "clean_sessions": float(len(clean)),
        "clean_unique_devices": clean_unique,
        "trusted_unique_devices": trusted_unique,
        "local_unique_devices": local_unique,
        "mean_signal_level": float(clean["signal_level"].mean()) if len(clean) else 0.0,
        "median_session_duration": float(clean["session_duration"].median())
        if len(clean)
        else 0.0,
        "zero_duration_ratio": float((clean["session_duration"] == 0).mean())
        if len(clean)
        else 0.0,
        "trusted_share_of_clean_unique": trusted_unique / clean_unique if clean_unique else 0.0,
        "local_share_of_clean_unique": local_unique / clean_unique if clean_unique else 0.0,
    }


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.nanstd(y_true) == 0:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _model_specs(calibration_df: pd.DataFrame) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for feature in DIAGNOSTIC_FEATURES:
        if feature not in calibration_df.columns or calibration_df[feature].std() == 0:
            continue
        specs.append(
            {
                "model_name": f"linear_{feature}",
                "model_type": "linear",
                "features": (feature,),
                "fit_intercept": True,
                "eligible_for_recommendation": feature in CANDIDATE_FEATURES,
            }
        )
        specs.append(
            {
                "model_name": f"origin_{feature}",
                "model_type": "linear",
                "features": (feature,),
                "fit_intercept": False,
                "eligible_for_recommendation": False,
            }
        )

    for feature in [
        "trusted_unique_devices",
        "clean_unique_devices",
        "local_unique_devices",
        "raw_unique_devices",
        "clean_sessions",
    ]:
        if feature not in calibration_df.columns or calibration_df[feature].std() == 0:
            continue
        for method in ("mean", "median"):
            specs.append(
                {
                    "model_name": f"capture_{method}_{feature}",
                    "model_type": f"capture_{method}",
                    "features": (feature,),
                    "fit_intercept": False,
                    "eligible_for_recommendation": False,
                }
            )

    blend_features = ("trusted_unique_devices", "clean_unique_devices")
    if all(calibration_df[f].std() > 0 for f in blend_features):
        specs.append(
            {
                "model_name": "blend_trusted_clean",
                "model_type": "linear",
                "features": blend_features,
                "fit_intercept": True,
                "eligible_for_recommendation": False,
            }
        )
    return specs


def _fit_and_predict(
    train: pd.DataFrame,
    predict_df: pd.DataFrame,
    spec: Dict[str, object],
    target: str = "manual_total_count",
) -> tuple[np.ndarray, float, Dict[str, float]]:
    features = list(spec["features"])
    model_type = str(spec["model_type"])

    if model_type == "linear":
        model = LinearRegression(fit_intercept=bool(spec["fit_intercept"]))
        model.fit(train[features].to_numpy(), train[target].to_numpy())
        pred = model.predict(predict_df[features].to_numpy())
        intercept = float(model.intercept_) if bool(spec["fit_intercept"]) else 0.0
        coefs = {f"coef_{feature}": float(coef) for feature, coef in zip(features, model.coef_)}
        return np.clip(pred.astype(float), 0, None), intercept, coefs

    feature = features[0]
    rates = train[target] / train[feature].clip(lower=1)
    if model_type == "capture_median":
        rate = float(rates.median())
    elif model_type == "capture_mean":
        rate = float(rates.mean())
    else:
        raise ValueError(f"Unsupported task 1 model type: {model_type}")
    pred = predict_df[feature].to_numpy(dtype=float) * rate
    return np.clip(pred, 0, None), 0.0, {f"coef_{feature}": rate}


def _beta_suffix(beta: float) -> str:
    return f"{int(round(beta * 100)):03d}"


def _trend_model_name(beta: float) -> str:
    return f"trend_clean_elasticity_beta_{_beta_suffix(beta)}"


def _trend_prediction_column(beta: float) -> str:
    return f"footfall_clean_elasticity_{_beta_suffix(beta)}"


def _nearest_calibrated_facility_map(
    facilities: pd.DataFrame | None,
    all_facilities: List[int],
    calibrated_facilities: List[int],
) -> Dict[int, int]:
    if facilities is None or facilities.empty:
        return {}
    required = {"facility_num", "latitude", "longitude"}
    if not required.issubset(facilities.columns):
        return {}

    coords = (
        facilities.loc[:, ["facility_num", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates("facility_num")
        .copy()
    )
    if coords.empty:
        return {}
    coords["facility_num"] = coords["facility_num"].astype(int)
    calibrated = coords.loc[coords["facility_num"].isin(calibrated_facilities)]
    if calibrated.empty:
        return {}

    nearest: Dict[int, int] = {}
    for facility in all_facilities:
        if facility in calibrated_facilities:
            continue
        row = coords.loc[coords["facility_num"] == facility]
        if row.empty:
            continue
        lat = float(row["latitude"].iloc[0])
        lon = float(row["longitude"].iloc[0])
        distances = (
            (calibrated["latitude"].astype(float) - lat) ** 2
            + (calibrated["longitude"].astype(float) - lon) ** 2
        )
        nearest[facility] = int(calibrated.loc[distances.idxmin(), "facility_num"])
    return nearest


def _trend_anchor_table(
    calibration_df: pd.DataFrame,
    facilities: pd.DataFrame | None = None,
    all_facilities: List[int] | None = None,
    target: str = "manual_total_count",
    capture_shrink_weight: float = CAPTURE_RATE_SHRINK_WEIGHT,
) -> pd.DataFrame:
    facility_stats = (
        calibration_df.groupby("facility_num", as_index=True)
        .agg(
            trend_anchor_manual_count=(target, "median"),
            trend_anchor_clean_unique_devices=("clean_unique_devices", "median"),
            trend_manual_sum=(target, "sum"),
            trend_clean_sum=("clean_unique_devices", "sum"),
            trend_calibration_windows=(target, "size"),
        )
        .copy()
    )

    global_manual = float(calibration_df[target].median())
    global_clean = max(float(calibration_df["clean_unique_devices"].median()), 1.0)
    global_rate = float(
        calibration_df[target].sum()
        / max(float(calibration_df["clean_unique_devices"].sum()), 1.0)
    )
    facility_stats["trend_source_capture_rate"] = (
        facility_stats["trend_manual_sum"]
        / facility_stats["trend_clean_sum"].clip(lower=1)
    )

    calibrated_facilities = [int(f) for f in facility_stats.index.tolist()]
    output_facilities = sorted(
        set(int(f) for f in (all_facilities or calibrated_facilities))
        | set(calibrated_facilities)
    )
    nearest = _nearest_calibrated_facility_map(
        facilities, output_facilities, calibrated_facilities
    )

    rows: List[Dict[str, object]] = []
    for facility in output_facilities:
        if facility in calibrated_facilities:
            source_facility = facility
            source_type = "manual_calibrated"
        elif facility in nearest:
            source_facility = nearest[facility]
            source_type = "nearest_calibrated_facility"
        else:
            source_facility = np.nan
            source_type = "global_calibration"

        if pd.notna(source_facility) and int(source_facility) in facility_stats.index:
            stats = facility_stats.loc[int(source_facility)]
            anchor_manual = float(stats["trend_anchor_manual_count"])
            anchor_clean = max(float(stats["trend_anchor_clean_unique_devices"]), 1.0)
            source_capture_rate = float(stats["trend_source_capture_rate"])
            calibration_windows = int(stats["trend_calibration_windows"])
        else:
            anchor_manual = global_manual
            anchor_clean = global_clean
            source_capture_rate = global_rate
            calibration_windows = 0

        if not np.isfinite(source_capture_rate) or source_capture_rate <= 0:
            source_capture_rate = global_rate
        shrunk_rate = (
            capture_shrink_weight * source_capture_rate
            + (1 - capture_shrink_weight) * global_rate
        )

        rows.append(
            {
                "facility_num": int(facility),
                "trend_source_facility": source_facility,
                "trend_source_type": source_type,
                "trend_anchor_manual_count": anchor_manual,
                "trend_anchor_clean_unique_devices": anchor_clean,
                "trend_source_capture_rate": source_capture_rate,
                "trend_global_capture_rate": global_rate,
                "trend_capture_rate_shrunk": shrunk_rate,
                "trend_capture_rate_shrink_weight": capture_shrink_weight,
                "trend_source_calibration_windows": calibration_windows,
            }
        )
    return pd.DataFrame(rows)


def _add_trend_preserving_predictions(
    frame: pd.DataFrame,
    calibration_df: pd.DataFrame,
    facilities: pd.DataFrame | None = None,
    betas: tuple[float, ...] = TREND_ELASTICITY_BETAS,
    target: str = "manual_total_count",
) -> pd.DataFrame:
    anchors = _trend_anchor_table(
        calibration_df,
        facilities=facilities,
        all_facilities=[int(f) for f in frame["facility_num"].dropna().unique()],
        target=target,
    )
    out = frame.merge(anchors, on="facility_num", how="left")

    fallback_manual = float(calibration_df[target].median())
    fallback_clean = max(float(calibration_df["clean_unique_devices"].median()), 1.0)
    fallback_rate = float(
        calibration_df[target].sum()
        / max(float(calibration_df["clean_unique_devices"].sum()), 1.0)
    )
    out["trend_anchor_manual_count"] = out["trend_anchor_manual_count"].fillna(
        fallback_manual
    )
    out["trend_anchor_clean_unique_devices"] = out[
        "trend_anchor_clean_unique_devices"
    ].fillna(fallback_clean)
    out["trend_source_capture_rate"] = out["trend_source_capture_rate"].fillna(
        fallback_rate
    )
    out["trend_global_capture_rate"] = out["trend_global_capture_rate"].fillna(
        fallback_rate
    )
    out["trend_capture_rate_shrunk"] = out["trend_capture_rate_shrunk"].fillna(
        fallback_rate
    )
    out["trend_source_type"] = out["trend_source_type"].fillna("global_calibration")
    out["trend_source_calibration_windows"] = out[
        "trend_source_calibration_windows"
    ].fillna(0)

    clean = out["clean_unique_devices"].to_numpy(dtype=float)
    anchor_clean = out["trend_anchor_clean_unique_devices"].clip(lower=1).to_numpy(
        dtype=float
    )
    anchor_manual = out["trend_anchor_manual_count"].to_numpy(dtype=float)
    ratio = np.divide(clean, anchor_clean, out=np.zeros_like(clean), where=anchor_clean > 0)

    out["footfall_clean_capture_global"] = (
        clean * out["trend_global_capture_rate"].to_numpy(dtype=float)
    ).round(2)
    out["footfall_clean_capture_source"] = (
        clean * out["trend_source_capture_rate"].to_numpy(dtype=float)
    ).round(2)
    out["footfall_clean_capture_shrunk"] = (
        clean * out["trend_capture_rate_shrunk"].to_numpy(dtype=float)
    ).round(2)

    for beta in betas:
        pred = anchor_manual * np.power(ratio, beta)
        pred = np.where(clean > 0, pred, 0.0)
        out[_trend_prediction_column(beta)] = np.clip(pred, 0, None).round(2)
    return out


def _predict_trend_preserving(
    train: pd.DataFrame,
    predict_df: pd.DataFrame,
    facilities: pd.DataFrame | None = None,
    beta: float = TREND_ELASTICITY_BETA,
    target: str = "manual_total_count",
) -> np.ndarray:
    scored = _add_trend_preserving_predictions(
        predict_df,
        train,
        facilities=facilities,
        betas=(beta,),
        target=target,
    )
    return scored[_trend_prediction_column(beta)].to_numpy(dtype=float)


def _evaluate_trend_models(
    calibration_df: pd.DataFrame,
    facilities: pd.DataFrame | None = None,
    target: str = "manual_total_count",
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    y = calibration_df[target].to_numpy(dtype=float)
    comparison_rows: List[Dict[str, object]] = []
    loo_rows: List[Dict[str, object]] = []

    for beta in TREND_ELASTICITY_BETAS:
        pred = _predict_trend_preserving(
            calibration_df, calibration_df, facilities=facilities, beta=beta, target=target
        )
        abs_error = np.abs(pred - y)
        model_name = _trend_model_name(beta)

        for idx in calibration_df.index:
            train = calibration_df.drop(index=idx)
            test = calibration_df.loc[[idx]]
            loo_pred = _predict_trend_preserving(
                train, test, facilities=facilities, beta=beta, target=target
            )
            manual = float(test[target].iloc[0])
            loo_abs_error = abs(float(loo_pred[0]) - manual)
            loo_rows.append(
                {
                    "model_name": model_name,
                    "facility_num": int(test["facility_num"].iloc[0]),
                    "started": test["started"].iloc[0],
                    "manual_total_count": manual,
                    "predicted": float(loo_pred[0]),
                    "abs_error": loo_abs_error,
                    "ape": loo_abs_error / max(manual, 1.0),
                }
            )

        loo_model = pd.DataFrame([r for r in loo_rows if r["model_name"] == model_name])
        comparison_rows.append(
            {
                "model_name": model_name,
                "model_type": "trend_elasticity",
                "features": "clean_unique_devices",
                "fit_intercept": False,
                "eligible_for_recommendation": True,
                "intercept": 0.0,
                "slope": beta,
                "coef_clean_unique_devices": beta,
                "mae": float(mean_absolute_error(y, pred)),
                "mape": float(mean_absolute_percentage_error(y, pred)),
                "r2": _safe_r2(y, pred),
                "loo_mae": float(loo_model["abs_error"].mean()),
                "loo_mape": float(loo_model["ape"].mean()),
                "loo_max_ape": float(loo_model["ape"].max()),
            }
        )
    return comparison_rows, loo_rows


def _evaluate_models(
    calibration_df: pd.DataFrame,
    facilities: pd.DataFrame | None = None,
    target: str = "manual_total_count",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = calibration_df[target].to_numpy(dtype=float)
    comparison_rows: List[Dict[str, object]] = []
    loo_rows: List[Dict[str, object]] = []

    for spec in _model_specs(calibration_df):
        pred, intercept, coefs = _fit_and_predict(calibration_df, calibration_df, spec, target)
        abs_error = np.abs(pred - y)
        ape = abs_error / np.clip(y, 1, None)

        model_name = str(spec["model_name"])
        features = tuple(spec["features"])
        for idx in calibration_df.index:
            train = calibration_df.drop(index=idx)
            test = calibration_df.loc[[idx]]
            loo_pred, _, _ = _fit_and_predict(train, test, spec, target)
            manual = float(test[target].iloc[0])
            loo_abs_error = abs(float(loo_pred[0]) - manual)
            loo_rows.append(
                {
                    "model_name": model_name,
                    "facility_num": int(test["facility_num"].iloc[0]),
                    "started": test["started"].iloc[0],
                    "manual_total_count": manual,
                    "predicted": float(loo_pred[0]),
                    "abs_error": loo_abs_error,
                    "ape": loo_abs_error / max(manual, 1.0),
                }
            )

        loo_model = pd.DataFrame([r for r in loo_rows if r["model_name"] == model_name])
        row: Dict[str, object] = {
            "model_name": model_name,
            "model_type": spec["model_type"],
            "features": "+".join(features),
            "fit_intercept": bool(spec["fit_intercept"]),
            "eligible_for_recommendation": bool(spec["eligible_for_recommendation"]),
            "intercept": intercept,
            "slope": float(coefs.get(f"coef_{features[0]}", 0.0)),
            "mae": float(mean_absolute_error(y, pred)),
            "mape": float(mean_absolute_percentage_error(y, pred)),
            "r2": _safe_r2(y, pred),
            "loo_mae": float(loo_model["abs_error"].mean()),
            "loo_mape": float(loo_model["ape"].mean()),
            "loo_max_ape": float(loo_model["ape"].max()),
        }
        row.update(coefs)
        comparison_rows.append(row)

    trend_comparison, trend_loo = _evaluate_trend_models(
        calibration_df, facilities=facilities, target=target
    )
    comparison_rows.extend(trend_comparison)
    loo_rows.extend(trend_loo)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["eligible_for_recommendation", "loo_mape", "mape"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    rank_map = {name: rank for rank, name in enumerate(comparison["model_name"])}
    loo = pd.DataFrame(loo_rows)
    loo["model_rank"] = loo["model_name"].map(rank_map)
    loo = loo.sort_values(["model_rank", "started"]).drop(columns=["model_rank"])
    return comparison, loo


def _bootstrap_selected_model(
    calibration_df: pd.DataFrame,
    selected_spec: Dict[str, object],
    target: str = "manual_total_count",
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    y = calibration_df[target].to_numpy(dtype=float)
    indices = np.arange(len(calibration_df))
    features = tuple(selected_spec["features"])

    for iteration in range(n):
        sample_positions = rng.choice(indices, size=len(indices), replace=True)
        train = calibration_df.iloc[sample_positions]
        pred, intercept, coefs = _fit_and_predict(train, calibration_df, selected_spec, target)
        rows.append(
            {
                "iteration": iteration,
                "intercept": intercept,
                "slope": float(coefs.get(f"coef_{features[0]}", 0.0)),
                "mae": float(mean_absolute_error(y, pred)),
                "mape": float(mean_absolute_percentage_error(y, pred)),
                "r2": _safe_r2(y, pred),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_trend_model(
    calibration_df: pd.DataFrame,
    facilities: pd.DataFrame | None = None,
    beta: float = TREND_ELASTICITY_BETA,
    target: str = "manual_total_count",
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    y = calibration_df[target].to_numpy(dtype=float)
    indices = np.arange(len(calibration_df))

    for iteration in range(n):
        sample_positions = rng.choice(indices, size=len(indices), replace=True)
        train = calibration_df.iloc[sample_positions]
        pred = _predict_trend_preserving(
            train, calibration_df, facilities=facilities, beta=beta, target=target
        )
        rows.append(
            {
                "iteration": iteration,
                "intercept": 0.0,
                "slope": beta,
                "beta": beta,
                "mae": float(mean_absolute_error(y, pred)),
                "mape": float(mean_absolute_percentage_error(y, pred)),
                "r2": _safe_r2(y, pred),
                "n_source_facilities": float(train["facility_num"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _select_recommended_model(comparison: pd.DataFrame) -> str:
    trend = comparison.loc[comparison["model_name"] == TREND_RECOMMENDED_MODEL]
    if not trend.empty:
        return TREND_RECOMMENDED_MODEL
    eligible = comparison.loc[comparison["eligible_for_recommendation"]].copy()
    if eligible.empty:
        eligible = comparison.copy()
    return str(eligible.sort_values(["loo_mape", "mape"]).iloc[0]["model_name"])


def _prediction_column(model_name: str) -> str:
    mapping = {
        "linear_trusted_unique_devices": "footfall_trusted_linear",
        "linear_clean_unique_devices": "footfall_clean_linear",
        "linear_local_unique_devices": "footfall_local_linear",
        "origin_clean_unique_devices": "footfall_clean_origin",
        "capture_median_clean_unique_devices": "footfall_clean_capture_median",
    }
    if model_name.startswith("trend_clean_elasticity_beta_"):
        return "footfall_clean_elasticity_" + model_name.rsplit("_", 1)[-1]
    return mapping.get(model_name, f"footfall_{model_name}")


def run_task1(
    sessions: pd.DataFrame,
    manuals: pd.DataFrame,
    facilities: pd.DataFrame | None = None,
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
    model_comparison, leave_one_out = _evaluate_models(
        calibration_df, facilities=facilities
    )
    recommended_name = _select_recommended_model(model_comparison)
    model_comparison["selected_for_delivery"] = model_comparison["model_name"].eq(
        recommended_name
    )
    model_comparison["selection_note"] = np.where(
        model_comparison["selected_for_delivery"],
        "Plan C delivered estimate: preserves clean-device hourly/daily trend while anchored to manual counts",
        "Diagnostic comparison model",
    )
    selected_row = model_comparison.loc[
        model_comparison["model_name"] == recommended_name
    ].iloc[0]
    if recommended_name.startswith("trend_clean_elasticity_beta_"):
        recommended_pred = _predict_trend_preserving(
            calibration_df,
            calibration_df,
            facilities=facilities,
            beta=TREND_ELASTICITY_BETA,
        )
        selected_spec = None
    else:
        selected_spec = next(
            spec
            for spec in _model_specs(calibration_df)
            if spec["model_name"] == recommended_name
        )
        recommended_pred, _, _ = _fit_and_predict(
            calibration_df, calibration_df, selected_spec
        )
    calibration_df["estimated_total_count"] = recommended_pred.round(2)
    calibration_df["recommended_model_name"] = recommended_name
    calibration_df["recommended_abs_error"] = (
        calibration_df["estimated_total_count"] - calibration_df["manual_total_count"]
    ).abs()
    calibration_df["recommended_ape"] = (
        calibration_df["recommended_abs_error"]
        / calibration_df["manual_total_count"].clip(lower=1)
    )

    for spec in _model_specs(calibration_df):
        name = str(spec["model_name"])
        if name != recommended_name and name not in HOURLY_PREDICTION_MODELS:
            continue
        pred, _, _ = _fit_and_predict(calibration_df, calibration_df, spec)
        calibration_df[f"pred_{name}"] = pred.round(2)

    trend_calibration = _add_trend_preserving_predictions(
        calibration_df,
        calibration_df,
        facilities=facilities,
        betas=TREND_ELASTICITY_BETAS,
    )
    for beta in TREND_ELASTICITY_BETAS:
        column = _trend_prediction_column(beta)
        calibration_df[f"pred_{_trend_model_name(beta)}"] = trend_calibration[column]
    calibration_df["pred_footfall_trend_preserving"] = calibration_df[
        f"pred_{TREND_RECOMMENDED_MODEL}"
    ]
    calibration_df["pred_plan_c_trend_preserving"] = calibration_df[
        "pred_footfall_trend_preserving"
    ]
    calibration_df["plan_c_abs_error"] = (
        calibration_df["pred_plan_c_trend_preserving"]
        - calibration_df["manual_total_count"]
    ).abs()
    calibration_df["plan_c_ape"] = (
        calibration_df["plan_c_abs_error"]
        / calibration_df["manual_total_count"].clip(lower=1)
    )

    if selected_spec is None:
        bootstrap = _bootstrap_trend_model(
            calibration_df,
            facilities=facilities,
            beta=TREND_ELASTICITY_BETA,
        )
    else:
        bootstrap = _bootstrap_selected_model(calibration_df, selected_spec)
    bootstrap_intercepts = bootstrap["intercept"] if len(bootstrap) else pd.Series(dtype=float)
    bootstrap_slopes = bootstrap["slope"] if len(bootstrap) else pd.Series(dtype=float)

    selected_feature = str(selected_row["features"]).split("+")[0]
    result = CalibrationResult(
        feature_name=selected_feature,
        intercept=float(selected_row["intercept"]),
        slope=float(selected_row["slope"]),
        mae=float(selected_row["mae"]),
        mape=float(selected_row["mape"]),
        r2=float(selected_row["r2"]),
        model_name=recommended_name,
        prediction_col=_prediction_column(recommended_name),
        selected_by=(
            "Plan C trend-preserving clean-device elasticity model selected to avoid "
            "flattening hourly/daily shape from four manual windows"
        ),
        n_calibration_windows=int(len(calibration_df)),
        loo_mae=float(selected_row["loo_mae"]),
        loo_mape=float(selected_row["loo_mape"]),
        loo_max_ape=float(selected_row["loo_max_ape"]),
        bootstrap_intercept_p05=float(bootstrap_intercepts.quantile(0.05))
        if len(bootstrap_intercepts)
        else np.nan,
        bootstrap_intercept_p50=float(bootstrap_intercepts.quantile(0.50))
        if len(bootstrap_intercepts)
        else np.nan,
        bootstrap_intercept_p95=float(bootstrap_intercepts.quantile(0.95))
        if len(bootstrap_intercepts)
        else np.nan,
        bootstrap_slope_p05=float(bootstrap_slopes.quantile(0.05))
        if len(bootstrap_slopes)
        else np.nan,
        bootstrap_slope_p50=float(bootstrap_slopes.quantile(0.50))
        if len(bootstrap_slopes)
        else np.nan,
        bootstrap_slope_p95=float(bootstrap_slopes.quantile(0.95))
        if len(bootstrap_slopes)
        else np.nan,
        model_comparison=model_comparison,
        leave_one_out=leave_one_out,
        bootstrap=bootstrap,
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
            zero_duration_ratio=("session_duration", lambda s: float((s == 0).mean())),
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
            "zero_duration_ratio": 0,
        }
    )
    hourly["trusted_share_of_clean_unique"] = (
        hourly["trusted_unique_devices"] / hourly["clean_unique_devices"].clip(lower=1)
    )
    hourly["local_share_of_clean_unique"] = (
        hourly["local_unique_devices"] / hourly["clean_unique_devices"].clip(lower=1)
    )

    prediction_names = set(HOURLY_PREDICTION_MODELS + [recommended_name])
    for spec in _model_specs(calibration_df):
        name = str(spec["model_name"])
        if name not in prediction_names:
            continue
        pred, _, _ = _fit_and_predict(calibration_df, hourly, spec)
        hourly[_prediction_column(name)] = pred.round(2)

    hourly = _add_trend_preserving_predictions(
        hourly,
        calibration_df,
        facilities=facilities,
        betas=TREND_ELASTICITY_BETAS,
    )
    hourly["footfall_trend_preserving"] = hourly[_prediction_column(recommended_name)]
    hourly["footfall_plan_c"] = hourly["footfall_trend_preserving"]
    hourly["footfall_plan_c_trend_preserving"] = hourly["footfall_trend_preserving"]

    recommended_col = _prediction_column(recommended_name)
    hourly["footfall_recommended"] = hourly[recommended_col]
    hourly["footfall_plan_a"] = hourly["footfall_trusted_linear"]
    hourly["estimated_total_footfall"] = hourly["footfall_recommended"]
    hourly["task1_recommended_model"] = recommended_name

    return calibration_df, result, hourly
