"""Tune Plan B HOD + capture parameters (grid search or Bayesian optimization)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.calibration import (
    apply_plan_b_estimates,
    assign_facility_neighbors,
    build_calibration_windows,
    capture_rates_per_facility,
    fit_global_linear,
    hour_of_day_profile,
    hourly_sensor_features,
    mall_hour_primary_facility_counts,
)
from src.cleaning import build_quality_mask
from src.hod_v2 import (
    HodV2Params,
    calibration_validation_hod_v2,
    loo_mae_hod_v2,
    tune_hod_v2_params,
)

CAPTURE_METHODS = ("mean", "median", "geo_mean", "latest")
TARGET_DAILY_RATIO = 0.66


def _params_from_vector(x: List[float], capture_idx: int) -> Tuple[HodV2Params, str]:
    """Decode optimizer vector into HodV2Params + capture method name."""
    (
        shrink_tau,
        smooth_tau,
        density_beta,
        profile_alpha,
        hod_min,
        hod_max,
        capture_scale,
    ) = x
    hod_min_f = float(hod_min)
    hod_max_f = float(max(hod_max, hod_min_f + 0.05))
    method = CAPTURE_METHODS[int(round(capture_idx)) % len(CAPTURE_METHODS)]
    p = HodV2Params(
        shrink_tau=float(shrink_tau),
        smooth_tau=float(smooth_tau),
        density_beta=float(density_beta),
        profile_alpha=float(profile_alpha),
        hod_min=hod_min_f,
        hod_max=hod_max_f,
        capture_global_scale=float(capture_scale),
    )
    return p, method


def evaluate_plan_b_config(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    manuals: pd.DataFrame,
    hod_params: HodV2Params,
    *,
    capture_method: str = "mean",
    include_loo: bool = True,
    target_daily_ratio: float = TARGET_DAILY_RATIO,
) -> Dict[str, Any]:
    """Score one Plan B configuration on calibration windows and full-week shape."""
    cal = build_calibration_windows(sessions, manuals, pedestrians_only=False)
    profile = hour_of_day_profile(sessions.loc[build_quality_mask(sessions)])
    neighbor_map = assign_facility_neighbors(facilities)
    global_trusted = fit_global_linear(cal, "trusted_unique_devices")
    capture_rates = capture_rates_per_facility(
        cal, method=capture_method, global_scale=hod_params.capture_global_scale
    )
    cap_map = dict(zip(capture_rates["facility_num"], capture_rates["mean_capture_rate"]))
    val = calibration_validation_hod_v2(cal, cap_map, neighbor_map, profile, hod_params)

    loo_mae = float("nan")
    if include_loo and len(cal) > 1:
        loo_mae = loo_mae_hod_v2(
            cal,
            neighbor_map,
            profile,
            hod_params,
            lambda df, method="mean", scale=1.0: capture_rates_per_facility(
                df, method=capture_method, global_scale=hod_params.capture_global_scale
            ),
        )

    hourly = hourly_sensor_features(sessions)
    dedup = mall_hour_primary_facility_counts(sessions)
    hourly = hourly.merge(dedup, on=["facility_num", "hour_start"], how="left")
    hourly["clean_unique_devices_dedup"] = hourly["clean_unique_devices_dedup"].fillna(0)
    hourly = apply_plan_b_estimates(
        hourly, cal, capture_rates, profile, global_trusted, neighbor_map, hod_params=hod_params
    )
    daily_b = hourly.groupby(hourly["hour_start"].dt.date)["footfall_plan_b"].sum()
    daily_a = hourly.groupby(hourly["hour_start"].dt.date)["footfall_plan_a"].sum()
    ratio_ba = float((daily_b / daily_a.clip(lower=1)).mean())

    insample_mae = float(val["abs_error"].mean())
    max_pct = float(val["pct_error"].max())
    ratio_penalty = abs(ratio_ba - target_daily_ratio) * 200.0
    loo_term = 0.25 * loo_mae if include_loo and np.isfinite(loo_mae) else 0.0

    return {
        "insample_mae": insample_mae,
        "max_pct_error": max_pct,
        "loo_mae": loo_mae,
        "daily_plan_b_over_a": ratio_ba,
        "composite_score": 0.40 * insample_mae + 0.25 * max_pct + loo_term + ratio_penalty,
        "calibration_validation": val,
        "hod_params": hod_params.__dict__,
        "capture_method": capture_method,
    }


def tune_plan_b_bayesian(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    manuals: pd.DataFrame,
    *,
    n_calls: int = 60,
    n_initial_points: int = 20,
    random_state: int = 42,
    target_daily_ratio: float = TARGET_DAILY_RATIO,
) -> Tuple[HodV2Params, str, pd.DataFrame, Dict[str, Any]]:
    """
    Bayesian optimization (Gaussian-process) over HOD v2 + capture hyperparameters.

    Uses scikit-optimize ``gp_minimize`` to minimize composite_score.
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Categorical, Real
    except ImportError as exc:
        raise ImportError(
            "Bayesian tuning requires scikit-optimize (import name: skopt).\n"
            "  pip install scikit-optimize\n"
            "Do NOT use: pip install skopt  (that package does not exist).\n"
            "Install into the same Python as your Jupyter kernel "
            "(Kernel → Change Kernel → .venv, or run the pip line in a notebook cell with !pip)."
        ) from exc

    dimensions = [
        Real(1.0, 8.0, name="shrink_tau"),
        Real(1.0, 5.0, name="smooth_tau"),
        Real(0.0, 0.25, name="density_beta"),
        Real(0.0, 0.35, name="profile_alpha"),
        Real(0.3, 0.7, name="hod_min"),
        Real(1.1, 2.5, name="hod_max"),
        Real(0.85, 1.15, name="capture_global_scale"),
        Categorical(list(range(len(CAPTURE_METHODS))), name="capture_method_idx"),
    ]

    history_rows: List[Dict[str, Any]] = []

    def objective(x: List[float]) -> float:
        capture_idx = int(x[7]) if len(x) > 7 else 0
        params, cap_method = _params_from_vector(x[:7], capture_idx)
        ev = evaluate_plan_b_config(
            sessions,
            facilities,
            manuals,
            params,
            capture_method=cap_method,
            target_daily_ratio=target_daily_ratio,
        )
        history_rows.append(
            {
                **params.__dict__,
                "capture_method": cap_method,
                "insample_mae": ev["insample_mae"],
                "max_pct_error": ev["max_pct_error"],
                "loo_mae": ev["loo_mae"],
                "daily_plan_b_over_a": ev["daily_plan_b_over_a"],
                "composite_score": ev["composite_score"],
            }
        )
        return ev["composite_score"]

    result = gp_minimize(
        objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        acq_func="EI",
        noise=1e-6,
        verbose=False,
    )

    best_x = list(result.x)
    best_params, best_method = _params_from_vector(best_x[:7], int(best_x[7]))
    history = pd.DataFrame(history_rows).sort_values("composite_score")
    best_eval = evaluate_plan_b_config(
        sessions,
        facilities,
        manuals,
        best_params,
        capture_method=best_method,
        target_daily_ratio=target_daily_ratio,
    )
    best_eval["optimizer"] = "bayesian_gp"
    best_eval["n_calls"] = n_calls
    best_eval["best_composite_score"] = float(result.fun)
    return best_params, best_method, history, best_eval


def tune_plan_b(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    manuals: pd.DataFrame,
    *,
    method: str = "grid",
    quick: bool = True,
    n_calls: int = 60,
    n_initial_points: int = 20,
    random_state: int = 42,
) -> Tuple[HodV2Params, str, pd.DataFrame, Dict[str, Any]]:
    """
    Find best Plan B parameters.

    method: ``"grid"`` (exhaustive grid on calibration LOO) or ``"bayesian"`` (GP search).

    Returns (hod_params, capture_method, history_df, best_evaluation).
    """
    if method.lower() in ("bayesian", "bayes", "gp"):
        return tune_plan_b_bayesian(
            sessions,
            facilities,
            manuals,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
        )

    cal = build_calibration_windows(sessions, manuals, pedestrians_only=False)
    profile = hour_of_day_profile(sessions.loc[build_quality_mask(sessions)])
    neighbor_map = assign_facility_neighbors(facilities)

    hod_params, _, hod_grid = tune_hod_v2_params(
        cal, neighbor_map, profile, capture_rates_per_facility, quick=quick
    )

    best_eval: Dict[str, Any] = {}
    best_score = float("inf")
    best_method = "mean"
    rows = []

    for cap_method in ("mean", "median") if quick else CAPTURE_METHODS:
        ev = evaluate_plan_b_config(
            sessions, facilities, manuals, hod_params, capture_method=cap_method
        )
        rows.append(
            {
                **ev["hod_params"],
                "capture_method": cap_method,
                **{k: ev[k] for k in (
                    "insample_mae",
                    "max_pct_error",
                    "loo_mae",
                    "daily_plan_b_over_a",
                    "composite_score",
                )},
            }
        )
        if ev["composite_score"] < best_score:
            best_score = ev["composite_score"]
            best_eval = ev
            best_method = cap_method

    summary_grid = pd.DataFrame(rows).sort_values("composite_score")
    best_eval["optimizer"] = "grid"
    return hod_params, best_method, summary_grid, best_eval
