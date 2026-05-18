"""
HOD v2 — manual-anchored hour adjustment (B) + shrink toward 1 away from calibration hours (D)
+ mild density correction vs profile (light A).

footfall = devices × capture_rate × hod_factor × density_factor
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.paths import CALIBRATED_FACILITIES

HOURS = list(range(24))


@dataclass
class HodV2Params:
    shrink_tau: float = 4.0
    smooth_tau: float = 3.0
    density_beta: float = 0.15
    profile_alpha: float = 0.2
    hod_min: float = 0.5
    hod_max: float = 1.25
    density_clip_min: float = 0.85
    density_clip_max: float = 1.15


def circular_hour_distance(h1: int, h2: int) -> float:
    d = abs(int(h1) - int(h2))
    return float(min(d, 24 - d))


def build_hod_anchors(cal_df: pd.DataFrame, cap_map: Dict[int, float]) -> pd.DataFrame:
    """Implied HOD at each manual calibration window: manual / (devices × capture).

    When the same facility × hour appears on multiple days, keep the latest window only
    so anchors are not averaged across very different device counts.
    """
    rows: List[Dict[str, Any]] = []
    for row in cal_df.itertuples(index=False):
        fac = int(row.facility_num)
        rate = float(cap_map.get(fac, 0.1))
        n = max(float(row.clean_unique_devices), 1.0)
        m = float(row.manual_total_count)
        hod = m / (n * rate) if rate > 1e-9 else 1.0
        rows.append(
            {
                "facility_num": fac,
                "hour_of_day": int(row.hour_of_day),
                "started": row.started,
                "hod_anchor": float(np.clip(hod, 0.25, 2.5)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("started").drop_duplicates(
        ["facility_num", "hour_of_day"], keep="last"
    )
    return df[["facility_num", "hour_of_day", "hod_anchor"]]


def _gaussian_weights(target_hour: int, anchor_hours: List[int], tau: float) -> np.ndarray:
    tau = max(float(tau), 1e-6)
    return np.array(
        [
            np.exp(-(circular_hour_distance(target_hour, ah) ** 2) / (2.0 * tau * tau))
            for ah in anchor_hours
        ]
    )


def smooth_hod_for_facility(
    anchors_fac: pd.DataFrame,
    smooth_tau: float,
    hod_min: float,
    hod_max: float,
) -> Dict[int, float]:
    if anchors_fac.empty:
        return {h: 1.0 for h in HOURS}
    anchor_hours = anchors_fac["hour_of_day"].astype(int).tolist()
    anchor_vals = anchors_fac["hod_anchor"].astype(float).tolist()
    curve: Dict[int, float] = {}
    for h in HOURS:
        w = _gaussian_weights(h, anchor_hours, smooth_tau)
        if w.sum() < 1e-9:
            curve[h] = 1.0
        else:
            val = float(np.dot(w, anchor_vals) / w.sum())
            curve[h] = float(np.clip(val, hod_min, hod_max))
    return curve


def build_hod_curves(
    anchors: pd.DataFrame,
    neighbor_map: Dict[int, int],
    all_facilities: List[int],
    params: HodV2Params,
) -> Dict[int, Dict[int, float]]:
    curves: Dict[int, Dict[int, float]] = {}
    for fac in CALIBRATED_FACILITIES:
        sub = anchors.loc[anchors["facility_num"] == fac]
        curves[fac] = smooth_hod_for_facility(
            sub, params.smooth_tau, params.hod_min, params.hod_max
        )
    for fac in all_facilities:
        if fac in curves:
            continue
        ref = int(neighbor_map.get(fac, fac))
        curves[fac] = curves.get(ref, {h: 1.0 for h in HOURS}).copy()
    return curves


def profile_hod_prior(
    fac: int,
    hour: int,
    ref_fac: int,
    cal_hours: List[int],
    prof: pd.Series,
) -> float:
    try:
        prof_val = max(float(prof.loc[(fac, hour)]), 1.0)
    except KeyError:
        prof_val = 1.0
    if cal_hours:
        refs = [
            float(prof.loc[(ref_fac, h)])
            for h in cal_hours
            if (ref_fac, h) in prof.index
        ]
        ref_prof = max(float(np.median(refs)), 1.0) if refs else prof_val
    else:
        ref_prof = prof_val
    return float(np.clip(ref_prof / prof_val, 0.5, 1.25))


def shrink_hod(hod_smooth: float, hour: int, cal_hours: List[int], shrink_tau: float) -> float:
    if not cal_hours:
        return 1.0
    dist = min(circular_hour_distance(hour, ch) for ch in cal_hours)
    if shrink_tau <= 0:
        w = 1.0 if dist < 0.5 else 0.0
    else:
        w = float(np.exp(-(dist ** 2) / (2.0 * shrink_tau * shrink_tau)))
    return float(1.0 + w * (hod_smooth - 1.0))


def density_factor(
    devices: float,
    profile_median: float,
    beta: float,
    clip_min: float,
    clip_max: float,
) -> float:
    if beta <= 0 or devices < 1 or profile_median < 1:
        return 1.0
    f = float((devices / profile_median) ** beta)
    return float(np.clip(f, clip_min, clip_max))


def cal_hours_for_facility(cal_df: pd.DataFrame, fac: int, ref_fac: int) -> List[int]:
    sub = cal_df.loc[cal_df["facility_num"].isin([fac, ref_fac])]
    if sub.empty:
        sub = cal_df.loc[cal_df["facility_num"] == ref_fac]
    return sorted(sub["hour_of_day"].astype(int).unique().tolist())


def hod_v2_components(
    fac: int,
    hour: int,
    devices: float,
    cal_df: pd.DataFrame,
    cap_map: Dict[int, float],
    prof: pd.Series,
    neighbor_map: Dict[int, int],
    curves: Dict[int, Dict[int, float]],
    params: HodV2Params,
) -> Tuple[float, float, float]:
    """Return (hod_factor, density_factor, hod_smooth_before_shrink)."""
    ref_fac = fac if fac in CALIBRATED_FACILITIES else int(neighbor_map.get(fac, fac))
    hod_smooth = float(curves.get(fac, {h: 1.0 for h in HOURS}).get(hour, 1.0))
    if params.profile_alpha > 0:
        ch = cal_hours_for_facility(cal_df, fac, ref_fac)
        prior = profile_hod_prior(fac, hour, ref_fac, ch, prof)
        hod_smooth = (1.0 - params.profile_alpha) * hod_smooth + params.profile_alpha * prior
        hod_smooth = float(np.clip(hod_smooth, params.hod_min, params.hod_max))
    ch = cal_hours_for_facility(cal_df, fac, ref_fac)
    hod = shrink_hod(hod_smooth, hour, ch, params.shrink_tau)
    hod = float(np.clip(hod, params.hod_min, params.hod_max))
    try:
        prof_med = max(float(prof.loc[(fac, hour)]), 1.0)
    except KeyError:
        prof_med = max(devices, 1.0)
    dens = density_factor(
        devices,
        prof_med,
        params.density_beta,
        params.density_clip_min,
        params.density_clip_max,
    )
    return hod, dens, hod_smooth


def predict_calibration_row(
    row: pd.Series,
    cal_df: pd.DataFrame,
    cap_map: Dict[int, float],
    prof: pd.Series,
    neighbor_map: Dict[int, int],
    params: HodV2Params,
    curves: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:
    fac = int(row["facility_num"])
    hour = int(row["hour_of_day"])
    devices = max(float(row["clean_unique_devices"]), 0.0)
    rate = float(cap_map.get(fac) or cap_map.get(neighbor_map.get(fac, fac), 0.1))
    if curves is None:
        anchors = build_hod_anchors(cal_df, cap_map)
        facilities = sorted(cal_df["facility_num"].astype(int).unique().tolist())
        curves = build_hod_curves(anchors, neighbor_map, facilities, params)
    hod, dens, _ = hod_v2_components(
        fac, hour, devices, cal_df, cap_map, prof, neighbor_map, curves, params
    )
    return max(0.0, devices * rate * hod * dens)


def loo_mae_hod_v2(
    cal_df: pd.DataFrame,
    neighbor_map: Dict[int, int],
    profile: pd.DataFrame,
    params: HodV2Params,
    capture_rates_fn,
) -> float:
    prof = profile.set_index(["facility_num", "hour_of_day"])["profile_median_devices"]
    errors: List[float] = []
    for i in range(len(cal_df)):
        train = cal_df.drop(index=cal_df.index[i])
        test = cal_df.iloc[i]
        cap_rates = capture_rates_fn(train)
        cap_map = dict(zip(cap_rates["facility_num"], cap_rates["mean_capture_rate"]))
        pred = predict_calibration_row(test, train, cap_map, prof, neighbor_map, params)
        errors.append(abs(pred - float(test["manual_total_count"])))
    return float(np.mean(errors)) if errors else float("inf")


def tune_hod_v2_params(
    cal_df: pd.DataFrame,
    neighbor_map: Dict[int, int],
    profile: pd.DataFrame,
    capture_rates_fn,
) -> Tuple[HodV2Params, float, pd.DataFrame]:
    """Leave-one-out grid search on manual calibration windows."""
    rows: List[Dict[str, Any]] = []
    best_mae = float("inf")
    best = HodV2Params()
    for shrink_tau in (2.0, 3.0, 4.0, 6.0):
        for smooth_tau in (2.0, 3.0, 4.0):
            for density_beta in (0.0, 0.1, 0.15, 0.2):
                for profile_alpha in (0.0, 0.15, 0.2):
                    p = HodV2Params(
                        shrink_tau=shrink_tau,
                        smooth_tau=smooth_tau,
                        density_beta=density_beta,
                        profile_alpha=profile_alpha,
                    )
                    mae = loo_mae_hod_v2(
                        cal_df, neighbor_map, profile, p, capture_rates_fn
                    )
                    rows.append({**asdict(p), "loo_mae": mae})
                    if mae < best_mae:
                        best_mae = mae
                        best = p
    grid = pd.DataFrame(rows).sort_values("loo_mae")
    return best, best_mae, grid


def calibration_validation_hod_v2(
    cal_df: pd.DataFrame,
    cap_map: Dict[int, float],
    neighbor_map: Dict[int, int],
    profile: pd.DataFrame,
    params: HodV2Params,
) -> pd.DataFrame:
    prof = profile.set_index(["facility_num", "hour_of_day"])["profile_median_devices"]
    anchors = build_hod_anchors(cal_df, cap_map)
    facilities = sorted(cal_df["facility_num"].astype(int).unique().tolist())
    curves = build_hod_curves(anchors, neighbor_map, facilities, params)
    rows = []
    for _, row in cal_df.iterrows():
        pred = predict_calibration_row(
            row, cal_df, cap_map, prof, neighbor_map, params, curves=curves
        )
        manual = float(row["manual_total_count"])
        rows.append(
            {
                "facility_num": int(row["facility_num"]),
                "started": row["started"],
                "hour_of_day": int(row["hour_of_day"]),
                "manual_total_count": manual,
                "predicted_plan_b_v2": round(pred, 2),
                "abs_error": round(abs(pred - manual), 2),
                "pct_error": round(abs(pred - manual) / max(manual, 1) * 100, 2),
            }
        )
    return pd.DataFrame(rows)
