"""Advanced Task 4 anomaly methods: matrix profile, STL, isolation forest, Markov journeys."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.cleaning import build_quality_mask
from src.paths import OUTPUT_DIR

IF_FEATURE_CANDIDATES = [
    "raw_sessions",
    "raw_unique_devices",
    "clean_sessions",
    "clean_unique_devices",
    "trusted_unique_devices",
    "local_unique_devices",
    "mean_signal_level",
    "median_session_duration",
]


def resolve_footfall_col(hourly: pd.DataFrame, footfall_col: Optional[str] = None) -> str:
    if footfall_col and footfall_col in hourly.columns:
        return footfall_col
    for col in ("footfall_plan_b", "estimated_total_footfall", "footfall_plan_a"):
        if col in hourly.columns:
            return col
    raise ValueError("No footfall column found in hourly DataFrame")


def _baseline_zscore_and_flag(anomaly_hours: pd.DataFrame) -> Tuple[str, str]:
    if "is_anomaly_hour" in anomaly_hours.columns:
        zcol = "zscore" if "zscore" in anomaly_hours.columns else "zscore_global"
        return zcol, "is_anomaly_hour"
    if "is_anomaly_hod" in anomaly_hours.columns:
        if "zscore_hod_baseline" in anomaly_hours.columns:
            return "zscore_hod_baseline", "is_anomaly_hod"
        if "zscore_global" in anomaly_hours.columns:
            return "zscore_global", "is_anomaly_hod"
    return "zscore", "is_anomaly_hour"


def _robust_abs_zscore(values: pd.Series) -> pd.Series:
    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = 1.4826 * mad if mad > 0 else 1.0
    return (values - median).abs() / scale


def _minmax(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)
    lo, hi = float(valid.min()), float(valid.max())
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
    valid = stds[:, 0] > 1e-9

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
    hourly: pd.DataFrame,
    footfall_col: Optional[str] = None,
    windows: Tuple[int, ...] = (6, 12, 24),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    col = resolve_footfall_col(hourly, footfall_col)
    summaries: List[pd.DataFrame] = []
    details: List[pd.DataFrame] = []

    for facility_num, grp in hourly.groupby("facility_num", sort=False):
        grp = grp.sort_values("hour_start").copy()
        values = grp[col].to_numpy(dtype=float)
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

            details.append(
                pd.DataFrame(
                    {
                        "facility_num": facility_num,
                        "hour_start": grp.iloc[centers]["hour_start"].to_numpy(),
                        "window_size": window,
                        "matrix_profile_distance": profile,
                        "matrix_profile_robust_z": mp_z,
                    }
                )
            )

        threshold = float(base["matrix_profile_score"].quantile(0.97))
        base["is_matrix_profile_anomaly"] = base["matrix_profile_score"] >= threshold
        summaries.append(base)

    summary_df = (
        pd.concat(summaries, ignore_index=True)
        if summaries
        else pd.DataFrame(
            columns=[
                "facility_num",
                "hour_start",
                "matrix_profile_score",
                "matrix_profile_window",
                "is_matrix_profile_anomaly",
            ]
        )
    )
    detail_df = (
        pd.concat(details, ignore_index=True)
        if details
        else pd.DataFrame(
            columns=[
                "facility_num",
                "hour_start",
                "window_size",
                "matrix_profile_distance",
                "matrix_profile_robust_z",
            ]
        )
    )
    return summary_df.sort_values(["facility_num", "hour_start"]), detail_df


def stl_hourly_anomalies(
    hourly: pd.DataFrame,
    footfall_col: Optional[str] = None,
    period: int = 24,
) -> pd.DataFrame:
    from statsmodels.tsa.seasonal import STL

    col = resolve_footfall_col(hourly, footfall_col)
    rows: List[pd.DataFrame] = []

    for _, grp in hourly.groupby("facility_num", sort=False):
        grp = grp.sort_values("hour_start").copy()
        series = grp[col].astype(float).to_numpy()

        if len(series) < period * 2:
            grp["stl_trend"] = np.nan
            grp["stl_seasonal"] = np.nan
            grp["stl_resid"] = series - np.mean(series)
            grp["stl_score"] = _robust_abs_zscore(grp["stl_resid"])
        else:
            fit = STL(series, period=period, robust=True).fit()
            grp["stl_trend"] = fit.trend
            grp["stl_seasonal"] = fit.seasonal
            grp["stl_resid"] = fit.resid
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
    hourly: pd.DataFrame,
    footfall_col: Optional[str] = None,
) -> pd.DataFrame:
    col = resolve_footfall_col(hourly, footfall_col)
    feature_cols = [c for c in IF_FEATURE_CANDIDATES if c in hourly.columns]
    if col not in feature_cols:
        feature_cols.append(col)
    if len(feature_cols) < 3:
        raise ValueError(f"Need at least 3 features for isolation forest; got {feature_cols}")

    rows: List[pd.DataFrame] = []
    for _, grp in hourly.groupby("facility_num", sort=False):
        grp = grp.sort_values("hour_start").copy()
        x = grp[feature_cols].fillna(0.0).to_numpy(dtype=float)
        if len(x) < 10:
            grp["isolation_forest_score"] = 0.0
            grp["is_isolation_forest_anomaly"] = False
            rows.append(grp[["facility_num", "hour_start", "isolation_forest_score", "is_isolation_forest_anomaly"]])
            continue
        model = IsolationForest(n_estimators=300, contamination=0.03, random_state=42)
        model.fit(x)
        scores = -model.score_samples(x)
        grp["isolation_forest_score"] = scores
        threshold = float(np.quantile(scores, 0.97))
        grp["is_isolation_forest_anomaly"] = grp["isolation_forest_score"] >= threshold
        rows.append(grp)

    out = pd.concat(rows, ignore_index=True)
    return out[
        ["facility_num", "hour_start", "isolation_forest_score", "is_isolation_forest_anomaly"]
    ].sort_values(["facility_num", "hour_start"])


def _build_journey_events(sessions: pd.DataFrame) -> pd.DataFrame:
    clean = sessions.loc[
        build_quality_mask(sessions), ["device_id", "facility_num", "session_start"]
    ]
    multi = clean.groupby("device_id")["facility_num"].nunique()
    multi_ids = multi.loc[multi > 1].index
    events = clean.loc[clean["device_id"].isin(multi_ids)].sort_values(
        ["device_id", "session_start"]
    )
    events = events.copy()
    events["prev_facility"] = events.groupby("device_id")["facility_num"].shift()
    events = events.loc[
        events["prev_facility"].isna()
        | (events["facility_num"] != events["prev_facility"])
    ]
    events["next_facility"] = events.groupby("device_id")["facility_num"].shift(-1)
    return events


def markov_journey_anomalies(
    sessions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    row_totals = (
        trans_counts.groupby("facility_num", as_index=False)["transition_count"]
        .sum()
        .rename(columns={"transition_count": "row_total"})
    )
    trans_probs = trans_counts.merge(row_totals, on="facility_num", how="left")
    trans_probs["transition_prob"] = (trans_probs["transition_count"] + alpha) / (
        trans_probs["row_total"] + alpha * state_count
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
            src, dst = seq[i], seq[i + 1]
            prob = prob_lookup.get(
                (src, dst),
                alpha / (row_lookup.get(src, 0) + alpha * state_count),
            )
            prob = max(float(prob), 1e-12)
            if prob < 0.02:
                rare_count += 1
            nll_values.append(float(-np.log(prob)))

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
                "markov_avg_neg_log_prob": float(np.mean(nll_values)),
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
    j_thresh = float(journey_df["markov_score"].quantile(0.97))
    journey_df["is_markov_journey_anomaly"] = journey_df["markov_score"] >= j_thresh
    journey_df = journey_df.sort_values("markov_score", ascending=False)

    markov_hour = (
        journey_df.groupby(["end_facility", "end_hour_start"], as_index=False)
        .agg(
            markov_score=("markov_score", "max"),
            markov_journey_count=("device_id", "size"),
        )
        .rename(columns={"end_facility": "facility_num", "end_hour_start": "hour_start"})
    )
    h_thresh = (
        float(markov_hour["markov_score"].quantile(0.97)) if len(markov_hour) else 0.0
    )
    markov_hour["is_markov_hour_anomaly"] = markov_hour["markov_score"] >= h_thresh
    return journey_df, markov_hour.sort_values(["facility_num", "hour_start"])


def compare_anomaly_methods(
    sessions: pd.DataFrame,
    hourly: pd.DataFrame,
    anomaly_hours: pd.DataFrame,
    footfall_col: Optional[str] = None,
    plan_name: str = "plan_a",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, float],
]:
    """Run advanced detectors and merge with baseline Task 4 flags into one comparison table."""
    col = resolve_footfall_col(hourly, footfall_col)
    zcol, flag_col = _baseline_zscore_and_flag(anomaly_hours)

    matrix_profile_df, matrix_profile_detail = matrix_profile_hourly_anomalies(hourly, col)
    stl_df = stl_hourly_anomalies(hourly, col)
    iso_df = isolation_forest_hourly_anomalies(hourly, col)
    markov_journey_df, markov_hour_df = markov_journey_anomalies(sessions)

    base_cols = ["facility_num", "hour_start", col]
    comparison = hourly[base_cols].copy()
    comparison["plan_name"] = plan_name
    comparison["footfall_col"] = col

    merge_cols = [
        (anomaly_hours, [zcol, flag_col]),
        (matrix_profile_df, ["matrix_profile_score", "is_matrix_profile_anomaly"]),
        (stl_df, ["stl_score", "is_stl_anomaly"]),
        (iso_df, ["isolation_forest_score", "is_isolation_forest_anomaly"]),
        (markov_hour_df, ["markov_score", "markov_journey_count", "is_markov_hour_anomaly"]),
    ]
    for df, cols in merge_cols:
        use = [c for c in cols if c in df.columns]
        comparison = comparison.merge(
            df[["facility_num", "hour_start"] + use],
            on=["facility_num", "hour_start"],
            how="left",
        )

    if "is_anomaly_consensus" in anomaly_hours.columns:
        comparison = comparison.merge(
            anomaly_hours[["facility_num", "hour_start", "is_anomaly_consensus"]],
            on=["facility_num", "hour_start"],
            how="left",
        )

    comparison["zscore_score"] = comparison[zcol].abs() if zcol in comparison.columns else 0.0
    comparison["zscore_norm"] = _minmax(comparison["zscore_score"].fillna(0))
    comparison["matrix_profile_norm"] = _minmax(
        comparison["matrix_profile_score"].fillna(0)
    )
    comparison["stl_norm"] = _minmax(comparison["stl_score"].fillna(0))
    comparison["isolation_forest_norm"] = _minmax(
        comparison["isolation_forest_score"].fillna(0)
    )
    comparison["markov_norm"] = _minmax(comparison["markov_score"].fillna(0))

    bool_cols = [
        flag_col,
        "is_matrix_profile_anomaly",
        "is_stl_anomaly",
        "is_isolation_forest_anomaly",
        "is_markov_hour_anomaly",
    ]
    if "is_anomaly_consensus" in comparison.columns:
        bool_cols.append("is_anomaly_consensus")

    for c in bool_cols:
        if c in comparison.columns:
            comparison[c] = comparison[c].fillna(False).astype(bool)

    norm_cols = [
        "zscore_norm",
        "matrix_profile_norm",
        "stl_norm",
        "isolation_forest_norm",
        "markov_norm",
    ]
    comparison["ensemble_score"] = comparison[norm_cols].mean(axis=1).round(6)
    present_bool = [c for c in bool_cols if c in comparison.columns]
    comparison["method_agreement_count"] = comparison[present_bool].sum(axis=1)
    comparison["is_consensus_anomaly"] = comparison["method_agreement_count"] >= 2
    comparison = comparison.sort_values("ensemble_score", ascending=False)

    summary = {
        "plan_name": plan_name,
        "footfall_col": col,
        "baseline_zscore_col": zcol,
        "baseline_flag_col": flag_col,
        "matrix_profile_anomaly_hours": int(
            matrix_profile_df["is_matrix_profile_anomaly"].sum()
        ),
        "stl_anomaly_hours": int(stl_df["is_stl_anomaly"].sum()),
        "isolation_forest_anomaly_hours": int(
            iso_df["is_isolation_forest_anomaly"].sum()
        ),
        "markov_journey_anomalies": int(markov_journey_df["is_markov_journey_anomaly"].sum())
        if len(markov_journey_df)
        else 0,
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


def run_advanced_anomaly_pipeline(
    sessions: pd.DataFrame,
    hourly: pd.DataFrame,
    anomaly_hours: pd.DataFrame,
    footfall_col: Optional[str] = None,
    plan_name: str = "plan_a",
    output_dir: Optional[Path] = None,
    *,
    write_outputs: bool = False,
) -> Dict[str, object]:
    """Run full advanced comparison; optionally write one comparison CSV."""
    out = output_dir or OUTPUT_DIR

    (
        matrix_profile_df,
        matrix_profile_detail,
        stl_df,
        iso_df,
        markov_journey_df,
        markov_hour_df,
        comparison,
        summary,
    ) = compare_anomaly_methods(sessions, hourly, anomaly_hours, footfall_col, plan_name)

    if write_outputs:
        out.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(
            out / f"task4_anomaly_method_comparison_{plan_name}.csv", index=False
        )

    return {
        "comparison": comparison,
        "matrix_profile": matrix_profile_df,
        "matrix_profile_detail": matrix_profile_detail,
        "stl": stl_df,
        "isolation_forest": iso_df,
        "markov_journeys": markov_journey_df,
        "markov_hourly": markov_hour_df,
        "summary": summary,
        "output_dir": out,
    }
