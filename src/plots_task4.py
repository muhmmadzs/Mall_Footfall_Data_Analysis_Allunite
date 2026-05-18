"""Task 4 — anomaly detection visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.paths import PLOTS_DIR
from src.plots_anomalies import generate_all_anomaly_plots


def _normalize_anomaly_summary(summary: Dict[str, float]) -> Dict[str, float]:
    out = dict(summary)
    if "high_session_threshold_top_0_1pct" not in out and "high_session_threshold" in out:
        out["high_session_threshold_top_0_1pct"] = out["high_session_threshold"]
    return out


def generate_task4_plots(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    device_stats: pd.DataFrame,
    suspicious: pd.DataFrame,
    anomaly_hours: pd.DataFrame,
    anomaly_summary: Dict[str, float],
    output_dir: Path | None = None,
) -> Path:
    out = output_dir or (PLOTS_DIR / "task4")
    ah = anomaly_hours.copy()
    if "zscore" not in ah.columns:
        for col in ("zscore_hod_baseline", "zscore_robust", "zscore_global"):
            if col in ah.columns:
                ah["zscore"] = ah[col]
                break
    generate_all_anomaly_plots(
        sessions,
        facilities,
        device_stats,
        suspicious,
        ah,
        _normalize_anomaly_summary(anomaly_summary),
        output_dir=out,
    )
    return out
