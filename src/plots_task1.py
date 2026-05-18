"""Task 1 — calibration and footfall estimation plots (Plan A and Plan B)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.paths import OUTPUT_DIR, PLOTS_DIR
from src.plots_footfall import (
    generate_all_footfall_plots,
    plot_calibration_overlay,
)


def task1_plot_dir(plan: str = "a", base: Path | None = None) -> Path:
    root = base or PLOTS_DIR
    sub = "plan_b" if plan.lower() == "b" else "plan_a"
    return root / "task1" / sub


def generate_task1_plots_plan_a(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    manuals: pd.DataFrame,
    hourly: pd.DataFrame,
    multi_sensor_rate: float | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any], Path]:
    out = output_dir or task1_plot_dir("a")
    out.mkdir(parents=True, exist_ok=True)
    daily_df, weekly_stats = generate_all_footfall_plots(
        sessions,
        facilities,
        manuals,
        hourly,
        multi_sensor_rate=multi_sensor_rate,
        output_dir=out,
    )
    plot_calibration_overlay(
        sessions,
        manuals,
        hourly,
        out / "manual_calibration_overlay.png",
    )
    return daily_df, weekly_stats, out


def generate_task1_plots_plan_b(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    hourly: pd.DataFrame,
    mall_daily: pd.DataFrame,
    cal_df: pd.DataFrame,
    comparison_daily: pd.DataFrame,
    hod_validation: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Static + comparison plots for Plan B task 1 outputs."""
    out = output_dir or task1_plot_dir("b")
    out.mkdir(parents=True, exist_ok=True)

    if not comparison_daily.empty:
        _plot_daily_comparison(comparison_daily, out / "daily_plan_a_vs_b.png")

    val = hod_validation
    if val is None or val.empty:
        val = cal_df
    if not val.empty and "predicted_plan_b_v2" in val.columns:
        _plot_hod_validation(val, out / "hod_v2_calibration_validation.png")
    elif not cal_df.empty:
        _plot_calibration_bars(cal_df, out / "calibration_windows.png")

    if not mall_daily.empty:
        _plot_mall_visitors_daily(mall_daily, out / "mall_visitors_daily.png")

    if not hourly.empty and sessions is not None:
        from src.plots_facilities import generate_all_facility_plots

        generate_all_facility_plots(
            sessions,
            facilities,
            hourly,
            out,
            footfall_col="footfall_plan_b",
            plan_label="Plan B",
        )

    return out


def _plot_daily_comparison(comp: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    if comp.empty:
        return
    fig, ax1 = plt.subplots(figsize=(12, 5))
    dates = pd.to_datetime(comp["date"])
    x = range(len(comp))
    width = 0.35
    if "footfall_plan_a" in comp.columns:
        ax1.bar(
            [i - width / 2 for i in x],
            comp["footfall_plan_a"],
            width,
            label="Plan A",
            color="#4a90d9",
        )
    ax1.bar(
        [i + width / 2 for i in x],
        comp["footfall_plan_b"],
        width,
        label="Plan B",
        color="#e67e22",
    )
    ax1.set_ylabel("Daily footfall (sensor-hour sum)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([d.strftime("%m-%d") for d in dates], rotation=45, ha="right")
    if "estimated_mall_visitors" in comp.columns:
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            comp["estimated_mall_visitors"],
            "o-",
            color="#27ae60",
            linewidth=2,
            label="Mall visitors",
        )
        ax2.set_ylabel("Mall visitors")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    else:
        ax1.legend(fontsize=8)
    ax1.set_title("Plan A vs Plan B — daily comparison")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_calibration_bars(cal_df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if "manual_total_count" not in cal_df.columns:
        return
    labels = [
        f"{int(r.facility_num)}\n{pd.Timestamp(r.started).strftime('%m-%d %H:%M')}"
        for r in cal_df.itertuples()
    ]
    manual = cal_df["manual_total_count"].to_numpy()
    pred_col = "pred_capture_rate" if "pred_capture_rate" in cal_df.columns else None
    pred = cal_df[pred_col].to_numpy() if pred_col else np.zeros_like(manual)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, manual, w, label="Manual", color="#2ecc71")
    if pred_col:
        ax.bar(x + w / 2, pred, w, label="Plan B estimate", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Calibration windows — manual vs Plan B")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_hod_validation(cal_df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    df = cal_df.copy()
    if "predicted_plan_b_v2" in df.columns:
        pred = df["predicted_plan_b_v2"]
    elif "pred_capture_rate" in df.columns:
        pred = df["pred_capture_rate"]
    else:
        return
    labels = [
        f"{int(r.facility_num)}\n{pd.Timestamp(r.started).strftime('%m-%d %H:%M')}"
        for r in df.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, df["manual_total_count"], w, label="Manual", color="#2ecc71")
    ax.bar(x + w / 2, pred, w, label="Plan B (HOD v2)", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("HOD v2 calibration validation")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_mall_visitors_daily(mall_daily: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    dates = pd.to_datetime(mall_daily["date"])
    ax.bar(dates, mall_daily["estimated_mall_visitors"], color="#27ae60", alpha=0.8)
    ax.set_ylabel("Estimated mall visitors")
    ax.set_title("Daily mall visitors (deduped devices × capture)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
