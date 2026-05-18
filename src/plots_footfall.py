"""Footfall and device-detection visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.cleaning import build_quality_mask
from src.eda import daily_device_footfall_summary, weekly_device_footfall_summary
from src.paths import OUTPUT_DIR
from src.plots_facilities import generate_all_facility_plots


def plot_daily_sessions_devices(daily_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    dates = pd.to_datetime(daily_df["date"])
    x = np.arange(len(daily_df))
    width = 0.2
    ax.bar(x - 1.5 * width, daily_df["total_sessions"], width, label="Total sessions")
    ax.bar(x - 0.5 * width, daily_df["clean_sessions"], width, label="Clean sessions")
    ax.bar(
        x + 0.5 * width,
        daily_df["unique_devices_mall"],
        width,
        label="Unique devices (mall dedup)",
    )
    ax.bar(
        x + 1.5 * width,
        daily_df["unique_devices_clean"],
        width,
        label="Unique devices (clean)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Daily sessions and unique devices detected")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_daily_device_types(daily_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    dates = [pd.Timestamp(d).strftime("%m-%d") for d in daily_df["date"]]
    bottom = np.zeros(len(daily_df))
    for col, color in [
        ("trusted_devices", "#2ecc71"),
        ("local_devices", "#3498db"),
        ("other_devices", "#95a5a6"),
    ]:
        vals = daily_df[col].to_numpy()
        ax.bar(dates, vals, bottom=bottom, label=col.replace("_", " "), color=color)
        bottom += vals
    ax.set_ylabel("Unique devices (clean)")
    ax.set_title("Daily device types (trusted / local / other)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_daily_footfall_vs_devices(daily_df: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    dates = pd.to_datetime(daily_df["date"])
    x = np.arange(len(daily_df))
    ax1.bar(x, daily_df["estimated_footfall"], color="steelblue", alpha=0.7, label="Est. footfall")
    ax1.set_ylabel("Estimated footfall (sum facility-hours)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        daily_df["unique_devices_clean"],
        color="coral",
        marker="o",
        linewidth=2,
        label="Unique devices (clean)",
    )
    ax2.set_ylabel("Unique devices (mall dedup, clean)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Daily estimated footfall vs detected unique devices")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_weekly_totals(
    daily_df: pd.DataFrame, weekly_stats: Dict[str, Any], path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics = [
        ("Total sessions", weekly_stats["total_sessions"]),
        ("Clean sessions", weekly_stats["clean_sessions"]),
        ("Est. footfall (week)", int(weekly_stats["estimated_footfall"])),
        ("Unique devices (week)", weekly_stats["unique_devices_clean_week"]),
    ]
    axes[0].barh(
        [m[0] for m in metrics],
        [m[1] for m in metrics],
        color=["#3498db", "#2ecc71", "#9b59b6", "#e67e22"],
    )
    axes[0].set_title("Week totals (Apr 20–26)")
    for i, (_, v) in enumerate(metrics):
        axes[0].text(v, i, f" {v:,.0f}", va="center", fontsize=9)

    dedup = weekly_stats["unique_devices_clean_week"]
    summed = weekly_stats["sum_facility_unique_devices_week"]
    axes[1].bar(
        ["Mall dedup\n(unique)", "Sum of facility\ndaily uniques"],
        [dedup, summed],
        color=["#2ecc71", "#e74c3c"],
    )
    axes[1].set_ylabel("Device count")
    axes[1].set_title("Double-counting effect (same phone, multiple sensors)")
    axes[1].text(
        0.5,
        max(dedup, summed) * 0.9,
        f"Multi-sensor overlap ~6.4% of devices (Task 2)",
        ha="center",
        fontsize=9,
        style="italic",
    )
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_facility_daily_heatmap(
    sessions: pd.DataFrame, facilities: pd.DataFrame, path: Path
) -> None:
    clean = sessions.loc[build_quality_mask(sessions)]
    pivot = (
        clean.groupby(["facility_num", "date"], as_index=False)["device_id"]
        .nunique()
        .pivot(index="facility_num", columns="date", values="device_id")
        .fillna(0)
    )
    lookup = facilities.set_index("facility_num")["facility_name"]
    pivot.index = pivot.index.map(lambda f: lookup.get(f, str(f)))

    fig, ax = plt.subplots(figsize=(10, 6))
    col_labels = [pd.Timestamp(c).strftime("%m-%d") for c in pivot.columns]
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax,
        xticklabels=col_labels,
    )
    ax.set_title("Daily unique devices (clean) per sensor")
    ax.set_xlabel("Date")
    ax.set_ylabel("Facility")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration_overlay(
    sessions: pd.DataFrame,
    manuals: pd.DataFrame,
    hourly: pd.DataFrame,
    path: Path,
) -> None:
    manual_total = (
        manuals.groupby(["facility_num", "started", "window_end"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "manual_total"})
    )
    rows = []
    for row in manual_total.itertuples(index=False):
        hour_rows = hourly.loc[
            (hourly["facility_num"] == row.facility_num)
            & (hourly["hour_start"] >= row.started)
            & (hourly["hour_start"] < row.window_end)
        ]
        rows.append(
            {
                "label": f"{row.facility_num}\n{row.started.strftime('%m-%d %H:%M')}",
                "manual_total": row.manual_total,
                "estimated_total": hour_rows["estimated_total_footfall"].sum(),
            }
        )
    cal_plot = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cal_plot))
    width = 0.35
    ax.bar(x - width / 2, cal_plot["manual_total"], width, label="Manual count", color="#2ecc71")
    ax.bar(
        x + width / 2,
        cal_plot["estimated_total"],
        width,
        label="Estimated (model)",
        color="#3498db",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cal_plot["label"], fontsize=8)
    ax.set_ylabel("People counted")
    ax.set_title("Manual counting vs estimated footfall (4 calibration windows)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def generate_all_footfall_plots(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    manuals: pd.DataFrame,
    hourly: pd.DataFrame,
    multi_sensor_rate: float | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Build daily summary CSV and all footfall/device PNGs."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    daily_df = daily_device_footfall_summary(sessions, hourly)
    weekly_stats = weekly_device_footfall_summary(daily_df, sessions)
    if multi_sensor_rate is not None:
        weekly_stats["multi_sensor_rate"] = multi_sensor_rate

    plot_daily_sessions_devices(daily_df, out / "footfall_daily_sessions_and_devices.png")
    plot_daily_device_types(daily_df, out / "footfall_daily_device_types_stacked.png")
    plot_daily_footfall_vs_devices(
        daily_df, out / "footfall_daily_estimated_vs_detected.png"
    )
    plot_weekly_totals(daily_df, weekly_stats, out / "footfall_weekly_totals.png")
    plot_facility_daily_heatmap(
        sessions, facilities, out / "footfall_daily_by_facility_heatmap.png"
    )
    plot_calibration_overlay(
        sessions, manuals, hourly, out / "footfall_manual_calibration_overlay.png"
    )
    generate_all_facility_plots(
        sessions,
        facilities,
        hourly,
        out,
        footfall_col="estimated_total_footfall",
        plan_label="Plan A",
    )

    return daily_df, weekly_stats
