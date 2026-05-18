"""Per-facility footfall and device plots for all sensors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.cleaning import build_quality_mask


def active_sensors(facilities: pd.DataFrame) -> pd.DataFrame:
    """Facilities with a sensor (exclude mall cluster row without box_macs)."""
    fac = facilities.copy()
    if "box_macs" in fac.columns:
        fac = fac.loc[fac["box_macs"].astype(str).str.len() > 2]
    return fac.sort_values("facility_num").drop_duplicates("facility_num")


def _facility_label(facilities: pd.DataFrame, facility_num: int) -> str:
    row = facilities.loc[facilities["facility_num"] == facility_num]
    if len(row):
        name = str(row.iloc[0]["facility_name"])
        return f"{facility_num}\n{name.replace('GB-LVO-', '')}"
    return str(facility_num)


def plot_all_facilities_hourly_grid(
    hourly: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
    value_col: str = "estimated_total_footfall",
    title: str = "Hourly estimated footfall — all sensors",
    ncols: int = 3,
) -> None:
    sensors = active_sensors(facilities)
    facility_nums: List[int] = sensors["facility_num"].astype(int).tolist()
    n = len(facility_nums)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True)
    axes = np.atleast_2d(axes)

    for idx, fac in enumerate(facility_nums):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        grp = hourly.loc[hourly["facility_num"] == fac].sort_values("hour_start")
        if grp.empty:
            ax.set_visible(False)
            continue
        ax.plot(grp["hour_start"], grp[value_col], color="steelblue", linewidth=0.9)
        ax.set_title(_facility_label(sensors, fac), fontsize=8)
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all_facilities_daily_bars(
    hourly: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
    value_col: str = "estimated_total_footfall",
    title: str = "Daily footfall total per sensor",
) -> None:
    sensors = active_sensors(facilities)
    daily = (
        hourly.assign(date=lambda d: d["hour_start"].dt.date)
        .groupby(["facility_num", "date"], as_index=False)[value_col]
        .sum()
    )
    daily["label"] = daily["facility_num"].map(
        lambda f: _facility_label(sensors, int(f)).replace("\n", " ")
    )
    pivot = daily.pivot(index="label", columns="date", values=value_col).fillna(0)
    pivot.columns = [pd.Timestamp(c).strftime("%m-%d") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, max(5, len(pivot) * 0.45)))
    pivot.plot(kind="barh", ax=ax, width=0.85)
    ax.set_xlabel("Daily estimated footfall (sum of hours)")
    ax.set_title(title)
    ax.legend(title="Date", bbox_to_anchor=(1.02, 1), fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all_facilities_footfall_heatmap(
    hourly: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
    value_col: str = "estimated_total_footfall",
    title: str = "Daily footfall per sensor",
) -> None:
    sensors = active_sensors(facilities)
    daily = (
        hourly.assign(date=lambda d: d["hour_start"].dt.date)
        .groupby(["facility_num", "date"], as_index=False)[value_col]
        .sum()
    )
    pivot = daily.pivot(index="facility_num", columns="date", values=value_col).fillna(0)
    pivot.index = pivot.index.map(lambda f: _facility_label(sensors, int(f)))
    col_labels = [pd.Timestamp(c).strftime("%m-%d") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.55)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        ax=ax,
        xticklabels=col_labels,
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Facility")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all_facilities_devices_grid(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
    title: str = "Hourly clean unique devices — all sensors",
    ncols: int = 3,
) -> None:
    clean = sessions.loc[build_quality_mask(sessions)]
    hourly = (
        clean.groupby(["facility_num", "hour_start"], as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "clean_unique_devices"})
    )
    plot_all_facilities_hourly_grid(
        hourly,
        facilities,
        path,
        value_col="clean_unique_devices",
        title=title,
        ncols=ncols,
    )


def plot_plan_a_vs_b_facilities(
    hourly_a: pd.DataFrame,
    hourly_b: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
    col_a: str = "estimated_total_footfall",
    col_b: str = "estimated_total_footfall",
) -> None:
    sensors = active_sensors(facilities)
    facility_nums = sensors["facility_num"].astype(int).tolist()
    ncols = 3
    nrows = math.ceil(len(facility_nums) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True)
    axes = np.atleast_2d(axes)

    for idx, fac in enumerate(facility_nums):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ga = hourly_a.loc[hourly_a["facility_num"] == fac].sort_values("hour_start")
        gb = hourly_b.loc[hourly_b["facility_num"] == fac].sort_values("hour_start")
        if len(ga):
            ax.plot(ga["hour_start"], ga[col_a], label="Plan A", alpha=0.85, linewidth=0.9)
        if len(gb):
            ax.plot(gb["hour_start"], gb[col_b], label="Plan B", alpha=0.85, linewidth=0.9)
        ax.set_title(_facility_label(sensors, fac), fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)

    for idx in range(len(facility_nums), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Plan A vs Plan B — hourly footfall by sensor", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_individual_facility_charts(
    hourly: pd.DataFrame,
    facilities: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sensors = active_sensors(facilities)
    for fac in sensors["facility_num"].astype(int):
        grp = hourly.loc[hourly["facility_num"] == fac].sort_values("hour_start")
        if grp.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(grp["hour_start"], grp[value_col], color="steelblue")
        ax.set_title(_facility_label(sensors, fac).replace("\n", " — "))
        ax.set_ylabel(value_col)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(out_dir / f"{prefix}_facility_{fac}.png", dpi=110)
        plt.close(fig)


def generate_all_facility_plots(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    hourly: pd.DataFrame,
    output_dir: Path,
    *,
    footfall_col: str = "estimated_total_footfall",
    plan_label: str = "Plan",
    hourly_plan_a: Optional[pd.DataFrame] = None,
    per_facility_charts: bool = False,
) -> None:
    """Write grid + heatmap + per-facility PNGs for all sensors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    label = plan_label.replace(" ", "_").lower()

    plot_all_facilities_hourly_grid(
        hourly,
        facilities,
        output_dir / f"{label}_all_facilities_hourly_footfall.png",
        value_col=footfall_col,
        title=f"{plan_label} — hourly footfall (all sensors)",
    )
    plot_all_facilities_footfall_heatmap(
        hourly,
        facilities,
        output_dir / f"{label}_all_facilities_daily_footfall_heatmap.png",
        value_col=footfall_col,
        title=f"{plan_label} — daily footfall per sensor",
    )
    plot_all_facilities_daily_bars(
        hourly,
        facilities,
        output_dir / f"{label}_all_facilities_daily_footfall_bars.png",
        value_col=footfall_col,
    )
    plot_all_facilities_devices_grid(
        sessions,
        facilities,
        output_dir / f"{label}_all_facilities_hourly_devices.png",
    )
    if per_facility_charts:
        _save_individual_facility_charts(
            hourly,
            facilities,
            output_dir / "by_facility",
            value_col=footfall_col,
            prefix=label,
        )

    if hourly_plan_a is not None and not hourly_plan_a.empty:
        plot_plan_a_vs_b_facilities(
            hourly_plan_a,
            hourly,
            facilities,
            output_dir / "plan_a_vs_b_all_facilities_hourly.png",
        )
