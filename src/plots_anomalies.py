"""Task 4 anomaly and pattern visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.paths import OUTPUT_DIR


def _footfall_column(df: pd.DataFrame) -> str:
    for col in ("footfall_plan_b", "estimated_total_footfall", "footfall_plan_a"):
        if col in df.columns:
            return col
    raise KeyError("No footfall column in anomaly_hours")


def _anomaly_mask(df: pd.DataFrame) -> pd.Series:
    if "is_anomaly_hour" in df.columns:
        return df["is_anomaly_hour"].astype(bool)
    if "is_anomaly_consensus" in df.columns:
        return df["is_anomaly_consensus"].astype(bool)
    if "is_anomaly_hod" in df.columns:
        return df["is_anomaly_hod"].astype(bool)
    return pd.Series(False, index=df.index)


def plot_flag_rates(flag_summary: Dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["is_excluded", "is_fake", "is_anomaly", "flagged_any"]
    rates = [flag_summary.get(k, 0) * 100 for k in labels]
    colors = ["#e74c3c", "#e67e22", "#9b59b6", "#34495e"]
    bars = ax.bar(labels, rates, color=colors)
    ax.set_ylabel("% of sessions")
    ax.set_title("Built-in session flag rates")
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{rate:.3f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_device_session_distribution(
    device_stats: pd.DataFrame, threshold: float, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = device_stats["total_sessions"]
    ax.hist(
        np.log10(vals.clip(lower=1)),
        bins=80,
        color="steelblue",
        alpha=0.75,
        edgecolor="white",
    )
    ax.axvline(
        np.log10(max(threshold, 1)),
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Top 0.01% threshold ({threshold:.0f} sessions)",
    )
    ax.set_xlabel("log10(total sessions per device)")
    ax.set_ylabel("Number of devices")
    ax.set_title("Device activity distribution (long-tail pattern)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_suspicious_device_patterns(
    device_stats: pd.DataFrame,
    suspicious: pd.DataFrame,
    threshold: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    normal = device_stats.loc[
        ~device_stats["device_id"].isin(suspicious["device_id"])
    ]
    if len(normal) > 8000:
        normal = normal.sample(8000, random_state=42)
    ax.scatter(
        normal["total_sessions"],
        normal["night_session_ratio"],
        s=8,
        alpha=0.15,
        c="gray",
        label="Other devices (sample)",
    )
    ax.scatter(
        suspicious["total_sessions"],
        suspicious["night_session_ratio"],
        s=40,
        alpha=0.85,
        c="crimson",
        edgecolors="black",
        linewidths=0.3,
        label=f"Suspicious unflagged (n={len(suspicious)})",
    )
    ax.axhline(0.6, color="orange", linestyle=":", label="Night ratio 60%")
    ax.axvline(threshold, color="navy", linestyle=":", label=f"Session threshold {threshold:.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("Total sessions (log scale)")
    ax.set_ylabel("Night session ratio (hours 0–5 UTC)")
    ax.set_title("Suspicious device patterns: high activity & night presence")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_facility_spread_patterns(
    device_stats: pd.DataFrame, suspicious: pd.DataFrame, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    normal = device_stats.loc[
        ~device_stats["device_id"].isin(suspicious["device_id"])
    ]
    if len(normal) > 8000:
        normal = normal.sample(8000, random_state=42)
    ax.scatter(
        normal["unique_facilities"],
        normal["active_days"],
        s=8,
        alpha=0.12,
        c="gray",
        label="Other devices (sample)",
    )
    ax.scatter(
        suspicious["unique_facilities"],
        suspicious["active_days"],
        s=50,
        alpha=0.9,
        c="crimson",
        label="Suspicious unflagged",
    )
    ax.set_xlabel("Unique facilities visited")
    ax.set_ylabel("Active days in week")
    ax.set_title("Multi-sensor spread vs persistence (suspicious devices)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_hourly_footfall_with_anomalies(
    anomaly_hours: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
    top_n_facilities: int = 4,
) -> None:
    footfall_col = _footfall_column(anomaly_hours)
    anomaly_mask = _anomaly_mask(anomaly_hours)
    lookup = facilities.set_index("facility_num")["facility_name"].to_dict()
    top_facilities = (
        anomaly_hours.groupby("facility_num")[footfall_col]
        .mean()
        .sort_values(ascending=False)
        .head(top_n_facilities)
        .index.tolist()
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    for ax, fac in zip(axes, top_facilities):
        grp = anomaly_hours.loc[anomaly_hours["facility_num"] == fac].sort_values(
            "hour_start"
        )
        ax.plot(
            grp["hour_start"],
            grp[footfall_col],
            color="steelblue",
            linewidth=1,
            alpha=0.8,
        )
        peaks = grp.loc[anomaly_mask.reindex(grp.index, fill_value=False)]
        if len(peaks):
            ax.scatter(
                peaks["hour_start"],
                peaks[footfall_col],
                color="crimson",
                s=35,
                zorder=5,
                label="Anomaly hour (|z|≥3)",
            )
        ax.set_title(lookup.get(fac, str(fac)), fontsize=9)
        ax.tick_params(axis="x", rotation=45)
        if len(peaks):
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Hourly estimated footfall with anomaly peaks", y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_anomaly_zscore_heatmap(
    anomaly_hours: pd.DataFrame, facilities: pd.DataFrame, path: Path
) -> None:
    df = anomaly_hours.copy()
    df["date"] = df["hour_start"].dt.date.astype(str)
    pivot = df.pivot_table(
        index="facility_num",
        columns="date",
        values="zscore",
        aggfunc="max",
    )
    lookup = facilities.set_index("facility_num")["facility_name"]
    pivot.index = pivot.index.map(lambda f: lookup.get(f, str(f)))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        center=0,
        vmin=-5,
        vmax=5,
        annot=True,
        fmt=".1f",
        ax=ax,
        cbar_kws={"label": "Max hourly z-score"},
    )
    ax.set_title("Peak footfall z-score per sensor per day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Facility")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_top_anomaly_hours(
    anomaly_hours: pd.DataFrame, facilities: pd.DataFrame, path: Path, n: int = 15
) -> None:
    top = anomaly_hours.nlargest(n, "zscore").copy()
    lookup = facilities.set_index("facility_num")["facility_name"]
    top["label"] = top.apply(
        lambda r: f"{lookup.get(r['facility_num'], r['facility_num'])}\n"
        f"{pd.Timestamp(r['hour_start']).strftime('%m-%d %H:%M')}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if z > 0 else "#3498db" for z in top["zscore"]]
    ax.barh(top["label"], top["zscore"], color=colors)
    ax.axvline(3, color="gray", linestyle="--", label="z = 3 threshold")
    ax.axvline(-3, color="gray", linestyle="--")
    ax.set_xlabel("Z-score (estimated footfall vs facility mean)")
    ax.set_title(f"Top {n} hourly footfall anomalies")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sessions_by_hour_flagged(
    sessions: pd.DataFrame, path: Path
) -> None:
    """Hour-of-day pattern: clean vs flagged session volume."""
    sessions = sessions.copy()
    sessions["flagged_any"] = sessions[
        ["is_excluded", "is_anomaly", "is_fake"]
    ].any(axis=1)
    hourly = (
        sessions.groupby(["hour_of_day", "flagged_any"], as_index=False)
        .size()
        .rename(columns={"size": "sessions"})
    )
    pivot = hourly.pivot(index="hour_of_day", columns="flagged_any", values="sessions").fillna(
        0
    )
    pivot.columns = ["Clean", "Flagged"]
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#3498db", "#e74c3c"])
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Sessions")
    ax.set_title("Session volume by hour: clean vs flagged")
    ax.legend(title="")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def generate_all_anomaly_plots(
    sessions: pd.DataFrame,
    facilities: pd.DataFrame,
    device_stats: pd.DataFrame,
    suspicious: pd.DataFrame,
    anomaly_hours: pd.DataFrame,
    anomaly_summary: Dict[str, float],
    output_dir: Path | None = None,
) -> None:
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    threshold = anomaly_summary["high_session_threshold_top_0_1pct"]

    plot_flag_rates(anomaly_summary, out / "task4_flag_rates.png")
    plot_device_session_distribution(
        device_stats, threshold, out / "task4_device_session_distribution.png"
    )
    plot_suspicious_device_patterns(
        device_stats, suspicious, threshold, out / "task4_suspicious_device_patterns.png"
    )
    plot_facility_spread_patterns(
        device_stats, suspicious, out / "task4_facility_spread_patterns.png"
    )
    plot_hourly_footfall_with_anomalies(
        anomaly_hours, facilities, out / "task4_hourly_footfall_anomalies.png"
    )
    plot_anomaly_zscore_heatmap(
        anomaly_hours, facilities, out / "task4_anomaly_zscore_heatmap.png"
    )
    plot_top_anomaly_hours(anomaly_hours, facilities, out / "task4_top_anomaly_hours.png")
    plot_sessions_by_hour_flagged(sessions, out / "task4_sessions_by_hour_flagged.png")
