"""Task 2 — multi-sensor overlap visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.paths import PLOTS_DIR


def plot_multi_sensor_summary(summary: Dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    total = summary.get("total_unique_devices_clean", 0)
    multi = summary.get("multi_sensor_devices", 0)
    single = max(total - multi, 0)
    ax.bar(
        ["Single sensor", "Multi-sensor"],
        [single, multi],
        color=["#3498db", "#e67e22"],
    )
    rate = summary.get("multi_sensor_rate", 0) * 100
    ax.set_ylabel("Unique devices (clean)")
    ax.set_title(f"Device overlap across sensors ({rate:.1f}% multi-sensor)")
    for i, v in enumerate([single, multi]):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_top_sensor_pairs(pair_counts: pd.DataFrame, path: Path, n: int = 15) -> None:
    top = pair_counts.nlargest(n, "shared_devices").copy()
    top["label"] = top.apply(
        lambda r: f"{r.get('facility_name_x', r['facility_num_x'])}\n↔\n"
        f"{r.get('facility_name_y', r['facility_num_y'])}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(top))))
    ax.barh(top["label"], top["shared_devices"], color="#9b59b6")
    ax.set_xlabel("Shared unique devices")
    ax.set_title(f"Top {n} sensor pairs by shared devices")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_jaccard_heatmap(
    pair_counts: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
) -> None:
    facs = sorted(facilities["facility_num"].unique())
    lookup = facilities.drop_duplicates("facility_num").set_index("facility_num")[
        "facility_name"
    ]
    mat = pd.DataFrame(0.0, index=facs, columns=facs)
    for row in pair_counts.itertuples(index=False):
        mat.loc[row.facility_num_x, row.facility_num_y] = row.jaccard_overlap
        mat.loc[row.facility_num_y, row.facility_num_x] = row.jaccard_overlap
    arr = mat.to_numpy(copy=True)
    np.fill_diagonal(arr, 1.0)
    mat = pd.DataFrame(arr, index=facs, columns=facs)
    labels = [f"{f}\n{lookup.get(f, '')[:12]}" for f in facs]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        mat,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Jaccard overlap"},
    )
    ax.set_title("Sensor pair overlap (Jaccard index)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_devices_per_facility(
    pair_counts: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
) -> None:
    """Unique devices per sensor (from pair table side counts)."""
    rows = []
    for fac in facilities["facility_num"].unique():
        sub_x = pair_counts.loc[pair_counts["facility_num_x"] == fac, "devices_facility_x"]
        sub_y = pair_counts.loc[pair_counts["facility_num_y"] == fac, "devices_facility_y"]
        val = None
        if len(sub_x):
            val = sub_x.iloc[0]
        elif len(sub_y):
            val = sub_y.iloc[0]
        if val is not None:
            name = facilities.loc[facilities["facility_num"] == fac, "facility_name"].iloc[0]
            rows.append({"facility_num": fac, "facility_name": name, "devices": val})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("devices", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r.facility_num}\n{str(r.facility_name)[:20]}" for r in df.itertuples()]
    ax.barh(labels, df["devices"], color="#3498db")
    ax.set_xlabel("Unique clean devices")
    ax.set_title("Devices detected per sensor (week)")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def generate_task2_plots(
    pair_counts: pd.DataFrame,
    summary: Dict[str, float],
    facilities: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    out = output_dir or (PLOTS_DIR / "task2")
    out.mkdir(parents=True, exist_ok=True)
    plot_multi_sensor_summary(summary, out / "multi_sensor_summary.png")
    plot_top_sensor_pairs(pair_counts, out / "top_sensor_pairs.png")
    plot_jaccard_heatmap(pair_counts, facilities, out / "jaccard_overlap_heatmap.png")
    plot_devices_per_facility(pair_counts, facilities, out / "devices_per_facility.png")
    try:
        from src.plots_mall_map import plot_task2_overlap_on_maps

        plot_task2_overlap_on_maps(
            pair_counts, facilities, out / "overlap_mall_map.png"
        )
    except FileNotFoundError as e:
        print(f"Task 2 mall map skipped: {e}")
    return out
