"""Task 3 — journey paths and transition visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.paths import PLOTS_DIR


def plot_top_journeys(top_paths: pd.DataFrame, path: Path, n: int = 15) -> None:
    count_col = "journey_count" if "journey_count" in top_paths.columns else "device_count"
    top = top_paths.nlargest(n, count_col).copy()
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(top))))
    ax.barh(top["path"], top[count_col], color="#16a085")
    ax.set_xlabel("Journey sessions" if count_col == "journey_count" else "Unique devices")
    ax.set_title(f"Top {n} multi-sensor journey paths")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_transition_heatmap(
    transitions: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
    top_n: int = 9,
) -> None:
    """Heatmap of transition counts between sensors (from → to)."""
    facs = list(facilities["facility_num"].unique())[:top_n]
    if not len(facs):
        return
    sub = transitions.loc[
        transitions["facility_num"].isin(facs) & transitions["next_facility"].isin(facs)
    ]
    pivot = sub.pivot_table(
        index="facility_num",
        columns="next_facility",
        values="transition_count",
        aggfunc="sum",
        fill_value=0,
    )
    lookup = facilities.drop_duplicates("facility_num").set_index("facility_num")[
        "facility_name"
    ]
    pivot.index = [f"{i}\n{str(lookup.get(i, i))[:10]}" for i in pivot.index]
    pivot.columns = [f"{i}\n{str(lookup.get(i, i))[:10]}" for i in pivot.columns]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Transition count"},
    )
    ax.set_xlabel("To sensor")
    ax.set_ylabel("From sensor")
    ax.set_title("Facility-to-facility transitions (multi-sensor devices)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_journey_length_distribution(
    sample_paths: pd.DataFrame,
    path: Path,
) -> None:
    if sample_paths.empty or "unique_facilities" not in sample_paths.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = sample_paths["unique_facilities"].clip(upper=12)
    ax.hist(vals, bins=range(2, int(vals.max()) + 2), color="#8e44ad", edgecolor="white")
    ax.set_xlabel("Unique facilities visited per journey session")
    ax.set_ylabel("Journey sessions")
    ax.set_title("Journey breadth (sessionized multi-sensor paths)")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_transition_flow_top(
    transitions: pd.DataFrame,
    path: Path,
    n: int = 12,
) -> None:
    top = transitions.nlargest(n, "transition_count").copy()
    top["edge"] = top.apply(
        lambda r: f"{int(r.facility_num)} → {int(r.next_facility)}", axis=1
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["edge"], top["transition_count"], color="#2980b9")
    ax.set_xlabel("Transition count")
    ax.set_title(f"Top {n} directed sensor transitions")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def generate_task3_plots(
    top_paths: pd.DataFrame,
    transitions: pd.DataFrame,
    sample_paths: pd.DataFrame,
    facilities: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    out = output_dir or (PLOTS_DIR / "task3")
    out.mkdir(parents=True, exist_ok=True)
    plot_top_journeys(top_paths, out / "top_journey_paths.png")
    plot_transition_heatmap(transitions, facilities, out / "transition_heatmap.png")
    plot_transition_flow_top(transitions, out / "top_transitions.png")
    if not sample_paths.empty:
        plot_journey_length_distribution(sample_paths, out / "journey_breadth_histogram.png")
    try:
        from src.plots_mall_map import (
            plot_task3_top_paths_on_map,
            plot_task3_transitions_on_maps,
        )

        plot_task3_transitions_on_maps(
            transitions, facilities, out / "transitions_mall_map.png"
        )
        plot_task3_top_paths_on_map(top_paths, facilities, out / "top_paths_mall_map.png")
    except FileNotFoundError as e:
        print(f"Task 3 mall map skipped: {e}")
    return out
