"""Task 4 advanced anomaly method comparison plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import PLOTS_DIR

METHOD_FLAG_COLUMNS: Dict[str, str] = {
    "baseline_zscore": "is_anomaly_hour",
    "baseline_hod": "is_anomaly_hod",
    "plan_b_consensus": "is_anomaly_consensus",
    "matrix_profile": "is_matrix_profile_anomaly",
    "stl": "is_stl_anomaly",
    "isolation_forest": "is_isolation_forest_anomaly",
    "markov_hour": "is_markov_hour_anomaly",
    "ensemble_consensus": "is_consensus_anomaly",
}


def _method_counts(comparison: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label, col in METHOD_FLAG_COLUMNS.items():
        if col in comparison.columns:
            counts[label] = int(comparison[col].fillna(False).astype(bool).sum())
    return counts


def plot_advanced_method_counts(
    comparison: pd.DataFrame,
    path: Path,
    *,
    title: str = "Task 4 — advanced anomaly methods",
) -> None:
    counts = _method_counts(comparison)
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    ax.bar(labels, values, color="#5b8def")
    ax.set_ylabel("Hours flagged")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_plan_ab_method_counts(
    comparison_a: pd.DataFrame,
    comparison_b: pd.DataFrame,
    path: Path,
) -> None:
    ca, cb = _method_counts(comparison_a), _method_counts(comparison_b)
    keys = sorted(set(ca) | set(cb))
    if not keys:
        return
    x = np.arange(len(keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width / 2, [ca.get(k, 0) for k in keys], width, label="Plan A footfall", color="#3498db")
    ax.bar(x + width / 2, [cb.get(k, 0) for k in keys], width, label="Plan B footfall", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=35, ha="right")
    ax.set_ylabel("Hours flagged")
    ax.set_title("Task 4 — anomaly methods (Plan A vs Plan B footfall)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_method_agreement_histogram(
    comparison: pd.DataFrame,
    path: Path,
    *,
    title: str = "Method agreement count",
) -> None:
    if "method_agreement_count" not in comparison.columns:
        return
    counts = comparison["method_agreement_count"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#9b59b6")
    ax.set_xlabel("Methods agreeing (boolean flags)")
    ax.set_ylabel("Sensor-hours")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_method_correlation_heatmap(
    comparison: pd.DataFrame,
    path: Path,
    *,
    title: str = "Anomaly method agreement (correlation)",
) -> None:
    flag_cols = [c for c in METHOD_FLAG_COLUMNS.values() if c in comparison.columns]
    if len(flag_cols) < 2:
        return
    flags = comparison[flag_cols].fillna(False).astype(int)
    corr = flags.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    short = [c.replace("is_", "").replace("_anomaly", "") for c in flag_cols]
    ax.set_xticks(range(len(flag_cols)))
    ax.set_yticks(range(len(flag_cols)))
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_top_ensemble_hours(
    comparison: pd.DataFrame,
    path: Path,
    *,
    top_n: int = 15,
    title: str = "Top consensus anomaly hours (ensemble score)",
) -> None:
    if "ensemble_score" not in comparison.columns:
        return
    top = comparison.nlargest(top_n, "ensemble_score")
    if top.empty:
        return
    labels = [
        f"{int(r.facility_num)} @ {pd.Timestamp(r.hour_start).strftime('%a %H:%M')}"
        for r in top.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, top["ensemble_score"].to_numpy(), color="#e74c3c")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Ensemble score (mean normalized method scores)")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def generate_task4_advanced_plots(
    comparison: pd.DataFrame,
    output_dir: Path | None = None,
    *,
    plan_label: str = "plan_a",
    comparison_other: Optional[pd.DataFrame] = None,
    comparison_other_label: str = "plan_b",
) -> Path:
    out = output_dir or (PLOTS_DIR / "task4")
    out.mkdir(parents=True, exist_ok=True)

    plot_advanced_method_counts(
        comparison,
        out / f"task4_advanced_method_counts_{plan_label}.png",
        title=f"Task 4 advanced methods — {plan_label}",
    )
    plot_method_agreement_histogram(
        comparison,
        out / f"task4_advanced_agreement_hist_{plan_label}.png",
        title=f"Method agreement — {plan_label}",
    )
    plot_method_correlation_heatmap(
        comparison,
        out / f"task4_advanced_method_correlation_{plan_label}.png",
        title=f"Method correlation — {plan_label}",
    )
    plot_top_ensemble_hours(
        comparison,
        out / f"task4_advanced_top_ensemble_{plan_label}.png",
        title=f"Top ensemble hours — {plan_label}",
    )

    if comparison_other is not None and plan_label == "plan_a":
        plot_plan_ab_method_counts(
            comparison,
            comparison_other,
            out / "task4_advanced_method_counts_plan_a_vs_b.png",
        )

    return out
