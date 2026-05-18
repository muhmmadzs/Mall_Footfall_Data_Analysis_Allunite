"""Plan B anomaly figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import OUTPUT_DIR


def plot_anomaly_method_counts(anomaly: pd.DataFrame, path: Path) -> None:
    cols = ["is_anomaly_global", "is_anomaly_robust", "is_anomaly_hod", "is_anomaly_consensus"]
    counts = {c.replace("is_anomaly_", ""): int(anomaly[c].sum()) for c in cols if c in anomaly.columns}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.keys(), counts.values(), color=["#95a5a6", "#3498db", "#2ecc71", "#e74c3c"])
    ax.set_ylabel("Anomaly hours flagged")
    ax.set_title("Plan B — anomaly detection methods")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_hod_vs_global_example(anomaly: pd.DataFrame, facility_num: int, path: Path) -> None:
    grp = anomaly.loc[anomaly["facility_num"] == facility_num].sort_values("hour_start")
    if grp.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    col = "footfall_plan_b" if "footfall_plan_b" in grp.columns else "estimated_total_footfall"
    ax.plot(grp["hour_start"], grp[col], label="Footfall (Plan B)", color="steelblue")
    hod_only = grp.loc[grp["is_anomaly_hod"] & ~grp["is_anomaly_global"]]
    both = grp.loc[grp["is_anomaly_hod"] & grp["is_anomaly_global"]]
    if len(both):
        ax.scatter(both["hour_start"], both[col], c="red", s=40, label="Both methods")
    if len(hod_only):
        ax.scatter(
            hod_only["hour_start"],
            hod_only[col],
            c="orange",
            s=40,
            label="HOD baseline only",
        )
    ax.set_title(f"Plan B anomalies — facility {facility_num}")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def generate_anomaly_plots(anomaly: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_anomaly_method_counts(anomaly, OUTPUT_DIR / "task4_anomaly_method_counts.png")
    plot_hod_vs_global_example(anomaly, 66330, OUTPUT_DIR / "task4_hod_vs_global_66330.png")
