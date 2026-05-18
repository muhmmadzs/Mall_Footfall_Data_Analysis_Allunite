"""Plan A vs Plan B comparison — two methods only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import OUTPUT_DIR, PLAN_A_OUTPUT


def load_plan_a_daily() -> pd.DataFrame:
    path = PLAN_A_OUTPUT / "task1_hourly_estimated_footfall.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["hour_start"])
    col = "footfall_plan_a" if "footfall_plan_a" in df.columns else "estimated_total_footfall"
    return (
        df.assign(date=lambda d: d["hour_start"].dt.date)
        .groupby("date", as_index=False)[col]
        .sum()
        .rename(columns={col: "footfall_plan_a"})
    )


def build_daily_comparison(plan_b_hourly: pd.DataFrame, mall_daily: pd.DataFrame) -> pd.DataFrame:
    b = (
        plan_b_hourly.assign(date=lambda d: d["hour_start"].dt.date)
        .groupby("date", as_index=False)
        .agg(
            footfall_plan_b=("footfall_plan_b", "sum"),
            footfall_plan_a_on_b_grid=("footfall_plan_a", "sum"),
        )
    )
    comp = b.copy()
    plan_a = load_plan_a_daily()
    if not plan_a.empty:
        comp = comp.merge(plan_a, on="date", how="left")
    else:
        comp["footfall_plan_a"] = comp["footfall_plan_a_on_b_grid"]

    if not mall_daily.empty and "estimated_mall_visitors" in mall_daily.columns:
        comp = comp.merge(
            mall_daily[["date", "estimated_mall_visitors", "unique_devices_mall"]],
            on="date",
            how="left",
        )
    comp["date"] = pd.to_datetime(comp["date"])
    return comp.sort_values("date")


def plot_daily_comparison(comp: pd.DataFrame, path: Path) -> None:
    if comp.empty:
        return
    fig, ax1 = plt.subplots(figsize=(12, 5))
    dates = comp["date"]
    x = range(len(comp))
    width = 0.35
    if "footfall_plan_a" in comp.columns:
        ax1.bar([i - width / 2 for i in x], comp["footfall_plan_a"], width, label="Plan A (sensor-hour)", color="#4a90d9")
    ax1.bar([i + width / 2 for i in x], comp["footfall_plan_b"], width, label="Plan B (sensor-hour)", color="#e67e22")
    ax1.set_ylabel("Daily footfall (sum of sensor-hours)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([d.strftime("%m-%d") for d in dates], rotation=45, ha="right")

    if "estimated_mall_visitors" in comp.columns:
        ax2 = ax1.twinx()
        ax2.plot(x, comp["estimated_mall_visitors"], "o-", color="#27ae60", linewidth=2, label="Mall visitors (Plan B)")
        ax2.set_ylabel("Estimated mall visitors (deduped devices × capture)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax1.legend(fontsize=8)

    ax1.set_title("Plan A vs Plan B — daily comparison")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def generate_comparison_outputs(
    plan_b_hourly: pd.DataFrame,
    mall_daily: pd.DataFrame,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comp = build_daily_comparison(plan_b_hourly, mall_daily)
    comp.to_csv(OUTPUT_DIR / "comparison_daily_footfall.csv", index=False)
    plot_daily_comparison(comp, OUTPUT_DIR / "comparison_daily_footfall.png")
    return comp
