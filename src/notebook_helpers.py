"""
Notebook helpers — Plan A (simple) and Plan B (sophisticated + mall visitors).

    from notebook_helpers import run_both_plans, load_comparison_daily, plot_comparison_interactive
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLAN_B_ROOT = REPO_ROOT / "plan_b"
PLAN_A_OUTPUT = REPO_ROOT / "outputs"
PLAN_B_OUTPUT = PLAN_B_ROOT / "outputs"

HOD_FACTOR_MIN = 0.5
HOD_FACTOR_MAX = 2.0


class _PlanBPath:
    def __enter__(self):
        self._saved_path = sys.path.copy()
        self._purged = [k for k in sys.modules if k == "src" or k.startswith("src.")]
        for k in self._purged:
            del sys.modules[k]
        sys.path.insert(0, str(PLAN_B_ROOT))
        return self

    def __exit__(self, *args):
        for k in self._purged:
            sys.modules.pop(k, None)
        sys.path[:] = self._saved_path


def setup_notebook_paths() -> Path:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    setup_notebook_paths()
    from src.load_data import load_all

    return load_all()


def run_plan_a(*, force_recompute: bool = False) -> Dict[str, Any]:
    setup_notebook_paths()
    out: Dict[str, Any] = {"plan": "A", "output_dir": PLAN_A_OUTPUT}
    path = PLAN_A_OUTPUT / "task1_hourly_estimated_footfall.csv"
    if not force_recompute and path.exists():
        out["hourly"] = pd.read_csv(path, parse_dates=["hour_start"])
        out["loaded_from_disk"] = True
        return out

    sessions, _, manuals = load_data()
    from src.task1 import run_task1

    cal_df, result, hourly = run_task1(sessions, manuals)
    PLAN_A_OUTPUT.mkdir(parents=True, exist_ok=True)
    cal_df.to_csv(PLAN_A_OUTPUT / "task1_calibration_windows.csv", index=False)
    hourly.to_csv(path, index=False)
    out["hourly"] = hourly
    out["calibration"] = cal_df
    out["task1_result"] = result
    out["loaded_from_disk"] = False
    return out


def run_plan_b(
    *,
    force_recompute: bool = False,
    hod_factor_min: float = HOD_FACTOR_MIN,
    hod_factor_max: float = HOD_FACTOR_MAX,
) -> Dict[str, Any]:
    setup_notebook_paths()
    out: Dict[str, Any] = {"plan": "B", "output_dir": PLAN_B_OUTPUT}
    if not force_recompute and (PLAN_B_OUTPUT / "task1_hourly_estimated_footfall.csv").exists():
        out["hourly"] = pd.read_csv(
            PLAN_B_OUTPUT / "task1_hourly_estimated_footfall.csv", parse_dates=["hour_start"]
        )
        out["mall_daily"] = pd.read_csv(PLAN_B_OUTPUT / "task1_mall_visitors_daily.csv")
        if (PLAN_B_OUTPUT / "task1_mall_visitors_hourly.csv").exists():
            out["mall_hourly"] = pd.read_csv(
                PLAN_B_OUTPUT / "task1_mall_visitors_hourly.csv", parse_dates=["hour_start"]
            )
        if (PLAN_B_OUTPUT / "comparison_daily_footfall.csv").exists():
            out["comparison_daily"] = pd.read_csv(PLAN_B_OUTPUT / "comparison_daily_footfall.csv")
        out["loaded_from_disk"] = True
        return out

    with _PlanBPath():
        import src.calibration as pb_cal
        from src.compare import generate_comparison_outputs
        from src.load_data import load_all as load_all_b
        from src.plots_anomalies import generate_anomaly_plots
        from src.task1 import run_task1_plan_b
        from src.task2 import run_task2
        from src.task4 import run_task4_plan_b

        pb_cal.HOD_FACTOR_MIN = hod_factor_min
        pb_cal.HOD_FACTOR_MAX = hod_factor_max

        sessions, facilities, manuals = load_all_b()
        _, t2 = run_task2(sessions, facilities)
        overlap = float(t2.get("multi_sensor_rate", 0.0))

        cal_df, models_df, loo, hourly, mall_daily, mall_hourly, capture_rates, meta = (
            run_task1_plan_b(sessions, facilities, manuals, overlap_rate=overlap)
        )
        suspicious, anomaly_hours, anomaly_summary, _ = run_task4_plan_b(sessions, hourly)

        PLAN_B_OUTPUT.mkdir(parents=True, exist_ok=True)
        cal_df.to_csv(PLAN_B_OUTPUT / "task1_calibration_windows.csv", index=False)
        models_df.to_csv(PLAN_B_OUTPUT / "task1_models_comparison.csv", index=False)
        loo.to_csv(PLAN_B_OUTPUT / "task1_leave_one_out_cv.csv", index=False)
        hourly.to_csv(PLAN_B_OUTPUT / "task1_hourly_estimated_footfall.csv", index=False)
        mall_daily.to_csv(PLAN_B_OUTPUT / "task1_mall_visitors_daily.csv", index=False)
        mall_hourly.to_csv(PLAN_B_OUTPUT / "task1_mall_visitors_hourly.csv", index=False)
        if not capture_rates.empty:
            capture_rates.to_csv(PLAN_B_OUTPUT / "task1_capture_rates_by_facility.csv", index=False)
        suspicious.head(200).to_csv(PLAN_B_OUTPUT / "task4_suspicious_unflagged_devices.csv", index=False)
        anomaly_hours.to_csv(PLAN_B_OUTPUT / "task4_hourly_anomalies.csv", index=False)
        with (PLAN_B_OUTPUT / "plan_b_meta.json").open("w") as f:
            json.dump(meta, f, indent=2, default=str)

        comp = generate_comparison_outputs(hourly, mall_daily)
        generate_anomaly_plots(anomaly_hours)

        out.update(
            {
                "hourly": hourly,
                "mall_daily": mall_daily,
                "mall_hourly": mall_hourly,
                "comparison_daily": comp,
                "meta": meta,
                "loaded_from_disk": False,
            }
        )
    return out


def run_both_plans(
    *,
    force_recompute: bool = False,
    hod_factor_min: float = HOD_FACTOR_MIN,
    hod_factor_max: float = HOD_FACTOR_MAX,
) -> Dict[str, Any]:
    a = run_plan_a(force_recompute=force_recompute)
    b = run_plan_b(
        force_recompute=force_recompute,
        hod_factor_min=hod_factor_min,
        hod_factor_max=hod_factor_max,
    )
    return {"plan_a": a, "plan_b": b, "comparison_daily": b.get("comparison_daily")}


def load_comparison_daily() -> pd.DataFrame:
    path = PLAN_B_OUTPUT / "comparison_daily_footfall.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_mall_visitors_daily() -> pd.DataFrame:
    path = PLAN_B_OUTPUT / "task1_mall_visitors_daily.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_mall_visitors_hourly() -> pd.DataFrame:
    path = PLAN_B_OUTPUT / "task1_mall_visitors_hourly.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["hour_start"])
    return pd.DataFrame()


def plot_comparison_interactive(comp: Optional[pd.DataFrame] = None):
    from src.plots_comparison import plot_comparison_interactive as _plot

    if comp is None:
        comp = load_comparison_daily()
    return _plot(comp)


def show_figure(output_dir: Path, filename: str):
    from IPython.display import Image, display

    path = output_dir / filename
    if path.exists():
        display(Image(filename=str(path)))
    else:
        print(f"Missing: {path}")
