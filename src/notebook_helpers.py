"""
Notebook helpers — single entry point for the full mall footfall pipeline.

    from notebook_helpers import run_full_analysis, plot_full_analysis

    results = run_full_analysis(force_recompute=True, write_outputs=False)
    plot_full_analysis(results, save_plots=True)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLAN_B_ROOT = REPO_ROOT / "plan_b"
PLAN_A_OUTPUT = REPO_ROOT / "outputs"
PLAN_B_OUTPUT = PLAN_B_ROOT / "outputs"  # legacy cache when write_outputs=True

from src.paths import PLOTS_DIR

HOD_FACTOR_MIN = 0.5
HOD_FACTOR_MAX = 2.0


def _reset_repo_src() -> None:
    """After Plan B imports, ensure repo-root ``src`` (with plots_task*) wins on sys.path."""
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    plan_b = str(PLAN_B_ROOT.resolve())
    repo = str(REPO_ROOT.resolve())
    cleaned = [p for p in sys.path if Path(p).resolve() != Path(plan_b)]
    if repo not in cleaned:
        cleaned.insert(0, repo)
    else:
        cleaned = [p for p in cleaned if Path(p).resolve() != Path(repo)]
        cleaned.insert(0, repo)
    sys.path[:] = cleaned


def _import_repo_src(module: str):
    """Load a module from ``REPO_ROOT/src/{module}.py`` (not plan_b/src)."""
    import importlib.util

    _reset_repo_src()
    path = REPO_ROOT / "src" / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"_repo_src.{module}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load repo module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        _reset_repo_src()


def setup_notebook_paths() -> Path:
    _reset_repo_src()
    return REPO_ROOT


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    setup_notebook_paths()
    from src.load_data import load_all

    return load_all()


def run_plan_a(
    *,
    force_recompute: bool = False,
    write_outputs: bool = False,
) -> Dict[str, Any]:
    setup_notebook_paths()
    out: Dict[str, Any] = {"plan": "A", "output_dir": PLAN_A_OUTPUT}
    path = PLAN_A_OUTPUT / "task1_hourly_estimated_footfall.csv"
    if not force_recompute and write_outputs and path.exists():
        out["hourly"] = pd.read_csv(path, parse_dates=["hour_start"])
        model_path = PLAN_A_OUTPUT / "task1_model_comparison.csv"
        loo_path = PLAN_A_OUTPUT / "task1_leave_one_out_cv.csv"
        boot_path = PLAN_A_OUTPUT / "task1_bootstrap_selected_model.csv"
        if model_path.exists():
            out["model_comparison"] = pd.read_csv(model_path)
        if loo_path.exists():
            out["leave_one_out"] = pd.read_csv(loo_path, parse_dates=["started"])
        if boot_path.exists():
            out["bootstrap"] = pd.read_csv(boot_path)
        out["loaded_from_disk"] = True
        return out

    sessions, facilities, manuals = load_data()
    from src.task1 import run_task1

    cal_df, result, hourly = run_task1(sessions, manuals, facilities)
    if write_outputs:
        PLAN_A_OUTPUT.mkdir(parents=True, exist_ok=True)
        cal_df.to_csv(PLAN_A_OUTPUT / "task1_calibration_windows.csv", index=False)
        hourly.to_csv(path, index=False)
        if not result.model_comparison.empty:
            result.model_comparison.to_csv(
                PLAN_A_OUTPUT / "task1_model_comparison.csv", index=False
            )
        if not result.leave_one_out.empty:
            result.leave_one_out.to_csv(
                PLAN_A_OUTPUT / "task1_leave_one_out_cv.csv", index=False
            )
        if not result.bootstrap.empty:
            result.bootstrap.to_csv(
                PLAN_A_OUTPUT / "task1_bootstrap_selected_model.csv", index=False
            )
    out["hourly"] = hourly
    out["calibration"] = cal_df
    out["task1_result"] = result
    out["model_comparison"] = result.model_comparison
    out["leave_one_out"] = result.leave_one_out
    out["bootstrap"] = result.bootstrap
    out["loaded_from_disk"] = False
    return out


def tune_plan_b_parameters(
    *,
    method: str = "bayesian",
    quick: bool = True,
    n_calls: int = 60,
    n_initial_points: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Tune Plan B HOD + capture parameters.

    method: ``"bayesian"`` (Gaussian-process, default) or ``"grid"`` (exhaustive grid).
    """
    setup_notebook_paths()
    with _PlanBPath():
        from src.tune_plan_b import tune_plan_b

        sessions, facilities, manuals = load_data()
        hod_params, capture_method, history, best_eval = tune_plan_b(
            sessions,
            facilities,
            manuals,
            method=method,
            quick=quick,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
        )
    return {
        "hod_params": hod_params.__dict__,
        "capture_method": capture_method,
        "evaluation": best_eval,
        "history": history,
        "optimizer": best_eval.get("optimizer", method),
    }


def run_plan_b(
    *,
    force_recompute: bool = False,
    write_outputs: bool = False,
    hod_factor_min: float = HOD_FACTOR_MIN,
    hod_factor_max: float = HOD_FACTOR_MAX,
    hod_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    setup_notebook_paths()
    out: Dict[str, Any] = {"plan": "B", "output_dir": PLAN_B_OUTPUT}
    hourly_path = PLAN_B_OUTPUT / "task1_hourly_estimated_footfall.csv"
    if not force_recompute and write_outputs and hourly_path.exists():
        out["hourly"] = pd.read_csv(hourly_path, parse_dates=["hour_start"])
        out["mall_daily"] = pd.read_csv(PLAN_B_OUTPUT / "task1_mall_visitors_daily.csv")
        if (PLAN_B_OUTPUT / "task1_mall_visitors_hourly.csv").exists():
            out["mall_hourly"] = pd.read_csv(
                PLAN_B_OUTPUT / "task1_mall_visitors_hourly.csv", parse_dates=["hour_start"]
            )
        out["comparison_daily"] = pd.read_csv(
            PLAN_B_OUTPUT / "comparison_daily_footfall.csv"
        ) if (PLAN_B_OUTPUT / "comparison_daily_footfall.csv").exists() else pd.DataFrame()
        out["loaded_from_disk"] = True
        return out

    with _PlanBPath():
        import src.calibration as pb_cal
        from src.compare import build_daily_comparison
        from src.load_data import load_all as load_all_b
        from src.task1 import run_task1_plan_b
        from src.task2 import run_task2
        from src.task4 import run_task4_plan_b

        pb_cal.HOD_FACTOR_MIN = hod_factor_min
        pb_cal.HOD_FACTOR_MAX = hod_factor_max

        sessions, facilities, manuals = load_all_b()
        _, t2 = run_task2(sessions, facilities)
        overlap = float(t2.get("multi_sensor_rate", 0.0))

        cal_df, models_df, loo, hourly, mall_daily, mall_hourly, capture_rates, meta = (
            run_task1_plan_b(
                sessions,
                facilities,
                manuals,
                overlap_rate=overlap,
                hod_params_override=hod_params,
            )
        )
        suspicious, anomaly_hours, anomaly_summary, device_stats = run_task4_plan_b(
            sessions, hourly
        )
        comp = build_daily_comparison(hourly, mall_daily)

        if write_outputs:
            PLAN_B_OUTPUT.mkdir(parents=True, exist_ok=True)
            hourly.to_csv(hourly_path, index=False)
            mall_daily.to_csv(PLAN_B_OUTPUT / "task1_mall_visitors_daily.csv", index=False)

        out.update(
            {
                "hourly": hourly,
                "mall_daily": mall_daily,
                "mall_hourly": mall_hourly,
                "comparison_daily": comp,
                "calibration": cal_df,
                "models": models_df,
                "loo": loo,
                "capture_rates": capture_rates,
                "meta": meta,
                "suspicious": suspicious,
                "anomaly_hours": anomaly_hours,
                "anomaly_summary": anomaly_summary,
                "device_stats": device_stats,
                "facilities": facilities,
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


def load_comparison_daily(results: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if results is not None:
        comp = results.get("comparison_daily")
        if comp is not None and len(comp):
            return comp
        pb = results.get("plan_b", {})
        if isinstance(pb, dict) and len(pb.get("comparison_daily", pd.DataFrame())):
            return pb["comparison_daily"]
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


def load_plan_b_hourly() -> pd.DataFrame:
    path = PLAN_B_OUTPUT / "task1_hourly_estimated_footfall.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["hour_start"])
    return pd.DataFrame()


def load_facilities() -> pd.DataFrame:
    setup_notebook_paths()
    from src.load_data import load_all

    _, facilities, _ = load_all()
    return facilities


def plot_comparison_interactive(comp: Optional[pd.DataFrame] = None):
    _plot = _import_repo_src("plots_comparison").plot_comparison_interactive

    if comp is None:
        comp = load_comparison_daily()
    return _plot(comp)


def plot_footfall_dashboard_interactive(
    comp: Optional[pd.DataFrame] = None,
    hourly: Optional[pd.DataFrame] = None,
    mall_hourly: Optional[pd.DataFrame] = None,
    facilities: Optional[pd.DataFrame] = None,
):
    """All interactive charts: daily, mall hourly, every location grid + heatmap + dropdown."""
    _dash = _import_repo_src("plots_comparison").plot_footfall_dashboard_interactive

    if comp is None:
        comp = load_comparison_daily()
    if hourly is None:
        hourly = load_plan_b_hourly()
    if mall_hourly is None:
        mall_hourly = load_mall_visitors_hourly()
    if facilities is None:
        facilities = load_facilities()
    return _dash(comp, hourly, mall_hourly, facilities)


def prepare_task1_abc(plan_a: Dict[str, Any], plan_b: Dict[str, Any]) -> Dict[str, Any]:
    """Build unified Plan A/B/C tables for Task 1 dashboards."""
    hourly = plan_a["hourly"].copy()
    hourly_b = plan_b["hourly"].copy()
    cal = plan_a["calibration"].copy()

    hourly["hour_start"] = pd.to_datetime(hourly["hour_start"], utc=True).dt.tz_localize(None)
    hourly_b["hour_start"] = pd.to_datetime(hourly_b["hour_start"], utc=True).dt.tz_localize(None)
    cal["started"] = pd.to_datetime(cal["started"], utc=True).dt.tz_localize(None)

    hourly["date"] = hourly["hour_start"].dt.date
    hourly_b["date"] = hourly_b["hour_start"].dt.date

    dashboard_hourly = hourly[[
        "facility_num",
        "hour_start",
        "date",
        "footfall_trusted_linear",
        "footfall_plan_c",
    ]].merge(
        hourly_b[["facility_num", "hour_start", "footfall_plan_b"]],
        on=["facility_num", "hour_start"],
        how="left",
    ).rename(columns={
        "footfall_trusted_linear": "plan_a_estimate",
        "footfall_plan_b": "plan_b_estimate",
        "footfall_plan_c": "plan_c_estimate",
    })

    daily_compare = (
        dashboard_hourly.groupby("date", as_index=False)[
            ["plan_a_estimate", "plan_b_estimate", "plan_c_estimate"]
        ]
        .sum()
        .sort_values("date")
    )

    daily_long = daily_compare.rename(columns={
        "plan_a_estimate": "Plan A",
        "plan_b_estimate": "Plan B",
        "plan_c_estimate": "Plan C",
    }).melt(
        id_vars="date",
        value_vars=["Plan A", "Plan B", "Plan C"],
        var_name="plan",
        value_name="daily_footfall",
    )

    facility_weekly = (
        dashboard_hourly.groupby("facility_num", as_index=False)[
            ["plan_a_estimate", "plan_b_estimate", "plan_c_estimate"]
        ]
        .sum()
        .sort_values("plan_c_estimate", ascending=False)
    )

    facility_weekly_long = facility_weekly.rename(columns={
        "plan_a_estimate": "Plan A",
        "plan_b_estimate": "Plan B",
        "plan_c_estimate": "Plan C",
    }).melt(
        id_vars="facility_num",
        value_vars=["Plan A", "Plan B", "Plan C"],
        var_name="plan",
        value_name="weekly_footfall",
    )

    location_daily = (
        dashboard_hourly.groupby(["date", "facility_num"], as_index=False)[
            ["plan_a_estimate", "plan_b_estimate", "plan_c_estimate"]
        ]
        .sum()
        .sort_values(["facility_num", "date"])
    )

    location_daily_long = location_daily.rename(columns={
        "plan_a_estimate": "Plan A",
        "plan_b_estimate": "Plan B",
        "plan_c_estimate": "Plan C",
    }).melt(
        id_vars=["date", "facility_num"],
        value_vars=["Plan A", "Plan B", "Plan C"],
        var_name="plan",
        value_name="daily_footfall",
    )

    plan_b_calibration = pd.DataFrame(plan_b.get("meta", {}).get("hod_v2_calibration_validation", []))
    if plan_b_calibration.empty:
        hod_path = PLAN_B_OUTPUT / "hod_v2_calibration_validation.csv"
        if hod_path.exists():
            plan_b_calibration = pd.read_csv(hod_path)
    if not plan_b_calibration.empty:
        plan_b_calibration["started"] = pd.to_datetime(
            plan_b_calibration["started"], utc=True
        ).dt.tz_localize(None)

    calibration_compare = cal[[
        "facility_num",
        "started",
        "manual_total_count",
        "pred_linear_trusted_unique_devices",
        "pred_plan_c_trend_preserving",
    ]].rename(columns={
        "pred_linear_trusted_unique_devices": "plan_a_pred",
        "pred_plan_c_trend_preserving": "plan_c_pred",
    })

    if not plan_b_calibration.empty:
        calibration_compare = calibration_compare.merge(
            plan_b_calibration[["facility_num", "started", "predicted_plan_b_v2"]],
            on=["facility_num", "started"],
            how="left",
        ).rename(columns={"predicted_plan_b_v2": "plan_b_pred"})
    else:
        calibration_compare["plan_b_pred"] = np.nan

    calibration_compare["window_label"] = (
        calibration_compare["facility_num"].astype(str)
        + " | "
        + calibration_compare["started"].dt.strftime("%Y-%m-%d %H:%M")
    )

    calibration_long = calibration_compare.rename(columns={
        "manual_total_count": "Manual actual",
        "plan_a_pred": "Plan A",
        "plan_b_pred": "Plan B",
        "plan_c_pred": "Plan C",
    }).melt(
        id_vars="window_label",
        value_vars=["Manual actual", "Plan A", "Plan B", "Plan C"],
        var_name="series",
        value_name="count",
    )

    mall_daily = plan_b.get("mall_daily", pd.DataFrame())
    mall_hourly = plan_b.get("mall_hourly", pd.DataFrame())

    overall_summary = pd.DataFrame([
        {
            "plan": "Plan A",
            "weekly_footfall": daily_compare["plan_a_estimate"].sum(),
            "daily_average": daily_compare["plan_a_estimate"].mean(),
        },
        {
            "plan": "Plan B",
            "weekly_footfall": daily_compare["plan_b_estimate"].sum(),
            "daily_average": daily_compare["plan_b_estimate"].mean(),
        },
        {
            "plan": "Plan C",
            "weekly_footfall": daily_compare["plan_c_estimate"].sum(),
            "daily_average": daily_compare["plan_c_estimate"].mean(),
        },
    ])

    mall_visitors_weekly = np.nan
    if not mall_daily.empty and "estimated_mall_visitors" in mall_daily.columns:
        mall_visitors_weekly = float(mall_daily["estimated_mall_visitors"].sum())

    return {
        "dashboard_hourly": dashboard_hourly,
        "daily_compare": daily_compare,
        "daily_long": daily_long,
        "facility_weekly": facility_weekly,
        "facility_weekly_long": facility_weekly_long,
        "location_daily": location_daily,
        "location_daily_long": location_daily_long,
        "calibration_compare": calibration_compare,
        "calibration_long": calibration_long,
        "mall_daily": mall_daily,
        "mall_hourly": mall_hourly,
        "overall_summary": overall_summary,
        "mall_visitors_weekly": mall_visitors_weekly,
        "task1_result": plan_a.get("task1_result"),
        "model_comparison": plan_a.get("model_comparison", pd.DataFrame()),
    }


def plot_task1_abc_interactive(
    task1_abc: Optional[Dict[str, Any]] = None,
    *,
    plan_a: Optional[Dict[str, Any]] = None,
    plan_b: Optional[Dict[str, Any]] = None,
    facilities: Optional[pd.DataFrame] = None,
):
    """Interactive Task 1 charts for Plan A, Plan B, and Plan C."""
    plot_abc = _import_repo_src("plots_comparison").plot_task1_abc_dashboard

    if task1_abc is None:
        if plan_a is None or plan_b is None:
            raise ValueError("Provide task1_abc or both plan_a and plan_b")
        task1_abc = prepare_task1_abc(plan_a, plan_b)

    if facilities is None:
        facilities = load_facilities()

    return plot_abc(
        task1_abc["daily_long"],
        task1_abc["facility_weekly_long"],
        task1_abc["location_daily_long"],
        task1_abc["calibration_long"],
        mall_daily=task1_abc.get("mall_daily"),
        mall_hourly=task1_abc.get("mall_hourly"),
        facilities=facilities,
    )


def show_figure(output_dir: Path, filename: str):
    from IPython.display import Image, display

    path = output_dir / filename
    if path.exists():
        display(Image(filename=str(path)))
    else:
        print(f"Missing: {path}")


def plots_dir(task: int, plan: str = "a") -> Path:
    """Directory of PNG plots for a task (plan_a or plan_b for task 1 only)."""
    if task == 1 and plan.lower() == "b":
        return PLOTS_DIR / "task1" / "plan_b"
    if task == 1:
        return PLOTS_DIR / "task1" / "plan_a"
    return PLOTS_DIR / f"task{task}"


def show_task_plots(task: int, plan: str = "a") -> None:
    """Display all PNGs under outputs/plots/task{N}/."""
    from IPython.display import Image, display

    d = plots_dir(task, plan)
    if not d.exists():
        print(f"No plots at {d}. Run plot_task{task}() first.")
        return
    files = sorted(d.glob("*.png"))
    if not files:
        print(f"No PNG files in {d}")
        return
    print(f"{len(files)} plots in {d}")
    for p in files:
        print(p.name)
        display(Image(filename=str(p)))


def run_task1(*, force_recompute: bool = False, plan: str = "both") -> Dict[str, Any]:
    """Task 1 — calibration & footfall (Plan A and/or Plan B)."""
    out: Dict[str, Any] = {}
    if plan in ("a", "both"):
        out["plan_a"] = run_plan_a(force_recompute=force_recompute)
    if plan in ("b", "both"):
        out["plan_b"] = run_plan_b(force_recompute=force_recompute)
    return out


def plot_task1(plan: str = "both") -> Dict[str, Path]:
    """Generate static task 1 plots."""
    _reset_repo_src()
    plots1 = _import_repo_src("plots_task1")
    generate_task1_plots_plan_a = plots1.generate_task1_plots_plan_a
    generate_task1_plots_plan_b = plots1.generate_task1_plots_plan_b

    sessions, facilities, manuals = load_data()
    dirs: Dict[str, Path] = {}
    if plan in ("a", "both"):
        hourly_path = PLAN_A_OUTPUT / "task1_hourly_estimated_footfall.csv"
        if hourly_path.exists():
            hourly = pd.read_csv(hourly_path, parse_dates=["hour_start"])
        else:
            from src.task1 import run_task1

            cal_df, _, hourly = run_task1(sessions, manuals, facilities)
        from src.task2 import run_task2 as _run_task2

        pair_counts, summary = _run_task2(sessions, facilities)
        _, _, d = generate_task1_plots_plan_a(
            sessions,
            facilities,
            manuals,
            hourly,
            multi_sensor_rate=summary.get("multi_sensor_rate"),
        )
        dirs["plan_a"] = d
    if plan in ("b", "both"):
        b = run_plan_b(force_recompute=False)
        hod_val = pd.DataFrame()
        hv = PLAN_B_OUTPUT / "hod_v2_calibration_validation.csv"
        if hv.exists():
            hod_val = pd.read_csv(hv)
        cal_path = PLAN_B_OUTPUT / "task1_calibration_windows.csv"
        cal_df = pd.read_csv(cal_path) if cal_path.exists() else pd.DataFrame()
        dirs["plan_b"] = generate_task1_plots_plan_b(
            sessions,
            facilities,
            b["hourly"],
            b.get("mall_daily", pd.DataFrame()),
            cal_df,
            b.get("comparison_daily", load_comparison_daily()),
            hod_validation=hod_val,
        )
    return dirs


def run_task2(
    *,
    force_recompute: bool = False,
    write_outputs: bool = False,
    sessions: Optional[pd.DataFrame] = None,
    facilities: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    if sessions is None or facilities is None:
        sessions, facilities, _ = load_data()
    pair_path = PLAN_A_OUTPUT / "task2_sensor_overlap_pairs.csv"
    if not force_recompute and write_outputs and pair_path.exists():
        pair_counts = pd.read_csv(pair_path)
        summary: Dict[str, Any] = {}
        summary_path = PLAN_A_OUTPUT / "analysis_summary.json"
        if summary_path.exists():
            with summary_path.open() as f:
                summary = json.load(f).get("task2", {})
        return {"pair_counts": pair_counts, "summary": summary, "loaded_from_disk": True}
    from src.task2 import run_task2 as _run

    pair_counts, summary = _run(sessions, facilities)
    if write_outputs:
        PLAN_A_OUTPUT.mkdir(parents=True, exist_ok=True)
        pair_counts.to_csv(pair_path, index=False)
    return {"pair_counts": pair_counts, "summary": summary, "loaded_from_disk": False}


def plot_task2(results: Optional[Dict[str, Any]] = None, *, save_plots: bool = True) -> Path:
    generate_task2_plots = _import_repo_src("plots_task2").generate_task2_plots

    t2 = results.get("task2") if results else run_task2()
    sessions, facilities, _ = load_data()
    if not save_plots:
        return PLOTS_DIR / "task2"
    return generate_task2_plots(t2["pair_counts"], t2["summary"], facilities, PLOTS_DIR / "task2")


def run_task3(
    *,
    force_recompute: bool = False,
    write_outputs: bool = False,
    sessions: Optional[pd.DataFrame] = None,
    facilities: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    if sessions is None or facilities is None:
        sessions, facilities, _ = load_data()
    top_path = PLAN_A_OUTPUT / "task3_top_journeys.csv"
    if not force_recompute and write_outputs and top_path.exists():
        return {
            "top_paths": pd.read_csv(top_path),
            "transitions": pd.read_csv(PLAN_A_OUTPUT / "task3_transition_matrix.csv"),
            "sample_paths": pd.read_csv(PLAN_A_OUTPUT / "task3_sample_device_journeys.csv"),
            "loaded_from_disk": True,
        }
    from src.task3 import run_task3 as _run

    top_paths, transitions, sample_paths, journey_devices = _run(sessions, facilities)
    if write_outputs:
        PLAN_A_OUTPUT.mkdir(parents=True, exist_ok=True)
        top_paths.head(100).to_csv(top_path, index=False)
        transitions.to_csv(PLAN_A_OUTPUT / "task3_transition_matrix.csv", index=False)
    return {
        "top_paths": top_paths,
        "transitions": transitions,
        "sample_paths": sample_paths,
        "journey_devices": journey_devices,
        "loaded_from_disk": False,
    }


def plot_task3(results: Optional[Dict[str, Any]] = None, *, save_plots: bool = True) -> Path:
    generate_task3_plots = _import_repo_src("plots_task3").generate_task3_plots

    t3 = results.get("task3") if results else run_task3()
    _, facilities, _ = load_data()
    if not save_plots:
        return PLOTS_DIR / "task3"
    return generate_task3_plots(
        t3["top_paths"],
        t3["transitions"],
        t3["sample_paths"],
        facilities,
        PLOTS_DIR / "task3",
    )


def run_task4(
    *,
    force_recompute: bool = False,
    plan: str = "b",
    results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Task 4 — anomalies.

    ``plan="c"`` uses the delivered Plan C estimate from the root Task 1
    pipeline (stored as ``estimated_total_footfall``). ``plan="a"`` is kept as
    a backwards-compatible alias for the same root-pipeline anomaly output.
    """
    normalized_plan = plan.lower()
    if results is not None:
        key = "task4_b" if normalized_plan == "b" else "task4_a"
        label = "B" if normalized_plan == "b" else "C"
        return {**results[key], "plan": label}

    sessions, _, _ = load_data()
    if normalized_plan == "b":
        pb = run_plan_b(force_recompute=force_recompute, write_outputs=False)
        return {
            "suspicious": pb["suspicious"],
            "anomaly_hours": pb["anomaly_hours"],
            "anomaly_summary": pb["anomaly_summary"],
            "device_stats": pb["device_stats"],
            "plan": "B",
        }
    pa = run_plan_a(force_recompute=force_recompute, write_outputs=False)
    from src.task4 import run_task4 as _run

    suspicious, anomaly_hours, anomaly_summary, device_stats = _run(sessions, pa["hourly"])
    return {
        "suspicious": suspicious,
        "anomaly_hours": anomaly_hours,
        "anomaly_summary": anomaly_summary,
        "device_stats": device_stats,
        "plan": "C",
    }


def plot_task4(
    plan: str = "b",
    results: Optional[Dict[str, Any]] = None,
    *,
    save_plots: bool = True,
) -> Path:
    generate_task4_plots = _import_repo_src("plots_task4").generate_task4_plots
    normalized_plan = plan.lower()

    if not save_plots:
        return PLOTS_DIR / "task4"
    if results is None:
        t4 = run_task4(force_recompute=False, plan=normalized_plan)
        sessions, facilities, _ = load_data()
    else:
        key = "task4_b" if normalized_plan == "b" else "task4_a"
        t4 = results[key]
        sessions, facilities = results["sessions"], results["facilities"]
    return generate_task4_plots(
        sessions,
        facilities,
        t4["device_stats"],
        t4["suspicious"],
        t4["anomaly_hours"],
        t4["anomaly_summary"],
        PLOTS_DIR / "task4",
    )


def run_task4_advanced(
    *,
    force_recompute: bool = False,
    plan: str = "both",
    write_outputs: bool = False,
    results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Advanced Task 4 methods (matrix profile, STL, isolation forest, Markov, ensemble)."""
    run_advanced = _import_repo_src("task4_advanced").run_advanced_anomaly_pipeline
    sessions, _, _ = load_data()
    out: Dict[str, Any] = {}

    if results is not None:
        hourly_a = results["plan_a"]["hourly"]
        anomaly_a = results["task4_a"]["anomaly_hours"]
        hourly_b = results["plan_b"]["hourly"]
        anomaly_b = results["task4_b"]["anomaly_hours"]
        if plan in ("a", "both"):
            out["plan_a"] = run_advanced(
                sessions,
                hourly_a,
                anomaly_a,
                footfall_col="estimated_total_footfall",
                plan_name="plan_a",
                output_dir=PLAN_A_OUTPUT,
                write_outputs=write_outputs,
            )
        if plan in ("b", "both"):
            out["plan_b"] = run_advanced(
                sessions,
                hourly_b,
                anomaly_b,
                footfall_col="footfall_plan_b",
                plan_name="plan_b",
                output_dir=PLAN_A_OUTPUT,
                write_outputs=write_outputs,
            )
        return out

    if plan in ("a", "both"):
        pa = run_plan_a(force_recompute=force_recompute, write_outputs=write_outputs)
        t4 = run_task4(force_recompute=force_recompute, plan="a")
        hourly = pa["hourly"]
        out["plan_a"] = run_advanced(
            sessions,
            hourly,
            t4["anomaly_hours"],
            footfall_col="estimated_total_footfall",
            plan_name="plan_a",
            output_dir=PLAN_A_OUTPUT,
            write_outputs=write_outputs,
        )
    if plan in ("b", "both"):
        pb = run_plan_b(force_recompute=force_recompute, write_outputs=write_outputs)
        t4 = run_task4(force_recompute=force_recompute, plan="b")
        out["plan_b"] = run_advanced(
            sessions,
            pb["hourly"],
            t4["anomaly_hours"],
            footfall_col="footfall_plan_b",
            plan_name="plan_b",
            output_dir=PLAN_A_OUTPUT,
            write_outputs=write_outputs,
        )
    return out


def plot_task4_advanced(
    plan: str = "both",
    results: Optional[Dict[str, Any]] = None,
    *,
    save_plots: bool = True,
) -> Path:
    """Plots for advanced anomaly method comparison."""
    if not save_plots:
        return PLOTS_DIR / "task4"
    generate = _import_repo_src("plots_task4_advanced").generate_task4_advanced_plots
    comp_a = comp_b = None
    if results and "task4_advanced" in results:
        adv = results["task4_advanced"]
        comp_a = adv.get("plan_a", {}).get("comparison")
        comp_b = adv.get("plan_b", {}).get("comparison")
    if comp_a is None and plan in ("a", "both"):
        p = PLAN_A_OUTPUT / "task4_anomaly_method_comparison_plan_a.csv"
        if p.exists():
            comp_a = pd.read_csv(p, parse_dates=["hour_start"])
    if comp_b is None and plan in ("b", "both"):
        p = PLAN_A_OUTPUT / "task4_anomaly_method_comparison_plan_b.csv"
        if p.exists():
            comp_b = pd.read_csv(p, parse_dates=["hour_start"])
    if comp_a is not None:
        generate(
            comp_a,
            PLOTS_DIR / "task4",
            plan_label="plan_a",
            comparison_other=comp_b if plan == "both" else None,
        )
    if comp_b is not None:
        generate(comp_b, PLOTS_DIR / "task4", plan_label="plan_b")
    return PLOTS_DIR / "task4"


def run_full_analysis(
    *,
    force_recompute: bool = True,
    write_outputs: bool = False,
    hod_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Tasks 1–4 (Plan A + B) and advanced anomalies in one pass (in-memory by default)."""
    setup_notebook_paths()
    sessions, facilities, manuals = load_data()

    from src.task1 import run_task1
    from src.task4 import run_task4 as run_task4_a

    cal_a, res_a, hourly_a = run_task1(sessions, manuals, facilities)
    t2 = run_task2(
        force_recompute=force_recompute,
        write_outputs=write_outputs,
        sessions=sessions,
        facilities=facilities,
    )
    t3 = run_task3(
        force_recompute=force_recompute,
        write_outputs=write_outputs,
        sessions=sessions,
        facilities=facilities,
    )
    plan_b = run_plan_b(
        force_recompute=force_recompute,
        write_outputs=write_outputs,
        hod_params=hod_params,
    )
    hourly_b = plan_b["hourly"]

    suspicious_a, anomaly_a, summary_a, dev_a = run_task4_a(sessions, hourly_a)
    suspicious_b = plan_b["suspicious"]
    anomaly_b = plan_b["anomaly_hours"]
    summary_b = plan_b["anomaly_summary"]
    dev_b = plan_b["device_stats"]

    run_advanced = _import_repo_src("task4_advanced").run_advanced_anomaly_pipeline
    adv_a = run_advanced(
        sessions,
        hourly_a,
        anomaly_a,
        footfall_col="estimated_total_footfall",
        plan_name="plan_a",
        write_outputs=write_outputs,
    )
    adv_b = run_advanced(
        sessions,
        hourly_b,
        anomaly_b,
        footfall_col="footfall_plan_b",
        plan_name="plan_b",
        write_outputs=write_outputs,
    )

    if write_outputs:
        PLAN_A_OUTPUT.mkdir(parents=True, exist_ok=True)
        hourly_a.to_csv(PLAN_A_OUTPUT / "task1_hourly_estimated_footfall.csv", index=False)
        cal_a.to_csv(PLAN_A_OUTPUT / "task1_calibration_windows.csv", index=False)
        if not res_a.model_comparison.empty:
            res_a.model_comparison.to_csv(
                PLAN_A_OUTPUT / "task1_model_comparison.csv", index=False
            )
        if not res_a.leave_one_out.empty:
            res_a.leave_one_out.to_csv(
                PLAN_A_OUTPUT / "task1_leave_one_out_cv.csv", index=False
            )
        if not res_a.bootstrap.empty:
            res_a.bootstrap.to_csv(
                PLAN_A_OUTPUT / "task1_bootstrap_selected_model.csv", index=False
            )

    return {
        "sessions": sessions,
        "facilities": facilities,
        "manuals": manuals,
        "plan_a": {
            "hourly": hourly_a,
            "calibration": cal_a,
            "task1_result": res_a,
            "model_comparison": res_a.model_comparison,
            "leave_one_out": res_a.leave_one_out,
            "bootstrap": res_a.bootstrap,
        },
        "plan_b": plan_b,
        "comparison_daily": plan_b["comparison_daily"],
        "task2": t2,
        "task3": t3,
        "task4_a": {
            "suspicious": suspicious_a,
            "anomaly_hours": anomaly_a,
            "anomaly_summary": summary_a,
            "device_stats": dev_a,
        },
        "task4_b": {
            "suspicious": suspicious_b,
            "anomaly_hours": anomaly_b,
            "anomaly_summary": summary_b,
            "device_stats": dev_b,
        },
        "task4_advanced": {"plan_a": adv_a, "plan_b": adv_b},
    }


def plot_full_analysis(
    results: Dict[str, Any],
    *,
    save_plots: bool = True,
    plan: str = "both",
) -> None:
    """Generate all static plots under outputs/plots/ from run_full_analysis() results."""
    if not save_plots:
        return
    _reset_repo_src()
    plots1 = _import_repo_src("plots_task1")
    sessions = results["sessions"]
    facilities = results["facilities"]
    manuals = results["manuals"]
    hourly_a = results["plan_a"]["hourly"]
    hourly_b = results["plan_b"]["hourly"]

    if plan in ("a", "both"):
        plots1.generate_task1_plots_plan_a(
            sessions,
            facilities,
            manuals,
            hourly_a,
            multi_sensor_rate=results["task2"]["summary"].get("multi_sensor_rate"),
            output_dir=PLOTS_DIR / "task1" / "plan_a",
        )
    if plan in ("b", "both"):
        plots1.generate_task1_plots_plan_b(
            sessions,
            facilities,
            hourly_b,
            results["plan_b"]["mall_daily"],
            results["plan_b"].get("calibration", pd.DataFrame()),
            results["comparison_daily"],
            output_dir=PLOTS_DIR / "task1" / "plan_b",
        )
    plot_task2(results, save_plots=True)
    plot_task3(results, save_plots=True)

    gen4 = _import_repo_src("plots_task4").generate_task4_plots
    gen4(
        sessions,
        facilities,
        results["task4_b"]["device_stats"],
        results["task4_b"]["suspicious"],
        results["task4_b"]["anomaly_hours"],
        results["task4_b"]["anomaly_summary"],
        PLOTS_DIR / "task4",
    )
    plot_task4_advanced(plan="both", results=results, save_plots=True)

    try:
        _import_repo_src("plots_mall_map").generate_mall_map_plots(
            results["task2"]["pair_counts"],
            results["task3"]["transitions"],
            results["task3"]["top_paths"],
            facilities,
            PLOTS_DIR,
        )
    except FileNotFoundError as e:
        print(f"Mall map plots skipped: {e}")


def run_all_tasks(
    *,
    force_recompute: bool = True,
    write_outputs: bool = False,
) -> Dict[str, Any]:
    return run_full_analysis(force_recompute=force_recompute, write_outputs=write_outputs)


def plot_mall_maps(results: Optional[Dict[str, Any]] = None) -> Path:
    """Task 2 & 3 overlays on assets/1.png and assets/2.png."""
    if results is None:
        results = {"task2": run_task2(), "task3": run_task3()}
    _, facilities, _ = load_data()
    return _import_repo_src("plots_mall_map").generate_mall_map_plots(
        results["task2"]["pair_counts"],
        results["task3"]["transitions"],
        results["task3"]["top_paths"],
        facilities,
        PLOTS_DIR,
    )


def plot_all_tasks(
    results: Optional[Dict[str, Any]] = None,
    *,
    plan: str = "both",
    save_plots: bool = True,
) -> None:
    if results is None:
        results = run_full_analysis(force_recompute=False, write_outputs=False)
    plot_full_analysis(results, save_plots=save_plots, plan=plan)
